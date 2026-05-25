"""Faithful ROCODE — vLLM LogitsProcessor backed by a Trie + Strategic
Rollback decider.

Pairs with `soundcode/rocode_trie.py` (the data structure) to reproduce
ROCODE (Jiang et al., 2024) Algorithms 1 and 2 — see
`lit-rev/papers/jiang2024_rocode.pdf` and the upstream reference in
`reproduce/algo/rocode/upstream/`.

Relationship to the MVP
-----------------------
- `soundcode/logits_processors.py::RocodeDecayingPenaltyProcessor` is the
  MVP / lite version. It applies the decayed penalty to a flat list of
  `(token_id, position)` pairs the caller registers manually. No tree,
  no recurrence detection, no high-entropy fallback.
- `RocodeTriePenaltyProcessor` (this file) is the FAITHFUL version used for
  the paper-baseline comparison. It walks a real Trie so penalties stack
  across attempts, and the matching `StrategicRollbackDecider` reproduces
  ROCODE Algorithm 1 (recurrence → r_h, else r_e).
- Both coexist; the algorithm dispatch in `soundcode/web/server.py` picks
  one. Suggested keys: `"rocode_lite"` for the MVP, `"rocode_faithful"` for
  this module. The MVP's existing `"rocode"` key can be kept as an alias
  for `"rocode_lite"` for backward compatibility (the main thread will
  decide).

Entropy contract (READ THIS BEFORE WIRING)
------------------------------------------
ROCODE Eq. 5 needs the per-step entropy H_t = -Σ p log p over the FULL
softmax distribution. The trie stores it; the decider reads it. This module
DOES NOT compute entropy from logits on its own — the orchestration layer
(DemoClient calling into the vLLM hook) is responsible. The contract is:

  After each token is sampled and committed, call
  `processor.record_emitted_token(token_id, entropy=H_t)`. If you only have
  the top-k logprobs (cheap path), feed in the entropy of that truncated
  distribution; the Strategic Rollback decider's argmax is invariant to
  monotonic transformations of the entropy axis, so a top-k approximation
  is usually fine in practice. If entropy is genuinely unavailable, pass
  0.0 — the decider will then pick the SHALLOWEST node when ties are forced
  (see `RocodeTrie.max_entropy_position`).

This processor does NOT pull entropy from vLLM's logprobs itself because
that wiring lives in DemoClient (different thread, different ownership).
That integration is punted to the main thread after this module lands.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from soundcode.rocode_trie import RocodeTrie

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:   # pragma: no cover — torch is a hard dep in practice
    torch = None       # type: ignore[assignment]
    _TORCH_AVAILABLE = False


log = logging.getLogger(__name__)


# ─── minimal Diagnostic surface used by the decider ─────────────────────


@runtime_checkable
class _LinenoBearing(Protocol):
    """Anything with a `.line` attribute that's either an int or None.

    `soundcode.code.Diagnostic` satisfies this (its field is `line`); we
    accept it duck-typed so this module doesn't import the heavy
    `soundcode.code` symbol cycle. Tests pass plain dataclasses.
    """

    line: int | None


# ─── the LogitsProcessor ────────────────────────────────────────────────


class RocodeTriePenaltyProcessor:
    """Faithful ROCODE Eq. 8-9 LogitsProcessor backed by a `RocodeTrie`.

    Called once per decode step. The contract from vLLM is:
      `__call__(token_ids, logits) -> logits`
    where `token_ids` is the prompt+generation prefix at the current step
    and `logits` is a 1-D tensor (or list/ndarray) of size `vocab_size`.

    What we do on each step:
      1. Compute the trie-depth of the NEXT token to be sampled:
           pos = len(token_ids) - prompt_offset + 1 — see `set_prompt_len`.
      2. Look at the current cursor's children: for each child token,
         multiply `logits[token_id]` by `child.penalty_applied`.
         Children with `penalty_applied == 1.0` (no prior attempt or
         distance-0 only) are skipped. Tokens never tried at this position
         are untouched.
      3. Return modified logits. We do NOT call softmax — vLLM's sampler
         does that. Multiplying logits by a positive scalar `m < 1`
         decreases that token's post-softmax probability monotonically,
         which matches Eq. 9 up to the global renormalisation constant.
         (The exact form in Eq. 9 multiplies the *probability* by PN; doing
         the same in logit-space — multiply by `PN` not `log(PN)` —
         deviates strictly from the paper's algebra, but matches what the
         MVP processor does and lets the two processors share a
         downstream-sampling assumption. We document this loud here so a
         future patch can swap to `logits[t] += math.log(child.penalty)`
         for exact Eq. 9 fidelity.)

    Coordinator API (called by the orchestration layer, not by vLLM):
      record_emitted_token(token_id, entropy)  — advance the cursor in
                                                  the trie. Call ONCE per
                                                  committed token.
      mark_error()                              — flag the current leaf as
                                                  a failed end. Called when
                                                  a blocking diagnostic
                                                  lands.
      rollback_and_penalize(pos, lam=None)      — move the cursor up to
                                                  `pos` and apply the
                                                  decayed penalty along
                                                  the abandoned path. lam
                                                  defaults to the value
                                                  passed at construction.
      reset()                                   — drop the entire trie
                                                  (e.g. fresh prompt).
    """

    def __init__(
        self,
        trie: RocodeTrie | None = None,
        lam: float = 0.9,
        *,
        prompt_lineno: int = 0,
    ) -> None:
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"lambda must be in (0, 1], got {lam}")
        self.lam = float(lam)
        self.trie = trie if trie is not None else RocodeTrie(prompt_lineno=prompt_lineno)
        # `prompt_len` is the number of token ids that constitute the prompt
        # (i.e. the offset at which the model's first generated token lives
        # in `token_ids`). Set via set_prompt_len(); if left at 0, we infer
        # by treating ALL prior tokens as generation.
        self._prompt_len: int = 0

    # ─── orchestration-layer entry points ──────────────────────────────

    def set_prompt_len(self, prompt_len: int) -> None:
        """Inform the processor how many leading token ids are the prompt
        (not generated by the model). Used to map vLLM's `len(token_ids)`
        to trie-depth.
        """
        if prompt_len < 0:
            raise ValueError(f"prompt_len must be >= 0, got {prompt_len}")
        self._prompt_len = int(prompt_len)

    def record_emitted_token(
        self,
        token_id: int,
        entropy: float = 0.0,
        lineno: int | None = None,
    ) -> None:
        """Advance the trie cursor by one token after the orchestration
        layer commits a sampled token to the buffer.

        Pass the FULL-distribution H_t if you have it (see module docstring).
        `lineno` defaults to the cursor's current lineno; pass
        `cursor.lineno + 1` when the just-emitted token closed a newline.
        """
        self.trie.add_token(token_id=int(token_id), entropy=float(entropy), lineno=lineno)

    def mark_error(self) -> None:
        """Mark the current leaf as a failed terminal. Display-only."""
        self.trie.mark_error_terminal()

    def rollback_and_penalize(self, rollback_pos: int, lam: float | None = None) -> int:
        """Move the cursor to depth `rollback_pos` AND apply the decayed
        penalty along the abandoned (now-orphaned-from-the-cursor's-PoV)
        path. Returns the number of penalized tokens.

        The two operations MUST happen together: penalty depends on
        knowing the rollback point, and the trie cursor needs to be at
        the same position for the next decode step to start from the
        right place.
        """
        if lam is None:
            lam = self.lam
        penalized = self.trie.apply_decayed_penalty(rollback_pos=rollback_pos, lam=lam)
        self.trie.rollback_to(rollback_pos)
        return penalized

    def reset(self) -> None:
        """Drop the trie entirely (e.g. fresh prompt). Preserves lam and
        prompt_len; only the search state goes away."""
        # Preserve the trie's prompt_lineno via the existing root's value so
        # downstream lineno math stays consistent.
        prompt_lineno = self.trie.root.lineno
        self.trie = RocodeTrie(prompt_lineno=prompt_lineno)
        self._prompt_len = 0

    # ─── vLLM LogitsProcessor contract ─────────────────────────────────

    def __call__(self, token_ids: list[int], logits: Any) -> Any:
        # Compute the trie-depth of the token to be sampled at this step.
        # If `prompt_len` was set, the next-token depth is the number of
        # tokens GENERATED so far plus 1. If it wasn't set, treat the entire
        # `token_ids` as generation (this matches the MVP's convention).
        generated_so_far = max(0, len(token_ids) - self._prompt_len)
        next_pos = generated_so_far + 1

        cursor = self.trie.cursor
        # Cheap exit: at the expected next-position, look at cursor's
        # children. If there's nothing there, no prior attempt has touched
        # this branch, and there's no penalty to apply. ALSO exit cheaply
        # when we've drifted off-cursor (e.g. record_emitted_token wasn't
        # called between steps) — better to silently no-op than to apply
        # wrong penalties.
        if not cursor.children:
            return logits
        if next_pos != cursor.depth + 1:
            # The orchestration layer is out of sync with the trie. Don't
            # try to second-guess; pass through. The Strategic decider
            # logs this as a contract violation if it's chronic.
            return logits

        for tid, child in cursor.children.items():
            multiplier = child.penalty_applied
            if multiplier == 1.0:
                continue
            try:
                logits[tid] = logits[tid] * multiplier
            except Exception:
                # `logits` may be list / ndarray / tensor. The [] assignment
                # works for all three. Exotic types: log + skip.
                log.warning(
                    "RocodeTriePenaltyProcessor: failed to apply multiplier "
                    "to logits[%d] (type=%s)", tid, type(logits).__name__,
                )
        return logits


# ─── Strategic Rollback (Algorithm 1) ───────────────────────────────────


@dataclass
class RollbackDecision:
    """What the decider tells the orchestration loop to do.

    rollback_pos    — trie-depth to roll back to. 0 = back to root (discard
                      all generation, retry from prompt). >0 = roll back to
                      that depth, keep tokens [0, rollback_pos) intact.
    used_recurrence — True if the decider picked r_h because the error
                      recurred. Useful for telemetry / unit tests.
    error_lineno    — the lineno of the diagnostic that triggered this
                      decision. Stored back by the decider so the next call
                      can detect recurrence.
    """

    rollback_pos: int
    used_recurrence: bool
    error_lineno: int | None


class StrategicRollbackDecider:
    """ROCODE Algorithm 1 (Strategic Rollback) — closed-form.

    Maintains a memory of the PREVIOUS diagnostic's lineno so the recurrence
    check ``e_n.lineno == e_{n-1}.lineno`` (paper Algorithm 1 line 5) is
    decoupled from the caller's bookkeeping. The decider is otherwise pure:
    given the current diagnostics and the trie state, it picks
      - r_e = error-line depth (Eq. 4)            if first/different error
      - r_h = high-entropy depth on the line       if recurrence (Eq. 6-7)

    Usage:
        decider = StrategicRollbackDecider(trie=trie)
        decision = decider.decide(diagnostics=[d1, d2, ...])
        processor.rollback_and_penalize(decision.rollback_pos)
    """

    def __init__(self, trie: RocodeTrie) -> None:
        self.trie = trie
        self._previous_error_lineno: int | None = None

    def reset(self) -> None:
        """Forget previous-error state. Called on a fresh prompt."""
        self._previous_error_lineno = None

    @staticmethod
    def _first_blocking_lineno(
        diagnostics: list[_LinenoBearing],
    ) -> int | None:
        """Pick the lineno that defines the rollback target.

        ROCODE always rolls back to ONE error point per attempt; when the
        compiler reports several, we use the FIRST one with a known line
        (earliest in source order, which is also the upstream's convention
        — their `report["error_lineno"]` is the first reported error).
        """
        for d in diagnostics:
            ln = getattr(d, "line", None)
            if ln is not None:
                return int(ln)
        return None

    def decide(
        self,
        diagnostics: list[_LinenoBearing],
        previous_diagnostics: list[_LinenoBearing] | None = None,
    ) -> RollbackDecision:
        """Decide where to roll back to.

        `diagnostics`          — the diagnostics from the FAILING attempt.
        `previous_diagnostics` — optional; if provided, OVERRIDES the
                                 internal memory for recurrence detection.
                                 Useful for unit tests that want to inject
                                 history without calling decide() twice.

        Returns a `RollbackDecision`. The decider DOES NOT mutate the trie
        — the caller (orchestration layer) is responsible for invoking
        `processor.rollback_and_penalize(decision.rollback_pos)` next.
        """
        current_lineno = self._first_blocking_lineno(diagnostics)
        if previous_diagnostics is not None:
            previous_lineno = self._first_blocking_lineno(previous_diagnostics)
        else:
            previous_lineno = self._previous_error_lineno

        # Update the internal memory for the NEXT call. We do this BEFORE
        # the branch decision so a thrown exception below doesn't desync
        # state. (decide() shouldn't throw on well-formed inputs, but
        # defensive is cheap.)
        self._previous_error_lineno = current_lineno

        if current_lineno is None:
            # No actionable diagnostic — roll back to root as a last resort.
            # Should be rare; the orchestration layer is expected to only
            # invoke decide() when there's at least one BLOCKING diagnostic.
            return RollbackDecision(
                rollback_pos=0, used_recurrence=False, error_lineno=None,
            )

        recurrence = RocodeTrie.same_error_recurred(
            latest_lineno=current_lineno,
            previous_lineno=previous_lineno,
        )

        if recurrence:
            # Eq. 6-7: jump to the high-entropy token's line. Upstream walks
            # from the line-start node UP to the root and picks the max-H
            # node; we mirror that by computing the line-start depth and
            # asking the trie for the max-entropy position on the *path*
            # (which traverses root → cursor, naturally including the
            # tokens above the error line — exactly upstream's behavior).
            line_start = self.trie.line_start_depth(current_lineno)
            rollback_pos = self.trie.max_entropy_position()
            # If the high-entropy position is BELOW the line-start (i.e.
            # the entropy spike is inside the error line itself), still
            # roll back to the line start — going below the line-start
            # doesn't escape the error context. Per upstream, the rollback
            # target should be at or above the line containing the error.
            if rollback_pos > line_start and line_start > 0:
                rollback_pos = line_start
            return RollbackDecision(
                rollback_pos=rollback_pos,
                used_recurrence=True,
                error_lineno=current_lineno,
            )

        # First error on this line — use r_e = line-start depth (Eq. 4).
        # Note: upstream stores `r_e = [e.lineno, e.offset]` (2-D); the trie
        # works in scalar depth, so we collapse to line-start (offset 0).
        # The fine-grained column-within-line rollback is a refinement that
        # only matters for languages where one statement spans many tokens
        # on one line; for Rust + statement-grained checkpoints this is
        # already what we want.
        rollback_pos = self.trie.line_start_depth(current_lineno)
        return RollbackDecision(
            rollback_pos=rollback_pos,
            used_recurrence=False,
            error_lineno=current_lineno,
        )


# Sanity: the canonical example must work out.
assert math.isclose(0.9 ** 3, 0.729, abs_tol=1e-9), "λ^3 != 0.729 (impossible)"
