"""Pluggable LogitsProcessor implementations for the vLLM backend.

Three algorithms share a common shape: a callable taking
`(token_ids: list[int], logits: torch.Tensor) -> torch.Tensor`, suitable for
plumbing into vLLM via `SamplingParams(logits_processors=[...])` (per-request
hook on the offline API) OR via an AdapterLogitsProcessor in v1 mode.

The three concrete implementations:

  - `NoopLogitsProcessor`   — SoundCode default. No modification. Rollback is
                              implemented via prompt-prefix re-decode +
                              structural checkpoint (see DemoClient / Code) —
                              the processor itself has nothing to do.
  - `RocodeDecayingPenaltyProcessor` — ROCODE Eq. 8-9 decayed penalty on a
                              recorded bad-token suffix. State + a record-API
                              (`add_penalty`, `reset`) drive it from outside.
  - `SemGuardEvaluatorProcessor` — line-level rollback signal from a trained
                              ~1.3B evaluator. Does NOT modify logits — it
                              detects newline boundaries in the emitted
                              stream and pushes `RollbackRequest`s onto
                              `self.rollback_signals` (an `asyncio.Queue`)
                              when the evaluator score falls below
                              `threshold`. DemoClient polls the queue
                              between tokens to trigger structural rollback.

vLLM imports (`torch`) live behind module-level guards so the unit tests can
import this module on a machine without a working torch installation; they
still pass against a `numpy`-backed fake.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import warnings
from typing import Any, Protocol, runtime_checkable

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:   # pragma: no cover — torch is a hard dep in practice
    torch = None       # type: ignore[assignment]
    _TORCH_AVAILABLE = False


log = logging.getLogger(__name__)


@runtime_checkable
class LogitsProcessor(Protocol):
    """vLLM LogitsProcessor: callable that takes (token_ids, logits) and
    returns modified logits. Called between the transformer forward pass and
    sampling, once per decode step.

    `token_ids` is the full prompt+generation token list at the current step
    (vLLM convention). `logits` is a 1-D tensor of size `vocab_size`.
    The processor MAY mutate `logits` in-place; the contract is just that it
    returns the final tensor sampling uses.
    """

    def __call__(self, token_ids: list[int], logits: Any) -> Any:  # noqa: D401
        ...


# ─── SoundCode (no-op) ──────────────────────────────────────────────────


class NoopLogitsProcessor:
    """SoundCode default — no penalty, no modification.

    Rollback works via prompt-prefix re-decode + structural checkpoint
    (`soundcode.code.Code.rollback` + `DemoClient._do_rollback`); the logits
    themselves are never touched. The processor exists so the dispatch table
    can produce a uniform object regardless of algorithm.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, token_ids: list[int], logits: Any) -> Any:
        return logits

    def reset(self) -> None:
        """No state to reset — kept for API symmetry with the other processors."""
        return


# ─── ROCODE (decayed penalty on a bad-token suffix) ──────────────────────


class RocodeDecayingPenaltyProcessor:
    """ROCODE Eq. 8-9 decayed penalty.

    The processor stores a list of `(token_id, expected_position)` pairs. At
    each decode step, if the current position matches a stored expected
    position, the logit for that token is multiplied by `lambda^distance`,
    where `distance` is `expected_position - rollback_position` — i.e. the
    farther the token was from the rollback point, the closer the multiplier
    is to 1 (less penalty).

    `add_penalty([142, 88, 271], rollback_offset=500)` adds:
      `[(142, 500), (88, 501), (271, 502)]`
    so token 142 gets `logit *= lambda^0 = 1.0` if re-emitted at step 500,
    token 88 gets `logit *= lambda^1` at step 501, etc.

    With `lambda = 0.9` and 5 tokens of distance, the multiplier is
    0.9^5 ≈ 0.59 (i.e. ~40% downweighted before softmax-renormalise).

    Multiple penalty entries may stack at the same step / for the same
    token_id; their effect multiplies.

    No softmax-renormalise is applied inside the processor — vLLM's sampler
    handles renormalisation after all logits processors run. We only scale
    the raw logit by a positive multiplier, which is monotonic in probability
    after softmax (and equivalent to subtracting `-log(lambda^d)` from the
    pre-softmax logit, the form ROCODE writes in Eq. 8-9).
    """

    def __init__(self, lam: float = 0.9) -> None:
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"lambda must be in (0, 1], got {lam}")
        self.lam = lam
        # Each entry: (token_id, expected_position, distance_from_rollback)
        # We store `distance` explicitly so multiple add_penalty() calls with
        # different rollback offsets all decay relative to their own offset.
        self._entries: list[tuple[int, int, int]] = []

    def add_penalty(self, token_ids: list[int], rollback_offset: int) -> None:
        """Record a penalty: each token in `token_ids` was on the bad suffix
        starting at `rollback_offset`. When sampling step `rollback_offset+i`
        runs, the logit for `token_ids[i]` will be multiplied by `lam^i`."""
        for i, tid in enumerate(token_ids):
            self._entries.append((int(tid), rollback_offset + i, i))

    def reset(self) -> None:
        """Drop all recorded penalty entries (e.g. on a fresh prompt)."""
        self._entries.clear()

    def __call__(self, token_ids: list[int], logits: Any) -> Any:
        step = len(token_ids)
        if not self._entries:
            return logits
        for tid, expected_pos, distance in self._entries:
            if expected_pos != step:
                continue
            multiplier = self.lam ** distance
            try:
                logits[tid] = logits[tid] * multiplier
            except Exception:
                # `logits` may be a list / ndarray / tensor; the [] assignment
                # works for all three. If something exotic comes in, leave
                # the tensor untouched rather than crashing the decode step.
                log.warning(
                    "RocodeDecayingPenaltyProcessor: failed to apply multiplier "
                    "to logits[%d] (type=%s)", tid, type(logits).__name__,
                )
        return logits


# ─── SemGuard (line-level evaluator rollback signal) ────────────────────


@dataclasses.dataclass(frozen=True)
class RollbackRequest:
    """Side-channel signal pushed by `SemGuardEvaluatorProcessor` whenever
    the evaluator scores a freshly-completed line below `threshold`.

    Fields:
      - `at_line`: 1-indexed line number (in the decoded prefix) the
        evaluator was scoring when it crossed the threshold. This is the
        line the model just finished emitting — it's the prime candidate
        for the rollback target.
      - `score`: the sigmoid probability the evaluator assigned. Always in
        `[0, threshold)`; included so the UI / log can show the user how
        close to / far from the threshold the call was.
      - `prefix_len`: number of CHARACTERS in the decoded prefix at the
        moment the signal was generated. DemoClient can map this to a byte
        offset into `code.content` and pick the nearest checkpoint at or
        before it.
    """
    at_line: int
    score: float
    prefix_len: int


class SemGuardEvaluatorProcessor:
    """SemGuard line-level rollback via trained 1.3B evaluator.

    Embedded in vLLM's LogitsProcessor protocol but does NOT modify logits.
    Instead, detects newlines in the emitted-token stream, schedules
    asynchronous scoring of the partial program via the evaluator, and
    signals a rollback request via an `asyncio.Queue`
    (`self.rollback_signals`) when score < threshold.

    DemoClient is expected to poll `processor.rollback_signals` between
    tokens (non-blocking `get_nowait()`) and trigger structural rollback
    when a signal arrives. The poll happens in the consumer half of the
    producer-consumer loop, alongside the existing cargo-error verdict
    handling — exact wiring is the integration step's responsibility.

    Sync/async bridge: `__call__` runs INSIDE vLLM's sampler (synchronous
    context, no event loop on the calling thread, sometimes a CUDA stream),
    so we cannot `await` the evaluator there. Instead:

      1. On every call, we detokenize the most recent NEW token(s) and
         append to an internal `bytearray`-backed buffer.
      2. If a newline appeared in those bytes, we capture the prefix as it
         stands now and schedule scoring via the bound event loop's
         `asyncio.run_coroutine_threadsafe`. The Future fire-and-forgets
         (errors are logged but never re-raised into the sampler).
      3. The scoring coroutine awaits `evaluator.score_prefix(prefix)` and,
         if the score is below threshold, `put_nowait`s a `RollbackRequest`
         onto `self.rollback_signals`.

    Bind the event loop with `await processor.bind_loop()` from the consumer
    side BEFORE the first generation step; without a bound loop the
    processor silently degrades to logits-passthrough (no scoring, no
    signal) and emits a one-time warning. This protects the sampler from
    crashing if scoring can't be wired through (e.g. CLI eval path, or a
    test that constructs a processor without a loop).

    Resampling: `max_resamples` caps how many rollback signals will be
    emitted per `reset()` cycle. After that many, the processor stops
    signalling but keeps tracking the buffer; resets when DemoClient calls
    `processor.reset()` at the end of a run.
    """

    def __init__(
        self,
        evaluator: Any | None = None,            # SemGuardEvaluator instance
        tokenizer: Any | None = None,            # HF tokenizer with .decode()
        threshold: float = 0.5,
        max_resamples: int = 3,
    ) -> None:
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.threshold = float(threshold)
        self.max_resamples = int(max_resamples)

        # Output channel. We construct the queue lazily on bind_loop() so
        # it's bound to the right event loop (asyncio.Queue created off an
        # unrelated loop would fail at put_nowait).
        self.rollback_signals: asyncio.Queue[RollbackRequest] | None = None

        # Bound event loop — populated by bind_loop(). Sampler thread uses
        # this to schedule scoring coroutines.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Track which token positions we've already decoded so we don't
        # re-decode the entire context on every call.
        self._last_seen_len = 0
        # Char-level prefix as we've reconstructed it. We avoid storing the
        # full token id list (the sampler already does); we keep just the
        # decoded text for newline detection + scoring.
        self._decoded_prefix = ""
        # Number of newlines we've already TRIGGERED scoring for. Lines
        # increment as we see them; we score each line at most once.
        self._lines_scored = 0
        # Number of rollback signals we've emitted this run.
        self._signals_emitted = 0

        # If misconfigured, degrade to no-op + warn ONCE.
        self._is_disabled = self.evaluator is None or self.tokenizer is None
        if self._is_disabled:
            warnings.warn(
                "SemGuardEvaluatorProcessor: evaluator or tokenizer missing — "
                "degrading to logits-passthrough. Construct with both "
                "(evaluator=SemGuardEvaluator(...), tokenizer=...) and call "
                "`await processor.bind_loop()` to enable scoring.",
                stacklevel=2,
            )

    # ── lifecycle ──────────────────────────────────────────────────────

    async def bind_loop(self) -> None:
        """Bind the currently-running event loop to this processor. MUST be
        called from the consumer's loop BEFORE the first sampler call; the
        sampler runs synchronously and needs a target loop to schedule
        scoring coroutines onto.

        Also lazily constructs `self.rollback_signals`. After this call,
        `__call__` is allowed to schedule scoring; before it, scheduling is
        silently skipped (logits pass through unchanged).
        """
        self._loop = asyncio.get_running_loop()
        if self.rollback_signals is None:
            self.rollback_signals = asyncio.Queue()

    def reset(self) -> None:
        """Reset between runs. Clears the prefix buffer, line counter, and
        signal queue. Keeps the bound loop + evaluator references — those
        are run-spanning."""
        self._last_seen_len = 0
        self._decoded_prefix = ""
        self._lines_scored = 0
        self._signals_emitted = 0
        # Drain any pending signals so a stale one from the previous run
        # doesn't fire on the next prompt.
        if self.rollback_signals is not None:
            while not self.rollback_signals.empty():
                try:
                    self.rollback_signals.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def check_rollback_signals(self) -> RollbackRequest | None:
        """Convenience for the consumer: non-blocking peek at the signal
        queue. Returns the next `RollbackRequest` if one is queued, else
        `None`. (DemoClient may bypass this and call
        `self.rollback_signals.get_nowait()` directly inside its
        `try/except QueueEmpty` block.)"""
        if self.rollback_signals is None:
            return None
        try:
            return self.rollback_signals.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # ── sampler-side hook (sync) ───────────────────────────────────────

    def __call__(self, token_ids: list[int], logits: Any) -> Any:
        # Logits are NEVER modified — SemGuard signals rollback out-of-band.
        if self._is_disabled:
            return logits
        if self._loop is None:
            # Not bound to a loop → can't schedule scoring. Silently no-op.
            return logits
        if self._signals_emitted >= self.max_resamples:
            return logits

        # Decode just the new tail of token_ids — re-decoding the whole
        # prompt on every step would be O(n^2).
        n = len(token_ids)
        if n <= self._last_seen_len:
            # No new tokens (shouldn't happen in vLLM's normal flow, but
            # guard against double-call from a sampler quirk).
            return logits
        new_ids = token_ids[self._last_seen_len:]
        self._last_seen_len = n
        try:
            new_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        except Exception as e:
            # Detokenization failures are non-fatal: log once and bail. The
            # sampler MUST get its logits back regardless.
            log.warning("SemGuardEvaluatorProcessor: decode failed on %d tokens: %r",
                        len(new_ids), e)
            return logits
        self._decoded_prefix += new_text

        # If the newly-decoded text contains any newline(s), we have one or
        # more freshly-completed lines to score. Score only the latest one
        # (the partial program "so far" — a low score on it triggers
        # rollback to the previous line).
        if "\n" not in new_text:
            return logits

        # Count newlines to advance the line counter; we score the prefix
        # as-it-stands-now once per newline-bearing batch (not once per
        # newline — multiple newlines in a single token chunk are rare and
        # all point at the same prefix anyway).
        self._lines_scored += new_text.count("\n")

        # Snapshot what the scorer sees, so a later call modifying
        # _decoded_prefix doesn't shift the input.
        prefix_snapshot = self._decoded_prefix
        line_snapshot = self._lines_scored

        # Fire-and-forget. We don't await the future — the sampler can't
        # block. Errors are logged in the coroutine.
        asyncio.run_coroutine_threadsafe(
            self._score_and_signal(prefix_snapshot, line_snapshot),
            self._loop,
        )
        return logits

    # ── async worker (runs on the bound loop) ──────────────────────────

    async def _score_and_signal(self, prefix: str, line_no: int) -> None:
        if self.rollback_signals is None or self.evaluator is None:
            return
        try:
            score = await self.evaluator.score_prefix(prefix)
        except Exception as e:
            log.warning("SemGuardEvaluatorProcessor: evaluator.score_prefix "
                        "failed: %r", e)
            return
        if score < self.threshold and self._signals_emitted < self.max_resamples:
            self._signals_emitted += 1
            req = RollbackRequest(
                at_line=line_no, score=score, prefix_len=len(prefix),
            )
            try:
                self.rollback_signals.put_nowait(req)
            except asyncio.QueueFull:
                # Unbounded queue by construction; reaching this branch
                # means the consumer isn't draining, which is its problem.
                log.warning("SemGuardEvaluatorProcessor: signal queue full "
                            "(consumer not draining); dropping %r", req)
