"""ROCODE Trie data structure — the faithful counterpart to the MVP
`RocodeDecayingPenaltyProcessor` in `soundcode/logits_processors.py`.

This module is the data-structure layer; the matching LogitsProcessor and the
Strategic Rollback decider live in `soundcode/rocode_processor.py`. The two
modules together reproduce ROCODE (Jiang et al., 2024) Algorithm 1 + 2 — see
`lit-rev/papers/jiang2024_rocode.pdf` §II and the upstream reference at
`reproduce/algo/rocode/upstream/tire_tree.py` (yes, "tire" is misspelled
upstream; we keep the standard English spelling here).

Why a Trie?
-----------
ROCODE accumulates rollback attempts. After the model tries a bad path and
the compiler rejects it, the decoder rolls back to some earlier position and
re-decodes — but the previously-tried tokens have a multiplicative penalty
applied so the model is *steered away* from re-emitting them. After many
rollbacks, those penalties COMPOUND (Eq. 8 says PN = λ^(t-r); if the same
token is hit on three separate attempts, it accumulates λ^d1 · λ^d2 · λ^d3).
A Trie lets us store every attempted path in one structure so the lookup at
sampling time is O(prefix-depth), and the multiplicative penalty falls out
naturally from walking the tree.

Granularity (1 node per TOKEN, not per statement)
-------------------------------------------------
Upstream stores one node per token (see `tire_tree.py:TreeNode`). The
*lineno* field on each node is what ties the token to its statement; rollback
operates in `(lineno, offset)` coordinates but the trie itself is token-
granular. We follow the same convention: each `TrieNode` is one token, and
`lineno` is the 1-indexed line number assigned when the token was added.

Memory characteristics
----------------------
The trie grows with TOTAL tokens emitted across ALL attempts, not just the
final program. For a HumanEval-sized prompt with at most ~5 rollback retries
and a ~200-token program, that bounds the tree at ~1000 nodes — trivial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# A sentinel for the root node's "token id". Vocabularies are non-negative, so
# any negative integer works; -1 matches upstream's `TreeNode(token_id=-1)`.
ROOT_TOKEN_ID: int = -1


@dataclass
class TrieNode:
    """A single token in the trie of attempted generation paths.

    Fields:
      token_id          — the vocabulary id of this token (`ROOT_TOKEN_ID` at
                          the root).
      parent            — None at root; otherwise the node one step up.
      children          — token_id → child node. dict for O(1) lookup.
      lineno            — 1-indexed line number of this token in the program
                          buffer. Used by the Strategic Rollback decider to
                          map error_lineno → which trie nodes to penalize.
                          Root lineno is the count of prompt lines (matches
                          upstream's `TireTree.__init__`).
      depth             — distance from root in token count. Equal to the
                          position `t` in ROCODE Eq. 8. Stored so penalty
                          computation is O(1) per node.
      entropy           — H_t at this position, computed by the caller from
                          the decoder's softmax distribution per Eq. 5. The
                          orchestration layer (DemoClient or vLLM hook) is
                          responsible for measuring entropy; the trie just
                          stores it. Default 0.0 = "unknown / not measured".
      penalty_applied   — cumulative multiplicative penalty for re-emitting
                          this exact token at this exact position. 1.0 means
                          "no penalty"; e.g. 0.81 means the next sampling
                          step at this position will multiply this token's
                          probability by 0.81. Accumulates ACROSS attempts:
                          if the same node is penalized twice with factors
                          λ^d1 and λ^d2, the field holds λ^(d1+d2).
      is_error_terminal — True if a blocking diagnostic was triggered AT or
                          AFTER this node and the path was rolled back. Used
                          for diagnostics / display, not for the math itself.
    """

    token_id: int
    parent: "TrieNode | None" = None
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    lineno: int = 0
    depth: int = 0
    entropy: float = 0.0
    penalty_applied: float = 1.0
    is_error_terminal: bool = False

    def is_root(self) -> bool:
        return self.parent is None


class RocodeTrie:
    """Trie of all attempted generation paths during one task.

    Reproduces the *shape* of upstream's `TireTree` but with a cleaner
    interface and modern Python (dataclasses, type hints, no global mutable
    state). Operations:

      add_token(token_id, entropy, lineno=None) — extend the current path
                                                 by one token.
      mark_error_terminal()                     — flag the current leaf as a
                                                 failed end (display-only).
      rollback_to(token_pos)                    — move the cursor up the
                                                 tree to absolute depth
                                                 `token_pos`. Subsequent
                                                 add_token() calls extend
                                                 from there.
      apply_decayed_penalty(rollback_pos, lam)  — walk the path from
                                                 `rollback_pos` down to the
                                                 last leaf of the most
                                                 recent attempt, applying
                                                 λ^(t - rollback_pos) to
                                                 each node's
                                                 penalty_applied. Implements
                                                 Eq. 8.
      get_penalty_for(token_id, pos)            — what multiplier should be
                                                 applied to this token's
                                                 logit if the next decode
                                                 step is at position `pos`?
                                                 Returns 1.0 if no prior
                                                 attempt visited that token
                                                 at that depth.
      max_entropy_position()                    — argmax_t H_t over the
                                                 current path from root to
                                                 cursor. Used for r_h (the
                                                 high-entropy fallback,
                                                 Eq. 6-7).

    The "current cursor" (`self.cursor`) is the deepest node on the path
    currently being extended. After a rollback, the cursor moves UP; the
    subtree below it stays in memory so its penalty contributions persist.
    """

    def __init__(self, prompt_lineno: int = 0) -> None:
        # Root represents the START of model-generated content. `prompt_lineno`
        # is the number of lines in the prompt (so the first model token lives
        # on `prompt_lineno + 1`). Upstream sets this from `code_context`.
        self.root = TrieNode(token_id=ROOT_TOKEN_ID, lineno=prompt_lineno)
        self.cursor: TrieNode = self.root
        # The deepest leaf of the MOST RECENT attempt — needed by
        # apply_decayed_penalty to know how far down to walk after rollback.
        # Updated by add_token; reset by rollback_to.
        self._last_attempt_leaf: TrieNode = self.root

    # ─── extension / rollback ────────────────────────────────────────────

    def add_token(
        self,
        token_id: int,
        entropy: float = 0.0,
        lineno: int | None = None,
    ) -> TrieNode:
        """Extend the current cursor by one token. Returns the new (or
        existing) child node.

        If a child with `token_id` already exists (because a previous attempt
        emitted the same token at this exact position), we REUSE it — the
        penalty field accumulates across attempts. Otherwise we create a new
        child. `entropy` is updated on every visit (most recent wins; the
        decoder's distribution can shift after penalties are applied, but for
        the Strategic Rollback decision we only need a *recent* H_t per
        position, not a historical average).

        `lineno` defaults to "same as parent" — callers that know the token
        completed a newline pass `parent.lineno + 1`.
        """
        parent = self.cursor
        if lineno is None:
            lineno = parent.lineno

        child = parent.children.get(token_id)
        if child is None:
            child = TrieNode(
                token_id=token_id,
                parent=parent,
                lineno=lineno,
                depth=parent.depth + 1,
                entropy=entropy,
            )
            parent.children[token_id] = child
        else:
            # Reuse: update entropy + lineno (in case the lineno bookkeeping
            # disagrees, the most recent visit wins). `penalty_applied`
            # is NOT reset here — penalties from prior attempts must persist.
            child.entropy = entropy
            child.lineno = lineno

        self.cursor = child
        self._last_attempt_leaf = child
        return child

    def mark_error_terminal(self) -> None:
        """Flag the current cursor as the end of a failed attempt.

        Display-only — not used in the penalty math. Useful for visualization
        and for unit tests that want to assert "this attempt ended in error".
        """
        self.cursor.is_error_terminal = True

    def rollback_to(self, token_pos: int) -> TrieNode:
        """Move the cursor up the tree to absolute depth `token_pos`.

        `token_pos` is in trie-coordinate-system: 0 = root, 1 = first token
        below root, etc. Raises ValueError if `token_pos` is outside
        [0, current_depth].

        The subtree below the new cursor is NOT deleted — its penalty data
        persists so when re-decoding visits the same (parent, token_id) pair,
        `add_token` will reuse the existing node and its accumulated penalty.
        """
        if token_pos < 0:
            raise ValueError(f"token_pos must be >= 0, got {token_pos}")
        if token_pos > self.cursor.depth:
            raise ValueError(
                f"token_pos {token_pos} exceeds current depth {self.cursor.depth}"
            )

        node = self.cursor
        while node.depth > token_pos:
            assert node.parent is not None, "depth>0 implies parent exists"
            node = node.parent
        self.cursor = node
        # `_last_attempt_leaf` is preserved — apply_decayed_penalty needs to
        # walk FROM the rollback point DOWN to the last leaf of the attempt
        # we're penalizing.
        return node

    # ─── penalty application + lookup ────────────────────────────────────

    def apply_decayed_penalty(
        self,
        rollback_pos: int,
        lam: float = 0.9,
    ) -> int:
        """Apply ROCODE Eq. 8 penalty to every node from `rollback_pos` down
        to the leaf of the most recent attempt.

        For each node at depth t in that path, multiplies its
        `penalty_applied` by `lam ** (t - rollback_pos)`:
          - the node AT the rollback point (t == rollback_pos) gets λ^0 = 1
          - the next token gets λ^1
          - the token after that gets λ^2, etc.

        Concrete example: with lam=0.9 and rollback_pos=5, tokens at trie
        depths 5, 6, 7 get cumulative multipliers 1.0, 0.9, 0.81. With
        lam=0.9 and distance=3, the multiplier is 0.9^3 = 0.729.

        Per upstream `decay_path`, we DO NOT touch the rollback-pos node
        itself (distance-0 penalty is the identity, so this is a no-op).
        Returns the number of nodes that received a non-identity penalty
        (i.e. rollback length in tokens) — useful for the caller to refund
        the generation-length budget.

        Requires: 0 <= rollback_pos <= last_attempt_leaf.depth.
        Requires: 0 < lam <= 1.
        """
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"lam must be in (0, 1], got {lam}")
        if rollback_pos < 0:
            raise ValueError(f"rollback_pos must be >= 0, got {rollback_pos}")
        if rollback_pos > self._last_attempt_leaf.depth:
            raise ValueError(
                f"rollback_pos {rollback_pos} exceeds last attempt leaf depth "
                f"{self._last_attempt_leaf.depth}"
            )

        # Walk from the leaf upward, collecting nodes, then apply penalties
        # top-down so the multiplier-by-distance is straightforward to compute.
        path: list[TrieNode] = []
        node: TrieNode | None = self._last_attempt_leaf
        while node is not None and node.depth >= rollback_pos:
            path.append(node)
            node = node.parent
        path.reverse()  # now ordered shallowest -> deepest

        penalized = 0
        for n in path:
            distance = n.depth - rollback_pos
            if distance == 0:
                continue  # identity multiplier; skip per upstream convention
            n.penalty_applied *= lam ** distance
            penalized += 1
        return penalized

    def get_penalty_for(self, token_id: int, pos: int) -> float:
        """Return the cumulative multiplier that should be applied to
        `logits[token_id]` if the next decode step is at trie-depth `pos`.

        Walks: at depth `pos-1` from the root along the CURRENT cursor path,
        find the node whose child includes `token_id`. That child's
        `penalty_applied` is the answer.

        Returns 1.0 (no penalty) if no prior attempt ever placed `token_id`
        at this exact (parent, depth) pair.

        Concretely: if the cursor is at depth 5 and `pos==6`, we look at
        cursor's children — that's "what penalty does token X get if it's
        emitted as the 6th token below root, given the current prefix?".
        """
        if pos <= 0:
            return 1.0
        if pos == self.cursor.depth + 1:
            # Fast path: next-token decision from the current cursor.
            child = self.cursor.children.get(token_id)
            return child.penalty_applied if child is not None else 1.0

        # General case: walk from the cursor toward the root until we find the
        # node at depth `pos - 1` on the current ancestry.
        node: TrieNode | None = self.cursor
        while node is not None and node.depth > pos - 1:
            node = node.parent
        if node is None or node.depth != pos - 1:
            return 1.0
        child = node.children.get(token_id)
        return child.penalty_applied if child is not None else 1.0

    # ─── analysis helpers used by the Strategic Rollback decider ─────────

    def max_entropy_position(self) -> int:
        """Return the trie-depth of the maximum-entropy node on the path
        from root to current cursor (exclusive of root).

        Implements Eq. 6: t* = argmax_{t ∈ [0,|y|]} H_t. We exclude the root
        because the root has no measured entropy (it represents the prompt
        boundary, not a sampled token). Ties broken by SHALLOWEST position
        (smallest depth) — matches "earliest cause" intuition; upstream's
        loop happens to favor the deepest tie because it iterates leaf→root
        and uses strict `>`. We mirror upstream's strict-> behavior so the
        decider yields the same answers on real inputs.

        If the path is empty (cursor == root), returns 0.
        """
        node: TrieNode | None = self.cursor
        best: TrieNode | None = None
        while node is not None and not node.is_root():
            if best is None or node.entropy > best.entropy:
                best = node
            node = node.parent
        return 0 if best is None else best.depth

    def max_entropy_position_below(self, ancestor_depth: int) -> int:
        """Like `max_entropy_position` but restricted to nodes STRICTLY
        BELOW `ancestor_depth` on the current path.

        Used by the recurrence branch of Strategic Rollback: if the error
        keeps recurring on the same line, we want the highest-entropy token
        WITHIN THE LINE (not anywhere on the whole path). Upstream computes
        this by starting from the line-start node and walking up; we mirror
        that by starting from the cursor and walking up, but stopping above
        `ancestor_depth`.
        """
        node: TrieNode | None = self.cursor
        best: TrieNode | None = None
        while node is not None and not node.is_root() and node.depth > ancestor_depth:
            if best is None or node.entropy > best.entropy:
                best = node
            node = node.parent
        return ancestor_depth if best is None else best.depth

    def line_start_depth(self, lineno: int) -> int:
        """Return the trie-depth of the FIRST token on `lineno`, walking up
        from the current cursor.

        Maps the compiler's error_lineno → trie coordinates. Returns 0 (root)
        if no token on `lineno` is on the current path (e.g. the error line
        is in the prompt, before any model-generated tokens).
        """
        node: TrieNode | None = self.cursor
        first_on_line: TrieNode | None = None
        while node is not None and not node.is_root():
            if node.lineno == lineno:
                first_on_line = node
            elif node.lineno < lineno and first_on_line is not None:
                # We've walked above the line; the most recent match is the
                # earliest one (we're going leaf→root, so "earliest" here
                # means smallest depth seen so far).
                break
            node = node.parent
        return 0 if first_on_line is None else first_on_line.depth

    # ─── utilities ───────────────────────────────────────────────────────

    def current_depth(self) -> int:
        """Distance in tokens from the root to the cursor."""
        return self.cursor.depth

    def path_token_ids(self) -> list[int]:
        """Return the token_ids on the current root→cursor path, in order.

        Useful for: (a) reconstructing what the LM has committed to so far,
        (b) tests that want to assert path-equality.
        """
        ids: list[int] = []
        node: TrieNode | None = self.cursor
        while node is not None and not node.is_root():
            ids.append(node.token_id)
            node = node.parent
        ids.reverse()
        return ids

    def total_nodes(self) -> int:
        """Count all nodes in the trie (for telemetry / memory bookkeeping)."""
        count = 0
        stack: list[TrieNode] = [self.root]
        while stack:
            n = stack.pop()
            count += 1
            stack.extend(n.children.values())
        return count

    @staticmethod
    def same_error_recurred(
        latest_lineno: int | None,
        previous_lineno: int | None,
    ) -> bool:
        """Algorithm 1 line 5 recurrence check: ``e_n.lineno == e_{n-1}.lineno``.

        Returns False if either lineno is None (e.g. the very first error has
        no predecessor). Static helper because the comparison is trivially
        decoupled from trie state — the decider in `rocode_processor.py`
        uses this to choose between r_e and r_h.
        """
        if latest_lineno is None or previous_lineno is None:
            return False
        return latest_lineno == previous_lineno


# ─── debugging helper (unused in tests, handy when iterating manually) ───


def render_trie(trie: RocodeTrie) -> str:
    """Pretty-print the trie as a multi-line string. Used for ad-hoc
    debugging; not called from the orchestration loop."""
    lines: list[str] = []

    def walk(node: TrieNode, prefix: str) -> None:
        marker = "*" if node is trie.cursor else " "
        err = "!" if node.is_error_terminal else " "
        lines.append(
            f"{prefix}{marker}{err}id={node.token_id} d={node.depth} "
            f"line={node.lineno} H={node.entropy:.3f} "
            f"penalty={node.penalty_applied:.3f}"
        )
        for child in node.children.values():
            walk(child, prefix + "  ")

    walk(trie.root, "")
    return "\n".join(lines)


# Sanity check the math doesn't get caught by precision issues on import.
assert math.isclose(0.9 ** 3, 0.729, abs_tol=1e-9), "λ^3 != 0.729 (impossible)"
