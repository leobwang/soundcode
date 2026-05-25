"""Unit tests for the faithful ROCODE implementation.

Covers `soundcode/rocode_trie.py` (the data structure), the
`StrategicRollbackDecider`, and `RocodeTriePenaltyProcessor` (the
LogitsProcessor) — all in `soundcode/rocode_processor.py`.

These tests are pure-Python — no torch, no GPU, no model. The processor's
logit modification path is exercised against plain `list[float]` inputs to
stay backend-agnostic.

Run: `uv run pytest tests/test_rocode_trie.py -xvs`
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from soundcode.rocode_processor import (
    RocodeTriePenaltyProcessor,
    StrategicRollbackDecider,
)
from soundcode.rocode_trie import RocodeTrie, TrieNode


# ─── tiny Diagnostic stand-in for the decider (duck-typed `.line`) ────


@dataclass
class _Diag:
    line: int | None
    message: str = ""


# ─── RocodeTrie: structural / extension tests ────────────────────────────


def test_trie_add_token_creates_child():
    """Adding a token below the root creates a new child with parent = root."""
    trie = RocodeTrie(prompt_lineno=5)
    assert trie.cursor is trie.root
    assert trie.root.lineno == 5
    assert trie.current_depth() == 0

    n = trie.add_token(token_id=42, entropy=1.234, lineno=6)
    assert n.token_id == 42
    assert n.parent is trie.root
    assert n.depth == 1
    assert n.entropy == pytest.approx(1.234)
    assert n.lineno == 6
    assert trie.cursor is n
    assert trie.root.children[42] is n


def test_trie_add_token_reuses_existing_child_and_preserves_penalty():
    """Adding the same token at the same depth twice must reuse the node so
    accumulated penalties from prior attempts survive."""
    trie = RocodeTrie()
    n1 = trie.add_token(token_id=7, entropy=0.5)
    n1.penalty_applied = 0.5  # pretend a previous attempt penalized it
    # Roll back to root and re-add the same token: must reuse n1.
    trie.rollback_to(0)
    n2 = trie.add_token(token_id=7, entropy=0.9)  # entropy update on revisit
    assert n2 is n1
    assert n2.penalty_applied == pytest.approx(0.5)  # preserved
    assert n2.entropy == pytest.approx(0.9)          # updated


def test_trie_rollback_to_returns_correct_ancestor():
    """rollback_to(k) moves the cursor to the unique ancestor at depth k."""
    trie = RocodeTrie()
    trie.add_token(token_id=1)
    n2 = trie.add_token(token_id=2)
    trie.add_token(token_id=3)
    trie.add_token(token_id=4)
    assert trie.current_depth() == 4

    ret = trie.rollback_to(2)
    assert ret is n2
    assert trie.cursor is n2
    assert trie.current_depth() == 2


def test_trie_rollback_to_root_works_and_is_idempotent():
    """rollback_to(0) walks all the way back to root, even from deep paths."""
    trie = RocodeTrie()
    for tid in range(10):
        trie.add_token(token_id=tid)
    trie.rollback_to(0)
    assert trie.cursor is trie.root
    # And again — should not error.
    trie.rollback_to(0)
    assert trie.cursor is trie.root


def test_trie_rollback_to_rejects_out_of_range():
    trie = RocodeTrie()
    trie.add_token(token_id=1)
    with pytest.raises(ValueError):
        trie.rollback_to(-1)
    with pytest.raises(ValueError):
        trie.rollback_to(5)  # cursor is at depth 1


# ─── RocodeTrie: penalty math (ROCODE Eq. 8) ─────────────────────────────


def test_trie_apply_decayed_penalty_correctness():
    """λ=0.9, rollback_pos=5; tokens at depths 5,6,7 should end up with
    penalties 0.9^0=1, 0.9^1=0.9, 0.9^2=0.81 (distance from rollback_pos).

    Specifically: the rollback-pos node itself gets λ^0 = no-op; the next
    token gets λ^1; the one after gets λ^2; etc.
    """
    trie = RocodeTrie()
    for tid in range(10):  # depths 1..10
        trie.add_token(token_id=tid)

    trie.apply_decayed_penalty(rollback_pos=5, lam=0.9)

    # Collect penalties keyed by depth.
    penalties: dict[int, float] = {}
    node = trie._last_attempt_leaf
    while node is not None and not node.is_root():
        penalties[node.depth] = node.penalty_applied
        node = node.parent

    # Distance 0 (at rollback_pos): unchanged.
    assert penalties[5] == pytest.approx(1.0)
    # Distance 1: 0.9
    assert penalties[6] == pytest.approx(0.9)
    # Distance 2: 0.81
    assert penalties[7] == pytest.approx(0.81)
    # Distance 3: 0.729
    assert penalties[8] == pytest.approx(0.729)
    # Distance 5: 0.9^5
    assert penalties[10] == pytest.approx(0.9 ** 5)
    # Above the rollback point: untouched.
    assert penalties[4] == pytest.approx(1.0)
    assert penalties[1] == pytest.approx(1.0)


def test_trie_apply_decayed_penalty_accumulates_across_attempts():
    """Two penalize-then-rollback cycles on the same token must multiply
    penalties together, not overwrite. This is the whole point of the trie.
    """
    trie = RocodeTrie()
    trie.add_token(token_id=99)        # depth 1
    trie.add_token(token_id=42)        # depth 2 — the bad token
    trie.apply_decayed_penalty(rollback_pos=1, lam=0.9)
    # Node at depth 2 should have multiplier 0.9^(2-1) = 0.9
    assert trie._last_attempt_leaf.penalty_applied == pytest.approx(0.9)

    # Pretend the second attempt also went down the same path and we
    # penalize again from the same rollback point.
    trie.rollback_to(1)
    trie.add_token(token_id=42)        # reuses the existing depth-2 node
    trie.apply_decayed_penalty(rollback_pos=1, lam=0.9)
    # Now: 0.9 * 0.9 = 0.81
    assert trie._last_attempt_leaf.penalty_applied == pytest.approx(0.81)


def test_trie_apply_decayed_penalty_rejects_bad_lambda():
    trie = RocodeTrie()
    trie.add_token(token_id=1)
    with pytest.raises(ValueError):
        trie.apply_decayed_penalty(rollback_pos=0, lam=0.0)
    with pytest.raises(ValueError):
        trie.apply_decayed_penalty(rollback_pos=0, lam=1.5)


def test_trie_get_penalty_for_returns_one_when_no_history():
    """Unvisited (token, position) pairs return multiplier 1.0."""
    trie = RocodeTrie()
    assert trie.get_penalty_for(token_id=42, pos=1) == 1.0
    assert trie.get_penalty_for(token_id=42, pos=99) == 1.0


def test_trie_get_penalty_for_returns_accumulated_value():
    """After a penalty is applied, get_penalty_for at the matching
    (parent_position+1) returns the same value."""
    trie = RocodeTrie()
    trie.add_token(token_id=10)  # depth 1
    trie.add_token(token_id=20)  # depth 2
    trie.apply_decayed_penalty(rollback_pos=1, lam=0.5)
    # Cursor is at depth 2 still; roll back to depth 1 to query its child.
    trie.rollback_to(1)
    # Next-token at pos 2: penalty for token 20 is 0.5 (distance 1, λ^1).
    assert trie.get_penalty_for(token_id=20, pos=2) == pytest.approx(0.5)
    # Other tokens at pos 2: unpenalized.
    assert trie.get_penalty_for(token_id=999, pos=2) == 1.0


# ─── RocodeTrie: high-entropy lookup (ROCODE Eq. 6-7) ───────────────────


def test_trie_max_entropy_position_returns_argmax():
    """max_entropy_position returns the depth of the highest-H node on the
    current path."""
    trie = RocodeTrie()
    trie.add_token(token_id=1, entropy=0.1)  # depth 1
    trie.add_token(token_id=2, entropy=2.5)  # depth 2 — winner
    trie.add_token(token_id=3, entropy=1.0)  # depth 3
    trie.add_token(token_id=4, entropy=0.5)  # depth 4
    assert trie.max_entropy_position() == 2


def test_trie_max_entropy_position_with_empty_path_returns_zero():
    trie = RocodeTrie()
    assert trie.max_entropy_position() == 0


def test_trie_line_start_depth_finds_first_token_on_line():
    """line_start_depth(L) returns the depth of the FIRST node on line L
    along the current path."""
    trie = RocodeTrie(prompt_lineno=0)
    # Line 1: tokens at depths 1, 2; line 2: depths 3, 4, 5.
    trie.add_token(token_id=11, lineno=1)
    trie.add_token(token_id=12, lineno=1)
    trie.add_token(token_id=21, lineno=2)
    trie.add_token(token_id=22, lineno=2)
    trie.add_token(token_id=23, lineno=2)
    assert trie.line_start_depth(1) == 1
    assert trie.line_start_depth(2) == 3
    # Line not on path: 0 (root).
    assert trie.line_start_depth(99) == 0


# ─── RocodeTrie: helpers ────────────────────────────────────────────────


def test_trie_path_token_ids_returns_root_to_cursor():
    trie = RocodeTrie()
    for tid in (5, 6, 7, 8):
        trie.add_token(token_id=tid)
    assert trie.path_token_ids() == [5, 6, 7, 8]
    trie.rollback_to(2)
    assert trie.path_token_ids() == [5, 6]


def test_trie_total_nodes_counts_all_subtrees():
    trie = RocodeTrie()
    trie.add_token(token_id=1)
    trie.add_token(token_id=2)
    # Diverge: roll back and try a different second token.
    trie.rollback_to(1)
    trie.add_token(token_id=3)
    # Root + 1 + (2,3) = 4 nodes
    assert trie.total_nodes() == 4


def test_trie_same_error_recurred_handles_none():
    """Helper static — None on either side means no recurrence."""
    assert RocodeTrie.same_error_recurred(None, 5) is False
    assert RocodeTrie.same_error_recurred(5, None) is False
    assert RocodeTrie.same_error_recurred(5, 5) is True
    assert RocodeTrie.same_error_recurred(5, 6) is False


def test_trie_mark_error_terminal_flags_cursor():
    trie = RocodeTrie()
    trie.add_token(token_id=1)
    trie.mark_error_terminal()
    assert trie.cursor.is_error_terminal is True


# ─── StrategicRollbackDecider: ROCODE Algorithm 1 ────────────────────────


def test_strategic_rollback_decider_uses_error_line_on_first_error():
    """First time we see an error on line L: rollback target = r_e =
    line-start depth of L (Eq. 4)."""
    trie = RocodeTrie()
    # Line 1 at depths 1-2, line 2 at depths 3-5, line 3 at depths 6-7.
    trie.add_token(token_id=1, lineno=1)
    trie.add_token(token_id=2, lineno=1)
    trie.add_token(token_id=3, lineno=2)
    trie.add_token(token_id=4, lineno=2)
    trie.add_token(token_id=5, lineno=2)
    trie.add_token(token_id=6, lineno=3)
    trie.add_token(token_id=7, lineno=3)

    decider = StrategicRollbackDecider(trie=trie)
    decision = decider.decide(diagnostics=[_Diag(line=2)])
    assert decision.used_recurrence is False
    assert decision.error_lineno == 2
    assert decision.rollback_pos == 3   # first token on line 2


def test_strategic_rollback_decider_uses_high_entropy_on_recurrence():
    """Same error line twice in a row → r_h (max-entropy depth, Eq. 6-7)."""
    trie = RocodeTrie()
    # Spread some entropies; the max is on line 1 (depth 2).
    trie.add_token(token_id=1, entropy=0.5, lineno=1)
    trie.add_token(token_id=2, entropy=3.0, lineno=1)   # max H on path
    trie.add_token(token_id=3, entropy=0.4, lineno=2)
    trie.add_token(token_id=4, entropy=0.6, lineno=2)
    trie.add_token(token_id=5, entropy=0.2, lineno=2)

    decider = StrategicRollbackDecider(trie=trie)
    # First call: not a recurrence (no history yet).
    d1 = decider.decide(diagnostics=[_Diag(line=2)])
    assert d1.used_recurrence is False
    assert d1.rollback_pos == 3   # r_e: line-start of line 2

    # Second call with same lineno: recurrence — must use r_h.
    d2 = decider.decide(diagnostics=[_Diag(line=2)])
    assert d2.used_recurrence is True
    # Highest-entropy depth on the current path is 2 (token 2's H = 3.0).
    assert d2.rollback_pos == 2


def test_strategic_rollback_decider_resets_recurrence_on_different_line():
    """Different lineno breaks the recurrence chain; the next-next call
    needs a same-lineno repeat to re-trigger r_h."""
    trie = RocodeTrie()
    trie.add_token(token_id=1, entropy=0.1, lineno=1)
    trie.add_token(token_id=2, entropy=2.0, lineno=1)
    trie.add_token(token_id=3, entropy=0.3, lineno=2)
    trie.add_token(token_id=4, entropy=0.4, lineno=3)

    decider = StrategicRollbackDecider(trie=trie)
    d1 = decider.decide(diagnostics=[_Diag(line=2)])
    d2 = decider.decide(diagnostics=[_Diag(line=3)])  # different line
    d3 = decider.decide(diagnostics=[_Diag(line=3)])  # now recurrence

    assert d1.used_recurrence is False
    assert d2.used_recurrence is False  # 3 != 2
    assert d3.used_recurrence is True   # 3 == 3


def test_strategic_rollback_decider_with_no_lineno_diagnostic_returns_root():
    """A diagnostic with no line info: fallback rollback to root."""
    trie = RocodeTrie()
    trie.add_token(token_id=1)
    decider = StrategicRollbackDecider(trie=trie)
    d = decider.decide(diagnostics=[_Diag(line=None)])
    assert d.rollback_pos == 0
    assert d.error_lineno is None


def test_strategic_rollback_decider_accepts_explicit_previous_diagnostics():
    """previous_diagnostics overrides the internal memory — useful for tests."""
    trie = RocodeTrie()
    trie.add_token(token_id=1, entropy=2.0, lineno=1)
    trie.add_token(token_id=2, entropy=0.5, lineno=2)
    decider = StrategicRollbackDecider(trie=trie)
    # Inject "previous lineno was 2" so the current call is a recurrence
    # even though decider's internal memory is empty.
    d = decider.decide(
        diagnostics=[_Diag(line=2)],
        previous_diagnostics=[_Diag(line=2)],
    )
    assert d.used_recurrence is True


# ─── RocodeTriePenaltyProcessor: __call__ modifies logits ───────────────


def test_rocode_trie_penalty_processor_modifies_logits_correctly():
    """Controlled input/output: after a penalty is applied to token X at
    position 2, calling the processor at step 2 (with prompt of len 0)
    must multiply logits[X] by the recorded penalty.
    """
    proc = RocodeTriePenaltyProcessor(lam=0.9)
    # Simulate one failed attempt that ended at depth 2 with token 42 there.
    proc.record_emitted_token(token_id=7, entropy=0.5)   # depth 1
    proc.record_emitted_token(token_id=42, entropy=0.2)  # depth 2
    # Roll back to depth 1 + penalize the abandoned suffix.
    penalized = proc.rollback_and_penalize(rollback_pos=1)
    assert penalized == 1   # only token 42 (depth 2) was penalized
    # Now token 42 at the cursor's child position has penalty 0.9^1 = 0.9.
    assert proc.trie.cursor.children[42].penalty_applied == pytest.approx(0.9)

    # Call the processor at the next decode step. token_ids length = 1
    # (one already-generated token, which is "token 7" living at depth 1
    # — the rollback didn't drop the trie node, but `token_ids` is the
    # decoder's input, so it has length 1 because we're about to sample
    # the next token at depth 2).
    token_ids = [7]
    logits = [1.0] * 100
    logits[42] = 5.0
    out = proc(token_ids, logits)
    # token 42 should be downweighted by 0.9.
    assert out[42] == pytest.approx(5.0 * 0.9)
    # token 7 (which was at depth 1, NOT in cursor's children anymore since
    # cursor is at depth 1 now and its children are token 42) is untouched.
    assert out[7] == pytest.approx(1.0)


def test_rocode_trie_penalty_processor_record_emitted_token_extends_trie():
    """record_emitted_token must move the trie cursor down one node per call."""
    proc = RocodeTriePenaltyProcessor()
    assert proc.trie.current_depth() == 0
    proc.record_emitted_token(token_id=10)
    assert proc.trie.current_depth() == 1
    proc.record_emitted_token(token_id=20)
    assert proc.trie.current_depth() == 2
    assert proc.trie.path_token_ids() == [10, 20]


def test_rocode_trie_penalty_processor_no_history_is_passthrough():
    """Before any penalty is applied, the processor is a no-op."""
    proc = RocodeTriePenaltyProcessor()
    logits_in = [1.0, 2.0, 3.0, 4.0]
    out = proc([0, 1], list(logits_in))
    assert out == logits_in


def test_rocode_trie_penalty_processor_reset_drops_history():
    """After reset(), no penalty fires even if the same step is re-played."""
    proc = RocodeTriePenaltyProcessor(lam=0.5)
    proc.record_emitted_token(token_id=5)
    proc.record_emitted_token(token_id=6)
    proc.rollback_and_penalize(rollback_pos=1)
    proc.reset()
    # Re-record the same prefix; with a fresh trie, no penalty applies.
    proc.record_emitted_token(token_id=5)
    logits = [1.0] * 10
    logits[6] = 9.0
    out = proc([5], list(logits))
    assert out[6] == pytest.approx(9.0)


def test_rocode_trie_penalty_processor_rejects_invalid_lambda():
    """lambda must be in (0, 1] — same contract as the MVP version."""
    with pytest.raises(ValueError):
        RocodeTriePenaltyProcessor(lam=0.0)
    with pytest.raises(ValueError):
        RocodeTriePenaltyProcessor(lam=1.1)


def test_rocode_trie_penalty_processor_set_prompt_len_offsets_position():
    """When prompt_len=3, only token_ids beyond index 3 count as generated."""
    proc = RocodeTriePenaltyProcessor(lam=0.9)
    proc.set_prompt_len(3)
    # Record one generated token at depth 1.
    proc.record_emitted_token(token_id=7)
    proc.record_emitted_token(token_id=42)
    proc.rollback_and_penalize(rollback_pos=1)

    # token_ids = prompt(3) + generated(1) = 4. Next pos in trie = 2.
    # Cursor depth is now 1. cursor.depth + 1 == 2. So a call should apply
    # the penalty to token 42 in cursor.children.
    token_ids = [99, 99, 99, 7]   # 3 prompt + 1 generated
    logits = [1.0] * 100
    logits[42] = 10.0
    out = proc(token_ids, list(logits))
    assert out[42] == pytest.approx(10.0 * 0.9)

    # token_ids drifted (no record_emitted_token between calls) — pass through.
    token_ids = [99, 99, 99, 7, 88, 88]
    logits = [1.0] * 100
    logits[42] = 10.0
    out = proc(token_ids, list(logits))
    assert out[42] == pytest.approx(10.0)  # unchanged


def test_rocode_trie_penalty_processor_distance_3_concrete_example():
    """Documentation sanity: λ=0.9, distance=3 → multiplier 0.729 (= 0.9^3).

    Build a chain of length 4 (depths 1..4), penalize from depth 1, then
    query the depth-4 node's penalty directly through the trie. The
    cursor is at depth 1 after rollback, but the trie subtree is
    preserved so we can walk it via children-lookup.
    """
    proc = RocodeTriePenaltyProcessor(lam=0.9)
    proc.record_emitted_token(token_id=1)   # depth 1
    proc.record_emitted_token(token_id=2)   # depth 2
    proc.record_emitted_token(token_id=3)   # depth 3
    proc.record_emitted_token(token_id=4)   # depth 4 — distance 3 from rb=1
    proc.rollback_and_penalize(rollback_pos=1)
    # Cursor is back at depth 1; walk the preserved subtree by children.
    node_d2 = proc.trie.cursor.children[2]
    node_d3 = node_d2.children[3]
    node_d4 = node_d3.children[4]
    assert node_d4.penalty_applied == pytest.approx(0.729)
    assert math.isclose(node_d4.penalty_applied, 0.9 ** 3)
    # And the intermediate distances:
    assert node_d2.penalty_applied == pytest.approx(0.9)
    assert node_d3.penalty_applied == pytest.approx(0.81)


# ─── algorithm dispatch contract — what the main thread will integrate ──


def test_rocode_trie_penalty_processor_conforms_to_logits_processor_protocol():
    """Same Protocol used by ALGORITHM_DISPATCH in soundcode/web/server.py."""
    from soundcode.logits_processors import LogitsProcessor
    p = RocodeTriePenaltyProcessor()
    assert isinstance(p, LogitsProcessor)
