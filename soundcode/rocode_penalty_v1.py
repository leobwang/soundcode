"""vLLM V1 engine-side ROCODE token-penalty processor.

vLLM's V1 engine runs in a subprocess, so a host-side stateful
LogitsProcessor cannot be shared by reference (the historic reason the
penalty arms were unwired no-ops — see notes/vllm-backend-followups.md).
The V1-native mechanism is an engine-registered processor plus per-request
state via ``SamplingParams.extra_args``:

* The HOST (scripts/paper_phase1.py, ``rollback="penalty"``) maintains the
  ROCODE trie (soundcode.rocode_processor.RocodeTriePenaltyProcessor),
  feeds it emitted tokens, applies ``rollback_and_penalize`` on rollback,
  and serializes the cursor's subtree into
  ``extra_args={"rocode_trie": ...}`` for the retry request.
* This adapter deserializes the subtree per request and walks it against
  the tokens generated so far; whenever the generation is still on a
  previously-attempted path, the children's ``penalty_applied``
  multipliers are applied to the logits — the same logit-space multiplier
  form as ``RocodeTriePenaltyProcessor.__call__`` (documented deviation
  from Eq. 9's probability-space algebra).

Serialized node format: ``{"c": {str(token_id): [penalty, child_node]}}``.
The root corresponds to the retry's start position.
"""

from __future__ import annotations

from typing import Any

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


def serialize_trie(node: Any, max_nodes: int = 2048) -> dict:
    """Serialize a soundcode.rocode_trie.TrieNode subtree for extra_args.

    Only subtrees containing at least one penalized node are kept — an
    unpenalized branch can never influence logits.
    """
    budget = [max_nodes]

    def _ser(n: Any) -> dict:
        out: dict = {"c": {}}
        for tid, child in getattr(n, "children", {}).items():
            if budget[0] <= 0:
                break
            sub = _ser(child)
            if child.penalty_applied != 1.0 or sub["c"]:
                budget[0] -= 1
                out["c"][str(tid)] = [float(child.penalty_applied), sub]
        return out

    return _ser(node)


class RocodePenaltyV1(AdapterLogitsProcessor):
    """Engine-side adapter: walks the host-serialized penalty trie."""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        extra = getattr(params, "extra_args", None) or {}
        trie = extra.get("rocode_trie")
        if not trie or not trie.get("c"):
            return None

        state: dict = {"node": trie, "consumed": 0}

        def proc(output_ids: list[int], logits):
            node = state["node"]
            i = state["consumed"]
            # Advance the cursor along tokens generated since last step.
            # Once the generation diverges from every previously-attempted
            # path, node becomes None and stays None (no penalties apply).
            while node is not None and i < len(output_ids):
                hit = node.get("c", {}).get(str(output_ids[i]))
                node = hit[1] if hit is not None else None
                i += 1
            state["node"] = node
            state["consumed"] = len(output_ids)
            if node is not None:
                for tid_s, (pen, _child) in node.get("c", {}).items():
                    if pen != 1.0:
                        tid = int(tid_s)
                        logits[tid] = logits[tid] * pen
            return logits

        return proc
