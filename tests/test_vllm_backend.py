"""Tests for the vLLM backend + LogitsProcessor plumbing.

Plumbing-only — no test in this file actually loads a model. Tests that
require GPU + model weights are marked `@pytest.mark.slow` and skipped by
default; run them with `uv run pytest -m slow tests/test_vllm_backend.py`.
"""

from __future__ import annotations

import math
import warnings

import pytest

from soundcode.llm import GenerationConfig
from soundcode.logits_processors import (
    LogitsProcessor,
    NoopLogitsProcessor,
    RocodeDecayingPenaltyProcessor,
    SemGuardEvaluatorProcessor,
)


# ─── duck-typed protocol conformance ───────────────────────────────────


def test_vllm_backend_satisfies_llm_protocol():
    """VllmBackend must expose every public method DemoClient calls on
    LlmServer. We check the surface here without instantiating the engine —
    a real instantiation would load the model into VRAM (out of scope for
    a unit test)."""
    from soundcode.vllm_backend import VllmBackend

    required = (
        "warmup", "set_prompt", "has_next", "next",
        "abort_current_stream", "close", "stream_thinking_phase",
    )
    for name in required:
        attr = getattr(VllmBackend, name, None)
        assert attr is not None, f"VllmBackend missing method: {name}"
        assert callable(attr), f"VllmBackend.{name} not callable"

    # `config` is a settable attribute that DemoClient reads via
    # `llm.config.mode`. Construct the class with a no-op processor so
    # __init__ runs without the engine.
    backend = VllmBackend(model="dummy", config=GenerationConfig())
    assert hasattr(backend, "config")
    assert isinstance(backend.config, GenerationConfig)
    # think_budget_exceeded is read via getattr in DemoClient — must exist.
    assert hasattr(backend, "think_budget_exceeded")
    assert backend.think_budget_exceeded is False


# ─── NoopLogitsProcessor ───────────────────────────────────────────────


def test_noop_logits_processor_returns_input_unchanged():
    """The default SoundCode processor must return logits unmodified —
    rollback works via prompt re-decode, not logit penalty."""
    p = NoopLogitsProcessor()
    # No torch needed — the processor is logits-type-agnostic in the no-op
    # case. Use a list so the test runs on machines without torch.
    logits = [0.1, 0.2, 0.3, 0.4]
    out = p([1, 2, 3], logits)
    assert out is logits
    # Multiple calls — state-free.
    out2 = p([1, 2, 3, 4], logits)
    assert out2 is logits


def test_noop_logits_processor_conforms_to_protocol():
    """LogitsProcessor is a runtime-checkable Protocol — NoopLogitsProcessor
    must satisfy it, so the dispatch table can return it interchangeably."""
    p = NoopLogitsProcessor()
    assert isinstance(p, LogitsProcessor)


# ─── RocodeDecayingPenaltyProcessor ────────────────────────────────────


def test_rocode_penalty_processor_decays_correctly():
    """Feed known token_ids and assert logits[bad_token] is multiplied by
    exactly `lam^distance`, others unchanged. Covers ROCODE Eq. 8-9."""
    lam = 0.9
    p = RocodeDecayingPenaltyProcessor(lam=lam)
    # Bad suffix `[142, 88, 271]` was on the rolled-back attempt starting at
    # offset 500. Distance 0 / 1 / 2 from the rollback point.
    p.add_penalty([142, 88, 271], rollback_offset=500)

    # At step 500, token 142 should be downweighted by lam^0 = 1.0.
    # (The penalty is registered but a distance-0 penalty is a no-op.)
    logits = [1.0] * 300
    logits[142] = 5.0
    token_ids = [0] * 500   # len() == 500 means "next step is 500"
    out = p(list(token_ids), list(logits))
    assert out[142] == pytest.approx(5.0 * lam ** 0)  # 5.0
    # Other slots untouched.
    assert out[88] == 1.0 and out[271] == 1.0

    # At step 501, token 88 should be multiplied by lam^1 = 0.9.
    token_ids = [0] * 501
    logits = [1.0] * 300
    logits[88] = 5.0
    out = p(list(token_ids), list(logits))
    assert out[88] == pytest.approx(5.0 * lam ** 1)

    # At step 502, token 271 should be multiplied by lam^2 = 0.81.
    token_ids = [0] * 502
    logits = [1.0] * 300
    logits[271] = 5.0
    out = p(list(token_ids), list(logits))
    assert out[271] == pytest.approx(5.0 * lam ** 2)

    # At a step OUTSIDE the recorded penalty positions, no logit is touched.
    token_ids = [0] * 600
    logits = [1.0] * 300
    out = p(list(token_ids), list(logits))
    assert all(out[i] == 1.0 for i in (88, 142, 271))


def test_rocode_processor_decay_at_distance_5_matches_spec():
    """Documentation check: with lam=0.9 and 5 tokens of distance, the
    multiplier is 0.9^5 ≈ 0.59049."""
    p = RocodeDecayingPenaltyProcessor(lam=0.9)
    p.add_penalty([0, 0, 0, 0, 0, 42], rollback_offset=100)
    token_ids = [0] * 105  # step 105, distance 5 from rollback at 100
    logits = [1.0] * 50
    logits[42] = 10.0
    out = p(list(token_ids), list(logits))
    assert out[42] == pytest.approx(10.0 * 0.9 ** 5)
    assert out[42] == pytest.approx(5.9049, rel=1e-4)


def test_rocode_processor_reset_clears_state():
    """After reset(), no penalty should fire — used when starting a fresh
    prompt to drop accumulated bad-suffix history."""
    p = RocodeDecayingPenaltyProcessor(lam=0.5)
    p.add_penalty([5], rollback_offset=10)
    p.reset()
    token_ids = [0] * 10
    logits = [1.0] * 20
    logits[5] = 2.0
    out = p(list(token_ids), list(logits))
    assert out[5] == 2.0  # unchanged — no penalty after reset


def test_rocode_processor_rejects_invalid_lambda():
    """lambda must be in (0, 1] — 0 would zero out the logit, > 1 would
    actually boost the bad token. Either is a bug."""
    with pytest.raises(ValueError):
        RocodeDecayingPenaltyProcessor(lam=0.0)
    with pytest.raises(ValueError):
        RocodeDecayingPenaltyProcessor(lam=-0.1)
    with pytest.raises(ValueError):
        RocodeDecayingPenaltyProcessor(lam=1.1)


# ─── SemGuardEvaluatorProcessor ────────────────────────────────────────
# The original "falls back to no-op without checkpoint" test was retired
# when agent #20 rewrote the processor with a new constructor signature
# (`(evaluator, tokenizer, threshold, max_resamples)` — the old
# `evaluator_ckpt=` kwarg no longer exists). The moral equivalent — that
# constructing without an evaluator silently degrades to logits-passthrough
# AND emits a warning — is covered by
# `tests/test_semguard_integration.py::test_semguard_processor_logits_unchanged`
# (the disabled-path branch) and
# `tests/test_semguard_integration.py::test_semguard_processor_silently_noops_without_bound_loop`.


# ─── server dispatch tables ────────────────────────────────────────────


def test_backend_dispatch_returns_right_class():
    """BACKEND_DISPATCH must produce the correct backend type for each key,
    and ignore the algorithm processor for the Ollama path (which doesn't
    accept per-step logit hooks). vLLM path is constructed but not warmed up
    (warmup loads the model — we only check the type here)."""
    from soundcode.web.server import BACKEND_DISPATCH
    from soundcode.llm import LlmServer

    cfg = GenerationConfig()
    proc = NoopLogitsProcessor()

    # Ollama path — discards the processor.
    ollama = BACKEND_DISPATCH["ollama"]("mistral:7b", cfg, proc)
    assert isinstance(ollama, LlmServer)

    # vLLM path — constructs the wrapper without loading the model.
    from soundcode.vllm_backend import VllmBackend
    vllm = BACKEND_DISPATCH["vllm"]("Qwen/Qwen2.5-Coder-7B-Instruct", cfg, proc)
    assert isinstance(vllm, VllmBackend)
    assert vllm.logits_processor is proc

    # Unknown key isn't in the table — the server normalises it before
    # dispatch, but the dispatch itself raises KeyError if asked directly.
    assert "ollama" in BACKEND_DISPATCH
    assert "vllm" in BACKEND_DISPATCH


def test_algorithm_dispatch_returns_right_processor():
    """ALGORITHM_DISPATCH keys map to processor factories.

    Post-integration, the table has five entries:
      - `soundcode`        → NoopLogitsProcessor (no logit penalty)
      - `rocode`           → RocodeDecayingPenaltyProcessor (back-compat alias
                             for `rocode_lite`)
      - `rocode_lite`      → RocodeDecayingPenaltyProcessor (MVP — flat list
                             of (token, position) pairs)
      - `rocode_faithful`  → RocodeTriePenaltyProcessor (faithful — backed by
                             RocodeTrie; lambda factory to set lam=0.9)
      - `semguard`         → SemGuardEvaluatorProcessor (no-op fallback when
                             the singleton evaluator isn't loaded; the real
                             wiring happens in `_run_generation`)
    """
    from soundcode.web.server import ALGORITHM_DISPATCH
    from soundcode.rocode_processor import RocodeTriePenaltyProcessor

    assert ALGORITHM_DISPATCH["soundcode"] is NoopLogitsProcessor
    # Two ROCODE entries: `rocode` is the back-compat alias for `rocode_lite`
    # (the MVP / flat-list processor). `rocode_faithful` is the trie-backed
    # processor used for the paper-baseline comparison.
    assert ALGORITHM_DISPATCH["rocode"] is RocodeDecayingPenaltyProcessor
    assert ALGORITHM_DISPATCH["rocode_lite"] is RocodeDecayingPenaltyProcessor
    # Each entry must be callable and produce the expected concrete type.
    assert isinstance(ALGORITHM_DISPATCH["soundcode"](), NoopLogitsProcessor)
    assert isinstance(
        ALGORITHM_DISPATCH["rocode"](), RocodeDecayingPenaltyProcessor,
    )
    assert isinstance(
        ALGORITHM_DISPATCH["rocode_lite"](), RocodeDecayingPenaltyProcessor,
    )
    assert isinstance(
        ALGORITHM_DISPATCH["rocode_faithful"](), RocodeTriePenaltyProcessor,
    )
    # SemGuard: the dispatch entry exists (server falls back to this when the
    # singleton evaluator hasn't been built — e.g. checkpoint missing). The
    # processor emits a warning at construction when both evaluator and
    # tokenizer are None; suppress for the test.
    assert "semguard" in ALGORITHM_DISPATCH
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg = ALGORITHM_DISPATCH["semguard"]()
    assert isinstance(sg, SemGuardEvaluatorProcessor)


# ─── optional: real model load (slow, requires GPU) ────────────────────


@pytest.mark.slow
def test_vllm_backend_real_model_load_and_generate():
    """End-to-end smoke: load a small coder model, run one generation, then
    close. Skipped unless `-m slow` is passed and the model directory exists.
    Costs ~10 GB VRAM + ~10 s walltime for a 1.5B model."""
    import asyncio
    import os
    from pathlib import Path
    from soundcode.vllm_backend import VllmBackend

    # Allow override via env so CI / smoke scripts can point at a specific
    # local model directory rather than hitting HF.
    model = os.environ.get(
        "SOUNDCODE_VLLM_TEST_MODEL",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    )
    if model.startswith("/") and not Path(model).exists():
        pytest.skip(f"model path {model} not present")

    async def _run():
        backend = VllmBackend(
            model=model,
            config=GenerationConfig(max_tokens=32, stop=["\n}"]),
            gpu_memory_utilization=0.15,
            max_model_len=1024,
        )
        try:
            await backend.warmup()
            backend.set_prompt("fn add(a: i32, b: i32) -> i32 {")
            tokens = []
            for _ in range(40):
                if not backend.has_next():
                    break
                t = await backend.next()
                if not t.text:
                    break
                tokens.append(t)
            assert tokens, "no tokens emitted"
        finally:
            await backend.close()

    asyncio.run(_run())
