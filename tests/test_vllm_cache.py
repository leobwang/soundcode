"""Tests for vLLM backend KV-cache reuse (prefix caching + rollback resume).

Plumbing-only — no test in this file loads a real model. Tests that require
GPU + model weights are marked `@pytest.mark.slow` and skipped by default;
run them with `uv run pytest -m slow tests/test_vllm_cache.py`.

The cache-reuse story relies on vLLM's automatic prefix caching: when a
rollback truncates the prompt to a shared prefix and we resubmit, PagedAttention
reuses the KV blocks for the common prefix without paying the prefill cost.
The unit tests below cover the plumbing — that prefix caching is enabled,
that `rollback_and_resume` aborts the in-flight request before resubmitting,
and that the stats surface exposes the right fields.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from soundcode.llm import GenerationConfig, Token
from soundcode.vllm_backend import BackendStats, VllmBackend


# ─── prefix caching enabled by default ─────────────────────────────────


def test_prefix_caching_enabled_by_default():
    """Constructing `VllmBackend` with default args must set
    `enable_prefix_caching=True` on the AsyncEngineArgs. This is the headline
    optimization that makes rollback-and-resume cheap."""
    backend = VllmBackend(model="dummy", config=GenerationConfig())
    assert backend._enable_prefix_caching is True


def test_prefix_caching_overridable_via_constructor():
    """Caller can disable prefix caching (e.g. for an apples-to-apples
    benchmark) by passing `enable_prefix_caching=False`."""
    backend = VllmBackend(
        model="dummy",
        config=GenerationConfig(),
        enable_prefix_caching=False,
    )
    assert backend._enable_prefix_caching is False


def test_prefix_caching_flag_threaded_into_engine_args():
    """Verify `_build_engine` forwards the flag into `AsyncEngineArgs`. We
    patch the vLLM imports so no model is loaded — we just inspect the
    kwargs the constructor received."""
    backend = VllmBackend(model="dummy", config=GenerationConfig())
    fake_engine = MagicMock()
    fake_args_cls = MagicMock()
    fake_engine_cls = MagicMock()
    fake_engine_cls.from_engine_args.return_value = fake_engine

    with patch.dict(
        "sys.modules",
        {"vllm": MagicMock(
            AsyncEngineArgs=fake_args_cls,
            AsyncLLMEngine=fake_engine_cls,
        )},
    ):
        backend._build_engine()

    # AsyncEngineArgs(...) was called once with enable_prefix_caching=True.
    assert fake_args_cls.call_count == 1
    _, kwargs = fake_args_cls.call_args
    assert kwargs.get("enable_prefix_caching") is True
    assert kwargs.get("model") == "dummy"
    # And the engine was built from those args.
    fake_engine_cls.from_engine_args.assert_called_once()
    assert backend._engine is fake_engine


# ─── rollback_and_resume aborts the in-flight request ──────────────────


def test_rollback_and_resume_aborts_in_flight_request():
    """When `rollback_and_resume` is called with a request already in flight,
    the existing request_id must be passed to `engine.abort` before the new
    request is submitted. This is what releases the dead-end KV blocks so
    the new request can reuse them."""
    backend = VllmBackend(model="dummy", config=GenerationConfig())
    # Simulate "request in flight": engine present, request_id set.
    fake_engine = MagicMock()
    fake_engine.abort = MagicMock(return_value=None)

    # `generate()` returns an async iterator that produces nothing — just a
    # finished output so the resumed stream terminates cleanly.
    async def _empty_generate(prompt, params, req_id):
        # Yield one finished output with no text → stream ends immediately.
        out = MagicMock()
        out.outputs = []
        out.finished = True
        out.prompt_token_ids = [1, 2, 3, 4, 5]
        out.num_cached_tokens = 3
        yield out

    fake_engine.generate = _empty_generate
    backend._engine = fake_engine
    backend._current_request_id = "soundcode-INFLIGHT"

    async def _drive():
        agen = backend.rollback_and_resume("partial prompt")
        async for _ in agen:
            pass

    asyncio.run(_drive())

    # The in-flight request must have been aborted.
    fake_engine.abort.assert_called_once_with("soundcode-INFLIGHT")
    # And the new request_id must have a rollback marker so logs can tell
    # them apart from fresh requests.
    assert backend.last_stats.last_request_id is not None
    assert backend.last_stats.last_request_id.startswith("soundcode-rb-")
    assert backend.last_stats.is_rollback is True


def test_rollback_and_resume_handles_no_inflight_request():
    """`rollback_and_resume` must not crash when there's no in-flight
    request (e.g. first call after a clean stream end). abort() shouldn't
    be invoked in that case."""
    backend = VllmBackend(model="dummy", config=GenerationConfig())
    fake_engine = MagicMock()
    fake_engine.abort = MagicMock()

    async def _empty_generate(prompt, params, req_id):
        out = MagicMock()
        out.outputs = []
        out.finished = True
        out.prompt_token_ids = None
        out.num_cached_tokens = None
        yield out

    fake_engine.generate = _empty_generate
    backend._engine = fake_engine
    backend._current_request_id = None  # no in-flight request

    async def _drive():
        async for _ in backend.rollback_and_resume("fresh prompt"):
            pass

    asyncio.run(_drive())
    # No prior request → no abort.
    fake_engine.abort.assert_not_called()


def test_rollback_and_resume_request_id_override():
    """When the caller passes `request_id=...`, that exact id is used (not a
    new uuid). Useful for log correlation in the demo overlay."""
    backend = VllmBackend(model="dummy", config=GenerationConfig())
    fake_engine = MagicMock()
    fake_engine.abort = MagicMock()

    captured_req_id = []

    async def _capturing_generate(prompt, params, req_id):
        captured_req_id.append(req_id)
        out = MagicMock()
        out.outputs = []
        out.finished = True
        out.prompt_token_ids = None
        out.num_cached_tokens = None
        yield out

    fake_engine.generate = _capturing_generate
    backend._engine = fake_engine

    async def _drive():
        async for _ in backend.rollback_and_resume(
            "p", request_id="my-custom-id",
        ):
            pass

    asyncio.run(_drive())
    assert captured_req_id == ["my-custom-id"]


# ─── BackendStats dataclass shape ──────────────────────────────────────


def test_backend_stats_includes_cache_metrics():
    """`BackendStats` must expose the fields the web demo overlay reads:
    prefix_tokens, cached_tokens, ttft_s, is_rollback, last_request_id."""
    stats = BackendStats()
    # Default-constructible (so DemoClient can read it before any stream
    # has opened).
    assert stats.prefix_tokens == 0
    assert stats.cached_tokens == 0
    assert stats.ttft_s is None
    assert stats.is_rollback is False
    assert stats.last_request_id is None
    # Hit rate is a derived property — 0.0 when prefix is empty.
    assert stats.cache_hit_rate == 0.0

    # Populated case — hit rate is cached_tokens / prefix_tokens, clamped.
    s2 = BackendStats(prefix_tokens=200, cached_tokens=150, ttft_s=0.012,
                     is_rollback=True, last_request_id="req-A")
    assert s2.cache_hit_rate == pytest.approx(0.75)
    assert s2.is_rollback is True
    assert s2.last_request_id == "req-A"


def test_backend_stats_cache_hit_rate_clamps_overflow():
    """If a buggy engine ever reports cached_tokens > prefix_tokens, the
    hit-rate must clamp to 1.0 rather than blow past it (cosmetic guard for
    the demo UI percentage gauge)."""
    s = BackendStats(prefix_tokens=100, cached_tokens=120)
    assert s.cache_hit_rate == 1.0


def test_backend_stats_exposed_on_fresh_backend():
    """A freshly-constructed backend must already have `.last_stats` set so
    callers can read it without any prior stream open."""
    backend = VllmBackend(model="dummy", config=GenerationConfig())
    assert hasattr(backend, "last_stats")
    assert isinstance(backend.last_stats, BackendStats)
    assert backend.last_stats.ttft_s is None


# ─── stats are populated when a stream runs ────────────────────────────


def test_stream_populates_stats_from_request_output():
    """When `_stream_request` runs against a mocked engine that emits
    `prompt_token_ids` and `num_cached_tokens` on its RequestOutputs, those
    values must land in `backend.last_stats`."""
    backend = VllmBackend(model="dummy", config=GenerationConfig())
    fake_engine = MagicMock()

    async def _gen_with_metrics(prompt, params, req_id):
        # First output: cumulative tokenized prompt + cached hit + 0 chars.
        out1 = MagicMock()
        out1.prompt_token_ids = [10, 20, 30, 40]
        out1.num_cached_tokens = 3
        out1.outputs = []
        out1.finished = False
        yield out1
        # Second output: one new token decoded.
        out2 = MagicMock()
        out2.prompt_token_ids = [10, 20, 30, 40]
        out2.num_cached_tokens = 3
        completion = MagicMock()
        completion.text = "hello"
        out2.outputs = [completion]
        out2.finished = True
        yield out2

    fake_engine.generate = _gen_with_metrics
    fake_engine.abort = MagicMock()
    backend._engine = fake_engine
    backend.set_prompt("seed prompt")

    async def _drive():
        # Pump until exhausted.
        tokens = []
        while backend.has_next():
            tok = await backend.next()
            if not tok.text:
                break
            tokens.append(tok)
        return tokens

    toks = asyncio.run(_drive())
    assert [t.text for t in toks] == ["hello"]
    assert backend.last_stats.prefix_tokens == 4
    assert backend.last_stats.cached_tokens == 3
    assert backend.last_stats.cache_hit_rate == pytest.approx(0.75)
    assert backend.last_stats.ttft_s is not None
    assert backend.last_stats.ttft_s >= 0.0
    # First stream isn't a rollback — set_prompt path.
    assert backend.last_stats.is_rollback is False


def test_rollback_and_resume_marks_is_rollback_true():
    """A stream opened via `rollback_and_resume` must report
    `last_stats.is_rollback == True` so the demo UI can attribute the TTFT
    savings to cache reuse."""
    backend = VllmBackend(model="dummy", config=GenerationConfig())
    fake_engine = MagicMock()

    async def _gen(prompt, params, req_id):
        out = MagicMock()
        out.prompt_token_ids = [1, 2, 3]
        out.num_cached_tokens = 3  # full hit — rollback to identical prefix
        completion = MagicMock()
        completion.text = "x"
        out.outputs = [completion]
        out.finished = True
        yield out

    fake_engine.generate = _gen
    fake_engine.abort = MagicMock()
    backend._engine = fake_engine

    async def _drive():
        async for _ in backend.rollback_and_resume("prompt"):
            pass

    asyncio.run(_drive())
    assert backend.last_stats.is_rollback is True
    assert backend.last_stats.cache_hit_rate == pytest.approx(1.0)


# ─── optional: real model load + microbenchmark ────────────────────────


@pytest.mark.slow
def test_prefix_cache_reuse_on_real_model():
    """End-to-end smoke: load a small coder model, generate once, then
    rollback to the prefix. The second TTFT should be smaller than the
    first AND `cached_tokens` should be > 0 on the second run.

    Skipped unless `-m slow` is passed. Costs ~10 GB VRAM + ~30s walltime
    for a 1.5B model.
    """
    import os
    from pathlib import Path

    model = os.environ.get(
        "SOUNDCODE_VLLM_TEST_MODEL",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    )
    if model.startswith("/") and not Path(model).exists():
        pytest.skip(f"model path {model} not present")

    async def _run():
        backend = VllmBackend(
            model=model,
            config=GenerationConfig(max_tokens=16),
            gpu_memory_utilization=0.15,
            max_model_len=1024,
        )
        try:
            await backend.warmup()
            prompt = "fn add(a: i32, b: i32) -> i32 {"
            # First run: cold cache.
            backend.set_prompt(prompt)
            for _ in range(20):
                if not backend.has_next():
                    break
                t = await backend.next()
                if not t.text:
                    break
            ttft_1 = backend.last_stats.ttft_s
            assert ttft_1 is not None
            # Second run: rollback to same prefix — should be a cache hit.
            async for _ in backend.rollback_and_resume(prompt):
                pass
            ttft_2 = backend.last_stats.ttft_s
            assert ttft_2 is not None
            assert backend.last_stats.is_rollback is True
            # Cache hit indicator should be non-zero — vLLM should report at
            # least the block-aligned prefix as cached.
            assert backend.last_stats.cached_tokens > 0, (
                f"expected cached_tokens > 0 after rollback, "
                f"got stats={backend.last_stats}"
            )
        finally:
            await backend.close()

    asyncio.run(_run())
