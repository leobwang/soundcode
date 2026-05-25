"""Stability tests for the vLLM backend.

Covers:
  - model load / unload + GPU-memory delta sanity
  - repeated load/unload cycles (catches gradual leaks)
  - long generation OOM headroom
  - mid-stream abort + resume on the same engine
  - clear error on bad model name (cheap, no GPU)
  - KV-cache correctness: prefix-cache equivalence to full prefill
  - KV-cache correctness: rollback_and_resume vs fresh generate

Most tests need GPU + model weights and are marked `slow+gpu`. They are
skipped by default. Run with:

    RUN_GPU_TESTS=1 uv run pytest tests/test_vllm_stability.py -v

The single non-GPU test (`test_vllm_backend_raises_clearly_on_bad_model_name`)
runs under the default invocation as a smoke test of error propagation.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any

import pytest

from soundcode.llm import GenerationConfig


# ─── markers + helpers ─────────────────────────────────────────────────


def _gpu_tests_enabled() -> bool:
    return os.environ.get("RUN_GPU_TESTS", "") == "1"


gpu_required = pytest.mark.skipif(
    not _gpu_tests_enabled(),
    reason="set RUN_GPU_TESTS=1 to run GPU-bound vLLM stability tests",
)


# Resolve the test model once. Allow override via env var so CI / smoke
# scripts can point at a smaller local model (e.g. 1.5B) without editing
# this file. Default is the MVP's 7B coder model.
TEST_MODEL = os.environ.get(
    "SOUNDCODE_VLLM_TEST_MODEL",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
)


def _nvidia_smi_used_mib() -> int:
    """Read total used memory across all GPUs (in MiB). Returns 0 if
    nvidia-smi is unavailable — caller must handle that (we skip the
    leak-check assertions in that case)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return sum(int(line) for line in out.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return 0


def _nvidia_smi_available() -> bool:
    try:
        subprocess.check_output(
            ["nvidia-smi", "--version"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


# ─── test_vllm_backend_raises_clearly_on_bad_model_name (cheap) ────────


@pytest.mark.asyncio
async def test_vllm_backend_raises_clearly_on_bad_model_name():
    """Passing a nonexistent model name should raise an exception with a
    recognisable error type — not crash with a cryptic deep-stack error
    from inside the engine threadpool.

    No GPU required: vLLM tries to resolve the model at engine-build time
    (warmup), and fails fast when the path / HF id doesn't exist. We just
    check that the failure surfaces as an exception the caller can catch.
    """
    from soundcode.vllm_backend import VllmBackend

    backend = VllmBackend(
        model="Nonexistent/DoesNotExistModel-12345",
        config=GenerationConfig(),
        gpu_memory_utilization=0.05,
        max_model_len=512,
    )
    # warmup() is the lazy load path. Any exception type is acceptable as
    # long as one is raised — the contract is "fail loudly, don't hang."
    with pytest.raises(Exception) as ei:
        await backend.warmup()
    # The error message should mention something model-resolution-shaped:
    # a path, a 404, "not found", or the model name itself. Looking for
    # *any* of these guards against the deep-stack cryptic case.
    msg = repr(ei.value).lower()
    likely_keywords = (
        "nonexistent", "not found", "no such", "does not exist",
        "404", "model", "huggingface", "hub", "repository", "repo",
    )
    assert any(k in msg for k in likely_keywords), (
        f"error message does not look model-resolution-shaped: {ei.value!r}"
    )
    # Defensive cleanup — close() on a never-initialised engine should be
    # a no-op (don't burn the next test on stale state).
    await backend.close()


# ─── test_vllm_backend_loads_and_unloads_without_leak (slow+gpu) ──────


@pytest.mark.slow
@pytest.mark.gpu
@gpu_required
@pytest.mark.asyncio
async def test_vllm_backend_loads_and_unloads_without_leak():
    """Load a 7B model, capture nvidia-smi mem; close; capture again; assert
    delta < 1 GB. Catches a missing engine.shutdown() / lingering CUDA
    context after close()."""
    if not _nvidia_smi_available():
        pytest.skip("nvidia-smi not on PATH — cannot measure VRAM delta")
    from soundcode.vllm_backend import VllmBackend

    baseline_mib = _nvidia_smi_used_mib()
    backend = VllmBackend(
        model=TEST_MODEL,
        config=GenerationConfig(max_tokens=8),
        gpu_memory_utilization=0.20,
        max_model_len=512,
    )
    try:
        await backend.warmup()
        # Verify the engine actually loaded — otherwise leak detection is
        # vacuous (no memory was acquired in the first place).
        after_load_mib = _nvidia_smi_used_mib()
        assert after_load_mib > baseline_mib + 1000, (
            f"engine warmup did not commit GPU memory: "
            f"baseline={baseline_mib} MiB after_load={after_load_mib} MiB"
        )
    finally:
        await backend.close()
    # Give the runtime a beat to release CUDA context buffers.
    await asyncio.sleep(2.0)
    after_close_mib = _nvidia_smi_used_mib()
    leaked_mib = after_close_mib - baseline_mib
    # < 1 GB tolerance covers CUDA context residue + measurement jitter.
    assert leaked_mib < 1024, (
        f"VRAM leak: baseline={baseline_mib} MiB, after_close={after_close_mib} MiB, "
        f"leaked={leaked_mib} MiB (>= 1024 MiB)"
    )


# ─── test_vllm_backend_repeated_load_unload_cycles (slow+gpu) ─────────


@pytest.mark.slow
@pytest.mark.gpu
@gpu_required
@pytest.mark.asyncio
async def test_vllm_backend_repeated_load_unload_cycles():
    """Load + close the backend 3 times; assert each cycle returns to within
    the baseline + 1 GB. Catches gradual leaks that a single-cycle test
    wouldn't notice."""
    if not _nvidia_smi_available():
        pytest.skip("nvidia-smi not on PATH — cannot measure VRAM delta")
    from soundcode.vllm_backend import VllmBackend

    baseline_mib = _nvidia_smi_used_mib()
    leak_samples: list[int] = []
    for cycle in range(3):
        backend = VllmBackend(
            model=TEST_MODEL,
            config=GenerationConfig(max_tokens=8),
            gpu_memory_utilization=0.20,
            max_model_len=512,
        )
        try:
            await backend.warmup()
        finally:
            await backend.close()
        await asyncio.sleep(2.0)
        after_close_mib = _nvidia_smi_used_mib()
        delta = after_close_mib - baseline_mib
        leak_samples.append(delta)
        # Per-cycle leak must stay bounded.
        assert delta < 1024, (
            f"cycle {cycle}: VRAM leaked {delta} MiB "
            f"(baseline={baseline_mib}, after_close={after_close_mib})"
        )
    # Final cycle should not be dramatically worse than the first — guards
    # against linear-in-cycle leaks that the per-cycle threshold tolerates.
    assert leak_samples[-1] - leak_samples[0] < 512, (
        f"gradual leak: cycle deltas {leak_samples}"
    )


# ─── test_vllm_backend_long_generation_no_oom (slow+gpu) ──────────────


@pytest.mark.slow
@pytest.mark.gpu
@gpu_required
@pytest.mark.asyncio
async def test_vllm_backend_long_generation_no_oom():
    """Generate up to ~4K tokens in one call; assert no exception is raised
    and that some tokens were produced. Uses a generic prompt that won't
    naturally stop early."""
    from soundcode.vllm_backend import VllmBackend

    backend = VllmBackend(
        model=TEST_MODEL,
        # Drop stop patterns to avoid early termination — we want to push the
        # generation length, not test the stop list.
        config=GenerationConfig(max_tokens=4096, stop=[]),
        gpu_memory_utilization=0.25,
        max_model_len=8192,
    )
    try:
        await backend.warmup()
        # An open-ended prompt that the model will happily continue at length.
        backend.set_prompt(
            "// Implement a comprehensive linked-list utility module in Rust.\n"
            "// Include push, pop, insert, remove, search, reverse, and length.\n"
            "// Document every function with a short doc comment.\n"
            "mod list {\n"
        )
        total_chars = 0
        token_count = 0
        # Cap iterations defensively — even at 1 char/token this can't run
        # forever, but better safe.
        for _ in range(8192):
            if not backend.has_next():
                break
            tok = await backend.next()
            if not tok.text:
                break
            total_chars += len(tok.text)
            token_count += 1
        assert token_count > 0, "no tokens emitted before EOS"
        # We don't enforce a hard minimum length (the model might stop
        # earlier on its own EOS) — the key assertion is "no OOM exception."
    finally:
        await backend.close()


# ─── test_vllm_backend_handles_engine_abort_during_generation ─────────


@pytest.mark.slow
@pytest.mark.gpu
@gpu_required
@pytest.mark.asyncio
async def test_vllm_backend_handles_engine_abort_during_generation():
    """Start a generation; abort mid-way; verify the engine is still usable
    for a subsequent generate. This is the rollback path: if the engine
    becomes wedged after an abort, every rollback breaks."""
    from soundcode.vllm_backend import VllmBackend

    backend = VllmBackend(
        model=TEST_MODEL,
        config=GenerationConfig(max_tokens=128, stop=[]),
        gpu_memory_utilization=0.20,
        max_model_len=1024,
    )
    try:
        await backend.warmup()
        # First generation — read a couple of tokens then abort.
        backend.set_prompt("fn add(a: i32, b: i32) -> i32 {")
        emitted_first = 0
        for _ in range(4):
            if not backend.has_next():
                break
            tok = await backend.next()
            if not tok.text:
                break
            emitted_first += 1
        assert emitted_first > 0, "first generation produced nothing"
        await backend.abort_current_stream()
        # Engine should now be cleanly idle. Start a second generation.
        backend.set_prompt("fn multiply(a: i32, b: i32) -> i32 {")
        emitted_second = 0
        for _ in range(20):
            if not backend.has_next():
                break
            tok = await backend.next()
            if not tok.text:
                break
            emitted_second += 1
        assert emitted_second > 0, (
            "engine became unusable after abort — second generation produced 0 tokens"
        )
    finally:
        await backend.close()


# ─── KV-cache correctness ─────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.gpu
@gpu_required
@pytest.mark.asyncio
async def test_prefix_cache_produces_identical_output_to_full_prefill():
    """Same prompt + same sampling params (greedy via temp=0); output token
    text identical between the cached and uncached engines.

    vLLM's prefix caching is exact (KV blocks are bit-identical), so for a
    deterministic decoding (temperature=0 → argmax sampling), the two runs
    must produce identical text.
    """
    from soundcode.vllm_backend import VllmBackend

    prompt = "fn fibonacci(n: u32) -> u64 {"
    # Greedy decoding so the outputs are deterministic. Drop stop patterns
    # to keep both runs going for the full max_tokens budget — that gives
    # the assertion more bytes to compare on.
    cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=32, stop=[])

    async def _collect(backend: VllmBackend) -> str:
        await backend.warmup()
        backend.set_prompt(prompt)
        chars: list[str] = []
        for _ in range(64):
            if not backend.has_next():
                break
            tok = await backend.next()
            if not tok.text:
                break
            chars.append(tok.text)
        return "".join(chars)

    # Cached run.
    cached_backend = VllmBackend(
        model=TEST_MODEL, config=cfg,
        gpu_memory_utilization=0.20, max_model_len=512,
        enable_prefix_caching=True,
    )
    try:
        cached_out = await _collect(cached_backend)
    finally:
        await cached_backend.close()

    await asyncio.sleep(1.0)

    # Uncached run — same engine instance shape, prefix caching disabled.
    uncached_backend = VllmBackend(
        model=TEST_MODEL, config=cfg,
        gpu_memory_utilization=0.20, max_model_len=512,
        enable_prefix_caching=False,
    )
    try:
        uncached_out = await _collect(uncached_backend)
    finally:
        await uncached_backend.close()

    assert cached_out, "cached run produced no tokens"
    assert uncached_out, "uncached run produced no tokens"
    assert cached_out == uncached_out, (
        "prefix-cache run diverged from uncached run:\n"
        f"  cached  ({len(cached_out)} chars): {cached_out!r}\n"
        f"  uncached ({len(uncached_out)} chars): {uncached_out!r}"
    )


@pytest.mark.slow
@pytest.mark.gpu
@gpu_required
@pytest.mark.asyncio
async def test_rollback_and_resume_does_not_corrupt_cache():
    """generate(A) → generate(A+B) → rollback_and_resume(A+C) on one engine;
    the rollback output must match a fresh "generate(A+C)" from a clean slate.

    Verifies that:
      1. Eviction of B's KV blocks doesn't corrupt subsequent C decoding.
      2. The cached prefix A is still usable after the rollback.
      3. Determinism holds across the engine's lifetime under churn.
    """
    from soundcode.vllm_backend import VllmBackend

    prompt_a = "fn helper(x: i32) -> i32 {"
    suffix_b = "\n    // B branch\n"
    suffix_c = "\n    // C branch\n"
    cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=24, stop=[])

    async def _drain(backend: VllmBackend) -> str:
        chars: list[str] = []
        for _ in range(64):
            if not backend.has_next():
                break
            tok = await backend.next()
            if not tok.text:
                break
            chars.append(tok.text)
        return "".join(chars)

    # Churned engine: A → A+B → A+C
    churned = VllmBackend(
        model=TEST_MODEL, config=cfg,
        gpu_memory_utilization=0.20, max_model_len=1024,
        enable_prefix_caching=True,
    )
    try:
        await churned.warmup()
        churned.set_prompt(prompt_a)
        await _drain(churned)
        await churned.abort_current_stream()
        churned.set_prompt(prompt_a + suffix_b)
        await _drain(churned)
        await churned.abort_current_stream()
        # rollback_and_resume on A+C — same engine, the cached blocks for A
        # should reuse cleanly.
        churned_resume_out: list[str] = []
        async for tok in churned.rollback_and_resume(prompt_a + suffix_c):
            if not tok.text:
                break
            churned_resume_out.append(tok.text)
            if len("".join(churned_resume_out)) > 200:
                break
        churned_out = "".join(churned_resume_out)
    finally:
        await churned.close()

    await asyncio.sleep(1.0)

    # Fresh engine: just generate(A+C).
    fresh = VllmBackend(
        model=TEST_MODEL, config=cfg,
        gpu_memory_utilization=0.20, max_model_len=1024,
        enable_prefix_caching=True,
    )
    try:
        await fresh.warmup()
        fresh.set_prompt(prompt_a + suffix_c)
        fresh_out = await _drain(fresh)
    finally:
        await fresh.close()

    assert churned_out, "churned-engine rollback_and_resume produced no tokens"
    assert fresh_out, "fresh-engine baseline produced no tokens"
    # Both runs use greedy decoding from the same prompt; the cache eviction
    # path must not change the output a single character.
    assert churned_out == fresh_out, (
        "rollback_and_resume diverged from fresh generate:\n"
        f"  churned ({len(churned_out)} chars): {churned_out!r}\n"
        f"  fresh   ({len(fresh_out)} chars): {fresh_out!r}"
    )
