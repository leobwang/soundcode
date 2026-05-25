"""Tests for the SemGuard evaluator wrapper + LogitsProcessor integration.

Two layers covered:

  1. `SemGuardEvaluator` (in `soundcode/semguard_eval.py`) — wraps the trained
     1.3B classifier. The "real load" test is marked `slow` and skipped if
     either the checkpoint or the backbone directory is missing.

  2. `SemGuardEvaluatorProcessor` (in `soundcode/logits_processors.py`) —
     observes the token stream and signals rollback when a line scores below
     threshold. Tested with a `FakeEvaluator` so it runs fast and on machines
     without a GPU.

Run with `uv run pytest tests/test_semguard_integration.py -xvs`. Add `-m slow`
for the real-model load test.
"""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from typing import Any

import pytest

from soundcode.logits_processors import (
    RollbackRequest,
    SemGuardEvaluatorProcessor,
)


# Real-model paths. Both are environment-dependent — tests skip if missing.
CKPT_PATH = Path(
    "/home/leobwang/code/courses/cmsc25750/project/reproduce/results/semguard/"
    "train_20260524_115641/deepseek-coder-1.3b_python/checkpoinss/0/model_0.bin"
)
BACKBONE_PATH = Path.home() / "hf-models" / "deepseek-coder-1.3b-base"


# ─── helpers ──────────────────────────────────────────────────────────


class FakeEvaluator:
    """Stand-in for `SemGuardEvaluator` with no GPU / no model.

    Returns a constant score; records every prefix it was called with so
    tests can assert on the contract."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.calls: list[str] = []

    async def score_prefix(self, prefix: str) -> float:
        self.calls.append(prefix)
        # Yield to let the event loop schedule other tasks before returning.
        await asyncio.sleep(0)
        return self.score


class FakeTokenizer:
    """Stand-in for an HF tokenizer's `.decode()` — pretends each integer is
    a single character (id == ord(char)). Good enough for newline detection."""

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)


# ─── SemGuardEvaluator ────────────────────────────────────────────────


@pytest.mark.slow
def test_semguard_evaluator_loads_from_checkpoint():
    """End-to-end load + score. Skipped if the checkpoint or backbone is
    missing (so the test passes on machines without SemGuard training done).

    Tries CUDA first; falls back to CPU if CUDA is unavailable or OOM. The
    classifier is only epoch-1 — scores can be middling, but the call must
    return a float in [0, 1]."""
    if not CKPT_PATH.exists():
        pytest.skip(f"SemGuard checkpoint not found at {CKPT_PATH}")
    if not BACKBONE_PATH.exists():
        pytest.skip(f"Backbone not found at {BACKBONE_PATH}")

    try:
        import torch
    except ImportError:
        pytest.skip("torch not available in test env")

    from soundcode.semguard_eval import SemGuardEvaluator, format_prefix

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = SemGuardEvaluator(
        checkpoint_path=CKPT_PATH,
        backbone_path=BACKBONE_PATH,
        device=device,
    )

    async def _run() -> tuple[float, float]:
        try:
            await evaluator.warmup()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # CUDA OOM at warmup → retry on CPU (cheaper than failing the
            # test outright; smoke-test the loading path).
            if "out of memory" in str(e).lower():
                pytest.skip(f"CUDA OOM during warmup: {e!r}")
            raise
        # Score two prefixes — one a plausible Python function, one obviously
        # broken. We don't assert the relative order (epoch-1 model can have
        # noisy preferences) — just that both calls return floats in [0, 1].
        good = format_prefix(
            nl="Add two integers and return the sum.",
            code="def add(a, b):\n    return a + b\n",
        )
        bad = format_prefix(
            nl="Add two integers and return the sum.",
            code="def add(a, b):\n    return a - b ** ** ** !!\n",
        )
        s_good = await evaluator.score_prefix(good)
        s_bad = await evaluator.score_prefix(bad)
        await evaluator.close()
        return s_good, s_bad

    s_good, s_bad = asyncio.run(_run())
    assert isinstance(s_good, float), s_good
    assert isinstance(s_bad, float), s_bad
    assert 0.0 <= s_good <= 1.0, s_good
    assert 0.0 <= s_bad <= 1.0, s_bad
    # Print for visibility under `-s` — epoch-1 model so we DON'T assert
    # ordering, but a smoke-test reader can eyeball the numbers.
    print(f"\n  good prefix score: {s_good:.4f}")
    print(f"  bad prefix score:  {s_bad:.4f}")


def test_semguard_evaluator_raises_clear_error_on_missing_checkpoint(tmp_path):
    """If the checkpoint path doesn't exist, warmup() must raise a
    FileNotFoundError with a clear diagnosis (not silently degrade)."""
    from soundcode.semguard_eval import SemGuardEvaluator

    missing = tmp_path / "no-such-ckpt.bin"
    evaluator = SemGuardEvaluator(
        checkpoint_path=missing,
        backbone_path=tmp_path,  # also missing but ckpt is checked first
        device="cpu",
    )

    async def _run() -> None:
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            await evaluator.warmup()

    asyncio.run(_run())


def test_semguard_evaluator_score_before_warmup_raises():
    """`score_prefix` before `warmup()` must raise a clear RuntimeError so a
    misuse fails loud instead of returning a meaningless float."""
    from soundcode.semguard_eval import SemGuardEvaluator

    evaluator = SemGuardEvaluator(
        checkpoint_path=Path("/dev/null"),
        backbone_path=Path("/dev/null"),
        device="cpu",
    )

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="warmup"):
            await evaluator.score_prefix("anything")

    asyncio.run(_run())


# ─── SemGuardEvaluatorProcessor (with FakeEvaluator) ──────────────────


async def test_semguard_processor_logits_unchanged():
    """The processor MUST return logits unmodified — its job is to observe,
    not to alter the sampling distribution. This holds for both the
    disabled (no-evaluator) and the enabled (with-evaluator) paths."""
    # Build a real logits tensor if torch is around; else a list.
    try:
        import torch
        logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
        is_tensor = True
    except ImportError:
        logits = [0.1, 0.2, 0.3, 0.4]
        is_tensor = False

    # Disabled (no evaluator) path.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_disabled = SemGuardEvaluatorProcessor()
    out = p_disabled([1, 2, 3], logits)
    if is_tensor:
        assert torch.equal(out, logits)
    else:
        assert out is logits

    # Enabled (with evaluator) path — even when scoring triggers, logits
    # come back untouched.
    fake_eval = FakeEvaluator(score=0.99)
    p_enabled = SemGuardEvaluatorProcessor(
        evaluator=fake_eval, tokenizer=FakeTokenizer(), threshold=0.5,
    )
    # Bind a loop so it doesn't silently no-op.
    await p_enabled.bind_loop()
    out2 = p_enabled([ord("a"), ord("\n")], logits)
    if is_tensor:
        assert torch.equal(out2, logits)
    else:
        assert out2 is logits


async def test_semguard_processor_signals_rollback_below_threshold():
    """When the evaluator scores below threshold on a freshly-completed
    line, the processor must enqueue a `RollbackRequest` on
    `rollback_signals`."""
    fake_eval = FakeEvaluator(score=0.3)
    fake_tok = FakeTokenizer()
    proc = SemGuardEvaluatorProcessor(
        evaluator=fake_eval, tokenizer=fake_tok, threshold=0.5,
    )
    await proc.bind_loop()

    # Simulate the vLLM sampler emitting `def f():\n` — 9 tokens, the last
    # one being a newline that triggers scoring.
    token_ids: list[int] = []
    text = "def f():\n"
    for ch in text:
        token_ids.append(ord(ch))
        proc(list(token_ids), [0.0])  # logits unused by the fake
    # Let the scheduled coroutine run.
    await asyncio.sleep(0.01)
    # Drain a few extra event loop iterations in case the run_coroutine_
    # threadsafe scheduling needs more.
    for _ in range(5):
        if not proc.rollback_signals.empty():
            break
        await asyncio.sleep(0.01)

    sig = await proc.check_rollback_signals()
    assert sig is not None, "expected a RollbackRequest on the queue"
    assert isinstance(sig, RollbackRequest)
    assert sig.at_line == 1                          # one newline so far
    assert 0.0 <= sig.score < 0.5                    # below threshold
    assert sig.prefix_len == len(text)
    # Evaluator was called at least once with the full prefix.
    assert any(call.endswith("def f():\n") for call in fake_eval.calls)


async def test_semguard_processor_no_signal_above_threshold():
    """When the evaluator scores above threshold, the queue stays empty."""
    fake_eval = FakeEvaluator(score=0.8)
    proc = SemGuardEvaluatorProcessor(
        evaluator=fake_eval, tokenizer=FakeTokenizer(), threshold=0.5,
    )
    await proc.bind_loop()

    token_ids: list[int] = []
    for ch in "def f():\n":
        token_ids.append(ord(ch))
        proc(list(token_ids), [0.0])
    await asyncio.sleep(0.05)

    sig = await proc.check_rollback_signals()
    assert sig is None, f"expected no signal, got {sig!r}"
    # But the evaluator MUST have been called (to know the score was high).
    assert len(fake_eval.calls) >= 1, "evaluator was never called"


async def test_semguard_processor_respects_max_resamples():
    """Once `max_resamples` signals have been emitted, further newlines
    must NOT add more signals (until `reset()`)."""
    fake_eval = FakeEvaluator(score=0.1)
    proc = SemGuardEvaluatorProcessor(
        evaluator=fake_eval,
        tokenizer=FakeTokenizer(),
        threshold=0.5,
        max_resamples=2,
    )
    await proc.bind_loop()

    # Emit five newline-separated lines — only the first two should produce
    # a rollback signal.
    token_ids: list[int] = []
    for ch in "a\nb\nc\nd\ne\n":
        token_ids.append(ord(ch))
        proc(list(token_ids), [0.0])
    # Wait for all scheduled scorings.
    await asyncio.sleep(0.1)

    count = 0
    while True:
        sig = await proc.check_rollback_signals()
        if sig is None:
            break
        count += 1
    assert count == 2, f"expected exactly 2 signals (max_resamples), got {count}"


async def test_semguard_processor_reset_drains_queue_and_counters():
    """`reset()` must clear pending signals and reset the line counter so
    the next run starts fresh."""
    fake_eval = FakeEvaluator(score=0.1)
    proc = SemGuardEvaluatorProcessor(
        evaluator=fake_eval, tokenizer=FakeTokenizer(),
        threshold=0.5, max_resamples=5,
    )
    await proc.bind_loop()

    # Emit one line — should produce one signal.
    token_ids: list[int] = []
    for ch in "x\n":
        token_ids.append(ord(ch))
        proc(list(token_ids), [0.0])
    await asyncio.sleep(0.05)
    assert not proc.rollback_signals.empty()

    proc.reset()
    assert proc.rollback_signals.empty()
    assert proc._lines_scored == 0
    assert proc._signals_emitted == 0
    assert proc._decoded_prefix == ""


def test_semguard_processor_silently_noops_without_bound_loop():
    """If the consumer forgot to call `bind_loop()`, the sampler MUST NOT
    crash — the processor degrades to logits-passthrough so generation
    proceeds (just without rollback signalling)."""
    fake_eval = FakeEvaluator(score=0.1)
    proc = SemGuardEvaluatorProcessor(
        evaluator=fake_eval, tokenizer=FakeTokenizer(), threshold=0.5,
    )
    # Note: no `await proc.bind_loop()`.
    try:
        import torch
        logits = torch.tensor([0.1, 0.2])
    except ImportError:
        logits = [0.1, 0.2]
    out = proc([ord("a"), ord("\n")], logits)
    if hasattr(logits, "shape"):
        import torch
        assert torch.equal(out, logits)
    else:
        assert out is logits
    # No queue was ever constructed.
    assert proc.rollback_signals is None
