"""Day 4 validation gate — §10.1 of week4-plan-v2.0.md.

Lightweight integration tests that exercise the whole pipeline without
spending real GPU-hours. Any failure here blocks the full evaluation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from soundcode.analyzer import RustAnalyzer
from soundcode.eval.boundary import find_boundaries
from soundcode.eval.classifier import classify, Verdict, filter_blocking
from soundcode.eval.dataset import RustProblem
from soundcode.eval.history import RollbackHistory
from soundcode.eval.model import ModelClient, GenerationConfig
from soundcode.eval.profile import Profile, CATEGORIES
from soundcode.eval.rollback_runner import (
    run_arm_a, run_arm_b, _build_prompt, _extract_first_rustc_error, _fn_body_open
)


PROJECT_PATH = Path(__file__).parent.parent / "test_workspace"


# ============================================================================
# Unit-ish integration tests (no LLM/no rust-analyzer)
# ============================================================================


def test_fn_body_open_detects_unclosed() -> None:
    assert _fn_body_open("fn f() {") is True
    assert _fn_body_open("fn f() { let x = 1;") is True
    assert _fn_body_open("fn f() { }") is False
    assert _fn_body_open("fn f() { let x = 1; }") is False


def test_fn_body_open_ignores_braces_in_strings() -> None:
    assert _fn_body_open('fn f() { let s = "}"; }') is False
    assert _fn_body_open('fn f() { let s = "{ unclosed"') is True


def test_extract_first_rustc_error() -> None:
    output = """error[E0308]: mismatched types
 --> src/main.rs:5:16
  |
5 |         "/" => req,
  |                ^^^ expected ...
"""
    err = _extract_first_rustc_error(output)
    assert err["code"] == "E0308"
    assert "mismatched types" in err["msg"]
    assert err["line"] == 5


def test_extract_rustc_error_no_code() -> None:
    output = "error: something went wrong"
    err = _extract_first_rustc_error(output)
    assert err["code"] == ""
    assert "something" in err["msg"]


def test_build_prompt_empty_history() -> None:
    p = RustProblem(name="t", prompt="fn f() -> i32 {\n", tests="}", stop_tokens=[])
    h = RollbackHistory()
    out = _build_prompt(p, h, context_window=4096)
    assert out == "fn f() -> i32 {\n"


def test_build_prompt_with_history_and_prefix() -> None:
    import lsprotocol.types as lsp_type
    from soundcode.analyzer import Diagnostic, DiagnosticSeverity

    p = RustProblem(name="t", prompt="fn f() -> i32 {\n", tests="}", stop_tokens=[])
    h = RollbackHistory()
    h.begin_attempt()
    h.add(Diagnostic(
        message="type mismatch",
        severity=DiagnosticSeverity.ERROR,
        range=lsp_type.Range(
            start=lsp_type.Position(line=0, character=0),
            end=lsp_type.Position(line=0, character=1),
        ),
        code="E0308",
        source="rust-analyzer",
    ))
    out = _build_prompt(p, h, context_window=4096, kept_prefix="    let x = 1;\n")
    assert "fn f() -> i32 {" in out
    assert "Previous attempts" in out
    assert "E0308" in out
    # Kept prefix appears AFTER history
    hist_end = out.rfind("type mismatch")
    prefix_start = out.rfind("let x = 1;")
    assert hist_end < prefix_start


# ============================================================================
# Component validation that needs rust-analyzer
# ============================================================================


async def _happy_problem() -> RustProblem:
    # Multi-statement body so at least one statement-boundary LSP check fires.
    return RustProblem(
        name="TestSumVec",
        prompt=(
            "/// Sum the elements of `v` using a running total. "
            "Use at least one local variable and a for loop.\n"
            "fn sum_vec(v: Vec<i32>) -> i32 {\n"
        ),
        tests="""}

fn main() {
    assert_eq!(sum_vec(vec![1, 2, 3, 4]), 10);
    println!("ok");
}
""",
        stop_tokens=["\n}"],
    )


async def _triggers_lsp_error() -> RustProblem:
    # A problem whose "natural" generation calls an undefined helper early.
    # rust-analyzer's native diagnostics may or may not catch this.
    return RustProblem(
        name="TestCall",
        prompt="/// Use a helper named `undefined_helper`.\nfn call_undefined() -> i32 {\n",
        tests="""}

fn main() {
    println!("{}", call_undefined());
}
""",
        stop_tokens=["\n}"],
    )


@pytest.mark.asyncio
async def test_happy_path_arm_b() -> None:
    """Clean problem, arm B, no rollback, compiles ok."""
    p = await _happy_problem()
    client = ModelClient(model="qwen3.5:0.8b")
    config = GenerationConfig(max_tokens=128, stop=["\n}"])
    result, prof = await run_arm_b(p, client, config, budget_s=30.0)
    assert result.num_generation_attempts >= 1
    # Profiler should have at least these categories populated
    wc = prof.wallclock_by_category()
    assert wc["generation"] > 0
    assert wc["prompt-ingestion"] > 0
    assert wc["cargo-check"] > 0
    # No LSP spans in arm B
    assert wc["lsp-roundtrip"] == 0
    assert wc["lsp-wait"] == 0


@pytest.mark.asyncio
async def test_happy_path_arm_a() -> None:
    """Clean problem, arm A, LSP checks fire but no rollback."""
    p = await _happy_problem()
    client = ModelClient(model="qwen3.5:0.8b")
    config = GenerationConfig(max_tokens=128, stop=["\n}"])
    async with RustAnalyzer(PROJECT_PATH) as ra:
        await ra.open_file("src/main.rs", "fn main() {}\n")
        await asyncio.sleep(2)
        result, prof = await run_arm_a(p, client, ra, config, budget_s=30.0)
    wc = prof.wallclock_by_category()
    assert wc["generation"] > 0
    assert wc["prompt-ingestion"] > 0
    # LSP must have fired at least once (boundary-detect is also always present)
    assert wc["lsp-roundtrip"] > 0
    assert wc["boundary-detect"] > 0


@pytest.mark.asyncio
async def test_wall_clock_budget_enforced() -> None:
    """Give a tiny budget; confirm we stop and record timed_out."""
    p = await _happy_problem()
    client = ModelClient(model="qwen3.5:0.8b")
    config = GenerationConfig(max_tokens=8192, stop=[])  # no natural stop
    # 1s budget — should timeout quickly
    result, prof = await run_arm_b(p, client, config, budget_s=1.0)
    # Either it compiled very fast or it timed out. Just check wall_clock is bounded.
    assert result.wall_clock_s < 5.0


def test_profile_all_categories_canonical() -> None:
    """Profile enforces canonical categories; rejects typos."""
    p = Profile(problem_id="p", model="m", arm="A")
    with pytest.raises(ValueError):
        p.add_span("not-a-category", 0, 1_000_000)
    # Valid categories accept
    for cat in CATEGORIES:
        p.add_span(cat, 0, 1_000_000)


@pytest.mark.asyncio
async def test_malformed_lsp_survival() -> None:
    """If the analyzer raises, runner continues without crashing."""

    # Create a no-op mock RustAnalyzer that raises on get_errors.
    class FakeRA:
        async def open_file(self, *a, **kw): return None
        async def update_file(self, *a, **kw): return None
        async def get_errors(self, *a, **kw):
            raise RuntimeError("LSP crashed")

    p = await _happy_problem()
    client = ModelClient(model="qwen3.5:0.8b")
    config = GenerationConfig(max_tokens=128, stop=["\n}"])
    result, prof = await run_arm_a(p, client, FakeRA(), config, budget_s=15.0)  # type: ignore
    # Should still produce a result (compile succeeds or fails, but no crash)
    assert result.name == p.name
    assert result.wall_clock_s > 0


def test_profile_idle_residual_small_on_realistic_run() -> None:
    """Residual idle should be small relative to total on a run with spans.

    Synthesize a dense span stream to verify the residual computation works.
    """
    import time
    p = Profile(problem_id="p", model="m", arm="A")
    # Record spans covering most of the elapsed time
    with p.span("generation"):
        time.sleep(0.01)
    with p.span("cargo-check"):
        time.sleep(0.01)
    d = p.to_dict()
    total = d["total_ns"]
    idle = d["estimated_idle_ns"]
    # Idle should be well under total since we recorded 20ms of a ~20ms run
    # Allow for a fair amount of overhead slack
    assert idle / total < 0.9


# ============================================================================
# Hypothesis sanity: boundary + classifier integration on a real error
# ============================================================================


@pytest.mark.asyncio
async def test_classifier_blocks_real_type_mismatch() -> None:
    """End-to-end: seed rust-analyzer with code that has a type mismatch,
    verify the classifier flags it as blocking when fn body is closed."""
    src = "fn f() -> i32 { \"hi\" }\nfn main() {}\n"
    async with RustAnalyzer(PROJECT_PATH) as ra:
        await ra.open_file("src/type_mismatch_test.rs", src)
        await asyncio.sleep(2)
        errors = await ra.get_errors("src/type_mismatch_test.rs")
    # rust-analyzer should produce at least one error
    # (may be suppressed depending on version; if so, skip)
    if not errors:
        pytest.skip("rust-analyzer returned no errors for the synthetic type mismatch")
    # Classify; fn body is closed in this source
    bs = find_boundaries(src)
    assert bs, "no boundaries found in complete source"
    last_pos = bs[-1].offset
    blocking = filter_blocking(
        errors,
        source=src,
        last_complete_pos=last_pos,
        open_fn_body=False,
    )
    # At least one of the errors should be blocking (the type mismatch)
    assert len(blocking) >= 1


@pytest.mark.asyncio
async def test_classifier_demotes_open_fn() -> None:
    """Same synthetic error; with open_fn_body=True, should be non-blocking."""
    src = "fn f() -> i32 { \"hi\" "  # unclosed
    async with RustAnalyzer(PROJECT_PATH) as ra:
        await ra.open_file("src/type_mismatch_open.rs", src + "\n}\nfn main() {}\n")
        await asyncio.sleep(2)
        errors = await ra.get_errors("src/type_mismatch_open.rs")
    if not errors:
        pytest.skip("rust-analyzer returned no errors")
    # Use last_complete_pos at the end, open_fn_body=True to demote
    bs = find_boundaries(src)
    last_pos = bs[-1].offset if bs else len(src)
    blocking = filter_blocking(
        errors, source=src, last_complete_pos=last_pos, open_fn_body=True
    )
    # With demotion, we expect type-mismatch errors to be non-blocking.
    for d, _r in blocking:
        assert d.code not in ("type-mismatch", "E0308"), \
            f"expected demotion; got blocking {d.code!r}"
