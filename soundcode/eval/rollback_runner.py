"""Rollback runner: implements arms A (LSP + compiler) and B (compiler-only)
from `notes/week4-plan-v2.0.md`.

Single-problem entry points:
  - run_arm_a(prob, client, ra, ...)
  - run_arm_b(prob, client, ...)

Both are wall-clock budget bounded (no retry-count cap). Both emit a
`Profile` with spans for §8.3 analysis.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from soundcode.analyzer import RustAnalyzer, Diagnostic
from soundcode.eval.boundary import BoundaryTracker, find_boundaries
from soundcode.eval.classifier import (
    DEFAULT_POLICY,
    DemotionPolicy,
    filter_blocking,
    line_col_to_offset,
)
from soundcode.eval.dataset import RustProblem
from soundcode.eval.history import RollbackHistory
from soundcode.eval.model import ModelClient, GenerationConfig
from soundcode.eval.profile import Profile
from soundcode.eval.sandbox import check_and_run


# Regex to detect open fn-body — very loose heuristic.
# The function body is "open" if the source contains more `{` than `}` at the
# top nesting (ignoring strings/comments). Use the BoundaryTracker's counter logic.


def _fn_body_open(source: str) -> bool:
    """True if there is at least one unclosed top-level function body.

    Uses a simplified approach: count `{` and `}` outside strings/comments.
    Works on the same scanner approach as boundary.py.
    """
    # Piggyback on find_boundaries for the scanner but count braces ourselves.
    # Rather than re-implementing, do a cheap depth scan here.
    in_string = False
    in_raw_string = False
    raw_hashes = 0
    in_line_comment = False
    in_block_comment = 0
    depth = 0
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if c == "/" and i + 1 < n and source[i + 1] == "*":
                in_block_comment += 1
                i += 2
                continue
            if c == "*" and i + 1 < n and source[i + 1] == "/":
                in_block_comment -= 1
                i += 2
                continue
            i += 1
            continue
        if in_raw_string:
            if c == '"':
                k = i + 1
                cnt = 0
                while k < n and cnt < raw_hashes and source[k] == "#":
                    cnt += 1
                    k += 1
                if cnt == raw_hashes:
                    in_raw_string = False
                    i = k
                    continue
            i += 1
            continue
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        # Normal
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            in_line_comment = True
            i += 2
            continue
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            in_block_comment = 1
            i += 2
            continue
        if c in ("r", "b"):
            # Simple raw-string detection
            j = i
            if source[j] == "b":
                j += 1
            if j < n and source[j] == "r":
                j += 1
                h = 0
                while j < n and source[j] == "#":
                    h += 1
                    j += 1
                if j < n and source[j] == '"':
                    in_raw_string = True
                    raw_hashes = h
                    i = j + 1
                    continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "'":
            # Possible char literal; skip forward a few chars to avoid counting braces inside
            # For depth counting purposes we just need to skip past char literals.
            # Try 3-char 'x', 4-char '\x', 6-char '\xHH', up to '\u{...}'
            if i + 2 < n and source[i + 1] != "\\" and source[i + 2] == "'":
                i += 3
                continue
            if i + 3 < n and source[i + 1] == "\\" and source[i + 3] == "'":
                i += 4
                continue
            if i + 5 < n and source[i + 1:i + 3] == "\\x" and source[i + 5] == "'":
                i += 6
                continue
            # Otherwise lifetime
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return depth > 0


@dataclass
class RollbackAttempt:
    """Record of a single generation attempt within arm A."""

    attempt_num: int
    completion: str
    errors: list[dict[str, Any]] = field(default_factory=list)
    rolled_back_to: int = -1  # offset rolled back to, -1 if full restart
    elapsed_s: float = 0.0


@dataclass
class ProblemRunResult:
    """Outcome for a single problem run in either arm."""

    name: str
    model: str
    arm: str
    compiled: bool
    passed: bool
    completion: str
    compile_output: str = ""
    run_output: str = ""
    wall_clock_s: float = 0.0
    wall_clock_to_compile_s: float = float("inf")  # time to first-pass-compile
    num_generation_attempts: int = 0
    num_rollbacks: int = 0
    num_lsp_rollbacks: int = 0       # rollbacks fired by LSP-tier (mid-stream)
    num_compiler_rollbacks: int = 0  # rollbacks fired by compiler-tier (after fn close)
    num_compiler_calls: int = 0
    tokens_generated: int = 0
    tokens_discarded: int = 0
    timed_out: bool = False
    attempts: list[dict[str, Any]] = field(default_factory=list)
    error_codes_seen: list[str] = field(default_factory=list)
    lsp_blocking_codes: list[str] = field(default_factory=list)  # codes that triggered LSP rollback

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self)}


# ---- Prompt rendering helpers ----


def _build_prompt(
    prob: RustProblem,
    history: RollbackHistory,
    *,
    context_window: int,
    reserved_completion: int = 512,
    kept_prefix: str = "",
) -> str:
    """Compose the next-attempt prompt.

    Layout:
        <problem prompt>         (signature + docstring; ends with `{\\n`)
        <rollback history>       (optional, as // comments)
        <kept prefix>            (optional, preserved valid code from prior attempt)

    The model continues from end of this prompt.
    """
    base = prob.prompt
    hist_text = history.render(
        context_window=context_window,
        reserved_completion_tokens=reserved_completion,
        prompt_so_far=base + kept_prefix,
    )
    # Rust problem prompts end with `{\n`. We want:
    #   <signature>{\n
    #   // history...\n
    #   <kept_prefix>
    out = base
    if hist_text:
        out = out + hist_text
    if kept_prefix:
        out = out + kept_prefix
    return out


# ---- Arm A: LSP + compiler ----


async def run_arm_a(
    prob: RustProblem,
    client: ModelClient,
    ra: RustAnalyzer,
    config: GenerationConfig | None = None,
    *,
    budget_s: float = 600.0,
    context_window: int = 8192,
    file_path_stem: str = "arm_a",
    demotion_policy: DemotionPolicy = DEFAULT_POLICY,
) -> tuple[ProblemRunResult, Profile]:
    """Run a single problem under arm A: statement-boundary LSP rollback + compiler.

    Loop bounded only by `budget_s` wall-clock. Profiler records all categories.
    """
    profile = Profile(problem_id=prob.name, model=client.model, arm="A")
    history = RollbackHistory()
    result = ProblemRunResult(
        name=prob.name, model=client.model, arm="A",
        compiled=False, passed=False, completion="",
    )

    problem_t0 = time.perf_counter()
    file_path = f"src/{file_path_stem}_{prob.name.replace('/', '_')}.rs"

    # Ensure file is open
    try:
        await ra.open_file(file_path, "fn main() {}\n")
    except Exception:
        pass  # already open from prior problem, ignore

    final_completion = ""
    kept_prefix = ""  # code preserved from prior attempt's valid portion

    while True:
        elapsed = time.perf_counter() - problem_t0
        remaining = budget_s - elapsed
        if remaining <= 0:
            result.timed_out = True
            break

        attempt_num = history.begin_attempt()
        result.num_generation_attempts = attempt_num

        # Build prompt with history + any kept prefix from prior rollback
        prompt = _build_prompt(prob, history, context_window=context_window, kept_prefix=kept_prefix)
        # Time from "blocking error" (prior attempt end) to first token of
        # the next request: this is the rollback overhead for attempts > 1.
        rollback_overhead_t0 = time.perf_counter_ns() if attempt_num > 1 else None

        # Stream generation
        completion = ""
        tracker = BoundaryTracker()
        aborted = False
        first_token_seen = False
        gen_span_start = time.perf_counter_ns()
        last_token_time = gen_span_start
        last_boundary_count = 0
        blocking_errors: list[Diagnostic] = []

        try:
            async for chunk in client.stream_raw(prompt, config):
                now = time.perf_counter_ns()
                if not first_token_seen:
                    # End the prompt-ingestion span (from request start to first token)
                    profile.add_span(
                        "prompt-ingestion",
                        t_start_ns=gen_span_start,
                        t_end_ns=now,
                        attempt=attempt_num,
                    )
                    if rollback_overhead_t0 is not None:
                        profile.add_span(
                            "rollback-overhead",
                            t_start_ns=rollback_overhead_t0,
                            t_end_ns=now,
                            attempt=attempt_num,
                        )
                    first_token_seen = True
                    last_token_time = now
                else:
                    # generation span covers the gap between prior token and this one
                    profile.add_span(
                        "generation",
                        t_start_ns=last_token_time,
                        t_end_ns=now,
                        attempt=attempt_num,
                    )
                    last_token_time = now

                completion += chunk
                result.tokens_generated += 1

                # Boundary detect
                bd_t0 = time.perf_counter_ns()
                all_boundaries = tracker.update(completion)
                profile.add_span(
                    "boundary-detect",
                    t_start_ns=bd_t0,
                    t_end_ns=time.perf_counter_ns(),
                )
                if len(all_boundaries) > last_boundary_count:
                    last_boundary_count = len(all_boundaries)
                    # Budget check
                    if (time.perf_counter() - problem_t0) > budget_s:
                        aborted = True
                        result.timed_out = True
                        break
                    # LSP check at this boundary
                    check_src = prob.check_source(completion)
                    lsp_t0 = time.perf_counter_ns()
                    try:
                        await ra.update_file(file_path, check_src)
                        errs = await ra.get_errors(file_path)
                    except Exception:
                        errs = []
                    profile.add_span(
                        "lsp-roundtrip",
                        t_start_ns=lsp_t0,
                        t_end_ns=time.perf_counter_ns(),
                        attempt=attempt_num,
                        num_errors=len(errs),
                    )
                    # lsp-wait = lsp-roundtrip here because generation blocks on it.
                    # Record it separately so profile shows the blocking cost.
                    profile.add_span(
                        "lsp-wait",
                        t_start_ns=lsp_t0,
                        t_end_ns=time.perf_counter_ns(),
                        attempt=attempt_num,
                    )
                    # Classify
                    classify_t0 = time.perf_counter_ns()
                    last_pos = all_boundaries[-1].offset
                    fn_open = _fn_body_open(completion)
                    blocking_pairs = filter_blocking(
                        errs, source=check_src,
                        last_complete_pos=last_pos,
                        open_fn_body=fn_open,
                        policy=demotion_policy,
                    )
                    profile.add_span(
                        "classifier",
                        t_start_ns=classify_t0,
                        t_end_ns=time.perf_counter_ns(),
                        attempt=attempt_num,
                        num_blocking=len(blocking_pairs),
                    )
                    if blocking_pairs:
                        blocking_errors = [p[0] for p in blocking_pairs]
                        aborted = True
                        result.num_rollbacks += 1
                        result.num_lsp_rollbacks += 1
                        # Record errors in history
                        for d in blocking_errors:
                            history.add(d)
                            code = d.code or ""
                            result.error_codes_seen.append(code)
                            result.lsp_blocking_codes.append(code)
                        # Compute earliest-error offset in the *completion*.
                        prompt_prefix_len = len(prob.prompt)
                        earliest_in_check = min(
                            line_col_to_offset(check_src, d.range.start.line, d.range.start.character)
                            for d in blocking_errors
                        )
                        earliest_in_completion = max(0, earliest_in_check - prompt_prefix_len)
                        # Find latest boundary strictly BEFORE the earliest error.
                        # Kept prefix = completion[:boundary.offset + 1].
                        target_end = 0
                        for b in all_boundaries:
                            if b.offset < earliest_in_completion:
                                target_end = b.offset + 1
                            else:
                                break
                        kept_prefix = completion[:target_end]
                        break
        except (httpx_aborted_exceptions()) as e:
            # Treat as abort — generation broken
            pass
        except Exception as e:
            # Log and proceed with what we have
            result.compile_output += f"\nStream error: {e}"

        if aborted:
            # Discarded tokens = completion after the kept prefix
            result.tokens_discarded += max(0, len(completion) - len(kept_prefix))
            continue  # new attempt

        # Stream completed without abort. Run compiler.
        final_completion = completion
        result.num_compiler_calls += 1
        source = prob.full_source(completion)
        comp_t0 = time.perf_counter_ns()
        # Budget-aware timeout
        remaining_budget = max(5.0, budget_s - (time.perf_counter() - problem_t0))
        run_result = await check_and_run(source, timeout=min(30.0, remaining_budget))
        profile.add_span(
            "cargo-check",
            t_start_ns=comp_t0,
            t_end_ns=time.perf_counter_ns(),
            invocation=result.num_compiler_calls,
        )

        if run_result.compiled:
            # Success — record wall-clock-to-first-pass-compile now
            result.wall_clock_to_compile_s = time.perf_counter() - problem_t0
            result.compiled = True
            result.passed = run_result.passed
            result.completion = completion
            result.compile_output = run_result.compile_output
            result.run_output = run_result.run_output
            break

        # Compile failed — add rustc error to history and try again.
        # Full restart: we can't tell from rustc output where the boundary
        # should be without more parsing, so reset kept_prefix to empty.
        rustc_err = _extract_first_rustc_error(run_result.compile_output)
        _add_compile_error_to_history(history, rustc_err)
        code = rustc_err.get("code", "") or ""
        result.error_codes_seen.append(code)
        result.num_rollbacks += 1
        result.num_compiler_rollbacks += 1
        result.tokens_discarded += len(completion)
        kept_prefix = ""

    result.wall_clock_s = time.perf_counter() - problem_t0
    if result.compiled is False and not result.completion:
        result.completion = final_completion
    return result, profile


# ---- Arm B: compiler-only ----


async def run_arm_b(
    prob: RustProblem,
    client: ModelClient,
    config: GenerationConfig | None = None,
    *,
    budget_s: float = 600.0,
    context_window: int = 8192,
) -> tuple[ProblemRunResult, Profile]:
    """Arm B: generate function, compile, on error retry with rustc feedback.

    Wall-clock-bounded, no retry count cap. Mirrors week3 compile-retry mode
    without the 3-retry cap."""
    profile = Profile(problem_id=prob.name, model=client.model, arm="B")
    history = RollbackHistory()
    result = ProblemRunResult(
        name=prob.name, model=client.model, arm="B",
        compiled=False, passed=False, completion="",
    )
    problem_t0 = time.perf_counter()
    final_completion = ""

    while True:
        elapsed = time.perf_counter() - problem_t0
        remaining = budget_s - elapsed
        if remaining <= 0:
            result.timed_out = True
            break

        attempt_num = history.begin_attempt()
        result.num_generation_attempts = attempt_num
        prompt = _build_prompt(prob, history, context_window=context_window)
        rollback_overhead_t0 = time.perf_counter_ns() if attempt_num > 1 else None

        # Stream generation (for parity with arm A and accurate profiling).
        # No LSP checks, no boundary-driven aborts.
        completion = ""
        first_token_seen = False
        gen_span_start = time.perf_counter_ns()
        last_token_time = gen_span_start

        try:
            async for chunk in client.stream_raw(prompt, config):
                now = time.perf_counter_ns()
                if not first_token_seen:
                    profile.add_span(
                        "prompt-ingestion",
                        t_start_ns=gen_span_start,
                        t_end_ns=now,
                        attempt=attempt_num,
                    )
                    if rollback_overhead_t0 is not None:
                        profile.add_span(
                            "rollback-overhead",
                            t_start_ns=rollback_overhead_t0,
                            t_end_ns=now,
                            attempt=attempt_num,
                        )
                    first_token_seen = True
                    last_token_time = now
                else:
                    profile.add_span(
                        "generation",
                        t_start_ns=last_token_time,
                        t_end_ns=now,
                        attempt=attempt_num,
                    )
                    last_token_time = now
                completion += chunk
                result.tokens_generated += 1
                if (time.perf_counter() - problem_t0) > budget_s:
                    result.timed_out = True
                    break
        except Exception as e:
            result.compile_output += f"\nStream error: {e}"

        if result.timed_out:
            break

        # Compile
        final_completion = completion
        result.num_compiler_calls += 1
        source = prob.full_source(completion)
        comp_t0 = time.perf_counter_ns()
        remaining_budget = max(5.0, budget_s - (time.perf_counter() - problem_t0))
        run_result = await check_and_run(source, timeout=min(30.0, remaining_budget))
        profile.add_span(
            "cargo-check",
            t_start_ns=comp_t0,
            t_end_ns=time.perf_counter_ns(),
            invocation=result.num_compiler_calls,
        )

        if run_result.compiled:
            result.wall_clock_to_compile_s = time.perf_counter() - problem_t0
            result.compiled = True
            result.passed = run_result.passed
            result.completion = completion
            result.compile_output = run_result.compile_output
            result.run_output = run_result.run_output
            break

        rustc_err = _extract_first_rustc_error(run_result.compile_output)
        _add_compile_error_to_history(history, rustc_err)
        code = rustc_err.get("code", "") or ""
        result.error_codes_seen.append(code)
        result.num_rollbacks += 1
        result.num_compiler_rollbacks += 1
        result.tokens_discarded += len(completion)

    result.wall_clock_s = time.perf_counter() - problem_t0
    if result.compiled is False and not result.completion:
        result.completion = final_completion
    return result, profile


# ---- Helpers ----


_RUSTC_ERROR_RE = re.compile(
    r"error(?:\[(?P<code>E\d+)\])?: (?P<msg>[^\n]+)(?:\n\s*-->\s*[^:]+:(?P<line>\d+):(?P<col>\d+))?",
    re.MULTILINE,
)


def _extract_first_rustc_error(compile_output: str) -> dict[str, Any]:
    """Parse the first rustc error from compile output. Returns a dict."""
    if not compile_output:
        return {"code": "", "msg": "compilation failed", "line": 0, "col": 0}
    m = _RUSTC_ERROR_RE.search(compile_output)
    if not m:
        # Fall back to the first line of output
        first_line = compile_output.strip().split("\n", 1)[0]
        return {"code": "", "msg": first_line[:200], "line": 0, "col": 0}
    return {
        "code": m.group("code") or "",
        "msg": m.group("msg") or "",
        "line": int(m.group("line") or 0),
        "col": int(m.group("col") or 0),
    }


def _add_compile_error_to_history(history: RollbackHistory, rustc_err: dict[str, Any]) -> None:
    """Append a parsed rustc error to the rollback history as a synthetic Diagnostic."""
    import lsprotocol.types as lsp_type
    from soundcode.analyzer import Diagnostic, DiagnosticSeverity

    line = max(0, rustc_err.get("line", 1) - 1)
    col = max(0, rustc_err.get("col", 1) - 1)
    d = Diagnostic(
        message=rustc_err.get("msg", "")[:300],
        severity=DiagnosticSeverity.ERROR,
        range=lsp_type.Range(
            start=lsp_type.Position(line=line, character=col),
            end=lsp_type.Position(line=line, character=col + 1),
        ),
        code=rustc_err.get("code", "") or None,
        source="rustc",
    )
    history.add(d)


def httpx_aborted_exceptions():
    """Tuple of httpx exception types that indicate an aborted stream.

    Kept as a function so the import is lazy (httpx may not be imported at
    module top in callers' contexts)."""
    import httpx
    return (httpx.ReadError, httpx.WriteError, httpx.CloseError, asyncio.CancelledError)
