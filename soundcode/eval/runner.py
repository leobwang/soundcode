"""Run arms (i) raw, (iii-mute) rollback-mute, (iii-instruct) rollback-instruct
on MultiPL-E HumanEval Rust problems.

Post-hoc evaluation per problem:
  - compile: `cargo build` on (prompt + completion + tests + closing brace + main)
  - pass: `cargo test` exit code 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from soundcode.cargo_check import CargoChecker
from soundcode.client import CodeClient, RunResult
from soundcode.eval.dataset import RustProblem, load_rust_humaneval
from soundcode.llm import LlmServer, GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE = PROJECT_ROOT / "rust-samples" / "sample0"
TEST_WORKSPACE = PROJECT_ROOT / "test_workspace"
RESULTS_ROOT = PROJECT_ROOT / "results" / "pilot"


def _ensure_test_workspace() -> Path:
    """Create a Cargo workspace dedicated to running tests post-hoc."""
    TEST_WORKSPACE.mkdir(exist_ok=True)
    (TEST_WORKSPACE / "Cargo.toml").write_text(
        '[package]\nname = "harness"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "harness"\npath = "src/main.rs"\n'
    )
    (TEST_WORKSPACE / "src").mkdir(exist_ok=True)
    return TEST_WORKSPACE


async def _post_hoc_test(prob: RustProblem, completion: str, workspace: Path) -> tuple[bool, bool]:
    """Build + test the assembled program; return (compiled, passed)."""
    src = prob.full_source(completion)
    (workspace / "src" / "main.rs").write_text(src)

    # 1. compile via cargo build
    proc = await asyncio.create_subprocess_exec(
        "cargo", "build", "--offline",
        cwd=str(workspace),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        proc.kill(); await proc.wait()
        return False, False
    compiled = proc.returncode == 0
    if not compiled:
        return False, False

    # 2. run as a binary (MultiPL-E tests panic on failure)
    proc = await asyncio.create_subprocess_exec(
        "cargo", "run", "--offline", "--quiet",
        cwd=str(workspace),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        proc.kill(); await proc.wait()
        return True, False
    return True, proc.returncode == 0


async def _run_raw(prob: RustProblem, model: str, budget_s: float, max_tokens: int) -> RunResult:
    """Arm (i): no verification in loop. Stream until EOS, then post-hoc test."""
    llm = LlmServer(model=model, config=GenerationConfig(max_tokens=max_tokens, stop=["\n}"]))
    llm.set_prompt(prob.prompt)
    t0 = time.perf_counter()
    completion = ""
    tokens = 0
    try:
        while llm.has_next() and time.perf_counter() - t0 < budget_s:
            tok = await llm.next()
            if not tok:
                break
            completion += tok
            tokens += 1
    finally:
        await llm.close()
    wall = time.perf_counter() - t0

    test_ws = _ensure_test_workspace()
    compiled, passed = await _post_hoc_test(prob, completion, test_ws)
    return RunResult(
        name=prob.name,
        arm="raw",
        completion=completion,
        compiled=compiled,
        passed=passed,
        wall_clock_s=wall,
        tokens_emitted=tokens,
        tokens_kept=len(completion),
        rollback_count=0,
        lsp_calls=0,
    )


async def _run_rollback(
    prob: RustProblem, model: str, budget_s: float, max_tokens: int,
    *, instruct: bool, workspace: Path,
) -> RunResult:
    """Arm (iii): rollback at boundaries using cargo check."""
    llm = LlmServer(model=model, config=GenerationConfig(max_tokens=max_tokens, stop=["\n}"]))
    checker = CargoChecker(workspace=workspace)
    client = CodeClient(
        llm=llm, checker=checker,
        instruct_on_rollback=instruct,
        wall_budget_s=budget_s,
        max_rollbacks=8,
    )
    t0 = time.perf_counter()
    try:
        completion, telem = await client.generate(prob.prompt, function_prefix=prob.prompt)
    finally:
        await llm.close()
    wall = time.perf_counter() - t0

    test_ws = _ensure_test_workspace()
    compiled, passed = await _post_hoc_test(prob, completion, test_ws)
    return RunResult(
        name=prob.name,
        arm="rollback_instruct" if instruct else "rollback_mute",
        completion=completion,
        compiled=compiled,
        passed=passed,
        wall_clock_s=wall,
        tokens_emitted=telem["tokens_seen"],
        tokens_kept=len(completion),
        rollback_count=telem["rollbacks"],
        lsp_calls=telem["lsp_calls"],
    )


async def run_arm(
    arm: str, problems: list[RustProblem], model: str,
    budget_s: float, max_tokens: int,
) -> list[RunResult]:
    """Run one arm over a list of problems and return per-problem results."""
    # Each arm gets a dedicated workspace to avoid cross-arm cargo cache races.
    arm_workspace = PROJECT_ROOT / "cargo_workspaces" / f"arm_{arm}"
    arm_workspace.mkdir(parents=True, exist_ok=True)
    (arm_workspace / "Cargo.toml").write_text(
        '[package]\nname = "scratch"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "scratch"\npath = "src/main.rs"\n'
    )
    (arm_workspace / "src").mkdir(exist_ok=True)
    (arm_workspace / "src" / "main.rs").write_text("fn main() {}\n")
    # Pre-warm: run cargo check once so subsequent calls are incremental.
    proc = await asyncio.create_subprocess_exec(
        "cargo", "check", "--offline", "--quiet",
        cwd=str(arm_workspace),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    await proc.communicate()

    results: list[RunResult] = []
    for i, prob in enumerate(problems):
        try:
            if arm == "raw":
                r = await _run_raw(prob, model, budget_s, max_tokens)
            elif arm == "rollback_mute":
                r = await _run_rollback(prob, model, budget_s, max_tokens, instruct=False, workspace=arm_workspace)
            elif arm == "rollback_instruct":
                r = await _run_rollback(prob, model, budget_s, max_tokens, instruct=True, workspace=arm_workspace)
            else:
                raise ValueError(arm)
        except Exception as e:
            r = RunResult(
                name=prob.name, arm=arm, completion="", compiled=False, passed=False,
                wall_clock_s=0.0, tokens_emitted=0, tokens_kept=0,
                rollback_count=0, lsp_calls=0,
                final_diagnostics=[f"runner error: {e!r}"],
            )
        results.append(r)
        print(
            f"  [{i+1:3d}/{len(problems)}] {prob.name[:36]:36} "
            f"arm={arm:18} P={'Y' if r.passed else 'N'} C={'Y' if r.compiled else 'N'} "
            f"wall={r.wall_clock_s:5.1f}s rb={r.rollback_count} "
            f"tok={r.tokens_emitted}/{r.tokens_kept} lsp={r.lsp_calls}",
            flush=True,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral-small3.2:24b")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--arms", nargs="+", default=["raw", "rollback_mute", "rollback_instruct"])
    parser.add_argument("--out", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = load_rust_humaneval()[: args.limit]
    print(f"Running {len(problems)} problems × {len(args.arms)} arms on {args.model}", flush=True)
    print(f"Budget {args.budget}s/problem, max_tokens {args.max_tokens}", flush=True)

    async def _go():
        for arm in args.arms:
            print(f"\n=== arm={arm} ===", flush=True)
            results = await run_arm(arm, problems, args.model, args.budget, args.max_tokens)
            safe_model = args.model.replace(":", "_").replace("/", "_")
            out_path = out_dir / f"{safe_model}_{arm}.json"
            out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
            print(f"Wrote {out_path} ({len(results)} results)", flush=True)

    asyncio.run(_go())


if __name__ == "__main__":
    main()
