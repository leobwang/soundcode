"""Week 4 experiment orchestrator: run arm A and arm B across models/problems.

Saves one JSON per (model, arm) under `./results/week4/{model}_{arm}.json`
containing per-problem results, and one JSON per problem per profile under
`./results/week4/profiles/{model}_{arm}_{problem}.json`.

Resumable: skips (model, arm) pairs whose output JSON already exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from soundcode.analyzer import RustAnalyzer
from soundcode.eval.dataset import load_rust_humaneval
from soundcode.eval.model import ModelClient, GenerationConfig
from soundcode.eval.rollback_runner import run_arm_a, run_arm_b


PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "week4"
PROFILES_DIR = RESULTS_DIR / "profiles"
WORKSPACE = PROJECT_ROOT / "test_workspace"


async def run_model_arm(
    model: str,
    arm: str,
    problems: list,
    *,
    budget_s: float,
    max_tokens: int,
) -> None:
    """Run a single (model, arm) × all problems."""
    out_file = RESULTS_DIR / f"{model.replace(':', '_').replace('/', '_')}_{arm}.json"
    if out_file.exists():
        print(f"[skip] {model} {arm} already done -> {out_file.name}", flush=True)
        return

    client = ModelClient(model=model)
    config = GenerationConfig(max_tokens=max_tokens, stop=["\n}"])
    problem_results = []
    start_time = time.perf_counter()
    print(f"[start] {model} {arm}: {len(problems)} problems, budget={budget_s}s each", flush=True)

    if arm == "A":
        async with RustAnalyzer(WORKSPACE) as ra:
            await ra.open_file("src/main.rs", "fn main() {}\n")
            await asyncio.sleep(2)  # let workspace load
            for i, prob in enumerate(problems):
                res, prof = await run_arm_a(prob, client, ra, config, budget_s=budget_s)
                problem_results.append(res)
                _save_profile(model, arm, prob.name, prof)
                elapsed = time.perf_counter() - start_time
                _log_problem(i + 1, len(problems), prob.name, res, elapsed)
    else:  # B
        for i, prob in enumerate(problems):
            res, prof = await run_arm_b(prob, client, config, budget_s=budget_s)
            problem_results.append(res)
            _save_profile(model, arm, prob.name, prof)
            elapsed = time.perf_counter() - start_time
            _log_problem(i + 1, len(problems), prob.name, res, elapsed)

    summary = _build_summary(model, arm, problem_results, time.perf_counter() - start_time)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"[done]  {model} {arm} -> {out_file.name}", flush=True)


def _log_problem(i: int, total: int, name: str, res, elapsed: float) -> None:
    tag = f"P={'Y' if res.passed else 'N'} C={'Y' if res.compiled else 'N'}"
    to_compile = f"{res.wall_clock_to_compile_s:.1f}s" if res.compiled else "∞"
    print(
        f"  [{i:3d}/{total}] {name[:36]:36} {tag} att={res.num_generation_attempts} "
        f"rb={res.num_rollbacks} cc={res.num_compiler_calls} "
        f"wall={res.wall_clock_s:5.1f}s tc={to_compile} (running {elapsed:.0f}s)",
        flush=True,
    )


def _build_summary(model: str, arm: str, results: list, total_time: float) -> dict:
    n = len(results)
    return {
        "model": model,
        "arm": arm,
        "num_problems": n,
        "total_wall_s": total_time,
        "compile_rate": sum(1 for r in results if r.compiled) / n if n else 0.0,
        "pass_rate": sum(1 for r in results if r.passed) / n if n else 0.0,
        "timeout_rate": sum(1 for r in results if r.timed_out) / n if n else 0.0,
        "avg_attempts": sum(r.num_generation_attempts for r in results) / n if n else 0.0,
        "avg_rollbacks": sum(r.num_rollbacks for r in results) / n if n else 0.0,
        "avg_compiler_calls": sum(r.num_compiler_calls for r in results) / n if n else 0.0,
        "avg_wall_s": sum(r.wall_clock_s for r in results) / n if n else 0.0,
        "median_wall_to_compile_s": _median(
            [r.wall_clock_to_compile_s for r in results if r.compiled]
        ),
        "total_tokens": sum(r.tokens_generated for r in results),
        "total_discarded": sum(r.tokens_discarded for r in results),
        "problems": [
            {
                **asdict(r),
                # dataclass serialization already works; just ensure no non-JSON-native types
            }
            for r in results
        ],
    }


def _median(values: list[float]) -> float:
    if not values:
        return float("inf")
    sv = sorted(values)
    n = len(sv)
    if n % 2:
        return sv[n // 2]
    return 0.5 * (sv[n // 2 - 1] + sv[n // 2])


def _save_profile(model: str, arm: str, prob_name: str, prof) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace(":", "_").replace("/", "_")
    safe_prob = prob_name.replace("/", "_")
    out = PROFILES_DIR / f"{safe_model}_{arm}_{safe_prob}.json"
    prof.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[
        "qwen3.5:0.8b",
        "nemotron-3-nano:4b",
        "qwen3.5:9b",
    ])
    parser.add_argument("--arms", nargs="*", default=["A", "B"])
    parser.add_argument("--limit", type=int, default=30, help="Num problems")
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    problems = load_rust_humaneval()[: args.limit]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    async def _run_all():
        for model in args.models:
            for arm in args.arms:
                try:
                    await run_model_arm(
                        model, arm, problems,
                        budget_s=args.budget, max_tokens=args.max_tokens,
                    )
                except Exception as e:
                    print(f"[error] {model} {arm}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
