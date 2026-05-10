"""Week 5 selective-demotion ablation orchestrator.

Runs arm A only (arm B is unaffected by demotion) on a subset of models,
under three demotion policies: ``both`` (current §6 default), ``ref_only``
(blog 3's proposal), and ``none`` (sanity ceiling).

Outputs land under ``./results/week5/<policy>/`` mirroring the
week-4 layout. Each (model, policy) pair produces one JSON.
Resumable: skips any (model, policy) pair whose output exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from soundcode.analyzer import RustAnalyzer
from soundcode.eval.classifier import policy_from_name
from soundcode.eval.dataset import load_rust_humaneval
from soundcode.eval.model import ModelClient, GenerationConfig
from soundcode.eval.rollback_runner import run_arm_a


PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "week5"
WORKSPACE = PROJECT_ROOT / "test_workspace"


async def run_model_policy(
    model: str,
    policy_name: str,
    problems: list,
    *,
    budget_s: float,
    max_tokens: int,
) -> None:
    safe_model = model.replace(":", "_").replace("/", "_")
    out_dir = RESULTS_ROOT / policy_name
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = out_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{safe_model}_A.json"
    if out_file.exists():
        print(f"[skip] {model} policy={policy_name} -> {out_file.name}", flush=True)
        return

    pol = policy_from_name(policy_name)
    client = ModelClient(model=model)
    config = GenerationConfig(max_tokens=max_tokens, stop=["\n}"])
    problem_results = []
    start_time = time.perf_counter()
    print(f"[start] {model} A policy={policy_name}: {len(problems)} problems, "
          f"budget={budget_s}s each", flush=True)

    async with RustAnalyzer(WORKSPACE) as ra:
        await ra.open_file("src/main.rs", "fn main() {}\n")
        await asyncio.sleep(2)
        for i, prob in enumerate(problems):
            res, prof = await run_arm_a(
                prob, client, ra, config,
                budget_s=budget_s,
                demotion_policy=pol,
                file_path_stem=f"arm_a_{policy_name}",
            )
            problem_results.append(res)
            safe_prob = prob.name.replace("/", "_")
            prof.save(profiles_dir / f"{safe_model}_A_{safe_prob}.json")
            elapsed = time.perf_counter() - start_time
            tag = f"P={'Y' if res.passed else 'N'} C={'Y' if res.compiled else 'N'}"
            tc = f"{res.wall_clock_to_compile_s:.1f}s" if res.compiled else "∞"
            print(
                f"  [{i+1:3d}/{len(problems)}] {prob.name[:36]:36} {tag} "
                f"att={res.num_generation_attempts} "
                f"lsp_rb={res.num_lsp_rollbacks} cc_rb={res.num_compiler_rollbacks} "
                f"cc={res.num_compiler_calls} wall={res.wall_clock_s:5.1f}s tc={tc} "
                f"(running {elapsed:.0f}s)",
                flush=True,
            )

    summary = _build_summary(model, policy_name, problem_results, time.perf_counter() - start_time)
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"[done]  {model} A policy={policy_name} -> {out_file.name}", flush=True)


def _build_summary(model: str, policy: str, results: list, total_time: float) -> dict:
    n = len(results)
    return {
        "model": model,
        "arm": "A",
        "policy": policy,
        "num_problems": n,
        "total_wall_s": total_time,
        "compile_rate": sum(1 for r in results if r.compiled) / n if n else 0.0,
        "pass_rate": sum(1 for r in results if r.passed) / n if n else 0.0,
        "timeout_rate": sum(1 for r in results if r.timed_out) / n if n else 0.0,
        "avg_attempts": sum(r.num_generation_attempts for r in results) / n if n else 0.0,
        "avg_rollbacks": sum(r.num_rollbacks for r in results) / n if n else 0.0,
        "avg_lsp_rollbacks": sum(r.num_lsp_rollbacks for r in results) / n if n else 0.0,
        "avg_compiler_rollbacks": sum(r.num_compiler_rollbacks for r in results) / n if n else 0.0,
        "total_lsp_rollbacks": sum(r.num_lsp_rollbacks for r in results),
        "total_compiler_rollbacks": sum(r.num_compiler_rollbacks for r in results),
        "avg_compiler_calls": sum(r.num_compiler_calls for r in results) / n if n else 0.0,
        "avg_wall_s": sum(r.wall_clock_s for r in results) / n if n else 0.0,
        "median_wall_to_compile_s": _median(
            [r.wall_clock_to_compile_s for r in results if r.compiled]
        ),
        "total_tokens": sum(r.tokens_generated for r in results),
        "total_discarded": sum(r.tokens_discarded for r in results),
        "problems": [asdict(r) for r in results],
    }


def _median(values: list[float]) -> float:
    if not values:
        return float("inf")
    sv = sorted(values)
    n = len(sv)
    return sv[n // 2] if n % 2 else 0.5 * (sv[n // 2 - 1] + sv[n // 2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[
        "nemotron-3-nano:4b",
        "qwen3.5:9b",
        "mistral-small3.2:24b",
        "nemotron-cascade-2:30b",
    ])
    parser.add_argument("--policies", nargs="*", default=["both", "ref_only", "none"],
                        choices=["both", "ref_only", "type_only", "none"])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    problems = load_rust_humaneval()[: args.limit]
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    async def _run_all():
        for policy in args.policies:
            for model in args.models:
                try:
                    await run_model_policy(
                        model, policy, problems,
                        budget_s=args.budget, max_tokens=args.max_tokens,
                    )
                except Exception as e:
                    print(f"[error] {model} policy={policy}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
