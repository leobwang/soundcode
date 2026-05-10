"""Main experiment script: run baseline and verified generation across models."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from soundcode.eval.dataset import load_rust_humaneval
from soundcode.eval.model import ModelClient, GenerationConfig
from soundcode.eval.runner import run_baseline, run_lsp, run_compiler, EvalResult

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


# Models to test, from smallest to largest
MODELS = [
    # Small models (~1-10B)
    ("qwen3.5:0.8b", "http://localhost:11434"),
    ("qwen3.5:9b", "http://localhost:11434"),
    # Medium models (~30B)
    ("qwen2.5-coder:32b", "http://localhost:11434"),
    # Large models (~120B)
    ("qwen3.5:122b", "http://localhost:11434"),
]


def make_config(temperature: float = 0.2) -> GenerationConfig:
    return GenerationConfig(
        temperature=temperature,
        max_tokens=1024,
        stop=["\n}"],
        top_p=0.95,
    )


async def run_experiment(
    model_name: str,
    base_url: str,
    problems: list,
    *,
    modes: list[str] = ["baseline", "lsp"],
    limit: int | None = None,
    concurrency: int = 4,
) -> list[EvalResult]:
    """Run all experiment modes for one model."""
    client = ModelClient(model=model_name, base_url=base_url)
    config = make_config()
    results = []

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  {model_name} — {mode}")
        print(f"{'='*60}")

        if mode == "baseline":
            result = await run_baseline(
                client, problems, config,
                concurrency=concurrency, limit=limit,
            )
        elif mode == "lsp":
            result = await run_lsp(
                client, problems, config,
                concurrency=min(concurrency, 2),  # Lower concurrency for verified
                limit=limit,
                max_retries=3,
            )
        elif mode == "compiler":
            result = await run_compiler(
                client, problems, config,
                concurrency=concurrency, limit=limit,
                max_retries=3,
            )
        else:
            print(f"  Unknown mode: {mode}, skipping")
            continue

        print(result.summary())

        # Save results
        safe_name = model_name.replace("/", "_").replace(":", "_")
        result.save(RESULTS_DIR / f"{safe_name}_{mode}.json")
        results.append(result)

    return results


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run SoundCode experiments")
    parser.add_argument("--models", nargs="*", help="Model names to test (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Limit problems per model")
    parser.add_argument("--modes", nargs="*", default=["baseline", "lsp"],
                        help="Modes to run (baseline, verified)")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests")
    args = parser.parse_args()

    print("Loading dataset...")
    problems = load_rust_humaneval()
    print(f"Loaded {len(problems)} problems")

    # Filter models if specified
    models = MODELS
    if args.models:
        models = [m for m in MODELS if m[0] in args.models]
        if not models:
            # Try as direct model names against Ollama
            models = [
                (name, "http://localhost:11434")
                for name in args.models
            ]

    all_results: list[EvalResult] = []

    for model_name, base_url in models:
        try:
            results = await run_experiment(
                model_name, base_url, problems,
                modes=args.modes,
                limit=args.limit,
                concurrency=args.concurrency,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\nERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Mode':<12} {'Compiled':>10} {'Passed':>10} {'Time':>8}")
    print(f"{'-'*30} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
    for r in all_results:
        print(
            f"{r.model:<30} {r.mode:<12} "
            f"{r.compilation_rate:>9.1%} {r.pass_rate:>9.1%} "
            f"{r.total_time:>7.1f}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
