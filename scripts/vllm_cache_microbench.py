"""Microbenchmark — vLLM automatic prefix caching for SoundCode rollback.

Story: when SoundCode rolls back to a previously-streamed prefix, vLLM's
automatic prefix caching reuses the KV-cache pages for the shared prefix,
so the second-and-subsequent stream opens skip the prefill cost. This script
quantifies the savings.

It runs three streams against the same model and prints time-to-first-token
(TTFT) + cache-hit indicators for each:

  1. `generate(prompt_1)`                — cold cache, full prefill.
  2. `generate(prompt_1 + extension)`    — same prefix, longer prompt.
     Expectation: TTFT_2 < TTFT_1 by roughly the prefill cost of `prompt_1`.
  3. `rollback_and_resume(prompt_1)`     — abort + resubmit truncated prefix.
     Expectation: TTFT_3 ≈ TTFT_2 (both reuse the cached prefix; #3 is also
     the production rollback path).

Run:
    uv run python scripts/vllm_cache_microbench.py
    uv run python scripts/vllm_cache_microbench.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct

Defaults to the 1.5B coder model so it fits in ~10 GB of VRAM alongside other
workloads. Override `--model` to test the 7B model (requires ~16 GB free).

Skip if you don't have ~10 GB free VRAM — set `--dry-run` to print the
expected behaviour table without loading any weights.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Local-import the backend without relying on installed package paths.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from soundcode.llm import GenerationConfig  # noqa: E402
from soundcode.vllm_backend import BackendStats, VllmBackend  # noqa: E402


# A non-trivial Rust prompt — long enough that prefill cost dominates a
# single-token TTFT measurement. Padded to ~1000 tokens so the prefill
# savings from the shared-prefix cache hit are large relative to the
# fixed scheduling overhead.
_BASE_PROMPT = """// Implement a Rust function that takes a string slice, splits it on
// whitespace, parses each token as an i64, and returns the sum. Skip tokens
// that fail to parse. The function signature is given below — fill in the
// body. Use idiomatic iterator chains and avoid intermediate allocations.
//
// Background context (intentionally verbose so the prompt is long enough
// that prefill cost dominates a single-token TTFT measurement):
//   * The function is part of a CLI tool that ingests log lines of the form
//     `<timestamp> <level> <count> ...`. Only the `<count>` field is summed.
//   * Lines may contain extra whitespace, tabs, or trailing punctuation.
//   * Negative integers are valid (e.g. "-5"); fractional and hex literals
//     are NOT valid and should be skipped, not coerced.
//   * The caller already split the file into lines; this function operates
//     on one line at a time.
//   * Performance matters: this is called on millions of log lines, so
//     avoid allocating intermediate Vec<String> or Vec<i64>; prefer
//     iterator chains that fuse.
//   * Robustness matters more than performance for the parse step; do not
//     swallow Result::Err silently in a way that hides bugs upstream.
//   * The output is consumed by a Prometheus exporter, which expects an
//     i64 counter (not f64); overflow is technically possible but unlikely
//     given log volumes, so we use plain `+` rather than `checked_add`.
//   * Style: prefer `&str::split_whitespace`, `str::parse::<i64>`,
//     `Iterator::filter_map`, and `Iterator::sum` over manual loops.
//
// Edge cases:
//   - empty string → 0
//   - all-whitespace string → 0
//   - "1 2 hello 3 4" → 10 (skip "hello")
//   - "  1  2  " → 3 (ignore leading/trailing/extra whitespace)
//   - "1.5 2 3" → 5 (skip "1.5"; parse::<i64> rejects non-integers)
//   - "-1 -2 3" → 0
//   - "9223372036854775808" → 0 (overflows i64, parse rejects)

use std::str::FromStr;

/// Sum all integers in a whitespace-separated string slice, skipping unparseable tokens.
///
/// # Examples
///
/// ```
/// assert_eq!(sum_integers("1 2 3"), 6);
/// assert_eq!(sum_integers("1 2 hello 3"), 6);
/// assert_eq!(sum_integers(""), 0);
/// assert_eq!(sum_integers("-1 -2 3"), 0);
/// ```
pub fn sum_integers(s: &str) -> i64 {"""

PROMPT_1 = _BASE_PROMPT

# Extension simulates "the rollback added a Rust comment and a few more
# instructive tokens" — the path the demo overlay actually takes.
PROMPT_1_EXT = (
    PROMPT_1
    + "\n    // (verifier feedback): use filter_map + parse::<i64> + sum\n"
)


@dataclass
class RunResult:
    label: str
    ttft_s: float
    prefix_tokens: int
    cached_tokens: int

    @property
    def cache_hit_rate(self) -> float:
        if self.prefix_tokens <= 0:
            return 0.0
        return self.cached_tokens / self.prefix_tokens


async def _drive_one_stream_via_set_prompt(
    backend: VllmBackend, prompt: str, label: str,
) -> RunResult:
    """Open a stream the normal way (set_prompt + next loop) and consume
    until the first ~32 tokens to capture TTFT + cache stats."""
    backend.set_prompt(prompt)
    n_consumed = 0
    while backend.has_next() and n_consumed < 32:
        tok = await backend.next()
        if not tok.text:
            break
        n_consumed += 1
    await backend.abort_current_stream()
    stats = backend.last_stats
    assert stats.ttft_s is not None, "no token emitted"
    return RunResult(
        label=label,
        ttft_s=stats.ttft_s,
        prefix_tokens=stats.prefix_tokens,
        cached_tokens=stats.cached_tokens,
    )


async def _drive_rollback(
    backend: VllmBackend, prompt: str, label: str,
) -> RunResult:
    """Open via rollback_and_resume so the cache-hit story matches what the
    production rollback path does."""
    n_consumed = 0
    async for tok in backend.rollback_and_resume(prompt):
        n_consumed += 1
        if n_consumed >= 32:
            break
    await backend.abort_current_stream()
    stats = backend.last_stats
    assert stats.ttft_s is not None, "no token emitted (rollback)"
    return RunResult(
        label=label,
        ttft_s=stats.ttft_s,
        prefix_tokens=stats.prefix_tokens,
        cached_tokens=stats.cached_tokens,
    )


def _print_table(results: list[RunResult]) -> None:
    hdr = (
        f"{'run':<28} {'TTFT (s)':>10} {'prefix_tok':>12} "
        f"{'cached_tok':>12} {'hit %':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.label:<28} {r.ttft_s:>10.4f} {r.prefix_tokens:>12d} "
            f"{r.cached_tokens:>12d} {100.0 * r.cache_hit_rate:>7.1f}%"
        )


def _print_expected_table() -> None:
    print("Expected behaviour with enable_prefix_caching=True:")
    print()
    print("  Run 1 (cold cache):")
    print("    TTFT_1 ≈ full prefill cost — first-token latency dominated by")
    print("    prompt-length-quadratic attention. cached_tokens = 0 (or just")
    print("    a small system-block hit).")
    print()
    print("  Run 2 (extended prompt, same prefix):")
    print("    TTFT_2 ≪ TTFT_1 — vLLM's PagedAttention reuses the KV blocks")
    print("    for the shared prefix; only the new suffix is prefilled.")
    print("    cached_tokens ≈ len(tokenize(PROMPT_1)) rounded to block size.")
    print()
    print("  Run 3 (rollback to original prefix):")
    print("    TTFT_3 ≈ TTFT_2 — same prefix-cache hit, no prefill on the")
    print("    common prefix. is_rollback=True on backend.last_stats.")
    print()
    print("  Speedup_2 = TTFT_1 / TTFT_2 ≈ 5–50× depending on prompt length.")
    print("  Speedup_3 = TTFT_1 / TTFT_3 ≈ same as Speedup_2.")


async def _run_bench(
    *,
    model: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    repeat: int,
) -> None:
    backend = VllmBackend(
        model=model,
        config=GenerationConfig(max_tokens=64),
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
    )
    print(f"Loading model: {model}")
    t0 = time.perf_counter()
    await backend.warmup()
    print(f"Loaded in {time.perf_counter() - t0:.1f} s")
    try:
        # Warmup pass to amortize any first-call CUDA graph cost — not
        # counted toward the reported TTFTs. Uses a DIFFERENT prompt so the
        # warmup doesn't accidentally prime the cache for PROMPT_1.
        print("Warmup pass (not measured)...")
        await _drive_one_stream_via_set_prompt(
            backend,
            "// warmup: print a Rust hello world\nfn main() {",
            "warmup",
        )

        all_runs: list[RunResult] = []
        for trial in range(repeat):
            # Use a per-trial unique-suffix prompt for the cold run so each
            # repetition actually misses the cache. The shared prefix PROMPT_1
            # tail is what runs 2 and 3 reuse.
            tag = f"trial-{trial}"
            cold_prompt = f"// trial id: {trial}  // cold-cache discriminator\n" + PROMPT_1

            print(f"\n--- {tag} ---")
            print(f"Run 1: generate(cold) — cold cache")
            r1 = await _drive_one_stream_via_set_prompt(
                backend, cold_prompt, f"{tag} 1: cold",
            )
            print(f"  TTFT_1 = {r1.ttft_s:.4f}s  "
                  f"prefix={r1.prefix_tokens}  cached={r1.cached_tokens}")

            print(f"Run 2: generate(cold + extension) — shared prefix cache hit")
            r2 = await _drive_one_stream_via_set_prompt(
                backend, cold_prompt + "\n    // (verifier feedback)\n",
                f"{tag} 2: extension",
            )
            print(f"  TTFT_2 = {r2.ttft_s:.4f}s  "
                  f"prefix={r2.prefix_tokens}  cached={r2.cached_tokens}")

            print(f"Run 3: rollback_and_resume(cold) — rollback to cached prefix")
            r3 = await _drive_rollback(
                backend, cold_prompt, f"{tag} 3: rollback",
            )
            print(f"  TTFT_3 = {r3.ttft_s:.4f}s  "
                  f"prefix={r3.prefix_tokens}  cached={r3.cached_tokens}")
            all_runs.extend([r1, r2, r3])

        # Aggregate across trials — group by suffix (1/2/3).
        print("\n=== All trials ===")
        _print_table(all_runs)

        cold_runs = [r for r in all_runs if "1: cold" in r.label]
        ext_runs = [r for r in all_runs if "2: extension" in r.label]
        rb_runs = [r for r in all_runs if "3: rollback" in r.label]
        if cold_runs and ext_runs and rb_runs:
            mean = lambda xs: sum(xs) / len(xs)
            mc = mean([r.ttft_s for r in cold_runs])
            me = mean([r.ttft_s for r in ext_runs])
            mr = mean([r.ttft_s for r in rb_runs])
            print()
            print(f"Mean TTFT_cold     = {mc * 1000:.2f} ms")
            print(f"Mean TTFT_extend   = {me * 1000:.2f} ms")
            print(f"Mean TTFT_rollback = {mr * 1000:.2f} ms")
            if me > 0:
                print(f"Speedup_extend / cold     = {mc / me:.2f}x")
            if mr > 0:
                print(f"Speedup_rollback / cold   = {mc / mr:.2f}x")
            print()
            ok_ext = all(r.cached_tokens > 0 for r in ext_runs)
            ok_rb = all(r.cached_tokens > 0 for r in rb_runs)
            ok_cold = all(r.cached_tokens == 0 for r in cold_runs)
            print(f"Cold runs report cached=0:        {'OK' if ok_cold else 'FAIL'}")
            print(f"Extend runs report cached>0:      {'OK' if ok_ext else 'FAIL'}")
            print(f"Rollback runs report cached>0:    {'OK' if ok_rb else 'FAIL'}")
    finally:
        await backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Model id or path. Default: 1.5B coder model.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.15,
        help="vLLM GPU memory budget fraction (default 0.15 = ~14 GB on a 96GB card).",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=2048,
        help="vLLM max model length (default 2048).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print expected behaviour table without loading any model.",
    )
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="How many cold/warm/rollback triples to run for stable timings "
             "(default 3). The cold run uses a fresh prompt suffix each time "
             "so it actually hits a cold cache.",
    )
    args = parser.parse_args()

    if args.dry_run:
        _print_expected_table()
        return

    asyncio.run(_run_bench(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        repeat=args.repeat,
    ))


if __name__ == "__main__":
    main()
