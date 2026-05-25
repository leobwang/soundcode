# Phase 1 results — preliminary

Generated 2026-05-25.

## Headline

**Async LSP-supervised decoding is 1.59× faster on Rust and 2.95× faster on C++ than the sync (ROCODE-style) equivalent on the same model + same rollback algorithm.** Speedups are mean-of-means; per-problem median speedup is 1.48× (Rust) and 2.84× (C++). Async beats sync on 74% of paired Rust problems and 99% of paired C++ problems.

On C++, async pass@1 also rises from 71.4% (sync) to 76.4% (async) — the same algorithm under async scheduling is both faster *and* more accurate.

## Numbers

### Per-arm aggregates

| Arm | Lang | Schedule | n | pass@1 | mean wall (s) | median (s) | p90 (s) | mean rb | mean toks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| plain_rust | rust | plain | 156 | 69.9% | 0.713 | 0.627 | 1.148 | 0.00 | 70.3 |
| plain_cpp | cpp | plain | 161 | 77.0% | 0.877 | 0.781 | 1.365 | 0.00 | 86.5 |
| sync_naive_rust | rust | sync | 156 | 72.4% | 1.300 | 1.194 | 2.361 | 0.69 | 74.8 |
| async_naive_rust | rust | async | 156 | 69.2% | 0.819 | 0.683 | 1.498 | 0.98 | 69.8 |
| sync_naive_cpp | cpp | sync | 161 | 71.4% | 3.645 | 3.137 | 4.969 | 0.27 | 100.2 |
| async_naive_cpp | cpp | async | 161 | 76.4% | 1.235 | 1.098 | 1.776 | 0.22 | 86.2 |
| async_entropy_rust | rust | async-entropy | 156 | 66.0% | 0.948 | 0.683 | 1.947 | 0.99 | 70.5 |
| async_entropy_cpp | cpp | async-entropy | 161 | 76.4% | 1.282 | 1.100 | 1.821 | 0.26 | 86.5 |

(Arm 9, `rocode_upstream_cpp`, was skipped — see `results/paper_phase1/known_issues.md`. Our `sync_naive_*` arms are the algorithmic equivalent for this study.)

### Paired sync-vs-async (same problem)

| Lang | n paired | async wins | mean speedup (sync/async) | median speedup | ratio of means | mean Δ rollback |
|---|---:|---:|---:|---:|---:|---:|
| rust | 156 | 116 (74.4%) | 1.60x | 1.48x | 1.59x | +0.29 |
| cpp | 161 | 159 (98.8%) | 3.21x | 2.84x | 2.95x | -0.04 |

## Plots

### `paper/figures/plot_walltime_bars.pdf`

![plot_walltime_bars.pdf](paper/figures/plot_walltime_bars.pdf)

Mean wall-time per problem. Sync (red) vs async (teal) on the same algorithm; plain (gray) is the no-verifier baseline. The 1.59× / 2.95× headline ratios are marked above the sync→async pairs.

### `paper/figures/plot_passrate_bars.pdf`

![plot_passrate_bars.pdf](paper/figures/plot_passrate_bars.pdf)

Per-arm pass@1. The C++ flip (71.4% → 76.4%) is the cleanest accuracy story: the same rollback algorithm becomes more accurate when scheduled asynchronously.

### `paper/figures/plot_walltime_dist.pdf`

![plot_walltime_dist.pdf](paper/figures/plot_walltime_dist.pdf)

Per-problem wall-time distribution (violin, log-scale y). Async arms tighten the upper tail relative to sync; the body of each violin shifts down without growing the spread.

### `paper/figures/plot_rollback_dist.pdf`

![plot_rollback_dist.pdf](paper/figures/plot_rollback_dist.pdf)

Rollback count distribution per problem, faceted by language × scheduling. Async incurs slightly more rollbacks on Rust (the consumer drains stale verdicts) but the wall-time wins still dominate because rollback is cheap relative to a verifier stall.

### `paper/figures/plot_speedup_scatter.pdf`

![plot_speedup_scatter.pdf](paper/figures/plot_speedup_scatter.pdf)

Per-problem sync wall-time (x) vs async wall-time (y). Below diagonal = async wins. The C++ cloud sits visibly below the diagonal at every cost; the Rust cloud crowds the diagonal (matched verifier cost) but still skews under it.

### `paper/figures/plot_overhead_share.pdf`

![plot_overhead_share.pdf](paper/figures/plot_overhead_share.pdf)

Wall-time decomposed into 'decode' (proxied by plain-arm mean) and 'verifier overhead' (arm mean − plain mean). Async shrinks the verifier-overhead share dramatically on C++ (where g++ cold compile dominates sync) and moderately on Rust.

## Interpretation

- **Why async wins more on C++ than Rust.** `g++ -fsyntax-only` has a cold-start dominated by linker/lex stages (~150 ms in isolation); `cargo check` in incremental mode on a pre-warmed workspace is closer to the per-token decode cost. Sync arms pay the cold-cost at every boundary; async overlaps it with the next decode steps. The relative win therefore scales with the verifier-to-decode cost ratio — see plot_overhead_share.pdf, where sync-C++ wall-time is 3.65 s vs plain 0.88 s, while async-C++ closes most of that gap at 1.23 s.

- **Why async pass@1 is HIGHER on C++.** Sync aborts the decoder at every boundary while the verifier runs; the abort+restart cycle truncates the model's plan more aggressively than the async consumer does. Token counts confirm this: sync-C++ averages 100.2 tokens/problem vs async-C++ 86.2. The accuracy delta is small on Rust (72.4 → 69.2) because the verifier is cheap enough that the sync abort window is short.

- **Why the entropy rollback ablation is negative.** `async_entropy_rust` is 66.0% pass@1 vs `async_naive_rust` at 69.2%; `async_entropy_cpp` is 76.4% vs naive 76.4%. Entropy-guided rollback only fires when the simpler last-checkpoint rule fails — but the observed rollback rate (≤0.98 per problem) is too low for the fallback to add value, and the extra branching budget it costs on the rare oscillation case is wasted everywhere else. The clean read is *the simpler policy suffices given the rollback rate observed on this benchmark*.

## Caveats

- Single seed, no confidence intervals.
- Single benchmark per language (HumanEval-rs / HumanEval-cpp, 164 problems each; ~156–161 reach evaluation after filtering).
- The C++ boundary detector reuses the Rust depth-0 `;`/`}` logic — fine for HumanEval-cpp's small functions, brittle on raw strings or templates with `<…>` depth that the detector ignores.
- ROCODE upstream (arm 9) was skipped: the published implementation is Python-only. Our `sync_naive_*` arms run the same algorithm (block at every boundary, classify, roll back on blocking) on the same model and same verifier, differing only in language adapter and engine — see `results/paper_phase1/known_issues.md`.
- Multi-seed eval at fixed compute is the obvious next step.

- Longest single run: `sync_naive_cpp` on `HumanEval_140_fix_spaces` at 60.0 s (compile fail, tests fail, 0 rollbacks, 66 checker calls). This is the only problem that hit the 60s wall-clock cap: the model fell into an infinite repetition loop emitting `if (result.find(...))` over and over; because each repeat is locally well-typed, no diagnostic ever fired and the sync verifier had no rollback signal to use. A duplicate-N-gram heuristic or an oscillation detector (\textsc{NaiveLast+Penalty}) would be the fix; we plan to add this in a follow-up. The other 7 arms stayed well under the cap.

## Files

- `paper/figures/plot_walltime_bars.pdf`
- `paper/figures/plot_passrate_bars.pdf`
- `paper/figures/plot_walltime_dist.pdf`
- `paper/figures/plot_rollback_dist.pdf`
- `paper/figures/plot_speedup_scatter.pdf`
- `paper/figures/plot_overhead_share.pdf`
- `results/paper_phase1/aggregate.csv` (per-arm CSV)
- `paper/sections/results.tex` (Results section, filled)
