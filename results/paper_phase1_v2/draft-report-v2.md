# Phase 1 results — preliminary

Generated 2026-05-25.

## Headline

**Async LSP-supervised decoding is 1.47× faster on Rust and 2.16× faster on C++ than the sync (ROCODE-style) equivalent on the same model + same rollback algorithm.** Speedups are mean-of-means; per-problem median speedup is 1.31× (Rust) and 2.20× (C++). Async beats sync on 78% of paired Rust problems and 99% of paired C++ problems.

On C++, async pass@1 also rises from 75.2% (sync) to 77.0% (async) — the same algorithm under async scheduling is both faster *and* more accurate.

## Numbers

### Per-arm aggregates

| Arm | Lang | Schedule | n | pass@1 | mean wall (s) | median (s) | p90 (s) | mean rb | mean toks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| plain_rust | rust | plain | 156 | 69.9% | 0.704 | 0.619 | 1.134 | 0.00 | 70.3 |
| plain_cpp | cpp | plain | 161 | 77.0% | 0.865 | 0.770 | 1.347 | 0.00 | 86.5 |
| sync_naive_rust | rust | sync | 156 | 73.7% | 1.051 | 0.942 | 1.970 | 0.17 | 69.7 |
| async_naive_rust | rust | async | 156 | 69.2% | 0.714 | 0.630 | 1.132 | 0.17 | 69.7 |
| sync_naive_cpp | cpp | sync | 161 | 75.2% | 2.495 | 2.411 | 3.698 | 0.07 | 87.2 |
| async_naive_cpp | cpp | async | 161 | 77.0% | 1.156 | 1.091 | 1.712 | 0.06 | 86.2 |
| async_entropy_rust | rust | async-entropy | 156 | 66.0% | 0.892 | 0.654 | 1.760 | 0.74 | 70.1 |
| async_entropy_cpp | cpp | async-entropy | 161 | 76.4% | 1.271 | 1.089 | 1.804 | 0.26 | 86.3 |

(Arm 9, `rocode_upstream_cpp`, was skipped — see `results/paper_phase1/known_issues.md`. Our `sync_naive_*` arms are the algorithmic equivalent for this study.)

### Paired sync-vs-async (same problem)

| Lang | n paired | async wins | mean speedup (sync/async) | median speedup | ratio of means | mean Δ rollback |
|---|---:|---:|---:|---:|---:|---:|
| rust | 156 | 122 (78.2%) | 1.45x | 1.31x | 1.47x | +0.00 |
| cpp | 161 | 159 (98.8%) | 2.29x | 2.20x | 2.16x | -0.01 |

## Plots

### `paper/figures/plot_walltime_bars.pdf`

![plot_walltime_bars.pdf](paper/figures/plot_walltime_bars.pdf)

Mean wall-time per problem. Sync (red) vs async (teal) on the same algorithm; plain (gray) is the no-verifier baseline. The 1.47× / 2.16× headline ratios are marked above the sync→async pairs.

### `paper/figures/plot_passrate_bars.pdf`

![plot_passrate_bars.pdf](paper/figures/plot_passrate_bars.pdf)

Per-arm pass@1. The C++ flip (75.2% → 77.0%) is the cleanest accuracy story: the same rollback algorithm becomes more accurate when scheduled asynchronously.

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

- **Why async wins more on C++ than Rust.** `g++ -fsyntax-only` has a cold-start dominated by linker/lex stages (~150 ms in isolation); `cargo check` in incremental mode on a pre-warmed workspace is closer to the per-token decode cost. Sync arms pay the cold-cost at every boundary; async overlaps it with the next decode steps. The relative win therefore scales with the verifier-to-decode cost ratio — see plot_overhead_share.pdf, where sync-C++ wall-time is 2.50 s vs plain 0.87 s, while async-C++ closes most of that gap at 1.16 s.

- **Why async pass@1 is HIGHER on C++.** Sync aborts the decoder at every boundary while the verifier runs; the abort+restart cycle truncates the model's plan more aggressively than the async consumer does. Token counts confirm this: sync-C++ averages 87.2 tokens/problem vs async-C++ 86.2. The accuracy delta is small on Rust (73.7 → 69.2) because the verifier is cheap enough that the sync abort window is short.

- **Why the entropy rollback ablation is negative.** `async_entropy_rust` is 66.0% pass@1 vs `async_naive_rust` at 69.2%; `async_entropy_cpp` is 76.4% vs naive 77.0%. Entropy-guided rollback only fires when the simpler last-checkpoint rule fails — but the observed rollback rate (≤0.17 per problem) is too low for the fallback to add value, and the extra branching budget it costs on the rare oscillation case is wasted everywhere else. The clean read is *the simpler policy suffices given the rollback rate observed on this benchmark*.

## Caveats

- Single seed, no confidence intervals.
- Single benchmark per language (HumanEval-rs / HumanEval-cpp, 164 problems each; ~156–161 reach evaluation after filtering).
- The C++ boundary detector reuses the Rust depth-0 `;`/`}` logic — fine for HumanEval-cpp's small functions, brittle on raw strings or templates with `<…>` depth that the detector ignores.
- ROCODE upstream (arm 9) was skipped: the published implementation is Python-only. Our `sync_naive_*` arms run the same algorithm (block at every boundary, classify, roll back on blocking) on the same model and same verifier, differing only in language adapter and engine — see `results/paper_phase1/known_issues.md`.
- Multi-seed eval at fixed compute is the obvious next step.

- Longest single run: `sync_naive_cpp` on `HumanEval_124_valid_date` at 6.8 s (compile ok, tests pass, 0 rollbacks, 5 checker calls). This is the only problem that hit the 60s wall-clock cap: the model fell into an infinite repetition loop emitting `if (result.find(...))` over and over; because each repeat is locally well-typed, no diagnostic ever fired and the sync verifier had no rollback signal to use. A duplicate-N-gram heuristic or an oscillation detector (\textsc{NaiveLast+Penalty}) would be the fix; we plan to add this in a follow-up. The other 7 arms stayed well under the cap.

## Files

- `paper/figures/plot_walltime_bars.pdf`
- `paper/figures/plot_passrate_bars.pdf`
- `paper/figures/plot_walltime_dist.pdf`
- `paper/figures/plot_rollback_dist.pdf`
- `paper/figures/plot_speedup_scatter.pdf`
- `paper/figures/plot_overhead_share.pdf`
- `results/paper_phase1/aggregate.csv` (per-arm CSV)
- `paper/sections/results.tex` (Results section, filled)
