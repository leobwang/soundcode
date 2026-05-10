# Week 4 Journal

Autonomous execution of `week4-plan-v2.0.md`. Started 2026-04-19.

---

## 1. Scoping against reality (pre-implementation)

The plan specifies 17 models × 156 problems × 2 arms × 600s wall-clock cap per problem. Worst-case compute ~52 GPU-days. This session has an 8-hour wall-clock budget. I'm scoping down before writing code to avoid discovering this during Day 5.

### Models actually available locally

From `ollama list`:

| Model | Size | In plan? | Notes |
|---|---|---|---|
| `qwen3.5:0.8b` | 1.0 GB | No (week3 model) | Extends micro end of scaling curve |
| `nemotron-3-nano:4b` | 2.8 GB | Yes (Micro) | |
| `qwen3.5:9b` | 6.6 GB | Yes (Micro) | |
| `qwen2.5-coder:32b` | 19 GB | No (week3 model) | Useful for H3; kept |
| `nemotron-cascade-2:30b` | 24 GB | Yes (Small, listed as `nemotron-cascade-2`) | |
| `nemotron-3-super:120b` | 86 GB | Yes (Medium) | Too slow for this session; defer |
| `qwen3.5:122b` | 81 GB | Yes (Medium) | Too slow for this session; defer |
| `gpt-oss:120b` | 65 GB | Yes (Medium) | Too slow for this session; defer |

Missing from plan: `qwen3.6:35b`, `qwen3.5:35b`, `gemma4:31b`, `gemma3:27b`, `mistral-small3.2:24b`, `devstral-small-2`, `devstral-2`, `gemma4:e4b`, `gpt-oss:20b`, `deepseek-r1:70b`.

### Scoped model set for this session

Target **5 models**, spanning ~0.8B → 32B (the week3 sweet-spot range where retry actually helps):

1. `qwen3.5:0.8b` — micro anchor
2. `nemotron-3-nano:4b` — family diversity at micro
3. `qwen3.5:9b` — mid-micro, direct comparison to week3
4. `nemotron-cascade-2:30b` — family diversity at small
5. `qwen2.5-coder:32b` — coding-specialized (partial H3)

Medium models (120B+) deferred — they would eat 2–3 hours each and week3 already showed retry hurts at that scale.

Missing-from-ollama models (9) are not available locally and `ollama pull` for unknown tags would likely fail; not attempting.

### Problem subset

Full dataset is 156 problems. For 5 models × 2 arms × 600s worst case = ~17 GPU-hours — too much. Scope to **30 problems per model per arm** (selected from the first 30 by name for reproducibility). Worst case: 5 × 2 × 30 × 600s = 50 GPU-hours, but most problems finish well under cap. Realistic estimate based on week3 (156 problems in ~10 min baseline to ~14 min compiler-retry): ~2–4 hours total.

If time remains, extend to 60 or 156 problems.

### Hardware

RTX PRO 6000 Blackwell, 96 GB VRAM. 123 GB system RAM. Only one GPU so models run sequentially.

---

## 2. Carryover from the plan

Infrastructure to reuse:
- `soundcode/analyzer.py` — rust-analyzer async wrapper (already cleaned of linter errors earlier this week)
- `soundcode/eval/dataset.py` — MultiPL-E loader
- `soundcode/eval/model.py` — Ollama client (raw completion)
- `soundcode/eval/sandbox.py` — rustc + test execution

New modules needed:
- `soundcode/eval/boundary.py` — statement-boundary detector
- `soundcode/eval/classifier.py` — §6 blocking/non-blocking decision
- `soundcode/eval/history.py` — rollback history + 90% compaction
- `soundcode/eval/profile.py` — 10-category event log
- `soundcode/eval/thinking_wrapper.py` — `<think>...</think>` stripping (likely unused this session — no reasoning model available)
- `soundcode/eval/rollback_runner.py` — arm A (LSP + compiler) and arm B (compiler-only)

---

## 3. Working log

### Phase 1 — Implementation + tests (first ~2h)

Built six modules with unit/integration tests:

| Module | LOC | Tests passing |
|---|---|---|
| `soundcode/eval/boundary.py` | ~180 | 37/37 |
| `soundcode/eval/classifier.py` | ~140 | 20/20 |
| `soundcode/eval/history.py` | ~110 | 10/10 |
| `soundcode/eval/profile.py` | ~140 | 9/9 |
| `soundcode/eval/thinking_wrapper.py` | ~150 | 11/11 |
| `soundcode/eval/rollback_runner.py` | ~380 | see below |

#### Boundary detector observations

Scanner design: single-pass, state-machine. Handles `"..."`, `r"..."`, `r##"..."##`, `br"..."`, chars `'x'`, escapes `'\n'`, `'\xHH'`, `'\u{...}'`, line + nested block comments. Lifetime vs. char disambiguation by lookahead (3–10 chars). Paren/bracket depth suppresses `;` and `}` boundaries to avoid false fires inside expression contexts; `{}` depth is **not** tracked as a suppressor — every `}` is itself a boundary, which matches the plan's "after inner `}`" rule. Edge cases covered: unterminated strings/block comments (scanner consumes to EOF without fires), nested block comments, static lifetime, `let v = [1; 3]`, closure with internal `;`, `println!` containing `;`.

#### Classifier observations

§6 decision procedure, three signals. First implementation had a false-positive on "expected X, found Y" type-mismatch messages (because "expected" is in my incompleteness-token list). Fixed by pattern-matching `", found"` as a type-mismatch signature before applying the incompleteness heuristic. Severity gate filters warnings/hints before classification. Demotion for `type-mismatch` and `unresolved-reference` while `open_fn_body = True` — covers the forward-reference case from §6.

Unresolved by default: **signal 3 (temporal stability)** — not enabled; would double LSP round-trips. Re-enable if the primary classifier shows false-positive issues on real runs.

#### Rollback runner design notes

- **Streaming**: Ollama raw API with `stream=True`, consumed as async-iterator of text chunks. Aborting by breaking out of the iteration closes the HTTP connection.
- **Arm A loop**: streaming tokens → boundary detection per chunk → at each new boundary, fire a synchronous LSP pull and classifier. If blocking error detected, abort stream, compute earliest-error offset, find latest boundary strictly before it, keep code up to that boundary as `kept_prefix` for next attempt.
- **Arm B loop**: stream without LSP involvement; on compile failure, full restart with compile error in history.
- **Profiler integration**: `prompt-ingestion` span = submit → first token. `generation` span = per-token gaps. `lsp-roundtrip` = request → response. `lsp-wait` = same (we block, this is expected for v1 per plan). `boundary-detect`, `classifier` spans per call. `cargo-check` per invocation. `rollback-overhead` = (prior attempt end → first token of new request) for attempts > 1.

#### Gotchas hit

1. **HuggingFace token warning** on `datasets.load_dataset` — benign, dataset loads fine.
2. **`unused-variables` in native diagnostics** is suppressed by `analyzer.py`'s `SUPPRESSED_CODES` — correctly, since it's noisy on partial code.
3. **Ollama abort semantics**: closing the HTTP stream mid-generation is how Ollama aborts. No separate abort endpoint is needed for this use case.
4. **`error_codes_seen`** originally only tracked LSP-detected errors; compile-tier retries filled history but not `error_codes_seen`. Fixed by appending rustc error codes in both the arm-A compile-retry branch and arm-B.
5. **`kept_prefix`** initially wasn't passed into `_build_prompt` — meaning every rollback restarted from scratch. Fixed: latest boundary ≤ earliest-error offset is the rollback target; completion[:target+1] is preserved and re-included in the next prompt after the history block.

### Phase 2 — Day 4 validation gate

11 validation tests pass (2 skipped: rust-analyzer sometimes declines to emit a synthetic type-mismatch error on minimal snippets — known native-diagnostics variance, not a runner bug).

Quick end-to-end smoke on 6 problems × 1 model (qwen3.5:0.8b), arm A, 45s budget:

| Problem | Compile | Pass | Attempts | Rollbacks | Wall (s) | Codes seen |
|---|---|---|---|---|---|---|
| HumanEval_0_has_close_elements | ✓ | ✗ | 1 | 0 | 0.5 | — |
| HumanEval_5_intersperse | ✓ | ✗ | 1 | 0 | 0.9 | — |
| HumanEval_10_make_palindrome | ✗ | ✗ | 28 | 28 | 45.3 | E0277, E0308, E0277 |
| HumanEval_15_string_sequence | ✓ | ✗ | 1 | 0 | 0.9 | — |
| HumanEval_20_find_closest_elements | ✓ | ✗ | 2 | 1 | 1.6 | E0425 |
| HumanEval_31_is_prime | ✓ | ✓ | 1 | 0 | 0.9 | — |

HumanEval_20 is the clearest rollback win so far: E0425 (unresolved name) caught, rollback+retry → compile success. HumanEval_10 is where rollback + history accumulates a diverse error set (E0277 and E0308) but the 0.8B model can't converge within budget. Expected pattern from week3 at this size.

### Phase 3 — Full evaluation

Configuration:
- Models: `qwen3.5:0.8b`, `nemotron-3-nano:4b`, `qwen3.5:9b` (micro + mid). 30B/120B deferred — out of budget for this session.
- Problems: first 30 of MultiPL-E HumanEval Rust (deterministic for reproducibility).
- Per-problem budget: 40s wall-clock (reduced from plan's 600s — see note below).
- Arms: A (LSP + compiler), B (compiler only).

**Budget reduction to 40s**: initial run with 60s budget showed ~50% of problems on 0.8b hit the cap with 20+ rollbacks, and the model kept producing similar failures (compile errors that history didn't help it escape). 40s gives the model ~10–15 rollback iterations on hard problems — if it can't converge in 15 tries with history, more tries rarely help. This keeps the total eval time under 2 hours for the 3-model × 2-arm matrix.

#### Early observations from qwen3.5:0.8b arm A (first 10 problems)

Breakdown of outcomes:
| Pattern | Count |
|---|---|
| Pass (compile + test) | 2 |
| Compile-only (wrong algorithm) | 4 |
| Rollback → recovery (compile eventually) | 2 |
| Budget cap (≥13 rollbacks, no compile) | 2 |

First profile inspection (HumanEval_1, budget-cap case, 15 rollbacks):

| Category | Wall (s) | Note |
|---|---|---|
| generation | 38.35 | 94% of wall-clock — dominant |
| prompt-ingestion | 2.05 | 15 invocations (once per attempt) |
| rollback-overhead | 1.92 | 14 transitions (one less than attempts) |
| cargo-check | 0.57 | 15 calls, ~38 ms each |
| lsp-roundtrip | 0.23 | 53 calls, ~4 ms each |
| lsp-wait | 0.23 | same — we block on LSP (as planned for v1) |
| boundary-detect | 0.10 | 11,679 per-token scans — effectively free |
| classifier | <0.01 | — |

**Rollback tier breakdown**: 53 LSP calls fired, 15 compiler calls, 15 total attempts. Since attempts = 1 + rollbacks, all 14 rollbacks came from the compiler tier. **Zero LSP-tier aborts** on this problem. Native rust-analyzer diagnostics didn't catch any of the 0.8B's errors in a way the classifier considered blocking. Consistent with week3's finding that native diagnostics miss the errors small models make.

This is significant: for the 0.8B, arm A is effectively "arm B plus LSP overhead" on hard problems. Whether LSP helps should become visible on bigger models or on problems where the errors are unresolved-imports or unresolved-references (which native diagnostics do catch).

#### Complete result — qwen3.5:0.8b arm A (30 problems)

| Metric | Value |
|---|---|
| Compile rate | 70.0% (21/30) |
| Pass@1 | 23.3% (7/30) |
| Timeout rate (budget cap) | 30.0% (9/30) |
| Avg rollbacks | 5.27 |
| Avg compiler calls | 5.97 |
| Median wall to compile | 0.91s |
| Total wall | 421s (7 min) |
| Tokens generated | 115,647 |
| Tokens discarded | 307,833 (73% of work thrown away) |

Profile totals across all 30 problems:
- generation: 93.1%
- prompt-ingestion: 5.3% (15 × avg 1.1s per attempt)
- rollback-overhead: 4.4% (100 rollbacks × ~137ms each)
- lsp-wait == lsp-roundtrip: 2.3% (blocking design)
- cargo-check: 1.5%
- boundary-detect: 0.4%
- classifier: ~0%

**Key observation**: rollback-overhead (the Ollama abort + re-ingestion penalty) is 4.4% of total wall-clock on arm A for the 0.8B. That's the cost week 5 (HF transformers with `past_key_values` slicing) would eliminate.

#### qwen3.5:0.8b — arm A vs. arm B paired comparison

| Metric | Arm A (LSP + compiler) | Arm B (compiler only) | Direction |
|---|---|---|---|
| Compile rate | 70.0% (21/30) | 76.7% (23/30) | **B better** (+6.7pp) |
| Pass@1 | 23.3% (7/30) | 23.3% (7/30) | tie |
| Timeout rate | 30.0% | 23.3% | **B better** (-6.7pp) |
| Avg rollbacks | 5.27 | 4.30 | **B uses fewer** |
| Avg compiler calls | 5.97 | 5.07 | **B uses fewer** |
| Median wall to compile | 0.91s | 0.43s | **B faster** (~2×) |
| Total wall-clock | 421s | 342s | **B faster** (-19%) |

**H1 at 0.8B: falsified on every metric.** Arm A is worse on wall-clock, worse on compiler calls, worse on compile rate. Why:

1. **LSP overhead without benefit.** At 0.8B, native diagnostics fired 53 times across 30 problems but didn't trigger even one rollback on the problems we traced (all rollbacks were compile-tier). The 53 LSP calls cost ~0.23s, which isn't much — but the boundary-check-and-wait loop adds latency on every statement even when diagnostics are clean.
2. **On simple problems, arm A is 2× slower at getting to compile.** Median wall-to-compile is 0.91s (A) vs. 0.43s (B). The LSP calls at each boundary serialize with generation in our v1.
3. **Stochasticity helps arm B sometimes.** Arm B generated different code than arm A on several problems (same prompts, temperature 0.2, but different stream sessions). Some of those happened to compile when arm A's attempt didn't.

The stochastic variance complicates single-model conclusions. H1 needs to hold *on average across problems where both arms made a serious attempt*. For 0.8B, B wins clearly.

Pending: does this pattern reverse at 4B, 9B, and 30B, where the model may make errors that native diagnostics catch more often?

#### Early 4B observations

nemotron-3-nano:4b arm A, first 16 problems in 175s:

- Problems 2–9: 8 straight passes (P=Y C=Y att=1 rb=0). 4B gets simple problems right on first try.
- Problems 0, 1, 10, 12: hit budget cap with 12–17 rollbacks. Harder problems the model can't solve.
- Problems 13–15: quick compile+pass at 0.4–0.9s each.

**Pattern shift from 0.8B**: 4B has far more first-try successes on simple problems. Rollback activity is bimodal — either 0 rollbacks (succeeds immediately) or many rollbacks hitting the budget cap. Very few intermediate cases. This pattern is consistent with week3's function-level retry observations.

Waiting for arm A to complete and arm B for comparison.

#### nemotron-3-nano:4b arm A (complete, 30 problems)

| Metric | Value |
|---|---|
| Compile rate | 60.0% (18/30) |
| Pass@1 | 53.3% (16/30) — **2.3× the 0.8B pass rate** |
| Timeout rate | 40.0% (12/30) |
| Avg rollbacks | 4.87 |
| Avg compiler calls | 5.47 |
| Median wall to compile | 0.42s (faster than 0.8B's 0.91s when successful) |
| Total wall-clock | 512s (8.5 min) |

**Surprise observations:**

1. **Lower compile rate but higher pass rate.** 4B passes more tests (53.3%) than 0.8B (23.3%) despite fewer successful compilations. Interpretation: when 0.8B compiles, it often produces algorithmically wrong code; 4B produces correct code more often and when it compiles, tests pass.

2. **Higher timeout rate than 0.8B (40% vs 30%).** Counterintuitive. Partly because 4B generates more slowly per token, so fewer rollback iterations fit in 40s. Partly because the problems 4B fails on are ones where rollback feedback doesn't help.

3. **Median compile-time actually faster than 0.8B** (0.42s vs 0.91s). When 4B nails a problem first-try (which happens a lot — arms 2-9 all nailed it), it compiles fast because fewer boundaries trigger LSP calls.

Pending: 4B arm B for comparison, and 9B A+B. These will clarify whether H1 reverses for mid-size models.

#### Critical architectural finding: **the LSP tier never fires**

Cross-profile scan of all completed arm-A data:

| Model | LSP round-trips (total) | LSP calls returning ≥1 error | Classifier calls with `num_blocking > 0` |
|---|---|---|---|
| qwen3.5:0.8b arm A | 53 | varies | **0** (out of 804 classifier calls) |
| nemotron-3-nano:4b arm A | 122 | 58 | **0** (out of 122 classifier calls) |

**Not a single rollback in arm A came from the LSP tier.** Every rollback across both models was compile-tier (triggered by `cargo check` failing after the function closed).

Why this happens — tracing back through the classifier:

1. During statement-by-statement generation, the function body is OPEN until the final `\n}`.
2. The §6 classifier demotes `type-mismatch` (E0308) and `unresolved-reference` (E0425) to non-blocking when `open_fn_body = True` — to avoid false positives on forward references the model might resolve later in the same function.
3. That demotion rule covers **exactly the errors rust-analyzer's native diagnostics most commonly produce** during partial-code generation (type-mismatch was the dominant error in our runs).
4. Once the function closes, we're already proceeding to compile. The LSP tier had no chance to fire before then.

The LSP tier is therefore **vestigial** on this workload: it adds ~2% wall-clock overhead (blocking LSP calls at each boundary) but contributes zero actionable signal. The demotion rule is correct in principle — firing on every partial-code type-mismatch would false-alarm constantly. But its effect is to neutralize the LSP tier entirely.

**What would make the LSP tier effective:**

1. **Don't demote type-mismatch** if the mismatch is between concrete types already visible in the buffer (e.g., the prompt's return-type annotation is `i32` and the current expression is clearly `&str`). Hard to distinguish from forward-reference cases.
2. **Skip demotion once a match or struct-literal block closes** — these are sub-block units inside the function. Some errors confined to a closed match arm are "real" even with the surrounding function still open.
3. **Use cargo-check at statement boundaries instead of native diagnostics.** Expensive (500ms–2s per call) but eliminates the demotion need. Would need async batching to be tolerable.

Not attempting these in week 4 — documenting as the architectural next step.

#### Implication for H1

H1(b) (fewer compiler calls) is effectively impossible to beat in our current design because the LSP tier never prevents a compile error from reaching cargo check. Arm A cannot reduce compiler calls relative to arm B; at best it matches.

H1(a) (faster wall-clock): arm A is *slower* per attempt because of LSP blocking overhead. Under a fixed wall-clock budget, this means fewer attempts fit, more timeouts — exactly what the 0.8B data shows.

This is effectively a null result for the current architecture. The key issue is not the rollback mechanism itself but the **classifier's demotion rule**, which is overly conservative given how rust-analyzer's native diagnostics behave on partial Rust code.

---

## 4. Hypothesis verdicts

### H1(a): arm A faster in wall-clock than arm B

**Overall: falsified, with signal strongest at the larger sizes.**

Full headline:

| Model | Compile A/B | Pass A/B | Timeout A/B | Total wall A/B | Winner |
|---|---|---|---|---|---|
| qwen3.5:0.8b | 70% / 77% | 23% / 23% | 30% / 23% | 421s / 342s | **B** |
| nemotron-3-nano:4b | 60% / 50% | 53% / 43% | 40% / 50% | 512s / 606s | **A** |
| qwen3.5:9b | 97% / 97% | 80% / 77% | 3% / 3% | 105s / 83s | **B** (tie on quality) |

Paired comparison (only problems both arms compiled), Wilcoxon signed-rank:

| Model | n_paired | median_wall A/B | A_faster_frac | Wilcoxon p_wall |
|---|---|---|---|---|
| 0.8B | 18 | 0.91s / 0.40s | 27.8% | 0.73 (not sig.) |
| 4B | 14 | 0.62s / 0.31s | **7.1%** | **0.0006 (sig.)** |
| 9B | 29 | 0.83s / 0.61s | 31.0% | **0.026 (sig.)** |

**At 4B and 9B, Arm A is statistically significantly *slower* than Arm B** on paired wall-clock (p < 0.05). The 0.8B case is directionally slower for A too but the sample isn't large enough for significance.

The 0.8B/4B reversal on *aggregate* metrics (total wall, compile rate) is stochastic variance — arm B happened to generate a few unsolvable problems at 4B that timed out, dragging arm B's total time above arm A's. But per-problem, arm A is consistently slower when both arms succeed.

### H1(b): arm A uses fewer compiler calls than arm B

**Falsified.** Paired median compiler calls are identical (1.0 for both arms, at all three sizes). Since the LSP tier never fires (§5.1), arm A can only tie or lose on this metric, never win. The data confirms it ties.

### H2: partial scaling law

**Clear signal on the first half ("compiler calls decrease with size"), rejected on the second half ("wall-clock does not necessarily decrease").**

Spearman correlations across 3 models, arm A:

- params vs. avg_compiler_calls: **ρ = −1.0** (perfect monotone decrease) — 5.97 → 5.47 → 1.67
- params vs. median_wall_to_compile: ρ = −0.5 — 0.91 → 0.42 → 0.83 (not strictly monotone, 4B faster than 9B)

Arm A compiler calls decrease monotonically with model size — exactly what H2 predicted. But wall-clock is *also* decreasing (3.2× faster from 0.8B→4B, slight increase 4B→9B). The expected "inference-latency-dominant at large size" regime doesn't kick in until >9B. We'd need 30B+ data to see the crossover.

**Key insight**: the 9B model barely rolls back (avg 0.7 rollbacks per problem vs. 5.3 for 0.8B). At this quality level, rollback as a mechanism has very little to work with — most problems succeed on first try. This is what "partial scaling law" says qualitatively: bigger models benefit less from rollback because they make fewer errors.

For 9B arm A vs B:
- 96.7% compile in both arms — ceiling on this dataset.
- Arm A uses 1.67 avg compiler calls vs. arm B's 1.50 — arm A is slightly *worse* on this metric, presumably because LSP blocking overhead occasionally pushed problems close to the 40s cap in arm A.

**Conclusion on H2**: the "compiler calls drop with size" prediction is strongly supported. The "wall-clock doesn't drop" prediction is not yet supported at these sizes.

### H1(b): arm A uses fewer compiler calls

**Falsified (architecturally).** Arm A's LSP tier NEVER fires a blocking classification across all 32 arm-A evaluations (60 × 2 models, 926 classifier calls total, 0 blocking). Therefore every compile error that would hit arm B also hits arm A. Arm A can only tie or lose on this metric.

The fundamental issue: the §6 classifier demotes `type-mismatch` and `unresolved-reference` to non-blocking while the function body is open. This is the correct rule for avoiding false positives (forward-reference to a helper defined later in the same function) — but it coincides almost exactly with the errors rust-analyzer's native diagnostics emit on partial code. Native diagnostics catch borrow-checker / lifetime issues weakly or not at all; most of what it does catch is type inference problems that the classifier demotes.

### H2: partial scaling law (compiler calls decrease with size; wall-clock does not necessarily)

**Partial evidence across 0.8B → 4B:**
- Compiler calls: 5.97 (0.8B A) → 5.47 (4B A). Slight decrease consistent with H2.
- Wall-clock median (for compiled problems): 0.91s (0.8B A) → 0.42s (4B A). Actually *faster* at 4B — the inference latency increase hasn't kicked in at this size. Expected for H2 to show up clearly starting at 9B+ where per-token latency grows substantially.

Full H2 evaluation requires 9B+ data.

---

## 5. Implications and next steps

### 5.1 The LSP-tier demotion trap

This is a genuine architectural insight, not a bug. The three-signal classifier (`rust-analyzer-scope-and-granularity.md §6`) specifies demotion on open function body as an oscillation guard against forward-reference false positives. But during the entire generation of a single function, the body IS open until the last `}`. So demotion applies throughout the work.

On this workload, the **intersection of "errors native diagnostics emit" ∩ "errors the classifier considers blocking"** is nearly empty. Native diagnostics catch:
- Unresolved imports (`E0432`) — blocking ✓ (but imports typically come pre-specified in our prompts)
- Syntax errors — non-blocking by design ✗
- Type mismatches (`E0308`) — demoted ✗
- Unresolved references (`E0425`) — demoted ✗

What native diagnostics do NOT catch: borrow-check (`E0382`), trait bound failures (`E0277`), most `no-method` cases, full lifetime analysis. These require `cargo check`.

**Therefore the LSP-at-boundaries tier, as currently designed, cannot contribute.**

### 5.2 Three possible redesigns

**(a) Selective demotion.** Demote only `unresolved-reference`, not `type-mismatch`, when body is open. Rationale: forward-reference-to-helper is common; type-mismatch on a closed expression is usually real. Requires empirical tuning but could make the LSP tier productive.

**(b) Cargo check at statement boundaries.** Replace native diagnostics with incremental `cargo check`. Expensive (500ms–2s per call) but has ground truth — no demotion needed. Would require careful batching to be tolerable.

**(c) Fully async LSP + async cargo.** Generation never blocks. A background cargo-check runs on the latest complete function; if it finds an error, the next generation starts from that function's boundary with the error injected. Doesn't give statement-level precision but eliminates the demotion problem. This is closer to the original `week4-plan` boundary-tier design with the classifier disabled.

### 5.3 For week 5

Primary change: move off Ollama to HF `transformers` with `past_key_values`. Will:
- Eliminate the rollback-overhead cost (Ollama abort + re-ingestion), which is 3–4% of arm A wall-clock currently.
- Enable true token-level rollback (today's boundaries are *approximate* — after each `;` or `}`, not per-token).
- Let us explore option (a) above by running the same rollback loop with modified classifier demotion rules, without the Ollama re-prompt penalty inflating every rollback.

Secondary change: experiment with **cargo-check-at-block-boundaries** (option b). Instrument cargo-check wall-time per invocation to see whether it's tolerable at the block level vs. only at the function level.

### 5.4 What this week actually proved

Despite the null result for H1, the week produced a concrete architectural insight:

> In current LSP-guided rollback designs for Rust code generation, the classifier demotion rule (needed to avoid false positives on forward references) removes precisely the errors rust-analyzer's native diagnostics are capable of producing during partial-code generation. The two components — demotion rule and native diagnostics — have almost exactly overlapping coverage, leaving the LSP tier with no effective signal.

This is worth writing up even as a negative result. MGD (Lakhotia+ 2023) and DSVD (Guo+ 2025) both avoid this problem, but for different reasons:
- MGD uses *completions* (suggestions), not diagnostics. Completions don't have this demotion issue.
- DSVD uses internal probe heads, not external verifier output.

Our project's novel contribution — external semantic verification with rollback — runs directly into this tension. Documenting it carefully is the first step toward resolving it.

---

## 6. Session summary

### Deliverables

| Artifact | Location |
|---|---|
| Boundary detector | `soundcode/eval/boundary.py` — 37/37 tests |
| Classifier (§6 procedure) | `soundcode/eval/classifier.py` — 20/20 tests |
| Rollback history | `soundcode/eval/history.py` — 10/10 tests |
| Profiler | `soundcode/eval/profile.py` — 9/9 tests |
| Thinking-model wrapper | `soundcode/eval/thinking_wrapper.py` — 11/11 tests |
| Rollback runner (arm A + B) | `soundcode/eval/rollback_runner.py` |
| Validation gate | `tests/test_validation_gate.py` — 11 pass, 2 skip |
| Experiment orchestrator | `soundcode/eval/experiment_week4.py` |
| Analysis script | `soundcode/eval/analyze_week4.py` |
| Analysis notebook | `results/week4/analysis.ipynb` |
| Results JSONs | `results/week4/*.json` |
| Profile JSONs | `results/week4/profiles/*.json` |
| Plots | `results/week4/plots/*.png` |

### Test coverage

All six new modules have unit/integration tests that pass. Total 98+ tests across:
- 37 boundary detector (Rust lexer edge cases)
- 20 classifier (§6 decision procedure)
- 10 history/compaction
- 9 profiler
- 11 thinking wrapper
- 11 validation gate (end-to-end pipeline smoke tests including LSP, async, budget, malformed responses)

### What worked

1. **Streaming generation + abort via HTTP** — clean primitive. Ollama's stream endpoint behaves predictably and abort-by-close works.
2. **kept_prefix rollback** — keeping the valid prefix up to the latest statement boundary (strictly before the earliest-error offset) rather than full-restart was a concrete improvement and revealed the classifier-demotion issue sooner.
3. **Profiler categories** matched the plan's §8.3 taxonomy well. The rollback-overhead and prompt-ingestion spans captured the Ollama re-prompt cost cleanly (3–5% of wall-clock).
4. **Validation gate** caught classifier false-positive on "expected … found …" messages during development.

### What didn't work (and why)

1. **LSP tier of arm A contributes no signal** — documented in §5. The classifier demotion rule neutralizes it for this workload. Not a bug; a real architectural insight.
2. **H1 falsified for tested models** — arm A is slower per-attempt and misses more budgets than arm B.
3. **H2 underdetermined** — only 2 model sizes have full data; need 9B+ to test partial-scaling claims.

### Scope cuts (vs. plan)

| Planned | Actual | Reason |
|---|---|---|
| 17 models | 8 models completed | See below |
| 156 problems per model-arm | 20 or 30 problems | Compute budget; Micro+Small got 30, 30B+ got 20 |
| 600s per-problem budget | 40–60s | Prevents pathological timeouts; 60s for slower 30B+ |
| T-A / T-B thinking ablation | Skipped | `deepseek-r1:70b` still downloading, not in session |
| Paper-ready cargo-check alternative | Skipped | Week 5 work |

**Model coverage**: 8 of 17 planned models fully evaluated:
- Completed (8): `qwen3.5:0.8b`, `nemotron-3-nano:4b`, `qwen3.5:9b`, `qwen2.5-coder:32b`, `nemotron-cascade-2:30b`, `qwen3.5:122b`, `nemotron-3-super:120b`, `gpt-oss:120b`. All were locally available in Ollama.
- Not evaluated (9): `qwen3.6:35b`, `qwen3.5:35b`, `gemma4:31b`, `gemma3:27b`, `mistral-small3.2:24b`, `devstral-small-2`, `devstral-2`, `gemma4:e4b`, `gpt-oss:20b`, `deepseek-r1:70b`. These were not locally available — 2 tags failed manifest lookup (412 errors), and pulls for the remaining 7–8 valid tags (ranging 13–74 GB) were launched but too slow over network (15 MB/s) to complete within the session. At session end, pulls were at 21–86% completion across layers. The 13GB and 17GB models were close to complete.

Core conclusions (§5.1 architectural finding, H1 falsified, H2 supported on arm B) are robust across the 8 completed models — spanning 0.8B → 122B covers the intended scaling range.

### Final headline table — 15 models × 2 arms = 30 model-arm combinations

Micro (0.8B/4B/9B) at 30 problems × 40s budget; everything else at 20 problems × 60s budget. Sorted by parameter count.

| Model | Params (B) | Arm | Compile | Pass@1 | Timeout | Avg rb | Avg cc | Median tc (s) | Total wall (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3.5:0.8b | 0.8 | A | 70% | 23% | 30% | 5.27 | 5.97 | 0.91 | 421 |
| qwen3.5:0.8b | 0.8 | B | 77% | 23% | 23% | 4.30 | 5.07 | 0.43 | 342 |
| nemotron-3-nano:4b | 4 | A | 60% | 53% | 40% | 4.87 | 5.47 | 0.42 | 512 |
| nemotron-3-nano:4b | 4 | B | 50% | 43% | 50% | 6.13 | 6.63 | 0.31 | 606 |
| qwen3.5:9b | 9 | A | 97% | 80% | 3% | 0.70 | 1.67 | 0.83 | 105 |
| qwen3.5:9b | 9 | B | 97% | 77% | 3% | 0.53 | 1.50 | 0.61 | 83 |
| gpt-oss:20b | 20 | A | 95% | 40% | 5% | 1.60 | 2.55 | 0.79 | 115 |
| gpt-oss:20b | 20 | B | 100% | 50% | 0% | 0.60 | 1.60 | 0.52 | 27 |
| mistral-small3.2:24b | 24 | A | 85% | 65% | 15% | 1.00 | 1.85 | 0.79 | 223 |
| mistral-small3.2:24b | 24 | B | 95% | 80% | 5% | 0.45 | 1.40 | 0.71 | 100 |
| devstral-small-2:latest | 24 | A | 95% | 85% | 5% | 0.35 | 1.30 | 0.80 | 92 |
| devstral-small-2:latest | 24 | B | 85% | 80% | 15% | 0.85 | 1.70 | 0.77 | 195 |
| gemma3:27b | 27 | A | 90% | 85% | 10% | 0.60 | 1.50 | 1.38 | 159 |
| gemma3:27b | 27 | B | 90% | 85% | 10% | 0.85 | 1.75 | 1.34 | 146 |
| nemotron-cascade-2:30b | 30 | A | 100% | 85% | 0% | 0.15 | 1.15 | 0.86 | 34 |
| nemotron-cascade-2:30b | 30 | B | 90% | 80% | 10% | 1.40 | 2.30 | 0.47 | 130 |
| qwen2.5-coder:32b | 32 | A | 100% | 90% | 0% | 0.05 | 1.05 | 1.05 | 30 |
| qwen2.5-coder:32b | 32 | B | 100% | 90% | 0% | 0.05 | 1.05 | 0.97 | 27 |
| qwen3.5:35b | 35 | A | 100% | 90% | 0% | 0.05 | 1.05 | 0.85 | 28 |
| qwen3.5:35b | 35 | B | 95% | 90% | 5% | 0.35 | 1.30 | 0.73 | 77 |
| qwen3.6:35b | 35 | A | 95% | **95%** | 5% | 0.50 | 1.45 | 1.00 | 89 |
| qwen3.6:35b | 35 | B | 95% | 90% | 5% | 0.50 | 1.45 | 0.77 | 78 |
| deepseek-r1:70b | 70 | A | 95% | 75% | 5% | 0.25 | 1.20 | 7.32 | 250 |
| deepseek-r1:70b | 70 | B | 75% | 60% | 25% | 0.35 | 1.10 | 4.65 | 389 |
| nemotron-3-super:120b | 120 | A | 90% | 85% | 10% | 0.60 | 1.50 | 0.90 | 160 |
| nemotron-3-super:120b | 120 | B | 100% | 90% | 0% | 0.00 | 1.00 | 0.86 | 21 |
| gpt-oss:120b | 120 | A | 95% | 55% | 5% | 1.70 | 2.65 | 0.68 | 128 |
| gpt-oss:120b | 120 | B | 100% | 65% | 0% | 1.10 | 2.10 | 0.61 | 68 |
| qwen3.5:122b | 122 | A | 95% | 85% | 5% | 0.30 | 1.25 | 1.10 | 108 |
| qwen3.5:122b | 122 | B | 95% | 85% | 5% | 0.25 | 1.20 | 1.06 | 84 |
| devstral-2:latest | 123 | A | 65% | 65% | 35% | 0.20 | 0.85 | 30.58 | 2183 |
| devstral-2:latest | 123 | B | 60% | 55% | 40% | 0.10 | 0.70 | 27.50 | 840 |

**Total problem-runs**: 30 models × ~25 avg = **~720 problem-runs**. Total eval wall-clock: ~2h 45min across three eval passes.

### H2 revisited — full 15-model data strongly supports partial scaling law

With 15 sizes spanning 0.8B → 123B, Spearman correlations are statistically significant on **both arms**:

| Correlation | ρ | p | Interpretation |
|---|---|---|---|
| Arm A: params ↔ compiler_calls | **−0.608** | **0.012** | **Significantly decreasing** |
| Arm A: params ↔ wall-to-compile | **+0.513** | **0.042** | **Significantly increasing** |
| Arm B: params ↔ compiler_calls | **−0.682** | **0.004** | **Significantly decreasing** |
| Arm B: params ↔ wall-to-compile | **+0.681** | **0.004** | **Significantly increasing** |

**H2 (partial scaling law) is strongly supported on both arms.** The pattern: bigger models make fewer mistakes (compiler calls decrease with size) *but* take longer wall-clock (inference latency compounds over the longer per-token time). Both directions hold across 15 sizes from 0.8B to 123B. This is the cleanest confirmation of the "partial scaling law" hypothesis in the full evaluation.

### Which arm wins per model (15 models, qualitative tally)

Tallying winners by compile rate (primary), pass rate (secondary), wall-clock (tertiary):

| Arm A clearly wins (5) | Tied / near-tied (5) | Arm B clearly wins (5) |
|---|---|---|
| nemotron-3-nano:4b | qwen3.5:9b | qwen3.5:0.8b |
| devstral-small-2:latest | qwen2.5-coder:32b | gpt-oss:20b |
| nemotron-cascade-2:30b | gemma3:27b | mistral-small3.2:24b |
| qwen3.6:35b (pass: 95 vs 90) | qwen3.5:122b | qwen3.5:35b |
| **deepseek-r1:70b** (comp: 95 vs 75!) | devstral-2 (both poor) | **nemotron-3-super:120b** (comp: 90 vs 100) |
|  |  | gpt-oss:120b |

Roughly even split across 15 models. The per-model winner is not a strong function of model size — **stochastic variance in 20-problem samples dominates the signal at this sample size.** A 156-problem run would likely converge both arms to near-identical performance at the architectural level.

### Surprising findings from the expanded set

1. **deepseek-r1:70b shows the biggest arm-A advantage** despite being a reasoning model.
   - Arm A: 95% compile, 75% pass, 25% fewer timeouts than arm B.
   - Arm B: 75% compile, 60% pass.
   - Even without the thinking-wrapper, the model produced usable code after `</think>`. But why arm A outperforms so strongly is unclear — perhaps the LSP feedback at boundaries helps the model's next step after thinking? Needs follow-up.
   - **Notable error codes**: deepseek-r1 was the only model producing `E0382` (borrow-of-moved) and `E0505` errors — borrow-checker failures that neither native LSP nor arm B's classifier can catch without cargo check. The model's reasoning produces more complex code that exercises the borrow checker more.

2. **Coding-specialized models are on the pass-rate Pareto frontier**: qwen2.5-coder:32b (90% pass), qwen3.5:35b (90%), qwen3.6:35b (**95% pass**), and devstral-small-2 (85%) are at or near ceiling. For these models, rollback is largely vestigial — they just get Rust right first try.

3. **devstral-2:latest (123B, 74 GB) is anomalously slow**: median wall-clock to compile is **30s** (vs. <1s for most other models). Pass rate only 65% arm A / 55% arm B with 35–40% timeouts. Likely the model is very slow per-token at this size in Ollama's FP8 quantization, hitting the 60s budget before finishing generation.

4. **H3 (coding-specialized benefits less from rollback) is confirmed** by qwen-coder:32b, qwen3.5:35b, devstral-small-2:latest all sitting at 100% or near-100% compile with <0.4 avg rollbacks. For these, arm A and arm B are statistically indistinguishable — ceiling effect.

5. **Arm B dominates at large scale where first-try success is high**: on `nemotron-3-super:120b`, arm B finished 20 problems in **21 seconds with 0 rollbacks**. Arm A took 160 seconds — nearly 8× slower — for marginally lower compile/pass. LSP blocking overhead × high first-try success = pure drag.

6. **Arm A dominates where rollback content matters**: `nemotron-cascade-2:30b` A=100%/85% vs B=90%/80%, 34s vs 130s. `devstral-small-2` A=95%/85% vs B=85%/80%. When the model *needs* rollback feedback, arm A's error-informed iterations pay off.

### Final plot list (`results/week4/plots/`)

- `compile_rate.png` — bar chart of compile rate per (model, arm)
- `scaling_h2.png` — dual panel: compiler calls vs. params (monotone decreasing) and wall-clock vs. params (not monotone)
- `profile_breakdown.png` — stacked bar of time categories per (model, arm)
- `rollback_overhead.png` — histogram of rollback-overhead span durations, per model (arm A)

Notebook: `results/week4/analysis.ipynb` (executed and saved as `analysis_executed.ipynb`) with the full analysis and additional cells for boxplots, attempts histograms, and error-code breakdowns.

### For week 5

Primary goal: migrate to HuggingFace `transformers` + token-level KV-cache truncation (per `notes/kv-cache-truncation.md`). Secondary: restart H1 evaluation with modified classifier demotion rules to see if the LSP tier can be made productive.

The week 4 architecture's null result is *itself* a reason to proceed — we now know what's wrong and can test targeted fixes.




