# SoundCode → publishable paper: a plan

**Compiled 2026-05-17 from a literature scan and a venue-calendar survey.**

This document is a working plan, not a sales pitch. It tries to be honest
about what is novel, what is not, what would have to be true for a NeurIPS-
class submission, and what realistic alternative venues exist.

---

## TL;DR

- The core mechanism is **closely** anticipated by two papers — **ROCODE**
  (Jiang et al., arXiv:2411.07112, Nov 2024) and **SemGuard** (Tian et al.,
  arXiv:2509.24507, Sept 2025). Both do statement / line-level mid-stream
  rollback during code generation. SoundCode is best framed as "ROCODE for
  Rust, **made asynchronous**, with the **real compiler / LSP** as the
  verifier, and a **last-surviving-checkpoint** rollback policy" — none of
  those four pieces is individually novel, but the combination is open.
- The week-4 result that **0 / 926 LSP calls were BLOCKING** (the
  classifier demoted everything to INCOMPLETE / NON_BLOCKING) is the most
  serious threat to the whole thesis. Any paper has to explain why the
  apparatus pays for itself. Without a fix, the work *strictly* doesn't
  beat aggressive resampling.
- Every 2026 main-conference deadline that's a good fit is **already
  past**. The two windows that line up with the remaining quarter are:
    1. **NeurIPS 2026 workshops** (paper deadlines ~Aug 29, 2026; suggested)
       — best venue fit, ~14 weeks away.
    2. **EMNLP 2026 Industry Track** (Jun 16, 2026) — only conference
       deadline in the quarter window; weaker venue fit.
- **Recommended path**: target a NeurIPS 2026 workshop (DL4C-successor or
  ML-for-Code / ML-for-Systems variant). Use the quarter to ship a
  defensible workshop paper; iterate during summer toward **ICSE 2027
  NIER** (Oct 23, 2026, 4 pages) or **MLSys 2027** (~Oct/Nov 2026, full
  paper) for a real conference outcome.

---

## 1. What you actually have, in one paragraph

A working end-to-end LSP-supervised Rust code-generation pipeline: producer-
consumer loop with boundary-based checkpointing (`;` and `}` at depth 0),
async `cargo check` / rust-analyzer verification, structural rollback to
the last surviving checkpoint, four orchestration modes for reasoning
models (RAW / RAW_THINK_INJECT / TWO_PHASE / CHAT_INSTRUCTED), 15 Ollama
models from 4B to 125B, ~8,500 lines of tested Python + JS, 78 unit /
playwright tests, JSONL run logging, and a live demo at
`https://soundcode-demo.psixyzt.com`. Engineering is solid. Evaluation is
not — week 4 produced indicative numbers across 15 × 2 × 30 = 900 runs
but the 2 arms compared an early baseline against an early prototype, and
the headline finding (MGD didn't transfer well from Java to Rust) is more
about MGD than about the current rollback design.

---

## 2. Related work — the actual neighbors

Two papers occupy nearly the same design point as SoundCode:

### ROCODE (Jiang et al., 2024) — direct competitor #1
arXiv:2411.07112. Streams tokens, runs program analysis after each
statement, on detected error rolls back the buffer and resumes with
**exponentially decaying token-penalty masking** on the bad span. Targets
Python and C++. Evaluates on HumanEval, MBPP, CodeForces2305, HumanEval-CPP
across nine Llama-2 / CodeGen / CodeLlama models. Reports ~19% token-cost
reduction vs. post-revising baselines.

### SemGuard (Tian et al., 2025) — direct competitor #2
arXiv:2509.24507. Lightweight learned semantic evaluator scores each
completed line; if score ≤ 0.5, backtracks to the start of the line and
resamples up to *N* times with a token penalty. Beats ROCODE by +2.2
pass@1 with 31% fewer tokens and 60% lower latency on MBPP / LiveCodeBench /
SemDiff (Python and Java), using 7B-scale Code LMs.

### The constrained-decoding family (verifier on the critical path)

| Paper | Verifier | Granularity | Async? | Rollback? |
|---|---|---|---|---|
| PICARD (Scholak et al., EMNLP 2021) | hand-written parser | per-token, beam-rerank | no | no (rejection) |
| Synchromesh (Poesia et al., ICLR 2022) | hand-coded DSL semantic engine | per-token | no | no |
| Grammar-Constrained Decoding (Geng et al., 2023) | CFG / EBNF | per-token | no | no |
| MGD (Agrawal et al., NeurIPS 2023) | LSP completions (via multilspy) | per-token, at triggers | no | no |
| DOMINO (Beurer-Kellner et al., ICML 2024) | grammar, tokenizer-aligned | per-token, ~0% overhead | no | no |
| Type-Constrained Code Generation (Mündler et al., PLDI 2025) | formal type system | per-token (sound) | no | no |
| IterGen (Ugare et al., ICLR 2025) | grammar | per grammar-symbol, with KV-cache backward nav | no | yes (grammar-level) |
| **ROCODE** (Jiang et al., 2024) | compiler / static analyzer | per-statement | **no** | **yes** |
| **SemGuard** (Tian et al., 2025) | learned semantic evaluator | per-line | no | yes |
| **SoundCode (you)** | LSP + `cargo check` | per-`;`/`}` boundary | **yes** | yes |

### The iterative-repair family (verifier off the critical path)

| Paper | Verifier | Granularity | Mid-stream? | Notes |
|---|---|---|---|---|
| Self-Refine (Madaan et al., NeurIPS 2023) | LLM self-critique | whole solution | no | no external signal |
| Self-Debug (Chen et al., ICLR 2024) | execution trace | whole solution | no | re-prompts with traces |
| Reflexion (Shinn et al., NeurIPS 2023) | task pass/fail + verbal | whole trajectory | no | episodic memory |
| CodeT (Chen et al., 2022) | generated tests, dual-exec voting | parallel resample | no | the resampling baseline |
| CodeRL (Le et al., NeurIPS 2022) | unit tests as RL reward | training-time | no | predicts bug-start span |
| INTERVENOR (Wang et al., ACL Findings 2024) | compiler errors → NL repair | whole solution | no | "Code Teacher" + "Code Learner" |
| LATS (Zhou et al., ICML 2024) | environment feedback + self-eval | MCTS over trajectories | no | heaviest TTC approach |
| AlphaCodium (Ridnik et al., 2024) | generated public + AI tests | whole solution flow | no | 19% → 44% on CodeContests |
| MGDebugger (Shi et al., 2024) | LLM-simulated executor | per subfunction in tree | no | post-hoc, hierarchical |
| SWE-Agent / AutoCodeRover | env. + tests | patch / repo level | no | repository scale |

The pivotal counter-result is **Olausson et al., "Is Self-Repair a Silver
Bullet?" (ICLR 2024, arXiv:2306.09896)**: across HumanEval / APPS,
self-repair only beats matched-budget i.i.d. resampling when the feedback
model is strictly stronger than the generator. Same-model self-critique
roughly cancels its own cost. This is the bar any rollback paper must
clear, and the bar that SoundCode's "swap LLM critique for an actual
compiler" framing is designed to clear.

### Control-flow neighbors (no semantics, but same loop topology)

| Paper | Notes |
|---|---|
| Speculative Decoding (Leviathan et al., ICML 2023) | draft model proposes, target verifies in parallel, rollback on rejection. **Pure logit verifier; no semantics; latency mechanism.** |
| Speculative Sampling (Chen et al., 2023) | DeepMind variant of the same idea. |
| Lookahead Decoding (Fu et al., 2024) | parallel verify branch *inside* the same LLM via Jacobi; structural blueprint for "verify in parallel + rollback." |

SoundCode's right framing is "speculative decoding with a *semantic*
verifier and a *structural* rollback unit," not "MGD with rollback."

### Adjacent / supporting

CodeFusion (diffusion-based AR code; can revise earlier tokens
architecturally), Coder-Reviewer Reranking (bidirectional likelihood),
LEVER (learned execution-conditioned verifier for reranking), Coffee
(arXiv:2311.07215), StepCoder (compiler-feedback RL).

---

## 3. What is genuinely novel — uncharitable read

**Overlapping with prior art:**

- Statement-boundary intervention → ROCODE has this exactly (Python, C++).
- Rollback-on-error → ROCODE, SemGuard, IterGen.
- Compiler-as-verifier → ROCODE, INTERVENOR, CoCoGen (Bi et al., ACL
  Findings 2024).
- Parallel verify-then-rollback control flow → Lookahead Decoding,
  Speculative Decoding.
- LSP feedback into a code LM → MGD is the canonical reference.
- Targeting Rust → CRUST-Bench (Khatry et al., 2025), RustRepoTrans (Ou et
  al., ASE 2025), C-to-Rust translation feedback loops (arXiv:2512.02567).

**What appears genuinely uncovered:**

1. **Asynchronous verification at structural boundaries.** Every prior
   boundary-rollback system (ROCODE) is synchronous-blocking; every
   async-parallel-verify system (Lookahead, Speculative) uses the LLM
   itself as verifier. Compiler off the critical path + speculation past
   a checkpoint until verification returns is the architectural novelty.
   The *reason* it's novel is also a virtue of the project: Rust's
   `cargo check` is too expensive (~100 ms – several seconds) for
   ROCODE-style synchronous blocking, but cheap enough to run in
   parallel with generation — Python/C++ verifiers are too fast to
   bother with async; Rust is in the sweet spot.

2. **Last-surviving-checkpoint rollback policy.** ROCODE rolls back to
   the *error site* (or highest-entropy token); SoundCode rolls back to
   the *last cargo-verified prefix*. Different correctness properties:
   SoundCode never re-emits past a checkpoint without a green light, so
   the failing tail is bounded by checkpoint depth and a "ratchet"
   property holds (no regression past the last verified offset).

3. **Rust as target.** Rust's borrow checker, trait resolution, and
   lifetime errors are the canonical "you can't catch this with a CFG
   or a Python AST" class of errors. No paper combines these signals
   with mid-stream rollback. SemGuard's *learned* evaluator strictly
   cannot match this (it has no formal type theory inside it).

4. **The four-mode orchestration matrix for thinking models** in a
   rollback context. Each mode (R0 / R1 / T2 / C3) has a distinct
   prompt protocol; you've shown empirically that they trade off
   reliability, latency, and code-phase cleanliness in non-obvious
   ways. No paper in the survey examines this interaction.

**Honest minimum bar for a method-paper contribution:** any one of (1)
or (3) alone, with rigorous evaluation, is publishable. Both together
plus a credible Olausson-style matched-budget comparison is enough for
a NeurIPS workshop. (1) + (2) + (3) + (4) with a strong evaluation grid
could plausibly clear a main-track bar at ICSE / FSE / MLSys.

---

## 4. The threat that has to be addressed

**Week 4 finding: classifier demoted 926 / 926 LSP calls to non-blocking,
zero rollbacks fired.** If the verifier never blocks, the entire apparatus
is overhead. Three possible resolutions, in order of preference:

1. **The classifier was too permissive on Rust-specific failure modes.**
   Re-audit the BLOCKING / INCOMPLETE / NON_BLOCKING decision rules
   against a manually-labelled subset of cargo / rust-analyzer
   diagnostics. Likely the INCOMPLETE bucket is absorbing real type
   errors that *should* trigger rollback (E0282/E0283 — type annotations
   needed — were flagged INCOMPLETE in the current code, but some of
   these are genuine bugs, not "wait for more tokens" signals).
2. **The week-4 prompts were too easy.** Run a harder benchmark
   (RustEvo² post-cutoff, the harder MBPP-Rust slices) where blocking
   errors should be more common.
3. **The mechanism is wrong** — rollback at any granularity doesn't
   help because compile errors are too rare on instruct-tuned coders.
   This is the worst case but plausible: large coders compile-pass on
   HumanEval-Rust >85% of the time without supervision, so a
   verification system has limited headroom unless evaluated on
   small or non-coder models.

The paper has to (a) measure the rollback-firing rate per (model,
benchmark) and (b) show that compile-pass rate goes up where rollback
fires, OR that the system's a no-op overhead-wise when it doesn't fire.
Either is publishable; neither has been demonstrated.

---

## 5. Next steps to enrich the project

Prioritized. Each item lists estimated effort, dependencies, and what it
unlocks for which venue.

### Tier 1 — must-have for any submission

**5.1 Re-audit the strikedown classifier (1-2 days).**
Manually label 100-200 real cargo / rust-analyzer diagnostics from the
existing `results/web_demo/*.jsonl` runs as truly-blocking vs.
mid-edit vs. warning. Re-tune the classifier. Re-measure rollback fire
rate on week-4 prompts. This is the cheapest experiment with the
highest leverage. *Unlocks*: defensible claim that the verifier
actually intervenes.

**5.2 Real evaluation grid (1-2 weeks of compute, 1 week of writeup).**
Following the methodology-survey recommendation:

| Axis | Values |
|---|---|
| Benchmarks | MultiPL-E HumanEval-rs (156 problems), MultiPL-E MBPP-rs (354 problems), RustEvo² post-cutoff slice (~150 problems) |
| Open models | Qwen2.5-Coder-7B-Instruct, Qwen2.5-Coder-32B-Instruct, StarCoder2-15B, CodeLlama-13B-Instruct, DeepSeek-Coder-V2-Lite |
| Closed reference | Claude Sonnet 4.x or GPT-4o (T=0, n=1 — reference only) |
| Conditions | A. Vanilla (greedy + nucleus, no verifier). B. Cargo-only rollback. C. SoundCode (LSP + cargo). D. Reflexion-style whole-solution retry (matched token budget). E. ROCODE re-implementation (synchronous, token-penalty masking). |
| Metrics (primary) | pass@1 (n=50, T=0.2), pass@10 (n=200, T=0.8), compile-pass rate |
| Metrics (secondary) | tokens-per-success, wall-clock-per-success, pass@k @ matched-token-budget |
| Statistics | 95% bootstrap CIs over problems (B=10,000), paired sign-flip permutation test for A-vs-C and C-vs-D |
| Hardware | 8×H100 with vLLM batching (open models); API calls (closed reference) |

This is the unambiguous spend that turns the project from "engineering
project" into "experimental ML paper." Order-of-magnitude cost: ~2M
generations for the open headline; runs in ~1-2 weeks on a single 8×H100
box.

*Unlocks*: NeurIPS / EMNLP / MLSys / ICSE workshop submission. Without
this, no venue beyond a demo track is realistic.

**5.3 Olausson-style matched-budget comparison (~3 days; piggybacks on 5.2).**
Re-plot results not as "method X vs. method Y at fixed N", but as
"pass@k at matched token budget" (Pareto curves of pass@1 vs. tokens
spent). This is what reviewers in this area now demand after Olausson
et al. (ICLR 2024) showed that fixed-N comparisons are misleading.

### Tier 2 — strong-to-have for a method paper

**5.4 Implement ROCODE on Rust as a head-to-head baseline (~3-5 days).**
ROCODE is the direct competitor; the paper *must* compare. Re-implement
ROCODE's "compile after each statement, on error mask the bad span with
exp-decaying penalty" mechanism using your existing `Code` /
`CargoChecker` plumbing — most of the code can be reused. This is the
single highest-leverage related-work response.

*Unlocks*: defensible "we beat ROCODE" claim if the async / LSP combo is
real; or, if not, a publishable "we replicate ROCODE on Rust and find
that async doesn't help in this regime" negative result.

**5.5 Add Python + pyright as a second language (~1 week).**
Your `multilspy` plumbing already supports it. Even 100 HumanEval-Python
problems would let you claim the method "generalizes beyond Rust." If
the rollback policy fails on Python, that's a publishable negative
result about the role of static-type richness in mid-stream
verification.

*Unlocks*: stronger generality claim, blunts the "this is just a
Rust-engineering paper" review.

**5.6 Bouncing-detector / adaptive rollback policy (~3 days).**
You've already observed that fine-grained rollback can bounce in a
narrow offset window (week 5 finding). Implement N-consecutive-near-
offset detection that escalates to coarser rollback (halve survivor,
jump to 0). This is a small but interesting methodological contribution
that distinguishes you from ROCODE's single-policy approach. Measure
how often it fires and whether it changes outcomes.

### Tier 3 — nice to have, polish

**5.7 Complete the chat-template registry (~1 day per family).**
Currently only Qwen3.x is in `CHAT_TEMPLATES` (used by T2 mode). Add
gpt-oss, deepseek-r1, nemotron-cascade-2, nemotron-3. This widens the
empirical surface for the four-mode analysis.

**5.8 Theoretical framing paragraph (~2 days of writing).**
A paper doesn't need a theorem, but it needs a paragraph that says
"rollback is justified because *X*, and *X* is non-obvious." Candidate
framings: (i) compute trade-off (rollback amortizes verifier cost vs.
rejection sampling); (ii) decoding-as-search (your verifier is a more
semantic acceptance check); (iii) uncertainty localization (rollback
target encodes where the model went wrong in a way pass/fail signals
don't). Pick one and defend it.

**5.9 Clippy-clean rate as a novel Rust-specific metric (~1 day).**
None of the surveyed papers reports this. Easy to add, gives you an
extra axis where reviewers can see your method shine independent of
correctness.

**5.10 Code-LM contamination check (~1 day).**
Standard hygiene for any 2026 code-LM paper. Run BigCode's contamination
detector or report per-model release dates vs. benchmark publication
dates so reviewers don't ask.

### Compute / wall-clock estimate

If all of Tier 1 + 5.4 (ROCODE re-implementation) gets done:
- Coding: ~2-3 weeks of focused work.
- Eval compute: ~2 weeks on 1× 8×H100 box.
- Writing: ~2 weeks.
- **Realistic timeline**: 6-8 weeks of half-time work, i.e., the rest of
  this quarter plus a slice of summer. Workshop-paper-ready by end of
  August 2026. Conference-paper-ready by October 2026.

---

## 6. Submission venues — calendar reality

As of 2026-05-17, **every 2026 main-conference deadline that's a good fit
is already past**. The actionable ones:

### Realistic targets in window

| Venue | Deadline | Length | Fit | Notes |
|---|---|---|---|---|
| **NeurIPS 2026 workshops** | ~Aug 29, 2026 (suggested; varies per workshop) | 4-8 pages | ★★★★★ | **Primary target.** DL4C-successor, ML4Code, ML for Systems, Foundation Models for Decision Making. Workshop list announced Jul 11. |
| **EMNLP 2026 Industry Track** | Jun 16, 2026 | 6 pages | ★★★★☆ | Only conference deadline in the quarter window. Weaker venue fit (NLP framing > systems framing) but achievable in 4 weeks. |
| **ICSE 2027 NIER** | Oct 23, 2026 | 4 pages | ★★★★★ | "New Ideas and Emerging Results" — explicitly the right venue for "engineering with promising empirical seed." Best 2026-cycle conference outcome. |
| **FSE 2027 Research, Cycle 1** | Oct 2, 2026 | 10+2 pages | ★★★★★ | Full conference paper with major-revision cycle (friendly to in-progress work). |
| **MLSys 2027** | ~Oct/Nov 2026 (mirrors MLSys 2026: Oct 30, 2025) | 10 pages | ★★★★☆ | Best ML-systems venue for "LSP-supervised decoding" as a systems contribution. |
| **TMLR** | Rolling | flexible | ★★★★☆ | Claims-based; no novelty/impact bar. 4-week review cycle. Real archival publication. |
| **ICLR 2027** | ~Sep 22-24, 2026 (predicted from ICLR 2026 dates) | 10 pages | ★★★★☆ | Main-track target if Tier 1 + 5.4 land by September. Tight. |
| **ACL 2027 via ARR** | ARR Oct 12, 2026 cycle | 8 pages | ★★★★☆ | If you can re-frame as NLP-flavoured. |

### Deadlines already past (don't bother this cycle)

| Venue | Deadline | Note |
|---|---|---|
| NeurIPS 2026 Main + Eval/Datasets | May 6, 2026 | Missed by 11 days. |
| COLM 2026 | Mar 31, 2026 | Missed. |
| ICLR 2026 | Sep 24, 2025 | Decisions out. |
| ICML 2026 | Jan 29, 2026 | Missed. |
| ASE 2026 Research | Apr 15, 2026 | Missed. |
| ASE 2026 NIER / Tools | May 12, 2026 | Missed by 5 days. |
| AAAI 2027 Main | Abstract Jul 21 / full Jul 28, 2026 | Too tight given current state. |

### Recommended path

1. **By end of quarter (mid-June 2026)**: ship Tier 1 (re-audit + real
   eval grid + Olausson-style matched-budget plot). Have a defensible
   4-6 page workshop paper draft.
2. **Mid-August 2026**: submit to a NeurIPS 2026 workshop. Use the
   intervening time to add Tier 2 items (ROCODE baseline, optionally
   Python+pyright) so the workshop paper extends naturally.
3. **October 2026**: if Tier 2 lands, submit a full version to
   **ICSE 2027 NIER** (Oct 23) and/or **FSE 2027 Research Cycle 1**
   (Oct 2). MLSys 2027 (~late Oct/Nov) is the back-up for the
   systems framing.
4. **Spring 2027**: with summer + fall expansion, target NeurIPS 2027
   main track (~May 2027) as the eventual "main story" publication.

### What to avoid

- **NeurIPS 2026 main track** — the deadline is past, and even if it
  weren't, the work is workshop-level not main-track-level right now.
- **AAAI 2027** — fit is poor for the systems-y framing; deadline too
  tight.
- **ICLR 2027 main track as primary near-term target** — possible but
  high-risk. Better as a fall-back if the workshop paper lands well.
- **Demo-only venues** as the *only* outcome — the live demo is
  legitimately strong (NeurIPS Demo Track would take it), but a demo
  track paper doesn't carry the same weight as a research track paper
  on the academic market.

---

## 7. The pitch, in one paragraph (for whichever venue)

> *"We introduce SoundCode, a speculative-decoding-style framework for
> LLM code generation where a semantic verifier (cargo check + LSP)
> runs asynchronously in parallel with the language model and triggers
> structural rollback to the last verified checkpoint on a blocking
> diagnostic. Unlike ROCODE (Jiang et al., 2024), which runs the
> verifier synchronously on the critical path and is therefore limited
> to fast verifiers (Python's compile(), g++ -fsyntax-only), our async
> design makes Rust's expensive borrow / trait / lifetime checks usable
> mid-stream. Unlike Monitor-Guided Decoding (Agrawal et al., NeurIPS
> 2023), which masks per-token logits to prevent invalid completions,
> we accept that bad prefixes will be generated and recover by
> rollback — this catches multi-token errors (borrow violations,
> trait-resolution failures) that no per-token mask can prevent. On
> MultiPL-E HumanEval-rs and MBPP-rs across 5 open code LMs, SoundCode
> improves pass@1 by X.X% over vanilla decoding at matched token
> budget and Y.Y% over a Reflexion-style whole-solution retry baseline,
> while reducing tokens-per-success by Z.Z%. We additionally study a
> four-mode interaction matrix for reasoning-model orchestration and
> show that two-phase thinking (chat-template-wrapped thinking trace
> baked into the prompt as a comment) is the only mode that preserves
> the rollback protocol without sacrificing thinking visibility."*

Numbers TBD by the experiments in §5.

---

## 8. References

### Direct competitors
- Jiang, X. et al. (2024). *ROCODE: Integrating Backtracking Mechanism and Program Analysis in LLMs for Code Generation*. arXiv:2411.07112.
- Tian, Z. et al. (2025). *SemGuard: Real-Time Semantic Evaluator for Correcting LLM-Generated Code*. arXiv:2509.24507.

### Constrained-decoding family
- Agrawal, L. A. et al. (2023). *Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context*. arXiv:2306.10763. NeurIPS 2023.
- Scholak, T., Schucher, N., Bahdanau, D. (2021). *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models*. arXiv:2109.05093. EMNLP 2021.
- Poesia, G. et al. (2022). *Synchromesh: Reliable Code Generation from Pre-trained Language Models*. arXiv:2201.11227. ICLR 2022.
- Geng, S. et al. (2023). *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*. arXiv:2305.13971. EMNLP 2023.
- Beurer-Kellner, L. et al. (2024). *Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation [DOMINO]*. arXiv:2403.06988. ICML 2024.
- Mündler, N. et al. (2025). *Type-Constrained Code Generation with Language Models*. arXiv:2504.09246. PLDI 2025.
- Ugare, S. et al. (2025). *IterGen: Iterative Semantic-aware Structured LLM Generation*. ICLR 2025. OpenReview ac93gRzxxV.
- Bi, Z. et al. (2024). *Iterative Refinement of Project-Level Code Context for Precise Code Generation with Compiler Feedback [CoCoGen]*. arXiv:2403.16792. ACL Findings 2024.

### Control-flow neighbors
- Leviathan, Y., Kalman, M., Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. arXiv:2211.17192. ICML 2023.
- Chen, C. et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*. arXiv:2302.01318.
- Fu, Y. et al. (2024). *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*. arXiv:2402.02057.

### Iterative-repair family
- Chen, X., Lin, M., Schärli, N., Zhou, D. (2024). *Teaching Large Language Models to Self-Debug*. arXiv:2304.05128. ICLR 2024.
- Madaan, A. et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. arXiv:2303.17651. NeurIPS 2023.
- Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. arXiv:2303.11366. NeurIPS 2023.
- Chen, B. et al. (2022). *CodeT: Code Generation with Generated Tests*. arXiv:2207.10397.
- Le, H. et al. (2022). *CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning*. arXiv:2207.01780. NeurIPS 2022.
- Zhou, A. et al. (2023). *Language Agent Tree Search Unifies Reasoning Acting and Planning [LATS]*. arXiv:2310.04406. ICML 2024.
- Wang, H. et al. (2023). *INTERVENOR: Prompting the Coding Ability of LLMs with the Interactive Chain of Repair*. arXiv:2311.09868. ACL Findings 2024.
- **Olausson, T. X. et al. (2023). *Is Self-Repair a Silver Bullet for Code Generation?* arXiv:2306.09896. ICLR 2024.**
- Ridnik, T., Kredo, D., Friedman, I. (2024). *Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering*. arXiv:2401.08500.
- Shi, Y. et al. (2024). *From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging [MGDebugger]*. arXiv:2410.01215.
- Yang, J. et al. (2024). *SWE-Agent: Agent-Computer Interfaces Enable Automated Software Engineering*. arXiv:2405.15793. NeurIPS 2024.
- Zhang, Y. et al. (2024). *AutoCodeRover: Autonomous Program Improvement*. arXiv:2404.05427. ISSTA 2024.
- Xia, C. S., Zhang, L. (2024). *Automated Program Repair via Conversation [ChatRepair]*. arXiv:2304.00385. ISSTA 2024.
- Bouzenia, I., Devanbu, P., Pradel, M. (2024). *RepairAgent: An Autonomous, LLM-Based Agent for Program Repair*. arXiv:2403.17134. ICSE 2025.
- Ni, A. et al. (2023). *LEVER: Learning to Verify Language-to-Code Generation with Execution*. arXiv:2302.08468. ICML 2023.

### Benchmarks and methodology
- Cassano, F. et al. (2023). *MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation*. arXiv:2208.08227. IEEE TSE 2023.
- Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code [HumanEval / pass@k]*. arXiv:2107.03374.
- Liu, J. et al. (2023). *Is Your Code Generated by ChatGPT Really Correct? [EvalPlus]*. NeurIPS 2023.
- Liang, S. et al. (2025). *RustEvo²: An Evolving Benchmark for API Evolution in LLM-based Rust Code Generation*. arXiv:2503.16922.
- Ou, R. et al. (2025). *RustRepoTrans: Repository-level Code Translation Benchmark Targeting Rust*. arXiv:2411.13990. ASE 2025.
- Khatry, A. et al. (2025). *CRUST-Bench: A Comprehensive Benchmark for C-to-safe-Rust Transpilation*. arXiv:2504.15254.
- Jain, N. et al. (2024). *LiveCodeBench: Holistic and Contamination Free Evaluation of LLMs for Code*. arXiv:2403.07974.
- Zhuo, T. Y. et al. (2025). *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions*. arXiv:2406.15877. ICLR 2025.
- Agarwal, R. et al. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. arXiv:2108.13264. NeurIPS 2021.
- Hui, B. et al. (2024). *Qwen2.5-Coder Technical Report*. arXiv:2409.12186.
- Lozhkov, A. et al. (2024). *StarCoder 2 and The Stack v2*. arXiv:2402.19173.
- Zhu, Q. et al. (2024). *DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence*. arXiv:2406.11931.
- Evtikhiev, M. et al. (2023). *Out of the BLEU: How Should We Assess Quality of the Code Generation Models?* arXiv:2208.03133.

### Sources
- [NeurIPS 2026 Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers)
- [NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops)
- [EMNLP 2026 Industry Track](https://2026.emnlp.org/calls/industry_track/)
- [ICLR 2026 Dates](https://iclr.cc/Conferences/2026/Dates)
- [ICML 2026 Workshops (incl. DL4C)](https://blog.icml.cc/2026/04/06/announcing-the-icml-2026-workshops-and-affinity-workshops/)
- [COLM 2026](https://colmweb.org/cfp.html)
- [ACL Rolling Review](http://aclrollingreview.org/dates)
- [MLSys 2026](https://mlsys.org/Conferences/2026/Dates)
- [ICSE 2027 Research Track](https://conf.researchr.org/track/icse-2027/icse-2027-research-track)
- [FSE 2027 Research](https://conf.researchr.org/track/fse-2027/fse-2027-papers)
- [TMLR](https://jmlr.org/tmlr/)
- [MultiPL-E (GitHub)](https://github.com/nuprl/MultiPL-E)
- [EvalPlus (GitHub)](https://github.com/evalplus/evalplus)
- [Big Code Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)

---

## 9. Bottom line

**The work has a defensible niche** — async parallel verification with a
real compiler + LSP, statement-boundary rollback, on Rust — but the
combination isn't *obviously* novel after a careful read of ROCODE
(2024), SemGuard (2025), MGD (NeurIPS 2023), and Lookahead Decoding
(2024). The contribution has to be defended on three fronts: (a) the
async design is forced by Rust's verifier cost and unaddressed by
synchronous prior work; (b) the LSP / real-compiler signal catches
errors that learned evaluators and grammars cannot; (c) the four-mode
orchestration interaction with thinking-model reasoning is unstudied.
Each of these needs a real experiment.

**Realistic outcome for this quarter**: a defensible NeurIPS 2026
workshop submission (~Aug 29) extending the current implementation with
the eval grid in §5.2 and the ROCODE re-implementation in §5.4.

**Realistic outcome for fall 2026**: ICSE 2027 NIER (Oct 23, 4 pages)
and/or FSE 2027 Cycle 1 (Oct 2, full paper).

**Realistic outcome for 2027**: ICLR 2027 (~Sep 2026) or NeurIPS 2027
(~May 2027) main track, contingent on the workshop / NIER paper landing
well and being extended with Python + pyright generalization (§5.5) and
a proper theoretical framing (§5.8).

What's *not* realistic right now is a NeurIPS 2026 main-track
publication. That ship sailed eleven days ago.
