# Evaluation of Proposed Research Ideas

Analysis of two proposed approaches for improving LLM code generation soundness, based on a survey of the current academic landscape. Written 2026-04-06.

---

## The Two Proposed Ideas

**Idea 1: Real-time documentation feeder.** A system that retrieves API documentation during generation at high frequency (10-50 times/second), injecting specs into the LLM's context as each line is generated. The LLM is fine-tuned or instructed to request information from the feeder.

**Idea 2: Judge + rollback.** An external verifier (LSP, type checker, linter) runs concurrently with generation, checkpointing at each code segment. If the verifier rejects a segment, all subsequently generated code is erased, the verifier's diagnostic is injected into the prompt, and the LLM regenerates from the last clean checkpoint.

---

## Idea 1: Real-Time Documentation Feeder

### What exists

| System | Relationship | Key difference |
|---|---|---|
| DocPrompting (ICLR 2023) | Retrieves API docs before generation | One-shot, not continuous |
| FLARE (EMNLP 2023) | Active retrieval during generation at sentence granularity | General text, not code; sentence-level, not token/line-level |
| kNN-LM (ICLR 2020) | Token-level retrieval from a datastore during generation | Code tokens in a datastore, not structured API docs |
| kNN-TRANX (2023) | Token-level retrieval for code with syntax constraints | Old seq2tree models, not modern autoregressive LLMs |
| NEST (NeurIPS 2024) | Token-level retrieval + speculative decoding | Focus is speed, not correctness; general text |
| Tok-RAG (ICLR 2025) | Token-level retrieval/LM harmonization framework | Theoretical, tested on QA not code |

### Novelty assessment: **Medium**

The specific application (API docs during code generation) is unstudied, but the mechanism (interleaved retrieval during autoregressive generation) is well-explored. A reviewer would likely see this as "FLARE/kNN-LM applied to API docs" -- an application contribution rather than a methodological one.

### Feasibility concerns

1. **Latency is the killer problem.** 10-50 queries/second means one retrieval per ~1-3 tokens. Transformers generate tokens autoregressively -- injecting new context mid-forward-pass requires either:
   - (a) Prefetching and hoping it arrives in time (event timeline (1)). But to inject new context, you must modify the KV cache, which means either appending docs at a fixed position or recomputing from the injection point. Neither is cheap.
   - (b) Accepting stale context (event timeline (2)). This degrades the value proposition -- if the doc arrives after the line is generated, it's equivalent to a post-hoc check.

2. **Context window bloat.** Continuously injecting API docs grows the prompt rapidly. You'd need a mechanism to evict stale docs, adding complexity.

3. **Marginal benefit over one-shot retrieval is unclear.** DocPrompting showed that a single retrieval pass captures most of the benefit for API correctness. The gain from continuous retrieval needs to be measured against the substantial engineering and latency costs.

4. **Fine-tuning or instruction-tuning for "request" behavior** adds scope. The model must learn to emit retrieval requests at the right times -- this is itself a research problem (related to FLARE's confidence-based trigger).

### Verdict

Not the strongest direction for a course project. The novelty is narrow (known technique + new data source), feasibility challenges are real without clear solutions, and the expected benefit over one-shot retrieval is uncertain.

---

## Idea 2: Judge + Rollback with LSP

### What exists

| System | What it has | What it lacks |
|---|---|---|
| MGD (NeurIPS 2023) | LSP during decoding, semantic constraints | **No rollback** -- forward masking only, at sparse trigger points |
| DSVD (EMNLP 2025) | Token-level rollback with KV-cache truncation | **No external verifier** -- uses internal probing heads, not LSP. Not applied to code |
| IterGen (ICLR 2025) | Grammar-symbol-level backtracking, KV-cache management | **Syntax only** -- no external semantic oracle. User must program semantic checks manually |
| PICARD (EMNLP 2021) | Incremental parsing in beam search | Forward masking only, SQL-specific |
| Type-Constrained (PLDI 2025) | Type inference during decoding | TypeScript subset only, no rollback, no LSP |

### Novelty assessment: **High**

This is the genuinely unstudied combination. The three closest papers each have exactly one of the required components:

```
System        | External semantic verifier | Rollback | Error-informed regeneration
--------------+---------------------------+----------+---------------------------
MGD           | Yes (LSP)                 | No       | No
DSVD          | No (internal probes)      | Yes      | Partial (penalty term)
IterGen       | No (user-programmed)      | Yes      | No (recurrence penalty)
This proposal | Yes (LSP)                 | Yes      | Yes (diagnostic injection)
```

No existing system does all three.

### Feasibility assessment: **Good**

1. **KV-cache checkpointing is solved.** Speculative decoding (Leviathan et al., 2023) already implements "generate speculatively, verify, rollback if wrong" at the token level. IterGen demonstrates grammar-symbol-level KV-cache truncation with no recomputation. The machinery exists.

2. **LSP latency is workable.** From notes/llm-code-soundness.md:
   - rust-analyzer completion: 10-50ms
   - rust-analyzer native diagnostics: 10-50ms
   - LLM generation at ~30 tok/s: ~33ms per token

   The LSP can roughly keep pace. The asynchronous design (let the model run ahead, rollback if needed) elegantly handles cases where it can't. This is the key insight that makes the idea practical.

3. **Partial program tolerance.** rust-analyzer is designed for editors where code is always half-written. Salsa's incremental computation framework handles partial programs efficiently. The main challenge is filtering spurious diagnostics from truly incomplete code vs. errors in "completed" regions.

4. **Checkpoint granularity has natural choices.** Per-statement or per-line are both practical. rust-analyzer's native diagnostics can analyze at these granularities. Statement boundaries are identifiable from the incremental parser.

5. **No retraining required.** Like PICARD, SynCode, and MGD, this is an inference-time-only method. Works with any autoregressive code LLM.

### What makes this a real research contribution

1. **Novel computational model.** Speculative generation + asynchronous external semantic verification + rollback. This can be cleanly formalized. The event timeline model (accept/reject decisions arriving asynchronously) is new in the constrained decoding literature.

2. **Directly addresses identified gap.** Between MGD (forward-only, trigger-point-only) and DSVD (rollback but internal-only). Reviewers of constrained decoding papers will immediately see the contribution.

3. **Empirically testable hypothesis: cascading errors.** How often does an error at line N cause errors at lines N+1, N+2, ...? If cascading is frequent, rollback has high value. If errors are independent, rollback matters less. This is measurable and the answer is independently interesting.

4. **Error-informed regeneration.** Injecting the LSP diagnostic into context before retry gives the model information it didn't have before. Unlike MGD (just masks tokens) or DSVD (re-samples from same distribution), the model learns *why* its output was wrong.

5. **Practical.** No model retraining, no custom tokenizer, no language-specific completion engine (unlike Type-Constrained's 11K lines for TypeScript). The LSP server is a black-box oracle.

### Risks

1. **Spurious diagnostics on partial programs.** Half-finished functions trigger many false errors. Mitigation: only rollback on diagnostics in "completed" code regions (e.g., statements that ended with `;` or `}`). Use rust-analyzer's native diagnostics (fast, 10-50ms) rather than full `cargo check` (2-30s).

2. **Rollback frequency could be too high.** If the model produces errors frequently, constant rollback makes generation very slow. Mitigation: tune checkpoint granularity; implement a "patience" parameter (tolerate N consecutive errors before rolling back); focus evaluation on settings where errors cascade most (multi-function generation with smaller models).

3. **Benefit may be small for strong models.** GPT-4/Claude rarely produce type errors in short functions. Mitigation: target smaller models (CodeLlama-7B, StarCoder-1B, Qwen2.5-Coder-1.5B) and longer/harder generation tasks (multi-function, repository-level) where cascading compounds. IterGen showed dramatic improvements with small models (Qwen2.5-0.5B).

4. **rust-analyzer's completion API behavior on highly partial code** needs empirical investigation. Some positions may return no completions or irrelevant completions. The original proposal already front-loaded this risk in Week 1.

---

## Recommendation: Focus on Idea 2

### Framing

**"Speculative Code Generation with Asynchronous Semantic Verification and Rollback"**

or

**"LSP-Guided Speculative Decoding with Semantic Rollback for Code Generation"**

### What to keep from the original proposal

- **Language: Rust.** rust-analyzer is the best LSP for this purpose (fastest, most mature incremental analysis, tolerant of partial code).
- **Evaluation benchmarks:** HumanEval-Rust, MultiPL-E Rust.
- **Ablation structure:** tokenizer-only, LSP with standard BPE, full system.

### What to change from the original proposal

- **Drop the custom tokenizer.** It's a separate contribution and not needed for the rollback mechanism. The token-grammar alignment problem is already handled by the asynchronous design -- the LSP doesn't need to map completions to individual tokens because it operates at the statement/line level.
- **Add the rollback mechanism** as the central contribution.
- **Add the asynchronous verification model** (the key novelty over MGD).
- **Add error-informed regeneration** (diagnostic injection into context).

### Core mechanism

1. Model generates token-by-token (standard autoregressive).
2. At **checkpoint boundaries** (end of statement/line), snapshot KV-cache state.
3. LSP analyzes the partial program **asynchronously** (in a separate process/thread).
4. If LSP diagnostics return errors in the newly generated code -> **rollback** to last clean checkpoint, inject error diagnostic into context, regenerate.
5. If clean -> advance checkpoint.
6. Optionally: **forward masking** (à la MGD) at trigger points (`.`, function calls) for additional constraint.

### LSP signals to use

- `textDocument/diagnostic` -- type errors, unresolved names, borrow checker violations
- `textDocument/completion` -- valid completions at cursor (for optional forward masking)
- `textDocument/hover` -- type information for context enrichment

### Baselines

1. Unconstrained generation (standard autoregressive)
2. MGD (LSP-constrained, no rollback) -- direct comparison to closest prior work
3. SynCode (CFG-only constraints)
4. Generate-then-fix loop (compile, get errors, retry entire program -- same compute budget)
5. IterGen (grammar-level backtracking with user-programmed semantic checks)

### Key metrics

- **Compilation rate** (syntax + type + borrow checker pass)
- **pass@k** on Rust benchmarks
- **Cascading error rate** (how often rollback prevents downstream errors)
- **Tokens wasted** (generated then rolled back) vs. tokens saved (by not cascading)
- **Wall-clock time** vs. generate-then-fix at equal compute budget
- **Rollback frequency** and **diagnostic filtering accuracy** (false positive rate)

### What makes this publishable

1. Novel computational model at the intersection of three established lines of work (constrained decoding, speculative decoding, LSP-in-the-loop).
2. Directly fills an identified gap in the literature (MGD + rollback + error feedback).
3. Empirically testable cascading error hypothesis.
4. Practical: no retraining, inference-time only, works with any autoregressive LLM.
5. Comparison to strong, recent baselines (MGD, IterGen, SynCode).

---

## Evaluation Benchmarks

### Primary benchmark: MultiPL-E (Rust split)

The most directly applicable benchmark. MultiPL-E translates HumanEval (164 problems) and MBPP (401 problems) from Python to 18 languages including Rust, with language-specific type mappings (`List` -> `Vec`, `int` -> `isize`, `Optional` -> `Option`, etc.). Evaluation uses pass@k with sandboxed test execution.

**Why it fits:** Provides a ready-made Rust test harness with unit tests. Codex achieves pass@1 above 40% on Rust, leaving significant room for improvement from constrained decoding.

**Limitation:** Problems are single-function, self-contained. No cross-file dependencies, crate imports, or complex ownership/lifetime challenges. Does not stress the scenarios where LSP verification adds the most value (cascading type errors, borrow checker violations across statements).

### Secondary benchmark: PRAGMATICCODE-style Rust benchmark (to construct)

MGD's PRAGMATICCODE (100 real-world Java repos, 10,538 dereference prompts) is the most relevant evaluation methodology, but it targets Java. Constructing a **Rust equivalent** would be the most impactful evaluation contribution:

1. Scrape popular Rust crates from crates.io / GitHub that `cargo check` successfully.
2. Extract method-completion prompts at points involving cross-file types, trait implementations, and lifetime-annotated APIs.
3. Measure compilation rate (the primary metric for LSP-guided verification) and identifier match.

This is feasible because Cargo makes builds reproducible, and rust-analyzer provides the same diagnostic signals used during constrained decoding.

### Complementary benchmarks

| Benchmark | Role in evaluation |
|---|---|
| **LiveCodeBench** (Jain et al., 2024) | Contamination guard. Ensures improvements are not from memorized solutions. Time-windowed design filters to problems after model training cutoff. |
| **CRUXEval** (Gu et al., 2024) | Code reasoning diagnostic. Tests whether the model can reason about execution behavior. Models that score poorly here may not benefit from constrained decoding since they lack underlying code reasoning ability. |
| **CrossCodeEval** (Ding et al., 2023) | Motivates why LSP matters. Shows in-file context alone achieves only 8.82% EM on cross-file completions; cross-file retrieval yields 3x improvement. LSP provides exactly this cross-file signal, but with semantic precision rather than lexical similarity. |
| **RepoBench** (Liu et al., 2023) | Demonstrates retrieval quality is the bottleneck in repository-level completion. LSP's structured analysis (go-to-definition, type hover) could outperform bag-of-tokens retrieval methods. |

### Metrics specific to this project

Beyond standard pass@k and compilation rate, the project introduces metrics that measure the **value of rollback specifically**:

| Metric | What it measures | How to compute |
|---|---|---|
| **Cascading error rate** | How often an error at line $N$ causes errors at lines $N+1, N+2, \ldots$ | For each rollback event, continue generation *without* rollback on a separate branch and count subsequent errors. Compare error count with vs. without rollback. |
| **Tokens wasted** | Tokens generated then discarded by rollback | Total tokens generated minus tokens in final output. Lower = more efficient verification. |
| **Rollback precision** | Fraction of rollbacks that actually prevented cascading errors | Among all rollback events, how many would have led to downstream errors if not rolled back. |
| **Diagnostic filtering accuracy** | False positive rate of LSP diagnostics on partial code | Fraction of LSP diagnostics on partial code that are spurious (caused by incompleteness, not real errors). |
| **Compute-normalized comparison** | pass@k at equal total compute budget | Compare: (a) rollback system generating $n$ tokens with $r$ rollbacks vs. (b) generate-then-fix loop using the same total token budget across full-program retries. |

---

## Comparison of All Approaches

| Approach | Intervention timing | Granularity | Constraint type | Rollback | Retraining |
|---|---|---|---|---|---|
| Grammar-guided (SynCode, Outlines) | Every token | Token | Syntax (CFG) | No | No |
| PICARD | Every token (beam) | Token | Syntax + SQL semantics | No | No |
| Synchromesh | Every token | Token | Syntax + semantics (CE) | No | No |
| MGD | Trigger points | Token | Semantic (LSP types) | No | No |
| IterGen | Grammar symbols | Grammar symbol | Syntax + user-programmed | Yes | No |
| Type-Constrained | Every token | Token | Types | No | No |
| DSVD | Every token | Token (window) | Factual (probing heads) | Yes | Probing heads |
| **This proposal** | Statement/line | Statement/line | Semantic (LSP) | Yes | No |

The proposal occupies a unique position: external semantic verification (like MGD) + rollback (like DSVD/IterGen) + error-informed regeneration (novel) + asynchronous design (novel).
