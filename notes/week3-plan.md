# Week 1 Plan: Feasibility Validation + Infrastructure

Goal: de-risk the core mechanism and build the infrastructure foundation.

---

## 1. Headless rust-analyzer client (Days 1-3)

Most critical dependency. Everything else builds on it.

- Write a Python library that spawns rust-analyzer over stdio, speaks LSP JSON-RPC, and supports:
  - `initialize` / `initialized` handshake
  - `textDocument/didOpen`, `textDocument/didChange` (incremental edits)
  - `textDocument/diagnostic` (pull diagnostics, LSP 3.17+)
  - `textDocument/completion` (for optional forward masking later)
- MGD's [`multilspy`](https://github.com/microsoft/monitors4codegen) already does this for Java/Python/C#/Rust. **Start by reading `multilspy`'s rust-analyzer integration** and decide: fork it, wrap it, or write from scratch. It handles lifecycle, incremental updates, and completion queries. The main question is whether it supports pull diagnostics (likely not -- MGD only uses completions).
- Wrap the client in a clean async interface: `update_file(content) -> None`, `get_diagnostics() -> List[Diagnostic]`, `get_completions(line, col) -> List[CompletionItem]`.
- Test against a real Cargo project. Verify you can open a file, push incremental edits, and receive diagnostics.

## 2. Partial-program diagnostic experiment (Days 2-4)

**Key feasibility risk.** rust-analyzer is designed for half-written code in editors, but we need to understand its behavior systematically.

- Create a Cargo project with 20-30 Rust functions of varying complexity (arithmetic, string manipulation, Vec/HashMap usage, trait implementations, lifetime-annotated functions).
- For each function, simulate incremental generation: feed the function line-by-line (or statement-by-statement) to rust-analyzer via the headless client. At each step, record:
  - Number of diagnostics returned
  - Which are **real errors** (type mismatch, unresolved name, borrow violation in completed code) vs. **spurious** (caused by incompleteness -- missing closing brace, incomplete expression, unresolved future reference)
  - Latency of each diagnostic request
- Build a small dataset: `(partial_program, diagnostics, spurious_or_real)`. This directly informs the diagnostic filtering strategy and measures the false positive rate -- one of the key metrics.
- **Specific things to test:**
  - Does rust-analyzer give useful diagnostics after a complete statement but before the function is closed?
  - What happens with `let x: Vec<i32> = vec![1, 2, 3];` followed by an incomplete next line?
  - How does it handle incomplete match arms, incomplete if/else, incomplete closures?

## 3. Model inference setup (Days 3-5)

- Set up local inference for a small model: **Qwen2.5-Coder-1.5B** (fits on a single GPU, IterGen showed dramatic improvements at this scale).
- Implement basic token-by-token generation with KV-cache access. Use HuggingFace `transformers` with `use_cache=True`. Verify you can:
  - Generate tokens one at a time
  - Access and store the KV-cache at arbitrary positions
  - Truncate the KV-cache to a prior position and resume generation from there (this is the rollback primitive)
- Run the model on 10 MultiPL-E Rust problems unconstrained to establish a baseline compilation rate and pass@1. This gives a "before" number.

## 4. Checkpoint boundary detection (Days 4-5)

- Implement a lightweight heuristic for detecting statement boundaries in the token stream. Options:
  - **Semicolon/brace tracking:** checkpoint after `;`, `}` (outside strings/comments). Simple, good enough for Week 1.
  - **Tree-sitter incremental parse:** more robust, handles all edge cases. Can defer to Week 2 if the heuristic works.
- Wire it into the generation loop: generate tokens, detect boundary, flag "checkpoint here."

## 5. Integration smoke test (Days 5-7)

- Wire together: model generates token-by-token -> at checkpoint boundary, snapshot KV-cache, push file content to rust-analyzer -> if diagnostic on completed code, print "WOULD ROLLBACK" (don't actually rollback yet, just log).
- Run this on 10 MultiPL-E Rust problems. Collect:
  - How many checkpoint boundaries per problem
  - How many diagnostics fire
  - How many are spurious vs. real
  - Latency of the full loop (generation + LSP round-trip)
- This is not the full system -- no actual rollback or regeneration. It's an instrumented dry run to validate that the pieces connect and to get real numbers on diagnostic frequency and latency.

---

## Deliverable

A short writeup (~1 page) with:

1. rust-analyzer diagnostic behavior on partial Rust programs (false positive rate, latency distribution)
2. Baseline compilation rate and pass@1 for Qwen2.5-Coder-1.5B on MultiPL-E Rust
3. Diagnostic frequency during unconstrained generation (how often would rollback trigger)
4. Go/no-go assessment: is the false positive rate manageable? Is latency workable?

These numbers directly inform Week 2 decisions: diagnostic filtering strategy, checkpoint granularity, and whether to adjust the approach.

---

## Blockers

- **GPU access** for model inference.
- **rust-analyzer on partial programs** -- high false positive rate would require rethinking diagnostic filtering.
- **`multilspy` coupling** -- if it's too tightly coupled to MGD's completion-only use case, writing a minimal LSP client from scratch over stdio is ~200 lines of Python. Don't over-invest in making `multilspy` work.
