# LLM Code Soundness: Compile-Time Guarantees, Formal Verification, and Agent Limitations

Notes from research discussion, 2026-04-04.

---

## 1. Static Assertions Across Languages

### General static assertions (compile-time checks)

**Top tier (CTFE as core feature):**
- **D** -- `static assert` + most powerful CTFE. Hash maps, arrays, strings all usable at compile time. Compile-time GC.
- **Zig** -- `comptime` blocks deeply integrated. Fixed-size buffers and slices at comptime, no heap.
- **Rust** -- `const { assert!(...) }` (stable 1.79+). Expanding but behind D/Zig: no heap, no trait methods in const fn (still unstable).

**Strong support:**
- **C++** -- `static_assert` since C++11, message optional since C++17. Powerful with `constexpr`/`consteval` but rough ergonomics.
- **C (C11+)** -- `_Static_assert`. Limited to integer constant expressions.
- **Nim** -- `static` blocks, `{.compileTime.}` pragma, full VM at compile time.

**Type-level proofs (ultimate static assertions):**
- Idris, Agda, Coq, Lean -- dependent type systems encode arbitrary invariants in types. Compiler proves satisfaction.

### Static assertions for mathematical constraints

Focus shifts to dependent types and formal verification:

- **Lean 4** -- best balance of usability and power. Mathlib library. Also a general-purpose language.
- **Coq** -- more mature (CompCert, sel4). Steeper learning curve.
- **Dafny** -- built-in `requires`/`ensures`/`invariant` with Z3 backend. Easiest entry point for pre/post-conditions.
- **F\*** -- dependent types + SMT. Verified crypto (HACL\*, EverCrypt). Extracts to C/OCaml.
- **SPARK/Ada** -- industrial-strength. Avionics, rail, nuclear.
- **Liquid Haskell** -- refinement types like `{v:Int | v > 0}` checked by SMT. Low annotation burden.

### Dynamic data structures in static assertions

Two paradigms:

**Compile-time execution (CTFE):**
- D (gold standard): hash maps, dynamic arrays, strings, GC at compile time
- Zig: arrays, structs, tagged unions at comptime; no heap allocator
- C++ (C++20/23): `std::vector`, `std::string` in constexpr with transient allocation (must free before runtime)
- Nim: sequences, tables, objects via compile-time VM

**Logical reasoning (SMT/provers):**
- Dafny: built-in `seq<T>`, `set<T>`, `map<K,V>` with quantifiers, proven by Z3
- Liquid Haskell: refinement types over lists
- Lean 4: proofs over any inductive type
- F\*: dependent types over lists, sequences

**Key distinction:** D/Zig/C++ can *build and query* data structures at compile time. Dafny/Lean can *prove universal properties* about data structures without executing them.

---

## 2. C++ vs Rust Compile-Time Checking

### C++23 vs Rust

| Dimension | Winner |
|---|---|
| Compile-time computation power | C++23 |
| Compile-time data structures | C++23 (transient heap: vector, string, map) |
| Memory safety guarantees | Rust (borrow checker, no UB) |
| Thread safety guarantees | Rust (Send/Sync) |
| Absence of undefined behavior | Rust |
| Generic expressiveness | C++23 (templates, SFINAE, concepts) |
| Generic usability / error quality | Rust (trait bounds) |
| Exhaustive matching | Rust (enforced match) |
| Overall "bugs caught at compile time" | Rust |

C++23 lets you *compute* more at compile time. Rust *proves* more about your program at compile time.

### C++26 changes

**Reflection (P2996):** Transformative. Compile-time introspection of types, members, enumerators. `^^` (reflect) and `[:..:]` (splice). Makes metaprogramming readable and composable. Rust has nothing equivalent -- proc macros are textual, not structural.

**Contracts (P2900):** `pre`, `post`, `contract_assert` as first-class syntax. Runtime-checked by default, not statically proven. Weaker than Dafny but now in the type system and toolable.

**Erroneous behavior (P2795):** New category between UB and well-defined. Uninitialized reads become erroneous (diagnosable, not UB). Shrinks UB surface incrementally.

**Other:** `std::inplace_vector<T,N>` (constexpr, no allocation), pack indexing (`Ts...[I]`), more constexpr stdlib, `= delete("reason")`.

**Net:** Reflection widens C++'s lead in compile-time computation/introspection. Safety gap persists -- no borrow checker, no ownership model. Profiles (Sutter's opt-in safety) did not make it into C++26.

---

## 3. Language Server Protocol (LSP)

### Basics
- JSON-RPC 2.0 over stdio (or TCP/pipes)
- Three message types: requests (client->server, must reply), notifications (fire-and-forget), responses
- rust-analyzer is the Rust LSP; clangd is the C/C++ LSP (built on Clang/LLVM)

### clangd vs rust-analyzer
- clangd needs `compile_commands.json` (CMake, Bear, Bazel tooling). rust-analyzer has zero config (Cargo).
- rust-analyzer generally better experience: faster incremental analysis, more refactorings, better macro handling, tighter compiler parity.
- C++'s textual `#include`, macros, and build system fragmentation make the problem fundamentally harder.

### LSP internals (inside the language server)

1. **Virtual File System (VFS):** in-memory file copies, patched by incremental diffs from editor
2. **Lexer/Parser -> Concrete Syntax Tree:** lossless CST (preserves whitespace, comments, malformed syntax). rust-analyzer uses `rowan` (immutable green/red trees). Must be error-recovering.
3. **Name resolution:** imports, scopes, macro expansion, trait/overload resolution. Most complex pass.
4. **Type inference/checking:** powers hover, inlay hints, diagnostics, completion filtering.
5. **Symbol index:** persistent on-disk database for cross-file operations (find references, workspace symbols).

### Incremental edits

**What the editor sends:** a diff (range + replacement text), not the whole file.

**Incremental parsing:**
- Tree-sitter: identifies smallest subtree containing edit, re-parses only that, reuses rest.
- rust-analyzer (rowan): immutable green trees with structural sharing. Unchanged subtrees share memory.
- clangd: no incremental parsing. Re-parses entire TU, but caches the preamble (#include expansions = ~90% of parse time).

**Incremental analysis (the hard part):**
- rust-analyzer uses **salsa** query framework: every analysis step is a memoized query with tracked inputs. When file changes, only queries whose inputs changed recompute. **Early cutoff**: if a query recomputes but produces the same result, dependents don't recompute. Adding a comment -> parse recomputes -> item_tree same -> everything downstream cached.
- clangd: no incremental analysis. Full Clang frontend re-run per change. Compensates with preamble caching, background indexing, debouncing.

**Diagnostics:** push-based (server sends full list) or pull-based (LSP 3.17+, client requests, server can reply "unchanged").

### Typical latencies

| Operation | rust-analyzer | clangd |
|---|---|---|
| Completion | 10-50ms | 50-300ms (1s+ in template-heavy code) |
| Go-to-definition | 5-20ms | 10-50ms |
| Hover | 5-30ms | 30-100ms |
| Find references (workspace) | 50-500ms | 100ms-2s |
| Diagnostics (native/fast path) | 10-50ms | 100ms-1s |
| Full diagnostics (cargo check / full TU) | 2-30s | 1-10s |

**Initial indexing:** small (10k LOC): 2-5s / 5-15s. Medium (100k): 10-30s / 30s-3min. Large (1M): 1-5min / 5-30min.

rust-analyzer's diagnostics split: fast native diagnostics (no borrow checker) + slow cargo check (full rustc). For full diagnostics, comparable to or slower than clangd.

---

## 4. LLMs and LSP Integration

### Current state: virtually none

No production LLM coding agent integrates directly with LSP during generation.

| Tool | LSP integration | What it actually does |
|---|---|---|
| Claude Code | No | Reads files, runs shell commands post-edit |
| Cursor | Partial | Reads editor's LSP diagnostics as text context |
| GitHub Copilot | No | Pure token prediction |
| Aider | No | Runs linter/compiler after edit |
| Continue | Partial | Can read IDE diagnostics (from LSP) |

The "partial" cases read diagnostics the editor's LSP client already computed. The LLM sees error messages as text, not structured LSP responses.

### Typical agent workflow: compiler-in-the-loop, not LSP-in-the-loop

```
LLM writes code -> tool runs cargo check / tsc / gcc -> errors as text -> LLM retries
```

Batch compilation after the fact, not interactive analysis during generation.

### Why no direct integration

1. **Architecture mismatch:** LLM generation is left-to-right ~30-100 tok/s. LSP is designed for interactive editing.
2. **Latency budget:** 50ms per LSP query x 200 tokens = 10s overhead.
3. **Running compiler after is good enough:** `cargo check` gives same errors, just in batch.
4. **LLMs rarely produce syntax errors:** type errors happen but aren't the main failure mode -- logic errors are, and LSP can't catch those.
5. **No standard headless LSP tooling:** LSP servers expect a real editor client.

### What LSP *would* help agents with

Not error checking -- navigation and understanding:
- `textDocument/definition`: follow symbols precisely
- `textDocument/references`: find all callers before refactoring
- `textDocument/completion`: know what methods/fields are available
- `textDocument/hover`: get exact types
- `workspace/symbol`: find symbols across project

---

## 5. Recursive Code Generation in Agents

### Does it exist?

Not as structured recursive decomposition. Current approaches:
- **Sequential, flat generation** (most common): one model, one pass, with retries.
- **Parallel subtask delegation** (emerging): Claude Code subagents, Devin -- parallelize across files, not within.

Nobody does: skeleton -> subagents fill methods -> sub-subagents fill helpers.

### Why not

1. **Context fragmentation:** subagents lose context about each other. Methods share implicit state.
2. **Interface specification problem:** specifying a subtask precisely enough = basically writing the code.
3. **Merge problem:** independently generated code sharing state has unresolvable semantic conflicts.
4. **LLMs are good enough at single-file generation:** 200-line class with 5 methods is within capability.

### Where decomposition works

- Across files/modules (loose coupling, explicit interfaces)
- Across abstraction layers with shared type contracts
- Not within tightly-coupled classes

### Related research

- CodePlan (MSR): dependency-aware multi-file edit planning
- MapCoder: sub-problem decomposition for competitive programming
- Self-collaboration papers: multiple LLM roles (architect, coder, tester)

---

## 6. Open Problems in Coding Agents

1. **Context window vs codebase size:** agent sees ~1-5% of code at any time. No RAG fully solves this.
2. **Specification bottleneck:** user gives a sentence, actual spec is 10x larger (in their head, Slack, tribal knowledge).
3. **Cannot verify own output:** compiles != correct. Agent-written tests can be consistent-but-wrong.
4. **Long task degradation:** success rate drops sharply with task complexity (single function ~80%, multi-day task ~5-15%).
5. **No persistent understanding:** each session starts from zero.
6. **Weak debugging:** pattern-match fixes instead of systematic root cause analysis.
7. **Blind to runtime:** can't see UI, interact with running app, use debugger interactively.
8. **External system coupling:** CI/CD, infra, external APIs, feature flags, monitoring -- agent sees code but not the system.
9. **Security and trust:** shell access, .env reading, package installation -- crude permission models.
10. **Evaluation unsolved:** benchmarks test what LLMs are already good at, not real-world difficulty.
11. **Weak planning:** reactive (code first, think later) instead of strategic.
12. **Doesn't learn from mistakes within session:** repeats failing patterns.

**Core tension:** coding is a reasoning problem that outputs text. LLMs are text generators simulating reasoning. Simulation breaks on long causal chains, novel situations, self-verification, strategic planning.

---

## 7. Mathematical Soundness in Agent-Generated Code

### Why agents fail at mathematical soundness

1. **No internal formal reasoning:** LLMs predict tokens that look like math, don't verify math.
2. **Numerical stability invisible to pattern matching:** agents default to naive formulas (e.g., `sum(x)/len(x)` instead of Kahan/Welford).
3. **Boundary conditions missed:** never checks condition number, division by near-zero, domain validity.
4. **Complexity claims unverified:** comments say O(n log n), implementation is O(n^2).
5. **Statistical reasoning is ritualistic:** applies t-test without checking normality, independence, sample size.
6. **Proof-adjacent code without proofs:** implements red-black tree, misses one rotation case, tests don't hit it.
7. **Off-by-one in math indexing:** drops normalization constants, wrong sign conventions, inclusive/exclusive bounds.
8. **Cannot reason about convergence:** no check for fixed point stability, basin of attraction, appropriate tolerance.
9. **No dimensional analysis:** doesn't track units through computation. `power = energy * time` instead of `energy / time`.
10. **Approximation problems:** defaults to naive implementations (softmax without max-shift, Taylor series for exp).

**Root cause:** mathematical correctness requires deductive reasoning (axioms + rules -> guaranteed conclusion). LLMs do abductive reasoning (examples + patterns -> plausible conclusion).

---

## 8. Languages for Mathematical Soundness

### Tier 1: Full formal verification

- **Lean 4:** largest math library (Mathlib). Also a real programming language. Used by Terence Tao.
- **Coq:** most mature. CompCert (verified C compiler), sel4 (verified OS kernel). Extracts to OCaml/Haskell.
- **Agda:** cleanest dependent type theory. Totality checker.
- **F\*:** dependent types + Z3. Verified crypto (HACL\*, Firefox, Linux kernel). Extracts to C, OCaml.

### Tier 2: Automated verification (SMT-backed)

- **Dafny:** Z3 does heavy lifting. Built-in seq/set/map with quantifiers. Best entry point for programmers.
- **SPARK/Ada:** avionics, rail, nuclear. Proves absence of runtime errors + functional correctness.
- **Why3:** verification platform, backends to Z3/CVC5/Alt-Ergo/Coq.

### Tier 3: Refinement types and contracts

- **Liquid Haskell:** `{v:Int | v > 0}` checked by SMT. Layered on Haskell.
- **ATS:** dependent types compiling to C. Tiny community.
- **F# with Units of Measure:** built-in dimensional analysis, zero runtime cost. Only covers dimensional correctness.

---

## 9. Lean 4 Deep Dive

### As general-purpose language

Yes, by design. Self-hosting compiler. Compiles to native via C (not interpreted). Reference-counted (not GC). Has: algebraic data types, pattern matching, monadic IO, mutable state (`let mut`), for loops, error handling, type classes, metaprogramming, C FFI.

**Performance:** faster than Haskell, comparable to OCaml, 2-5x slower than Rust/C.

**What's missing:** database drivers, HTTP client/server (basic only), async IO, GUI, image/audio, ML/numeric libraries. Small but growing package ecosystem.

**Trajectory:** roughly where Rust was in 2014 -- clearly good, clearly the future of its niche, not yet ready to replace mainstream choices for most teams.

### Coding agents in Lean 4: poor

**Why:**
- Training data scarcity (~5-10k repos vs Python's ~50M+)
- Proof generation != code generation: partial proofs are useless, pattern matching fails because proofs are problem-specific, errors cascade
- Tactic mode adversarial: each tactic transforms invisible proof state; agent generates text without seeing intermediate states
- Search space enormous: unbounded branching factor (any of 150k+ Mathlib lemmas could be relevant)

**Benchmarks:** miniF2F ~30-35% single-pass (vs ~90% human). ProofNet ~50% with retrieval. LeanDojo ~52%.

**Specialized tools outperform general agents:** ReProver (retrieval-augmented), LLMStep (tactic copilot), AlphaProof (RL-trained, solved IMO 2024 problems). Key: they have access to proof state.

**What would help most:** interactive tactic generation with proof state feedback -- integrating the agent with Lean's LSP.

### Rewriting PyTorch in Lean 4

**Full rewrite: impractical.**
- CUDA kernels: no GPU backend, no SIMD. Would FFI to same libraries.
- C++ core (ATen): possible but questionable value. Shape-checked tensors are interesting but most real shapes are dynamic.
- Performance: 2-5x slower than C++ for compute.

**Realistic path: verified spec layer over PyTorch.**
- Shape safety: dimension mismatches become compile errors
- Gradient correctness: prove backward pass computes correct derivative
- Optimizer convergence: prove SGD converges for convex objectives

**The floating-point gap:** Lean proves theorems about R (reals). Computers use Float64. `(a+b)+c != a+(b+c)` in IEEE 754. Proofs about idealized math don't catch numerical errors. Would need Flocq-equivalent (exists in Coq, not Lean).

**Closest existing work:** SciLean project (scientific computing in Lean 4, verified derivatives). Very early.

---

## 10. Relevant Benchmarks

### Formal reasoning / theorem proving
- **miniF2F:** 244 math competition problems in Lean/Isabelle. SOTA ~60% with search, ~30% single-pass.
- **ProofNet:** undergraduate math in Lean. ~50% with retrieval.
- **LeanDojo:** ~100k Lean 4 theorems from Mathlib. ReProver ~52%.
- **FIMO:** formal IMO problems. AlphaProof solved 4/6 IMO 2024.
- **PutnamBench:** graduate-level. SOTA ~5-10%.

### Mathematical reasoning (informal)
- **MATH:** 12.5k competition problems. SOTA ~90%+ (contamination concerns; answer != proof).
- **GSM8K / GSM-Symbolic:** grade school math. GSM-Symbolic (same problems, different numbers) drops models ~15%, exposing pattern matching.
- **FrontierMath:** research-level math (Epoch AI). SOTA ~2-5%. Catastrophically low.

### Code generation
- **SWE-bench Verified:** real GitHub issues. SOTA ~55-65%. Plateauing.
- **HumanEval/MBPP:** function-level. ~95%+ now. Saturated.
- **LiveCodeBench:** new competitive programming problems (post-cutoff). No contamination.
- **CRUXEval:** predict output / predict input. SOTA ~70-80%. Models struggle with complex state.
- **SciCode:** scientific computing from research papers. SOTA ~10-20% on hard problems.

### Major benchmark gaps

| Gap | What's missing |
|---|---|
| NumericalSoundness-bench | Floating-point correctness, stability, precision |
| ProofRepair-bench | Fix broken Lean/Coq proofs |
| InvariantGen-bench | Generate loop invariants, pre/post conditions |
| TypeSystemExploit-bench | Use advanced type features to catch bugs |
| ConvergenceBench | Iterative algorithms with provable convergence |
| Long-horizon debug | Debug requiring 20+ investigation steps |

**Core problem:** benchmarks test what LLMs are good at. The areas we identified as weak (numerical soundness, formal verification, mathematical correctness in code) have thin or absent benchmarks. No measurement -> no optimization pressure -> no improvement.

---

## 11. Practical Approaches to Mathematical Soundness in Agentic Coding

### The core problem

When an agent writes math-heavy code, it compiles, passes tests, looks right -- but may be silently mathematically wrong. You don't get an error; you get bad science.

### Level 0: costs nothing, do today

- **Property-based testing** (hypothesis): test invariants on random inputs, not example values. softmax: non-negative, sums to 1, preserves ordering, no NaN/Inf.
- **Gradient checking:** `torch.autograd.gradcheck` for every custom autograd function. Non-negotiable. Put in CI.
- **Use stable versions by default:** lookup table of naive vs stable (log_softmax not log(softmax), np.linalg.solve not inv(A)@b, Welford not naive variance).

### Level 1: moderate effort, high value

- **Condition number awareness:** check `cond(A)` before linear algebra. If > 10^12, fall back to regularized/pseudoinverse.
- **Dimensional analysis:** pint (Python units), jaxtyping+beartype (tensor shape checking).
- **Error propagation tracking:** uncertainties library, interval arithmetic (mpmath).

### Level 2: significant effort, catches deep bugs

- **Herbie:** automated numerical stability analysis. Feed formula, get stable rewrite.
- **Symbolic differentiation + codegen:** derive gradients with sympy, generate code. No manual derivatives.
- **Metamorphic testing:** test mathematical properties across related inputs (translation invariance, scale equivariance, monotonicity).

### Agent-specific interventions

- **Oracle-based verification:** compare agent's implementation against trusted reference (scipy, torch builtins) on 1000 random inputs.
- **Property specs as prompts:** give the agent mathematical properties the function must satisfy, require it to test them.
- **Numerical stress testing as agent tool:** test on normal, large (1e15), small (1e-15), near-cancellation, constant, singleton inputs.
- **Symbolic ground truth:** provide symbolic expression as spec, auto-check agent's code against it.
- **Constrained agent architecture:** compiler -> property tests -> stress tests -> gradient checks -> oracle comparison -> ALL PASS -> done.
- **Dimensional annotations:** require jaxtyping shape annotations on all tensor functions.
- **Two-agent adversarial:** separate implementer and verifier agents.
- **CLAUDE.md rules:** checklist the agent follows every time (cite source formula, gradient check, stress test, oracle compare, never implement naive versions).

**Key insight:** you can't make agents mathematically sound. You build a gauntlet of automated checks. You provide the math specs (properties, formulas, invariants), the agent provides implementation + testing labor.

---

## 12. Why Numerical Stability Isn't in Mathlib

- **R != Float64 algebraically.** Float64 is not a field, not associative for addition, not distributive. Mathlib's framework assumes field axioms.
- **Different questions.** Mathlib: "this algorithm is correct." Numerical analysis: "by how much is it wrong?"
- **Need IEEE 754 formalization first.** Rounding model, error propagation, special values (NaN, Inf, subnormals). Done in Coq (Flocq, ~50k lines, 15+ years). Not done in Lean 4.
- **Community mismatch.** Mathlib contributors are pure mathematicians. Numerical analysis is applied math, culturally distant from proof assistants.
- **Ugly to formalize.** Typical numerical result (Higham): error bound involving gamma_n, machine epsilon, componentwise absolute value, induction over summation tracking rounding at each step. Tedious, aesthetically unrewarding for volunteer contributors.
- **Estimated effort to reach Flocq parity in Lean 4:** ~5-8 person-years. Nobody has started. Small academic ROI (porting project), Lean's Float type is opaque, funding agencies don't prioritize formalization infrastructure.
- **Will it happen?** Likely eventually. AI safety, scientific reproducibility, Lean's trajectory all push toward it. Needs someone to do the unglamorous IEEE 754 foundation work.
