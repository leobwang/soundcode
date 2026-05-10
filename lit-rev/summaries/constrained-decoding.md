# Constrained Decoding for Code Generation

Papers on enforcing syntactic and semantic constraints during LLM decoding, from grammar-guided approaches to type-system-level constraints.

---

## 1. A Syntactic Neural Model for General-Purpose Code Generation (Yin & Neubig, 2017)

**ArXiv:** https://arxiv.org/abs/1704.01696

### 1.1 Problem Statement

Given a natural language description $x$, generate a code snippet $c$ in a general-purpose language (Python). Prior seq2seq approaches treat code as a flat token sequence, ignoring formal grammar constraints, leading to syntactically invalid outputs.

### 1.2 Core Idea: Generate the AST, Not the Code

Instead of generating code token by token, the model generates an **abstract syntax tree (AST)** via a sequence of **actions** applied in depth-first, left-to-right order:

1. **ApplyRule[$r$]** -- expands the current AST node using a grammar production rule $r$. The set of applicable rules is restricted to those whose LHS matches the current node's type, guaranteeing syntactic validity.
2. **GenToken[$v$]** -- emits a terminal token $v$ at a leaf node, drawn from a fixed vocabulary or copied from the NL input via a pointer network.

The AST is decomposed into a sequence of $T$ actions $a_1, a_2, \ldots, a_T$, and the joint probability factorizes autoregressively:

$$p(y \mid x) = \prod_{t=1}^{T} p(a_t \mid a_{<t}, x)$$

where $a_t$ is the action at step $t$ (either ApplyRule or GenToken) and $a_{<t}$ denotes all preceding actions. Because the decoder only selects grammar-valid actions, **every generated AST is syntactically valid by construction**.

### 1.3 Neural Architecture

Encoder-decoder with attention. Decoder hidden state incorporates **parent feeding** (hidden state of the parent AST node) and frontier node type, providing structural information beyond flat left-to-right sequence.

### 1.4 Results

| System | HearthStone Acc. | Django Acc. |
|---|---|---|
| LPN (Ling et al., 2016) | 4.5% | 62.3% |
| SEQ2TREE | 1.5% | 28.9% |
| **Yin & Neubig** | **16.2%** | **71.6%** |

SEQ2TREE produces grammatically incorrect ASTs 10.9-21.2% of the time. This model: **0% syntax errors**. Removing the copy mechanism causes the largest accuracy drop (16.2% -> 3.0% on HearthStone).

### 1.5 Limitations

- No semantic checks (type correctness, variable scoping, logical soundness).
- Language-specific (requires explicit grammar + action decomposition per language).
- 2017-era LSTM architecture; does not leverage modern pre-trained transformers.

---

## 2. PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding (Scholak et al., EMNLP 2021)

**ArXiv:** https://arxiv.org/abs/2109.05093

### 2.1 Problem Statement

Fine-tuned T5 models for text-to-SQL have an unconstrained output space, frequently producing syntactically or semantically invalid SQL. Prior approaches require custom architectures or impractically large beam sizes for post-hoc filtering.

### 2.2 Core Idea: Incremental Parser as a Filter

At each beam search step, for each candidate token in top-$k$, PICARD checks whether appending it keeps the partial output parseable. Invalid tokens get $-\infty$ scores.

**Four checking modes** (increasingly strict):
1. **Off** -- baseline.
2. **Lexing** -- valid SQL lexical items only.
3. **Parsing (no guards)** -- full grammatical structure + valid table.column references.
4. **Parsing with guards** -- eager semantic guards: column resolution, alias resolution, FROM clause consistency.

**Key properties:** Inference-time only (no retraining). Works on surface-form SQL. Compatible with any autoregressive decoder.

### 2.3 Results

| System | Dev EM% | Dev EX% |
|---|---|---|
| T5-Base | 57.2 | 57.9 |
| **T5-Base + PICARD** | **65.8** | **68.4** |
| T5-3B | 69.9 | 71.4 |
| **T5-3B + PICARD** | **75.5** | **79.3** |

**T5-Base + PICARD outperforms T5-Large without it.** T5-Large + PICARD outperforms T5-3B without it. PICARD gives the benefit of a 3-10x larger model for free.

**Speed:** ~24% overhead on A100 (2.5 -> 3.1 sec/sample for T5-3B with beam 4).

### 2.4 Limitations

- SQL-specific: parser and guards hand-crafted for SQL.
- No rollback -- forward masking only.

---

## 3. Synchromesh: Reliable Code Generation from Pre-trained Language Models (Poesia et al., ICLR 2022)

**ArXiv:** https://arxiv.org/abs/2201.11227

### 3.1 Problem Statement

Pre-trained LLMs (GPT-3, Codex) produce two error classes: conceptual errors (wrong intent) and implementation errors (syntax, type mismatches, nonexistent references). Austin et al. (2021) found 47% of failures were implementation errors.

### 3.2 Core Idea: Completion Engines + TST

**Target Similarity Tuning (TST):** Fine-tune sentence embeddings so inputs with structurally similar *target programs* (by AST tree edit distance) map close together, improving few-shot example selection.

**Constrained Semantic Decoding (CSD):** A **Completion Engine (CE)** $C_L$ takes any partial program string $s$ and returns a regex over the base character alphabet $\Sigma$ describing which characters can validly extend $s$:

$$C_L : \Sigma_L^* \to \text{RegEx}(\Sigma)$$

where $\Sigma$ is the character alphabet (e.g., ASCII) and $\Sigma_L \subseteq \Sigma^*$ is the set of tokens of the target language $L$ (keywords, identifiers, operators, etc.).

CEs are implemented in two layers:
1. **Context-free:** Derived from the grammar via ANTLR.
2. **Context-sensitive:** Hand-written semantic rules (column names must exist, aliases must resolve, type compatibility).

**Token misalignment solved via Brzozowski derivatives.** LLM tokens (BPE sub-words) and language tokens have misaligned boundaries. CSD resolves this using Brzozowski derivatives: given a regular language $S$ (the set of valid continuations from the CE) and a string prefix $u$ (e.g., a partial BPE token already generated), the derivative $u^{-1}S = \{v : u \cdot v \in S\}$ gives the set of strings that can follow $u$ to produce a member of $S$. This lets CSD check each candidate BPE token against the valid continuation set, even when token boundaries don't align with language token boundaries.

### 3.3 Results

| Model | SQL Exec. | SQL Valid | SMCalFlow Acc. | SMCalFlow Valid |
|---|---|---|---|---|
| Codex 175B | 56% | 73% | 45% | 79% |
| **Codex + CSD + TST** | **64%** | **85%** | **63%** | **99%** |

- Validity jumps: SQL 73% -> 85%, SMCalFlow 79% -> 99%.
- CSD and TST are complementary (better prompt + constrained output).
- ~8% overhead using OpenAI API logit bias.
- Advantage grows with program length.

### 3.4 Limitations

- CSD cannot fix conceptual errors; constraining to valid programs produces valid-but-wrong programs.
- Hand-written CEs required per language.
- No rollback capability.

---

## 4. SynCode: LLM Generation with Grammar Augmentation (Ugare et al., NAACL 2024)

**ArXiv:** https://arxiv.org/abs/2403.01632

### 4.1 Problem Statement

Prior constrained decoding approaches have three shortcomings: (1) token-grammar misalignment, (2) computational overhead, (3) lack of generality across languages. SynCode addresses all three with a **language-agnostic** framework.

### 4.2 Core Idea: DFA Mask Store + Incremental Parser

**Two-step masking.** Let $\Gamma$ denote the set of grammar terminals (keywords, operators, literal patterns), each recognized by a **deterministic finite automaton (DFA)** $D_\gamma$ with states $Q$, transition function $\delta$, initial state $q_0$, and accepting states $F$.

1. **Compute accept sequences and remainder.** The incremental parser (Lark) processes partial output and returns: (a) the set $S_\mathcal{A}$ of **accept sequences** -- sequences of terminals $\mathcal{A} = (\gamma_1, \gamma_2, \ldots)$ that can legally follow the current partial output according to the grammar; and (b) the **remainder** $r$ -- the trailing suffix of the partial output that might change its terminal classification when more characters are appended (e.g., `"12"` could become an integer or a float prefix depending on what follows).

2. **Look up masks from pre-computed DFA mask store.** For each accept sequence $\mathcal{A} = (\gamma_1, \ldots)$, take the DFA $D_{\gamma_1}$ for the first terminal, walk it from its initial state using the remainder $r$ to reach state $q_r = \delta^*(q_0, r)$, and look up the pre-computed mask $\tilde{m}_\mathcal{A}$ for that (DFA, state) pair. The final mask is the element-wise OR across all accept sequences:

$$\tilde{m}_{S_\mathcal{A}} = \bigvee_{\mathcal{A} \in S_\mathcal{A}} \tilde{m}_\mathcal{A}$$

where each $\tilde{m}_\mathcal{A}[v] = 1$ iff LLM token $v$ does not lead to a dead state when walking $D_{\gamma_1}$ from $q_r$.

**DFA mask store:** Pre-computed offline lookup table mapping each (terminal DFA $D_\gamma$, DFA state $q$) pair to a binary mask over the LLM vocabulary $V$. O(1) lookup at inference time, GPU-offloadable as a tensor.

### 4.3 Theoretical Guarantees

**Theorem 1 (Soundness):** Valid tokens are never blocked.
**Theorem 2 (Completeness):** Under mild conditions, invalid tokens are always blocked.

### 4.4 Results

- **JSON:** 0% syntax errors across all models (vs. various non-zero baselines).
- **SQL (Spider):** CodeLlama-7B compilation 80.0% -> 84.4%, execution 49.5% -> 52.1%.
- **Python/Go (HumanEval, MBXP):** Syntax errors reduced by 96.07%.
- **Overhead:** 10-20% with GPU-offloaded DFA mask store.

### 4.5 Limitations

- **Syntax only** -- no semantic constraints (types, scoping, schema awareness).
- No rollback.
- Language-agnostic but requires EBNF grammar specification.

---

## 5. Outlines: Efficient Guided Generation for Large Language Models (Willard & Louf, NeurIPS Workshop 2023)

**ArXiv:** https://arxiv.org/abs/2307.09702

Outlines constrains LLM sampling to a CFG or regex by building a finite-state machine index that maps token IDs to grammar states. Provides efficient logit masking at each step. Widely adopted in practice (dottxt). Limited to syntax-level constraints (regex/CFG), no semantic awareness.

---

## 6. IterGen: Iterative Structured LLM Generation with Backtracking (Park et al., ICLR 2025)

**ArXiv:** https://arxiv.org/abs/2410.07295

### 6.1 Problem Statement

Existing grammar-guided tools (SynCode, Outlines, etc.) guarantee syntactic correctness but (1) cannot enforce semantic properties (valid column names, no privacy leaks) and (2) have no backtracking -- if the LLM produces semantically incorrect fragments, the only recourse is discarding the entire output.

### 6.2 Core Idea: Three Primitives

IterGen exposes three operations at the **grammar symbol** level (not raw tokens):

- **`forward(stop_symbol, count)`** -- generate until `count` new occurrences of `stop_symbol`.
- **`backward(stop_symbol, count)`** -- remove the last `count` occurrences, restoring KV-cache, parser, and symbol map.
- **`view(symbol)`** -- return all substrings matching `symbol` in current output.

**Grammar symbols** (terminals and non-terminals from the CFG) provide a natural, interpretable abstraction for navigation, unlike raw BPE tokens.

### 6.3 Backtracking Mechanism

Backtracking is **user-programmed**, not automatic. The user writes a loop: `forward` -> `view` -> semantic check -> `backward` if invalid.

**KV-cache management:** On `backward`, the KV cache is simply **truncated** to the backtrack token position (O(1)). No recomputation. Remaining entries are valid because autoregressive KV entries are independent of future positions.

**Recurrence penalty:** After backtracking, the model tends to regenerate the same tokens that were just rejected. To encourage exploration, each previously generated token $t$ has its score multiplied by $(1-\gamma)^\alpha$, where $\gamma \in [0, 1]$ is the penalty strength (default $\gamma = 0.7$) and $\alpha$ is the number of times token $t$ has been backtracked over. Higher $\alpha$ means the token has been rejected more often, so its score is progressively suppressed.

**Symbol position map:** Maintained incrementally via the LR parser's reduce operations. Each grammar symbol is mapped to character-position spans.

### 6.4 Results

**SQL (Spider, 1034 problems, averaged across 9 models):**

| Method | Accuracy | Execution Success | Avg. Time |
|---|---|---|---|
| Standard | 28.90% | 50.28% | 0.81s |
| SynCode | 35.22% | 63.72% | 1.56s |
| **IterGen** | **41.63%** | **75.84%** | **1.19s** |

IterGen is **faster** than SynCode on average because it catches errors early rather than generating long invalid suffixes.

**Privacy leakage (Enron email extraction):** Reduces leaks from 51.4% to **0%** across all 10 models.

**Vega-Lite:** +17.8% accuracy over SynCode on average.

### 6.5 Limitations

- Semantic constraints must be **manually programmed** by the user.
- Single-sequence generation only (no batching).
- Recurrence penalty is a heuristic that can skew distributions.
- No convergence guarantee (may hit max_iter).
- Grammar must be LR-parseable.
- Semantic checks are shallow (schema validation, email matching) -- cannot check logical correctness.

### 6.6 Key Figures

- **Figure 1:** Architecture showing KV cache, decoding trace (token tree), and symbol position map.
- **Figure 3:** 18-line SQL generation program demonstrating API simplicity.

---

## 7. Type-Constrained Code Generation with Language Models (Mundler et al., PLDI 2025)

**ArXiv:** https://arxiv.org/abs/2504.09246

### 7.1 Problem Statement

Syntax accounts for only ~6% of compilation errors in LLM-generated TypeScript; 94% are type errors. Can we enforce type-system rules during decoding so that every generated program is well-typed?

### 7.2 Core Idea: Prefix Automata for Type Inference

1. **Prefix automata:** Non-deterministic automata satisfying a *prefix property*: every reachable state can lead to an accepting state. Guarantees that at any point during generation, the partial program can still be completed to a well-typed program.

2. **Simply-typed core calculus $L_B$:** Types `number`, `string`, `boolean`, function types `(p) => T`. Automaton incrementally builds an AST, annotating states with the type environment.

3. **Type reachability search:** The hardest sub-problem is determining whether a partial expression currently of type $T$ (e.g., `number`) can be extended via available operations (binary operators, member access, function calls) to produce a value of a required goal type $G$ (e.g., `string`). This is solved by DFS over an abstracted type graph where nodes are types and edges are operations (e.g., `number -> string` via `.toString()`).

4. **Extension to TypeScript:** Arrays, loops, `const`/`let`, additional operators, polymorphic built-in members. ~11,200 lines of Python.

### 7.3 How It Differs from Syntax-Only

Syntax-only catches stray semicolons and unmatched brackets. Type-constrained catches undeclared identifiers, wrong argument types, missing return statements, and type mismatches -- **94% of actual compilation errors**.

### 7.4 Results

Across 6 LLMs (Gemma 2B/9B/27B, DeepSeek Coder 33B, CodeLlama 34B, Qwen2.5 32B):
- **Compilation errors reduced by 75.3%** on HumanEval, 52.1% on MBPP (vs. 9.0% / 4.8% for syntax-only).
- **pass@1 improves** by 3.5% (synthesis), 5.0% (translation), 37.0% (repair).
- **Overhead:** 20-55% median; 99.4% of cases need only 1 sample-and-check iteration.

### 7.5 Limitations

- Covers a **subset** of TypeScript (no union/intersection types, limited generics).
- Sound but incomplete: may reject valid completions involving complex type chains.
- Requires building a language-specific completion engine (~11K lines).
- No rollback capability -- forward masking only.

---

## Comparative Summary

| Paper | Year | Constraint Level | Rollback? | Language-Agnostic? | Retraining? |
|---|---|---|---|---|---|
| **Yin & Neubig** | 2017 | Syntax (CFG via AST) | No | No (per-language) | Yes |
| **PICARD** | 2021 | Syntax + SQL semantics | No | No (SQL only) | No |
| **Synchromesh** | 2022 | Syntax + rich semantics | No | Partially (per-language CE) | No |
| **Outlines** | 2023 | Syntax (CFG/regex) | No | Yes | No |
| **SynCode** | 2024 | Syntax (CFG) | No | Yes (any EBNF) | No |
| **IterGen** | 2025 | Syntax + user-programmed semantics | **Yes** (grammar-symbol level) | Yes (any LR grammar) | No |
| **Type-Constrained** | 2025 | Syntax + types | No | No (TypeScript subset) | No |

**Evolution:**
1. **2017:** Grammar-guided generation eliminates syntax errors but requires custom architectures.
2. **2021-2022:** Inference-time-only constrained decoding (PICARD, Synchromesh) with off-the-shelf LMs.
3. **2023-2024:** Language-agnostic approaches (Outlines, SynCode) with formal guarantees.
4. **2025:** Beyond syntax -- IterGen adds backtracking with user-programmed semantic checks; Type-Constrained adds type-system enforcement.

**Frontier:** The field is moving from syntax to semantics. IterGen and Type-Constrained represent two approaches: IterGen makes semantic checking user-programmable with backtracking; Type-Constrained formalizes type inference into the automaton. Neither uses an external semantic oracle (like an LSP server) as a comprehensive constraint source.
