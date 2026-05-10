# Verification and Monitoring During Code Generation

Papers on systems that verify, monitor, or correct LLM-generated code during or immediately after the generation process, using external tools (LSP, type checkers, execution) or internal model signals.

---

## 1. Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context (Lakhotia et al., NeurIPS 2023)

**ArXiv:** https://arxiv.org/abs/2306.10763
**Code:** https://github.com/microsoft/monitors4codegen

### 1.1 Problem Statement

LMs generate high-quality completions when local context suffices, but frequently hallucinate identifiers when code references types, methods, or APIs defined in other files, packages, or external libraries. IDEs solve this for human developers via static analysis exposed through the Language Server Protocol (LSP). MGD asks: can we give LMs the same assistance at decoding time, without retraining?

### 1.2 Core Idea: Monitors

MGD introduces a **monitor** -- a stateful interface between the LM and a static analysis engine -- that observes code being generated token-by-token and, at predefined **trigger points**, queries the static analysis to obtain valid completions. These are converted into a binary mask over the LM's vocabulary, reshaping logits so that only type-consistent tokens can be sampled.

The key distinction from prior constrained decoding (PICARD, Synchromesh, SynCode) is that MGD enforces **semantic** constraints derived from **repository-wide static analysis** (type hierarchies, cross-file imports, build-time artifacts), not just syntactic validity from a context-free grammar.

### 1.3 Formal Definitions

**Monitor.** A monitor $M_\varphi$ for property $\varphi$ is a tuple $(A_\varphi, s_0, S, \texttt{pre}, \texttt{update}, \texttt{maskgen})$ where:
- $S$ is the set of states; $s_0 \in S$ is the **wait state**
- $\texttt{pre}(s; x_1, \ldots, x_n)$ determines when the monitor activates
- $\texttt{update}(s, x_{n+1})$ transitions state after each token
- $\texttt{maskgen}(s, V)$ produces a binary mask $m \in \{0, 1\}^{|V|}$

**Guided sampling (Eq. 1).** Let $L_\theta$ denote the language model with parameters $\theta$, $C$ the repository context (all files, dependencies, build artifacts), $p$ any additional prompt context (e.g., fill-in-the-middle suffix), and $\ell \in \mathbb{R}^{|V|}$ the raw logit vector over vocabulary $V$. The composition $L_\theta || M_\varphi$ (model guided by monitor) samples as:

$$(L_\theta || M_\varphi)(x_{n+1} | x_{1:n}; C, p, s) = \begin{cases} \text{softmax}(\ell)[x_{n+1}] & \text{if } s = s_0 \\ \text{softmax}(\ell \oplus m)[x_{n+1}] & \text{otherwise} \end{cases}$$

where $\oplus$ denotes the masking operation: for each token $x$ in the vocabulary, if $m[x] = 0$ then $\ell[x]$ is set to $-K$ (a large negative constant, effectively $-\infty$), leaving valid tokens ($m[x] = 1$) unchanged.

**State transition (Eq. 4):** When in wait state and pre-condition fires, invoke static analysis $A_\varphi$ on the partial code within the full repository context $C$. Otherwise, $\texttt{update}$ prunes the suggestion set as tokens are generated.

### 1.4 Monitor Types

**Primary: Type-consistent identifier dereferences.** Triggers on the `.` operator. Queries the language server to resolve the type $T$ of the object, returns all accessible fields/methods. Mask allows only tokens that are prefixes of valid identifiers.

**Additional monitors (MGDMICROBENCH):**
1. Instantiation of valid classes (triggers on `new `)
2. Switch over enum (triggers on `case `)
3. Correct number of arguments (stack-based tracking)
4. Joint monitoring for multiple properties (product of state spaces)

**Richer analyses:** Typestates (API call ordering via FSMs) and session types (communication protocol enforcement).

### 1.5 The `multilspy` Library

Cross-platform Python library abstracting LSP interaction. Supports Java (Eclipse JDT.LS), Python (JEDI), C# (OmniSharp), Rust (rust-analyzer). Manages language server lifecycle, incremental document updates, and completion queries.

### 1.6 Key Design: Trigger Points, Not Every Token

Monitors fire **only when specific pre-conditions are met** (e.g., `.` for dereferences). Between triggers, the LM generates freely. This is fundamentally different from grammar-constrained approaches that intervene at every step.

### 1.7 Results

Evaluated on **PRAGMATICCODE** (100 real-world Java repos) with **DOTPROMPTS** (10,538 dereference prompts).

**Models:** CodeGen-350M/2B/6B, SantaCoder-1.1B, text-davinci-003 (~175B).

| Config | Compilation Rate | Next Identifier Match |
|---|---|---|
| CG-350M | 52.43 | 76.94 |
| **CG-350M-MGD** | **65.37 (+24.7%)** | **83.80 (+8.9%)** |
| CG-2B | 57.01 | 81.11 |
| **CG-2B-MGD** | **70.91 (+24.4%)** | **87.32 (+7.7%)** |
| SC (1.1B) | 59.97 | 82.40 |
| **SC-MGD** | **73.03 (+21.8%)** | **88.42 (+7.3%)** |
| TD-3 (175B) | 62.66 | 86.18 |
| **TD-3-MGD** | **74.26 (+18.5%)** | **91.19 (+5.8%)** |

Key: **SC-MGD (1.1B) outperforms TD-3 (175B)** on compilation rate (73.03 vs. 62.66). MGD is complementary to prompt augmentation and fill-in-the-middle decoding.

**Overhead:** ~83% mean slowdown (22.57s -> 41.34s per sample on CG-6B).

### 1.8 Limitations

1. **No rollback.** If static analysis returns empty suggestions, the generation run is abandoned. No mechanism to undo previously generated tokens. The authors explicitly identify "backtracking and beam-search" as future work.
2. **Trigger points only.** Cannot prevent errors between trigger points (wrong variable names, incorrect logic, wrong statement structure).
3. **Dependent on static analysis quality.** Language servers use heuristics for partial code; can be imprecise or incomplete.
4. **No functional correctness.** Improves compilability, not semantic correctness.
5. **Repository build environment required.**
6. **Java-focused evaluation.**

### 1.9 Key Figures

- **Figure 1:** Motivating example -- text-davinci-003 and SantaCoder generate wrong identifiers for `ServerNode.Builder`; SantaCoder+MGD gets it right.
- **Figure 4 (Appendix A):** Step-by-step monitor state transitions during generation. Essential for understanding the mechanism.
- **Table 1:** Main results across all configurations.

---

## 2. DSVD: Dynamic Self-Verify Decoding (Guo et al., 2025)

**ArXiv:** https://arxiv.org/abs/2503.03149

### 2.1 Problem Statement

LLMs hallucinate during text generation. Existing approaches either intervene preemptively (ITI, DoLa, TruthX) without exploiting the model's ability to recognize errors after the fact, or verify post-hoc (Self-Refine, Reflexion) at high computational cost. DSVD's thesis: LLMs are better at *detecting* errors after generation than *preventing* them ("delayed awareness of hallucinations"), so the optimal intervention point is *during* generation, immediately after an error is produced.

### 2.2 Core Idea

Two components running **in parallel** with the LM head during autoregressive generation:

1. **Hallucination detector:** Lightweight probing heads (two-layer MLPs) trained on the model's hidden states, producing a binary hallucination probability for each token.
2. **Dynamic rollback:** When the detector flags a hallucination, roll back to a position before the error, sample candidate continuations, score with a penalty, and select the best.

The probing heads share the model's hidden states (already computed for next-token prediction), making verification essentially free.

### 2.3 Rollback Mechanism

**Sliding window** $W$ of size $r$ tracks recent probing probabilities $\{z_{t-r+1}, \ldots, z_t\}$. Rollback triggers when any $z_i^{\text{hallu}} > 0.5$.

**On rollback** to position $t_0 = t - r$:
1. Truncate KV-cache to position $t_0$ (O(1) operation -- simply discard entries for positions $t_0 + 1$ through $t$).
2. Sample $k$ candidate continuations $\{s_1, \ldots, s_k\}$, each of length $m$ tokens, from position $t_0$.
3. Score each candidate $s_j$ using a penalized log-probability that combines the LM's confidence with the probing heads' hallucination assessment:

   $$f(s_j) = \sum_{i=t_0}^{t_0 + m} \left[\log p(x_i | x_{<i}) - \alpha \cdot \log(z_i^{\text{hallu}})\right]$$

   The first term is the standard LM log-probability; the second penalizes tokens that the probing heads flag as likely hallucinated. The hyperparameter $\alpha \in \mathbb{R}^+$ controls penalty intensity.
4. Select $s_{\text{best}} = \arg\max_{s_j} f(s_j)$; resume generation from position $t_0 + m$.

**Default hyperparameters:** $r = 10$ (window size), $k = 5$ (number of candidate continuations), $m = 20$ (continuation length), $\alpha = 0.1$ (penalty intensity).

### 2.4 Probing Head Training

The detector consists of $L$ probing heads (one per transformer layer), where each probing head $\phi^l$ is a **two-layer MLP** with binary classification output. During the forward pass, the model produces hidden states $h_i^l$ at each token position $i$ and layer $l$. The probing heads reuse these hidden states (already computed for next-token prediction) to produce a binary hallucination probability:

$$z_i = \text{softmax}\left(\frac{1}{L} \sum_{l=0}^{L} \phi^l(h_i^l)\right)$$

where $z_i = (z_i^{\text{hallu}}, z_i^{\text{correct}})$ are the probabilities that token $i$ is hallucinated vs. correct. Aggregating across all layers outperforms using any single layer.

Trained on the model's own generations with semi-supervised labeling: correct responses (Rouge-L F1 > 0.8 against ground truth) label all tokens as non-hallucinated; incorrect responses (Rouge-L F1 < 0.2) identify the hallucination onset position via log-probability analysis. Focal loss with $\gamma = 2$.

### 2.5 Results

**Models:** Llama-2-7B-Chat, Llama-3-8B-IT, Qwen2.5-7B-IT.

| Method | TruthfulQA (T*I) | StrQA | SciQ | EntQ |
|---|---|---|---|---|
| Greedy | 31.9 | 63.6 | 59.8 | 29.3 |
| DoLa | 41.4 | 62.1 | 61.3 | 29.5 |
| TruthX | 45.2 | 64.1 | 59.8 | 30.1 |
| Self-Refine | 36.9 | 58.2 | 55.4 | 26.1 |
| **DSVD** | **48.4** | **67.7** | **61.8** | **30.7** |

(Llama-2-7B-Chat numbers shown; DSVD leads across all models.)

**Composability:** DSVD stacks on top of DoLa and ITI for additive gains.

### 2.6 Latency

| Rollbacks | Overhead vs. Greedy |
|---|---|
| 0 (probing only) | ~5% |
| 5 | ~10% |
| 10 | ~20% |

Compare to Self-Refine's ~435% overhead.

### 2.7 Limitations

1. **Depends on internal knowledge.** Cannot correct hallucinations about facts not in the model's weights.
2. **Not applied to code.** Evaluated on factual QA only. The probing heads detect factual hallucinations, not code errors.
3. **Requires probing head training** per model architecture.
4. **No external verifier.** Uses only internal model signals, not LSP/compiler/type checker.

### 2.8 Key Figures

- **Figure 2:** Full DSVD framework: parallel decoding + verifying, rollback, candidate sampling, resumption.
- **Algorithm 1:** Complete pseudocode.
- **Table 5:** Latency comparison showing ~5-20% overhead.

---

## 3. LEVER: Learning to Verify Language-to-Code Generation with Execution (Ni et al., ICML 2023)

**ArXiv:** https://arxiv.org/abs/2302.08468

### 3.1 Problem Statement

Code LLMs in few-shot settings often produce the correct program among sampled candidates, but standard decoding selects the wrong one. Prior heuristics (error pruning, majority voting) cannot capture rich semantic features of execution outputs. LEVER trains a learned verifier that uses execution results to rerank samples.

### 3.2 Core Mechanism

Three stages:
1. **Generation:** Sample $k$ programs $\{\hat{y}_1, \ldots, \hat{y}_k\}$ from the LLM with temperature sampling; deduplicate.
2. **Execution:** Execute each candidate $\hat{y}$ via an executor $E(\cdot)$ to obtain its execution result $E(\hat{y})$ -- this may be a SQL result table, a numerical answer, a Python return value, or an error message.
3. **Verification:** A trained binary classifier $P_\theta(v | x, \hat{y}, E(\hat{y}))$ takes the natural language task description $x$, the candidate program $\hat{y}$, and its execution result $E(\hat{y})$, and outputs the probability that the program is correct ($v = 1$). The final **reranking score** $P_R$ combines the LLM's generation probability with the verifier's probability:

$$P_R(\hat{y}, v=1 | x) = P_{\text{LM}}(\hat{y} | x) \cdot P_\theta(v=1 | x, \hat{y}, E(\hat{y}))$$

Programs with identical execution results are grouped and their scores summed.

**Verifier architecture:** T5-base/T5-large or RoBERTa-large (0.5% the size of the generator). Trained on the model's own generations labeled by comparing execution results to ground truth.

### 3.3 Results

With Codex (code-davinci-002):

| Benchmark | Greedy | EP+ML (heuristic) | LEVER |
|---|---|---|---|
| Spider (text-to-SQL) | 75.3 | 77.3 | **81.9** |
| WikiTQ (table QA) | 53.0 | 54.9 | **65.8** |
| GSM8k (math) | 67.2 | 72.1 | **84.5** |
| MBPP (Python) | 61.1 | 62.2 | **68.9** |

All were SOTA at time of publication.

### 3.4 Key Findings

- Execution results are critical: removing them drops performance 1.2-6.6%.
- Data-efficient: improvements persist with as few as 250 training examples.
- Cross-LLM transfer: verifiers trained on one LLM improve programs from another.

### 3.5 Limitations

- Requires programs to be executable (test inputs or database must be available).
- Safety concerns from executing arbitrary generated code.
- Post-generation reranking, not online verification during generation.
- Sampling budget (20-100 programs) adds computational cost.

---

## Comparative Summary

| Paper | Verification Signal | When | Rollback? | Scope | Applied to Code? |
|---|---|---|---|---|---|
| **MGD** | LSP static analysis (types, cross-file resolution) | During decoding, at trigger points | No (abandons on failure) | Semantic (identifiers) | Yes (Java, C#, Rust) |
| **DSVD** | Internal probing heads on hidden states | During decoding, every token | Yes (KV-cache truncation) | Factual correctness | No (QA tasks only) |
| **LEVER** | Execution results + trained verifier | After generation (reranking) | No (selects among samples) | Functional correctness | Yes (SQL, Python, math) |

**Key gap:** No system combines external semantic verification (MGD's LSP) with token-level rollback (DSVD's mechanism). MGD constrains forward but can't undo; DSVD rolls back but uses internal signals, not external tools.
