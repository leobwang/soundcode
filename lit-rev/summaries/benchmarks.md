# Benchmarks for Code Generation

Papers on benchmarks used to evaluate LLM code generation, code reasoning, and repository-level completion.

---

## 1. HumanEval / Codex: Evaluating Large Language Models Trained on Code (Chen et al., 2021)

**ArXiv:** https://arxiv.org/abs/2107.03374

### 1.1 What It Measures

164 hand-written Python programming problems measuring **functional correctness** of generated code. Each problem consists of a function signature, docstring, and unit tests (7.7 tests on average). A generated sample is correct iff it passes all unit tests.

### 1.2 The pass@k Metric

Given $n$ generated samples per problem (with $n = 200$ in practice), let $c$ denote the number that pass all tests. The unbiased estimator of pass@k is:

$$\text{pass@}k := \mathbb{E}_{\text{problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]$$

where $\binom{a}{b}$ is the binomial coefficient ("$a$ choose $b$"), $n$ is total samples, $c$ is correct samples, and $k$ is the draw budget. This computes the probability that at least one of $k$ random draws is correct. Optimal temperature: $T = 0.2$ for pass@1 (greedy-like), $T = 0.8$ for pass@100 (diversity).

### 1.3 Key Findings

- Codex-12B: 28.8% pass@1, 77.5% pass@100.
- Performance scales as a sigmoid in log-parameters.
- BLEU score is a poor proxy for functional correctness.
- Performance degrades exponentially with number of chained operations in the docstring.

### 1.4 Limitations

Small (164 problems), Python-only, static (susceptible to contamination), relatively easy.

---

## 2. MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation (Cassano et al., 2023)

**ArXiv:** https://arxiv.org/abs/2208.08227

### 2.1 What It Measures

Functional correctness of LLM-generated code across **18 programming languages** including Rust. Translates HumanEval (164 problems) and MBPP (401 problems) from Python to each target language via rule-based compilers (~200 LOC each).

### 2.2 How Translation Works

The *benchmarks themselves* are translated (function signatures, docstrings, doctests, unit tests), not the solutions. The model is prompted in the target language and must synthesize the function body from scratch.

**Rust-specific handling:**
- Targets Rust 1.59.0, uses `///` doc comments
- Type mappings: `str` -> `String`, `List` -> `Vec`, `Tuple` -> Rust tuple, `dict` -> `HashMap`, `Optional` -> `Option`, `int` -> `isize`, `float` -> `f64`
- All values are owned (not borrowed)
- Problems using `Union`, `Any`, or `Ellipsis` are excluded (up to 5 HumanEval problems dropped)
- Rust is categorized as **Low**-frequency (1.1% GitHub share)

### 2.3 Key Findings

- Codex achieves pass@1 above 40% on Rust HumanEval, upper-middle tier among languages.
- JavaScript often matches or exceeds Python performance despite benchmarks originating in Python.
- No overall effect of type annotations on pass@1 ($p = 0.33$ for HumanEval).
- Language frequency correlates with performance but not deterministically.
- Translating Python-specific terminology (e.g., "list" -> "vector") improves Rust performance.

### 2.4 Relevance to This Project

Most directly applicable functional correctness benchmark for Rust. Provides ready-made Rust test harness with unit tests and pass@k evaluation. However, problems are **single-function, self-contained** -- no cross-file dependencies, crate imports, or complex ownership/lifetime challenges. Serves as a baseline for standalone function synthesis.

---

## 3. CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution (Gu et al., 2024)

**ArXiv:** https://arxiv.org/abs/2401.03065

### 3.1 What It Measures

800 short Python functions (3-13 lines) testing **code reasoning** rather than code generation:

- **CRUXEval-O (output prediction):** Given function `f` and input, predict the output. Tests mental execution.
- **CRUXEval-I (input prediction):** Given function `f` and output, predict any input producing that output. Tests semantic inversion.

Both use execution-based correctness (prediction is run as Python code).

### 3.2 Key Findings

- GPT-4 with CoT: 74.8% on input prediction, 81.9% on output prediction -- far from perfect on simple functions.
- **HumanEval gains do not transfer to CRUXEval.** Models distilled on GPT-3.5/4 data dramatically improve HumanEval but show no improvement on CRUXEval. E.g., WizardCoder 34B improves HumanEval from 53.7% to 73.2% but does not outperform Code Llama 34B on CRUXEval.
- CoT helps output prediction more than input prediction.
- GPT-4 consistently fails on simple string manipulations (likely tokenization-related).

### 3.3 Relevance

Demonstrates that strong generation performance is necessary but not sufficient for code understanding. Provides a complementary signal that guards against models scoring well through memorization.

---

## 4. LiveCodeBench: Holistic and Contamination Free Evaluation (Jain et al., 2024)

**ArXiv:** https://arxiv.org/abs/2403.07974

### 4.1 What It Measures

511+ coding problems from weekly contests on LeetCode, AtCoder, and CodeForces (May 2023 -- May 2024). Four evaluation scenarios:

1. **Code Generation:** Generate correct program from NL description. Pass@1 via 10 samples.
2. **Self-Repair:** Fix an incorrect program given error feedback. Tests debugging.
3. **Code Execution:** Predict output of a function given input (adapted from CRUXEval).
4. **Test Output Prediction:** Predict expected output from NL description + test input, without code.

### 4.2 How It Avoids Contamination

Each problem is tagged with a release date $D$. For any model with known training cutoff, the benchmark filters to problems released *after* that cutoff. This revealed contamination: DeepSeek-Instruct-33B shows a stark performance drop on LeetCode problems after August 2023.

### 4.3 Key Findings

- **HumanEval overfitting is real.** Fine-tuned open models lie above the diagonal when plotting HumanEval+ vs. LCB performance -- disproportionately good on HumanEval without matching on fresh problems.
- GPT-4-Turbo leads by 16.2 points over DeepSeek-Instruct-33B on LCB despite only 4.3-point gap on HumanEval+.
- CoT substantially improves code execution (GPT-4-Turbo: 64.8% -> 83.6%).

### 4.4 Relevance

Essential for credible evaluation: ensures improvements from constrained decoding are not due to memorized solutions.

---

## 5. CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion (Ding et al., 2023)

**ArXiv:** https://arxiv.org/abs/2310.11248

### 5.1 What It Measures

**Statement-level code completion where the correct completion depends on information in a different file.** Supports Python, Java, TypeScript, C# (not Rust). ~9,928 examples.

### 5.2 How It Works

Dataset construction via static analysis:
1. Replace intra-project imports with empty class stubs.
2. Run static analyzers (Pylint, `javac`, `tsc`, `csc`) to identify undefined-name errors -- these pinpoint cross-file dependencies.
3. Map error locations back to original file to determine completion targets.

**Evaluation metrics:** Exact Match (EM), Edit Similarity (ES, defined as $1 - \text{normalized Levenshtein distance}$), and Identifier Match (F1 over extracted identifier lists).

### 5.3 Key Findings

- **In-file context alone is grossly insufficient.** StarCoder-15.5B: 8.82% EM on Python with in-file only.
- Cross-file retrieval yields up to 3x improvement (8.82% -> 15.72% EM with BM25, 21.01% with oracle).
- Even oracle retrieval remains far from perfect (~21% EM).
- BM25 is a strong retrieval baseline, often beating neural encoders.

### 5.4 Relevance

Directly motivates LSP-based approaches: an LSP server can provide precisely the cross-file information that BM25 approximates crudely. The static-analysis-based dataset construction methodology could be adapted to Rust using `rustc` or rust-analyzer diagnostics.

---

## 6. RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems (Liu et al., 2023)

**ArXiv:** https://arxiv.org/abs/2306.03091

### 6.1 What It Measures

Repository-level code completion in Python and Java (not Rust), decomposed into three sub-tasks:

- **RepoBench-R (Retrieval):** Select the most relevant cross-file code snippet from candidates. Evaluated with acc@$k$ (fraction where gold snippet appears in top $k$).
- **RepoBench-C (Completion):** Predict the next line given cross-file context $C_x$ and in-file context $C_{\text{in}}$. Evaluated with Exact Match (EM) and Edit Similarity (ES).
- **RepoBench-P (Pipeline):** End-to-end retrieval + completion.

### 6.2 Key Findings

- Cross-file context substantially improves completion (37.64 EM vs. 26.64 EM for in-file only on Python).
- Placing most-relevant snippet closest to completion point yields best results.
- Even random cross-file context helps vs. in-file-only.
- StarCoder shows inconsistent performance with long contexts.

### 6.3 Relevance

An LSP server performs a structured version of RepoBench-R: instead of token-level similarity, it uses semantic analysis (go-to-definition, type hover) to identify exactly the right cross-file information. RepoBench demonstrates retrieval quality is a bottleneck, suggesting LSP's structured context could improve over bag-of-tokens retrieval.

---

## Comparative Summary

| Benchmark | Year | Languages | Granularity | Cross-file? | Contamination-resistant? | Key metric |
|---|---|---|---|---|---|---|
| **HumanEval** | 2021 | Python | Function | No | No | pass@k |
| **MultiPL-E** | 2023 | 18 (incl. Rust) | Function | No | No | pass@k |
| **CRUXEval** | 2024 | Python | Function | No | No | Execution accuracy |
| **LiveCodeBench** | 2024 | Python | Function/Program | No | **Yes** (time-windowed) | pass@1 |
| **CrossCodeEval** | 2023 | Python, Java, TS, C# | Statement | **Yes** | Yes (post-cutoff repos) | EM, ES, Identifier F1 |
| **RepoBench** | 2023 | Python, Java | Line | **Yes** | Yes (post-cutoff repos) | EM, ES, acc@k |

**Gap for this project:** No existing cross-file benchmark covers Rust. MultiPL-E covers Rust but only single-function generation. A Rust-specific repository-level benchmark -- using rust-analyzer's diagnostics to identify cross-file dependencies (analogous to CrossCodeEval's methodology) -- would be the ideal evaluation target for an LSP-guided system.
