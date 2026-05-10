# Retrieval-Augmented Code Generation

Papers on retrieving documentation, code examples, or other external knowledge to improve LLM code generation quality.

---

## 1. DocPrompting: Generating Code by Retrieving the Docs (Zhou et al., ICLR 2023)

**ArXiv:** https://arxiv.org/abs/2207.05987

### 1.1 Problem Statement

Code generation models are trained on a fixed corpus and cannot generalize to unseen libraries, functions, or APIs that appear after training. Human programmers routinely consult documentation. DocPrompting asks: can we give code generation models the same ability by retrieving relevant documentation at inference time?

### 1.2 Core Mechanism

A **retrieve-then-generate** framework with two components:

1. **Retriever R.** Given a natural language intent $n$, scores every document $d_i$ in a documentation pool $D$ and returns the top-$k$ most relevant documents.
   - *Sparse*: BM25 via Elasticsearch.
   - *Dense*: Neural encoder (SimCSE or CodeT5-base) trained contrastively with positive pairs $(n, d_i^+)$ and in-batch negatives.

2. **Generator G.** Produces code conditioned on intent and retrieved docs.
   - GPT-Neo (125M, 1.3B) and Codex: concatenate docs with intent as prompt.
   - T5/CodeT5: use Fusion-in-Decoder (FiD) -- each $(n, d_i)$ pair encoded independently, decoder attends to all.

### 1.3 Documentation Types

- **Shell scripting (tldr benchmark):** 400k paragraphs from Unix `man` pages for 1,879 Bash commands. Train/test splits use completely disjoint commands.
- **Python (CoNaLa, re-split):** 35,763 documents from DevDocs for built-in and popular library functions. Test set requires at least one unseen function.

### 1.4 Key Results

- **Shell (tldr):** T5+DocPrompting achieves >2x higher command name accuracy (30.28% vs. 10.02%) and +9% absolute exact match over vanilla T5.
- **Python (CoNaLa):** CodeT5+DocPrompting improves BLEU by 1.65 points and >2x unseen function recall (18.30 vs. 9.03). 2.85% pass@1 improvement (52% relative).
- **Documentation vs. examples:** Retrieving documentation outperforms retrieving NL-code example pairs, because documentation is more reliably available.
- **Why it works:** Documentation increases n-gram overlap between input and target code (e.g., unigram overlap 12% -> 24% on tldr).

### 1.5 Limitations

- Retriever quality is a bottleneck: irrelevant docs from similarly named functions cause errors.
- Gains are smaller for very strong models (Codex), possibly due to pretraining data overlap.
- Documentation pool must be manually curated and maintained.
- **Retrieval happens once, before generation** -- not interleaved during generation.

---

## 2. FLARE: Forward-Looking Active REtrieval Augmented Generation (Jiang et al., EMNLP 2023)

**ArXiv:** https://arxiv.org/abs/2305.06983

### 2.1 Problem Statement

Standard RAG retrieves once based on the user's input. For long-form generation, information needs emerge *during* generation and are not evident from the input alone. FLARE asks: can we build a system that actively decides *when* and *what* to retrieve throughout generation?

### 2.2 Core Mechanism

**FLARE_direct** (primary method) operates at **sentence granularity**:

1. **Generate a temporary next sentence** $\hat{s}_t$ without new retrieval.
2. **Decide whether to retrieve:** If any token in $\hat{s}_t$ has probability below threshold $\theta$, trigger retrieval. Low probability indicates the LM lacks knowledge.
3. **Formulate a query** from $\hat{s}_t$:
   - *Implicit (masking):* Mask out low-confidence tokens (probability below $\beta$) and use remaining tokens as query.
   - *Explicit (question generation):* Prompt GPT-3.5-turbo to generate questions answerable by low-confidence spans.
4. **Retrieve and regenerate:** Use query to retrieve documents, prepend to context, regenerate the sentence.

Repeat sentence by sentence until completion. Only current retrieval step's documents are used (no accumulation).

**FLARE_instruct:** LM generates explicit `[Search(query)]` calls during generation, triggered by few-shot prompting.

### 2.3 What Triggers Retrieval

- **Trigger:** Token-level probability below threshold $\theta$.
- **Granularity:** Sentence-level (each iteration generates up to 64 tokens, first sentence extracted).
- **Empirically:** Triggering retrieval for 40-80% of sentences works best.

### 2.4 Key Results

| Task | No Retrieval | Single Retrieval | FLARE |
|---|---|---|---|
| 2WikiMultihopQA (EM) | 37.2 | 39.4 | **51.0** |
| StrategyQA (EM) | 72.9 | 71.5 | **77.3** |
| ASQA (EM) | 29.1 | 36.3 | **41.3** |

- Forward-looking retrieval (using the *next* sentence) consistently outperforms backward-looking (using the *previous* sentence).
- Masking low-confidence tokens improves over raw sentence queries.

### 2.5 Limitations

- Does not help on short-form generation tasks (~20 tokens).
- Increases latency: multiple LM activations per output.
- Can propagate errors: incorrect temporary sentence may retrieve confirming (but wrong) documents.
- **Sentence-level granularity, not token-level.** Not designed for code generation.
- Evaluated on general text QA/summarization, not code.

---

## Comparative Summary

| Paper | Retrieval Timing | Granularity | What's Retrieved | Applied to Code? |
|---|---|---|---|---|
| **DocPrompting** | Before generation (one-shot) | Prompt-level | API documentation | Yes (Bash, Python) |
| **FLARE** | During generation (active) | Sentence-level | Wikipedia/web documents | No (general text QA) |

**Key gap for code generation:** No existing system continuously retrieves API documentation at the token/line level during autoregressive code generation. DocPrompting retrieves once before generation. FLARE retrieves during generation but at sentence granularity for general text, not structured API docs for code.
