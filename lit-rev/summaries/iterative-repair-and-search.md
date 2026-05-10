# Iterative Repair and Search-Based Code Generation

Papers on systems that improve code through iterative debugging loops, self-reflection, or tree search over candidate programs.

---

## 1. Self-Debugging: Teaching Large Language Models to Self-Debug (Chen et al., ICLR 2024)

**ArXiv:** https://arxiv.org/abs/2304.05128

### 1.1 Problem Statement

LLMs frequently produce incorrect programs on the first attempt. Prior approaches sample many candidates and rerank, or train separate repair models. Self-Debugging asks: can a single pretrained LLM iteratively debug its own code without human feedback or additional training?

### 1.2 Core Mechanism

A few-shot prompting framework with three steps per debugging turn:

1. **Generation.** LLM produces a candidate program.
2. **Explanation.** LLM processes its prediction via:
   - *Code explanation:* Line-by-line natural language explanation (rubber duck debugging).
   - *Execution trace:* Simulated intermediate execution steps.
3. **Feedback.** Correctness signal assembled from:
   - *Simple feedback:* Binary correct/wrong sentence.
   - *Unit test (UT) feedback:* Actual execution against tests; error messages included in prompt.
   - *Code explanation (Expl) feedback:* Model reads its own explanation, compares to problem statement.

Loop terminates when model declares prediction correct or max turns (10) reached. Greedy decoding, no fine-tuning.

### 1.3 Key Results

| Benchmark | Task | Best Self-Debugging Gain |
|---|---|---|
| Spider (text-to-SQL) | No unit tests | +2-3%; +9% on "extra hard" (81.3 -> 84.1) |
| TransCoder (C++ -> Python) | Unit tests available | +12% (80.4 -> 92.5 with UT+Expl) |
| MBPP (text-to-Python) | Partial unit tests | +9.4% (61.4 -> 70.8 with Trace) |

- Self-Debugging with 1 sample matches baselines sampling 10-32 candidates.
- Without code execution: Trace feedback alone still improves by up to 5%.
- Most common fixes: wrong WHERE conditions (25.7%), missing DISTINCT (17.1%), wrong JOINs (14.3%).

### 1.4 Limitations

- Code explanation less useful when initial code is far from correct.
- Simple feedback without explanation does not help on tasks without unit tests.
- Limited to models strong enough to generate useful code explanations.
- **Full-program-level iteration.** Does not intervene during generation.

---

## 2. Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., NeurIPS 2023)

**ArXiv:** https://arxiv.org/abs/2303.11366

### 2.1 Problem Statement

LLM agents cannot efficiently learn from trial-and-error. Traditional RL requires gradient updates. Reflexion asks: can an LLM agent improve across episodes using only natural language feedback stored in memory, without weight updates?

### 2.2 Core Mechanism: Verbal Reinforcement

Three components:

1. **Actor** ($M_a$): Standard LLM agent generating actions conditioned on task, observations, and episodic memory.
2. **Evaluator** ($M_e$): Scores trajectories via exact-match, heuristics, unit test execution, or LLM classification.
3. **Self-Reflection** ($M_{sr}$): Generates natural-language summary of what went wrong and what to try differently. Appended to long-term memory buffer.

**The loop:**
- Trial 0: Actor generates; Evaluator scores; Self-Reflection produces verbal reflection; store in memory.
- Trial $t$: Actor conditioned on past reflections; if failed, reflect and append.
- Terminate when correct or max trials reached. Memory bounded to last 1-3 reflections.

**For code generation:** Evaluator uses **self-generated unit tests** (not ground-truth). Tests generated via Chain-of-Thought, filtered by AST parsing, executed against candidate code.

### 2.3 Key Results

| Benchmark | Base (GPT-4) | Reflexion (GPT-4) |
|---|---|---|
| HumanEval (Python) | 80.1 | **91.0** |
| HumanEval (Rust) | 60.0 | **68.0** |
| LeetcodeHard (Python) | 7.5 | **15.0** |

- 91% pass@1 on HumanEval surpassed previous GPT-4 SOTA.
- Ablation: removing self-reflection (keeping test execution) yields no improvement; removing test generation drops accuracy from 68% to 52%. Both components are needed.

### 2.4 Limitations

- No formal convergence guarantee; can get stuck in local minima.
- Relies on self-evaluation quality; flaky self-generated tests cause false positives (16.3% on MBPP Python).
- Emergent capability: no benefit with weaker models.
- **Full-program retry**, not fine-grained intervention during generation.

---

## 3. LATS: Language Agent Tree Search (Zhou et al., ICML 2024)

**ArXiv:** https://arxiv.org/abs/2310.04406

### 3.1 Problem Statement

Existing agent frameworks (ReAct, Reflexion, ToT) each address some aspects of reasoning, acting, and planning, but none unifies all three. ReAct and Reflexion are greedy (single trajectory). ToT explores multiple paths but lacks environment feedback. LATS asks: can we systematically search over action paths while incorporating environment feedback and learning from experience?

### 3.2 Core Mechanism: MCTS for LLM Agents

LATS adapts **Monte Carlo Tree Search (MCTS)** -- a search algorithm that builds a tree of possible action sequences and uses random simulations to estimate the value of each node -- to the LLM agent setting.

Each node in the search tree is a state $s = [x, a_{1..i}, o_{1..i}]$, where $x$ is the original input (e.g., problem description), $a_{1..i}$ is the sequence of actions taken so far, and $o_{1..i}$ is the corresponding sequence of observations (environment feedback). The algorithm runs $K$ iterations, each consisting of six operations:

1. **Selection:** Starting from the root, traverse the tree by selecting the child with the highest **UCT (Upper Confidence bound applied to Trees)** score:

   $$\text{UCT}(s) = V(s) + w \sqrt{\frac{\ln N(\text{parent}(s))}{N(s)}}$$

   where $V(s)$ is the estimated value (expected reward) of state $s$, $N(s)$ is the number of times state $s$ has been visited during search, $N(\text{parent}(s))$ is the visit count of $s$'s parent node, and $w$ is the exploration weight (default $w = 1$). The first term favors high-value nodes (exploitation); the second term favors under-visited nodes (exploration).

2. **Expansion:** At the selected leaf node, sample $n$ candidate actions from the LLM $p_\theta$; execute each action in the environment to obtain an observation, creating $n$ new child nodes.

3. **Evaluation:** Each new child node receives a value from a composite value function:

   $$V(s) = \lambda \cdot \text{LM}(s) + (1-\lambda) \cdot \text{SC}(s)$$

   where $\text{LM}(s)$ is the LLM's evaluation of the trajectory's correctness (prompted to output a scalar score 1-10 after seeing environment feedback), $\text{SC}(s)$ is a **self-consistency score** (actions sampled multiple times at the same state are considered more reliable), and $\lambda$ is a mixing hyperparameter ($\lambda = 0.5$ for reasoning tasks, $\lambda = 0.8$ for programming).

4. **Simulation:** Expand the most promising node (by value) until a terminal state is reached.

5. **Backpropagation:** When a terminal state is reached with reward $r$, update all nodes along the path: increment visit counts $N(s)$ and update value estimates $V(s)$ as running averages.

6. **Reflection:** On failure, the LLM generates a verbal self-reflection summarizing errors and proposing alternatives (as in Reflexion). Stored for subsequent iterations.

**For code generation:** Each "action" is a complete solution (no intermediate steps). Evaluation uses LLM-generated internal tests (4-6 tests); the percentage of tests passed serves as the reward $r$.

### 3.3 Key Results

| Method | Model | HumanEval | MBPP |
|---|---|---|---|
| Reflexion | GPT-3.5 | 68.1 | 70.0 |
| **LATS** | GPT-3.5 | **83.8** | **81.1** |
| Reflexion | GPT-4 | 91.0 | -- |
| **LATS** | GPT-4 | **92.7** | -- |

92.7% pass@1 on HumanEval was highest reported at publication.

**Ablation:** Removing LM heuristic (value function): 0.63 -> 0.37 on HotPotQA (massive drop). Removing reflection: 0.63 -> 0.58 (modest).

### 3.4 Limitations

- Higher computational cost than greedy approaches (many LLM calls).
- Requires environment reversion (ability to reset to earlier states).
- **Operates at the program/action level**, not within the decoding process. Each node is a complete program, not a partial token sequence.

---

## Comparative Summary

| Paper | Intervention Level | Search Strategy | Feedback Signal | Multiple Attempts? |
|---|---|---|---|---|
| **Self-Debugging** | Full program | Linear retry | Execution + self-explanation | Yes (sequential) |
| **Reflexion** | Full program | Linear retry with memory | Execution + verbal reflection | Yes (sequential, with memory) |
| **LATS** | Full program | MCTS tree search | Execution + LM value function | Yes (parallel branches) |

**Common pattern:** All three operate at the full-program level -- generate a complete program, evaluate it, then retry or branch. None intervenes *during* the token-level generation process. The feedback loop is coarse-grained (program-level), not fine-grained (token/line-level).

**Key gap:** These approaches waste computation by generating complete programs before discovering errors. A finer-grained approach that detects errors as they emerge during generation could prevent cascading mistakes and reduce wasted tokens.
