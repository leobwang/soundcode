# Project Guidelines

## Python environment
- Use the `uv` environment for all Python execution: run scripts with `uv run python <script>`, install packages with `uv add <package>`.
- Do not use `pip install`, `python` directly, or create separate virtual environments.

## Presenting Papers

### Source material
- Always read the original `.pdf` of the paper (in `lit-rev/papers/`), not just `.md` summaries, before presenting or discussing a paper — unless the PDF is already in context.

### Overview structure
- Preserve the structure of the paper: section titles, hierarchy of definitions, theorems, algorithms.
- Be clear about:
  1. The problem the paper tries to solve
  2. Background knowledge needed to understand the approach
  3. The method proposed by the authors
  4. Results and evaluation of the method
  5. Final takeaways
- Do not assume the reader has background beyond the following. When using symbols or terms outside this scope, expand their definitions.
  - **Math** (proficient undergrad): multivariable calculus, linear algebra, probability & statistics, real analysis, differential equations, stochastic processes, discrete math
  - **CS** (proficient graduate, select areas): machine learning, computer vision, NLP, reinforcement learning, algorithms, databases, systems programming
  - **Languages/tools**: Python, C/C++, Rust, Java, JavaScript, LaTeX, PyTorch
- Every symbol must be defined before or at its first use in an equation. Do not introduce symbols (e.g., $V(s)$, $\alpha$, $D_\gamma$) in a formula without explaining what they denote.
- Identify which figures and tables are important to the overall understanding of the paper and include them in the presentation.
