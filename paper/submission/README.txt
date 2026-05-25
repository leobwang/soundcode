SoundCode: Async LSP-Supervised Decoding for Code Generation
============================================================

Title:  Async LSP-Supervised Decoding for Code Generation
Author: Leo Wang (University of Chicago, psixyzt@gmail.com)

One-line abstract
-----------------
At fixed model, fixed verifier, and fixed rollback algorithm, running
the verifier asynchronously instead of synchronously on the decoding
critical path yields 1.59x wall-time speedup on Rust and 2.95x on C++
on MultiPL-E HumanEval with Qwen2.5-Coder-7B-Instruct, while raising
C++ pass@1 from 71.4% to 76.4%.

Build instructions
------------------
From this directory:

    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex

The pre-built PDF is `main.pdf`.

Artifacts
---------
Live demo: https://soundcode-demo.psixyzt.com
Source:    https://github.com/leobwang/soundcode

Directory contents
------------------
main.tex          - top-level LaTeX scaffold
main.pdf          - pre-built PDF (17 pages)
references.bib    - 26 BibTeX entries
neurips_2024.sty  - NeurIPS 2024 style file (preprint mode)
sections/         - intro, related_work, method, results, conclusion
figures/          - 6 workflow figures + 6 result plots
