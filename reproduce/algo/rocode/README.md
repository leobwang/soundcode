# ROCODE reproduction

**Paper.** *ROCODE: Integrating Backtracking Mechanism and Program Analysis in Large Language Models for Code Generation* — [arXiv:2411.07112](https://arxiv.org/abs/2411.07112)

**Upstream.** https://github.com/jiangxxxue/ROCODE
**Pinned commit.** `49037ff6b98843d68138d3c6761f60c30f1381ce` (2025-12-16) — see `UPSTREAM_COMMIT.txt`.

## What's in `upstream/`

Top-level files vendored from the upstream repo:

| File / dir                     | Purpose                                                    |
|--------------------------------|------------------------------------------------------------|
| `main.py`                      | Entry point for HumanEval / MBPP runs                      |
| `main_codeforces.py`           | Entry point for the CodeForces2305 benchmark               |
| `run.sh`                       | Shell wrapper for HumanEval / MBPP                         |
| `run_codeforces.sh`            | Shell wrapper for CodeForces2305                           |
| `evaluate_generated_code.py`   | Metrics (PassRate, AvgPassRate, Compiler Correctness %)    |
| `evaluate/`                    | Per-benchmark evaluation harnesses                         |
| `data/`                        | Bundled benchmark inputs                                   |
| `config.py`                    | Hyperparameters / model selection                          |
| `detect_repeat.py`             | Repetition detection used by the backtracking mechanism    |
| `PA_tools.py`                  | Program-analysis utilities (the "PA" in ROCODE)            |
| `tire_tree.py`                 | Trie used by the backtracking search (file is misspelled in upstream) |
| `utils.py`                     | Misc helpers                                               |
| `environment.yml`              | Conda environment spec — *this is the canonical install path* |
| `figures/`                     | Paper figures (not needed for execution)                   |
| `LICENSE`                      | Upstream license                                           |
| `README.md`                    | Upstream README                                            |

## Environment

ROCODE ships a conda `environment.yml`. The main SoundCode project uses `uv`;
do **not** mix the two. Create a dedicated conda env:

```bash
# from this directory (reproduce/algo/rocode/)
conda env create -f upstream/environment.yml -n rocode-repro
conda activate rocode-repro
```

(Not run by the scaffolding step — see project plan.)

## Original run instructions (quoted from `upstream/README.md`)

> Run our approach and evaluate it by executing the `run.sh` script or the
> `run_codeforces.sh` script.
>
> ```bash
> # Run ROCODE on HumanEval and MBPP datasets
> sh run.sh
>
> # Run ROCODE on CodeForces2305 dataset
> run_codeforces.sh
> ```
>
> - `main.py`: Executes our ROCODE approach.
> - `evaluate_generated_code.py`: Calculates metrics such as PassRate, AvgPassRate, and Compiler Correctness Percentage.

The upstream backend is plain HuggingFace `transformers`; we keep this backend
for the canonical reproduction (per the project plan).

## Anticipated local modifications

Tracked in `MODIFICATIONS.md`. Planned items:

1. **JSONL logging hook** — emit one `run_<timestamp>_<hash>.jsonl` per benchmark
   run, matching the schema under `<project_root>/results/web_demo/`, so ROCODE
   runs can be ingested by the same analysis notebooks as SoundCode runs.
2. **Optional Ollama backend adapter** — for apples-to-apples wall-clock
   comparison with SoundCode (which serves models via Ollama). Only added if
   the HF-vs-Ollama latency gap turns out to dominate the comparison.

## Configs

Project-side config overrides (sweep grids, model lists, seeds) go in `configs/`
and are loaded by the dispatcher in `reproduce/scripts/` (added later). Nothing
there yet.
