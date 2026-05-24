# SemGuard reproduction

**Paper.** SemGuard — semantic-evaluator-guided decoding with rollback.

**Upstream.** https://github.com/wwwql/SemGuard
**Pinned commit.** `18dadd7578dfbdb8d087eb12ef08d817c28e26ec` (2026-04-07) — see `UPSTREAM_COMMIT.txt`.

## What's in `upstream/`

Top-level files vendored from the upstream repo:

| File / dir              | Purpose                                                              |
|-------------------------|----------------------------------------------------------------------|
| `README.md`             | Upstream README                                                      |
| `train_base_model.py`   | Step 1: fine-tunes the base model on full code in SemDiff            |
| `critic/`               | Step 2: trains the partial-code evaluator (see `critic/run.sh`)      |
| `generate.py`           | Vanilla generation baseline                                          |
| `generate_sem.py`       | Step 3: generation under evaluator-guided rollback (the SemGuard loop) |
| `metric/`               | Step 4: evaluation harness; wraps Evalplus / LiveCodeBench           |
| `configs/`              | Upstream training/eval configs                                       |
| `data/`                 | `SemDiff.rar`, `SemDiff-Java.rar` — the SemDiff dataset (compressed) |
| `Datasets/`             | Additional dataset glue                                              |
| `assets/`               | Paper figures                                                        |
| `transformers/`         | **Vendored HuggingFace transformers (fork of v4.45.2)** — installed editable; the decoding hooks SemGuard needs aren't in mainline `transformers` |

## Environment

SemGuard is pip-based and depends on its **vendored transformers fork**. Do
not mix this env with the main project's `uv` env (the transformers pin will
collide).

Suggested setup, from this directory:

```bash
# from reproduce/algo/semguard/
uv venv .venv --python 3.10
source .venv/bin/activate
pip install -r upstream/requirements.txt
pip install -e upstream/transformers
```

(Not run by the scaffolding step — see project plan.)

## Dataset

`upstream/data/SemDiff.rar` and `upstream/data/SemDiff-Java.rar` ship in the
repo and contain the training data. **Not yet extracted** — the user will
extract them (requires `unrar`) into `reproduce/data/semguard/` so they live
outside the gitignored upstream tree but inside the gitignored `data/` tree.

## ⚠️ Reproduction requires retraining the evaluator

> The paper claims "we publicly release the SemDiff dataset, trained
> evaluators, and the complete, open-source implementation."

The published GitHub README **does not** link evaluator checkpoints. The
dataset and code are present; the trained evaluator weights are not. Full
reproduction therefore requires running steps 1 + 2 from the upstream README
locally:

```bash
python upstream/train_base_model.py
cd upstream/critic && bash run.sh
```

**Cost estimate (rough).** The classifier is ~1.3B params. Per
`(backbone × language)` evaluator: ~6–12 GPU-hours on one H100. With the four
backbones × two languages used in the paper, that's ~2–4 GPU-days on a single
H100. This is gated on the user's compute decision — do not start training
during scaffolding.

## Original run instructions (quoted from `upstream/README.md`)

> ## 1. Finetune Base Models
> First, fine-tune the base model on the complete code of the SemDiff dataset by running the following code:
> ```bash
> python train_base_model.py
> ```
> ## 2. Training Evaluator
> Second, train the evaluator on the partial code of the SemDiff dataset by running the following code:
> ```bash
> cd critic
> bash run.sh
> ```
> ## 3. Generate Code
> Third, control the decoding and generation of code with the help of an evaluator:
> ```bash
> python generate_sem.py
> ```
> ## 4. Test Results
> We use [Evalplus](https://github.com/evalplus/evalplus), [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) for evaluation of MBPP, LiveCodeBench respectively.
> Final, you can use the following code to test SemDiff.
> ```bash
> cd metric
> bash test_one_solution.sh
> ```

## Anticipated local modifications

Tracked in `MODIFICATIONS.md`. Planned items:

1. **JSONL logging hook** — emit one `run_<timestamp>_<hash>.jsonl` per
   generation run, matching the schema under
   `<project_root>/results/web_demo/`, so SemGuard runs can be ingested by the
   same analysis notebooks as SoundCode runs. The hook attaches inside
   `generate_sem.py`.

## Configs

Project-side config overrides go in `configs/` and are loaded by the
dispatcher in `reproduce/scripts/` (added later). Nothing there yet.
