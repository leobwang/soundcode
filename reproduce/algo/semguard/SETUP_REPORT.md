# SemGuard reproduction — setup report

Generated: 2026-05-24
Author: claude-code (CMSC 25750 reproduction track)

## Tooling

| Tool        | Status           | Notes                                                                  |
|-------------|------------------|------------------------------------------------------------------------|
| `uv`        | present          | `/home/leobwang/.local/bin/uv`                                         |
| `unrar`     | missing (system) | Worked around via static `unrar` binary from rarlab — see "SemDiff dataset" below |
| `nvidia-smi`| present          | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB total / 97184 MiB free |

To install `unrar`: `sudo apt install unrar` (Ubuntu/Debian). Alternative: `7z x SemDiff.rar` after `apt install p7zip-rar`.

## Per-paper venv

- Location: `reproduce/algo/semguard/.venv/`
- Python: CPython 3.10.20 (selected via `uv venv --python 3.10`)
- Already covered by `reproduce/.gitignore` (`algo/*/.venv/`)
- **Activation note.** The main project's `pyproject.toml` declares `requires-python >=3.12`. As a result, running `uv run` (with or without `--no-project`) inside `reproduce/algo/semguard/` ignores the local `.venv` and falls back to the main project env. **For SemGuard, invoke the interpreter directly: `reproduce/algo/semguard/.venv/bin/python ...`** (or `source .venv/bin/activate` first, then plain `python`). `uv pip install` honours the activated venv correctly.

## Dependencies

Installed inside `.venv`:

| Package      | Version | Source                                                          |
|--------------|---------|-----------------------------------------------------------------|
| torch        | 2.12.0  | PyPI (CUDA 13 wheel)                                            |
| transformers | 4.45.2  | **vendored fork** `upstream/transformers/` (installed editable) |
| peft         | 0.19.1  | PyPI                                                            |
| accelerate   | 1.13.0  | PyPI                                                            |
| bitsandbytes | 0.49.2  | PyPI                                                            |
| deepspeed    | 0.19.0  | PyPI                                                            |
| datasets     | (latest)| PyPI                                                            |
| sentencepiece, protobuf, scikit-learn, numpy, pandas, tqdm, pyext | (latest) | PyPI |

**Status: PASS** — no version conflicts, all packages resolved on first attempt.

### `requirements.txt` was missing upstream

The upstream `README.md` references `pip install -r requirements.txt`, but the file is not present at the pinned commit `18dadd75`. We reconstructed it from the imports actually used by `train_base_model.py`, `generate*.py`, `critic/run.py`, `metric/test_one_solution.py`, and the `Datasets/` glue, and wrote it back to `upstream/requirements.txt`. See `MODIFICATIONS.md` for the entry.

## SemDiff dataset

**Status: EXTRACTED** (2026-05-24, see `RUN_REPORT.md` and `MODIFICATIONS.md`).

Path forward without sudo: download the official static `unrar` from
rarlab.com and run it locally — no system install required.

```bash
wget -q https://www.rarlab.com/rar/rarlinux-x64-700.tar.gz -O /tmp/rarlinux.tar.gz
tar -xzf /tmp/rarlinux.tar.gz -C /tmp/
/tmp/rar/unrar x -y upstream/data/SemDiff.rar upstream/data/
/tmp/rar/unrar x -y upstream/data/SemDiff-Java.rar upstream/data/
```

(`pip install rarfile` won't help on its own — `rarfile` shells out to a
system `unrar` binary, so it inherits the same blocker. `patool` and
`pyunpack` were not exercised.)

Post-extract layout (after a small rename pass so both archives have the
same shape):

```
upstream/data/SemDiff/        # Python
  train.json    (40 453 rows)
  valid.json    ( 2 371 rows)
  test.json     ( 1 791 rows)
  train1/       per-example problem statements + reference solutions
  test1/
upstream/data/SemDiff-Java/   # Java
  train.json    (38 893 rows)
  valid.json    ( 2 121 rows)
  test.json     ( 1 837 rows)
  train/
  test/
```

These are the inputs `critic/run.py --data_dir <SemDiff_dir>` consumes via
`DatasetCA`, which reads `pos_code` / `neg_code` / `text` per row.

## GPU

```
NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  total VRAM: 97 887 MiB (~96 GiB)
  free VRAM:  97 184 MiB at time of report
```

This single GPU has more VRAM than an H100-80G, so any single-backbone training step should fit. Multi-GPU runs (`critic/run.sh` uses `accelerate launch --num_processes $gpucount`) are not relevant — we have one device.

## Training script: what it expects

`upstream/train_base_model.py` + `upstream/configs/train_base_model_configs.py`:

| Arg / setting                | Default                                            |
|------------------------------|----------------------------------------------------|
| `--model`                    | `bigcode/starcoder2-7b` *(not DeepSeek; see below)*|
| `--model_path`               | `None` (must be set explicitly)                    |
| `--train-path`               | `/data/SemDiff/train`                              |
| `--epochs`                   | 5                                                  |
| `--batch-size-per-replica`   | 2                                                  |
| `--grad-acc-steps`           | 16                                                 |
| `--lr`                       | 2e-5                                               |
| `--fp16`                     | True                                               |
| `--save-freq`                | 200                                                |

Backbone loading uses `load_in_8bit=True` + LoRA (`r=8`, `alpha=32`, `q_proj` + `v_proj`), so `bitsandbytes` and `peft` are mandatory at training time. **The evaluator** (`upstream/critic/run.py`, the actual SemGuard contribution) instead defaults to `--load deepseek-coder-1.3b-base`, which matches the "~1.3B classifier" figure in the project README.

Quick reading of the script confirms it can be imported cleanly; we did not run any training.

## Smoke test

```bash
cd reproduce/algo/semguard/upstream
../.venv/bin/python -c "import train_base_model; print('IMPORT_OK')"
# IMPORT_OK
```

**Status: PASS** — but only after two patches to upstream code (see `MODIFICATIONS.md`):
1. `Datasets/apps_dataset.py` imported `dataset_lm.reindent`, but the file is at `Datasets/reindent.py`.
2. `transformers/src/transformers/generation/utils.py` had top-level `from Transformers.src.transformers...` (capital T, sibling-dir layout) and `from Models.models import CodeT5ClassBCE`, which (a) don't resolve at this commit and (b) cause a circular import even if they did. The symbols are only used by the semantic-rollback hook; we moved them to lazy / function-scope imports.

## Cost estimate for the full retraining grid

Per the upstream paper and the project's own README:

- 4 LM backbones × 2 languages = 8 evaluator runs
- ~6–12 GPU-hours per evaluator on one H100
- **~2–4 GPU-days total** on our single 96 GiB GPU (RTX PRO 6000 ≈ H100-class for fine-tuning a 1.3B classifier with LoRA / 8-bit; throughput similar to H100, with more VRAM)

This excludes Step 1 (`train_base_model.py`) base-model fine-tunes, which are also paper-grid (4 backbones × 2 languages); those involve full 7B-class LMs and would add several more GPU-days.

**Not run during scaffolding.** Decision deferred to user.

## Commands to trigger full retraining (do not run unprompted)

```bash
cd reproduce/algo/semguard

# Step 1 — base-model fine-tune (one per (backbone, language)):
.venv/bin/python upstream/train_base_model.py \
    --model bigcode/starcoder2-7b \
    --model_path <local_path_or_hf_id> \
    --save_dir ../../../model/semguard/<run_name> \
    --train-path ../../../data/semguard/SemDiff/train

# Step 2 — evaluator training (one per (backbone, language)):
cd upstream/critic
bash run.sh <PORT> <GPU_IDS>          # e.g. bash run.sh 29500 0
#   internally: accelerate launch --num_processes N --main_process_port PORT \
#               --config_file ./acc_config.yaml ./run.py
#   default --load deepseek-coder-1.3b-base

# Step 3 — guided generation (after evaluator weights exist):
cd ..
.venv/bin/python generate_sem.py

# Step 4 — evaluate (Evalplus / LiveCodeBench harness):
cd metric
bash test_one_solution.sh
```

Substitute `<backbone>` over `{starcoder2-7b, deepseek-coder-6.7b-base, codellama-7b, qwen2.5-coder-7b}` (the four reported by the paper) and `<lang>` over `{python (SemDiff), java (SemDiff-Java)}`. Adjust `--train-path` for the Java set.

## What we did NOT do

- Did not download any model weights (would be tens of GiB; deferred).
- Did not extract the SemDiff RAR archives (no `unrar`; user to install).
- Did not train anything.
- Did not patch `generate_sem.py` for our JSONL logging convention (Week-5 task per `MODIFICATIONS.md` planning section; out of scope for setup).

(Items 1-3 in this list were addressed in the subsequent Week-5 kickoff
session — see `RUN_REPORT.md`. The `generate_sem.py` patch remains open.)
