# SemGuard reproduction — run report (Week 5 kickoff)

Generated: 2026-05-24
Author: claude-code (CMSC 25750 reproduction track)
Builds on: `SETUP_REPORT.md` (environment + patch log), `MODIFICATIONS.md`.

## 1. RAR extraction — **SUCCESS**

Method that worked: **official static `unrar` from rarlab**, downloaded without
sudo.

```bash
wget -q https://www.rarlab.com/rar/rarlinux-x64-700.tar.gz -O /tmp/rarlinux.tar.gz
tar -xzf /tmp/rarlinux.tar.gz -C /tmp/
/tmp/rar/unrar x -y upstream/data/SemDiff.rar upstream/data/
/tmp/rar/unrar x -y upstream/data/SemDiff-Java.rar upstream/data/
```

Did **not** need to fall back to `rarfile`, `patool`, or `pyunpack`. (`rarfile`
shells out to `unrar`, so it would have failed anyway on this box. Patool was
not exercised.)

After unpacking, the two archives use different layouts. I reorganized them
into the canonical layout the upstream README implies:

| Path                                                        | Rows (train/valid/test) |
|-------------------------------------------------------------|-------------------------|
| `upstream/data/SemDiff/{train,valid,test}.json` (+ `train1/`, `test1/`) | 40 453 / 2 371 / 1 791 |
| `upstream/data/SemDiff-Java/{train,valid,test}.json` (+ `train/`, `test/`) | 38 893 / 2 121 / 1 837 |

Confirmed by reading one record from each: both have the `pos_code` / `neg_code` / `text` / `pos` / `neg` / `Deviate` / `id` columns that
`critic/data_utils.DatasetCA` keys off.

(See `MODIFICATIONS.md` "SemDiff data extraction" entry for the layout-rename details.)

## 2. Model downloads — **SUCCESS**

Both fetched via `huggingface-cli` from the per-paper venv:

| Model                                  | Local path                                        | Size  | Notes                                                       |
|----------------------------------------|---------------------------------------------------|-------|-------------------------------------------------------------|
| `deepseek-ai/deepseek-coder-1.3b-base` | `~/hf-models/deepseek-coder-1.3b-base`            | 2.6 G | Evaluator backbone (paper default for `critic/run.py`).    |
| `bigcode/starcoder2-7b`                | `~/hf-models/starcoder2-7b`                       | 14 G  | Base-model fine-tune target for `train_base_model.py`.     |

Neither is gated; no HF auth was needed. The `qwen2.5-coder-7b-instruct/`
directory under `~/hf-models/` is from a different paper's run and is not used
here.

## 3. Patches applied beyond the original setup

Three additional patches inside `upstream/critic/` were needed to make
`critic/run.py` runnable. Full diffs are recorded in `MODIFICATIONS.md`
("evaluator-training patches" entry):

1. `critic/models.py`: replaced `local_dir = 'TODO'` (hardcoded placeholder)
   with `local_dir = ''` so `args.model_name_or_path` can be a real path.
2. `critic/run.py`: fixed `args.model_type = args.load` → `args.model_type =
   'deepseek'` (`MODEL_CLASSES` is keyed on `'deepseek'`, so the original would
   `KeyError`). Added `SEMGUARD_BACKBONE` env-var override for the backbone
   directory.
3. `critic/run.py`: replaced the broken `args.cache_path = ''` + `os.makedirs('')`
   stanza with a sensible default (`<output_dir>/cache`).

## 4. Detached training cell — **RUNNING**

The headline cell (DeepSeek-Coder 1.3B evaluator on SemDiff Python) is alive
under `nohup`.

| Field                | Value |
|----------------------|-------|
| Worker PID           | `185033` (python `run.py`) |
| Parent wrapper PID   | `184947` (bash) → `184949` (accelerate launcher) |
| Log file             | `/home/leobwang/code/courses/cmsc25750/project/reproduce/results/semguard/train_20260524_115641/train.log` |
| Launch script        | `/home/leobwang/code/courses/cmsc25750/project/reproduce/results/semguard/train_20260524_115641/launch.sh` |
| Output / checkpoints | `/home/leobwang/code/courses/cmsc25750/project/reproduce/results/semguard/train_20260524_115641/deepseek-coder-1.3b_python/` |
| Backbone             | `~/hf-models/deepseek-coder-1.3b-base` |
| Data                 | `upstream/data/SemDiff/` |
| Config               | `--load deepseek-coder-1.3b-base --batch_size 8 --grad_acc_steps 4 --lr 5e-6 --epochs 20 --max_source_len 600 --do_eval 1` |
| Distributed          | `accelerate launch --num_processes 1 --mixed_precision fp16 --dynamo_backend no` (overrides upstream's `num_processes: 6` in `acc_config.yaml`); deepspeed ZeRO-2 inside |
| GPU footprint        | 19 GB used of 96 GB (lots of slack — could grow `batch_size`) |
| Throughput at first epoch | ~5 it/s × 10 113 steps/epoch ≈ 33 min/epoch ⇒ ~11 hr for the full 20-epoch schedule |
| Loss trajectory      | step 4: 0.695 (≈ ln 2), step 228: 0.745 (oscillating early; expected for BCE on a fresh classifier head). Falls back below 0.69 around step 68. |

The job will continue under `nohup` after this agent exits. To check on it
later:

```bash
tail -F /home/leobwang/code/courses/cmsc25750/project/reproduce/results/semguard/train_20260524_115641/train.log
ps -p 185033 -o pid,etime,stat,cmd
```

Best-epoch model weights will land at
`.../deepseek-coder-1.3b_python/checkpoint-best/model.bin` (and per-epoch
checkpoints at `checkpoinss/<epoch>/model_<epoch>.bin` — sic, the upstream
folder name has the typo).

## 5. Smoke-test re-runs (regardless of training)

| Check | Outcome |
|-------|---------|
| `cd upstream && python -c "import train_base_model"` | **PASS** (no new patches needed beyond `SETUP_REPORT.md`'s three) |
| `cd upstream/critic && python -c "import run"` | **PASS** after patches 4–6 above |

The naive `cd upstream && python -c "from critic import run as critic_run"`
form fails because `critic/run.py` does `from models import build_or_load_gen_model`
(relative to its own dir). That's not a bug we should chase — the upstream
`run.sh` always invokes it with `cwd = critic/`.

## 6. What still needs to happen to reproduce the full paper grid

The paper reports:

> 4 LM backbones × 2 languages × {base-model fine-tune step + evaluator
> training step} = **16 expensive training runs**, plus a guided-generation
> sweep and an evaluation harness pass.

State after this session:

| Step                       | Backbones    | Langs        | Done | Remaining                       |
|----------------------------|--------------|--------------|------|---------------------------------|
| **Base FT** (`train_base_model.py`) | 4            | 2            | 0/8  | Need to download `deepseek-coder-6.7b-base`, `codellama-7b`, `qwen2.5-coder-7b-base`. StarCoder2-7B is downloaded. ~6-12 GPU-hr each on this GPU. |
| **Evaluator** (`critic/run.py`) | 4            | 2            | 1/8 in-flight | Then repeat with codet5bce / codebert / additional deepseek scale. Most are smaller than 1.3B so faster per cell. |
| **Generation** (`generate_sem.py`)  | per evaluator | per evaluator | 0/8  | Needs lazy `CodeT5ClassBCE` import in `transformers/generation/utils.py` re-pointed at `critic.models.CodeT5ClassBCE` (open todo from `MODIFICATIONS.md` patch 3). |
| **Eval harness** (`metric/test_one_solution.sh`) | – | – | 0/2 | Plumb in Evalplus + LiveCodeBench. Outside the paper repo. |

At the rate of one ~11-hour training cell in flight, the **evaluator** column
alone is ~3 GPU-days of wall-clock if run sequentially on this one device.
The **base FT** column is the larger expense (7B-class LoRA + 8-bit). Both
fit in 96 GB VRAM with margin to spare.

## 7. Time/resource accounting for this session

| Resource           | Spent                                   |
|--------------------|-----------------------------------------|
| Network egress     | ~17 GB (deepseek 2.6 GB + starcoder 14 GB) |
| Disk under `~/hf-models/` | 17 GB                            |
| Disk under `upstream/data/` | ~270 MB extracted from 11 MB compressed |
| Disk under `reproduce/results/semguard/` | Growing; cache + checkpoints |
| GPU             | 1 × RTX PRO 6000 Blackwell, 19/96 GB in use, will run for ~11 hr after this report ends |
| Sudo            | None used |
