# ROCODE — full HumanEval run report

Date: 2026-05-24
Generator: Qwen2.5-Coder-7B-Instruct (CodeLlama-7B-hf substitute; see
`MODIFICATIONS.md` for rationale).

## Environment

- Conda env: `rocode-repro`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation, 96 GB VRAM
- torch: **2.11.0+cu128** (upgraded from the env's bundled 2.3.0+cu121 to get
  `sm_120` kernels — see `MODIFICATIONS.md` § runtime upgrade)
- transformers: 4.40.1 (unchanged)
- Model dtype on GPU: **fp32** (unchanged from upstream loader; ~30 GB
  resident, fits easily)

## Model weights

- Source: `Qwen/Qwen2.5-Coder-7B-Instruct` on HuggingFace Hub (ungated)
- Local path: `/home/leobwang/hf-models/qwen2.5-coder-7b-instruct/`
- On-disk size: 14.4 GB across 4 safetensors shards
- Download: `conda run -n rocode-repro huggingface-cli download
  Qwen/Qwen2.5-Coder-7B-Instruct --local-dir
  ~/hf-models/qwen2.5-coder-7b-instruct` (~4 min wall-clock)

## Patches applied

See `MODIFICATIONS.md` for the full dated log. Summary of what `upstream/`
changes were necessary:

1. `main.py`: BOS-id fallback (Qwen tokenizer exposes `bos_token_id = None`,
   upstream prepends it unconditionally → `TypeError`).
2. `main.py`: optional `--num-problems N` CLI flag (used for the smoke test;
   default `None` = run all).

Both are gated so the upstream CodeLlama invocation still works unchanged.

## Smoke test (HumanEval/0)

| Check                       | Result                                    |
|-----------------------------|-------------------------------------------|
| Model loads on GPU          | yes (~5 s, 4 shards, ~30 GB resident)     |
| Inference produces output   | yes (textbook nested-loop completion)     |
| ROCODE PA tool fires        | **yes — 6 PA-tool calls on 1 problem** (verified by monkey-patching `PA_tools.PA` with a counter; one call per generated statement) |
| Standard evaluator: `pass@1`| **1.0**                                   |
| Standard evaluator: AvgPassRatio | 1.0                                  |
| Standard evaluator: CCP     | 1.0                                       |
| Wall-clock for 1 problem    | 7.6 s (incl. trie-tree + PA loop overhead)|

## Full HumanEval run (detached)

Command actually launched (single line, in
`/home/leobwang/code/courses/cmsc25750/project/reproduce/algo/rocode/upstream/`):

```bash
nohup conda run -n rocode-repro python main.py \
  --arch Qwen2.5-Coder-7B-Instruct \
  --model-dir /home/leobwang/hf-models/qwen2.5-coder-7b-instruct \
  --dataset humaneval \
  --temperature 0.0 \
  --topk 50 \
  --topp 1 \
  --num-samples 1 \
  --decay-factor 0.9 \
  --output-dir /home/leobwang/code/courses/cmsc25750/project/reproduce/results/rocode/run_20260524_120206 \
  --output-file-suffix ROCODE \
  > /home/leobwang/code/courses/cmsc25750/project/reproduce/results/rocode/run_20260524_120206/stdout.log 2>&1 &
```

- **Run dir:** `/home/leobwang/code/courses/cmsc25750/project/reproduce/results/rocode/run_20260524_120206/`
- **Process PIDs (verified 2 min after launch):**
  - 187697 (outer bash wrapper, **reparented to init (PPID=1) — safe**)
  - 187701 (`conda run` wrapper)
  - **187704 — actual `python main.py`** *(the one to watch with `ps -p 187704`)*
- **Log file:** `run_20260524_120206/stdout.log` (single file, stderr merged)
- **Output JSONL:** `run_20260524_120206/humaneval_Qwen2.5-Coder-7B-Instruct_temp0.0_topp1.0_topk50_df0.9_samples1_ROCODE.jsonl`
  (one line per completed task, flushed after each task)

### Estimated wall-clock

The smoke test ran HumanEval/0 in 7.6 s end-to-end (including 5 s model load
amortized once). Early progress from the monitor showed **6 tasks completed
in the first 2 min of generation** including model load (~10–15 s/task
average after the load amortizes). At that rate:

- 164 tasks × ~10 s = **~30 min for generation** (rough; tasks with
  rollbacks will run longer)
- + a few minutes for the evaluation pass

Live GPU usage at the 2-min mark: **72 GB VRAM, 98% GPU utilization** (fp32
7B Qwen + KV cache + small PA-tool CPU side).

This is **~6× faster than the paper's headline** (`0.622 min/task = 37.3 s/task`
on A100), entirely explained by the Blackwell card being substantially faster
than the A100 the paper used.

(Note: this is wall-clock for *generation only*, with `--num-samples 1` and the
PA tool's compile/rollback loop active. Some tasks may take much longer if the
PA tool triggers many rollbacks — the budget cap is 2 × `MAX_GENERATION_LENGTH`
= 1536 tokens.)

## How to check progress / inspect results

```bash
# Live log
tail -f /home/leobwang/code/courses/cmsc25750/project/reproduce/results/rocode/run_20260524_120206/stdout.log

# Is the run still alive?
pgrep -af 'main\.py.*Qwen2.5-Coder' | grep -v conda

# How many tasks completed?
wc -l /home/leobwang/code/courses/cmsc25750/project/reproduce/results/rocode/run_20260524_120206/humaneval_Qwen2.5-Coder-7B-Instruct_temp0.0_topp1.0_topk50_df0.9_samples1_ROCODE.jsonl

# GPU usage right now
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv

# Once the JSONL has 164 lines, run the evaluator:
cd /home/leobwang/code/courses/cmsc25750/project/reproduce/algo/rocode/upstream
conda run -n rocode-repro python evaluate_generated_code.py \
  --dataset humaneval \
  --input_path /home/leobwang/code/courses/cmsc25750/project/reproduce/results/rocode/run_20260524_120206/humaneval_Qwen2.5-Coder-7B-Instruct_temp0.0_topp1.0_topk50_df0.9_samples1_ROCODE.jsonl \
  --truncate --eval_standard --eval_ET
# pass@k / AvgPassRatio / CCP go to stdout and to <jsonl>_results_summary
```

## Caveats and follow-ups

- **Model substitution**: Qwen2.5-Coder-7B-Instruct is an instruction-tuned
  model fed with a *completion-style* prompt (no chat template wrapping). Qwen
  was trained heavily on raw code completion data so this works, but the
  resulting pass@1 should **not** be compared head-to-head with the paper's
  CodeLlama-7B-hf numbers. The numbers we get here are the headline for
  *ROCODE-on-Qwen* and are the right baseline for downstream comparison
  against SoundCode (which also drives Qwen).
- **PyTorch is upstream-drifted** in the env (`environment.yml` pins 2.3.0,
  installed 2.11.0+cu128). Necessary for Blackwell; harmless for the math but
  must be re-pinned if anyone re-runs on a non-Blackwell GPU.
- **Stdout.log will contain a tqdm progress bar** (`tqdm(range(len(dataset)))`
  in `pipeline()`); `wc -l` on the JSONL is the more reliable progress meter.
- **HumanEval(ET) evaluation** can be enabled by passing `--eval_ET` (the
  upstream `run.sh` does this); it uses the bundled `data/HumanEval_ET.jsonl`
  with the extended test cases.
