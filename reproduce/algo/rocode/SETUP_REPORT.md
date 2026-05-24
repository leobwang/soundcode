# ROCODE reproduction — setup report

Date: 2026-05-24
Pinned upstream commit: `49037ff6b98843d68138d3c6761f60c30f1381ce` (see `UPSTREAM_COMMIT.txt`)

## Tooling

| Tool         | Path                                       | Notes                              |
|--------------|--------------------------------------------|------------------------------------|
| `conda`      | `/home/leobwang/miniconda3/condabin/conda` | miniconda3, base env in PATH       |
| `nvidia-smi` | `/usr/bin/nvidia-smi`                      | Driver 590.48.01, CUDA 13.1        |

## GPU

- **Device:** 1× NVIDIA RTX PRO 6000 Blackwell
- **Total VRAM:** 97,887 MiB (~96 GB)
- **Free VRAM at setup:** 97,853 MiB (no processes running on GPU)
- **Power cap:** 600 W

This is substantially more VRAM than the paper's hardware required, so the
reproduction will not be VRAM-constrained for any 7B model the paper used.

## Conda environment

- **Name:** `rocode-repro`
- **Python:** 3.11.9
- **Status:** **success** — created from `upstream/environment.yml` without modification.

Key pinned versions (verified via `conda run -n rocode-repro python -c ...`):

| Package        | Version           |
|----------------|-------------------|
| `torch`        | 2.3.0+cu121       |
| `transformers` | 4.40.1            |
| `tokenizers`   | 0.19.1            |
| `accelerate`   | 0.29.3            |
| `peft`         | 0.10.0            |
| `datasets`     | 2.19.0            |
| `tree-sitter`  | 0.22.0            |
| `numpy`        | 1.26.4            |
| `pandas`       | 2.2.2             |
| CUDA wheels    | `nvidia-*-cu12`   |

Note: the env ships its own bundled CUDA 12.1 runtime via the `nvidia-*-cu12`
wheels. The driver (590.48 / CUDA 13.1) is forward-compatible, so PyTorch
reports `cuda_avail = True` and `device_count = 1`.

## Smoke test

Both checks pass with no patches needed:

```bash
$ conda run -n rocode-repro python -c "import torch; import transformers; \
    print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
2.3.0+cu121 4.40.1 True

$ cd upstream && conda run -n rocode-repro python -c "import main"
main imported OK
```

`main.py --help` also runs cleanly, listing all expected CLI flags.

## Default model the upstream code expects

From `upstream/run.sh`:

```bash
Dataset="humaneval"
Model="CodeLlama-7b-hf"
model_dir="/home/dongyh/LLMs/CodeLlama-7b-hf"   # paper author's local path
```

`upstream/main.py` defaults `--arch=CodeLlama-7b-hf` and resolves the path via
`--model-dir` (or falls back to `meta-llama/CodeLlama-7b-hf` / `Salesforce/...`
for codegen). The model is loaded in **fp32** (`torch_dtype=torch.float32`),
placed on `cuda:0`, with `low_cpu_mem_usage=True`.

## Estimated GPU memory & wall-clock for a full HumanEval(ET) run

**Memory (CodeLlama-7B in fp32):**
- 7B params × 4 B/param = ~28 GB weights
- KV cache + activations for ~768-token contexts at batch 1: a few GB more
- Total working set: **~30–34 GB** — comfortably fits in the 96 GB GPU. Could
  also fit in bf16 at ~14 GB if we patch the loader.

**Wall-clock:**
- Paper headline: **~0.622 min/task** on CodeLlama-7B over HumanEval (164 tasks)
  → 164 × 0.622 ≈ **102 min ≈ 1.7 h** on the paper's hardware (NVIDIA A100 40GB
  per their setup).
- The Blackwell RTX PRO 6000 is roughly 2–3× faster than an A100 40GB for
  fp32/bf16 transformer decode, so a single HumanEval(ET) run with one
  `--num-samples` should land at **~35–60 min wall-clock**, dominated by
  per-token decode latency rather than rollback overhead.
- MBPP (974 tasks) under the same protocol would be ~6× longer (~3–6 h here).

These are the *generation* phase only; the subsequent
`evaluate_generated_code.py` (sandboxed test execution) typically adds a few
minutes for HumanEval, considerably more for MBPP.

## Triggering a full inference run

The upstream entry point is `bash upstream/run.sh`, but that script hard-codes
the paper author's local model path. To trigger the full HumanEval(ET) run on
this machine, the user needs to:

1. **Decide the model source.** Either (a) download CodeLlama-7B from
   HuggingFace (`huggingface-cli download codellama/CodeLlama-7b-hf
   --local-dir ~/models/CodeLlama-7b-hf` — about 13 GB), or (b) point at an
   already-cached copy.
2. **Edit `upstream/run.sh`** to set `model_dir=<path>`, **or** invoke
   `main.py` directly:
   ```bash
   conda activate rocode-repro
   cd reproduce/algo/rocode/upstream
   python main.py \
     --arch CodeLlama-7b-hf \
     --model-dir <path-to-weights> \
     --dataset humaneval \
     --temperature 0.0 --topk 50 --topp 1 \
     --num-samples 1 --decay-factor 0.9 \
     --output-dir outputs/humaneval \
     --output-file-suffix ROCODE
   python evaluate_generated_code.py \
     --dataset humaneval \
     --input_path outputs/humaneval/humaneval_CodeLlama-7b-hf_temp0.0_topp1.0_topk50_df0.9_samples1_ROCODE.jsonl \
     --truncate --eval_standard --eval_ET
   ```
3. **Authorize the GPU burn.** Expect ~35–60 min of GPU time for HumanEval(ET)
   generation + a few minutes for evaluation, ~30 GB resident VRAM. No
   modifications to the upstream code are required to make the above invocation
   succeed; only the model weights and the `--model-dir` path.

## What is *not* covered by this setup

- Model weights are **not** downloaded (deferred to the full-run phase).
- No GPU compute has been exercised — only `cuda.is_available()` was probed.
- The planned JSONL logging hook and optional Ollama backend (see
  `MODIFICATIONS.md`) are **not** implemented yet.
- CodeForces2305 (`run_codeforces.sh`) was not exercised; it should work with
  the same env, but the data directory `data/CodeForces2305/` was not validated
  end-to-end here.
