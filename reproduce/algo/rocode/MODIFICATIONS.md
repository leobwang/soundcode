# ROCODE — local modifications

## Convention

When a modification is made, add a dated entry below describing the change and
the file(s) touched.

## Planned

- **JSONL logging hook.** Emit one `run_<YYYYMMDD>_<HHMMSS>_<hash>.jsonl` file
  per benchmark run into `reproduce/results/`, matching the schema used by the
  SoundCode web demo under `<project_root>/results/web_demo/`. Likely touches
  `main.py`, `main_codeforces.py`, and `evaluate_generated_code.py`.
- **Optional Ollama backend adapter.** Wrap the HuggingFace generation call so
  that ROCODE can drive the same Ollama-hosted models we use in SoundCode, for
  apples-to-apples wall-clock comparison. Only added if needed; would touch
  `main.py` and `config.py`.

## Log

### 2026-05-24 — env setup, no upstream patches

Created conda env `rocode-repro` directly from `upstream/environment.yml`; no
version-pin conflicts, no source modifications. Verified `import torch`,
`import transformers`, and `import main` (from `upstream/`) all succeed. See
`SETUP_REPORT.md` for details. No files under `upstream/` were touched.

### 2026-05-24 — model substitution: Qwen2.5-Coder-7B-Instruct ↔ CodeLlama-7B-hf

**Why.** Paper uses `codellama/CodeLlama-7b-hf`, which is **gated on HuggingFace**
and the user has no HF auth. Qwen2.5-Coder-7B-Instruct was chosen as the
substitute: it is the user's primary generator family in SoundCode (per project
memory), so this is a strictly better baseline for downstream
ROCODE-vs-SoundCode comparison.

**Weights.** Downloaded once to `~/hf-models/qwen2.5-coder-7b-instruct/`
(14.4 GB on disk; 4 safetensors shards). Backed by the HF cache under
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/`.

**Prompt format.** Option (b) from the task spec — *raw completion-style
prompt*, no chat template. Rationale: ROCODE's `pipeline()` is built on
statement-level decoding (`generate_code_stat`) that hand-rolls the token loop
on top of `model.generate(max_length=N+1)`. Wrapping the prompt in Qwen's
`<|im_start|>user…<|im_end|><|im_start|>assistant\n` markers would break the
PA tool's lineno bookkeeping (the PA tool parses the buffered code as a
top-level Python module). Since Qwen2.5-Coder is heavily trained on raw FIM
and continuation data, it still produces clean code completions from the
unwrapped HumanEval prompt — verified end-to-end below.

**Smoke evidence.** On HumanEval/0 (single problem, 1 sample, temperature 0):
- Model loads (4 shards, ~5 s).
- Pipeline issues 6 PA-tool calls (one per generated statement); all returned
  `"Code Test Passed."`.
- Final completion is the textbook nested-loop solution; standard evaluator
  reports `pass@1 = 1.0`, `AvgPassRatio = 1.0`, `CCP = 1.0`.

**Patches applied to `upstream/main.py`.** Two minimal edits, both gated so
they're no-ops for CodeLlama (so the upstream `run.sh` invocation still works
unchanged):

1. *BOS fallback for tokenizers that omit `bos_token_id`.*
   `tokenizer.bos_token_id` is `None` for the Qwen2.5 tokenizer (even though
   `config.json` lists `bos_token_id: 151643`). Upstream prepends
   `tokenizer.bos_token_id` to `input_ids` unconditionally, which crashes with
   `TypeError: unsupported operand type(s) for +: 'NoneType' and 'list'`.
   Patch: if `bos_token_id is None`, fall back to `tokenizer.pad_token_id`
   (which equals `151643 = <|endoftext|>`, i.e. the same id `config.json`
   advertises).
   File: `upstream/main.py` (`generate_code_stat`, lines ~40-43).

2. *`--num-problems N` flag for smoke tests.*
   Upstream `main.py` iterates the entire dataset; there is no built-in way to
   limit to the first N tasks. Added a CLI flag that slices the HF Dataset to
   the first N rows after `load_dataset_my()`. Default `None` (=run all
   problems), so behavior is unchanged for any pre-existing invocation.
   File: `upstream/main.py` (`parse_args`, `main`).

### 2026-05-24 — runtime upgrade: torch 2.3.0+cu121 → 2.11.0+cu128

**Why.** The conda env shipped torch 2.3.0+cu121, which only compiles kernels
for `sm_50…sm_90`. The target GPU is an RTX PRO 6000 Blackwell (compute
capability 12.0 / `sm_120`), so the first model.generate() call died with
`CUDA error: no kernel image is available for execution on the device`.

**Fix.** Upgraded torch in the `rocode-repro` env via
`pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch`.
The cu128 wheel set ships `sm_75…sm_120`, and verifies clean (matmul + model
forward both succeed on the Blackwell card). Other env packages
(`transformers==4.40.1`, `tokenizers==0.19.1`, `tree-sitter==0.22.0`, etc.) are
unchanged; transformers 4.40 is compatible with torch 2.11.

**Caveat.** This drifts the env away from the upstream `environment.yml`. If
the reproduction is later attempted on a non-Blackwell GPU (A100, H100,
RTX 4090, etc.), the original torch 2.3.0+cu121 should work as-is and the
upgrade is unnecessary. The patched env is preserved in `rocode-repro`; no
other env is touched.
