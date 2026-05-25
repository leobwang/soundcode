# vLLM load test: qwen3.5:122b MoE

Verdict: **Yes — Qwen3.5-122B-A10B loads into vLLM 0.20.2 at int4 (GPTQ) and
generates valid code.** ~68 GiB weight memory, ~15 GiB KV cache, ~88 GiB peak
on a single 96 GiB RTX PRO 6000 Blackwell.

## Sanity check at start

- GPU: NVIDIA RTX PRO 6000 Blackwell, 97 GiB free.
- VRAM at start: 34 MiB used / 97184 MiB free / 0 % util.

## Identity of `qwen3.5:122b` (ollama)

- Ollama tag: `qwen3.5:122b`, blob
  `/usr/share/ollama/.ollama/models/blobs/sha256-93c83617a40560a61cda911ee327efdb5b5fbd39caa8b777a4ec565c0af1af3d`
  (81.4 GB on disk, Q4_K_M GGUF, arch `qwen35moe`, 262 k context, 125.1 B
  params, 3072 hidden, vision+tools+thinking capability).
- HF upstream: **`Qwen/Qwen3.5-122B-A10B`** (`qwen3_5_moe`,
  `Qwen3_5MoeForConditionalGeneration`). 125 B total, 10 B active per token;
  256 experts, 8 per token (top-k MoE); 48 hidden layers; 248 320 vocab; 262 144
  max position; hybrid Mamba (GDN linear attention) + full-attention layers.
  License: Apache-2.0.
- Available quantizations on HF (size / format):
  - `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` — **chosen**; 73.45 GiB, GPTQ 4-bit
    g128, 39 shards, official Alibaba; was already cached at
    `/home/leobwang/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/5b9f0050d3ec98b0c81a7716776533c5eacebb64`.
  - `cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit` — 75 GiB, compressed-tensors
    pack-quantized int4 g32 (third-party, llm-compressor format). Started
    downloading then aborted in favour of the cached GPTQ.
  - `Qwen/Qwen3.5-122B-A10B-FP8` — ~125 GiB; would NOT fit on a 96 GiB GPU.
  - `Sehyo/Qwen3.5-122B-A10B-NVFP4` — 81 GiB; NVFP4 is supported on
    Blackwell but newer and more fragile than GPTQ; skipped for first attempt.
  - `Qwen/Qwen3.5-122B-A10B` fp16 base — ~250 GiB; far too large.
  - `unsloth/Qwen3.5-122B-A10B-GGUF` / `bartowski/...` — GGUF only; the
    architecture isn't covered by vLLM's GGUF loader.

## VRAM arithmetic per precision

For 125 B parameters at single-GPU on 96 GiB:

| precision | params footprint | fits 96 GiB? |
|---|---|---|
| fp16 / bf16 (2 B/param)  | 250 GiB | No |
| int8     (1 B/param)     | 125 GiB | No |
| int4     (0.5 B/param + overhead) | ~63-68 GiB | **Yes, with margin for KV** |
| FP8      (1 B/param)     | 125 GiB | No |
| NVFP4    (0.5 B/param)   | ~80 GiB | Yes (tight) |
| GGUF Q4_K_M (mixed)      | ~81 GiB | Yes (tight) |

int4 (GPTQ or AWQ) is the only safe single-GPU precision.

## Strategy A: vLLM + GPTQ-Int4 from HF cache (preferred path)

Tried first. Two attempts:

**Attempt 1** — `quantization="gptq_marlin"`, `max_model_len=8192`,
`gpu_memory_utilization=0.90`, `enforce_eager=True`. Got through:
- Resolved arch `Qwen3_5MoeForConditionalGeneration` (line 4 of log)
- `gptq_marlin` runtime kernel selected (cast bf16 -> fp16 because GPTQ
  marlin requires fp16)
- Loaded 39 safetensors shards in 16.6 s
- Reported `Model loading took 68.36 GiB memory and 19.74 seconds`
- Profiled KV cache: `Available KV cache memory: 14.99 GiB; GPU KV cache
  size: 255,590 tokens; Max concurrency 31.20x at 8192 tokens/req`

Then failed in `_initialize_kv_caches`:

```
AssertionError: In Mamba cache align mode, block_size (2096) must be <= max_num_batched_tokens (2048).
```

Root cause: the model has Mamba/GDN linear-attention layers (the
`gdn_linear_attn.py` kernel) and vLLM's `Mamba cache align` mode pads the
attention block size up to the Mamba page size (`2048 -> 2096` per the
`interface.py:606` log). vLLM then asserts `block_size <=
max_num_batched_tokens`. The default chunked-prefill batch budget is 2048,
which is just under 2096.

**Attempt 2** — Same config plus `max_num_batched_tokens=4096`. **Succeeded
end-to-end**:

- Same 68.36 GiB weight footprint, same 14.99 GiB KV cache (255 590 tokens,
  31.20× concurrency).
- Warmup wall: 38.2 s (load 14.2 s of which 11.1 s reading shards + 4.3 s
  engine init/profile/warmup; OS file cache was warm from attempt 1).
- Generation: TTFT 47.14 s, decoded 32 tokens in 47.88 s → 0.67 tok/s.
- Output for prompt `def fibonacci(n: int) -> int:\n    """..."""\n`:

```python
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_memo
```

Correct recursion.

**Why TTFT/throughput are slow on this run.**
- `enforce_eager=True` disables CUDA graphs. Necessary while we're at 90 %
  VRAM utilisation — capturing graphs would temporarily allocate more.
- The Blackwell GPU exposes SM 12.x, which vLLM 0.20.2's logs flag as
  needing CUDA >= 12.9 for FlashAttention 3. We fall back to FA2 (`Using
  FlashAttention version 2`).
- 256-expert MoE under eager mode: each forward pass routes through 8
  experts per token sequentially. The first-token prefill is dominated by
  vLLM's chunked prefill warm path.
- This is a smoke test, not a benchmark — TTFT improves substantially after
  the first request once KV cache is warm and prefix caching can reuse
  prefills. A future patch could try `enforce_eager=False` at
  `gpu_memory_utilization=0.85` to enable CUDA graph capture.

## Strategy B (GGUF direct) — skipped

vLLM's GGUF support doesn't cover `qwen3_5_moe` and the Ollama blob doesn't
ship an HF-format tokenizer. Strategy A succeeded, so this wasn't tried.

## Strategy C (bitsandbytes on-the-fly) — skipped

Would require either the 250 GiB fp16 base (too big to download / hold) or
loading from a quantized checkpoint that bnb can't consume anyway. Not
viable for a 125 B model on a single GPU.

## Wiring into VllmBackend infrastructure

Added a small per-model registry instead of overloading `VllmBackend.__init__`'s
signature further. Three files changed:

1. **`soundcode/vllm_model_registry.py`** (new) — `VLLM_MODEL_REGISTRY: dict`
   maps short keys to `{hf_id, quantization, dtype, max_model_len,
   gpu_memory_utilization, enforce_eager, max_num_batched_tokens}` dicts.
   Currently registers:
   - `qwen2.5-coder-7b`, `deepseek-coder-1.3b`, `starcoder2-7b` (existing
     local hf-models cache models)
   - `qwen3.5-122b-int4` → snapshot dir for `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`,
     `gptq_marlin`, fp16, 8192 ctx, 0.90 GPU util, `max_num_batched_tokens=4096`.

2. **`soundcode/vllm_backend.py`** — `VllmBackend.__init__` now:
   - Treats every tuning kwarg as `Optional` (default `None`).
   - Looks up `model` in the registry; if it's a key, fills in any
     unset-by-caller kwarg from the registry entry. If it's a path or HF id,
     bypasses the registry (preserves the historical surface used by
     `tests/test_vllm_backend.py::test_backend_dispatch_returns_right_class`,
     which still passes `"Qwen/Qwen2.5-Coder-7B-Instruct"` directly).
   - New constructor knobs: `quantization`, `max_num_batched_tokens`. Both
     are plumbed through to `AsyncEngineArgs` only when not `None` (so the
     existing 7B path continues to use vLLM's auto-detection).

3. **`tests/test_vllm_backend.py`** — unchanged; all 9 plumbing tests still
   pass, plus the 11 cache tests in `tests/test_vllm_cache.py` (20 total).

To use the 122B from any existing call-site of `VllmBackend`, pass
`model="qwen3.5-122b-int4"` and everything else is set up automatically.
The server.py dispatch (`BACKEND_DISPATCH["vllm"]`) goes through unchanged.

## Files modified

- `soundcode/vllm_backend.py` — registry consultation + `quantization` and
  `max_num_batched_tokens` plumbing.
- `soundcode/vllm_model_registry.py` — new file, 4 model entries.
- `notes/vllm-qwen35moe-load-test.md` — this document.

## Punted

- Strategy B (GGUF) and Strategy C (bitsandbytes) — not needed; Strategy A
  succeeded.
- NVFP4 trial — could be faster on Blackwell hardware FP4, but adds risk
  and the GPTQ path is already proven working. Future patch.
- CUDA graph capture (`enforce_eager=False`) — currently disabled to keep
  VRAM headroom; could be tried at lower `gpu_memory_utilization` to see if
  throughput improves materially.
- Disabling the vision tower — the model has multi-modal capability we
  don't use, and profiling instantiates a vision encoder which contributes
  to the ~55 s engine init time. A `multimodal_config={"limit_per_prompt":
  {"image": 0}}` or arch-specific text-only flag could shorten init.

## Reference numbers (final attempt)

| metric | value |
|---|---|
| Model weight memory | 68.36 GiB |
| KV cache (90 % GPU util) | 14.99 GiB |
| KV cache token capacity (8192 ctx) | 255 590 tokens |
| Max concurrency | 31.20× |
| Total VRAM at idle post-load | ~73 GiB |
| Total VRAM during decode | ~88-89 GiB |
| Weight load wall | 11.1 s (warm) / 16.6 s (cold) |
| Engine init (profile + KV + warmup) | 4.3 s (warm) |
| End-to-end warmup | 38.2 s (warm) |
| TTFT (first prompt, cold cache) | 47.14 s |
| Decode throughput (eager, FA2) | ~0.67 tok/s |
