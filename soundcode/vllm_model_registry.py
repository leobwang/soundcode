"""Registry of vLLM-loadable models, keyed by short human-readable name.

`VllmBackend.__init__` consults this registry: when `model=` is a key here,
the registry's settings (HF id, quantization, dtype, max_model_len, GPU mem
util) are used to fill in any unspecified constructor kwargs. When `model=`
is a plain path / HF id (i.e. not a registry key), the registry is bypassed
and the constructor's explicit kwargs win — preserving the historical surface
used by `tests/test_vllm_backend.py::test_backend_dispatch_returns_right_class`
which passes `"Qwen/Qwen2.5-Coder-7B-Instruct"` directly.

Each entry is a dict with the following recognised keys:

    "hf_id"                 — str, the model path or HF repo id vLLM loads
    "quantization"          — str | None, vLLM `quantization=` arg (e.g.
                              "compressed-tensors", "awq", "gptq", "fp8");
                              None means "let vLLM auto-detect from config"
    "dtype"                 — str, vLLM `dtype=` (default "bfloat16")
    "max_model_len"         — int, prompt+gen context cap
    "gpu_memory_utilization"— float in (0, 1], fraction of free VRAM vLLM
                              may claim. Cap aggressively when sharing the
                              GPU with other workloads.
    "enforce_eager"         — bool, disable CUDA graph capture (saves VRAM,
                              costs throughput). Default True (matches
                              VllmBackend's existing default).

Add a new model by appending a new entry; nothing else needs to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


VLLM_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # ─── small coder models (local hf-models cache) ────────────────────
    "qwen2.5-coder-7b": {
        "hf_id": str(Path.home() / "hf-models" / "qwen2.5-coder-7b-instruct"),
        "quantization": None,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.30,
        "enforce_eager": True,
    },
    "deepseek-coder-1.3b": {
        "hf_id": str(Path.home() / "hf-models" / "deepseek-coder-1.3b-base"),
        "quantization": None,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.15,
        "enforce_eager": True,
    },
    "starcoder2-7b": {
        "hf_id": str(Path.home() / "hf-models" / "starcoder2-7b"),
        "quantization": None,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.30,
        "enforce_eager": True,
    },
    # ─── Qwen3.5-122B-A10B MoE (GPTQ Int4, official Alibaba release) ────
    # Source: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 on HF.
    # Architecture: Qwen3_5MoeForConditionalGeneration (qwen3_5_moe). 125B
    # total params, ~10B active per token; 256 experts, 8 per token;
    # vocab 248k; native context 262k.
    # Quantization: GPTQ Int4, group_size=128. ~79 GB on disk; loads at
    # ~80-85 GB VRAM with margin for KV cache + activations.
    # max_model_len capped at 8192 — the full 262k context would blow the
    # KV-cache budget on a single 96 GB GPU.
    "qwen3.5-122b-int4": {
        # HF cache snapshot path — populated by an earlier snapshot_download.
        # vLLM accepts a snapshot directory as `model=` (it reads
        # config.json + model.safetensors.index.json from there).
        "hf_id": "/home/leobwang/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/5b9f0050d3ec98b0c81a7716776533c5eacebb64",
        "quantization": "gptq_marlin",
        "dtype": "float16",
        # Throughput-tuned config (see notes/vllm-122b-throughput.md):
        # ~96-98 tok/s decode @ batch=1 on RTX PRO 6000 Blackwell, vs
        # ~40 tok/s in the prior eager+0.90 config. Knobs:
        #   - `enforce_eager=False` lets vLLM capture CUDA graphs for the
        #     decode path (the dominant lever).
        #   - `max_model_len=2048` keeps the KV-cache budget small enough
        #     that the marlin MoE GEMM workspace + graph capture fits in
        #     VRAM; full 8192 ctx OOMs during graph capture.
        #   - `gpu_memory_utilization=0.78` leaves ~20 GiB headroom for the
        #     graph capture workspace and any concurrent GPU workload (the
        #     paused SemGuard training).
        #   - `compilation_config={"cudagraph_capture_sizes": [1,2,4],
        #     "cudagraph_mode": 2}` (FULL_DECODE_ONLY) restricts captured
        #     batch shapes to the decode side; vLLM's default PIECEWISE
        #     captures up to ~67 shapes whose combined workspace OOMs.
        #   - `max_num_batched_tokens=4096` absorbs the Mamba-block-size
        #     alignment padding (~2096 from the GDN linear-attention layer).
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.78,
        "enforce_eager": False,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 4,
        "compilation_config": {
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_mode": 2,  # CUDAGraphMode.FULL_DECODE_ONLY
        },
    },
    # ─── Qwen3.5-122B-A10B NVFP4 (RedHatAI repack, Blackwell-tuned) ────
    # Same architecture, NVFP4 (4-bit float) instead of GPTQ-Int4.
    # Throughput is comparable to the GPTQ-Int4 entry (89-110 tok/s on a
    # Blackwell RTX PRO 6000), but with two hard requirements:
    #   1. dtype MUST be bfloat16 — vLLM's SM 120 cutlass NVFP4 MoE
    #      explicitly rejects fp16 output ("SM120 NVFP4 MOE only supports
    #      bfloat16 output").
    #   2. The default flashinfer NVFP4 backend tries to JIT-compile cubins
    #      for `compute_120f`, which requires CUDA >= 12.9 nvcc. The system
    #      nvcc on the Blackwell host is 12.0, so flashinfer's NVFP4 path
    #      fails. Use vLLM's native CUTLASS NVFP4 kernels instead by setting
    #      these env vars *before* engine construction:
    #          VLLM_NVFP4_GEMM_BACKEND=cutlass
    #          VLLM_USE_FLASHINFER_MOE_FP4=0
    #      (We don't set them in the registry — callers do it.)
    # VRAM is ~82 GiB (vs ~75 GiB for GPTQ-Int4) because NVFP4 weights
    # ship with FP8 scale tensors, not 16-bit. Use the GPTQ entry by
    # default; switch here only if the SM 12.0 NVFP4 kernel gap closes
    # upstream or if you specifically need NVFP4-quality scales.
    "qwen3.5-122b-nvfp4": {
        "hf_id": "/home/leobwang/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4/snapshots/49d19c108259a21450c40b8af38828b0a97390d8",
        "quantization": "compressed-tensors",
        "dtype": "bfloat16",  # see (1) above — NOT fp16 on SM 120
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": False,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 4,
        "compilation_config": {
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_mode": 2,  # CUDAGraphMode.FULL_DECODE_ONLY
        },
    },
    # ─── Qwen3.6-35B-A3B MoE (GPTQ Int4, palmfuture community quant) ────
    # Source: palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 on HF (no official Alibaba
    # GPTQ-Int4 exists for 3.6; the FP8 and NVFP4 variants are the only
    # first-party packs, and a 35B fp8 is ~35 GB so int4 is preferred for
    # head-room). Architecture: `Qwen3_5MoeForConditionalGeneration`
    # (`qwen3_5_moe`) — the *same* family as the 122B and same vLLM model
    # code path. 36B total params, ~3B active per token; the Ollama tag
    # reports it as Q4_K_M GGUF (23 GB on disk; this GPTQ pack is ~22 GB).
    # 262 144 native context; native vision+tools+thinking; bf16 base.
    # Quantization: GPTQ Int4, group_size=128, sym=True. Attention,
    # shared experts, MoE gate, MTP head, and vision encoder are excluded
    # (kept at higher precision) — the `dynamic` block in `quantize_config.json`
    # is from gptqmodel 6.0.3.
    # max_model_len=2048 mirrors the 122B winner; KV cache budget is huge
    # at 0.78 util on a 96 GiB GPU (54-57 GB free, ~3-7x what we'd need).
    # Headline throughput: 155-158 tok/s decode on the spec prompt
    # `def fibonacci(n):` with `max_tokens=200`, T=0, ignore_eos — ~60% above
    # the 122B winner thanks to ~3x smaller active-params footprint per
    # token. See notes/vllm-35b-throughput.md for the per-strategy sweep.
    "qwen3.6-35b-int4": {
        "hf_id": "/home/leobwang/.cache/huggingface/hub/models--palmfuture--Qwen3.6-35B-A3B-GPTQ-Int4/snapshots/d1fef185160f938fca00c3c664f21250dd544d63",
        "quantization": "gptq_marlin",
        "dtype": "float16",
        # Throughput-tuned config (see notes/vllm-35b-throughput.md):
        # ~156 tok/s decode @ batch=1 on RTX PRO 6000 Blackwell. Knobs are
        # carried over verbatim from the 122B winner because the architecture,
        # the Mamba page-size constraint, and the SM 12.x kernel gaps are
        # all identical between the 35B and the 122B (same `qwen3_5_moe`
        # arch). What changes is just the magnitude: 35B has so much
        # head-room (peak VRAM 75 GiB out of 96 GiB) that we could loosen
        # gpu_memory_utilization, max_model_len, and max_num_seqs, but
        # measured throughput is the *same* (152-158 tok/s) — the model is
        # already memory-bandwidth-bound at batch=1, so more KV slots and
        # more graph captures don't help further. Keep the conservative
        # 122B-mirror knobs for predictable VRAM (room to share the GPU
        # with SemGuard training).
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.78,
        "enforce_eager": False,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 4,
        "compilation_config": {
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_mode": 2,  # CUDAGraphMode.FULL_DECODE_ONLY
        },
    },
}


def resolve(model: str) -> dict[str, Any] | None:
    """Look up `model` in the registry. Returns the entry dict, or None if
    the key is not registered (caller treats the string as a raw path / HF id).
    """
    return VLLM_MODEL_REGISTRY.get(model)
