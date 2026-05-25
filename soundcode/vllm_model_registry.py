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
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        # Qwen3.5-MoE has GDN linear-attention (Mamba-style) layers. vLLM
        # pads the attention block size up to the Mamba page size (e.g. 2096
        # for this config) and then asserts `block_size <=
        # max_num_batched_tokens`. The 0.20.2 default of 2048 fails that
        # check; bump to 4096 to absorb the alignment padding and leave
        # headroom for chunked-prefill batches.
        "max_num_batched_tokens": 4096,
    },
}


def resolve(model: str) -> dict[str, Any] | None:
    """Look up `model` in the registry. Returns the entry dict, or None if
    the key is not registered (caller treats the string as a raw path / HF id).
    """
    return VLLM_MODEL_REGISTRY.get(model)
