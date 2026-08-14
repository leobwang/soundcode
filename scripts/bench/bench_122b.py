"""Single-strategy throughput probe for Qwen3.5-122B-A10B.

Loads vLLM with a given config, runs `def fibonacci(n):` prompt with
max_tokens=200 / temp=0.0, prints warmup / TTFT / decode tok/sec, then
exits. Run with `uv run python scripts/bench/bench_122b.py --strategy A`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any

# Suppress noisy logging from vLLM during probe.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


GPTQ_INT4 = (
    "/home/leobwang/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4/"
    "snapshots/5b9f0050d3ec98b0c81a7716776533c5eacebb64"
)
NVFP4 = (
    "/home/leobwang/.cache/huggingface/hub/"
    "models--RedHatAI--Qwen3.5-122B-A10B-NVFP4/"
    "snapshots/49d19c108259a21450c40b8af38828b0a97390d8"
)


PROMPT = "def fibonacci(n):"
MAX_TOKENS = 200
TEMPERATURE = 0.0
# Prompt that won't naturally stop after a few dozen tokens (no stop reached).
PROMPT_LONG = (
    "Write a self-contained Python program that implements a high-performance "
    "in-memory key-value store with TTL eviction, an LRU policy, and a "
    "lightweight HTTP API exposing GET/PUT/DELETE. Include error handling, "
    "docstrings, and a small CLI entry point. Begin the implementation now:\n\n"
    "import time\nimport heapq\nimport threading\n\nclass KVStore:\n    "
)


def gpu_mem_used_mib() -> int:
    """Read GPU memory used (MiB) from nvidia-smi."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return int(out.split("\n")[0])
    except Exception:
        return -1


def run_one(name: str, kwargs: dict[str, Any], use_short_prompt: bool = False) -> dict[str, Any]:
    """Run one probe. Returns dict of metrics."""
    print(f"\n=== strategy {name} ===", flush=True)
    print(f"config: {json.dumps({k: v for k, v in kwargs.items() if k != 'model'}, indent=2)}",
          flush=True)
    print(f"VRAM before load: {gpu_mem_used_mib()} MiB", flush=True)
    prompt_used = PROMPT if use_short_prompt else PROMPT_LONG

    from vllm import LLM, SamplingParams

    t_load_start = time.perf_counter()
    try:
        llm = LLM(**kwargs)
    except Exception as e:
        print(f"LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        return {
            "strategy": name,
            "status": "load_failed",
            "error": f"{type(e).__name__}: {str(e)[:500]}",
        }
    t_load_end = time.perf_counter()
    warmup_s = t_load_end - t_load_start
    vram_after_load = gpu_mem_used_mib()
    print(f"warmup: {warmup_s:.2f}s, VRAM after load: {vram_after_load} MiB",
          flush=True)

    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=1.0,
                        ignore_eos=True)

    # First do a true 1-token warmup to separate prefill/TTFT from steady-state.
    sp_one = SamplingParams(temperature=TEMPERATURE, max_tokens=1, top_p=1.0,
                            ignore_eos=True)
    try:
        t_ttft_start = time.perf_counter()
        _ = llm.generate([prompt_used], sp_one, use_tqdm=False)
        t_ttft_end = time.perf_counter()
        ttft = t_ttft_end - t_ttft_start
    except Exception:
        ttft = None

    # Warmup generation (single token) to get past kernel-tune.
    try:
        t_gen_start = time.perf_counter()
        outputs = llm.generate([prompt_used], sp, use_tqdm=False)
        t_gen_end = time.perf_counter()
    except Exception as e:
        print(f"GENERATE FAILED: {type(e).__name__}: {e}", flush=True)
        # Clean up.
        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {
            "strategy": name,
            "status": "generate_failed",
            "warmup_s": warmup_s,
            "vram_after_load_mib": vram_after_load,
            "error": f"{type(e).__name__}: {str(e)[:500]}",
        }
    gen_s = t_gen_end - t_gen_start
    out = outputs[0]
    gen_text = out.outputs[0].text
    gen_tokens = len(out.outputs[0].token_ids)
    # Use the explicit single-token call as our TTFT measurement (above).
    # Decode tok/sec = (n - 1) / (gen_s - ttft), assuming first-token cost
    # in `gen_s` is roughly the same as the single-token run's ttft.
    if ttft is not None and gen_s > ttft and gen_tokens > 1:
        decode_tps = (gen_tokens - 1) / (gen_s - ttft)
    else:
        decode_tps = gen_tokens / gen_s if gen_s > 0 else 0.0
    total_tps = gen_tokens / gen_s if gen_s > 0 else 0.0

    print(f"gen: {gen_tokens} tok in {gen_s:.2f}s, "
          f"TTFT={ttft if ttft is None else f'{ttft:.2f}s'}, "
          f"decode={decode_tps:.2f} tok/s, total={total_tps:.2f} tok/s",
          flush=True)
    print(f"text[:80]: {gen_text[:80]!r}", flush=True)
    vram_peak = gpu_mem_used_mib()
    print(f"VRAM during gen: {vram_peak} MiB", flush=True)

    # Clean up.
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "strategy": name,
        "status": "ok",
        "warmup_s": warmup_s,
        "ttft_s": ttft,
        "gen_s": gen_s,
        "gen_tokens": gen_tokens,
        "decode_tps": decode_tps,
        "total_tps": total_tps,
        "vram_after_load_mib": vram_after_load,
        "vram_peak_mib": vram_peak,
        "text_preview": gen_text[:120],
    }


STRATEGIES: dict[str, dict[str, Any]] = {
    # A: enforce_eager=False on GPTQ-int4
    "A": dict(
        model=GPTQ_INT4,
        quantization="gptq_marlin",
        dtype="float16",
        gpu_memory_utilization=0.82,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        enforce_eager=False,
    ),
    # A-fallback: more conservative if A OOMs at graph capture
    "A_fallback": dict(
        model=GPTQ_INT4,
        quantization="gptq_marlin",
        dtype="float16",
        gpu_memory_utilization=0.78,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enforce_eager=False,
    ),
    # A-tight: tight margins with expandable_segments env (set externally)
    "A_tight": dict(
        model=GPTQ_INT4,
        quantization="gptq_marlin",
        dtype="float16",
        gpu_memory_utilization=0.72,
        max_model_len=2048,
        max_num_batched_tokens=2096,
        enforce_eager=False,
        max_num_seqs=4,
    ),
    # A-min-graphs: minimal CUDA graph capture (just batch 1 + 8)
    "A_min_graphs": dict(
        model=GPTQ_INT4,
        quantization="gptq_marlin",
        dtype="float16",
        gpu_memory_utilization=0.78,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enforce_eager=False,
        max_num_seqs=4,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_mode": 2,  # FULL_DECODE_ONLY
        },
    ),
    # A with C stacking: minimal graphs + swap_space + log_stats off
    "A_stacked": dict(
        model=GPTQ_INT4,
        quantization="gptq_marlin",
        dtype="float16",
        gpu_memory_utilization=0.78,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enforce_eager=False,
        max_num_seqs=4,
        swap_space=4,
        disable_log_stats=True,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_mode": 2,
        },
    ),
    # A with more graph sizes (decode batches 1-8)
    "A_more_graphs": dict(
        model=GPTQ_INT4,
        quantization="gptq_marlin",
        dtype="float16",
        gpu_memory_utilization=0.78,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enforce_eager=False,
        max_num_seqs=4,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4, 8],
            "cudagraph_mode": 2,
        },
    ),
    # Baseline reproduction for sanity
    "baseline": dict(
        model=GPTQ_INT4,
        quantization="gptq_marlin",
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        max_num_batched_tokens=4096,
        enforce_eager=True,
    ),
    # B: NVFP4
    "B": dict(
        model=NVFP4,
        # nvfp4-pack-quantized is the compressed-tensors format -> use it.
        quantization="compressed-tensors",
        dtype="float16",
        gpu_memory_utilization=0.82,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        enforce_eager=False,
    ),
    "B_eager": dict(
        model=NVFP4,
        quantization="compressed-tensors",
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        enforce_eager=True,
    ),
    # B with cutlass + marlin (skipping flashinfer SM12.x compile)
    "B_cutlass_marlin": dict(
        model=NVFP4,
        quantization="compressed-tensors",
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        enforce_eager=True,
    ),
    "B_cutlass_graphs": dict(
        model=NVFP4,
        quantization="compressed-tensors",
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enforce_eager=False,
        max_num_seqs=4,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_mode": 2,  # FULL_DECODE_ONLY
        },
    ),
    "B_emulation": dict(
        model=NVFP4,
        quantization="compressed-tensors",
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        enforce_eager=True,
    ),
    # C: Stack tuning knobs on top of winning A config
    "C_on_A": dict(
        model=GPTQ_INT4,
        quantization="gptq_marlin",
        dtype="float16",
        gpu_memory_utilization=0.82,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        enforce_eager=False,
        max_num_seqs=8,
        swap_space=4,
        disable_log_stats=True,
    ),
    # C on B
    "C_on_B": dict(
        model=NVFP4,
        quantization="compressed-tensors",
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enforce_eager=False,
        max_num_seqs=4,
        swap_space=4,
        disable_log_stats=True,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_mode": 2,
        },
    ),
    # B with more graph sizes (capture extra batches)
    "B_more_graphs": dict(
        model=NVFP4,
        quantization="compressed-tensors",
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enforce_eager=False,
        max_num_seqs=4,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
            "cudagraph_mode": 2,
        },
    ),
    # B with FULL_AND_PIECEWISE (graphs for everything)
    "B_full_graphs": dict(
        model=NVFP4,
        quantization="compressed-tensors",
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enforce_eager=False,
        max_num_seqs=4,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_mode": 4,  # FULL_AND_PIECEWISE
        },
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    parser.add_argument("--json-out", default=None,
                        help="Write JSON metrics to this path")
    parser.add_argument("--short-prompt", action="store_true",
                        help="Use 'def fibonacci(n):' instead of the long prompt")
    args = parser.parse_args()

    cfg = STRATEGIES[args.strategy]
    result = run_one(args.strategy, cfg, use_short_prompt=args.short_prompt)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nresult JSON: {args.json_out}", flush=True)
    print(f"\nFINAL: {json.dumps(result, indent=2)}", flush=True)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
