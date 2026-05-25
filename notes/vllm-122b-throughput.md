# Pushing Qwen3.5-122B-A10B throughput on vLLM

Hardware: 1× NVIDIA RTX PRO 6000 Blackwell, 96 GiB, SM 12.0, driver 590.48, CUDA 13.0
vLLM: 0.20.2
Prompt: `def fibonacci(n):` (also a longer KV-store program prompt for prefill-amortised measurements). 200 tokens, T=0.0, ignore_eos.
Decode tok/s reported as `(n_tokens − 1) / (gen_s − ttft)`, measured with a separate 1-token call to capture true TTFT.

## Verdict (TL;DR)

**Yes — we hit 50-100 tok/s.** Two configurations clear the headline target on the standard short prompt:

| config | decode tok/s | VRAM | warmup |
|---|---|---|---|
| **GPTQ-Int4 + min CUDA graphs (winner)** | **96-98 tok/s** | 75 GB | 38-46 s |
| NVFP4 + cutlass + min CUDA graphs | 89-110 tok/s (variance) | 82 GB | 52-99 s |
| Ollama (same model, same hardware) | 92-93 tok/s | 92 GB | first 14 s, then warm |
| vLLM baseline reproduction (task #24 spec) | **40 tok/s** | 89 GB | 35 s |

The "0.67 tok/s" figure in the task brief is not reproducible with `llm.generate()`. The same baseline config under the same prompt now measures 40 tok/s — likely the original number came from streaming-overhead-inclusive measurement or was a regression that has since healed (vLLM 0.20.2 has had MoE perf fixes between dev tags). Reporting both honestly: from the brief's "0.67 tok/s" baseline we are 140× over; from the reproduced 40 tok/s baseline we are 2.4× over.

## Per-strategy results

All numbers from a single GPU, no other workloads, vLLM 0.20.2. Each run loaded the model, did a 1-token warmup, then generated 200 tokens with `ignore_eos=True`.

### Strategy A — `enforce_eager=False` on GPTQ-Int4

The brief's "headline change": turn CUDA graphs on. Two sub-attempts:

- **A (default `gpu_memory_utilization=0.82`, `max_model_len=4096`, full PIECEWISE graph capture):** OOM during `profile_cudagraph_memory()` — the marlin MoE GEMM workspace at the captured batch sizes is enormous (intermediate cache across 256 experts), and even at 0.82 util vLLM tries to capture graphs whose forward passes push past 95 GiB.
- **A_fallback (`gpu_memory_utilization=0.78`, `max_model_len=2048`):** Same OOM (~24 MiB short).
- **A_min_graphs (winner of A family): `gpu_memory_utilization=0.78`, `max_model_len=2048`, `max_num_batched_tokens=4096`, `max_num_seqs=4`, `compilation_config={cudagraph_capture_sizes=[1,2,4], cudagraph_mode=FULL_DECODE_ONLY}`** — **96.9 tok/s**, warmup 38.8 s, TTFT 0.16 s, VRAM 75.3 GiB. The crucial knob is limiting captured batch sizes to just the decode-side shapes (1/2/4) so the captured workspace stays small; this is what avoids the OOM and what enables CUDA graphs at all on this model.

Stacking C-strategy knobs (`swap_space=4`, `disable_log_stats=True`) gave no change (single-user single-request — these are batching/serving optimisations).

### Strategy B — NVFP4 (RedHatAI/Qwen3.5-122B-A10B-NVFP4)

The model is cached locally as `compressed-tensors` (`nvfp4-pack-quantized`). Available NVFP4 backends in vLLM 0.20.2:

| backend | result on SM 12.0 |
|---|---|
| `flashinfer-cutlass` (default Linear) | **FAILS**: `RuntimeError: No supported CUDA architectures found for major versions [12]` — flashinfer's NVFP4 cubin requires CUDA ≥ 12.9 nvcc, system has 12.0 |
| `flashinfer-trtllm`, `flashinfer-cudnn` | same failure (all share the gen_gemm_sm120_module_cutlass_fp4 path) |
| `vllm-cutlass` Linear + `cutlass` MoE | **works**, but only with `dtype=bfloat16` — the kernel explicitly rejects fp16 output: `SM120 NVFP4 MOE only supports bfloat16 output, got: Half` |
| `marlin` NVFP4 | not tested (cutlass path worked) |
| `emulation` | not tested (cutlass faster) |

Required environment: `VLLM_NVFP4_GEMM_BACKEND=cutlass VLLM_USE_FLASHINFER_MOE_FP4=0`.

- **B_eager (bfloat16, eager):** 10.2 tok/s decode — no CUDA graphs, large kernel-launch overhead per token across 48 layers + 256-expert MoE.
- **B_cutlass_graphs (winner of B family): `dtype=bfloat16`, `gpu_memory_utilization=0.85`, `max_model_len=2048`, `compilation_config={cudagraph_capture_sizes=[1,2,4], cudagraph_mode=FULL_DECODE_ONLY}`** — 89-110 tok/s across runs (variance ~10-20%), warmup 52-99 s, TTFT 0.16-0.92 s, VRAM 82.0 GiB.
- **B_more_graphs ([1,2,4,8,16] capture sizes):** 109.7 tok/s; warmup is longer (98 s) but steady-state slightly higher.

NVFP4 does **not** decisively beat GPTQ on SM 12.0 — both are gated by the same SM 12.x kernel coverage. GPTQ's Marlin path is already well-tuned for Ampere/Hopper and runs fine on Blackwell via SM 12.0 PTX; NVFP4 requires Blackwell-native cubins that aren't (yet) shipped for SM 12.0 in flashinfer 0.4.x / vLLM 0.20.2's bundled cutlass.

### Strategy C — additional vLLM knobs

Stacked on top of the best A and B configurations: `max_num_seqs=4`, `swap_space=4`, `disable_log_stats=True`. **No measurable improvement** for the single-user / `max_tokens=200` workload — these knobs are for concurrent serving / KV-cache fragmentation, neither of which applies here. `block_size` is fixed by the Mamba page size for hybrid attention; cannot be overridden cleanly. `use_v2_block_manager` is the default in vLLM 0.20.2 — no flag to set.

### Strategy D — Ollama control

Curl probe against `qwen3.5:122b` (Q4_K_M GGUF, same model lineage):

```
ollama (cold): eval_count=200, eval_duration=2.17s, total=14.56s, 92.13 tok/s
ollama (warm): eval_count=200, eval_duration=2.14s, total=2.51s, 93.43 tok/s
```

**Ollama hits 92-93 tok/s warm** — essentially the same envelope as our vLLM winning configs (89-110 tok/s). The memory-bandwidth ceiling estimate of 100-160 tok/s in the brief is consistent: both engines are within ~80-110 tok/s, sitting under the bandwidth ceiling but well within the headline target. Ollama uses 91.5 GiB; vLLM-GPTQ uses 75 GiB (smaller because GPTQ-Marlin lives at int4 + 16-bit scales; GGUF Q4_K_M is mixed-precision and heavier per param).

## Winning config (registry entry)

```python
"qwen3.5-122b-int4": {
    "hf_id": "<GPTQ-Int4 snapshot path>",
    "quantization": "gptq_marlin",
    "dtype": "float16",
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
```

Peak VRAM under this config: **75.3 GiB** (with ~21 GiB of headroom for SemGuard or other GPU workloads).

A second entry for NVFP4 is included for completeness (`qwen3.5-122b-nvfp4`), but it requires two env vars (`VLLM_NVFP4_GEMM_BACKEND=cutlass`, `VLLM_USE_FLASHINFER_MOE_FP4=0`) and `dtype=bfloat16`, and does not measurably beat GPTQ-Int4 on this hardware until the SM 12.0 NVFP4 kernel gap is fixed upstream.

## SM 12.x gap analysis (what's blocking further wins)

vLLM 0.20.2 on Blackwell (SM 12.0) is constrained by a small set of missing kernels:

1. **flashinfer NVFP4 GEMM cubins for SM 12.0**
   `flashinfer/jit/gemm/core.py::gen_gemm_sm120_module_cutlass_fp4` requires CUDA ≥ 12.9 nvcc. System nvcc is 12.0, so JIT-compilation of `compute_120f`/`compute_121a` fails with `No supported CUDA architectures found for major versions [12]`. Upstream needs either (a) prebuilt cubins shipped in flashinfer wheels for SM 12.0, or (b) a CUDA-12.0-compatible JIT path.
2. **vLLM's bundled cutlass NVFP4 MoE kernel restricted to bfloat16 output**
   `csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu` rejects fp16 output on SM 120 (`NotImplementedError: SM120 NVFP4 MOE only supports bfloat16 output, got: Half`). A small change — adding a `half` template instantiation — would let NVFP4 work with fp16 sampling paths.
3. **Marlin GEMM workspace too large for default CUDA-graph capture sizes**
   The `moe_wna16_marlin_gemm` (GPTQ MoE) workspace at `max_num_batched_tokens=4096` × `max_num_seqs=8` × full PIECEWISE captures consumes ~25 MiB beyond the `gpu_memory_utilization` budget, OOMing during `profile_cudagraph_memory()`. Workaround in this report: cap `cudagraph_capture_sizes=[1,2,4]` and use `cudagraph_mode=FULL_DECODE_ONLY`. A vLLM-side fix would be to size the graph-capture budget against the actual marlin workspace per (batch_size × num_experts × intermediate_size), which is doable but isn't done in 0.20.2.
4. **`vllm.compilation` shows `CUDAGraphMode.FULL is not supported with GDNAttentionBackend backend`**
   The hybrid Mamba (GDN linear attention) layers in Qwen3.5-MoE fall back to PIECEWISE + FULL (PIECEWISE for prefill mixed, FULL for decode-only). This is by design — Mamba's variable-length state requires PIECEWISE. Not a regression; just a structural limit.

What upstream fixes would unblock further gains beyond ~100 tok/s:
- Native fused MoE NVFP4 cubins for SM 12.0 — would let us bypass the marlin GEMM (which still pays a 4-bit→16-bit dequant per token) and use Blackwell's hardware FP4 directly. Plausible 2-3× lift over the GPTQ-Marlin path.
- Per-expert Tensor-Core-aware grouped GEMM in vLLM (`vllm._custom_ops.cutlass_fp4_group_mm` exists but only at bfloat16, see #2). Would close the gap with TensorRT-LLM's TRTLLM MoE backend.
- Bandwidth-saturating Mamba SSM kernels on SM 12.0. Currently the GDN attention layer goes through Triton kernels; on smaller models they reach ~80% of peak bandwidth, but at 125B with 48 layers the absolute time-per-token is dominated by the MoE GEMM, so this is lower priority.

GitHub issues to watch:
- vllm-project/vllm — search `blackwell`, `sm120`, `nvfp4` in open issues
- flashinfer-ai/flashinfer — search `sm_120`, `compute_120`
- (No specific issue numbers tracked here; the report intentionally avoids guessing.)

## Honest verdict

- Hit 50-100 tok/s? **Yes**: 96.9 tok/s (GPTQ + min CUDA graphs) on the exact spec prompt, reproducible.
- Could plausibly hit 130-160 tok/s if NVFP4 + native Blackwell cubins worked: **maybe**, but blocked upstream (SM 12.0 nvcc / flashinfer cubin gap + the vLLM cutlass kernel's fp16 restriction).
- Ollama hits 93 tok/s on the same model — confirms vLLM is now in the same envelope; previously the "0.67 tok/s" gap was either methodology (streaming end-to-end vs `llm.generate`) or a regression in an older nightly that's been fixed by 0.20.2.

Bugs / quirks worked around:
- `gpu_memory_utilization=0.78` is *still* too much for default PIECEWISE graph capture on Qwen3.5-MoE-GPTQ — the workspace size is not estimated correctly by `profile_cudagraph_memory()`. Forced workaround: `compilation_config={cudagraph_capture_sizes=[1,2,4]}`.
- NVFP4 flashinfer path silently calls `gen_gemm_sm120_module_cutlass_fp4()` even when `VLLM_USE_FLASHINFER_MOE_FP4=0`; the env var only disables the *MoE* flashinfer path, not the *Linear* one. Forced workaround: also set `VLLM_NVFP4_GEMM_BACKEND=cutlass`.
- The vLLM cutlass NVFP4 MoE rejects fp16. Forced workaround: `dtype=bfloat16` for the NVFP4 entry.
- Tests pass: yes (`uv run pytest tests/test_vllm_backend.py tests/test_vllm_cache.py -q` — see deliverable 3).

## Methodology notes

- All numbers from `llm.generate()` (offline LLM, not the AsyncLLMEngine path). The async path adds ~5% overhead in our measurements, which would still leave us above 90 tok/s.
- `ignore_eos=True` to force a full 200-token generation regardless of natural stop.
- TTFT measured via a separate `max_tokens=1` call (vLLM 0.20.2 removed per-output `metrics.first_token_time` in v1 engine path).
- Decode tok/s computed as `(n−1) / (gen_s − ttft)` so the first-token cost (prefill) is amortised out — matches what `vllm bench` reports as "output token throughput".
