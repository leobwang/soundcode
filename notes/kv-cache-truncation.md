# KV-Cache Truncation for Token-Level Rollback

Reference for implementing the rollback primitive that `week3-plan.md §3` and the two-tier architecture in `rust-analyzer-scope-and-granularity.md §5` depend on.

---

## 1. The core question

To roll back generation to an earlier statement boundary, we need to restore the transformer's KV-cache to the state it had at that boundary. Options depend on the inference framework.

| Framework | Token-level KV truncation | Notes |
|---|---|---|
| HuggingFace `transformers` | ✅ exposed directly via `past_key_values` | exact, but slow; single-stream |
| vLLM | ❌ no public API | has workarounds via abort + prefix caching |
| SGLang | partial (via RadixAttention cache reuse) | similar tradeoff to vLLM |
| llama.cpp | ✅ `llama_kv_cache_seq_rm` exists | but no Python bindings for tensor-level access |

---

## 2. Path 1: vLLM — abort + re-prefill with prefix caching

vLLM's PagedAttention block manager is not designed for user-initiated mid-sequence rollback. The pragmatic substitute: treat rollback as "kill the current request, re-issue with a shorter prompt." With `enable_prefix_caching=True`, vLLM detects that the new prompt shares a prefix with the aborted request's blocks and reuses them — so the re-prefill is cheap (only the suffix + error feedback gets recomputed).

### Sketch

```python
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="Qwen/Qwen2.5-Coder-1.5B",
    enable_prefix_caching=True,      # critical — makes re-prefill cheap
    block_size=16,                   # smaller blocks → finer cache-hit granularity
))

async def generate_with_rollback(prompt: str, sampling: SamplingParams):
    tokens: list[int] = []
    checkpoints: list[tuple[int, list[int]]] = []  # (token_pos, token_ids)

    rid = str(uuid.uuid4())
    stream = engine.generate(prompt, sampling, request_id=rid)

    async for out in stream:
        new_tokens = out.outputs[0].token_ids[len(tokens):]
        for tok in new_tokens:
            tokens.append(tok)
            if is_statement_boundary(tokens):       # ; / } detector
                checkpoints.append((len(tokens), tokens[:]))

            errors = await check_with_rust_analyzer(tokens)
            if errors:
                target_pos, target_tokens = pick_rollback_target(checkpoints, errors)
                await engine.abort(rid)             # frees this request's blocks
                prefix_text = tokenizer.decode(target_tokens)
                new_prompt = prefix_text + f"\n// Error: {errors[0].message}\n"
                rid = str(uuid.uuid4())
                stream = engine.generate(new_prompt, sampling, request_id=rid)
                tokens = target_tokens[:]
                checkpoints = [c for c in checkpoints if c[0] <= target_pos]
                break  # restart the outer async for on new stream

    return tokens
```

### What prefix caching gives you vs. doesn't

| Rollback lands on... | Re-prefill cost |
|---|---|
| token boundary aligned with a block (e.g., 16 tokens) | near-zero — cached blocks reused |
| mid-block (rollback 8 tokens into a 16-block) | that one block recomputed; rest are cache hits |
| shorter than full prompt + error text | error text is new, original prompt stays hot |

**Design guidance:** small `block_size` (8 or 16) + checkpoint on statement boundaries. Rollbacks that land near block boundaries get the full speedup.

### What `abort` actually does

`AsyncLLMEngine.abort(request_id)` removes the sequence from the scheduler and frees its blocks via the BlockManager. Blocks belonging to shared prefixes stay in cache (reference-counted). The aborted tokens are gone from the engine's view; you reintroduce them as text in the new prompt.

---

## 3. Path 2: HuggingFace transformers — true token-level rollback

Exposes `past_key_values` as an ordinary Python tuple of tensors. This is what `week3-plan.md §3` calls for.

### Sketch

```python
outputs = model(input_ids, past_key_values=kv, use_cache=True)
# kv shape: tuple of (keys, values) per layer,
# each tensor shape [batch, heads, seq, head_dim]

# Checkpoint: keep a reference (optionally .clone() if you want to mutate safely).
checkpoint_kv = tuple((k.clone(), v.clone()) for k, v in kv)

# Rollback to position N: slice along the seq dim.
truncated_kv = tuple((k[:, :, :N, :], v[:, :, :N, :]) for k, v in kv)

# Resume: feed the next token with truncated_kv as past_key_values.
```

**Memory note:** naive checkpointing keeps every past state in GPU memory. For long generations, checkpoint by *copying the slice you need* (not the full tensor) or store checkpoints on CPU and move back on rollback.

---

## 4. Tradeoffs

| | HuggingFace | vLLM (abort + prefix cache) |
|---|---|---|
| Token-level precision | exact | block-aligned (recompute mid-block tokens) |
| Throughput (single stream) | baseline | 5–10× faster |
| Throughput (batch) | limited | continuous batching wins heavily |
| Memory | grows linearly with seq_len | PagedAttention, bounded |
| Implementation effort | trivial | moderate (stream handling, abort/re-issue) |
| KV-cache observability | full | opaque |
| Usable for FLOPs-cost research | yes | hard — block abstraction hides exact cost |

---

## 5. Which to pick

| Scenario | Pick |
|---|---|
| Research prototype, MultiPL-E eval (156 problems, single-stream) | **HuggingFace** — `week3-plan.md` already chose this |
| Production serving / throughput-bound | **vLLM + abort + prefix caching** |
| FLOP-accounted research for a paper | **HuggingFace** — vLLM hides the cost |
| Fine-grained KV research | **HuggingFace** |

**Decision for this project:** HuggingFace transformers for the prototype. Migrate to vLLM only if throughput or concurrent-request serving becomes the bottleneck.

---

## 6. Middle-ground hacks if vLLM is non-negotiable

1. **Stop-token heuristic**: set `stop=[";", "}", "\n"]` so generation halts at boundaries, then make a fresh request with more context + error feedback. Avoids abort plumbing at the cost of constantly restarting. Simple; loses some throughput.
2. **Fork vLLM, expose `BlockManager.free_last_n_blocks`**: this method exists internally. Fragile — the block-manager code is rewritten every few vLLM versions.
3. **vLLM built-in speculative decoding** (`speculative_model=...`): internally does token rollback, but only against a draft model's token probabilities. Not usable for LSP-gated rollback — the acceptance criterion is internal to the engine.

---

## 7. Relationship to the rollback mechanism

Checkpoint granularity from `rust-analyzer-scope-and-granularity.md §5` is **statement boundary**. That cadence determines the KV-cache access pattern:

- **Checkpoint cost**: one `past_key_values` reference (HF) or nothing explicit (vLLM relies on block reuse).
- **Rollback cost**: one tensor slice (HF, ~microseconds) or one `abort` + re-prefill of the suffix (vLLM, milliseconds dominated by block recomputation).
- **Frequency**: one checkpoint per statement. A 20-line function produces ~20 checkpoints. Rollback depth is usually 1–3 statements based on the pattern of cascading errors in the week3 experiment.
