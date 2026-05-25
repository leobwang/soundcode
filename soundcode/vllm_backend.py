"""VllmBackend — async vLLM-based backend, interface-compatible with LlmServer.

Replaces the Ollama HTTP path in `soundcode.llm.LlmServer` with an in-process
vLLM `AsyncLLMEngine`. Same public surface DemoClient consumes:

    backend.warmup() / set_prompt() / has_next() / next() / abort_current_stream()
    backend.close() / backend.stream_thinking_phase(prompt)
    backend.config: GenerationConfig

so the two are swappable behind a duck-typed `backend` parameter on DemoClient.

What's different from LlmServer:
  - Hosts the model in-process (vs Ollama's HTTP server) — startup is one-time
    + non-trivial GPU memory commit, vs Ollama's per-request load.
  - Supports per-step LogitsProcessor for ROCODE-style penalty application,
    plus the SemGuard side-channel rollback signal.
  - Token streaming uses vLLM's incremental RequestOutput diff (each yielded
    output carries `token_ids` over the full generation so far; we diff the
    new tail and decode just that slice).

What it does NOT do (MVP):
  - No TWO_PHASE-style thinking-phase wrapping — `stream_thinking_phase` is a
    no-op async generator that yields nothing (the dispatcher falls through
    to the main producer-consumer loop, as it does for RAW mode in Ollama).
  - No CHAT_INSTRUCTED-style `/api/chat` channel splitting — vLLM has no
    server-side `thinking` field. A future patch can use a chat-template
    wrap + ThinkSplitter on the raw stream to recover similar behaviour.
  - No `think_budget_exceeded` flag — set to False so DemoClient's
    getattr-guarded read keeps working.

The Token class is reused from `soundcode.llm` so DemoClient's existing
`token.kind` / `token.text` consumption is bit-identical between backends.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from soundcode.llm import GenerationConfig, Token
from soundcode.logits_processors import LogitsProcessor, NoopLogitsProcessor


log = logging.getLogger(__name__)


@dataclass
class BackendStats:
    """Lightweight stats surface DemoClient (and the web UI) can read off the
    backend to display KV-cache reuse evidence.

    Populated on every stream open (`set_prompt` / `rollback_and_resume`):
      - `prefix_tokens`: length of the prompt token sequence we asked vLLM to
        prefill. For rollback_and_resume this is the truncated prefix.
      - `cached_tokens`: number of tokens vLLM reported as a prefix-cache hit
        (vLLM populates `RequestOutput.num_cached_tokens` for v1 engines with
        `enable_prefix_caching=True`). 0 if the engine version doesn't expose
        the field or prefix caching is off.
      - `ttft_s`: wall-clock seconds from stream open to first non-empty
        Token yielded. None until the first token arrives.
      - `is_rollback`: True if this stream was opened via `rollback_and_resume`
        (so consumers can attribute the TTFT savings to cache reuse).
      - `last_request_id`: vLLM request id of the most recent stream — useful
        for log correlation when debugging rollback sequences.
    """
    prefix_tokens: int = 0
    cached_tokens: int = 0
    ttft_s: float | None = None
    is_rollback: bool = False
    last_request_id: str | None = None

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of the prompt prefix that hit the KV cache. 0.0 if the
        engine didn't report it or the prefix was empty."""
        if self.prefix_tokens <= 0:
            return 0.0
        return min(1.0, max(0.0, self.cached_tokens / self.prefix_tokens))


class VllmBackend:
    """In-process vLLM AsyncLLMEngine backend.

    Construct, then `await warmup()` once to load the model into VRAM, then
    drive the same `set_prompt() / has_next() / next()` loop DemoClient uses
    against `LlmServer`.

    The logits processor is supplied at construction time and re-used across
    every `set_prompt()` — its state (e.g. ROCODE's penalty list) survives
    across stream opens, which is what the rollback path needs: a stream
    aborted mid-generation should re-decode with the same processor still
    holding the penalty entries the previous attempt accumulated.
    """

    def __init__(
        self,
        model: str,
        config: GenerationConfig | None = None,
        logits_processor: LogitsProcessor | None = None,
        *,
        # vLLM tuning knobs surfaced to the caller. Defaults are conservative
        # so a 7B model fits alongside other workloads.
        dtype: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        enforce_eager: bool | None = None,
        quantization: str | None = None,
        # Chunked-prefill batch budget — must be >= the attention block size.
        # For hybrid Mamba-attention MoEs like Qwen3.5-122B-A10B the attention
        # block size is padded up to the Mamba page size (e.g. 2096), and
        # vLLM asserts `block_size <= max_num_batched_tokens` at startup; set
        # this to >= the expected block size when loading such models. None
        # → let vLLM pick the default (2048 in 0.20.2).
        max_num_batched_tokens: int | None = None,
        # Per-step concurrency cap. Single-user demos want a small value
        # (~4-8) to keep scheduling overhead minimal. None → let vLLM use
        # its default (256 in 0.20.2). For Qwen3.5-MoE-GPTQ with CUDA graphs
        # enabled, the marlin GEMM workspace scales with this, so cap it.
        max_num_seqs: int | None = None,
        # vLLM CompilationConfig dict (e.g. cudagraph_capture_sizes,
        # cudagraph_mode). Passed through unchanged to AsyncEngineArgs.
        # None → vLLM picks the default (PIECEWISE, ~67 graph sizes).
        # For Qwen3.5-MoE we cap captures to [1,2,4] decode shapes to
        # avoid OOM during graph profiling — see
        # notes/vllm-122b-throughput.md.
        compilation_config: dict[str, Any] | None = None,
        # KV-cache reuse — on by default for SoundCode's rollback story. When
        # a rollback truncates the prompt to a previously-streamed prefix,
        # PagedAttention's automatic prefix cache reuses the prefill cost for
        # the shared prefix. Disable only for benchmark comparisons.
        enable_prefix_caching: bool = True,
    ) -> None:
        # If `model` is a registry key, look up the per-model defaults
        # (HF id, quantization, dtype, max_model_len, gpu_mem_util) and use
        # them for any kwarg the caller left as None. This lets the web demo
        # / dispatch table refer to models by short name (e.g.
        # "qwen3.5-122b-int4") without each call-site having to know the
        # full HF id + tuning parameters.
        from soundcode.vllm_model_registry import resolve

        reg = resolve(model)
        if reg is not None:
            resolved_model = reg.get("hf_id", model)
            dtype = dtype if dtype is not None else reg.get("dtype", "bfloat16")
            max_model_len = (
                max_model_len if max_model_len is not None
                else reg.get("max_model_len", 2048)
            )
            gpu_memory_utilization = (
                gpu_memory_utilization if gpu_memory_utilization is not None
                else reg.get("gpu_memory_utilization", 0.30)
            )
            enforce_eager = (
                enforce_eager if enforce_eager is not None
                else reg.get("enforce_eager", True)
            )
            quantization = (
                quantization if quantization is not None
                else reg.get("quantization")
            )
            max_num_batched_tokens = (
                max_num_batched_tokens if max_num_batched_tokens is not None
                else reg.get("max_num_batched_tokens")
            )
            max_num_seqs = (
                max_num_seqs if max_num_seqs is not None
                else reg.get("max_num_seqs")
            )
            compilation_config = (
                compilation_config if compilation_config is not None
                else reg.get("compilation_config")
            )
        else:
            resolved_model = model

        self.model = resolved_model
        self.config = config or GenerationConfig()
        self.logits_processor: LogitsProcessor = (
            logits_processor or NoopLogitsProcessor()
        )
        # vLLM tuning — apply built-in fallbacks for anything still None.
        self._dtype = dtype if dtype is not None else "bfloat16"
        self._max_model_len = max_model_len if max_model_len is not None else 2048
        self._gpu_memory_utilization = (
            gpu_memory_utilization if gpu_memory_utilization is not None else 0.30
        )
        self._enforce_eager = enforce_eager if enforce_eager is not None else True
        self._quantization = quantization
        self._max_num_batched_tokens = max_num_batched_tokens
        self._max_num_seqs = max_num_seqs
        self._compilation_config = compilation_config
        self._enable_prefix_caching = enable_prefix_caching
        # Lazily-created engine; build on first warmup() / open_stream().
        self._engine: Any = None
        # Iteration state, mirrors LlmServer's surface.
        self._prompt: str = ""
        self._stream_iter: AsyncIterator[Token] | None = None
        self._current_request_id: str | None = None
        self._exhausted: bool = True
        # DemoClient reads this via getattr; keep present-but-False so its
        # branch on "C3 think-budget exceeded" never fires for the vLLM path.
        self.think_budget_exceeded: bool = False
        # KV-cache stats surface — populated on each stream open and updated
        # as the first token arrives. Public attribute so DemoClient / the
        # web UI can read it directly off the backend instance.
        self.last_stats: BackendStats = BackendStats()
        # Per-stream bookkeeping for TTFT measurement. Reset by `_open_stream`
        # and `rollback_and_resume`.
        self._stream_open_time: float | None = None
        self._next_is_rollback: bool = False

    # ─── prompt control (mirror of LlmServer's surface) ────────────────

    def set_prompt(self, prompt: str) -> None:
        """Set the prompt to use on the *next* stream open."""
        self._prompt = prompt
        self._exhausted = False

    def has_next(self) -> bool:
        return not self._exhausted

    async def next(self) -> Token:
        if self._stream_iter is None:
            await self._open_stream()
        assert self._stream_iter is not None
        try:
            token = await anext(self._stream_iter)
        except StopAsyncIteration:
            self._exhausted = True
            await self._close_stream()
            return Token("", "code")
        return token

    async def abort_current_stream(self) -> None:
        """Cancel any in-flight vLLM request and mark exhausted so DemoClient
        knows to call set_prompt() again before resuming."""
        self._exhausted = True
        await self._close_stream()

    async def rollback_and_resume(
        self,
        prompt_at_rollback: str,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[Token]:
        """Abort the current in-flight request (if any) and resume from the
        rollback point with a fresh request.

        `prompt_at_rollback` is the prompt prefix up to the rollback point —
        normally the original prompt plus whatever tokens DemoClient kept
        before the bad suffix. vLLM's automatic prefix caching transparently
        reuses the KV blocks shared with the aborted request, so the prefill
        cost for the common prefix is effectively zero. The shorter the
        rollback, the larger the cache hit.

        Returns an async iterator of `Token` objects, same shape as the one
        DemoClient pumps via `next()`. Callers can either:
          - `async for tok in backend.rollback_and_resume(prefix): ...` to
            consume directly, OR
          - `backend.set_prompt(prefix); await backend.next()` to fall back
            into the existing has_next()/next() driver (which is what the
            web demo does today).

        Stats for the resumed stream are exposed on `backend.last_stats`:
        `cached_tokens / prefix_tokens` gives the cache hit rate, and `ttft_s`
        captures the wall-clock time to first decoded character — both of
        which the demo overlay can display to make the speedup visible.
        """
        # Abort whatever was running. `_close_stream` handles the missing-
        # request-id case (e.g. if the stream already finished cleanly).
        await self._close_stream()

        new_req_id = request_id or f"soundcode-rb-{uuid.uuid4().hex[:12]}"
        self._current_request_id = new_req_id
        self._prompt = prompt_at_rollback
        self._reset_stats(req_id=new_req_id, is_rollback=True)
        # Build the underlying stream and stash it on the instance so
        # subsequent has_next()/next() calls find it. Also yield Tokens
        # directly to callers that prefer the async-iterator shape.
        if self._engine is None:
            await self.warmup()
        stream = self._stream_request(prompt_at_rollback, new_req_id)
        self._stream_iter = stream
        self._exhausted = False
        async for token in stream:
            yield token
        # Stream exhausted — match `next()`'s end-of-stream semantics so
        # callers can chain rollback_and_resume directly into set_prompt.
        self._exhausted = True
        self._stream_iter = None

    async def warmup(self) -> None:
        """Build the AsyncLLMEngine (loads weights into VRAM). Idempotent —
        a second call is a no-op."""
        if self._engine is None:
            await asyncio.to_thread(self._build_engine)

    async def close(self) -> None:
        """Release the engine + GPU memory. After close(), warmup() can be
        called again to re-load; but most callers won't bother."""
        await self._close_stream()
        if self._engine is not None:
            try:
                # vLLM engines expose a shutdown() in v1 mode; gracefully
                # handle older APIs that don't.
                shut = getattr(self._engine, "shutdown", None)
                if shut is not None:
                    res = shut()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception as e:
                log.warning("vLLM engine shutdown failed: %r", e)
            self._engine = None

    async def stream_thinking_phase(self, prompt: str) -> AsyncIterator[Token]:
        """TWO_PHASE phase-1 hook. The vLLM backend doesn't implement chat-
        template-wrapped thinking yet — yield nothing so DemoClient's
        TWO_PHASE dispatch falls through cleanly (it tests for an empty
        trace and skips the comment-bake step in that case).

        Defined as `async def` returning an iterator so the duck-typed
        protocol matches LlmServer's `stream_thinking_phase`.
        """
        return
        yield  # pragma: no cover — keeps this a generator function

    # ─── internals ─────────────────────────────────────────────────────

    def _build_engine(self) -> None:
        """Build the AsyncLLMEngine synchronously (run inside asyncio.to_thread
        so the event loop stays responsive). Once built, kept for the backend's
        lifetime."""
        # Imported here so the unit tests can import vllm_backend without a
        # live vLLM installation (the import only happens at warmup time).
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_kwargs: dict[str, Any] = dict(
            model=self.model,
            dtype=self._dtype,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enforce_eager=self._enforce_eager,
            # Automatic prefix caching — PagedAttention reuses KV blocks for
            # any new request whose prompt shares a prefix with a previously-
            # seen one. This is the mechanism that makes rollback_and_resume
            # cheap: aborting then resubmitting with a truncated prompt costs
            # ~0 prefill on the shared prefix.
            enable_prefix_caching=self._enable_prefix_caching,
        )
        # `quantization=None` lets vLLM infer from config.json — pass through
        # only when explicitly set (e.g. "compressed-tensors" for the
        # Qwen3.5-122B-A10B-AWQ-4bit weights, "awq", "gptq", "fp8", etc.).
        if self._quantization is not None:
            engine_kwargs["quantization"] = self._quantization
        # `max_num_batched_tokens` — bumped above 2048 (the vLLM default) for
        # hybrid Mamba-attention MoEs like Qwen3.5-122B-A10B, whose attention
        # block size is padded to the Mamba page size (~2096); the engine
        # asserts `block_size <= max_num_batched_tokens` at startup.
        if self._max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = self._max_num_batched_tokens
        # `max_num_seqs` — concurrent-request cap; for single-user demos
        # we keep this small so the marlin MoE GEMM workspace stays small
        # (graphs capture at multiples of `max_num_seqs`).
        if self._max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = self._max_num_seqs
        # `compilation_config` — pass-through dict (vLLM accepts a dict and
        # parses it into CompilationConfig). For Qwen3.5-MoE-GPTQ we cap
        # `cudagraph_capture_sizes` to small decode shapes to avoid OOM
        # during graph profiling; see the registry entry's comment.
        if self._compilation_config is not None:
            engine_kwargs["compilation_config"] = self._compilation_config
        engine_args = AsyncEngineArgs(**engine_kwargs)
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def _open_stream(self) -> None:
        """Open a vLLM generate stream for `self._prompt`. The async iterator
        diffs new tokens off each RequestOutput and yields them as Tokens."""
        if self._engine is None:
            await self.warmup()
        assert self._engine is not None
        req_id = f"soundcode-{uuid.uuid4().hex[:12]}"
        self._current_request_id = req_id
        # Stats reset — first token will fill in TTFT, generate loop will
        # fill in cached_tokens. `is_rollback` was set by the most recent
        # caller (defaults to False after a fresh set_prompt).
        is_rollback = self._next_is_rollback
        self._next_is_rollback = False
        self._reset_stats(req_id=req_id, is_rollback=is_rollback)
        self._stream_iter = self._stream_request(self._prompt, req_id)
        self._exhausted = False

    def _build_sampling_params(self) -> Any:
        """SamplingParams shared between `_open_stream` and `rollback_and_resume`.

        Wires the per-step logits processor through vLLM's SamplingParams when
        the running vLLM version still accepts the `logits_processors` kwarg.
        Newer v1 engines (0.10+) moved per-request processors to engine-level
        `ModelConfig.logits_processors` — in that case we silently fall back
        to a plain SamplingParams; the engine's processor list (if registered)
        still runs. The processor's __call__(token_ids, logits) signature
        matches both calling conventions.
        """
        from vllm import SamplingParams

        base_kwargs: dict[str, Any] = dict(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=(self.config.max_tokens
                        if self.config.max_tokens > 0 else 2048),
            stop=list(self.config.stop) if self.config.stop else None,
        )
        try:
            return SamplingParams(
                **base_kwargs,
                logits_processors=[self.logits_processor],
            )
        except TypeError:
            # vLLM dropped per-request logits_processors kwarg (v1 engine
            # registration path). Caller is expected to have registered the
            # processor at engine-construction time.
            return SamplingParams(**base_kwargs)

    def _reset_stats(self, *, req_id: str, is_rollback: bool) -> None:
        """Clear `last_stats` and start the TTFT clock for a fresh stream."""
        self.last_stats = BackendStats(
            prefix_tokens=0,
            cached_tokens=0,
            ttft_s=None,
            is_rollback=is_rollback,
            last_request_id=req_id,
        )
        self._stream_open_time = time.perf_counter()

    async def _stream_request(
        self, prompt: str, req_id: str,
    ) -> AsyncIterator[Token]:
        """Submit `prompt` under `req_id` and yield decoded Tokens.

        Detokenizes incrementally by tracking the accumulated text rather
        than the per-token text — the latter is fragile across BPE-style
        vocabs where one-byte tokens decode to garbage in isolation but
        join cleanly. vLLM's RequestOutput exposes `.outputs[0].text`
        which is the accumulated detokenized string for the whole
        generation; we slice off the new tail.

        Side effects on `self.last_stats`:
          - `prefix_tokens` set from the first RequestOutput's
            `prompt_token_ids` length (vLLM exposes this once the engine
            has tokenized the prompt).
          - `cached_tokens` set from `RequestOutput.num_cached_tokens` if
            the engine populates it (v1 engine + prefix caching on).
          - `ttft_s` set on the first non-empty Token yielded.
        """
        assert self._engine is not None
        engine = self._engine
        params = self._build_sampling_params()
        # Per-stream detokenizer cursor — kept local so concurrent or
        # back-to-back streams can't trample each other's state.
        prev_text_len = 0
        try:
            async for output in engine.generate(prompt, params, req_id):
                # Capture prompt-length + cache-hit metrics once they're
                # available. `prompt_token_ids` is populated as soon as the
                # engine has tokenized; `num_cached_tokens` lands on v1
                # engines with prefix caching enabled. Both are monotonic,
                # so overwriting every iteration is safe.
                pti = getattr(output, "prompt_token_ids", None)
                if pti is not None and self.last_stats.prefix_tokens == 0:
                    self.last_stats.prefix_tokens = len(pti)
                nct = getattr(output, "num_cached_tokens", None)
                if nct is not None:
                    self.last_stats.cached_tokens = int(nct)

                if not output.outputs:
                    continue
                out0 = output.outputs[0]
                full_text = out0.text or ""
                if len(full_text) > 0 and self.last_stats.ttft_s is None:
                    # Record TTFT on first decoded character — this is what
                    # the user perceives as "the model started responding".
                    if self._stream_open_time is not None:
                        self.last_stats.ttft_s = (
                            time.perf_counter() - self._stream_open_time
                        )
                # Slice new tail off the cumulative detokenized text.
                if len(full_text) > prev_text_len:
                    new_text = full_text[prev_text_len:]
                    prev_text_len = len(full_text)
                    yield Token(new_text, "code")
                if output.finished:
                    break
        except asyncio.CancelledError:
            # External abort — propagate (the engine cancellation hook
            # in `_close_stream` already aborted the request).
            raise

    async def _close_stream(self) -> None:
        """Cancel the in-flight request (if any) and drop the iterator."""
        if self._stream_iter is not None:
            self._stream_iter = None
        if self._engine is not None and self._current_request_id is not None:
            try:
                abort = getattr(self._engine, "abort", None)
                if abort is not None:
                    res = abort(self._current_request_id)
                    if asyncio.iscoroutine(res):
                        await res
            except Exception as e:
                log.warning(
                    "vLLM abort(%s) failed: %r",
                    self._current_request_id, e,
                )
            self._current_request_id = None
