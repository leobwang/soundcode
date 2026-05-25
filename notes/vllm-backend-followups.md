# vLLM backend — follow-ups

Items deliberately left out of the MVP (week-6 vLLM backend + LogitsProcessor
plumbing). Each line is one issue + roughly what it'd take to land it.

- **KV-cache crop on rollback** — current `_close_stream()` just aborts the
  request and re-decodes from scratch on the next `set_prompt()`. The
  perf win is that the survivor prefix's KV cache is already on-GPU; vLLM's
  AsyncLLMEngine has no public knob for "truncate this request's KV to N
  tokens" yet. Requires either a custom Scheduler hook or a fork.

- **SemGuard evaluator wiring** — currently no-op + warning. Once the
  trained ~1.3B checkpoint stabilises (epoch-2+ at
  `reproduce/results/semguard/train_20260524_211624_resume/...`), load it
  in `SemGuardEvaluatorProcessor.__init__`, then in `__call__` score the
  partial program at each newline and set `pending_rollback_offset` to the
  last-newline offset when score < threshold.

- **Full ROCODE trie integration** — `RocodeDecayingPenaltyProcessor` does
  the penalty math correctly, but nothing today feeds penalties to it. The
  trie that records bad-suffix tokens on every rollback (ROCODE §3.2) needs
  to be added to `DemoClient._do_rollback`: extract the rolled-back token
  ids from the LLM's tokenizer, call `processor.add_penalty(ids, offset)`.
  Requires DemoClient to know the backend's tokenizer (or have the backend
  expose a `tokenize_suffix(text, offset) -> list[int]` helper).

- **SemGuard side-channel rollback signal** — `pending_rollback_offset` is
  set by the processor but no consumer polls it. DemoClient's main producer
  loop would need a new branch between `_emit_token` and `body_closed`
  check: "if backend.logits_processor has a pending rollback, fire the
  rollback path." Couples DemoClient to a backend-specific attribute, so
  the cleaner design is a `backend.poll_rollback_signal() -> int | None`
  helper that hides where the signal came from.

- **Integration tests** — `tests/test_vllm_backend.py` covers plumbing
  only. Need a slow-marker test that drives DemoClient end-to-end against a
  real vLLM-backed model with each of the three logits processors (no-op,
  ROCODE with stub penalty, SemGuard placeholder) and asserts the
  expected number of rollbacks / token kinds.

- **TWO_PHASE / CHAT_INSTRUCTED on vLLM** — `stream_thinking_phase` is a
  no-op generator on `VllmBackend`. Equivalent functionality requires a
  chat-template wrap + `</think>` stop token on the raw stream (vLLM has no
  server-side thinking-channel split like Ollama's `/api/chat`). For now,
  RAW mode is the only one fully supported on the vLLM path; the dispatch
  silently uses RAW for non-RAW modes.
