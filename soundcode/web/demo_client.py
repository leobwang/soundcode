"""DemoClient — instrumented variant of CodeClient that emits structured
events to a callback so the web frontend can animate them.

Wraps the same producer-consumer loop as `soundcode.client.CodeClient`,
but adds:
  - `emit(event_dict)` hook called at every loop interest point.
  - Optional throttling per emitted token (for slow/visual demos).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from soundcode.code import Category, Code, State
from soundcode.llm import LlmServer

EmitFn = Callable[[dict], Awaitable[None]]


def _is_body_trivially_empty(content: str) -> bool:
    """True iff `content` has no characters outside whitespace and Rust
    comments (line `//…` or block `/* … */`). The function body essentially
    has no code — gluing on a `}` closer would produce `fn f() { }`, which
    compiles but doesn't constitute a solution."""
    i = 0
    n = len(content)
    while i < n:
        c = content[i]
        if c.isspace():
            i += 1
            continue
        if c == "/" and i + 1 < n and content[i + 1] == "/":
            j = content.find("\n", i + 2)
            i = n if j == -1 else j + 1
            continue
        if c == "/" and i + 1 < n and content[i + 1] == "*":
            j = content.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        return False
    return True


@dataclass
class DemoConfig:
    instruct_on_rollback: bool = False
    wall_budget_s: float = 180.0
    token_delay_s: float = 0.0  # 0 = real-time
    repetition_window: int = 60        # 0 disables the watchdog
    repetition_threshold: int = 3      # abort when same window appears N+ times
    max_continuations: int = 3         # if final check is incomplete, how many
                                       # times to re-prompt for continuation


class DemoClient:
    def __init__(self, llm: LlmServer, checker, *, config: DemoConfig, emit: EmitFn):
        self.llm = llm
        self.checker = checker
        self.config = config
        self.emit = emit

    async def generate(self, prompt: str) -> str:
        code = Code(prefix=prompt, suffix="}", checker=self.checker)
        tasks: deque[asyncio.Task] = deque()
        self.llm.set_prompt(prompt)
        rollbacks = 0
        t0 = time.perf_counter()

        # Repetition watchdog: count how many times any
        # `repetition_window`-sized substring has appeared in the buffer.
        # Reset on rollback (the buffer shrinks).
        window_hashes: dict[int, int] = {}

        async def _do_rollback(suffix_msg: str, reason: str,
                               survivor_offset: int | None = None) -> int:
            """Shared rollback path. If `survivor_offset` is None, uses the
            latest checkpoint (default cargo-error behaviour). If supplied,
            truncates explicitly to that offset and prunes any later
            checkpoints — used by the repetition watchdog to slice out the
            entire repetition basin, not just to the in-basin checkpoint."""
            nonlocal rollbacks
            rollbacks += 1
            if survivor_offset is None:
                survivor_offset = code.ckpt[-1]
            else:
                survivor_offset = max(0, min(survivor_offset, len(code.content)))
            discarded = code.content[survivor_offset:]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            tasks.clear()
            await self.llm.abort_current_stream()
            new_prompt = prompt + code.content_up_to(survivor_offset) + suffix_msg
            self.llm.set_prompt(new_prompt)
            code.rollback(to_offset=survivor_offset)
            window_hashes.clear()
            # Snapshot reflects the *pre-rollback* buffer so the UI can
            # show the discarded state when the entry is clicked.
            pre_rollback_snapshot = code.content_up_to(survivor_offset) + discarded
            await self.emit({
                "type": "rollback",
                "to_offset": survivor_offset,
                "discarded": discarded,
                "rollback_count": rollbacks,
                "reason": reason,
                "code_snapshot": pre_rollback_snapshot,
            })
            return rollbacks

        await self.emit({"type": "status", "phase": "generating", "message": "Streaming tokens"})
        await self.emit({"type": "reset_buffer"})

        while True:
            elapsed = time.perf_counter() - t0
            if elapsed > self.config.wall_budget_s:
                await self.emit({
                    "type": "status", "phase": "done",
                    "message": f"wall budget ({self.config.wall_budget_s:.0f}s) exceeded",
                })
                break

            # 1. Produce.
            if self.llm.has_next():
                token = await self.llm.next()
                if token:
                    if self.config.token_delay_s > 0:
                        await asyncio.sleep(self.config.token_delay_s)
                    code.append(token)
                    await self.emit({
                        "type": "token_emitted",
                        "text": token,
                        "offset": len(code.content),
                    })
                    # Fix #3: structural termination on function-body close.
                    if code.body_closed:
                        for t in tasks:
                            t.cancel()
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)
                        tasks.clear()
                        await self.llm.abort_current_stream()
                        await self.emit({
                            "type": "status", "phase": "done",
                            "message": "function body closed; generation complete",
                        })
                        break
                    # Fix #2: repetition watchdog → trigger a rollback (not
                    # an abort). User feedback: retrying with the buffer
                    # truncated and a "you were repeating" hint in the
                    # re-prompt actually breaks the loop in many cases.
                    if self.config.repetition_window > 0 and len(code.content) >= self.config.repetition_window:
                        window = code.content[-self.config.repetition_window:]
                        h = hash(window)
                        window_hashes[h] = window_hashes.get(h, 0) + 1
                        if window_hashes[h] >= self.config.repetition_threshold:
                            await self.emit({
                                "type": "status", "phase": "generating",
                                "message": (
                                    f"repetition detected (a {self.config.repetition_window}-char window appeared "
                                    f"{self.config.repetition_threshold}×) — rolling back"
                                ),
                            })
                            # Rollback target: truncate to BEFORE the first
                            # occurrence of the repeating window so the
                            # entire repetition basin is discarded, not just
                            # the latest copy (which the latest checkpoint
                            # may already sit inside).
                            first_occ = code.content.find(window)
                            if first_occ >= 0:
                                # Snap to the most recent checkpoint at or before
                                # `first_occ`. Falls back to 0 if none.
                                survivor = max(
                                    [c for c in code.ckpt if c <= first_occ] + [0]
                                )
                            else:
                                survivor = code.ckpt[-1]
                            # Always inject an instruction here, regardless of
                            # `instruct_on_rollback` — without a context change
                            # the LM would re-enter the repetition basin.
                            suffix_msg = (
                                "\n// previous attempt got stuck repeating itself; "
                                "take a different approach to finish the function.\n"
                            )
                            await _do_rollback(suffix_msg, reason="repetition",
                                               survivor_offset=survivor)
                            # Restart the outer producer-consumer loop —
                            # `break` here would exit `while True` entirely
                            # (this branch is NOT inside the inner consume
                            # loop the way the cargo-error rollback is).
                            continue
                    if code.at_boundary():
                        await self.emit({
                            "type": "check_started",
                            "offset": len(code.content),
                            "code_snapshot": code.content,
                        })
                        tasks.append(asyncio.create_task(self._wrapped_check(code, len(code.content))))

            # 2. Consume.
            while tasks and tasks[0].done():
                offset_at_check, result = tasks.popleft().result()
                verdict = result.verdict
                diags = [
                    {
                        "category": d.category.value,
                        "code": d.code,
                        "message": d.message,
                        "line": d.line,
                        "column": d.column,
                    }
                    for d in result.diagnostics
                ]
                await self.emit({
                    "type": "verdict",
                    "verdict": verdict.value,  # "inactive" | "ok" | "error"
                    "diagnostics": diags,
                    "offset": offset_at_check,
                    "code_snapshot": code.content[:offset_at_check],
                })
                if verdict is State.OK:
                    await self.emit({"type": "checkpoint", "offset": code.ckpt[-1]})
                if verdict is State.ERROR:
                    suffix_msg = ""
                    if self.config.instruct_on_rollback and result.message:
                        # result.message is already deduplicated (see CheckResult.message).
                        # Take up to 5 unique error lines.
                        err_lines = [ln for ln in result.message.splitlines() if ln.strip()][:5]
                        suffix_msg = (
                            "\n// previous attempt failed cargo check with:\n"
                            + "\n".join(f"// {ln}" for ln in err_lines) + "\n"
                        )
                    # Boundary-based checkpointing makes `ckpt[-1]` the latest
                    # `;`/`}` boundary, which may sit AT or past the cargo-
                    # error site. Pick the largest checkpoint strictly before
                    # `offset_at_check` so the failing slice is actually
                    # discarded. Falls back to 0 if no such ckpt exists.
                    survivor = max([c for c in code.ckpt if c < offset_at_check] + [0])
                    await _do_rollback(suffix_msg, reason="cargo_error",
                                       survivor_offset=survivor)
                    break

            # 3. Termination.
            if not self.llm.has_next():
                if not tasks:
                    await self.emit({"type": "status", "phase": "done", "message": "stream complete"})
                    break
                try:
                    await tasks[0]
                except asyncio.CancelledError:
                    pass

        # Drain any remaining.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Run a final post-hoc cargo check on the *full* function body — no
        # unreachable!() shim — to surface errors the mid-gen check structurally
        # cannot detect: missing-return, unclosed-brace, comments-only body, etc.
        # If it reports the solution is incomplete, re-prompt the LM with a
        # "finish the function" hint and run another stream. Bounded by
        # `max_continuations` (default 3) so we don't loop forever on a LM
        # that keeps emitting EOS too early.
        continuations = 0
        while True:
            is_complete = await self._emit_final_check(prompt, code.content)
            if is_complete or continuations >= self.config.max_continuations:
                break
            if time.perf_counter() - t0 > self.config.wall_budget_s:
                break
            continuations += 1
            await self.emit({
                "type": "status", "phase": "generating",
                "message": (
                    f"final check incomplete — re-prompting the LM to continue "
                    f"(attempt {continuations}/{self.config.max_continuations})"
                ),
            })
            # No truncation; just append a "please continue" hint and resume.
            suffix_msg = (
                "\n// the function body above is incomplete; "
                "continue from here and finish the function.\n"
            )
            self.llm.set_prompt(prompt + code.content + suffix_msg)
            await self.emit({
                "type": "continuation",
                "attempt": continuations,
                "of": self.config.max_continuations,
                "code_snapshot": code.content,
            })
            # Run another producer-consumer round, identical to the main loop
            # but bounded by has_next() and wall budget. We reuse the same
            # `code`, `tasks`, `window_hashes` state.
            window_hashes.clear()
            await self._run_stream_phase(code, tasks, prompt, window_hashes, t0)

        await self.emit({"type": "final", "content": code.content,
                         "rollback_count": rollbacks,
                         "continuations": continuations,
                         "code_snapshot": code.content})
        return code.content

    async def _run_stream_phase(self, code, tasks, prompt, window_hashes, t0):
        """The producer-consumer loop body, factored out so it can be re-run
        for continuation attempts after a failed final_check."""
        while True:
            if time.perf_counter() - t0 > self.config.wall_budget_s:
                return
            if self.llm.has_next():
                token = await self.llm.next()
                if token:
                    if self.config.token_delay_s > 0:
                        await asyncio.sleep(self.config.token_delay_s)
                    code.append(token)
                    await self.emit({
                        "type": "token_emitted",
                        "text": token,
                        "offset": len(code.content),
                    })
                    if code.body_closed:
                        for t in tasks:
                            t.cancel()
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)
                        tasks.clear()
                        await self.llm.abort_current_stream()
                        await self.emit({
                            "type": "status", "phase": "done",
                            "message": "function body closed; generation complete",
                        })
                        return
                    if self.config.repetition_window > 0 and len(code.content) >= self.config.repetition_window:
                        window = code.content[-self.config.repetition_window:]
                        h = hash(window)
                        window_hashes[h] = window_hashes.get(h, 0) + 1
                        # Note: we DON'T trigger a full rollback here in the
                        # continuation phase — repetition within continuation
                        # output is likely the LM stalling; better to just
                        # abort this attempt and let `generate()` decide.
                        if window_hashes[h] >= self.config.repetition_threshold:
                            await self.emit({
                                "type": "status", "phase": "generating",
                                "message": (
                                    f"continuation repetition detected — abandoning attempt"
                                ),
                            })
                            await self.llm.abort_current_stream()
                            return
            # Termination.
            if not self.llm.has_next():
                if not tasks:
                    return
                try:
                    await tasks[0]
                except asyncio.CancelledError:
                    pass

    async def _emit_final_check(self, prompt: str, content: str) -> bool:
        """Run cargo / rust-analyzer on (prompt + content + real closer) and
        emit a `final_check` event. Returns True iff the solution is
        considered complete (no blocking + no structural-incompleteness
        signals)."""
        # An empty / comments-only body trivially compiles when we paste a
        # `}` closer ("`fn foo() { }`"), but it's not a real solution. Flag
        # it so the continuation loop re-prompts the LM. Without this, a
        # run where every rollback retreats to offset 0 (no surviving code)
        # returns is_complete=True and emits the empty content as the answer.
        if _is_body_trivially_empty(content):
            await self.emit({
                "type": "final_check",
                "verdict": "error",
                "is_complete": False,
                "blocking_count": 0,
                "incomplete_signal_count": 1,
                "non_blocking_count": 0,
                "diagnostics": [{
                    "category": "incomplete",
                    "code": "empty-body",
                    "message": "function body has no code (only whitespace / comments)",
                    "line": None, "column": None,
                }],
                "code_snapshot": content,
            })
            return False
        body_closed = content.rstrip().endswith("}")
        closer = "" if body_closed else "\n}\n"
        real_source = prompt + content + closer + "\n\nfn main() {}\n"
        await self.emit({"type": "status", "phase": "generating",
                         "message": "running final post-hoc cargo check"})
        try:
            raw_diags = await self.checker.check(real_source)
        except Exception as e:
            await self.emit({
                "type": "final_check", "verdict": "error",
                "message": f"checker failed: {e!r}",
                "diagnostics": [],
                "is_complete": False,
            })
            return False
        # Classify (same rule the in-loop check uses — but BLOCKING here is
        # the result we actually care about for "is the solution complete").
        from soundcode.code import Category, Code as _Code
        classified = [_Code._classify(d) for d in raw_diags]
        blocking = [d for d in classified if d.category is Category.BLOCKING]
        non_blocking = [d for d in classified if d.category is Category.NON_BLOCKING]
        incomplete = [d for d in classified if d.category is Category.INCOMPLETE]
        # Detect the "obviously incomplete solution" pattern: any cargo error
        # whose message hints at structural incompleteness gets surfaced.
        keywords = (
            "mismatched types", "expected `", "expected expression",
            "expected one of", "missing return", "expected `}`",
            "expected `;`", "unclosed delimiter", "function body required",
            "expected return type", "this function should return",
        )
        incomplete_signals = [
            d for d in (blocking + incomplete)
            if any(k in (d.message or "").lower() for k in (s.lower() for s in keywords))
        ]
        verdict = "ok" if (not blocking and not incomplete_signals) else "error"
        is_complete = verdict == "ok"
        await self.emit({
            "type": "final_check",
            "verdict": verdict,
            "is_complete": is_complete,
            "blocking_count": len(blocking),
            "incomplete_signal_count": len(incomplete_signals),
            "non_blocking_count": len(non_blocking),
            "diagnostics": [
                {
                    "category": d.category.value,
                    "code": d.code,
                    "message": d.message,
                    "line": d.line,
                    "column": d.column,
                }
                for d in (blocking[:8] + incomplete_signals[:4] + non_blocking[:3])
            ],
            "code_snapshot": content,
        })
        return is_complete

    async def _wrapped_check(self, code: Code, offset_at_start: int):
        result = await code.check()
        return offset_at_start, result
