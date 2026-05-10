"""Thinking-model wrapper — strips `<think>...</think>` from token streams.

Models like deepseek-r1 emit reasoning traces before the final answer. For
LSP-guided generation we care about the post-thinking *code*, not the
thinking itself. This wrapper:

- Buffers tokens seen inside `<think>...</think>`.
- Emits downstream only tokens after `</think>`.
- Measures thinking time separately (for fair T-A vs. T-B comparison).
- Enforces a `thinking_timeout_s` on how long we wait for `</think>`.

Two rollback modes (§8.1):
- T-A: re-enter thinking after rollback (inject instruction "reconsider").
- T-B: skip thinking after rollback (inject "directly continue from cursor").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Mode(str, Enum):
    T_A = "re-enter-thinking"
    T_B = "skip-thinking"


class _State(Enum):
    BEFORE_THINK = "before"
    IN_THINK = "in"
    AFTER_THINK = "after"


@dataclass
class ThinkingWrapper:
    """Streaming filter that strips `<think>...</think>` tags.

    Intended usage: call `feed(chunk)` with each streamed text fragment; it
    returns the portion that should be visible downstream. `is_thinking`
    reports whether we're currently inside a think block.
    """

    thinking_timeout_s: float = 60.0
    open_tag: str = "<think>"
    close_tag: str = "</think>"
    state: _State = _State.BEFORE_THINK
    _buffer: str = ""                      # unemitted text being inspected for tags
    _think_start_time: float | None = None
    _think_end_time: float | None = None
    _thinking_time_s: float = 0.0
    _thinking_text: str = ""
    _visible_text: str = ""

    def feed(self, chunk: str) -> str:
        """Consume `chunk`; return the visible portion to forward downstream.

        Non-visible portions (inside <think>...</think>) are retained only
        in the thinking counters.
        """
        now = time.monotonic()
        # Append to buffer for tag scanning.
        self._buffer += chunk
        output_parts: list[str] = []

        while self._buffer:
            if self.state is _State.BEFORE_THINK:
                idx = self._buffer.find(self.open_tag)
                if idx == -1:
                    # If the tail could be the start of an open tag, hold it.
                    hold = self._partial_tail(self.open_tag)
                    out = self._buffer[: len(self._buffer) - hold]
                    self._buffer = self._buffer[len(self._buffer) - hold:]
                    if out:
                        self._visible_text += out
                        output_parts.append(out)
                    break
                # Emit up to the `<think>`, then transition.
                out = self._buffer[:idx]
                if out:
                    self._visible_text += out
                    output_parts.append(out)
                self._buffer = self._buffer[idx + len(self.open_tag):]
                self.state = _State.IN_THINK
                self._think_start_time = now
            elif self.state is _State.IN_THINK:
                idx = self._buffer.find(self.close_tag)
                if idx == -1:
                    # Retain whatever might be a partial close tag; buffer rest as thinking.
                    hold = self._partial_tail(self.close_tag)
                    consumed = self._buffer[: len(self._buffer) - hold]
                    self._buffer = self._buffer[len(self._buffer) - hold:]
                    self._thinking_text += consumed
                    # check timeout
                    if self._think_start_time is not None and (now - self._think_start_time) >= self.thinking_timeout_s:
                        # Do not raise — caller checks `thinking_timed_out`
                        pass
                    break
                self._thinking_text += self._buffer[:idx]
                self._buffer = self._buffer[idx + len(self.close_tag):]
                self._think_end_time = now
                if self._think_start_time is not None:
                    self._thinking_time_s += (self._think_end_time - self._think_start_time)
                self.state = _State.AFTER_THINK
            else:  # AFTER_THINK: everything flows through
                out = self._buffer
                self._buffer = ""
                self._visible_text += out
                output_parts.append(out)

        return "".join(output_parts)

    def _partial_tail(self, tag: str) -> int:
        """Return the length of the buffer's suffix that could be a prefix of `tag`.

        If buffer ends with a full `tag`, we'd already have matched it. This
        covers partial matches like '<thi' where we can't yet tell whether
        the next chunk will complete the tag.
        """
        # Check suffix lengths from len(tag)-1 down to 1
        max_len = min(len(tag) - 1, len(self._buffer))
        for k in range(max_len, 0, -1):
            if self._buffer.endswith(tag[:k]):
                return k
        return 0

    @property
    def is_thinking(self) -> bool:
        return self.state is _State.IN_THINK

    @property
    def thinking_timed_out(self) -> bool:
        if self._think_start_time is None or self.state is _State.AFTER_THINK:
            return False
        return (time.monotonic() - self._think_start_time) >= self.thinking_timeout_s

    @property
    def thinking_time_s(self) -> float:
        if self.state is _State.IN_THINK and self._think_start_time is not None:
            return self._thinking_time_s + (time.monotonic() - self._think_start_time)
        return self._thinking_time_s

    @property
    def visible_text(self) -> str:
        return self._visible_text

    @property
    def thinking_text(self) -> str:
        return self._thinking_text


def rollback_instruction(mode: Mode, error_summary: str) -> str:
    """Produce the rollback-prompt suffix for either T-A or T-B.

    For T-A: explicitly ask the model to think again.
    For T-B: instruct direct continuation without thinking.
    """
    if mode is Mode.T_A:
        return (
            f"\n// Error from previous attempt: {error_summary}\n"
            f"// Reconsider the design, then continue from the cursor.\n"
        )
    else:  # T_B
        return (
            f"\n// Error from previous attempt: {error_summary}\n"
            f"// Directly continue writing Rust from the cursor; do not re-think.\n"
        )
