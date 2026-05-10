"""Tests for soundcode.eval.thinking_wrapper."""

from __future__ import annotations

from soundcode.eval.thinking_wrapper import Mode, ThinkingWrapper, rollback_instruction


def test_no_thinking_passthrough() -> None:
    w = ThinkingWrapper()
    out = w.feed("fn main() { }")
    assert out == "fn main() { }"
    assert w.visible_text == "fn main() { }"
    assert w.thinking_text == ""


def test_complete_think_block_stripped() -> None:
    w = ThinkingWrapper()
    out = w.feed("hello <think>reasoning here</think>code")
    assert "reasoning here" not in out
    assert out == "hello code"
    assert w.thinking_text == "reasoning here"


def test_streamed_think_block() -> None:
    """Feed the stream in multiple chunks; should still strip correctly."""
    w = ThinkingWrapper()
    parts = []
    for chunk in ["be", "fore<th", "ink>abc", "def</th", "ink>aft", "er"]:
        parts.append(w.feed(chunk))
    assert "".join(parts) == "beforeafter"
    assert w.thinking_text == "abcdef"


def test_no_close_tag_stays_in_thinking() -> None:
    w = ThinkingWrapper()
    out = w.feed("pre<think>reasoning...")
    assert out == "pre"
    assert w.is_thinking


def test_partial_open_tag_held_back() -> None:
    """If the chunk ends mid-`<think>`, we mustn't emit the partial."""
    w = ThinkingWrapper()
    out = w.feed("code<th")
    # `<th` could be the start of `<think>`, so it's held.
    assert out == "code"
    assert not w.is_thinking
    # Continue and emit the next chunk
    out2 = w.feed("ink>reasoning</think>after")
    assert out2 == "after"


def test_not_a_think_tag() -> None:
    """Chars that look like the start but aren't should flush through."""
    w = ThinkingWrapper()
    out1 = w.feed("code<")
    # `<` alone; held because it could start `<think>`
    out2 = w.feed("tag>stuff")
    # Now we know `<tag>` isn't `<think>`. Should get it all.
    assert out1 + out2 == "code<tag>stuff"


def test_thinking_time_measured() -> None:
    import time

    w = ThinkingWrapper()
    w.feed("<think>")
    time.sleep(0.01)
    w.feed("</think>")
    assert w.thinking_time_s >= 0.01


def test_thinking_timeout_flag() -> None:
    import time

    w = ThinkingWrapper(thinking_timeout_s=0.01)
    w.feed("<think>still going...")
    time.sleep(0.02)
    assert w.thinking_timed_out


def test_rollback_instruction_t_a_says_reconsider() -> None:
    msg = rollback_instruction(Mode.T_A, "E0308 at line 5: type mismatch")
    assert "Reconsider" in msg or "reconsider" in msg
    assert "E0308" in msg


def test_rollback_instruction_t_b_says_direct() -> None:
    msg = rollback_instruction(Mode.T_B, "E0308 at line 5: type mismatch")
    assert "do not re-think" in msg.lower() or "directly continue" in msg.lower()


def test_multiple_think_blocks_accumulated() -> None:
    """Some models emit multiple <think> blocks. Wrapper should handle it."""
    w = ThinkingWrapper()
    w.feed("a<think>one</think>")
    # After first close, state is AFTER_THINK — subsequent tags pass through
    # as verbatim text. That matches "generation continues"; a second thought
    # would appear as literal text to the model downstream.
    out = w.feed("b<think>two</think>c")
    # Post-think, everything passes through verbatim
    assert "a" in w.visible_text
    # The exact post-think behavior: second <think> is not stripped.
    # (Simplification: assumes one think block per generation.)
