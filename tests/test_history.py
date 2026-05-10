"""Tests for soundcode.eval.history."""

from __future__ import annotations

import lsprotocol.types as lsp_type

from soundcode.analyzer import Diagnostic, DiagnosticSeverity
from soundcode.eval.history import RollbackHistory


def _d(msg: str, code: str = "", line: int = 0) -> Diagnostic:
    return Diagnostic(
        message=msg,
        severity=DiagnosticSeverity.ERROR,
        range=lsp_type.Range(
            start=lsp_type.Position(line=line, character=0),
            end=lsp_type.Position(line=line, character=1),
        ),
        code=code,
        source="rust-analyzer",
    )


def test_empty_renders_empty() -> None:
    h = RollbackHistory()
    assert h.render_full() == ""
    assert h.render_compact() == ""


def test_single_entry() -> None:
    h = RollbackHistory()
    h.begin_attempt()  # attempt 1
    h.add(_d("expected `i32`, found `&str`", code="E0308", line=4))
    out = h.render_full()
    assert "Attempt 1" in out
    assert "E0308" in out
    assert "line 5" in out  # 0-indexed → 1-indexed
    assert "expected `i32`, found `&str`" in out


def test_multi_attempt_accumulation() -> None:
    h = RollbackHistory()
    h.begin_attempt()
    h.add(_d("err a", code="E0308"))
    h.begin_attempt()
    h.add(_d("err b", code="E0425"))
    h.begin_attempt()
    h.add(_d("err c", code="E0277"))
    assert len(h) == 3
    out = h.render_full()
    assert "Attempt 1" in out and "Attempt 2" in out and "Attempt 3" in out
    assert "E0308" in out and "E0425" in out and "E0277" in out


def test_render_compact_keeps_recent() -> None:
    h = RollbackHistory()
    for i in range(5):
        h.begin_attempt()
        h.add(_d(f"err{i}", code=f"E030{i}"))
    out = h.render_compact(keep_recent=2)
    assert "[Attempts 1–3]" in out
    # Recent entries should be full text
    assert "err3" in out and "err4" in out
    # Older counts in compact summary
    assert "E0300" in out and "E0301" in out and "E0302" in out


def test_render_compact_small_history_equals_full() -> None:
    h = RollbackHistory()
    for i in range(2):
        h.begin_attempt()
        h.add(_d(f"err{i}", code=f"E030{i}"))
    assert h.render_compact(keep_recent=3) == h.render_full()


def test_render_compact_groups_duplicates() -> None:
    h = RollbackHistory()
    for i in range(6):
        h.begin_attempt()
        # Alternate two codes
        h.add(_d(f"err{i}", code="E0308" if i % 2 == 0 else "E0425"))
    out = h.render_compact(keep_recent=2)
    # Older 4 entries: 2 E0308 + 2 E0425
    assert "E0308×2" in out
    assert "E0425×2" in out


def test_render_chooses_compact_when_budget_exceeded() -> None:
    h = RollbackHistory()
    # Enough long entries to exceed 90% of a small window.
    long_msg = "a" * 500
    for _ in range(20):
        h.begin_attempt()
        h.add(_d(long_msg, code="E0308"))
    rendered = h.render(
        context_window=1024,
        reserved_completion_tokens=128,
        prompt_so_far="",
    )
    # Should have compacted: look for the "[Attempts 1–N]:" marker
    assert "[Attempts 1" in rendered


def test_render_chooses_full_when_small() -> None:
    h = RollbackHistory()
    h.begin_attempt()
    h.add(_d("small error", code="E0308"))
    rendered = h.render(
        context_window=32768,
        reserved_completion_tokens=512,
        prompt_so_far="",
    )
    assert "[Attempts" not in rendered  # not compacted
    assert "Attempt 1" in rendered


def test_code_distribution() -> None:
    h = RollbackHistory()
    h.begin_attempt(); h.add(_d("a", code="E0308"))
    h.begin_attempt(); h.add(_d("b", code="E0425"))
    h.begin_attempt(); h.add(_d("c", code="E0308"))
    dist = h.code_distribution()
    assert dist == {"E0308": 2, "E0425": 1}


def test_multiline_message_trimmed() -> None:
    h = RollbackHistory()
    h.begin_attempt()
    h.add(_d("first line\nsecond line\nthird", code="E0308"))
    out = h.render_full()
    assert "first line" in out
    assert "second line" not in out
