"""Smoke tests for the new soundcode pipeline (run with `uv run pytest tests/test_smoke.py -xvs`)."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from soundcode.cargo_check import CargoChecker
from soundcode.code import Category, Code, State
from soundcode.eval.boundary import find_boundaries


# ─── boundary detection ────────────────────────────────────────────────

def test_boundary_simple_semicolons():
    src = "let x = 1; let y = 2;"
    b = find_boundaries(src)
    assert [bb.kind for bb in b] == [";", ";"]
    assert [bb.offset for bb in b] == [9, 20]


def test_boundary_string_with_semicolon():
    src = 'let x = ";"; let y = 1;'
    b = find_boundaries(src)
    # The `;` inside the string should not count.
    assert len(b) == 2  # two `;` outside strings
    assert b[0].offset == src.index(";", src.index('"')+2)


def test_boundary_inside_parens_suppressed():
    src = "foo(a; b);"  # the inner ; is inside parens
    b = find_boundaries(src)
    # Only the outer ; at offset 9 counts.
    assert [bb.kind for bb in b] == [";"]
    assert b[0].offset == 9


def test_boundary_raw_string_with_semi():
    src = 'let x = r#"a;b"#; let y = 0;'
    b = find_boundaries(src)
    # Two `;` outside the raw string.
    assert len(b) == 2


# ─── cargo check ───────────────────────────────────────────────────────


def _make_workspace(tmp: Path) -> Path:
    (tmp / "Cargo.toml").write_text(
        '[package]\nname = "scratch"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "scratch"\npath = "src/main.rs"\n'
    )
    (tmp / "src").mkdir()
    (tmp / "src" / "main.rs").write_text("fn main() {}\n")
    return tmp


@pytest.fixture
def workspace(tmp_path):
    return _make_workspace(tmp_path)


def test_cargo_check_clean(workspace):
    chk = CargoChecker(workspace=workspace)
    diags = asyncio.run(chk.check("fn main() { let x: i32 = 1; let _ = x; }"))
    blocking = [d for d in diags if d.category is Category.BLOCKING]
    assert len(blocking) == 0


def test_cargo_check_finds_type_mismatch(workspace):
    chk = CargoChecker(workspace=workspace)
    diags = asyncio.run(chk.check('fn main() { let x: i32 = "oops"; let _ = x; }'))
    blocking = [d for d in diags if d.category is Category.BLOCKING]
    assert len(blocking) >= 1
    assert any(d.code == "E0308" for d in blocking)


# ─── Code class end-to-end ────────────────────────────────────────────


def test_code_check_clean_creates_checkpoint(workspace):
    chk = CargoChecker(workspace=workspace)
    # Use a HumanEval-shaped prefix: function signature ending with `{`.
    # `Code.check` will append `\n}\n\nfn main() {}\n` automatically.
    code = Code(prefix="fn f(x: i32) -> i32 {", suffix="}\n", checker=chk)
    code.append(" let y: i32 = x + 1; y")
    res = asyncio.run(code.check())
    assert res.verdict in (State.OK, State.INACTIVE), f"got {res.verdict}: {res.diagnostics}"


def test_code_check_blocks_on_error(workspace):
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="fn f(x: i32) -> i32 {", suffix="}\n", checker=chk)
    code.append(' let y: i32 = "oops"; y')
    res = asyncio.run(code.check())
    assert res.verdict is State.ERROR
    assert any(d.is_blocking for d in res.diagnostics)


def test_check_result_message_dedupes_repeated_diagnostics():
    """CheckResult.message should collapse identical (code, message) pairs so
    rollback-instruct doesn't inject 3× the same E0434 line and trigger the
    prompt-echo trap (draft-report-0.md §5.3)."""
    from soundcode.code import CheckResult, Diagnostic, Category, State
    diags = [
        Diagnostic(category=Category.BLOCKING, code="E0434",
                   message="can't capture dynamic environment in a fn item"),
        Diagnostic(category=Category.BLOCKING, code="E0434",
                   message="can't capture dynamic environment in a fn item"),
        Diagnostic(category=Category.BLOCKING, code="E0434",
                   message="can't capture dynamic environment in a fn item"),
        Diagnostic(category=Category.BLOCKING, code="E0308",
                   message="mismatched types"),
        Diagnostic(category=Category.NON_BLOCKING, code="warn",
                   message="some warning"),  # ignored — not blocking
    ]
    result = CheckResult(verdict=State.ERROR, diagnostics=diags)
    lines = result.message.splitlines()
    assert len(lines) == 2, f"expected 2 unique blocking lines, got {len(lines)}: {lines}"
    assert any("E0434" in l for l in lines)
    assert any("E0308" in l for l in lines)


def test_body_closed_signals_function_close(workspace):
    """Code.body_closed should fire iff the LM has emitted more `}` than `{`
    — i.e. it has closed the function body whose opening `{` lives in the
    prompt."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="fn f() -> i32 {", suffix="}", checker=chk)
    code.append("    let x = 1;")          # balance == 0
    assert not code.body_closed
    code.append(" if x > 0 { x } else { -x }")  # balance still 0 (nested)
    assert not code.body_closed
    code.append("\n}")                     # closing the function body
    assert code.body_closed, code.content


def test_body_closed_ignores_braces_in_strings(workspace):
    """Closing braces inside string / char / comment literals must NOT
    contribute to the balance — otherwise the LM emitting `"}"` as part of
    its code would falsely terminate generation."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="fn f() {", suffix="}", checker=chk)
    code.append('    let s = "hello }";')   # `}` in string — ignored
    assert not code.body_closed
    code.append("    let c = '}';")         # `}` in char literal — ignored
    assert not code.body_closed
    code.append("    // foo } bar")         # `}` in comment — ignored
    assert not code.body_closed
    code.append("\n}")
    assert code.body_closed


def test_brace_balance_signed(workspace):
    """The signed balance helper should return negative when there are more
    `}` than `{` in the buffer."""
    from soundcode.code import _content_brace_balance
    assert _content_brace_balance("{ {") == 2
    assert _content_brace_balance("{ } {") == 1
    assert _content_brace_balance("} }") == -2
    assert _content_brace_balance("// }\n") == 0
    assert _content_brace_balance('"}"') == 0


def test_code_rollback_restores_buffer(workspace):
    """`rollback()` with no args returns the buffer to `ckpt[-1]`. With the
    new boundary-based checkpointing, `;` and `}` create checkpoints during
    `append()` — so we append a `;`-terminated chunk (boundary auto-ckpt at
    offset 10) and then a chunk with NO boundary (latest ckpt stays at 10),
    so the no-arg rollback restores the original."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("good text;")           # `;` boundary → ckpt at offset 10
    assert code.ckpt[-1] == 10
    code.append(" extra junk")           # no boundary → ckpt unchanged
    assert code.ckpt[-1] == 10
    code.rollback()
    assert code.content == "good text;"
    assert code.state is State.INACTIVE


def test_append_adds_checkpoint_at_semicolon(workspace):
    """Every `;` outside strings/parens/brackets must add a checkpoint at
    the offset right after the `;`."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("let x = 1;")
    assert code.ckpt == [0, 10]
    code.append(" let y = 2;")
    assert code.ckpt == [0, 10, 21]


def test_append_adds_checkpoint_at_closing_brace(workspace):
    """A `}` outside strings/comments also creates a checkpoint at the offset
    right after the `}` — even when nested inside another block."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("if x { y; }")
    # boundaries: `;` at offset 8 → ckpt 9; `}` at offset 10 → ckpt 11.
    assert code.ckpt == [0, 9, 11]


def test_append_skips_boundaries_in_strings(workspace):
    """`;`/`}` inside string literals must not create checkpoints."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append('let s = "; in } string";')
    # Only the trailing `;` is a real boundary.
    real_semi_offset = code.content.rfind(";")
    assert code.ckpt == [0, real_semi_offset + 1]


def test_append_skips_boundaries_inside_parens(workspace):
    """`;` inside `()` is suppressed (`find_boundaries` semantics)."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("foo(a; b);")
    # Only outer `;` at offset 9 → ckpt 10.
    assert code.ckpt == [0, 10]


def test_rollback_floor_resets_to_current_target(workspace):
    """After successive rollbacks, the auto-ckpt floor must reflect the
    CURRENT target — not the historical max. Otherwise the LM regenerating
    fresh content past a lower target gets its NEW boundaries silently
    skipped because they land below the historical floor."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    # Build content with boundaries every 10 chars.
    code.append("a; b; c; d; e;")   # ckpts at 2, 5, 8, 11, 14
    assert code.ckpt == [0, 2, 5, 8, 11, 14]
    code.rollback(to_offset=14)     # peel back the latest; floor = 14
    code.rollback(to_offset=11)     # then peel back another; floor must = 11 (not 14)
    code.rollback(to_offset=5)      # and another; floor must = 5
    # Now append fresh content whose boundary lands at offset 8 (BELOW the
    # historical max of 14). The auto-ckpt MUST add it — otherwise a
    # monotonic floor would block it.
    code.append(" x;")               # `;` at offset 7, ckpt at 8
    assert 8 in code.ckpt, f"new boundary in regenerated content was skipped: {code.ckpt}"


def test_rollback_floor_blocks_boundary_readdition(workspace):
    """Regression: after rolling back through a boundary, the next `append`'s
    auto-ckpt scan must NOT re-add that boundary as a checkpoint — otherwise
    the cargo-error rollback survivor (= largest ckpt < offset_at_check)
    keeps picking the same offset on every retry, and the producer spin-loops
    on one survivor until the wall budget kills it (observed in the qwen3.5:9b
    sudoku run with 588 / 43 same-offset rollbacks)."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("let x = 1;")              # ckpt at 10
    code.append(" let y = 2;")             # ckpt at 21
    code.append(" let z = 3;")             # ckpt at 32
    assert code.ckpt == [0, 10, 21, 32]
    # Rollback to 32 (latest boundary). The ckpt at 32 should be removed.
    code.rollback(to_offset=32)
    assert 32 not in code.ckpt
    # LM regenerates non-boundary text — even a re-scan of the unchanged
    # content[:32] (which still has `;` at offsets 9, 20, 31) MUST NOT
    # re-introduce ckpt at 32.
    code.append(" no")
    code.append("_boundary")
    assert 32 not in code.ckpt, f"ckpt at 32 was re-added: {code.ckpt}"
    # New boundary past the floor still gets a ckpt.
    code.append(";")
    assert code.ckpt[-1] == len(code.content)
    assert code.ckpt[-1] > 32
