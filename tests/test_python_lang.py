"""Tests for `soundcode.lang.python` (Phase 3 multilingual: Python pack).

Three categories:
  - `PythonCompileChecker`: clean / SyntaxError / IndentationError
  - `PythonBoundaryDetector`: top-level def boundaries, suppression inside
    nested blocks, triple-quoted strings, open parens
  - `PythonWorkspace`: scaffold creates pyproject + src/main.py
  - `PyrightLspChecker`: smoke (skips if pyright-langserver is missing)
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from soundcode.code import Category
from soundcode.lang.python import (
    PyrightLspChecker,
    PythonBoundaryDetector,
    PythonCompileChecker,
    PythonWorkspace,
)


# ─── PythonCompileChecker ────────────────────────────────────────────────


def test_python_compile_checker_clean_program_passes():
    """A syntactically valid program produces zero diagnostics."""
    checker = PythonCompileChecker()
    diags = asyncio.run(checker.check("def f():\n    return 1\n"))
    assert diags == [], diags


def test_python_compile_checker_syntax_error_fails():
    """A SyntaxError produces one BLOCKING diagnostic with line/col."""
    checker = PythonCompileChecker()
    diags = asyncio.run(checker.check("def f(: return 1\n"))
    assert len(diags) == 1, diags
    d = diags[0]
    assert d.category is Category.BLOCKING
    assert d.code == "syntax-error"
    # The error is on line 1; column varies by Python minor version but
    # should be a positive int when set.
    assert d.line == 1, d
    if d.column is not None:
        assert d.column >= 1, d


def test_python_compile_checker_indentation_error_fails():
    """IndentationError (subclass of SyntaxError) is also caught."""
    checker = PythonCompileChecker()
    src = "def f():\n  return 1\n    return 2\n"
    diags = asyncio.run(checker.check(src))
    assert len(diags) == 1, diags
    d = diags[0]
    assert d.category is Category.BLOCKING
    assert d.code == "syntax-error"


# ─── PythonBoundaryDetector ──────────────────────────────────────────────


def test_python_boundary_detector_finds_top_level_statements():
    """Two top-level defs separated by a newline → one boundary after each."""
    src = "def f():\n    return 1\ndef g():\n    return 2\n"
    bds = PythonBoundaryDetector().find_boundaries(src)
    # Boundaries fire at the offset immediately after the newline whose
    # NEXT non-blank line is at indent 0. That's just before `def g`. There
    # may or may not be a boundary at the trailing newline (depends on EOF
    # handling — we don't fire at EOF, see docstring).
    starts = [src.find("def f"), src.find("def g")]
    # Every offset returned must land at the start of a top-level def.
    assert all(src[b:].startswith("def ") for b in bds), (bds, [src[b:b+6] for b in bds])
    # We expect at least the boundary BEFORE `def g`.
    assert src.find("def g") in bds, bds


def test_python_boundary_detector_does_not_fire_in_nested_blocks():
    """Body lines (indent > 0) are not boundaries even at the newlines that
    terminate them."""
    src = "def f():\n    x = 1\n    y = 2\n    return x + y\n"
    bds = PythonBoundaryDetector().find_boundaries(src)
    # None of the boundaries should land inside the function body — every
    # boundary must point at indent-0 content.
    for b in bds:
        # The text starting at the boundary should not begin with whitespace.
        if b < len(src):
            assert not src[b].isspace() or src[b] == "\n", (
                f"boundary at offset {b} points into indented body: {src[b:b+10]!r}"
            )


def test_python_boundary_detector_handles_multiline_strings():
    """A triple-quoted string interior with indent-0-looking lines does not
    fire a boundary; the boundary only fires AFTER the string closes."""
    src = (
        'def f():\n'
        '    s = """\n'
        'top\n'           # this WOULD look like indent-0 but is inside the string
        'level\n'         # ditto
        '"""\n'
        'def g():\n'      # real top-level statement
        '    return 1\n'
    )
    bds = PythonBoundaryDetector().find_boundaries(src)
    # No boundary should point inside the triple-string body.
    for b in bds:
        assert src[b:].startswith("def "), (
            f"boundary at offset {b} should be at a top-level def, but got: {src[b:b+10]!r}"
        )
    # The boundary BEFORE `def g` should be present.
    assert src.find("def g") in bds, bds


def test_python_boundary_detector_handles_open_parens():
    """Mid-call newlines (bracket depth > 0) do not fire boundaries."""
    src = "f(\n  x,\n  y,\n)\n"
    bds = PythonBoundaryDetector().find_boundaries(src)
    # No boundary should land between `f(` and `)`. The only candidate
    # boundary would be after the final `\n` — but that's EOF and we don't
    # fire on the final newline (no next line to check). So we expect [].
    # If a boundary IS returned, it must point at the very end (after `)\n`)
    # which would be len(src), and there's no statement after it.
    for b in bds:
        assert b >= src.index(")"), f"boundary at {b} fired inside parens"


def test_python_boundary_detector_handles_open_parens_then_top_level():
    """Sanity check: after the paren closes, a subsequent top-level statement
    DOES fire a boundary."""
    src = "f(\n  x,\n  y,\n)\ndef g():\n    return 1\n"
    bds = PythonBoundaryDetector().find_boundaries(src)
    assert src.find("def g") in bds, bds


# ─── PythonWorkspace ─────────────────────────────────────────────────────


def test_python_workspace_setup_creates_pyproject_and_src():
    """Workspace materializes a usable Python project layout."""
    ws = PythonWorkspace()
    path = ws.setup()
    try:
        assert path.is_dir()
        assert (path / "pyproject.toml").is_file()
        assert (path / "pyrightconfig.json").is_file()
        assert (path / "src" / "main.py").is_file()
        # Content sanity: pyproject has a project section.
        toml = (path / "pyproject.toml").read_text()
        assert "[project]" in toml
        assert "name" in toml
    finally:
        ws.teardown()
    # Owned temp dir is cleaned up.
    assert not path.exists(), f"teardown failed to remove {path}"


def test_python_workspace_setup_idempotent_on_external_path():
    """When given an explicit path, setup is idempotent and teardown is a
    no-op (matches RustWorkspace semantics for persistent dirs)."""
    with tempfile.TemporaryDirectory() as td:
        ws = PythonWorkspace(path=Path(td))
        p1 = ws.setup()
        p2 = ws.setup()
        assert p1 == p2 == Path(td)
        ws.teardown()
        # External path is preserved.
        assert Path(td).exists()


# ─── PyrightLspChecker (smoke; skip if pyright-langserver missing) ───────


def test_pyright_lsp_checker_smoke():
    """Start, run one check, stop. Skipped if pyright-langserver isn't
    discoverable on PATH (e.g., CI without Node)."""
    if shutil.which("pyright-langserver") is None:
        pytest.skip("pyright-langserver not on PATH")

    ws = PythonWorkspace()
    workspace_path = ws.setup()
    checker = PyrightLspChecker(workspace=workspace_path, settle_s=2.0)
    try:
        try:
            checker.start()
        except RuntimeError as e:
            pytest.skip(f"pyright-langserver failed to start: {e}")
        # Clean program → expect zero blocking diagnostics.
        diags = asyncio.run(checker.check("def f():\n    return 1\n"))
        # We don't assert == [] because Pyright may emit hints; but no
        # diagnostic should be BLOCKING.
        blocking = [d for d in diags if d.category is Category.BLOCKING]
        assert blocking == [], blocking
    finally:
        checker.stop()
        ws.teardown()
