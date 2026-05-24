"""Tests for `soundcode/lang/cpp.py` — C++ language pack (Phase 2).

Each test uses `tempfile.mkdtemp()` for isolation. The clangd LSP test is
skipped automatically if `clangd` is not on `PATH`.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from soundcode.code import Category
from soundcode.lang.boundary import BoundaryDetector
from soundcode.lang.checker import Checker
from soundcode.lang.cpp import (
    ClangdLspChecker,
    CppBoundaryDetector,
    CppWorkspace,
    GccChecker,
)
from soundcode.lang.workspace import Workspace


# ─── GccChecker ────────────────────────────────────────────────────────


def test_gcc_checker_clean_program_passes():
    """A trivially correct program must produce zero BLOCKING diagnostics."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not installed")
    tmp = Path(tempfile.mkdtemp(prefix="soundcode_cpp_"))
    try:
        ws = CppWorkspace(path=tmp)
        ws.setup()
        chk = GccChecker(workspace=tmp)
        assert isinstance(chk, Checker)
        diags = asyncio.run(chk.check("int main() { int x = 1; return x; }\n"))
        blocking = [d for d in diags if d.category is Category.BLOCKING]
        assert blocking == [], f"unexpected blocking diags: {blocking}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gcc_checker_syntax_error_fails():
    """An obvious syntax error (missing `;`) must surface as BLOCKING."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not installed")
    tmp = Path(tempfile.mkdtemp(prefix="soundcode_cpp_"))
    try:
        ws = CppWorkspace(path=tmp)
        ws.setup()
        chk = GccChecker(workspace=tmp)
        diags = asyncio.run(chk.check(
            "int main() { int x = 1 return x; }\n"  # missing `;` after 1
        ))
        blocking = [d for d in diags if d.category is Category.BLOCKING]
        assert len(blocking) >= 1, f"expected blocking diags, got: {diags}"
        # Every parsed diag should carry line/col info.
        for d in blocking:
            assert d.line is not None and d.line >= 1
            assert d.column is not None and d.column >= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gcc_checker_skips_if_no_gcc():
    """If the configured compiler binary is missing from PATH, `check`
    returns `[]` rather than raising — same defensive behaviour as the
    rust-analyzer wrapper when the LSP process dies."""
    tmp = Path(tempfile.mkdtemp(prefix="soundcode_cpp_"))
    try:
        ws = CppWorkspace(path=tmp)
        ws.setup()
        chk = GccChecker(workspace=tmp, compiler="definitely-not-a-real-compiler-xyz")
        diags = asyncio.run(chk.check("int main() { return 0; }\n"))
        assert diags == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── CppBoundaryDetector ───────────────────────────────────────────────


def test_cpp_boundary_detector_finds_depth_0_semicolons():
    """`;` and `}` at depth 0 are boundaries; those inside `()`/`[]` are
    suppressed; string contents are skipped."""
    det = CppBoundaryDetector()
    assert isinstance(det, BoundaryDetector)
    src = 'int x = 1; std::string s = "; in } string"; if (a) { b; }'
    bs = det.find_boundaries(src)
    # Expected boundaries:
    #   after `int x = 1;`  → offset 10
    #   after the second `;` (post-string assignment) → just after that `;`
    #   after `b;` inside the if-block → offset of that `;` + 1
    #   after the closing `}` of the if-block → end
    assert 10 in bs, bs
    # Make sure the in-string `;` and `}` were NOT picked up.
    sem_in_string = src.index("; in")
    assert (sem_in_string + 1) not in bs
    brace_in_string = src.index("} string")
    assert (brace_in_string + 1) not in bs


def test_cpp_boundary_detector_handles_template_brackets():
    """`std::vector<int> v;` must produce exactly ONE boundary — after the
    trailing `;`. Angle brackets are NOT depth-tracked, so neither `<`
    nor `>` can spuriously suppress (or introduce) a boundary."""
    det = CppBoundaryDetector()
    src = "std::vector<int> v;"
    bs = det.find_boundaries(src)
    assert bs == [len(src)], bs

    # Nested templates: also a single boundary at the trailing `;`.
    src2 = "std::map<std::string, std::vector<int>> m;"
    bs2 = det.find_boundaries(src2)
    assert bs2 == [len(src2)], bs2

    # `<` used as a binary operator must not confuse the detector either.
    src3 = "bool b = a < c; bool d = e > f;"
    bs3 = det.find_boundaries(src3)
    assert len(bs3) == 2, bs3
    assert bs3[0] == src3.index("; bool") + 1
    assert bs3[1] == len(src3)


def test_cpp_boundary_detector_skips_block_and_line_comments():
    """`;` and `}` inside `//` and `/* */` comments must not register."""
    det = CppBoundaryDetector()
    src = (
        "int x = 1; // trailing ; and } here\n"
        "/* multi\n  line ; and } */\n"
        "int y = 2;\n"
    )
    bs = det.find_boundaries(src)
    # Two real `;` boundaries — after `1;` and after `2;`.
    assert len(bs) == 2, bs


# ─── CppWorkspace ──────────────────────────────────────────────────────


def test_cpp_workspace_setup_creates_main_cpp_and_compile_commands():
    """`setup()` is idempotent and writes the seed file + compile DB."""
    tmp = Path(tempfile.mkdtemp(prefix="soundcode_cpp_"))
    try:
        ws = CppWorkspace(path=tmp)
        assert isinstance(ws, Workspace)
        out = ws.setup()
        assert out == tmp
        seed = tmp / "main.cpp"
        cc = tmp / "compile_commands.json"
        assert seed.exists(), "main.cpp not created"
        assert "int main()" in seed.read_text()
        assert cc.exists(), "compile_commands.json not created"
        entries = json.loads(cc.read_text())
        assert isinstance(entries, list) and len(entries) == 1
        entry = entries[0]
        assert entry["file"].endswith("main.cpp")
        assert any("-std=c++17" in a for a in entry["arguments"])
        # Idempotent: a second setup must not raise.
        ws.setup()
        assert seed.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cpp_workspace_teardown_removes_directory():
    tmp = Path(tempfile.mkdtemp(prefix="soundcode_cpp_"))
    ws = CppWorkspace(path=tmp)
    ws.setup()
    assert tmp.exists()
    ws.teardown()
    assert not tmp.exists()


# ─── ClangdLspChecker ──────────────────────────────────────────────────


def test_clangd_lsp_checker_smoke():
    """End-to-end: start clangd, send a clean and a broken file, expect
    BLOCKING diagnostics on the latter. Skipped if clangd is missing."""
    if shutil.which("clangd") is None:
        pytest.skip("clangd not installed")
    tmp = Path(tempfile.mkdtemp(prefix="soundcode_cpp_"))
    try:
        ws = CppWorkspace(path=tmp)
        ws.setup()
        chk = ClangdLspChecker(workspace=tmp, settle_s=2.0)
        assert isinstance(chk, Checker)
        chk.start()
        try:
            clean = asyncio.run(chk.check(
                "int main() { int x = 1; return x; }\n"
            ))
            assert all(d.category is not Category.BLOCKING for d in clean), clean
            broken = asyncio.run(chk.check(
                "int main() { int x = 1 return x; }\n"  # missing `;`
            ))
            assert any(d.category is Category.BLOCKING for d in broken), broken
        finally:
            chk.stop()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
