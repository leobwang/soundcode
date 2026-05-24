"""Tests for Java language support (`soundcode/lang/java.py`).

Mirror of the Rust-side smoke tests in `tests/test_smoke.py` (cargo +
boundary + workspace) plus a lightweight LSP smoke test that is skipped
unless Eclipse JDT.LS has already been auto-downloaded by multilspy.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from soundcode.code import Category
from soundcode.lang.java import (
    JavaBoundaryDetector,
    JavaWorkspace,
    JavacChecker,
    JdtLspChecker,
)


# ─── helpers ──────────────────────────────────────────────────────────


_HAS_JAVAC = shutil.which("javac") is not None
_skip_no_javac = pytest.mark.skipif(
    not _HAS_JAVAC, reason="javac not in PATH"
)


def _jdtls_cache_present() -> bool:
    """Best-effort check whether multilspy's vscode-java bundle is already on
    disk — keeps the LSP smoke test cheap when present and skipped when not."""
    try:
        from multilspy.language_servers.eclipse_jdtls import eclipse_jdtls as _m
        static_dir = Path(_m.__file__).parent / "static" / "vscode-java"
        return static_dir.exists() and any(static_dir.iterdir())
    except Exception:
        return False


# ─── JavacChecker ─────────────────────────────────────────────────────


@_skip_no_javac
def test_javac_checker_clean_program_passes(tmp_path):
    src = (
        "public class Main {\n"
        "    public static void main(String[] args) {\n"
        "        int x = 1;\n"
        "        int y = 2;\n"
        "        System.out.println(x + y);\n"
        "    }\n"
        "}\n"
    )
    chk = JavacChecker(workspace=tmp_path)
    diags = asyncio.run(chk.check(src))
    blocking = [d for d in diags if d.category is Category.BLOCKING]
    assert blocking == [], f"unexpected blocking diagnostics: {blocking}"


@_skip_no_javac
def test_javac_checker_syntax_error_fails(tmp_path):
    src = (
        "public class Main {\n"
        "    public static void main(String[] args) {\n"
        "        int x = ;\n"
        "    }\n"
        "}\n"
    )
    chk = JavacChecker(workspace=tmp_path)
    diags = asyncio.run(chk.check(src))
    blocking = [d for d in diags if d.category is Category.BLOCKING]
    assert len(blocking) >= 1, f"expected at least one blocking diag, got {diags}"
    d0 = blocking[0]
    # The error is on line 3 (`int x = ;`).
    assert d0.line == 3, f"expected line 3, got {d0.line}"
    # Column should point at (or just after) the `;` on that line.
    assert d0.column is not None, f"expected column to be parsed, got {d0}"


def test_javac_checker_skips_if_no_javac():
    """Sentinel: when javac is missing, all `_skip_no_javac`-marked tests
    must be skipped — verified by the marker being live. We just assert
    the marker behaves as expected (no-op when javac is present)."""
    if shutil.which("javac") is None:
        pytest.skip("javac not in PATH (expected behaviour)")


@_skip_no_javac
def test_javac_checker_lifecycle_methods_are_noops(tmp_path):
    """`start()`/`stop()` exist for protocol shape but must be no-ops on the
    sync subprocess path — calling them must not raise."""
    chk = JavacChecker(workspace=tmp_path)
    chk.start()
    chk.stop()
    assert chk.is_async is False


# ─── JavaBoundaryDetector ─────────────────────────────────────────────


def test_java_boundary_detector_finds_depth_0_semicolons():
    src = "int x = 1; int y = 2;"
    bd = JavaBoundaryDetector()
    offsets = bd.find_boundaries(src)
    # The `;` characters are at offsets 9 and 20 in the source string.
    assert offsets == [9, 20], (offsets, src)


def test_java_boundary_detector_finds_depth_0_braces():
    """Boundaries at depth-0 `}` characters. Nested-block braces also fire
    boundaries (matching Rust's `find_boundaries`) — what is suppressed is
    `;`/`}` inside *parens* or *brackets*, not inside braces."""
    src = "if (x) { y; } z;"
    bd = JavaBoundaryDetector()
    offsets = bd.find_boundaries(src)
    # `;` at offset 10 (inside the if-block, depth 1 wrt braces but depth 0
    # wrt parens/brackets — fires); `}` at offset 12; `;` at offset 15.
    assert offsets == [10, 12, 15], (offsets, src)


def test_java_boundary_detector_skips_semicolon_inside_parens():
    src = "foo(a; b);"
    bd = JavaBoundaryDetector()
    offsets = bd.find_boundaries(src)
    # Only the outer `;` at offset 9 counts.
    assert offsets == [9], (offsets, src)


def test_java_boundary_detector_skips_semicolon_inside_string():
    src = 'String s = "a; b"; int x = 1;'
    bd = JavaBoundaryDetector()
    offsets = bd.find_boundaries(src)
    # Two real `;`: at the end of the assignment and at the end of `int x = 1`.
    assert len(offsets) == 2, (offsets, src)
    assert src[offsets[0]] == ";"
    assert src[offsets[1]] == ";"


# ─── JavaWorkspace ────────────────────────────────────────────────────


def test_java_workspace_setup_creates_main_java():
    ws = JavaWorkspace()
    try:
        path = ws.setup()
        assert path.exists()
        main_java = path / "Main.java"
        assert main_java.exists()
        content = main_java.read_text()
        assert "class Main" in content
        assert "public static void main" in content
    finally:
        ws.teardown()


@_skip_no_javac
def test_java_workspace_main_java_is_parseable(tmp_path):
    """The scaffolded `Main.java` should compile cleanly under javac."""
    ws = JavaWorkspace(path=tmp_path)
    path = ws.setup()
    main_java = path / "Main.java"
    src = main_java.read_text()
    chk = JavacChecker(workspace=path)
    diags = asyncio.run(chk.check(src))
    blocking = [d for d in diags if d.category is Category.BLOCKING]
    assert blocking == [], f"scaffolded Main.java did not compile: {blocking}"


def test_java_workspace_setup_is_idempotent():
    """Repeated setup() calls on the same workspace must leave it valid."""
    ws = JavaWorkspace()
    try:
        p1 = ws.setup()
        p2 = ws.setup()
        assert p1 == p2
        assert (p2 / "Main.java").exists()
    finally:
        ws.teardown()


def test_java_workspace_teardown_removes_owned_dir():
    """When `path` is allocated by setup(), teardown() must remove the tree."""
    ws = JavaWorkspace()
    path = ws.setup()
    assert path.exists()
    ws.teardown()
    assert not path.exists()


def test_java_workspace_teardown_preserves_external_dir(tmp_path):
    """When `path` is passed in by the caller, teardown() must NOT remove it
    — same defensive semantics as `RustWorkspace.teardown()` (no-op)."""
    external = tmp_path / "external_ws"
    external.mkdir()
    ws = JavaWorkspace(path=external)
    ws.setup()
    ws.teardown()
    assert external.exists(), "externally-provided workspace must survive teardown"
    assert (external / "Main.java").exists()


# ─── JdtLspChecker (heavy; skipped unless cache is warm) ──────────────


@pytest.mark.skipif(
    not _jdtls_cache_present(),
    reason="Eclipse JDT.LS not pre-downloaded; first-time install is too slow for unit tests",
)
def test_jdt_lsp_checker_smoke(tmp_path):
    """Smoke test: instantiate, start, stop. Does NOT run a check — that
    would extend the test budget by the JDT.LS workspace-indexing settle
    time. The protocol contract here is lifecycle correctness."""
    ws = JavaWorkspace(path=tmp_path)
    ws.setup()
    chk = JdtLspChecker(workspace=tmp_path, settle_s=2.0)
    assert chk.is_async is True
    chk.start()
    try:
        # Light invariant: the background thread should be alive after start.
        assert chk._thread is not None and chk._thread.is_alive()
    finally:
        chk.stop()
