"""Cross-language parity tests for the four language packs.

For each (lang, valid_program, invalid_program) triple, verifies:
1. The sync checker (subprocess- or compile()-based) accepts the valid one
   and rejects the invalid one with a parseable BLOCKING diagnostic.
2. The boundary detector emits at least N boundaries in the valid program.
3. The workspace's setup() / teardown() lifecycle works end-to-end.
4. A per-lang sync-checker timing is captured and printed (no assertion) so
   we have a per-commit record of where each language lives on the
   verifier-cost spectrum that motivates SoundCode's async-for-Rust pitch.

Note on the API: every `Checker.check` is async (the Protocol unifies sync
subprocess and async LSP backends behind a single `async def check` —
see `soundcode/lang/checker.py`). Tests therefore use `asyncio.run`.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from soundcode.code import Category, Diagnostic
from soundcode.lang import (
    CppBoundaryDetector,
    CppWorkspace,
    GccChecker,
    JavaBoundaryDetector,
    JavaWorkspace,
    JavacChecker,
    PythonBoundaryDetector,
    PythonCompileChecker,
    PythonWorkspace,
    RustBoundaryDetector,
    RustCargoChecker,
    RustWorkspace,
)


@dataclass(frozen=True)
class LangCase:
    name: str
    tool: str  # binary that must be on PATH (used for pytest.skip)
    make_checker: Callable[[Path], Any]
    boundary_cls: Any
    workspace_cls: Any
    valid_program: str
    invalid_program: str
    boundary_min: int


def _rust_checker(ws_path: Path) -> Any:
    return RustCargoChecker(workspace=ws_path)


def _java_checker(ws_path: Path) -> Any:
    return JavacChecker(workspace=ws_path)


def _cpp_checker(ws_path: Path) -> Any:
    return GccChecker(workspace=ws_path)


def _python_checker(_ws_path: Path) -> Any:
    # PythonCompileChecker is in-memory; ignores workspace path
    return PythonCompileChecker()


CASES: list[LangCase] = [
    LangCase(
        name="rust",
        tool="cargo",
        make_checker=_rust_checker,
        boundary_cls=RustBoundaryDetector,
        workspace_cls=RustWorkspace,
        valid_program="fn main() {\n    let x = 1;\n    let y = 2;\n    let _ = x + y;\n}\n",
        invalid_program='fn main() {\n    let x: i32 = "hello";\n}\n',
        boundary_min=2,
    ),
    LangCase(
        name="java",
        tool="javac",
        make_checker=_java_checker,
        boundary_cls=JavaBoundaryDetector,
        workspace_cls=JavaWorkspace,
        valid_program=(
            "public class Main {\n"
            "    public static void main(String[] args) {\n"
            "        int x = 1;\n"
            "        int y = 2;\n"
            "        System.out.println(x + y);\n"
            "    }\n"
            "}\n"
        ),
        invalid_program=(
            "public class Main {\n"
            "    public static void main(String[] args) {\n"
            '        int x = "not an int";\n'
            "    }\n"
            "}\n"
        ),
        boundary_min=2,
    ),
    LangCase(
        name="cpp",
        tool="g++",
        make_checker=_cpp_checker,
        boundary_cls=CppBoundaryDetector,
        workspace_cls=CppWorkspace,
        valid_program=(
            "int main() {\n"
            "    int x = 1;\n"
            "    int y = 2;\n"
            "    return x + y;\n"
            "}\n"
        ),
        invalid_program=(
            "int main() {\n"
            "    int x = ;\n"  # syntax error
            "    return 0;\n"
            "}\n"
        ),
        boundary_min=2,
    ),
    LangCase(
        name="python",
        tool="python3",  # always present
        make_checker=_python_checker,
        boundary_cls=PythonBoundaryDetector,
        workspace_cls=PythonWorkspace,
        valid_program=(
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "def main():\n"
            "    print(add(1, 2))\n"
        ),
        invalid_program=(
            "def add(a, b)\n"  # missing colon
            "    return a + b\n"
        ),
        boundary_min=1,
    ),
]


def _blocking(diags: list[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diags if d.category is Category.BLOCKING]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_sync_checker_accepts_valid(case: LangCase, tmp_path: Path):
    if shutil.which(case.tool) is None:
        pytest.skip(f"{case.tool} not in PATH")
    ws = case.workspace_cls(path=tmp_path / case.name)
    ws_path = ws.setup()
    try:
        checker = case.make_checker(ws_path)
        diags = asyncio.run(checker.check(case.valid_program))
        blockers = _blocking(diags)
        assert blockers == [], (
            f"{case.name}: valid program rejected — BLOCKING diagnostics: {blockers}"
        )
    finally:
        ws.teardown()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_sync_checker_rejects_invalid(case: LangCase, tmp_path: Path):
    if shutil.which(case.tool) is None:
        pytest.skip(f"{case.tool} not in PATH")
    ws = case.workspace_cls(path=tmp_path / case.name)
    ws_path = ws.setup()
    try:
        checker = case.make_checker(ws_path)
        diags = asyncio.run(checker.check(case.invalid_program))
        blockers = _blocking(diags)
        assert len(blockers) >= 1, (
            f"{case.name}: invalid program produced no BLOCKING diags "
            f"(all diags: {diags}) — can't drive a rollback"
        )
    finally:
        ws.teardown()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_boundary_detector_finds_at_least_min_boundaries(case: LangCase):
    detector = case.boundary_cls()
    boundaries = detector.find_boundaries(case.valid_program)
    assert len(boundaries) >= case.boundary_min, (
        f"{case.name}: expected ≥{case.boundary_min} boundaries in valid program, "
        f"got {len(boundaries)}: {boundaries}"
    )
    for b in boundaries:
        assert 0 <= b <= len(case.valid_program), (
            f"{case.name}: boundary {b} out of range [0, {len(case.valid_program)}]"
        )


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_workspace_lifecycle(case: LangCase, tmp_path: Path):
    ws = case.workspace_cls(path=tmp_path / case.name)
    ws_path = ws.setup()
    try:
        assert ws_path.exists() and ws_path.is_dir(), (
            f"{case.name}: workspace path {ws_path} does not exist"
        )
    finally:
        ws.teardown()


def test_verifier_cost_spectrum_record(capsys, tmp_path: Path):
    """Record per-language sync-checker latency. No assertion — just a
    per-commit record so reviewers can see the verifier-cost-spectrum
    framing isn't theoretical. SoundCode's "async pays off for Rust"
    pitch is grounded in these numbers staying ordered the way we expect:
    Python << C++ < Java < Rust.

    This test always passes; check stdout (run with `-s`) for the timings.
    """
    timings: list[tuple[str, float]] = []
    for case in CASES:
        if shutil.which(case.tool) is None:
            timings.append((case.name, float("nan")))
            continue
        ws = case.workspace_cls(path=tmp_path / case.name)
        ws_path = ws.setup()
        try:
            checker = case.make_checker(ws_path)
            # Warm-up call (excluded from timing)
            asyncio.run(checker.check(case.valid_program))
            t0 = time.perf_counter()
            asyncio.run(checker.check(case.valid_program))
            dt = time.perf_counter() - t0
            timings.append((case.name, dt * 1000.0))
        finally:
            ws.teardown()
    print("\n=== verifier-cost spectrum (ms, single-file valid program) ===")
    for name, ms in timings:
        marker = "n/a (tool missing)" if ms != ms else f"{ms:7.1f} ms"
        print(f"  {name:8s}  {marker}")
