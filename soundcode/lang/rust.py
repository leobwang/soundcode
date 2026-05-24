"""Rust implementations of the `Checker` / `BoundaryDetector` / `Workspace`
Protocols. Phase 0 keeps the original Rust modules
(`soundcode/cargo_check.py`, `soundcode/ra_check.py`,
`soundcode/eval/boundary.py`) in place and wraps them here — so all
existing Rust tests still exercise the same code paths, and the wrappers
are pure plumbing.

Two checker variants are exported:
  - `RustCargoChecker`: sync, one-shot subprocess per call (wraps CargoChecker).
  - `RustAnalyzerLspChecker`: async, persistent LSP backend (wraps RustAnalyzerChecker).
The web server picks one based on the `verifier` query param ("cargo" / "ra").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from soundcode.cargo_check import CargoChecker
from soundcode.code import Diagnostic
from soundcode.eval.boundary import find_boundaries as _rust_find_boundaries
from soundcode.ra_check import RustAnalyzerChecker


@dataclass
class RustCargoChecker:
    """Cargo-based Rust checker. Wraps `CargoChecker`. Sync lifecycle —
    `start`/`stop` are no-ops; every `check` spawns a fresh `cargo check`
    subprocess."""

    workspace: Path
    is_async: bool = False
    _inner: CargoChecker | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._inner = CargoChecker(workspace=self.workspace)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def check(self, source: str) -> list[Diagnostic]:
        assert self._inner is not None
        return await self._inner.check(source)


@dataclass
class RustAnalyzerLspChecker:
    """rust-analyzer-based Rust checker. Wraps `RustAnalyzerChecker`. Has a
    real start/stop lifecycle — the LSP process lives in a background
    thread shared across calls."""

    workspace: Path
    settle_s: float = 1.5
    is_async: bool = True
    _inner: RustAnalyzerChecker | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._inner = RustAnalyzerChecker(workspace=self.workspace, settle_s=self.settle_s)

    def start(self) -> None:
        assert self._inner is not None
        self._inner.start()

    def stop(self) -> None:
        assert self._inner is not None
        self._inner.stop()

    async def check(self, source: str) -> list[Diagnostic]:
        assert self._inner is not None
        return await self._inner.check(source)


class RustBoundaryDetector:
    """Rust boundary detector. Wraps `soundcode.eval.boundary.find_boundaries`
    to satisfy the `BoundaryDetector` protocol — returns offsets only
    (drops the `kind` field), matching the protocol's flat-int signature."""

    def find_boundaries(self, source: str) -> list[int]:
        return [b.offset for b in _rust_find_boundaries(source)]


@dataclass
class RustWorkspace:
    """Rust cargo-project workspace. Lifts the inline `_ensure_workspace()`
    from `soundcode/web/server.py`. `setup()` scaffolds Cargo.toml +
    src/main.rs idempotently and returns the workspace path. `teardown()`
    is a no-op — the demo workspace is persistent so cargo's incremental
    cache survives across runs (warm cargo check is ~300ms vs ~3s cold)."""

    path: Path

    def setup(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "Cargo.toml").write_text(
            '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n\n'
            '[[bin]]\nname = "demo"\npath = "src/main.rs"\n'
        )
        (self.path / "src").mkdir(exist_ok=True)
        (self.path / "src" / "main.rs").write_text("fn main() {}\n")
        return self.path

    def teardown(self) -> None:
        pass
