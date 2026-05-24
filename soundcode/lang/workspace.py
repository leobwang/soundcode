"""Workspace protocol — language-agnostic scaffolding for the on-disk
project that the Checker writes candidate code into.

Phase 0: Rust needs a `Cargo.toml` + `src/main.rs`. C++ will need a
`CMakeLists.txt` + `src/main.cpp` (or whatever the chosen build system
provides). Python may need nothing more than a single `.py` file.

The protocol is two methods, mirroring the existing inline
`_ensure_workspace()` (setup) and a teardown hook for impls that need to
clean transient state (Phase 0's RustWorkspace does not — it points at a
persistent directory under `cargo_workspaces/`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Workspace(Protocol):
    """Materialize an on-disk workspace usable by a `Checker`.

    `setup()` is idempotent: re-running it on an existing workspace must
    leave it in a valid state (the existing `_ensure_workspace()` uses
    `mkdir(parents=True, exist_ok=True)` and `write_text` for this reason
    — every server restart re-runs setup).
    """

    def setup(self) -> Path: ...
    def teardown(self) -> None: ...
