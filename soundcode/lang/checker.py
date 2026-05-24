"""Checker protocol — language-agnostic surface over compiler/LSP backends.
Phase 0: extracted from `CargoChecker` (sync, one-shot subprocess per call)
and `RustAnalyzerChecker` (async, long-running LSP server with start/stop)
so per-language implementations in Phase 1+ can drop in without touching
the orchestration loop.

The two existing implementations expose slightly different lifecycles:
  - sync (CargoChecker): no start/stop needed; every `check` spawns a fresh
    `cargo check` subprocess.
  - async (RustAnalyzerChecker): explicit `start()` (boots a thread + LSP
    process) and `stop()` (joins/teardown). Required because the LSP
    process is shared across calls.
We unify them under a single Protocol with no-op default lifecycle methods.
Implementations override `start`/`stop` only if they actually need them.
The `is_async` flag is a label used by callers/UI; orchestration code never
branches on it — it just calls `check`/`start`/`stop` uniformly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from soundcode.code import Diagnostic


@runtime_checkable
class Checker(Protocol):
    """A code checker that returns a list of Diagnostics for a source string.

    Lifecycle:
      start(): boot any persistent backend (LSP process, daemon, etc.).
               No-op for one-shot checkers like cargo. Called from a sync
               context (web server uses `asyncio.to_thread`).
      stop():  tear down the backend. Mirror of start.
      check(source): the actual diagnostic call — async because both
               existing implementations are async (one awaits a subprocess,
               the other awaits an LSP notification).

    `is_async` is a self-description label: True = stateful long-running
    backend that benefits from explicit start/stop (the LSP path); False =
    one-shot per-call backend (the cargo path). Phase 0 doesn't branch on
    this — Phase 1+ might use it to wire up UI badges or routing.
    """

    is_async: bool

    def start(self) -> None: ...
    def stop(self) -> None: ...
    async def check(self, source: str) -> list[Diagnostic]: ...
