"""rust-analyzer-based diagnostics verifier (alternative to cargo_check).

Wraps multilspy. Threaded async loop so the sync `check()` can be called
from the same producer-consumer loop that uses CargoChecker.

Same return shape as CargoChecker.check: list[Diagnostic].
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path

from soundcode.code import Category, Diagnostic


@dataclass
class RustAnalyzerChecker:
    workspace: Path
    file_in_workspace: str = "src/main.rs"
    settle_s: float = 2.0
    _lock: asyncio.Lock | None = None

    def __post_init__(self) -> None:
        from multilspy import LanguageServer
        from multilspy.multilspy_config import MultilspyConfig
        from multilspy.multilspy_logger import MultilspyLogger

        self._lock = asyncio.Lock()
        self._config = MultilspyConfig.from_dict({"code_language": "rust"})
        self._logger = MultilspyLogger()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lsp = None
        self._lsp_ctx = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._captured: dict[str, list[dict]] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            raise RuntimeError("rust-analyzer failed to start within 30s")

    def _thread_main(self) -> None:
        from multilspy import LanguageServer
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _start():
            self._lsp = LanguageServer.create(self._config, self._logger, str(self.workspace))
            self._lsp_ctx = self._lsp.start_server()
            await self._lsp_ctx.__aenter__()
            # Intercept publishDiagnostics (multilspy installs a no-op handler).
            self._lsp.server.on_notification(
                "textDocument/publishDiagnostics",
                lambda params: self._captured.__setitem__(
                    params["uri"], params.get("diagnostics", [])
                ),
            )
            self._ready.set()

        loop.run_until_complete(_start())
        try:
            loop.run_until_complete(self._wait_until_stop())
        finally:
            try:
                loop.run_until_complete(self._lsp_ctx.__aexit__(None, None, None))
            except Exception:
                pass
            loop.close()

    async def _wait_until_stop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    async def check(self, source: str) -> list[Diagnostic]:
        """Write `source` to src/main.rs, wait for rust-analyzer to publish."""
        if self._lsp is None:
            return []
        assert self._lock is not None
        async with self._lock:
            target = self.workspace / self.file_in_workspace
            target.write_text(source)
            uri = target.as_uri()
            self._captured.pop(uri, None)

            # Touch the file via open_file/close cycle so rust-analyzer re-analyzes.
            fut = asyncio.run_coroutine_threadsafe(
                self._touch_and_wait(uri), self._loop,
            )
            try:
                diags = fut.result(timeout=self.settle_s + 2.0)
            except Exception:
                diags = []

        return _convert(diags)

    async def _touch_and_wait(self, uri: str) -> list[dict]:
        try:
            with self._lsp.open_file("src/main.rs"):
                # Wait for analysis.
                await asyncio.sleep(self.settle_s)
                return self._captured.get(uri, [])
        except Exception:
            return []


_SEVERITY = {1: "ERROR", 2: "WARNING", 3: "INFO", 4: "HINT"}


def _convert(raw: list[dict]) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for d in raw:
        sev = d.get("severity")
        code = d.get("code")
        if isinstance(code, dict):
            code = code.get("value") or code.get("code") or code
        if not isinstance(code, str):
            code = str(code) if code is not None else None
        # Map severity. rust-analyzer:
        #   1 (Error) → BLOCKING by default (refined later by classifier)
        #   2 (Warning) / 3 (Info) / 4 (Hint) → NON_BLOCKING
        if sev == 1:
            category = Category.BLOCKING
        else:
            category = Category.NON_BLOCKING
        rng = d.get("range") or {}
        start = rng.get("start") or {}
        out.append(Diagnostic(
            category=category,
            message=d.get("message", ""),
            code=code,
            line=(start.get("line") or 0) + 1,
            column=(start.get("character") or 0) + 1,
        ))
    return out
