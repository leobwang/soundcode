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
    # Max wait for the publish stream to settle. Fresh analysis after a
    # reopen typically lands in ~2-3s; the quiet-grace heuristic in
    # _cycle_and_wait usually returns well before this deadline.
    settle_s: float = 6.0
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
        # uri -> (published document version | None, diagnostics)
        self._captured: dict[str, tuple[int | None, list[dict]]] = {}
        self._version = 0
        self._last_text = ""  # overlay content as of the last didChange

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
            # Intercept publishDiagnostics (multilspy installs a no-op
            # handler). Record the published document VERSION alongside the
            # diagnostics so check() can correlate a publish with the
            # didChange that triggered it — without this, a check can read
            # the re-published diagnostics of the PREVIOUS content (stale
            # off-by-one; rust-analyzer re-sends known diagnostics eagerly).
            self._lsp.server.on_notification(
                "textDocument/publishDiagnostics",
                lambda params: self._captured.__setitem__(
                    params["uri"],
                    (params.get("version"), params.get("diagnostics", [])),
                ),
            )
            # Open the document ONCE and keep it open for the checker's
            # lifetime. Content updates flow through multilspy's own
            # incremental-edit API (delete + insert), the only sync path
            # this rust-analyzer/multilspy combination reliably applies —
            # hand-rolled didOpen/didChange/didChangeWatchedFiles overlays
            # are silently ignored by the server (see git history).
            self._open_cm = self._lsp.open_file(self.file_in_workspace)
            self._open_cm.__enter__()
            self._ready.set()

        loop.run_until_complete(_start())
        try:
            loop.run_until_complete(self._wait_until_stop())
        finally:
            try:
                self._open_cm.__exit__(None, None, None)
            except Exception:
                pass
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
        """Replace the open document's content with `source` via multilspy's
        incremental-edit API, then await the version-matched publish."""
        if self._lsp is None:
            return []
        assert self._lock is not None
        async with self._lock:
            target = self.workspace / self.file_in_workspace
            target.write_text(source)  # keep disk in sync (post-hoc tools)
            uri = target.as_uri()

            fut = asyncio.run_coroutine_threadsafe(
                self._edit_and_wait(uri, source), self._loop,
            )
            try:
                diags = await asyncio.wrap_future(fut)
            except Exception:
                diags = []

        return _convert(diags)

    async def _edit_and_wait(self, uri: str, text: str) -> list[dict]:
        """Delete the whole buffer, insert `text` (both via multilspy's own
        versioned incremental didChange path — the only sync mechanism this
        server/client combination reliably applies), then wait up to
        `settle_s` for a publishDiagnostics whose version reaches the
        buffer's post-edit version. Falls back to the newest publish at the
        deadline if the server omits versions."""
        try:
            buf = self._lsp.open_file_buffers.get(uri)
            if buf is None:
                return []
            old = buf.contents
            if old:
                old_lines = old.split("\n")
                self._lsp.delete_text_between_positions(
                    self.file_in_workspace,
                    {"line": 0, "character": 0},
                    {"line": len(old_lines) - 1,
                     "character": len(old_lines[-1])},
                )
            self._lsp.insert_text_at_position(
                self.file_in_workspace, 0, 0, text)
            v = buf.version
            # didSave triggers rust-analyzer's flycheck (embedded cargo
            # check) — without it, rustc-class diagnostics (E0308 etc.) are
            # computed once at startup and never refresh, which was the
            # historic staleness of this driver. Disk was synced by check().
            self._lsp.server.notify.did_save_text_document({
                "textDocument": {"uri": uri},
                "text": text,
            })
            # After a save, r-a publishes a transient version-matched set
            # (native quick pass) followed by the flycheck-merged set a
            # moment later. Take the newest version-matched publish once
            # the stream has been quiet for 0.4s (or at the deadline).
            loop = asyncio.get_event_loop()
            deadline = loop.time() + self.settle_s
            matched: tuple[int | None, list[dict]] | None = None
            matched_at: float | None = None
            while loop.time() < deadline:
                got = self._captured.get(uri)
                if (got is not None and got[0] is not None
                        and got[0] >= v and got is not matched):
                    matched = got
                    matched_at = loop.time()
                if matched_at is not None and loop.time() - matched_at > 0.4:
                    break
                await asyncio.sleep(0.05)
            if matched is not None:
                return matched[1]
            got = self._captured.get(uri)
            return got[1] if got is not None else []
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
