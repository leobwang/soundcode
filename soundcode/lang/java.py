"""Java implementations of the `Checker` / `BoundaryDetector` / `Workspace`
Protocols. Multilingual Phase 1 — parallels `soundcode/lang/rust.py`.

Two checker variants:
  - `JavacChecker`: sync, one-shot `javac` subprocess per call (analogous
    to `RustCargoChecker`). Parses the `<file>:<line>: <severity>: <msg>`
    diagnostic format that `javac` writes to stderr.
  - `JdtLspChecker`: async, persistent Eclipse JDT.LS via multilspy
    (analogous to `RustAnalyzerLspChecker`). Boots in a background thread
    so the sync `start()`/`stop()` lifecycle matches the protocol.

`JavaBoundaryDetector` reuses the Rust boundary detector — `;` and `}`
outside strings/comments/parens/brackets have identical semantics in
Java. Java does have a few constructs (annotations, lambdas with `->`)
that the Rust scanner doesn't recognize specially, but none of them
introduce new boundary characters or alter the suppression rules, so
sharing the implementation is safe. Re-implementing would be pure
duplication.

`JavaWorkspace` materializes a `Main.java` skeleton in a temp directory.
Unlike Rust's cargo workspace (persistent on disk for incremental
caching), the Java workspace defaults to a transient `tempfile.mkdtemp()`
location and `teardown()` removes it — javac has no per-workspace cache
that benefits from persistence at this granularity.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from soundcode.code import Category, Diagnostic
from soundcode.eval.boundary import find_boundaries as _shared_find_boundaries


# ─── JavacChecker (sync, one-shot) ────────────────────────────────────


@dataclass
class JavacChecker:
    """`javac`-based Java checker. Synchronous lifecycle — `start`/`stop` are
    no-ops; every `check()` writes `source` to `Main.java` in `workspace`,
    spawns `javac -Xlint:none -d <tmp> Main.java`, and parses stderr.

    The compiled `.class` files are written into a throw-away temp dir per
    call so they don't accumulate (and so concurrent checks on different
    workspaces don't collide via shared output dirs)."""

    workspace: Path
    file_in_workspace: str = "Main.java"
    timeout_s: float = 30.0
    is_async: bool = False
    _lock: asyncio.Lock | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def check(self, source: str) -> list[Diagnostic]:
        """Write `source` to `Main.java` and run `javac`."""
        assert self._lock is not None
        async with self._lock:
            target = self.workspace / self.file_in_workspace
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
            out_dir = tempfile.mkdtemp(prefix="javac_out_", dir=str(self.workspace))
            try:
                proc = await asyncio.create_subprocess_exec(
                    "javac",
                    "-Xlint:none",
                    "-d", out_dir,
                    str(target),
                    cwd=str(self.workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ},
                )
                try:
                    _stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.timeout_s,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return []
            finally:
                shutil.rmtree(out_dir, ignore_errors=True)

        return _parse_javac_output(stderr.decode("utf-8", errors="replace"))


# javac stderr lines look like:
#   <file>:<line>: error: <message>
#   <file>:<line>: warning: <message>
#   <file>:<line>: note: <message>
# followed by a source-line echo and a caret line ("    ^") whose
# leading-space count gives us the column (1-indexed). The summary
# trailer ("N error(s)" / "N warning(s)") is ignored.
_DIAG_HEADER_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):\s+(?P<sev>error|warning|note):\s+(?P<msg>.*)$"
)


def _parse_javac_output(stderr: str) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    lines = stderr.splitlines()
    i = 0
    while i < len(lines):
        m = _DIAG_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        sev = m.group("sev")
        if sev == "error":
            cat = Category.BLOCKING
        elif sev == "warning":
            cat = Category.NON_BLOCKING
        else:
            # "note:" lines are informational — skip, matching cargo's
            # treatment of `note`/`help`.
            i += 1
            continue
        try:
            line_no = int(m.group("line"))
        except ValueError:
            line_no = None
        # Look ahead for a caret-bearing line — it's typically lines[i+2]
        # (after the source-line echo), but be lenient: scan forward at
        # most 3 non-header lines for a "<spaces>^" pattern.
        col: int | None = None
        j = i + 1
        while j < len(lines) and j <= i + 3:
            if _DIAG_HEADER_RE.match(lines[j]):
                break
            stripped = lines[j].rstrip()
            if stripped and stripped.lstrip(" \t") == "^":
                # column = 1-indexed position of `^`
                col = len(lines[j]) - len(lines[j].lstrip(" \t")) + 1
                j += 1
                break
            j += 1
        out.append(Diagnostic(
            category=cat,
            message=m.group("msg"),
            code=None,
            line=line_no,
            column=col,
        ))
        i = j
    return out


# ─── JdtLspChecker (async, persistent LSP) ────────────────────────────


@dataclass
class JdtLspChecker:
    """Eclipse JDT.LS-based Java checker via multilspy. Mirrors
    `RustAnalyzerChecker` (in `soundcode/ra_check.py`) and its wrapper
    `RustAnalyzerLspChecker`: a background thread owns an asyncio loop
    that drives the long-running LSP process; the sync `start()`/`stop()`
    lifecycle blocks until the server is ready / torn down.

    First-time use will trigger multilspy to download Gradle, the
    vscode-java JRE+JDTLS bundle, and intellicode — total ~hundreds of
    MB. Subsequent runs reuse the cache under `~/.multilspy/`.
    """

    workspace: Path
    file_in_workspace: str = "Main.java"
    settle_s: float = 3.0
    is_async: bool = True
    _lock: asyncio.Lock | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from multilspy.multilspy_config import MultilspyConfig
        from multilspy.multilspy_logger import MultilspyLogger

        self._lock = asyncio.Lock()
        self._config = MultilspyConfig.from_dict({"code_language": "java"})
        self._logger = MultilspyLogger()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lsp = None
        self._lsp_ctx = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._start_error: BaseException | None = None
        self._captured: dict[str, list[dict]] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._stop.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        # JDT.LS startup is heavier than rust-analyzer (JVM boot + workspace
        # indexing). 120s covers a warm start; cold first-run that has to
        # download the bundle will exceed this — callers should warm the
        # multilspy cache out of band.
        if not self._ready.wait(timeout=120.0):
            if self._start_error is not None:
                raise RuntimeError(
                    f"Eclipse JDT.LS failed to start: {self._start_error!r}"
                )
            raise RuntimeError("Eclipse JDT.LS failed to start within 120s")
        if self._start_error is not None:
            raise RuntimeError(
                f"Eclipse JDT.LS failed to start: {self._start_error!r}"
            )

    def _thread_main(self) -> None:
        from multilspy import LanguageServer
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _start():
            self._lsp = LanguageServer.create(
                self._config, self._logger, str(self.workspace)
            )
            self._lsp_ctx = self._lsp.start_server()
            await self._lsp_ctx.__aenter__()
            # Intercept publishDiagnostics — same hook as the Rust path.
            self._lsp.server.on_notification(
                "textDocument/publishDiagnostics",
                lambda params: self._captured.__setitem__(
                    params["uri"], params.get("diagnostics", [])
                ),
            )
            self._ready.set()

        try:
            loop.run_until_complete(_start())
        except BaseException as e:
            self._start_error = e
            self._ready.set()  # unblock waiter
            loop.close()
            return
        try:
            loop.run_until_complete(self._wait_until_stop())
        finally:
            try:
                if self._lsp_ctx is not None:
                    loop.run_until_complete(
                        self._lsp_ctx.__aexit__(None, None, None)
                    )
            except Exception:
                pass
            loop.close()

    async def _wait_until_stop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    async def check(self, source: str) -> list[Diagnostic]:
        """Write `source` to Main.java, wait for JDT.LS to publish diagnostics."""
        if self._lsp is None:
            return []
        assert self._lock is not None
        async with self._lock:
            target = self.workspace / self.file_in_workspace
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
            uri = target.as_uri()
            self._captured.pop(uri, None)

            fut = asyncio.run_coroutine_threadsafe(
                self._touch_and_wait(uri), self._loop,
            )
            try:
                diags = fut.result(timeout=self.settle_s + 5.0)
            except Exception:
                diags = []

        return _convert_lsp(diags)

    async def _touch_and_wait(self, uri: str) -> list[dict]:
        try:
            with self._lsp.open_file(self.file_in_workspace):
                await asyncio.sleep(self.settle_s)
                return self._captured.get(uri, [])
        except Exception:
            return []


def _convert_lsp(raw: list[dict]) -> list[Diagnostic]:
    """Convert LSP diagnostic dicts to `Diagnostic`. Same mapping as the
    Rust path — severity 1 (Error) → BLOCKING, everything else → NON_BLOCKING."""
    out: list[Diagnostic] = []
    for d in raw:
        sev = d.get("severity")
        code = d.get("code")
        if isinstance(code, dict):
            code = code.get("value") or code.get("code") or code
        if not isinstance(code, str):
            code = str(code) if code is not None else None
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


# ─── JavaBoundaryDetector ─────────────────────────────────────────────


class JavaBoundaryDetector:
    """Java boundary detector. Reuses
    `soundcode.eval.boundary.find_boundaries`: depth-0 `;` or `}` outside
    strings / chars / comments / parens / brackets — semantically identical
    to Rust for the purposes of statement/block termination.

    Quirks of Java that the shared scanner does the right thing on:
      - `//` line comments and `/* … */` block comments: handled.
      - String literals `"…"`: handled (Java escape rules are a strict
        subset of Rust's for the boundary-suppression purpose).
      - Char literals `'x'`: handled (Java doesn't have Rust lifetimes,
        but the scanner's char-vs-lifetime heuristic still accepts valid
        Java chars).
      - Lambdas `(x, y) -> { … }`: `->` is two ordinary tokens; the `{`
        opens a block whose `}` is a depth-0 boundary, which is the right
        behaviour (the body of a lambda is a statement-level chunk).
      - Text blocks (`\"\"\" … \"\"\"`): NOT specially handled — the scanner
        treats the opening `\"\"\"` as three empty/one-char strings.
        Boundaries are still detected correctly in surrounding code; the
        only risk is mis-classifying a `;` or `}` inside the text block,
        which is acceptable for Phase 1 (text blocks are uncommon in the
        synthetic code the LM produces for HumanEval-style problems).
    """

    def find_boundaries(self, source: str) -> list[int]:
        return [b.offset for b in _shared_find_boundaries(source)]


# ─── JavaWorkspace ────────────────────────────────────────────────────


_MAIN_JAVA_SKELETON = (
    "public class Main {\n"
    "    public static void main(String[] args) {\n"
    "    }\n"
    "}\n"
)


@dataclass
class JavaWorkspace:
    """Transient Java workspace: a directory containing `Main.java`.

    `path` is optional — when omitted (the default), `setup()` allocates
    a fresh `tempfile.mkdtemp()`. `teardown()` removes the workspace
    directory tree if (and only if) we own it (i.e., it was allocated by
    `setup()`). An externally-provided `path` is left in place on
    `teardown()` so callers that point at a persistent location aren't
    surprised by data loss."""

    path: Path | None = None
    _owned: bool = field(default=False, init=False, repr=False)

    def setup(self) -> Path:
        if self.path is None:
            self.path = Path(tempfile.mkdtemp(prefix="soundcode_java_"))
            self._owned = True
        self.path.mkdir(parents=True, exist_ok=True)
        main_java = self.path / "Main.java"
        # Idempotent: only write if missing or different — repeated
        # setup() calls on the same workspace don't perturb LSP state.
        if not main_java.exists() or main_java.read_text() != _MAIN_JAVA_SKELETON:
            main_java.write_text(_MAIN_JAVA_SKELETON)
        return self.path

    def teardown(self) -> None:
        if self._owned and self.path is not None and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
            self.path = None
            self._owned = False
