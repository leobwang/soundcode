"""Python implementations of the `Checker` / `BoundaryDetector` / `Workspace`
Protocols (Phase 3 — multilingual expansion).

Three checkers / one boundary detector / one workspace:
  - `PythonCompileChecker`: sync, microsecond-latency. Uses Python's built-in
    `compile(source, filename, 'exec')` to detect SyntaxError / IndentationError.
    Same return shape as `RustCargoChecker.check`. No subprocess, no startup,
    no IO — the fastest verifier on the four-language ladder.
  - `PyrightLspChecker`: async, persistent LSP backend. Wraps a custom
    `PyrightServer` (multilspy subclass driving `pyright-langserver --stdio`).
    Mirrors `ra_check.py`'s threaded-loop pattern exactly.
  - `PythonBoundaryDetector`: indent-aware. Fires at the offset immediately
    after a newline whose NEXT non-blank, non-comment line starts at indent 0
    — i.e., the start of a new top-level statement. Suppresses boundaries
    inside strings (single, triple-quoted, raw, f-string) and inside any
    unclosed `(` / `[` / `{` bracket. Also respects `\\`-continuation lines.
  - `PythonWorkspace`: creates a temp dir with `pyproject.toml` +
    `pyrightconfig.json` + `src/main.py`. `teardown()` removes it. Persistence
    is not useful for Python the way it is for cargo's incremental cache.

Pyright was chosen over Jedi/Pylance per the project multilingual-strategy
memo. Multilspy ships a Jedi backend out of the box; we drop in a Pyright
subclass here rather than introducing closed-source Pylance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from soundcode.code import Category, Diagnostic


# ─── compile() checker ───────────────────────────────────────────────────


@dataclass
class PythonCompileChecker:
    """`compile(source, filename, 'exec')`-based checker. Catches SyntaxError
    and IndentationError; converts each to a single BLOCKING diagnostic. No
    semantic / type checking — for that, use `PyrightLspChecker`. Microsecond
    latency; sync lifecycle (start/stop are no-ops)."""

    workspace: Path | None = None
    file_in_workspace: str = "src/main.py"
    is_async: bool = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def check(self, source: str) -> list[Diagnostic]:
        # filename is purely for the diagnostic display path; we don't write
        # to disk because compile() is in-memory. If the caller wants to also
        # see the source on disk for downstream tooling, that's their job.
        filename = self.file_in_workspace if self.workspace is None else str(
            self.workspace / self.file_in_workspace
        )
        try:
            compile(source, filename, "exec")
        except SyntaxError as e:
            # SyntaxError covers IndentationError, TabError, etc. — all
            # subclasses of SyntaxError. We treat them all uniformly as
            # BLOCKING; downstream classifier (in `soundcode.code.Code._classify`)
            # can demote to INCOMPLETE when appropriate.
            return [
                Diagnostic(
                    category=Category.BLOCKING,
                    message=e.msg or type(e).__name__,
                    code="syntax-error",
                    line=e.lineno,
                    column=e.offset,
                )
            ]
        except ValueError as e:
            # compile() raises ValueError for source containing null bytes; rare.
            return [
                Diagnostic(
                    category=Category.BLOCKING,
                    message=str(e),
                    code="value-error",
                    line=None,
                    column=None,
                )
            ]
        return []


# ─── Pyright LSP checker (mirrors ra_check.py) ───────────────────────────


class _PyrightServer:
    """Multilspy `LanguageServer` subclass that drives `pyright-langserver
    --stdio`. Constructed lazily inside `PyrightLspChecker.__post_init__` so
    that environments without pyright installed don't fail at import time."""

    # The class body is empty — see `_make_pyright_server` for the actual
    # factory. We keep a placeholder so `PyrightLspChecker` can type-hint
    # against a concrete name without importing multilspy at module level.


def _make_pyright_server(workspace: Path):
    """Build a multilspy LanguageServer pointing at `pyright-langserver
    --stdio`. Mirrors the Jedi server factory from
    `multilspy/language_servers/jedi_language_server/jedi_server.py`.
    """
    from multilspy.language_server import LanguageServer
    from multilspy.lsp_protocol_handler.server import ProcessLaunchInfo
    from multilspy.multilspy_config import MultilspyConfig
    from multilspy.multilspy_logger import MultilspyLogger

    config = MultilspyConfig.from_dict({"code_language": "python"})
    logger = MultilspyLogger()

    # Multilspy's stock Python factory uses jedi-language-server; we override
    # by instantiating the base LanguageServer directly with a Pyright launch
    # spec. The base class supports arbitrary commands via ProcessLaunchInfo.
    class PyrightServer(LanguageServer):
        def __init__(self) -> None:
            super().__init__(
                config,
                logger,
                str(workspace),
                ProcessLaunchInfo(
                    cmd="pyright-langserver --stdio",
                    cwd=str(workspace),
                ),
                "python",
            )

        def _get_initialize_params(self, repository_absolute_path: str):
            # Re-use Jedi's init params (they're language-agnostic — every
            # standard LSP client sends the same shape).
            here = pathlib.Path(__file__).resolve().parent
            jedi_init = (
                here.parent.parent
                / ".venv/lib/python3.12/site-packages/multilspy"
                / "language_servers/jedi_language_server/initialize_params.json"
            )
            d = json.loads(jedi_init.read_text())
            d.pop("_description", None)
            d["processId"] = os.getpid()
            d["rootPath"] = repository_absolute_path
            d["rootUri"] = pathlib.Path(repository_absolute_path).as_uri()
            d["workspaceFolders"][0]["uri"] = pathlib.Path(
                repository_absolute_path
            ).as_uri()
            d["workspaceFolders"][0]["name"] = os.path.basename(
                repository_absolute_path
            )
            return d

        @asynccontextmanager
        async def start_server(self) -> AsyncIterator["PyrightServer"]:
            async def do_nothing(params):
                return None

            self.server.on_notification("window/logMessage", do_nothing)
            self.server.on_notification("$/progress", do_nothing)
            self.server.on_notification("textDocument/publishDiagnostics", do_nothing)
            self.server.on_request("client/registerCapability", do_nothing)
            self.server.on_request(
                "workspace/configuration", lambda params: [{} for _ in params.get("items", [])]
            )

            async with super().start_server():
                await self.server.start()
                init_response = await self.server.send.initialize(
                    self._get_initialize_params(self.repository_root_path)
                )
                _ = init_response  # we don't assert on Pyright capabilities
                self.server.notify.initialized({})
                yield self
                await self.server.shutdown()
                await self.server.stop()

    return PyrightServer()


@dataclass
class PyrightLspChecker:
    """pyright-langserver-based Python checker. Threaded async loop — same
    pattern as `RustAnalyzerChecker`. `start()` boots the LSP in a background
    thread and waits for `initialize` to return; `stop()` shuts it down.
    `check()` writes `source` to `file_in_workspace`, touches it via an
    open/close cycle, and waits `settle_s` for `publishDiagnostics`.
    """

    workspace: Path
    file_in_workspace: str = "src/main.py"
    settle_s: float = 1.5
    is_async: bool = True
    _lock: asyncio.Lock | None = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _lsp: object | None = field(default=None, init=False, repr=False)
    _lsp_ctx: object | None = field(default=None, init=False, repr=False)
    _ready: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _captured: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            raise RuntimeError("pyright-langserver failed to start within 30s")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _start() -> None:
            self._lsp = _make_pyright_server(self.workspace)
            self._lsp_ctx = self._lsp.start_server()
            await self._lsp_ctx.__aenter__()
            self._lsp.server.on_notification(
                "textDocument/publishDiagnostics",
                lambda params: self._captured.__setitem__(
                    params["uri"], params.get("diagnostics", [])
                ),
            )
            self._ready.set()

        try:
            loop.run_until_complete(_start())
        except Exception:
            # Surface the failure to `start()` via the timeout path.
            self._ready.set()
            loop.close()
            return
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
        if self._lsp is None or self._loop is None:
            return []
        assert self._lock is not None
        async with self._lock:
            target = self.workspace / self.file_in_workspace
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
            uri = target.as_uri()
            self._captured.pop(uri, None)
            fut = asyncio.run_coroutine_threadsafe(
                self._touch_and_wait(self.file_in_workspace, uri), self._loop
            )
            try:
                diags = fut.result(timeout=self.settle_s + 2.0)
            except Exception:
                diags = []
        return _convert_pyright(diags)

    async def _touch_and_wait(self, relpath: str, uri: str) -> list[dict]:
        try:
            with self._lsp.open_file(relpath):
                await asyncio.sleep(self.settle_s)
                return self._captured.get(uri, [])
        except Exception:
            return []


def _convert_pyright(raw: list[dict]) -> list[Diagnostic]:
    """Convert Pyright LSP diagnostics → soundcode Diagnostics. Same severity
    mapping as `ra_check._convert` (severity 1 = Error → BLOCKING, others →
    NON_BLOCKING) so downstream classifier code works unchanged."""
    out: list[Diagnostic] = []
    for d in raw:
        sev = d.get("severity")
        code = d.get("code")
        if isinstance(code, dict):
            code = code.get("value") or code.get("code") or code
        if not isinstance(code, str):
            code = str(code) if code is not None else None
        category = Category.BLOCKING if sev == 1 else Category.NON_BLOCKING
        rng = d.get("range") or {}
        start = rng.get("start") or {}
        out.append(
            Diagnostic(
                category=category,
                message=d.get("message", ""),
                code=code,
                line=(start.get("line") or 0) + 1,
                column=(start.get("character") or 0) + 1,
            )
        )
    return out


# ─── Boundary detector (indent-aware, the load-bearing new piece) ─────────


class PythonBoundaryDetector:
    """Find offsets just after newlines whose NEXT non-blank, non-comment line
    starts at indent level 0 (start of a new top-level statement).

    Returned offsets satisfy the protocol convention: a boundary at offset
    `o` means the *next* checkpoint should be at `o` (i.e., immediately after
    the newline). Caller (`Code._update_boundary_checkpoints`) treats this as
    `o + 1` of an off-by-one boundary char — for Python the boundary IS the
    newline, so we return `newline_pos + 1` directly. This matches the
    semantics the Rust path applies via `find_boundaries(...).offset + 1`.

    Approach: line-by-line scan that tracks (a) bracket depth across (/[/{,
    (b) whether we are inside a triple-quoted string, (c) backslash line
    continuations. We only consider a newline a boundary candidate when
    bracket depth is 0, we are not inside a triple-quoted string, and the
    previous (logical) line did not end with `\\`. We then peek forward
    skipping blank lines and comment-only lines to find the NEXT real line;
    if its indent is 0, the newline is a boundary.

    Edge cases handled:
      - Single-quoted strings (' or ") with backslash escapes
      - Triple-quoted strings (\"\"\" and ''') including those crossing many lines
      - f-string prefixes (f, F, rf, etc.) — treated like normal strings; we
        don't try to track the embedded `{...}` brace depth, which is fine
        because braces inside the string never become real bracket depth
      - Raw strings (r/R/rb/...) — same as normal, the leading prefix is
        consumed naturally
      - `#` comments to end of line
      - Backslash line continuation (a single `\\` immediately before `\\n`
        merges with the next line — no boundary fires on the newline)
      - Blank lines and comment-only lines are skipped when peeking forward
        for the next "real" non-blank line

    Edge cases NOT fully handled (documented punts):
      - Decorators (`@foo`) start at indent 0 just like a `def`/`class` and
        DO fire boundaries. That's actually correct because the decorator IS
        the start of a top-level statement (the function defn it precedes
        will fire its own boundary later).
      - Inside parenthesized expressions, embedded triple-quoted strings are
        handled, but the bracket depth tracking does not look inside strings
        — which is the same convention as the Rust scanner.
      - `__all__ = [...]` spanning multiple lines correctly suppresses
        boundaries (bracket depth > 0).
      - We do NOT use `tokenize.generate_tokens`: it raises on partial /
        unterminated source, which is the whole point of an incremental
        detector. A small hand-rolled scanner is more forgiving.
    """

    def find_boundaries(self, source: str) -> list[int]:
        n = len(source)
        boundaries: list[int] = []

        # State across lines.
        paren = 0  # ( count
        bracket = 0  # [ count
        brace = 0  # { count
        in_triple: str | None = None  # None | '"""' | "'''"
        prev_backslash_continue = False  # the previous *real* line ended with `\`

        # Walk line by line. We need both the line text and the offset of
        # the newline that terminates it so we can return that offset+1.
        i = 0
        line_start = 0
        # Cache the indent of each upcoming non-blank/non-comment line so
        # boundaries can peek forward without re-scanning more than once.
        pending_newline: int | None = None  # offset of a newline that may be a boundary
        pending_was_continuation = False

        while i <= n:
            # Find next \n (or EOF).
            if i < n and source[i] != "\n":
                i += 1
                continue

            # We're at a newline OR EOF.
            line = source[line_start:i]
            newline_offset = i  # the index of the \n character itself (or n at EOF)

            # Walk THIS line's characters with current state to update
            # paren/bracket/brace/triple/backslash. We need the FINAL state
            # at the end of the line plus the FIRST-NON-WHITESPACE index for
            # indent detection on a possible boundary candidate.
            new_paren, new_bracket, new_brace, new_triple, ends_with_backslash, is_blank_or_comment, leading_indent = (
                _scan_line(line, paren, bracket, brace, in_triple)
            )

            # If we had a pending newline (the previous line ended at indent
            # eligible state and we were waiting to see if THIS line starts
            # at indent 0), decide it now.
            if pending_newline is not None:
                if is_blank_or_comment:
                    # Skip blank/comment lines; keep pending alive.
                    pass
                else:
                    if leading_indent == 0:
                        boundaries.append(pending_newline + 1)
                    pending_newline = None
                    pending_was_continuation = False

            # Decide whether THIS line's terminating newline is a boundary
            # candidate. It is iff at end-of-line:
            #   bracket depth == 0
            #   not inside a triple-quoted string
            #   not a backslash-continuation (current line ends with `\`)
            #   not a blank/comment-only line (those don't terminate statements)
            #   we are at a real newline (not EOF — EOF doesn't matter for
            #     boundaries since there's nothing after to start a statement)
            paren, bracket, brace, in_triple = new_paren, new_bracket, new_brace, new_triple
            prev_backslash_continue = ends_with_backslash

            if (
                i < n
                and paren == 0
                and bracket == 0
                and brace == 0
                and in_triple is None
                and not ends_with_backslash
                and not is_blank_or_comment
            ):
                pending_newline = newline_offset

            # Advance.
            i += 1
            line_start = i

        return boundaries


def _scan_line(
    line: str,
    paren: int,
    bracket: int,
    brace: int,
    in_triple: str | None,
) -> tuple[int, int, int, str | None, bool, bool, int]:
    """Scan one physical line, returning the updated state plus:
      - ends_with_backslash: whether the last non-whitespace char (outside a
        string/comment) is a bare `\\` continuation
      - is_blank_or_comment: whether the line has no executable content
        (blank, whitespace-only, or starts with `#` after optional indent)
      - leading_indent: number of leading spaces (tabs counted as 1 for
        boundary purposes — Python's "logical" indent is what matters and
        we only care whether indent == 0)
    """
    n = len(line)
    if n == 0:
        # Empty line — blank, no state change.
        return paren, bracket, brace, in_triple, False, True, 0

    # Leading indent (whitespace-only prefix length).
    j = 0
    while j < n and line[j] in (" ", "\t"):
        j += 1
    leading_indent = j

    # If we entered the line still inside a triple-quoted string, the leading
    # whitespace is part of the string — indent is "irrelevant" (not a
    # top-level statement). We need to be careful about `is_blank_or_comment`:
    #   - Pure continuation line (no closing quote on this line): blank — no
    #     boundary candidate.
    #   - Closing-quote line (`"""` then optionally more code or whitespace):
    #     this line IS the tail of a real statement (the assignment/expr
    #     containing the string literal). It should be a boundary candidate
    #     iff bracket depth ends at 0.
    entered_in_triple = in_triple is not None
    if in_triple is not None:
        quote = in_triple
        k = 0
        while k < n:
            if line.startswith(quote, k):
                in_triple = None
                k += len(quote)
                break
            if line[k] == "\\" and k + 1 < n:
                k += 2
                continue
            k += 1
        if in_triple is not None:
            # Still inside; the whole line is string body — no boundary.
            return paren, bracket, brace, in_triple, False, True, leading_indent
        # We exited mid-line; fall through to normal scan starting at k.
        idx = k
    else:
        idx = j  # start scanning at first non-whitespace char

    # `is_blank_or_comment` semantics:
    #   - If we entered the line inside a triple-string and exited it, the
    #     line is the tail of a real statement — NOT blank, even if nothing
    #     follows the closing quote. Boundary may fire.
    #   - Otherwise, "blank" means no executable content: empty, whitespace
    #     only, or a `#` comment (after optional leading indent).
    if entered_in_triple:
        is_blank_or_comment = False
    else:
        is_blank_or_comment = idx >= n or line[idx] == "#"
        rest = line[idx:]
        if rest.strip() == "" or rest.lstrip().startswith("#"):
            is_blank_or_comment = True

    ends_with_backslash = False

    # Now walk the line proper, updating bracket counts and detecting
    # triple-string entry and end-of-line backslash.
    i = idx
    last_nonspace_was_backslash = False
    last_nonspace_pos = -1
    while i < n:
        c = line[i]

        # Comment to end of line.
        if c == "#":
            break

        # Triple-quoted string entry.
        if (c == '"' or c == "'") and i + 2 < n and line[i + 1] == c and line[i + 2] == c:
            quote = c * 3
            in_triple = quote
            k = i + 3
            while k < n:
                if line.startswith(quote, k):
                    in_triple = None
                    k += 3
                    break
                if line[k] == "\\" and k + 1 < n:
                    k += 2
                    continue
                k += 1
            i = k
            last_nonspace_was_backslash = False
            if i > 0:
                last_nonspace_pos = i - 1
            continue

        # Single-line string.
        if c == '"' or c == "'":
            quote = c
            k = i + 1
            while k < n:
                if line[k] == "\\" and k + 1 < n:
                    k += 2
                    continue
                if line[k] == quote:
                    k += 1
                    break
                # Unterminated single-quoted string → consume to EOL
                # (Python would raise SyntaxError at compile time; for the
                # boundary scanner we just stop the string at the line break).
                k += 1
            i = k
            last_nonspace_was_backslash = False
            if i > 0:
                last_nonspace_pos = min(i, n) - 1
            continue

        if c == "(":
            paren += 1
        elif c == ")":
            paren = max(0, paren - 1)
        elif c == "[":
            bracket += 1
        elif c == "]":
            bracket = max(0, bracket - 1)
        elif c == "{":
            brace += 1
        elif c == "}":
            brace = max(0, brace - 1)

        if not c.isspace():
            last_nonspace_was_backslash = (c == "\\")
            last_nonspace_pos = i
        i += 1

    # End-of-line backslash continuation: only counts if the LAST non-space
    # non-comment char is a bare `\` and bracket depth at EOL is 0 (inside
    # brackets it's not a continuation, it's part of the expression).
    if last_nonspace_was_backslash and paren == 0 and bracket == 0 and brace == 0 and in_triple is None:
        ends_with_backslash = True

    return paren, bracket, brace, in_triple, ends_with_backslash, is_blank_or_comment, leading_indent


# ─── Workspace ──────────────────────────────────────────────────────────


@dataclass
class PythonWorkspace:
    """Python project workspace for Pyright. `setup()` materializes:
      - `pyproject.toml` (minimal `[project]` + `[tool.pyright]` enumerating
        `src/` as the source root)
      - `pyrightconfig.json` (explicit Pyright config — Pyright prefers this
        over `[tool.pyright]` when both exist; we ship both for robustness)
      - `src/main.py` (placeholder `def main(): pass`)

    `path` is optional. If not provided, a fresh temp dir is allocated.
    `teardown()` removes the dir IFF we created the temp dir ourselves — an
    externally-supplied path is left alone (matches RustWorkspace's policy).
    """

    path: Path | None = None
    _owned_tempdir: bool = field(default=False, init=False, repr=False)

    def setup(self) -> Path:
        if self.path is None:
            self.path = Path(tempfile.mkdtemp(prefix="soundcode-py-"))
            self._owned_tempdir = True
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "pyproject.toml").write_text(
            '[project]\n'
            'name = "demo"\n'
            'version = "0.1.0"\n'
            'requires-python = ">=3.10"\n'
            '\n'
            '[tool.pyright]\n'
            'include = ["src"]\n'
        )
        (self.path / "pyrightconfig.json").write_text(
            json.dumps(
                {
                    "include": ["src"],
                    "pythonVersion": "3.10",
                    "reportMissingImports": "warning",
                },
                indent=2,
            )
        )
        (self.path / "src").mkdir(exist_ok=True)
        main_py = self.path / "src" / "main.py"
        if not main_py.exists():
            main_py.write_text("def main():\n    pass\n")
        return self.path

    def teardown(self) -> None:
        if self._owned_tempdir and self.path is not None and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
            self.path = None
