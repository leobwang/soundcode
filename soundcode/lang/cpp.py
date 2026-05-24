"""C++ implementations of the `Checker` / `BoundaryDetector` / `Workspace`
Protocols. Phase 2 multilingual support.

Two checker variants:
  - `GccChecker`: sync, one-shot subprocess per call (`g++ -fsyntax-only`).
    Mirrors `RustCargoChecker` — no LSP daemon, every `check` spawns a
    fresh compiler invocation. Parses stderr for `file:line:col: severity:`
    diagnostics.
  - `ClangdLspChecker`: async, persistent clangd LSP. Driven by hand via
    asyncio subprocess + JSON-RPC because multilspy does not bundle a C++
    backend (only Rust, Go, Java, etc. — `multilspy/language_servers/`
    has no clangd entry). Needs a `compile_commands.json` in the workspace
    for meaningful semantic analysis; `CppWorkspace.setup()` writes one.

The boundary detector treats angle brackets as **non-depth-tracking** so
`std::vector<int>` does not push `<` onto a bracket stack that would then
have to be reconciled with `>` (which doubles as the shift operator,
greater-than, etc.). C++ statements live inside `()`/`[]`/`{}` exactly as
Rust does — templates are statement-internal noise, so a depth-0 `;`/`}`
remains the right signal.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

from soundcode.code import Category, Diagnostic


# ─── GccChecker (sync, one-shot subprocess) ────────────────────────────


@dataclass
class GccChecker:
    """`g++ -fsyntax-only` syntactic + semantic check.

    Writes the candidate source to `<workspace>/<file_in_workspace>` and
    runs g++ in `--syntax-only` mode so we get diagnostics without an
    object file. Same return shape as `CargoChecker.check`.

    `compiler` defaults to `g++`; pass `"clang++"` to use clang. The flag
    set (`-std=c++17 -Wall -fsyntax-only`) is accepted by both.
    """

    workspace: Path
    file_in_workspace: str = "main.cpp"
    compiler: str = "g++"
    std: str = "c++17"
    timeout_s: float = 15.0
    is_async: bool = False
    _lock: asyncio.Lock | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def check(self, source: str) -> list[Diagnostic]:
        """Write `source` to the workspace file and run the compiler."""
        assert self._lock is not None
        async with self._lock:
            target = self.workspace / self.file_in_workspace
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
            try:
                proc = await asyncio.create_subprocess_exec(
                    self.compiler,
                    "-fsyntax-only",
                    f"-std={self.std}",
                    "-Wall",
                    str(target),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ},
                )
            except FileNotFoundError:
                return []
            try:
                _stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return []

        return _parse_gcc_stderr(
            stderr.decode("utf-8", errors="replace"),
            str(target),
        )


# Match `<file>:<line>:<col>: <severity>: <message>` where severity is one
# of error / warning / note / fatal error. The file portion may be the
# absolute path or a relative one — we don't anchor on it.
_DIAG_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<sev>fatal error|error|warning|note):\s+(?P<msg>.*)$"
)


def _parse_gcc_stderr(stderr: str, file_path: str) -> list[Diagnostic]:
    """Parse `g++` stderr into Diagnostics.

    Only lines that look like `file:line:col: severity: message` are
    captured; carets, source quotes, and `In function ‘...’:` headers are
    ignored. Maps:
      fatal error / error → BLOCKING (classifier may refine to INCOMPLETE)
      warning             → NON_BLOCKING
      note                → skipped (we already have the primary diag)
    """
    out: list[Diagnostic] = []
    for raw_line in stderr.splitlines():
        m = _DIAG_RE.match(raw_line)
        if m is None:
            continue
        sev = m.group("sev")
        if sev in ("error", "fatal error"):
            cat = Category.BLOCKING
        elif sev == "warning":
            cat = Category.NON_BLOCKING
        else:  # note
            continue
        out.append(Diagnostic(
            category=cat,
            message=m.group("msg").strip(),
            code=None,
            line=int(m.group("line")),
            column=int(m.group("col")),
        ))
    return out


# ─── ClangdLspChecker (async, persistent LSP) ──────────────────────────


@dataclass
class ClangdLspChecker:
    """clangd-backed C++ checker. Mirrors `RustAnalyzerChecker` but talks
    JSON-RPC to a `clangd` subprocess directly (multilspy has no C++
    backend — verified against `multilspy/language_servers/`).

    `workspace` must contain a `compile_commands.json` (see
    `CppWorkspace.setup`) so clangd can resolve include paths and run
    semantic analysis. Without one, clangd falls back to a fallback
    command and reports almost nothing useful.
    """

    workspace: Path
    file_in_workspace: str = "main.cpp"
    settle_s: float = 1.5
    is_async: bool = True
    _lock: asyncio.Lock | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()
        self._proc: asyncio.subprocess.Process | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._captured: dict[str, list[dict]] = {}
        self._reader_task: asyncio.Task | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}

    def start(self) -> None:
        if shutil.which("clangd") is None:
            raise RuntimeError("clangd not installed")
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            raise RuntimeError("clangd failed to start within 30s")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._spawn_and_init())
            self._ready.set()
            loop.run_until_complete(self._wait_until_stop())
        finally:
            try:
                loop.run_until_complete(self._shutdown())
            except Exception:
                pass
            loop.close()

    async def _spawn_and_init(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            "clangd",
            "--background-index=false",
            "--log=error",
            cwd=str(self.workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        await self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": self.workspace.as_uri(),
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {"relatedInformation": False},
                    "synchronization": {"didSave": True},
                },
            },
        })
        await self._notify("initialized", {})

    async def _wait_until_stop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.2)

    async def _shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            await self._notify("exit", None)
        except Exception:
            pass
        try:
            self._proc.terminate()
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    async def check(self, source: str) -> list[Diagnostic]:
        if self._proc is None or self._loop is None:
            return []
        assert self._lock is not None
        async with self._lock:
            target = self.workspace / self.file_in_workspace
            target.write_text(source)
            uri = target.as_uri()
            self._captured.pop(uri, None)

            fut = asyncio.run_coroutine_threadsafe(
                self._open_and_wait(uri, source), self._loop,
            )
            try:
                diags = fut.result(timeout=self.settle_s + 5.0)
            except Exception:
                diags = []
        return _convert_lsp(diags)

    async def _open_and_wait(self, uri: str, source: str) -> list[dict]:
        try:
            await self._notify("textDocument/didOpen", {
                "textDocument": {
                    "uri": uri,
                    "languageId": "cpp",
                    "version": 1,
                    "text": source,
                },
            })
            await asyncio.sleep(self.settle_s)
            diags = self._captured.get(uri, [])
            await self._notify("textDocument/didClose", {
                "textDocument": {"uri": uri},
            })
            return diags
        except Exception:
            return []

    # --- JSON-RPC plumbing ---

    async def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        while True:
            header = await stdout.readline()
            if not header:
                return
            if not header.startswith(b"Content-Length:"):
                continue
            length = int(header.split(b":", 1)[1].strip())
            # consume the rest of headers (blank line)
            while True:
                ln = await stdout.readline()
                if not ln or ln in (b"\r\n", b"\n"):
                    break
            body = await stdout.readexactly(length)
            try:
                msg = json.loads(body.decode("utf-8"))
            except Exception:
                continue
            if "method" in msg and msg["method"] == "textDocument/publishDiagnostics":
                params = msg.get("params") or {}
                self._captured[params.get("uri", "")] = params.get("diagnostics", [])
            elif "id" in msg and msg["id"] in self._pending:
                self._pending.pop(msg["id"]).set_result(msg)

    async def _request(self, method: str, params: dict) -> dict:
        assert self._proc is not None and self._proc.stdin is not None
        msg_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = fut
        payload = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        self._write(payload)
        return await asyncio.wait_for(fut, timeout=10.0)

    async def _notify(self, method: str, params) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._write(payload)

    def _write(self, payload: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header + body)


def _convert_lsp(raw: list[dict]) -> list[Diagnostic]:
    """LSP `publishDiagnostics.diagnostics[]` → soundcode `Diagnostic`."""
    out: list[Diagnostic] = []
    for d in raw:
        sev = d.get("severity")
        code = d.get("code")
        if isinstance(code, dict):
            code = code.get("value") or code.get("code")
        if not isinstance(code, str):
            code = str(code) if code is not None else None
        # LSP severities: 1 Error, 2 Warning, 3 Info, 4 Hint
        category = Category.BLOCKING if sev == 1 else Category.NON_BLOCKING
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


# ─── CppBoundaryDetector ───────────────────────────────────────────────


class CppBoundaryDetector:
    """Statement/block boundary detector for C++.

    Same rule as Rust: a `;` or `}` outside strings, char literals, line
    or block comments, parens `()`, or square brackets `[]`. Braces are
    NOT suppressed (a `}` closing any block is itself a boundary).

    Angle brackets `<>` are deliberately NOT depth-tracked. In C++ they
    overload as both template delimiters (`std::vector<int>`,
    `make_unique<T>()`) and binary operators (`<`, `>`, `<=`, `>=`,
    `<<`, `>>`). A naive depth count immediately desyncs — and templates
    are statement-internal noise anyway: any `;`/`}` we'd want to flag
    sits *outside* the template argument list. So we treat `<` and `>`
    as ordinary characters.

    C++ block comments are NOT nestable (unlike Rust), so `/*` is closed
    by the first `*/` without depth tracking. Raw string literals follow
    C++11 syntax: `R"delim(...)delim"`.
    """

    def find_boundaries(self, source: str) -> list[int]:
        return [b for b in _cpp_find_boundaries(source)]


def _cpp_find_boundaries(source: str) -> list[int]:
    boundaries: list[int] = []
    i = 0
    n = len(source)
    paren = 0
    bracket = 0  # `[...]`

    while i < n:
        c = source[i]

        # Line comment
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            j = source.find("\n", i + 2)
            i = n if j == -1 else j + 1
            continue
        # Block comment (NOT nestable in C++)
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            j = source.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        # C++11 raw string: R"delim(...)delim" (optionally with u8/u/U/L prefix)
        if c == "R" and i + 1 < n and source[i + 1] == '"':
            end = _skip_raw_string(source, i + 1)
            i = end
            continue
        if c in ("u", "U", "L") and i + 1 < n:
            # u8R"..." / uR"..." / UR"..." / LR"..."
            j = i + 1
            if c == "u" and j < n and source[j] == "8":
                j += 1
            if j < n and source[j] == "R" and j + 1 < n and source[j + 1] == '"':
                end = _skip_raw_string(source, j + 1)
                i = end
                continue
        # Regular string literal (with optional u8/u/U/L prefix handled below
        # by allowing the prefix char to fall through; we only act on `"`).
        if c == '"':
            i = _skip_regular_string(source, i)
            continue
        # Char literal (no lifetime ambiguity in C++ — `'` always opens a char)
        if c == "'":
            i = _skip_char_literal(source, i)
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
            pass  # not suppressing — `}` still counts as boundary
        elif c == "}":
            if paren == 0 and bracket == 0:
                boundaries.append(i + 1)
        elif c == ";":
            if paren == 0 and bracket == 0:
                boundaries.append(i + 1)
        i += 1
    return boundaries


def _skip_regular_string(source: str, open_quote: int) -> int:
    """Return index *past* the closing `"`."""
    n = len(source)
    i = open_quote + 1
    while i < n:
        c = source[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == '"':
            return i + 1
        if c == "\n":
            # Unterminated — bail.
            return i + 1
        i += 1
    return n


def _skip_char_literal(source: str, open_quote: int) -> int:
    """Return index past the closing `'`."""
    n = len(source)
    i = open_quote + 1
    while i < n:
        c = source[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "'":
            return i + 1
        if c == "\n":
            return i + 1
        i += 1
    return n


def _skip_raw_string(source: str, open_quote: int) -> int:
    """Skip a C++11 raw string literal. `open_quote` points at the `"`.

    Form: `"delim(...)delim"` where `delim` is up to 16 chars from a
    restricted set. We just look for the `(` after the opening quote,
    capture the delimiter, then search for `)delim"`.
    """
    n = len(source)
    paren_idx = source.find("(", open_quote + 1)
    if paren_idx == -1:
        return n
    delim = source[open_quote + 1: paren_idx]
    needle = ")" + delim + '"'
    end_idx = source.find(needle, paren_idx + 1)
    if end_idx == -1:
        return n
    return end_idx + len(needle)


# ─── CppWorkspace ──────────────────────────────────────────────────────


@dataclass
class CppWorkspace:
    """C++ workspace scaffold.

    `setup()` writes a minimal `main.cpp` (`int main() { return 0; }`)
    and a `compile_commands.json` that points clangd at that file with a
    plain `g++ -std=c++17` command. The compile_commands entry is
    required for clangd to do meaningful semantic analysis; without it
    clangd falls back to a guess command and reports missing-include
    noise.

    Idempotent: re-running `setup()` overwrites the seed file (the
    checker overwrites it on every `check()` anyway). `teardown()`
    removes the entire workspace directory — use a `tempfile.mkdtemp()`
    path if you want isolation.
    """

    path: Path
    file_name: str = "main.cpp"
    std: str = "c++17"

    def setup(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        seed = self.path / self.file_name
        seed.write_text("int main() { return 0; }\n")
        cc = [{
            "directory": str(self.path),
            "file": str(seed),
            "arguments": [
                "g++", f"-std={self.std}", "-Wall", "-c", str(seed),
            ],
        }]
        (self.path / "compile_commands.json").write_text(json.dumps(cc, indent=2))
        return self.path

    def teardown(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
