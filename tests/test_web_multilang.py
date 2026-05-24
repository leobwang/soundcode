"""Multilingual Phase 4 tests for the web demo server.

Covers:
  - The server starts cleanly on a non-production port.
  - `/api/prompts?lang=…` returns a non-empty list for each supported
    language (rust + the three new ones).
  - `/api/prompts/{id}?lang=…` returns the full prompt for a known id.
  - The (language, mode) dispatch table maps to the correct checker /
    boundary / workspace classes.
  - The dispatch surfaces a `status: error` event (rather than
    crashing) when the required compiler/LSP is missing.

These tests do NOT spin up a real LLM. The end-to-end token-streaming
behaviour is covered by `tests/test_web_demo.py` (Rust-only Playwright
tests) and `tests/test_demo_client.py` (mock LLM unit tests).

Run:
    uv run pytest tests/test_web_multilang.py -v

The server fixture binds to a free port (typically 8081) — explicitly
NOT 8080, which is reserved for the systemd-managed production instance.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from soundcode.lang.cpp import (
    ClangdLspChecker,
    CppBoundaryDetector,
    CppWorkspace,
    GccChecker,
)
from soundcode.lang.java import (
    JavaBoundaryDetector,
    JavaWorkspace,
    JavacChecker,
    JdtLspChecker,
)
from soundcode.lang.python import (
    PyrightLspChecker,
    PythonBoundaryDetector,
    PythonCompileChecker,
    PythonWorkspace,
)
from soundcode.lang.rust import (
    RustAnalyzerLspChecker,
    RustBoundaryDetector,
    RustCargoChecker,
    RustWorkspace,
)
from soundcode.web import server as srv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── helpers ──────────────────────────────────────────────────────────


def _free_port() -> int:
    """Pick a free local port for the test server. Explicitly avoids
    8080 (production systemd instance) — `_free_port` returns a kernel-
    assigned port, which is almost never 8080."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert port != 8080, "refusing to use the production port"
    return port


@pytest.fixture(scope="module")
def server_proc():
    """Spin up `soundcode.web.server` on a free local port. Module-scoped
    so the suite shares one server (cuts ~3s of startup per test)."""
    port = _free_port()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "soundcode.web.server",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 25
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    break
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    else:
        proc.kill()
        out = proc.stdout.read().decode("utf-8", errors="replace")
        pytest.fail(f"server did not start in 25s:\n{out}")

    yield port, url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _get_json(url: str) -> tuple[int, object]:
    """Fetch a URL, return (status, parsed-json-or-text)."""
    try:
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


# ─── 1. server startup ────────────────────────────────────────────────


def test_server_starts(server_proc):
    port, url = server_proc
    code, body = _get_json(url + "/")
    assert code == 200, (code, body)
    # The body is the HTML — verify it actually mentions the soundcode
    # title so we know we're hitting the right server.
    assert "SoundCode" in body if isinstance(body, str) else False


# ─── 2. /api/prompts per language ────────────────────────────────────


@pytest.mark.parametrize("lang", ["rust", "java", "cpp", "python"])
def test_api_prompts_per_language(server_proc, lang):
    """Each supported language returns a non-empty prompt list with the
    expected per-prompt shape."""
    port, url = server_proc
    code, body = _get_json(f"{url}/api/prompts?lang={lang}")
    assert code == 200, (lang, code, body)
    assert isinstance(body, list), (lang, type(body), body)
    assert len(body) > 0, f"no prompts returned for lang={lang}"
    # Every entry has the expected keys.
    for item in body:
        for k in ("id", "title", "prompt_len", "is_custom"):
            assert k in item, (lang, item)


def test_api_prompts_default_lang_is_rust(server_proc):
    """Backward compat: a request without `?lang=` defaults to Rust."""
    _, url = server_proc
    code, body = _get_json(f"{url}/api/prompts")
    assert code == 200
    assert isinstance(body, list)
    # Rust ships the LeetCode 37 custom prompt; verify it's present so we
    # know we got the Rust list (not an empty fallback).
    ids = [p["id"] for p in body]
    assert "LeetCode_37_solve_sudoku" in ids, ids


@pytest.mark.parametrize("lang,sample_id", [
    ("rust", "LeetCode_37_solve_sudoku"),
    ("java", "java_reverse_list"),
    ("cpp", "cpp_reverse_vec"),
    ("python", "py_reverse_list"),
])
def test_api_prompts_get_single(server_proc, lang, sample_id):
    """A known custom-prompt id resolves to the full prompt text."""
    _, url = server_proc
    code, body = _get_json(f"{url}/api/prompts/{sample_id}?lang={lang}")
    assert code == 200, (lang, sample_id, code, body)
    assert body["id"] == sample_id
    assert len(body["prompt"]) > 50, (lang, sample_id, body)


def test_api_prompts_unknown_lang_falls_back_to_rust(server_proc):
    """Unknown `lang=` falls back to Rust rather than erroring — keeps the
    UI alive if someone hand-types a typo in the URL."""
    _, url = server_proc
    code, body = _get_json(f"{url}/api/prompts?lang=cobol")
    assert code == 200, (code, body)
    assert isinstance(body, list)
    # Should match the Rust list (which is what `?lang=rust` returns).
    code2, body2 = _get_json(f"{url}/api/prompts?lang=rust")
    assert [p["id"] for p in body] == [p["id"] for p in body2]


# ─── 3. dispatch table maps to correct classes ───────────────────────


def test_dispatch_table_covers_all_languages_and_modes():
    """Sanity: the dispatch table has an entry for every (language, mode)
    combo we expose to the UI. 4 langs × 2 modes = 8 entries."""
    expected = {
        (lang, mode)
        for lang in srv.SUPPORTED_LANGS
        for mode in ("compiler", "lsp")
    }
    assert set(srv.LANG_DISPATCH.keys()) == expected


@pytest.mark.parametrize("language,verifier,checker_cls,boundary_cls", [
    ("rust",   "compiler", RustCargoChecker,        RustBoundaryDetector),
    ("rust",   "cargo",    RustCargoChecker,        RustBoundaryDetector),
    ("rust",   "lsp",      RustAnalyzerLspChecker,  RustBoundaryDetector),
    ("rust",   "ra",       RustAnalyzerLspChecker,  RustBoundaryDetector),
    ("java",   "compiler", JavacChecker,            JavaBoundaryDetector),
    ("java",   "lsp",      JdtLspChecker,           JavaBoundaryDetector),
    ("cpp",    "compiler", GccChecker,              CppBoundaryDetector),
    ("cpp",    "lsp",      ClangdLspChecker,        CppBoundaryDetector),
    ("python", "compiler", PythonCompileChecker,    PythonBoundaryDetector),
    ("python", "lsp",      PyrightLspChecker,       PythonBoundaryDetector),
])
def test_dispatch_resolves_to_expected_classes(
    language, verifier, checker_cls, boundary_cls
):
    """Drive `_normalize_mode` + `LANG_DISPATCH` directly to verify the
    intended (Checker, BoundaryDetector) classes are picked. Doesn't
    instantiate them — that's covered by the lang-pack tests."""
    mode = srv._normalize_mode(verifier)
    entry = srv.LANG_DISPATCH[(language, mode)]
    assert entry["checker_cls"] is checker_cls, (language, verifier, entry)
    assert entry["boundary_cls"] is boundary_cls, (language, verifier, entry)


# ─── 4. WS-level dispatch via the start event ────────────────────────


@pytest.mark.parametrize("language,verifier,expected_checker", [
    ("rust",   "compiler", "RustCargoChecker"),
    ("java",   "compiler", "JavacChecker"),
    ("cpp",    "compiler", "GccChecker"),
    ("python", "compiler", "PythonCompileChecker"),
])
def test_ws_start_event_dispatches_correct_workspace(
    monkeypatch, language, verifier, expected_checker
):
    """End-to-end dispatch: simulate the WS `start` payload, intercept
    `DemoClient.generate` so we don't actually call any LLM, and verify
    the checker class that `_run_generation` constructed matches the
    expected per-language type.

    This is the linchpin test for Phase 4: it asserts the full pipeline
    from `{"type":"start","language":<l>,"verifier":<v>}` through to
    the checker class the orchestrator ends up using. If a future change
    breaks the dispatch (typo in the table, mismatched key, etc.) this
    will fail with the wrong class name in the assertion.
    """
    captured = {}

    # Intercept DemoClient construction & generate so we observe the
    # wiring without booting an LLM.
    class _FakeLlm:
        config = type("C", (), {"reenter_thinking_on_rollback": False,
                                "mode": None,
                                "stop": []})()
        async def warmup(self): pass
        async def close(self): pass
        def set_prompt(self, p): pass
        def has_next(self): return False
        async def next(self): return None
        async def abort_current_stream(self): pass

    async def _fake_generate(self, prompt):
        captured["checker_cls"] = type(self.checker).__name__
        captured["boundary_cls"] = type(self.boundary).__name__
        return ""

    monkeypatch.setattr(srv, "LlmServer", lambda **kw: _FakeLlm())
    monkeypatch.setattr(
        "soundcode.web.demo_client.DemoClient.generate", _fake_generate
    )
    # Bypass the cargo cache warm-up — we don't have a real workspace
    # set up for non-Rust paths and the warmup spawns a subprocess.
    async def _noop_warm(_ws): pass
    monkeypatch.setattr(srv, "_warm_cargo", _noop_warm)

    # Tool-availability gate: force-True so the test runs even when the
    # local box lacks (e.g.) javac. The construction itself will work as
    # long as the lang-pack import succeeded.
    monkeypatch.setattr(srv, "_tool_available", lambda tool: True)

    req = {
        "type": "start",
        "prompt": "fn f() {",
        "language": language,
        "verifier": verifier,
        "model": "mistral-small3.2:24b",
        "token_delay_ms": 0,
        "instruct": False,
        "mode": "raw",
    }
    events = []

    async def emit(ev):
        events.append(ev)

    asyncio.run(srv._run_generation(req, emit))

    assert captured.get("checker_cls") == expected_checker, (
        f"expected {expected_checker}, got {captured}; events: {events}"
    )


# ─── 5. graceful skip when the required tool is missing ──────────────


def test_missing_tool_surfaces_status_error(monkeypatch):
    """When the required compiler/LSP is absent, `_run_generation` must
    emit a `status: error` event and return cleanly (NOT raise / crash).
    """
    # Force the tool-availability probe to claim everything is missing.
    monkeypatch.setattr(srv, "_tool_available", lambda tool: False)

    req = {
        "type": "start",
        "prompt": "// hello",
        "language": "cpp",
        "verifier": "lsp",
        "model": "mistral-small3.2:24b",
        "token_delay_ms": 0,
        "instruct": False,
        "mode": "raw",
    }
    events = []

    async def emit(ev):
        events.append(ev)

    asyncio.run(srv._run_generation(req, emit))
    err = [e for e in events if e.get("phase") == "error"]
    assert err, [e for e in events]
    assert "clangd" in err[0]["message"].lower(), err[0]


# ─── 6. per-language smoke check via the actual lang packs ───────────


@pytest.mark.skipif(
    shutil.which("javac") is None, reason="javac not installed"
)
def test_java_workspace_setup_and_checker_runs():
    """End-to-end on the Java lang pack: workspace materialises, JavacChecker
    runs against a small program, returns the expected error category."""
    ws = JavaWorkspace()
    path = ws.setup()
    try:
        chk = JavacChecker(workspace=path)
        # A program with a clear type error.
        bad = (
            "public class Main {\n"
            "  public static void main(String[] args) {\n"
            "    int x = \"hello\";\n"
            "  }\n"
            "}\n"
        )
        diags = asyncio.run(chk.check(bad))
        assert any(d.is_blocking for d in diags), diags
    finally:
        ws.teardown()


@pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not installed"
)
def test_cpp_workspace_setup_and_checker_runs():
    """End-to-end on the C++ lang pack: workspace materialises, GccChecker
    runs against a small program with a syntax error."""
    tmp = Path(tempfile.mkdtemp(prefix="soundcode_cpp_test_"))
    try:
        ws = CppWorkspace(path=tmp)
        path = ws.setup()
        chk = GccChecker(workspace=path)
        bad = "int main() {\n  return \"oops\";\n}\n"
        diags = asyncio.run(chk.check(bad))
        assert any(d.is_blocking for d in diags), diags
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def test_python_workspace_setup_and_checker_runs():
    """End-to-end on the Python lang pack — PythonCompileChecker doesn't
    need any external tool (uses `compile()` in-process)."""
    ws = PythonWorkspace()
    path = ws.setup()
    try:
        chk = PythonCompileChecker(workspace=path)
        # SyntaxError.
        bad = "def f(:\n    pass\n"
        diags = asyncio.run(chk.check(bad))
        assert any(d.is_blocking for d in diags), diags
    finally:
        ws.teardown()
