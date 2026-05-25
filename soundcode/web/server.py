"""Web-demo server for the rollback generator.

Single aiohttp server:
  - GET  /                    -> index.html
  - GET  /static/*            -> static assets
  - GET  /api/prompts?lang=…  -> list of bundled prompts for `lang` (default rust)
  - GET  /api/prompts/{id}?lang=…  -> full prompt text for `id` within `lang`
  - WS   /ws                  -> bidirectional protocol (see Events below)

Run:
    uv run python -m soundcode.web.server [--host 127.0.0.1] [--port 8765]

Events client->server:
  {"type": "start", "prompt": str,
   "verifier": "compiler"|"lsp"|"cargo"|"ra",  # cargo/ra are legacy aliases
   "language": "rust"|"java"|"cpp"|"python",   # default "rust"
   "model": str, "token_delay_ms": int, "instruct": bool}
  {"type": "stop"}

Events server→client (see DemoClient for the full set):
  {"type": "status", "phase": "loading"|"ready"|"generating"|"done"|"warning", "message": str}
  {"type": "reset_buffer"}
  {"type": "token_emitted", "text": str, "offset": int}
  {"type": "check_started", "offset": int}
  {"type": "verdict", "verdict": "ok"|"error"|"inactive", "diagnostics": [...], "offset": int}
  {"type": "checkpoint", "offset": int}
  {"type": "rollback", "to_offset": int, "discarded": str, "rollback_count": int}
  {"type": "backend_stats", "prefix_tokens": int, "cached_tokens": int,
   "cache_hit_pct": float, "ttft_s": float|None, "is_rollback": bool}
  {"type": "final", "content": str, "rollback_count": int}
  {"type": "error", "message": str}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from aiohttp import web, WSMsgType

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
from soundcode.llm import LlmServer, GenerationConfig
from soundcode.logits_processors import (
    NoopLogitsProcessor,
    RocodeDecayingPenaltyProcessor,
    SemGuardEvaluatorProcessor,
)
from soundcode.rocode_processor import RocodeTriePenaltyProcessor
from soundcode.semguard_eval import SemGuardEvaluator
from soundcode.eval.dataset import load_rust_humaneval
from soundcode.web.demo_client import DemoClient, DemoConfig
from soundcode.web.extra_prompts import CUSTOM_PROBLEMS
from soundcode.web.extra_prompts_multi import PROBLEMS_BY_LANG


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_WS = PROJECT_ROOT / "cargo_workspaces" / "web_demo"
LOGS_DIR = PROJECT_ROOT / "results" / "web_demo"

SUPPORTED_LANGS = ("rust", "java", "cpp", "python")


# Map UI-level verifier names ("compiler"/"lsp" — the new generalized
# names, plus legacy Rust-only "cargo"/"ra") to the canonical internal
# tags. The legacy aliases keep backward compat with older clients still
# sending "cargo"/"ra" in the WS start payload.
_MODE_ALIASES: dict[str, str] = {
    "cargo": "compiler",
    "compiler": "compiler",
    "ra": "lsp",
    "lsp": "lsp",
}


def _normalize_mode(verifier: str) -> str:
    """Map any of {compiler, cargo, lsp, ra} to {compiler, lsp}."""
    return _MODE_ALIASES.get(verifier, "compiler")


# Friendly install hints surfaced as a status:error event when the
# required binary is missing on this machine.
_INSTALL_HINTS: dict[str, str] = {
    "cargo": "cargo not installed; install via rustup (https://rustup.rs).",
    "rust-analyzer": "rust-analyzer not installed; install via "
                     "`rustup component add rust-analyzer`.",
    "javac": "javac not installed; install via `apt install default-jdk` "
             "(or equivalent).",
    "jdt.ls": "Eclipse JDT.LS bundle missing; first multilspy run will "
              "download it (~hundreds of MB).",
    "g++": "g++ not installed; install via `apt install g++` (or equivalent).",
    "clangd": "clangd not installed; install via `apt install clangd` "
              "(or equivalent).",
    "python": "python interpreter not available (this should not happen).",
    "pyright-langserver": "pyright not installed; install via "
                          "`uv add pyright` or `npm i -g pyright`.",
}


def _tool_available(tool: str) -> bool:
    """Best-effort `which`-based probe."""
    if tool == "python":
        return True
    if tool == "jdt.ls":
        # multilspy downloads JDT.LS lazily on first use; we can't probe
        # for it without booting the server. Trust that it will work; if
        # not, JdtLspChecker.start() surfaces the failure.
        return True
    binary = {
        "cargo": "cargo",
        "rust-analyzer": "rust-analyzer",
        "javac": "javac",
        "g++": "g++",
        "clangd": "clangd",
        "pyright-langserver": "pyright-langserver",
    }.get(tool, tool)
    return shutil.which(binary) is not None


# Workspace factories — return a fresh Workspace instance whose `path`
# (if applicable) is already pointed at the right directory. Rust points
# at a persistent dir under `cargo_workspaces/` so cargo's incremental
# cache survives across runs; the others use a fresh tempdir per run.


def _rust_workspace_factory() -> RustWorkspace:
    return RustWorkspace(path=DEMO_WS)


def _java_workspace_factory() -> JavaWorkspace:
    return JavaWorkspace()


def _cpp_workspace_factory() -> CppWorkspace:
    return CppWorkspace(path=Path(tempfile.mkdtemp(prefix="soundcode_cpp_demo_")))


def _python_workspace_factory() -> PythonWorkspace:
    return PythonWorkspace()


# Per-language function-closer string. Glued onto the model-generated
# `code.content` inside `Code.check()` (in-loop verifier hits) and inside
# `DemoClient._emit_final_check` (post-hoc full-body check) so the file
# fed to the checker is syntactically complete. Each closer must match
# the corresponding workspace's seed shape (see `LANG_DISPATCH` below):
#
#   - Rust: prompt ends with `fn foo(...) -> T {`. Closer closes the body
#     with `unreachable!()` (universal-return shim that satisfies any
#     return type so only in-content errors surface) plus a `}` plus a
#     trailing `fn main() {}` so the cargo crate has an entry point.
#     This is the historical default — preserved byte-identically.
#   - C++: prompt ends with `T foo(...) {`. Closer uses `return {};`
#     (value-init that satisfies any non-void return type via brace-init,
#     and is accepted by g++ for void too) plus the body `}` plus an
#     `int main() { return 0; }` entry point.
#   - Java: prompt ends with `public class Main { ... static T foo(...) {`
#     where the workspace seeds the Main class itself. Closer therefore
#     has to close the method (`}`) AND the class (`}`). No separate
#     `main()` needed — `JavaWorkspace._MAIN_JAVA_SKELETON` already wires
#     one in (and the model body is the static `foo`, not `main`).
#   - Python: prompt ends with `def foo(...) -> T:\n    """docs"""`.
#     The closer is `\n    pass\n` — adds a benign no-op statement at the
#     function-body indent so `compile()` doesn't choke on a dangling `:`
#     when the body is empty / partial. Python has no separate `main()`
#     entry point.
LANG_CLOSER: dict[str, str] = {
    "rust":   "\n    unreachable!()\n}\n\nfn main() {}\n",
    "cpp":    "\n    return {};\n}\n\nint main() { return 0; }\n",
    "java":   "\n    }\n}\n",
    "python": "\n    pass\n",
}


# Dispatch table: (language, mode) -> dict of factories. `checker_cls`
# is invoked as `checker_cls(workspace=<path>)`; `boundary_cls()` takes
# no args; `workspace_factory()` returns the Workspace instance whose
# `setup()` we then call.
LANG_DISPATCH = {
    ("rust", "compiler"): {
        "workspace_factory": _rust_workspace_factory,
        "checker_cls": RustCargoChecker,
        "boundary_cls": RustBoundaryDetector,
        "tool": "cargo",
    },
    ("rust", "lsp"): {
        "workspace_factory": _rust_workspace_factory,
        "checker_cls": RustAnalyzerLspChecker,
        "boundary_cls": RustBoundaryDetector,
        "tool": "rust-analyzer",
    },
    ("java", "compiler"): {
        "workspace_factory": _java_workspace_factory,
        "checker_cls": JavacChecker,
        "boundary_cls": JavaBoundaryDetector,
        "tool": "javac",
    },
    ("java", "lsp"): {
        "workspace_factory": _java_workspace_factory,
        "checker_cls": JdtLspChecker,
        "boundary_cls": JavaBoundaryDetector,
        "tool": "jdt.ls",
    },
    ("cpp", "compiler"): {
        "workspace_factory": _cpp_workspace_factory,
        "checker_cls": GccChecker,
        "boundary_cls": CppBoundaryDetector,
        "tool": "g++",
    },
    ("cpp", "lsp"): {
        "workspace_factory": _cpp_workspace_factory,
        "checker_cls": ClangdLspChecker,
        "boundary_cls": CppBoundaryDetector,
        "tool": "clangd",
    },
    ("python", "compiler"): {
        "workspace_factory": _python_workspace_factory,
        "checker_cls": PythonCompileChecker,
        "boundary_cls": PythonBoundaryDetector,
        "tool": "python",
    },
    ("python", "lsp"): {
        "workspace_factory": _python_workspace_factory,
        "checker_cls": PyrightLspChecker,
        "boundary_cls": PythonBoundaryDetector,
        "tool": "pyright-langserver",
    },
}


# ─── backend + algorithm dispatch ─────────────────────────────────────


def _make_ollama_backend(model: str, config: GenerationConfig, _processor):
    """Ollama backend ignores the logits_processor — Ollama's HTTP API
    doesn't accept per-step logit hooks. The processor is constructed for
    API symmetry with the vLLM path, then dropped here."""
    return LlmServer(model=model, config=config)


def _make_vllm_backend(model: str, config: GenerationConfig, processor):
    """vLLM backend: imports the module lazily so the production Ollama
    deployment doesn't have to pay the vLLM import cost on every server
    startup. Forwards the configured logits_processor through."""
    from soundcode.vllm_backend import VllmBackend
    return VllmBackend(
        model=model, config=config, logits_processor=processor,
    )


BACKEND_DISPATCH = {
    "ollama": _make_ollama_backend,
    "vllm":   _make_vllm_backend,
}


ALGORITHM_DISPATCH = {
    # SoundCode default — no penalty, no observation. Rollback works via
    # prompt-prefix re-decode + structural checkpoint (the processor never
    # touches logits).
    "soundcode":       NoopLogitsProcessor,
    # MVP ROCODE — flat list of (token_id, expected_position) pairs, no trie.
    # The `"rocode"` key is the historical name; `"rocode_lite"` is the new
    # explicit alias post-faithful-version. Both point at the same class so
    # existing UI clients keep working.
    "rocode":          RocodeDecayingPenaltyProcessor,
    "rocode_lite":     RocodeDecayingPenaltyProcessor,
    # Faithful ROCODE — backed by RocodeTrie; tracks per-token nodes with
    # accumulated penalty across attempts. The lambda factory pins
    # `lam=0.9` (paper default); a future patch can plumb lambda through
    # the WS request payload if we want per-run tuning.
    "rocode_faithful": lambda: RocodeTriePenaltyProcessor(lam=0.9),
    # SemGuard fallback — the actual singleton-backed construction happens
    # in `_run_generation` (see `get_semguard_evaluator()` below). This
    # dispatch entry is the no-op fallback: if a caller goes through the
    # generic factory path (e.g. a test), they get a placeholder that
    # silently degrades to logits-passthrough rather than crashing.
    "semguard":        SemGuardEvaluatorProcessor,
}


# ─── SemGuard singleton ────────────────────────────────────────────────
#
# The trained evaluator (1.3B classifier on top of DeepSeek-Coder-1.3B-base)
# is expensive to construct — it loads the full backbone into VRAM, plus
# the BCE head on top. We build it once at server startup and reuse it
# across every SemGuard run. The per-request processor wraps the singleton
# so its `score_prefix` calls all serialise through the evaluator's internal
# `asyncio.Lock`.
#
# Paths are hard-coded to the training output. If either is missing (e.g.
# training hasn't finished or the user hasn't downloaded the backbone), the
# factory returns None and the server falls back to constructing a no-op
# SemGuardEvaluatorProcessor — generation still works, just without
# rollback signalling. A `status` WS event surfaces this to the UI so the
# user knows they're getting a degraded run.
_SEMGUARD_CKPT_PATH = (
    PROJECT_ROOT
    / "reproduce" / "results" / "semguard"
    / "train_20260524_115641" / "deepseek-coder-1.3b_python"
    / "checkpoinss" / "0" / "model_0.bin"
)
_SEMGUARD_BACKBONE_PATH = Path.home() / "hf-models" / "deepseek-coder-1.3b-base"

_SEMGUARD_EVALUATOR: SemGuardEvaluator | None = None  # lazy-init singleton
_SEMGUARD_LOAD_LOCK = asyncio.Lock()                  # one-time init guard


async def get_semguard_evaluator() -> SemGuardEvaluator | None:
    """Lazily construct + warmup the SemGuardEvaluator singleton.

    Returns the singleton instance, or None if the checkpoint / backbone
    isn't present on disk (caller decides to fall back to a no-op processor).
    Subsequent calls return the cached singleton — `await warmup()` runs
    only on the very first call, under `_SEMGUARD_LOAD_LOCK` so two
    concurrent SemGuard runs don't both pay the load cost."""
    global _SEMGUARD_EVALUATOR
    if _SEMGUARD_EVALUATOR is not None:
        return _SEMGUARD_EVALUATOR
    if not _SEMGUARD_CKPT_PATH.exists() or not _SEMGUARD_BACKBONE_PATH.exists():
        return None
    async with _SEMGUARD_LOAD_LOCK:
        # Double-checked-locking: another coroutine may have initialised
        # us while we were waiting for the lock.
        if _SEMGUARD_EVALUATOR is not None:
            return _SEMGUARD_EVALUATOR
        evaluator = SemGuardEvaluator(
            checkpoint_path=_SEMGUARD_CKPT_PATH,
            backbone_path=_SEMGUARD_BACKBONE_PATH,
        )
        await evaluator.warmup()
        _SEMGUARD_EVALUATOR = evaluator
    return _SEMGUARD_EVALUATOR


# ─── workspace + checker wiring ───────────────────────────────────────


def _ensure_workspace() -> Path:
    """Materialize the Rust demo workspace. Thin wrapper over
    `RustWorkspace.setup()` — kept as a module-level helper so tests and
    other callers can still import it under its historical name."""
    return RustWorkspace(path=DEMO_WS).setup()


async def _warm_cargo(ws: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        "cargo", "check", "--offline", "--quiet",
        cwd=str(ws),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    await proc.communicate()


# ─── HTTP routes ──────────────────────────────────────────────────────


_PROBLEMS_CACHE = None


def _problems():
    """Rust HumanEval problems (lazy-loaded; cached). Only Rust ships with
    a bundled HumanEval set — the other languages get just the curated
    prompt list from `extra_prompts_multi.py`."""
    global _PROBLEMS_CACHE
    if _PROBLEMS_CACHE is None:
        try:
            _PROBLEMS_CACHE = load_rust_humaneval()
        except Exception:
            # Dataset/HF unavailable (e.g., offline test env). Degrade to
            # just the custom prompt list rather than crashing the server.
            _PROBLEMS_CACHE = []
    return _PROBLEMS_CACHE


async def index(request):
    return web.FileResponse(STATIC_DIR / "index.html")


def _request_lang(request) -> str:
    """Pull the `lang` query param, defaulting to "rust" (= backward
    compat with clients that don't pass one)."""
    lang = request.query.get("lang", "rust").lower()
    if lang not in SUPPORTED_LANGS:
        lang = "rust"
    return lang


def _custom_problems_for(lang: str) -> list:
    """Curated prompts for the given language. Rust draws from
    `extra_prompts.CUSTOM_PROBLEMS`; the others from
    `extra_prompts_multi.PROBLEMS_BY_LANG`."""
    if lang == "rust":
        return list(CUSTOM_PROBLEMS)
    return list(PROBLEMS_BY_LANG.get(lang, []))


def _custom_first(extra=CUSTOM_PROBLEMS) -> list:
    """Return a single combined list: custom problems first, then HumanEval.
    Rust-only (kept for callers that import this directly)."""
    return list(extra) + list(_problems())


async def list_prompts(request):
    lang = _request_lang(request)
    out = []
    for p in _custom_problems_for(lang):
        out.append({
            "id": p.name,
            "title": p.title[:90],
            "prompt_len": len(p.prompt),
            "is_custom": True,
        })
    # Only Rust ships the MultiPL-E HumanEval set. Other languages get
    # just the curated prompt list above.
    if lang == "rust":
        for p in _problems():
            first_line = next(
                (ln for ln in p.prompt.split("\n") if ln.startswith("///")),
                p.name,
            ).strip("/ ").strip()
            out.append({
                "id": p.name,
                "title": first_line[:90],
                "prompt_len": len(p.prompt),
                "is_custom": False,
            })
    return web.json_response(out)


async def get_prompt(request):
    pid = request.match_info["id"]
    lang = _request_lang(request)
    for p in _custom_problems_for(lang):
        if p.name == pid:
            return web.json_response({"id": p.name, "prompt": p.prompt})
    if lang == "rust":
        for p in _problems():
            if p.name == pid:
                return web.json_response({"id": p.name, "prompt": p.prompt})
    return web.json_response({"error": "not found"}, status=404)


# ─── WebSocket ────────────────────────────────────────────────────────


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)

    async def emit(event: dict) -> None:
        if not ws.closed:
            await ws.send_json(event)

    current_task: asyncio.Task | None = None

    async def stop_current():
        nonlocal current_task
        if current_task is not None and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except (asyncio.CancelledError, Exception):
                pass
        current_task = None

    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                req = json.loads(msg.data)
            except json.JSONDecodeError:
                await emit({"type": "error", "message": "bad json"})
                continue
            kind = req.get("type")
            if kind == "start":
                await stop_current()
                current_task = asyncio.create_task(
                    _run_generation(req, emit)
                )
            elif kind == "stop":
                await stop_current()
                await emit({"type": "status", "phase": "done", "message": "stopped"})
        elif msg.type == WSMsgType.ERROR:
            break
    await stop_current()
    return ws


REASONING_MODELS = frozenset({
    "qwen3.5:9b", "qwen3.5:35b", "qwen3.5:122b",
    "qwen3.6:35b", "qwen3.6:latest",
    "gpt-oss:20b", "gpt-oss:120b",
    "deepseek-r1:70b",
    "nemotron-cascade-2:30b",
    "nemotron-3-super:120b",
    "nemotron-3-nano:4b",
})


async def _run_generation(req: dict, emit) -> None:
    import time, uuid, datetime
    from soundcode.llm import OrchestrationMode, template_for_model
    try:
        prompt = req["prompt"]
        verifier = req.get("verifier", "cargo")
        # Backward compat: clients that don't send a `language` field get
        # Rust (the only language pre-multilingual). Validate against the
        # supported set and fall back to "rust" on unknown values so a
        # typo doesn't fail the run.
        language = (req.get("language") or "rust").lower()
        if language not in SUPPORTED_LANGS:
            language = "rust"
        mode_tag = _normalize_mode(verifier)
        model = req.get("model", "mistral-small3.2:24b")
        token_delay_ms = int(req.get("token_delay_ms", 0))
        instruct = bool(req.get("instruct", False))
        # Backend + algorithm dispatch. Missing → ollama + soundcode (= the
        # historical defaults, byte-identical to pre-vLLM behaviour). Unknown
        # values fall back to the default rather than failing the run, so a
        # stale UI talking to a new server still works.
        backend_req = (req.get("backend") or "ollama").lower()
        if backend_req not in BACKEND_DISPATCH:
            backend_req = "ollama"
        algorithm_req = (req.get("algorithm") or "soundcode").lower()
        if algorithm_req not in ALGORITHM_DISPATCH:
            algorithm_req = "soundcode"
        mode_req = req.get("mode")
        # Backward compat: pre-mode clients sent `thinking: true` which maps
        # to RAW_THINK_INJECT.
        if mode_req is None:
            mode_req = ("raw_think_inject"
                        if bool(req.get("thinking", False))
                        else "raw")
        try:
            mode_requested = OrchestrationMode(mode_req)
        except ValueError:
            mode_requested = OrchestrationMode.RAW
    except Exception as e:
        await emit({"type": "error", "message": f"bad start request: {e!r}"})
        return

    # Coerce R1/T2/C3 → RAW for non-reasoning models. R1 silently no-ops on a
    # non-reasoning model anyway; T2/C3 would crash or thrash. TWO_PHASE
    # additionally requires a chat template — fall back to RAW if none.
    mode_active = mode_requested
    if mode_requested != OrchestrationMode.RAW and model not in REASONING_MODELS:
        mode_active = OrchestrationMode.RAW
    if mode_active == OrchestrationMode.TWO_PHASE and template_for_model(model) is None:
        mode_active = OrchestrationMode.RAW

    # When `instruct on rollback` is on AND the active orchestration mode is
    # one of the thinking modes, the model gets BOTH an instructive cargo-
    # error comment in the rolled-back prompt AND a fresh thinking phase
    # before re-attempting. The LlmServer detects this flag and skips its
    # one-shot mode-flip; DemoClient detects it and re-runs phase-1 for
    # TWO_PHASE specifically. R1 / C3 re-enter automatically via the next
    # `_open_stream` call once the mode stops flipping.
    reenter_thinking = (
        instruct
        and mode_active in (
            OrchestrationMode.RAW_THINK_INJECT,
            OrchestrationMode.TWO_PHASE,
            OrchestrationMode.CHAT_INSTRUCTED,
        )
    )

    # Open a per-run JSONL log so the entire event stream can be replayed
    # post-hoc for diagnosis.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    log_path = LOGS_DIR / f"run_{run_id}.jsonl"
    log_fp = log_path.open("w", encoding="utf-8")
    t_start = time.time()

    def _write_log(event: dict) -> None:
        try:
            line = json.dumps({"t": round(time.time() - t_start, 4), **event}, ensure_ascii=False)
            log_fp.write(line + "\n")
            log_fp.flush()
        except Exception:
            pass

    # Write a single "session" record at the top so the file is self-contained.
    _write_log({
        "type": "session",
        "run_id": run_id,
        "iso_time": datetime.datetime.now().isoformat(),
        "request": {
            "prompt": prompt,
            "language": language,
            "verifier": verifier,
            "mode_tag": mode_tag,
            "model": model,
            "token_delay_ms": token_delay_ms,
            "instruct": instruct,
            "mode_requested": mode_requested.value,
            "mode_active": mode_active.value,
            "reenter_thinking_on_rollback": reenter_thinking,
            "backend": backend_req,
            "algorithm": algorithm_req,
        },
    })

    # Tee the WebSocket emit through the log file.
    async def emit_and_log(event: dict) -> None:
        _write_log(event)
        await emit(event)

    # Resolve the dispatch entry for (language, mode_tag). If the user
    # picked an unsupported combo (shouldn't happen — the UI restricts to
    # the cells in the table) we surface a clean error rather than
    # crashing.
    dispatch = LANG_DISPATCH.get((language, mode_tag))
    if dispatch is None:
        await emit_and_log({
            "type": "status", "phase": "error",
            "message": f"unsupported combination: language={language}, mode={mode_tag}",
        })
        log_fp.close()
        return

    tool = dispatch["tool"]
    if not _tool_available(tool):
        hint = _INSTALL_HINTS.get(tool, f"required tool `{tool}` missing")
        await emit_and_log({
            "type": "status", "phase": "error",
            "message": hint,
        })
        log_fp.close()
        return

    # Materialize the workspace (idempotent for Rust's persistent dir;
    # fresh tempdir for the others). Then wire up checker + boundary.
    workspace_obj = dispatch["workspace_factory"]()
    workspace_path = workspace_obj.setup()
    boundary = dispatch["boundary_cls"]()

    await emit_and_log({
        "type": "status", "phase": "loading",
        "message": f"setting up {language} workspace at {workspace_path}",
        "log_path": str(log_path),
    })

    # Rust-specific warmup: the cargo incremental cache benefits from a
    # pre-flight `cargo check --offline`. Other languages don't need this.
    if language == "rust":
        await _warm_cargo(workspace_path)

    if mode_tag == "lsp":
        await emit_and_log({
            "type": "status", "phase": "loading",
            "message": f"starting LSP backend ({tool})",
        })
        try:
            checker = dispatch["checker_cls"](workspace=workspace_path)
        except Exception as e:
            await emit_and_log({
                "type": "status", "phase": "error",
                "message": f"could not construct {tool} checker: {e!r}",
            })
            log_fp.close()
            return
        try:
            await asyncio.to_thread(checker.start)
        except Exception as e:
            await emit_and_log({
                "type": "status", "phase": "error",
                "message": f"{tool} failed to start: {e!r}",
            })
            log_fp.close()
            return
    else:
        try:
            checker = dispatch["checker_cls"](workspace=workspace_path)
        except Exception as e:
            await emit_and_log({
                "type": "status", "phase": "error",
                "message": f"could not construct {tool} checker: {e!r}",
            })
            log_fp.close()
            return

    await emit_and_log({"type": "status", "phase": "loading", "message": f"connecting to ollama / {model}"})
    # The actual load (weights → VRAM) happens lazily inside Ollama on the
    # first generate call. For 120B+ models that's tens of seconds where the
    # UI would otherwise sit on "generating" with no tokens. Explicit warmup:
    # max_tokens = -1 = "predict until EOS or context limit" (Ollama's default).
    # Stop patterns: any of these substrings in the stream terminate generation.
    #   `\n}`        — column-0 closing brace (most common: closes the fn body)
    #   `\nfn `      — column-0 start of a new top-level fn (LM moved on)
    #   `\npub fn `  — column-0 start of a new public fn
    #   `\nimpl `    — column-0 start of an impl block
    #   `\n}\n\n//`  — closing brace followed by a blank-line comment
    # `RAW_THINK_INJECT` and `CHAT_INSTRUCTED` need looser stop patterns —
    # the model often emits a `\nfn ` (re-declared signature) just after the
    # boundary, which would prematurely terminate generation under the default
    # stop list. Drop the function-start patterns for those modes.
    if mode_active in (OrchestrationMode.RAW_THINK_INJECT, OrchestrationMode.CHAT_INSTRUCTED):
        stop_patterns = ["\n}\n\n//"]
    else:
        stop_patterns = ["\n}", "\nfn ", "\npub fn ", "\nimpl ", "\n}\n\n//"]
    # Build the logits processor first (most are nullary; if a future entry
    # needs config it can read req fields here). Then hand it to the
    # backend factory, which decides whether to actually wire it through.
    #
    # SemGuard takes a custom path: it needs a singleton SemGuardEvaluator
    # (the trained 1.3B classifier — expensive to construct), so we look it
    # up via `get_semguard_evaluator()` rather than going through the
    # nullary-factory dispatch entry. If the singleton can't be loaded
    # (checkpoint or backbone missing on disk), we still construct a
    # placeholder processor — `__call__` becomes a no-op + emits a warning
    # at construction time — and surface the degradation to the UI so the
    # user knows they're not actually getting SemGuard's rollback signal.
    try:
        if algorithm_req == "semguard":
            evaluator = await get_semguard_evaluator()
            if evaluator is None:
                # No checkpoint / no backbone — placeholder no-op processor.
                processor = SemGuardEvaluatorProcessor(
                    evaluator=None, tokenizer=None,
                )
                await emit_and_log({
                    "type": "status", "phase": "warning",
                    "message": (
                        "SemGuard checkpoint not loaded "
                        f"(missing {_SEMGUARD_CKPT_PATH.name} or backbone); "
                        "algorithm degraded to no-op — generation proceeds "
                        "without semantic rollback signals"
                    ),
                })
            else:
                threshold = float(req.get("threshold", 0.5))
                processor = SemGuardEvaluatorProcessor(
                    evaluator=evaluator,
                    tokenizer=evaluator._tokenizer,
                    threshold=threshold,
                )
                # Bind to the current event loop so the sampler can schedule
                # scoring coroutines via `run_coroutine_threadsafe`. The
                # DemoClient polls `processor.rollback_signals` between
                # tokens (see `demo_client.py`).
                await processor.bind_loop()
        else:
            processor = ALGORITHM_DISPATCH[algorithm_req]()
    except Exception as e:
        await emit_and_log({
            "type": "status", "phase": "error",
            "message": f"could not construct algorithm `{algorithm_req}`: {e!r}",
        })
        log_fp.close()
        return
    gen_config = GenerationConfig(
        max_tokens=-1,
        stop=stop_patterns,
        mode=mode_active,
        reenter_thinking_on_rollback=reenter_thinking,
    )
    try:
        llm = BACKEND_DISPATCH[backend_req](model, gen_config, processor)
    except Exception as e:
        await emit_and_log({
            "type": "status", "phase": "error",
            "message": f"could not construct backend `{backend_req}`: {e!r}",
        })
        log_fp.close()
        return

    await emit_and_log({
        "type": "status", "phase": "loading",
        "message": (
            f"loading model into memory: {model} "
            f"(backend={backend_req}, algorithm={algorithm_req})"
        ),
    })
    try:
        await llm.warmup()
    except Exception as e:
        await emit_and_log({"type": "status", "phase": "error", "message": f"model load failed: {e!r}"})
        await llm.close()
        return

    demo = DemoClient(
        llm=llm,
        checker=checker,
        config=DemoConfig(
            instruct_on_rollback=instruct,
            wall_budget_s=180.0,
            token_delay_s=token_delay_ms / 1000.0,
        ),
        emit=emit_and_log,
        boundary=boundary,
        # Per-language source tail (see `LANG_CLOSER` above). Threaded into
        # `DemoClient` so both the in-loop check (via `Code.function_closer`)
        # and the post-hoc final check (via `DemoClient.function_closer`)
        # use the language-appropriate closer instead of the Rust-only shim
        # that previously leaked into every language's checker source.
        function_closer=LANG_CLOSER.get(language, LANG_CLOSER["rust"]),
    )

    await emit_and_log({"type": "status", "phase": "ready", "message": "model ready, starting generation"})
    try:
        await demo.generate(prompt)
    except asyncio.CancelledError:
        await emit_and_log({"type": "status", "phase": "done", "message": "cancelled"})
        raise
    except Exception as e:
        import traceback
        await emit_and_log({"type": "error", "message": f"{e!r}", "traceback": traceback.format_exc()})
    finally:
        try:
            await llm.close()
        except Exception:
            pass
        if mode_tag == "lsp":
            try:
                await asyncio.to_thread(checker.stop)
            except Exception:
                pass
        # Workspace teardown: Rust's `RustWorkspace.teardown()` is a no-op
        # (persistent dir for incremental cache); the others remove their
        # tempdirs. Calling teardown unconditionally is safe.
        try:
            workspace_obj.teardown()
        except Exception:
            pass
        _write_log({"type": "session_end"})
        log_fp.close()


# ─── App factory ──────────────────────────────────────────────────────


@web.middleware
async def _no_cache_middleware(request: web.Request, handler):
    """Send `Cache-Control: no-store` on every response so neither the
    Cloudflare edge nor the browser caches anything. We want every request
    to reach this machine — when a CSS or JS change ships, viewers see it
    on the next refresh, not 4 hours later when the edge happens to
    revalidate.

    Header semantics (this trips people up — including me, the first time):
      - `no-cache`  = "you may store it, but revalidate via
                     If-None-Match before serving the cached copy."
                     Cloudflare's edge still STORES the response and
                     reports `cf-cache-status: MISS/HIT`.
      - `no-store`  = "don't store it at all." Cloudflare's edge skips
                     caching entirely and reports `cf-cache-status: BYPASS`.
                     This is what we want for the dev demo — no edge
                     caching, no browser caching, every viewer always
                     gets the live origin.

    Pinned with `private, max-age=0, must-revalidate` for older clients
    that don't honor `no-store` alone (rare, mostly belt-and-suspenders).
    """
    resp = await handler(request)
    resp.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, private, max-age=0"
    )
    resp.headers["Pragma"] = "no-cache"   # HTTP/1.0 legacy
    resp.headers["Expires"] = "0"         # ditto
    return resp


def make_app() -> web.Application:
    app = web.Application(middlewares=[_no_cache_middleware])
    app.router.add_get("/", index)
    app.router.add_get("/api/prompts", list_prompts)
    app.router.add_get("/api/prompts/{id}", get_prompt)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", STATIC_DIR, show_index=False)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = make_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
