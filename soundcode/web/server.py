"""Web-demo server for the rollback generator.

Single aiohttp server:
  - GET  /             → index.html
  - GET  /static/*     → static assets
  - GET  /api/prompts  → list of bundled HumanEval prompts (id + truncated docstring)
  - GET  /api/prompts/{id} → full prompt text
  - WS   /ws           → bidirectional protocol (see Events below)

Run:
    uv run python -m soundcode.web.server [--host 127.0.0.1] [--port 8765]

Events client→server:
  {"type": "start", "prompt": str, "verifier": "cargo"|"ra",
   "model": str, "token_delay_ms": int, "instruct": bool}
  {"type": "stop"}

Events server→client (see DemoClient for the full set):
  {"type": "status", "phase": "loading"|"ready"|"generating"|"done", "message": str}
  {"type": "reset_buffer"}
  {"type": "token_emitted", "text": str, "offset": int}
  {"type": "check_started", "offset": int}
  {"type": "verdict", "verdict": "ok"|"error"|"inactive", "diagnostics": [...], "offset": int}
  {"type": "checkpoint", "offset": int}
  {"type": "rollback", "to_offset": int, "discarded": str, "rollback_count": int}
  {"type": "final", "content": str, "rollback_count": int}
  {"type": "error", "message": str}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from aiohttp import web, WSMsgType

from soundcode.cargo_check import CargoChecker
from soundcode.llm import LlmServer, GenerationConfig
from soundcode.eval.dataset import load_rust_humaneval
from soundcode.web.demo_client import DemoClient, DemoConfig
from soundcode.web.extra_prompts import CUSTOM_PROBLEMS


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_WS = PROJECT_ROOT / "cargo_workspaces" / "web_demo"
LOGS_DIR = PROJECT_ROOT / "results" / "web_demo"


# ─── workspace + checker wiring ───────────────────────────────────────


def _ensure_workspace() -> Path:
    DEMO_WS.mkdir(parents=True, exist_ok=True)
    (DEMO_WS / "Cargo.toml").write_text(
        '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "demo"\npath = "src/main.rs"\n'
    )
    (DEMO_WS / "src").mkdir(exist_ok=True)
    (DEMO_WS / "src" / "main.rs").write_text("fn main() {}\n")
    return DEMO_WS


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
    global _PROBLEMS_CACHE
    if _PROBLEMS_CACHE is None:
        _PROBLEMS_CACHE = load_rust_humaneval()
    return _PROBLEMS_CACHE


async def index(request):
    return web.FileResponse(STATIC_DIR / "index.html")


def _custom_first(extra=CUSTOM_PROBLEMS) -> list:
    """Return a single combined list: custom problems first, then HumanEval."""
    return list(extra) + list(_problems())


async def list_prompts(request):
    out = []
    for p in CUSTOM_PROBLEMS:
        out.append({
            "id": p.name,
            "title": p.title[:90],
            "prompt_len": len(p.prompt),
            "is_custom": True,
        })
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
    for p in CUSTOM_PROBLEMS:
        if p.name == pid:
            return web.json_response({"id": p.name, "prompt": p.prompt})
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


async def _run_generation(req: dict, emit) -> None:
    import time, uuid, datetime
    try:
        prompt = req["prompt"]
        verifier = req.get("verifier", "cargo")
        model = req.get("model", "mistral-small3.2:24b")
        token_delay_ms = int(req.get("token_delay_ms", 0))
        instruct = bool(req.get("instruct", False))
    except Exception as e:
        await emit({"type": "error", "message": f"bad start request: {e!r}"})
        return

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
            "verifier": verifier,
            "model": model,
            "token_delay_ms": token_delay_ms,
            "instruct": instruct,
        },
    })

    # Tee the WebSocket emit through the log file.
    async def emit_and_log(event: dict) -> None:
        _write_log(event)
        await emit(event)

    workspace = _ensure_workspace()
    await emit_and_log({"type": "status", "phase": "loading", "message": "warming cargo cache", "log_path": str(log_path)})
    await _warm_cargo(workspace)

    if verifier == "ra":
        from soundcode.ra_check import RustAnalyzerChecker
        await emit_and_log({"type": "status", "phase": "loading", "message": "starting rust-analyzer"})
        checker = RustAnalyzerChecker(workspace=workspace, settle_s=1.5)
        await asyncio.to_thread(checker.start)
    else:
        checker = CargoChecker(workspace=workspace)

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
    llm = LlmServer(
        model=model,
        config=GenerationConfig(
            max_tokens=-1,
            stop=["\n}", "\nfn ", "\npub fn ", "\nimpl ", "\n}\n\n//"],
        ),
    )

    await emit_and_log({"type": "status", "phase": "loading", "message": f"loading model into memory: {model}"})
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
        if verifier == "ra":
            try:
                await asyncio.to_thread(checker.stop)
            except Exception:
                pass
        _write_log({"type": "session_end"})
        log_fp.close()


# ─── App factory ──────────────────────────────────────────────────────


def make_app() -> web.Application:
    app = web.Application()
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
