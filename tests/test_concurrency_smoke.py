"""Concurrency smoke tests.

Three flavours:
  - Two concurrent generations on the Ollama backend (no GPU; uses a
    pure stub for LlmServer since the real Ollama isn't always up in CI).
  - Two concurrent generations on the vLLM backend (slow+gpu).
  - Two concurrent WebSocket connections to a test aiohttp server bound to
    port 8084 (no GPU — uses the stubbed-backend path via dispatch).

The vLLM AsyncLLMEngine batches concurrent requests naturally; the goal is
to confirm DemoClient and the WS layer don't introduce a single-stream
serialisation bottleneck.
"""

from __future__ import annotations

import asyncio
import os
import socket
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import pytest

from soundcode.llm import GenerationConfig, OrchestrationMode, Token


# ─── GPU gating + helpers ─────────────────────────────────────────────


def _gpu_tests_enabled() -> bool:
    return os.environ.get("RUN_GPU_TESTS", "") == "1"


gpu_required = pytest.mark.skipif(
    not _gpu_tests_enabled(),
    reason="set RUN_GPU_TESTS=1 to run GPU-bound concurrency tests",
)


TEST_MODEL = os.environ.get(
    "SOUNDCODE_VLLM_TEST_MODEL",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
)


# Port 8084 is reserved for this agent (8080/8082/8083 are taken).
WS_TEST_PORT = 8084


# ─── stub backends ────────────────────────────────────────────────────


class _FakeLlmStream:
    """Reusable stub mimicking LlmServer/VllmBackend's set_prompt/has_next/next
    surface. Each `set_prompt` resets the cursor; the supplied token list is
    streamed once."""

    def __init__(self, tokens: list[str]):
        self._tokens = list(tokens)
        self._pos = 0
        self._exhausted = False
        self.config = GenerationConfig(mode=OrchestrationMode.RAW)
        self.think_budget_exceeded = False

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt
        self._pos = 0
        self._exhausted = False

    def has_next(self) -> bool:
        return not self._exhausted

    async def next(self) -> Token:
        # Small await so concurrent streams actually interleave on the loop.
        await asyncio.sleep(0)
        if self._pos >= len(self._tokens):
            self._exhausted = True
            return Token("", "code")
        t = self._tokens[self._pos]
        self._pos += 1
        return Token(t, "code")

    async def abort_current_stream(self) -> None:
        self._exhausted = True

    async def warmup(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def stream_thinking_phase(self, prompt: str):
        return
        yield  # pragma: no cover


class _FakeChecker:
    async def check(self, source: str) -> list:
        return []


# ─── test_ollama_backend_two_concurrent_streams ───────────────────────


@pytest.mark.asyncio
async def test_ollama_backend_two_concurrent_streams():
    """Two DemoClient.generate() coroutines driving independent stub LlmServer
    instances must both complete via asyncio.gather. Verifies DemoClient
    has no shared mutable global state.

    We use stubs (not real Ollama) so the test doesn't require a live
    Ollama server. The race-conditions we're hunting are in DemoClient's
    event loop, not in Ollama's HTTP server.
    """
    from soundcode.web.demo_client import DemoClient, DemoConfig

    async def _run_one(client_id: int) -> list[dict]:
        backend = _FakeLlmStream([f"    let id = {client_id};\n", "\n}"])
        events: list[dict] = []

        async def emit(ev: dict) -> None:
            events.append(ev)
            # Slow the emit a tiny bit so the two coroutines actually
            # interleave on the event loop rather than running serially.
            await asyncio.sleep(0)

        demo = DemoClient(
            llm=backend, checker=_FakeChecker(),
            config=DemoConfig(token_delay_s=0, repetition_window=0,
                              wall_budget_s=10, max_continuations=0),
            emit=emit,
        )
        await demo.generate(f"fn f{client_id}() -> i32 {{")
        return events

    # Two coroutines launched via gather — both must reach the "final" event.
    results = await asyncio.gather(_run_one(0), _run_one(1))
    for cid, events in enumerate(results):
        event_types = [e["type"] for e in events]
        assert "final" in event_types, (
            f"client {cid} did not reach final; types={event_types[-10:]}"
        )
        emitted_texts = [e["text"] for e in events if e["type"] == "token_emitted"]
        # Each client must have seen its own id token (no cross-talk).
        assert any(f"let id = {cid};" in t for t in emitted_texts), (
            f"client {cid} did not see its own token in {emitted_texts}"
        )
        # And NOT seen the other client's id token.
        other = 1 - cid
        assert not any(f"let id = {other};" in t for t in emitted_texts), (
            f"client {cid} saw the OTHER client's token (cross-talk): "
            f"{emitted_texts}"
        )


# ─── test_vllm_backend_two_concurrent_streams ─────────────────────────


@pytest.mark.slow
@pytest.mark.gpu
@gpu_required
@pytest.mark.asyncio
async def test_vllm_backend_two_concurrent_streams():
    """Two separate VllmBackend instances driving generate() in parallel
    must both complete. vLLM's AsyncLLMEngine handles batching internally;
    if DemoClient introduces a single-stream lock, this would deadlock.

    Note: we use two *separate* backend instances rather than two streams
    against one backend, because the current VllmBackend stores a single
    `_stream_iter` — sharing would race. Two backend instances is the
    realistic UI scenario (two browser tabs)."""
    from soundcode.vllm_backend import VllmBackend
    from soundcode.web.demo_client import DemoClient, DemoConfig

    async def _run_one(prompt: str) -> str:
        backend = VllmBackend(
            model=TEST_MODEL,
            config=GenerationConfig(max_tokens=32, stop=["\n}"]),
            gpu_memory_utilization=0.15,
            max_model_len=512,
        )
        events: list[dict] = []

        async def emit(ev: dict) -> None:
            events.append(ev)

        try:
            await backend.warmup()
            demo = DemoClient(
                llm=backend, checker=_FakeChecker(),
                config=DemoConfig(token_delay_s=0, repetition_window=0,
                                  wall_budget_s=60, max_continuations=0),
                emit=emit,
            )
            await demo.generate(prompt)
        finally:
            await backend.close()
        # Return concatenated code tokens for sanity.
        return "".join(
            e["text"] for e in events
            if e["type"] == "token_emitted" and e.get("kind") == "code"
        )

    out_a, out_b = await asyncio.gather(
        _run_one("fn add(a: i32, b: i32) -> i32 {"),
        _run_one("fn mul(a: i32, b: i32) -> i32 {"),
    )
    assert out_a, "concurrent stream A produced no code"
    assert out_b, "concurrent stream B produced no code"


# ─── test_web_demo_handles_two_simultaneous_ws_connections ────────────


def _port_free(port: int) -> bool:
    """Best-effort: True iff `port` is bindable on 127.0.0.1.

    Uses SO_REUSEADDR so a recent TIME_WAIT socket (left behind by the
    previous test on the same port) doesn't make us spuriously skip —
    aiohttp's TCPSite also binds with REUSEADDR, so this matches the
    actual server behaviour."""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


@asynccontextmanager
async def _ws_test_server():
    """Spin up the aiohttp app under test on WS_TEST_PORT. Yields the URL
    base. Cleans up gracefully on exit. Uses SO_REUSEADDR so TIME_WAIT
    sockets from a previous test don't block the bind."""
    if not _port_free(WS_TEST_PORT):
        pytest.skip(
            f"port {WS_TEST_PORT} is already in use — another agent or a "
            f"stale server is holding it. This test reserves 8084 by "
            f"convention; verify nothing else has it."
        )
    from aiohttp import web
    from soundcode.web.server import make_app

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    # reuse_address=True lets us bind even when a previous test on the same
    # port left a connection in TIME_WAIT (~60s linger by default).
    site = web.TCPSite(runner, "127.0.0.1", WS_TEST_PORT,
                       reuse_address=True)
    await site.start()
    try:
        yield f"http://127.0.0.1:{WS_TEST_PORT}"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_web_demo_handles_two_simultaneous_ws_connections():
    """Open two WebSocket connections to the demo server's /ws endpoint
    in parallel. Both should accept; both should be able to send a
    `stop` message and receive at least one event back (the server's
    "done: stopped" status). Verifies the ws_handler isn't serialised on
    some module-level singleton.

    We don't send a `start` event with a real prompt — that would require
    a live Ollama / vLLM backend. A `stop` against an idle session is a
    cheap protocol-level smoke test of "the WS layer doesn't block one
    connection on another."
    """
    async with _ws_test_server() as url:
        ws_url = url.replace("http://", "ws://") + "/ws"

        async def _one_connection(client_id: int) -> list[dict]:
            received: list[dict] = []
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(ws_url) as ws:
                    # Send a stop while idle — the handler emits a
                    # `status: done: stopped` event in response. This is
                    # the cheapest way to verify two-way liveness without
                    # touching a model.
                    await ws.send_json({"type": "stop"})
                    # Wait briefly for the status response.
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=5)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            import json as _json
                            received.append(_json.loads(msg.data))
                    except asyncio.TimeoutError:
                        pass
                    await ws.close()
            return received

        results = await asyncio.gather(
            _one_connection(0), _one_connection(1),
            return_exceptions=True,
        )
        # Both connections must have completed cleanly (no exceptions).
        for i, r in enumerate(results):
            assert not isinstance(r, BaseException), (
                f"connection {i} raised: {r!r}"
            )
        # And each must have received the `done: stopped` status event.
        for i, events in enumerate(results):
            statuses = [
                e for e in events
                if e.get("type") == "status" and "stopped" in e.get("message", "")
            ]
            assert statuses, (
                f"connection {i} did not receive a stop ack; events={events}"
            )


@pytest.mark.asyncio
async def test_web_demo_ws_handles_bad_json_gracefully():
    """Send malformed JSON to /ws — the server should emit an error event
    and keep the connection alive (so the user can retry). Regression guard
    that the JSON-decode path doesn't crash the entire connection."""
    async with _ws_test_server() as url:
        ws_url = url.replace("http://", "ws://") + "/ws"
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(ws_url) as ws:
                await ws.send_str("not valid json {{{")
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                except asyncio.TimeoutError:
                    pytest.fail("server did not respond to malformed JSON")
                assert msg.type == aiohttp.WSMsgType.TEXT
                import json as _json
                evt = _json.loads(msg.data)
                assert evt.get("type") == "error", evt
                assert "bad json" in evt.get("message", "").lower()
                # Connection still alive — send a stop and read its ack.
                await ws.send_json({"type": "stop"})
                try:
                    msg2 = await asyncio.wait_for(ws.receive(), timeout=5)
                    assert msg2.type == aiohttp.WSMsgType.TEXT
                    evt2 = _json.loads(msg2.data)
                    assert evt2.get("type") == "status"
                except asyncio.TimeoutError:
                    pytest.fail(
                        "server died after malformed JSON — should stay alive"
                    )
                await ws.close()
