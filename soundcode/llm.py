"""LlmServer — thin async wrapper over Ollama's streaming completion API.

API matches the producer-consumer loop from draft-plan-0.md §3.2:

  llm = LlmServer(model="...")
  llm.set_prompt(prompt)
  while llm.has_next():
      token = await llm.next()
  llm.abort_current_stream()
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = -1   # -1 = predict until EOS / context limit (Ollama default)
    top_p: float = 0.95
    stop: list[str] = field(default_factory=lambda: ["\n}"])


class LlmServer:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        config: GenerationConfig | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.config = config or GenerationConfig()
        self.timeout = timeout
        self._prompt: str = ""
        self._client: httpx.AsyncClient | None = None
        self._stream_ctx = None
        self._stream_iter: AsyncIterator[str] | None = None
        self._exhausted = True  # nothing to read until set_prompt() + first next()
        self._tokens_in_current_stream = 0

    # ─── prompt control ────────────────────────────────────────────────

    def set_prompt(self, prompt: str) -> None:
        """Set the prompt to use on the *next* stream open."""
        self._prompt = prompt
        self._exhausted = False  # there's a fresh stream waiting to be opened

    def has_next(self) -> bool:
        """True if the LM might emit another token."""
        return not self._exhausted

    async def next(self) -> str:
        """Get the next token (text chunk) from the LM.

        Opens a stream on first call after set_prompt(). Returns "" only at EOS.
        """
        if self._stream_iter is None:
            await self._open_stream()
        assert self._stream_iter is not None
        try:
            token = await anext(self._stream_iter)
        except StopAsyncIteration:
            self._exhausted = True
            await self._close_stream()
            return ""
        self._tokens_in_current_stream += 1
        return token

    async def abort_current_stream(self) -> None:
        """Close the in-flight HTTP stream. Next has_next() returns False until
        set_prompt() is called again."""
        self._exhausted = True
        await self._close_stream()

    async def warmup(self) -> None:
        """Force Ollama to load this model into VRAM before generation starts.

        Sends `POST /api/generate` with an empty prompt and `stream=false`,
        which Ollama treats as a "load this model" call and returns once the
        weights are resident. Without this, the load delay (seconds for small
        models, tens of seconds to minutes for 120B+) happens lazily on the
        first token of the real stream — the UI sees `generating` while
        nothing is actually flowing yet. Surfacing the load as its own phase
        gives the frontend something explicit to display.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        resp = await self._client.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": "", "stream": False},
        )
        resp.raise_for_status()

    # ─── internals ─────────────────────────────────────────────────────

    async def _open_stream(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        payload = {
            "model": self.model,
            "prompt": self._prompt,
            "raw": True,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
                "stop": self.config.stop,
            },
        }
        self._stream_ctx = self._client.stream(
            "POST", f"{self.base_url}/api/generate", json=payload,
        )
        resp = await self._stream_ctx.__aenter__()
        resp.raise_for_status()
        self._stream_iter = _iter_ollama_tokens(resp)
        self._tokens_in_current_stream = 0

    async def _close_stream(self) -> None:
        if self._stream_iter is not None:
            self._stream_iter = None
        if self._stream_ctx is not None:
            try:
                await self._stream_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._stream_ctx = None

    async def close(self) -> None:
        await self._close_stream()
        if self._client is not None:
            await self._client.aclose()
            self._client = None


async def _iter_ollama_tokens(resp: httpx.Response) -> AsyncIterator[str]:
    async for line in resp.aiter_lines():
        if not line.strip():
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("response"):
            yield evt["response"]
        if evt.get("done"):
            break
