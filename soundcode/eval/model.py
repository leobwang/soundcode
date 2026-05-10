"""LLM client for code generation via Ollama's native API."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=lambda: ["\n}"])
    top_p: float = 0.95


@dataclass
class ModelClient:
    """Client for Ollama's API, supporting both raw completion and chat modes."""

    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 300.0
    mode: str = "raw"  # "raw" for completion-style, "chat" for chat-style

    async def complete(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> str:
        """Generate a completion for the given prompt."""
        if config is None:
            config = GenerationConfig()

        if self.mode == "raw":
            return await self._complete_raw(prompt, config)
        else:
            return await self._complete_chat(prompt, config)

    async def _complete_raw(
        self, prompt: str, config: GenerationConfig
    ) -> str:
        """Completion via Ollama native API with raw=True (no template)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "stop": config.stop,
                "top_p": config.top_p,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["response"]

    async def _complete_chat(
        self, prompt: str, config: GenerationConfig
    ) -> str:
        """Completion via chat API with code extraction from response."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Rust programming expert. Complete the given function body. "
                    "Output ONLY the function body code. Do not include the function signature, "
                    "closing brace, markdown formatting, or explanations."
                ),
            },
            {
                "role": "user",
                "content": f"Complete the body of this Rust function:\n\n{prompt}",
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"]
            return self._extract_code(content, prompt)

    def _extract_code(self, response: str, prompt: str) -> str:
        """Extract function body from a chat model response."""
        # Try to find code in markdown blocks
        code_blocks = re.findall(r"```(?:rust)?\s*\n(.*?)```", response, re.DOTALL)
        if code_blocks:
            code = code_blocks[0].strip()
        else:
            code = response.strip()

        # If the model repeated the full function, extract just the body
        # Look for the opening brace from the prompt's function signature
        lines = code.split("\n")
        body_lines = []
        in_body = False
        brace_depth = 0

        for line in lines:
            if not in_body:
                if line.strip().endswith("{"):
                    in_body = True
                    brace_depth = 1
                    continue
                # If the line is just code (no fn signature), treat as body
                if not line.strip().startswith("fn ") and not line.strip().startswith("///"):
                    body_lines.append(line)
            else:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    break
                body_lines.append(line)

        if body_lines:
            return "\n".join(body_lines)
        return code

    async def stream_raw(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens (text chunks) from Ollama's raw completion endpoint.

        Yields each text chunk as it arrives. Caller can break out of the
        loop to abort generation (closes the HTTP connection).
        """
        if config is None:
            config = GenerationConfig()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "stop": config.stop,
                "top_p": config.top_p,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        return

    async def complete_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        *,
        concurrency: int = 4,
    ) -> list[str]:
        """Generate completions for multiple prompts with bounded concurrency."""
        sem = asyncio.Semaphore(concurrency)
        results: list[str | None] = [None] * len(prompts)

        async def _do(idx: int, prompt: str) -> None:
            async with sem:
                results[idx] = await self.complete(prompt, config)

        async with asyncio.TaskGroup() as tg:
            for i, p in enumerate(prompts):
                tg.create_task(_do(i, p))

        return results  # type: ignore
