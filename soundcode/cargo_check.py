"""Run `cargo check --message-format=json` on a Rust source string.

A persistent Cargo workspace at `rust-samples/sample0` is used; we overwrite
`src/main.rs` with each candidate, run cargo check, parse JSON diagnostics.

This is faster than rust-analyzer push diagnostics for our cadence (~300ms
warm) and gives us full rustc diagnostics — no demotion / writing-edge
heuristics needed.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from soundcode.code import Category, Diagnostic


@dataclass
class CargoChecker:
    workspace: Path
    file_in_workspace: str = "src/main.rs"
    timeout_s: float = 15.0
    _lock: asyncio.Lock | None = None

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    async def check(self, source: str) -> list[Diagnostic]:
        """Write `source` to src/main.rs and run cargo check."""
        # Serialize: there's only one src/main.rs per workspace.
        async with self._lock:
            target = self.workspace / self.file_in_workspace
            target.write_text(source)
            proc = await asyncio.create_subprocess_exec(
                "cargo", "check",
                "--message-format=json",
                "--offline",
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "CARGO_TERM_COLOR": "never"},
            )
            try:
                stdout, _stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return []

        return _parse_cargo_output(stdout.decode("utf-8", errors="replace"))


def _parse_cargo_output(stdout: str) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("reason") != "compiler-message":
            continue
        msg = evt.get("message") or {}
        level = msg.get("level", "")
        # rustc levels: error, warning, note, help. We map:
        #   error -> BLOCKING (the classifier refines to INCOMPLETE later)
        #   warning -> NON_BLOCKING
        #   anything else -> skip
        if level == "error":
            cat = Category.BLOCKING
        elif level == "warning":
            cat = Category.NON_BLOCKING
        else:
            continue

        code = None
        cobj = msg.get("code")
        if isinstance(cobj, dict):
            code = cobj.get("code")
        text = msg.get("message", "")
        spans = msg.get("spans") or []
        primary = next((s for s in spans if s.get("is_primary")), None)
        if primary is None and spans:
            primary = spans[0]
        line = col = None
        if primary:
            line = primary.get("line_start")
            col = primary.get("column_start")
        out.append(Diagnostic(category=cat, message=text, code=code, line=line, column=col))
    return out
