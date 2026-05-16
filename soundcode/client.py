"""CodeClient — the producer-consumer loop from draft-plan-0.md §3.2.

  tasks = deque()
  while True:
      if llm.has_next():
          token = await llm.next()       # blocking on LM
          code.append(token)             # sync
          if code.at_boundary():
              tasks.append(create_task(code.check()))
      while tasks and tasks[0].done():
          result = tasks.popleft().result()
          if result.is_error:
              for t in tasks: t.cancel()
              tasks.clear()
              await llm.abort_current_stream()
              llm.set_prompt(prompt + content_up_to(ckpt[-1]) + format_error(...))
              code.rollback()
              break
      if not llm.has_next():
          if not tasks: return code.content
          await tasks[0]
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from soundcode.cargo_check import CargoChecker
from soundcode.code import Code, State
from soundcode.llm import LlmServer, GenerationConfig


@dataclass
class RunResult:
    name: str
    arm: str
    completion: str            # the buffer the LM produced
    compiled: bool             # post-hoc cargo check on (prompt + completion + tests)
    passed: bool               # cargo test passed
    wall_clock_s: float
    tokens_emitted: int        # total over all attempts
    tokens_kept: int           # len(completion) measured in chars (proxy)
    rollback_count: int        # how many times we rolled back
    lsp_calls: int             # how many cargo-check calls during generation
    final_diagnostics: list[str] = field(default_factory=list)


class CodeClient:
    def __init__(
        self,
        llm: LlmServer,
        checker: CargoChecker,
        *,
        instruct_on_rollback: bool = False,
        max_rollbacks: int = 8,
        wall_budget_s: float = 60.0,
    ) -> None:
        self.llm = llm
        self.checker = checker
        self.instruct_on_rollback = instruct_on_rollback
        self.max_rollbacks = max_rollbacks
        self.wall_budget_s = wall_budget_s

    async def generate(
        self,
        prompt: str,
        function_prefix: str,
        function_suffix: str = "\n}\n\nfn main() {}\n",
    ) -> tuple[str, dict]:
        """Generate code with the rollback loop.

        Returns (final_completion_string, telemetry_dict).
        """
        code = Code(prefix=function_prefix, suffix=function_suffix, checker=self.checker)
        tasks: deque[asyncio.Task] = deque()
        self.llm.set_prompt(prompt)
        rollbacks = 0
        t0 = time.perf_counter()

        while True:
            # Time budget.
            if time.perf_counter() - t0 > self.wall_budget_s:
                for t in tasks:
                    t.cancel()
                tasks.clear()
                await self.llm.abort_current_stream()
                break

            # 1. produce
            if self.llm.has_next():
                token = await self.llm.next()
                if token:
                    code.append(token)
                    if code.at_boundary():
                        tasks.append(asyncio.create_task(code.check()))

            # 2. consume
            while tasks and tasks[0].done():
                result = tasks.popleft().result()
                if result.is_error:
                    rollbacks += 1
                    # Cancel pending checks (their content is about to disappear).
                    for t in tasks:
                        t.cancel()
                    # Await cancellations so they don't fire mutations later.
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    tasks.clear()
                    await self.llm.abort_current_stream()
                    # Build the new prompt: original + verified prefix [+ error msg].
                    last_ckpt = code.ckpt[-1]
                    survivor = code.content_up_to(last_ckpt)
                    if self.instruct_on_rollback:
                        suffix_msg = (
                            "\n// previous attempt failed cargo check with:\n"
                            + "\n".join(f"// {line}" for line in result.message.splitlines())
                            + "\n"
                        )
                        new_prompt = prompt + survivor + suffix_msg
                    else:
                        new_prompt = prompt + survivor
                    self.llm.set_prompt(new_prompt)
                    code.rollback()
                    if rollbacks >= self.max_rollbacks:
                        break
                    break  # restart producer
            else:
                pass

            if rollbacks >= self.max_rollbacks:
                for t in tasks:
                    t.cancel()
                tasks.clear()
                break

            # 3. termination / flush
            if not self.llm.has_next():
                if not tasks:
                    break
                try:
                    await tasks[0]
                except asyncio.CancelledError:
                    pass
                # loop back to drain

        # Done generating. Drain any remaining task results.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        telemetry = {
            "rollbacks": rollbacks,
            "tokens_seen": code.tokens_seen,
            "lsp_calls": code.lsp_calls,
            "checkpoints": len(code.ckpt) - 1,
        }
        return code.content, telemetry
