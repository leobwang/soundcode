"""Trace a single problem through the rollback loop with verbose output."""

import asyncio
import sys
from pathlib import Path

from soundcode.cargo_check import CargoChecker
from soundcode.client import CodeClient
from soundcode.code import Code
from soundcode.eval.dataset import load_rust_humaneval
from soundcode.llm import LlmServer, GenerationConfig


async def main(problem_name: str) -> None:
    problems = load_rust_humaneval()
    prob = next(p for p in problems if problem_name in p.name)
    print("=== prompt last 200 ===")
    print(repr(prob.prompt[-200:]))
    print()

    workspace = Path("cargo_workspaces/trace")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "Cargo.toml").write_text(
        '[package]\nname = "scratch"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "scratch"\npath = "src/main.rs"\n'
    )
    (workspace / "src").mkdir(exist_ok=True)
    (workspace / "src" / "main.rs").write_text("fn main() {}\n")

    llm = LlmServer(model="mistral-small3.2:24b",
                    config=GenerationConfig(max_tokens=400, stop=["\n}"]))
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix=prob.prompt, suffix="}", checker=chk)

    # Manually step through to see each boundary check.
    llm.set_prompt(prob.prompt)
    rb = 0
    while rb < 3:  # only first few rollbacks
        last_check_pos = 0
        while llm.has_next():
            tok = await llm.next()
            if not tok:
                break
            code.append(tok)
            if code.at_boundary():
                print(f"\n--- boundary at offset {len(code.content)}, content tail: {repr(code.content[-60:])} ---")
                result = await code.check()
                print(f"verdict: {result.verdict.value}, "
                      f"diags: {len(result.diagnostics)}")
                for d in result.diagnostics[:3]:
                    print(f"  [{d.category.value}] {d.code}: {d.message[:120]}")
                if result.is_error:
                    print(f"\n*** ROLLBACK #{rb+1} ***")
                    survivor = code.content_up_to(code.ckpt[-1])
                    print(f"survivor: {repr(survivor)}")
                    rb += 1
                    await llm.abort_current_stream()
                    llm.set_prompt(prob.prompt + survivor)
                    code.rollback()
                    break
        else:
            # Stream ended cleanly
            print(f"\n=== Stream ended cleanly. Final completion ({len(code.content)} chars) ===")
            print(code.content)
            await llm.close()
            return
    await llm.close()
    print(f"\n=== Stopped after {rb} rollbacks. Final content ({len(code.content)} chars):")
    print(repr(code.content[:300]))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
