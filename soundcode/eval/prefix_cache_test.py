"""Verify whether Ollama survives prefix-cache across stream abort.

Procedure:
  1. Send a long-prefix completion request, abort mid-stream.
  2. Re-send the same prefix; measure TTFT.
  3. Repeat for a fresh prefix as control.

If the warm-second-call TTFT is materially lower than the cold-third-call
TTFT, prefix-caching is in play; otherwise re-prompting after abort costs
the same as a fresh prompt.
"""

from __future__ import annotations

import asyncio
import time
from soundcode.llm import LlmServer, GenerationConfig


LONG_PROMPT = (
    "// A long, varied prompt to exercise the prefix cache. " * 20
    + "\n\nfn fibonacci(n: u32) -> u64 {\n    let mut a: u64 = 0;\n"
)


async def time_first_token(llm: LlmServer, prompt: str) -> tuple[float, str]:
    llm.set_prompt(prompt)
    t0 = time.perf_counter()
    tok = await llm.next()
    return time.perf_counter() - t0, tok


async def main() -> None:
    llm = LlmServer(
        model="mistral-small3.2:24b",
        config=GenerationConfig(max_tokens=32, stop=[]),
    )

    print("Trial 1: cold (first ever call with this prompt)")
    ttft_cold, _ = await time_first_token(llm, LONG_PROMPT)
    print(f"  TTFT = {ttft_cold:.3f}s")
    # consume a few tokens then abort
    for _ in range(8):
        await llm.next()
    await llm.abort_current_stream()

    # Same prompt: should hit cache if cache exists
    print("\nTrial 2: warm (same prompt, after abort)")
    ttft_warm, _ = await time_first_token(llm, LONG_PROMPT)
    print(f"  TTFT = {ttft_warm:.3f}s")
    for _ in range(8):
        await llm.next()
    await llm.abort_current_stream()

    # Slightly different prompt: should NOT hit cache for the differing part
    print("\nTrial 3: fresh (similar prompt with different tail)")
    fresh_prompt = LONG_PROMPT + "\n    let mut b: u64 = 1;\n"
    ttft_fresh, _ = await time_first_token(llm, fresh_prompt)
    print(f"  TTFT = {ttft_fresh:.3f}s")

    await llm.close()

    print("\n--- summary ---")
    print(f"cold:  {ttft_cold:.3f}s")
    print(f"warm:  {ttft_warm:.3f}s  ({100*(1-ttft_warm/ttft_cold):+.1f}% vs cold)")
    print(f"fresh: {ttft_fresh:.3f}s ({100*(1-ttft_fresh/ttft_cold):+.1f}% vs cold)")
    if ttft_warm < 0.5 * ttft_cold:
        print("\nVERDICT: prefix cache is active — re-prompts after abort are cheap.")
    elif ttft_warm < 0.8 * ttft_cold:
        print("\nVERDICT: some prefix-cache benefit observable, but not dominant.")
    else:
        print("\nVERDICT: NO meaningful prefix-cache benefit on Ollama.")


if __name__ == "__main__":
    asyncio.run(main())
