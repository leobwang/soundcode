"""Run all 4 arms — raw, MGD, rollback, MGD+rollback — using HuggingFace
transformers (NOT vLLM). This matches MGD's reference implementation
exactly: in-process generation + `LogitsProcessor` callable.

Tradeoffs vs vllm_runner.py:
  + No subprocess / IPC. The LogitsProcessor runs in the same process
    as the rest of the pilot, so monitor state and telemetry actually
    persist across token-generation steps.
  + Same API as MGD's reference (microsoft/monitors4codegen/hf_gen.py).
  - Slower per-token generation than vLLM (no continuous batching).
  - Doesn't use prefix caching.

For a 1.5B-class model on a Blackwell GPU these tradeoffs are minor.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

from soundcode.cargo_check import CargoChecker
from soundcode.code import Code, State
from soundcode.eval.dataset import RustProblem, load_rust_humaneval
from soundcode.mgd import (
    DereferenceMonitor,
    MgdTelemetry,
    RustAnalyzerCompletionsProvider,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "pilot"
TEST_WORKSPACE = PROJECT_ROOT / "test_workspace"


@dataclass
class HfRunResult:
    name: str
    arm: str
    completion: str
    compiled: bool
    passed: bool
    wall_clock_s: float
    tokens_emitted: int
    tokens_kept: int
    rollback_count: int
    lsp_calls: int
    mgd_triggers: int
    mgd_constrained_steps: int
    mgd_empty_masks: int
    mgd_ra_calls: int


class MgdLogitsProcessor(LogitsProcessor):
    """HF-compatible logits processor that wraps a DereferenceMonitor.

    HF passes `input_ids` as a 2-D LongTensor (batch, seq_len) and
    `scores` as a 2-D FloatTensor (batch, vocab). For batch=1 we
    extract the single sequence's tokens.
    """

    def __init__(self, monitor: DereferenceMonitor, prompt_len: int) -> None:
        self.monitor = monitor
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids shape: (batch=1, seq_len). scores shape: (batch=1, vocab).
        # Slice off the prompt — we only want generated tokens.
        gen_ids = input_ids[0, self.prompt_len:].tolist()
        try:
            mask = self.monitor.maskgen(gen_ids)
        except Exception as e:
            self.monitor.log.last_message = f"maskgen error: {e!r}"
            return scores
        if mask:
            idx = torch.tensor(list(mask), dtype=torch.long, device=scores.device)
            scores[0, idx] = float("-inf")
        return scores


class StopOnText(StoppingCriteria):
    """Stop generation when decoded text ends with one of the stop strings."""

    def __init__(self, tokenizer, prompt_len: int, stops: list[str]) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        gen_ids = input_ids[0, self.prompt_len:].tolist()
        if not gen_ids:
            return False
        text = self.tokenizer.decode(gen_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
        return any(text.endswith(s) for s in self.stops)


def _ensure_test_workspace() -> Path:
    TEST_WORKSPACE.mkdir(exist_ok=True)
    (TEST_WORKSPACE / "Cargo.toml").write_text(
        '[package]\nname = "harness"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "harness"\npath = "src/main.rs"\n'
    )
    (TEST_WORKSPACE / "src").mkdir(exist_ok=True)
    return TEST_WORKSPACE


async def _post_hoc_test(prob: RustProblem, completion: str, workspace: Path) -> tuple[bool, bool]:
    src = prob.full_source(completion)
    (workspace / "src" / "main.rs").write_text(src)
    proc = await asyncio.create_subprocess_exec(
        "cargo", "build", "--offline",
        cwd=str(workspace),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        proc.kill(); await proc.wait()
        return False, False
    if proc.returncode != 0:
        return False, False
    proc = await asyncio.create_subprocess_exec(
        "cargo", "run", "--offline", "--quiet",
        cwd=str(workspace),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        proc.kill(); await proc.wait()
        return True, False
    return True, proc.returncode == 0


def _generate(
    model, tokenizer, prompt: str, *,
    max_tokens: int, processor: MgdLogitsProcessor | None,
) -> tuple[str, int]:
    """One HF generation pass."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    stopper = StopOnText(tokenizer, prompt_len, stops=["\n}"])

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stopper]),
    )
    if processor is not None:
        gen_kwargs["logits_processor"] = transformers.LogitsProcessorList([processor])

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = out_ids[0, prompt_len:].tolist()
    text = tokenizer.decode(new_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
    # Truncate at the first \n} that closes the function — the stopping
    # criteria doesn't catch all tokenizations, so post-truncate here.
    idx = text.find("\n}")
    if idx >= 0:
        text = text[:idx]
    return text, len(new_ids)


async def run_arm_for_problem(
    model, tokenizer, provider,
    prob: RustProblem, arm: str,
    workspace: Path, test_workspace: Path, max_tokens: int,
    *, wall_budget_s: float = 90.0,
) -> HfRunResult:
    t0 = time.perf_counter()
    text = ""
    n_tokens = 0
    rollbacks = 0
    lsp_calls = 0
    tele = MgdTelemetry()

    # Per-problem wall-clock budget: if MGD takes too long, the
    # rust-analyzer cache pollution problem from earlier becomes
    # extreme. Wrap the slow arms with a budget that aborts.
    budgeted = arm in ("mgd", "mgd_rollback")

    async def _budgeted_run(coro_or_callable):
        if not budgeted:
            return await coro_or_callable() if callable(coro_or_callable) else await coro_or_callable
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(coro_or_callable) if callable(coro_or_callable) else coro_or_callable,
                timeout=wall_budget_s,
            )
        except asyncio.TimeoutError:
            print(f"    BUDGET EXCEEDED ({wall_budget_s}s)", flush=True)
            return None

    try:
        if arm == "raw":
            text, n_tokens = _generate(model, tokenizer, prob.prompt, max_tokens=max_tokens, processor=None)
        elif arm == "mgd":
            tele = MgdTelemetry()
            monitor = DereferenceMonitor(
                tokenizer=tokenizer, completions_provider=provider,
                prompt=prob.prompt, log=tele,
            )
            inputs = tokenizer(prob.prompt, return_tensors="pt")
            prompt_len = inputs.input_ids.shape[1]
            processor = MgdLogitsProcessor(monitor, prompt_len)
            result = await _budgeted_run(
                lambda: _generate(
                    model, tokenizer, prob.prompt,
                    max_tokens=max_tokens, processor=processor,
                )
            )
            if result is not None:
                text, n_tokens = result
        elif arm in ("rollback_mute", "rollback_instruct"):
            text, n_tokens, rollbacks, lsp_calls = await _run_rollback_hf(
                model, tokenizer, prob, max_tokens,
                instruct=(arm == "rollback_instruct"), workspace=workspace,
            )
        elif arm == "mgd_rollback":
            try:
                result = await asyncio.wait_for(
                    _run_mgd_rollback_hf(
                        model, tokenizer, provider, prob, max_tokens, workspace=workspace,
                    ),
                    timeout=wall_budget_s,
                )
                text, n_tokens, rollbacks, lsp_calls, tele = result
            except asyncio.TimeoutError:
                print(f"    BUDGET EXCEEDED ({wall_budget_s}s)", flush=True)
        else:
            raise ValueError(arm)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    ERROR in arm={arm}: {e!r}", flush=True)

    wall = time.perf_counter() - t0
    compiled = passed = False
    if text:
        try:
            compiled, passed = await _post_hoc_test(prob, text, test_workspace)
        except Exception as e:
            print(f"    POST-HOC ERROR: {e!r}", flush=True)

    return HfRunResult(
        name=prob.name, arm=arm, completion=text,
        compiled=compiled, passed=passed,
        wall_clock_s=wall,
        tokens_emitted=n_tokens, tokens_kept=len(text),
        rollback_count=rollbacks, lsp_calls=lsp_calls,
        mgd_triggers=tele.triggers,
        mgd_constrained_steps=tele.constrained_steps,
        mgd_empty_masks=tele.empty_masks,
        mgd_ra_calls=tele.rust_analyzer_calls,
    )


async def _run_rollback_hf(
    model, tokenizer, prob: RustProblem, max_tokens: int,
    *, instruct: bool, workspace: Path,
) -> tuple[str, int, int, int]:
    checker = CargoChecker(workspace=workspace)
    code = Code(prefix=prob.prompt, suffix="}", checker=checker)
    rollbacks = 0
    total_tokens = 0
    current_prompt = prob.prompt
    max_rollbacks = 8

    while rollbacks < max_rollbacks:
        new_text, n = _generate(
            model, tokenizer, current_prompt,
            max_tokens=max(max_tokens - len(code.content) // 4, 32),
            processor=None,
        )
        total_tokens += n
        code.content += new_text

        result = await code.check()
        if not result.is_error:
            break
        rollbacks += 1
        survivor = code.content_up_to(code.ckpt[-1])
        if instruct:
            err_lines = result.message.splitlines()[:5]
            suffix_msg = (
                "\n// previous attempt failed cargo check with:\n"
                + "\n".join(f"// {line}" for line in err_lines) + "\n"
            )
            current_prompt = prob.prompt + survivor + suffix_msg
        else:
            current_prompt = prob.prompt + survivor
        code.rollback()

    return code.content, total_tokens, rollbacks, code.lsp_calls


async def _run_mgd_rollback_hf(
    model, tokenizer, provider, prob: RustProblem, max_tokens: int,
    *, workspace: Path,
) -> tuple[str, int, int, int, MgdTelemetry]:
    tele = MgdTelemetry()
    checker = CargoChecker(workspace=workspace)
    code = Code(prefix=prob.prompt, suffix="}", checker=checker)
    rollbacks = 0
    total_tokens = 0
    current_prompt = prob.prompt
    max_rollbacks = 8

    while rollbacks < max_rollbacks:
        monitor = DereferenceMonitor(
            tokenizer=tokenizer, completions_provider=provider,
            prompt=current_prompt, log=tele,
        )
        inputs = tokenizer(current_prompt, return_tensors="pt")
        prompt_len = inputs.input_ids.shape[1]
        processor = MgdLogitsProcessor(monitor, prompt_len)
        new_text, n = _generate(
            model, tokenizer, current_prompt,
            max_tokens=max(max_tokens - len(code.content) // 4, 32),
            processor=processor,
        )
        total_tokens += n
        code.content += new_text

        result = await code.check()
        if not result.is_error:
            break
        rollbacks += 1
        survivor = code.content_up_to(code.ckpt[-1])
        current_prompt = prob.prompt + survivor
        code.rollback()

    return code.content, total_tokens, rollbacks, code.lsp_calls, tele


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--arms", nargs="+",
                        default=["raw", "rollback_mute", "rollback_instruct", "mgd", "mgd_rollback"])
    parser.add_argument("--out", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_rust_humaneval()[: args.limit]
    test_workspace = _ensure_test_workspace()

    # rust-analyzer provider
    arm_ws = PROJECT_ROOT / "cargo_workspaces" / "hf_mgd"
    arm_ws.mkdir(parents=True, exist_ok=True)
    (arm_ws / "Cargo.toml").write_text(
        '[package]\nname = "scratch"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "scratch"\npath = "src/main.rs"\n'
    )
    (arm_ws / "src").mkdir(exist_ok=True)
    (arm_ws / "src" / "main.rs").write_text("fn main() {}\n")
    print("Starting rust-analyzer completion provider...", flush=True)
    provider = RustAnalyzerCompletionsProvider(arm_ws)
    provider.start()
    print(f"  provider ready", flush=True)

    print(f"Loading {args.model} via transformers...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    print(f"  model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)", flush=True)

    safe = args.model.replace("/", "_").replace(":", "_")
    try:
        async def _run_all():
            nonlocal provider
            for arm in args.arms:
                print(f"\n=== arm={arm} ===", flush=True)
                results: list[HfRunResult] = []
                for i, prob in enumerate(problems):
                    # Restart the provider every 5 MGD problems to keep
                    # rust-analyzer state from accumulating.
                    if arm in ("mgd", "mgd_rollback") and i > 0 and i % 5 == 0:
                        print(f"  [restarting provider after {i} problems]", flush=True)
                        provider.stop()
                        provider = RustAnalyzerCompletionsProvider(arm_ws)
                        provider.start()

                    r = await run_arm_for_problem(
                        model, tokenizer, provider, prob, arm,
                        workspace=arm_ws, test_workspace=test_workspace,
                        max_tokens=args.max_tokens,
                    )
                    results.append(r)
                    print(
                        f"  [{i+1:3d}/{len(problems)}] {prob.name[:36]:36} "
                        f"arm={arm:18} P={'Y' if r.passed else 'N'} "
                        f"C={'Y' if r.compiled else 'N'} "
                        f"wall={r.wall_clock_s:5.1f}s rb={r.rollback_count} "
                        f"tok={r.tokens_emitted}/{r.tokens_kept} "
                        f"trig={r.mgd_triggers} con={r.mgd_constrained_steps} "
                        f"emp={r.mgd_empty_masks} ra={r.mgd_ra_calls}",
                        flush=True,
                    )
                out_path = out_dir / f"hf_{safe}_{arm}.json"
                out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
                print(f"Wrote {out_path}", flush=True)

        asyncio.run(_run_all())
    finally:
        provider.stop()


if __name__ == "__main__":
    main()
