"""Run all 4 arms — raw, MGD, rollback, MGD+rollback — on the SAME model
via vLLM's offline `LLM()` API.

Why offline vs HTTP: vLLM's offline `LLM` exposes `SamplingParams(logits_processors=...)`
synchronously, which is what we need to wire up the MGD mask. The OpenAI-
compatible server doesn't accept arbitrary logits processors over the wire.

Trade-off: offline mode is per-process — each call is one batch then teardown.
For our pilot (one prompt at a time, n=30) the overhead is acceptable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from soundcode.cargo_check import CargoChecker
from soundcode.code import Category, Code, State
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
class VllmRunResult:
    name: str
    arm: str
    completion: str
    compiled: bool
    passed: bool
    wall_clock_s: float
    tokens_emitted: int
    tokens_kept: int
    rollback_count: int
    lsp_calls: int       # cargo-check calls (rollback arms)
    mgd_triggers: int    # `.` triggers in MGD arms
    mgd_constrained_steps: int
    mgd_empty_masks: int
    mgd_ra_calls: int


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


# ─── arms ──────────────────────────────────────────────────────────────


def _run_raw_vllm(llm, prob: RustProblem, max_tokens: int) -> tuple[str, int]:
    """Arm (i) — no constraints, no rollback. One generate() call."""
    from vllm import SamplingParams
    params = SamplingParams(
        temperature=0.2, top_p=0.95, max_tokens=max_tokens,
        stop=["\n}"],
    )
    out = llm.generate([prob.prompt], params, use_tqdm=False)[0]
    text = out.outputs[0].text
    n_tokens = len(out.outputs[0].token_ids)
    return text, n_tokens


_REQ_COUNTER = {"n": 0}


def _next_req_id() -> int:
    _REQ_COUNTER["n"] += 1
    return _REQ_COUNTER["n"]


def _run_mgd_vllm(llm, tokenizer, provider, prob: RustProblem, max_tokens: int) -> tuple[str, int, MgdTelemetry]:
    """Arm (ii) — MGD mask via vLLM v1 logits-processor adapter."""
    from vllm import SamplingParams
    from soundcode.mgd import fetch_telemetry

    req_id = _next_req_id()
    params = SamplingParams(
        temperature=0.2, top_p=0.95, max_tokens=max_tokens,
        stop=["\n}"],
        extra_args={"mgd_prompt": prob.prompt, "mgd_req_id": req_id},
    )
    out = llm.generate([prob.prompt], params, use_tqdm=False)[0]
    text = out.outputs[0].text
    n_tokens = len(out.outputs[0].token_ids)
    tele = fetch_telemetry(req_id) or MgdTelemetry()
    return text, n_tokens, tele


async def _run_rollback_vllm(llm, prob: RustProblem, max_tokens: int, *,
                              instruct: bool, workspace: Path) -> tuple[str, int, int, int]:
    """Arm (iii) — cargo-check-driven rollback. Multiple LLM.generate() calls."""
    from vllm import SamplingParams

    checker = CargoChecker(workspace=workspace)
    code = Code(prefix=prob.prompt, suffix="}", checker=checker)
    rollbacks = 0
    total_tokens = 0
    current_prompt = prob.prompt
    max_rollbacks = 8

    while rollbacks < max_rollbacks:
        params = SamplingParams(
            temperature=0.2, top_p=0.95,
            max_tokens=max(max_tokens - len(code.content) // 4, 32),
            stop=["\n}"],
        )
        out = llm.generate([current_prompt], params, use_tqdm=False)[0]
        new_text = out.outputs[0].text
        total_tokens += len(out.outputs[0].token_ids)

        # Append the whole new text into the buffer at once (we don't have
        # streaming here — vLLM offline returns at-once). Then check at the
        # final position only — boundaries that the LM may have crossed
        # are abstracted away.
        code.content += new_text
        # Single end-of-generation check.
        result = await code.check()
        if not result.is_error:
            break  # done
        # Otherwise: rollback to last clean checkpoint and re-prompt.
        rollbacks += 1
        survivor = code.content_up_to(code.ckpt[-1])
        if instruct:
            err_msg = result.message.splitlines()[:5]
            suffix_msg = (
                "\n// previous attempt failed cargo check with:\n"
                + "\n".join(f"// {line}" for line in err_msg) + "\n"
            )
            current_prompt = prob.prompt + survivor + suffix_msg
        else:
            current_prompt = prob.prompt + survivor
        code.rollback()

    return code.content, total_tokens, rollbacks, code.lsp_calls


# ─── orchestration ────────────────────────────────────────────────────


async def run_problem(
    llm, tokenizer, provider, prob: RustProblem, arm: str,
    workspace: Path, test_workspace: Path, max_tokens: int,
) -> VllmRunResult:
    t0 = time.perf_counter()
    text = ""
    n_tokens = 0
    rollbacks = 0
    lsp_calls = 0
    mgd_tele = MgdTelemetry()
    err = None
    try:
        if arm == "raw":
            text, n_tokens = _run_raw_vllm(llm, prob, max_tokens)
        elif arm == "mgd":
            text, n_tokens, mgd_tele = _run_mgd_vllm(llm, tokenizer, provider, prob, max_tokens)
        elif arm == "rollback_mute":
            text, n_tokens, rollbacks, lsp_calls = await _run_rollback_vllm(
                llm, prob, max_tokens, instruct=False, workspace=workspace,
            )
        elif arm == "rollback_instruct":
            text, n_tokens, rollbacks, lsp_calls = await _run_rollback_vllm(
                llm, prob, max_tokens, instruct=True, workspace=workspace,
            )
        elif arm == "mgd_rollback":
            # Combined: MGD masking inside each generation pass, plus
            # cargo-check rollback if final state errs. We do the simplest
            # implementation: MGD inside the inner generate, full-text check
            # at end, rollback at the prompt level if needed.
            text, n_tokens, rollbacks, lsp_calls, mgd_tele = await _run_mgd_rollback(
                llm, tokenizer, provider, prob, max_tokens, workspace,
            )
        else:
            raise ValueError(arm)
    except Exception as e:
        import traceback
        err = repr(e)
        print(f"    ERROR in arm={arm}: {err}", flush=True)
        traceback.print_exc()

    wall = time.perf_counter() - t0
    compiled = passed = False
    if text and not err:
        try:
            compiled, passed = await _post_hoc_test(prob, text, test_workspace)
        except Exception as e:
            err = repr(e)
            print(f"    POST-HOC ERROR: {err}", flush=True)

    return VllmRunResult(
        name=prob.name, arm=arm, completion=text,
        compiled=compiled, passed=passed,
        wall_clock_s=wall,
        tokens_emitted=n_tokens, tokens_kept=len(text),
        rollback_count=rollbacks, lsp_calls=lsp_calls,
        mgd_triggers=mgd_tele.triggers,
        mgd_constrained_steps=mgd_tele.constrained_steps,
        mgd_empty_masks=mgd_tele.empty_masks,
        mgd_ra_calls=mgd_tele.rust_analyzer_calls,
    )


async def _run_mgd_rollback(
    llm, tokenizer, provider, prob: RustProblem, max_tokens: int, workspace: Path,
) -> tuple[str, int, int, int, MgdTelemetry]:
    """Arm (iv) — MGD masking + rollback at end of generation.

    Each pass: MGD logits processor enabled; emit text; run cargo check;
    rollback on error; re-prompt. MGD telemetry accumulated across attempts.
    """
    from vllm import SamplingParams

    tele = MgdTelemetry()
    checker = CargoChecker(workspace=workspace)
    code = Code(prefix=prob.prompt, suffix="}", checker=checker)
    rollbacks = 0
    total_tokens = 0
    current_prompt = prob.prompt
    max_rollbacks = 8

    from soundcode.mgd import fetch_telemetry

    while rollbacks < max_rollbacks:
        req_id = _next_req_id()
        params = SamplingParams(
            temperature=0.2, top_p=0.95,
            max_tokens=max(max_tokens - len(code.content) // 4, 32),
            stop=["\n}"],
            extra_args={"mgd_prompt": current_prompt, "mgd_req_id": req_id},
        )
        out = llm.generate([current_prompt], params, use_tqdm=False)[0]
        new_text = out.outputs[0].text
        total_tokens += len(out.outputs[0].token_ids)
        code.content += new_text

        # Accumulate telemetry across attempts.
        attempt_tele = fetch_telemetry(req_id)
        if attempt_tele:
            tele.triggers += attempt_tele.triggers
            tele.constrained_steps += attempt_tele.constrained_steps
            tele.masks_built += attempt_tele.masks_built
            tele.empty_masks += attempt_tele.empty_masks
            tele.total_completions += attempt_tele.total_completions
            tele.rust_analyzer_calls += attempt_tele.rust_analyzer_calls

        result = await code.check()
        if not result.is_error:
            break
        rollbacks += 1
        survivor = code.content_up_to(code.ckpt[-1])
        current_prompt = prob.prompt + survivor

    return code.content, total_tokens, rollbacks, code.lsp_calls, tele


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--arms", nargs="+",
                        default=["raw", "rollback_mute", "rollback_instruct", "mgd", "mgd_rollback"])
    parser.add_argument("--out", default=str(RESULTS_ROOT))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_rust_humaneval()[: args.limit]
    test_workspace = _ensure_test_workspace()

    # Start rust-analyzer FIRST so the adapter sees a ready provider.
    print(f"Starting rust-analyzer completion provider...", flush=True)
    arm_ws = PROJECT_ROOT / "cargo_workspaces" / "vllm_mgd"
    arm_ws.mkdir(parents=True, exist_ok=True)
    (arm_ws / "Cargo.toml").write_text(
        '[package]\nname = "scratch"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "scratch"\npath = "src/main.rs"\n'
    )
    (arm_ws / "src").mkdir(exist_ok=True)
    (arm_ws / "src" / "main.rs").write_text("fn main() {}\n")
    provider = RustAnalyzerCompletionsProvider(arm_ws)
    provider.start()
    print(f"  provider ready", flush=True)

    print(f"Loading {args.model} via vLLM...", flush=True)
    from vllm import LLM
    from transformers import AutoTokenizer
    from soundcode.mgd import set_global_provider, build_mgd_adapter

    set_global_provider(provider)
    mgd_adapter_cls = build_mgd_adapter()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        dtype="bfloat16",
        logits_processors=[mgd_adapter_cls],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    safe = args.model.replace("/", "_").replace(":", "_")
    try:
        async def _run_all():
            for arm in args.arms:
                print(f"\n=== arm={arm} ===", flush=True)
                results: list[VllmRunResult] = []
                for i, prob in enumerate(problems):
                    r = await run_problem(
                        llm, tokenizer, provider, prob, arm,
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
                        f"mgd_trig={r.mgd_triggers} ra={r.mgd_ra_calls}",
                        flush=True,
                    )
                out_path = out_dir / f"{safe}_{arm}.json"
                out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
                print(f"Wrote {out_path}", flush=True)

        asyncio.run(_run_all())
    finally:
        provider.stop()


if __name__ == "__main__":
    main()
