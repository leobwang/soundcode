"""Paper Phase 1 — headline experiment for the NeurIPS-style writeup.

Runs 9 arms × ~160 problems × Qwen2.5-Coder-7B-Instruct (vLLM), greedy decoding
(T=0), max_tokens=512, 1 sample per problem. Per-problem 60s wall-time cap.

Headline comparison: arms 3 vs 4 (Rust) and 5 vs 6 (C++) — sync vs async LSP
verification under naive last-checkpoint rollback. The claim being tested:
async LSP-supervised decoding gives meaningful wall-time speedup over sync
(ROCODE-style) verification on the same algorithm.

The "sync" arm mirrors ROCODE's algorithm: the decode stream is ABORTED at
every depth-0 `;` / `}` boundary, the checker runs to completion, and the
stream is restarted (possibly from a rollback target). The "async" arm spawns
the checker as a task and KEEPS DECODING; the verdict is harvested whenever
it lands, and rollback only fires for a real BLOCKING error.

Per-arm output: results/paper_phase1/<arm_id>.jsonl, one JSON-per-line:
    {"arm", "problem_id", "lang", "wall_time_s", "n_tokens", "n_rollbacks",
     "n_checker_calls", "compile_ok", "tests_pass", "generated_code", ...}
Summary: results/paper_phase1/summary.json with aggregated per-arm stats.

Idempotent: existing arm files are skipped on resume.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_phase1"
KNOWN_ISSUES_PATH = RESULTS_DIR / "known_issues.md"
SENTINEL_PATH = RESULTS_DIR / "DONE"

MODEL_NAME = "qwen2.5-coder-7b"  # registry key (see soundcode/vllm_model_registry.py)

# Pinned nuprl/MultiPL-E revision — the revision the paper's data was
# generated from (HF cache refs/main at run time). Unpinned loads track
# upstream main and may drift.
MULTIPLE_REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"

DEFAULT_PROBLEM_TIMEOUT_S = 60.0
DEFAULT_MAX_TOKENS = 512
MAX_ROLLBACKS_PER_PROBLEM = 6


# ─── arms ──────────────────────────────────────────────────────────────────


ARMS: list[dict[str, Any]] = [
    {"id": "plain_rust",          "lang": "rust", "mode": "plain"},
    {"id": "plain_cpp",           "lang": "cpp",  "mode": "plain"},
    {"id": "sync_naive_rust",     "lang": "rust", "mode": "soundcode",
     "scheduling": "sync",  "rollback": "naive"},
    {"id": "async_naive_rust",    "lang": "rust", "mode": "soundcode",
     "scheduling": "async", "rollback": "naive"},
    {"id": "sync_naive_cpp",      "lang": "cpp",  "mode": "soundcode",
     "scheduling": "sync",  "rollback": "naive"},
    {"id": "async_naive_cpp",     "lang": "cpp",  "mode": "soundcode",
     "scheduling": "async", "rollback": "naive"},
    {"id": "async_entropy_rust",  "lang": "rust", "mode": "soundcode",
     "scheduling": "async", "rollback": "entropy"},
    {"id": "async_entropy_cpp",   "lang": "cpp",  "mode": "soundcode",
     "scheduling": "async", "rollback": "entropy"},
    {"id": "rocode_upstream_cpp", "lang": "cpp",  "mode": "rocode_upstream"},
]

# Post-deadline arms — NOT part of the default grid (so replication runs of
# the paper grid are unaffected). Selectable only via --arms.
#   entropy2: ROCODE's FULL entropy rule, properly wired — real per-token
#             Shannon entropies from vLLM logprobs, error-line offset r_e on
#             first rollback, max-entropy line r_h on recurrence (same error
#             code at same line as the previous rollback).
#   lsp:      rust-analyzer (via soundcode.ra_check) as the in-loop verifier
#             instead of cargo check; NaiveLast rollback.
EXTRA_ARMS: list[dict[str, Any]] = [
    {"id": "async_entropy2_rust", "lang": "rust", "mode": "soundcode",
     "scheduling": "async", "rollback": "entropy2"},
    {"id": "async_entropy2_cpp",  "lang": "cpp",  "mode": "soundcode",
     "scheduling": "async", "rollback": "entropy2"},
    {"id": "async_lsp_rust",      "lang": "rust", "mode": "soundcode",
     "scheduling": "async", "rollback": "naive", "verifier": "lsp"},
    {"id": "sync_lsp_rust",       "lang": "rust", "mode": "soundcode",
     "scheduling": "sync",  "rollback": "naive", "verifier": "lsp"},
    # Third language: Java / javac.
    {"id": "plain_java",          "lang": "java", "mode": "plain"},
    {"id": "sync_naive_java",     "lang": "java", "mode": "soundcode",
     "scheduling": "sync",  "rollback": "naive"},
    {"id": "async_naive_java",    "lang": "java", "mode": "soundcode",
     "scheduling": "async", "rollback": "naive"},
    # Matched-budget resampling baseline (compiler as post-hoc filter).
    {"id": "resample_rust",       "lang": "rust", "mode": "resample"},
    {"id": "resample_cpp",        "lang": "cpp",  "mode": "resample"},
    # NaiveLast+Penalty (ROCODE Eq. 8-9 via engine-side V1 processor).
    {"id": "async_penalty_rust",  "lang": "rust", "mode": "soundcode",
     "scheduling": "async", "rollback": "penalty"},
    {"id": "async_penalty_cpp",   "lang": "cpp",  "mode": "soundcode",
     "scheduling": "async", "rollback": "penalty"},
    {"id": "sync_penalty_rust",   "lang": "rust", "mode": "soundcode",
     "scheduling": "sync",  "rollback": "penalty"},
    {"id": "sync_penalty_cpp",    "lang": "cpp",  "mode": "soundcode",
     "scheduling": "sync",  "rollback": "penalty"},
]


# ─── language adapters ─────────────────────────────────────────────────────


class LangAdapter:
    """Per-language pieces — checker, boundary detector, function closer,
    workspace setup, post-hoc pass@1 testing.
    """
    lang: str
    function_closer: str
    stop_strings: list[str]

    def fresh_workspace(self) -> Path:
        raise NotImplementedError

    def make_checker(self, workspace: Path):
        raise NotImplementedError

    def post_hoc_test(self, prompt: str, completion: str, tests: str) -> tuple[bool, bool]:
        raise NotImplementedError

    def compile_only(self, prompt: str, completion: str) -> bool:
        """Does prompt+completion (+ function closer trimmed) compile?
        Used by the resample-baseline arm as its selection filter — tests
        are never consulted during selection."""
        raise NotImplementedError


class RustAdapter(LangAdapter):
    def __init__(self) -> None:
        self.lang = "rust"
        self.function_closer = "\n    unreachable!()\n}\n\nfn main() {}\n"
        # `\n}` is the canonical end-of-function delimiter for HumanEval-rs.
        self.stop_strings = ["\n}"]

    def fresh_workspace(self) -> Path:
        ws = Path(tempfile.mkdtemp(prefix="pp1-rust-"))
        (ws / "Cargo.toml").write_text(
            '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n\n'
            '[[bin]]\nname = "demo"\npath = "src/main.rs"\n'
        )
        (ws / "src").mkdir(exist_ok=True)
        (ws / "src" / "main.rs").write_text("fn main() {}\n")
        return ws

    def make_checker(self, workspace: Path):
        from soundcode.cargo_check import CargoChecker
        return CargoChecker(workspace=workspace)

    def post_hoc_test(self, prompt: str, completion: str, tests: str) -> tuple[bool, bool]:
        ws = self.fresh_workspace()
        try:
            body = completion
            # Strip trailing stop tokens that aren't part of the function body.
            for stop in ("\n}",):
                if body.endswith(stop):
                    body = body[: -len(stop)]
            src = prompt + body + "\n" + tests
            (ws / "src" / "main.rs").write_text(src)
            try:
                proc = subprocess.run(
                    ["cargo", "build", "--offline"],
                    cwd=str(ws), capture_output=True, timeout=30,
                    env={**os.environ, "CARGO_TERM_COLOR": "never"},
                )
            except subprocess.TimeoutExpired:
                return False, False
            if proc.returncode != 0:
                return False, False
            try:
                proc = subprocess.run(
                    ["cargo", "run", "--offline", "--quiet"],
                    cwd=str(ws), capture_output=True, timeout=15,
                    env={**os.environ, "CARGO_TERM_COLOR": "never"},
                )
            except subprocess.TimeoutExpired:
                return True, False
            return True, (proc.returncode == 0)
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def compile_only(self, prompt: str, completion: str) -> bool:
        ws = self.fresh_workspace()
        try:
            body = completion
            for stop in ("\n}",):
                if body.endswith(stop):
                    body = body[: -len(stop)]
            src = prompt + body + "\n}\n\nfn main() {}\n"
            (ws / "src" / "main.rs").write_text(src)
            try:
                proc = subprocess.run(
                    ["cargo", "build", "--offline"],
                    cwd=str(ws), capture_output=True, timeout=30,
                    env={**os.environ, "CARGO_TERM_COLOR": "never"},
                )
            except subprocess.TimeoutExpired:
                return False
            return proc.returncode == 0
        finally:
            shutil.rmtree(ws, ignore_errors=True)


class CppAdapter(LangAdapter):
    def __init__(self) -> None:
        self.lang = "cpp"
        self.function_closer = "\n    return {};\n}\n\nint main() { return 0; }\n"
        self.stop_strings = ["\n}"]

    def fresh_workspace(self) -> Path:
        ws = Path(tempfile.mkdtemp(prefix="pp1-cpp-"))
        (ws / "main.cpp").write_text("int main() { return 0; }\n")
        return ws

    def make_checker(self, workspace: Path):
        from soundcode.lang.cpp import GccChecker
        return GccChecker(workspace=workspace, file_in_workspace="main.cpp")

    def post_hoc_test(self, prompt: str, completion: str, tests: str) -> tuple[bool, bool]:
        ws = self.fresh_workspace()
        try:
            body = completion
            for stop in ("\n}",):
                if body.endswith(stop):
                    body = body[: -len(stop)]
            src = prompt + body + "\n" + tests
            srcfile = ws / "main.cpp"
            srcfile.write_text(src)
            exe = ws / "main"
            try:
                proc = subprocess.run(
                    ["g++", "-std=c++17", "-O0", "-w", str(srcfile), "-o", str(exe)],
                    capture_output=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                return False, False
            if proc.returncode != 0:
                return False, False
            try:
                proc = subprocess.run(
                    [str(exe)],
                    capture_output=True, timeout=15,
                )
            except subprocess.TimeoutExpired:
                return True, False
            return True, (proc.returncode == 0)
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def compile_only(self, prompt: str, completion: str) -> bool:
        ws = self.fresh_workspace()
        try:
            body = completion
            for stop in ("\n}",):
                if body.endswith(stop):
                    body = body[: -len(stop)]
            src = prompt + body + "\n}\n\nint main() { return 0; }\n"
            srcfile = ws / "main.cpp"
            srcfile.write_text(src)
            try:
                proc = subprocess.run(
                    ["g++", "-std=c++17", "-O0", "-w", "-fsyntax-only",
                     str(srcfile)],
                    capture_output=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                return False
            return proc.returncode == 0
        finally:
            shutil.rmtree(ws, ignore_errors=True)


class JavaAdapter(LangAdapter):
    def __init__(self) -> None:
        self.lang = "java"
        # Close the method with a throw (satisfies any return type), then
        # the class — the Java analog of Rust's `unreachable!()` shim.
        self.function_closer = (
            "\n        throw new RuntimeException(\"incomplete\");\n    }\n}\n")
        # MultiPL-E humaneval-java stop token is the method-closing brace at
        # one indent level.
        self.stop_strings = ["\n    }"]
        # MultiPL-E Java depends on javatuples (Pair/Triplet in prompts and
        # tests); the official evaluation container ships it on the
        # classpath. javac and java both honor the CLASSPATH env var, which
        # also reaches JavacChecker's in-loop subprocess (env passthrough).
        jar = PROJECT_ROOT / "vendor" / "javatuples-1.2.jar"
        if jar.exists():
            os.environ["CLASSPATH"] = f".:{jar}"

    def fresh_workspace(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="pp1-java-"))

    def make_checker(self, workspace: Path):
        from soundcode.lang.java import JavacChecker
        return JavacChecker(workspace=workspace,
                            file_in_workspace="Problem.java")

    def post_hoc_test(self, prompt: str, completion: str, tests: str) -> tuple[bool, bool]:
        ws = self.fresh_workspace()
        try:
            body = completion
            for stop in ("\n    }",):
                if body.endswith(stop):
                    body = body[: -len(stop)]
            src = prompt + body + "\n" + tests
            (ws / "Problem.java").write_text(src)
            try:
                proc = subprocess.run(
                    ["javac", "Problem.java"],
                    cwd=str(ws), capture_output=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                return False, False
            if proc.returncode != 0:
                return False, False
            try:
                proc = subprocess.run(
                    ["java", "-ea", "Problem"],
                    cwd=str(ws), capture_output=True, timeout=15,
                )
            except subprocess.TimeoutExpired:
                return True, False
            return True, (proc.returncode == 0)
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def compile_only(self, prompt: str, completion: str) -> bool:
        ws = self.fresh_workspace()
        try:
            body = completion
            for stop in ("\n    }",):
                if body.endswith(stop):
                    body = body[: -len(stop)]
            src = prompt + body + "\n    }\n}\n"
            (ws / "Problem.java").write_text(src)
            try:
                proc = subprocess.run(
                    ["javac", "Problem.java"],
                    cwd=str(ws), capture_output=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                return False
            return proc.returncode == 0
        finally:
            shutil.rmtree(ws, ignore_errors=True)


ADAPTERS: dict[str, type[LangAdapter]] = {
    "rust": RustAdapter, "cpp": CppAdapter, "java": JavaAdapter,
}


# ─── dataset loading ───────────────────────────────────────────────────────


@dataclass
class Problem:
    name: str
    prompt: str
    tests: str
    stop_tokens: list[str]
    language: str


def load_humaneval(lang: str, family: str = "humaneval") -> list[Problem]:
    from datasets import load_dataset
    suffix = {"rust": "rs", "cpp": "cpp", "java": "java"}.get(lang)
    if suffix is None:
        raise ValueError(lang)
    if family not in ("humaneval", "mbpp"):
        raise ValueError(family)
    ds_name = f"{family}-{suffix}"
    ds = load_dataset("nuprl/MultiPL-E", ds_name, split="test",
                      revision=MULTIPLE_REVISION)
    out: list[Problem] = []
    for row in ds:
        out.append(Problem(
            name=row["name"],
            prompt=row["prompt"],
            tests=row["tests"],
            stop_tokens=list(row["stop_tokens"]),
            language=lang,
        ))
    return out


# ─── vLLM engine ───────────────────────────────────────────────────────────


def build_vllm_engine(model_name: str = MODEL_NAME,
                      with_penalty: bool = False):
    """Build an AsyncLLMEngine with the winning config: enforce_eager=False,
    gpu_memory_utilization=0.78, cudagraph_capture_sizes=[1,2,4].

    `with_penalty` registers the engine-side ROCODE penalty processor
    (soundcode.rocode_penalty_v1). Only set for penalty arms — it is
    per-request inert without extra_args, but keeping the default engine
    config byte-identical preserves replication comparability.
    """
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from soundcode.vllm_model_registry import resolve

    reg = resolve(model_name)
    assert reg is not None, f"model {model_name} not in registry"
    engine_kwargs: dict[str, Any] = dict(
        model=reg["hf_id"],
        dtype=reg.get("dtype", "bfloat16"),
        max_model_len=reg.get("max_model_len", 4096),
        gpu_memory_utilization=0.78,
        enforce_eager=False,
        max_num_seqs=4,
        enable_prefix_caching=True,
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_mode": 2,  # FULL_DECODE_ONLY
        },
    )
    if reg.get("quantization") is not None:
        engine_kwargs["quantization"] = reg["quantization"]
    if with_penalty:
        # vLLM's custom-logitsproc loader expects entrypoint syntax
        # ("module.path:ClassName"), not a dotted attribute path.
        engine_kwargs["logits_processors"] = [
            "soundcode.rocode_penalty_v1:RocodePenaltyV1"]
    args = AsyncEngineArgs(**engine_kwargs)
    return AsyncLLMEngine.from_engine_args(args)


def make_lsp_checker(workspace: Path):
    """rust-analyzer via multilspy — the AsyncLSP/SyncLSP verifier.

    Same interface as CargoChecker (async check(source) -> list[Diagnostic],
    plus start()/stop()). Uses the shipped driver defaults, including its
    fixed 2.0s publish-settle wait per check; see soundcode/ra_check.py.
    """
    from soundcode.ra_check import RustAnalyzerChecker
    return RustAnalyzerChecker(workspace=workspace)


# ─── result dataclass ──────────────────────────────────────────────────────


@dataclass
class RunResult:
    arm: str
    problem_id: str
    lang: str
    wall_time_s: float
    n_tokens: int
    n_rollbacks: int
    n_checker_calls: int
    compile_ok: bool
    tests_pass: bool
    timeout: bool
    generated_code: str
    error: str | None = None


# ─── plain arm ─────────────────────────────────────────────────────────────


async def run_plain(engine, prob: Problem, adapter: LangAdapter, *,
                     timeout_s: float, max_tokens: int) -> RunResult:
    """Plain arm — no checker, no rollback. One greedy generate, then test."""
    from vllm import SamplingParams

    t0 = time.perf_counter()
    req_id = f"pp1-plain-{uuid.uuid4().hex[:10]}"
    full_text = ""
    n_tokens = 0
    timed_out = False
    err = None

    params = SamplingParams(
        temperature=0.0, top_p=1.0,
        max_tokens=max_tokens,
        stop=adapter.stop_strings or None,
    )

    async def _gen():
        nonlocal full_text, n_tokens
        async for out in engine.generate(prob.prompt, params, req_id):
            if out.outputs:
                full_text = out.outputs[0].text or ""
                n_tokens = len(out.outputs[0].token_ids or [])
            if out.finished:
                break

    try:
        await asyncio.wait_for(_gen(), timeout=timeout_s)
    except asyncio.TimeoutError:
        timed_out = True
        with contextlib.suppress(Exception):
            await engine.abort(req_id)
    except Exception as e:
        err = repr(e)
        with contextlib.suppress(Exception):
            await engine.abort(req_id)
    wall = time.perf_counter() - t0

    compile_ok = tests_pass = False
    if full_text and not err and not timed_out:
        try:
            compile_ok, tests_pass = await asyncio.to_thread(
                adapter.post_hoc_test, prob.prompt, full_text, prob.tests,
            )
        except Exception as e:
            err = f"post_hoc: {e!r}"

    return RunResult(
        arm="", problem_id=prob.name, lang=prob.language,
        wall_time_s=wall, n_tokens=n_tokens,
        n_rollbacks=0, n_checker_calls=0,
        compile_ok=compile_ok, tests_pass=tests_pass,
        timeout=timed_out, generated_code=full_text, error=err,
    )


# ─── resample-baseline arm ─────────────────────────────────────────────────


async def run_resample(engine, prob: Problem, adapter: LangAdapter, *,
                       timeout_s: float, max_tokens: int,
                       temperature: float = 0.8,
                       max_samples: int = 32) -> RunResult:
    """Matched-budget i.i.d. resampling baseline (Olausson-style).

    Sample full completions at T=temperature (seeded per sample) until one
    COMPILES or the wall budget / sample cap is exhausted. The compiler is
    a post-hoc filter, not an in-loop supervisor; unit tests are never
    consulted during selection. The submission is the first compiling
    sample, else the last sample drawn. Fields: n_rollbacks = resamples
    performed, n_checker_calls = compile-filter invocations.
    """
    from vllm import SamplingParams

    t0 = time.perf_counter()
    total_tokens = 0
    samples = 0
    chosen = ""
    timed_out = False
    err = None

    try:
        while samples < max_samples:
            elapsed = time.perf_counter() - t0
            if elapsed >= timeout_s:
                timed_out = chosen == ""
                break
            params = SamplingParams(
                temperature=temperature, top_p=0.95,
                seed=samples,  # deterministic-ish reproducibility
                max_tokens=max_tokens,
                stop=adapter.stop_strings or None,
            )
            req_id = f"pp1-rs-{uuid.uuid4().hex[:10]}"
            text = ""
            n_tok = 0

            async def _gen():
                nonlocal text, n_tok
                async for out in engine.generate(prob.prompt, params, req_id):
                    if out.outputs:
                        text = out.outputs[0].text or ""
                        n_tok = len(out.outputs[0].token_ids or [])
                    if out.finished:
                        break

            budget = max(0.5, timeout_s - (time.perf_counter() - t0))
            try:
                await asyncio.wait_for(_gen(), timeout=budget)
            except asyncio.TimeoutError:
                with contextlib.suppress(Exception):
                    await engine.abort(req_id)
                if chosen == "":
                    chosen = text
                break
            samples += 1
            total_tokens += n_tok
            chosen = text
            compiled = await asyncio.to_thread(
                adapter.compile_only, prob.prompt, text)
            if compiled:
                break
    except Exception as e:
        err = repr(e)

    wall = time.perf_counter() - t0
    compile_ok = tests_pass = False
    if chosen and not err:
        try:
            compile_ok, tests_pass = await asyncio.to_thread(
                adapter.post_hoc_test, prob.prompt, chosen, prob.tests)
        except Exception as e:
            err = f"post_hoc: {e!r}"

    return RunResult(
        arm="", problem_id=prob.name, lang=prob.language,
        wall_time_s=wall, n_tokens=total_tokens,
        n_rollbacks=max(0, samples - 1), n_checker_calls=samples,
        compile_ok=compile_ok, tests_pass=tests_pass,
        timeout=timed_out, generated_code=chosen, error=err,
    )


# ─── soundcode arms ────────────────────────────────────────────────────────


async def run_soundcode(
    engine, prob: Problem, adapter: LangAdapter,
    *, scheduling: str, rollback: str,
    timeout_s: float, max_tokens: int,
    verifier: str = "cargo",
) -> RunResult:
    """SoundCode arm — checker-supervised decoding with rollback.

    Producer-consumer loop driven by vLLM's streaming RequestOutput. For each
    yielded chunk, we append the new text delta to a `Code` buffer (which
    re-detects boundaries). When `at_boundary()` fires:

      sync:   abort vLLM, await checker.check(), then either rollback+restart
              OR restart from the next-prompt position (which extends the
              prefix and reuses KV cache via vLLM's automatic prefix caching).
      async:  spawn `code.check()` as a task and KEEP DECODING. Harvest the
              task whenever it completes; if it returned an error, abort
              vLLM and roll back.

    Both modes share the same Code / Checker / rollback machinery — the only
    difference is whether we block on the check.
    """
    from soundcode.code import Code

    t0 = time.perf_counter()
    workspace = adapter.fresh_workspace()
    rocode_proc = None
    rocode_decider = None
    if rollback == "entropy":
        from soundcode.rocode_processor import (
            RocodeTriePenaltyProcessor,
            StrategicRollbackDecider,
        )
        rocode_proc = RocodeTriePenaltyProcessor(lam=0.9)
        rocode_decider = StrategicRollbackDecider(trie=rocode_proc.trie)
    elif rollback == "penalty":
        # Host-side trie for the NaiveLast+Penalty policy; the engine-side
        # multiplier lives in soundcode.rocode_penalty_v1 and receives the
        # serialized cursor subtree via SamplingParams.extra_args.
        from soundcode.rocode_processor import RocodeTriePenaltyProcessor
        rocode_proc = RocodeTriePenaltyProcessor(lam=0.9)

    n_tokens = 0
    n_rollbacks = 0
    timed_out = False
    err: str | None = None

    # entropy2 bookkeeping: per-token entropy spans in content coordinates
    # ([start_char, end_char, mean_H]), and the (code, line) key of the
    # previous rollback's first blocking diagnostic (recurrence detection).
    ent_spans: list[list[float]] = []
    last_rb_key: tuple | None = None
    prev_n_lp = 0
    # penalty bookkeeping: char span per committed token (maps char-offset
    # rollback targets to trie token depths), cumulative across restarts.
    tok_spans: list[tuple[int, int]] = []
    prev_n_tok = 0
    # Highest boundary offset already checked/spawned. Without this gate,
    # at_boundary() re-fires on the SAME trailing `;`/`}` whenever a newly
    # appended chunk is pure whitespace (rstrip makes it invisible) — on
    # deep-indent languages (Java) the sync arm thrashed at ~80 checks and
    # aborts per problem. Reset downward on rollback (truncated offsets may
    # hold new code needing fresh checks).
    checked_boundary_floor = -1

    try:
        if verifier == "lsp":
            checker = make_lsp_checker(workspace)
            # A failed LSP start must surface as an arm error, not silently
            # degrade the arm to plain decoding (check() would return []).
            checker.start()
            # Exclude per-problem rust-analyzer startup from measured wall —
            # the cargo arms pay no analogous per-problem server cost, and
            # the vLLM engine build is likewise excluded globally.
            t0 = time.perf_counter()
        else:
            checker = adapter.make_checker(workspace)
            if hasattr(checker, "start"):
                with contextlib.suppress(Exception):
                    checker.start()

        code = Code(
            prefix=prob.prompt, suffix="}",
            checker=checker,
            function_closer=adapter.function_closer,
        )

        cur_prompt = prob.prompt
        pending_check: asyncio.Task | None = None

        async def _do_rollback(result) -> None:
            """Apply rollback policy. Mutates `code` and `cur_prompt`."""
            nonlocal cur_prompt, last_rb_key, checked_boundary_floor
            if (rollback == "entropy" and rocode_decider is not None
                    and rocode_proc is not None):
                try:
                    decision = rocode_decider.decide(result.diagnostics)
                    ckpts = [c for c in code.ckpt if c < len(code.content)]
                    if decision.used_recurrence and len(ckpts) >= 2:
                        target = ckpts[-2]
                    elif ckpts:
                        target = ckpts[-1]
                    else:
                        target = 0
                except Exception:
                    target = code.ckpt[-1] if code.ckpt else 0
            elif rollback == "penalty" and rocode_proc is not None:
                # NaiveLast target + ROCODE decayed penalty on the
                # abandoned suffix (applied engine-side on the retry).
                target = code.ckpt[-1] if code.ckpt else 0
                try:
                    tok_pos = sum(1 for s in tok_spans if s[1] <= target)
                    tok_pos = min(tok_pos, rocode_proc.trie.cursor.depth)
                    rocode_proc.mark_error()
                    rocode_proc.rollback_and_penalize(tok_pos)
                    del tok_spans[tok_pos:]
                except Exception:
                    pass
            elif rollback == "entropy2":
                # ROCODE's full rule (Eqs. 5-7): r_e (error-line offset) on
                # first rollback for an error; r_h (start of the line
                # containing the max-entropy token) when the same error code
                # recurs at the same line as the previous rollback.
                target = None
                blocking = [d for d in result.diagnostics
                            if d.is_blocking and d.line is not None]
                key = None
                r_e = None
                if blocking:
                    d0 = blocking[0]
                    key = (d0.code, d0.line)
                    # Map 1-based line in the checked file (prompt + content
                    # + closer) to a char offset in content coordinates.
                    checked = prob.prompt + code.content
                    lines = checked.split("\n")
                    if 1 <= d0.line <= len(lines):
                        off = sum(len(l) + 1 for l in lines[: d0.line - 1])
                        r_e = max(0, min(off - len(prob.prompt),
                                         len(code.content)))
                recurrence = key is not None and key == last_rb_key
                last_rb_key = key
                if recurrence and ent_spans:
                    smax = max(ent_spans, key=lambda s: s[2])
                    target = code.content.rfind("\n", 0, int(smax[0])) + 1
                elif r_e is not None and r_e < len(code.content):
                    target = r_e
                if target is None or target >= len(code.content):
                    target = code.ckpt[-1] if code.ckpt else 0
            else:
                target = code.ckpt[-1] if code.ckpt else 0
            code.rollback(to_offset=target)
            checked_boundary_floor = target - 1
            if rollback == "entropy2":
                # Prune entropy spans past the rollback target.
                kept = []
                for s in ent_spans:
                    if s[0] >= target:
                        continue
                    kept.append([s[0], min(s[1], float(target)), s[2]])
                ent_spans[:] = kept
            cur_prompt = prob.prompt + code.content_up_to(target)

        attempt = 0
        # Top loop: each iteration opens a fresh vLLM stream from `cur_prompt`.
        # We exit when the stream finishes (no rollback fired during it) or
        # the rollback / timeout budget is exhausted.
        while attempt <= MAX_ROLLBACKS_PER_PROBLEM:
            attempt += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= timeout_s:
                timed_out = True
                break

            req_id = f"pp1-sc-{uuid.uuid4().hex[:10]}"
            stream_start_text_len = len(code.content)
            prev_full = ""  # cumulative text vLLM has emitted this stream

            from vllm import SamplingParams
            if rollback == "penalty" and rocode_proc is not None:
                from soundcode.rocode_penalty_v1 import serialize_trie
                _extra = {"rocode_trie":
                          serialize_trie(rocode_proc.trie.cursor)}
            else:
                _extra = None
            params = SamplingParams(
                temperature=0.0, top_p=1.0,
                # Leave headroom for what we've already emitted; cap absolute.
                max_tokens=max(64, max_tokens),
                stop=adapter.stop_strings or None,
                # entropy2 needs per-token distributional entropy; top-20
                # logprobs give the standard truncated approximation.
                logprobs=20 if rollback == "entropy2" else None,
                extra_args=_extra,
            )
            prev_n_lp = 0   # cumulative-logprob cursor for this stream
            prev_n_tok = 0  # cumulative-token cursor for this stream

            rollback_triggered = False
            sync_check_ok_continue = False  # sync-only: check passed, restart stream

            async def _consume():
                nonlocal n_tokens, n_rollbacks, pending_check, prev_full
                nonlocal rollback_triggered, sync_check_ok_continue
                nonlocal cur_prompt, prev_n_lp, prev_n_tok
                nonlocal checked_boundary_floor
                async for out in engine.generate(cur_prompt, params, req_id):
                    if time.perf_counter() - t0 >= timeout_s:
                        with contextlib.suppress(Exception):
                            await engine.abort(req_id)
                        return
                    if out.outputs:
                        full = out.outputs[0].text or ""
                        if out.outputs[0].token_ids is not None:
                            n_tokens = max(n_tokens, len(out.outputs[0].token_ids))
                        if len(full) > len(prev_full):
                            new_text = full[len(prev_full):]
                            prev_full = full
                            if rollback == "entropy2":
                                # Record this delta's char span with the mean
                                # Shannon entropy of its newly arrived tokens
                                # (streaming yields ~1 token per chunk).
                                span_start = float(len(code.content))
                                lps = out.outputs[0].logprobs or []
                                new_lps = lps[prev_n_lp:]
                                prev_n_lp = len(lps)
                                hs = []
                                for tok_lp in new_lps:
                                    if tok_lp:
                                        hs.append(-sum(
                                            math.exp(l.logprob) * l.logprob
                                            for l in tok_lp.values()))
                                h = sum(hs) / len(hs) if hs else 0.0
                                ent_spans.append(
                                    [span_start,
                                     span_start + len(new_text), h])
                            if (rollback == "penalty"
                                    and rocode_proc is not None):
                                tids = out.outputs[0].token_ids or []
                                new_ids = list(tids[prev_n_tok:])
                                prev_n_tok = len(tids)
                                if new_ids:
                                    base = len(code.content)
                                    per = len(new_text) / len(new_ids)
                                    for k, tid in enumerate(new_ids):
                                        s = base + int(k * per)
                                        e = (base + len(new_text)
                                             if k == len(new_ids) - 1
                                             else base + int((k + 1) * per))
                                        tok_spans.append((s, e))
                                        rocode_proc.record_emitted_token(
                                            int(tid))
                            code.append(new_text)
                            if code.body_closed:
                                # Function body closed — natural end.
                                with contextlib.suppress(Exception):
                                    await engine.abort(req_id)
                                return
                            b_off = len(code.content.rstrip()) - 1
                            if (code.at_boundary()
                                    and b_off > checked_boundary_floor):
                                if scheduling == "sync":
                                    # Abort the stream, run check inline,
                                    # then return to let outer loop restart.
                                    checked_boundary_floor = b_off
                                    with contextlib.suppress(Exception):
                                        await engine.abort(req_id)
                                    result = await code.check()
                                    if result.is_error:
                                        n_rollbacks += 1
                                        rollback_triggered = True
                                        await _do_rollback(result)
                                    else:
                                        # OK — extend prefix to include all
                                        # verified text and restart stream
                                        # from there. KV cache hit on the
                                        # shared prefix is automatic.
                                        cur_prompt = prob.prompt + code.content
                                        sync_check_ok_continue = True
                                    return
                                else:  # async
                                    # If there's no in-flight check, spawn
                                    # one. If the previous one is done,
                                    # harvest first.
                                    if pending_check is not None and pending_check.done():
                                        try:
                                            done_res = pending_check.result()
                                        except Exception:
                                            done_res = None
                                        pending_check = None
                                        if done_res is not None and done_res.is_error:
                                            n_rollbacks += 1
                                            rollback_triggered = True
                                            with contextlib.suppress(Exception):
                                                await engine.abort(req_id)
                                            await _do_rollback(done_res)
                                            return
                                    if pending_check is None:
                                        checked_boundary_floor = b_off
                                        pending_check = asyncio.create_task(
                                            code.check()
                                        )
                        # Also harvest if a result landed between tokens.
                        if (scheduling == "async"
                                and pending_check is not None
                                and pending_check.done()):
                            try:
                                done_res = pending_check.result()
                            except Exception:
                                done_res = None
                            pending_check = None
                            if done_res is not None and done_res.is_error:
                                n_rollbacks += 1
                                rollback_triggered = True
                                with contextlib.suppress(Exception):
                                    await engine.abort(req_id)
                                await _do_rollback(done_res)
                                return
                    if out.finished:
                        break

            attempt_budget = max(0.5, timeout_s - (time.perf_counter() - t0))
            try:
                await asyncio.wait_for(_consume(), timeout=attempt_budget)
            except asyncio.TimeoutError:
                timed_out = True
                with contextlib.suppress(Exception):
                    await engine.abort(req_id)
                break
            except Exception as e:
                err = repr(e)
                with contextlib.suppress(Exception):
                    await engine.abort(req_id)
                break

            # After consume returns: drain any outstanding async check (for
            # async-mode only; sync already harvested inline).
            if pending_check is not None:
                # If it's not yet done, wait — but bounded by the per-problem
                # budget. We need its verdict because async mode's correctness
                # relies on harvesting end-of-stream checks too.
                try:
                    remaining = max(0.2, timeout_s - (time.perf_counter() - t0))
                    done_res = await asyncio.wait_for(
                        pending_check, timeout=remaining,
                    )
                    if done_res is not None and done_res.is_error:
                        n_rollbacks += 1
                        rollback_triggered = True
                        await _do_rollback(done_res)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pending_check.cancel()
                except Exception:
                    pass
                pending_check = None

            if rollback_triggered:
                # outer loop restarts the stream from the rollback target
                continue
            if sync_check_ok_continue:
                # sync mode: check passed, restart stream to extend.
                # NOTE: this does NOT count toward MAX_ROLLBACKS budget — we
                # only restart because the algorithm requires aborting the
                # stream at each boundary, not because we hit an error. We
                # reset `attempt` so a long no-error run doesn't terminate
                # prematurely. The wall-clock budget remains the true cap.
                attempt = max(0, attempt - 1)
                continue
            break  # natural stream end (body_closed or stop string or EOS)

        if hasattr(checker, "stop"):
            with contextlib.suppress(Exception):
                checker.stop()

        wall = time.perf_counter() - t0
        compile_ok = tests_pass = False
        if code.content and not err and not timed_out:
            try:
                compile_ok, tests_pass = await asyncio.to_thread(
                    adapter.post_hoc_test, prob.prompt, code.content, prob.tests,
                )
            except Exception as e:
                err = f"post_hoc: {e!r}"

        return RunResult(
            arm="", problem_id=prob.name, lang=prob.language,
            wall_time_s=wall, n_tokens=n_tokens,
            n_rollbacks=n_rollbacks, n_checker_calls=code.checks_run,
            compile_ok=compile_ok, tests_pass=tests_pass,
            timeout=timed_out, generated_code=code.content, error=err,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# ─── arm runner ────────────────────────────────────────────────────────────


async def run_arm(
    arm: dict[str, Any],
    engine,
    problems: list[Problem],
    out_path: Path,
    *,
    timeout_s: float,
    max_tokens: int,
    log_every: int = 5,
) -> None:
    """Run an arm over all problems, writing one JSONL line per problem."""
    lang = arm["lang"]
    adapter: LangAdapter = ADAPTERS[lang]()

    out_f = out_path.open("a", buffering=1)
    try:
        for i, prob in enumerate(problems):
            mode = arm["mode"]
            try:
                if mode == "plain":
                    r = await run_plain(
                        engine, prob, adapter,
                        timeout_s=timeout_s, max_tokens=max_tokens,
                    )
                elif mode == "soundcode":
                    r = await run_soundcode(
                        engine, prob, adapter,
                        scheduling=arm["scheduling"],
                        rollback=arm["rollback"],
                        timeout_s=timeout_s, max_tokens=max_tokens,
                        verifier=arm.get("verifier", "cargo"),
                    )
                elif mode == "resample":
                    r = await run_resample(
                        engine, prob, adapter,
                        timeout_s=timeout_s, max_tokens=max_tokens,
                    )
                else:
                    raise ValueError(f"unknown mode: {mode}")
            except Exception as e:
                err = repr(e) + "\n" + traceback.format_exc()
                r = RunResult(
                    arm=arm["id"], problem_id=prob.name, lang=lang,
                    wall_time_s=0.0, n_tokens=0, n_rollbacks=0,
                    n_checker_calls=0, compile_ok=False, tests_pass=False,
                    timeout=False, generated_code="", error=err,
                )
            r.arm = arm["id"]
            out_f.write(json.dumps(r.__dict__) + "\n")
            if (i + 1) % log_every == 0 or i == 0 or i == len(problems) - 1:
                print(
                    f"  [{i+1:3d}/{len(problems)}] {prob.name[:38]:38} "
                    f"wall={r.wall_time_s:5.1f}s rb={r.n_rollbacks} "
                    f"chk={r.n_checker_calls} tok={r.n_tokens} "
                    f"C={'Y' if r.compile_ok else 'N'} "
                    f"P={'Y' if r.tests_pass else 'N'} "
                    f"{'TIMEOUT' if r.timeout else ''} "
                    f"{'ERR' if r.error else ''}",
                    flush=True,
                )
    finally:
        out_f.close()


def _aggregate(arm_results: dict[str, list[dict]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm_id, results in arm_results.items():
        if not results:
            summary[arm_id] = {"n": 0}
            continue
        walls = [r["wall_time_s"] for r in results]
        rbs = [r["n_rollbacks"] for r in results]
        toks = [r["n_tokens"] for r in results]
        chks = [r.get("n_checker_calls", 0) for r in results]
        passes = sum(1 for r in results if r["tests_pass"])
        compiles = sum(1 for r in results if r["compile_ok"])
        timeouts = sum(1 for r in results if r["timeout"])
        errors = sum(1 for r in results if r.get("error"))
        n = len(results)
        summary[arm_id] = {
            "n": n,
            "pass_at_1": passes / n if n else 0.0,
            "compile_rate": compiles / n if n else 0.0,
            "n_passed": passes,
            "n_compiled": compiles,
            "n_timeout": timeouts,
            "n_error": errors,
            "mean_wall_s": statistics.mean(walls) if walls else 0.0,
            "median_wall_s": statistics.median(walls) if walls else 0.0,
            "p90_wall_s": (
                statistics.quantiles(walls, n=10)[-1] if len(walls) >= 10
                else max(walls, default=0.0)
            ),
            "mean_rollbacks": statistics.mean(rbs) if rbs else 0.0,
            "mean_tokens": statistics.mean(toks) if toks else 0.0,
            "mean_checker_calls": statistics.mean(chks) if chks else 0.0,
        }
    return summary


# ─── rocode upstream ───────────────────────────────────────────────────────


async def run_rocode_upstream_cpp() -> tuple[list[RunResult], str | None]:
    """Per task instructions, this arm should run the upstream ROCODE main.py
    on HumanEval-cpp. But the upstream codebase (`reproduce/algo/rocode/
    upstream/`) supports only Python HumanEval and MBPP — its PA_tools runs
    Python `exec()` on the candidate and parses `SyntaxError` /
    `AssertionError`. There is no C++ analogue, and porting it (g++ syntax
    errors + dynamic test execution wrapper for C++) is a multi-day effort.

    We document this in known_issues and skip the arm. The headline
    comparison (arms 3 vs 4, 5 vs 6) does not depend on this arm.
    """
    note = (
        "## Arm 9 (rocode_upstream_cpp): SKIPPED\n\n"
        "The upstream ROCODE codebase (`reproduce/algo/rocode/upstream/`) "
        "supports only Python HumanEval and MBPP. Its `PA_tools.py` runs "
        "Python `exec()` on a candidate snippet and parses `SyntaxError` / "
        "`AssertionError`; there is no C++ program analyzer in the upstream "
        "or any way to compile/run C++ test cases through its interface. "
        "Porting PA to C++ (g++ syntax errors + dynamic test execution) is "
        "a multi-day effort and is out of scope for the paper deadline.\n\n"
        "The headline comparison (arms 3 vs 4, 5 vs 6) does not depend on "
        "this arm — those are 'sync_naive' (our own ROCODE-faithful sync "
        "scheduler) vs 'async_naive' (SoundCode). The sync_naive arm is the "
        "algorithmic equivalent of ROCODE: it blocks on the program analyzer "
        "at every statement boundary, applies a rollback on error, and "
        "restarts. The only difference is the language (C++/Rust vs Python) "
        "and the engine (vLLM vs HuggingFace transformers).\n"
    )
    return [], note


# ─── main ──────────────────────────────────────────────────────────────────


def append_known_issue(text: str) -> None:
    KNOWN_ISSUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KNOWN_ISSUES_PATH.open("a") as f:
        f.write(text.rstrip() + "\n\n")


def _log_toolchain() -> None:
    """Record verifier-toolchain versions in the run log — verifier latency
    is the measured quantity, so these are part of the experiment config."""
    for cmd in (["rustc", "--version"], ["cargo", "--version"],
                ["g++", "--version"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=10).stdout.splitlines()[0]
        except Exception as e:
            out = f"{cmd[0]}: unavailable ({e!r})"
        print(f"  toolchain: {out}", flush=True)


async def amain(args) -> None:
    global RESULTS_DIR, KNOWN_ISSUES_PATH, SENTINEL_PATH
    if args.out_dir:
        RESULTS_DIR = Path(args.out_dir).resolve()
        KNOWN_ISSUES_PATH = RESULTS_DIR / "known_issues.md"
        SENTINEL_PATH = RESULTS_DIR / "DONE"
    _log_toolchain()
    print(f"  results dir: {RESULTS_DIR}", flush=True)
    print(f"  model: {args.model}  dataset: {args.dataset}  "
          f"multipl-e revision: {MULTIPLE_REVISION[:12]}", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not KNOWN_ISSUES_PATH.exists():
        KNOWN_ISSUES_PATH.write_text(
            "# Paper Phase 1 — known issues\n\n"
            "Appended to by `scripts/paper_phase1.py` when an arm is "
            "skipped, partially run, or encounters an infrastructure "
            "failure.\n\n"
        )

    arms = ARMS
    if args.arms:
        wanted = set(args.arms.split(","))
        arms = [a for a in ARMS + EXTRA_ARMS if a["id"] in wanted]
        if not arms:
            print(f"No arms match --arms={args.arms}", file=sys.stderr)
            sys.exit(1)

    arms_to_run: list[dict[str, Any]] = []
    for arm in arms:
        out_path = RESULTS_DIR / f"{arm['id']}.jsonl"
        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            print(f"skip {arm['id']} — {out_path} already exists "
                  f"({out_path.stat().st_size} bytes)", flush=True)
            continue
        arms_to_run.append(arm)

    # rocode_upstream_cpp — handle first (no vLLM needed).
    upstream_arm = next((a for a in arms_to_run
                         if a["id"] == "rocode_upstream_cpp"), None)
    if upstream_arm is not None:
        print(f"\n=== arm: {upstream_arm['id']} (no vLLM) ===", flush=True)
        out_path = RESULTS_DIR / f"{upstream_arm['id']}.jsonl"
        _results, note = await run_rocode_upstream_cpp()
        if note:
            append_known_issue(note)
            print(note, flush=True)
        # Touch the file so we don't re-attempt the arm; mark explicitly.
        out_path.write_text(json.dumps({
            "arm": "rocode_upstream_cpp",
            "skipped": True,
            "reason": "upstream is python-only; see known_issues.md",
        }) + "\n")
        arms_to_run = [a for a in arms_to_run if a["id"] != "rocode_upstream_cpp"]

    engine = None
    if arms_to_run:
        print(f"\nLoading datasets...", flush=True)
        problems_by_lang: dict[str, list[Problem]] = {}
        for lang in {a["lang"] for a in arms_to_run}:
            probs = load_humaneval(lang, family=args.dataset)
            if args.limit:
                probs = probs[: args.limit]
            problems_by_lang[lang] = probs
            print(f"  {lang}: {len(probs)} problems", flush=True)

        print(f"\nBuilding vLLM engine ({args.model})...", flush=True)
        t0 = time.perf_counter()
        with_penalty = any(a.get("rollback") == "penalty"
                           for a in arms_to_run)
        engine = build_vllm_engine(args.model, with_penalty=with_penalty)
        print(f"  ready in {time.perf_counter()-t0:.1f}s", flush=True)

        # Warmup pass (not measured): amortizes cudagraph capture.
        try:
            from vllm import SamplingParams
            req_id = f"pp1-warm-{uuid.uuid4().hex[:6]}"
            async for out in engine.generate(
                "fn main() {",
                SamplingParams(temperature=0.0, max_tokens=8),
                req_id,
            ):
                if out.finished:
                    break
        except Exception as e:
            print(f"  warmup failed (non-fatal): {e!r}", flush=True)

        for arm in arms_to_run:
            print(f"\n=== arm: {arm['id']} ===", flush=True)
            out_path = RESULTS_DIR / f"{arm['id']}.jsonl"
            if out_path.exists():
                out_path.unlink()
            t_arm = time.perf_counter()
            try:
                await run_arm(
                    arm, engine, problems_by_lang[arm["lang"]], out_path,
                    timeout_s=args.timeout, max_tokens=args.max_tokens,
                )
            except Exception as e:
                note = (
                    f"arm `{arm['id']}` raised: {e!r}\n"
                    f"traceback:\n{traceback.format_exc()}"
                )
                append_known_issue(note)
                print(f"  ARM FAILED: {note}", flush=True)
            print(f"  arm wall: {time.perf_counter()-t_arm:.1f}s", flush=True)

        try:
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                res = shutdown()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass

    print("\nAggregating summary...", flush=True)
    arm_results: dict[str, list[dict]] = {}
    for arm in arms:
        out_path = RESULTS_DIR / f"{arm['id']}.jsonl"
        if not out_path.exists():
            continue
        rows: list[dict] = []
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Skip the upstream-arm placeholder.
            if d.get("skipped"):
                continue
            rows.append(d)
        arm_results[arm["id"]] = rows
    summary = _aggregate(arm_results)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    SENTINEL_PATH.write_text(json.dumps({
        "finished_at": time.time(),
        "arms_run": [a["id"] for a in arms],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=DEFAULT_PROBLEM_TIMEOUT_S,
                        help="per-problem wall-clock cap (seconds)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="max tokens per generate call")
    parser.add_argument("--limit", type=int, default=None,
                        help="optional: limit problems per arm (smoke test)")
    parser.add_argument("--arms", type=str, default=None,
                        help="optional: comma-separated arm ids to run")
    parser.add_argument("--force", action="store_true",
                        help="re-run arms even if jsonl exists")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="override results dir "
                             "(default: results/paper_phase1)")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="model registry key "
                             "(see soundcode/vllm_model_registry.py)")
    parser.add_argument("--dataset", type=str, default="humaneval",
                        choices=["humaneval", "mbpp"],
                        help="MultiPL-E benchmark family")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
