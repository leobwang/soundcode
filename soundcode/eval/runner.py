"""Evaluation runner: generate completions and measure compilation rate + pass@k."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from soundcode.eval.dataset import RustProblem, load_rust_humaneval
from soundcode.eval.model import ModelClient, GenerationConfig
from soundcode.eval.sandbox import check_and_run, RunResult


@dataclass
class ProblemResult:
    name: str
    compiled: bool
    passed: bool
    completion: str
    compile_output: str = ""
    run_output: str = ""
    generation_time: float = 0.0
    attempts: int = 1
    rollbacks: int = 0
    analyzer_errors: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    model: str
    mode: str  # "baseline", "lsp", etc.
    num_problems: int = 0
    num_compiled: int = 0
    num_passed: int = 0
    total_time: float = 0.0
    total_attempts: int = 0
    total_rollbacks: int = 0
    problem_results: list[ProblemResult] = field(default_factory=list)

    @property
    def compilation_rate(self) -> float:
        return self.num_compiled / self.num_problems if self.num_problems else 0.0

    @property
    def pass_rate(self) -> float:
        return self.num_passed / self.num_problems if self.num_problems else 0.0

    @property
    def avg_attempts(self) -> float:
        return self.total_attempts / self.num_problems if self.num_problems else 0.0

    def summary(self) -> str:
        lines = [
            f"Model: {self.model} | Mode: {self.mode}",
            f"Problems: {self.num_problems}",
            f"Compiled: {self.num_compiled}/{self.num_problems} ({self.compilation_rate:.1%})",
            f"Passed:   {self.num_passed}/{self.num_problems} ({self.pass_rate:.1%})",
            f"Time:     {self.total_time:.1f}s",
        ]
        if self.mode != "baseline":
            lines.append(f"Avg attempts: {self.avg_attempts:.2f}")
            lines.append(f"Total rollbacks: {self.total_rollbacks}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "mode": self.mode,
            "num_problems": self.num_problems,
            "num_compiled": self.num_compiled,
            "num_passed": self.num_passed,
            "compilation_rate": self.compilation_rate,
            "pass_rate": self.pass_rate,
            "total_time": self.total_time,
            "total_attempts": self.total_attempts,
            "total_rollbacks": self.total_rollbacks,
            "avg_attempts": self.avg_attempts,
            "problems": [asdict(p) for p in self.problem_results],
        }
        path.write_text(json.dumps(data, indent=2))


async def run_baseline(
    client: ModelClient,
    problems: list[RustProblem],
    config: GenerationConfig | None = None,
    *,
    concurrency: int = 4,
    limit: int | None = None,
) -> EvalResult:
    """Run baseline (unconstrained) generation and evaluation."""
    if limit is not None:
        problems = problems[:limit]

    result = EvalResult(model=client.model, mode="baseline", num_problems=len(problems))
    start = time.time()
    sem = asyncio.Semaphore(concurrency)
    done_count = 0

    async def process_one(prob: RustProblem) -> ProblemResult:
        nonlocal done_count
        async with sem:
            t0 = time.time()
            try:
                completion = await client.complete(prob.prompt, config)
            except Exception as e:
                done_count += 1
                return ProblemResult(
                    name=prob.name, compiled=False, passed=False,
                    completion="", compile_output=f"Generation error: {e}",
                )
            gen_time = time.time() - t0

            source = prob.full_source(completion)
            run_result = await check_and_run(source)

            done_count += 1
            if done_count % 10 == 0:
                print(f"  [{done_count}/{len(problems)}] ...", flush=True)

            return ProblemResult(
                name=prob.name,
                compiled=run_result.compiled,
                passed=run_result.passed,
                completion=completion,
                compile_output=run_result.compile_output,
                run_output=run_result.run_output,
                generation_time=gen_time,
            )

    tasks = [process_one(p) for p in problems]
    problem_results = await asyncio.gather(*tasks)

    for pr in problem_results:
        result.problem_results.append(pr)
        result.total_attempts += pr.attempts
        if pr.compiled:
            result.num_compiled += 1
        if pr.passed:
            result.num_passed += 1

    result.total_time = time.time() - start
    return result


async def run_lsp(
    client: ModelClient,
    problems: list[RustProblem],
    config: GenerationConfig | None = None,
    *,
    concurrency: int = 2,
    limit: int | None = None,
    max_retries: int = 3,
) -> EvalResult:
    """Run generation with rust-analyzer verification and retry.

    For each problem:
    1. Generate the completion
    2. Check with rust-analyzer for type/name errors
    3. If errors, regenerate with error diagnostics in prompt
    4. Repeat up to max_retries times
    5. Finally, compile and run tests
    """
    from soundcode.analyzer import RustAnalyzer

    if limit is not None:
        problems = problems[:limit]

    result = EvalResult(model=client.model, mode="lsp", num_problems=len(problems))
    start = time.time()
    sem = asyncio.Semaphore(concurrency)
    done_count = 0

    project_path = Path(__file__).parent.parent.parent / "test_workspace"

    async def process_one(prob: RustProblem, ra: RustAnalyzer) -> ProblemResult:
        nonlocal done_count
        async with sem:
            t0 = time.time()
            attempts = 0
            rollbacks = 0
            all_errors: list[str] = []
            completion = ""
            error_context = ""

            for attempt in range(max_retries + 1):
                attempts += 1

                # Build prompt with error feedback from previous attempt
                if error_context:
                    augmented = (
                        prob.prompt
                        + "// Error: " + error_context.replace("\n", " ").strip()
                        + "\n"
                    )
                else:
                    augmented = prob.prompt

                try:
                    completion = await client.complete(augmented, config)
                except Exception as e:
                    done_count += 1
                    return ProblemResult(
                        name=prob.name, compiled=False, passed=False,
                        completion="", compile_output=f"Generation error: {e}",
                        attempts=attempts, rollbacks=rollbacks,
                    )

                # Check with rust-analyzer
                check_src = prob.check_source(completion)
                file_path = f"src/{prob.name}.rs"
                try:
                    await ra.update_file(file_path, check_src)
                    # Poll diagnostics: try multiple times with increasing delay
                    errors = []
                    for wait in [0.3, 0.5, 0.7]:
                        await asyncio.sleep(wait)
                        errors = await ra.get_errors(file_path)
                        if errors:
                            break  # Found errors, no need to wait more
                except Exception:
                    # Analyzer failed — just proceed with what we have
                    break

                if not errors:
                    break  # Clean — proceed to final test

                # Record errors and prepare feedback for retry
                rollbacks += 1
                error_msgs = [
                    f"// Line {e.start_line}: {e.message}" for e in errors[:5]
                ]
                error_context = "\n".join(error_msgs) + "\n"
                all_errors.extend(error_msgs)

            gen_time = time.time() - t0

            # Final evaluation: compile and run tests
            source = prob.full_source(completion)
            run_result = await check_and_run(source)

            done_count += 1
            if done_count % 10 == 0:
                print(f"  [{done_count}/{len(problems)}] ...", flush=True)

            return ProblemResult(
                name=prob.name,
                compiled=run_result.compiled,
                passed=run_result.passed,
                completion=completion,
                compile_output=run_result.compile_output,
                run_output=run_result.run_output,
                generation_time=gen_time,
                attempts=attempts,
                rollbacks=rollbacks,
                analyzer_errors=all_errors,
            )

    async with RustAnalyzer(project_path) as ra:
        # Pre-open a placeholder to ensure analyzer is ready
        await ra.open_file("src/main.rs", "fn main() {}\n")
        await asyncio.sleep(2)  # Let workspace load

        tasks = [process_one(p, ra) for p in problems]
        problem_results = await asyncio.gather(*tasks)

    for pr in problem_results:
        result.problem_results.append(pr)
        result.total_attempts += pr.attempts
        result.total_rollbacks += pr.rollbacks
        if pr.compiled:
            result.num_compiled += 1
        if pr.passed:
            result.num_passed += 1

    result.total_time = time.time() - start
    return result


async def run_compiler(
    client: ModelClient,
    problems: list[RustProblem],
    config: GenerationConfig | None = None,
    *,
    concurrency: int = 4,
    limit: int | None = None,
    max_retries: int = 3,
) -> EvalResult:
    """Generate-then-fix baseline: compile, get errors, retry with error feedback."""
    if limit is not None:
        problems = problems[:limit]

    result = EvalResult(model=client.model, mode="compiler", num_problems=len(problems))
    start = time.time()
    sem = asyncio.Semaphore(concurrency)
    done_count = 0

    async def process_one(prob: RustProblem) -> ProblemResult:
        nonlocal done_count
        async with sem:
            t0 = time.time()
            attempts = 0
            rollbacks = 0
            all_errors: list[str] = []
            completion = ""
            error_context = ""

            for attempt in range(max_retries + 1):
                attempts += 1
                if error_context:
                    augmented = prob.prompt + "// Error: " + error_context.replace("\n", " ").strip() + "\n"
                else:
                    augmented = prob.prompt

                try:
                    completion = await client.complete(augmented, config)
                except Exception as e:
                    done_count += 1
                    return ProblemResult(
                        name=prob.name, compiled=False, passed=False,
                        completion="", compile_output=f"Generation error: {e}",
                        attempts=attempts, rollbacks=rollbacks,
                    )

                check_src = prob.check_source(completion)
                run_result = await check_and_run(check_src)
                if run_result.compiled:
                    break

                rollbacks += 1
                err_lines = [l.strip() for l in run_result.compile_output.splitlines() if l.strip().startswith("error")][:3]
                error_context = "; ".join(err_lines)
                all_errors.extend(err_lines)

            gen_time = time.time() - t0
            source = prob.full_source(completion)
            run_result = await check_and_run(source)

            done_count += 1
            if done_count % 10 == 0:
                print(f"  [{done_count}/{len(problems)}] ...", flush=True)

            return ProblemResult(
                name=prob.name, compiled=run_result.compiled, passed=run_result.passed,
                completion=completion, compile_output=run_result.compile_output,
                run_output=run_result.run_output, generation_time=gen_time,
                attempts=attempts, rollbacks=rollbacks, analyzer_errors=all_errors,
            )

    tasks = [process_one(p) for p in problems]
    problem_results = await asyncio.gather(*tasks)

    for pr in problem_results:
        result.problem_results.append(pr)
        result.total_attempts += pr.attempts
        result.total_rollbacks += pr.rollbacks
        if pr.compiled:
            result.num_compiled += 1
        if pr.passed:
            result.num_passed += 1

    result.total_time = time.time() - start
    return result
