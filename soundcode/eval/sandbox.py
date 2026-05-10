"""Sandboxed Rust compilation and test execution."""

from __future__ import annotations

import asyncio
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    compiled: bool
    passed: bool
    compile_output: str
    run_output: str
    source: str


async def check_and_run(source: str, *, timeout: float = 30.0) -> RunResult:
    """Compile and run a Rust source file in a temporary directory.

    Returns compilation status, test pass status, and output.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="soundcode_"))
    try:
        # Write source
        src_file = tmpdir / "main.rs"
        src_file.write_text(source)
        out_bin = tmpdir / "main"

        # Compile
        proc = await asyncio.create_subprocess_exec(
            "rustc", str(src_file), "-o", str(out_bin),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return RunResult(
                compiled=False, passed=False,
                compile_output="TIMEOUT", run_output="",
                source=source,
            )

        compile_output = (stdout + stderr).decode(errors="replace")
        if proc.returncode != 0:
            return RunResult(
                compiled=False, passed=False,
                compile_output=compile_output, run_output="",
                source=source,
            )

        # Run tests
        proc = await asyncio.create_subprocess_exec(
            str(out_bin),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return RunResult(
                compiled=True, passed=False,
                compile_output=compile_output, run_output="TIMEOUT",
                source=source,
            )

        run_output = (stdout + stderr).decode(errors="replace")
        passed = proc.returncode == 0

        return RunResult(
            compiled=True, passed=passed,
            compile_output=compile_output, run_output=run_output,
            source=source,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
