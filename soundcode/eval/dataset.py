"""MultiPL-E Rust dataset loading and problem representation."""

from __future__ import annotations

from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class RustProblem:
    name: str
    prompt: str
    tests: str
    stop_tokens: list[str]

    def full_source(self, completion: str) -> str:
        """Assemble the full Rust source file for compilation and testing.

        The dataset tests field starts with '}' (the function's closing brace),
        so we just concatenate: prompt + completion + newline + tests.
        """
        body = completion
        # Strip trailing stop tokens if present (Ollama excludes them,
        # but other backends may include them)
        for tok in self.stop_tokens:
            if body.endswith(tok):
                body = body[: -len(tok)]
        return self.prompt + body + "\n" + self.tests

    def check_source(self, completion: str) -> str:
        """Assemble source for rust-analyzer checking (no tests, just the function)."""
        body = completion
        for tok in self.stop_tokens:
            if body.endswith(tok):
                body = body[: -len(tok)]
        return self.prompt + body + "\n}\n\nfn main() {}\n"


def load_rust_humaneval() -> list[RustProblem]:
    """Load the MultiPL-E HumanEval Rust split."""
    ds = load_dataset("nuprl/MultiPL-E", "humaneval-rs", split="test")
    problems = []
    for row in ds:
        problems.append(
            RustProblem(
                name=row["name"],
                prompt=row["prompt"],
                tests=row["tests"],
                stop_tokens=row["stop_tokens"],
            )
        )
    return problems
