"""Trace what rust-analyzer returns at trigger sites during MGD generation.

Goal: identify whether MGD's harm on Rust comes from
  (a) the LSP returning wrong/missing completions,
  (b) my port's tokenizer-trie / mask-construction logic,
  (c) the model emitting tokens that aren't prefixes of any valid completion
      even when the LSP set looks reasonable.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from soundcode.mgd import (
    RustAnalyzerCompletionsProvider,
    _trailing_post_dot_identifier,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def probe(provider, full_text: str, label: str) -> list[str]:
    """Single probe with a (warm) provider."""
    completions = provider(full_text)
    print(f"\n=== {label} ===")
    print(f"  text-tail = {full_text[-60:]!r}")
    print(f"  trigger:   {_trailing_post_dot_identifier(full_text)!r}")
    print(f"  completions ({len(completions)}):")
    for c in completions[:30]:
        print(f"    {c!r}")
    if len(completions) > 30:
        print(f"    ... and {len(completions) - 30} more")
    return completions


def main() -> None:
    ws = PROJECT_ROOT / "cargo_workspaces" / "diagnose_mgd"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "Cargo.toml").write_text(
        '[package]\nname = "scratch"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "scratch"\npath = "src/main.rs"\n'
    )
    (ws / "src").mkdir(exist_ok=True)
    (ws / "src" / "main.rs").write_text("fn main() {}\n")

    print("Starting (warm) provider...")
    provider = RustAnalyzerCompletionsProvider(ws)
    provider.start()
    # Initial dummy call to give rust-analyzer time to index.
    import time
    time.sleep(5)
    print("  warm.")

    try:
        _run_probes(provider)
    finally:
        provider.stop()


def _run_probes(provider) -> None:
    # ── A. Trivial: typed Vec<f32>, query at `.` ───────────────────────
    probe(
        provider,
        "fn has_close_elements(numbers: Vec<f32>, threshold: f32) -> bool {\n"
        "    numbers.",
        "A: numbers.<cursor> (typed Vec<f32>)",
    )

    # ── B. After let assignment, query at `.` ──────────────────────────
    probe(
        provider,
        "fn separate(s: String) -> Vec<String> {\n"
        "    let mut result: Vec<String> = Vec::new();\n"
        "    let chars: Vec<char> = s.chars().collect();\n"
        "    result.",
        "B: result.<cursor> after typed Vec<String> let",
    )

    # ── C. After `Vec::new()` (un-typed) — most likely to fail ─────────
    probe(
        provider,
        "fn separate(s: String) -> Vec<String> {\n"
        "    let mut result = Vec::new();\n"
        "    result.",
        "C: result.<cursor> after untyped Vec::new",
    )

    # ── D. Mid-method, expression position ─────────────────────────────
    probe(
        provider,
        "fn has_close_elements(numbers: Vec<f32>, threshold: f32) -> bool {\n"
        "    for i in 0..numbers.len() {\n"
        "        for j in (i+1)..numbers.len() {\n"
        "            if (numbers[i] - numbers[j]).",
        "D: (numbers[i] - numbers[j]).<cursor>",
    )

    # ── E. After method chain partial start ────────────────────────────
    probe(
        provider,
        "fn separate(s: String) -> Vec<String> {\n"
        "    s.split_whitespace().",
        "E: s.split_whitespace().<cursor>",
    )

    # ── F. Mock the MGD scenario where small model wrote `re` after dot
    probe(
        provider,
        "fn has_close_elements(numbers: Vec<f32>, threshold: f32) -> bool {\n"
        "    numbers.re",
        "F: numbers.re (partial identifier after dot)",
    )


if __name__ == "__main__":
    main()
