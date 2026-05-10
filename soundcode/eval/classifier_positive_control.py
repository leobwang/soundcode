"""Positive control: verify the §6 classifier flags real blocking errors
on closed Rust code.

Each snippet has a fully-closed function body (so the open-fn demotion does
NOT apply) and contains a single obvious error that rust-analyzer's native
diagnostics SHOULD catch. We feed each through the actual analyzer +
classifier pipeline used by arm A.

Outcome interpretation:
- If a snippet produces a blocking classification: classifier is alive on
  that error code. Strengthens the null result during generation.
- If a snippet produces an LSP error but no blocking classification: there
  is a classifier bug or misconfigured demotion.
- If a snippet produces NO LSP error at all: rust-analyzer's native
  diagnostics don't catch this error class. (cargo check would, but native
  doesn't.) Still informative — strengthens the "native is too weak"
  narrative.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from soundcode.analyzer import RustAnalyzer
from soundcode.eval.boundary import find_boundaries
from soundcode.eval.classifier import filter_blocking


PROJECT = Path(__file__).parent.parent.parent
WORKSPACE = PROJECT / "test_workspace"


@dataclass
class Snippet:
    name: str
    code: str
    expected_code: str  # rustc error code we'd expect (E0xxx)
    why: str            # short description


# Closed-body Rust snippets. Each has obvious errors. The body is closed,
# so `open_fn_body=False` — classifier should not demote.
SNIPPETS: list[Snippet] = [
    Snippet(
        "type_mismatch_str_for_i32",
        'fn f() -> i32 { "hello" }\nfn main() {}\n',
        "E0308",
        "Function returns i32 but body is &str — pure type-mismatch on closed body.",
    ),
    Snippet(
        "type_mismatch_assignment",
        "fn main() { let _x: i32 = \"hello\"; }\n",
        "E0308",
        "Assigning &str to i32 in a let binding.",
    ),
    Snippet(
        "unresolved_import",
        "use std::collections::HashMop;\nfn main() {}\n",
        "E0432",
        "Misspelled import — module-scope, never demoted.",
    ),
    Snippet(
        "unresolved_function_call",
        "fn main() { definitely_not_defined(); }\n",
        "E0425",
        "Calling an undefined function from main, no fn body still open.",
    ),
    Snippet(
        "no_such_method",
        "fn main() { let x: i32 = 1; x.no_such_method(); }\n",
        "E0599",
        "Method not found on i32.",
    ),
    Snippet(
        "missing_match_arm",
        """fn f(x: Option<i32>) -> i32 {
    match x {
        Some(n) => n,
    }
}
fn main() {}
""",
        "E0004",
        "Non-exhaustive match on Option (no None arm).",
    ),
    Snippet(
        "trait_bound_not_satisfied",
        """fn require_clone<T: Clone>(_x: T) {}
struct NoClone;
fn main() { require_clone(NoClone); }
""",
        "E0277",
        "NoClone does not implement Clone — trait bound failure.",
    ),
    Snippet(
        "wrong_arity",
        "fn f(_a: i32, _b: i32) {}\nfn main() { f(1); }\n",
        "E0061",
        "Function called with wrong number of arguments.",
    ),
    Snippet(
        "wrong_return_type_int_for_string",
        "fn g() -> String { 42 }\nfn main() {}\n",
        "E0308",
        "Returns i32 but signature says String — closed body type-mismatch.",
    ),
    Snippet(
        "use_after_move",
        """fn main() {
    let s = String::from("a");
    let _t = s;
    let _u = s;
}
""",
        "E0382",
        "Use after move — borrow-checker error, native LSP unlikely to catch.",
    ),
]


async def evaluate(ra: RustAnalyzer, snip: Snippet) -> dict:
    """Run a single snippet through the analyzer + classifier.

    We write to ``src/main.rs`` (the bin target of test_workspace) so
    rust-analyzer actually analyzes it, mirroring the path the arm-A
    runner takes.
    """
    file_path = "src/main.rs"
    await ra.update_file(file_path, snip.code)
    # Allow rust-analyzer time to analyze.
    errors = []
    for wait in (0.3, 0.5, 0.7, 1.0, 1.5):
        await asyncio.sleep(wait)
        errors = await ra.get_errors(file_path)
        if errors:
            break

    # Compute last_complete_pos (very last `;` or `}` in the source) and
    # open_fn_body status. All snippets have closed bodies.
    bs = find_boundaries(snip.code)
    last_pos = bs[-1].offset if bs else len(snip.code) - 1

    blocking_pairs = filter_blocking(
        errors,
        source=snip.code,
        last_complete_pos=last_pos,
        open_fn_body=False,  # by construction, body is closed
    )

    return {
        "name": snip.name,
        "expected_code": snip.expected_code,
        "why": snip.why,
        "n_errors": len(errors),
        "error_codes": [d.code for d in errors],
        "error_messages": [d.message[:120] for d in errors],
        "n_blocking": len(blocking_pairs),
        "blocking_codes": [d.code for d, _r in blocking_pairs],
        "blocking_reasons": [r.reason for _d, r in blocking_pairs],
    }


async def main() -> None:
    results: list[dict] = []
    async with RustAnalyzer(WORKSPACE) as ra:
        await ra.open_file("src/main.rs", "fn main() {}\n")
        await asyncio.sleep(2)
        for snip in SNIPPETS:
            r = await evaluate(ra, snip)
            results.append(r)
            print(_format(r))

    # Save raw report
    out = PROJECT / "results" / "week4" / "positive_control.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    # Summary stats
    n = len(results)
    n_lsp_caught = sum(1 for r in results if r["n_errors"] > 0)
    n_blocked = sum(1 for r in results if r["n_blocking"] > 0)
    print(f"\nSummary: {n} snippets")
    print(f"  rust-analyzer native produced any error:  {n_lsp_caught}/{n}")
    print(f"  classifier flagged at least one blocking: {n_blocked}/{n}")


def _format(r: dict) -> str:
    icon_lsp = "✓" if r["n_errors"] > 0 else "·"
    icon_block = "✓" if r["n_blocking"] > 0 else "·"
    codes = ",".join(c for c in r["error_codes"] if c) or "(none)"
    block_codes = ",".join(c for c in r["blocking_codes"] if c) or "(none)"
    return (
        f"  [{icon_lsp} LSP][{icon_block} BLOCK] {r['name']:36} "
        f"expected={r['expected_code']:5} got_lsp={codes:30} blocking={block_codes}"
    )


if __name__ == "__main__":
    asyncio.run(main())
