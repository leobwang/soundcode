"""Classifier confusion matrix per code category.

Goal: quantify when the §6 classifier correctly blocks (TP), correctly
passes (TN), false-alarms (FP), or misses (FN) across distinct code
conditions:

  A) Closed body, real semantic error          -> ideally BLOCKING
  B) Closed body, no error (clean Rust)        -> ideally NOT BLOCKING
  C) Open body, real semantic error in
     ALREADY-COMPLETED statement                -> per §6 demoted to
                                                   not-blocking; counts as
                                                   FN (a real error we miss)
  D) Open body, apparent error is a forward
     reference (helper defined later)           -> demotion is CORRECT
                                                   -> not-blocking is TN
  E) In-progress code with only syntax errors  -> ideally NOT BLOCKING (TN)

The output is a per-category TP/FP/TN/FN table plus an overall summary.
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
class Sample:
    name: str
    code: str
    open_fn: bool   # is the writing fn body still open?
    label: str      # "blocking" if real error that should be flagged, "non_blocking" otherwise
    category: str
    note: str


# ---------------- Category A: closed body, real error ----------------
A: list[Sample] = [
    Sample("A1_type_mismatch", 'fn f() -> i32 { "hello" }\nfn main() {}\n', False, "blocking", "A_closed_real_error", "i32 vs &str"),
    Sample("A2_unresolved_import", "use std::collections::HashMop;\nfn main() {}\n", False, "blocking", "A_closed_real_error", "typo'd import"),
    Sample("A3_unresolved_call", "fn main() { definitely_not_defined(); }\n", False, "blocking", "A_closed_real_error", "undefined call"),
    Sample("A4_no_method", "fn main() { let x: i32 = 1; x.no_such_method(); }\n", False, "blocking", "A_closed_real_error", "missing method"),
    Sample("A5_match_nonexh", "fn f(x: Option<i32>) -> i32 {\n    match x {\n        Some(n) => n,\n    }\n}\nfn main() {}\n", False, "blocking", "A_closed_real_error", "non-exhaustive match"),
    Sample("A6_wrong_arity", "fn f(_a: i32, _b: i32) {}\nfn main() { f(1); }\n", False, "blocking", "A_closed_real_error", "wrong number of args"),
    Sample("A7_return_type", "fn g() -> String { 42 }\nfn main() {}\n", False, "blocking", "A_closed_real_error", "i32 vs String"),
    Sample("A8_assign_mismatch", 'fn main() { let _x: i32 = "hello"; }\n', False, "blocking", "A_closed_real_error", "let-binding type mismatch"),
]

# ---------------- Category B: closed body, no error ----------------
B: list[Sample] = [
    Sample("B1_simple_int", 'fn main() { let x: i32 = 42; println!("{}", x); }\n', False, "non_blocking", "B_closed_clean", "trivial valid"),
    Sample("B2_generic_fn", "fn pair<T>(x: T, y: T) -> (T, T) { (x, y) }\nfn main() { let _p = pair(1, 2); }\n", False, "non_blocking", "B_closed_clean", "generic fn"),
    Sample("B3_vec_collect", "fn main() {\n    let v: Vec<i32> = (1..=10).map(|x| x * 2).collect();\n    println!(\"{:?}\", v);\n}\n", False, "non_blocking", "B_closed_clean", "iterator chain"),
    Sample("B4_struct_use", "struct P { x: i32, y: i32 }\nfn main() {\n    let p = P { x: 1, y: 2 };\n    println!(\"{} {}\", p.x, p.y);\n}\n", False, "non_blocking", "B_closed_clean", "struct"),
    Sample("B5_match_full", "fn f(x: Option<i32>) -> i32 {\n    match x {\n        Some(n) => n,\n        None => 0,\n    }\n}\nfn main() { let _ = f(Some(1)); }\n", False, "non_blocking", "B_closed_clean", "exhaustive match"),
    Sample("B6_trait_impl", "struct W;\nimpl std::fmt::Debug for W { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, \"W\") } }\nfn main() { println!(\"{:?}\", W); }\n", False, "non_blocking", "B_closed_clean", "trait impl"),
]

# ---------------- Category C: open body, real error in completed stmt ----------------
C: list[Sample] = [
    Sample("C1_open_type_mismatch", 'fn f() {\n    let _x: i32 = "hello";\n', True, "blocking", "C_open_real_in_closed_stmt", "completed let with type mismatch — real, but classifier demotes"),
    Sample("C2_open_unresolved_import", "use std::collections::HashMop;\nfn f() {\n", True, "blocking", "C_open_real_in_closed_stmt", "module-scope import error; should NOT demote (always blocking)"),
    Sample("C3_open_no_method", "fn f() {\n    let x: i32 = 1;\n    x.no_such_method();\n", True, "blocking", "C_open_real_in_closed_stmt", "no_method on closed expr"),
]

# ---------------- Category D: open body, forward reference ----------------
D: list[Sample] = [
    Sample("D1_forward_call", "fn f() {\n    let _r = helper();\n", True, "non_blocking", "D_open_forward_ref", "calls helper that will be defined below — correct demotion"),
    Sample("D2_forward_struct_use", "fn f() {\n    let _p = MyStruct { x: 1 };\n", True, "non_blocking", "D_open_forward_ref", "uses MyStruct, defined later"),
    Sample("D3_forward_chained", "fn f() {\n    let v = make_vec();\n    let s = sum(&v);\n    let _r = s.checked_div(2);\n", True, "non_blocking", "D_open_forward_ref", "calls multiple helpers"),
]

# ---------------- Category E: in-progress code, syntax errors only ----------------
E: list[Sample] = [
    Sample("E1_missing_semi", "fn f() {\n    let x = 1\n", True, "non_blocking", "E_inprogress_syntax", "missing semicolon"),
    Sample("E2_unterm_string", 'fn f() {\n    let s = "hello\n', True, "non_blocking", "E_inprogress_syntax", "unterminated string"),
    Sample("E3_unclosed_if", "fn f() {\n    if x { \n", True, "non_blocking", "E_inprogress_syntax", "unclosed if"),
    Sample("E4_open_paren", "fn f() {\n    println!(\"{}\", (1 + 2\n", True, "non_blocking", "E_inprogress_syntax", "unclosed paren"),
]

ALL = A + B + C + D + E


async def evaluate(ra: RustAnalyzer, s: Sample) -> dict:
    file_path = "src/main.rs"
    await ra.update_file(file_path, s.code)
    errors = []
    for wait in (0.3, 0.5, 0.8, 1.2):
        await asyncio.sleep(wait)
        errors = await ra.get_errors(file_path)
        if errors:
            break
    bs = find_boundaries(s.code)
    last_pos = bs[-1].offset if bs else len(s.code) - 1
    blocking_pairs = filter_blocking(
        errors,
        source=s.code,
        last_complete_pos=last_pos,
        open_fn_body=s.open_fn,
    )
    n_block = len(blocking_pairs)
    classifier_blocked = n_block > 0
    actually_blocking = (s.label == "blocking")
    if actually_blocking and classifier_blocked:
        outcome = "TP"
    elif actually_blocking and not classifier_blocked:
        outcome = "FN"
    elif not actually_blocking and classifier_blocked:
        outcome = "FP"
    else:
        outcome = "TN"
    return {
        "name": s.name,
        "category": s.category,
        "open_fn": s.open_fn,
        "label": s.label,
        "n_errors": len(errors),
        "error_codes": [d.code for d in errors],
        "n_blocking": n_block,
        "blocking_codes": [d.code for d, _r in blocking_pairs],
        "blocking_reasons": [r.reason for _d, r in blocking_pairs],
        "outcome": outcome,
        "note": s.note,
    }


async def main() -> None:
    results: list[dict] = []
    async with RustAnalyzer(WORKSPACE) as ra:
        await ra.open_file("src/main.rs", "fn main() {}\n")
        await asyncio.sleep(2)
        for s in ALL:
            r = await evaluate(ra, s)
            results.append(r)
            print(_fmt(r))

    out = PROJECT / "results" / "week4" / "classifier_confusion.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")
    _summary(results)


def _fmt(r: dict) -> str:
    icons = {"TP": "✓ TP", "TN": "✓ TN", "FP": "✗ FP", "FN": "✗ FN"}
    return (
        f"  [{icons[r['outcome']]}] {r['name']:30} cat={r['category']:30} "
        f"open={str(r['open_fn']):5} label={r['label']:12} got_codes={','.join(c for c in r['error_codes'] if c)[:25]:25} "
        f"-> blocking={r['n_blocking']}"
    )


def _summary(results: list[dict]) -> None:
    from collections import defaultdict
    cat_outcomes: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    for r in results:
        cat_outcomes[r["category"]][r["outcome"]] += 1
    print("\n=== Confusion matrix per category ===")
    print(f"{'category':32} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'total':>6}")
    print("-" * 60)
    totals = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for cat in sorted(cat_outcomes):
        x = cat_outcomes[cat]
        n = sum(x.values())
        for k in totals:
            totals[k] += x[k]
        print(f"{cat:32} {x['TP']:>4} {x['FP']:>4} {x['TN']:>4} {x['FN']:>4} {n:>6}")
    n = sum(totals.values())
    print("-" * 60)
    print(f"{'TOTAL':32} {totals['TP']:>4} {totals['FP']:>4} {totals['TN']:>4} {totals['FN']:>4} {n:>6}")
    # rates
    n_pos = totals["TP"] + totals["FN"]
    n_neg = totals["TN"] + totals["FP"]
    if n_pos:
        recall = totals["TP"] / n_pos
        print(f"  Recall  (TP / (TP+FN)) = {recall:.2f}")
    if n_neg:
        spec = totals["TN"] / n_neg
        print(f"  Specificity (TN / (TN+FP)) = {spec:.2f}")
    if totals["TP"] + totals["FP"]:
        prec = totals["TP"] / (totals["TP"] + totals["FP"])
        print(f"  Precision (TP / (TP+FP)) = {prec:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
