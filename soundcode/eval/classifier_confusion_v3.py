"""Classifier confusion matrix with rust-analyzer-derived ground truth.

Methodology fix: do NOT hand-label samples. For each sample, provide both:
  - ``code``: the as-tested form (may be open mid-generation)
  - ``closed_version``: the same code completed (helpers added, fn body
    closed, etc.) — what the model would write once finished

Ground truth label is determined by running rust-analyzer on
``closed_version`` (with ``open_fn_body=False`` so no demotion). If the
analyzer reports any blocking diagnostic on the closed code, ground
truth = "blocking"; otherwise "non_blocking".

This removes hand judgment from the labeling. The classifier is then
evaluated on the original (possibly open) ``code``, with whatever
``open_fn`` flag matches its actual state, and outcomes are scored
against the rust-analyzer-derived ground truth.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
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
    code: str               # as-tested (may be open)
    closed_version: str     # syntactically complete; ground truth comes from
                            # rust-analyzer running on this
    open_fn: bool           # is the as-tested form an open fn body?
    condition: str          # "closed" or "open"
    err_class: str


def _wrap_main() -> str:
    return "fn main() {}\n"


SAMPLES: list[Sample] = [
    # ============ CLOSED FN BODY ============
    # Already closed; closed_version is identical
    Sample("CB_typeM_return", 'fn f() -> i32 { "hi" }\nfn main() {}\n',
           'fn f() -> i32 { "hi" }\nfn main() {}\n',
           False, "closed", "type-mismatch"),
    Sample("CB_typeM_let", 'fn main() { let _x: i32 = "hi"; }\n',
           'fn main() { let _x: i32 = "hi"; }\n',
           False, "closed", "type-mismatch"),
    Sample("CB_typeM_arith", 'fn main() { let _x: i32 = 1.5; }\n',
           'fn main() { let _x: i32 = 1.5; }\n',
           False, "closed", "type-mismatch"),
    Sample("CB_typeM_returnString", "fn g() -> String { 42 }\nfn main() {}\n",
           "fn g() -> String { 42 }\nfn main() {}\n",
           False, "closed", "type-mismatch"),
    Sample("CB_unrIMPORT", "use std::collections::HashMop;\nfn main() {}\n",
           "use std::collections::HashMop;\nfn main() {}\n",
           False, "closed", "unresolved-import"),
    Sample("CB_unrREF_fn", "fn main() { definitely_not_defined(); }\n",
           "fn main() { definitely_not_defined(); }\n",
           False, "closed", "unresolved-ref"),
    Sample("CB_unrREF_var", "fn main() { let _ = some_variable; }\n",
           "fn main() { let _ = some_variable; }\n",
           False, "closed", "unresolved-ref"),
    Sample("CB_unrTYPE", "fn main() { let _x: NotAType = (); }\n",
           "fn main() { let _x: NotAType = (); }\n",
           False, "closed", "unresolved-type"),
    Sample("CB_noMethod_int", "fn main() { let x: i32 = 1; x.no_such_method(); }\n",
           "fn main() { let x: i32 = 1; x.no_such_method(); }\n",
           False, "closed", "no-method"),
    Sample("CB_noMethod_str", 'fn main() { let s = "hi"; s.totally_fake(); }\n',
           'fn main() { let s = "hi"; s.totally_fake(); }\n',
           False, "closed", "no-method"),
    Sample("CB_match_nonexh",
           "fn f(x: Option<i32>) -> i32 {\n    match x { Some(n) => n, }\n}\nfn main() {}\n",
           "fn f(x: Option<i32>) -> i32 {\n    match x { Some(n) => n, }\n}\nfn main() {}\n",
           False, "closed", "missing-arm"),
    Sample("CB_arity_few", "fn f(_a: i32, _b: i32) {}\nfn main() { f(1); }\n",
           "fn f(_a: i32, _b: i32) {}\nfn main() { f(1); }\n",
           False, "closed", "wrong-arity"),
    Sample("CB_arity_many", "fn f(_a: i32) {}\nfn main() { f(1, 2); }\n",
           "fn f(_a: i32) {}\nfn main() { f(1, 2); }\n",
           False, "closed", "wrong-arity"),
    Sample("CB_traitBound",
           "fn need<T: Clone>(_x: T) {}\nstruct N;\nfn main() { need(N); }\n",
           "fn need<T: Clone>(_x: T) {}\nstruct N;\nfn main() { need(N); }\n",
           False, "closed", "trait-bound"),
    Sample("CB_useAfterMove",
           "fn main() {\n    let s = String::from(\"a\");\n    let _t = s;\n    let _u = s;\n}\n",
           "fn main() {\n    let s = String::from(\"a\");\n    let _t = s;\n    let _u = s;\n}\n",
           False, "closed", "borrow-check"),
    # closed clean — closed_version identical
    Sample("CC_simple_int", 'fn main() { let x: i32 = 42; println!("{}", x); }\n',
           'fn main() { let x: i32 = 42; println!("{}", x); }\n',
           False, "closed", "none"),
    Sample("CC_generic_fn",
           "fn pair<T>(x: T, y: T) -> (T, T) { (x, y) }\nfn main() { let _p = pair(1, 2); }\n",
           "fn pair<T>(x: T, y: T) -> (T, T) { (x, y) }\nfn main() { let _p = pair(1, 2); }\n",
           False, "closed", "none"),
    Sample("CC_iterator",
           "fn main() {\n    let v: Vec<i32> = (1..=10).map(|x| x * 2).collect();\n    println!(\"{:?}\", v);\n}\n",
           "fn main() {\n    let v: Vec<i32> = (1..=10).map(|x| x * 2).collect();\n    println!(\"{:?}\", v);\n}\n",
           False, "closed", "none"),
    Sample("CC_struct",
           "struct P { x: i32, y: i32 }\nfn main() {\n    let p = P { x: 1, y: 2 };\n    println!(\"{} {}\", p.x, p.y);\n}\n",
           "struct P { x: i32, y: i32 }\nfn main() {\n    let p = P { x: 1, y: 2 };\n    println!(\"{} {}\", p.x, p.y);\n}\n",
           False, "closed", "none"),
    Sample("CC_match_full",
           "fn f(x: Option<i32>) -> i32 {\n    match x { Some(n) => n, None => 0 }\n}\nfn main() { let _ = f(Some(1)); }\n",
           "fn f(x: Option<i32>) -> i32 {\n    match x { Some(n) => n, None => 0 }\n}\nfn main() { let _ = f(Some(1)); }\n",
           False, "closed", "none"),
    Sample("CC_explicit_ret", "fn f() -> i32 { return 42; }\nfn main() { let _ = f(); }\n",
           "fn f() -> i32 { return 42; }\nfn main() { let _ = f(); }\n",
           False, "closed", "none"),
    Sample("CC_iter_collect", "fn f() -> Vec<i32> { (0..10).collect() }\nfn main() { let _v = f(); }\n",
           "fn f() -> Vec<i32> { (0..10).collect() }\nfn main() { let _v = f(); }\n",
           False, "closed", "none"),
    Sample("CC_match_at",
           "fn f(x: i32) -> i32 { match x { n @ 0..=10 => n, _ => 0 } }\nfn main() {}\n",
           "fn f(x: i32) -> i32 { match x { n @ 0..=10 => n, _ => 0 } }\nfn main() {}\n",
           False, "closed", "none"),
    Sample("CC_unused_var", 'fn main() { let _unused = 42; }\n',
           'fn main() { let _unused = 42; }\n',
           False, "closed", "warning-only"),

    # ============ OPEN FN BODY ============
    # Open with real error in completed sub-statement; closed_version closes
    # the body but keeps the error. Rust-analyzer should flag closed_version.
    Sample("OB_unrIMPORT_open",
           "use std::collections::HashMop;\nfn f() {\n    let _x: i32 = 1;\n",
           "use std::collections::HashMop;\nfn f() {\n    let _x: i32 = 1;\n}\nfn main() { f(); }\n",
           True, "open", "unresolved-import"),
    Sample("OB_noMethod_closed_expr",
           "fn f() {\n    let x: i32 = 1;\n    x.no_such_method();\n",
           "fn f() {\n    let x: i32 = 1;\n    x.no_such_method();\n}\nfn main() { f(); }\n",
           True, "open", "no-method"),
    Sample("OB_match_nonexh_closed",
           "fn f() {\n    let x: Option<i32> = Some(1);\n    let _r = match x {\n        Some(n) => n,\n    };\n",
           "fn f() {\n    let x: Option<i32> = Some(1);\n    let _r = match x {\n        Some(n) => n,\n    };\n}\nfn main() { f(); }\n",
           True, "open", "missing-arm"),
    Sample("OB_traitBound_closed",
           "fn need<T: Clone>(_x: T) {}\nstruct N;\nfn f() {\n    need(N);\n",
           "fn need<T: Clone>(_x: T) {}\nstruct N;\nfn f() {\n    need(N);\n}\nfn main() { f(); }\n",
           True, "open", "trait-bound"),
    Sample("OB_typeM_demoted",
           'fn f() -> i32 {\n    let _x: i32 = "hello";\n',
           'fn f() -> i32 {\n    let _x: i32 = "hello";\n    0\n}\nfn main() { let _ = f(); }\n',
           True, "open", "type-mismatch"),
    Sample("OB_unrREF_demoted",
           "fn f() {\n    nonexistent_helper_v2();\n",
           "fn f() {\n    nonexistent_helper_v2();\n}\nfn main() { f(); }\n",
           True, "open", "unresolved-ref"),
    Sample("OB_arity_in_open",
           "fn helper(_a: i32, _b: i32) {}\nfn f() {\n    helper(1);\n",
           "fn helper(_a: i32, _b: i32) {}\nfn f() {\n    helper(1);\n}\nfn main() { f(); }\n",
           True, "open", "wrong-arity"),
    Sample("OB_useAfterMove_open",
           "fn f() {\n    let s = String::from(\"a\");\n    let _t = s;\n    let _u = s;\n",
           "fn f() {\n    let s = String::from(\"a\");\n    let _t = s;\n    let _u = s;\n}\nfn main() { f(); }\n",
           True, "open", "borrow-check"),

    # Open with forward references; closed_version DEFINES the helpers so
    # rust-analyzer should be clean.
    Sample("OC_fwd_call",
           "fn f() {\n    let _r = helper_below();\n",
           "fn f() {\n    let _r = helper_below();\n}\nfn helper_below() -> i32 { 42 }\nfn main() { f(); }\n",
           True, "open", "forward-ref"),
    Sample("OC_fwd_struct",
           "fn f() {\n    let _p = MyStructDefBelow { x: 1 };\n",
           "fn f() {\n    let _p = MyStructDefBelow { x: 1 };\n}\nstruct MyStructDefBelow { x: i32 }\nfn main() { f(); }\n",
           True, "open", "forward-ref"),
    Sample("OC_fwd_chain",
           "fn f() {\n    let v = make_vec_below();\n    let _s = sum_below(&v);\n",
           "fn f() {\n    let v = make_vec_below();\n    let _s = sum_below(&v);\n}\nfn make_vec_below() -> Vec<i32> { vec![1,2,3] }\nfn sum_below(v: &Vec<i32>) -> i32 { v.iter().sum() }\nfn main() { f(); }\n",
           True, "open", "forward-ref"),

    # Open with syntax errors near writing edge; closed_version is a
    # fixed-syntax completion.
    Sample("OC_missing_semi",
           "fn f() {\n    let x = 1\n",
           "fn f() {\n    let x = 1;\n    let _ = x;\n}\nfn main() { f(); }\n",
           True, "open", "syntax-only"),
    Sample("OC_unterm_string",
           'fn f() {\n    let s = "hello\n',
           'fn f() {\n    let s = "hello";\n    let _ = s;\n}\nfn main() { f(); }\n',
           True, "open", "syntax-only"),
    Sample("OC_unclosed_if",
           "fn f() {\n    if true {\n",
           "fn f() {\n    if true {\n        let _ = 1;\n    }\n}\nfn main() { f(); }\n",
           True, "open", "syntax-only"),
    Sample("OC_unclosed_paren",
           'fn f() {\n    println!("{}", (1 + 2\n',
           'fn f() {\n    println!("{}", (1 + 2));\n}\nfn main() { f(); }\n',
           True, "open", "syntax-only"),

    # Open with clean partial code; closed_version closes cleanly.
    Sample("OC_clean_partial_1stmt",
           "fn f() {\n    let x: i32 = 1;\n",
           "fn f() {\n    let x: i32 = 1;\n    let _ = x;\n}\nfn main() { f(); }\n",
           True, "open", "none"),
    Sample("OC_clean_partial_2stmt",
           "fn f() {\n    let x: i32 = 1;\n    let y: i32 = x + 1;\n",
           "fn f() {\n    let x: i32 = 1;\n    let y: i32 = x + 1;\n    let _ = y;\n}\nfn main() { f(); }\n",
           True, "open", "none"),
    Sample("OC_clean_loop",
           "fn f() {\n    for i in 0..10 {\n        let _ = i;\n    }\n",
           "fn f() {\n    for i in 0..10 {\n        let _ = i;\n    }\n}\nfn main() { f(); }\n",
           True, "open", "none"),

    # Past-edge: error after the last `;`
    Sample("OC_past_edge_typo",
           'fn f() {\n    let x: i32 = 1;\n    Definitelynotatype\n',
           'fn f() {\n    let x: i32 = 1;\n    let _ = x;\n}\nfn main() { f(); }\n',
           True, "open", "past-edge"),

    Sample("OC_unused_only",
           "fn f() {\n    let _x = 1;\n    let unused_var = 99;\n",
           "fn f() {\n    let _x = 1;\n    let unused_var = 99;\n    let _ = unused_var;\n}\nfn main() { f(); }\n",
           True, "open", "warning-only"),
]


async def get_ground_truth(ra: RustAnalyzer, s: Sample) -> tuple[str, list[str]]:
    """Run rust-analyzer on the closed version. Errors → blocking.

    Returns (label, list_of_error_codes).
    """
    file_path = "src/main.rs"
    await ra.update_file(file_path, s.closed_version)
    errors = []
    for wait in (0.4, 0.6, 0.9, 1.3):
        await asyncio.sleep(wait)
        errors = await ra.get_errors(file_path)
        if errors:
            break
    label = "blocking" if errors else "non_blocking"
    codes = [d.code for d in errors]
    return label, codes


async def get_classifier_verdict(ra: RustAnalyzer, s: Sample) -> tuple[str, list[str], list[str]]:
    """Run rust-analyzer + classifier on the as-tested code."""
    file_path = "src/main.rs"
    await ra.update_file(file_path, s.code)
    errors = []
    for wait in (0.4, 0.6, 0.9, 1.3):
        await asyncio.sleep(wait)
        errors = await ra.get_errors(file_path)
        if errors:
            break
    bs = find_boundaries(s.code)
    last_pos = bs[-1].offset if bs else len(s.code) - 1
    blocking_pairs = filter_blocking(
        errors, source=s.code,
        last_complete_pos=last_pos,
        open_fn_body=s.open_fn,
    )
    verdict = "blocking" if blocking_pairs else "non_blocking"
    return verdict, [d.code for d in errors], [d.code for d, _r in blocking_pairs]


async def evaluate(ra: RustAnalyzer, s: Sample) -> dict:
    gt_label, gt_codes = await get_ground_truth(ra, s)
    verdict, raw_codes, block_codes = await get_classifier_verdict(ra, s)
    if gt_label == "blocking" and verdict == "blocking": outcome = "TP"
    elif gt_label == "blocking": outcome = "FN"
    elif verdict == "blocking": outcome = "FP"
    else: outcome = "TN"
    return {
        "name": s.name,
        "condition": s.condition,
        "err_class": s.err_class,
        "open_fn": s.open_fn,
        "ground_truth": gt_label,
        "ground_truth_codes": gt_codes,
        "classifier_verdict": verdict,
        "raw_lsp_codes_on_open": raw_codes,
        "blocking_codes_on_open": block_codes,
        "outcome": outcome,
    }


def _print_table(results: list[dict]) -> None:
    print("\n=== 2x2 confusion matrix per code condition ===\n")
    by_cond: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    for r in results:
        by_cond[r["condition"]][r["outcome"]] += 1
    print(f"{'condition':12}  {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'n':>3}  {'precision':>10} {'recall':>8} {'specificity':>12}")
    print("-" * 80)
    totals = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for cond in ("closed", "open"):
        x = by_cond[cond]
        n = sum(x.values())
        p = x["TP"] / (x["TP"] + x["FP"]) if x["TP"] + x["FP"] else float("nan")
        r = x["TP"] / (x["TP"] + x["FN"]) if x["TP"] + x["FN"] else float("nan")
        s = x["TN"] / (x["TN"] + x["FP"]) if x["TN"] + x["FP"] else float("nan")
        for k in totals: totals[k] += x[k]
        print(f"{cond:12}  {x['TP']:>3} {x['FP']:>3} {x['TN']:>3} {x['FN']:>3} {n:>3}  {p:>10.2f} {r:>8.2f} {s:>12.2f}")
    n = sum(totals.values())
    p = totals["TP"] / (totals["TP"] + totals["FP"]) if totals["TP"] + totals["FP"] else float("nan")
    r = totals["TP"] / (totals["TP"] + totals["FN"]) if totals["TP"] + totals["FN"] else float("nan")
    s = totals["TN"] / (totals["TN"] + totals["FP"]) if totals["TN"] + totals["FP"] else float("nan")
    print("-" * 80)
    print(f"{'TOTAL':12}  {totals['TP']:>3} {totals['FP']:>3} {totals['TN']:>3} {totals['FN']:>3} {n:>3}  {p:>10.2f} {r:>8.2f} {s:>12.2f}")

    print("\n=== Breakdown by error class ===\n")
    by_cls: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    for r in results:
        by_cls[r["err_class"]][r["outcome"]] += 1
    print(f"{'err_class':22}  {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'n':>3}")
    print("-" * 50)
    for ec in sorted(by_cls):
        x = by_cls[ec]
        n = sum(x.values())
        print(f"{ec:22}  {x['TP']:>3} {x['FP']:>3} {x['TN']:>3} {x['FN']:>3} {n:>3}")


async def main() -> None:
    results: list[dict] = []
    async with RustAnalyzer(WORKSPACE) as ra:
        await ra.open_file("src/main.rs", "fn main() {}\n")
        await asyncio.sleep(2)
        for s in SAMPLES:
            r = await evaluate(ra, s)
            results.append(r)
            ic = {"TP": "✓ TP", "TN": "✓ TN", "FP": "✗ FP", "FN": "✗ FN"}[r["outcome"]]
            gtcodes = ",".join(c for c in r["ground_truth_codes"] if c)[:18]
            opcodes = ",".join(c for c in r["raw_lsp_codes_on_open"] if c)[:18]
            print(f"  [{ic}] {r['name']:28} cond={r['condition']:6} cls={r['err_class']:18} "
                  f"GT={r['ground_truth']:12} ({gtcodes:18}) → V={r['classifier_verdict']:12} ({opcodes:18})")

    out = PROJECT / "results" / "week4" / "classifier_confusion_v3.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")
    _print_table(results)


if __name__ == "__main__":
    asyncio.run(main())
