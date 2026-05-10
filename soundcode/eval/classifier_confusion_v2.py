"""Diversified classifier confusion matrix.

Fix: previous version's categories were label-determined (each category had
either all positives or all negatives), so FP+TN or TP+FN cells were
structurally empty. This version uses two condition rows that each contain
both positive (should-block) and negative (should-not-block) samples, plus
sub-rows by error class. Result: every cell can be populated.

Conditions × labels:
  R1. Closed fn body × should-block          -> TP candidate
  R1. Closed fn body × should-NOT-block      -> TN candidate
  R2. Open fn body   × should-block          -> TP candidate (mixed: some
                                                demote → FN, others always
                                                blocking → TP)
  R2. Open fn body   × should-NOT-block      -> TN candidate (forward ref,
                                                syntax, past-edge, clean)

Sub-rows expose error class so the matrix shows where misses happen.
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
    open_fn: bool
    label: str        # "blocking" or "non_blocking"
    condition: str    # "closed" or "open"
    err_class: str    # type-mismatch, unresolved-ref, unresolved-import, no-method, missing-arm, trait-bound, syntax-only, none, forward-ref, past-edge, etc.


SAMPLES: list[Sample] = [
    # ============ R1. CLOSED FN BODY × SHOULD-BLOCK (12 samples) ============
    Sample("CB_typeM_return", 'fn f() -> i32 { "hi" }\nfn main() {}\n',
           False, "blocking", "closed", "type-mismatch"),
    Sample("CB_typeM_let", 'fn main() { let _x: i32 = "hi"; }\n',
           False, "blocking", "closed", "type-mismatch"),
    Sample("CB_typeM_arith", 'fn main() { let _x: i32 = 1.5; }\n',
           False, "blocking", "closed", "type-mismatch"),
    Sample("CB_typeM_returnString", "fn g() -> String { 42 }\nfn main() {}\n",
           False, "blocking", "closed", "type-mismatch"),
    Sample("CB_unrIMPORT", "use std::collections::HashMop;\nfn main() {}\n",
           False, "blocking", "closed", "unresolved-import"),
    Sample("CB_unrMODULE", "mod nope; fn main() {}\n",
           False, "blocking", "closed", "unresolved-import"),
    Sample("CB_unrREF_fn", "fn main() { definitely_not_defined(); }\n",
           False, "blocking", "closed", "unresolved-ref"),
    Sample("CB_unrREF_var", "fn main() { let _ = some_variable; }\n",
           False, "blocking", "closed", "unresolved-ref"),
    Sample("CB_unrTYPE", "fn main() { let _x: NotAType = (); }\n",
           False, "blocking", "closed", "unresolved-type"),
    Sample("CB_noMethod_int", "fn main() { let x: i32 = 1; x.no_such_method(); }\n",
           False, "blocking", "closed", "no-method"),
    Sample("CB_noMethod_str", 'fn main() { let s = "hi"; s.totally_fake(); }\n',
           False, "blocking", "closed", "no-method"),
    Sample("CB_match_nonexh", "fn f(x: Option<i32>) -> i32 {\n    match x { Some(n) => n, }\n}\nfn main() {}\n",
           False, "blocking", "closed", "missing-arm"),
    Sample("CB_arity_few", "fn f(_a: i32, _b: i32) {}\nfn main() { f(1); }\n",
           False, "blocking", "closed", "wrong-arity"),
    Sample("CB_arity_many", "fn f(_a: i32) {}\nfn main() { f(1, 2); }\n",
           False, "blocking", "closed", "wrong-arity"),
    Sample("CB_traitBound", "fn need<T: Clone>(_x: T) {}\nstruct N;\nfn main() { need(N); }\n",
           False, "blocking", "closed", "trait-bound"),
    Sample("CB_useAfterMove", "fn main() {\n    let s = String::from(\"a\");\n    let _t = s;\n    let _u = s;\n}\n",
           False, "blocking", "closed", "borrow-check"),

    # ============ R1. CLOSED FN BODY × SHOULD-NOT-BLOCK (10 samples) ============
    Sample("CC_simple_int", 'fn main() { let x: i32 = 42; println!("{}", x); }\n',
           False, "non_blocking", "closed", "none"),
    Sample("CC_generic_fn", "fn pair<T>(x: T, y: T) -> (T, T) { (x, y) }\nfn main() { let _p = pair(1, 2); }\n",
           False, "non_blocking", "closed", "none"),
    Sample("CC_iterator", "fn main() {\n    let v: Vec<i32> = (1..=10).map(|x| x * 2).collect();\n    println!(\"{:?}\", v);\n}\n",
           False, "non_blocking", "closed", "none"),
    Sample("CC_struct", "struct P { x: i32, y: i32 }\nfn main() {\n    let p = P { x: 1, y: 2 };\n    println!(\"{} {}\", p.x, p.y);\n}\n",
           False, "non_blocking", "closed", "none"),
    Sample("CC_match_full", "fn f(x: Option<i32>) -> i32 {\n    match x { Some(n) => n, None => 0 }\n}\nfn main() { let _ = f(Some(1)); }\n",
           False, "non_blocking", "closed", "none"),
    Sample("CC_explicit_ret", "fn f() -> i32 { return 42; }\nfn main() { let _ = f(); }\n",
           False, "non_blocking", "closed", "none"),
    Sample("CC_iter_collect", "fn f() -> Vec<i32> { (0..10).collect() }\nfn main() { let _v = f(); }\n",
           False, "non_blocking", "closed", "none"),
    Sample("CC_match_at", "fn f(x: i32) -> i32 { match x { n @ 0..=10 => n, _ => 0 } }\nfn main() {}\n",
           False, "non_blocking", "closed", "none"),
    Sample("CC_unused_var", 'fn main() { let _unused = 42; }\n',
           False, "non_blocking", "closed", "warning-only"),
    Sample("CC_dead_code", "fn unused_fn() {}\nfn main() {}\n",
           False, "non_blocking", "closed", "warning-only"),

    # ============ R2. OPEN FN BODY × SHOULD-BLOCK (8 samples) ============
    # Subset: errors that are NOT demoted (unresolved-import, no-method, missing-arm, etc.)
    # Expected outcome: classifier should still flag these as blocking.
    Sample("OB_unrIMPORT_open", "use std::collections::HashMop;\nfn f() {\n    let _x: i32 = 1;\n",
           True, "blocking", "open", "unresolved-import"),
    Sample("OB_noMethod_closed_expr", "fn f() {\n    let x: i32 = 1;\n    x.no_such_method();\n",
           True, "blocking", "open", "no-method"),
    Sample("OB_match_nonexh_closed_match",
           "fn f() {\n    let x: Option<i32> = Some(1);\n    let _r = match x {\n        Some(n) => n,\n    };\n",
           True, "blocking", "open", "missing-arm"),
    Sample("OB_traitBound_closed_call",
           "fn need<T: Clone>(_x: T) {}\nstruct N;\nfn f() {\n    need(N);\n",
           True, "blocking", "open", "trait-bound"),
    # Subset: errors that ARE demoted (type-mismatch, unresolved-reference)
    # Expected outcome: classifier returns non-blocking → FN by ground-truth label
    Sample("OB_typeM_demoted", 'fn f() -> i32 {\n    let _x: i32 = "hello";\n',
           True, "blocking", "open", "type-mismatch"),
    Sample("OB_unrREF_demoted", "fn f() {\n    nonexistent_helper_v2();\n",
           True, "blocking", "open", "unresolved-ref"),
    Sample("OB_arity_in_open",
           "fn helper(_a: i32, _b: i32) {}\nfn f() {\n    helper(1);\n",
           True, "blocking", "open", "wrong-arity"),
    Sample("OB_useAfterMove_open",
           "fn f() {\n    let s = String::from(\"a\");\n    let _t = s;\n    let _u = s;\n",
           True, "blocking", "open", "borrow-check"),

    # ============ R2. OPEN FN BODY × SHOULD-NOT-BLOCK (12 samples) ============
    # Forward references (legit demotion candidates)
    Sample("OC_fwd_call", "fn f() {\n    let _r = helper_below();\n",
           True, "non_blocking", "open", "forward-ref"),
    Sample("OC_fwd_struct", "fn f() {\n    let _p = MyStructDefBelow { x: 1 };\n",
           True, "non_blocking", "open", "forward-ref"),
    Sample("OC_fwd_chain",
           "fn f() {\n    let v = make_vec_below();\n    let _s = sum_below(&v);\n",
           True, "non_blocking", "open", "forward-ref"),
    # Pure syntax errors near writing edge
    Sample("OC_missing_semi", "fn f() {\n    let x = 1\n",
           True, "non_blocking", "open", "syntax-only"),
    Sample("OC_unterm_string", 'fn f() {\n    let s = "hello\n',
           True, "non_blocking", "open", "syntax-only"),
    Sample("OC_unclosed_if", "fn f() {\n    if some_cond {\n",
           True, "non_blocking", "open", "syntax-only"),
    Sample("OC_unclosed_paren", 'fn f() {\n    println!("{}", (1 + 2\n',
           True, "non_blocking", "open", "syntax-only"),
    # Clean partial code (no error yet)
    Sample("OC_clean_partial_1stmt", "fn f() {\n    let x: i32 = 1;\n",
           True, "non_blocking", "open", "none"),
    Sample("OC_clean_partial_2stmt",
           "fn f() {\n    let x: i32 = 1;\n    let y: i32 = x + 1;\n",
           True, "non_blocking", "open", "none"),
    Sample("OC_clean_loop", "fn f() {\n    for i in 0..10 {\n        let _ = i;\n    }\n",
           True, "non_blocking", "open", "none"),
    # Diagnostic past writing edge: error in tail that hasn't been "completed"
    Sample("OC_past_edge_typo",
           # the typo is at the writing edge (after the last `;`)
           'fn f() {\n    let x: i32 = 1;\n    Definitelynotatype\n',
           True, "non_blocking", "open", "past-edge"),
    # Severity warning only — should not block
    Sample("OC_unused_only", "fn f() {\n    let _x = 1;\n    let unused_var = 99;\n",
           True, "non_blocking", "open", "warning-only"),
]


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
        errors, source=s.code,
        last_complete_pos=last_pos,
        open_fn_body=s.open_fn,
    )
    classifier_blocked = len(blocking_pairs) > 0
    actually_blocking = (s.label == "blocking")
    if actually_blocking and classifier_blocked: outcome = "TP"
    elif actually_blocking: outcome = "FN"
    elif classifier_blocked: outcome = "FP"
    else: outcome = "TN"
    return {
        "name": s.name,
        "condition": s.condition,
        "label": s.label,
        "err_class": s.err_class,
        "n_errors": len(errors),
        "error_codes": [d.code for d in errors],
        "n_blocking": len(blocking_pairs),
        "blocking_codes": [d.code for d, _r in blocking_pairs],
        "blocking_reasons": [r.reason for _d, r in blocking_pairs],
        "outcome": outcome,
    }


def _print_table(results: list[dict]) -> None:
    """Print 2x2 confusion matrix per condition + breakdown by error class."""
    from collections import defaultdict

    print("\n=== 2x2 confusion matrix per code condition ===\n")
    by_cond: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    for r in results:
        by_cond[r["condition"]][r["outcome"]] += 1
    print(f"{'condition':18}  {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'n':>4}  {'precision':>10} {'recall':>8} {'specificity':>12}")
    print("-" * 90)
    totals = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for cond in ("closed", "open"):
        x = by_cond[cond]
        n = sum(x.values())
        prec = x["TP"] / (x["TP"] + x["FP"]) if x["TP"] + x["FP"] else float("nan")
        rec = x["TP"] / (x["TP"] + x["FN"]) if x["TP"] + x["FN"] else float("nan")
        spec = x["TN"] / (x["TN"] + x["FP"]) if x["TN"] + x["FP"] else float("nan")
        for k in totals: totals[k] += x[k]
        print(f"{cond:18}  {x['TP']:>4} {x['FP']:>4} {x['TN']:>4} {x['FN']:>4} {n:>4}  "
              f"{prec:>10.2f} {rec:>8.2f} {spec:>12.2f}")
    n = sum(totals.values())
    prec = totals["TP"] / (totals["TP"] + totals["FP"]) if totals["TP"] + totals["FP"] else float("nan")
    rec = totals["TP"] / (totals["TP"] + totals["FN"]) if totals["TP"] + totals["FN"] else float("nan")
    spec = totals["TN"] / (totals["TN"] + totals["FP"]) if totals["TN"] + totals["FP"] else float("nan")
    print("-" * 90)
    print(f"{'TOTAL':18}  {totals['TP']:>4} {totals['FP']:>4} {totals['TN']:>4} {totals['FN']:>4} {n:>4}  "
          f"{prec:>10.2f} {rec:>8.2f} {spec:>12.2f}")

    print("\n=== Breakdown by error class (where misses occur) ===\n")
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    for r in results:
        by_class[r["err_class"]][r["outcome"]] += 1
    print(f"{'err_class':22}  {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'n':>4}")
    print("-" * 60)
    for ec in sorted(by_class):
        x = by_class[ec]
        n = sum(x.values())
        print(f"{ec:22}  {x['TP']:>4} {x['FP']:>4} {x['TN']:>4} {x['FN']:>4} {n:>4}")


async def main() -> None:
    results: list[dict] = []
    async with RustAnalyzer(WORKSPACE) as ra:
        await ra.open_file("src/main.rs", "fn main() {}\n")
        await asyncio.sleep(2)
        for s in SAMPLES:
            r = await evaluate(ra, s)
            results.append(r)
            ic = {"TP": "✓ TP", "TN": "✓ TN", "FP": "✗ FP", "FN": "✗ FN"}[r["outcome"]]
            codes = ",".join(c for c in r["error_codes"] if c)[:30]
            print(f"  [{ic}] {r['name']:32} cond={r['condition']:6} cls={r['err_class']:18} got_codes={codes}")

    out = PROJECT / "results" / "week4" / "classifier_confusion_v2.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")
    _print_table(results)


if __name__ == "__main__":
    asyncio.run(main())
