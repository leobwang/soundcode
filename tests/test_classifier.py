"""Tests for soundcode.eval.classifier.

Constructs synthetic Diagnostic objects and verifies the §6 decision
procedure classifies them correctly.
"""

from __future__ import annotations

import lsprotocol.types as lsp_type

from soundcode.analyzer import Diagnostic, DiagnosticSeverity
from soundcode.eval.classifier import (
    classify,
    filter_blocking,
    line_col_to_offset,
    Verdict,
)


def _range(sl: int, sc: int, el: int, ec: int) -> lsp_type.Range:
    return lsp_type.Range(
        start=lsp_type.Position(line=sl, character=sc),
        end=lsp_type.Position(line=el, character=ec),
    )


def _err(
    message: str,
    *,
    code: str | None = None,
    sl: int = 0,
    sc: int = 0,
    el: int = 0,
    ec: int = 1,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
) -> Diagnostic:
    return Diagnostic(
        message=message,
        severity=severity,
        range=_range(sl, sc, el, ec),
        code=code,
        source="rust-analyzer",
    )


# ----- line_col_to_offset -----


def test_offset_first_line_first_col() -> None:
    src = "hello\nworld"
    assert line_col_to_offset(src, 0, 0) == 0


def test_offset_second_line() -> None:
    src = "hello\nworld"
    assert line_col_to_offset(src, 1, 0) == 6  # after "hello\n"


def test_offset_mid_line() -> None:
    src = "hello\nworld"
    assert line_col_to_offset(src, 1, 3) == 9


def test_offset_past_eof_clamps() -> None:
    src = "hello"
    assert line_col_to_offset(src, 100, 100) == len(src)


# ----- classify -----


def test_severity_gate_warning_is_non_blocking() -> None:
    d = _err("warning message", severity=DiagnosticSeverity.WARNING)
    r = classify(d, source="...", last_complete_pos=100, open_fn_body=False)
    assert r.verdict == Verdict.NON_BLOCKING
    assert r.reason == "severity-not-error"


def test_past_writing_edge_is_non_blocking() -> None:
    src = "fn main() { let x = 1;"
    d = _err("mismatched types", code="E0308", sl=0, sc=22, el=0, ec=25)
    r = classify(d, source=src, last_complete_pos=20, open_fn_body=True)
    assert r.verdict == Verdict.NON_BLOCKING
    assert r.reason == "past-writing-edge"


def test_syntactic_code_always_non_blocking() -> None:
    src = "fn main() { let x = 1; }"
    d = _err("expected `;`", code="syntax-error")
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=False)
    assert r.verdict == Verdict.NON_BLOCKING
    assert r.reason == "syntactic-code"


def test_unterminated_message_is_non_blocking() -> None:
    src = 'fn main() { let s = "oops'
    d = _err('unterminated double-quote string', code=None)
    r = classify(d, source=src, last_complete_pos=10, open_fn_body=True)
    assert r.verdict == Verdict.NON_BLOCKING
    assert r.reason == "incompleteness-message"


def test_expected_in_message_is_non_blocking() -> None:
    src = "fn main() { let x = "
    d = _err("expected expression", code=None)
    r = classify(d, source=src, last_complete_pos=10, open_fn_body=True)
    assert r.verdict == Verdict.NON_BLOCKING


def test_type_mismatch_in_completed_fn_blocks() -> None:
    # Fn body is closed — demotion doesn't apply.
    src = "fn f() -> i32 { \"hi\" }"
    d = _err(
        "expected `i32`, found `&str`",
        code="E0308",
        sl=0, sc=16, el=0, ec=20,
    )
    # last_complete_pos points to final `}` — everything is behind it
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=False)
    assert r.verdict == Verdict.BLOCKING
    assert r.reason == "semantic-in-completed-code"


def test_type_mismatch_with_open_fn_body_demoted() -> None:
    src = "fn f() -> i32 { \"hi\""
    d = _err(
        "expected `i32`, found `&str`",
        code="E0308",
        sl=0, sc=16, el=0, ec=20,
    )
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=True)
    assert r.verdict == Verdict.NON_BLOCKING
    assert r.reason == "demoted-type-mismatch"


def test_unresolved_reference_open_fn_demoted() -> None:
    src = "fn main() { foo();"
    d = _err(
        "cannot find function `foo` in this scope",
        code="E0425",
        sl=0, sc=12, el=0, ec=15,
    )
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=True)
    assert r.verdict == Verdict.NON_BLOCKING


def test_unresolved_reference_closed_fn_blocks() -> None:
    src = "fn main() { foo(); }"
    d = _err(
        "cannot find function `foo` in this scope",
        code="E0425",
        sl=0, sc=12, el=0, ec=15,
    )
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=False)
    assert r.verdict == Verdict.BLOCKING


def test_unresolved_import_always_blocks() -> None:
    """E0432 is module-scope; demotion doesn't apply even with open fn body."""
    src = "use std::collections::HashMop;"
    d = _err(
        "unresolved import",
        code="E0432",
        sl=0, sc=4, el=0, ec=10,
    )
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=True)
    assert r.verdict == Verdict.BLOCKING


def test_missing_match_arm_blocks() -> None:
    src = "fn f(x: Option<i32>) -> i32 { match x { Some(n) => n, } }"
    d = _err(
        "non-exhaustive patterns: `None` not covered",
        code="E0004",
        sl=0, sc=30, el=0, ec=35,
    )
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=False)
    # E0004 isn't in the allowlist but it's also not demotable / syntactic
    assert r.verdict == Verdict.BLOCKING


def test_no_method_always_blocks() -> None:
    src = "fn main() { let x = 1; x.not_a_method(); }"
    d = _err(
        "no method named `not_a_method` found for type `{integer}`",
        code="E0599",
        sl=0, sc=25, el=0, ec=37,
    )
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=True)
    assert r.verdict == Verdict.BLOCKING


def test_unresolved_proc_macro_non_blocking() -> None:
    src = "#[derive(Debug, Clone)]"
    d = _err("can't find proc-macro for `Clone`", code="unresolved-proc-macro")
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=False)
    assert r.verdict == Verdict.NON_BLOCKING
    assert r.reason == "syntactic-code"


def test_filter_blocking_mixed() -> None:
    src = "fn main() { let x = 1; foo();"
    diags = [
        _err("expected `;`", code="syntax-error", sl=0, sc=28),  # non-blocking
        _err("cannot find `foo`", code="E0425", sl=0, sc=23, el=0, ec=26),  # demoted
    ]
    filtered = filter_blocking(
        diags,
        source=src,
        last_complete_pos=len(src) - 1,
        open_fn_body=True,
    )
    assert filtered == []

    # With closed fn body, the E0425 is blocking.
    diags2 = [_err("cannot find `foo`", code="E0425", sl=0, sc=23, el=0, ec=26)]
    filtered2 = filter_blocking(
        diags2,
        source=src,
        last_complete_pos=len(src) - 1,
        open_fn_body=False,
    )
    assert len(filtered2) == 1


def test_multi_line_position_computation() -> None:
    # Multi-line source; ensure offsets are correct after \n
    src = "fn f() {\n    let x = 1;\n    let y = 2;\n}"
    # Line 2 starts at offset 24 (after "fn f() {\n" = 9 and "    let x = 1;\n" = 15).
    # char 8 on line 2 lands on 'y'; char 10 lands on '='.
    assert src[line_col_to_offset(src, 2, 8)] == "y"
    assert src[line_col_to_offset(src, 2, 10)] == "="


def test_past_edge_exact_boundary_not_past() -> None:
    """last_complete_pos == diag_offset means diag is AT the boundary — counts as in-completed (not past)."""
    src = "fn main() { x;"
    # diag at offset 13 (the `;`)
    d = _err("something", code="E0308", sl=0, sc=13, el=0, ec=14)
    # last_complete_pos = 13 — equal to diag offset, not past.
    r = classify(d, source=src, last_complete_pos=13, open_fn_body=True)
    # Open fn body => demoted. But reason should NOT be "past-writing-edge".
    assert r.reason != "past-writing-edge"


# ----- DemotionPolicy ablation -----


def test_policy_ref_only_blocks_type_mismatch() -> None:
    """policy=ref_only: type-mismatch should be BLOCKING even with open fn body."""
    from soundcode.eval.classifier import DemotionPolicy
    src = "fn f() -> i32 { \"hi\""
    d = _err("expected `i32`, found `&str`", code="E0308", sl=0, sc=16, el=0, ec=20)
    pol = DemotionPolicy(demote_type_mismatch=False, demote_unresolved_ref=True)
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=True, policy=pol)
    assert r.verdict == Verdict.BLOCKING


def test_policy_ref_only_still_demotes_unresolved_ref() -> None:
    """policy=ref_only: unresolved-reference should still be demoted with open fn body."""
    from soundcode.eval.classifier import DemotionPolicy
    src = "fn main() { foo();"
    d = _err("cannot find function `foo`", code="E0425", sl=0, sc=12, el=0, ec=15)
    pol = DemotionPolicy(demote_type_mismatch=False, demote_unresolved_ref=True)
    r = classify(d, source=src, last_complete_pos=len(src) - 1, open_fn_body=True, policy=pol)
    assert r.verdict == Verdict.NON_BLOCKING
    assert r.reason == "demoted-unresolved-ref"


def test_policy_none_blocks_everything_real() -> None:
    """policy=none: both type-mismatch and unresolved-reference should block."""
    from soundcode.eval.classifier import DemotionPolicy
    src = "fn f() -> i32 { foo()"
    d_tm = _err("expected `i32`, found `&str`", code="E0308", sl=0, sc=16, el=0, ec=20)
    d_ur = _err("cannot find function `foo`", code="E0425", sl=0, sc=16, el=0, ec=19)
    pol = DemotionPolicy(demote_type_mismatch=False, demote_unresolved_ref=False)
    r1 = classify(d_tm, source=src, last_complete_pos=len(src) - 1, open_fn_body=True, policy=pol)
    r2 = classify(d_ur, source=src, last_complete_pos=len(src) - 1, open_fn_body=True, policy=pol)
    assert r1.verdict == Verdict.BLOCKING
    assert r2.verdict == Verdict.BLOCKING


def test_policy_default_matches_section6() -> None:
    """Default policy demotes both — current §6 behaviour."""
    from soundcode.eval.classifier import DEFAULT_POLICY
    assert DEFAULT_POLICY.demote_type_mismatch is True
    assert DEFAULT_POLICY.demote_unresolved_ref is True
    assert DEFAULT_POLICY.name == "both"


def test_policy_from_name_roundtrip() -> None:
    from soundcode.eval.classifier import policy_from_name
    for n in ("both", "ref_only", "type_only", "none"):
        p = policy_from_name(n)
        assert p.name == n
