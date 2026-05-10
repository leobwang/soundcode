"""Tests for soundcode.eval.boundary.

Covers: simple statements, nested blocks, strings, raw strings, chars,
lifetimes, comments, macros, incomplete code."""

from __future__ import annotations

from soundcode.eval.boundary import find_boundaries, Boundary, BoundaryTracker


def _kinds(src: str) -> list[str]:
    return [b.kind for b in find_boundaries(src)]


def _offsets(src: str) -> list[int]:
    return [b.offset for b in find_boundaries(src)]


def test_empty() -> None:
    assert find_boundaries("") == []


def test_single_statement() -> None:
    src = "let x = 1;"
    bs = find_boundaries(src)
    assert len(bs) == 1
    assert bs[0].kind == ";"
    assert bs[0].offset == src.index(";")


def test_simple_function() -> None:
    src = "fn main() { let x = 1; let y = 2; }"
    # Boundaries: two ; and one closing }
    assert _kinds(src) == [";", ";", "}"]


def test_string_contains_semicolons() -> None:
    src = 'fn main() { let s = "a;b;c;"; }'
    # Only the ; after `;"` (statement terminator) and the final } count
    assert _kinds(src) == [";", "}"]


def test_string_with_escaped_quote() -> None:
    src = 'fn main() { let s = "he\\"llo;"; }'
    assert _kinds(src) == [";", "}"]


def test_raw_string_no_hashes() -> None:
    src = 'fn main() { let s = r"a;b"; }'
    assert _kinds(src) == [";", "}"]


def test_raw_string_with_hashes() -> None:
    src = 'fn main() { let s = r##"a"#b;"##; }'
    assert _kinds(src) == [";", "}"]


def test_byte_string() -> None:
    src = 'fn main() { let s = b"a;b"; }'
    assert _kinds(src) == [";", "}"]


def test_byte_raw_string() -> None:
    src = 'fn main() { let s = br#"a;"#; }'
    assert _kinds(src) == [";", "}"]


def test_line_comment_hides_semicolon() -> None:
    src = "fn main() { // let x = 1;\nlet y = 2; }"
    # Only the real ; on line 2 and the final } should count
    assert _kinds(src) == [";", "}"]


def test_block_comment_hides_semicolon() -> None:
    src = "fn main() { /* let x = 1; */ let y = 2; }"
    assert _kinds(src) == [";", "}"]


def test_nested_block_comment() -> None:
    src = "fn main() { /* /* ; */ */ let y = 2; }"
    # Rust supports nested block comments. The ; inside the inner comment is hidden.
    assert _kinds(src) == [";", "}"]


def test_char_literal_semicolon() -> None:
    src = "fn main() { let c = ';'; }"
    # The semicolon inside the char literal must be ignored
    assert _kinds(src) == [";", "}"]


def test_char_literal_with_escape() -> None:
    src = "fn main() { let c = '\\n'; }"
    assert _kinds(src) == [";", "}"]


def test_char_literal_escaped_quote() -> None:
    src = r"fn main() { let c = '\''; }"
    # The apostrophe is escaped; still a char literal
    assert _kinds(src) == [";", "}"]


def test_lifetime_not_char() -> None:
    src = "fn f<'a>(x: &'a str) -> &'a str { x }"
    # Two lifetimes, one closing brace. Nothing should become a char literal.
    assert _kinds(src) == ["}"]


def test_static_lifetime() -> None:
    src = "fn main() { let s: &'static str = \"hello\"; }"
    assert _kinds(src) == [";", "}"]


def test_paren_suppresses_semicolon() -> None:
    # Inside `()` we suppress — though `;` here isn't legal Rust,
    # the detector shouldn't emit.
    src = "fn main() { f(a; b); }"
    assert _kinds(src) == [";", "}"]
    # First ; is suppressed (inside parens), so we see only the outer ; and }


def test_bracket_suppresses() -> None:
    src = "fn main() { let v = [1; 3]; }"
    # The `;` inside [1; 3] is a repeat-initializer, suppressed by our rule.
    # Only the statement-terminator ; and the closing } appear.
    assert _kinds(src) == [";", "}"]


def test_nested_blocks() -> None:
    src = "fn main() { if x { 1; } else { 2; } }"
    # Boundaries: 1;  }  2;  }  }  — five total
    assert _kinds(src) == [";", "}", ";", "}", "}"]


def test_closure_with_semicolon() -> None:
    src = "fn main() { let f = |x| { x + 1; }; }"
    # Inside closure: x + 1 ;  }.  Then outer ; for the let, and final }
    assert _kinds(src) == [";", "}", ";", "}"]


def test_macro_invocation() -> None:
    src = 'fn main() { println!("hello; world"); }'
    # Macro body contains a string with ; — suppressed.
    # Final `)` ;, then `}` boundary.
    assert _kinds(src) == [";", "}"]


def test_vec_macro() -> None:
    src = "fn main() { let v = vec![1, 2, 3]; }"
    assert _kinds(src) == [";", "}"]


def test_unterminated_string() -> None:
    # String runs to EOF: rest of file is consumed as string content
    src = 'fn main() { let s = "oops; y }'
    # After the unterminated quote, everything is string, so no boundaries
    assert find_boundaries(src) == []


def test_unterminated_block_comment() -> None:
    src = "fn main() { /* unclosed\nlet x = 1; }"
    # After /* everything is consumed as comment; no boundaries emitted.
    assert find_boundaries(src) == []


def test_incomplete_function_body() -> None:
    src = "fn main() { let x = 1;"
    # Only the ; is a boundary; no closing }
    bs = find_boundaries(src)
    assert [b.kind for b in bs] == [";"]


def test_complete_function_then_incomplete() -> None:
    src = """fn first() { let x = 1; }
fn second() { let y ="""
    # Boundaries: first fn's `;`, first fn's `}`. Nothing for incomplete `let y =`.
    assert _kinds(src) == [";", "}"]


def test_struct_definition() -> None:
    src = "struct Point { x: f64, y: f64 }"
    # Struct defs don't use `;` inside; just the closing `}`
    assert _kinds(src) == ["}"]


def test_struct_with_items() -> None:
    src = "struct Point { x: f64, y: f64 } fn origin() -> Point { Point { x: 0.0, y: 0.0 } }"
    # Struct's `}`, then inner struct literal `}`, then fn `}`
    assert _kinds(src) == ["}", "}", "}"]


def test_match_expression() -> None:
    src = "fn f(x: u32) -> u32 { match x { 0 => 1, _ => 2, } }"
    # No `;` — the match is the trailing expr. Just the two `}`.
    assert _kinds(src) == ["}", "}"]


def test_match_with_statement_arms() -> None:
    src = """fn f(x: u32) { match x {
        0 => { let a = 1; println!("{}", a); }
        _ => { println!("other"); }
    } }"""
    # Inside first arm: `;` `;` `}`; second arm: `;` `}`; match `}`; fn `}`.
    # Expected: ; ; } ; } } }
    assert _kinds(src) == [";", ";", "}", ";", "}", "}", "}"]


def test_boundary_offsets_monotone() -> None:
    src = "fn main() { let x = 1; let y = 2; let z = 3; }"
    bs = find_boundaries(src)
    offsets = [b.offset for b in bs]
    assert offsets == sorted(offsets)
    assert len(bs) == 4  # three ; and one }


def test_tracker_incremental() -> None:
    t = BoundaryTracker()
    t.update("fn main() {")
    assert t.boundaries == []
    assert t.latest == -1
    t.update("fn main() { let x = 1;")
    assert len(t.boundaries) == 1
    assert t.boundaries[0].kind == ";"
    t.update("fn main() { let x = 1; }")
    kinds = [b.kind for b in t.boundaries]
    assert kinds == [";", "}"]


def test_double_quote_in_char() -> None:
    src = "fn main() { let c = '\"'; }"
    # Char containing a literal double-quote
    assert _kinds(src) == [";", "}"]


def test_realistic_multi_stmt_fn() -> None:
    src = """
fn factorial(n: u64) -> u64 {
    if n <= 1 {
        return 1;
    }
    let mut result = 1;
    for i in 2..=n {
        result *= i;
    }
    result
}
""".strip()
    # Boundaries within: return ; (1),  inner } (2), result = 1 ; (3),
    # result *= i ; (4), for body } (5), fn } (6) = 6 boundaries
    kinds = _kinds(src)
    assert kinds == [";", "}", ";", ";", "}", "}"]


def test_unterminated_char_treated_as_lifetime() -> None:
    # Edge case: a stray ' that looks like a lifetime
    src = "fn f<'a>(x: &'a [i32]) { for y in x { println!(\"{}\", y); } }"
    # '_a and '_a lifetimes; then 3 boundaries: ;  }  }
    assert _kinds(src) == [";", "}", "}"]


def test_unicode_escape_in_char() -> None:
    src = 'fn main() { let c = \'\\u{1F600}\'; }'
    assert _kinds(src) == [";", "}"]
