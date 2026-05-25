"""Regression test for the Rust-shim leak in `Code.check()` / `DemoClient`.

Previously `soundcode.code.Code.check()` hard-coded the Rust closer
`"\n    unreachable!()\n}\n\nfn main() {}\n"` and appended it to every
source handed to the checker — regardless of language. When `Code` was
wired with `GccChecker` (C++) the leak produced
`'fn' does not name a type` and `'unreachable' was not declared in this
scope` C++ errors. The parallel leak at `soundcode/web/demo_client.py:570`
(`real_source = prompt + content + closer + "\n\nfn main() {}\n"`) had
the same problem.

The fix parameterizes the closer via the new `function_closer` kwarg on
`Code.__init__` (default = Rust shim for back-compat) and threads it
through `DemoClient` from a per-language `LANG_CLOSER` table in
`soundcode.web.server`.

This test pins both fronts:
  - For each non-Rust language (java, cpp, python), constructing `Code`
    with the language's closer and calling `check()` must produce a
    source that contains the language-appropriate closer fragment AND
    does NOT contain any Rust-only tokens (`fn ` with trailing space,
    `unreachable!`).
  - For Rust the default closer is unchanged (back-compat sanity).
  - For `DemoClient`, generating against a per-language closer and
    capturing the source handed to `_emit_final_check`'s checker
    confirms the leak does not reappear at the final-check site.
"""

from __future__ import annotations

import asyncio

import pytest

from soundcode.code import Code
from soundcode.web.demo_client import DemoClient, DemoConfig
from soundcode.web.server import LANG_CLOSER


class RecordingChecker:
    """Records every `source` argument it sees. Returns no diagnostics so
    the Code path falls through to the OK branch (irrelevant here — we
    care only about the source string the checker was invoked with)."""

    def __init__(self) -> None:
        self.sources: list[str] = []

    async def check(self, source: str) -> list:
        self.sources.append(source)
        return []


@pytest.mark.parametrize(
    "lang,fragment_present",
    [
        ("cpp", "return {};"),
        ("java", "    }\n}"),
        ("python", "    pass"),
    ],
)
def test_code_check_uses_per_language_closer(lang: str, fragment_present: str):
    """For each non-Rust language: `Code(function_closer=LANG_CLOSER[lang])`
    must hand the checker a source that contains the language-appropriate
    closer fragment, and must NOT contain the Rust-only tokens that the
    old hard-coded shim leaked."""
    closer = LANG_CLOSER[lang]
    chk = RecordingChecker()
    code = Code(
        prefix=f"// {lang} prefix\n",
        suffix="",
        checker=chk,
        function_closer=closer,
    )
    code.append("some_call();")
    asyncio.run(code.check())
    assert chk.sources, "checker was never invoked"
    src = chk.sources[-1]
    # Rust-only tokens must NOT appear in the per-language source.
    assert "fn " not in src, f"[{lang}] leaked Rust `fn ` token:\n{src!r}"
    assert "unreachable!" not in src, (
        f"[{lang}] leaked Rust `unreachable!` token:\n{src!r}"
    )
    # The language-appropriate closer fragment must appear.
    assert fragment_present in src, (
        f"[{lang}] expected closer fragment {fragment_present!r} in source:\n{src!r}"
    )


def test_code_check_rust_closer_is_byte_identical_default():
    """The default `function_closer` MUST match the historical Rust shim
    so Rust callers and the 134 existing tests stay byte-identical."""
    expected = "\n    unreachable!()\n}\n\nfn main() {}\n"
    assert LANG_CLOSER["rust"] == expected, LANG_CLOSER["rust"]
    chk = RecordingChecker()
    # Construct Code with NO `function_closer` override — default applies.
    code = Code(prefix="fn foo() {", suffix="}", checker=chk)
    code.append(" let x = 1;")
    asyncio.run(code.check())
    assert chk.sources, "checker was never invoked"
    src = chk.sources[-1]
    assert src.endswith(expected), (
        f"Rust default closer was not appended byte-identically:\n"
        f"tail={src[-len(expected):]!r}\nexpected={expected!r}"
    )
    # The Rust path should obviously still contain the Rust tokens — this
    # is the back-compat sanity check, not a leak.
    assert "fn main()" in src
    assert "unreachable!" in src


@pytest.mark.parametrize(
    "lang,fragment_present",
    [
        ("cpp", "return {};"),
        ("java", "    }\n}"),
        ("python", "    pass"),
    ],
)
def test_demo_client_final_check_uses_per_language_closer(lang: str,
                                                          fragment_present: str):
    """`DemoClient._emit_final_check` must compose the source from the
    per-language closer too — the pre-fix leak at line 570 hard-coded
    `\\n\\nfn main() {}\\n` after the body close, polluting every
    language's final-check source with Rust syntax."""
    closer = LANG_CLOSER[lang]
    chk = RecordingChecker()

    async def _emit(_ev: dict) -> None:
        return None

    # The DemoConfig defaults are fine for this test — we never enter
    # `generate()`, just `_emit_final_check` directly.
    demo = DemoClient(
        llm=None,  # type: ignore[arg-type]  # not touched by _emit_final_check
        checker=chk,
        config=DemoConfig(),
        emit=_emit,
        function_closer=closer,
    )
    # Pick content that does NOT end with `}` so the body_closed shortcut
    # falls through and the closer is actually appended. For Python, no
    # closing brace makes sense anyway; for the brace languages, an
    # unfinished statement is the realistic case.
    prompt = f"// {lang} prompt\n"
    content = "some_partial_statement"
    asyncio.run(demo._emit_final_check(prompt, content))
    assert chk.sources, "checker was never invoked by _emit_final_check"
    src = chk.sources[-1]
    assert "fn " not in src, f"[{lang}] leaked Rust `fn ` token:\n{src!r}"
    assert "unreachable!" not in src, (
        f"[{lang}] leaked Rust `unreachable!` token:\n{src!r}"
    )
    assert fragment_present in src, (
        f"[{lang}] expected closer fragment {fragment_present!r} in source:\n{src!r}"
    )
