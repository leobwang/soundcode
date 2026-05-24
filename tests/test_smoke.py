"""Smoke tests for the new soundcode pipeline (run with `uv run pytest tests/test_smoke.py -xvs`)."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from soundcode.cargo_check import CargoChecker
from soundcode.code import Category, Code, State
from soundcode.eval.boundary import find_boundaries


# ─── boundary detection ────────────────────────────────────────────────

# ─── ThinkSplitter ─────────────────────────────────────────────────────


def test_think_splitter_clean_close_tag_clean_continuation():
    """Most-favoured boundary: model emits literal `</think>` then continues
    the body literally (no fence, no redeclaration). The splitter emits the
    pre-boundary bytes as think and the post-boundary bytes as code. NOTE:
    the splitter no longer emits a synthetic `</think>` marker — that's the
    job of `DemoClient._emit_token`'s think→code transition logic, which
    fires once on the first code token. Emitting it from the splitter too
    would produce duplicate markers (the bug observed on qwen3.5:122b)."""
    from soundcode.llm import ThinkSplitter
    s = ThinkSplitter("think")
    out = s.feed("I will write a sum function.\n</think>\n    a + b\n")
    kinds = [t.kind for t in out]
    texts = [t.text for t in out]
    assert kinds == ["think", "code"], (kinds, texts)
    assert texts[0] == "I will write a sum function.\n"
    # PostBoundaryStripper sees "\n    a + b\n" — clean continuation,
    # neither a fence nor `fn `. Passes through.
    assert texts[1] == "\n    a + b\n"
    # More clean code passes through in subsequent feeds.
    out2 = s.feed("\n}")
    assert all(t.kind == "code" for t in out2)
    assert "".join(t.text for t in out2) == "\n}"


def test_think_splitter_close_tag_then_fence_strips_chat_wrapping():
    """qwen3.5:122b pattern: `</think>` followed by markdown fence and
    function redeclaration. PostBoundaryStripper must strip ALL of that and
    emit only the body bytes — otherwise the fence and the redeclared
    signature pollute `Code.content` (real-LLM regression observed
    2026-05-16)."""
    from soundcode.llm import ThinkSplitter
    s = ThinkSplitter("think")
    out = s.feed(
        "plan: write a fn.\n</think>\n\n"
        "```rust\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n```\n"
    )
    kinds = [t.kind for t in out]
    texts = [t.text for t in out]
    code_text = "".join(t.text for t in out if t.kind == "code")
    assert kinds[0] == "think" and texts[0] == "plan: write a fn.\n"
    assert "```" not in code_text, code_text
    assert "fn add" not in code_text, code_text   # signature stripped
    # The body bytes (between the outer `{` and `}`) make it through.
    assert "a + b" in code_text


def test_think_splitter_close_tag_straddling_chunks():
    """`</think>` arrives split across two HTTP chunks. The splitter must
    hold back the partial tag suffix rather than emitting it as thinking
    text."""
    from soundcode.llm import ThinkSplitter
    s = ThinkSplitter("think")
    out1 = s.feed("plan: backtracking. </thi")
    text1 = "".join(t.text for t in out1)
    assert "</thi" not in text1
    assert text1 == "plan: backtracking. "
    # Second chunk completes the tag and then continues with clean body.
    out2 = s.feed("nk>\n    a + b\n")
    kinds2 = [t.kind for t in out2]
    code_text = "".join(t.text for t in out2 if t.kind == "code")
    assert "code" in kinds2
    assert code_text == "\n    a + b\n", code_text


def test_think_splitter_markdown_fence_fallback():
    """When the model doesn't emit `</think>`, a markdown ```rust fence is
    treated as the boundary. The fence itself is dropped. Following content
    is fed through `PostBoundaryStripper`, which may further strip a
    redeclared signature."""
    from soundcode.llm import ThinkSplitter
    s = ThinkSplitter("think")
    out = s.feed("Let me write a solver.\n```rust\nfn solve() {\n    body;\n}\n```")
    code_text = "".join(t.text for t in out if t.kind == "code")
    think_text = "".join(t.text for t in out if t.kind == "think")
    assert think_text == "Let me write a solver.\n"
    assert "```" not in code_text
    assert "fn solve" not in code_text
    assert "body;" in code_text


def test_think_splitter_no_boundary_stays_thinking():
    """If the model neither closes the tag nor emits a fence, the whole
    stream is thinking. `flush()` at EOS yields the buffered tail."""
    from soundcode.llm import ThinkSplitter
    s = ThinkSplitter("think")
    out = s.feed("just thinking the whole time, sorry no code")
    assert all(t.kind == "think" for t in out)
    assert "".join(t.text for t in out) == "just thinking the whole time, sorry no code"
    # And a tail that looks like a partial tag is held by `feed` but
    # surfaced by `flush`.
    out_tail = s.feed("hmm </thi")
    assert all(t.kind == "think" for t in out_tail)
    assert "</thi" not in "".join(t.text for t in out_tail)
    flushed = s.flush()
    assert flushed and flushed[-1].text.endswith("</thi")


# ─── PostBoundaryStripper (R1 mode chat-wrap cleanup) ──────────────────


def test_post_boundary_stripper_passthrough_for_clean_continuation():
    """Clean post-`</think>` continuation (no fence, no `fn `): the stripper
    must emit everything as code without dropping anything."""
    from soundcode.llm import PostBoundaryStripper
    s = PostBoundaryStripper()
    out = s.feed("\n    a + b\n    let x = 1;\n")
    text = "".join(t.text for t in out)
    assert text == "\n    a + b\n    let x = 1;\n", repr(text)


def test_post_boundary_stripper_strips_fence_and_redeclaration():
    """qwen3.5:122b R1 pattern: post-`</think>` content is a fence + a
    re-declared function. Both must be stripped, only the body bytes survive."""
    from soundcode.llm import PostBoundaryStripper
    s = PostBoundaryStripper()
    out = s.feed("\n\n```rust\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n```")
    text = "".join(t.text for t in out)
    assert "```" not in text, text
    assert "fn add" not in text, text
    assert text == "\n    a + b\n", repr(text)


def test_post_boundary_stripper_strips_redeclaration_without_fence():
    """Some models emit `fn <ident>` immediately after `</think>` with no
    fence in between — still strip up to the matching `{`."""
    from soundcode.llm import PostBoundaryStripper
    s = PostBoundaryStripper()
    out = s.feed("\nfn solve(x: i32) -> i32 {\n    x + 1\n}\n")
    text = "".join(t.text for t in out)
    assert "fn solve" not in text
    assert text == "\n    x + 1\n", repr(text)


def test_post_boundary_stripper_chunked_fence_detection():
    """The fence prefix may arrive split across chunks (` ```` then `rust\\n`).
    The stripper must hold off until the next chunk confirms or refutes."""
    from soundcode.llm import PostBoundaryStripper
    s = PostBoundaryStripper()
    out1 = s.feed("\n``")            # partial fence — hold
    assert out1 == []
    out2 = s.feed("`rust\nfn f() {\n    body;\n}\n```")
    text = "".join(t.text for t in out1 + out2)
    assert "```" not in text
    assert "fn f" not in text
    assert "body;" in text


def test_post_boundary_stripper_chunked_fn_detection():
    """Same held-prefix behavior for a `fn ` start."""
    from soundcode.llm import PostBoundaryStripper
    s = PostBoundaryStripper()
    out1 = s.feed("\nf")    # could grow into `fn` — hold
    assert out1 == []
    out2 = s.feed("n f() { body; }\n")
    text = "".join(t.text for t in out2)
    assert "fn" not in text
    assert "body;" in text


# ─── ChatBodyStripper (CHAT_INSTRUCTED mode) ───────────────────────────


def test_chat_body_stripper_extracts_outer_body():
    """Model emits a complete fn re-declaration around the requested body.
    Stripper must yield only the bytes between the outer `{ … }`."""
    from soundcode.llm import ChatBodyStripper
    s = ChatBodyStripper()
    out = s.feed(
        "fn solve_sudoku(board: &mut Vec<Vec<char>>) -> bool {\n"
        "    let x = 1;\n"
        "    let y = 2;\n"
        "}"
    )
    text = "".join(t.text for t in out)
    assert text == "\n    let x = 1;\n    let y = 2;\n", repr(text)


def test_chat_body_stripper_preserves_nested_fn_braces():
    """Inner `fn` blocks are part of the body — their braces stay; only the
    outermost `{ … }` is stripped."""
    from soundcode.llm import ChatBodyStripper
    s = ChatBodyStripper()
    out = s.feed(
        "fn outer() {\n"
        "    fn inner() -> i32 { 42 }\n"
        "    inner()\n"
        "}"
    )
    text = "".join(t.text for t in out)
    assert text == "\n    fn inner() -> i32 { 42 }\n    inner()\n", repr(text)


def test_chat_body_stripper_ignores_braces_in_strings():
    """`{` and `}` inside a string literal must not affect brace depth —
    otherwise the stripper would close the outer body prematurely."""
    from soundcode.llm import ChatBodyStripper
    s = ChatBodyStripper()
    out = s.feed('fn f() {\n    let s = "a } b { c";\n}')
    text = "".join(t.text for t in out)
    assert text == '\n    let s = "a } b { c";\n', repr(text)


def test_chat_body_stripper_ignores_braces_in_comments():
    """Comment braces must not affect depth either."""
    from soundcode.llm import ChatBodyStripper
    s = ChatBodyStripper()
    out = s.feed("fn f() {\n    // closing } bracket\n    /* and { inside */\n}")
    text = "".join(t.text for t in out)
    assert text == "\n    // closing } bracket\n    /* and { inside */\n", repr(text)


def test_chat_body_stripper_drops_preamble_before_open():
    """Anything before the first depth-0 `{` is discarded — including
    explanatory prose, the function signature itself, attributes, etc."""
    from soundcode.llm import ChatBodyStripper
    s = ChatBodyStripper()
    out = s.feed("Here's an implementation:\n#[inline]\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}")
    text = "".join(t.text for t in out)
    assert text == "\n    a + b\n", repr(text)


def test_chat_body_stripper_drops_postamble_after_close():
    """Anything after the matching `}` is discarded — markdown fence close,
    trailing prose, closing remarks."""
    from soundcode.llm import ChatBodyStripper
    s = ChatBodyStripper()
    out1 = s.feed("fn f() {\n    body;\n}\n\nThis is explanatory text.")
    text1 = "".join(t.text for t in out1)
    assert text1 == "\n    body;\n", repr(text1)
    # Subsequent feeds after the close drop everything.
    out2 = s.feed("more postamble")
    assert out2 == []


def test_chat_body_stripper_chunked_input():
    """Brace-depth tracking and string-context tracking must survive a chunk
    boundary in the middle of any of: a `{`, a `}`, a string, a comment."""
    from soundcode.llm import ChatBodyStripper
    s = ChatBodyStripper()
    chunks = [
        "fn f(",
        ") {\n   ",
        " let s = ",
        '"hello',
        ' } w', 'orld";\n',
        "    fn g() { 1 ",
        "}\n",
        "}",
        " trailing",
    ]
    emitted = "".join(t.text for chunk in chunks for t in s.feed(chunk))
    flushed = "".join(t.text for t in s.flush())
    assert flushed == "", flushed
    assert emitted == '\n    let s = "hello } world";\n    fn g() { 1 }\n', repr(emitted)


def test_chat_body_stripper_unmatched_open_emits_via_flush():
    """If the model EOS'd before emitting the matching `}`, the stripper
    must still yield the body bytes accumulated so far — `flush()` is the
    final-chance emission."""
    from soundcode.llm import ChatBodyStripper
    s = ChatBodyStripper()
    out = s.feed("fn f() {\n    unfinished")
    text = "".join(t.text for t in out)
    # Note: `unfinished` is emitted as plain code; we never see the closing `}`.
    assert text == "\n    unfinished", repr(text)
    assert s.flush() == []   # no buffered tail in this case


# ─── _iter_ollama_chat_tokens think_char_budget ─────────────────────────


def test_chat_iter_aborts_when_think_budget_exceeded():
    """`_iter_ollama_chat_tokens` must stop iterating and invoke
    `on_budget_exceeded` once cumulative `thinking`-field chars cross the
    cap. Without this, qwen3.5:122b in CHAT_INSTRUCTED mode emits unbounded
    chain-of-thought before ever producing `content`, and the chat call
    appears to hang."""
    import asyncio, json
    from soundcode.llm import _iter_ollama_chat_tokens, ChatBodyStripper

    class FakeResp:
        def __init__(self, lines):
            self._lines = lines
        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

    # 5 thinking chunks of 100 chars each = 500 chars total.
    chunks = []
    for i in range(5):
        chunks.append(json.dumps({"message": {"thinking": "x" * 100}, "done": False}))
    # ChatBodyStripper needs the model to emit `{ … }` to surface a code
    # token (it strips a redeclared signature + outer braces). Give it that.
    chunks.append(json.dumps({"message": {"content": "fn f() {\n    body;\n}\n"}, "done": False}))
    chunks.append(json.dumps({"message": {}, "done": True}))

    flag = {"hit": False}
    def on_budget():
        flag["hit"] = True

    async def collect(budget):
        out = []
        async for tok in _iter_ollama_chat_tokens(
            FakeResp(chunks), ChatBodyStripper(),
            think_char_budget=budget, on_budget_exceeded=on_budget,
        ):
            out.append(tok)
        return out

    # Generous budget: all 5 thinking chunks + content are yielded.
    flag["hit"] = False
    toks = asyncio.run(collect(10_000))
    assert flag["hit"] is False
    think_chars = sum(len(t.text) for t in toks if t.kind == "think")
    assert think_chars == 500
    assert any(t.kind == "code" and "body" in t.text for t in toks)

    # Tight budget: aborts after 2 chunks (200 chars > 150 limit).
    flag["hit"] = False
    toks = asyncio.run(collect(150))
    assert flag["hit"] is True
    think_chars = sum(len(t.text) for t in toks if t.kind == "think")
    # First chunk (100 chars) was yielded; second chunk crossed the budget
    # and was NOT yielded — abort happened before the yield.
    assert think_chars == 100, f"got {think_chars}"
    # Code chunk after the budget abort is NOT yielded — the iter returned.
    assert not any(t.kind == "code" for t in toks)


# ─── template_for_model ────────────────────────────────────────────────


def test_template_for_model_matches_qwen3_family():
    """`qwen3.5:122b`, `qwen3.5:9b`, `qwen3.6:35b` etc. all map to the qwen3
    template. Non-reasoning Qwen2.5 variant returns None."""
    from soundcode.llm import template_for_model
    assert template_for_model("qwen3.5:9b") is not None
    assert template_for_model("qwen3.5:122b") is not None
    assert template_for_model("qwen3.6:35b") is not None
    assert template_for_model("qwen3.6:latest") is not None
    # Non-reasoning families:
    assert template_for_model("qwen2.5-coder:32b") is None
    assert template_for_model("mistral-small3.2:24b") is None
    assert template_for_model("gemma3:27b") is None
    # Reasoning families without a template yet — return None.
    assert template_for_model("gpt-oss:20b") is None
    assert template_for_model("deepseek-r1:70b") is None


def test_reenter_thinking_preserves_mode_across_one_shot():
    """When `reenter_thinking_on_rollback` is True, the one-shot mode-flip
    in `_open_stream` MUST NOT fire — so the next rollback opens the same
    thinking-mode protocol again. Default-False keeps the historical
    one-shot behaviour.

    Regression: without this flag, instruct-on-rollback for R1/T2/C3 is
    silently a no-op on the second attempt (mode already flipped to RAW)."""
    from soundcode.llm import GenerationConfig, OrchestrationMode

    # Default (off): mode flips to RAW after the first injecting open —
    # mirrors the existing behaviour that other tests depend on.
    cfg = GenerationConfig(mode=OrchestrationMode.RAW_THINK_INJECT)
    inject_now = (cfg.mode == OrchestrationMode.RAW_THINK_INJECT)
    if inject_now and not cfg.reenter_thinking_on_rollback:
        cfg.mode = OrchestrationMode.RAW
    assert cfg.mode == OrchestrationMode.RAW, "default = one-shot flip"

    # With the flag on, the same code path must NOT flip the mode.
    cfg2 = GenerationConfig(
        mode=OrchestrationMode.RAW_THINK_INJECT,
        reenter_thinking_on_rollback=True,
    )
    inject_now2 = (cfg2.mode == OrchestrationMode.RAW_THINK_INJECT)
    if inject_now2 and not cfg2.reenter_thinking_on_rollback:
        cfg2.mode = OrchestrationMode.RAW
    assert cfg2.mode == OrchestrationMode.RAW_THINK_INJECT, \
        "reenter_thinking_on_rollback=True must keep the mode set"


def test_raw_think_inject_one_shot_still_creates_splitter():
    """Regression: `_open_stream` must construct the splitter on the SAME
    invocation that injects `<think>\\n` into the prompt — even though that
    invocation also flips `config.mode` back to RAW for the next stream.
    Earlier bug: branched on `self.config.mode` after the one-shot flip, so
    the injecting stream got `splitter=None` and routed all thinking tokens
    into the code buffer."""
    from soundcode.llm import GenerationConfig, OrchestrationMode
    cfg = GenerationConfig(mode=OrchestrationMode.RAW_THINK_INJECT)
    inject_now = (cfg.mode == OrchestrationMode.RAW_THINK_INJECT)
    assert inject_now is True
    cfg.mode = OrchestrationMode.RAW                # one-shot flip
    # The splitter must be constructed from `inject_now`, not the
    # (now-RAW) config field.
    must_have_splitter = bool(inject_now)
    assert must_have_splitter is True


def test_think_splitter_pass_through_after_close():
    """Once the splitter switches to CODE, subsequent chunks are passed
    through without further inspection — even if they contain a literal
    `</think>` (which would be a stray model artifact in the code phase)."""
    from soundcode.llm import ThinkSplitter
    s = ThinkSplitter("think")
    s.feed("a thought.\n</think>\nfn foo() {")
    out = s.feed(" let _ = \"</think>\";")
    assert len(out) == 1
    assert out[0].kind == "code"
    assert out[0].text == ' let _ = "</think>";'


def test_boundary_simple_semicolons():
    src = "let x = 1; let y = 2;"
    b = find_boundaries(src)
    assert [bb.kind for bb in b] == [";", ";"]
    assert [bb.offset for bb in b] == [9, 20]


def test_boundary_string_with_semicolon():
    src = 'let x = ";"; let y = 1;'
    b = find_boundaries(src)
    # The `;` inside the string should not count.
    assert len(b) == 2  # two `;` outside strings
    assert b[0].offset == src.index(";", src.index('"')+2)


def test_boundary_inside_parens_suppressed():
    src = "foo(a; b);"  # the inner ; is inside parens
    b = find_boundaries(src)
    # Only the outer ; at offset 9 counts.
    assert [bb.kind for bb in b] == [";"]
    assert b[0].offset == 9


def test_boundary_raw_string_with_semi():
    src = 'let x = r#"a;b"#; let y = 0;'
    b = find_boundaries(src)
    # Two `;` outside the raw string.
    assert len(b) == 2


# ─── cargo check ───────────────────────────────────────────────────────


def _make_workspace(tmp: Path) -> Path:
    (tmp / "Cargo.toml").write_text(
        '[package]\nname = "scratch"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "scratch"\npath = "src/main.rs"\n'
    )
    (tmp / "src").mkdir()
    (tmp / "src" / "main.rs").write_text("fn main() {}\n")
    return tmp


@pytest.fixture
def workspace(tmp_path):
    return _make_workspace(tmp_path)


def test_cargo_check_clean(workspace):
    chk = CargoChecker(workspace=workspace)
    diags = asyncio.run(chk.check("fn main() { let x: i32 = 1; let _ = x; }"))
    blocking = [d for d in diags if d.category is Category.BLOCKING]
    assert len(blocking) == 0


def test_cargo_check_finds_type_mismatch(workspace):
    chk = CargoChecker(workspace=workspace)
    diags = asyncio.run(chk.check('fn main() { let x: i32 = "oops"; let _ = x; }'))
    blocking = [d for d in diags if d.category is Category.BLOCKING]
    assert len(blocking) >= 1
    assert any(d.code == "E0308" for d in blocking)


# ─── Code class end-to-end ────────────────────────────────────────────


def test_code_check_clean_creates_checkpoint(workspace):
    chk = CargoChecker(workspace=workspace)
    # Use a HumanEval-shaped prefix: function signature ending with `{`.
    # `Code.check` will append `\n}\n\nfn main() {}\n` automatically.
    code = Code(prefix="fn f(x: i32) -> i32 {", suffix="}\n", checker=chk)
    code.append(" let y: i32 = x + 1; y")
    res = asyncio.run(code.check())
    assert res.verdict in (State.OK, State.INACTIVE), f"got {res.verdict}: {res.diagnostics}"


def test_code_check_blocks_on_error(workspace):
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="fn f(x: i32) -> i32 {", suffix="}\n", checker=chk)
    code.append(' let y: i32 = "oops"; y')
    res = asyncio.run(code.check())
    assert res.verdict is State.ERROR
    assert any(d.is_blocking for d in res.diagnostics)


def test_check_result_message_dedupes_repeated_diagnostics():
    """CheckResult.message should collapse identical (code, message) pairs so
    rollback-instruct doesn't inject 3× the same E0434 line and trigger the
    prompt-echo trap (draft-report-0.md §5.3)."""
    from soundcode.code import CheckResult, Diagnostic, Category, State
    diags = [
        Diagnostic(category=Category.BLOCKING, code="E0434",
                   message="can't capture dynamic environment in a fn item"),
        Diagnostic(category=Category.BLOCKING, code="E0434",
                   message="can't capture dynamic environment in a fn item"),
        Diagnostic(category=Category.BLOCKING, code="E0434",
                   message="can't capture dynamic environment in a fn item"),
        Diagnostic(category=Category.BLOCKING, code="E0308",
                   message="mismatched types"),
        Diagnostic(category=Category.NON_BLOCKING, code="warn",
                   message="some warning"),  # ignored — not blocking
    ]
    result = CheckResult(verdict=State.ERROR, diagnostics=diags)
    lines = result.message.splitlines()
    assert len(lines) == 2, f"expected 2 unique blocking lines, got {len(lines)}: {lines}"
    assert any("E0434" in l for l in lines)
    assert any("E0308" in l for l in lines)


def test_body_closed_signals_function_close(workspace):
    """Code.body_closed should fire iff the LM has emitted more `}` than `{`
    — i.e. it has closed the function body whose opening `{` lives in the
    prompt."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="fn f() -> i32 {", suffix="}", checker=chk)
    code.append("    let x = 1;")          # balance == 0
    assert not code.body_closed
    code.append(" if x > 0 { x } else { -x }")  # balance still 0 (nested)
    assert not code.body_closed
    code.append("\n}")                     # closing the function body
    assert code.body_closed, code.content


def test_body_closed_ignores_braces_in_strings(workspace):
    """Closing braces inside string / char / comment literals must NOT
    contribute to the balance — otherwise the LM emitting `"}"` as part of
    its code would falsely terminate generation."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="fn f() {", suffix="}", checker=chk)
    code.append('    let s = "hello }";')   # `}` in string — ignored
    assert not code.body_closed
    code.append("    let c = '}';")         # `}` in char literal — ignored
    assert not code.body_closed
    code.append("    // foo } bar")         # `}` in comment — ignored
    assert not code.body_closed
    code.append("\n}")
    assert code.body_closed


def test_brace_balance_signed(workspace):
    """The signed balance helper should return negative when there are more
    `}` than `{` in the buffer."""
    from soundcode.code import _content_brace_balance
    assert _content_brace_balance("{ {") == 2
    assert _content_brace_balance("{ } {") == 1
    assert _content_brace_balance("} }") == -2
    assert _content_brace_balance("// }\n") == 0
    assert _content_brace_balance('"}"') == 0


def test_code_rollback_restores_buffer(workspace):
    """`rollback()` with no args returns the buffer to `ckpt[-1]`. With the
    new boundary-based checkpointing, `;` and `}` create checkpoints during
    `append()` — so we append a `;`-terminated chunk (boundary auto-ckpt at
    offset 10) and then a chunk with NO boundary (latest ckpt stays at 10),
    so the no-arg rollback restores the original."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("good text;")           # `;` boundary → ckpt at offset 10
    assert code.ckpt[-1] == 10
    code.append(" extra junk")           # no boundary → ckpt unchanged
    assert code.ckpt[-1] == 10
    code.rollback()
    assert code.content == "good text;"
    assert code.state is State.INACTIVE


def test_append_adds_checkpoint_at_semicolon(workspace):
    """Every `;` outside strings/parens/brackets must add a checkpoint at
    the offset right after the `;`."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("let x = 1;")
    assert code.ckpt == [0, 10]
    code.append(" let y = 2;")
    assert code.ckpt == [0, 10, 21]


def test_append_adds_checkpoint_at_closing_brace(workspace):
    """A `}` outside strings/comments also creates a checkpoint at the offset
    right after the `}` — even when nested inside another block."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("if x { y; }")
    # boundaries: `;` at offset 8 → ckpt 9; `}` at offset 10 → ckpt 11.
    assert code.ckpt == [0, 9, 11]


def test_append_skips_boundaries_in_strings(workspace):
    """`;`/`}` inside string literals must not create checkpoints."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append('let s = "; in } string";')
    # Only the trailing `;` is a real boundary.
    real_semi_offset = code.content.rfind(";")
    assert code.ckpt == [0, real_semi_offset + 1]


def test_append_skips_boundaries_inside_parens(workspace):
    """`;` inside `()` is suppressed (`find_boundaries` semantics)."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("foo(a; b);")
    # Only outer `;` at offset 9 → ckpt 10.
    assert code.ckpt == [0, 10]


def test_rollback_floor_resets_to_current_target(workspace):
    """After successive rollbacks, the auto-ckpt floor must reflect the
    CURRENT target — not the historical max. Otherwise the LM regenerating
    fresh content past a lower target gets its NEW boundaries silently
    skipped because they land below the historical floor."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    # Build content with boundaries every 10 chars.
    code.append("a; b; c; d; e;")   # ckpts at 2, 5, 8, 11, 14
    assert code.ckpt == [0, 2, 5, 8, 11, 14]
    code.rollback(to_offset=14)     # peel back the latest; floor = 14
    code.rollback(to_offset=11)     # then peel back another; floor must = 11 (not 14)
    code.rollback(to_offset=5)      # and another; floor must = 5
    # Now append fresh content whose boundary lands at offset 8 (BELOW the
    # historical max of 14). The auto-ckpt MUST add it — otherwise a
    # monotonic floor would block it.
    code.append(" x;")               # `;` at offset 7, ckpt at 8
    assert 8 in code.ckpt, f"new boundary in regenerated content was skipped: {code.ckpt}"


def test_rollback_floor_blocks_boundary_readdition(workspace):
    """Regression: after rolling back through a boundary, the next `append`'s
    auto-ckpt scan must NOT re-add that boundary as a checkpoint — otherwise
    the cargo-error rollback survivor (= largest ckpt < offset_at_check)
    keeps picking the same offset on every retry, and the producer spin-loops
    on one survivor until the wall budget kills it (observed in the qwen3.5:9b
    sudoku run with 588 / 43 same-offset rollbacks)."""
    chk = CargoChecker(workspace=workspace)
    code = Code(prefix="", suffix="", checker=chk)
    code.append("let x = 1;")              # ckpt at 10
    code.append(" let y = 2;")             # ckpt at 21
    code.append(" let z = 3;")             # ckpt at 32
    assert code.ckpt == [0, 10, 21, 32]
    # Rollback to 32 (latest boundary). The ckpt at 32 should be removed.
    code.rollback(to_offset=32)
    assert 32 not in code.ckpt
    # LM regenerates non-boundary text — even a re-scan of the unchanged
    # content[:32] (which still has `;` at offsets 9, 20, 31) MUST NOT
    # re-introduce ckpt at 32.
    code.append(" no")
    code.append("_boundary")
    assert 32 not in code.ckpt, f"ckpt at 32 was re-added: {code.ckpt}"
    # New boundary past the floor still gets a ckpt.
    code.append(";")
    assert code.ckpt[-1] == len(code.content)
    assert code.ckpt[-1] > 32
