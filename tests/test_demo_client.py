"""Tests for the three structural-termination fixes in `soundcode.web.demo_client`.

Each test drives `DemoClient.generate` with a *fake* LlmServer + a *fake*
CargoChecker. The fake LlmServer streams a pre-canned token sequence; the
fake CargoChecker returns whatever diagnostics the test wants. We collect
emitted events and assert on them.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from soundcode.llm import GenerationConfig, OrchestrationMode, Token
from soundcode.web.demo_client import DemoClient, DemoConfig


class FakeLlm:
    """Minimal mock of LlmServer that streams a fixed list of tokens once,
    then reports has_next() == False until set_prompt() is called again
    (mimicking the real server's behavior)."""

    def __init__(self, tokens: list[str]):
        self._all_token_lists = [tokens]
        self._idx = 0
        self._pos = 0
        self._exhausted = False
        self._prompt = ""
        self._set_prompt_count = 0
        # `DemoClient` reads `llm.config.mode` to decide whether to run
        # TWO_PHASE phase 1 before the main loop. The fake LM doesn't have a
        # real config; default to RAW so the dispatch is a no-op.
        self.config = GenerationConfig(mode=OrchestrationMode.RAW)

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt
        self._exhausted = False
        self._set_prompt_count += 1
        # The first set_prompt() (made by `generate()` at start) should NOT
        # advance — we want to play the initial token list. Only subsequent
        # set_prompt() calls (from the rollback path) advance to the next
        # queued attempt.
        if self._set_prompt_count > 1 and self._idx < len(self._all_token_lists) - 1:
            self._idx += 1
        self._pos = 0

    def add_attempt(self, tokens: list[str]) -> None:
        """Queue up another sequence to be streamed after the next rollback."""
        self._all_token_lists.append(tokens)

    def has_next(self) -> bool:
        return not self._exhausted

    async def next(self) -> Token:
        toks = self._all_token_lists[self._idx]
        if self._pos >= len(toks):
            self._exhausted = True
            return Token("", "code")
        t = toks[self._pos]
        self._pos += 1
        # Each fake entry may be a bare str (= code token) or a Token already
        # (= explicit thinking / code). Coerce so test cases can mix.
        return t if isinstance(t, Token) else Token(t, "code")

    async def abort_current_stream(self) -> None:
        self._exhausted = True

    async def close(self) -> None:
        pass


class FakeChecker:
    """Always returns no diagnostics. Used when we want `body_closed` or the
    watchdog to be the termination trigger."""

    async def check(self, source: str) -> list:
        return []


class ScriptedChecker:
    """Returns a pre-scripted list of diagnostics per `check()` call. Used to
    exercise the cargo-error rollback path with a known sequence of verdicts."""

    def __init__(self, scripts: list[list]):
        # Each element is a list of Diagnostic objects to return for one call.
        self._scripts = list(scripts)
        self._call = 0

    async def check(self, source: str) -> list:
        if self._call < len(self._scripts):
            res = self._scripts[self._call]
        else:
            res = []
        self._call += 1
        return res


async def _collect(coro):
    events: list[dict] = []

    async def emit(ev: dict) -> None:
        events.append(ev)

    return events, await coro(emit)


# ─── fix #3: body_closed termination ──────────────────────────────────


@pytest.mark.asyncio
async def test_body_closed_terminates_generation():
    """If the LM emits `}` at column 0 (closing the function body), the loop
    must stop without waiting for any further token, stop string, or wall budget."""
    # The prompt's opening `{` is implicit; content starts at balance 0.
    # Emit some real-looking statements, then close.
    tokens = [
        "    let x = 1;",
        "\n    return x;",
        "\n}",                # this closes the function body — balance goes negative
        "\n\nfn other() {",   # the LM might keep going — we should NOT receive this
        " unreachable",
    ]
    llm = FakeLlm(tokens)
    demo = DemoClient(
        llm=llm, checker=FakeChecker(),
        config=DemoConfig(token_delay_s=0, wall_budget_s=10,
                          repetition_window=0),  # disable watchdog for this test
        emit=lambda ev: asyncio.sleep(0),  # placeholder
    )
    events: list[dict] = []
    async def emit(ev):
        events.append(ev)
    demo.emit = emit
    await demo.generate("fn f() -> i32 {")

    # The 4th token should NEVER be emitted because we stopped at the 3rd.
    emitted_texts = [e["text"] for e in events if e["type"] == "token_emitted"]
    assert "\n\nfn other() {" not in emitted_texts, emitted_texts
    assert any(
        e["type"] == "status" and "function body closed" in e.get("message", "")
        for e in events
    ), [e for e in events if e["type"] == "status"]


@pytest.mark.asyncio
async def test_body_closed_ignores_string_brace():
    """A `}` inside a string literal must not trigger termination."""
    tokens = [
        '    let s = "abc}xyz";',   # `}` inside string — should not terminate
        "\n    let y = 2;",
        "\n}",                       # this is the real body close
    ]
    llm = FakeLlm(tokens)
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=FakeChecker(),
        config=DemoConfig(token_delay_s=0, repetition_window=0),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() {")

    # All 3 tokens should be emitted (we only stop at the LAST one).
    emitted_texts = [e["text"] for e in events if e["type"] == "token_emitted"]
    assert len(emitted_texts) == 3, emitted_texts


# ─── fix #2: repetition watchdog ──────────────────────────────────────


@pytest.mark.asyncio
async def test_repetition_watchdog_triggers_rollback():
    """The watchdog now triggers a rollback (not an abort). We expect:
       - status "repetition detected … rolling back"
       - a rollback event with reason="repetition"
       - rollback_count == 1 after the first detection
       - the buffer is truncated to the latest checkpoint (offset 0 here)
       - a "you were repeating" comment is injected for the retry
    """
    pattern = "    for i in 0..9 { board[i] = 1; }\n"  # 38 chars
    tokens = [pattern] * 5
    llm = FakeLlm(tokens)
    # Push an empty second attempt so the loop terminates after rollback.
    llm.add_attempt([])
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=FakeChecker(),
        config=DemoConfig(token_delay_s=0,
                          repetition_window=60, repetition_threshold=3,
                          wall_budget_s=10),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() {")

    statuses = [e for e in events if e["type"] == "status"]
    assert any("repetition detected" in s.get("message", "") and "rolling back" in s.get("message", "")
               for s in statuses), [s["message"] for s in statuses]

    rb = [e for e in events if e["type"] == "rollback"]
    assert len(rb) >= 1, "expected a rollback event"
    assert rb[0]["reason"] == "repetition"
    assert rb[0]["rollback_count"] == 1
    # Either way: to_offset must be < the buffer length at the time of detection,
    # so SOMETHING got discarded (otherwise the rollback was a no-op).
    assert rb[0]["to_offset"] >= 0
    assert len(rb[0]["discarded"]) > 0


@pytest.mark.asyncio
async def test_repetition_watchdog_resumes_after_rollback():
    """After the watchdog fires + rolls back, the LM must get to stream a
    second attempt — generation MUST NOT terminate at the watchdog point."""
    pattern = "    for i in 0..9 { board[i] = 1; }\n"
    # Attempt 1: pure repetition that fires the watchdog.
    # Attempt 2: a small clean body that closes the function — gives us a
    # clean termination so the test doesn't have to wait for wall budget.
    llm = FakeLlm([pattern] * 5)
    llm.add_attempt(["    let x = 1;\n", "\n}"])
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=FakeChecker(),
        config=DemoConfig(token_delay_s=0,
                          repetition_window=60, repetition_threshold=3,
                          wall_budget_s=10),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() {")
    # Must have fired the watchdog AND then emitted tokens from attempt #2.
    statuses = [s.get("message","") for s in events if s["type"] == "status"]
    assert any("repetition detected" in m for m in statuses), statuses
    # Final body-close status from attempt #2 — proves the loop kept running
    # after the watchdog rolled back.
    assert any("function body closed" in m for m in statuses), statuses
    # The "let x = 1" tokens from attempt 2 must have been emitted.
    emitted = [e["text"] for e in events if e["type"] == "token_emitted"]
    assert "    let x = 1;\n" in emitted


@pytest.mark.asyncio
async def test_cargo_error_rollback_targets_boundary_before_error():
    """With boundary-based checkpointing, a cargo-error rollback should land
    on the boundary BEFORE the check's offset (not at-or-past it). And on a
    second failure with the LM regenerating without crossing a new boundary,
    the rollback must retreat further (not re-pick the same survivor) —
    that's what stops the producer from spin-looping on one offset."""
    from soundcode.code import Category, Diagnostic

    # 3 statements; only stmt #3 errors. stmt #1 ends at offset 10
    # (boundary ckpt at 11), stmt #2 ends at offset 21 (ckpt at 22),
    # stmt #3 ends at offset 32 (ckpt at 33).
    tokens = [
        "let x = 1;",     # offset 0..10, `;` at 9 → ckpt at 10
        " let y = 2;",    # offset 10..21, `;` at 20 → ckpt at 21
        " bad_call();",   # offset 21..33, `;` at 32 → ckpt at 33
    ]
    err = Diagnostic(category=Category.BLOCKING, code="E0425",
                     message="cannot find function `bad_call`")
    # 1st check (after stmt #1): clean.
    # 2nd check (after stmt #2): clean.
    # 3rd check (after stmt #3): error. Triggers rollback to ckpt < 33.
    # Latest ckpt < 33 is 21 → survivor = 21 (after stmt #2).
    llm = FakeLlm(tokens)
    # After rollback, the LM has nothing more queued; the loop exits.
    llm.add_attempt([])
    checker = ScriptedChecker([[], [], [err]])
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=checker,
        config=DemoConfig(token_delay_s=0, repetition_window=0,
                          wall_budget_s=5, max_continuations=0),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() {")
    rbs = [e for e in events if e["type"] == "rollback"]
    assert len(rbs) >= 1, [e["type"] for e in events]
    # survivor must be strictly before the erroring check's offset (33).
    assert rbs[0]["to_offset"] == 21, rbs[0]
    assert rbs[0]["reason"] == "cargo_error"


def test_is_body_trivially_empty_recognises_whitespace_and_comments():
    """The empty-body detector must classify whitespace and comments as
    'no code' so the final-check path can re-prompt the LM instead of
    emitting an empty solution as 'complete'."""
    from soundcode.web.demo_client import _is_body_trivially_empty as fn
    assert fn("")
    assert fn("   \n\t\n")
    assert fn("// just a comment\n")
    assert fn("/* block */\n// line\n")
    assert fn("\n/* a */ \n   // b\n  /* c */\n")
    # Real code must NOT be classified as empty.
    assert not fn("42")
    assert not fn("let x = 1;")
    assert not fn("    // explanatory\n    return 1;\n")
    assert not fn('let s = "// not a comment";')


@pytest.mark.skipif(
    pytest.importorskip("os").environ.get("RUN_REAL_LLM") != "1",
    reason="Set RUN_REAL_LLM=1 to run the real-LLM per-mode integration test",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [
    OrchestrationMode.RAW,
    OrchestrationMode.RAW_THINK_INJECT,
    OrchestrationMode.TWO_PHASE,
    OrchestrationMode.CHAT_INSTRUCTED,
])
async def test_real_llm_per_mode(mode):
    """Drive DemoClient against qwen3.5:9b in every orchestration mode.

    Assertions per mode:
    - All modes: generation terminates within wall budget, final content is
      non-trivial (≥10 chars of actual code).
    - RAW: zero `kind="think"` tokens emit.
    - R1/T2/C3: at least one `kind="think"` token emits AND the run produces
      code (kind="code" tokens > 0).
    """
    import time
    from pathlib import Path
    from soundcode.cargo_check import CargoChecker
    from soundcode.llm import LlmServer, GenerationConfig

    # Set up the same workspace pattern the server uses.
    ws_root = Path("/tmp") / f"soundcode_test_ws_{mode.value}"
    ws_root.mkdir(parents=True, exist_ok=True)
    (ws_root / "Cargo.toml").write_text(
        '[package]\nname = "scratch"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[[bin]]\nname = "scratch"\npath = "src/main.rs"\n'
    )
    (ws_root / "src").mkdir(parents=True, exist_ok=True)
    (ws_root / "src" / "main.rs").write_text("fn main() {}\n")

    checker = CargoChecker(workspace=ws_root)
    # CHAT_INSTRUCTED and RAW_THINK_INJECT need looser stops because the
    # model may re-declare the function with `\nfn `.
    if mode in (OrchestrationMode.CHAT_INSTRUCTED, OrchestrationMode.RAW_THINK_INJECT):
        stop = ["\n}\n\n//"]
    else:
        stop = ["\n}", "\nfn ", "\npub fn ", "\nimpl ", "\n}\n\n//"]
    llm = LlmServer(
        model="qwen3.5:9b",
        config=GenerationConfig(max_tokens=-1, stop=stop, mode=mode),
    )
    await llm.warmup()
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=checker,
        config=DemoConfig(
            instruct_on_rollback=False, wall_budget_s=180.0,
            token_delay_s=0, repetition_window=60, max_continuations=2,
        ),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    prompt = (
        "/// Add two integers.\n"
        "fn add(a: i32, b: i32) -> i32 {"
    )
    t0 = time.perf_counter()
    final = await demo.generate(prompt)
    elapsed = time.perf_counter() - t0
    await llm.close()

    think_tokens = [e for e in events
                    if e["type"] == "token_emitted" and e.get("kind") == "think"]
    code_tokens  = [e for e in events
                    if e["type"] == "token_emitted" and e.get("kind") == "code"]
    print(f"\n[{mode.value}] elapsed={elapsed:.1f}s "
          f"think={len(think_tokens)} code={len(code_tokens)} "
          f"final_len={len(final)}")

    # All modes: must terminate (events have a final) and produce code.
    assert any(e["type"] == "final" for e in events), \
        f"[{mode.value}] no final event; events: {[e['type'] for e in events[-10:]]}"
    assert len(code_tokens) > 0, f"[{mode.value}] zero code tokens emitted"

    if mode == OrchestrationMode.RAW:
        assert len(think_tokens) == 0, \
            f"[{mode.value}] expected zero think tokens, got {len(think_tokens)}"
    else:
        # R1/T2/C3 should produce at least SOME thinking content on qwen3.5:9b.
        assert len(think_tokens) > 0, \
            f"[{mode.value}] expected think tokens, got zero"


def _make_two_phase_llm(token_attempts, *, phase1_call_counter):
    """Build a FakeLlm wired for TWO_PHASE mode. Pre-pads with an empty
    attempt because `DemoClient.generate` calls `set_prompt` twice before
    the main loop in TWO_PHASE (initial + post-phase-1), each call advancing
    the FakeLlm's attempt index. The empty pad absorbs that initial flip
    so the test's `token_attempts[0]` runs first."""
    class _Fake(FakeLlm):
        async def stream_thinking_phase(self, prompt):
            phase1_call_counter[0] += 1
            yield Token("consider", "think")
            yield Token(" the error", "think")
    llm = _Fake([])           # empty slot absorbs the initial set_prompt
    for atts in token_attempts:
        llm.add_attempt(atts)
    llm.config.mode = OrchestrationMode.TWO_PHASE
    return llm


@pytest.mark.asyncio
async def test_two_phase_reenter_thinking_on_rollback():
    """When `reenter_thinking_on_rollback` is on AND mode is TWO_PHASE,
    every cargo-error rollback must invoke `stream_thinking_phase` a SECOND
    (and third, …) time on the rolled-back prompt. This is the path where
    "instruct on rollback + a thinking mode" composes into "re-reason about
    the cargo error before re-attempting"."""
    from soundcode.code import Category, Diagnostic
    phase1 = [0]
    err = Diagnostic(category=Category.BLOCKING, code="E0425",
                     message="cannot find function `bad_call`")
    # Two real attempts: first hits cargo error, second exits via body_closed.
    llm = _make_two_phase_llm(
        # First attempt: no closing brace → boundary at `;` triggers a check;
        # the stream exhausts cleanly so the producer loop awaits the verdict
        # instead of body_closed-short-circuiting. Second attempt: closes the
        # body so the post-rollback run terminates without piling on rollbacks.
        [["let x = 1;"], ["    let y = 2;", "\n}"]],
        phase1_call_counter=phase1,
    )
    llm.config.reenter_thinking_on_rollback = True
    checker = ScriptedChecker([[err], []])
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=checker,
        config=DemoConfig(token_delay_s=0, repetition_window=0,
                          wall_budget_s=5, max_continuations=0,
                          instruct_on_rollback=True),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() {")

    rbs = [e for e in events if e["type"] == "rollback"]
    assert len(rbs) >= 1, [e["type"] for e in events]
    # Phase 1 ran at least twice: once at the start, once on the rollback.
    assert phase1[0] >= 2, (
        f"expected ≥2 phase-1 invocations (initial + 1 per rollback), got {phase1[0]}"
    )
    # The "re-thinking" status surfaced in the event stream.
    statuses = [e.get("message","") for e in events if e["type"] == "status"]
    assert any("re-thinking" in m for m in statuses), statuses


@pytest.mark.asyncio
async def test_two_phase_no_reenter_when_flag_off():
    """Reenter is OPT-IN: with the flag off, phase 1 must run exactly once
    even when a rollback fires."""
    from soundcode.code import Category, Diagnostic
    phase1 = [0]
    err = Diagnostic(category=Category.BLOCKING, code="E0425",
                     message="cannot find function `bad_call`")
    llm = _make_two_phase_llm(
        # First attempt: no closing brace → boundary at `;` triggers a check;
        # the stream exhausts cleanly so the producer loop awaits the verdict
        # instead of body_closed-short-circuiting. Second attempt: closes the
        # body so the post-rollback run terminates without piling on rollbacks.
        [["let x = 1;"], ["    let y = 2;", "\n}"]],
        phase1_call_counter=phase1,
    )
    llm.config.reenter_thinking_on_rollback = False
    checker = ScriptedChecker([[err], []])
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=checker,
        config=DemoConfig(token_delay_s=0, repetition_window=0,
                          wall_budget_s=5, max_continuations=0,
                          instruct_on_rollback=False),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() {")

    rbs = [e for e in events if e["type"] == "rollback"]
    assert len(rbs) >= 1, (
        f"expected a rollback to actually fire so this isn't a vacuous "
        f"test, got events {[e['type'] for e in events]}"
    )
    assert phase1[0] == 1, (
        f"expected exactly 1 phase-1 call when reenter is off, got {phase1[0]}"
    )


@pytest.mark.asyncio
async def test_two_phase_dispatch_runs_phase1_then_phase2():
    """TWO_PHASE: DemoClient must call `llm.stream_thinking_phase(prompt)`
    before the main producer loop, emit each yielded token as kind="think",
    inject the trace as a comment block, and only then start the code phase."""
    thinking_chunks = ["the model ", "thinks ", "for a bit"]

    class FakeLlmWithThinkingPhase(FakeLlm):
        async def stream_thinking_phase(self, prompt):
            for ch in thinking_chunks:
                yield Token(ch, "think")

    llm = FakeLlmWithThinkingPhase(["    let x = 1;\n", "\n}"])
    llm.config.mode = OrchestrationMode.TWO_PHASE
    captured_prompts: list[str] = []
    orig_set_prompt = llm.set_prompt

    def capture_set_prompt(p):
        captured_prompts.append(p)
        orig_set_prompt(p)
    llm.set_prompt = capture_set_prompt

    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=FakeChecker(),
        config=DemoConfig(token_delay_s=0, repetition_window=0,
                          wall_budget_s=5, max_continuations=0),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() {")

    # Phase 1 status appears once.
    statuses = [e.get("message","") for e in events if e["type"] == "status"]
    assert any("phase 1" in m for m in statuses), statuses
    assert any("phase 2" in m for m in statuses), statuses

    # Synthetic <think> / </think> markers wrap the thinking trace.
    think_texts = [e["text"] for e in events
                   if e["type"] == "token_emitted" and e.get("kind") == "think"]
    assert "<think>" in think_texts
    assert "</think>" in think_texts
    for ch in thinking_chunks:
        assert ch in think_texts

    # After phase 1, the prompt was rewritten to include the comment block.
    rewritten = captured_prompts[-1]
    assert "// ───── reasoning ─────" in rewritten
    assert "// the model" in rewritten


@pytest.mark.asyncio
async def test_raw_mode_skips_phase1():
    """RAW: DemoClient must NOT call stream_thinking_phase and must NOT emit
    any think tokens — only code tokens flow."""
    class WatchfulFakeLlm(FakeLlm):
        def __init__(self, tokens):
            super().__init__(tokens)
            self.phase1_calls = 0
        async def stream_thinking_phase(self, prompt):
            self.phase1_calls += 1
            if False:
                yield   # make this a generator without yielding anything

    llm = WatchfulFakeLlm(["    let x = 1;\n", "\n}"])
    llm.config.mode = OrchestrationMode.RAW
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=FakeChecker(),
        config=DemoConfig(token_delay_s=0, repetition_window=0,
                          wall_budget_s=5, max_continuations=0),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() {")
    assert llm.phase1_calls == 0
    think_count = sum(1 for e in events
                      if e["type"] == "token_emitted" and e.get("kind") == "think")
    assert think_count == 0


@pytest.mark.asyncio
async def test_continuation_after_empty_content_skips_misleading_hint():
    """When the first stream produces no code (e.g., thinking-mode dumped
    everything inside `<think>` and the splitter never saw a boundary, so
    `code.content` is empty by the time final_check runs), the continuation
    must NOT append the "the function body above is incomplete; continue
    from here" comment. With empty content the comment lands right after
    the function's opening `{` and the LM frequently interprets it as the
    function's body, then emits just `}` to close.
    """
    # Sequence:
    #   1) First stream emits nothing (LM EOS immediately) → empty content.
    #   2) Final check fires, body is empty → is_complete=False, continuation
    #      attempt 1 begins. The fix routes the empty case through the
    #      original prompt without the misleading hint.
    #   3) Continuation stream emits one body token + `\n}` → body closes.
    llm = FakeLlm([])
    llm.add_attempt(["    let x = 1;\n", "}"])
    captured_prompts: list[str] = []

    class CapturingLlm(FakeLlm):
        def set_prompt(self, prompt: str) -> None:
            captured_prompts.append(prompt)
            super().set_prompt(prompt)

    llm2 = CapturingLlm([])
    llm2.add_attempt(["    let x = 1;\n", "}"])
    events: list[dict] = []
    demo = DemoClient(
        llm=llm2, checker=FakeChecker(),
        config=DemoConfig(token_delay_s=0, repetition_window=0,
                          wall_budget_s=5, max_continuations=2),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() {")
    # Final check must have fired at least once + a continuation must have run.
    assert any(e["type"] == "continuation" for e in events), \
        [e["type"] for e in events]
    # The prompt passed to the continuation stream must NOT contain the
    # misleading "above is incomplete" comment when content is empty.
    cont_prompt = captured_prompts[-1]
    assert "above is incomplete" not in cont_prompt, cont_prompt
    # The continuation stream did produce actual code (`let x = 1;`).
    emitted = [e["text"] for e in events
               if e["type"] == "token_emitted" and e.get("kind", "code") == "code"]
    assert "    let x = 1;\n" in emitted, emitted


@pytest.mark.asyncio
async def test_final_check_flags_empty_body_as_incomplete():
    """A run where every rollback discarded everything ends with empty
    content; `_emit_final_check` must report incomplete (NOT 'ok') so the
    continuation loop re-prompts the LM instead of returning an empty
    function body as the final answer (observed in run_20260515_221852)."""
    llm = FakeLlm(["// trying to think\n"])  # only a comment, then EOS
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=FakeChecker(),
        config=DemoConfig(token_delay_s=0, repetition_window=0,
                          wall_budget_s=5, max_continuations=0),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn solve() {")
    fc = [e for e in events if e["type"] == "final_check"]
    assert fc, "no final_check event emitted"
    assert fc[-1]["is_complete"] is False, fc[-1]
    assert fc[-1]["verdict"] == "error"


@pytest.mark.asyncio
async def test_repetition_watchdog_does_not_trigger_on_normal_code():
    """Normal non-repetitive code (with `body_closed` happening at the end)
    should terminate via body_closed, NOT via repetition."""
    tokens = [
        "    let x = 1;",
        "\n    let y = 2;",
        "\n    let z = x + y;",
        "\n    return z;",
        "\n}",
    ]
    llm = FakeLlm(tokens)
    events: list[dict] = []
    demo = DemoClient(
        llm=llm, checker=FakeChecker(),
        config=DemoConfig(token_delay_s=0,
                          repetition_window=60, repetition_threshold=3),
        emit=lambda ev: events.append(ev) or asyncio.sleep(0),
    )
    await demo.generate("fn f() -> i32 {")
    statuses = [e for e in events if e["type"] == "status"]
    msgs = [s.get("message", "") for s in statuses]
    assert any("function body closed" in m for m in msgs), msgs
    assert not any("repetition detected" in m for m in msgs), msgs
