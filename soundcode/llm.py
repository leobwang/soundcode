"""LlmServer — thin async wrapper over Ollama's streaming completion API.

API matches the producer-consumer loop from draft-plan-0.md §3.2:

  llm = LlmServer(model="...")
  llm.set_prompt(prompt)
  while llm.has_next():
      token = await llm.next()
  llm.abort_current_stream()
"""

from __future__ import annotations

import asyncio
import enum
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx


class OrchestrationMode(str, enum.Enum):
    """Strategy for getting (optional) thinking + code out of a reasoning model.

    - `RAW`: legacy path. `<think>` is never injected; Ollama receives the bare
      prompt with `raw: true`. No thinking trace; clean literal continuation.
    - `RAW_THINK_INJECT`: append `<think>\\n` to the raw prompt and use
      `ThinkSplitter` to parse a `</think>` / markdown-fence boundary. Fragile
      across models (qwen3.5:9b clean, qwen3.5:122b never closes the tag).
    - `TWO_PHASE`: phase 1 = chat-template-wrapped raw call with `stop=["</think>"]`
      to harvest a reliable thinking trace; phase 2 = bake the trace into the
      original prompt as a `// reasoning` comment block and run the existing
      raw producer-consumer loop on the augmented prompt. Two round trips;
      reliable thinking + clean code phase; rollback works unchanged.
    - `CHAT_INSTRUCTED`: `/api/chat` with `think: true` plus a system prompt
      forbidding markdown fences / preamble / closing brace. One call;
      reliable thinking; model still often re-declares the function signature,
      so the response is passed through `ChatBodyStripper` to extract only
      the body. Rollback retries fall back to RAW (one-shot).
    """
    RAW = "raw"
    RAW_THINK_INJECT = "raw_think_inject"
    TWO_PHASE = "two_phase"
    CHAT_INSTRUCTED = "chat_instructed"


@dataclass(frozen=True)
class ChatTemplate:
    """Per-family chat-template literals used by TWO_PHASE phase 1.

    Phase 1's job is to get a reliable thinking trace out of a reasoning model
    in raw mode. We do that by manually wrapping the user task in the model's
    own assistant-turn prefix and adding `</think>` as a stop sequence —
    Ollama returns the bytes between `<think>` and `</think>`, which is the
    thinking trace, and stops there.
    """
    user_open: str            # e.g. "<|im_start|>user\n"
    user_close: str           # e.g. "\n<|im_end|>\n"
    assistant_think_open: str # e.g. "<|im_start|>assistant\n<think>\n"
    think_close: str          # e.g. "</think>" — the stop sequence for phase 1


# Only families empirically verified to engage thinking via manual chat-template
# wrap in raw mode. Add new entries here when verified per-family.
CHAT_TEMPLATES: dict[str, ChatTemplate] = {
    "qwen3": ChatTemplate(
        user_open="<|im_start|>user\n",
        user_close="\n<|im_end|>\n",
        assistant_think_open="<|im_start|>assistant\n<think>\n",
        think_close="</think>",
    ),
}


def template_for_model(model: str) -> ChatTemplate | None:
    """Look up the chat template for a model by family prefix.

    `qwen3.5:122b` → `qwen3` template, `qwen2.5-coder:32b` → None (the qwen2.5
    family doesn't support thinking; treat as no-template).
    """
    name = model.lower()
    # Most specific match first: qwen3.6 maps to "qwen3" template iff we have
    # one for the family. Keep this list short and explicit.
    if name.startswith("qwen3.") or name.startswith("qwen3:"):
        return CHAT_TEMPLATES.get("qwen3")
    return None


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = -1   # -1 = predict until EOS / context limit (Ollama default)
    top_p: float = 0.95
    stop: list[str] = field(default_factory=lambda: ["\n}"])
    # `enable_thinking` toggles Ollama's `think` top-level option. Under
    # `raw: true` it's a no-op (Ollama bypasses the chat template that would
    # inject `<think>`). Used by `CHAT_INSTRUCTED` mode where raw=false.
    enable_thinking: bool = False
    # Orchestration strategy — see `OrchestrationMode` for descriptions.
    # Off-the-cliff one-shot semantics: after the first stream open for any
    # mode != RAW, `_current_mode` (a runtime field on LlmServer) flips back
    # to RAW so rollback / continuation retries get plain literal completion.
    mode: OrchestrationMode = OrchestrationMode.RAW
    # Opt-in nudge: prepend a Rust-comment directive immediately before the
    # injected `<think>` in RAW_THINK_INJECT mode. Marginally helps some
    # models; empirically a no-op on others.
    r1_directive_comment: bool = False
    # CHAT_INSTRUCTED-only safety cap. Larger reasoning models (qwen3.5:122b
    # observed) can emit unbounded thinking traces before producing any
    # `content` field — the chat call effectively never completes. When the
    # cumulative `thinking`-field character count exceeds this budget, the
    # chat stream is aborted and the run falls through to RAW continuation,
    # which produces actual code without thinking. Default 8000 chars (~2500
    # tokens) covers normal reasoning runs on small models and bounds the
    # large ones to a hard wall-clock ceiling.
    think_char_budget: int = 8000
    # When True, R1 / TWO_PHASE / CHAT_INSTRUCTED do NOT one-shot-flip to
    # RAW after the first stream open. Every rollback re-runs the thinking-
    # mode protocol (re-inject `<think>`, re-run phase 1, or re-open chat).
    # Wired up by the server when `instruct_on_rollback` is on AND mode is a
    # thinking mode — the idea is that if we're already paying for an
    # instructive comment in the rollback prompt, we should also give the
    # model another chance to reason about it before continuing.
    reenter_thinking_on_rollback: bool = False


@dataclass
class Token:
    """One streamed unit. `kind` distinguishes thinking tokens (which must NOT
    be appended to the code buffer) from code tokens (the real continuation).

    Optional fields `token_id` and `entropy` carry sampler-level information
    for processors that want it (notably ROCODE's trie, which records token
    + entropy per node). The Ollama path doesn't expose these (HTTP API hides
    them), so both default to None — consumers must treat them as optional.
    The vLLM path may populate them in a future patch; for now the trie
    bookkeeping uses a text-hash fallback when `token_id` is None.
    """
    text: str
    kind: str = "code"           # "code" | "think"
    token_id: int | None = None  # vocab id when known; None for Ollama
    entropy: float | None = None # H_t when computed; None for Ollama

    def __bool__(self) -> bool:
        return bool(self.text)


class PostBoundaryStripper:
    """Sub-stripper used by `ThinkSplitter` after the think→code boundary
    fires. It looks at the first ~40 non-whitespace bytes the model emits in
    its "code phase" and decides whether the model entered chat-style mode
    (markdown fence and/or function redeclaration — qwen3.5:122b's pattern)
    or just continued the prompt literally (qwen3.5:9b's pattern):

    - **passthrough**: no fence, no `fn ` start → emit everything as-is.
    - **extract_body**: fence and/or `fn <ident>(…) {` redeclaration detected
      → delegate to `ChatBodyStripper` to extract only the bytes inside the
      outer `{ … }`. Drops the fence open, the redeclared signature, the
      matching outer `}`, the fence close, and any postamble.

    Without this, a R1 run on qwen3.5:122b puts ~10K chars of fence-wrapped
    re-declared function code into the code buffer; with it, only the actual
    function body reaches `Code.content`.
    """

    DETECT_THRESHOLD = 40   # non-whitespace chars to see before committing to passthrough

    def __init__(self) -> None:
        self.mode = "detect"   # "detect" | "passthrough" | "extract_body"
        self.buf = ""
        self._body_stripper: ChatBodyStripper | None = None

    def feed(self, chunk: str) -> list[Token]:
        if self.mode == "passthrough":
            return [Token(chunk, "code")] if chunk else []
        if self.mode == "extract_body":
            assert self._body_stripper is not None
            return self._body_stripper.feed(chunk)
        # mode == "detect"
        self.buf += chunk
        stripped = self.buf.lstrip()
        # The model often emits a partial fence prefix that needs the next
        # chunk to confirm (e.g., `\`` or `\`\``). Hold off on commit if the
        # non-whitespace prefix could still grow into a fence.
        if stripped.startswith("```"):
            return self._switch_to_extract_body()
        if any(stripped == p[:len(stripped)] for p in ("```rust\n", "```rs\n", "```\n")) \
                and len(stripped) < 3:
            # partial fence prefix — hold
            return []
        if stripped.startswith("fn "):
            return self._switch_to_extract_body()
        if stripped.startswith("fn") and len(stripped) < 3:
            return []  # could grow into `fn ` — hold
        # Not chat-style. Commit to passthrough once we have enough
        # non-whitespace content to be confident — or we'll never detect
        # late-arriving boilerplate.
        if len(stripped) >= self.DETECT_THRESHOLD or "\n" in stripped:
            self.mode = "passthrough"
            out = self.buf
            self.buf = ""
            return [Token(out, "code")] if out else []
        return []

    def _switch_to_extract_body(self) -> list[Token]:
        self._body_stripper = ChatBodyStripper()
        buf = self.buf
        self.buf = ""
        self.mode = "extract_body"
        return self._body_stripper.feed(buf)

    def flush(self) -> list[Token]:
        if self.mode == "extract_body":
            assert self._body_stripper is not None
            return self._body_stripper.flush()
        if self.mode == "passthrough":
            return []
        # Still in detect at EOS — emit whatever we have.
        out = []
        if self.buf:
            out.append(Token(self.buf, "code"))
        self.buf = ""
        return out


class ThinkSplitter:
    """State machine that splits Ollama's single `response` field into think
    vs. code tokens when we injected `<think>\\n` into the raw prompt.

    Reasoning models (Qwen 3.x, gpt-oss, deepseek-r1, nemotron-cascade-2,
    nemotron-3-super, nemotron-3-nano) emit chain-of-thought directly after a
    `<think>` token, then either close with the literal `</think>` tag or
    transition into chat-style output (often a ```rust fence). Both signals
    are treated as the think→code boundary; the rest of the stream is code.

    All boundaries are searched on the accumulated buffer (`pending` + chunk)
    so a tag straddling two HTTP chunks (e.g., `</thi` arrives in chunk N and
    `nk>` in chunk N+1) is still detected.
    """

    END_TAG = "</think>"
    # Longest first so `\`\`\`rust\\n` matches before `\`\`\`\\n`.
    FENCE_PATTERNS = ("```rust\n", "```rs\n", "```\n")
    _ALL_PATTERNS = (END_TAG,) + FENCE_PATTERNS
    MAX_HOLD = max(len(p) for p in _ALL_PATTERNS)

    def __init__(self, start_state: str = "think") -> None:
        self.state = start_state   # "think" or "code"
        self.pending = ""          # held-back tail awaiting boundary decision
        # After the think→code boundary fires, post-boundary content flows
        # through `PostBoundaryStripper`, which detects chat-style output
        # (fence / fn-redeclaration) and strips the boilerplate.
        self._post: PostBoundaryStripper | None = None

    def feed(self, chunk: str) -> list[Token]:
        # In CODE state, EVERYTHING goes through the post-boundary stripper
        # (which may be in passthrough, detect, or extract-body mode).
        if self.state == "code":
            assert self._post is not None
            return self._post.feed(chunk)

        text = self.pending + chunk
        self.pending = ""
        out: list[Token] = []

        # 1) Literal closing tag — the cleanest boundary. NOTE: we do NOT
        #    emit `</think>` as a token here. `DemoClient._emit_token` will
        #    synthesise one when it sees the think→code transition. Emitting
        #    it from the splitter too would produce duplicate markers.
        idx = text.find(self.END_TAG)
        if idx >= 0:
            if idx > 0:
                out.append(Token(text[:idx], "think"))
            rest = text[idx + len(self.END_TAG):]
            self.state = "code"
            self._post = PostBoundaryStripper()
            if rest:
                out.extend(self._post.feed(rest))
            return out

        # 2) Fallback: markdown fence. Drop the fence itself; let
        #    `_emit_token` synthesise the `</think>` marker on the first
        #    code-kind transition.
        for pat in self.FENCE_PATTERNS:
            idx = text.find(pat)
            if idx >= 0:
                if idx > 0:
                    out.append(Token(text[:idx], "think"))
                rest = text[idx + len(pat):]
                self.state = "code"
                self._post = PostBoundaryStripper()
                # The fence is already gone; `PostBoundaryStripper` may
                # detect a follow-up `fn <ident>` redeclaration.
                if rest:
                    out.extend(self._post.feed(rest))
                return out

        # 3) No boundary yet. Hold back the longest tail that could be the
        # PREFIX of any boundary pattern, so a tag straddling chunks isn't
        # mis-emitted as plain thinking text.
        hold = 0
        for n in range(min(len(text), self.MAX_HOLD), 0, -1):
            tail = text[-n:]
            if any(p.startswith(tail) for p in self._ALL_PATTERNS):
                hold = n
                break
        if hold > 0:
            self.pending = text[-hold:]
            text = text[:-hold]
        if text:
            out.append(Token(text, "think"))
        return out

    def flush(self) -> list[Token]:
        """Called at EOS. Emit any held-back tail as a final token in the
        current state — without this, a stream that ends mid-buffer would
        silently drop those characters."""
        out: list[Token] = []
        if self.pending:
            out.append(Token(self.pending, self.state))
            self.pending = ""
        if self._post is not None:
            out.extend(self._post.flush())
        return out


class ChatBodyStripper:
    """Mode CHAT_INSTRUCTED only. Consumes the `content` field of a streaming
    chat response and emits only the bytes INSIDE the outermost `{ … }` of the
    model's response. The model usually re-declares the outer function
    signature (sometimes with a different return type) even when the system
    prompt forbids it — we drop that signature and the matching outer braces,
    yielding just the function body.

    Brace depth ignores braces inside string literals, char literals, and
    Rust comments (line + block). Nested fns inside the body keep their
    braces and pass through as code.

    States:
      BEFORE_OPEN: looking for the first depth-0 `{`. All bytes dropped.
      INSIDE     : inside the outer body. Emit bytes as kind="code", track
                   brace depth. When depth returns to 0 (i.e. we saw the
                   matching `}`), strip the `}` itself and switch to AFTER_CLOSE.
      AFTER_CLOSE: drop everything (closing markdown fence, postamble).
    """

    def __init__(self) -> None:
        self.state = "before_open"
        self.depth = 0           # brace depth within INSIDE
        # Lexer state for the byte we're currently looking at — needed because
        # chunks can split mid-string / mid-comment.
        self._in_str = False     # inside "…"
        self._in_char = False    # inside '…'
        self._in_line_cmt = False
        self._in_block_cmt = False
        self._str_escape = False
        # Buffer for `content` bytes that haven't been emitted yet — used to
        # hold tail characters whose interpretation depends on the next byte
        # (e.g., `\\` inside a string, `/` that might start `//` or `/*`).
        self._pending = ""

    def feed(self, chunk: str) -> list[Token]:
        if self.state == "after_close":
            return []
        out_chars: list[str] = []
        # Process character by character. Cheap — chunks are small.
        text = self._pending + chunk
        self._pending = ""
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if self._in_line_cmt:
                if self.state == "inside":
                    out_chars.append(c)
                if c == "\n":
                    self._in_line_cmt = False
                i += 1
                continue
            if self._in_block_cmt:
                # need 2-char lookahead for `*/`
                if c == "*":
                    if i + 1 >= n:
                        # hold for next chunk
                        self._pending = c
                        break
                    if text[i + 1] == "/":
                        if self.state == "inside":
                            out_chars.append("*/")
                        self._in_block_cmt = False
                        i += 2
                        continue
                if self.state == "inside":
                    out_chars.append(c)
                i += 1
                continue
            if self._in_str:
                if self.state == "inside":
                    out_chars.append(c)
                if self._str_escape:
                    self._str_escape = False
                elif c == "\\":
                    self._str_escape = True
                elif c == '"':
                    self._in_str = False
                i += 1
                continue
            if self._in_char:
                if self.state == "inside":
                    out_chars.append(c)
                if self._str_escape:
                    self._str_escape = False
                elif c == "\\":
                    self._str_escape = True
                elif c == "'":
                    self._in_char = False
                i += 1
                continue
            # Not currently in any lexical context.
            if c == "/":
                # Lookahead 1 char for `//` or `/*`. Hold if at end of buffer.
                if i + 1 >= n:
                    self._pending = c
                    break
                nxt = text[i + 1]
                if nxt == "/":
                    self._in_line_cmt = True
                    if self.state == "inside":
                        out_chars.append("//")
                    i += 2
                    continue
                if nxt == "*":
                    self._in_block_cmt = True
                    if self.state == "inside":
                        out_chars.append("/*")
                    i += 2
                    continue
                # plain slash
                if self.state == "inside":
                    out_chars.append(c)
                i += 1
                continue
            if c == '"':
                self._in_str = True
                if self.state == "inside":
                    out_chars.append(c)
                i += 1
                continue
            if c == "'":
                # Could be lifetime or char literal. Treat conservatively:
                # if next char is alphanumeric and NOT followed by `'` within
                # a few chars, it's a lifetime, not a char literal. For
                # safety on a streamed buffer we treat `'` as char-literal
                # start only when we can confirm a closing `'` is nearby;
                # otherwise just pass through as plain text. The brace counter
                # never inspects `'` so this is harmless either way.
                if self.state == "inside":
                    out_chars.append(c)
                i += 1
                continue
            if c == "{":
                if self.state == "before_open":
                    # Start of the outer body — switch to INSIDE. Don't emit
                    # the `{` itself; the caller's prompt ended with `{`.
                    self.state = "inside"
                    self.depth = 1
                else:  # state == "inside"
                    self.depth += 1
                    out_chars.append(c)
                i += 1
                continue
            if c == "}":
                if self.state == "inside":
                    self.depth -= 1
                    if self.depth == 0:
                        # Matching close — drop it, switch to AFTER_CLOSE.
                        self.state = "after_close"
                        i += 1
                        # Anything else in this chunk is postamble — discard.
                        break
                    out_chars.append(c)
                # else: stray `}` in BEFORE_OPEN — drop (it's part of
                # whatever preamble the model emitted).
                i += 1
                continue
            # Everything else.
            if self.state == "inside":
                out_chars.append(c)
            i += 1
        emitted = "".join(out_chars)
        return [Token(emitted, "code")] if emitted else []

    def flush(self) -> list[Token]:
        """Stream ended (or the chat call returned `done`). If we're still
        INSIDE — model EOS'd before emitting the matching `}` — emit whatever
        we have so far as code; the function-body completion logic on the
        caller side (cargo, body_closed) will handle the unterminated body.
        Any leftover lookahead-pending byte is treated similarly."""
        out: list[Token] = []
        if self._pending and self.state == "inside":
            out.append(Token(self._pending, "code"))
        self._pending = ""
        return out


CHAT_INSTRUCTED_SYSTEM_PROMPT = (
    "You are a Rust code-completion engine. The user gives you a function "
    "signature with the opening brace. You MUST output ONLY the function body — "
    "the raw Rust code that goes between the opening `{` and the closing `}`. "
    "NO markdown fences (no ```rust or ```). "
    "NO preamble (no \"Here's the function:\", no explanatory English). "
    "NO closing `}` for the outer function — that will be appended by the caller. "
    "Your first character must be the first character of the function body."
)


class LlmServer:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        config: GenerationConfig | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.config = config or GenerationConfig()
        self.timeout = timeout
        self._prompt: str = ""
        self._client: httpx.AsyncClient | None = None
        self._stream_ctx = None
        self._stream_iter: AsyncIterator[Token] | None = None
        self._exhausted = True  # nothing to read until set_prompt() + first next()
        self._tokens_in_current_stream = 0
        # Set to True when a CHAT_INSTRUCTED stream is aborted because the
        # cumulative `thinking` chars exceeded `config.think_char_budget`.
        # DemoClient reads this after the chat stream finishes to emit a
        # status event explaining why the run produced no code on the first
        # attempt and the continuation path is now active.
        self.think_budget_exceeded = False

    # ─── prompt control ────────────────────────────────────────────────

    def set_prompt(self, prompt: str) -> None:
        """Set the prompt to use on the *next* stream open."""
        self._prompt = prompt
        self._exhausted = False  # there's a fresh stream waiting to be opened

    def has_next(self) -> bool:
        """True if the LM might emit another token."""
        return not self._exhausted

    async def next(self) -> Token:
        """Get the next token from the LM.

        Returns a `Token(text, kind)` where `kind` is "think" or "code". At
        EOS returns `Token("", "code")`. Opens the stream lazily on first call.
        """
        if self._stream_iter is None:
            await self._open_stream()
        assert self._stream_iter is not None
        try:
            token = await anext(self._stream_iter)
        except StopAsyncIteration:
            self._exhausted = True
            await self._close_stream()
            return Token("", "code")
        self._tokens_in_current_stream += 1
        return token

    async def abort_current_stream(self) -> None:
        """Close the in-flight HTTP stream. Next has_next() returns False until
        set_prompt() is called again."""
        self._exhausted = True
        await self._close_stream()

    async def warmup(self) -> None:
        """Force Ollama to load this model into VRAM before generation starts.

        Sends `POST /api/generate` with an empty prompt; Ollama returns once
        the weights are resident. Without this, the load delay (seconds for
        small models, tens of seconds to minutes for 120B+) happens lazily on
        the first token of the real stream, and the UI sees `generating` with
        nothing flowing yet.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        resp = await self._client.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": "", "stream": False},
        )
        resp.raise_for_status()

    # ─── internals ─────────────────────────────────────────────────────

    async def _open_stream(self) -> None:
        """Mode-dispatched stream opener.

        - `RAW`, `TWO_PHASE`: open a raw `/api/generate` stream with no
          `<think>` injection. `TWO_PHASE` looks the same here because its
          phase-1 thinking call is a separate method (`stream_thinking_phase`)
          invoked by `DemoClient` BEFORE this producer-consumer loop starts —
          by the time we hit `_open_stream`, the prompt has already been
          augmented with the thinking-as-comment block.
        - `RAW_THINK_INJECT`: append `<think>\\n` to the outgoing prompt and
          use `ThinkSplitter` to parse a `</think>` / markdown-fence boundary.
          One-shot: mode flips to `RAW` after this stream open.
        - `CHAT_INSTRUCTED`: open `/api/chat` with `think: true` plus a system
          prompt forbidding markdown fences / preamble. Use `ChatBodyStripper`
          to extract just the body from the response field. One-shot.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        mode = self.config.mode
        if mode == OrchestrationMode.CHAT_INSTRUCTED:
            await self._open_chat_instructed_stream()
            # One-shot flip unless reenter-thinking is on, in which case the
            # next rollback gets another chat call with the updated prompt.
            if not self.config.reenter_thinking_on_rollback:
                self.config.mode = OrchestrationMode.RAW
            return
        # All non-chat modes share the raw `/api/generate` path. Inject
        # `<think>` only for `RAW_THINK_INJECT`.
        inject_now = (mode == OrchestrationMode.RAW_THINK_INJECT)
        outgoing_prompt = self._prompt
        if inject_now:
            if self.config.r1_directive_comment:
                # Lightweight nudge: place a Rust comment immediately before
                # the `<think>` trigger. Some models read this as a directive
                # to keep their post-`</think>` output as a literal body
                # continuation; many ignore it. Off by default.
                outgoing_prompt = (
                    outgoing_prompt
                    + "    // Reasoning will follow. After </think>, continue"
                    + " the function body literally — no markdown, no preamble.\n"
                )
            outgoing_prompt = outgoing_prompt + "<think>\n"
            # One-shot: after `</think>` the model often emits a fresh
            # response (re-declaring the function, opening a markdown fence,
            # etc.) rather than continuing the prompt literally. Subsequent
            # rollback/continuation attempts get plain raw completion.
            # When reenter-thinking is on, skip the flip — each rollback
            # gets another `<think>` injection (the model re-reasons about
            # the error comment that the instruct path just added).
            if not self.config.reenter_thinking_on_rollback:
                self.config.mode = OrchestrationMode.RAW
        payload = {
            "model": self.model,
            "prompt": outgoing_prompt,
            "raw": True,
            "stream": True,
            "think": self.config.enable_thinking,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
                "stop": self.config.stop,
            },
        }
        self._stream_ctx = self._client.stream(
            "POST", f"{self.base_url}/api/generate", json=payload,
        )
        resp = await self._stream_ctx.__aenter__()
        resp.raise_for_status()
        # Branch on the captured `inject_now` (not on `self.config.mode`,
        # which has already been flipped to RAW by the one-shot above).
        splitter = ThinkSplitter("think") if inject_now else None
        self._stream_iter = _iter_ollama_tokens(resp, splitter=splitter)
        self._tokens_in_current_stream = 0

    async def _open_chat_instructed_stream(self) -> None:
        """Open `/api/chat` for `CHAT_INSTRUCTED` mode. Streams chunks where
        each has `message.thinking` (yielded as kind='think') and/or
        `message.content` (routed through `ChatBodyStripper`).

        Aborts mid-stream if cumulative thinking chars exceed
        `config.think_char_budget` — sets `self.think_budget_exceeded` so
        `DemoClient` can surface a status event."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": CHAT_INSTRUCTED_SYSTEM_PROMPT},
                {"role": "user", "content": self._prompt},
            ],
            "stream": True,
            "think": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }
        self._stream_ctx = self._client.stream(
            "POST", f"{self.base_url}/api/chat", json=payload,
        )
        resp = await self._stream_ctx.__aenter__()
        resp.raise_for_status()
        # Reset the budget flag for this run; the iter will set it on abort.
        self.think_budget_exceeded = False
        def _on_budget():
            self.think_budget_exceeded = True
        self._stream_iter = _iter_ollama_chat_tokens(
            resp, ChatBodyStripper(),
            think_char_budget=self.config.think_char_budget,
            on_budget_exceeded=_on_budget,
        )
        self._tokens_in_current_stream = 0

    async def stream_thinking_phase(self, prompt: str) -> AsyncIterator[Token]:
        """`TWO_PHASE` phase 1: chat-template-wrapped raw call with `stop=
        [<think_close>]`. Yields `kind="think"` Tokens as the model emits
        chain-of-thought. Stops cleanly at `</think>` (the bytes containing
        the stop sequence are NOT yielded — Ollama returns content up to but
        not including the stop pattern).

        Does NOT touch `self._stream_iter` / `self._prompt` — those belong to
        the main producer-consumer loop, which `DemoClient` will drive after
        this method completes. Returns silently with no yields when the
        model has no known chat template (caller treats that as "no
        thinking available").
        """
        template = template_for_model(self.model)
        if template is None:
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        wrapped = (
            template.user_open + prompt + template.user_close
            + template.assistant_think_open
        )
        payload = {
            "model": self.model,
            "prompt": wrapped,
            "raw": True,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": -1,
                "top_p": self.config.top_p,
                "stop": [template.think_close],
            },
        }
        async with self._client.stream(
            "POST", f"{self.base_url}/api/generate", json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response = evt.get("response")
                if response:
                    yield Token(response, "think")
                if evt.get("done"):
                    break

    async def _close_stream(self) -> None:
        if self._stream_iter is not None:
            self._stream_iter = None
        if self._stream_ctx is not None:
            try:
                await self._stream_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._stream_ctx = None

    async def close(self) -> None:
        await self._close_stream()
        if self._client is not None:
            await self._client.aclose()
            self._client = None


async def _iter_ollama_tokens(
    resp: httpx.Response,
    splitter: ThinkSplitter | None = None,
) -> AsyncIterator[Token]:
    """Stream Ollama chunks as `Token` instances.

    Two ways thinking can arrive:
    - `thinking` field (chat-mode path): Ollama already parsed `<think>…
      </think>` server-side and split the channel. Yielded directly as
      `kind="think"`.
    - `response` field with raw-mode injection (`splitter is not None`):
      everything goes through the splitter, which finds the `</think>` /
      markdown-fence boundary and emits `kind="think"` then `kind="code"`.
      Without a splitter, `response` is always `kind="code"`.
    """
    async for line in resp.aiter_lines():
        if not line.strip():
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        thinking = evt.get("thinking")
        if thinking:
            yield Token(thinking, "think")
        response = evt.get("response")
        if response:
            if splitter is not None:
                for tok in splitter.feed(response):
                    yield tok
            else:
                yield Token(response, "code")
        if evt.get("done"):
            if splitter is not None:
                for tok in splitter.flush():
                    yield tok
            break


async def _iter_ollama_chat_tokens(
    resp: httpx.Response,
    stripper: ChatBodyStripper,
    *,
    think_char_budget: int | None = None,
    on_budget_exceeded=None,
) -> AsyncIterator[Token]:
    """Stream `/api/chat` chunks as `Token` instances. Chat-mode chunks wrap
    their per-token fields inside a `message` object (rather than at the top
    level like `/api/generate`):

        {"message": {"role": "assistant", "content": "...", "thinking": "..."},
         "done": false}

    `thinking` → kind="think" Tokens (passed straight through).
    `content`  → routed through `ChatBodyStripper` to extract only the
                 outermost function body, yielded as kind="code" Tokens.

    If `think_char_budget` is set, cumulative `thinking` chars are tracked.
    When the cap is exceeded, `on_budget_exceeded` is invoked and the stream
    is aborted — caller's continuation path (RAW mode) will produce code.
    """
    think_chars_seen = 0
    async for line in resp.aiter_lines():
        if not line.strip():
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = evt.get("message") or {}
        thinking = msg.get("thinking")
        if thinking:
            think_chars_seen += len(thinking)
            if (think_char_budget is not None
                    and think_chars_seen > think_char_budget):
                if on_budget_exceeded is not None:
                    on_budget_exceeded()
                # Stop here — caller's `LlmServer.next()` will see EOS, the
                # producer loop will exit, final_check will mark the body
                # incomplete, and the continuation path (RAW mode) will run.
                return
            yield Token(thinking, "think")
        content = msg.get("content")
        if content:
            for tok in stripper.feed(content):
                yield tok
        if evt.get("done"):
            for tok in stripper.flush():
                yield tok
            break
