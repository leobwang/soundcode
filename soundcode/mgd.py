# type: ignore
"""MGD-style dereference monitor adapted from microsoft/monitors4codegen
for the rust-analyzer + vLLM stack.

Differences from the reference DereferencesMonitor:

- Reference depends on multilspy's `insert_text_at_position` / `delete_text_between_positions`
  (multilspy versions that ship those mutators are older). We instead:
    write the buffer (prompt + generation_so_far) to src/main.rs, then call
    rust-analyzer for completions at the cursor.
- Reference uses code_tokenize for break-char detection — Java-flavored.
  For Rust we hard-code break chars from rustc lexer.
- Reference uses a `pygtrie` over the tokenizer vocabulary to find tokens
  whose prefix matches a legal completion. We use a precomputed dict
  mapping prefix-strings to token-ids built at monitor construction time.

The algorithm itself is preserved:

  state = UnInitialized -> S0 (free generation)
  on every step:
    if just-emitted text ends with `.`:
      query LSP completions; if non-empty, transition to Constrained.
    if Constrained:
      maintain a list `legal_completions` of suffixes still valid.
      mask logits so only tokens that (a) prefix a legal suffix, or
      (b) match a legal suffix followed by a break-char, can be sampled.
    on break-char in newly generated token:
      transition back to S0.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch


# Rust break-chars: any character that "closes" an identifier when appended.
# Spaces, punctuation, operators, brackets, control chars. Anything that is
# NOT [A-Za-z0-9_].
def _is_break_char(c: str) -> bool:
    return not (c.isalnum() or c == "_")


def _extract_leading_identifier(text: str) -> str:
    """Return the leading Rust-identifier-prefix of `text`.

    Strips trailing `()`, `(…)`, ` (as TraitName)`, and so on — anything
    after the first non-identifier character. Required because multilspy
    returns the LSP `label` (display text) as `completionText`, not the
    bare identifier.
    """
    n = 0
    for c in text:
        if c.isalnum() or c == "_":
            n += 1
        else:
            break
    return text[:n]


def _trailing_post_dot_identifier(text: str) -> str | None:
    """If text ends with `.<identifier-chars>` (possibly zero chars), return the
    identifier-chars suffix. Otherwise None.

    Handles `..` (range operator) and `.0` (tuple field) cases by rejecting
    them as triggers (the LM is probably constructing a numeric / range,
    not a method call)."""
    n = len(text)
    if n == 0:
        return None
    # Walk back over identifier chars
    i = n - 1
    while i >= 0 and (text[i].isalnum() or text[i] == "_"):
        i -= 1
    # i now points at first non-identifier char, or -1
    if i < 0 or text[i] != ".":
        return None
    # Reject `..` (range)
    if i > 0 and text[i - 1] == ".":
        return None
    # Reject `<digit>.<digits>` (float literal)
    if i > 0 and text[i - 1].isdigit() and i + 1 < n and text[i + 1].isdigit():
        return None
    return text[i + 1:]


class MonitorState(Enum):
    UNINITIALIZED = "uninitialized"
    S0 = "s0"                # free generation
    CONSTRAINED = "constrained"  # masking logits to a legal suffix


@dataclass
class MgdTelemetry:
    triggers: int = 0          # how many times `.` triggered an LSP query
    constrained_steps: int = 0 # how many tokens were sampled under a mask
    masks_built: int = 0       # number of mask operations
    empty_masks: int = 0       # times the mask was empty (would force abort)
    total_completions: int = 0 # sum of `len(legal_completions)` at each trigger
    rust_analyzer_calls: int = 0
    last_message: str = ""


class DereferenceMonitor:
    """Per-prompt monitor instance.

    Construction is lightweight; first `maskgen` call does the heavy work
    (prompt-len capture, LSP cursor positioning).
    """

    def __init__(
        self,
        tokenizer,
        completions_provider,
        prompt: str,
        log: MgdTelemetry | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.completions_provider = completions_provider
        self.prompt = prompt
        self.state = MonitorState.UNINITIALIZED
        self.prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        self.legal_completions: list[str] = []   # suffixes still valid
        self.last_generated_text = ""             # text from index prompt_len onward, last seen
        self.log = log or MgdTelemetry()
        # Build a prefix-string -> token-ids index over the model vocab so
        # mask construction is O(|legal|) rather than O(|vocab|).
        self._vocab_size = tokenizer.vocab_size
        self._tok_text: list[str] = self._build_token_text_table()

    def _build_token_text_table(self) -> list[str]:
        """Map token-id -> string (decoded). Slow at construction (~1 s) but O(1) lookup."""
        decoded = []
        for tid in range(self._vocab_size):
            try:
                s = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False, skip_special_tokens=False)
            except Exception:
                s = ""
            decoded.append(s)
        return decoded

    # ─── core update on each generation step ──────────────────────────

    def maskgen(self, input_ids: list[int]) -> set[int]:
        """Return the set of token-ids to MASK (block). Empty set means no constraint.

        Note: in vLLM v1, the per-request callable receives *only* the
        generated tokens (not the prompt). So input_ids IS gen_so_far_ids.
        """
        gen_so_far_ids = input_ids  # vLLM v1 passes only generated tokens
        gen_text = self.tokenizer.decode(
            gen_so_far_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True,
        )

        # 2) Compute the delta added since last call.
        new_text = gen_text[len(self.last_generated_text):]
        self.last_generated_text = gen_text

        # 3) Update the monitor state based on new_text + transitions.
        if self.state is MonitorState.UNINITIALIZED:
            self.state = MonitorState.S0
        elif self.state is MonitorState.CONSTRAINED:
            self.log.constrained_steps += 1
            if any(_is_break_char(c) for c in new_text):
                # End of identifier — drop constraint.
                self.state = MonitorState.S0
                self.legal_completions = []
            else:
                # Still emitting identifier characters: shrink the suffix set.
                self.legal_completions = [
                    s[len(new_text):]
                    for s in self.legal_completions
                    if s.startswith(new_text)
                ]
                if not self.legal_completions:
                    self.state = MonitorState.S0

        # 4) If in S0, check if cursor is in a "post-dot identifier" position.
        # Two cases:
        #   (a) gen_text ends with `.` exactly (sub-token break right after the dot)
        #   (b) gen_text ends with `.<identifier-chars>` — common when the
        #       tokenizer encodes `.method` or `.field` as a single token.
        if self.state is MonitorState.S0:
            full_text = self.prompt + gen_text
            partial_suffix = _trailing_post_dot_identifier(full_text)
            if partial_suffix is not None:
                self.log.triggers += 1
                # Always query at the position immediately after `.`
                cursor_text = full_text[: len(full_text) - len(partial_suffix)]
                completions = self.completions_provider(cursor_text)
                self.log.rust_analyzer_calls += 1
                self.log.total_completions += len(completions)
                # Filter to those compatible with the already-emitted prefix
                # (when partial_suffix is non-empty).
                if partial_suffix:
                    completions = [c[len(partial_suffix):] for c in completions
                                   if c.startswith(partial_suffix)]
                    completions = [c for c in completions if c]
                if completions:
                    self.legal_completions = completions
                    self.state = MonitorState.CONSTRAINED

        # 5) Build mask if constrained.
        if self.state is MonitorState.CONSTRAINED:
            self.log.masks_built += 1
            allowed: set[int] = set()
            for suffix in self.legal_completions:
                if not suffix:
                    continue
                # A token is allowed if its text is a prefix of `suffix`,
                # OR token text == suffix + (some break-char-leading tail).
                for tid, ttext in enumerate(self._tok_text):
                    if not ttext:
                        continue
                    if suffix.startswith(ttext):
                        allowed.add(tid)
                    elif ttext.startswith(suffix) and len(ttext) > len(suffix) and _is_break_char(ttext[len(suffix)]):
                        allowed.add(tid)
            if not allowed:
                self.log.empty_masks += 1
                self.log.last_message = f"empty mask after suffix-shrink (suffixes={self.legal_completions})"
                # Empty mask -> drop the constraint to avoid forcing infinite EOS;
                # this matches the reference behavior of "abandon on empty".
                self.state = MonitorState.S0
                self.legal_completions = []
                return set()

            block_set = set(range(self._vocab_size)) - allowed
            return block_set

        return set()


class RustAnalyzerCompletionsProvider:
    """Thread-safe wrapper around multilspy's completion query.

    Spins up a single rust-analyzer instance and reuses it. Each `__call__`
    writes the current buffer to src/main.rs and queries completions at
    (line, col) = end of buffer.

    Because vLLM's LogitsProcessor is called synchronously from the model's
    forward pass, this provider runs multilspy from a background thread that
    owns its own event loop.
    """

    def __init__(self, workspace: Path):
        from multilspy import LanguageServer
        from multilspy.multilspy_config import MultilspyConfig
        from multilspy.multilspy_logger import MultilspyLogger

        self.workspace = workspace
        self.target = workspace / "src" / "main.rs"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lsp = None
        self._lsp_ctx = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.config = MultilspyConfig.from_dict({"code_language": "rust"})
        self.logger = MultilspyLogger()

    def start(self):
        """Start the background event loop + rust-analyzer."""
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=60.0)

    def _thread_main(self):
        from multilspy import LanguageServer
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _start():
            self._lsp = LanguageServer.create(self.config, self.logger, str(self.workspace))
            self._lsp_ctx = self._lsp.start_server()
            await self._lsp_ctx.__aenter__()
            self._ready.set()

        loop.run_until_complete(_start())
        try:
            loop.run_until_complete(self._wait_until_stop())
        finally:
            try:
                loop.run_until_complete(self._lsp_ctx.__aexit__(None, None, None))
            except Exception:
                pass
            loop.close()

    async def _wait_until_stop(self):
        while not self._stop.is_set():
            await asyncio.sleep(0.5)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __call__(self, full_text: str) -> list[str]:
        """Write buffer to disk, query completions at (line, col) at end of buffer."""
        if self._lsp is None:
            return []
        with self._lock:
            # Append `unreachable!()` closer so the file parses while we have
            # a partial function body. The cursor for completion query is
            # still the actual end of `full_text` (before the closer).
            cursor_text = full_text
            closer = "\n    unreachable!()\n}\n\nfn main() {}\n"
            file_text = cursor_text + closer
            # cursor line/col = end of cursor_text (0-indexed)
            line = cursor_text.count("\n")
            last_nl = cursor_text.rfind("\n")
            col = len(cursor_text) - (last_nl + 1) if last_nl >= 0 else len(cursor_text)
            self.target.write_text(file_text)

            fut = asyncio.run_coroutine_threadsafe(
                self._async_request_completions(line, col), self._loop
            )
            try:
                return fut.result(timeout=5.0)
            except Exception:
                return []

    async def _async_request_completions(self, line: int, col: int) -> list[str]:
        try:
            with self._lsp.open_file("src/main.rs"):
                items = await self._lsp.request_completions("src/main.rs", line, col, allow_incomplete=True)
        except Exception:
            return []
        # LSP CompletionItemKind values we accept (method/field/etc.).
        # Skip keywords (14), text snippets (1), snippets (15),
        # color/file/folder/etc. We want only "real" name completions.
        # Reference: https://microsoft.github.io/language-server-protocol/specifications/specification-3-17/#completionItemKind
        IDENTIFIER_KINDS = {
            2,  # Method
            3,  # Function
            4,  # Constructor
            5,  # Field
            6,  # Variable
            7,  # Class
            8,  # Interface
            9,  # Module
            10, # Property
            13, # Enum
            20, # EnumMember
            21, # Constant
            22, # Struct
            25, # TypeParameter
        }
        out = []
        for item in items:
            text = item.get("completionText") or item.get("insertText") or item.get("label")
            kind = item.get("kind")
            if kind not in IDENTIFIER_KINDS:
                continue
            if not text:
                continue
            # Strip decoration: multilspy returns the *label* (e.g.
            # `iter()`, `drain(…)`, `to_owned() (as ToOwned)`).
            # We want just the leading identifier.
            ident = _extract_leading_identifier(text)
            if ident:
                out.append(ident)
        # Deduplicate while preserving order.
        seen = set()
        deduped = []
        for s in out:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped


# ─── vLLM v1 LogitsProcessor adapter ─────────────────────────────────────
#
# vLLM 0.20+ requires a class-based LogitsProcessor registered at LLM
# construction time. Per-request behavior is selected via SamplingParams.
# We attach our monitor through `extra_args["mgd_prompt"]` so the adapter
# can build a fresh DereferenceMonitor for each request.

_GLOBAL_PROVIDER: RustAnalyzerCompletionsProvider | None = None
_GLOBAL_TELEMETRY: dict[int, MgdTelemetry] = {}


def set_global_provider(p: RustAnalyzerCompletionsProvider) -> None:
    global _GLOBAL_PROVIDER
    _GLOBAL_PROVIDER = p


def fetch_telemetry(req_id: int) -> MgdTelemetry | None:
    return _GLOBAL_TELEMETRY.pop(req_id, None)


def build_mgd_adapter():
    """Build the AdapterLogitsProcessor subclass once vLLM is imported."""
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
    from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
    from transformers import AutoTokenizer

    _tokenizer_cache: dict[str, object] = {}

    class MgdAdapter(AdapterLogitsProcessor):
        def __init__(self, vllm_config, device, is_pin_memory: bool):
            super().__init__(vllm_config, device, is_pin_memory)
            model = vllm_config.model_config.model
            if model not in _tokenizer_cache:
                _tokenizer_cache[model] = AutoTokenizer.from_pretrained(model)
            self._tokenizer = _tokenizer_cache[model]

        def is_argmax_invariant(self) -> bool:
            return False

        def new_req_logits_processor(self, params):
            extra = getattr(params, "extra_args", None) or {}
            prompt = extra.get("mgd_prompt")
            req_id = extra.get("mgd_req_id")
            if not prompt or req_id is None or _GLOBAL_PROVIDER is None:
                return None
            tele = MgdTelemetry()
            _GLOBAL_TELEMETRY[req_id] = tele
            monitor = DereferenceMonitor(
                tokenizer=self._tokenizer,
                completions_provider=_GLOBAL_PROVIDER,
                prompt=prompt, log=tele,
            )

            def _per_request(token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
                try:
                    mask = monitor.maskgen(token_ids)
                except Exception as e:
                    tele.last_message = f"maskgen error: {e!r}"
                    return logits
                if mask:
                    idx = torch.tensor(list(mask), dtype=torch.long, device=logits.device)
                    logits[idx] = float("-inf")
                return logits

            return _per_request

    return MgdAdapter
