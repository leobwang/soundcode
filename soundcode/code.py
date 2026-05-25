"""Code buffer with checkpoint + diagnostic state — the Code/Diagnostic classes
from draft-plan-0.md §3.5.

The buffer is mutated only by the producer loop (synchronous appends). A
single async `check()` may run at a time; if the loop schedules a new check
before the previous one finishes, the previous one is cancelled by the
caller. `check()` therefore does not need to be thread-safe with respect to
itself.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

from soundcode.eval.boundary import find_boundaries

if TYPE_CHECKING:
    from soundcode.cargo_check import CargoChecker


def _content_brace_balance(s: str, end: int | None = None) -> int:
    """Signed brace balance of `s[:end]`, ignoring braces inside strings,
    char literals, and comments. Positive = more `{` than `}`; negative =
    the `}`s have outnumbered the `{`s, which means the buffer has closed
    something that was opened *before* it (e.g. the function body whose
    opening `{` lives in the prompt — see `Code.body_closed`)."""
    if end is None:
        end = len(s)
    balance = 0
    i = 0
    while i < end:
        c = s[i]
        if c == "/" and i + 1 < end and s[i + 1] == "/":
            j = s.find("\n", i + 2)
            i = end if j == -1 else j + 1
            continue
        if c == "/" and i + 1 < end and s[i + 1] == "*":
            j = s.find("*/", i + 2)
            i = end if j == -1 else j + 2
            continue
        if c == '"':
            j = i + 1
            while j < end:
                if s[j] == "\\":
                    j += 2; continue
                if s[j] == '"':
                    break
                j += 1
            i = j + 1
            continue
        if c == "'":
            j = i + 1
            # char literal up to 4 chars; otherwise lifetime
            while j < end and s[j] not in ("'", "\n"):
                if s[j] == "\\":
                    j += 2; continue
                j += 1
            i = j + 1 if j < end and s[j] == "'" else i + 1
            continue
        if c == "{":
            balance += 1
        elif c == "}":
            balance -= 1
        i += 1
    return balance


def _content_brace_depth(s: str, end: int) -> int:
    """Non-negative brace depth — clamped variant of `_content_brace_balance`.

    Used by `at_boundary` to gate boundary-cadence checks (we only want to
    check at function-body top level, where depth == 0 but balance hasn't
    yet gone negative)."""
    return max(0, _content_brace_balance(s, end))


class State(str, enum.Enum):
    INACTIVE = "inactive"  # no actionable signal yet
    OK = "ok"              # checked: no blocking diagnostics
    ERROR = "error"        # checked: ≥1 blocking diagnostic


class Category(str, enum.Enum):
    INCOMPLETE = "incomplete"      # syntax-error-class, "expected", "unterminated", ...
    BLOCKING = "blocking"          # real error past the writing edge
    NON_BLOCKING = "non_blocking"  # warnings, lints, hints


@dataclass(frozen=True)
class Diagnostic:
    category: Category
    message: str
    code: str | None = None
    line: int | None = None
    column: int | None = None

    @property
    def is_blocking(self) -> bool:
        return self.category is Category.BLOCKING


@dataclass(frozen=True)
class CheckResult:
    verdict: State
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.verdict is State.ERROR

    @property
    def message(self) -> str:
        """Rendered diagnostic text used for rollback-instruct.

        Deduplicates by (code, message) so when cargo reports the same error
        multiple times (e.g. one E0434 per offending inner fn), the LM is told
        about it once. Repeating identical lines collapses the probability of
        the LM continuing in comment-echo mode — see the rollback-instruct
        "prompt-echo trap" described in draft-report-0.md §5.3.
        """
        seen: set[tuple[str | None, str]] = set()
        lines: list[str] = []
        for d in self.diagnostics:
            if not d.is_blocking:
                continue
            key = (d.code, d.message.strip())
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  - {d.code or '?'}: {d.message}")
        return "\n".join(lines)


class Code:
    """Append-only buffer of generated code, with checkpoint history.

    See draft-plan-0.md §3.4 for the state machine.
    """

    def __init__(
        self,
        *,
        prefix: str,
        suffix: str,
        checker: "CargoChecker",
        function_closer: str = "\n    unreachable!()\n}\n\nfn main() {}\n",
    ) -> None:
        # `prefix` is the prompt/function signature already on disk.
        # `suffix` is what closes the function (the test harness etc.)
        # — only used for the FINAL cargo-check; not for in-loop checks.
        # `function_closer` is the per-language source-tail glued onto the
        # snapshot inside `check()` so the surface fed to the checker is a
        # syntactically complete file. Default is the Rust shim (preserves
        # historic behaviour for Rust callers); the multilingual web demo
        # supplies the per-language closer from `web.server.LANG_CLOSER`.
        self.prefix = prefix
        self.suffix = suffix
        self.function_closer = function_closer
        self.content = ""             # what the LM has generated so far
        self.state: State = State.INACTIVE
        self.diagnostics: list[Diagnostic] = []
        self.ckpt: list[int] = [0]    # offsets into self.content; 0 is always valid
        # Offset cursor for boundary-based auto-checkpointing. `append()` only
        # adds ckpts for boundaries at offsets strictly past this cursor;
        # `rollback()` advances it to the rollback target so that the boundary
        # we just walked back through doesn't immediately get re-added as a
        # ckpt by the next `find_boundaries` rescan (which would re-trap the
        # cargo-error retry on the same survivor).
        self._ckpt_scan_floor: int = 0
        self._checker = checker
        # Track how many tokens / boundary checks have happened (telemetry).
        self.tokens_seen: int = 0
        self.checks_run: int = 0
        self.lsp_calls: int = 0

    # ─── sync API used by the producer loop ───────────────────────────

    def append(self, token: str) -> None:
        """Append a token to the buffer.

        Also adds a checkpoint at every newly-completed structural boundary
        (a `;` or `}` outside strings/comments/parens/brackets). Boundary
        checkpoints are eager — they don't wait for cargo to verify the slice.
        This makes rollbacks fine-grained: the cargo-error path can target the
        boundary just before the error site instead of the last verified-OK
        position (which may be many statements back). Callers that want a
        verified-clean rollback target must filter on `offset_at_check`
        themselves; see `DemoClient._do_rollback`'s cargo-error branch.
        """
        self.content += token
        self.tokens_seen += 1
        self._update_boundary_checkpoints()

    def _update_boundary_checkpoints(self) -> None:
        """Append a checkpoint at offset `b.offset + 1` for every NEW boundary
        — one that is strictly past both the latest existing checkpoint and
        the consumed-boundary floor (`_ckpt_scan_floor`, advanced on rollback).

        The floor is what prevents the cargo-error retry loop from looping
        forever: `rollback()` removes the survivor ckpt AND raises the floor
        to that offset, so a subsequent rescan does NOT re-add ckpts at
        offsets ≤ target even though the `;`/`}` chars are still in the
        buffer. Without the floor, the next `find_boundaries` rescan would
        re-add the just-removed ckpt and pin the retry to the same survivor."""
        bounds = find_boundaries(self.content)
        last = self.ckpt[-1] if self.ckpt else 0
        for b in bounds:
            offset = b.offset + 1
            if offset <= self._ckpt_scan_floor:
                continue
            if offset > last:
                self.ckpt.append(offset)
                last = offset

    @property
    def body_closed(self) -> bool:
        """True iff the LM has emitted more `}` than `{` in its content —
        i.e. it has closed the function body whose opening `{` lives in
        the prompt. This is the "structurally I am done" signal; the
        generator should stop sampling once this fires."""
        return _content_brace_balance(self.content) < 0

    def at_boundary(self) -> bool:
        """True iff the most recent token closed a statement at the function-body
        top level (brace depth == 0 in the *content*, since the prefix opened the
        function body — so content-depth 0 = inside the function, not nested).

        Boundaries inside `for`/`if`/match/etc. blocks are skipped because the
        `unreachable!()` closer would land in the wrong scope.
        """
        if not self.content:
            return False
        last = self.content.rstrip()
        if not last or last[-1] not in (";", "}"):
            return False
        bounds = find_boundaries(self.content)
        if not bounds:
            return False
        last_char_offset = len(self.content.rstrip()) - 1
        if bounds[-1].offset != last_char_offset:
            return False
        # Compute brace depth at this offset in self.content. Function-body
        # interior is depth 0 (prefix opened the function). Mid-loop/if is
        # depth ≥ 1.
        depth = _content_brace_depth(self.content, last_char_offset + 1)
        return depth == 0

    def content_up_to(self, offset: int) -> str:
        return self.content[:offset]

    def rollback(self, to_offset: int | None = None) -> None:
        """Truncate content back to a checkpoint and reset state.

        If `to_offset` is None, uses the latest checkpoint (`ckpt[-1]`). If
        `to_offset` is given, truncates explicitly to that offset.

        In both cases, ALSO removes the target checkpoint from `ckpt` (along
        with anything past it). This is what makes the cargo-error retry loop
        progress: after rolling back to X, if the LM regenerates without
        crossing a new boundary, the next rollback picks the next-older
        ckpt (one further step back) instead of the same X — without this
        the generator can loop on the same survivor offset until the wall
        budget kills it.
        """
        if to_offset is None:
            target = self.ckpt[-1] if self.ckpt else 0
        else:
            target = max(0, min(to_offset, len(self.content)))
        self.ckpt = [c for c in self.ckpt if c < target]
        if not self.ckpt:
            self.ckpt = [0]
        self.content = self.content[:target]
        # Block the next `append`'s auto-ckpt scan from re-discovering the
        # boundary we just rolled back through. The floor is set to the
        # current target (NOT max(prev, target)) — content past `target` has
        # been truncated, so the "old vs new" distinction depends only on the
        # current cut point. Keeping the floor monotonic would silently swallow
        # NEW boundaries in regenerated content whenever the LM rolls back to
        # an offset lower than the previous floor.
        self._ckpt_scan_floor = target
        self.diagnostics = []
        self.state = State.INACTIVE

    # ─── async API used by the consumer queue ─────────────────────────

    async def check(self) -> CheckResult:
        """Run cargo check on the current buffer; update state + checkpoints.

        Returns a CheckResult. Cancellation is a no-op — the buffer is not
        mutated, and the caller (loop) is expected to handle the cancellation.
        """
        # Snapshot the content offset at which we are checking. After we
        # await the subprocess, content may have grown; but we only checkpoint
        # at the snapshot offset, so the checkpoint corresponds to verified code.
        snapshot_offset = len(self.content)
        snapshot = self.content
        self.checks_run += 1
        try:
            self.lsp_calls += 1
            # `function_closer` is the per-language source-tail glued onto
            # the snapshot so the surface fed to the checker is a complete
            # file. For Rust (the historical default) this is the
            # `unreachable!()` shim that satisfies any return type while
            # leaving only *in-content* errors visible; for the multilingual
            # web demo the per-language closer comes from `LANG_CLOSER` in
            # `soundcode.web.server` (e.g. `\n    return {};\n}\n\nint
            # main() { return 0; }\n` for C++).
            diagnostics = await self._checker.check(
                self.prefix + snapshot + self.function_closer
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Checker error treated as INACTIVE (don't block on infrastructure issues).
            return CheckResult(verdict=State.INACTIVE, diagnostics=[])

        # Classify each diagnostic.
        classified = [self._classify(d) for d in diagnostics]
        blocking = [d for d in classified if d.is_blocking]
        non_blocking = [d for d in classified if d.category is Category.NON_BLOCKING]

        if blocking:
            self.diagnostics = classified
            self.state = State.ERROR
            return CheckResult(verdict=State.ERROR, diagnostics=classified)

        # No blocking → either OK (at least one non-blocking signal, or clean)
        # or INACTIVE (only incomplete-class diagnostics, i.e. unparseable code).
        only_incomplete = all(d.category is Category.INCOMPLETE for d in classified)
        if classified and only_incomplete:
            return CheckResult(verdict=State.INACTIVE, diagnostics=classified)

        # OK: clean or only non-blocking diagnostics. Checkpoint here.
        self.diagnostics = classified
        self.state = State.OK
        if snapshot_offset > self.ckpt[-1]:
            self.ckpt.append(snapshot_offset)
        return CheckResult(verdict=State.OK, diagnostics=classified)

    # ─── classifier (open-question #4 answer: simple version) ─────────

    @staticmethod
    def _classify(d: Diagnostic) -> Diagnostic:
        """Re-classify a raw diagnostic into INCOMPLETE / BLOCKING / NON_BLOCKING.

        Per the answer to open Q4: "any error not attributed to incomplete code"
        is BLOCKING. We mark INCOMPLETE for syntax-error class and messages
        with classic incompleteness tokens. NON_BLOCKING covers warnings.
        """
        # The cargo_check layer already pre-marks category for warnings vs errors;
        # this just refines errors into INCOMPLETE vs BLOCKING.
        if d.category is Category.NON_BLOCKING:
            return d
        msg_lower = d.message.lower()

        # Codes that strictly indicate "more code is needed to resolve":
        #   E0282/E0283/E0284 — type annotations needed (inference cannot proceed
        #                       until the variable is used in a typed context).
        #   E0698             — type annotations needed for closure.
        #   E0601             — main not found (only happens during checks).
        #   syntax-error      — by definition.
        INCOMPLETE_CODES = {
            "syntax-error", "E0601",
            "E0282", "E0283", "E0284", "E0698",
        }
        if d.code in INCOMPLETE_CODES:
            return Diagnostic(
                category=Category.INCOMPLETE,
                message=d.message,
                code=d.code,
                line=d.line,
                column=d.column,
            )
        for tok in ("unterminated", "expected", "unclosed", "missing trailing",
                    "unexpected eof", "unexpected end of file", ", found"):
            # "expected X, found Y" is type-mismatch — keep as BLOCKING.
            if tok == ", found":
                if ", found" in msg_lower:
                    return d  # keep BLOCKING
                continue
            if tok in msg_lower:
                return Diagnostic(
                    category=Category.INCOMPLETE,
                    message=d.message,
                    code=d.code,
                    line=d.line,
                    column=d.column,
                )
        return d
