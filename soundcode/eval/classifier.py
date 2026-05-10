"""Classify rust-analyzer diagnostics as blocking or non-blocking.

Implements the three-signal procedure from
`notes/rust-analyzer-scope-and-granularity.md §6`:

1. Position relative to last completion boundary
2. Error class (syntactic incompleteness vs. semantic)
3. Optional temporal stability (not used by default; see README)

Exposes `classify(diag, ...) -> Verdict` and `filter_blocking(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from soundcode.analyzer import Diagnostic, DiagnosticSeverity


class Verdict(str, Enum):
    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"


# Codes that, by themselves, always indicate incompleteness — never block.
_SYNTACTIC_CODES: frozenset[str] = frozenset({
    "syntax-error",
    "unresolved-proc-macro",   # macros commonly break on partial code
})

# Substrings that indicate an incompleteness / lex-level problem when seen
# in a diagnostic message.
_INCOMPLETENESS_TOKENS: tuple[str, ...] = (
    "unterminated",
    "expected",
    "unclosed",
    "missing trailing",
    "incomplete",
    "unexpected end of file",
    "unexpected eof",
)

# Semantic codes that can be demoted to non-blocking when the enclosing
# function body is still open (they often resolve once the body completes).
# This is the *full* default set; `DemotionPolicy` (below) selects a subset.
_TYPE_MISMATCH_CODES: frozenset[str] = frozenset({"type-mismatch", "E0308"})
_UNRESOLVED_REF_CODES: frozenset[str] = frozenset({"unresolved-reference", "E0425"})


@dataclass(frozen=True)
class DemotionPolicy:
    """Controls which semantic error classes are downgraded to non-blocking
    when the enclosing function body is still open.

    The §6 default demotes both `type-mismatch` and `unresolved-reference`
    on the theory that both are commonly resolved by code written later in
    the same function (forward references and not-yet-typed expressions).

    Blog 3's validation showed that this overlap with rust-analyzer's
    actual native-diagnostic output during partial-code generation
    neutralizes the LSP tier — every emitted error is demoted. The blog
    proposed splitting the two: continue demoting `unresolved-reference`
    (forward refs are common and benign) but stop demoting `type-mismatch`
    (a type error on a *completed* expression is usually real).
    """
    demote_type_mismatch: bool = True
    demote_unresolved_ref: bool = True

    @property
    def name(self) -> str:
        if self.demote_type_mismatch and self.demote_unresolved_ref:
            return "both"
        if self.demote_unresolved_ref and not self.demote_type_mismatch:
            return "ref_only"
        if self.demote_type_mismatch and not self.demote_unresolved_ref:
            return "type_only"
        return "none"


DEFAULT_POLICY = DemotionPolicy()


# Backwards-compat: the original frozenset that callers used.
_DEMOTABLE_WHILE_FN_OPEN: frozenset[str] = _TYPE_MISMATCH_CODES | _UNRESOLVED_REF_CODES

# Semantic codes that always block (imports are module-scope, etc.).
# Listed for documentation; anything not syntactic and not demotable
# falls into this category by default.
_ALWAYS_BLOCKING: frozenset[str] = frozenset({
    "unresolved-import",
    "unresolved-module",
    "no-method",
    "trait-impl-incorrect",
    "missing-match-arm",
    "missing-unsafe",
    "E0432",
    "E0599",
    "E0277",
    "E0133",
})


def line_col_to_offset(source: str, line: int, character: int) -> int:
    """Convert an LSP (line, character) position to a character offset in source.

    Both line and character are 0-indexed. `character` is interpreted as a
    Python-string (unicode-scalar) offset. For non-BMP characters the LSP
    default is UTF-16 code units; since we negotiate UTF-8 in
    `analyzer.py` initialization and most generated code is ASCII, this
    aligns correctly for our workload. We clamp to source length on overflow.
    """
    if line < 0:
        return 0
    # splitlines(keepends=True) preserves \n so offsets add correctly.
    lines = source.splitlines(keepends=True)
    if line >= len(lines):
        # Past EOF — return source length.
        return len(source)
    prefix_len = sum(len(l) for l in lines[:line])
    line_text = lines[line].rstrip("\n").rstrip("\r")
    col = min(max(0, character), len(line_text))
    return prefix_len + col


@dataclass(frozen=True)
class ClassifyResult:
    verdict: Verdict
    reason: str  # short human-readable tag: why this decision

    @property
    def is_blocking(self) -> bool:
        return self.verdict == Verdict.BLOCKING


def _message_looks_incomplete(message: str) -> bool:
    msg_lower = message.lower()
    # "expected X, found Y" is the standard rustc type-mismatch signature and
    # is *semantic*, not incomplete. Exclude that pattern before checking the
    # incompleteness substrings.
    if ", found" in msg_lower:
        return False
    return any(tok in msg_lower for tok in _INCOMPLETENESS_TOKENS)


def classify(
    diag: Diagnostic,
    *,
    source: str,
    last_complete_pos: int,
    open_fn_body: bool,
    policy: DemotionPolicy = DEFAULT_POLICY,
) -> ClassifyResult:
    """Classify a single diagnostic.

    Args:
        diag: Diagnostic from rust-analyzer.
        source: Full current source buffer.
        last_complete_pos: Index of the most recent `;` or `}` that closed
            a statement or block at depth 0 (parens/brackets). Anything
            beyond this index is "in progress."
        open_fn_body: True if there is an unclosed top-level function body
            at the writing edge. Enables demotion of some semantic codes.
        policy: Which semantic codes to demote when ``open_fn_body``.
            Default §6 behaviour demotes both type-mismatch and
            unresolved-reference; pass alternative policies to ablate.

    Returns:
        ClassifyResult with a verdict and a short reason tag.
    """
    # Severity gate — only ERROR triggers rollback. Warnings/infos ignored.
    if diag.severity != DiagnosticSeverity.ERROR:
        return ClassifyResult(Verdict.NON_BLOCKING, "severity-not-error")

    code = diag.code or ""

    # Signal 1: position past the writing edge -> definitely in-progress.
    diag_offset = line_col_to_offset(source, diag.range.start.line, diag.range.start.character)
    if diag_offset > last_complete_pos:
        return ClassifyResult(Verdict.NON_BLOCKING, "past-writing-edge")

    # Signal 2a: syntactic-class codes are never blocking.
    if code in _SYNTACTIC_CODES:
        return ClassifyResult(Verdict.NON_BLOCKING, "syntactic-code")

    # Signal 2b: message mentions incompleteness tokens.
    if _message_looks_incomplete(diag.message):
        return ClassifyResult(Verdict.NON_BLOCKING, "incompleteness-message")

    # Signal 2c: demotion while function body is still open — controlled
    # by the active policy.
    if open_fn_body:
        if policy.demote_type_mismatch and code in _TYPE_MISMATCH_CODES:
            return ClassifyResult(Verdict.NON_BLOCKING, "demoted-type-mismatch")
        if policy.demote_unresolved_ref and code in _UNRESOLVED_REF_CODES:
            return ClassifyResult(Verdict.NON_BLOCKING, "demoted-unresolved-ref")

    # Default: remaining semantic errors in completed code are blocking.
    return ClassifyResult(Verdict.BLOCKING, "semantic-in-completed-code")


def filter_blocking(
    diags: list[Diagnostic],
    *,
    source: str,
    last_complete_pos: int,
    open_fn_body: bool,
    policy: DemotionPolicy = DEFAULT_POLICY,
) -> list[tuple[Diagnostic, ClassifyResult]]:
    """Run `classify` on a list of diagnostics; return only the blocking ones.

    Non-blocking diagnostics are filtered out. Useful as the hot-path
    predicate in the rollback runner.
    """
    out: list[tuple[Diagnostic, ClassifyResult]] = []
    for d in diags:
        r = classify(
            d,
            source=source,
            last_complete_pos=last_complete_pos,
            open_fn_body=open_fn_body,
            policy=policy,
        )
        if r.is_blocking:
            out.append((d, r))
    return out


# ---- Convenience: parse policy from a CLI string ----

def policy_from_name(name: str) -> DemotionPolicy:
    """Map a CLI string ('both', 'ref_only', 'type_only', 'none') to a policy."""
    mapping = {
        "both": DemotionPolicy(demote_type_mismatch=True, demote_unresolved_ref=True),
        "ref_only": DemotionPolicy(demote_type_mismatch=False, demote_unresolved_ref=True),
        "type_only": DemotionPolicy(demote_type_mismatch=True, demote_unresolved_ref=False),
        "none": DemotionPolicy(demote_type_mismatch=False, demote_unresolved_ref=False),
    }
    return mapping[name]
