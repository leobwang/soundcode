"""Statement/block boundary detector for incremental Rust source.

A boundary is a `;` or `}` that terminates a statement or block in Rust source:
outside any string/char/comment context and outside any unbalanced
parens `()` or brackets `[]`. Braces `{}` are themselves block-structural, so
they do *not* suppress boundaries — every `}` that closes a block is itself
a boundary.

Used by the rollback runner to decide when to snapshot generation state
and fire an async LSP pull.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Boundary:
    """A position in source where generation may checkpoint."""

    offset: int      # byte/char offset (UTF-8 aware via Python str indexing)
    kind: str        # ";" or "}"


def find_boundaries(source: str) -> list[Boundary]:
    """Scan a complete Rust source string and return all boundary positions.

    Offsets are in Python str indices (unicode scalars), not bytes.
    Returns boundaries in order of occurrence.
    """
    boundaries: list[Boundary] = []
    i = 0
    n = len(source)

    # Counters. `paren`, `bracket` suppress boundaries when > 0.
    # `brace` is tracked but doesn't suppress (boundaries fire on every `}`).
    paren = 0
    bracket = 0

    while i < n:
        c = source[i]

        # --- Comments ---
        if c == "/" and i + 1 < n:
            nxt = source[i + 1]
            if nxt == "/":
                # Line comment — skip to newline
                j = source.find("\n", i + 2)
                i = n if j == -1 else j + 1
                continue
            if nxt == "*":
                # Block comment (nestable in Rust)
                depth = 1
                j = i + 2
                while j < n and depth > 0:
                    if source[j] == "/" and j + 1 < n and source[j + 1] == "*":
                        depth += 1
                        j += 2
                    elif source[j] == "*" and j + 1 < n and source[j + 1] == "/":
                        depth -= 1
                        j += 2
                    else:
                        j += 1
                i = j
                continue

        # --- Raw strings: r"...", r#"..."#, br"...", br#"..."#, etc. ---
        # Longest-first check.
        if c in ("r", "b") and _try_raw_string_start(source, i) is not None:
            end = _try_raw_string_start(source, i)
            if end is not None:
                i = end
                continue

        # --- Byte string: b"..." ---
        if c == "b" and i + 1 < n and source[i + 1] == '"':
            i = _skip_regular_string(source, i + 1) + 1
            continue

        # --- Regular string: "..." ---
        if c == '"':
            i = _skip_regular_string(source, i) + 1
            continue

        # --- Char literal vs. lifetime (`'`) ---
        if c == "'":
            end = _try_char_literal_end(source, i)
            if end is not None:
                i = end
                continue
            # Lifetime: just consume the apostrophe; the identifier
            # characters after it will be consumed as normal tokens.
            i += 1
            continue

        # --- Delimiters ---
        if c == "(":
            paren += 1
        elif c == ")":
            paren = max(0, paren - 1)
        elif c == "[":
            bracket += 1
        elif c == "]":
            bracket = max(0, bracket - 1)
        elif c == "{":
            pass  # brace tracking intentionally omitted; no suppression
        elif c == "}":
            if paren == 0 and bracket == 0:
                boundaries.append(Boundary(offset=i, kind="}"))
        elif c == ";":
            if paren == 0 and bracket == 0:
                boundaries.append(Boundary(offset=i, kind=";"))

        i += 1

    return boundaries


def _skip_regular_string(source: str, open_quote_idx: int) -> int:
    """Return the index of the closing `"` of a regular (non-raw) string.

    Handles `\\x` escapes. If the string is unterminated, returns len-1.
    """
    n = len(source)
    i = open_quote_idx + 1
    while i < n:
        c = source[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == '"':
            return i
        i += 1
    return n - 1  # unterminated — consume to EOF


def _try_raw_string_start(source: str, i: int) -> int | None:
    """Detect a raw string at position `i`. Return index after the closing quote+hashes.

    Patterns: r"...", r#"..."#, r##"..."##, br"...", br#"..."#, ...
    Returns None if the position isn't a raw-string start.
    """
    n = len(source)
    j = i
    # Optional `b` prefix
    if j < n and source[j] == "b":
        j += 1
    # Must start with `r`
    if j >= n or source[j] != "r":
        return None
    j += 1
    # Count hashes
    hashes = 0
    while j < n and source[j] == "#":
        hashes += 1
        j += 1
    if j >= n or source[j] != '"':
        return None
    # Now scan until closing `"` followed by `hashes` `#`s
    j += 1
    while j < n:
        if source[j] == '"':
            # Check following hashes
            k = j + 1
            count = 0
            while k < n and count < hashes and source[k] == "#":
                count += 1
                k += 1
            if count == hashes:
                return k  # index after all closing hashes
        j += 1
    return n  # unterminated


def _try_char_literal_end(source: str, i: int) -> int | None:
    """Check if source[i] starts a char literal; return index after closing `'`.

    Returns None if this is a lifetime, not a char.
    Patterns accepted:
      'x'         (3 chars)
      '\\n', '\\\\'  (4 chars for simple escapes)
      '\\xHH'        (6 chars)
      '\\u{HHHH}'    (up to ~10 chars)
    """
    n = len(source)
    assert source[i] == "'"

    # Try simple: 'X'
    if i + 2 < n and source[i + 1] != "\\" and source[i + 2] == "'":
        return i + 3

    # Try simple escape: '\X'
    if i + 3 < n and source[i + 1] == "\\" and source[i + 3] == "'":
        return i + 4

    # Try hex escape: '\xHH'
    if i + 5 < n and source[i + 1] == "\\" and source[i + 2] == "x" and source[i + 5] == "'":
        return i + 6

    # Try unicode escape: '\u{...}'
    if i + 4 < n and source[i + 1] == "\\" and source[i + 2] == "u" and source[i + 3] == "{":
        # Find the closing `}` within a reasonable span then expect `'`
        for j in range(i + 4, min(i + 12, n)):
            if source[j] == "}":
                if j + 1 < n and source[j + 1] == "'":
                    return j + 2
                return None
            if source[j] == "'" or source[j] == "\n":
                return None
    return None


class BoundaryTracker:
    """Incremental tracker: call `update(source)` as tokens arrive.

    Exposes `latest` — the index of the latest boundary in the source so far.
    A returned index of -1 means no boundary seen yet.
    """

    def __init__(self) -> None:
        self._boundaries: list[Boundary] = []

    def update(self, source: str) -> list[Boundary]:
        """Rescan the full source. O(n) per call; source is small enough per problem.

        Returns the current list of all boundaries (not just new ones).
        """
        self._boundaries = find_boundaries(source)
        return self._boundaries

    @property
    def latest(self) -> int:
        return self._boundaries[-1].offset if self._boundaries else -1

    @property
    def boundaries(self) -> list[Boundary]:
        return list(self._boundaries)
