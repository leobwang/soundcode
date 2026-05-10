"""Rollback history: accumulate past errors, render them into the prompt.

Per `notes/week4-plan-v2.0.md §5.2, §5.3`:
- Every rollback appends a new entry.
- The full history is rendered into the prompt on subsequent attempts.
- When `rendered_prompt + code_prefix + completion_budget` approaches 90%
  of the model's context window, compact older entries into a single line.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from soundcode.analyzer import Diagnostic


@dataclass(frozen=True)
class HistoryEntry:
    attempt: int          # 1-indexed attempt number this error belongs to
    line: int             # 1-indexed source line of the error (for display)
    code: str             # e.g. "E0308", or "" if unknown
    message: str          # diagnostic message, trimmed


@dataclass
class RollbackHistory:
    """Accumulates errors across rollbacks for a single problem."""

    entries: list[HistoryEntry] = field(default_factory=list)
    _attempt_counter: int = 0

    def begin_attempt(self) -> int:
        """Return the new attempt number; call once per generation attempt."""
        self._attempt_counter += 1
        return self._attempt_counter

    def add(self, diag: Diagnostic, *, attempt: int | None = None) -> None:
        """Record a diagnostic under the current (or specified) attempt."""
        a = attempt if attempt is not None else self._attempt_counter
        msg = diag.message.strip().split("\n", 1)[0]  # first line only
        code = (diag.code or "").strip()
        line = diag.range.start.line + 1  # human-facing 1-indexed
        self.entries.append(HistoryEntry(attempt=a, line=line, code=code, message=msg))

    def __len__(self) -> int:
        return len(self.entries)

    # ---- Rendering ----

    def render_full(self) -> str:
        """Render all entries verbatim, chronologically."""
        if not self.entries:
            return ""
        lines = ["// Previous attempts produced the following errors:"]
        for e in self.entries:
            code = f"{e.code} " if e.code else ""
            lines.append(f"// Attempt {e.attempt}: {code}at line {e.line}: {e.message}")
        return "\n".join(lines) + "\n"

    def render_compact(self, keep_recent: int = 3) -> str:
        """Render with the oldest entries collapsed into a one-line summary.

        Keeps the `keep_recent` most recent entries verbatim and compresses
        the rest into a single line listing distinct error codes and counts.
        """
        if len(self.entries) <= keep_recent:
            return self.render_full()

        recent = self.entries[-keep_recent:]
        older = self.entries[:-keep_recent]

        # Count distinct codes (use "(no-code)" sentinel for entries without code)
        counts: Counter[str] = Counter()
        for e in older:
            counts[e.code or "(no-code)"] += 1

        code_summary_parts = []
        for code, n in counts.most_common():
            code_summary_parts.append(f"{code}×{n}" if n > 1 else code)
        code_summary = ", ".join(code_summary_parts)

        lines = [
            "// Previous attempts produced the following errors:",
            f"// [Attempts 1–{len(older)}]: {code_summary}",
        ]
        for e in recent:
            code = f"{e.code} " if e.code else ""
            lines.append(f"// Attempt {e.attempt}: {code}at line {e.line}: {e.message}")
        return "\n".join(lines) + "\n"

    def render(
        self,
        *,
        context_window: int,
        reserved_completion_tokens: int,
        prompt_so_far: str,
    ) -> str:
        """Render history, choosing full or compact based on estimated token usage.

        Rough heuristic: 4 chars per token. If prompt_so_far + full_history
        would exceed 90% of (context_window - reserved_completion), compact.
        """
        full = self.render_full()
        # Rough 4-chars-per-token heuristic
        est_tokens = (len(prompt_so_far) + len(full)) // 4
        budget = int(0.9 * (context_window - reserved_completion_tokens))
        if est_tokens <= budget:
            return full
        # Compaction reduces by collapsing older entries into a single line.
        return self.render_compact(keep_recent=3)

    def code_distribution(self) -> dict[str, int]:
        """Distinct error-code counts. Useful for profiling / reporting."""
        return dict(Counter(e.code for e in self.entries))

    def as_dict_list(self) -> list[dict]:
        return [
            {"attempt": e.attempt, "line": e.line, "code": e.code, "message": e.message}
            for e in self.entries
        ]

    def clone(self) -> "RollbackHistory":
        h = RollbackHistory(
            entries=list(self.entries),
            _attempt_counter=self._attempt_counter,
        )
        return h
