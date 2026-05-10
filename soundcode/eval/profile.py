"""Per-problem, per-event timing profiler.

Per `notes/week4-plan-v2.0.md §8.3`. Records spans with raw nanosecond
timestamps; aggregation happens at analysis time to allow multiple slicings.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# Canonical categories — the §8.3 table. Validated at write time.
CATEGORIES: frozenset[str] = frozenset({
    "generation",         # time between emitted tokens
    "prompt-ingestion",   # submit → first token
    "lsp-roundtrip",      # LSP request → response, per call
    "lsp-wait",           # generation blocked waiting for LSP (should be ~0)
    "classifier",         # §6 decision procedure
    "boundary-detect",    # ;/}` tracker
    "cargo-check",        # cargo check wall time
    "rollback-overhead",  # error detected → new request first token
    "thinking",           # <think>...</think> blocks (reasoning models)
    "idle/other",         # residual; catch-all
})


@dataclass
class Span:
    category: str
    t_start_ns: int
    t_end_ns: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        return self.t_end_ns - self.t_start_ns

    @property
    def duration_s(self) -> float:
        return self.duration_ns / 1e9

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "t_start_ns": self.t_start_ns,
            "t_end_ns": self.t_end_ns,
            "duration_s": self.duration_s,
            "meta": self.meta,
        }


@dataclass
class Profile:
    """Event log for a single (problem, model, arm) triple."""

    problem_id: str
    model: str
    arm: str
    spans: list[Span] = field(default_factory=list)
    t0_ns: int = field(default_factory=lambda: time.perf_counter_ns())

    @contextlib.contextmanager
    def span(self, category: str, **meta: Any) -> Iterator[None]:
        """Record a span for the duration of the context block."""
        if category not in CATEGORIES:
            raise ValueError(f"unknown category: {category!r} (valid: {sorted(CATEGORIES)})")
        t_start = time.perf_counter_ns()
        try:
            yield
        finally:
            t_end = time.perf_counter_ns()
            self.spans.append(Span(category=category, t_start_ns=t_start, t_end_ns=t_end, meta=meta))

    def add_span(
        self,
        category: str,
        t_start_ns: int,
        t_end_ns: int,
        **meta: Any,
    ) -> None:
        """Manually record a span (for events that don't fit a context block)."""
        if category not in CATEGORIES:
            raise ValueError(f"unknown category: {category!r}")
        self.spans.append(Span(category=category, t_start_ns=t_start_ns, t_end_ns=t_end_ns, meta=meta))

    # ---- Aggregation ----

    def total_wall_ns(self) -> int:
        """Wall-clock elapsed since profile started."""
        return time.perf_counter_ns() - self.t0_ns

    def duration_by_category(self) -> dict[str, int]:
        """Sum durations per category (ns). Sums, not merged — may overlap."""
        out: dict[str, int] = {c: 0 for c in CATEGORIES}
        for s in self.spans:
            out[s.category] += s.duration_ns
        return out

    def wallclock_by_category(self) -> dict[str, int]:
        """Merge overlapping spans per category, return total elapsed per category (ns).

        Useful when spans within a category overlap in time (e.g., concurrent LSP
        calls) — this reports the actual wall-clock portion attributable to each
        category, not the summed durations.
        """
        out: dict[str, int] = {c: 0 for c in CATEGORIES}
        # Group spans by category, sort by start, merge overlapping.
        by_cat: dict[str, list[tuple[int, int]]] = {c: [] for c in CATEGORIES}
        for s in self.spans:
            by_cat[s.category].append((s.t_start_ns, s.t_end_ns))
        for cat, intervals in by_cat.items():
            intervals.sort()
            merged_total = 0
            cur_start, cur_end = -1, -1
            for a, b in intervals:
                if a > cur_end:
                    if cur_start >= 0:
                        merged_total += cur_end - cur_start
                    cur_start, cur_end = a, b
                else:
                    cur_end = max(cur_end, b)
            if cur_start >= 0:
                merged_total += cur_end - cur_start
            out[cat] = merged_total
        return out

    def rollback_overhead_ns(self) -> list[int]:
        """Return durations of all `rollback-overhead` spans."""
        return [s.duration_ns for s in self.spans if s.category == "rollback-overhead"]

    def to_dict(self) -> dict[str, Any]:
        total_ns = self.total_wall_ns()
        by_cat_merged = self.wallclock_by_category()
        accounted_ns = sum(by_cat_merged[c] for c in by_cat_merged if c != "idle/other")
        idle_ns = max(0, total_ns - accounted_ns)
        return {
            "problem_id": self.problem_id,
            "model": self.model,
            "arm": self.arm,
            "total_ns": total_ns,
            "total_s": total_ns / 1e9,
            "wallclock_by_category_ns": by_cat_merged,
            "duration_by_category_ns": self.duration_by_category(),
            "estimated_idle_ns": idle_ns,
            "num_spans": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
