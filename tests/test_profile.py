"""Tests for soundcode.eval.profile."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from soundcode.eval.profile import CATEGORIES, Profile, Span


def test_empty_profile() -> None:
    p = Profile(problem_id="p", model="m", arm="A")
    d = p.to_dict()
    assert d["num_spans"] == 0
    # All categories should appear with zero duration
    for cat in CATEGORIES:
        assert d["wallclock_by_category_ns"][cat] == 0


def test_span_records_duration() -> None:
    p = Profile(problem_id="p", model="m", arm="A")
    with p.span("generation"):
        time.sleep(0.01)
    assert len(p.spans) == 1
    s = p.spans[0]
    assert s.category == "generation"
    assert s.duration_ns > 0
    assert s.duration_s >= 0.01


def test_unknown_category_raises() -> None:
    p = Profile(problem_id="p", model="m", arm="A")
    with pytest.raises(ValueError):
        with p.span("not-a-real-category"):
            pass


def test_duration_vs_wallclock_non_overlapping() -> None:
    """For non-overlapping spans within a single category, sum == merged."""
    p = Profile(problem_id="p", model="m", arm="A")
    with p.span("generation"):
        time.sleep(0.005)
    with p.span("generation"):
        time.sleep(0.005)
    dur = p.duration_by_category()["generation"]
    wall = p.wallclock_by_category()["generation"]
    assert abs(dur - wall) < 1_000_000  # within 1 ms


def test_wallclock_merges_overlapping_spans() -> None:
    """Overlapping spans in same category collapse to their union."""
    p = Profile(problem_id="p", model="m", arm="A")
    # Manually construct overlapping spans
    t0 = 1_000_000_000
    p.add_span("generation", t_start_ns=t0, t_end_ns=t0 + 1_000_000_000)
    p.add_span("generation", t_start_ns=t0 + 500_000_000, t_end_ns=t0 + 1_500_000_000)
    dur = p.duration_by_category()["generation"]
    wall = p.wallclock_by_category()["generation"]
    # Sum = 2s, merged = 1.5s
    assert dur == 2_000_000_000
    assert wall == 1_500_000_000


def test_rollback_overhead_list() -> None:
    p = Profile(problem_id="p", model="m", arm="A")
    p.add_span("rollback-overhead", 0, 100_000_000)
    p.add_span("rollback-overhead", 200_000_000, 500_000_000)
    p.add_span("generation", 0, 50_000_000)
    overhead = p.rollback_overhead_ns()
    assert overhead == [100_000_000, 300_000_000]


def test_save_roundtrip(tmp_path: Path) -> None:
    p = Profile(problem_id="HumanEval_0", model="qwen3.5:9b", arm="A")
    with p.span("generation"):
        time.sleep(0.001)
    with p.span("lsp-roundtrip"):
        time.sleep(0.001)
    with p.span("rollback-overhead"):
        time.sleep(0.001)
    out = tmp_path / "profile.json"
    p.save(out)
    loaded = json.loads(out.read_text())
    assert loaded["problem_id"] == "HumanEval_0"
    assert loaded["model"] == "qwen3.5:9b"
    assert loaded["arm"] == "A"
    assert loaded["num_spans"] == 3
    # Each span present
    cats = {s["category"] for s in loaded["spans"]}
    assert cats == {"generation", "lsp-roundtrip", "rollback-overhead"}


def test_meta_preserved() -> None:
    p = Profile(problem_id="p", model="m", arm="A")
    with p.span("cargo-check", invocation=1, code_len=250):
        time.sleep(0.001)
    meta = p.spans[0].meta
    assert meta == {"invocation": 1, "code_len": 250}


def test_idle_estimation() -> None:
    """idle/other should be total - accounted."""
    p = Profile(problem_id="p", model="m", arm="A")
    # Sleep without recording → goes into idle
    time.sleep(0.02)
    with p.span("generation"):
        time.sleep(0.01)
    d = p.to_dict()
    # Idle should be > 0
    assert d["estimated_idle_ns"] > 0
