"""Paper Phase 3 — analysis of the post-deadline experiment battery.

Consumes:
  results/paper_phase1/            original run (run 1)
  results/paper_phase1_run{2..5}/  timing replications (same grid)
  results/paper_new_arms/          entropy2, LSP, Java, resample arms
  results/paper_phase1_1p5b/       Qwen2.5-Coder-1.5B full grid
  results/paper_phase1_mbpp/       MBPP-rs/-cpp full grid

Emits a single JSON blob (results/paper_phase3_analysis.json) with:
  multirun   — per-run per-arm mean walls, sync/async ratios, mean+sd,
               and cross-run pass@1 stability for every arm
  new_arms   — per-arm aggregates + McNemar pairwise vs relevant baselines
  scaling    — 7B vs 1.5B ratio comparison (cost-spectrum prediction)
  mbpp       — per-arm aggregates + McNemar vs plain, paired-win rates

Run: uv run python scripts/paper_phase3_analysis.py
"""

from __future__ import annotations

import json
import statistics
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

CORE_ARMS = ["plain", "sync_naive", "async_naive", "async_entropy"]
LANGS = ["rust", "cpp"]


def load_arm(d: Path, arm_id: str) -> dict[str, dict]:
    recs = {}
    p = d / f"{arm_id}.jsonl"
    if not p.exists():
        return recs
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("skipped"):
            continue
        recs[r["problem_id"]] = r
    return recs


def agg(recs: dict[str, dict]) -> dict:
    if not recs:
        return {"n": 0}
    walls = [r["wall_time_s"] for r in recs.values()]
    return {
        "n": len(recs),
        "pass": sum(r["tests_pass"] for r in recs.values()),
        "compile": sum(r["compile_ok"] for r in recs.values()),
        "timeout": sum(r["timeout"] for r in recs.values()),
        "mean_wall": statistics.mean(walls),
        "median_wall": statistics.median(walls),
        "p90_wall": (statistics.quantiles(walls, n=10)[-1]
                     if len(walls) >= 10 else max(walls)),
        "mean_rb": statistics.mean(r["n_rollbacks"] for r in recs.values()),
        "mean_chk": statistics.mean(
            r.get("n_checker_calls", 0) for r in recs.values()),
        "mean_tok": statistics.mean(r["n_tokens"] for r in recs.values()),
    }


def mcnemar(a: dict[str, dict], b: dict[str, dict], field: str) -> dict:
    ids = sorted(set(a) & set(b))
    oa = sum(1 for i in ids if a[i][field] and not b[i][field])
    ob = sum(1 for i in ids if b[i][field] and not a[i][field])
    n = oa + ob
    p = 1.0 if n == 0 else min(
        1.0, sum(comb(n, k) for k in range(min(oa, ob) + 1)) / 2**n * 2)
    return {"n_paired": len(ids), "only_a": oa, "only_b": ob,
            "p_exact": round(p, 5)}


def paired_win_rate(fast: dict[str, dict], slow: dict[str, dict]) -> dict:
    ids = sorted(set(fast) & set(slow))
    wins = sum(1 for i in ids
               if fast[i]["wall_time_s"] < slow[i]["wall_time_s"])
    ratios = sorted(slow[i]["wall_time_s"] / fast[i]["wall_time_s"]
                    for i in ids if fast[i]["wall_time_s"] > 0)
    return {"n": len(ids), "wins": wins,
            "median_ratio": ratios[len(ratios) // 2] if ratios else None}


def multirun_analysis() -> dict:
    runs = {"run1": R / "paper_phase1"}
    for k in range(2, 6):
        d = R / f"paper_phase1_run{k}"
        if d.exists():
            runs[f"run{k}"] = d
    out: dict = {"runs_found": list(runs)}
    per_run: dict = {}
    ratios: dict[str, list[float]] = {"rust": [], "cpp": []}
    pass_counts: dict[str, dict[str, int]] = {}
    for name, d in runs.items():
        row: dict = {}
        for lang in LANGS:
            arms = {a: load_arm(d, f"{a}_{lang}") for a in CORE_ARMS}
            if not arms["plain"]:
                continue
            row[lang] = {a: agg(recs) for a, recs in arms.items()}
            s, a_ = row[lang]["sync_naive"], row[lang]["async_naive"]
            if s.get("n") and a_.get("n"):
                ratio = s["mean_wall"] / a_["mean_wall"]
                row[lang]["sync_async_ratio"] = round(ratio, 4)
                ratios[lang].append(ratio)
            for arm in CORE_ARMS:
                key = f"{arm}_{lang}"
                pass_counts.setdefault(key, {})[name] = \
                    row[lang][arm].get("pass")
        per_run[name] = row
    out["per_run"] = per_run
    out["ratio_summary"] = {
        lang: {
            "n_runs": len(v),
            "mean": round(statistics.mean(v), 4) if v else None,
            "sd": round(statistics.stdev(v), 4) if len(v) > 1 else None,
            "min": round(min(v), 4) if v else None,
            "max": round(max(v), 4) if v else None,
        } for lang, v in ratios.items()
    }
    out["pass_stability"] = {
        k: {"values": v,
            "identical": len(set(v.values())) == 1}
        for k, v in pass_counts.items()
    }
    return out


def new_arms_analysis() -> dict:
    d = R / "paper_new_arms"
    base = R / "paper_phase1"
    out: dict = {}
    if not d.exists():
        return {"missing": True}
    # entropy2 vs baselines
    for lang in LANGS:
        e2 = load_arm(d, f"async_entropy2_{lang}")
        if not e2:
            continue
        naive = load_arm(base, f"async_naive_{lang}")
        recb = load_arm(base, f"async_entropy_{lang}")
        plain = load_arm(base, f"plain_{lang}")
        out[f"entropy2_{lang}"] = {
            "agg": agg(e2),
            "vs_async_naive_pass": mcnemar(e2, naive, "tests_pass"),
            "vs_async_naive_compile": mcnemar(e2, naive, "compile_ok"),
            "vs_recbackoff_pass": mcnemar(e2, recb, "tests_pass"),
            "vs_plain_pass": mcnemar(e2, plain, "tests_pass"),
        }
    # LSP arms
    for arm in ["async_lsp_rust", "sync_lsp_rust"]:
        recs = load_arm(d, arm)
        if not recs:
            continue
        plain = load_arm(base, "plain_rust")
        out[arm] = {
            "agg": agg(recs),
            "vs_plain_pass": mcnemar(recs, plain, "tests_pass"),
            "problems_with_rollback": sum(
                1 for r in recs.values() if r["n_rollbacks"] > 0),
            "total_rollbacks": sum(
                r["n_rollbacks"] for r in recs.values()),
        }
    if "async_lsp_rust" in out and "sync_lsp_rust" in out:
        a = load_arm(d, "async_lsp_rust")
        s = load_arm(d, "sync_lsp_rust")
        out["lsp_sync_async"] = {
            "ratio_of_means":
                round(agg(s)["mean_wall"] / agg(a)["mean_wall"], 4),
            "paired": paired_win_rate(a, s),
        }
    # Java arms
    jarms = {a: load_arm(d, f"{a}_java")
             for a in ["plain", "sync_naive", "async_naive"]}
    if jarms["plain"]:
        out["java"] = {a: agg(recs) for a, recs in jarms.items()}
        if jarms["sync_naive"] and jarms["async_naive"]:
            out["java"]["sync_async_ratio"] = round(
                agg(jarms["sync_naive"])["mean_wall"]
                / agg(jarms["async_naive"])["mean_wall"], 4)
            out["java"]["paired"] = paired_win_rate(
                jarms["async_naive"], jarms["sync_naive"])
            out["java"]["pass_mcnemar"] = mcnemar(
                jarms["async_naive"], jarms["sync_naive"], "tests_pass")
    # resample baseline
    for lang in LANGS:
        rs = load_arm(d, f"resample_{lang}")
        if not rs:
            continue
        an = load_arm(base, f"async_naive_{lang}")
        plain = load_arm(base, f"plain_{lang}")
        out[f"resample_{lang}"] = {
            "agg": agg(rs),
            "vs_async_naive_pass": mcnemar(rs, an, "tests_pass"),
            "vs_async_naive_compile": mcnemar(rs, an, "compile_ok"),
            "vs_plain_pass": mcnemar(rs, plain, "tests_pass"),
        }
    return out


def grid_analysis(d: Path) -> dict:
    out: dict = {}
    if not d.exists():
        return {"missing": True}
    for lang in LANGS:
        arms = {a: load_arm(d, f"{a}_{lang}") for a in CORE_ARMS}
        if not arms["plain"]:
            continue
        row = {a: agg(recs) for a, recs in arms.items()}
        s, a_ = arms["sync_naive"], arms["async_naive"]
        if s and a_:
            row["sync_async_ratio"] = round(
                agg(s)["mean_wall"] / agg(a_)["mean_wall"], 4)
            row["paired"] = paired_win_rate(a_, s)
            row["pass_mcnemar_sync_async"] = mcnemar(
                a_, s, "tests_pass")
            row["pass_mcnemar_async_plain"] = mcnemar(
                a_, arms["plain"], "tests_pass")
        out[lang] = row
    return out


def main() -> None:
    result = {
        "multirun": multirun_analysis(),
        "new_arms": new_arms_analysis(),
        "grid_1p5b": grid_analysis(R / "paper_phase1_1p5b"),
        "grid_mbpp": grid_analysis(R / "paper_phase1_mbpp"),
    }
    out_path = R / "paper_phase3_analysis.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
