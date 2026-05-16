"""Summarize pilot results into a Markdown table for the report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scipy.stats import wilcoxon


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def aggregate(arm_name: str, rows: list[dict]) -> dict:
    n = len(rows)
    compiled = [r for r in rows if r["compiled"]]
    passed = [r for r in rows if r["passed"]]
    return {
        "arm": arm_name,
        "n": n,
        "pass_rate": len(passed) / n,
        "compile_rate": len(compiled) / n,
        "mean_wall": statistics.mean(r["wall_clock_s"] for r in rows),
        "median_wall": statistics.median(r["wall_clock_s"] for r in rows),
        "mean_tokens_emitted": statistics.mean(r["tokens_emitted"] for r in rows),
        "mean_tokens_kept": statistics.mean(r["tokens_kept"] for r in rows),
        "mean_rollbacks": statistics.mean(r["rollback_count"] for r in rows),
        "total_rollbacks": sum(r["rollback_count"] for r in rows),
        "mean_lsp_calls": statistics.mean(r["lsp_calls"] for r in rows),
        "problems_with_rollback": sum(1 for r in rows if r["rollback_count"] > 0),
    }


def paired_compare(base: list[dict], chal: list[dict], chal_name: str, base_name: str) -> dict:
    """Paired by problem name."""
    bdict = {r["name"]: r for r in base}
    cdict = {r["name"]: r for r in chal}
    common = set(bdict) & set(cdict)
    pairs = [(bdict[n], cdict[n]) for n in sorted(common)]
    n = len(pairs)
    # McNemar-style for binary outcomes
    bp = [b["passed"] for b, _ in pairs]
    cp = [c["passed"] for _, c in pairs]
    bc = [b["compiled"] for b, _ in pairs]
    cc = [c["compiled"] for _, c in pairs]
    pass_diff = sum(c - b for b, c in zip(bp, cp))
    comp_diff = sum(c - b for b, c in zip(bc, cc))
    # Wilcoxon paired on wall-clock
    diffs = [c["wall_clock_s"] - b["wall_clock_s"] for b, c in pairs]
    p_wall = float("nan")
    if n >= 5 and any(d != 0 for d in diffs):
        try:
            p_wall = float(wilcoxon(diffs).pvalue)
        except Exception:
            pass
    return {
        "comparison": f"{chal_name} vs {base_name}",
        "n_pairs": n,
        "delta_pass_pp": 100 * pass_diff / n if n else 0,
        "delta_compile_pp": 100 * comp_diff / n if n else 0,
        "mean_delta_wall_s": statistics.mean(diffs) if diffs else 0,
        "wilcoxon_p_wall": p_wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/pilot")
    ap.add_argument("--model", default="mistral-small3.2_24b")
    args = ap.parse_args()

    root = Path(args.dir)
    arms = ["raw", "rollback_mute", "rollback_instruct"]
    data = {}
    for arm in arms:
        path = root / f"{args.model}_{arm}.json"
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        data[arm] = load(path)

    print("\n## §4.2 Per-arm aggregate\n")
    print("| Arm | n | pass@1 | compile | mean wall (s) | median wall | mean rollbacks | total rollbacks | problems w/ rollback | mean cargo-check calls | mean tokens emitted | mean tokens kept |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        if arm not in data: continue
        a = aggregate(arm, data[arm])
        print(
            f"| {arm} | {a['n']} | {a['pass_rate']:.3f} | {a['compile_rate']:.3f} | "
            f"{a['mean_wall']:.2f} | {a['median_wall']:.2f} | {a['mean_rollbacks']:.2f} | "
            f"{a['total_rollbacks']} | {a['problems_with_rollback']} | "
            f"{a['mean_lsp_calls']:.2f} | {a['mean_tokens_emitted']:.1f} | {a['mean_tokens_kept']:.1f} |"
        )

    print("\n## §4.2b Paired comparisons (matched by problem)\n")
    print("| Comparison | n pairs | Δ pass (pp) | Δ compile (pp) | mean Δ wall (s) | Wilcoxon p (wall) |")
    print("|---|---:|---:|---:|---:|---:|")
    for chal in ("rollback_mute", "rollback_instruct"):
        if chal not in data or "raw" not in data: continue
        c = paired_compare(data["raw"], data[chal], chal, "raw")
        wp = f"{c['wilcoxon_p_wall']:.4f}" if c['wilcoxon_p_wall'] == c['wilcoxon_p_wall'] else "n/a"
        print(
            f"| {c['comparison']} | {c['n_pairs']} | "
            f"{c['delta_pass_pp']:+.1f} | {c['delta_compile_pp']:+.1f} | "
            f"{c['mean_delta_wall_s']:+.2f} | {wp} |"
        )

    # Also paired: instruct vs mute
    if "rollback_mute" in data and "rollback_instruct" in data:
        c = paired_compare(data["rollback_mute"], data["rollback_instruct"],
                           "rollback_instruct", "rollback_mute")
        wp = f"{c['wilcoxon_p_wall']:.4f}" if c['wilcoxon_p_wall'] == c['wilcoxon_p_wall'] else "n/a"
        print(
            f"| {c['comparison']} | {c['n_pairs']} | "
            f"{c['delta_pass_pp']:+.1f} | {c['delta_compile_pp']:+.1f} | "
            f"{c['mean_delta_wall_s']:+.2f} | {wp} |"
        )

    print("\n## §4.3 Per-problem (P/C/rb/wall_s)\n")
    by_prob: dict[str, dict[str, dict]] = {}
    for arm in arms:
        if arm not in data: continue
        for r in data[arm]:
            by_prob.setdefault(r["name"], {})[arm] = r

    print("| Problem | raw | rollback_mute | rollback_instruct |")
    print("|---|---|---|---|")
    for name in sorted(by_prob):
        row = [name]
        for arm in arms:
            if arm not in data:
                row.append("—")
                continue
            r = by_prob[name].get(arm)
            if not r:
                row.append("—")
            else:
                p = "P" if r["passed"] else "F"
                c = "C" if r["compiled"] else "X"
                row.append(
                    f"{p}{c} rb={r['rollback_count']} {r['wall_clock_s']:.1f}s"
                )
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
