"""Week 4 analysis: load results/week4/*.json and profiles, compute metrics,
test hypotheses H1/H2, produce plots.

Usage:
    uv run python -m soundcode.eval.analyze_week4 [--plots]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "week4"
PROFILES_DIR = RESULTS_DIR / "profiles"
PLOTS_DIR = RESULTS_DIR / "plots"


# Approximate parameter counts per model (B = billions).
# Best-effort from Ollama tag sizes; for H2 plotting only.
MODEL_PARAMS_B = {
    "qwen3.5:0.8b": 0.8,
    "nemotron-3-nano:4b": 4.0,
    "qwen3.5:9b": 9.0,
    "gpt-oss:20b": 20.0,
    "mistral-small3.2:24b": 24.0,
    "devstral-small-2:latest": 24.0,
    "gemma3:27b": 27.0,
    "nemotron-cascade-2:30b": 30.0,
    "qwen2.5-coder:32b": 32.0,
    "qwen3.5:35b": 35.0,
    "qwen3.6:35b": 35.0,
    "deepseek-r1:70b": 70.0,
    "devstral-2:latest": 123.0,  # 123B per Mistral
    "nemotron-3-super:120b": 120.0,
    "gpt-oss:120b": 120.0,
    "qwen3.5:122b": 122.0,
}


def load_summary() -> pd.DataFrame:
    """Load per-(model, arm) summaries into a DataFrame row per problem."""
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        for prob in data["problems"]:
            rows.append({
                "model": data["model"],
                "arm": data["arm"],
                "params_b": MODEL_PARAMS_B.get(data["model"], float("nan")),
                "problem": prob["name"],
                "compiled": prob["compiled"],
                "passed": prob["passed"],
                "wall_s": prob["wall_clock_s"],
                "to_compile_s": prob["wall_clock_to_compile_s"],
                "attempts": prob["num_generation_attempts"],
                "rollbacks": prob["num_rollbacks"],
                "compiler_calls": prob["num_compiler_calls"],
                "tokens": prob["tokens_generated"],
                "discarded": prob["tokens_discarded"],
                "timed_out": prob["timed_out"],
                "error_codes": prob["error_codes_seen"],
            })
    return pd.DataFrame(rows)


def per_model_arm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compile the headline summary table: per (model, arm)."""
    def median_finite(series: pd.Series) -> float:
        finite = series[np.isfinite(series)]
        return float(finite.median()) if len(finite) else float("inf")

    grp = df.groupby(["model", "arm"])
    return grp.agg(
        n=("problem", "count"),
        compile_rate=("compiled", "mean"),
        pass_rate=("passed", "mean"),
        timeout_rate=("timed_out", "mean"),
        median_wall_s=("wall_s", "median"),
        median_compile_s=("to_compile_s", median_finite),
        avg_rollbacks=("rollbacks", "mean"),
        avg_compiler_calls=("compiler_calls", "mean"),
        avg_tokens=("tokens", "mean"),
        avg_discarded=("discarded", "mean"),
    ).reset_index()


def hypothesis_h1(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model paired comparison between arms A and B.

    For each problem compiled by BOTH arms, report:
      - whether arm A is faster (lower wall-clock to first compile)
      - whether arm A uses fewer compiler calls
      - Wilcoxon signed-rank p-value for both (paired, two-sided)
    """
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        wilcoxon = None  # type: ignore

    key_cols = ["model", "problem"]
    a = df[df["arm"] == "A"][key_cols + ["compiled", "to_compile_s", "compiler_calls", "wall_s", "rollbacks"]]
    b = df[df["arm"] == "B"][key_cols + ["compiled", "to_compile_s", "compiler_calls", "wall_s", "rollbacks"]]
    merged = a.merge(b, on=key_cols, suffixes=("_A", "_B"))
    both = merged[(merged["compiled_A"]) & (merged["compiled_B"])].copy()
    both["delta_compile_s"] = both["to_compile_s_A"] - both["to_compile_s_B"]
    both["delta_compiler_calls"] = both["compiler_calls_A"] - both["compiler_calls_B"]
    both["A_faster"] = both["delta_compile_s"] < 0
    both["A_fewer_calls"] = both["delta_compiler_calls"] < 0

    rows = []
    for model, sub in both.groupby("model"):
        p_wall = p_calls = float("nan")
        if wilcoxon is not None and len(sub) >= 3:
            # All-zero deltas are not testable by Wilcoxon
            if (sub["delta_compile_s"] != 0).any():
                try:
                    p_wall = float(wilcoxon(sub["to_compile_s_A"], sub["to_compile_s_B"]).pvalue)
                except Exception:
                    pass
            if (sub["delta_compiler_calls"] != 0).any():
                try:
                    p_calls = float(wilcoxon(sub["compiler_calls_A"], sub["compiler_calls_B"]).pvalue)
                except Exception:
                    pass
        rows.append({
            "model": model,
            "n_paired": len(sub),
            "median_wall_A": float(sub["to_compile_s_A"].median()),
            "median_wall_B": float(sub["to_compile_s_B"].median()),
            "median_calls_A": float(sub["compiler_calls_A"].median()),
            "median_calls_B": float(sub["compiler_calls_B"].median()),
            "A_faster_frac": float(sub["A_faster"].mean()),
            "A_fewer_calls_frac": float(sub["A_fewer_calls"].mean()),
            "wilcoxon_p_wall": p_wall,
            "wilcoxon_p_calls": p_calls,
        })
    return pd.DataFrame(rows)


def error_code_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, arm): distribution of rustc/LSP error codes encountered."""
    from collections import Counter

    rows = []
    for (model, arm), sub in df.groupby(["model", "arm"]):
        codes: Counter = Counter()
        for lst in sub["error_codes"]:
            codes.update(lst)
        # Top 5 by count
        top5 = dict(codes.most_common(5))
        rows.append({
            "model": model,
            "arm": arm,
            "total_errors_recorded": sum(codes.values()),
            "distinct_codes": len(codes),
            "top_codes": ", ".join(f"{c}={n}" for c, n in top5.items()),
        })
    return pd.DataFrame(rows)


def discarded_tokens_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, arm): ratio of discarded tokens to tokens generated.
    A measure of how much work is thrown away by the rollback loop."""
    rows = []
    for (model, arm), sub in df.groupby(["model", "arm"]):
        gen = sub["tokens"].sum()
        disc = sub["discarded"].sum()
        ratio = disc / gen if gen else 0.0
        rows.append({
            "model": model,
            "arm": arm,
            "tokens": int(gen),
            "discarded": int(disc),
            "discard_ratio": ratio,
        })
    return pd.DataFrame(rows)


def hypothesis_h2(df: pd.DataFrame) -> dict[str, float]:
    """H2: compiler calls decreases monotonically with model size; wall-clock does not.

    Compute Spearman correlations between params and (compiler_calls, wall_clock).
    """
    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception:
        return {}

    summ = per_model_arm_summary(df)
    summ["params"] = summ["model"].map(MODEL_PARAMS_B)
    summ = summ.dropna(subset=["params"])

    out: dict[str, float] = {}
    for arm in ("A", "B"):
        sub = summ[summ["arm"] == arm]
        if len(sub) >= 3:
            r, p = spearmanr(sub["params"], sub["avg_compiler_calls"])
            out[f"rho_params_vs_compiler_calls_arm_{arm}"] = float(r)
            out[f"p_params_vs_compiler_calls_arm_{arm}"] = float(p)
            r2, p2 = spearmanr(sub["params"], sub["median_compile_s"].replace([float("inf")], float("nan")))
            out[f"rho_params_vs_wall_arm_{arm}"] = float(r2)
            out[f"p_params_vs_wall_arm_{arm}"] = float(p2)
    return out


def profile_breakdown(model: str, arm: str) -> dict[str, float]:
    """Aggregate wall-clock by category across all profile files for a (model, arm)."""
    safe_model = model.replace(":", "_").replace("/", "_")
    totals: dict[str, float] = {}
    n_profiles = 0
    for f in PROFILES_DIR.glob(f"{safe_model}_{arm}_*.json"):
        data = json.loads(f.read_text())
        for cat, ns in data.get("wallclock_by_category_ns", {}).items():
            totals[cat] = totals.get(cat, 0) + ns / 1e9
        n_profiles += 1
    # Also estimate idle
    return totals


def rollback_overhead_distribution() -> dict[str, list[float]]:
    """Collect all rollback-overhead span durations (seconds) per model (arm A)."""
    by_model: dict[str, list[float]] = {}
    for f in PROFILES_DIR.glob("*_A_*.json"):
        data = json.loads(f.read_text())
        model = data["model"]
        by_model.setdefault(model, [])
        for s in data.get("spans", []):
            if s["category"] == "rollback-overhead":
                by_model[model].append(s["duration_s"])
    return by_model


def make_plots(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = per_model_arm_summary(df)
    if summary.empty:
        print("No data to plot.")
        return

    # 1. Compile rate bar: per-model, per-arm
    fig, ax = plt.subplots(figsize=(9, 4))
    models = sorted(summary["model"].unique(), key=lambda m: MODEL_PARAMS_B.get(m, 1e9))
    x = np.arange(len(models))
    w = 0.35
    a_rates = [summary[(summary["model"] == m) & (summary["arm"] == "A")]["compile_rate"].iloc[0]
               if len(summary[(summary["model"] == m) & (summary["arm"] == "A")]) else 0 for m in models]
    b_rates = [summary[(summary["model"] == m) & (summary["arm"] == "B")]["compile_rate"].iloc[0]
               if len(summary[(summary["model"] == m) & (summary["arm"] == "B")]) else 0 for m in models]
    ax.bar(x - w/2, a_rates, w, label="Arm A (LSP + compiler)")
    ax.bar(x + w/2, b_rates, w, label="Arm B (compiler only)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Compile rate")
    ax.set_title("Compile rate per model, by arm")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "compile_rate.png", dpi=120)
    plt.close()

    # 2. H2 scaling plot: compiler calls vs params, wall-clock vs params
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm, marker in [("A", "o"), ("B", "s")]:
        sub = summary[summary["arm"] == arm].sort_values("model", key=lambda s: s.map(MODEL_PARAMS_B).fillna(1e9))
        params = sub["model"].map(MODEL_PARAMS_B).values
        ax1.plot(params, sub["avg_compiler_calls"], marker=marker, label=f"Arm {arm}")
        ax2.plot(params, sub["median_compile_s"], marker=marker, label=f"Arm {arm}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Params (B)")
    ax1.set_ylabel("Avg compiler calls per problem")
    ax1.set_title("H2: compiler calls vs model size")
    ax1.legend()
    ax2.set_xscale("log")
    ax2.set_xlabel("Params (B)")
    ax2.set_ylabel("Median wall-clock to compile (s)")
    ax2.set_title("H2: wall-clock vs model size")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scaling_h2.png", dpi=120)
    plt.close()

    # 3. Profile stacked bar: per-model, per-arm
    fig, ax = plt.subplots(figsize=(12, 5))
    categories = ["prompt-ingestion", "generation", "lsp-roundtrip", "lsp-wait",
                  "cargo-check", "rollback-overhead", "classifier", "boundary-detect"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    labels = []
    stacks: dict[str, list[float]] = {c: [] for c in categories}
    for model in models:
        for arm in ("A", "B"):
            breakdown = profile_breakdown(model, arm)
            label = f"{model}\n({arm})"
            labels.append(label)
            for c in categories:
                stacks[c].append(breakdown.get(c, 0))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for c, color in zip(categories, colors):
        ax.bar(x, stacks[c], label=c, bottom=bottom, color=color)
        bottom += np.array(stacks[c])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Total wall-clock (s) across all problems")
    ax.set_title("Time breakdown by category, per model × arm")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "profile_breakdown.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 4. Rollback overhead distribution (arm A)
    fig, ax = plt.subplots(figsize=(9, 4))
    by_model = rollback_overhead_distribution()
    for model, vals in sorted(by_model.items(), key=lambda kv: MODEL_PARAMS_B.get(kv[0], 1e9)):
        if vals:
            ax.hist(vals, bins=30, alpha=0.5, label=f"{model} (n={len(vals)})")
    ax.set_xlabel("Rollback-overhead per event (s)")
    ax.set_ylabel("Count")
    ax.set_title("Arm A: rollback-overhead distribution (abort→first-token)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rollback_overhead.png", dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    df = load_summary()
    if df.empty:
        print("No results found under", RESULTS_DIR)
        return
    print("\n=== Per-model, per-arm summary ===")
    summ = per_model_arm_summary(df)
    print(summ.to_string(index=False))
    print("\n=== H1 paired comparison (problems compiled by BOTH arms) ===")
    h1 = hypothesis_h1(df)
    print(h1.to_string(index=False))
    print("\n=== H2: scaling correlations (Spearman) ===")
    h2 = hypothesis_h2(df)
    for k, v in h2.items():
        print(f"  {k}: {v:+.3f}")
    print("\n=== Error code breakdown per (model, arm) ===")
    print(error_code_breakdown(df).to_string(index=False))
    print("\n=== Token discard ratio (discarded / generated) ===")
    print(discarded_tokens_ratio(df).to_string(index=False))

    if args.plots:
        print("\nGenerating plots to", PLOTS_DIR)
        make_plots(df)
        print("Plots saved.")


if __name__ == "__main__":
    main()
