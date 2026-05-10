"""Analyze the week-5 selective-demotion ablation.

Loads ``results/week5/<policy>/<model>_A.json`` for the three policies and
computes per-(model, policy) metrics, then a summary across models.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).parent.parent.parent
RESULTS_ROOT = PROJECT / "results" / "week5"
PLOTS_DIR = RESULTS_ROOT / "plots"


def load_all() -> pd.DataFrame:
    rows = []
    for pol_dir in sorted(RESULTS_ROOT.glob("*")):
        if not pol_dir.is_dir() or pol_dir.name == "plots":
            continue
        policy = pol_dir.name
        for f in sorted(pol_dir.glob("*_A.json")):
            data = json.loads(f.read_text())
            for prob in data["problems"]:
                rows.append({
                    "policy": policy,
                    "model": data["model"],
                    "problem": prob["name"],
                    "compiled": prob["compiled"],
                    "passed": prob["passed"],
                    "wall_s": prob["wall_clock_s"],
                    "to_compile_s": prob["wall_clock_to_compile_s"],
                    "attempts": prob["num_generation_attempts"],
                    "rollbacks": prob["num_rollbacks"],
                    "lsp_rollbacks": prob.get("num_lsp_rollbacks", 0),
                    "compiler_rollbacks": prob.get("num_compiler_rollbacks", 0),
                    "compiler_calls": prob["num_compiler_calls"],
                    "tokens": prob["tokens_generated"],
                    "discarded": prob["tokens_discarded"],
                    "timed_out": prob["timed_out"],
                    "error_codes": prob.get("error_codes_seen", []),
                    "lsp_blocking_codes": prob.get("lsp_blocking_codes", []),
                })
    return pd.DataFrame(rows)


def per_model_policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    def median_finite(s: pd.Series) -> float:
        f = s[np.isfinite(s)]
        return float(f.median()) if len(f) else float("inf")

    g = df.groupby(["model", "policy"])
    return g.agg(
        n=("problem", "count"),
        compile_rate=("compiled", "mean"),
        pass_rate=("passed", "mean"),
        timeout_rate=("timed_out", "mean"),
        median_to_compile=("to_compile_s", median_finite),
        avg_lsp_rollbacks=("lsp_rollbacks", "mean"),
        avg_compiler_rollbacks=("compiler_rollbacks", "mean"),
        avg_compiler_calls=("compiler_calls", "mean"),
        total_lsp_rollbacks=("lsp_rollbacks", "sum"),
        total_compiler_rollbacks=("compiler_rollbacks", "sum"),
    ).reset_index()


def lsp_blocking_code_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Count which error codes triggered LSP-tier blocking, per (model, policy)."""
    rows = []
    for (model, policy), sub in df.groupby(["model", "policy"]):
        codes = Counter()
        for lst in sub["lsp_blocking_codes"]:
            codes.update(lst)
        rows.append({
            "model": model,
            "policy": policy,
            "total_lsp_blocking_events": sum(codes.values()),
            "top_codes": ", ".join(f"{c}={n}" for c, n in codes.most_common(6)) or "(none)",
        })
    return pd.DataFrame(rows)


def paired_compare(df: pd.DataFrame, baseline: str = "both", challenger: str = "ref_only") -> pd.DataFrame:
    """Per-model paired comparison of two policies on the same problems."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None  # type: ignore

    keys = ["model", "problem"]
    a = df[df["policy"] == baseline][keys + ["compiled", "passed", "to_compile_s", "compiler_calls", "lsp_rollbacks"]]
    b = df[df["policy"] == challenger][keys + ["compiled", "passed", "to_compile_s", "compiler_calls", "lsp_rollbacks"]]
    m = a.merge(b, on=keys, suffixes=(f"_{baseline}", f"_{challenger}"))
    rows = []
    for model, sub in m.groupby("model"):
        n = len(sub)
        compile_d = sub[f"compiled_{challenger}"].astype(int) - sub[f"compiled_{baseline}"].astype(int)
        pass_d = sub[f"passed_{challenger}"].astype(int) - sub[f"passed_{baseline}"].astype(int)
        # paired wall-clock on problems both compiled
        both = sub[(sub[f"compiled_{baseline}"]) & (sub[f"compiled_{challenger}"])]
        p_wall = float("nan")
        if wilcoxon is not None and len(both) >= 3 and (both[f"to_compile_s_{baseline}"] - both[f"to_compile_s_{challenger}"]).any():
            try:
                p_wall = float(wilcoxon(both[f"to_compile_s_{challenger}"], both[f"to_compile_s_{baseline}"]).pvalue)
            except Exception:
                pass
        rows.append({
            "model": model,
            f"compile_{baseline}": float(sub[f"compiled_{baseline}"].mean()),
            f"compile_{challenger}": float(sub[f"compiled_{challenger}"].mean()),
            "Δ_compile_pp": float(compile_d.mean() * 100),
            f"pass_{baseline}": float(sub[f"passed_{baseline}"].mean()),
            f"pass_{challenger}": float(sub[f"passed_{challenger}"].mean()),
            "Δ_pass_pp": float(pass_d.mean() * 100),
            f"lsp_rb_{baseline}": float(sub[f"lsp_rollbacks_{baseline}"].sum()),
            f"lsp_rb_{challenger}": float(sub[f"lsp_rollbacks_{challenger}"].sum()),
            "n_paired_compiled": len(both),
            "wilcoxon_p_wall": p_wall,
        })
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summ = per_model_policy_summary(df)
    models = sorted(summ["model"].unique())
    policies = ["both", "ref_only", "none"]
    palette = {"both": "tab:blue", "ref_only": "tab:orange", "none": "tab:red"}

    # Plot 1: total LSP-tier rollbacks per (model, policy)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(models))
    w = 0.27
    for i, pol in enumerate(policies):
        vals = [
            summ[(summ["model"] == m) & (summ["policy"] == pol)]["total_lsp_rollbacks"].iloc[0]
            if not summ[(summ["model"] == m) & (summ["policy"] == pol)].empty else 0
            for m in models
        ]
        ax.bar(x + (i - 1) * w, vals, w, label=f"{pol}", color=palette[pol])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Total LSP-tier blocking events (sum across problems)")
    ax.set_title("LSP-tier rollback events vs. demotion policy")
    ax.legend(title="policy")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ablation_lsp_blocks.png", dpi=120)
    plt.close()

    # Plot 2: compile rate per (model, policy)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, pol in enumerate(policies):
        vals = [
            summ[(summ["model"] == m) & (summ["policy"] == pol)]["compile_rate"].iloc[0] * 100
            if not summ[(summ["model"] == m) & (summ["policy"] == pol)].empty else 0
            for m in models
        ]
        ax.bar(x + (i - 1) * w, vals, w, label=f"{pol}", color=palette[pol])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Compile rate (%)")
    ax.set_title("Compile rate by demotion policy")
    ax.legend(title="policy")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ablation_compile.png", dpi=120)
    plt.close()

    # Plot 3: pass rate per (model, policy)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, pol in enumerate(policies):
        vals = [
            summ[(summ["model"] == m) & (summ["policy"] == pol)]["pass_rate"].iloc[0] * 100
            if not summ[(summ["model"] == m) & (summ["policy"] == pol)].empty else 0
            for m in models
        ]
        ax.bar(x + (i - 1) * w, vals, w, label=f"{pol}", color=palette[pol])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Pass@1 (%)")
    ax.set_title("Pass@1 by demotion policy")
    ax.legend(title="policy")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ablation_pass.png", dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    df = load_all()
    if df.empty:
        print("No results found under", RESULTS_ROOT)
        return

    summ = per_model_policy_summary(df)
    print("\n=== Per-(model, policy) summary ===")
    print(summ.to_string(index=False))

    print("\n=== LSP-tier blocking-event distribution ===")
    print(lsp_blocking_code_dist(df).to_string(index=False))

    print("\n=== Paired comparison: ref_only vs both ===")
    cmp1 = paired_compare(df, baseline="both", challenger="ref_only")
    print(cmp1.to_string(index=False))

    print("\n=== Paired comparison: none vs both ===")
    cmp2 = paired_compare(df, baseline="both", challenger="none")
    print(cmp2.to_string(index=False))

    if args.plots:
        make_plots(df)
        print(f"\nPlots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
