"""Generate additional plots for the week-4 blog.

Produces (into results/week4/plots/blog_*.png):
  blog_pass_compile_by_size.png  — dual-axis: compile & pass rate vs. params
  blog_paired_wall.png          — per-model wall-clock A vs B (bar)
  blog_rollback_scaling.png     — avg rollbacks vs params
  blog_profile_share.png        — stacked relative share of wall-clock categories
  blog_arm_winners.png          — per-model arm A vs B delta in pass rate
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soundcode.eval.analyze_week4 import (
    load_summary,
    per_model_arm_summary,
    profile_breakdown,
    MODEL_PARAMS_B,
)

OUT = Path(__file__).parent.parent.parent / "results" / "week4" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

df = load_summary()
summ = per_model_arm_summary(df)
summ["params"] = summ["model"].map(MODEL_PARAMS_B)
summ = summ.dropna(subset=["params"]).sort_values("params")
models_ordered = summ["model"].drop_duplicates().tolist()


# 1. Compile rate + pass rate vs params, per arm
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for arm, marker, color in [("A", "o", "tab:blue"), ("B", "s", "tab:orange")]:
    sub = summ[summ["arm"] == arm].sort_values("params")
    ax1.plot(sub["params"], sub["compile_rate"] * 100, marker=marker, color=color, label=f"Arm {arm}", lw=1.5)
    ax2.plot(sub["params"], sub["pass_rate"] * 100, marker=marker, color=color, label=f"Arm {arm}", lw=1.5)
for ax, title, ylabel in [(ax1, "Compile rate vs. model size", "Compile rate (%)"),
                          (ax2, "Pass@1 vs. model size", "Pass@1 (%)")]:
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "blog_pass_compile_by_size.png")
plt.close()


# 2. Per-model paired wall-clock (arm A vs arm B)
pairs = summ.pivot(index="model", columns="arm", values="median_compile_s").reset_index()
pairs["params"] = pairs["model"].map(MODEL_PARAMS_B)
pairs = pairs.dropna(subset=["params"]).sort_values("params")
x = np.arange(len(pairs))
w = 0.4
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - w / 2, pairs["A"], w, label="Arm A (LSP + compiler)", color="tab:blue")
ax.bar(x + w / 2, pairs["B"], w, label="Arm B (compiler only)", color="tab:orange")
ax.set_xticks(x)
ax.set_xticklabels(pairs["model"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Median wall-clock to first-pass-compile (s)")
ax.set_title("Arm A vs. Arm B — median wall-clock to compile, per model")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT / "blog_paired_wall.png")
plt.close()


# 3. Avg rollbacks vs params
fig, ax = plt.subplots(figsize=(8, 4.5))
for arm, marker, color in [("A", "o", "tab:blue"), ("B", "s", "tab:orange")]:
    sub = summ[summ["arm"] == arm].sort_values("params")
    ax.plot(sub["params"], sub["avg_rollbacks"], marker=marker, color=color, label=f"Arm {arm}", lw=1.5)
ax.set_xscale("log")
ax.set_xlabel("Parameters (B)")
ax.set_ylabel("Average rollbacks per problem")
ax.set_title("Rollback frequency decreases with model capability")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "blog_rollback_scaling.png")
plt.close()


# 4. Profile: share of wall-clock per category, per (model, arm)
categories = ["generation", "prompt-ingestion", "lsp-wait", "cargo-check", "rollback-overhead", "boundary-detect"]
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
labels: list[str] = []
stacks: dict[str, list[float]] = {c: [] for c in categories}
for model in models_ordered:
    for arm in ("A", "B"):
        bd = profile_breakdown(model, arm)
        total = sum(bd.values())
        if total == 0:
            continue
        labels.append(f"{model}\n({arm})")
        for c in categories:
            stacks[c].append(bd.get(c, 0) / total * 100)
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(15, 5.5))
bottom = np.zeros(len(labels))
for c, color in zip(categories, colors):
    ax.bar(x, stacks[c], label=c, bottom=bottom, color=color)
    bottom += np.array(stacks[c])
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Share of wall-clock (%)")
ax.set_title("Where does the time go? (share of wall-clock per category)")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(OUT / "blog_profile_share.png")
plt.close()


# 5. Arm winners: pass-rate delta (A − B) per model
pass_ab = summ.pivot(index="model", columns="arm", values="pass_rate").reset_index()
pass_ab["params"] = pass_ab["model"].map(MODEL_PARAMS_B)
pass_ab = pass_ab.dropna(subset=["params"]).sort_values("params")
pass_ab["delta"] = (pass_ab["A"] - pass_ab["B"]) * 100
fig, ax = plt.subplots(figsize=(13, 4.5))
colors_bar = ["tab:blue" if v > 0 else "tab:orange" for v in pass_ab["delta"]]
ax.bar(np.arange(len(pass_ab)), pass_ab["delta"], color=colors_bar)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(np.arange(len(pass_ab)))
ax.set_xticklabels(pass_ab["model"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Pass@1 (Arm A) − Pass@1 (Arm B), pp")
ax.set_title("Per-model pass-rate delta: positive = Arm A wins, negative = Arm B wins")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT / "blog_arm_winners.png")
plt.close()

print("Blog plots written to", OUT)
