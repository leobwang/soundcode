"""Plot the pilot results: per-model per-arm compile/pass + rollback counts."""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path("results/pilot")
OUT = RESULTS / "plots"
OUT.mkdir(exist_ok=True)

ARMS = ["raw", "rollback_mute", "rollback_instruct"]
ARM_LABELS = {"raw": "raw", "rollback_mute": "rollback (mute)", "rollback_instruct": "rollback (instruct)"}
COLORS = {"raw": "tab:gray", "rollback_mute": "tab:blue", "rollback_instruct": "tab:orange"}


def load_model(model_slug: str) -> dict[str, list[dict]]:
    out = {}
    for arm in ARMS:
        p = RESULTS / f"{model_slug}_{arm}.json"
        if p.exists():
            out[arm] = json.loads(p.read_text())
    return out


def aggregate_pass_compile(rows: list[dict]) -> tuple[float, float, float, int]:
    n = len(rows)
    pas = sum(r["passed"] for r in rows) / n
    com = sum(r["compiled"] for r in rows) / n
    wall = statistics.mean(r["wall_clock_s"] for r in rows)
    rb = sum(r["rollback_count"] for r in rows)
    return pas, com, wall, rb


def main():
    models = []
    if (RESULTS / "mistral-small3.2_24b_raw.json").exists():
        models.append(("mistral-small3.2:24b", "mistral-small3.2_24b"))
    if (RESULTS / "nemotron-3-nano_4b_raw.json").exists():
        models.append(("nemotron-3-nano:4b", "nemotron-3-nano_4b"))
    if (RESULTS / "qwen3.5_9b_raw.json").exists():
        models.append(("qwen3.5:9b", "qwen3.5_9b"))

    n_models = len(models)
    if n_models == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(n_models)
    w = 0.27

    # Plot 1: pass@1
    ax = axes[0]
    for i, arm in enumerate(ARMS):
        vals = []
        for _, slug in models:
            data = load_model(slug)
            if arm in data:
                p, _, _, _ = aggregate_pass_compile(data[arm])
                vals.append(p * 100)
            else:
                vals.append(0)
        ax.bar(x + (i - 1) * w, vals, w, label=ARM_LABELS[arm], color=COLORS[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in models], rotation=15, ha="right")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("pass@1 by arm")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")

    # Plot 2: compile rate
    ax = axes[1]
    for i, arm in enumerate(ARMS):
        vals = []
        for _, slug in models:
            data = load_model(slug)
            if arm in data:
                _, c, _, _ = aggregate_pass_compile(data[arm])
                vals.append(c * 100)
            else:
                vals.append(0)
        ax.bar(x + (i - 1) * w, vals, w, label=ARM_LABELS[arm], color=COLORS[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in models], rotation=15, ha="right")
    ax.set_ylabel("compile rate (%)")
    ax.set_title("compile rate by arm")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")

    # Plot 3: total rollback count
    ax = axes[2]
    for i, arm in enumerate(ARMS):
        vals = []
        for _, slug in models:
            data = load_model(slug)
            if arm in data:
                _, _, _, rb = aggregate_pass_compile(data[arm])
                vals.append(rb)
            else:
                vals.append(0)
        ax.bar(x + (i - 1) * w, vals, w, label=ARM_LABELS[arm], color=COLORS[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in models], rotation=15, ha="right")
    ax.set_ylabel("total rollback events (30 problems)")
    ax.set_title("rollback firing rate by arm")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT / "pilot_summary.png", dpi=120)
    plt.close()
    print(f"Wrote {OUT / 'pilot_summary.png'}")


if __name__ == "__main__":
    main()
