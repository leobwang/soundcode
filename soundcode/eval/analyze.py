"""Analyze experiment results and produce summary tables."""

from __future__ import annotations

import json
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def load_results() -> list[dict]:
    """Load all result JSON files."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        data["_file"] = f.name
        results.append(data)
    return results


def summary_table(results: list[dict]) -> str:
    """Generate a markdown summary table."""
    lines = []
    lines.append("| Model | Mode | Compiled | Passed | Avg Attempts | Rollbacks | Time |")
    lines.append("|---|---|---|---|---|---|---|")

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    for model, runs in sorted(by_model.items()):
        for r in sorted(runs, key=lambda x: x["mode"]):
            compiled = f"{r['compilation_rate']:.1%}"
            passed = f"{r['pass_rate']:.1%}"
            avg_att = f"{r.get('avg_attempts', 1.0):.2f}"
            rollbacks = str(r.get("total_rollbacks", 0))
            time_s = f"{r['total_time']:.0f}s"
            lines.append(
                f"| {r['model']} | {r['mode']} | {compiled} | {passed} | {avg_att} | {rollbacks} | {time_s} |"
            )

    return "\n".join(lines)


def error_analysis(results: list[dict]) -> str:
    """Analyze common error patterns across models and modes."""
    lines = []

    for r in results:
        if r["mode"] != "baseline":
            continue

        model = r["model"]
        problems = r["problems"]
        total = len(problems)
        compiled = sum(1 for p in problems if p["compiled"])
        failed = [p for p in problems if not p["compiled"]]

        if not failed:
            continue

        # Categorize errors
        error_types: dict[str, int] = {}
        for p in failed:
            output = p.get("compile_output", "")
            for line in output.splitlines():
                line = line.strip()
                if line.startswith("error["):
                    code = line.split("]")[0] + "]"
                    error_types[code] = error_types.get(code, 0) + 1
                elif line.startswith("error:"):
                    msg = line[:60]
                    error_types[msg] = error_types.get(msg, 0) + 1

        lines.append(f"\n### {model} — {total - compiled} compilation failures")
        top_errors = sorted(error_types.items(), key=lambda x: -x[1])[:10]
        for err, count in top_errors:
            lines.append(f"- {err}: {count}")

    return "\n".join(lines)


def improvement_analysis(results: list[dict]) -> str:
    """Analyze per-problem improvements from verification modes."""
    lines = []

    by_model: dict[str, dict[str, dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], {})[r["mode"]] = r

    for model, modes in sorted(by_model.items()):
        if "baseline" not in modes:
            continue
        baseline = {p["name"]: p for p in modes["baseline"]["problems"]}

        for mode_name in ["lsp", "compiler"]:
            if mode_name not in modes:
                continue
            other = {p["name"]: p for p in modes[mode_name]["problems"]}

            fixed = []  # Didn't compile in baseline, compiles now
            broken = []  # Compiled in baseline, doesn't compile now
            for name in baseline:
                b = baseline[name]
                o = other.get(name)
                if o is None:
                    continue
                if not b["compiled"] and o["compiled"]:
                    fixed.append(name)
                elif b["compiled"] and not o["compiled"]:
                    broken.append(name)

            lines.append(f"\n### {model}: {mode_name} vs baseline")
            lines.append(f"- Fixed (now compiles): {len(fixed)}")
            lines.append(f"- Broken (no longer compiles): {len(broken)}")
            lines.append(f"- Net improvement: {len(fixed) - len(broken)}")

    return "\n".join(lines)


def main():
    results = load_results()
    if not results:
        print("No results found in results/")
        return

    print("# Experiment Results\n")
    print("## Summary\n")
    print(summary_table(results))
    print("\n## Error Analysis\n")
    print(error_analysis(results))
    print("\n## Improvement Analysis\n")
    print(improvement_analysis(results))


if __name__ == "__main__":
    main()
