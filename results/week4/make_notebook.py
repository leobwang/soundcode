"""Build analysis.ipynb."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("""# Week 4 Analysis

Compare arm A (LSP + compiler) vs. arm B (compiler-only) across models.
Tests H1(a/b): arm A faster, fewer compiler calls.
Tests H2: partial scaling law — compiler calls drop with size, wall-clock may not."""),
    nbf.v4.new_code_cell("""import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from soundcode.eval.analyze_week4 import (
    load_summary, per_model_arm_summary, hypothesis_h1,
    profile_breakdown, rollback_overhead_distribution, MODEL_PARAMS_B,
)

df = load_summary()
print(f'Problems loaded: {len(df)} rows across {df["model"].nunique()} models, {df["arm"].nunique()} arms')
df.head()"""),
    nbf.v4.new_markdown_cell("## Per-model, per-arm summary"),
    nbf.v4.new_code_cell("""summary = per_model_arm_summary(df)
summary"""),
    nbf.v4.new_markdown_cell("""## H1 paired comparison (problems compiled by both arms)

Reports: median wall-clock to first-pass-compile, median compiler calls,
fraction of paired problems where arm A wins, Wilcoxon signed-rank p-value."""),
    nbf.v4.new_code_cell("""h1 = hypothesis_h1(df)
h1"""),
    nbf.v4.new_markdown_cell("## H2: scaling correlations (Spearman)"),
    nbf.v4.new_code_cell("""from soundcode.eval.analyze_week4 import hypothesis_h2
h2 = hypothesis_h2(df)
for k, v in h2.items():
    print(f'  {k}: {v:+.3f}')"""),
    nbf.v4.new_markdown_cell("## Compile rate per model, by arm"),
    nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 4))
models = sorted(summary['model'].unique(), key=lambda m: MODEL_PARAMS_B.get(m, 1e9))
x = np.arange(len(models))
w = 0.35
for i, arm in enumerate(['A', 'B']):
    rates = [summary[(summary['model']==m) & (summary['arm']==arm)]['compile_rate'].iloc[0]
             if len(summary[(summary['model']==m) & (summary['arm']==arm)]) else 0
             for m in models]
    ax.bar(x + (i - 0.5) * w, rates, w, label=f'Arm {arm}')
ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha='right')
ax.set_ylabel('Compile rate'); ax.set_title('Compile rate per model, by arm')
ax.legend(); plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("## H2: scaling behaviour"),
    nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for arm, marker in [('A', 'o'), ('B', 's')]:
    sub = summary[summary['arm']==arm].copy()
    sub['params'] = sub['model'].map(MODEL_PARAMS_B)
    sub = sub.sort_values('params')
    ax1.plot(sub['params'], sub['avg_compiler_calls'], marker=marker, label=f'Arm {arm}')
    ax2.plot(sub['params'], sub['median_compile_s'], marker=marker, label=f'Arm {arm}')
ax1.set_xscale('log'); ax1.set_xlabel('Params (B)'); ax1.set_ylabel('Avg compiler calls per problem')
ax1.set_title('H2(a): compiler calls vs size'); ax1.legend()
ax2.set_xscale('log'); ax2.set_xlabel('Params (B)'); ax2.set_ylabel('Median wall-clock to compile (s)')
ax2.set_title('H2(b): wall-clock vs size'); ax2.legend()
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("## Profile breakdown"),
    nbf.v4.new_code_cell("""categories = ['prompt-ingestion', 'generation', 'lsp-roundtrip', 'lsp-wait',
              'cargo-check', 'rollback-overhead', 'classifier', 'boundary-detect']
fig, ax = plt.subplots(figsize=(13, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
labels, stacks = [], {c: [] for c in categories}
for model in models:
    for arm in ('A', 'B'):
        bd = profile_breakdown(model, arm)
        if sum(bd.values()) == 0: continue
        labels.append(f'{model}\\n({arm})')
        for c in categories:
            stacks[c].append(bd.get(c, 0))
x = np.arange(len(labels))
bottom = np.zeros(len(labels))
for c, color in zip(categories, colors):
    ax.bar(x, stacks[c], label=c, bottom=bottom, color=color)
    bottom += np.array(stacks[c])
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Total wall-clock (s) across all problems')
ax.set_title('Where does the time go? (per model × arm)')
ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("## Rollback overhead distribution (arm A)"),
    nbf.v4.new_code_cell("""by_model = rollback_overhead_distribution()
fig, ax = plt.subplots(figsize=(10, 4))
for model, vals in sorted(by_model.items(), key=lambda kv: MODEL_PARAMS_B.get(kv[0], 1e9)):
    if vals:
        ax.hist(vals, bins=25, alpha=0.5, label=f'{model} (n={len(vals)})')
ax.set_xlabel('Rollback-overhead per event (s)'); ax.set_ylabel('Count')
ax.set_title('Arm A: rollback-overhead distribution (abort→first-token)')
ax.legend(); plt.tight_layout(); plt.show()
print('Per-model median:')
for model, vals in sorted(by_model.items(), key=lambda kv: MODEL_PARAMS_B.get(kv[0], 1e9)):
    if vals: print(f'  {model}: {np.median(vals):.2f}s (n={len(vals)})')"""),
    nbf.v4.new_markdown_cell("## Error-code distribution"),
    nbf.v4.new_code_cell("""from collections import Counter
for arm in ('A', 'B'):
    print(f'=== Arm {arm} ===')
    for model in models:
        codes = Counter()
        sub = df[(df['model'] == model) & (df['arm'] == arm)]
        for lst in sub['error_codes']:
            codes.update(lst)
        if codes:
            top = ', '.join(f'{c}={n}' for c, n in codes.most_common(6))
            print(f'  {model}: {top}')
        else:
            print(f'  {model}: (no codes)')"""),
    nbf.v4.new_markdown_cell("## Discard ratio (tokens thrown away by rollback)"),
    nbf.v4.new_code_cell("""from soundcode.eval.analyze_week4 import discarded_tokens_ratio
dr = discarded_tokens_ratio(df)
dr"""),
    nbf.v4.new_markdown_cell("""## Per-problem wall-clock distribution

Boxplot per (model, arm). The tails show budget-cap failures (40s cap)."""),
    nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 5))
df2 = df.copy()
df2['label'] = df2['model'] + ' (' + df2['arm'] + ')'
df2 = df2.sort_values('params_b')
labels_order = df2['label'].unique().tolist()
data = [df2[df2['label']==l]['wall_s'].values for l in labels_order]
ax.boxplot(data, labels=labels_order, showmeans=True)
ax.set_xticklabels(labels_order, rotation=30, ha='right')
ax.set_ylabel('Wall-clock per problem (s)')
ax.set_title('Wall-clock distribution per (model, arm)')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Attempts distribution (histogram)"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(len(models), 2, figsize=(11, 2.5 * len(models)), sharex=True, sharey=True)
if len(models) == 1:
    axes = axes.reshape(1, 2)
for i, model in enumerate(models):
    for j, arm in enumerate(['A', 'B']):
        sub = df[(df['model'] == model) & (df['arm'] == arm)]
        axes[i, j].hist(sub['attempts'], bins=20, edgecolor='black')
        axes[i, j].set_title(f'{model} (arm {arm})')
        axes[i, j].set_xlabel('Generation attempts')
        axes[i, j].set_ylabel('# problems')
plt.tight_layout(); plt.show()"""),
]
nb.cells = cells

out_path = Path(__file__).parent / "analysis.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path}")
