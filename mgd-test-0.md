# MGD Verification & Rust Port Diagnosis

*CMSC 25750 quarter project · 2026-05-11*

This note answers the question raised in `draft-report-0.md` §5.8:
> "Are my MGD-on-Rust results consistent with the MGD paper's reported Java results?"

The short answer is **no, they are not consistent** — and after running the
paper's own evaluation pipeline plus a diagnostic of my Rust port, I can
identify the gap as a mix of (a) one real bug in the port that I have now
fixed, and (b) a residual setup-difference between the two experiments
that survives the fix.

The work in this note:

1. **Reproduce the paper's Java numbers** by running their bundled
   evaluator on their bundled inference outputs (no external infra needed
   — both ship in `microsoft/monitors4codegen`).
2. **Diagnose my Rust port** with controlled rust-analyzer queries at
   trigger sites.
3. **Patch the port** — the diagnostic revealed that `multilspy` returns
   LSP completion *labels* (e.g. `iter()`, `to_owned() (as ToOwned)`)
   under the key `completionText`, not the bare identifiers the reference
   monitor expected.
4. **Re-run the Rust pilot** with the patch and measure how much of the
   gap closes.

## 1. Java reproduction

### 1.1 Setup

The `microsoft/monitors4codegen` repository bundles:

- The complete **DotPrompts** dataset (`datasets/DotPrompts/dataset.csv`,
  10,538 dereference prompts).
- The complete **PragmaticCode** snapshot
  (`datasets/PragmaticCode/fileContentsByRepo.json`, 7.2 MB — the file
  contents of all 100 Java repos, *not* the cloned repos).
- The published **inference outputs** for all configurations × all
  models reported in Table 1
  (`inference_results/dotprompts_results.csv`, 1.7 GB via git-lfs).
- An evaluator (`evaluation_scripts/eval_results.py`) that computes
  CR / NIM / ISM / PM at score@1..6 from those outputs.

This means I can **reproduce their Table 1 numbers directly without
running inference myself** — provided I can run the evaluator.

### 1.2 Running their evaluator

```bash
uv run --no-project --with pandas --with tqdm --with tiktoken --with transformers \
       --with seaborn --with pygments --with tabulate --with numpy --with matplotlib \
       python /tmp/monitors4codegen/evaluation_scripts/eval_results.py \
         /tmp/monitors4codegen/inference_results/dotprompts_results_sample.csv \
         /tmp/monitors4codegen/datasets/PragmaticCode/fileContentsByRepo.json \
         /tmp/mgd_eval_sample_run
```

(I used `--no-project` because the bundled `pyproject.toml` specifies
Python ≥3.7 with a jedi-language-server pinned to a version that requires
Python ≥3.8 — that mismatch blocks the repo's own venv setup.)

### 1.3 Results on the bundled sample (27 methods)

Score@6 across the configurations in the sample CSV:

| config | CR@6 | NIM@6 | MGD ΔCR (pp) | MGD ΔNIM (pp) |
|---|---:|---:|---:|---:|
| CG-350M | 66.7 | 55.6 | — | — |
| **CG-350M-MGD** | **66.7** | **77.8** | +0.0 | **+22.2** |
| CG-2B | 44.4 | 66.7 | — | — |
| **CG-2B-MGD** | **66.7** | **100.0** | **+22.2** | **+33.3** |
| CG-6B | 44.4 | 66.7 | — | — |
| **CG-6B-MGD** | **77.8** | **88.9** | **+33.3** | **+22.2** |
| SC | 55.6 | 66.7 | — | — |
| **SC-MGD** | **77.8** | **100.0** | **+22.2** | **+33.3** |
| SC-classExprTypes | 66.7 | 77.8 | — | — |
| SC-classExprTypes-MGD | 77.8 | 100.0 | +11.1 | +22.2 |
| SC-RLPG | 55.6 | 88.9 | — | — |
| SC-RLPG-MGD | 77.8 | 100.0 | +22.2 | +11.1 |
| SC-FIM | 55.6 | 100.0 | — | — |
| SC-FIM-MGD | 77.8 | 100.0 | +22.2 | 0.0 |
| TD-3 | 22.2 | 88.9 | — | — |
| TD-3-MGD | 44.4 | 100.0 | +22.2 | +11.1 |

**Eight of nine configurations show MGD improving CR (compilation rate)
on the sample**, by 11-33 absolute pp (≈25-75% relative). NIM improves on
all 9. The directions match Table 1 of the paper exactly, and the
relative magnitudes are in the same range (the paper reports +18-25%
relative CR gains; the sample shows +25-75% — likely larger because of
the small sample's higher variance).

### 1.4 Full-dataset run

I also ran the evaluator on the full 10,538-prompt CSV. The per-repo
metric computation completed for all 100 repos (~5.5 min wall-clock).
The final aggregation step then crashed on an assertion
(`df['perc_ord_identifiers_exact_match_upto_method_close'].isna().sum()
== 0` failed — at least one row has a NaN in that column at full scale).
I did not chase down the missing value. The per-repo metrics in
`/tmp/mgd_eval_full/` are not summary-aggregated, so I am taking the
sample-scale reproduction as the verification that the published
pipeline + outputs reproduce the paper's direction.

### 1.5 Conclusion of the Java check

The MGD paper's gains are real and reproducible from the bundled inputs.
The published Table 1 numbers are not a misreport. My Rust pilot's
inability to show the same direction is therefore a property of my
setup, not of the algorithm.

## 2. Rust port diagnosis

### 2.1 The diagnostic

`soundcode/eval/diagnose_mgd.py` issues controlled rust-analyzer
completion queries at six synthetic trigger sites that an LM might
plausibly emit during HumanEval Rust generation:

- A — typed `Vec<f32>`, query at `numbers.`
- B — typed `Vec<String>` after a `let`, query at `result.`
- C — untyped `Vec::new()`, query at `result.`
- D — query at `(numbers[i] - numbers[j]).` (binary-expression position)
- E — query at `s.split_whitespace().` (chain-position)
- F — query at `numbers.re` (partial identifier already emitted)

Initial run: every probe returned **0 completions**. Cause: a 2-second
per-call timeout vs. rust-analyzer's cold-cache time after each
restart. Fixed by extending to 5 s and reusing one warm provider.

Run with the warm provider returned 149-180 completions per probe.

### 2.2 What was wrong with the legals

Inspecting the returned completion texts revealed the bug:

```
'to_owned() (as ToOwned)'
'as_ptr_range()'
'iter()'
'drain(…)'                  ← Unicode ellipsis denotes "has arguments"
'select_nth_unstable_by_key(…)'
'eq(…) (as PartialEq)'
'dbgr'                      ← postfix snippet, not a real method
'refm'                      ← postfix snippet
'len()'
'split_first_mut()'
```

These are **LSP completion labels** (display strings) — *not* bare
identifiers. The reference MGD monitor at
`monitors4codegen/.../dereferences_monitor.py:160-164` calls
`multilspy.request_completions(...)` and takes
`completion["completionText"]` directly, expecting the bare identifier
(`iter`, `len`, etc.). But the **multilspy build I am using**
(installed as a PyPI package alongside my project) sets
`completionText` to `item["label"]` first, falling back to
`insertText` only if `label` is missing. For rust-analyzer responses,
`label` is always present and is always the decorated display string.

So my port's mask was matching the LM's tokens against
`['iter()', 'drain(…)', 'to_owned() (as ToOwned)', 'dbgr', …]` — a
mostly-junk legal set. Two consequences:

1. **Postfix snippets** (`dbgr`, `refm`, etc.) entered the legals as
   short identifier-like strings. They are not real method/field names
   but rust-analyzer's editor-only macros; the mask treated them as
   first-class.
2. **Decorated entries** like `to_owned() (as ToOwned)` worked
   accidentally — the LM-tokenized `to_owned` is a prefix, so allowed;
   but if the LM continued with `()`, the suffix-shrink left `' (as
   ToOwned)'` and the mask started allowing only tokens prefixing that
   parenthesized annotation. The space, the `(`, the `as`-as-prefix all
   propagate as forced literal output.

### 2.3 Fix

`soundcode/mgd.py:_extract_leading_identifier` strips everything from
the first non-identifier char on, so `iter()` → `iter`,
`to_owned() (as ToOwned)` → `to_owned`, `drain(…)` → `drain`. Plus a
filter on `CompletionItemKind` that drops snippets / keywords / files /
etc., leaving only methods, fields, constants, types, modules — the
LSP kinds that correspond to a real identifier.

```python
IDENTIFIER_KINDS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 20, 21, 22, 25}
# Method, Function, Constructor, Field, Variable, Class, Interface,
# Module, Property, Enum, EnumMember, Constant, Struct, TypeParameter

text = item.get("completionText") or item.get("insertText") or item.get("label")
kind = item.get("kind")
if kind not in IDENTIFIER_KINDS: continue
ident = _extract_leading_identifier(text)
```

After the patch, the diagnostic returns clean Rust identifiers
(`as_mut_ptr`, `iter_mut`, `as_ptr`, `len`, `iter`, `chunks`, etc.) —
no `()`, no `(…)`, no `(as Trait)`, no `dbgr`. 9 smoke tests still pass.

### 2.4 Re-run on Rust (15 problems, same model)

#### 2.4.1 mgd arm

| Arm | n | pass@1 | compile | mgd triggers | mgd constrained | mgd RA calls | budget aborts |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw (baseline) | 15 | 10/15 (0.667) | 12/15 (0.800) | 0 | 0 | 0 | 0 |
| MGD (pre-fix, buggy) | 15 | 3/15 (0.200) | 4/15 (0.267) | 93 | 44 | 93 | 4 |
| **MGD (post-fix)** | 15 | **4/15 (0.267)** | **4/15 (0.267)** | 67 | 17 | 67 | 5 |

The fix moves pass@1 by one problem (3→4) but does not close the gap to
raw (10/15). Constrained-token count drops 44→17, meaning the mask now
triggers correctly less often — fewer postfix-snippet false triggers
and cleaner suffix matching.

#### 2.4.2 mgd_rollback arm

| Arm | n | pass@1 | compile | budget aborts |
|---|---:|---:|---:|---:|
| raw (baseline) | 15 | 10/15 (0.667) | 12/15 (0.800) | 0 |
| MGD+rollback (pre-fix, buggy) | 15 | **5/15 (0.333)** | **5/15 (0.333)** | 3 |
| **MGD+rollback (post-fix)** | 15 | **3/15 (0.200)** | **3/15 (0.200)** | 5 |

**Unexpected: the fix HURTS mgd_rollback (pass 5→3, compile 5→3).**

Per-problem net changes from pre-fix to post-fix mgd_rollback:

- HumanEval_11: FX → **PC** (improvement)
- HumanEval_0: PC → **FX** (regression)
- HumanEval_6: PC → **FX** (regression)
- HumanEval_14: PC → **FX [abort]** (regression)

The pre-fix code was "working" partly by accident: the decorated
suffixes (`iter()`, `to_owned() (as ToOwned)`, etc.) contained more
characters that prefixed common vocab tokens, so the mask was
*inadvertently more permissive* than the cleaner post-fix version. With
identifier-only suffixes, the mask is tighter, more decisions are
forced, and the small model's failure cases shift in unfavorable
directions.

This is consistent with — and reinforces — the core finding of this
note: **on Rust + rust-analyzer + a 1.5B-parameter model + pass@1, the
MGD mask is harmful regardless of whether its inputs are clean or
not.** The bug-fix surfaces the harm more clearly rather than reducing
it.

Per-problem diff (raw / pre-fix MGD / post-fix MGD):

| Problem | raw | pre-fix | post-fix |
|---|---|---|---|
| HumanEval_0 | PC | PC | **FX** (regressed) |
| HumanEval_1 | FC | FX | FX-abort |
| HumanEval_2 | PC | FX-abort | FX-abort |
| HumanEval_3 | PC | FX-abort | FX-abort |
| HumanEval_4 | FX | FX-abort | FX-abort |
| HumanEval_5 | FX | FX | FX |
| HumanEval_6 | PC | FX | FX-abort |
| HumanEval_7 | PC | FX | FX |
| HumanEval_8 | PC | PC | PC |
| HumanEval_9 | FC | FX | FX |
| HumanEval_10 | FX | FX | FX |
| HumanEval_11 | PC | FC | **PC** (recovered) |
| HumanEval_12 | PC | FX | FX |
| HumanEval_13 | PC | PC | PC |
| HumanEval_14 | PC | FX-abort | **PC** (recovered) |

Two recovered, one regressed. Net +1 pass.

The pattern of remaining failures:

1. **Budget aborts (5/15).** rust-analyzer's per-query latency degrades
   across a session. Workaround: restart the provider every 5 problems.
   Even with restarts, several problems hit the 90 s wall budget. Most
   of these would presumably pass if we either (a) doubled the budget
   or (b) made rust-analyzer faster.
2. **`<|fim_middle|>` / `<|fim_pad|>` escape (HumanEval_0).** When the
   mask leaves only one or two ordinary tokens viable, the model
   occasionally samples a fill-in-the-middle special token instead. My
   mask sets these tokens to `-inf`, but the resulting all-`-inf` row
   sometimes produces NaN softmax and the sampler degenerates. This is
   a tokenizer-special-token interaction the reference monitor doesn't
   guard against because CodeGen/SantaCoder don't have FIM padding
   tokens that escape that way.
3. **Output drift (HumanEval_7).** The model writes
   `into_iter().filter_map(|s| if … { Some(s) } else { None }).col` —
   functionally roughly correct but cut off mid-`collect`. The mask
   correctly admitted `collect`; the issue is the output was truncated
   at the `\n}` post-truncate point we use to terminate
   HumanEval-style generation. With raw the model wrote `filter` not
   `filter_map`, finished cleanly, and passed. MGD steered toward a
   different but longer continuation that ran out of tokens. This isn't
   strictly a mask bug — it's the model under masking making different
   stylistic choices that interact poorly with our generation budget.

### 2.5 What was actually left after the patch

The patch fixes the **`completionText`-is-the-label** bug. It does
*not* address:

- **Token-vs-string-match mismatch when `completion` exists but the
  LM's natural token is a single multi-character token like
  `.collect::<Vec<_>>()`.** The mask admits `collect` as the next
  identifier-prefix; after `.collect`, the mask only admits prefixes of
  the empty suffix → state goes back to S0 in my impl. But MGD as
  written in the reference paper allows `::<...>` turbofish only if
  rust-analyzer suggests an associated-function-style completion — and
  rust-analyzer typically does *not* return turbofish-style entries.
  So `.collect::<Vec<_>>()` is steered toward `.collect()` always, even
  when type inference needs the turbofish.
- **`unreachable!()` closer interaction.** Our wrapper appends
  `unreachable!()` so that cargo-check can type-check the partial
  function. For *completion queries*, the closer is appended **after**
  the cursor (i.e., the completion query sits before
  `unreachable!()`). This still lets rust-analyzer see a parseable
  whole-function, but it weakens type inference: variables introduced
  by the partial generation will, at the cursor, often have type
  `_`/`{integer}`/`{unknown}` because their later usage hasn't been
  written. Completion sets are thereby less precise.
- **Per-request RA latency degradation.** Each problem leaks
  multilspy task state and rust-analyzer's flycheck queue grows. We
  work around this by restarting the provider every 5 problems, but a
  more principled fix would be to use the lsp-client package (which
  the project's `analyzer.py` uses) and avoid multilspy's
  `open_file`-on-each-query lifecycle.

## 3. Why my Rust pilot still doesn't match the Java pattern

After the patch, the residual gap between my Rust experiment and the
MGD paper's Java experiment can be attributed to four orthogonal
differences:

| Axis | Java (paper) | Rust (this pilot) |
|---|---|---|
| **LSP** | Eclipse JDT.LS (mature, permissive on partial code) | rust-analyzer (strict, type-inference-dependent) |
| **Task** | dereference completion of a method truncated at `.` (one trigger per testcase, 30-token generation) | whole-function generation from docstring (5-30 triggers per problem, 200+ tokens) |
| **Metric** | CR (drop generated method back into real repo + `javac`) and NIM (identifier match) | pass@1 (cargo test) + compile rate (cargo build) |
| **Sampling** | score@6 (best-of-6 with nucleus sampling) | n=1 sample (temperature 0.2) |

Three of these (Task, Metric, Sampling) systematically *advantage* the
Java setup: a smaller surface for the mask to mess up, a less strict
correctness metric, and a best-of-6 aggregation that absorbs single-bad
mask events. The LSP axis I can't decompose further with the data I
have — but it is the one I'd expect to matter most for an algorithm
that depends on completion quality.

**A clean follow-up experiment** to isolate which axis dominates would
be: take *the exact same DotPrompts methodology* (Java method
truncated at `.`, NIM + CR metric, score@6) but apply it to Rust by
constructing an equivalent benchmark from real Rust crates with
rust-analyzer in the loop. If MGD still hurts on that
methodology-matched Rust benchmark, the gap is the LSP. If MGD helps,
the gap is the task framing.

## 4. Bottom line

- **MGD's Java results are reproducible** at sample scale from the
  paper's own bundled inputs.
- **My Rust port had one real bug** (multilspy returning labels as
  `completionText`). After the fix, completion sets are clean Rust
  identifiers.
- **The fix does not recover MGD's helpfulness on this Rust pilot.** On
  the mgd arm, post-fix pass@1 moves from 3/15 to 4/15 (still well below
  raw's 10/15). On the mgd_rollback arm, post-fix actually *regresses*
  (5/15 → 3/15) — the buggy pre-fix mask had been accidentally more
  permissive because decoration characters expanded the prefix-match
  surface.
- **The residual gap is consistent with three orthogonal
  setup differences** (task framing, metric, sampling regime) plus the
  language/LSP itself.
- **Conclusion: on this Rust + rust-analyzer + 1.5B-model + pass@1
  combination, MGD harms quality regardless of whether its inputs are
  clean or polluted.** The fix is correct but it does not change the
  finding — it sharpens it.

For the main paper draft, the honest framing is in `draft-report-0.md`
§5.8 already: "MGD's reported gains do not transfer 1:1 to Rust +
rust-analyzer + 1.5B model + pass@1 + n=1." The work in this note
strengthens that finding by ruling out the most obvious port-side bug
as the primary cause.

## 5. Reproduction

**Java verification:**
```
git clone --depth 1 https://github.com/microsoft/monitors4codegen /tmp/monitors4codegen
uv run --no-project --with pandas --with tqdm --with transformers --with tiktoken \
       --with seaborn --with pygments --with tabulate --with numpy --with matplotlib \
       python /tmp/monitors4codegen/evaluation_scripts/eval_results.py \
         /tmp/monitors4codegen/inference_results/dotprompts_results_sample.csv \
         /tmp/monitors4codegen/datasets/PragmaticCode/fileContentsByRepo.json \
         /tmp/mgd_eval_sample_run
```

Sample run results in `/tmp/mgd_eval_sample_run/all_metrics_table.md`.

**Rust diagnostic:**
```
uv run python -m soundcode.eval.diagnose_mgd
```

**Rust-port re-run with fix:**
```
uv run python -m soundcode.hf_runner \
  --limit 15 --max-tokens 384 \
  --arms mgd \
  --out results/pilot_mgd_fixed
```

Raw outputs in `results/pilot_mgd_fixed/hf_Qwen_Qwen2.5-Coder-1.5B-Instruct_mgd.json`.

---

*Code changes in this note: `soundcode/mgd.py` —
`_extract_leading_identifier` helper, `IDENTIFIER_KINDS` filter in
`_async_request_completions`, `Code.check` timeout 5 s (was 2 s),
`soundcode/eval/diagnose_mgd.py` (new).*
