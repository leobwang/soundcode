# reproduce/

Reproduction harnesses for the two paper baselines we compare SoundCode against:

- **ROCODE** — *ROCODE: Integrating Backtracking Mechanism and Program Analysis in Large Language Models for Code Generation* (arXiv:2411.07112)
- **SemGuard** — semantic-evaluator-guided decoding with rollback

Both baselines target the same problem family as SoundCode (verifier-in-the-loop decoding for code generation), so reproducing them on a common bench is required for a head-to-head comparison in the final report.

## Layout

```
reproduce/
  algo/        per-paper code: vendored upstream + our wrappers
    rocode/
      upstream/         pinned copy of github.com/jiangxxxue/ROCODE
      configs/          our config overrides
      README.md         setup + run notes specific to ROCODE
      MODIFICATIONS.md  log of every change we make to upstream/
      UPSTREAM_COMMIT.txt
    semguard/
      upstream/         pinned copy of github.com/wwwql/SemGuard
                        (includes vendored HF transformers v4.45.2)
      configs/
      README.md
      MODIFICATIONS.md
      UPSTREAM_COMMIT.txt
  bench/       benchmark task definitions (HumanEval, MBPP, etc.) staged for both algos
  data/        extracted SemDiff / SemDiff-Java + other large data artifacts (gitignored)
  model/       custom-tuned weights we produce (e.g. retrained SemGuard evaluators); gitignored
  results/     per-run JSONL logs (gitignored)
  scripts/     top-level dispatcher / sweep runner (added in a later phase)
```

## Conventions

- **Vendor-in, not submodules.** Each paper's upstream code lives at a pinned commit
  under `algo/<paper>/upstream/`. The SHA + commit date are recorded in
  `UPSTREAM_COMMIT.txt`. We do not use git submodules: they complicate environment
  reproduction and the upstream repos are small. Local edits go through
  `MODIFICATIONS.md`.

- **Per-paper environments, isolated from the main project.** The main SoundCode
  project uses `uv`. ROCODE ships a conda `environment.yml`, and SemGuard pip-installs
  a vendored transformers fork. Each paper gets its own dedicated environment
  (see the per-paper README) so dependency conflicts can't bleed into the main project.

- **JSONL logging.** All reproduction runs emit one JSONL file per run into `results/`,
  matching the schema used by the SoundCode web demo
  (`<project_root>/results/web_demo/run_<YYYYMMDD>_<HHMMSS>_<hash>.jsonl`).
  The logging hook is one of the planned `MODIFICATIONS.md` entries for each upstream.

## Per-paper status

| Paper    | Vendored | Env created | Data ready                         | Training done | Ready to run |
|----------|----------|-------------|------------------------------------|---------------|--------------|
| ROCODE   | yes      | no          | upstream `data/` shipped           | n/a           | no           |
| SemGuard | yes      | no          | `.rar` archives bundled, unextracted | no (required) | no           |

### ROCODE
Scaffolded. Conda env `rocode-repro` still needs to be created from
`upstream/environment.yml`. No weights or datasets to download beyond what
upstream ships and what HuggingFace serves at run time.

### SemGuard
Scaffolded. **Retraining is required**: despite the paper's claim that
"we publicly release the SemDiff dataset, trained evaluators, and the complete,
open-source implementation," the GitHub README does **not** link evaluator
checkpoints. The SemDiff dataset *is* bundled (as `.rar` archives), so retraining
the 1.3B classifier per backbone × language is the path forward.
