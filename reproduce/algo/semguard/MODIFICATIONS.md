# SemGuard — local modifications

## Active

Modifications applied to `upstream/` during environment setup (2026-05-24).
All three were required to get `train_base_model.py` to import cleanly under
the per-paper venv. See `SETUP_REPORT.md` for context.

## Planned

- **JSONL logging hook.** Emit one `run_<YYYYMMDD>_<HHMMSS>_<hash>.jsonl` file
  per generation run into `reproduce/results/`, matching the schema used by
  the SoundCode web demo under `<project_root>/results/web_demo/`. Likely
  touches `generate_sem.py` (and `metric/test_one_solution.sh` if we want
  evaluation events in the same stream).

## Convention

When a modification is made, add a dated entry below describing the change and
the file(s) touched. Note: the vendored `upstream/transformers/` fork is part
of the install, but we should avoid editing it — if we need decoding-side
changes, prefer subclassing in `generate_sem.py` over patching transformers.

### Log

#### 2026-05-24 — environment setup patches (claude-code)

**1. `upstream/requirements.txt` — created (file was missing from the pinned commit).**
The upstream README instructs `pip install -r requirements.txt`, but no such file
exists at commit `18dadd75`. Reconstructed from imports observed across
`train_base_model.py`, `generate.py`, `generate_sem.py`, `critic/run.py`,
`metric/test_one_solution.py`, and `Datasets/`. Contents:

```
torch
accelerate
peft
bitsandbytes
deepspeed
datasets
sentencepiece
protobuf
numpy
pandas
scikit-learn
tqdm
pyext
```

(`transformers` is installed separately from the vendored fork.)

**2. `upstream/Datasets/apps_dataset.py` — fixed broken import.**
Replaced

```python
from dataset_lm.reindent import run as run_reindent
```

with

```python
from Datasets.reindent import run as run_reindent
```

The `dataset_lm/` package does not exist in the repo; `reindent.py` lives in
`Datasets/`. The pinned commit ships this import broken — `train_base_model.py`
fails to import without the fix.

**3. `upstream/transformers/src/transformers/generation/utils.py` — removed
unresolvable top-level imports.**

Upstream's added lines (1× `sys`, 3× `Transformers`, 1× `Models`) at the top of
this file are unresolvable as-is and cause a circular import even when rewritten
to lowercase `transformers`. Original:

```python
import sys

from Transformers.src.transformers.models.auto import AutoTokenizer, AutoConfig
from Transformers.src.transformers import RobertaForSequenceClassification

sys.path.append('../../../..')

from Models.models import CodeT5ClassBCE

from Transformers.src.transformers.models.roberta.tokenization_roberta import RobertaTokenizer
```

Replaced with a comment block explaining the situation. The four symbols
(`AutoTokenizer`, `AutoConfig`, `RobertaForSequenceClassification`,
`RobertaTokenizer`, `CodeT5ClassBCE`) are only referenced by the
semantic-rollback path further down the file (around lines 3018 and 3091), where
they're already imported lazily in the function body — so the top-level imports
were redundant in addition to being broken. **Note:** `CodeT5ClassBCE` is also
present in `upstream/critic/models.py`; before we exercise the
semantic-rollback path at generation time, we'll likely need to point that
lazy import at `critic.models.CodeT5ClassBCE` or write a small shim.

These patches are intentionally minimal — they unblock training-script import
and leave the generation path untouched. We expect to revisit (3) when
`generate_sem.py` is actually run.

#### 2026-05-24 — evaluator-training patches (claude-code)

Needed to actually launch `critic/run.py` end-to-end. Three changes inside
`upstream/critic/`:

**4. `critic/models.py` — replaced the `'TODO'` placeholder path prefix.**
`build_or_load_gen_model()` prepends `local_dir + args.model_name_or_path`
before calling `from_pretrained(...)`. Upstream ships `local_dir = 'TODO'`,
which then resolves to a non-existent directory like `TODOdeepseek-coder-...`.
Changed to `local_dir = ''` so callers can pass an absolute or HF-hub path as
`args.model_name_or_path`.

**5. `critic/run.py` — fixed `model_type` lookup and added env-var hook for
backbone path.**
The DeepSeek branch (around line 269) did
```python
elif 'deepseek' in args.output_dir.lower():
    args.model_type = args.load          # e.g. 'deepseek-coder-1.3b-base'
    args.model_name_or_path = 'deepseek-coder-1.3b-base'
```
but `MODEL_CLASSES` in `models.py` only keys on the literal `'deepseek'` —
the assignment would raise `KeyError`. Fixed to `args.model_type = 'deepseek'`,
and added an `os.environ.get('SEMGUARD_BACKBONE', ...)` override so we can
point at a local HF download (`~/hf-models/deepseek-coder-1.3b-base`) without
editing the file each run.

**6. `critic/run.py` — fixed broken `cache_path` initialization at entry point.**
Upstream had
```python
args.cache_path = ''
os.makedirs(args.cache_path, exist_ok=True)   # FileNotFoundError on ''
```
right after `parse_args()`, which negates the `--cache_path` CLI flag and then
crashes on the `makedirs`. Replaced with: honour `--cache_path` when given,
otherwise default to `<output_dir>/cache`. `data_utils.DatasetCA` writes
`{cache_path}/{task}_wordsize_*_rank_*.exps` here.

These three patches let the training cell launch cleanly. The headline cell —
`deepseek-coder-1.3b × SemDiff (Python)` — is verified producing loss in the
log; see `RUN_REPORT.md` for the PID and log path.

#### 2026-05-24 — validation-loader bug + resume-from-checkpoint (claude-code)

The 2026-05-24_115641 run completed epoch 1 (10,113 steps, loss 0.69 → 0.68)
and saved `checkpoinss/0/model_0.bin`, then crashed in `validation_multigpu`
with `TypeError: object of type 'NoneType' has no len()` at run.py:83.

**7. `critic/run.py` — construct validation_set / testing_set on demand.**
Upstream's `main()` hardcoded `validation_set = testing_set = None` in every
branch, so `validation_loader` was always `None` even when `--do_eval=1`.
`DatasetCA` already accepts `data_type='valid'` and `data_type='test'`; the
fix is just to build them when the corresponding flag is set:

```python
validation_set = DatasetCA(args, tokenizer, MAX_LEN, data_type='valid') if args.do_eval else None
testing_set    = DatasetCA(args, tokenizer, MAX_LEN, data_type='test')  if args.do_test else None
```

The `valid.json` / `test.json` files at `data/SemDiff/{valid,test}.json` are
already in the schema `DatasetCA` expects (they have a `text` column and
`pos_code`/`neg_code` columns, so the `if 'pos_code' in self.data.columns:`
branch fires and yields 2 × N examples). Dry-run row counts:

- valid: 4742 (= 2 × 2371)
- test:  3582 (= 2 × 1791)

The `valid1/` / `test1/` per-example dirs (the prior agent's hypothesis) are
**not** consumed by `critic/data_utils.py` at all — only `Datasets/` /
`generate_sem.py` use those. So no symlinking / dir-fabrication was needed.

**8. `critic/run.py` — added `--resume_from_checkpoint` and `--start_epoch`.**
Upstream has no resume flag. Added two CLI args and a load block right after
`accelerator.prepare(model, optimizer)`:

```python
parser.add_argument('--resume_from_checkpoint', default='', type=str, ...)
parser.add_argument('--start_epoch', default=0, type=int, ...)
...
if args.resume_from_checkpoint:
    state_dict = torch.load(args.resume_from_checkpoint, map_location='cpu')
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.load_state_dict(state_dict, strict=False)
```

Also threaded `args.start_epoch` through both training loops so the epoch
counter and `checkpoinss/<epoch>/model_<epoch>.bin` paths line up with the
prior run (resumed run starts at epoch=1, writes `checkpoinss/1/...`,
`checkpoinss/2/...`, etc.):

```python
for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
```

This is the minimum patch needed to resume the 2026-05-24_115641 run from
`checkpoinss/0/model_0.bin` for the remaining 19 epochs.

#### 2026-05-24 — SemDiff data extraction (claude-code)

Extracted `upstream/data/SemDiff.rar` and `upstream/data/SemDiff-Java.rar`
using the official static `unrar` from rarlab (download path:
`https://www.rarlab.com/rar/rarlinux-x64-700.tar.gz`, extracted to
`/tmp/rar/unrar`). No system package install was needed.

The two archives use different internal layouts:
- `SemDiff.rar` extracts into `./data/` (with `train.json`, `valid.json`,
  `test.json`, and per-example `train1/`, `test1/` source dirs).
- `SemDiff-Java.rar` extracts directly into the cwd (`train/`, `test/`, and
  the three JSON files at the root).

Reorganized to a consistent layout that matches the upstream README's
expectations:
- `upstream/data/SemDiff/{train,valid,test}.json` + `train1/`, `test1/`
- `upstream/data/SemDiff-Java/{train,valid,test}.json` + `train/`, `test/`

(Renamed `./data/` → `./SemDiff/` and moved the Java root files into
`./SemDiff-Java/`.) Row counts: 40k/2.4k/1.8k train/valid/test for Python,
39k/2.1k/1.8k for Java. The `critic/data_utils.py:get_data()` reads `train.json`
etc. directly off `args.data_dir`, so pointing `--data_dir` at the new
`SemDiff/` (or `SemDiff-Java/`) folder works without further changes.
