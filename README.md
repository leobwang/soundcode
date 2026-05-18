# SoundCode

**LSP-supervised LLM code generation.** The model writes Rust one token at a
time; a "judge" (cargo, or rust-analyzer) runs in parallel at every
structural checkpoint and forces a restart from the last surviving
checkpoint when it sees a blocking error. The result is code that the
compiler actually accepts, even from small LMs that would otherwise produce
nonsense.

> ## 🎛️ Live interactive demo: **<https://soundcode-demo.psixyzt.com>**
>
> 15 models from 4B to 125B, four orchestration modes, two verifiers
> (cargo check and rust-analyzer), and a 3-pane view of the LLM stream,
> the code buffer, and the verifier log. Click any LSP entry to see the
> code state at that moment in the run. Watch tokens fly between modules
> in real time and watch rollbacks strike out failed code segments.

UChicago CMSC 25750 quarter project (week 5). The framework, the
orchestration modes, and the demo were all built incrementally over the
course of the quarter.

---

## What's in the box

### The framework
Three modules in a tight feedback loop:

| Module | File | Role |
|---|---|---|
| **LlmServer** | `soundcode/llm.py` | streams tokens from Ollama; exposes `.next() → Token`, `.set_prompt(p)`, `.abort_current_stream()`; supports four orchestration modes for thinking models |
| **Code** | `soundcode/code.py` | append-only buffer with `at_boundary()`, `check() → CheckResult`, `rollback(offset)`; auto-checkpoints at every `;` / `}` boundary; tracks `body_closed` and runs a repetition watchdog |
| **Verifier** | `soundcode/cargo_check.py`, `soundcode/ra_check.py` | invokes `cargo check` (~300 ms/call) or `rust-analyzer` over multilspy (~1–3 s/call); classifies each diagnostic into `BLOCKING` (triggers rollback) / `INCOMPLETE` (silent — buffer is mid-edit) / `NON_BLOCKING` (silent — warnings) |

The orchestrator (`DemoClient.generate` in `soundcode/web/demo_client.py`)
runs the producer-consumer loop on top of these primitives, emitting
WebSocket events that the web UI renders live.

### Four orchestration modes

Reasoning models can be coaxed into emitting a chain-of-thought trace
before producing code. None of the obvious approaches is strictly best, so
the demo lets you switch between four:

| Mode | Approach | Latency | Thinking visible | Rollback compat |
|---|---|---|---|---|
| **R0** raw | bare prompt, no thinking | 1× call | none | full |
| **R1** raw + `<think>` injection | append `<think>\n` to the prompt, parse `</think>` boundary | 1× call | per-model fragile (works on 9B, fails on 122B) | full (one-shot) |
| **T2** two-phase | phase 1 = chat-template-wrapped call with `stop=["</think>"]` to harvest thinking; phase 2 = bake trace into the prompt as a `// reasoning` comment block, run raw | 2× calls | reliable | full |
| **C3** chat + sysprompt | `/api/chat` with `think:true` plus a system prompt forbidding markdown / preamble; brace-matched body extractor strips the redeclared signature | 1× call | reliable; `think_char_budget` caps overthink on big models | one-shot then raw |

Default is **T2** on `qwen3.5:122b`.

### The web demo

`soundcode/web/server.py` is an `aiohttp` WebSocket server hosting:

- **Title + abstract** explaining the project.
- **Settings | Prompt** split 50/50 across the full page width — 15 Ollama
  models in three size groups (Micro 0–15B, Small 15–50B, Medium 50–150B),
  21 sample HumanEval-Rust prompts plus the LeetCode 37 Sudoku Solver as
  the default.
- **Pipeline pane** with capsules flying LLM → CODE → LSP per event so
  the orchestration is visible without reading logs.
- **3-pane live view**: LLM stream (tokens flow inline as text), code
  buffer (with cross-out → fade animation on rollback), LSP/verifier log
  (entries click-to-pin with a past-state preview overlay).
- **Blog section below the demo**: animated walkthrough of one rollback
  cycle, a vertical static flowchart, the four-mode card layout, and a
  findings section recapping the design decisions of the project.

Every run writes a JSONL trace to `results/web_demo/run_<ts>_<id>.jsonl`
with every event (tokens, verdicts, rollbacks, code snapshots, mode
metadata) so any run can be replayed and diagnosed.

---

## Quickstart

### Prerequisites

- **Python ≥3.12**, managed by [`uv`](https://docs.astral.sh/uv/).
- **[Ollama](https://ollama.com)** running on `localhost:11434` with at least
  one model pulled (`ollama pull qwen3.5:9b` for a quick start, or
  `qwen3.5:122b` for the default).
- **Rust toolchain** for cargo check. `rustup default stable` is enough.
- *(optional)* `rust-analyzer` on `PATH` if you want to switch the
  verifier from cargo to LSP.

### Run the demo locally

```bash
uv sync                                              # install Python deps
uv run python -m soundcode.web.server \              # serve on 0.0.0.0:8080
  --host 0.0.0.0 --port 8080
```

Open <http://localhost:8080> in a browser. Click **Start** with the
defaults to watch qwen3.5:122b solve LeetCode 37 in two-phase mode with
checkpoint-driven rollback. JSONL traces land in `results/web_demo/`.

### Run the test suite

```bash
uv run pytest tests/test_smoke.py tests/test_demo_client.py   # 52 unit tests
uv run pytest tests/test_web_demo.py                           # 26 playwright tests
                                                               # (needs the server up
                                                               #  at localhost:8080)

# real-LLM gated tests (require Ollama + qwen3.5:9b loaded):
RUN_REAL_LLM=1 uv run pytest tests/test_web_demo.py::test_real_qwen_sudoku_merge_and_click
RUN_REAL_LLM=1 uv run pytest tests/test_demo_client.py::test_real_llm_per_mode
```

### Expose to the internet via Cloudflare Tunnel

The public demo at <https://soundcode-demo.psixyzt.com> runs through a
**Cloudflare Named Tunnel** (`cloudflared`) terminated on a personal GPU
host. No port is exposed; the tunnel is an outbound persistent QUIC
connection. WebSocket pass-through and `Cache-Control: no-store` in the
aiohttp middleware mean the demo behaves as a pure pipe — every request
reaches the origin, no edge caching.

---

## Project layout

```
.
├── soundcode/
│   ├── llm.py              # LlmServer + 4 orchestration modes
│   │                       #   (RAW, RAW_THINK_INJECT, TWO_PHASE,
│   │                       #    CHAT_INSTRUCTED) + ThinkSplitter +
│   │                       #    PostBoundaryStripper + ChatBodyStripper
│   ├── code.py             # Code buffer, ckpt list, body_closed,
│   │                       #   classifier (BLOCKING / INCOMPLETE /
│   │                       #   NON_BLOCKING), repetition watchdog
│   ├── cargo_check.py      # CargoChecker — invoke `cargo check` on a
│   │                       #   workspace snapshot, parse JSON diagnostics
│   ├── ra_check.py         # RustAnalyzerChecker — push diagnostics over
│   │                       #   multilspy, same .check() interface
│   ├── client.py           # CodeClient — non-demo production loop (used
│   │                       #   by the eval runner)
│   ├── mgd.py              # Monitor-Guided Decoding port for Rust
│   │                       #   (week 4 comparison baseline)
│   ├── eval/               # week 4 eval infrastructure (15 models × 2
│   │                       #   arms × 30 problems)
│   └── web/                # demo server + frontend
│       ├── server.py            # aiohttp WS server + per-run JSONL logging
│       ├── demo_client.py       # DemoClient.generate — instrumented
│       │                        #   producer-consumer loop, emits UI events
│       ├── extra_prompts.py     # LeetCode 37 Sudoku Solver custom prompt
│       └── static/
│           ├── index.html       # blog + demo layout
│           ├── style.css        # dark theme, HF-paper-style cards
│           ├── app.js           # WebSocket client, 3-pane renderer,
│           │                    #   token + LSP-entry interaction
│           └── flow-anim.js     # animated walkthrough above the static
│                                #   flow chart
├── tests/
│   ├── test_smoke.py            # 41 tests: Code, classifier, ThinkSplitter,
│   │                            #   PostBoundaryStripper, ChatBodyStripper,
│   │                            #   template registry
│   ├── test_demo_client.py      # 11 tests: producer-loop dispatch per mode,
│   │                            #   rollback survivor rule, empty-body
│   │                            #   continuation, think-budget abort
│   └── test_web_demo.py         # 26 playwright tests + 2 RUN_REAL_LLM
│                                #   gated end-to-end runs
├── blog/                        # weekly project blogs
│   └── blog5.md                 # this week's writeup (matches the live page)
├── results/
│   ├── week4/                   # week 4 eval results (15 models × 2 arms)
│   └── web_demo/                # per-run JSONL traces from the live demo
├── notes/                       # planning + journals
├── lit-rev/                     # literature review summaries
└── pyproject.toml               # uv-managed deps
```

---

## Status — what's implemented

- ✅ Producer-consumer rollback loop with boundary-based checkpointing.
- ✅ Cargo-error and repetition-watchdog rollback triggers.
- ✅ Four orchestration modes for thinking models (R0 / R1 / T2 / C3),
  each with one-shot fallback to raw mode for subsequent attempts.
- ✅ Continuation loop after incomplete final cargo check.
- ✅ Web demo with live token streaming, cross-out rollback animation,
  click-to-pin LSP entries with past-code preview, animated walkthrough.
- ✅ Cloudflare Tunnel deployment with `Cache-Control: no-store` so
  changes ship immediately, no edge caching.
- ✅ JSONL run logging for replay / diagnosis.
- ✅ 78 unit + playwright tests + 6 real-LLM gated tests.

## Roadmap

- Bouncing-detector for fine-grained rollback: when N rollbacks cluster in
  a narrow offset window, retreat further (halve survivor, jump to 0)
  to break the model out of a local minimum.
- Chat-template registry entries for the remaining reasoning families
  (gpt-oss, deepseek-r1, nemotron-cascade-2, nemotron-3 — currently
  only Qwen3.x is registered for T2 mode).
- Benchmark numbers across all four modes on a fixed problem set
  (the per-mode live-LLM tests in this session were sanity probes, not
  measurements).

---

## License & acknowledgements

UChicago CMSC 25750 (Topics in ML Systems), Spring 2026 quarter project
by Leo Wang.

The MGD baseline in `soundcode/mgd.py` is a Rust-targeted port of
[microsoft/monitors4codegen](https://github.com/microsoft/monitors4codegen)
(Agrawal et al., *Monitor-Guided Decoding of Code LMs*, NeurIPS 2023).
LSP wiring uses [multilspy](https://github.com/microsoft/multilspy).
