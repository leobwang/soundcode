# SoundCode — LSP-supervised LLM code generation

*Leo Wang · UChicago CMSC 25750 · week 5*

Live demo: <https://soundcode-demo.psixyzt.com>

---

## Abstract

LLMs don't understand the logic behind the code they write. A *judge*
with knowledge of the programming language can help LLMs generate correct
code. We introduce **SoundCode**, an orchestration framework that uses
Language Server Protocol (LSP) tooling to inspect the code in real time,
in parallel, at structural checkpoints while the LLM generates. If the
code passes the check, the judge stays out of the way; if an error is
found, the judge strikes down the code segment back to the last
checkpoint and forces a restart from there.

---

## The interactive demo

The page at the URL above runs an instrumented build of the rollback
generator. Models are organised by size:

- **Micro (0–15B)**: `nemotron-3-nano:4b`, `qwen3.5:9b`
- **Small (15–50B)**: `gpt-oss:20b`, `mistral-small3.2:24b`, `devstral-small-2`, `gemma3:27b`, `nemotron-cascade-2:30b`, `qwen2.5-coder:32b`, `qwen3.5:35b`, `qwen3.6:35b`
- **Medium (50–150B)** *(default `qwen3.5:122b`)*: `deepseek-r1:70b`, `gpt-oss:120b`, `nemotron-3-super:120b`, `qwen3.5:122b`, `devstral-2` (125B)

Orchestration modes (this week's headline addition) are R0/R1/T2/C3 — see
the [Findings](#findings-from-this-session) section for what each does.

---

## How it works

### Code-generation framework

Three modules call each other in a short loop.

```
   LlmServer ──.next() → token → .append()──►  Code  ──if .at_boundary(): .check()──►  Verifier
   (llm.py)                                   (code.py)                                (cargo_check.py
        ▲                                       ▲                                       / ra_check.py)
        │                                       │ CheckResult                                  │
        │                                       │                                              │
        │                              ◄────────┘  BLOCKING → .rollback(ckpt)  ────────────────┤
        │                                                                                      │
        └──── BLOCKING → .abort_current_stream() + .set_prompt(new_prompt) ────────────────────┘
```

Method-level mapping:

| Where | Real method | Role |
|---|---|---|
| `soundcode/llm.py` | `LlmServer.next()` | yield the next streamed `Token` |
| `soundcode/llm.py` | `LlmServer.set_prompt(p)` | reset what the next stream open will send |
| `soundcode/llm.py` | `LlmServer.abort_current_stream()` | close the in-flight HTTP stream |
| `soundcode/code.py` | `Code.append(token)` | extend the buffer; auto-add `;` / `}` checkpoints |
| `soundcode/code.py` | `Code.at_boundary() → bool` | true iff the last token closed a depth-0 boundary |
| `soundcode/code.py` | `Code.check() → CheckResult` | run the verifier on the snapshot; classify verdict |
| `soundcode/code.py` | `Code.rollback(offset)` | truncate `content` to `offset`; prune `ckpt` |
| `soundcode/cargo_check.py` | `CargoChecker.check(source)` | invoke `cargo check` on source; return raw diagnostics |
| `soundcode/ra_check.py` | `RustAnalyzerChecker.check(source)` | same interface, via rust-analyzer over multilspy |

The orchestrator (`DemoClient.generate`) ties them together:

1. Pull the next streamed token from the LLM.
2. Append it to a `Code` buffer that tracks every `;` / `}` boundary in
   the content as a checkpoint offset.
3. If the latest token closed a depth-0 boundary, fire an async
   `check()` against the buffer's snapshot.
4. As verdicts arrive, color the corresponding token (green = OK, red =
   error, grey = inactive/incomplete). On a blocking error, rollback to
   the largest checkpoint strictly before the check's offset.
5. Terminate when the function body closes structurally, or the wall-clock
   budget runs out, or the repetition watchdog catches the model looping.

### What triggers a strikedown

Every diagnostic cargo (or rust-analyzer) emits is sorted into one of three
categories by `Code._classify` in `soundcode/code.py`. Only the first
triggers a rollback; the other two stay silent and let the producer loop
continue.

| Category | Triggers strikedown? | Examples |
|---|---|---|
| **BLOCKING** | ✅ yes — rollback fires | E0308 mismatched types · E0425 cannot find function · E0061 wrong number of arguments · E0277 trait bound not satisfied · "expected X, found Y" type errors · any cargo error not flagged as INCOMPLETE below |
| **INCOMPLETE** | ❌ silent — buffer is mid-edit, not broken | syntax-error · E0601 main not found · E0282/E0283/E0284/E0698 type annotations needed (inference can't proceed yet) · messages containing "expected", "unterminated", "unclosed", "missing trailing", "unexpected eof" |
| **NON_BLOCKING** | ❌ silent — warning, not an error | unused_variables · dead_code · clippy lints |

The classifier is deliberately conservative on INCOMPLETE — it treats partial
code as a "wait for more tokens" signal rather than a failure signal, because
aborting on every transient parse error would prevent the model from ever
finishing a statement.

### Why LeetCode 37 (Sudoku Solver) is the default prompt

The signature `fn solve_sudoku(board: &mut Vec<Vec<char>>)` is short but a
working body is ~1.5 KB of Rust — long enough to cross many checkpoints
and actually exercise rollback. It also reliably trips the failure modes
the pipeline cares about: *E0308* char-vs-int mismatches, *E0594* on the
mutable borrow, and *E0434* when the model tries nested `fn` helpers that
close over locals. The full LeetCode prompt (~1.7 KB with constraints and
an example board) is itself a decent stress test for context handling on
larger models.

---

## Four modes of thinking separation

Reasoning models (Qwen 3.x, gpt-oss, deepseek-r1, nemotron) can be coaxed
into emitting a chain-of-thought trace before producing code. There's no
single right way to do it — every approach trades off *latency*, *boundary
reliability*, and *rollback compatibility* differently. SoundCode supports
four; switch live from the **Orchestration** dropdown in the demo.

### R0 — raw, no thinking

Bare prompt to `/api/generate` with `raw: true`. Single call. The model
just continues the prompt literally; no thinking trace appears. Always
works on every model in the dropdown.

| Latency | Thinking | Rollback | Reliability |
|---|---|---|---|
| 1× call | none | standard literal-continuation | always |

### R1 — raw + `<think>` injection

Append `<think>\n` to the outgoing prompt. Parse the stream for a literal
`</think>` tag or a markdown fence as the think→code boundary. After the
boundary, `PostBoundaryStripper` detects and strips any redeclared
signature so only the function body reaches the code buffer.

| Latency | Thinking | Rollback | Reliability |
|---|---|---|---|
| 1× call | visible (when model closes the tag) | one-shot then raw | per-model — qwen3.5:9b yes, 122b no |

### T2 — two-phase: think → code  *(default)*

Phase 1: a separate chat-template-wrapped call with `stop=["</think>"]`
harvests a reliable thinking trace. Phase 2: bake the trace into the
prompt as a Rust-comment block, then run the existing raw producer-
consumer loop. Thinking actually conditions the code that follows;
rollback works unchanged because phase 2 is plain raw mode.

| Latency | Thinking | Rollback | Reliability |
|---|---|---|---|
| 2× calls (one extra cold-start) | reliable (cleanly stops at `</think>`) | standard (phase 2 is literal raw) | gold path |

### C3 — chat + code-only instruction

`/api/chat` with `think: true` plus a system prompt forbidding markdown,
preamble, and the closing brace. Ollama splits `thinking` and `content`
fields per chunk server-side. `ChatBodyStripper` brace-matches the
`content` to extract only the body — the model still re-declares the
signature.

| Latency | Thinking | Rollback | Reliability |
|---|---|---|---|
| 1× call | reliable (Ollama splits channels) | one-shot then raw | high; large models overthink → `think_char_budget` |

---

## Findings from this session

### 1. Boundary-based checkpointing peels rollbacks one statement at a time

The original policy ("checkpoint only at cargo-verified positions") caused
rollbacks to leap back many lines. The new policy ("checkpoint at every `;`
and `}` boundary in the buffer, regardless of verification") lets a failing
run retreat by ~one statement per rollback.

It only works because `_ckpt_scan_floor` is **non-monotonic**: it resets to
the current rollback target rather than tracking the historical max. With a
monotonic floor (the initial implementation), regenerated content past a
lower target gets its new boundaries silently swallowed by the auto-ckpt
scan — boundaries below the historical max get filtered out as "already
seen", and the system stops adding new ckpts. The fix (`self._ckpt_scan_floor
= target`, not `max(self._ckpt_scan_floor, target)`) is one line; the
diagnostic that surfaced it was an unrelated 477-rollback wall-budget timeout.

### 2. Empty bodies pass cargo but aren't solutions

A completion that produces no code (the prompt-echo trap fires and rolls
everything back to offset 0) used to be returned as "complete" because
`fn solve_sudoku(...) { }` compiles fine — no blocking diagnostics, just
an `unused_variables` warning. The final-check pass would report `verdict:
"ok"`, mark `is_complete=True`, and the `final` event would emit `content=""`.

`_is_body_trivially_empty` now classifies whitespace + Rust comments as
empty content; `_emit_final_check` reports `is_complete=False` and the
continuation loop fires a re-prompt. **Subtler half of the fix**: the
continuation path skips its misleading "above is incomplete" comment when
content is empty — otherwise the comment lands right after the function's
`{` and the model reads it as the body, emitting just `}` to close an
empty function. Pre-fix that path produced `final = '}'`; post-fix it
produces real code from a fresh code-completion attempt.

### 3. The right rollback target is "largest ckpt strictly less than offset_at_check"

Picking `ckpt[-1]` can land at or past the failing slice (because the
buffer has grown since the cargo check fired), causing a tight retry loop
on the same offset. The retry must also **remove** the survivor from
`ckpt` — otherwise re-emitting an identical boundary in the next attempt
re-establishes the same target, the next rollback picks the same survivor,
and the loop never makes progress.

Empirically: pre-fix, a real-LLM run on qwen3.5:9b sudoku hit 588 rollbacks
all targeting offset 1962 within 180 seconds (wall budget). Post-fix, the
same run goes 43 rollbacks across 67 distinct offsets, monotonically
backing up when the model re-emits broken code, until either it converges
or the budget runs out.

### 4. Four orchestration modes; three live, one decorative

| Mode | Approach | Reliable? | Cost |
|---|---|---|---|
| **R0** raw | bare prompt, raw mode, no thinking | always | baseline |
| **R1** raw + `<think>` injection | append `<think>\n` to prompt; parse `</think>` or markdown fence as boundary | per-model fragile (qwen3.5:9b yes, 122b no) | 1 call |
| **T2** two-phase | phase 1 = chat-template-wrapped raw call with `stop=["</think>"]`; phase 2 = raw continuation with thinking trace baked in as a Rust-comment block | reliable | 2 calls |
| **C3** chat + sysprompt | `/api/chat` with `think:true` + system prompt forbidding markdown / preamble / closing brace; brace-matched body extractor strips re-declared signature | reliable | 1 call |

T2 is the gold path: it preserves the rollback prompt-splicing protocol
(rollback retries are bare raw mode), gets thinking that actually conditions
the code (the comment is in the prompt the model continues), and gives a
reliable `</think>` boundary (Ollama's stop sequence terminates phase 1
cleanly). Cost is one extra round trip; for a 122B cold load that's
significant, but for a warm model it's <1 s.

C3 is the simplest to wire up (one call, no per-family template registry,
no boundary parsing — Ollama returns `thinking` and `content` as separate
JSON fields per chunk) but the model still re-declares the function and
sometimes picks its own return type, leading to type-mismatch errors that
the rollback path then has to recover from.

R1 is mostly there for the experimental contrast — it's the cheapest way
to get *some* thinking visibility, and it works on small models. It fails
on large models that don't close `</think>` reliably.

### 5. Thinking-mode reliability is template-level, not prompt-level

Reasoning models emit `<think>…</think>` because they were trained on
assistant-turn chat templates that inject those tokens. In raw mode (which
the rollback pipeline needs for literal prompt continuation), no in-prompt
directive reliably gets the model to close `</think>`. Confirmed by direct
probe: `raw=true + think=true` on qwen3.5:9b is bit-identical to
`raw=true` alone:

| | `eval_count` | `eval_duration` | Response |
|---|---|---|---|
| `raw=true, think=true` | 15 | 81.8 ms | `\n    a + b\n}\n\nfn main() {\n    let` |
| `raw=true` (no think) | 15 | 82.1 ms | `\n    a + b\n}\n\nfn main() {\n    let` |

Same token count, same generation time, bit-identical output. The `think`
flag is a placebo under raw mode. The chat-template parser is what
materialises `<think>` into the model's input — bypass it and you bypass
thinking, regardless of the API flag.

### 6. Large reasoning models overthink in C3 and need a wall

qwen3.5:122b in CHAT_INSTRUCTED mode emitted **22,950 chars of thinking**
(~7,500 tokens) on the sudoku prompt and never produced a single `content`
chunk — the chat call effectively never returns. By contrast, qwen3.5:9b
on the same prompt thinks ~350 chars and then produces code. 60× ratio,
not bug.

`GenerationConfig.think_char_budget` (default 8000 chars ≈ 2500 tokens)
caps cumulative `thinking`-field chars. When exceeded, the chat stream is
aborted and the run falls through to the RAW-mode continuation path,
which produces actual code without thinking. Verified end-to-end:
qwen3.5:122b sudoku, budget=3000, runs cleanly in 9.5 s with the status
event `thinking exceeded budget (3000 chars) — chat stream aborted;
falling back to RAW continuation`.

### 7. R1's post-`</think>` content needs cleaning

After the model closes thinking, it often emits chat-style output: a
markdown fence ```` ```rust ```` followed by a re-declared function
signature. `ThinkSplitter` correctly detects `</think>` as the boundary,
but the chars that follow are not literal body continuation — they're
the model's chat-mode answer, fence and all, which used to pollute
`Code.content`.

`PostBoundaryStripper` (new) detects this within the first ~40
non-whitespace bytes after the boundary. If a fence or a `fn ` start is
present, it delegates to `ChatBodyStripper` to extract only the body
bytes; if not, it passes the clean continuation through. End-to-end on
qwen3.5:122b in R1: pre-fix, `Code.content` was 10,243 chars of
fence-wrapped redeclared function; post-fix, 19 chars of clean body
(`'\n        a + b\n    '` for `fn add`).

---

## Pipeline notes

- The full test suite is 78 tests (52 unit, 26 playwright, ~6 live-LLM
  gated on `RUN_REAL_LLM=1`). All pass on the current branch.
- Per-run JSONL logs at `results/web_demo/run_<TS>_<id>.jsonl` record every
  emitted event including `mode_active`, `code_snapshot`s for the click-to-
  preview UI, and the full token stream. They're sufficient to replay any
  run for diagnosis.
- The live demo is fronted by a Cloudflare named tunnel; no public port is
  exposed on the host. WebSocket keepalive is tuned at the tunnel level
  (`keepAliveTimeout: 600s`) so 122B cold-loads don't trip the edge's
  default 100 s idle timeout.

## What's next

- A "bouncing detector" for the rollback policy: when N consecutive rollbacks
  cluster in a narrow offset window, retreat much further (halve the
  survivor offset, or jump to 0) to break the model out of a local minimum.
- More chat-template registry entries for T2 (currently only Qwen3.x is
  verified; gpt-oss, deepseek-r1, nemotron-cascade-2, nemotron-3 families
  need their `<|start|>` / `<|im_start|>` variants documented).
- Real benchmark numbers across the four modes on a fixed problem set —
  the per-mode live-LLM tests in this session were sanity probes, not
  measurements.
