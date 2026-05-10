# rust-analyzer Scope and Granularity

Practical reference for what rust-analyzer sees, what has to be completed before it produces useful diagnostics, what those diagnostics attach to, and what this implies for the rollback mechanism.

Complements `rust-analyzer-usage.md` (LSP-protocol-level reference) with **semantic** observations.

---

## 1. At what level does rust-analyzer produce diagnostics?

**Character-range level.** Every diagnostic carries a `range` with `{start: {line, character}, end: {line, character}}` — 0-indexed character positions, not whole lines. Example:

```json
"range": { "start": {"line": 3, "character": 12}, "end": {"line": 3, "character": 17} }
```

Whole-line or whole-statement diagnostics are just ranges that happen to span those limits. The underlying granularity is *character spans*.

---

## 2. What has to be completed to get partial diagnostics?

The **current syntactic unit** must parse cleanly, even if the rest of the file does not.

| Unit | Completeness criterion |
|---|---|
| Statement | ends with `;` |
| Block (function/match/if/loop body) | closed with `}` |
| Struct / enum / impl | closed with `}` |
| Function body | can remain open — error-recovering parser still analyzes completed statements inside it |

Key rules from §4.6 of `rust-analyzer-usage.md`:

1. **Error-recovering parser**: produces a CST even for invalid syntax. Statements before an unclosed `}` are still semantically analyzed.
2. **Semantic suppression on heavy syntax damage**: when too many tokens fail to parse, rust-analyzer drops *all* semantic diagnostics to avoid noise. So "a little incompleteness" is fine; "a lot" gives you nothing.
3. **Cascading from macros**: a syntax error upstream can kill macro expansion and generate spurious errors across the whole file. Filter by range overlap with the region you just changed.

**Project-relevant rule** (§9.2): treat code as "completed" once it ends in `;` or `}`. Diagnostics whose range overlaps completed code are trustworthy; diagnostics beyond the last terminator are likely from "still generating" and should be ignored.

---

## 3. What do diagnostics point to?

Ranges can be as tight as a single identifier or as wide as a block — whichever is most faithful to the error. In the five-level taxonomy:

| Level | Supported? | Example codes | Example span |
|---|---|---|---|
| identifier | yes | `unresolved-import`, `unused-variables`, `incorrect-ident-case`, `unresolved-reference` | `HashMop` in `use std::collections::HashMop;` |
| expression / sub-expression | yes | `type-mismatch` | The `42` in `let x: String = 42;` |
| statement / line | yes | `syntax-error` (missing `;`), `missing-unsafe` | The full `*raw_ptr` dereference statement |
| block | yes | `missing-match-arm` | The whole `match x { ... }` block |
| module | **no** | — | Module-level concerns still attach to a specific item *inside* the module (e.g., `unresolved-module` points to the `mod foo;` statement, not the module itself) |

Diagnostics are always anchored to concrete source tokens, never to "the whole module" as a concept.

---

## 4. Worked examples

### 4.1 Unterminated string literal (mid-function)

```rust
fn main() {
    let collected_iterator: Vec<i32> = (0..10).collect();
    println!("Collected (0..10) into: {:?}", collected_iterator);

    let mut xs = vec![1i32, 2, 3];
    println!("Initial vector: {:?}", xs);

    println!("Push 4 into the
```

| Line(s) | Verdict |
|---|---|
| `let collected_iterator: Vec<i32> = (0..10).collect();` | ✅ clean — parses as complete statement, `Vec<i32>` resolves `collect()` turbofish |
| `println!("Collected (0..10) into: {:?}", collected_iterator);` | ✅ clean — `Vec<i32>` implements `Debug` |
| `let mut xs = vec![1i32, 2, 3];` | ✅ clean (rustc would flag `unused_mut` as a lint, but that's not in native diagnostics) |
| `println!("Initial vector: {:?}", xs);` | ✅ clean |
| `println!("Push 4 into the` | ❌ `syntax-error` — unterminated string literal; consumes rest of buffer as string content |

**Editor view:** one red squiggle from the opening `"` on the last line extending to EOF. No other squiggles. Cascading `"expected )"`, `"expected ;"`, `"expected }"` errors are typically collapsed into the single unterminated-string message because the lexer has no follow-up tokens.

**Rollback signal:** the §9.2 filter skips this error because it lives in code that hasn't been terminated with `;` or `}`. Correct — the model is still generating this statement.

### 4.2 Incomplete function with mismatched return type

```rust
use std::mem;

#[derive(Debug, Clone, Copy)]
struct Point { x: f64, y: f64 }

struct Rectangle { top_left: Point, bottom_right: Point }

fn origin() -> Point {
```

| Section | Verdict |
|---|---|
| `use std::mem;` | ✅ parses clean (rustc would warn `unused_imports`, but that's not native) |
| `struct Point { ... }` with derives | ✅ clean — `f64` implements `Debug`, `Clone`, `Copy` |
| `struct Rectangle { ... }` | ✅ clean |
| `fn origin() -> Point {` | ❌ body opened but never closed |

**Diagnostics on the tail:**

1. **`syntax-error`** (guaranteed): "expected `}`" / "unexpected EOF." Anchored at EOF or the unclosed `{`.
2. **`type-mismatch`** (probable): "expected `Point`, found `()`." The signature declares `Point`; the empty body returns unit. Whether this fires depends on parser recovery — rust-analyzer usually does type-check the signature against a synthesized empty block, but may suppress if it treats the body as "still incomplete."

**Filter implication:** the missing-`}` error is outside completed code → skip. But the `type-mismatch` attaches to the signature `-> Point`, which *is* completed → would wrongly trigger rollback. **Additional heuristic needed**: suppress type-mismatch errors that overlap a function header whose body is still open.

### 4.3 Unresolved external crate (detached-file mode)

```rust
use http::{Request, Response};

fn response(req: Request<()>) -> http::Result<Response<()>> {
    match req.uri().path() {
```

In rust-analyzer **detached-file mode** (no `Cargo.toml` providing `http` as a dependency), `http` is unknown. One root error cascades into many:

| Token | Verdict |
|---|---|
| `use http::{Request, Response};` | ❌ `unresolved-import` on `http` |
| `Request<()>`, `Response<()>`, `http::Result` | ❌ "unresolved type" each — cascading from the missing crate |
| `req.uri()`, `.path()` | ❌ "unknown method" — receiver type is unresolved |
| EOF | ❌ `syntax-error ×2` for unclosed match and function bodies |

**Essentially every line squiggled.** This is the §4.6 item 4 warning in action: one unresolved crate can break resolution across the entire file.

If the same snippet is placed in a Cargo project with `http = "1"` in `[dependencies]`:

| Token | Verdict |
|---|---|
| `use http::...` | ✅ resolves |
| Type expressions, method calls | ✅ resolve |
| match block open, then EOF | ❌ single `syntax-error`: "expected `}`" |

**Rollback signal difference:** detached mode would blast the model with 4+ error messages on every iteration, all spurious. This demonstrates why the LSP client in `soundcode/` runs against a real Cargo project (`test_workspace/`), not against standalone files.

### 4.4 Undefined helper functions (complete file, missing definitions)

```rust
use http::{Request, Response};

fn response(req: Request<()>) -> http::Result<Response<()>> {
    match req.uri().path() {
        "/" => index(req),
        "/foo" => foo(req),
        "/bar" => bar(req),
        _ => not_found(req),
    }
}

fn main() {}
```

File is syntactically complete. Diagnostics:

- **Four `unresolved-reference` errors** — one per match arm, on `index`, `foo`, `bar`, `not_found`. These functions aren't defined.
- **No exhaustiveness error** — `_ => not_found(req)` covers the rest; match is exhaustive.
- **No move error** — `req` is moved in four arms, but only one arm runs. Legal.
- **Cascading consequence**: the match expression's type can't be inferred (all arm types unknown), so the return-type check on `-> http::Result<Response<()>>` may or may not fire a secondary error.
- **`dead_code`** warning on `response` (rustc lint via flycheck; not native).

**Rollback signal:** all four errors land inside completed statements → trustworthy. The model forgot to define helpers; error-informed regeneration should work.

### 4.5 Module privacy (cross-file resolution)

```rust
// main.rs
mod my_module;
use my_module::response;

// my_module.rs
fn response(req: Request<()>) -> http::Result<Response<()>> { ... }
```

The `use` statement in `main.rs` gets ❌ `E0603` "function `response` is private." Items default to private; crossing a module boundary requires `pub`.

**Note on the help diagnostic:** rust-analyzer will also emit a **★** info-severity pointer in `my_module.rs` ("the function `response` is defined here") attached to the E0603 chain. This is severity-3 (information), not an error — shouldn't trigger rollback.

---

## 5. Rollback granularity — design decision

**Primary checkpoint: statement boundary** (after `;` or after `}` that closes an inner block).
**Secondary checkpoint: function-body boundary** (for heavier `cargo check`).

### Why statement-level is the floor

1. **Finer than statement gives no semantic signal.** Partial expressions don't type-check; you'd get syntax errors the filter has to suppress anyway. Pure overhead.
2. **Statements are where diagnostics become trustworthy.** §9.2 filter: "act on diagnostics overlapping code ending in `;` or `}`" — exactly the statement boundary.
3. **Error granularity matches rollback granularity.** Most common errors (E0308 type mismatch, E0425 unresolved name, E0599 no method) attach to a single RHS expression or identifier inside one statement. Rolling back to the start of that statement is the minimum undo that clears the error.

### Why not finer than statement

| Granularity | Problem |
|---|---|
| identifier | Error detection requires the containing statement to parse. Checkpointing on every identifier gives no signal a statement checkpoint doesn't. |
| expression | Partial expressions don't type-check. 5–10× more checkpoints with no added signal. |

### Why not coarser than statement as primary

| Granularity | Problem |
|---|---|
| block (function body, match body) | Too much work discarded per rollback. Useful as a *secondary* heavier check — that's where `cargo check` runs for borrow-checker coverage (E0382, lifetimes, full trait resolution). |
| whole function / module | This is the `compiler` baseline from the week3 experiment — generate-then-fix, not rollback. No KV-cache reuse. Measured: +12pp compile on 1B–32B but *hurts* pass@1 at 122B. |

### Two-tier architecture

```
Token stream ─┬── after `;`      → snapshot KV-cache, pull native diagnostics
              │                    (10-50ms, catches type/name errors — ~30% of errors)
              ├── after inner `}` → same + re-pull (scoped bindings now visible)
              └── after fn `}`    → snapshot + trigger cargo check (2-30s, async)
                                    (catches borrow checker, lifetimes, full traits — the rest)
```

Maps directly to the week3-journal conclusion: "fast native diagnostics inline + cargo check at function boundaries."

### Oscillation risk

If multiple errors surface at the same boundary, roll back to the **earliest** one (furthest-back checkpoint whose range doesn't overlap any reported error), not the latest. Otherwise: generate → hit error A → rollback → regenerate → hit pre-existing error B → rollback further → oscillate.

---

## 6. Distinguishing non-blocking from blocking errors

Rollback must act on **real semantic errors** and ignore **artifacts of incomplete code**. The distinction comes from two orthogonal signals (plus an optional third), combined in a decision procedure.

### Signal 1: position relative to the last completion boundary

Track the **last completion position** — byte offset of the most recent `;` or `}` that closed a statement or block at the current nesting level, ignoring string/comment contexts. Everything beyond it is "in progress."

```
 line 1: fn response(req: Request<()>) -> http::Result<Response<()>> {
 line 2:     match req.uri().path() {
 line 3:         "/" => index(req),
                                    ^— last_complete_pos (this `,`)
 line 4:         "/foo" =>
                                  ^— writing edge
```

**Rule**: diagnostics whose range starts at or after `last_complete_pos` are **non-blocking** (they're about code the model is still writing). Diagnostics whose range lies entirely before `last_complete_pos` are *candidates* for blocking — pass them to signal 2.

This is the §9.2 filter from `rust-analyzer-usage.md` stated precisely.

### Signal 2: error class

rust-analyzer's `code` field and message text separate "parser ran out of tokens" from "semantics are wrong."

| Pattern | Class | Meaning |
|---|---|---|
| `syntax-error` + message contains "expected" / "unclosed delimiter" / "missing" | **syntactic incompleteness** | parser hit a boundary or EOF mid-construct |
| message contains "unterminated" (string/char/block comment/raw string) | **syntactic incompleteness** | lexer consumed to EOF looking for closer |
| `type-mismatch` (E0308) | **semantic** | completed expression has wrong type |
| `unresolved-reference` (E0425), `unresolved-import` (E0432), `unresolved-module` | **semantic** | name doesn't exist in scope |
| `no-method` (E0599), `trait-impl-incorrect` (E0277) | **semantic** | trait/method resolution failed |
| `missing-match-arm` | **semantic** | fires only on *closed* match blocks → always real |
| `missing-unsafe` | **semantic** | unsafe op outside `unsafe` block |

**Key asymmetry**: semantic diagnostics fire only when the relevant unit parses. So if you see `type-mismatch`, the containing statement already terminated — the question is only whether the error is real or will be fixed by subsequent context.

### Signal 3: temporal stability (optional, stronger)

After a `didChange`, rust-analyzer's incremental analysis is asynchronous. Transient "not-yet-analyzed" diagnostics can appear and then disappear on a re-pull. Per §9.3 of `rust-analyzer-usage.md`:

```python
errors_1 = await ra.get_errors(file, line_range=just_completed_range)
await asyncio.sleep(0.1)
errors_2 = await ra.get_errors(file, line_range=just_completed_range)
persistent = [e for e in errors_2 if e in errors_1]  # by (range, code, message)
```

Doubles LSP round-trips; use only if the fast classifier is giving false positives in practice.

### Decision procedure

```python
def classify(
    diag: Diagnostic,
    *,
    last_complete_pos: int,      # byte offset of last ;/} at scope
    open_fn_body: bool,          # is there an unclosed fn body?
    diag_offset: int,            # byte offset of diag.range.start
) -> Literal["blocking", "non_blocking"]:

    # 1. Range past the writing edge — definitely in-progress.
    if diag_offset >= last_complete_pos:
        return "non_blocking"

    # 2. Syntax-class errors are never real semantic errors.
    if diag.code == "syntax-error":
        return "non_blocking"
    msg = diag.message.lower()
    if "unterminated" in msg or "expected" in msg or "unclosed" in msg:
        return "non_blocking"

    # 3. Demotions for transient semantic errors while a fn body is still open.
    if open_fn_body:
        if diag.code in ("type-mismatch", "unresolved-reference"):
            return "non_blocking"   # defer until fn closes

    # 4. Remaining semantic errors in completed code — blocking.
    return "blocking"
```

Track `open_fn_body` with a small brace-depth counter over emitted tokens (ignore braces inside strings/comments). The same counter gives you `last_complete_pos`.

### Edge cases

**Unresolved reference that will be defined below.** When the model writes a function calling helpers before defining them (e.g. `response()` calling `index`/`foo`/`bar`/`not_found` in example 4.4), every call triggers `unresolved-reference`. But the model is likely about to define them. → Rule 3 handles this: while the enclosing function body is open, defer these errors until close.

**`type-mismatch` on a return signature with empty body.** Case 4.2: `fn origin() -> Point {` with no body → "expected `Point`, found `()`." Range sits on the signature (completed), but the error is an artifact of the empty body. → Same demotion; re-evaluate once the body is written and closed.

**Semantic suppression after heavy damage** (§4.6 item 3). If a syntax error is severe, *all* semantic diagnostics vanish — you get silence that looks clean. → If diagnostic count drops dramatically coinciding with a parse error, don't trust "no errors" — treat as non-blocking and wait for next boundary.

**Macro expansion cascades** (§4.6 item 4). An unresolved import can make `println!` appear broken because the macro fails to expand. → Restrict to diagnostics whose range overlaps tokens you just emitted; ignore cascades in code far from the edit region.

**Cross-module forward references.** rust-analyzer resolves across files with caching lag. → Temporal stability check (signal 3) helps.

### Two-tier application

| Tier | When | Classifier behavior |
|---|---|---|
| Fast | after every `;` / inner `}` | decision procedure above; `open_fn_body = True` → demotions active |
| Slow | after every outer `}` (function close) | `open_fn_body = False` → demotions lift, trust remaining semantic errors; additionally trigger `cargo check` for borrow checker |

Errors demoted at the fast tier get re-evaluated at the slow tier under stricter rules. If they're still there when the function closes, they're real.

### How this applies to the week3 error distribution

From the 9B baseline failures in `week3-journal.md`:

| Code | Count | Classifier behavior |
|---|---|---|
| E0308 type mismatch | 39 | Demoted while fn body open. Most resolve if model was mid-expression. |
| E0425 unresolved name | 30 | Demoted while fn body open. Catches forward-reference-to-helper pattern. |
| E0277 trait not satisfied | 22 | Blocking immediately. Trait resolution doesn't change with further writes in the same fn. |
| E0599 no method | 6 | Blocking immediately. |
| E0382 borrow of moved | — | Push diagnostic only (cargo check). Fires at slow tier exclusively. |

Fast tier fires for E0277/E0599 immediately (correct: real errors) and defers E0308/E0425 until function close (correct: often false alarms during generation).
