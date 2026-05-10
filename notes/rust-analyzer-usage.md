# rust-analyzer Usage Reference

Practical reference for building a headless rust-analyzer client over LSP. Focused on the operations needed for this project: incremental edits, diagnostics, completions, and hover.

---

## 1. Transport

rust-analyzer communicates over **stdio** using JSON-RPC 2.0. Spawn the binary with no arguments.

Each message is framed with an HTTP-style header:

```
Content-Length: <byte-count>\r\n
\r\n
<JSON payload>
```

- `Content-Length` is in **bytes** (UTF-8 encoded), not characters.
- Header lines must end with `\r\n`, not bare `\n`.
- For debug logging, set `RA_LOG=trace` (not `RUST_LOG`).

Three message types:

| Type | Has `id`? | Direction | Reply expected? |
|---|---|---|---|
| Request | Yes | client→server | Yes (matching `id`) |
| Response | Yes | server→client | No (it is the reply) |
| Notification | No | either | No |

The client must demux incoming messages: responses (have `id`, match to pending requests) vs. notifications (no `id`, dispatched by `method`).

---

## 2. Initialization

### 2.1 Initialize request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "processId": 12345,
    "rootUri": "file:///path/to/cargo/project",
    "capabilities": {
      "general": {
        "positionEncodings": ["utf-8", "utf-16"]
      },
      "textDocument": {
        "diagnostic": {},
        "completion": {
          "completionItem": {
            "snippetSupport": false,
            "resolveSupport": {
              "properties": ["additionalTextEdits"]
            }
          }
        },
        "publishDiagnostics": {
          "relatedInformation": true
        }
      },
      "workspace": {
        "diagnostics": {
          "refreshSupport": true
        }
      },
      "experimental": {
        "serverStatusNotification": true
      }
    },
    "initializationOptions": {
      "cargo": {
        "buildScripts": { "enable": true },
        "allTargets": true
      },
      "procMacro": { "enable": true },
      "checkOnSave": false,
      "diagnostics": {
        "enable": true,
        "experimental": { "enable": false }
      }
    }
  }
}
```

Key `initializationOptions`:

| Key | Type | Default | Why it matters |
|---|---|---|---|
| `checkOnSave` | bool | `true` | **Set to `false`** for headless use. Prevents rust-analyzer from spawning `cargo check` on every save, which is slow (2-30s) and unnecessary when we pull diagnostics on demand. |
| `diagnostics.enable` | bool | `true` | Master switch for native diagnostics (type errors, unresolved names). Must be `true`. |
| `diagnostics.experimental.enable` | bool | `false` | Extra diagnostics with higher false positive rate. Leave `false` initially. |
| `cargo.buildScripts.enable` | bool | `true` | Needed for proc macro expansion. Without it, `derive` macros won't resolve. |
| `procMacro.enable` | bool | `true` | Same — required for `#[derive(...)]` to work. |
| `check.command` | string | `"check"` | Can set to `"clippy"` for lint diagnostics. Not needed for our use case. |

Key `capabilities` to declare:

| Capability | Why |
|---|---|
| `textDocument.diagnostic: {}` | Enables **pull diagnostics** (LSP 3.17). Without this, you only get push diagnostics from cargo check. |
| `workspace.diagnostics.refreshSupport: true` | Server sends `workspace/diagnostic/refresh` when native diagnostics change. Tells you when to re-pull. |
| `experimental.serverStatusNotification: true` | Enables `experimental/serverStatus` notifications. Needed to know when the server is ready. |
| `general.positionEncodings: ["utf-8"]` | Negotiate UTF-8 positions instead of default UTF-16. Simpler if your client works in bytes. |

### 2.2 Initialized notification

After receiving the `initialize` response, immediately send:

```json
{"jsonrpc": "2.0", "method": "initialized", "params": {}}
```

### 2.3 Waiting for readiness

After `initialized`, rust-analyzer loads the workspace asynchronously (parses Cargo.toml, resolves dependencies, expands macros, builds symbol index). **Semantic queries return empty or wrong results until loading completes.**

Monitor readiness via `experimental/serverStatus` notifications:

```json
{
  "jsonrpc": "2.0",
  "method": "experimental/serverStatus",
  "params": {
    "health": "ok",
    "quiescent": true,
    "message": null
  }
}
```

- `quiescent: false` → background work in progress.
- `quiescent: true` → ready for queries.
- `health: "ok"` → fully functional.
- `health: "warning"` → partially functional (e.g., missing dependency). Results may be incomplete.
- `health: "error"` → fatal config problem. Most queries will fail.

**Initial indexing time:** ~2-5s for a small project (10k LOC), ~10-30s for medium (100k LOC).

### 2.4 Shutdown

```
Client                          Server
  |--- shutdown (request) ----->|
  |<-- shutdown response -------|
  |--- exit (notification) ---->|
```

Without proper shutdown, rust-analyzer logs "client exited without proper shutdown." Use a signal handler to send shutdown on SIGINT/SIGTERM.

---

## 3. Document Sync

rust-analyzer uses **incremental** text document sync. The client maintains the authoritative document state and sends diffs.

### 3.1 Open a file

```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/didOpen",
  "params": {
    "textDocument": {
      "uri": "file:///path/to/src/main.rs",
      "languageId": "rust",
      "version": 1,
      "text": "fn main() {\n\n}\n"
    }
  }
}
```

- The `uri` must correspond to a file within the Cargo project (under `rootUri`).
- The file does not need to exist on disk. rust-analyzer's VFS uses the in-memory content from `didOpen`.
- `version` starts at any integer; increment on each `didChange`.

### 3.2 Incremental edit

```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/didChange",
  "params": {
    "textDocument": {
      "uri": "file:///path/to/src/main.rs",
      "version": 2
    },
    "contentChanges": [
      {
        "range": {
          "start": {"line": 1, "character": 0},
          "end": {"line": 1, "character": 0}
        },
        "text": "    let x: i32 = 42;\n"
      }
    ]
  }
}
```

- **Positions are 0-indexed** (line 0 is the first line, character 0 is the first column).
- **Position encoding** defaults to UTF-16 code units. If you negotiated UTF-8 in `initialize`, use byte offsets instead.
- **Insert** = `start == end`, text is non-empty.
- **Delete** = `start != end`, text is empty.
- **Replace** = `start != end`, text is non-empty.
- **Full document replacement**: omit `range`, set `text` to the entire file content. Simpler but less efficient.
- Multiple `contentChanges` in one notification are applied sequentially. Ranges in later changes reference positions after earlier changes have been applied.

### 3.3 Close a file

```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/didClose",
  "params": {
    "textDocument": {
      "uri": "file:///path/to/src/main.rs"
    }
  }
}
```

After close, rust-analyzer reverts to the on-disk version.

### 3.4 Full-document updates (simpler alternative)

For this project, sending the full document on each change may be simpler than computing incremental diffs, at the cost of slightly more work for rust-analyzer:

```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/didChange",
  "params": {
    "textDocument": {"uri": "file:///path/to/src/main.rs", "version": 3},
    "contentChanges": [
      {"text": "fn main() {\n    let x: i32 = 42;\n    let y = x + 1;\n}\n"}
    ]
  }
}
```

Omitting `range` means "replace entire document." rust-analyzer accepts this even though it advertises incremental sync.

---

## 4. Diagnostics

### 4.1 Two independent diagnostic streams

rust-analyzer produces diagnostics from two independent sources. They do not overlap.

| Property | Native diagnostics | Flycheck diagnostics |
|---|---|---|
| Source | rust-analyzer's own analysis | `cargo check` / `cargo clippy` |
| `source` field | `"rust-analyzer"` | `"rustc"` |
| Delivery | **Pull** (`textDocument/diagnostic`) | **Push** (`textDocument/publishDiagnostics`) |
| Latency | 10-50ms | 2-30s |
| Coverage | Syntax errors, type mismatches, unresolved names/imports, missing match arms, unused variables | All of the above + borrow checker, lifetime errors, lint warnings |
| Partial code tolerance | Good (error-recovering parser) | Poor (requires compilable code) |

**For this project:** use native (pull) diagnostics for the real-time loop (fast, tolerant of partial code). Optionally run flycheck at function boundaries for borrow checker coverage.

### 4.2 Pull diagnostics

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "method": "textDocument/diagnostic",
  "params": {
    "textDocument": {"uri": "file:///path/to/src/main.rs"}
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "result": {
    "kind": "full",
    "resultId": "abc123",
    "items": [
      {
        "range": {
          "start": {"line": 3, "character": 12},
          "end": {"line": 3, "character": 17}
        },
        "severity": 1,
        "code": "type-mismatch",
        "source": "rust-analyzer",
        "message": "expected `String`, found `i32`",
        "relatedInformation": []
      }
    ]
  }
}
```

Severity levels:

| Value | Meaning | Action |
|---|---|---|
| 1 | Error | Rollback candidate |
| 2 | Warning | Log, don't rollback |
| 3 | Information | Ignore |
| 4 | Hint | Ignore |

### 4.3 Push diagnostics

Arrive as unsolicited notifications:

```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/publishDiagnostics",
  "params": {
    "uri": "file:///path/to/src/main.rs",
    "version": 5,
    "diagnostics": [
      {
        "range": { ... },
        "severity": 1,
        "code": "E0308",
        "source": "rustc",
        "message": "mismatched types\nexpected `String`, found `i32`"
      }
    ]
  }
}
```

Push diagnostics are triggered by:
- File save (if `checkOnSave: true`)
- Manual `rust-analyzer/runFlycheck` notification

### 4.4 Triggering flycheck manually

```json
{
  "jsonrpc": "2.0",
  "method": "rust-analyzer/runFlycheck",
  "params": {
    "textDocument": {"uri": "file:///path/to/src/main.rs"}
  }
}
```

Results arrive via `textDocument/publishDiagnostics`. Cancel with:

```json
{"jsonrpc": "2.0", "method": "rust-analyzer/cancelFlycheck", "params": {}}
```

### 4.5 Diagnostic refresh

When native diagnostics change (e.g., after a `didChange` is processed), the server sends:

```json
{
  "jsonrpc": "2.0",
  "method": "workspace/diagnostic/refresh",
  "params": null
}
```

This tells the client: "your cached pull diagnostics may be stale; re-pull if needed." Requires `workspace.diagnostics.refreshSupport: true` in client capabilities.

### 4.6 Behavior on partial/incomplete code

rust-analyzer's parser is **error-recovering** — it produces a CST even for invalid syntax. Key behaviors:

1. **Incomplete functions**: A function missing its closing `}` produces a syntax error diagnostic at the expected position. Statements before the incomplete end are still analyzed semantically.

2. **Incomplete expressions**: `let x = ` (no RHS) produces a syntax error. The binding `x` may or may not be visible to subsequent code depending on parser recovery.

3. **Semantic suppression on heavy syntax damage**: When a file has many syntax errors, rust-analyzer **suppresses semantic diagnostics entirely** to avoid noise from cascading parser recovery failures. This means you may get fewer diagnostics than expected on heavily incomplete code.

4. **Cascading from macros**: Invalid syntax can cause macro expansion failures, which cascade into errors like "cannot find function `main`" even when `main` exists. Filter diagnostics by range to avoid acting on cascading errors outside the region you just changed.

### 4.7 Native diagnostic codes

Common codes relevant to this project:

| Code | Meaning | Example |
|---|---|---|
| `syntax-error` | Parse failure | Missing `;`, unmatched `{` |
| `type-mismatch` | Type error | Expected `String`, found `i32` |
| `unresolved-import` | Bad import | `use std::collections::HashMop;` |
| `unresolved-module` | Missing module | `mod foo;` with no `foo.rs` |
| `missing-match-arm` | Non-exhaustive match | Missing `None` arm |
| `missing-unsafe` | Unsafe operation outside `unsafe` block | Dereferencing raw pointer |
| `unused-variables` | Unused binding | `let x = 5;` (x never read) |
| `incorrect-ident-case` | Naming convention | `let MyVar = 5;` (should be snake_case) |

Suppress specific codes via config:

```json
{"diagnostics": {"disabled": ["unused-variables", "incorrect-ident-case"]}}
```

### 4.8 Recommended configuration for this project

```json
{
  "checkOnSave": false,
  "diagnostics": {
    "enable": true,
    "experimental": {"enable": false},
    "disabled": ["unused-variables", "incorrect-ident-case"]
  }
}
```

- `checkOnSave: false`: we control when flycheck runs.
- Disable `unused-variables` and `incorrect-ident-case`: noisy on partial code, never worth a rollback.

---

## 5. Completions

### 5.1 Request

```json
{
  "jsonrpc": "2.0",
  "id": 20,
  "method": "textDocument/completion",
  "params": {
    "textDocument": {"uri": "file:///path/to/src/main.rs"},
    "position": {"line": 5, "character": 7}
  }
}
```

Trigger characters: `:`, `.`, `'`, `(`.

### 5.2 Response

```json
{
  "jsonrpc": "2.0",
  "id": 20,
  "result": {
    "isIncomplete": false,
    "items": [
      {
        "label": "push",
        "kind": 2,
        "detail": "fn push(&mut self, value: T)",
        "insertText": "push($0)",
        "insertTextFormat": 2,
        "sortText": "0000push",
        "filterText": "push",
        "data": { ... }
      },
      {
        "label": "len",
        "kind": 2,
        "detail": "fn len(&self) -> usize",
        "insertText": "len()",
        "insertTextFormat": 1,
        "sortText": "0001len"
      }
    ]
  }
}
```

`kind` values (subset):

| Value | Meaning |
|---|---|
| 1 | Text |
| 2 | Method |
| 3 | Function |
| 5 | Field |
| 6 | Variable |
| 7 | Class (struct/enum) |
| 8 | Interface (trait) |
| 9 | Module |
| 14 | Keyword |
| 21 | Constant |

### 5.3 Completion resolve

For auto-import completions, the initial response omits the import edit. To get it:

```json
{
  "jsonrpc": "2.0",
  "id": 21,
  "method": "completionItem/resolve",
  "params": { <the completion item with its `data` field> }
}
```

The response includes `additionalTextEdits` with the import statement. Only needed if you use completions for forward masking.

### 5.4 Latency

Typical: **10-50ms**. Faster after initial indexing when the symbol cache is warm. Can spike on first completion in a new scope or after large edits.

---

## 6. Hover

### 6.1 Request

```json
{
  "jsonrpc": "2.0",
  "id": 30,
  "method": "textDocument/hover",
  "params": {
    "textDocument": {"uri": "file:///path/to/src/main.rs"},
    "position": {"line": 3, "character": 8}
  }
}
```

### 6.2 Response

```json
{
  "jsonrpc": "2.0",
  "id": 30,
  "result": {
    "contents": {
      "kind": "markdown",
      "value": "```rust\nlet x: Vec<i32>\n```\n\n---\n\nA contiguous growable array type."
    },
    "range": {
      "start": {"line": 3, "character": 8},
      "end": {"line": 3, "character": 9}
    }
  }
}
```

Returns type information and documentation for the symbol at the cursor. Useful for context enrichment — inject the type of a variable into the LLM's context before regeneration.

Latency: **5-30ms**.

---

## 7. Go-to-definition and References

### 7.1 Go-to-definition

```json
{
  "jsonrpc": "2.0",
  "id": 40,
  "method": "textDocument/definition",
  "params": {
    "textDocument": {"uri": "file:///path/to/src/main.rs"},
    "position": {"line": 10, "character": 15}
  }
}
```

Response: a `Location` or `LocationLink[]` pointing to the definition site. Latency: **5-20ms**.

### 7.2 Find references

```json
{
  "jsonrpc": "2.0",
  "id": 41,
  "method": "textDocument/references",
  "params": {
    "textDocument": {"uri": "file:///path/to/src/main.rs"},
    "position": {"line": 2, "character": 7},
    "context": {"includeDeclaration": true}
  }
}
```

Response: `Location[]`. Latency: **50-500ms** (workspace-wide search).

---

## 8. Custom Extensions

### 8.1 Server status

Enabled by `experimental.serverStatusNotification: true` in client capabilities. Notification from server:

```json
{
  "method": "experimental/serverStatus",
  "params": {"health": "ok", "quiescent": true}
}
```

### 8.2 View syntax tree (debugging)

```json
{
  "id": 50,
  "method": "rust-analyzer/viewSyntaxTree",
  "params": {
    "textDocument": {"uri": "file:///path/to/src/main.rs"}
  }
}
```

Returns a text dump of the CST. Useful for understanding parser recovery behavior on partial code.

### 8.3 Expand macro

```json
{
  "id": 51,
  "method": "rust-analyzer/expandMacro",
  "params": {
    "textDocument": {"uri": "file:///path/to/src/main.rs"},
    "position": {"line": 1, "character": 0}
  }
}
```

Returns the expanded form of a macro invocation. Useful for debugging `vec![]`, `derive`, etc.

### 8.4 Reload workspace

```json
{"id": 52, "method": "rust-analyzer/reloadWorkspace", "params": null}
```

Forces re-reading of `Cargo.toml` and dependency resolution. Use after modifying `Cargo.toml`.

---

## 9. Practical Patterns for This Project

### 9.1 Minimal generation loop

```
1. spawn rust-analyzer, send initialize, wait for quiescent
2. didOpen("src/main.rs", skeleton code with function signature)
3. for each generated statement:
     a. didChange — append statement to file content
     b. sleep ~10ms (let rust-analyzer process the change)
     c. textDocument/diagnostic — pull native diagnostics
     d. filter: severity == 1, range overlaps new statement
     e. if errors → rollback; inject diagnostic message into LLM context
```

### 9.2 Diagnostic filtering heuristic

Not all severity-1 diagnostics on partial code are actionable. Filter by:

1. **Range**: only consider diagnostics whose range overlaps with code you consider "completed" (ended with `;` or `}`). Ignore diagnostics past the cursor.
2. **Code**: ignore `syntax-error` at the very end of the file (likely from missing closing braces that will be added later).
3. **Cascading**: if the diagnostic references a symbol defined after the error point, it's likely a cascade. Ignore.

### 9.3 Determining when diagnostics are ready after an edit

After sending `didChange`, rust-analyzer processes the edit asynchronously. Two approaches:

1. **Wait for `workspace/diagnostic/refresh`**: the server sends this when diagnostics may have changed. Then pull.
2. **Poll with a short delay**: send `textDocument/diagnostic` after ~10-20ms. If the edit hasn't been processed yet, you'll get stale diagnostics (which is fine — you'll catch the error on the next checkpoint).

Option 2 is simpler and matches the asynchronous verification design: the LSP doesn't need to be perfectly synchronous with generation.

### 9.4 Position tracking

As you append code, track the line and character offset of the end of the document. This is where you insert next. For full-document updates (simpler), just maintain the full string and send it each time — no position math needed.

---

## Sources

- [rust-analyzer manual](https://rust-analyzer.github.io/book/)
- [rust-analyzer configuration reference](https://rust-analyzer.github.io/book/configuration.html)
- [LSP extensions documentation](https://github.com/rust-lang/rust-analyzer/blob/master/docs/dev/lsp-extensions.md)
- [Server capabilities source](https://github.com/rust-lang/rust-analyzer/blob/master/crates/rust-analyzer/src/lsp/capabilities.rs)
- [LSP specification 3.17](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)
- [Suppress semantic diagnostics on syntax errors (PR #17536)](https://github.com/rust-lang/rust-analyzer/pull/17536)
- [Pull diagnostics identifier (PR #19266)](https://github.com/rust-lang/rust-analyzer/pull/19266)
