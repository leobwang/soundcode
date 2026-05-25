"use strict";

// ─── DOM ───────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const statusEl       = $("status");
const wsStatusEl     = $("ws-status");
const promptSelect   = $("prompt-select");
const langSelect     = $("lang-select");
const verifierSelect = $("verifier-select");
const backendSelect  = $("backend-select");
const algorithmSelect = $("algorithm-select");
const modelSelect    = $("model-select");
const cadenceInput   = $("cadence");
const cadenceLabel   = $("cadence-label");
const instructCheck  = $("instruct");
const modeSelect     = $("mode-select");

// Multilingual support: when the language dropdown changes we reload
// `/api/prompts?lang=…` and switch the code panel's syntax-class on
// `#code-body` (`language-rust`, `language-java`, …). The panel is a
// plain `<pre>` today (no Prism/Highlight.js), but tagging the class
// keeps the option open for a future highlighter drop-in without
// further JS changes.
function currentLang() {
  return (langSelect && langSelect.value) || "rust";
}

function applyLanguageClass() {
  const lang = currentLang();
  if (!codeBody) return;
  // Strip any prior language-* class so we don't accumulate them.
  codeBody.classList.forEach((c) => {
    if (c.startsWith("language-")) codeBody.classList.remove(c);
  });
  codeBody.classList.add(`language-${lang}`);
  const cp = $("code-preview");
  if (cp) {
    cp.classList.forEach((c) => {
      if (c.startsWith("language-")) cp.classList.remove(c);
    });
    cp.classList.add(`language-${lang}`);
  }
}

// Mirror of server-side `REASONING_MODELS`. Models that support reasoning
// (and therefore can engage R1/T2/C3). Non-reasoning models silently fall
// back to RAW on the server; UI greys the non-RAW options to surface that.
const REASONING_MODELS = new Set([
  "qwen3.5:9b", "qwen3.5:35b", "qwen3.5:122b",
  "qwen3.6:35b", "qwen3.6:latest",
  "gpt-oss:20b", "gpt-oss:120b",
  "deepseek-r1:70b",
  "nemotron-cascade-2:30b",
  "nemotron-3-super:120b",
  "nemotron-3-nano:4b",
]);
// Subset that has a chat-template registry entry for TWO_PHASE phase 1.
// Mirror of `CHAT_TEMPLATES` keys in soundcode/llm.py — only qwen3.x for now.
const TWO_PHASE_OK_PREFIXES = ["qwen3."];

function _refreshModeOptionsForModel() {
  const model = modelSelect.value || "";
  const isReasoning = REASONING_MODELS.has(model);
  const isTwoPhaseOk = TWO_PHASE_OK_PREFIXES.some(p => model.startsWith(p));
  for (const opt of modeSelect.options) {
    if (opt.value === "raw") { opt.disabled = false; continue; }
    if (opt.value === "two_phase") {
      opt.disabled = !(isReasoning && isTwoPhaseOk);
    } else {
      opt.disabled = !isReasoning;
    }
  }
  // If the current selection is now disabled, coerce to RAW.
  const cur = modeSelect.options[modeSelect.selectedIndex];
  if (cur && cur.disabled) modeSelect.value = "raw";
  _updateModeHint();
}

function _updateModeHint() {
  const hint = $("mode-hint");
  if (!hint) return;
  const m = modeSelect.value;
  const model = modelSelect.value || "";
  const isReasoning = REASONING_MODELS.has(model);
  const instructOn = instructCheck && instructCheck.checked;
  const msgs = {
    raw:               "Plain code completion. No thinking.",
    raw_think_inject:  "Append <think>\\n to the prompt; split <think>/</think> from the raw stream. Per-model fragile.",
    two_phase:         "Phase 1: harvest thinking via chat-template wrap. Phase 2: bake trace into prompt as comment, raw completion. Two round trips.",
    chat_instructed:   "/api/chat with system prompt forbidding markdown. Reliable thinking + body extraction. One round trip.",
  };
  let text = msgs[m] || "";
  if (m !== "raw" && !isReasoning) {
    text = "Selected model is not a reasoning model — server will coerce to RAW. " + text;
  } else if (m !== "raw" && instructOn) {
    text += " RE-ENTER ON: each rollback re-runs thinking with the cargo-error comment in context.";
  }
  hint.textContent = text;
}
const startBtn       = $("start-btn");
const stopBtn        = $("stop-btn");
const resetBtn       = $("reset-btn");
const promptEditor   = $("prompt-editor");
const modelReadout   = $("model-readout-name");
const codeBody       = $("code-body");
const lspBody        = $("lsp-body");
const llmStack       = $("llm-stack");
const pipelineTrack  = $("pipeline-track");
const pipeLlmBlock   = $("pipe-llm-block");
const pipeCodeBlock  = $("pipe-code-block");
const pipeLspBlock   = $("pipe-lsp-block");
const tokenCounter   = $("token-counter");
const rollbackCounter = $("rollback-counter");
const checkpointCounter = $("checkpoint-counter");

// Backend-stats panel (next to the status pill). Populated from the
// `backend_stats` WS event so the demo can show vLLM's KV-cache hit rate +
// TTFT — the headline number for the rollback-resume speedup story.
// Ollama backends never emit `backend_stats`, so the panel stays hidden.
const backendStatsEl   = $("backend-stats");
const bsHitPctEl       = $("bs-hit-pct");
const bsCachedEl       = $("bs-cached");
const bsPrefixEl       = $("bs-prefix");
const bsTtftEl         = $("bs-ttft");
const bsRollbackBadgeEl = $("bs-rollback-badge");

// ─── state ─────────────────────────────────────────────────────────────
const state = {
  ws: null,
  acceptedCode: "",                  // committed text (sum of token texts)
  tokensByEndOffset: new Map(),      // offset (after token) -> {text, codeSpan, llmPill}
  tokens: 0,
  rollbacks: 0,
  checkpoints: 1,
  isRunning: false,
};

function offsetToRowCol(text, offset) {
  // 1-indexed (row, column) within `text`, treating LF as line break.
  const clamped = Math.min(Math.max(offset, 0), text.length);
  let row = 1, col = 1;
  for (let i = 0; i < clamped; i++) {
    if (text.charCodeAt(i) === 10) { row++; col = 1; }
    else { col++; }
  }
  return [row, col];
}

function rcLabel(text, offset) {
  const [r, c] = offsetToRowCol(text, offset);
  return `(${r},${c})`;
}

// ─── code-preview overlay (driven by LSP-entry hover OR click-to-pin) ─
let pinnedLspEntry = null;

function pinLspEntry(entry) {
  // Toggle the persistent preview to the given entry (or off if null).
  if (pinnedLspEntry && pinnedLspEntry !== entry) {
    pinnedLspEntry.classList.remove("lsp-entry-pinned");
  }
  if (entry === null) {
    pinnedLspEntry = null;
    hideCodePreview();
    const lbl = $("code-preview-label");
    if (lbl) lbl.textContent = "▸ hover preview";
    return;
  }
  pinnedLspEntry = entry;
  entry.classList.add("lsp-entry-pinned");
  showCodePreview(entry._codeSnapshot, entry._snapOffset, entry._verdict);
  const lbl = $("code-preview-label");
  if (lbl) lbl.textContent = "📌 pinned — click again to release";
}

function showCodePreview(snapshot, highlightOffset, kind) {
  const preview = $("code-preview");
  const label = $("code-preview-label");
  if (!preview) return;
  preview.innerHTML = "";
  if (snapshot == null) {
    preview.appendChild(document.createTextNode("(no snapshot for this entry)"));
  } else {
    // Highlight the last line up to `highlightOffset` so the diagnostic's
    // location is visually clear. Choose green for "ok", red for "error",
    // and a neutral grey for "inactive".
    const off = highlightOffset != null
      ? Math.min(Math.max(highlightOffset, 0), snapshot.length)
      : snapshot.length;
    const klass = kind === "error"    ? "preview-highlight-error"
                : kind === "ok"       ? "preview-highlight-ok"
                : kind === "rollback" ? "preview-highlight-rollback"
                : null;
    if (klass && off > 0) {
      const lineStart = Math.max(0, snapshot.lastIndexOf("\n", off - 1) + 1);
      preview.appendChild(document.createTextNode(snapshot.slice(0, lineStart)));
      const span = document.createElement("span");
      span.className = klass;
      span.textContent = snapshot.slice(lineStart, off);
      preview.appendChild(span);
      preview.appendChild(document.createTextNode(snapshot.slice(off)));
    } else {
      preview.textContent = snapshot;
    }
  }
  preview.classList.add("active");
  if (label) label.classList.add("active");
}
function hideCodePreview() {
  const preview = $("code-preview");
  const label = $("code-preview-label");
  if (preview) preview.classList.remove("active");
  if (label) label.classList.remove("active");
}

// ─── cross-panel hover highlighting ────────────────────────────────────
function hookHoverHighlight(rootEl) {
  rootEl.addEventListener("mouseover", (e) => {
    const t = e.target.closest("[data-offset]");
    if (!t) return;
    const off = t.dataset.offset;
    document.querySelectorAll(`[data-offset='${CSS.escape(off)}']`).forEach((el) => {
      el.classList.add("tok-hover");
    });
  });
  rootEl.addEventListener("mouseout", (e) => {
    const t = e.target.closest("[data-offset]");
    if (!t) return;
    const off = t.dataset.offset;
    document.querySelectorAll(`[data-offset='${CSS.escape(off)}']`).forEach((el) => {
      el.classList.remove("tok-hover");
    });
  });
}

function formatDiagsTooltip(verdict, diagnostics, fallbackMsg) {
  // Return a multi-line string suitable for a `title` attribute.
  const verdictLine = `verdict: ${verdict.toUpperCase()}`;
  if (!diagnostics || diagnostics.length === 0) {
    return `${verdictLine}\n${fallbackMsg || "(no diagnostics)"}`;
  }
  const lines = [verdictLine];
  for (const d of diagnostics) {
    const tag = (d.category || "?").toUpperCase();
    const code = d.code ? `[${d.code}] ` : "";
    const loc = d.line ? ` @${d.line}:${d.column || 0}` : "";
    lines.push(`  ${tag}${loc} ${code}${d.message}`);
  }
  return lines.join("\n");
}

function setStatus(phase, message) {
  statusEl.className = `status ${phase}`;
  statusEl.textContent = message ? `${phase} — ${message}` : phase;
}

function setWsStatus(text, klass) {
  wsStatusEl.textContent = text;
  wsStatusEl.className = `legend-text ${klass || ""}`;
}

// ─── load samples ──────────────────────────────────────────────────────
let allPrompts = [];   // cached list of {id, title, prompt_len}
const MIN_PROMPT_LEN = 800;  // filter: hide trivially-short problems

function rebuildPromptSelect() {
  const orderEl = document.getElementById("prompt-order");
  const order = orderEl ? orderEl.value : "id";
  // Custom prompts always pass the filter — they're hand-picked, short on
  // purpose. Only HumanEval entries get the >MIN_PROMPT_LEN treatment.
  const filtered = allPrompts.filter(
    (p) => p.is_custom || (p.prompt_len || 0) > MIN_PROMPT_LEN
  );
  if (order === "length") {
    filtered.sort((a, b) => (b.prompt_len || 0) - (a.prompt_len || 0));
  }
  promptSelect.innerHTML = "";
  const blank = document.createElement("option");
  blank.value = ""; blank.textContent = "— free text —";
  promptSelect.appendChild(blank);
  for (const p of filtered) {
    const opt = document.createElement("option");
    opt.value = p.id;
    const lenTag = p.prompt_len != null ? ` [${p.prompt_len}ch]` : "";
    opt.textContent = `${p.id.replace(/^HumanEval_/, "HE_")}: ${p.title}${lenTag}`;
    promptSelect.appendChild(opt);
  }
}

async function loadPrompts() {
  try {
    const lang = currentLang();
    const res = await fetch(`/api/prompts?lang=${encodeURIComponent(lang)}`);
    allPrompts = await res.json();
    rebuildPromptSelect();
    // Auto-select the first non-blank entry actually in the dropdown.
    if (promptSelect.options.length > 1) {
      promptSelect.selectedIndex = 1;
      await onPromptSelected();
    } else {
      // Non-Rust languages may have only the curated 3-prompt set, in
      // which case the MIN_PROMPT_LEN filter trims everything. Re-show
      // all custom prompts (they're short by design) when that happens.
      promptEditor.value = "";
    }
  } catch (e) {
    console.error("loadPrompts failed:", e);
  }
}

async function onPromptSelected() {
  const id = promptSelect.value;
  if (!id) return;
  try {
    const lang = currentLang();
    const res = await fetch(
      `/api/prompts/${encodeURIComponent(id)}?lang=${encodeURIComponent(lang)}`
    );
    const obj = await res.json();
    if (obj.prompt) {
      promptEditor.value = obj.prompt;
    }
  } catch (e) {
    console.error(e);
  }
}

// ─── WebSocket ─────────────────────────────────────────────────────────
function openWs() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws.onopen = () => setWsStatus("connected", "connected");
  state.ws.onclose = () => setWsStatus("disconnected", "");
  state.ws.onerror = () => setWsStatus("ws error", "error");
  state.ws.onmessage = (msg) => {
    try { handleEvent(JSON.parse(msg.data)); }
    catch (e) { console.error("bad event", msg.data); }
  };
}

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
  } else {
    setWsStatus("not connected", "error");
  }
}

// ─── pipeline capsules ─────────────────────────────────────────────────
let pipeSeq = 0;

function blockCenterX(blockEl) {
  // Center X of a pipeline block, expressed in pixels relative to the track.
  const trackRect = pipelineTrack.getBoundingClientRect();
  const r = blockEl.getBoundingClientRect();
  return (r.left + r.right) / 2 - trackRect.left;
}

function spawnCapsule(label, klass, fromBlock, toBlock, duration_ms) {
  const cap = document.createElement("div");
  cap.className = `pipe-capsule ${klass || ""}`;
  cap.textContent = label;
  // Stagger vertically so concurrent capsules don't overlap.
  const top = 6 + (pipeSeq % 5) * 14;
  pipeSeq++;
  const startX = blockCenterX(fromBlock);
  const endX = blockCenterX(toBlock);
  cap.style.setProperty("--start-x", `${startX}px`);
  cap.style.setProperty("--end-x", `${endX}px`);
  cap.style.setProperty("--top", `${top}px`);
  cap.style.setProperty("--dur", `${duration_ms || 600}ms`);
  // Offset for hit-testing the center of the capsule on the start position.
  cap.style.transform = "translateX(-50%)";
  pipelineTrack.appendChild(cap);
  cap.addEventListener("animationend", () => cap.remove());
}

// Convenience: short visible label for tokens that are mostly whitespace.
function pipeTokenLabel(text) {
  const { display } = visualizeToken(text);
  return display.length > 12 ? display.slice(0, 11) + "…" : display;
}

// Render whitespace tokens with visible glyphs so the LLM panel doesn't
// look like a column of empty bubbles. Returns the visible string + a
// boolean indicating whether the token was non-empty / printable.
function visualizeToken(text) {
  if (text === "") return { display: "∅", isEmpty: true };
  // Replace control chars with visible substitutes.
  let out = text
    .replace(/\n/g, "↵")
    .replace(/\r/g, "↵")
    .replace(/\t/g, "→");
  // If the original token is pure whitespace, render runs of spaces as
  // middle-dots so they're visible.
  if (/^\s+$/.test(text)) {
    out = out.replace(/ /g, "·");
  }
  return { display: out, isEmpty: false };
}

function ingestToken(text, endOffset) {
  // Code-panel span (preserves whitespace because parent <pre> does).
  const codeSpan = document.createElement("span");
  codeSpan.className = "tok";
  codeSpan.dataset.offset = endOffset;
  codeSpan.textContent = text;
  codeBody.appendChild(codeSpan);

  // LLM-panel pill — display whitespace with visible glyphs.
  const { display } = visualizeToken(text);
  const pill = document.createElement("span");
  pill.className = "token-pill";
  pill.dataset.offset = endOffset;
  pill.textContent = display;
  // Tooltip = JSON-quoted original (so users can see what's really there).
  pill.setAttribute("title", JSON.stringify(text));
  if (/^\s+$/.test(text) || text === "") pill.classList.add("token-pill-ws");
  llmStack.appendChild(pill);

  // A code token arriving "anchors" any pending thinking pills — they
  // belong to the thought run that produced this code, and should share
  // its fate under rollback (see `applyRollback`).
  if (state.unanchoredThinkPills && state.unanchoredThinkPills.length) {
    for (const tp of state.unanchoredThinkPills) {
      tp.dataset.anchorOffset = String(endOffset);
    }
    state.unanchoredThinkPills.length = 0;
  }

  _trimLlmStack();

  state.tokensByEndOffset.set(endOffset, { text, codeSpan, llmPill: pill });
  state.acceptedCode += text;
  state.tokens += 1;
  tokenCounter.textContent = state.tokens;
  codeBody.scrollTop = codeBody.scrollHeight;
  // Scroll the LLM stack's parent to show the latest pill (newest is at end).
  llmStack.parentElement.scrollTop = llmStack.parentElement.scrollHeight;
}

function ingestThinkToken(text) {
  // Thinking tokens render in the LLM pane only — they never advance the
  // code buffer, never become checkpoints, and never get a verdict. They
  // sit "unanchored" until the next code token claims them; on rollback
  // a thinking pill is marked discarded iff its anchor is in the
  // discarded range (see `applyRollback`).
  const { display } = visualizeToken(text);
  const pill = document.createElement("span");
  // `<think>` / `</think>` are synthesised by the server to bracket each
  // thinking run — distinguish them visually from thinking-body tokens so
  // the user can read the structure.
  const isMarker = text === "<think>" || text === "</think>";
  pill.className = "token-pill " + (isMarker ? "tok-think-marker" : "tok-think");
  pill.textContent = display;
  pill.setAttribute("title", JSON.stringify(text));
  if (/^\s+$/.test(text) || text === "") pill.classList.add("token-pill-ws");
  llmStack.appendChild(pill);

  state.unanchoredThinkPills = state.unanchoredThinkPills || [];
  state.unanchoredThinkPills.push(pill);

  _trimLlmStack();
  llmStack.parentElement.scrollTop = llmStack.parentElement.scrollHeight;
}

function _trimLlmStack() {
  // Cap at 200 pills (was 80). Thinking traces can be long; we want enough
  // history to still see the latest code tokens in context.
  while (llmStack.children.length > 200) {
    const first = llmStack.firstChild;
    // If we trimmed an unanchored thinking pill, drop it from the tracking
    // list too so it doesn't get orphaned-anchored later.
    if (state.unanchoredThinkPills) {
      const i = state.unanchoredThinkPills.indexOf(first);
      if (i >= 0) state.unanchoredThinkPills.splice(i, 1);
    }
    llmStack.removeChild(first);
  }
}

function colorTokenAt(endOffset, verdict, tooltip) {
  // Look up the token whose ending offset matches the verdict's offset.
  // `verdict` ∈ {"ok", "error", "inactive"}; non-ckpt stays unstyled.
  const entry = state.tokensByEndOffset.get(endOffset);
  if (!entry) return;
  if (verdict === "ok") {
    entry.codeSpan.classList.add("tok-ok");
    entry.llmPill.classList.add("tok-ok");
  } else if (verdict === "error") {
    entry.codeSpan.classList.add("tok-error");
    entry.llmPill.classList.add("tok-error");
  } else {
    // inactive → leave white. Still attach tooltip for diagnosability.
  }
  if (tooltip) {
    entry.codeSpan.setAttribute("title", tooltip);
    entry.llmPill.setAttribute("title", tooltip);
  }
}

function applyRollback(toOffset, discarded) {
  state.rollbacks += 1;
  rollbackCounter.textContent = state.rollbacks;
  // Code-panel: discarded tokens get the strikedown animation (line-through,
  // then drop + fade), then are removed from the DOM. The user briefly sees
  // what's being struck out before it disappears, matching the live buffer
  // state (which has already been truncated to `toOffset` server-side).
  // LLM-panel: tokens past the survivor are NOT removed — they stay visible
  // marked `tok-discarded` (dark red, struck through), so the history of
  // what got rolled back is auditable.
  const toDrop = [];
  const STRIKE_DURATION_MS = 1100;
  for (const [offset, entry] of state.tokensByEndOffset) {
    if (offset > toOffset) {
      toDrop.push(offset);
      const span = entry.codeSpan;
      span.classList.remove("tok-ok", "tok-error");
      span.classList.add("tok-strikedown");
      setTimeout(() => { try { span.remove(); } catch (_) {} }, STRIKE_DURATION_MS);
      entry.llmPill.classList.remove("tok-ok", "tok-error");
      entry.llmPill.classList.add("tok-discarded");
    }
  }
  for (const offset of toDrop) state.tokensByEndOffset.delete(offset);

  // Thinking pills: a `<think>` block belongs to the FIRST code token that
  // followed it (their `anchor_offset`). If that anchor is in the discarded
  // range the whole block is discarded too; otherwise it stays clean.
  // Unanchored pills (in-flight thinking that hasn't produced code yet)
  // are left alone — they may still belong to a code token yet to arrive.
  for (const pill of llmStack.querySelectorAll(".tok-think, .tok-think-marker")) {
    const a = pill.dataset.anchorOffset;
    if (a != null && parseInt(a, 10) > toOffset) {
      pill.classList.add("tok-discarded");
    }
  }

  state.acceptedCode = state.acceptedCode.slice(0, toOffset);
}

// Two parallel merge tables:
//   `lspEntriesByOffset` — same buffer offset → same entry (check + verdict).
//   `lspEntriesByBody`   — same rendered body text → same entry (cargo
//                          repeatedly emitting the same diagnostic on
//                          successive boundary checks).
const lspEntriesByOffset = new Map();
const lspEntriesByBody   = new Map();

function _detachEntryFromMaps(el) {
  for (const [off, e] of lspEntriesByOffset) {
    if (e === el) { lspEntriesByOffset.delete(off); break; }
  }
  for (const [body, e] of lspEntriesByBody) {
    if (e === el) { lspEntriesByBody.delete(body); break; }
  }
}

function _trimLspLog() {
  while (lspBody.children.length > 60) {
    const removed = lspBody.firstChild;
    _detachEntryFromMaps(removed);
    if (pinnedLspEntry === removed) pinnedLspEntry = null;
    lspBody.removeChild(removed);
  }
}

function _buildLspEntry(verdict, meta, body, offset, snapshot) {
  const entry = document.createElement("div");
  entry.className = `lsp-entry ${verdict}`;
  if (offset != null) entry.dataset.offset = offset;
  if (snapshot != null) {
    entry._codeSnapshot = snapshot;
    entry._verdict = verdict;
    entry._snapOffset = offset;
  }
  const metaSpan = document.createElement("div");
  metaSpan.className = "meta";
  metaSpan.textContent = meta;
  const bodySpan = document.createElement("div");
  bodySpan.className = "body";
  bodySpan.textContent = body;
  entry.appendChild(metaSpan);
  entry.appendChild(bodySpan);
  return entry;
}

function _removeIfPresent(el) {
  if (!el) return;
  if (pinnedLspEntry === el) pinnedLspEntry = null;
  el.remove();
  _detachEntryFromMaps(el);
}

function _scrollLspToBottom() {
  // Auto-scroll only if the user is already at (or near) the bottom — if
  // they've scrolled up to inspect an older entry, don't yank the view.
  const slack = 40;
  const atBottom = lspBody.scrollTop + lspBody.clientHeight >= lspBody.scrollHeight - slack;
  if (atBottom) lspBody.scrollTop = lspBody.scrollHeight;
}

function logLsp(verdict, message, code, meta, offset, snapshot) {
  // Stand-alone log entry — dedupes by BOTH body and offset so cargo's
  // repeated emissions and back-to-back rollbacks targeting the same
  // checkpoint collapse to a single "freshened" entry at the bottom.
  const body = code ? `${code}: ${message}` : message;
  _removeIfPresent(lspEntriesByBody.get(body));
  if (offset != null) {
    _removeIfPresent(lspEntriesByOffset.get(offset));
  }
  const entry = _buildLspEntry(verdict, meta || verdict.toUpperCase(), body, offset, snapshot);
  lspBody.appendChild(entry);
  lspEntriesByBody.set(body, entry);
  if (offset != null) {
    lspEntriesByOffset.set(offset, entry);
  }
  _trimLspLog();
  _scrollLspToBottom();
}

function upsertLspEntry(offset, verdict, meta, body, snapshot) {
  // Two-axis merge:
  //   1) Same offset → same entry (check_started → verdict at the same place).
  //   2) Same body text → same entry (cargo re-emits the same dead_code /
  //      unused_variables warning on every successive boundary check;
  //      collapse all those duplicates into one entry that "freshens" to
  //      the bottom of the log each time).
  _removeIfPresent(lspEntriesByOffset.get(offset));
  _removeIfPresent(lspEntriesByBody.get(body));
  const entry = _buildLspEntry(verdict, meta, body, offset, snapshot);
  lspBody.appendChild(entry);
  lspEntriesByOffset.set(offset, entry);
  lspEntriesByBody.set(body, entry);
  _trimLspLog();
  _scrollLspToBottom();
  return entry;
}

// ─── events from server ────────────────────────────────────────────────
function handleEvent(ev) {
  switch (ev.type) {
    case "status":
      setStatus(ev.phase, ev.message);
      if (ev.phase === "ready" || ev.phase === "generating") {
        state.isRunning = true;
        startBtn.disabled = true; stopBtn.disabled = false;
      } else if (ev.phase === "done" || ev.phase === "error") {
        state.isRunning = false;
        startBtn.disabled = false; stopBtn.disabled = true;
      }
      break;
    case "reset_buffer":
      state.acceptedCode = "";
      state.tokensByEndOffset.clear();
      if (state.unanchoredThinkPills) state.unanchoredThinkPills.length = 0;
      codeBody.textContent = "";
      llmStack.innerHTML = "";
      break;
    case "token_emitted": {
      // Server-emitted tokens carry `kind`: "code" or "think".
      // Thinking tokens render in the LLM pane only and do NOT flow through
      // the LLM → CODE pipeline capsule (they never become code).
      const kind = ev.kind || "code";
      if (kind === "think") {
        ingestThinkToken(ev.text);
      } else {
        // ev.offset is the buffer length AFTER this token is appended.
        ingestToken(ev.text, ev.offset);
        spawnCapsule(pipeTokenLabel(ev.text), "token",
                     pipeLlmBlock, pipeCodeBlock, 500);
      }
      break;
    }
    case "check_started": {
      const rc = rcLabel(state.acceptedCode, ev.offset);
      spawnCapsule(`tok ${rc}`, "token", pipeCodeBlock, pipeLspBlock, 500);
      // No log entry — the verdict that follows shortly is the informative
      // event. The capsule animation is enough visual cue that a check is
      // in flight.
      break;
    }
    case "verdict": {
      const rc = rcLabel(state.acceptedCode, ev.offset);
      // Pipeline: LSP → CODE with the verdict label.
      spawnCapsule(`${ev.verdict}${rc}`, ev.verdict, pipeLspBlock, pipeCodeBlock, 700);
      // Build a tooltip from diagnostics and attach to the boundary token.
      const tooltip = formatDiagsTooltip(ev.verdict, ev.diagnostics, null);
      colorTokenAt(ev.offset, ev.verdict, tooltip);
      // Collapse ALL diagnostics into one merged entry per offset, and
      // deduplicate body lines so repeats of the same (code, line, col,
      // message) at the same offset don't show as visual duplicates.
      const kind = ev.verdict === "ok" ? "ok"
                 : ev.verdict === "error" ? "error" : "inactive";
      const meta = `${ev.verdict.toUpperCase()} ${rc}`;
      const lines = [];
      const seenLine = new Set();
      if (ev.diagnostics && ev.diagnostics.length) {
        for (const d of ev.diagnostics) {
          const cat = (d.category || "?").toUpperCase();
          const loc = d.line ? ` (${d.line},${d.column || 0})` : "";
          const text = `[${cat}]${loc} ${d.code ? d.code + ": " : ""}${d.message}`;
          if (seenLine.has(text)) continue;
          seenLine.add(text);
          lines.push(text);
        }
      } else {
        lines.push("(no diagnostics)");
      }
      upsertLspEntry(ev.offset, kind, meta, lines.join("\n"), ev.code_snapshot);
      break;
    }
    case "checkpoint": {
      // Bump the counter but don't add a log entry — the OK verdict at the
      // same offset already conveys "this position is a checkpoint."
      state.checkpoints += 1;
      checkpointCounter.textContent = state.checkpoints;
      break;
    }
    case "rollback": {
      const rcFrom = rcLabel(state.acceptedCode, ev.to_offset + (ev.discarded || "").length);
      const rcTo = rcLabel(state.acceptedCode, ev.to_offset);
      const label = `rollback ${rcFrom}→${rcTo}`;
      spawnCapsule(label, "rollback", pipeLspBlock, pipeCodeBlock, 600);
      setTimeout(() =>
        spawnCapsule(label, "rollback", pipeCodeBlock, pipeLlmBlock, 600), 250);
      // Attach the pre-rollback snapshot (truncated code + discarded suffix)
      // so clicking the entry shows what was just discarded.
      logLsp("rollback",
             `rollback → offset ${ev.to_offset}, discarded ${ev.discarded.length} chars`,
             null, `ROLLBACK #${ev.rollback_count} ${rcFrom}→${rcTo}`,
             ev.to_offset, ev.code_snapshot);
      applyRollback(ev.to_offset, ev.discarded);
      break;
    }
    case "final":
      logLsp("ok",
             `final length ${ev.content.length}, rollbacks ${ev.rollback_count}, `
             + `continuations ${ev.continuations || 0}`,
             null, "FINAL", null, ev.code_snapshot);
      break;
    case "continuation":
      logLsp("inactive",
             `continuation attempt ${ev.attempt}/${ev.of}`,
             null, "CONTINUATION", null, ev.code_snapshot);
      break;
    case "final_check": {
      // Post-hoc cargo check on the full body (no unreachable!() shim).
      // Surface whether the solution is actually complete + compiles.
      const ok = ev.is_complete;
      const klass = ok ? "ok" : "error";
      const summary = ok
        ? "post-hoc cargo check: complete & compiles"
        : `post-hoc cargo check: INCOMPLETE — ${ev.blocking_count || 0} blocking error(s), ${ev.incomplete_signal_count || 0} structural signal(s)`;
      logLsp(klass, summary, null, "FINAL CHECK", null, ev.code_snapshot);
      setStatus(ok ? "done" : "error", summary);
      for (const d of (ev.diagnostics || [])) {
        const dkClass = d.category === "blocking" ? "error"
                      : d.category === "non_blocking" ? "ok"
                      : "inactive";
        const where = d.line ? ` (${d.line},${d.column || 0})` : "";
        logLsp(dkClass,
               `${d.code ? d.code + ": " : ""}${d.message}`,
               null,
               `POST-HOC ${d.category.toUpperCase()}${where}`,
               null, ev.code_snapshot);
      }
      break;
    }
    case "backend_stats":
      // KV-cache evidence from the vLLM backend. Reveal the panel on first
      // event; update every ~32 tokens / every rollback thereafter.
      if (backendStatsEl) {
        backendStatsEl.hidden = false;
        if (bsHitPctEl) {
          const pct = (ev.cache_hit_pct != null) ? ev.cache_hit_pct.toFixed(1) : "—";
          bsHitPctEl.textContent = `${pct}%`;
        }
        if (bsCachedEl) bsCachedEl.textContent = (ev.cached_tokens != null) ? String(ev.cached_tokens) : "—";
        if (bsPrefixEl) bsPrefixEl.textContent = (ev.prefix_tokens != null) ? String(ev.prefix_tokens) : "—";
        if (bsTtftEl) {
          if (ev.ttft_s != null) {
            const ms = ev.ttft_s * 1000;
            bsTtftEl.textContent = ms < 1000
              ? `${ms.toFixed(0)} ms`
              : `${(ms / 1000).toFixed(2)} s`;
          } else {
            bsTtftEl.textContent = "—";
          }
        }
        if (bsRollbackBadgeEl) {
          bsRollbackBadgeEl.hidden = !ev.is_rollback;
        }
      }
      break;
    case "error":
      setStatus("error", ev.message || "error");
      logLsp("error", ev.message, null, "ERROR");
      break;
    default:
      console.warn("unknown event", ev);
  }
}

// ─── button wiring ────────────────────────────────────────────────────
function resetUi() {
  pinLspEntry(null);  // clear any persistent preview
  lspEntriesByOffset.clear();
  lspEntriesByBody.clear();
  state.acceptedCode = "";
  state.tokensByEndOffset.clear();
  if (state.unanchoredThinkPills) state.unanchoredThinkPills.length = 0;
  state.tokens = 0;
  state.rollbacks = 0;
  state.checkpoints = 1;
  tokenCounter.textContent = "0";
  rollbackCounter.textContent = "0";
  checkpointCounter.textContent = "1";
  codeBody.textContent = "";
  lspBody.innerHTML = "";
  llmStack.innerHTML = "";
  if (pipelineTrack) pipelineTrack.innerHTML = "";
  // Hide the backend-stats panel until the next backend_stats event
  // arrives — Ollama runs never populate it, so the panel should not
  // linger with stale numbers from a previous vLLM run.
  if (backendStatsEl) {
    backendStatsEl.hidden = true;
    if (bsRollbackBadgeEl) bsRollbackBadgeEl.hidden = true;
  }
  setStatus("idle", "");
}

function onStart() {
  const prompt = promptEditor.value.trim();
  if (!prompt) { alert("Prompt is empty."); return; }
  resetUi();
  modelReadout.textContent = modelSelect.value;
  const req = {
    type: "start",
    prompt,
    language: currentLang(),
    verifier: verifierSelect.value,
    // Back-compat defaults if the new selectors aren't in the DOM (e.g. an
    // older index.html cached in the browser): server reads "ollama" /
    // "soundcode" when these fields are missing, so we don't have to
    // bother sending them — but we do, for the explicit-config log entry.
    backend: backendSelect ? backendSelect.value : "ollama",
    algorithm: algorithmSelect ? algorithmSelect.value : "soundcode",
    model: modelSelect.value,
    token_delay_ms: parseInt(cadenceInput.value, 10) || 0,
    instruct: instructCheck.checked,
    mode: modeSelect ? modeSelect.value : "raw",
  };
  send(req);
  setStatus("loading", "starting…");
  startBtn.disabled = true; stopBtn.disabled = false;
}

function onStop() {
  send({ type: "stop" });
  startBtn.disabled = false; stopBtn.disabled = true;
}

function onReset() {
  send({ type: "stop" });
  resetUi();
  startBtn.disabled = false; stopBtn.disabled = true;
}

function onCadenceInput() {
  const v = parseInt(cadenceInput.value, 10);
  cadenceLabel.textContent = v === 0 ? "real-time" : `${v} ms / token`;
}

// ─── init ──────────────────────────────────────────────────────────────
function init() {
  startBtn.addEventListener("click", onStart);
  stopBtn.addEventListener("click", onStop);
  resetBtn.addEventListener("click", onReset);
  cadenceInput.addEventListener("input", onCadenceInput);
  promptSelect.addEventListener("change", onPromptSelected);
  if (modeSelect) {
    modeSelect.addEventListener("change", _updateModeHint);
    modelSelect.addEventListener("change", _refreshModeOptionsForModel);
    if (instructCheck) instructCheck.addEventListener("change", _updateModeHint);
    _refreshModeOptionsForModel();
  }
  const orderEl = document.getElementById("prompt-order");
  if (orderEl) orderEl.addEventListener("change", rebuildPromptSelect);
  // Cross-panel hover highlight: any token/log-entry in these three regions
  // highlights its counterparts in the others.
  hookHoverHighlight(codeBody);
  hookHoverHighlight(llmStack);
  hookHoverHighlight(lspBody);
  // LSP-entry hover → ephemeral preview (only if nothing is pinned).
  lspBody.addEventListener("mouseover", (e) => {
    if (pinnedLspEntry) return;
    const t = e.target.closest(".lsp-entry");
    if (!t) return;
    const snap = t._codeSnapshot;
    if (snap == null) return;
    showCodePreview(snap, t._snapOffset, t._verdict);
  });
  lspBody.addEventListener("mouseout", (e) => {
    if (pinnedLspEntry) return;
    const t = e.target.closest(".lsp-entry");
    if (!t) return;
    if (t.contains(e.relatedTarget)) return;
    hideCodePreview();
  });
  // LSP-entry click → pin/unpin the preview for that entry.
  lspBody.addEventListener("click", (e) => {
    const t = e.target.closest(".lsp-entry");
    if (!t) return;
    if (t._codeSnapshot == null) return;  // nothing to show
    if (pinnedLspEntry === t) {
      pinLspEntry(null);  // toggle off
    } else {
      pinLspEntry(t);
    }
  });
  // Escape clears the pin.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && pinnedLspEntry) {
      pinLspEntry(null);
    }
  });
  onCadenceInput();
  stopBtn.disabled = true;
  // Language change re-loads the prompt list and updates the syntax-class
  // on the code panel. Defer the prompt fetch slightly so a fast user
  // doesn't fire two in flight in parallel.
  if (langSelect) {
    langSelect.addEventListener("change", async () => {
      applyLanguageClass();
      await loadPrompts();
    });
    applyLanguageClass();
  }
  loadPrompts();
  openWs();
  setStatus("idle", "");
}
document.addEventListener("DOMContentLoaded", init);
