/* ============================================================
 *  Animated walkthrough above the static flow chart.
 *  Plays one full cycle (tokens → ckpt → Ok, then tokens → ckpt
 *  → Error → strikedown → re-generate → Ok), then loops.
 *
 *  Coordinates are PERCENTAGES of the .anim-flow-stage container,
 *  which matches the 1000×400 SVG viewBox via aspect-ratio.
 *  Module centres in viewBox: LLM (180,85), Code (500,85), LSP (820,85).
 *  → as percentages: LLM (18%, 21.25%), Code (50%, 21.25%), LSP (82%, 21.25%).
 * ============================================================ */

(function () {
  const POS = {
    llm:  { x: 18.0, y: 21.25 },
    code: { x: 50.0, y: 21.25 },
    lsp:  { x: 82.0, y: 21.25 },
    off:  { x: 108.0, y: 21.25 },   // off-right, used for "Ok → nowhere"
  };

  const T_TOKEN   = 700;   // ms per token capsule flight
  const T_TOK_GAP = 110;   // stagger between successive tokens in a burst
  const T_BURST   = 5 * T_TOK_GAP + T_TOKEN;  // wall time of a 6-token burst
  const T_CKPT    = 900;   // ckpt capsule flight time
  const T_VERDICT = 900;   // verdict capsule flight time

  function $(id) { return document.getElementById(id); }

  function init() {
    const overlay = $("anim-overlay");
    const stage   = document.querySelector(".anim-flow-stage");
    if (!overlay || !stage) return;

    const slots = [$("anim-slot-0"), $("anim-slot-1"), $("anim-slot-2")];
    const llmBlock  = $("anim-llm");
    const codeBlock = $("anim-code");

    function spawnCap(text, klass, from, to, dur) {
      const cap = document.createElement("div");
      cap.className = "anim-cap " + klass;
      cap.textContent = text;
      cap.style.left = from.x + "%";
      cap.style.top  = from.y + "%";
      cap.style.transitionDuration = dur + "ms, " + dur + "ms, 200ms";
      overlay.appendChild(cap);
      // Force layout then trigger the appearance + the move on the next frame.
      requestAnimationFrame(() => {
        cap.classList.add("appeared");
        requestAnimationFrame(() => {
          cap.style.left = to.x + "%";
          cap.style.top  = to.y + "%";
        });
      });
      setTimeout(() => { cap.classList.add("fading"); }, dur - 50);
      setTimeout(() => { try { cap.remove(); } catch (_) {} }, dur + 350);
    }

    function tokenBurst(startMs) {
      for (let i = 0; i < 6; i++) {
        setTimeout(() => spawnCap("tok", "cap-token",
                                   POS.llm, POS.code, T_TOKEN),
                   startMs + i * T_TOK_GAP);
      }
    }

    function ckpt(startMs, label) {
      setTimeout(() => spawnCap(label, "cap-ckpt",
                                 POS.code, POS.lsp, T_CKPT),
                 startMs);
    }

    function verdictOk(startMs, label) {
      setTimeout(() => spawnCap(label, "cap-ok",
                                 POS.lsp, POS.off, T_VERDICT),
                 startMs);
    }

    function verdictErrorFanout(startMs, label) {
      setTimeout(() => {
        spawnCap(label, "cap-error", POS.lsp, POS.llm,  T_VERDICT);
        spawnCap(label, "cap-error", POS.lsp, POS.code, T_VERDICT);
      }, startMs);
    }

    function towerAdd(idx, startMs) {
      setTimeout(() => slots[idx].classList.add("visible"), startMs);
    }
    function towerStrike(idx, startMs) {
      setTimeout(() => slots[idx].classList.add("struck"), startMs);
    }
    function towerRemove(idx, startMs) {
      setTimeout(() => {
        slots[idx].classList.remove("visible");
        slots[idx].classList.remove("struck");
      }, startMs);
    }
    function towerClearAll(startMs) {
      setTimeout(() => slots.forEach(s => {
        s.classList.remove("visible");
        s.classList.remove("struck");
      }), startMs);
    }

    function shake(blockEl, startMs, extraClass) {
      setTimeout(() => {
        blockEl.classList.remove("shake");
        // restart animation by forcing reflow
        void blockEl.getBoundingClientRect();
        blockEl.classList.add("shake");
        if (extraClass) blockEl.classList.add(extraClass);
      }, startMs);
      setTimeout(() => {
        blockEl.classList.remove("shake");
        if (extraClass) blockEl.classList.remove(extraClass);
      }, startMs + 700);
    }

    function shout(text, where, startMs) {
      setTimeout(() => {
        const el = document.createElement("div");
        el.className = "anim-shout";
        el.textContent = text;
        // Sit above the centre of the target module (subtract ~10% from y).
        el.style.left = where.x + "%";
        el.style.top  = (where.y - 12) + "%";
        overlay.appendChild(el);
        setTimeout(() => { try { el.remove(); } catch (_) {} }, 1700);
      }, startMs);
    }

    /* ─── timeline ──────────────────────────────────────
     * Steps mirror the storyboard in the README:
     *   1. 6 tokens → Code, add Code[0]
     *   2. ckpt Code[0] → LSP                     (concurrent with 3)
     *   3. 6 more tokens → Code, add Code[1]
     *   4. Ok(0) → off
     *   5. 6 tokens → Code, add Code[2]
     *   6. ckpt Code[1] → LSP                     (lagging Code[1])
     *   7. 6 more tokens → Code (will get struck)
     *   8. Error(1) → LLM + Code → shake + strikedown, lose [1] [2]
     *   9. re-generate [1] and [2], both Ok this time
     * --------------------------------------------------- */
    function runCycle() {
      let t = 0;
      // step 1: tokens → Code[0]
      tokenBurst(t);
      const t1End = t + T_BURST;
      towerAdd(0, t1End + 100);

      // step 2: ckpt(0) → LSP
      const t2 = t1End + 350;
      ckpt(t2, "Code[0]");

      // step 3 (concurrent with 2): tokens → Code[1]
      tokenBurst(t2);
      const t3End = t2 + T_BURST;
      towerAdd(1, t3End + 100);

      // step 4: after step 2's ckpt arrives, Ok(0) → off
      const t4 = t2 + T_CKPT + 200;
      verdictOk(t4, "Ok(0)");

      // step 5: tokens → Code[2]
      const t5 = Math.max(t3End + 350, t4 + 250);
      tokenBurst(t5);
      const t5End = t5 + T_BURST;
      towerAdd(2, t5End + 100);

      // step 6: ckpt(1) → LSP   (the one that will fail)
      const t6 = t5 + 250;
      ckpt(t6, "Code[1]");

      // step 7 (concurrent with 5): more tokens (will get struck)
      tokenBurst(t5 + 350);

      // step 8: Error(1) returns, fans out to LLM + Code
      const t8 = t6 + T_CKPT + 200;
      verdictErrorFanout(t8, "Error(1)");
      const t8Hit = t8 + T_VERDICT;
      shake(llmBlock,  t8Hit, "error");
      shake(codeBlock, t8Hit, "error");
      shout("Startover(1)", POS.llm,  t8Hit);
      shout("Strikedown(1)", POS.code, t8Hit);
      // Strike out the lost rows for ~600 ms, then remove them.
      towerStrike(1, t8Hit);
      towerStrike(2, t8Hit);
      towerRemove(1, t8Hit + 700);
      towerRemove(2, t8Hit + 700);

      // step 9: regenerate [1] and [2], both Ok
      const t9 = t8Hit + 1300;
      // tokens → Code[1]
      tokenBurst(t9);
      const t9aEnd = t9 + T_BURST;
      towerAdd(1, t9aEnd + 100);
      const t9b = t9aEnd + 350;
      ckpt(t9b, "Code[1]");
      verdictOk(t9b + T_CKPT + 200, "Ok(1)");
      // tokens → Code[2]
      const t9c = t9b + 350;
      tokenBurst(t9c);
      const t9cEnd = t9c + T_BURST;
      towerAdd(2, t9cEnd + 100);
      const t9d = t9cEnd + 350;
      ckpt(t9d, "Code[2]");
      verdictOk(t9d + T_CKPT + 200, "Ok(2)");

      const total = t9d + T_CKPT + T_VERDICT + 1400;
      // Pause ~1.6s with all three rows visible, then clear and loop.
      setTimeout(() => towerClearAll(0), total - 200);
      setTimeout(runCycle, total + 200);
    }

    // Kick off the first cycle once the page has settled.
    setTimeout(runCycle, 600);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
