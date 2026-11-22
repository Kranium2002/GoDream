// The story band under the demo screen, always in view: how the demo on show learned, as three inline steps with the
// numbers of its saved run (real tries, situations imagined by TabPFN, the spiking brain and its training time), and
// how it did: the saved test results before and after, or, for a demo without saved results, this session's real
// rounds. While both lanes run, the steps light up one after another; with reduced motion they stay still.
import { learnedIn, rowValue, sessionRows, storySteps } from "./showcase.js";
import { $, el, setText } from "./dom.js";

const STEP_MS = 3000; // each step stays lit this long while a round runs
const NOUNS = { hills: "hills", dodge: "punches" }; // what one test situation is called, per demo; else "situations"

export function initBandView() {
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let stepsKey = "", resultsKey = "", timer = null, lit = -1;

  function render(demo, { running = false, counts = null } = {}) {
    const band = $("band");
    band.hidden = !demo;
    if (!demo) { cycle(false); return; }
    const l = demo.showcase.learning;
    const steps = storySteps(l), learned = learnedIn(l);
    const sk = JSON.stringify([demo.id, steps, learned]);
    if (sk !== stepsKey) {
      stepsKey = sk;
      $("steps").replaceChildren(...steps.map((s, k) => el("li", { class: "step", "data-key": s.key, "data-n": String(k) },
        s.strong ? el("span", { class: "step__n" }, s.strong) : null, el("span", { class: "step__w" }, s.rest))));
      $("learned-in").hidden = !learned;
      setText($("learned-in"), learned ?? "");
      lit = -1;
    }
    renderResults(demo, counts);
    cycle(running);
  }

  function renderResults(demo, counts) {
    const res = demo.showcase.results, lanes = demo.lanes ?? {};
    const noun = NOUNS[demo.id] ?? "situations";
    let title, rows, empty = null;
    if (res) {
      title = res.n ? `Tested on ${res.n} new ${noun}` : `Tested on new ${noun}`;
      rows = res.rows.map((r) => ({ label: r.label, before: rowValue(r, "before"), after: rowValue(r, "after"), good: true }));
    } else {
      const s = sessionRows(counts);
      title = s.rounds ? `This session: ${s.rounds} ${s.rounds === 1 ? noun.replace(/es$|s$/, "") : noun}` : "This session";
      rows = s.rows;
      if (!s.rounds) empty = "Press Run both to start the count.";
    }
    const key = JSON.stringify([demo.id, title, rows, empty, lanes]);
    if (key === resultsKey) return;
    resultsKey = key;
    setText($("tested-title"), title);
    const table = $("tested-table"), note = $("tested-empty");
    note.hidden = !empty;
    setText(note, empty ?? "");
    table.hidden = !rows.length;
    if (!rows.length) { table.replaceChildren(); return; }
    table.replaceChildren(
      el("thead", {}, el("tr", {}, el("td", {}), el("th", { scope: "col", class: "col-before" }, lanes.before ?? ""), el("th", { scope: "col", class: "col-after" }, lanes.after ?? ""))),
      // The learned lane's number is large where it counts a good result; a session's "Hit 1 of 1" is not drawn like a win.
      el("tbody", {}, ...rows.map((r) => el("tr", {}, el("th", { scope: "row" }, r.label), el("td", { class: "v-before" }, r.before),
        el("td", { class: r.good ? "v-after" : "v-after v-after--plain" }, r.after)))));
  }

  // One calm motion: while a round runs, the steps light up in turn.
  function cycle(running) {
    const band = $("band");
    const go = running && !reduceMotion.matches;
    band.dataset.running = String(go);
    if (go && !timer) {
      const tick = () => { const n = $("steps").children.length || 1; lit = (lit + 1) % n; for (const li of $("steps").children) li.dataset.on = String(Number(li.dataset.n) === lit); };
      tick();
      timer = setInterval(tick, STEP_MS);
    } else if (!go && timer) {
      clearInterval(timer);
      timer = null;
      lit = -1;
      for (const li of $("steps").children) delete li.dataset.on;
    }
  }

  return { render };
}
