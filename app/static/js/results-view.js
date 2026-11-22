// Screen 3, Results: one headline sentence from the run's own test numbers, two to four before/after tiles, calm notes
// from the data, "Watch before" and "Watch after" on the stage (the episode player plays each test with the baseline,
// then with what it learned; the caption says which is on stage), and the full numbers behind a disclosure.
import { numbersTable, resultHeadline, resultNotes, resultTiles, skillKey, skillTitle } from "./plain.js";
import { $, el, setText } from "./dom.js";

const ARROW = { better: "Better", worse: "Worse", same: "No change" };

export function initResultsView(ctx) {
  const { api, notes } = ctx;
  let runId = null, body = null, loading = false, active = false, watching = null, phase = null, asked = null;

  async function enter(route) {
    active = true;
    const id = route.runId ?? ctx.openRun();
    runId = id;
    if (!id) return render();
    ctx.setOpenRun(id);
    loading = true;
    body = null;
    render();
    const b = await ctx.runBody(id);
    if (!active || runId !== id) return;
    loading = false;
    body = b;
    render();
  }

  function leave() {
    active = false;
    if (watching) {
      watching = null;
      api.episodesStop().then((r) => { if (!r.ok) notes.set("episodes", "warn", `Stop episodes: ${r.message}`, "Episodes"); });
    }
  }

  async function watch(which) {
    asked = which;
    if (watching === runId && ctx.state()?.episodes?.playing) return render();
    const r = await api.episodesStart(runId);
    if (r.ok) { watching = runId; notes.clear("episodes"); ctx.reloadState(); }
    else { notes.set("episodes", r.unavailable ? "warn" : "danger", `This run's episodes cannot be shown: ${r.message} (${r.code})`, "Episodes"); asked = null; }
    render();
  }
  async function stopWatching() {
    watching = null; asked = null;
    await api.episodesStop();
    ctx.reloadState();
    render();
  }

  // The caption on stage says which is playing: before or after learning.
  function onEpisode(v) {
    phase = v?.phase ?? null;
    if (active) renderWatch();
  }

  function render() {
    if (!active) return;
    const results = body?.results;
    const head = results ? resultHeadline(results, typeof results.timing?.attempts === "number" ? results.timing.attempts : null) : null;
    const empty = !runId;
    const none = !empty && !loading && !head;
    $("res-empty").hidden = !(empty || none);
    $("res-main").hidden = empty || none;
    if (empty || none) {
      setText($("res-empty-h"), empty ? "Nothing learned yet." : "This lesson has no test results.");
      setText($("res-empty-p"), empty ? "Teach it something first." : "It may have stopped before its test, or it is still running.");
      $("res-foot").replaceChildren(el("button", { class: "btn btn--accent btn--lg", type: "button", on: { click: () => ctx.go("teach") } }, empty ? "Teach it something" : "Teach something else"));
      if (none && body === null) notes.set("results", "danger", `GET /api/runs/${runId} gave no readable run.`, "Results");
      return;
    }
    notes.clear("results");
    const key = skillKey(results?.skill ?? body?.results?.skill, ctx.catalogue()) ?? skillKey(ctx.cardOf(runId)?.skill, ctx.catalogue());
    setText($("res-eyebrow"), loading ? "" : `${skillTitle(key)} · tested in simulation`);
    setText($("res-h"), loading ? "Reading the test results…" : head.text);
    if (loading) { $("res-tiles").replaceChildren(); $("res-notes").replaceChildren(); $("res-watch").hidden = true; $("res-numbers").hidden = true; $("res-foot").replaceChildren(); return; }
    $("res-tiles").replaceChildren(...resultTiles(results).map((t) => el("li", { class: "tile", "data-change": t.change ?? "none" },
      el("p", { class: "tile__label" }, t.label),
      el("p", { class: "tile__nums" },
        el("span", { class: "tile__before" }, el("span", { class: "sr-only" }, "Before: "), t.before),
        el("span", { class: "tile__arrow", "aria-hidden": "true" }, "→"),
        el("span", { class: "tile__after" }, el("span", { class: "sr-only" }, "After: "), t.after)),
      el("p", { class: "tile__basis" }, [t.change ? ARROW[t.change] : null, t.basis].filter(Boolean).join(" · ")))));
    const calm = resultNotes(results);
    $("res-notes").replaceChildren(...calm.map((n) => el("p", { class: "calm" }, n.text)));
    $("res-notes").hidden = !calm.length;
    renderWatch();
    renderNumbers(results);
    const isDodge = key === "DODGE";
    $("res-foot").replaceChildren(el("button", { class: "btn btn--accent btn--lg", type: "button", on: { click: () => ctx.go(isDodge ? "live" : "teach") } },
      isDodge ? "Try it live" : "Teach something else"));
    $("res-foot").hidden = false;
  }

  function renderWatch() {
    const offered = Boolean(ctx.state()?.episodes) && Boolean(body?.results);
    $("res-watch").hidden = !offered;
    if (!offered) return;
    const on = watching === runId && ctx.state()?.episodes?.playing !== false;
    const btn = (which, label) => el("button", { class: `btn ${on && phase === which ? "btn--accent" : "btn--glass"}`, type: "button", "aria-pressed": String(on && asked === which),
      on: { click: () => watch(which) } }, label, on && phase === which ? el("span", { class: "btn__tag" }, "on stage") : null);
    $("res-watch-row").replaceChildren(btn("before", "Watch before"), btn("after", "Watch after"),
      on ? el("button", { class: "btn btn--quiet btn--sm", type: "button", on: { click: stopWatching } }, "Stop") : null);
    setText($("res-watch-hint"), "Each test plays twice on the stage: first before learning, then after.");
  }

  function renderNumbers(results) {
    const t = numbersTable(results);
    $("res-numbers").hidden = !t.rows.length;
    $("res-numbers-body").replaceChildren(
      el("table", { class: "numbers" },
        el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Measure"), el("th", { scope: "col" }, "Before learning"), el("th", { scope: "col" }, "After learning"))),
        el("tbody", {}, ...t.rows.map((r) => el("tr", {}, el("th", { scope: "row" }, r.label), el("td", {}, r.before), el("td", {}, r.after))))),
      ...(t.paired.length ? [el("p", { class: "numbers__h" }, "Change on the same tests"), el("ul", { class: "numbers__list" }, ...t.paired.map((p) => el("li", {}, p)))] : []),
      el("ul", { class: "numbers__list numbers__list--facts" }, ...t.facts.map((f) => el("li", {}, f))));
  }

  return { enter, leave, onEpisode, runId: () => runId, stateChanged: () => { if (active && !loading) renderWatch(); } };
}
