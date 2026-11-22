// Screen 2, Learn: the open run as five plain steps with live progress from its events, while the stage plays the run's
// own episodes (the episode player, started on entry and stopped on leaving). A finished run plays back the same way
// from its recorded event log, labelled as recorded. Every number comes from the run's events.
import { applyLine, emptyTimeline, fromRunResponse, lossCurve, trainSeries } from "./learn.js";
import { replayPlan } from "./flow.js";
import { lessonSteps, replaySpeed, skillKey, skillTask, skillTitle } from "./plain.js";
import { $, el, setText, svgEl } from "./dom.js";

const ENDED = ["done", "failed", "refused", "stopped"];
const SPARK = { width: 132, height: 36, pad: 3 };
const count = (n) => Math.round(n).toLocaleString("en-US");

export function initLearnView(ctx) {
  const { api, notes } = ctx;
  const live = new Map();     // run id -> SSE lines received this session (memory only)
  const recorded = new Map(); // run id -> its recorded event log, as read for a replay
  let runId = null, tl = emptyTimeline(), runState = null, loading = false, problem = null;
  let replay = null;          // {id, speed, timers, finished}
  let active = false, playing = null; // the run whose episodes this screen started

  // ---- entering and leaving --------------------------------------------------------------------------------------
  async function enter(route) {
    active = true;
    const id = route.runId ?? ctx.openRun();
    if (!id) { runId = null; return render(); }
    ctx.setOpenRun(id);
    if (id !== runId) { stopReplay(); runId = id; tl = { ...emptyTimeline(), runId: id }; runState = null; problem = null; }
    loading = true;
    render();
    const r = await api.run(id);
    if (!active || runId !== id) return;
    loading = false;
    if (!r.ok) {
      problem = r.unavailable ? "This lesson could not be found." : "This lesson could not be read.";
      notes.set("learn", "danger", `GET /api/runs/${id}: ${r.message} (${r.code})`, "Learn");
      return render();
    }
    notes.clear("learn");
    runState = typeof r.data.state === "string" ? r.data.state : null;
    if (ENDED.includes(runState)) {
      if (!replay || replay.id !== id) await startReplay(id);
    } else {
      tl = (live.get(id) ?? []).reduce(applyLine, fromRunResponse(id, r.data));
    }
    startEpisodes(id);
    render();
  }

  function leave() {
    active = false;
    if (playing) {
      const id = playing;
      playing = null;
      const ep = ctx.state()?.episodes;
      if (!ep || ep.playing !== false) api.episodesStop().then((r) => { if (!r.ok) notes.set("episodes", "warn", `Stop episodes: ${r.message}`, "Episodes"); });
      ctx.episodesStopped?.(id);
    }
  }

  async function startEpisodes(id) {
    if (!ctx.state()?.episodes) return; // the server offers no episode player: the stage shows the live simulation
    if (ctx.state().episodes.playing && ctx.state().episodes.run_id === id) { playing = id; return; }
    const r = await api.episodesStart(id);
    if (!active || runId !== id) { if (r.ok) api.episodesStop(); return; }
    if (r.ok) { playing = id; notes.clear("episodes"); ctx.reloadState(); }
    else notes.set("episodes", r.unavailable ? "warn" : "info", `This run's episodes cannot be shown on the stage: ${r.message} (${r.code})`, "Episodes");
  }

  // ---- a recorded lesson, played back ----------------------------------------------------------------------------
  async function startReplay(id) {
    stopReplay();
    const r = await api.runEvents(id);
    if (!active || runId !== id) return;
    if (!r.ok || !Array.isArray(r.data)) {
      notes.set("replay", "warn", `GET /api/runs/${id}/events: ${r.ok ? "not a list" : r.message}. The final recorded state is shown.`, "Learn");
      const body = await ctx.runBody(id);
      tl = fromRunResponse(id, body ?? {});
      replay = { id, speed: 1, timers: [], finished: true };
      return;
    }
    notes.clear("replay");
    const lines = r.data.filter((l) => l && typeof l === "object");
    recorded.set(id, lines);
    const speed = replaySpeed(lines);
    const plan = replayPlan(lines, speed);
    const mine = replay = { id, speed, timers: [], finished: false };
    tl = { ...emptyTimeline(), runId: id };
    for (const p of plan) mine.timers.push(setTimeout(() => { if (replay === mine) { tl = applyLine(tl, p.line); render(); } }, p.atMs));
    mine.timers.push(setTimeout(() => { if (replay === mine) { mine.finished = true; render(); } }, (plan.at(-1)?.atMs ?? 0) + 400));
  }
  function stopReplay() { if (replay) { replay.timers.forEach(clearTimeout); replay = null; } }
  function skipReplay() {
    if (!replay || replay.finished) return;
    replay.timers.forEach(clearTimeout);
    tl = (recorded.get(replay.id) ?? []).reduce(applyLine, { ...emptyTimeline(), runId: replay.id });
    replay.finished = true;
    render();
  }

  // ---- live lines ------------------------------------------------------------------------------------------------
  async function onRunEvent(line) {
    if (typeof line?.run_id !== "string" || !line.run_id) return;
    if (!live.has(line.run_id)) live.set(line.run_id, []);
    live.get(line.run_id).push(line);
    if (ENDED.includes(line.state)) ctx.cardsMayHaveChanged();
    if (line.run_id !== runId || replay) return;
    tl = applyLine(tl, line);
    if (ENDED.includes(line.state)) {
      const r = await api.run(runId);
      if (r.ok && runId === line.run_id) { runState = r.data.state ?? runState; ctx.forgetRunBody(runId); }
    }
    if (active) render();
  }

  // ---- drawing ---------------------------------------------------------------------------------------------------
  function render() {
    if (!active) return;
    const empty = !runId;
    $("learn-empty").hidden = !empty;
    $("learn-main").hidden = empty;
    if (empty) {
      $("learn-foot").replaceChildren(el("button", { class: "btn btn--accent btn--lg", type: "button", on: { click: () => ctx.go("teach") } }, "Teach it something"));
      return;
    }
    const key = skillKey(tl.skill, ctx.catalogue()) ?? skillKey(ctx.cardOf(runId)?.skill, ctx.catalogue());
    const task = key ? skillTask(key) : null;
    setText($("learn-h"), problem ?? (key ? (task ? `Learning to ${task}` : `Learning: ${skillTitle(key)}`) : "Loading the lesson…"));
    const shownState = replay && !replay.finished ? "running" : runState;
    setText($("learn-eyebrow"), replay ? `Recorded earlier${replay.speed > 1 ? ` · shown ${replay.speed}× faster` : ""}` : runState === "running" || runState === "starting" ? "Live lesson" : runState ? "Lesson" : "");
    $("learn-eyebrow").dataset.live = String(!replay && (runState === "running" || runState === "starting"));
    const steps = problem || loading ? [] : lessonSteps(tl, shownState);
    $("lesson").replaceChildren(...steps.map(stepItem));
    $("lesson").hidden = !steps.length;
    const stoppedAt = steps.find((s) => s.state === "stopped");
    const ended = ENDED.includes(runState) && (!replay || replay.finished);
    $("learn-note").hidden = !(ended && runState !== "done");
    setText($("learn-note"), stoppedAt ? `This lesson stopped before it finished, at “${stoppedAt.title}”.` : "This lesson stopped before it finished.");
    const foot = [];
    if (problem) foot.push(el("button", { class: "btn btn--accent btn--lg", type: "button", on: { click: () => ctx.go("teach") } }, "Teach something else"));
    else if (ended && runState === "done") foot.push(el("button", { class: "btn btn--accent btn--lg", type: "button", on: { click: () => ctx.go("results", runId) } }, "See what it learned"));
    else if (ended) foot.push(el("button", { class: "btn btn--accent btn--lg", type: "button", on: { click: () => ctx.go("teach") } }, "Teach something else"));
    else if (replay && !replay.finished) foot.push(el("button", { class: "btn btn--quiet", type: "button", on: { click: skipReplay } }, "Skip to the end"));
    $("learn-foot").replaceChildren(...foot);
    $("learn-foot").hidden = !foot.length;
    ctx.drawerChanged();
  }

  function stepItem(s, k) {
    const extra = [];
    if (s.current || s.state === "running") extra.push(el("p", { class: "lesson__what" }, s.what));
    if (s.progress && (s.state === "running" || s.current)) {
      const frac = Math.min(1, s.progress.done / s.progress.total);
      const bar = el("span", { class: "meter", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": String(s.progress.total), "aria-valuenow": String(s.progress.done) },
        el("span", { class: "meter__fill" }));
      bar.firstChild.style.transform = `scaleX(${frac.toFixed(4)})`;
      extra.push(el("div", { class: "lesson__progress" }, bar, el("span", { class: "lesson__count" }, `${count(s.progress.done)} of ${count(s.progress.total)}${s.progress.unit ? ` ${s.progress.unit}` : ""}`)));
    }
    if (s.key === "brain" && (s.state === "running" || s.current || s.state === "done")) { const spark = sparkline(); if (spark) extra.push(spark); }
    const mark = s.state === "done" ? "✓" : s.state === "stopped" ? "!" : String(k + 1);
    const stateText = { waiting: "Waiting", running: "Now", done: "Done", stopped: "Stopped", unlogged: "Not in the log" }[s.state];
    return el("li", { class: "lesson__step", "data-state": s.state, "data-current": String(s.current), ...(s.current ? { "aria-current": "step" } : {}) },
      el("span", { class: "lesson__mark", "aria-hidden": "true" }, mark),
      el("div", { class: "lesson__body" },
        el("p", { class: "lesson__title" }, s.title, el("span", { class: "sr-only" }, ` (${stateText})`)),
        ...extra));
  }

  // The training curve as a small line: the validation loss falling over the rounds.
  function sparkline() {
    const series = trainSeries(tl);
    if (series.length < 2) return null;
    const c = lossCurve(series, SPARK);
    const line = c?.validation ?? c?.train;
    if (!line) return null;
    const key = c?.validation ? "validation_loss" : "train_loss";
    const pts = series.filter((d) => typeof d[key] === "number");
    const svg = svgEl("svg", { viewBox: `0 0 ${SPARK.width} ${SPARK.height}`, class: "spark", role: "img",
      "aria-label": `Training error, from ${Number(pts[0][key].toPrecision(2))} to ${Number(pts.at(-1)[key].toPrecision(2))}; lower is better` });
    svg.append(svgEl("polyline", { points: line, class: "spark__line" }));
    return el("div", { class: "lesson__spark" }, svg, el("span", { class: "lesson__count" }, "Its error while training (lower is better)"));
  }

  return {
    enter, leave, onRunEvent,
    runId: () => runId,
    log: () => (replay ? recorded.get(replay.id) ?? [] : live.get(runId) ?? []),
    timeline: () => tl,
    runState: () => runState,
  };
}
