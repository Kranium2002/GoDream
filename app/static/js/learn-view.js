// Learning as one flow on the robot stage: the pipeline track under the dog (live runs from the chat, or a replay of a
// recorded run), the learnt-skill cards, the training curve, the results and the step details. Every value comes from a
// response or a recorded event; a replay is always labelled as one; nothing is stored.
import { applyLine, emptyTimeline, flatRows, fromRunResponse, groupRows, inferLine, lossCurve, rampMode, resultView, runIds, timelineRows, timingLine, timingView, trainSeries } from "./learn.js";
import { cardsFrom, replayPlan, skillLabel, trackStages } from "./flow.js";

const SVG = "http://www.w3.org/2000/svg"; // the SVG namespace name (an identifier, never fetched)
const FINAL = ["done", "failed", "refused", "stopped"];
const BOX = { width: 640, height: 260, pad: 52 };
const fmt = (v) => (typeof v === "number" && Number.isFinite(v) ? String(Number(v.toPrecision(4))) : String(v)); // display rounding only

export function initLearnView({ api, el, notes, episodes, catalogue = () => null }) {
  const $ = (id) => document.getElementById(id);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const runs = new Map(); // run id -> SSE lines received this session (memory only)
  let selected = null, tl = emptyTimeline(), mode = null, gateKey = null, visible = false;
  let replay = null; // {id, speed, timers, finished}
  let lastCards = [];

  const status = (text) => { $("learn-status").textContent = text; };
  const kv = (rows) => el("dl", { class: "kv" }, ...rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));

  // ---- server state ---------------------------------------------------------------------------------------------
  // Listing and replaying saved runs work without the runs capability; only starting one needs it, and the chat
  // says so where a run is started.
  function setState() {}

  function setVisible(v) {
    const entering = v && !visible;
    visible = v;
    if (entering) { loadCards(); loadRunList(); }
  }

  // ---- learnt skills: one card per run on disk ------------------------------------------------------------------
  async function loadCards() {
    let r = await api.runCards();
    let cards = r.ok ? cardsFrom(r.data) : [];
    if (!r.ok || (Array.isArray(r.data) && r.data.length && !cards.length)) {
      const plain = await api.runs(); // the bare list of ids, when the summary is not offered
      if (plain.ok) { r = plain; cards = cardsFrom(plain.data); }
    }
    if (r.ok) notes.clear("cards");
    else notes.set("cards", r.unavailable ? "warn" : "danger", `Saved runs cannot be listed: ${r.message}`, r.unavailable ? "Unavailable" : "Error");
    $("cards-empty").hidden = !(r.ok && !cards.length);
    lastCards = cards;
    $("cards").replaceChildren(...cards.map(card));
    episodes?.refresh();
    for (const c of cards) addRun(c.runId);
  }

  const localTime = (iso) => {
    const d = typeof iso === "string" ? new Date(iso) : null;
    return d && !Number.isNaN(d.getTime()) ? d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : null;
  };

  function card(c) {
    const label = c.skill ? skillLabel(catalogue(), c.skill) : null;
    const when = localTime(c.updated);
    const facts = [c.state && `state ${c.state}`, c.stepsDone !== null && `${c.stepsDone} ${c.stepsDone === 1 ? "step" : "steps"} done`, when && `updated ${when}`].filter(Boolean);
    const badges = [c.hasResult === true && "result saved", c.hasStudent === true && "student trained"].filter(Boolean);
    const timed = timingLine(c.timing); // measured by Dev B's run_skill worker, wall clock
    const replayBtn = el("button", { class: "btn btn--primary btn--sm", type: "button", "aria-describedby": `card-${c.runId}` }, "Replay on the robot");
    replayBtn.addEventListener("click", () => playRun(c.runId));
    const epBtn = el("button", { class: "btn btn--secondary btn--sm", type: "button", "data-episodes": c.runId, "aria-describedby": `card-${c.runId}` }, "Show this run's episodes");
    return el("li", {}, el("article", { class: "card run-card", "aria-label": `${label?.name ?? c.skill ?? "Run"} ${c.runId}` },
      el("div", { class: "run-card__head" },
        el("h3", {}, label?.name ?? c.skill ?? `Run ${c.runId}`),
        el("p", { class: "run-card__id" }, el("code", {}, c.skill && label ? `${c.skill} · run ${c.runId}` : `run ${c.runId}`))),
      label?.description ? el("p", { class: "run-card__desc" }, label.description) : el("span", {}),
      el("div", { class: "run-card__body" },
        el("p", { class: "run-card__facts", id: `card-${c.runId}` }, facts.join(" · ") || "No summary from the server."),
        timed ? el("p", { class: "run-card__facts" }, `Measured: ${timed}.`) : null,
        badges.length ? el("p", { class: "badges" }, ...badges.map((t) => el("span", { class: "badge badge--neutral" }, t))) : null),
      el("div", { class: "run-card__actions" }, replayBtn, epBtn)));
  }

  // ---- runs -----------------------------------------------------------------------------------------------------
  async function loadRunList() {
    const r = await api.runs();
    if (r.ok) notes.clear("run-list");
    else notes.set("run-list", r.unavailable ? "warn" : "danger", `Runs saved on disk cannot be listed: ${r.message} Runs from this session and a typed id still work.`, r.unavailable ? "Unavailable" : "Error");
    for (const id of r.ok ? runIds(r.data) : []) addRun(id);
  }

  function addRun(id, open = false) {
    if (!runs.has(id)) {
      runs.set(id, []);
      const select = $("run-select");
      if (select.options.length === 1 && !select.options[0].value) select.options[0].textContent = "Choose a run";
      select.append(el("option", { value: id }, id));
    }
    if (open) { stopReplay(false); openRun(id); showStage(); loadCards(); } // open is used when a run has just started
  }

  async function openRun(id) {
    selected = id;
    $("run-select").value = runs.has(id) ? id : "";
    tl = { ...emptyTimeline(), runId: id };
    render();
    status(`Loading run ${id}.`);
    await refresh(id);
  }

  async function refresh(id) {
    const r = await api.run(id);
    if (selected !== id || (replay && !replay.finished)) return; // another run, or a replay is drawing this one
    if (r.ok) {
      tl = (runs.get(id) ?? []).reduce(applyLine, fromRunResponse(id, r.data)); // lines that arrived live, applied again
      status(`Run ${id}${tl.state ? `: ${tl.state}` : ""}.`);
    } else {
      status(`${r.unavailable ? "The run's record is unavailable" : "The run could not be read"}: ${r.message}${runs.get(id)?.length ? " Live step updates are still shown." : ""}`);
    }
    render();
  }

  function onRunEvent(line) {
    if (typeof line?.run_id !== "string" || !line.run_id) return;
    addRun(line.run_id);
    runs.get(line.run_id).push(line);
    if (line.run_id !== selected || replay) return;
    tl = applyLine(tl, line);
    render();
    if (FINAL.includes(line.state)) refresh(selected); // a finished step may have written results
  }

  // ---- replay of a recorded run ---------------------------------------------------------------------------------
  async function playRun(id) {
    stopReplay(false);
    addRun(id);
    selected = id;
    $("run-select").value = id;
    showStage();
    const speed = Number($("replay-speed").value);
    const r = await api.runEvents(id);
    if (selected !== id) return;
    if (!r.ok || !Array.isArray(r.data)) {
      status(`Replay needs the run's recorded events, which the server ${r.ok ? "did not send as a list" : `does not offer: ${r.message}`} The latest recorded state is shown instead.`);
      return openRun(id);
    }
    const plan = replayPlan(r.data.filter((l) => l && typeof l === "object"), speed);
    replay = { id, speed, timers: [], finished: false };
    tl = { ...emptyTimeline(), runId: id };
    render();
    status(`Replaying recorded run ${id} at ${speed}x: ${plan.length} recorded events.`);
    const mine = replay;
    for (const p of plan) mine.timers.push(setTimeout(() => { if (replay === mine) { tl = applyLine(tl, p.line); render(); } }, p.atMs));
    mine.timers.push(setTimeout(async () => {
      if (replay !== mine) return;
      mine.finished = true;
      status(`Replay of recorded run ${id} finished.`);
      const res = await api.run(id); // the results file, read once the replay reaches the end
      if (replay === mine && res.ok) { tl = { ...tl, results: res.data.results ?? null, baseline: res.data.baseline ?? null, state: res.data.state ?? tl.state }; }
      render();
    }, (plan.at(-1)?.atMs ?? 0) + 50));
  }

  function stopReplay(reload = true) {
    if (!replay) return;
    replay.timers.forEach(clearTimeout);
    const id = replay.id;
    replay = null;
    $("replay-stop").hidden = true;
    if (reload) { status(`Replay stopped. The latest recorded state of run ${id} is shown.`); refresh(id); }
  }

  function showStage() {
    scrollTo({ top: 0, behavior: reduceMotion.matches ? "auto" : "smooth" }); // the stage, with the header
  }

  // ---- stage indicators: only what an event actually carries -----------------------------------------------------
  function onObservation(obs) {
    const m = rampMode(obs);
    if (m === null || m === mode) return;
    mode = m;
    $("l-mode").textContent = m === "SLOPE" ? "SLOPE, the learned correction may act" : "NORMAL, the nominal command passes through";
    $("l-mode-wrap").hidden = false;
    $("track-meta").hidden = false;
  }

  function onInfer(data) {
    const text = inferLine(data);
    if (text === null || text === gateKey) return;
    gateKey = text;
    $("l-gate").textContent = text;
    $("l-gate-wrap").hidden = false;
    $("track-meta").hidden = false;
  }

  // ---- rendering ------------------------------------------------------------------------------------------------
  function render() {
    const rows = timelineRows(tl);
    renderTrack(rows);
    const meta = [tl.skill && `Skill ${tl.skill}`, tl.state && `run state ${tl.state}`, tl.unreadable && `${tl.unreadable} event lines could not be read`].filter(Boolean);
    $("tl-meta").textContent = selected ? meta.join(" · ") : "No run open.";
    $("timeline").replaceChildren(...rows.map(stepItem));
    renderTraining();
    renderResult($("baseline-body"), tl.baseline);
    renderResult($("results-body"), tl.results);
  }

  function renderTrack(rows) {
    const track = $("track");
    track.hidden = !selected;
    document.body.classList.toggle("track-open", Boolean(selected));
    if (!selected) return;
    $("replay-stop").hidden = !replay || replay.finished;
    $("track-episodes").dataset.episodes = selected;
    $("track-episodes").hidden = false;
    episodes?.refresh();
    const skill = tl.skill ? skillLabel(catalogue(), tl.skill)?.name ?? tl.skill : null;
    $("track-title").textContent = replay
      ? `Replay of recorded run ${replay.id} at ${replay.speed}x${replay.finished ? ", finished" : ""}${skill ? ` · ${skill}` : ""}`
      : `Run ${selected}${skill ? ` · ${skill}` : ""}${tl.state ? ` · ${tl.state}` : ""}`;
    $("track-steps").replaceChildren(...trackStages(rows, tl.state).map((s) => {
      const bad = ["failed", "refused", "stopped"].includes(s.state);
      const cls = s.state === "done" ? "is-done" : bad ? "is-bad" : s.running ? "is-running" : "";
      const tone = s.state === "done" ? "ok" : bad ? "danger" : s.running ? "accent" : "neutral";
      const progress = s.fraction === null ? (s.progress ? el("span", { class: "ts__count" }, s.progress) : null)
        : el("span", { class: "ts__progress" }, el("span", { class: "ts__count" }, s.progress),
          el("span", { class: "ts__bar", role: "img", "aria-label": `${s.progress}` }, el("span", { class: "ts__fill" })));
      if (s.fraction !== null) progress.querySelector(".ts__fill").style.transform = `scaleX(${s.fraction.toFixed(4)})`;
      return el("li", { class: `ts ${cls}` },
        el("span", { class: "ts__label" }, s.label),
        el("span", { class: `chip chip--${tone} ts__state` }, el("span", { class: "chip__dot", "aria-hidden": "true" }), s.stateText),
        progress ?? el("span", {}),
        s.explain ? el("span", { class: "ts__explain" }, s.explain) : el("span", {}));
    }));
  }

  function stepItem(r) {
    const cls = r.state === "done" ? "is-done" : ["failed", "refused", "stopped"].includes(r.state) ? "is-bad" : "";
    const facts = [r.state ?? "no event yet", r.progress, r.time].filter(Boolean).join(" · ");
    let data = null;
    if (r.isTrainObservation) data = el("p", { class: "small muted" }, "Training observation: shown in the training chart.");
    else if (r.dataRows.length) data = kv(r.dataRows);
    return el("li", { class: cls },
      el("span", { class: `dot${cls ? "" : " dot--hollow"}`, "aria-hidden": "true" }),
      el("div", {}, el("p", { class: "timeline__label" }, r.label), el("p", { class: "timeline__facts" }, facts),
        r.reason ? el("p", { class: "problem" }, `Reason: ${r.reason}`) : null, data));
  }

  function renderTraining() {
    const series = trainSeries(tl);
    const body = $("train-body");
    if (!series.length) return body.replaceChildren(el("p", { class: "muted" }, "No training observations in this run yet."));
    const last = series[series.length - 1];
    const facts = [
      ["Step", typeof last.max_steps === "number" ? `${last.step} of ${last.max_steps}` : String(last.step)],
      ["Training loss", last.train_loss], ["Validation loss", last.validation_loss],
      ["Best validation loss", last.best_validation_loss], ["Best step", last.best_step],
    ].filter(([, v]) => v !== undefined && v !== null).map(([k, v]) => [k, typeof v === "number" ? fmt(v) : v]);
    const rateRows = flatRows(last.firing_rate_per_layer).map(([k, v]) => [`Firing rate, layer ${k === "value" ? "" : k}`.trim(), v]);
    const headRows = flatRows(last.validation_loss_per_head).map(([k, v]) => [`Validation loss, ${k}`, v]);
    body.replaceChildren(chart(series),
      el("dl", { class: "stats" }, ...[...facts, ...headRows, ...rateRows].flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])));
  }

  function chart(series) {
    const c = lossCurve(series, BOX);
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", `0 0 ${BOX.width} ${BOX.height}`);
    svg.setAttribute("class", "curve");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `Loss over training steps ${c.x[0]} to ${c.x[1]}, loss from ${fmt(c.y[0])} to ${fmt(c.y[1])}. Solid line: training loss; dashed line: validation loss.`);
    const add = (tag, attrs, text) => {
      const n = document.createElementNS(SVG, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      if (text !== undefined) n.textContent = text;
      svg.append(n);
    };
    const { width: w, height: h, pad: p } = BOX;
    add("line", { x1: p, y1: h - p, x2: w - p, y2: h - p, class: "curve__axis" });
    add("line", { x1: p, y1: p, x2: p, y2: h - p, class: "curve__axis" });
    if (c.train) add("polyline", { points: c.train, class: "curve__train" });
    if (c.validation) add("polyline", { points: c.validation, class: "curve__val" });
    add("text", { x: p, y: h - p + 22, class: "curve__label" }, `step ${c.x[0]}`);
    add("text", { x: w - p, y: h - p + 22, class: "curve__label", "text-anchor": "end" }, `step ${c.x[1]}`);
    add("text", { x: p - 8, y: p + 5, class: "curve__label", "text-anchor": "end" }, fmt(c.y[1]));
    add("text", { x: p - 8, y: h - p + 5, class: "curve__label", "text-anchor": "end" }, fmt(c.y[0]));
    add("text", { x: p - 8, y: p - 16, class: "curve__label", "text-anchor": "end" }, "loss");
    return el("figure", { class: "curve-wrap" }, svg, el("figcaption", { class: "small muted" }, "Solid: training loss. Dashed: validation loss."));
  }

  const sections = (rows) => groupRows(rows).map((g) => el("details", { class: "section-group" },
    el("summary", {}, `${g.key} (${g.rows.length} ${g.rows.length === 1 ? "value" : "values"})`), kv(g.rows)));

  // How long learning took, as Dev B's run_skill worker measured it on this laptop (wall clock per step).
  function timingBlock(result) {
    const t = timingView(result);
    if (!t) return null;
    const s = (v) => `${v.toFixed(1)} s`;
    const learnParts = t.steps.filter(([name]) => t.learnSteps.includes(name)).map(([name, v]) => `${name} ${s(v)}`);
    const rows = [
      ...(t.mode ? [["Mode", t.mode]] : []),
      ...(t.attempts !== null ? [["Real simulator attempts", String(t.attempts)]] : []),
      ["Learning", `${s(t.learnS)}${learnParts.length ? ` (${learnParts.join(", ")})` : ""}`],
      ...(t.evaluateS !== null ? [["Held-out evaluation", s(t.evaluateS)]] : []),
      ...(t.machine || t.cpuCount !== null ? [["Machine", [t.machine, t.cpuCount !== null && `${t.cpuCount} logical CPUs`].filter(Boolean).join(", ")]] : []),
      ...(t.clock ? [["Clock", t.clock]] : []),
    ];
    return el("div", { class: "timing" }, el("p", { class: "block-title" }, "Measured on this laptop"), kv(rows));
  }

  function renderResult(body, result) {
    const v = resultView(result);
    if (v.kind === "none") return body.replaceChildren(el("p", { class: "muted" }, replay && !replay.finished ? "Appears when the replay reaches the end." : "Not reported yet."));
    const timing = body === $("results-body") ? timingBlock(result) : null;
    if (v.kind === "raw") {
      return body.replaceChildren(...(timing ? [timing] : []), el("p", { class: "small muted" }, "The server sent a result without a metrics table; its sections are shown as sent."), ...sections(v.rows));
    }
    const facts = [v.scenarioCount !== null && `${v.scenarioCount} held-out scenarios`, v.seeds && `seeds ${v.seeds}`].filter(Boolean).join(", ");
    const table = el("table", { class: "specs" },
      el("caption", {}, `Measured in simulation${facts ? `: ${facts}` : ""}. Values with their interval and n where the server gives them.`),
      el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Metric"), ...v.methods.map((m) => el("th", { scope: "col" }, m.label)))),
      el("tbody", {}, ...v.rows.map((row) => el("tr", {},
        el("th", { scope: "row" }, row.name, row.better ? el("span", { class: "note" }, row.better) : null),
        ...row.cells.map((c) => el("td", {}, c))))));
    const paired = v.paired.length ? el("table", { class: "specs" },
      el("caption", {}, "Paired differences on the same scenarios"),
      el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Metric"), el("th", { scope: "col" }, "Comparison"), el("th", { scope: "col" }, "Difference"))),
      el("tbody", {}, ...v.paired.map((p) => el("tr", {}, el("th", { scope: "row" }, p.name), el("td", {}, p.comparison), el("td", {}, p.diff))))) : null;
    const extra = v.extra.length ? [el("p", { class: "block-title" }, "Other sections of the result, as sent"), ...sections(v.extra)] : [];
    body.replaceChildren(...(timing ? [timing] : []), el("div", { class: "table-wrap" }, table), ...(paired ? [el("div", { class: "table-wrap" }, paired)] : []), ...extra);
  }

  // ---- controls -------------------------------------------------------------------------------------------------
  $("run-select").addEventListener("change", (e) => { if (e.target.value) { stopReplay(false); openRun(e.target.value); } });
  $("run-pick").addEventListener("submit", (e) => {
    e.preventDefault();
    const id = $("run-id").value.trim();
    if (id) { stopReplay(false); addRun(id); openRun(id); }
  });
  $("replay-stop").addEventListener("click", () => stopReplay(true));
  $("track-close").addEventListener("click", () => { stopReplay(false); selected = null; render(); status("No run open."); });
  render();
  loadCards(); // once at start too, so the cards never wait for the section to scroll into view

  // The catalogue arrived or changed: the cards name their skills from it.
  const catalogueChanged = () => { $("cards").replaceChildren(...lastCards.map(card)); episodes?.refresh(); };
  return { setState, setVisible, addRun, onRunEvent, onObservation, onInfer, catalogueChanged, showStage, selected: () => selected };
}
