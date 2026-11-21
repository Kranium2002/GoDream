// The Learn area: a run picker, the six-step timeline from GET /api/runs/{id} and SSE run lines, the training curve,
// the before/after tables and the simulator stage. Every value shown comes from a response; nothing is stored.
import { applyLine, emptyTimeline, flatRows, fromRunResponse, inferLine, lossCurve, rampMode, resultView, runIds, timelineRows, trainSeries } from "./learn.js";
import { createStage } from "./stage.js";

const SVG = "http://www.w3.org/2000/svg"; // the SVG namespace name (an identifier, never fetched)
const FINAL = ["done", "failed", "refused", "stopped"];
const BOX = { width: 640, height: 240, pad: 28 };

export function initLearnView({ api, el }) {
  const $ = (id) => document.getElementById(id);
  const stage = createStage({ root: $("l-stage"), trackingImg: $("l-tracking"), trackingNote: $("l-trk-note"), headImg: $("l-head"), headNote: $("l-head-note") });
  let visible = false, headCapability = null;
  const runs = new Map(); // run id -> SSE lines received this session (memory only)
  let selected = null, tl = emptyTimeline(), mode = null, gateKey = null;

  const status = (text) => { $("learn-status").textContent = text; };

  // ---- server state ---------------------------------------------------------------------------------------------
  function setState(state, capabilities) {
    headCapability = capabilities.find((c) => c.name === "head_camera") ?? null;
    stage.setAspect(state?.frame?.width, state?.frame?.height);
    const down = capabilities.filter((c) => (c.name === "runs" || c.name === "results") && !c.available);
    $("learn-cap").hidden = !down.length;
    $("learn-cap").textContent = down.map((c) => `${c.name === "runs" ? "Learning runs" : "Results"} unavailable: ${c.reason}`).join(" ");
    stage.update(visible, headCapability);
  }

  function setVisible(v) {
    const entering = v && !visible;
    visible = v;
    stage.update(visible, headCapability);
    if (entering) loadRunList();
  }

  // The runs found on disk (GET /api/runs), added to the ones seen in this session.
  async function loadRunList() {
    const r = await api.runs();
    const note = $("run-list-note");
    note.hidden = r.ok;
    if (!r.ok) note.textContent = `Runs saved on disk cannot be listed: ${r.message} Runs from this session and a typed id still work.`;
    for (const id of r.ok ? runIds(r.data) : []) addRun(id);
  }

  // ---- runs -----------------------------------------------------------------------------------------------------
  function addRun(id, open = false) {
    if (!runs.has(id)) {
      runs.set(id, []);
      const select = $("run-select");
      if (select.options.length === 1 && !select.options[0].value) select.options[0].textContent = "Choose a run";
      select.append(el("option", { value: id }, id));
    }
    if (open) { openRun(id); loadRunList(); } // open is used when a run has just started
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
    if (selected !== id) return; // another run was opened meanwhile
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
    if (line.run_id !== selected) return;
    tl = applyLine(tl, line);
    render();
    if (FINAL.includes(line.state)) refresh(selected); // a finished step may have written baseline or results
  }

  // ---- stage indicators: only what an event actually carries -----------------------------------------------------
  function onObservation(obs) {
    const m = rampMode(obs);
    if (m === null || m === mode) return;
    mode = m;
    $("l-mode").textContent = m === "SLOPE" ? "SLOPE: the learned correction may act" : "NORMAL: the nominal command passes through";
  }

  function onInfer(data) {
    const text = inferLine(data);
    if (text === null || text === gateKey) return;
    gateKey = text;
    $("l-gate").textContent = text;
  }

  // ---- rendering ------------------------------------------------------------------------------------------------
  function render() {
    const meta = [tl.skill && `Skill ${tl.skill}`, tl.state && `run state ${tl.state}`, tl.unreadable && `${tl.unreadable} event lines could not be read`].filter(Boolean);
    $("tl-meta").textContent = selected ? meta.join(" · ") : "No run open.";
    $("timeline").replaceChildren(...timelineRows(tl).map(stepItem));
    renderTraining();
    renderResult($("baseline-body"), tl.baseline);
    renderResult($("results-body"), tl.results);
  }

  function stepItem(r) {
    const mark = r.state === "done" ? "✓" : ["failed", "refused", "stopped"].includes(r.state) ? "✕" : r.state ? "●" : "○";
    const cls = r.state === "done" ? "is-good" : ["failed", "refused", "stopped"].includes(r.state) ? "is-bad" : "";
    const facts = [r.state ?? "not started", r.progress, r.time].filter(Boolean).join(" · ");
    let data = null;
    if (r.isTrainObservation) data = el("p", { class: "hint" }, "Training observation: shown in the training chart.");
    else if (r.dataRows.length) data = el("dl", { class: "kv" }, ...r.dataRows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));
    return el("li", { class: cls },
      el("span", { class: "mark", "aria-hidden": "true" }, mark),
      el("div", {}, el("p", { class: "timeline__label" }, r.label), el("p", { class: "timeline__facts" }, facts),
        r.reason ? el("p", { class: "problem" }, `Reason: ${r.reason}`) : null, data));
  }

  function renderTraining() {
    const series = trainSeries(tl);
    const body = $("train-body");
    if (!series.length) return body.replaceChildren(el("p", { class: "empty" }, "No training observations in this run yet."));
    const last = series[series.length - 1];
    const facts = [
      ["Step", typeof last.max_steps === "number" ? `${last.step} of ${last.max_steps}` : String(last.step)],
      ["Training loss", last.train_loss], ["Validation loss", last.validation_loss],
      ["Best validation loss", last.best_validation_loss], ["Best step", last.best_step],
    ].filter(([, v]) => v !== undefined && v !== null).map(([k, v]) => [k, typeof v === "number" ? String(v) : v]);
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
    svg.setAttribute("aria-label", `Loss over training steps ${c.x[0]} to ${c.x[1]}, loss from ${c.y[0]} to ${c.y[1]}. Solid line: training loss; dashed line: validation loss.`);
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
    add("text", { x: p, y: h - 6, class: "curve__label" }, `step ${c.x[0]}`);
    add("text", { x: w - p, y: h - 6, class: "curve__label", "text-anchor": "end" }, `step ${c.x[1]}`);
    add("text", { x: p + 6, y: p - 10, class: "curve__label" }, `loss ${c.y[1]}`);
    add("text", { x: p + 6, y: h - p - 6, class: "curve__label" }, `loss ${c.y[0]}`);
    const legend = el("p", { class: "hint" }, "Solid: training loss. Dashed: validation loss.");
    return el("figure", { class: "curve-wrap" }, svg, legend);
  }

  function renderResult(body, result) {
    const v = resultView(result);
    if (v.kind === "none") return body.replaceChildren(el("p", { class: "empty" }, "Not reported yet."));
    if (v.kind === "raw") {
      return body.replaceChildren(el("p", { class: "hint" }, "The server sent a result this page does not recognise; it is shown as sent."),
        el("dl", { class: "kv" }, ...v.rows.flatMap(([k, val]) => [el("dt", {}, k), el("dd", {}, val)])));
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
    const extra = v.extra.length ? el("div", {}, el("p", { class: "block-title" }, "Other sections of the result, as sent"),
      el("dl", { class: "kv" }, ...v.extra.flatMap(([k, val]) => [el("dt", {}, k), el("dd", {}, val)]))) : null;
    body.replaceChildren(table, ...[paired, extra].filter(Boolean));
  }

  // ---- controls -------------------------------------------------------------------------------------------------
  $("run-select").addEventListener("change", (e) => { if (e.target.value) openRun(e.target.value); });
  $("run-pick").addEventListener("submit", (e) => {
    e.preventDefault();
    const id = $("run-id").value.trim();
    if (id) { addRun(id); openRun(id); }
  });
  render();

  return { setState, setVisible, addRun, onRunEvent, onObservation, onInfer };
}
