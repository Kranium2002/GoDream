// The Learn view's pure parts. Certain: run event lines {run_id, skill, step, state, done, total, time_utc, data} and
// GET /api/runs/{id} {state, steps, baseline, results}. Recognised only when their fields are present: Dev B's training
// observer dict in train data, and the proposed result shape (resultView, the one adapter to change if it differs).
// Anything else is shown as key and value, never dropped and never interpreted.

export const STEPS = ["baseline", "attempts", "teacher_fit", "dream", "train", "evaluate"];
const LABELS = { baseline: "Baseline measured", attempts: "Real simulator attempts", teacher_fit: "PFN teacher fit",
  dream: "PFN dreams", train: "SNN training", evaluate: "Held-out evaluation" };
const STOPPED = ["refused", "failed", "stopped"];

import { readout } from "./readout.js";

const finite = (v) => typeof v === "number" && Number.isFinite(v);
const text = (v) => (typeof v === "string" ? v : JSON.stringify(v));

// Nested data as [dotted key, text] rows: the fallback for anything not recognised.
export function flatRows(value, prefix = "") {
  if (value === null || value === undefined) return [];
  if (typeof value !== "object" || Array.isArray(value)) return [[prefix || "value", text(value)]];
  return Object.entries(value).flatMap(([k, v]) => {
    const key = prefix ? `${prefix}.${k}` : k;
    return v !== null && typeof v === "object" && !Array.isArray(v) ? flatRows(v, key) : [[key, text(v)]];
  });
}

// ---- timeline ------------------------------------------------------------------------------------------------------
export const emptyTimeline = () => ({ runId: null, skill: null, state: null, steps: {}, train: {}, baseline: null, results: null, unreadable: 0, arrivals: 0 });

const isTrainObservation = (d) => d !== null && typeof d === "object" && finite(d.step) && (finite(d.train_loss) || finite(d.validation_loss));

// One event line into the timeline (pure). Per step the line with the latest time_utc wins (the later arrival on a tie);
// a line without a readable time never overrides one with a time. Train observations are kept by their own step.
export function applyLine(tl, line) {
  if (line === null || typeof line !== "object" || typeof line.step !== "string" || !line.step) {
    return { ...tl, unreadable: tl.unreadable + 1 };
  }
  const t = typeof line.time_utc === "string" ? Date.parse(line.time_utc) : NaN;
  const entry = { state: line.state, done: line.done, total: line.total, time: line.time_utc, t: Number.isNaN(t) ? null : t, data: line.data };
  const old = tl.steps[line.step];
  const wins = !old || (entry.t !== null && (old.t === null || entry.t >= old.t)) || (entry.t === null && old.t === null);
  const next = { ...tl, arrivals: tl.arrivals + 1 };
  if (typeof line.run_id === "string") next.runId = line.run_id;
  if (typeof line.skill === "string") next.skill = line.skill;
  if (wins) next.steps = { ...tl.steps, [line.step]: entry };
  if (line.step === "train" && isTrainObservation(line.data)) next.train = { ...tl.train, [line.data.step]: line.data };
  return next;
}

export function fromRunResponse(runId, body) {
  const lines = Array.isArray(body?.steps) ? body.steps : [];
  const tl = lines.reduce(applyLine, { ...emptyTimeline(), runId });
  return { ...tl, runId, state: typeof body?.state === "string" ? body.state : null, baseline: body?.baseline ?? null, results: body?.results ?? null };
}

export const trainSeries = (tl) => Object.values(tl.train).sort((a, b) => a.step - b.step);

// The six steps, then any other step the run reported, each as a display row.
export function timelineRows(tl) {
  const names = [...STEPS, ...Object.keys(tl.steps).filter((s) => !STEPS.includes(s))];
  return names.map((step) => {
    const e = tl.steps[step];
    const d = e?.data;
    const why = d && typeof d === "object" ? [d.reason, d.message, d.error].find((v) => typeof v === "string" && v) : undefined;
    return {
      step,
      label: LABELS[step] ?? step,
      state: typeof e?.state === "string" ? e.state : null,
      progress: finite(e?.done) && finite(e?.total) && e.total > 0 ? `${e.done} of ${e.total}` : null,
      time: typeof e?.time === "string" ? e.time : null,
      reason: STOPPED.includes(e?.state) && why ? why : null,
      isTrainObservation: step === "train" && isTrainObservation(d),
      dataRows: flatRows(d),
    };
  });
}

// ---- training curve --------------------------------------------------------------------------------------------
// Polyline points for train and validation loss over step inside a width x height box with `pad` inside each edge;
// lower loss is drawn lower. null when no point has a loss.
export function lossCurve(series, { width, height, pad }) {
  const keys = ["train_loss", "validation_loss"];
  const pts = series.filter((d) => finite(d.step) && keys.some((k) => finite(d[k])));
  if (!pts.length) return null;
  const losses = pts.flatMap((d) => keys.map((k) => d[k]).filter(finite));
  const x = [Math.min(...pts.map((d) => d.step)), Math.max(...pts.map((d) => d.step))];
  const y = [Math.min(...losses), Math.max(...losses)];
  const scale = ([lo, hi], v, a, b) => (hi === lo ? (a + b) / 2 : a + ((v - lo) / (hi - lo)) * (b - a));
  const line = (k) => {
    const p = pts.filter((d) => finite(d[k])).map((d) => `${round(scale(x, d.step, pad, width - pad))},${round(scale(y, d[k], height - pad, pad))}`);
    return p.length ? p.join(" ") : null;
  };
  return { x, y, train: line("train_loss"), validation: line("validation_loss") };
}
const round = (v) => Math.round(v * 100) / 100;

// ---- results ---------------------------------------------------------------------------------------------------
const METHOD_LABELS = { no_op: "No-op", physics: "Physics only", physics_only: "Physics only", teacher: "PFN teacher", pfn_teacher: "PFN teacher", snn: "SNN" };
const num = (v) => String(Number(v.toPrecision(4)));

function cell(c, unit) {
  if (!c || !finite(c.value)) return "not measured";
  let s = num(c.value) + (unit ? ` ${unit}` : "");
  if (finite(c.ci_low) && finite(c.ci_high)) s += ` (${num(c.ci_low)} to ${num(c.ci_high)})`;
  if (finite(c.n)) s += `, n = ${c.n}`;
  return s;
}

// The proposed result shape {run_id, skill, declaration, seeds, scenario_ids, metrics: [{name, unit, higher_is_better,
// values: {method: {value, ci_low, ci_high, n}}}], paired: [{name, a, b, diff, ci_low, ci_high, n}]} -> a table;
// anything else -> {kind: "raw", rows}. This is the one function to change if Dev B's result file differs.
export function resultView(result) {
  if (result === null || result === undefined) return { kind: "none" };
  const metrics = result?.metrics;
  const fits = Array.isArray(metrics) && metrics.length > 0
    && metrics.every((m) => m && typeof m.name === "string" && m.values && typeof m.values === "object" && !Array.isArray(m.values));
  if (!fits) return { kind: "raw", rows: flatRows(result) };
  const keys = [...new Set(metrics.flatMap((m) => Object.keys(m.values)))];
  const { metrics: _m, paired: _p, seeds: _s, scenario_ids: _i, ...rest } = result;
  return {
    kind: "table",
    extra: flatRows(rest), // every key the table does not show, as sent
    seeds: Array.isArray(result.seeds) ? result.seeds.map(text).join(", ") : null,
    scenarioCount: Array.isArray(result.scenario_ids) ? result.scenario_ids.length : null,
    methods: keys.map((key) => ({ key, label: METHOD_LABELS[key] ?? key })),
    rows: metrics.map((m) => ({
      name: m.name,
      unit: typeof m.unit === "string" ? m.unit : "",
      better: m.higher_is_better === true ? "higher is better" : m.higher_is_better === false ? "lower is better" : null,
      cells: keys.map((k) => cell(m.values[k], m.unit)),
    })),
    paired: (Array.isArray(result.paired) ? result.paired : []).map((p) => ({
      name: text(p?.name),
      comparison: `${text(p?.a)} minus ${text(p?.b)}`,
      diff: cell({ value: p?.diff, ci_low: p?.ci_low, ci_high: p?.ci_high, n: p?.n }, ""),
    })),
  };
}

// GET /api/runs: the run ids found on disk (a list of strings); anything else gives none.
export function runIds(data) {
  return Array.isArray(data) ? [...new Set(data.filter((id) => typeof id === "string" && id))] : [];
}

// ---- stage indicators ------------------------------------------------------------------------------------------
// One line from an SSE infer event: its action and the confidences it carries, the gate only when it carries one.
export function inferLine(data) {
  if (typeof data?.action !== "string") return null;
  const view = readout({ ok: true, data });
  const parts = [`${view.action.label} (${view.action.code})`];
  for (const [label, key] of [["threat confidence", "threat_confidence"], ["action confidence", "action_confidence"]]) {
    if (finite(data[key])) parts.push(`${label} ${data[key].toFixed(2)}`);
  }
  if (view.gate.length) parts.push(view.gate[view.gate.length - 1].label.toLowerCase());
  return parts.join(", ");
}

// RAMP mode, only when an observation actually carries it (field name assumed: ramp_mode, or mode).
export function rampMode(obs) {
  const v = obs?.ramp_mode ?? obs?.mode;
  return v === "NORMAL" || v === "SLOPE" ? v : null;
}
