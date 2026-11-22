// "How it learned": the four beats of a lesson, pure. From GET /api/story/<run_id> when the server has it, else from
// what a recorded run's own files hold (GET /api/runs/<run_id>): then only beats 3 and 4 have pictures and beats 1 and
// 2 keep the counts the run recorded. Every number is the files'; a part the files do not hold is null, never guessed.

const finite = (v) => typeof v === "number" && Number.isFinite(v);
const str = (v) => (typeof v === "string" && v.trim() ? v.trim() : null);
const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : null);
const pairs = (v) => (Array.isArray(v) ? v.filter((p) => Array.isArray(p) && p.length >= 2 && finite(p[0]) && finite(p[1])) : []);

export function humanize(name) {
  const s = String(name).replace(/_(m_s|m|cm|deg|s|rad)$/, "").replace(/_/g, " ").trim().toLowerCase();
  return s ? s[0].toUpperCase() + s.slice(1) : String(name);
}

// ---- beat 1: the real tries ------------------------------------------------------------------------------------------
function readTries(a) {
  a = obj(a);
  if (!a) return null;
  const paths = Array.isArray(a.paths) ? a.paths.map((p) => ({ points: pairs(p?.points), ended: str(p?.ended) ?? "" })).filter((p) => p.points.length >= 2) : null;
  const outcomes = Array.isArray(a.outcomes) ? a.outcomes.map((o) => ({ kind: str(o?.kind) ?? "", ended: str(o?.ended) ?? "" })) : null;
  const obstacles = Array.isArray(a.obstacles) ? a.obstacles.filter((o) => finite(o?.x) && finite(o?.y) && finite(o?.r) && o.r > 0).map((o) => ({ x: o.x, y: o.y, r: o.r })) : [];
  return { nTries: Number.isInteger(a.n_tries) ? a.n_tries : null, nDecisions: Number.isInteger(a.n_decisions) ? a.n_decisions : null,
    paths: paths?.length ? paths : null, outcomes: outcomes?.length ? outcomes : null, obstacles };
}

// How the tries ended, as groups in first-seen order with their counts: [{ended, count}].
export function endings(items) {
  const m = new Map();
  for (const it of items ?? []) m.set(it.ended, (m.get(it.ended) ?? 0) + 1);
  return [...m].map(([ended, count]) => ({ ended, count }));
}

// Paths seen from above, fitted into a w x h box with `pad` (x forward to the right, y to the left drawn up), one scale
// for both axes: {paths: [{d, ended, end: [x, y]}], obstacles: [{cx, cy, r}]}. Obstacles count toward the extent.
export function topView(tries, { w, h, pad }) {
  const pts = [...tries.paths.flatMap((p) => p.points), ...tries.obstacles.flatMap((o) => [[o.x - o.r, o.y - o.r], [o.x + o.r, o.y + o.r]])];
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const k = Math.min((w - 2 * pad) / Math.max(x1 - x0, 1e-6), (h - 2 * pad) / Math.max(y1 - y0, 1e-6));
  const ox = pad + ((w - 2 * pad) - (x1 - x0) * k) / 2, oy = pad + ((h - 2 * pad) - (y1 - y0) * k) / 2;
  const X = (x) => Math.round((ox + (x - x0) * k) * 10) / 10, Y = (y) => Math.round((h - oy - (y - y0) * k) * 10) / 10;
  return {
    paths: tries.paths.map((p) => ({ d: p.points.map(([x, y], i) => `${i ? "L" : "M"}${X(x)} ${Y(y)}`).join(" "), ended: p.ended, end: [X(p.points.at(-1)[0]), Y(p.points.at(-1)[1])] })),
    obstacles: tries.obstacles.map((o) => ({ cx: X(o.x), cy: Y(o.y), r: Math.round(o.r * k * 10) / 10 })),
  };
}

// ---- beat 2: TabPFN's what-ifs ---------------------------------------------------------------------------------------
function readEstimate(e) {
  e = obj(e);
  if (!e || !finite(e.value)) return null;
  return { value: e.value, low: finite(e.low) ? e.low : null, high: finite(e.high) ? e.high : null, unit: typeof e.unit === "string" ? e.unit : "", label: str(e.label) };
}
function readWhatIf(list) {
  const first = Array.isArray(list) ? list.find((w) => Array.isArray(w?.choices) && w.choices.length) : null;
  if (!first) return null;
  const choices = first.choices.map((c) => ({ label: str(c?.label) ?? "", tried: c?.tried === true, predicted: readEstimate(c?.predicted), measured: readEstimate(c?.measured) }))
    .filter((c) => c.label && (c.predicted || c.measured));
  if (!choices.length) return null;
  const m = obj(first.moment);
  return { moment: { tryNo: Number.isInteger(m?.try) ? m.try : null, t: finite(m?.t_s) ? m.t_s : null }, choices };
}

// The what-if choices on one axis: {lo, hi, x(v) -> 0..1}, from every value, low, high and measured value shown.
export function choiceAxis(choices) {
  const vals = choices.flatMap((c) => [c.predicted?.value, c.predicted?.low, c.predicted?.high, c.measured?.value]).filter(finite);
  const min = Math.min(...vals), max = Math.max(...vals);
  let lo = min, hi = max;
  if (hi === lo) { lo -= 1; hi += 1; }
  const padding = (hi - lo) * 0.08;
  lo -= padding; hi += padding;
  // Values that all lie in 0..1 (chances, shares) keep that range on the axis, never past it.
  if (min >= 0 && max <= 1) { lo = Math.max(0, lo); hi = Math.min(1, hi); }
  else if (min >= 0) lo = Math.max(0, lo);
  return { lo, hi, x: (v) => (v - lo) / (hi - lo) };
}

function readImagination(im) {
  const rows = Array.isArray(obj(im)?.rows) ? im.rows : [];
  const out = rows.filter((r) => str(r?.label) && finite(r?.tabpfn_error) && finite(r?.formula_error))
    .map((r) => ({ label: str(r.label), unit: typeof r.unit === "string" ? r.unit : "", tabpfn: r.tabpfn_error, formula: r.formula_error, n: Number.isInteger(r.n) ? r.n : null }));
  return out.length ? out : null;
}

// ---- beat 3: the spiking brain ---------------------------------------------------------------------------------------
function readBrain(b) {
  b = obj(b);
  if (!b) return null;
  const curve = Array.isArray(b.curve) ? b.curve.filter((p) => finite(p?.step) && (finite(p?.train) || finite(p?.validation)))
    .map((p) => ({ step: p.step, train: finite(p.train) ? p.train : null, validation: finite(p.validation) ? p.validation : null })) : [];
  // neurons: a count, or the hidden layer sizes; choices: a count, or the names of the choices.
  const neurons = Number.isInteger(b.neurons) ? b.neurons : Array.isArray(b.neurons) && b.neurons.length && b.neurons.every(Number.isInteger) ? b.neurons.reduce((x, y) => x + y, 0) : null;
  const choices = Number.isInteger(b.choices) ? b.choices : Array.isArray(b.choices) && b.choices.length ? b.choices.length : null;
  const out = { inputs: Number.isInteger(b.inputs) ? b.inputs : null, neurons, choices,
    curve, spikesPerDecision: finite(b.spikes_per_decision) ? b.spikes_per_decision : null, decisionMs: finite(b.decision_ms) ? b.decision_ms : null, stepsRun: null, layers: Array.isArray(b.neurons) ? b.neurons.length : null };
  return Object.values(out).some((v) => (Array.isArray(v) ? v.length : v !== null)) ? out : null;
}

// What a recorded run's files say about its student: hidden layer sizes, its training curves (train and validation
// loss by step; a validation loss per head is averaged over the heads it names), spikes per decision from the parity
// check (spikes counted over the inputs it checked) and the measured decision time.
function brainFromRun(results) {
  const train = obj(results?.train), training = obj(train?.training), sweep = obj(results?.sweep);
  const hidden = Array.isArray(train?.hidden_sizes) ? train.hidden_sizes : Array.isArray(sweep?.settings?.hidden_sizes) ? sweep.settings.hidden_sizes : null;
  const curve = new Map();
  for (const [s, v] of pairs(training?.train_loss_curve)) curve.set(s, { step: s, train: v, validation: null });
  for (const p of Array.isArray(training?.validation_loss_curve) ? training.validation_loss_curve : []) {
    if (!Array.isArray(p) || !finite(p[0])) continue;
    const v = finite(p[1]) ? p[1] : obj(p[1]) ? mean(Object.values(p[1]).filter(finite)) : null;
    if (v === null) continue;
    curve.set(p[0], { ...(curve.get(p[0]) ?? { step: p[0], train: null }), validation: v });
  }
  const par = obj(train?.parity?.action);
  const spikes = Array.isArray(par?.total_spikes_per_layer_numpy) && finite(par?.n_inputs_checked) && par.n_inputs_checked > 0
    ? par.total_spikes_per_layer_numpy.filter(finite).reduce((a, b) => a + b, 0) / par.n_inputs_checked : null;
  const latency = obj(results?.runtime_evidence?.latency);
  const steps = finite(training?.steps_run) ? training.steps_run : finite(sweep?.training?.steps_run) ? sweep.training.steps_run : null;
  const out = { inputs: null, neurons: hidden && hidden.every(Number.isInteger) ? hidden.reduce((a, b) => a + b, 0) : null, layers: hidden?.length ?? null, choices: null,
    curve: [...curve.values()].sort((a, b) => a.step - b.step), spikesPerDecision: spikes, decisionMs: finite(latency?.median_ms) ? latency.median_ms : null, stepsRun: steps };
  return out.neurons !== null || out.curve.length || out.spikesPerDecision !== null || out.decisionMs !== null ? out : null;
}
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);

// A training curve as two polylines in a w x h box (lower loss drawn lower): {train, validation, steps: [first, last]}.
export function curvePaths(curve, { w, h, pad }) {
  const vals = curve.flatMap((p) => [p.train, p.validation]).filter(finite);
  if (!curve.length || !vals.length) return null;
  const s0 = curve[0].step, s1 = curve.at(-1).step, lo = Math.min(...vals), hi = Math.max(...vals);
  const X = (s) => (s1 === s0 ? w / 2 : pad + ((s - s0) / (s1 - s0)) * (w - 2 * pad)), Y = (v) => (hi === lo ? h / 2 : h - pad - ((v - lo) / (hi - lo)) * (h - 2 * pad));
  const line = (k) => { const p = curve.filter((q) => finite(q[k])).map((q) => `${Math.round(X(q.step) * 10) / 10},${Math.round(Y(q[k]) * 10) / 10}`); return p.length >= 2 ? p.join(" ") : null; };
  return { train: line("train"), validation: line("validation"), steps: [s0, s1] };
}

// ---- beat 4: the test on unseen situations ---------------------------------------------------------------------------
// The two methods a result compares: the learned one (snn, else student, else the last) and the one before it.
export function pairOf(metric) {
  const keys = Object.keys(metric?.values ?? {});
  const after = ["snn", "student"].find((k) => keys.includes(k)) ?? keys.at(-1) ?? null;
  return { before: keys.find((k) => k !== after) ?? null, after };
}
const usable = (m) => m && typeof m.name === "string" && obj(m.values);

function cellOf(metric, cell) {
  if (!cell || !finite(cell.value)) return null;
  const n = Number.isInteger(cell.n) && cell.n > 0 ? cell.n : null;
  const lo = finite(cell.ci_low) ? cell.ci_low : null, hi = finite(cell.ci_high) ? cell.ci_high : null;
  const unit = typeof metric.unit === "string" ? metric.unit : "";
  let text, range = null;
  if (unit === "fraction" && n) {
    text = `${Math.round(cell.value * n)} of ${n}`;
    if (lo !== null && hi !== null) range = `likely ${Math.round(lo * n)} to ${Math.round(hi * n)}`;
  } else if (unit === "fraction") {
    text = `${Math.round(cell.value * 100)}%`;
    if (lo !== null && hi !== null) range = `likely ${Math.round(lo * 100)}% to ${Math.round(hi * 100)}%`;
  } else if (unit === "count") text = String(Math.round(cell.value));
  else {
    const u = unit === "1/min" ? " per minute" : unit === "deg" ? "°" : unit ? ` ${unit}` : "";
    const f = (v) => `${Number(v.toPrecision(3))}${u}`;
    text = f(cell.value);
    if (lo !== null && hi !== null) range = `likely ${f(lo)} to ${f(hi)}`;
  }
  return { value: cell.value, low: lo, high: hi, n, text, range };
}

// The metrics with a learned value -> rows {name, label, unit, higherIsBetter, before, after, change}; at most `max`.
export function testRows(metrics, max = 4) {
  return (Array.isArray(metrics) ? metrics.filter(usable) : []).map((m) => {
    const { before, after } = pairOf(m);
    const b = cellOf(m, m.values[before]), a = cellOf(m, m.values[after]);
    const better = m.higher_is_better === true ? "higher" : m.higher_is_better === false ? "lower" : null;
    let change = null;
    if (a && b && better) change = a.text === b.text ? "same" : (a.value > b.value) === (better === "higher") ? "better" : "worse";
    return { name: m.name, label: str(m.label) ?? humanize(m.name), unit: typeof m.unit === "string" ? m.unit : "", higherIsBetter: better, before: b, after: a, change };
  }).filter((r) => r.after).slice(0, max);
}

// Calm notes from the data: a goal (any number named *_target in the result) the learned value did not reach, and a
// headline change whose paired interval still includes no change.
export function testNotes(result, rows) {
  const notes = [];
  const targets = new Map();
  (function walk(v) {
    if (!v || typeof v !== "object") return;
    for (const [k, x] of Object.entries(v)) {
      if (/_target$/.test(k) && finite(x)) targets.set(`${k}:${x}`, { stem: k.replace(/_target$/, ""), value: x });
      else if (x && typeof x === "object") walk(x);
    }
  })(result);
  const metrics = Array.isArray(result?.metrics) ? result.metrics.filter(usable) : [];
  for (const { stem, value } of targets.values()) {
    const m = metrics.find((x) => x.name.includes(stem));
    if (!m) continue;
    const row = testRows([m])[0];
    if (!row?.after) continue;
    const met = m.higher_is_better === false ? row.after.value <= value : row.after.value >= value;
    if (met) continue;
    const goal = m.unit === "fraction" ? `${Math.round(value * 100)}%` : String(value);
    const got = m.unit === "fraction" ? `${Math.round(row.after.value * 100)}%${row.after.n ? ` (${row.after.text})` : ""}` : row.after.text;
    notes.push({ kind: "goal", text: `Not there yet. The goal for ${row.label.toLowerCase()} is ${goal}; in testing it reached ${got}.` });
  }
  const head = rows[0];
  // Another method in the same test that did better than the learned one on the first metric is said, not hidden.
  const hm = head ? metrics.find((m) => m.name === head.name) : null;
  if (hm && head.after && head.higherIsBetter) {
    const { before, after } = pairOf(hm);
    for (const [k, cell] of Object.entries(hm.values)) {
      if (k === before || k === after) continue;
      const c = cellOf(hm, cell);
      if (c && (head.higherIsBetter === "higher" ? c.value > head.after.value : c.value < head.after.value)) {
        notes.push({ kind: "other", text: `In the same test, ${humanize(k).toLowerCase()} did better: ${c.text}.` });
      }
    }
  }
  const paired = Array.isArray(result?.paired) ? result.paired.find((p) => p?.name === head?.name) : null;
  if (paired && finite(paired.ci_low) && finite(paired.ci_high) && paired.ci_low <= 0 && paired.ci_high >= 0) {
    notes.push({ kind: "chance", text: "With this many tests, this change could still be chance." });
  }
  return notes;
}

// ---- the whole story -------------------------------------------------------------------------------------------------
// GET /api/story/<run_id> -> the beats.
export function readStory(raw, runId = null) {
  const s = obj(raw) ?? {};
  const test = obj(s.test);
  const rows = testRows(test?.metrics);
  return {
    source: "story", runId,
    tries: readTries(s.attempts),
    whatIf: readWhatIf(s.what_if),
    imagination: readImagination(s.imagination),
    brain: readBrain(s.brain),
    test: rows.length ? { rows, notes: testNotes(test, rows) } : null,
  };
}

// GET /api/runs/<run_id> (a recorded run) -> the beats it can fill: the count of real tries its timing recorded and the
// rows TabPFN learned from, the student, and the test.
export function storyFromRun(body, runId = null) {
  const r = obj(body?.results) ?? {};
  const rows = testRows(r.metrics);
  const attempts = Number.isInteger(r.timing?.attempts) ? r.timing.attempts : null;
  const contextRows = Number.isInteger(r.teach?.teacher?.n_context_rows) ? r.teach.teacher.n_context_rows : null;
  const shape = Array.isArray(r.teach?.scores?.shape) && r.teach.scores.shape.length === 2 && r.teach.scores.shape.every(Number.isInteger) ? r.teach.scores.shape : null;
  return {
    source: "run", runId,
    tries: attempts !== null || contextRows !== null ? { nTries: attempts, nDecisions: null, contextRows, paths: null, outcomes: null, obstacles: [] } : null,
    whatIf: null,
    imagined: shape ? { situations: shape[0], choices: shape[1] } : null,
    imagination: null,
    brain: brainFromRun(r),
    test: rows.length ? { rows, notes: testNotes(r, rows) } : null,
  };
}

// The four beats in order, each {n, key, title, has}: has is whether the story holds anything for it.
export const BEATS = [
  { n: 1, key: "tries", title: "A few real tries" },
  { n: 2, key: "whatIf", title: "TabPFN fills in the what-ifs" },
  { n: 3, key: "brain", title: "A tiny spiking brain learns the choices" },
  { n: 4, key: "test", title: "Tested on cases it never saw" },
];
export function beatsOf(story) {
  return BEATS.map((b) => ({ ...b, has: b.key === "whatIf" ? Boolean(story.whatIf || story.imagination || story.imagined) : Boolean(story[b.key]) }));
}
