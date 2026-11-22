// GET /api/demos -> the demos the page shows, and the plain words built from their typed fields. Pure. Nothing here
// knows a demo by name: tabs, handles, outcome lines and the result sentence come from what each demo declares, and a
// part a demo does not declare is left out, never filled in.
import { readLanes, readShowcase } from "./showcase.js";

export const HANDLE_KINDS = ["slope", "offset", "direction", "amount"];
export const OUTCOME_KINDS = ["flag", "count", "seconds", "metres"];
export const SENSES = ["feel", "eye"];

const finite = (v) => typeof v === "number" && Number.isFinite(v);
const str = (v) => (typeof v === "string" && v.trim() ? v.trim() : null);

// One handle declaration -> {ok: true, handle} or {ok: false, problem}. A handle needs a name, a known kind, a label,
// a range with min below max, a step above 0 and a default inside the range.
export function readHandle(h) {
  if (!h || typeof h !== "object") return { ok: false, problem: "A handle is not an object." };
  const name = str(h.name);
  if (!name) return { ok: false, problem: "A handle has no name." };
  const bad = (why) => ({ ok: false, problem: `Handle ${name}: ${why}` });
  if (!HANDLE_KINDS.includes(h.kind)) return bad(`its kind ${JSON.stringify(h.kind)} is not one of ${HANDLE_KINDS.join(", ")}.`);
  if (!str(h.label)) return bad("it has no label.");
  if (!finite(h.min) || !finite(h.max) || !(h.min < h.max)) return bad("its range needs a min below its max.");
  if (!finite(h.step) || !(h.step > 0)) return bad("its step must be a number above 0.");
  if (!finite(h.default) || h.default < h.min || h.default > h.max) return bad("its default must lie inside its range.");
  return { ok: true, handle: { name, kind: h.kind, label: str(h.label), unit: typeof h.unit === "string" ? h.unit : "", min: h.min, max: h.max, step: h.step, default: h.default } };
}

function readField(f) {
  if (!f || typeof f !== "object" || !str(f.name) || !OUTCOME_KINDS.includes(f.kind) || !str(f.label)) return null;
  return { name: str(f.name), kind: f.kind, label: str(f.label), tone: f.tone === "bad" || f.tone === "good" ? f.tone : null };
}

// A label sent as an identifier (collision_rate) is shown in words (Collision rate); a label in words is kept.
export const wordsOf = (label) => (/^[a-z0-9]+(_[a-z0-9]+)+$/.test(label) ? label.replace(/_(m_s|m|s|deg)$/, "").replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()) : label);

function readHeadline(h) {
  if (!h || typeof h !== "object" || !str(h.metric_label) || !finite(h.before) || !finite(h.after)) return null;
  return { metricLabel: wordsOf(str(h.metric_label)), unit: typeof h.unit === "string" ? h.unit : "", before: h.before, after: h.after, n: Number.isInteger(h.n) && h.n > 0 ? h.n : null };
}

// The whole answer -> {ok, demos, problems}. ok is false only when the answer is not a list; a demo that cannot be
// read is left out with a problem, and so is a handle or an outcome field that cannot be read.
export function readDemos(raw) {
  if (!Array.isArray(raw)) return { ok: false, demos: [], problems: ["The demo list is not a list."] };
  const problems = [], demos = [], seen = new Set();
  for (const d of raw) {
    const id = str(d?.id), title = str(d?.title);
    if (!id || !title) { problems.push("A demo without an id or a title was left out."); continue; }
    if (seen.has(id)) { problems.push(`Demo ${id} is listed twice; the second was left out.`); continue; }
    seen.add(id);
    const handles = [];
    for (const h of Array.isArray(d.handles) ? d.handles : []) {
      const r = readHandle(h);
      if (r.ok) handles.push(r.handle);
      else problems.push(`Demo ${id}: ${r.problem}`);
    }
    const outcomeFields = (Array.isArray(d.outcome_fields) ? d.outcome_fields : []).map(readField);
    if (outcomeFields.some((f) => f === null)) problems.push(`Demo ${id}: an outcome field without a name, a known kind or a label was left out.`);
    demos.push({
      id, title,
      name: str(d.name),
      lanes: readLanes(d.lanes),
      showcase: readShowcase(d.showcase),
      tagline: str(d.tagline),
      skill: str(d.skill),
      runId: str(d.run_id),
      headline: readHeadline(d.headline),
      handles,
      outcomeFields: outcomeFields.filter(Boolean),
      sense: SENSES.includes(d.sense) ? d.sense : null,
      liveCamera: d.live_camera === true,
      available: d.available === true,
      missing: (Array.isArray(d.missing) ? d.missing : []).map(str).filter(Boolean),
    });
  }
  return { ok: true, demos, problems };
}

// ---- handles ---------------------------------------------------------------------------------------------------------
const decimals = (step) => { const s = String(step); return s.includes("e-") ? Number(s.split("e-")[1]) : (s.split(".")[1] ?? "").length; };

// A value snapped to the handle's grid (min + k x step) and kept inside its range.
export function snap(handle, v) {
  if (!finite(v)) return handle.default;
  const k = Math.round((v - handle.min) / handle.step);
  const x = Math.min(handle.max, Math.max(handle.min, handle.min + k * handle.step));
  return Number(x.toFixed(decimals(handle.step)));
}

export const defaultParams = (demo) => Object.fromEntries(demo.handles.map((h) => [h.name, h.default]));

// Where a value sits in its range, 0..1.
export const fraction = (handle, v) => (snap(handle, v) - handle.min) / (handle.max - handle.min);

const UNIT_GLUE = { deg: "°", "°": "°", "%": "%" };
// A number with its unit as people read it: 12°, 0.3 m, 1.5 m/s; a minus sign, not a hyphen.
export function valueText(v, unit = "", digits = null) {
  if (!finite(v)) return "";
  const d = digits ?? (Number.isInteger(v) ? 0 : Math.abs(v) >= 10 ? 1 : 2);
  const t = Number(v.toFixed(d)).toString().replace("-", "−");
  if (!unit) return t;
  return UNIT_GLUE[unit] ? `${t}${UNIT_GLUE[unit]}` : `${t} ${unit}`;
}
export const handleText = (handle, v) => valueText(snap(handle, v), handle.unit, decimals(handle.step));

// A slope handle's value as an angle in degrees, when its unit says it is one; null otherwise.
export function angleDeg(handle, v) {
  const u = handle.unit.toLowerCase();
  if (u === "deg" || u === "°" || u === "degree" || u === "degrees") return v;
  if (u === "rad") return (v * 180) / Math.PI;
  return null;
}

// The cross-section of a smooth hill (a spherical cap) whose steepest slope, at its rim, is `slopeDeg`, drawn with a
// footprint half-width `a` in a box: {path, top, rim}. A cap of rim slope s has sphere radius R = a / sin s and height
// h = R (1 - cos s). Returns null for an angle outside (0, 90).
export function capProfile(slopeDeg, { cx, ground, a }) {
  if (!(slopeDeg > 0 && slopeDeg < 90)) return null;
  const s = (slopeDeg * Math.PI) / 180, R = a / Math.sin(s), h = R * (1 - Math.cos(s));
  const r = (n) => Math.round(n * 100) / 100;
  return { path: `M ${r(cx - a)} ${r(ground)} A ${r(R)} ${r(R)} 0 0 1 ${r(cx + a)} ${r(ground)} Z`, top: r(ground - h), height: h };
}

// ---- outcome lines ---------------------------------------------------------------------------------------------------
const NUMBER_UNIT = { seconds: "s", metres: "m", count: "" };

// One lane's outcome -> {text, detail, tone}: text is the labels of the flags that are true, in the declared order
// ("Reached", "Got hit"); detail is each remaining number field as "<label> <value> <unit>".
// With no true flag the numbers are the text. tone: "bad" when a true flag is declared bad, "good" when one is
// declared good, else "neutral". null when the outcome holds none of the declared fields.
export function outcomeLine(fields, outcome) {
  if (!outcome || typeof outcome !== "object" || !Array.isArray(fields)) return null;
  const flags = fields.filter((f) => f.kind === "flag" && (outcome[f.name] === true || outcome[f.name] === 1));
  const numbers = fields.filter((f) => f.kind !== "flag" && finite(outcome[f.name]))
    .map((f, k) => `${k ? f.label.replace(/^./, (c) => c.toLowerCase()) : f.label} ${valueText(outcome[f.name], NUMBER_UNIT[f.kind], f.kind === "count" ? 0 : 1)}`);
  if (!flags.length && !numbers.length) return null;
  const tone = flags.some((f) => f.tone === "bad") ? "bad" : flags.some((f) => f.tone === "good") ? "good" : "neutral";
  if (!flags.length) return { text: numbers.join(", "), detail: null, tone };
  return { text: flags.map((f) => f.label).join(", "), detail: numbers.length ? numbers.join(", ") : null, tone };
}

// ---- the result sentence under the handles -------------------------------------------------------------------------
// A demo's headline -> one sentence, or null: "Tested on 41 situations it never saw while learning. Reached the spot:
// 25 of 41 before, 31 of 41 with the learned reflex."
export function headlineSentence(h) {
  if (!h) return null;
  const show = (v) => (h.unit === "fraction" && h.n ? `${Math.round(v * h.n)} of ${h.n}` : h.unit === "fraction" ? `${Math.round(v * 100)}%`
    : h.unit === "count" ? String(Math.round(v)) : valueText(v, h.unit));
  const lead = h.n ? `Tested on ${h.n} situations it never saw while learning. ` : "Tested on situations it never saw while learning. ";
  return `${lead}${h.metricLabel}: ${show(h.before)} before, ${show(h.after)} with the learned reflex.`;
}
