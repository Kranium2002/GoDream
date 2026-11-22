// Plain words for visitors, built only from what the server sends: skill names, the five lesson steps, result headlines
// and tiles, goal notes, episode captions, lesson choices and the cost question. Pure. Labels are presentation; every
// number comes from the data and a missing one is left out, never filled in.

const finite = (v) => typeof v === "number" && Number.isFinite(v);
const str = (v) => (typeof v === "string" && v ? v : null);

// ---- skills ---------------------------------------------------------------------------------------------------------
const SKILLS = {
  RAMP: { title: "Climbing a ramp", task: "climb a ramp" },
  MOVE: { title: "Walking to a mark", task: "walk to a mark and stop" },
  DODGE: { title: "Dodging", task: "dodge what comes at it" },
};
const SPEC_TO_SKILL = { ramp: "RAMP", move: "MOVE", dodge_vision: "DODGE" };

// A skill id as runs, events and the catalogue name it (RAMP, ramp, dodge_vision) -> the catalogue's name, or null.
export function skillKey(id, catalogue = null) {
  if (!str(id)) return null;
  const entry = Array.isArray(catalogue) ? catalogue.find((s) => s && (s.skill === id || s.spec === id)) : null;
  if (entry && str(entry.skill)) return entry.skill;
  return SPEC_TO_SKILL[id] ?? id.toUpperCase();
}
export const skillTitle = (key) => SKILLS[key]?.title ?? (str(key) ? humanize(key) : "A skill");
export const skillTask = (key) => SKILLS[key]?.task ?? null;

export function humanize(name) {
  const s = String(name).replace(/_(m_s|m|cm|deg|s|rad)$/, "").replace(/_/g, " ").trim().toLowerCase();
  return s ? s[0].toUpperCase() + s.slice(1) : String(name);
}

// ---- the request, as understood -------------------------------------------------------------------------------------
const PARAMS = { incline_deg: "Ramp angle", nominal_speed_m_s: "Walking speed", forward_m: "Distance forward", sideways_m: "Distance sideways",
  turn_deg: "Turn", threats: "What comes at it" };
const UNIT_TEXT = { deg: "°" };
const valueText = (v, unit) => {
  const t = Array.isArray(v) ? v.map(String).join(" and ") : typeof v === "number" ? String(Number(v.toPrecision(4))) : String(v);
  if (!unit) return t;
  return UNIT_TEXT[unit] ? `${t}${UNIT_TEXT[unit]}` : `${t} ${unit}`;
};

// A chat answer -> {ok, heard, settings: [{label, value}], usual, refused: [text], problems: [text], reply}.
// heard is "Got it: <the person's own words>." when a declaration came back.
export function understood(chat, catalogue) {
  const decl = chat?.declared_spec && typeof chat.declared_spec === "object" ? chat.declared_spec : null;
  const entry = decl && Array.isArray(catalogue) ? catalogue.find((s) => s?.skill === decl.skill) : null;
  const declared = decl?.params && typeof decl.params === "object" ? decl.params : {};
  const settings = Object.entries(declared).map(([name, v]) => ({ label: PARAMS[name] ?? humanize(name), value: valueText(v, entry?.params?.[name]?.unit) }));
  const words = str(decl?.source_text)?.trim().replace(/[.!?\s]+$/, "");
  const problems = (Array.isArray(chat?.problems) ? chat.problems : []).map((p) => (typeof p?.message === "string" ? p.message : null)).filter(Boolean);
  const refused = (Array.isArray(chat?.missing) ? chat.missing : []).map((m) => (typeof m === "string" ? m : null)).filter(Boolean);
  return {
    ok: Boolean(decl) && !problems.length && !refused.length && entry?.available !== false,
    skill: decl ? skillKey(decl.skill, catalogue) : null,
    heard: decl ? `Got it: ${words ? words[0].toLowerCase() + words.slice(1) : skillTask(skillKey(decl.skill, catalogue)) ?? "that"}.` : null,
    settings,
    usual: Boolean(decl) && settings.length === 0,
    unsupported: entry?.available === false ? (Array.isArray(entry.missing) ? entry.missing.map(String) : []) : [],
    refused,
    problems,
    reply: str(chat?.reply),
  };
}

// ---- lessons: modes, durations, the cost question -------------------------------------------------------------------
export function aboutTime(seconds) {
  if (!finite(seconds) || seconds < 0) return null;
  if (seconds < 90) return "about 1 minute";
  if (seconds < 90 * 60) return `about ${Math.round(seconds / 60)} minutes`;
  return `about ${Math.round(seconds / 360) / 10} hours`;
}

// The newest finished run (state done) of a skill, optionally of one mode (from its recorded timing). cards: flow.cardsFrom.
export function latestDone(cards, skill, mode = null, catalogue = null) {
  return [...cards].filter((c) => c.state === "done" && skillKey(c.skill, catalogue) === skill && (mode === null || c.timing?.mode === mode))
    .sort((a, b) => String(b.updated ?? "").localeCompare(String(a.updated ?? "")))[0] ?? null;
}

// /api/state run_modes for one skill -> {choices: [{mode, label, learn, test, timedBy}], quickRefused: reason | null}.
// learn and test are durations measured on this computer by the newest finished run of that skill and mode, or null.
export function lessonChoices(runModes, skill, cards, catalogue = null) {
  const modes = runModes && typeof runModes === "object" ? runModes[skill] : null;
  if (!modes || typeof modes !== "object") return { choices: [], quickRefused: null };
  const order = ["fast", "full", ...Object.keys(modes).filter((m) => m !== "fast" && m !== "full")];
  const choices = order.filter((m) => modes[m] === "").map((mode) => {
    const run = latestDone(cards, skill, mode, catalogue);
    const t = run?.timing;
    return { mode, label: mode === "fast" ? "Quick lesson" : mode === "full" ? "Full lesson" : humanize(mode),
      learn: aboutTime(t?.learn_s), test: aboutTime(t?.evaluate_s), timedBy: run?.runId ?? null };
  });
  return { choices, quickRefused: typeof modes.fast === "string" && modes.fast !== "" ? modes.fast : null };
}

const thousands = (n) => Math.round(n).toLocaleString("en-US");
// The 409 cost_confirmation_required body -> one plain question, from its estimate only.
export function costQuestion(body) {
  const e = body?.estimate;
  const tries = finite(e?.episodes) ? `about ${thousands(e.episodes)} simulated tries` : null;
  const credits = finite(e?.tabpfn_tokens?.value) ? `about ${thousands(e.tabpfn_tokens.value)} teacher credits` : null;
  if (tries && credits) return `This runs ${tries} and uses ${credits}. Start?`;
  if (tries) return `This runs ${tries} and uses the teacher service; its credits could not be estimated here. Start?`;
  if (credits) return `This uses ${credits}. Start?`;
  return "This lesson uses simulator time and teacher credits. Start?";
}

// ---- the five lesson steps --------------------------------------------------------------------------------------------
export const LESSON = [
  { key: "try", title: "Try it for real", what: "The dog tries the task in the simulator.", steps: ["baseline", "attempts"], unit: "tries" },
  { key: "learn", title: "Learn from the tries", what: "A teacher model learns where a simple physics guess goes wrong.", steps: ["teacher_fit"], unit: "" },
  { key: "imagine", title: "Imagine many more situations", what: "The teacher scores thousands of made-up situations.", steps: ["dream"], unit: "" },
  { key: "brain", title: "Train the dog's small brain", what: "A tiny spiking network learns the best move for each one.", steps: ["train"], unit: "rounds" },
  { key: "test", title: "Test on new situations", what: "The dog faces situations it has never seen.", steps: ["evaluate"], unit: "tests" },
];
const ENDED = ["done", "failed", "refused", "stopped"];
const BAD = ["failed", "refused", "stopped"];

// A learn.js timeline -> the five steps: {key, title, what, state, progress: {done, total, unit} | null, current}.
// state: waiting | running | done | stopped (the run ended here early) | unlogged (the run's log has no line for it).
// runState: the run's state when known. A step whose last line says running in a run that is done finished with the
// run (run_skill logs a step's start; some commands log nothing after it).
export function lessonSteps(tl, runState = null) {
  const ended = ENDED.includes(runState);
  const logged = LESSON.map((s) => s.steps.some((k) => tl?.steps?.[k]));
  const rows = LESSON.map((s, i) => {
    const entries = s.steps.map((k) => tl?.steps?.[k]).filter(Boolean);
    const last = entries.reduce((a, b) => (a === null || (b.t ?? -Infinity) >= (a.t ?? -Infinity) ? b : a), null);
    let state = last ? last.state : ended ? "unlogged" : "waiting";
    // Started and never closed in an ended run: finished if the run is done or a later step began, else stopped here.
    if (state === "running" && ended) state = runState === "done" || logged.slice(i + 1).some(Boolean) ? "done" : "stopped";
    if (BAD.includes(state)) state = "stopped";
    if (!["waiting", "running", "done", "stopped", "unlogged"].includes(state)) state = "running";
    const withCount = [...entries].reverse().find((e) => finite(e.done) && finite(e.total) && e.total > 0);
    return { key: s.key, title: s.title, what: s.what, state, current: false,
      progress: withCount ? { done: withCount.done, total: withCount.total, unit: s.unit } : null };
  });
  const running = rows.findIndex((r) => r.state === "running");
  const stopped = rows.findIndex((r) => r.state === "stopped");
  if (running >= 0) rows[running].current = true;
  else if (stopped >= 0) rows[stopped].current = true;
  else if (!ended) {
    const next = rows.findIndex((r) => r.state === "waiting" && rows.slice(0, rows.indexOf(r)).some((p) => p.state === "done"));
    if (next >= 0) rows[next].current = true;
  }
  return rows;
}

// ---- results --------------------------------------------------------------------------------------------------------
const METRICS = {
  ascent_success_rate: { label: "Reaches the top of the ramp", says: (a, n) => `reaches the top of the ramp ${a} times out of ${n}` },
  fall_rate: { label: "Falls", says: (a, n) => `falls ${a} times out of ${n}` },
  collision_rate: { label: "Gets hit", says: (a, n) => `gets hit ${a} times out of ${n}` },
  recall: { label: "Threats it moved away from", says: (a, n) => `moves away from ${a} of ${n} threats` },
  active_precision: { label: "Moves that were needed" },
  mean_progress_m: { label: "Distance up the ramp, on average" },
  mean_slip_m: { label: "Foot slip, on average" },
  flat_false_activations: { label: "False alarms on flat ground" },
  mean_error_forward_cm: { label: "Miss when walking forward or back", says: (a) => `misses its mark by ${a} on average when walking forward or back` },
  mean_error_sideways_cm: { label: "Miss when stepping sideways", says: (a) => `misses its mark by ${a} on average when stepping sideways` },
  mean_error_turn_deg: { label: "Miss when turning", says: (a) => `misses its turn by ${a} on average` },
  mean_min_separation_m: { label: "Closest distance to the object, on average" },
  false_activations_per_safe_minute: { label: "Moves for nothing, per minute" },
};
export const metricLabel = (name) => METRICS[name]?.label ?? humanize(name);

// The two methods a result compares: the learned one (snn, else student, else the last) and the one before it.
function pairOf(metric) {
  const keys = Object.keys(metric?.values ?? {});
  const after = ["snn", "student"].find((k) => keys.includes(k)) ?? keys.at(-1) ?? null;
  const before = keys.find((k) => k !== after) ?? null;
  return { before, after };
}

// One value of a metric for people: {text, count, n}. A fraction with its n is shown as the count it was measured
// from (value x n, which gives back the whole number counted); a count as a whole number; a mean with its unit.
function shown(metric, cell) {
  const v = cell?.value, n = finite(cell?.n) ? cell.n : null;
  if (!finite(v)) return { text: "—", count: null, n, value: null };
  const unit = typeof metric.unit === "string" ? metric.unit : "";
  if (unit === "fraction" && n !== null && n > 0) { const c = Math.round(v * n); return { text: String(c), count: c, n, value: v }; }
  if (unit === "fraction") return { text: `${Math.round(v * 100)}%`, count: null, n, value: v };
  if (unit === "count") return { text: String(Math.round(v)), count: Math.round(v), n, value: v };
  const digits = unit === "m" || unit === "1/min" ? 2 : 1;
  const u = unit === "deg" ? "°" : unit === "1/min" ? " per minute" : unit ? ` ${unit}` : "";
  return { text: `${v.toFixed(digits)}${u}`, count: null, n, value: Number(v.toFixed(digits)) };
}

const usable = (m) => m && typeof m.name === "string" && m.values && typeof m.values === "object";
export const metricsOf = (result) => (Array.isArray(result?.metrics) ? result.metrics.filter(usable) : []);

// A metric -> a before/after tile: {name, label, before, after, change: better | worse | same | null, basis}.
export function tileOf(metric) {
  const { before, after } = pairOf(metric);
  const b = shown(metric, metric.values[before]), a = shown(metric, metric.values[after]);
  let change = null;
  if (a.value !== null && b.value !== null) {
    const x = a.count ?? a.value, y = b.count ?? b.value;
    change = x === y ? "same" : (x > y) === (metric.higher_is_better !== false) ? "better" : "worse";
  }
  const n = a.n ?? b.n;
  const basis = n === null ? null : metric.unit === "fraction" ? `out of ${n} tests` : metric.unit === "count" ? `in ${n} tests` : `over ${n} tests`;
  return { name: metric.name, label: metricLabel(metric.name), before: b.text, after: a.text, change, basis, beforeKey: before, afterKey: after };
}

// Two to four tiles: the result's own metrics in its order, those with a learned value first-come.
export function resultTiles(result, max = 4) {
  return metricsOf(result).map(tileOf).filter((t) => t.after !== "—").slice(0, max);
}

// The headline: one sentence from the first metric with both values. attempts: the run's real tries, when recorded.
export function resultHeadline(result, attempts = null) {
  const m = metricsOf(result).find((x) => { const t = tileOf(x); return t.after !== "—" && t.before !== "—"; });
  if (!m) return null;
  const t = tileOf(m), a = shown(m, m.values[t.afterKey]);
  const lead = finite(attempts) ? `After ${attempts} tries,` : "After learning,";
  const says = METRICS[m.name]?.says;
  const body = says ? `the dog ${says(t.after, a.n)}` : `${t.label.toLowerCase()}: ${t.after}`;
  return { metric: m.name, text: `${lead} ${body}. Before: ${t.before}.` };
}

// One line for a skill card: "Reaches the top of the ramp: 31 of 41, up from 25".
export function cardLine(result) {
  const m = metricsOf(result).find((x) => { const t = tileOf(x); return t.after !== "—" && t.before !== "—"; });
  if (!m) return null;
  const t = tileOf(m), a = shown(m, m.values[t.afterKey]);
  const of = a.count !== null && a.n !== null && m.unit === "fraction" ? ` of ${a.n}` : "";
  const x = shown(m, m.values[t.afterKey]), y = shown(m, m.values[t.beforeKey]);
  const dir = (x.count ?? x.value) > (y.count ?? y.value) ? "up from" : (x.count ?? x.value) < (y.count ?? y.value) ? "down from" : "same as before,";
  return `${t.label}: ${t.after}${of}, ${dir} ${t.before}`;
}

// Calm notes from the data: a goal (any number named *_target in the result) the learned value did not reach, and a
// headline change whose paired interval still includes no change.
export function resultNotes(result) {
  const notes = [];
  const targets = new Map();
  (function walk(v) {
    if (!v || typeof v !== "object") return;
    for (const [k, x] of Object.entries(v)) {
      if (/_target$/.test(k) && finite(x)) targets.set(`${k}:${x}`, { stem: k.replace(/_target$/, ""), value: x });
      else if (x && typeof x === "object") walk(x);
    }
  })(result);
  for (const { stem, value } of targets.values()) {
    const m = metricsOf(result).find((x) => x.name.includes(stem));
    if (!m) continue;
    const { after } = pairOf(m), a = shown(m, m.values[after]);
    if (a.value === null) continue;
    const met = m.higher_is_better === false ? a.value <= value : a.value >= value;
    if (met) continue;
    const pct = (v) => `${Math.round(v * 100)}%`;
    const goal = m.unit === "fraction" ? pct(value) : String(value);
    const got = m.unit === "fraction" ? `${pct(a.value)}${a.count !== null ? ` (${a.count} of ${a.n})` : ""}` : a.text;
    notes.push({ kind: "goal", text: `Not there yet. Goal for ${metricLabel(m.name).toLowerCase()}: ${goal}. In testing: ${got}.` });
  }
  const head = resultHeadline(result);
  const paired = Array.isArray(result?.paired) ? result.paired.find((p) => p?.name === head?.metric) : null;
  if (paired && finite(paired.ci_low) && finite(paired.ci_high) && paired.ci_low <= 0 && paired.ci_high >= 0) {
    notes.push({ kind: "chance", text: "With this many tests, this change could still be chance." });
  }
  return notes;
}

// "The numbers": every metric with its interval and count, the paired changes, and how it was measured, in plain words.
const METHODS = { no_op: "without the learned decisions (its usual command)", nominal: "without the learned decisions (its usual command)",
  physics_only: "a simple physics rule", physics: "a simple physics rule", snn: "the small spiking brain it trained", student: "the small brain it trained" };
const pct = (v) => `${Math.round(v * 100)}%`;
function detailed(metric, cell) {
  const s = shown(metric, cell);
  if (s.value === null) return "none";
  const ci = finite(cell?.ci_low) && finite(cell?.ci_high);
  if (metric.unit === "fraction") {
    const head = s.count !== null ? `${s.count} of ${s.n} (${pct(cell.value)})` : pct(cell.value);
    return ci ? `${head}; likely between ${pct(cell.ci_low)} and ${pct(cell.ci_high)}` : head;
  }
  const n = s.n !== null ? ` over ${s.n} tests` : "";
  return ci ? `${s.text}${n}; likely between ${shown(metric, { value: cell.ci_low }).text} and ${shown(metric, { value: cell.ci_high }).text}` : `${s.text}${n}`;
}
// The confidence level the run states for its intervals (any bootstrap.level in the result), or null.
function levelOf(result) {
  let found = null;
  (function walk(v) {
    if (!v || typeof v !== "object" || found !== null) return;
    if (v.bootstrap && typeof v.bootstrap === "object" && finite(v.bootstrap.level)) { found = v.bootstrap.level; return; }
    for (const x of Object.values(v)) walk(x);
  })(result);
  return found;
}
export function numbersTable(result) {
  const metrics = metricsOf(result);
  const rows = metrics.map((m) => { const { before, after } = pairOf(m); return { label: metricLabel(m.name), before: detailed(m, m.values[before]), after: detailed(m, m.values[after]) }; });
  const first = metrics[0] ? pairOf(metrics[0]) : { before: null, after: null };
  const paired = (Array.isArray(result?.paired) ? result.paired : []).filter((p) => finite(p?.diff)).map((p) => {
    const m = metrics.find((x) => x.name === p.name);
    const f = m?.unit === "fraction" ? (v) => `${v > 0 ? "+" : ""}${pct(v)}` : (v) => `${v > 0 ? "+" : ""}${Number(v.toPrecision(3))}`;
    const range = finite(p.ci_low) && finite(p.ci_high) ? `; likely between ${f(p.ci_low)} and ${f(p.ci_high)}` : "";
    return `${m ? metricLabel(m.name) : humanize(p.name)}: ${f(p.diff)} after learning, on the same ${finite(p.n) ? `${p.n} ` : ""}tests${range}.`;
  });
  const seeds = result?.seeds && typeof result.seeds === "object" ? result.seeds : null;
  const facts = [
    first.before ? `Before: ${METHODS[first.before] ?? humanize(first.before).toLowerCase()}.` : null,
    first.after ? `After: ${METHODS[first.after] ?? humanize(first.after).toLowerCase()}.` : null,
    Array.isArray(result?.scenario_ids) ? `${result.scenario_ids.length} test situations the dog had not seen while learning.` : null,
    finite(seeds?.runtime_seed) ? `Random seed ${seeds.runtime_seed}${typeof seeds.split === "string" ? `, ${seeds.split} set` : ""}${typeof seeds.world_preset === "string" ? `, ${seeds.world_preset} world` : ""}.` : null,
    levelOf(result) !== null ? `Ranges are ${Math.round(levelOf(result) * 100)}% intervals, as the run computed them.` : "Ranges are the intervals the run reported.",
    "Everything was measured in simulation.",
  ].filter(Boolean);
  return { rows, paired, facts };
}

// ---- episodes on the stage ------------------------------------------------------------------------------------------
const DODGE_KINDS = { empty: "nothing in view", static: "an object standing still", receding: "an object moving away", off_path: "an object passing by",
  background_motion: "movement in the background", ambiguous: "an unclear movement", incoming_left: "something coming from the left",
  incoming_center: "something coming straight at it", incoming_right: "something coming from the right" };
function situation(skill, kind) {
  if (!str(kind)) return null;
  if (skill === "RAMP") return kind === "slope" ? "a ramp" : kind === "flat" ? "flat ground" : humanize(kind).toLowerCase();
  if (skill === "MOVE") return { forward: "walking forward or back", sideways: "stepping sideways", turn: "turning" }[kind] ?? humanize(kind).toLowerCase();
  if (skill === "DODGE") return DODGE_KINDS[kind] ?? humanize(kind).toLowerCase();
  return humanize(kind).toLowerCase();
}
function outcomeText(skill, kind, o) {
  if (!o || typeof o !== "object") return null;
  const parts = [];
  if (skill === "RAMP") {
    if (o.reached_top === true || o.reached_top === 1) parts.push("reached the top");
    else if (kind === "slope") parts.push("did not reach the top");
  } else if (skill === "MOVE" && finite(o.error_units)) {
    parts.push(o.unit === "rad" ? `missed by ${Math.round((o.error_units * 180) / Math.PI)}°` : `missed the mark by ${Math.round(o.error_units * 100)} cm`);
  } else if (skill === "DODGE" && (typeof o.collision === "boolean" || finite(o.collision))) {
    parts.push(o.collision ? "got hit" : "not hit");
  }
  if (o.fall === true || o.fall === 1) parts.push("fell");
  return parts.length ? parts.join(", ") : null;
}

// An SSE episode event -> {hidden} or {title, phase, phaseText, outcome, detail}. phase: before | after | practice.
export function episodeCaption(ev, catalogue = null) {
  if (!ev || typeof ev !== "object" || !["start", "end"].includes(ev.state)) return { hidden: true };
  const skill = skillKey(ev.skill, catalogue);
  const phase = ev.step === "attempts" ? "practice" : ev.action === "student" ? "after" : "before";
  const word = ev.step === "attempts" ? "Try" : "Test";
  const count = Number.isInteger(ev.index) ? `${word} ${ev.index + 1}${Number.isInteger(ev.total) ? ` of ${ev.total}` : ""}` : word;
  const where = situation(skill, ev.kind);
  return {
    hidden: false,
    title: where ? `${count}: ${where}` : count,
    phase,
    phaseText: { before: "Before learning", after: "After learning", practice: "Practice" }[phase],
    outcome: ev.state === "end" ? outcomeText(skill, ev.kind, ev.outcome) : null,
    detail: [str(ev.caption), str(ev.note)].filter(Boolean).join(" "),
  };
}

// ---- recorded lessons -----------------------------------------------------------------------------------------------
// The speed at which a recorded event log replays in about targetMs (never slower than real time): a whole number.
export function replaySpeed(lines, targetMs = 40000) {
  const times = lines.map((l) => (typeof l?.time_utc === "string" ? Date.parse(l.time_utc) : NaN)).filter((t) => !Number.isNaN(t));
  if (times.length < 2) return 1;
  const span = Math.max(...times) - Math.min(...times);
  return Math.max(1, Math.round(span / targetMs));
}

// ---- where the page is: #teach, #learn/<run>, #results/<run>, #live -------------------------------------------------
export const SCREENS = ["teach", "learn", "results", "live"];
export function parseRoute(hash) {
  const [screen, ...rest] = String(hash ?? "").replace(/^#\/?/, "").split("/");
  const runId = rest.length ? decodeURIComponent(rest.join("/")) : null;
  return SCREENS.includes(screen) ? { screen, runId: runId || null } : { screen: "teach", runId: null };
}
export const routeHash = (screen, runId = null) => `#${screen}${runId && (screen === "learn" || screen === "results") ? `/${encodeURIComponent(runId)}` : ""}`;
