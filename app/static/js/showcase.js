// The showcase: a demo's saved results and how it learned (GET /api/demos "showcase", "lanes", "name"), the result of
// one lane in plain words, the live readout of the spiking brain ("brain" events) and the numbers formatted as people
// read them. Pure. Every number comes from the server's data; a part the data leaves out is null and is left out of the
// page, never filled in.

const finite = (v) => typeof v === "number" && Number.isFinite(v);
const str = (v) => (typeof v === "string" && v.trim() ? v.trim() : null);
const count = (v) => (Number.isInteger(v) && v >= 0 ? v : null);
const amount = (v) => (finite(v) && v >= 0 ? v : null);
const truthy = (v) => v === true || v === 1;
const lowerFirst = (s) => s.replace(/^./, (c) => c.toLowerCase());
const MINUS = "−";

// ---- reading -----------------------------------------------------------------------------------------------------------
export const ROW_KINDS = ["count", "metres", "seconds", "fraction"];

// lanes -> {before, after}, each a label or null.
export const readLanes = (raw) => ({ before: str(raw?.before), after: str(raw?.after) });

function readRow(r) {
  if (!r || typeof r !== "object" || !str(r.field) || !str(r.label) || !ROW_KINDS.includes(r.kind) || !finite(r.before) || !finite(r.after)) return null;
  const of = count(r.of);
  if (r.kind === "count" && !(of > 0)) return null; // "12 of ?" cannot be said
  return { field: str(r.field), label: str(r.label), kind: r.kind, before: r.before, after: r.after, of, better: r.better === "lower" || r.better === "higher" ? r.better : null };
}

function readResults(raw) {
  if (!raw || typeof raw !== "object") return null;
  const rows = (Array.isArray(raw.rows) ? raw.rows : []).map(readRow).filter(Boolean);
  if (!rows.length) return null;
  return { n: count(raw.n) || null, split: str(raw.split), rows };
}

// showcase -> {results: null | {n, split, rows}, learning: {realTries, imagined, neurons, trainS, learnS, decisionMs, spikesPerDecision}}
export function readShowcase(raw) {
  const l = raw?.learning && typeof raw.learning === "object" ? raw.learning : {};
  return {
    results: readResults(raw?.results),
    learning: {
      realTries: count(l.real_tries) || null,
      imagined: count(l.imagined) || null,
      neurons: count(l.neurons) || null,
      trainS: amount(l.train_s),
      learnS: amount(l.learn_s),
      decisionMs: amount(l.decision_ms),
      spikesPerDecision: amount(l.spikes_per_decision),
    },
  };
}

// ---- numbers as people read them ---------------------------------------------------------------------------------------
const sign = (t) => t.replace("-", MINUS);

// 1385 -> "1,385"
export const fmtInt = (n) => (finite(n) ? sign(Math.round(n).toLocaleString("en-US")) : "");

// Milliseconds with the digits that matter: 0.1312 -> "0.13", 0.0123 -> "0.012", 1.54 -> "1.5", 1.02 -> "1.0" (one
// decimal always, so a live readout does not jump between "1 ms" and "1.1 ms"), 12.3 -> "12".
export function fmtMs(ms) {
  if (!finite(ms)) return "";
  if (ms >= 10) return fmtInt(ms);
  if (ms >= 1) return sign(ms.toFixed(1));
  return sign(String(Number(ms.toPrecision(2))));
}

// Seconds: 14.4 -> "14 s", 4.25 -> "4.3 s", 0.46 -> "0.46 s".
export function fmtSeconds(s) {
  if (!finite(s)) return "";
  const t = s >= 10 ? fmtInt(s) : s >= 1 ? String(Number(s.toFixed(1))) : String(Number(s.toPrecision(2)));
  return `${sign(t)} s`;
}

// A duration as minutes and seconds: 393.2 -> "6:33", 59.7 -> "1:00".
export function fmtClock(s) {
  if (!finite(s) || s < 0) return "";
  const total = Math.round(s), m = Math.floor(total / 60), r = total % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

// A duration in minutes, one decimal under ten: 393.2 -> "6.6", 1500 -> "25".
export function fmtMinutes(s) {
  if (!finite(s) || s < 0) return "";
  const m = s / 60;
  return m >= 10 ? String(Math.round(m)) : String(Number(m.toFixed(1)));
}

// Metres with two significant digits: 0.678 -> "0.68 m", 4.105 -> "4.1 m", 12.3 -> "12 m".
export function fmtMetres(m) {
  if (!finite(m)) return "";
  const a = Math.abs(m), t = a >= 10 ? String(Math.round(m)) : String(Number(m.toPrecision(2)));
  return `${sign(t)} m`;
}

// One side of a results row: "28 of 30", "0.68 m", "12 s", "93%".
export function rowValue(row, side) {
  const v = row?.[side];
  if (!finite(v)) return "";
  if (row.kind === "count") return `${fmtInt(v)} of ${fmtInt(row.of)}`;
  if (row.kind === "metres") return fmtMetres(v);
  if (row.kind === "seconds") return fmtSeconds(v);
  return `${fmtInt(v * 100)}%`;
}

// ---- the action the brain chose, in plain words --------------------------------------------------------------------------
// The reflex's choices are named after their command, "yaw+0.70_vx0.60": the yaw rate in rad/s, positive turning
// anticlockwise seen from above, that is to the left (config.yaml commands.fields.yaw_rate). The dodge choices are the
// command names of skills.dodge_vision; UNCERTAIN is the gate's "not sure yet", which sends the zero command.
const ACTION_WORDS = { NO_OP: "Watching", UNCERTAIN: "Not sure yet", DODGE_LEFT: "Step left", DODGE_RIGHT: "Step right", RETREAT: "Step back" };

export function choiceWords(name) {
  const n = str(name);
  if (!n) return null;
  const yaw = /^yaw([+-]?\d+(?:\.\d+)?)(?:_|$)/.exec(n);
  if (yaw) { const y = Number(yaw[1]); return y > 0 ? "Steer left" : y < 0 ? "Steer right" : "Straight on"; }
  if (ACTION_WORDS[n]) return ACTION_WORDS[n];
  return n.replace(/[_-]+/g, " ").toLowerCase().replace(/^./, (c) => c.toUpperCase());
}

// ---- one lane's result ---------------------------------------------------------------------------------------------------
const WORD_TONE = { Hit: "bad", Fell: "bad", Dodged: "good" };

// A lane's measured outcome -> {word, big, detail, tone} or null when it holds nothing this can say: one plain result,
// no numbers. word is what the session tally counts. The server's result word comes first (the dodge demo: Hit,
// Dodged, Not hit, Fell). A finish-line round (the hills demo) is one yes or no: "Reached the finish line" or "Did not
// reach the finish line", and a fall shows as "Fell" (it did not reach it); its tally word is Reached or Not reached.
export function laneResult(outcome) {
  if (!outcome || typeof outcome !== "object") return null;
  const word = str(outcome.result);
  if (word) return { word, big: word, detail: null, tone: WORD_TONE[word] ?? "neutral" };
  if ("crossed_goal_line" in outcome) {
    if (truthy(outcome.fall)) return { word: "Not reached", big: "Fell", detail: null, tone: "bad" };
    if (truthy(outcome.crossed_goal_line)) return { word: "Reached", big: "Reached the finish line", detail: null, tone: "good" };
    return { word: "Not reached", big: "Did not reach the finish line", detail: null, tone: "neutral" };
  }
  if (truthy(outcome.fall)) return { word: "Fell", big: "Fell", detail: null, tone: "bad" };
  return null;
}

// ---- the spiking brain, live -----------------------------------------------------------------------------------------------
export const BRAIN_KEEP = 40;   // decisions kept for the sparkline
export const BRAIN_MEDIAN = 20; // decisions in the live median

export const emptyBrain = () => ({ list: [] });

// One "brain" event -> the next state. Only the learned lane's events with a measured decision time count.
export function applyBrain(state, ev, keep = BRAIN_KEEP) {
  if (!ev || typeof ev !== "object" || ev.side !== "after" || !finite(ev.decision_ms) || ev.decision_ms < 0) return state;
  const item = { ms: ev.decision_ms, choice: str(ev.choice), issued: typeof ev.issued === "boolean" ? ev.issued : null, spikes: finite(ev.spikes) && ev.spikes >= 0 ? ev.spikes : null, t: finite(ev.t_s) ? ev.t_s : null };
  return { list: [...state.list, item].slice(-keep) };
}

export function median(values) {
  const v = values.filter(finite).sort((a, b) => a - b);
  if (!v.length) return null;
  const k = v.length >> 1;
  return v.length % 2 ? v[k] : (v[k - 1] + v[k]) / 2;
}

// A dodge move stays on screen this long (simulated seconds) after the gate issued it, so the audience can read it.
export const MOVE_SHOW_S = 1.0;

// What the dog is doing, in plain words. The hills reflex sends a command at every decision (issued null): its newest
// choice. The dodge network gives an opinion every frame but the dog only acts when the gate issues a command, so the
// screen shows the last issued move for MOVE_SHOW_S, and "Watching" otherwise; opinions not acted on are not shown.
function doing(list) {
  const now = list.at(-1);
  if (now.issued === null) return choiceWords(now.choice);
  const move = [...list].reverse().find((x) => x.issued === true);
  const fresh = move && finite(move.t) && finite(now.t) && now.t >= move.t && now.t - move.t <= MOVE_SHOW_S;
  return fresh ? choiceWords(move.choice) : "Watching";
}

// -> null before the first decision, else {ms: median of the last BRAIN_MEDIAN decision times, choice: what the dog is
// doing in plain words (see doing), spikes: the spike counts of the kept decisions (null when none was sent), n}.
export function brainView(state, window = BRAIN_MEDIAN) {
  const list = state?.list ?? [];
  if (!list.length) return null;
  const spikes = list.map((x) => x.spikes);
  return { ms: median(list.slice(-window).map((x) => x.ms)), choice: doing(list), spikes: spikes.some((s) => s !== null) ? spikes : null, n: list.length };
}

// ---- how it learned: the steps ------------------------------------------------------------------------------------------
// The demo screen's inline steps: [{key, strong, rest}], strong null when the number is not in the data. A fourth step,
// how fast the spiking brain decides (its saved timing, the network alone), only when the run measured it.
export function storySteps(learning) {
  const l = learning ?? {};
  const trained = finite(l.trainS) ? `, trained in ${fmtSeconds(l.trainS)}` : "";
  const steps = [
    l.realTries !== null && l.realTries !== undefined ? { key: "tries", strong: fmtInt(l.realTries), rest: " real tries" } : { key: "tries", strong: null, rest: "Real tries in the simulator" },
    l.imagined !== null && l.imagined !== undefined ? { key: "imagine", strong: fmtInt(l.imagined), rest: " situations imagined by TabPFN" } : { key: "imagine", strong: null, rest: "Situations imagined by TabPFN" },
    l.neurons !== null && l.neurons !== undefined ? { key: "brain", strong: fmtInt(l.neurons), rest: ` neurons in a spiking brain${trained}` } : { key: "brain", strong: null, rest: `A spiking brain${trained}` },
  ];
  if (finite(l.decisionMs)) steps.push({ key: "decide", strong: `${fmtMs(l.decisionMs)} ms`, rest: " for the brain alone to decide" });
  return steps;
}

// The whole lesson's time, real tries to trained brain: "Learned in 6.5 minutes, start to finish".
export function learnedIn(learning) {
  const s = learning?.learnS;
  if (!finite(s)) return null;
  return `Learned in ${s < 60 ? fmtSeconds(s) : `${fmtMinutes(s)} minutes`}, start to finish`;
}

// The landing page's three steps, each with the numbers of every demo that has them:
// [{key, title, sentence, values: [{demo, value, unit}]}].
export function homeSteps(demos) {
  const of = (get, fmt, unit) => demos.map((d) => ({ d, v: get(d.showcase?.learning ?? {}) }))
    .filter(({ v }) => v !== null && v !== undefined && finite(v)).map(({ d, v }) => ({ demo: d.name ?? d.title, value: fmt(v), unit }));
  return [
    { key: "tries", title: "Real tries", sentence: "The dog practises for real in the simulator. Real tries are slow, so they are the smallest part of what it learns from.",
      values: of((l) => l.realTries, fmtInt, "real tries") },
    { key: "imagine", title: "TabPFN imagines", sentence: "TabPFN, an AI model trained in advance to read tables of numbers, predicts how many more situations would turn out, without trying them.",
      values: of((l) => l.imagined, fmtInt, "situations imagined") },
    { key: "brain", title: "A spiking brain learns", sentence: "A spiking neural network, a tiny brain that works in pulses like real neurons, learns the best choice from those imagined situations.",
      values: of((l) => l.neurons, fmtInt, "neurons") },
  ];
}

// The hero's sentence. The last clause only when every published decision time is under a millisecond.
// "many times more" only when every demo with both counts imagined at least ten times its real tries.
export function heroSub(demos) {
  const times = demos.map((d) => d.showcase?.learning?.decisionMs).filter(finite);
  const fast = times.length > 0 && times.every((t) => t < 1);
  const pairs = demos.map((d) => d.showcase?.learning ?? {}).filter((l) => finite(l.realTries) && finite(l.imagined));
  const many = pairs.length > 0 && pairs.every((l) => l.imagined >= 10 * l.realTries);
  return `Real practice is slow, so it gets only a little. TabPFN imagines ${many ? "many times more situations" : "more situations"}, and a tiny spiking brain learns the right choices from them.${fast ? " On its own, the brain decides in a fraction of a millisecond." : ""}`;
}

// The landing page's big numbers: at most three, each only when its data exists: [{key, value, label, sub}].
export function statsBand(demos) {
  const out = [];
  const first = (get) => demos.find((d) => get(d) !== null && get(d) !== undefined);
  const dec = first((d) => d.showcase?.learning?.decisionMs);
  if (dec) out.push({ key: "decision", value: fmtMs(dec.showcase.learning.decisionMs), label: "milliseconds for the spiking brain to decide, on this laptop", sub: null });
  const learn = first((d) => d.showcase?.learning?.learnS);
  if (learn) {
    const s = learn.showcase.learning.learnS, what = `the ${(learn.name ?? learn.title).toLowerCase()}`;
    out.push(s < 60 ? { key: "learn", value: fmtInt(s), label: `seconds to learn ${what}, start to finish`, sub: null }
      : { key: "learn", value: fmtMinutes(s), label: `minutes to learn ${what}, start to finish`, sub: null });
  }
  // "28 of 30 / reached the finish line with the learned reflex / 12 of 30 walking alone"
  const res = demos.map((d) => ({ d, row: d.showcase?.results?.rows.find((r) => r.kind === "count") })).find((x) => x.row);
  if (res) {
    const { d, row } = res, lanes = d.lanes ?? {};
    out.push({ key: "result", value: rowValue(row, "after"), label: lanes.after ? `${lowerFirst(row.label)} ${lowerFirst(lanes.after)}` : lowerFirst(row.label),
      sub: `${rowValue(row, "before")} ${lanes.before ? lowerFirst(lanes.before) : "before it learned"}` });
  }
  return out;
}

// ---- this session, from real rounds only -------------------------------------------------------------------------------
// A lane's counts {word: n} -> rows [{label, before, after}] with "n of total", in a fixed order, at most `max` rows.
// The good word of the round's kind (Dodged, Reached) always has a row, even at 0. A finish-line round counts one
// row only: reached the finish line or not.
const SESSION_ORDER = ["Dodged", "Hit", "Not hit", "Reached", "Fell"];
const GOOD_WORD = { Hit: "Dodged", "Not hit": "Dodged", Dodged: "Dodged", Reached: "Reached", "Not reached": "Reached" };
const SESSION_LABEL = { Reached: "Reached the finish line" };
export function sessionRows(counts, max = 3) {
  const before = counts?.before ?? {}, after = counts?.after ?? {};
  const total = (c) => Object.values(c).reduce((a, n) => a + n, 0);
  const tb = total(before), ta = total(after);
  if (!tb && !ta) return { rounds: 0, rows: [] };
  const seen = [...Object.keys(before), ...Object.keys(after)];
  const words = [...new Set([...seen, ...seen.map((w) => GOOD_WORD[w]).filter(Boolean)])].filter((w) => w !== "Not reached")
    .sort((a, b) => (SESSION_ORDER.indexOf(a) + 1 || 99) - (SESSION_ORDER.indexOf(b) + 1 || 99)).slice(0, max);
  // good: the row counts a good result (Dodged, Reached).
  return { rounds: Math.max(tb, ta), rows: words.map((w) => ({ label: SESSION_LABEL[w] ?? w, before: tb ? `${before[w] ?? 0} of ${tb}` : "", after: ta ? `${after[w] ?? 0} of ${ta}` : "", good: GOOD_WORD[w] === w })) };
}
