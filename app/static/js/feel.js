// What the dog feels and sees, for the sense strips under each lane. Pure: reading the "feel" and "eye" events, keeping
// the last few seconds on the simulator's clock, a scale shared by both lanes, and the strip's drawing geometry.
import { sideOf } from "./duel.js";

const finite = (v) => typeof v === "number" && Number.isFinite(v);
const nums = (v, n) => Array.isArray(v) && v.length === n && v.every(finite);

// A "feel" event -> {side, t, tilt: [noseUp, roll], turn: [x, y, z], feet: [4 booleans]} or null when unreadable.
export function readFeel(ev) {
  if (!ev || !finite(ev.t_s) || !nums(ev.tilt_deg, 2) || !nums(ev.turn_rate, 3) || !Array.isArray(ev.feet) || ev.feet.length !== 4) return null;
  return { side: sideOf(ev), t: ev.t_s, tilt: [...ev.tilt_deg], turn: [...ev.turn_rate], feet: ev.feet.map((f) => f === 1 || f === true) };
}

// An "eye" event -> {side, t, cells: Float32Array} (values 0..1, row-major) or null. size: the cells the server's
// image_window declares (height x width), or null to accept any square.
export function readEye(ev, size = null) {
  const c = ev?.cells;
  if (!Array.isArray(c) || !c.length || !c.every((v) => finite(v) && v >= 0 && v <= 1)) return null;
  if (size !== null ? c.length !== size : !Number.isInteger(Math.sqrt(c.length))) return null;
  return { side: sideOf(ev), t: finite(ev.t_s) ? ev.t_s : null, cells: Float32Array.from(c) };
}

// The last spanS seconds of samples {t, ...}, oldest first; a clock that runs back (a new run) starts again.
export function createTrace(spanS) {
  let items = [];
  return {
    push(s) {
      if (items.length && s.t < items.at(-1).t) items = [];
      items.push(s);
      const from = s.t - spanS;
      let k = 0;
      while (k < items.length - 1 && items[k + 1].t <= from) k++;
      if (k) items = items.slice(k);
    },
    samples: () => items,
    last: () => items.at(-1) ?? null,
    clear() { items = []; },
  };
}

// The largest |value| of one channel over the given sample lists (both lanes), so the two strips share one scale and
// can be compared by eye. null when there is no value or every value is 0.
export function sharedScale(lists, get) {
  let m = 0;
  for (const list of lists) for (const s of list) { const v = get(s); if (finite(v)) m = Math.max(m, Math.abs(v)); }
  return m > 0 ? m : null;
}

// A channel as a polyline: time runs left to right over the last spanS seconds ending at tEnd; 0 sits on yMid and
// +scale reaches yMid - half (up). Points outside the window are left out. "x,y x,y ..." or "" with fewer than two.
export function tracePath(samples, get, { x0, x1, yMid, half, tEnd, spanS, scale }) {
  if (!(spanS > 0) || !(x1 > x0)) return "";
  const pts = [];
  for (const s of samples) {
    const v = get(s);
    if (!finite(v) || s.t < tEnd - spanS || s.t > tEnd) continue;
    const x = x0 + ((s.t - (tEnd - spanS)) / spanS) * (x1 - x0);
    const y = yMid - (scale ? Math.max(-1, Math.min(1, v / scale)) : 0) * half;
    pts.push(`${Math.round(x * 10) / 10},${Math.round(y * 10) / 10}`);
  }
  return pts.length >= 2 ? pts.join(" ") : "";
}

// One foot's contact as runs [[xStart, xEnd], ...] over the same window: a run lasts from the first sample in contact
// to the first sample out of contact (or the newest sample).
export function contactRuns(samples, foot, { x0, x1, tEnd, spanS }) {
  const xOf = (t) => x0 + ((Math.max(t, tEnd - spanS) - (tEnd - spanS)) / spanS) * (x1 - x0);
  const runs = [];
  let start = null;
  for (const s of samples) {
    if (s.t > tEnd) break;
    if (s.feet[foot] && start === null) start = s.t;
    else if (!s.feet[foot] && start !== null) { if (s.t >= tEnd - spanS) runs.push([xOf(start), xOf(s.t)]); start = null; }
  }
  if (start !== null && samples.length) runs.push([xOf(start), xOf(Math.min(tEnd, samples.at(-1).t))]);
  return runs.map(([a, b]) => [Math.round(a * 10) / 10, Math.round(b * 10) / 10]).filter(([a, b]) => b > a);
}
