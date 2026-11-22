// Simulator poses -> where to draw the dog now. Each pose carries the simulator's clock (t_s). Poses reach the page
// unevenly: the server publishes at app.pose_hz of simulated time, the simulator runs slower than real time on a
// loaded laptop, and events arrive in bursts. So the drawing plays the poses back on the simulator's clock, at the rate
// the simulator is measured to run, a small delay behind the newest pose (the largest recent gap between arrivals,
// with a margin): a late pose then slows the motion a little instead of stopping it. Playback speeds up when poses
// pile up and slows down when they run short; it never goes past the newest pose and never invents one. A pose without
// t_s is placed on its arrival time. Scene objects (the contract's "scene" event) ride on the poses: a scene belongs to
// the newest pose when it arrives (or to the pose whose t_s it names), and an object is interpolated by its id.
export function createPoseBuffer({ maxEntries = 96, rateWindowMs = 3000, minDelayMs = 50, maxDelayMs = 600, margin = 1.3 } = {}) {
  let entries = [];   // oldest first: {pose, t (simulator s), at (arrival ms), scene}
  let play = null;    // the simulator time being drawn
  let lastNow = null, lastScene = null;

  const timeOf = (pose, at) => (typeof pose?.t_s === "number" && Number.isFinite(pose.t_s) ? pose.t_s : at / 1000);

  function recent(nowAt) {
    const r = entries.filter((e) => nowAt - e.at <= rateWindowMs);
    return r.length >= 2 ? r : entries.slice(-2);
  }
  // Simulated seconds per wall second over the recent arrivals; 1 until two poses span at least 150 ms.
  function rate() {
    const r = recent(entries.at(-1).at);
    const wall = (r.at(-1).at - r[0].at) / 1000, sim = r.at(-1).t - r[0].t;
    return wall >= 0.15 && sim > 0 ? Math.min(8, Math.max(0.02, sim / wall)) : 1;
  }
  function delayMs() {
    const r = recent(entries.at(-1).at);
    let gap = 0;
    for (let k = 1; k < r.length; k++) gap = Math.max(gap, r[k].at - r[k - 1].at);
    return Math.min(maxDelayMs, Math.max(minDelayMs, gap * margin));
  }

  return {
    push(pose, arrivedMs) {
      const t = timeOf(pose, arrivedMs), last = entries.at(-1);
      // Another world (a different body count), or the clock going back (a new episode): start again.
      if (last && (last.pose.pos.length !== pose.pos.length || t < last.t - 1e-9)) { entries = []; play = null; }
      const entry = { pose, t, at: arrivedMs, scene: lastScene };
      if (entries.length && Math.abs(entries.at(-1).t - t) < 1e-9) entries[entries.length - 1] = entry;
      else entries.push(entry);
      // Keep what playback may still need: one pose before the drawn time, and at most maxEntries.
      while (entries.length > maxEntries || (entries.length > 2 && play !== null && entries[1].t <= play)) entries.shift();
    },
    pushScene(scene) {
      lastScene = scene;
      if (!entries.length) return;
      const named = typeof scene?.t_s === "number" ? entries.findIndex((e) => Math.abs(e.t - scene.t_s) < 1e-6) : -1;
      for (let k = named >= 0 ? named : entries.length - 1; k < entries.length; k++) entries[k].scene = scene;
    },
    sample(nowMs) {
      if (!entries.length) return null;
      const newest = entries.at(-1);
      if (entries.length === 1) { play = newest.t; lastNow = nowMs; return { ...newest.pose, scene: newest.scene }; }
      const r = rate(), lag0 = (delayMs() / 1000) * r;
      const dt = lastNow === null ? 0 : Math.min(0.25, Math.max(0, (nowMs - lastNow) / 1000));
      lastNow = nowMs;
      if (play === null || play > newest.t || newest.t - play > Math.max(4 * lag0, 2 * r)) {
        play = Math.max(entries[0].t, newest.t - lag0); // the start, or too far behind (a hidden tab): rejoin
      } else {
        const lag = newest.t - play, err = (lag - lag0) / Math.max(lag0, 1e-3);
        play = Math.min(newest.t, play + dt * r * Math.min(2.5, Math.max(0.25, 1 + 0.8 * err)));
      }
      let i = entries.length - 2;
      while (i > 0 && entries[i].t > play) i--;
      const a = entries[i], b = entries[i + 1];
      const f = b.t > a.t ? Math.min(1, Math.max(0, (play - a.t) / (b.t - a.t))) : 1;
      return { ...blend(a.pose, b.pose, f), scene: blendScene(a.scene, b.scene, f) };
    },
    // The newest pose, without moving the playback clock.
    peek: () => (entries.length ? entries.at(-1).pose : null),
    // True while there is still motion to draw (playback behind the newest pose).
    moving() { return entries.length >= 2 && (play === null || play < entries.at(-1).t - 1e-9); },
    clear() { entries = []; play = null; lastNow = null; },
  };
}

function lerp3(p, q, f) { return p.map((x, k) => x + (q[k] - x) * f); }
function nlerpQuat(u, v, f) {
  const sign = u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3] < 0 ? -1 : 1; // the short way round
  const r = u.map((x, k) => x + (sign * v[k] - x) * f);
  const l = Math.hypot(...r) || 1;
  return r.map((x) => x / l);
}

function blend(p, q, f) {
  return { t_s: q.t_s, pos: q.pos.map((v, i) => lerp3(p.pos[i], v, f)), quat: q.quat.map((v, i) => nlerpQuat(p.quat[i], v, f)) };
}

// Objects present in both scenes move between them; the object list and the markers are the nearer scene's, so an
// object the newer scene drops disappears with it.
export function blendScene(a, b, f) {
  if (!a || !b) return f < 0.5 ? (a ?? b ?? null) : (b ?? a ?? null);
  const near = f < 0.5 ? a : b;
  const other = new Map((f < 0.5 ? b : a).objects?.map((o) => [o.id, o]) ?? []);
  return {
    objects: (near.objects ?? []).map((o) => {
      const p = other.get(o.id);
      if (!p || !Array.isArray(p.pos) || !Array.isArray(o.pos)) return o;
      const [from, to] = f < 0.5 ? [o, p] : [p, o];
      return { ...o, pos: lerp3(from.pos, to.pos, f), quat: Array.isArray(from.quat) && Array.isArray(to.quat) ? nlerpQuat(from.quat, to.quat, f) : o.quat };
    }),
    markers: near.markers ?? [],
  };
}
