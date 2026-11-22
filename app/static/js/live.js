// The live camera's pure parts: rolling window, in-flight policy, sampling, the camera's pace, rate meters, recent
// statistics and the event picture. Memory only.

// The newest `frames` frames of `size` values each, oldest first.
export function createFrameWindow(frames, size) {
  let buf = [];
  return {
    push(values, capturedAt) {
      if (values.length !== size) throw new Error(`A frame must hold ${size} values, not ${values.length}.`);
      buf.push({ values, capturedAt });
      if (buf.length > frames) buf.shift();
    },
    full: () => buf.length === frames,
    // {window: Float32Array of frames * size, capturedAt: the newest frame's capture time}, or null until full.
    snapshot() {
      if (buf.length < frames) return null;
      const window = new Float32Array(frames * size);
      buf.forEach((f, k) => window.set(f.values, k * size));
      return { window, capturedAt: buf[buf.length - 1].capturedAt };
    },
    list: () => buf.map((f) => f.values), // the frames, oldest first
    clear() { buf = []; },
  };
}

// One request in flight at most; while it runs only the newest offered item waits, older ones are dropped and counted.
// onAnswer(item, result) runs for every answer that arrives before the next reset().
export function createSender(send, onAnswer) {
  let inFlight = false, waiting = null, dropped = 0, generation = 0;
  async function go(item) {
    inFlight = true;
    const mine = generation;
    const result = await send(item);
    if (mine !== generation) return; // reset while in flight: the answer belongs to a stopped session
    inFlight = false;
    onAnswer(item, result);
    if (waiting !== null && !inFlight) {
      const next = waiting;
      waiting = null;
      go(next);
    }
  }
  return {
    offer(item) {
      if (!inFlight) go(item);
      else {
        if (waiting !== null) dropped++;
        waiting = item;
      }
    },
    dropped: () => dropped,
    busy: () => inFlight,
    reset() { generation++; inFlight = false; waiting = null; dropped = 0; },
  };
}

// Decides which camera frames to take at `hz`: a frame up to a quarter period early counts (camera jitter; half a
// period would take every frame of a camera at twice the rate); a long gap restarts the clock. `frame` names the camera
// frame (its presented-frame count or media time): the same frame is never taken twice, whatever the times say.
export function createSampler(hz) {
  const period = 1000 / hz;
  let next = null, last;
  return {
    take(now, frame) {
      if (frame !== undefined && frame === last) return false;
      if (next !== null && now < next - period / 4) return false;
      next = next === null || now - next > period ? now + period : next + period;
      last = frame;
      return true;
    },
  };
}

// Frames per second over the last second, from the frames' own times; null with fewer than two.
export function createRateMeter() {
  let times = [];
  return {
    add(t) { times.push(t); },
    rate(now) {
      times = times.filter((t) => now - t <= 1000);
      return times.length < 2 ? null : ((times.length - 1) * 1000) / (times[times.length - 1] - times[0]);
    },
    clear() { times = []; },
  };
}

// Values of the last `spanMs`: add(t, value); summary(now) -> {n, p50, p95} (nearest-rank), or null without values.
export function createRecent(spanMs = 5000) {
  let items = [];
  return {
    add(t, value) { if (typeof value === "number" && Number.isFinite(value)) items.push([t, value]); },
    summary(now) {
      items = items.filter(([t]) => now - t <= spanMs);
      if (!items.length) return null;
      const v = items.map(([, x]) => x).sort((a, b) => a - b);
      const rank = (p) => v[Math.min(v.length - 1, Math.max(0, Math.ceil(p * v.length) - 1))];
      return { n: v.length, p50: rank(0.5), p95: rank(0.95) };
    },
    clear() { items = []; },
  };
}

// A time on the page's monotonic clock (performance.now) as epoch milliseconds, anchored at a pair of readings of both
// clocks taken together now; immune to the drift between performance.timeOrigin and the wall clock.
export const toEpochMs = (t, nowMono, nowEpoch) => nowEpoch - (nowMono - t);

// Whether the camera keeps up with the rate the model expects. update(now, rate) with the camera's measured frames per
// second (null when unknown) -> {slow, warn}: slow when below `tolerance` of hz now; warn once it has been slow for more
// than afterMs, and it stays until the camera has kept up for more than afterMs (no flicker).
export function createPace(hz, { tolerance = 0.9, afterMs = 1000 } = {}) {
  let slowSince = null, okSince = null, warn = false;
  return {
    update(now, rate) {
      const slow = !(typeof rate === "number" && rate >= hz * tolerance);
      if (slow) { okSince = null; slowSince ??= now; if (now - slowSince > afterMs) warn = true; }
      else { slowSince = null; okSince ??= now; if (now - okSince > afterMs) warn = false; }
      return { slow, warn };
    },
    reset() { slowSince = okSince = null; warn = false; },
  };
}

// What the newest pair of frames looks like to the model, for people: one value per cell in 0..1. "events" is
// learned_judgment.observed.encode_window_features' event rule for one pair (float32: a change above the threshold is an
// ON event, 1, white; below minus the threshold an OFF event, 0, black; else 0.5, mid grey). Any other encoding is
// shown as the signed change, 0.5 + 0.5 * change / max(largest |change|, floor), floor the server's diff_floor if sent.
export function changePicture(older, newer, encoding, floor) {
  if (older.length !== newer.length) throw new Error("The two frames differ in size.");
  const out = new Float32Array(newer.length);
  const change = Array.from(newer, (v, k) => Math.fround(v - older[k]));
  if (encoding === "events") {
    if (!(typeof floor === "number" && floor > 0)) throw new Error("The event picture needs the server's event threshold.");
    const thr = Math.fround(floor);
    change.forEach((c, k) => { out[k] = c > thr ? 1 : c < -thr ? 0 : 0.5; });
    return out;
  }
  const scale = Math.max(...change.map(Math.abs), typeof floor === "number" && floor > 0 ? floor : 0);
  change.forEach((c, k) => { out[k] = scale > 0 ? 0.5 + (0.5 * c) / scale : 0.5; });
  return out;
}

// Why the camera cannot start, as one plain sentence; null when nothing stands in the way.
// {secure, hasApi} before asking, {error} after getUserMedia refused.
export function cameraProblem({ secure, hasApi, error }) {
  if (error) {
    const name = error.name ?? "Error";
    if (name === "NotAllowedError" || name === "SecurityError") return "Camera permission was denied. Allow the camera for this page in the browser's site settings, then press Start camera again.";
    if (name === "NotFoundError" || name === "OverconstrainedError") return "No camera was found on this computer.";
    if (name === "NotReadableError" || name === "AbortError") return "The camera could not be opened; another program may be using it.";
    return `The camera could not start (${name}: ${error.message ?? "no detail"}).`;
  }
  if (!secure) return "The browser allows the camera only on a secure page; open this page at 127.0.0.1 or localhost.";
  if (!hasApi) return "This browser does not offer camera access.";
  return null;
}
