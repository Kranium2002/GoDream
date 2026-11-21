// The live camera's pure parts: rolling window, in-flight policy, sampling clock and rate meter. Memory only.

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
// period would take every frame of a camera at twice the rate); a long gap restarts the clock.
export function createSampler(hz) {
  const period = 1000 / hz;
  let next = null;
  return {
    take(now) {
      if (next !== null && now < next - period / 4) return false;
      next = next === null || now - next > period ? now + period : next + period;
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
  if (!hasApi) return "This browser does not offer camera access (getUserMedia).";
  return null;
}
