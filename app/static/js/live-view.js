// "Your fist": the laptop camera on an explicit click. Each new camera frame (at most at the rate the model expects,
// never the same frame twice) becomes the dog's grey cells; the newest full window goes to the server at once, and
// while one request is in flight only the newest window waits. The big event view is drawn from frames already in
// memory; the answer's command moves the simulated dog on the right (the server applies it). Frames live only in
// memory and are released on Stop or when the person leaves this mode; nothing is stored.
import { cameraProblem, changePicture, createFrameWindow, createPace, createRateMeter, createSampler, createSender, toEpochMs } from "./live.js";
import { preprocess, readWindowSettings } from "./vision.js";
import { plainHonesty, plainState, plainWhy, readout } from "./readout.js";
import { cropRect } from "./drive.js";
import { paint } from "./sense-view.js";
import { $, setText } from "./dom.js";

const RETRY_MS = 3000; // after an unavailable or failed answer, wait this long before offering the next window
const SLOW_TOLERANCE = 0.75; // the camera counts as slow below this share of the rate the model expects

export function initLiveView({ api, notes, setLine }) {
  const video = $("cam-video");
  const capture = document.createElement("canvas"); // never in the page, never exported
  const captureCtx = capture.getContext("2d", { willReadFrequently: true });
  let settings = null, inferCap = null, active = false, problem = null;
  let stream = null, session = 0, timer = null, run = null, starting = false;
  const cam = createRateMeter(); // the camera's own frame rate, for the slow-camera hint
  let pace = null;

  // ---- server state -------------------------------------------------------------------------------------------
  function setState(state, capabilities) {
    const r = readWindowSettings(state?.image_window);
    settings = r.ok ? r.settings : null;
    if (r.ok) notes.clear("window");
    else notes.set("window", "warn", `image_window: ${r.problems.join(" ")}`, "Camera");
    inferCap = capabilities.find((c) => c.name === "infer") ?? null;
    if (inferCap && !inferCap.available) notes.set("infer", "warn", `infer unavailable: ${inferCap.reason}`, "Camera");
    else notes.clear("infer");
    setText($("eye-key"), settings?.windowEncoding === "events"
      ? "Like an event camera, it only notices change: white got brighter, black got darker, grey stayed the same."
      : "It notices what changes between frames: lighter got brighter, darker got darker.");
    render();
  }

  // Whether the camera can be used here: null, or one plain sentence saying why not.
  function blocker() {
    if (!settings) return "The server has not said what the dog's eye expects, so the camera cannot be used yet.";
    if (!inferCap?.available) return "The dog's reflex is not loaded on this server, so it cannot react to your camera yet.";
    return null;
  }

  function render() {
    if (!active) return;
    const b = $("cam-toggle");
    b.hidden = false;
    b.disabled = starting || (!stream && Boolean(blocker()));
    setText(b, starting ? "Asking for the camera" : stream ? "Stop camera" : "Start camera");
    b.dataset.on = String(Boolean(stream));
    if (!stream) setLine(problem ?? blocker() ?? "Push your fist toward the camera. The dog sees only the outlined square, 12 by 12 dots.");
  }

  // ---- camera ----------------------------------------------------------------------------------------------------
  async function start() {
    if (stream || starting || blocker()) return;
    const why = cameraProblem({ secure: window.isSecureContext, hasApi: Boolean(navigator.mediaDevices?.getUserMedia) });
    if (why) return showProblem(why);
    starting = true;
    problem = null;
    render();
    let s;
    try {
      // The rate the model expects, asked of the camera (it may give less in dim light).
      s = await navigator.mediaDevices.getUserMedia({ video: { frameRate: { ideal: settings.hz }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    } catch (error) {
      starting = false;
      return showProblem(cameraProblem({ error }));
    }
    starting = false;
    if (!active) { s.getTracks().forEach((t) => t.stop()); return render(); }
    stream = s;
    session++;
    video.srcObject = stream;
    try { await video.play(); } catch { /* a muted inline video plays; a refusal shows as no frames */ }
    const n = settings.height * settings.width;
    run = { settings, frameWindow: createFrameWindow(settings.frames, n), sampler: createSampler(settings.hz), retryAt: 0, sessionStart: true, sender: null };
    const r = run;
    r.sender = createSender(async (snap) => {
      const first = r.sessionStart; // until the server has answered one window of this session
      const result = await api.infer(snap.window, r.settings.frames, toEpochMs(snap.capturedAt, performance.now(), Date.now()), first);
      if (first && result.ok) r.sessionStart = false;
      return { result };
    }, (snap, answer) => showDecision(r, answer));
    pace = createPace(settings.hz, { tolerance: SLOW_TOLERANCE });
    document.body.dataset.cam = "on";
    $("react").hidden = false;
    setText($("state-line"), "Watching");
    $("state-line").dataset.tone = "calm";
    setText($("why-line"), "It needs a moment of camera frames before its first look.");
    setDanger(null, null);
    $("honesty").hidden = true;
    setLine("Push your fist toward the camera, quickly. Keep the rest of the picture still.");
    render();
    schedule(session);
  }

  function stop() {
    session++;
    clearTimeout(timer);
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
    if (run) { run.frameWindow.clear(); run.sender.reset(); }
    run = null;
    capture.width = capture.height = 0; // drops the last frame's pixels
    const big = $("eye-events"); big.width = big.height = 1;
    cam.clear();
    pace?.reset();
    delete document.body.dataset.cam;
    $("react").hidden = true;
    $("slow-hint").hidden = true;
    $("cam-crop").hidden = true;
    render();
  }

  function showProblem(text) {
    problem = text;
    notes.report("danger", text, "Camera");
    render();
  }

  // ---- frames ----------------------------------------------------------------------------------------------------
  function schedule(mine) {
    if (mine !== session) return;
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback((now, meta) => onFrame(mine, now, meta));
    else timer = setTimeout(() => onFrame(mine, performance.now(), null), 1000 / (run?.settings.hz ?? 30));
  }

  function onFrame(mine, now, meta) {
    if (mine !== session) return;
    const frame = meta ? meta.presentedFrames : video.currentTime; // which camera frame this is
    // When the frame was captured: the camera's own time when the browser gives it, else when it was handed over for
    // display, else this callback's time (the latest, so the age measured is never flattered).
    const captured = typeof meta?.captureTime === "number" ? meta.captureTime : typeof meta?.presentationTime === "number" ? meta.presentationTime : now;
    cam.add(now);
    const p = pace.update(now, cam.rate(now));
    $("slow-hint").hidden = !p.warn;
    if (run && run.sampler.take(now, frame)) take(run, captured);
    schedule(mine);
  }

  function take(r, t) {
    const W = video.videoWidth, H = video.videoHeight;
    if (!W || !H) return;
    // Only the centre square is read back: the same pixels preprocess() would crop, so the cells are identical.
    const S = Math.min(W, H), x0 = Math.floor((W - S) / 2), y0 = Math.floor((H - S) / 2);
    if (capture.width !== S || capture.height !== S) { capture.width = S; capture.height = S; }
    captureCtx.drawImage(video, x0, y0, S, S, 0, 0, S, S); // same size: no scaling, no mirroring
    let cells;
    try {
      cells = preprocess(captureCtx.getImageData(0, 0, S, S).data, S, S, r.settings);
    } catch (e) {
      notes.report("danger", e.message, "Camera");
      return;
    }
    r.frameWindow.push(cells, t);
    const snap = r.frameWindow.snapshot();
    if (snap && performance.now() >= r.retryAt) r.sender.offer(snap); // the newest window, at once
    const s = r.settings, frames = r.frameWindow.list();
    if (frames.length >= 2) {
      try { paint($("eye-events"), changePicture(frames.at(-2), frames.at(-1), s.windowEncoding, s.diffFloor), s.width); }
      catch (e) { notes.set("events", "warn", `Event view: ${e.message}`, "Camera"); }
    }
    placeCrop(W, H);
  }

  let cropKey = "";
  function placeCrop(W, H) {
    const c = cropRect(W, H, video.clientWidth, video.clientHeight), box = $("cam-crop");
    box.hidden = !c;
    const key = c ? `${c.left},${c.top},${c.size}` : "";
    if (c && key !== cropKey) Object.assign(box.style, { left: `${c.left}px`, top: `${c.top}px`, width: `${c.size}px`, height: `${c.size}px` });
    cropKey = key;
  }

  // ---- answers ---------------------------------------------------------------------------------------------------
  function setDanger(value, threshold) {
    const fill = $("danger-fill");
    const v = typeof value === "number" ? Math.min(1, Math.max(0, value)) : 0;
    fill.style.transform = `scaleX(${v.toFixed(3)})`;
    fill.dataset.over = String(typeof threshold === "number" && v >= threshold);
    const mark = $("danger-mark");
    mark.hidden = typeof threshold !== "number";
    if (typeof threshold === "number") {
      const x = Math.min(1, Math.max(0, threshold));
      mark.style.left = `${(x * 100).toFixed(1)}%`;
      mark.dataset.side = x > 0.62 ? "end" : x < 0.38 ? "start" : "mid";
    }
    $("danger").setAttribute("aria-valuenow", typeof value === "number" ? value.toFixed(2) : "0");
  }

  function showDecision(r, { result }) {
    if (r !== run) return;
    const view = readout(result);
    const st = plainState(view);
    setText($("state-line"), st.text);
    $("state-line").dataset.tone = st.tone;
    setText($("why-line"), plainWhy(view));
    if (view.state !== "decision") {
      r.retryAt = performance.now() + RETRY_MS; // unavailable or failed: pause instead of repeating the request at camera rate
      if (view.state === "unavailable") notes.set("infer-answer", "warn", `infer: ${view.message}`, "Camera");
      else notes.report("danger", `infer failed: ${view.message} (${view.code})`, "Camera");
      return;
    }
    notes.clear("infer-answer");
    setDanger(view.values.threat, view.thresholds?.threat ?? null);
    const honest = plainHonesty(view);
    setText($("honesty"), honest.join(" "));
    $("honesty").hidden = !honest.length;
  }

  addEventListener("pagehide", () => { if (stream) stop(); });

  return {
    setState,
    toggle: () => (stream ? stop() : start()),
    enter() { active = true; problem = null; render(); },
    leave() { active = false; if (stream) stop(); $("cam-toggle").hidden = true; $("slow-hint").hidden = true; },
    on: () => Boolean(stream),
  };
}
