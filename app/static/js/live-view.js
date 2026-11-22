// Screen 4, Try it live: the camera on an explicit click; each new camera frame (at most at the rate the model expects,
// never the same frame twice) becomes grey cells; the newest full window goes to /api/infer at once, and while one is in
// flight only the newest waits. The dog's eye (12 x 12 and the event view) is drawn from frames already in memory; the
// answer moves the simulated dog on the stage (the server applies its command). Frames live only in memory and are
// released on Stop or on leaving the screen; nothing is stored.
import { cameraProblem, changePicture, createFrameWindow, createPace, createRateMeter, createRecent, createSampler, createSender, toEpochMs } from "./live.js";
import { preprocess, readWindowSettings } from "./vision.js";
import { plainHonesty, plainState, plainWhy, readout } from "./readout.js";
import { cropRect } from "./drive.js";
import { $, setText } from "./dom.js";

const RETRY_MS = 3000; // after an unavailable or failed answer, wait this long before offering the next window

export function initLiveView(ctx) {
  const { api, notes } = ctx;
  const video = $("cam-video");
  const capture = document.createElement("canvas"); // never in the page, never exported
  const captureCtx = capture.getContext("2d", { willReadFrequently: true });
  let settings = null, inferCap = null, active = false;
  let stream = null, session = 0, timer = null, run = null, lastView = null, lastAnswer = null, starting = false;
  const cam = createRateMeter(), sent = createRateMeter();
  const ageAtSend = createRecent(5000), rtt = createRecent(5000), serverAge = createRecent(5000);
  let pace = null;

  // ---- server state -------------------------------------------------------------------------------------------
  function setState(state, capabilities) {
    const r = readWindowSettings(state?.image_window);
    settings = r.ok ? r.settings : null;
    if (r.ok) notes.clear("window");
    else notes.set("window", "warn", `image_window: ${r.problems.join(" ")}`, "Live");
    inferCap = capabilities.find((c) => c.name === "infer") ?? null;
    if (inferCap && !inferCap.available) notes.set("infer", "warn", `infer unavailable: ${inferCap.reason}`, "Live");
    else notes.clear("infer");
    renderIdle();
  }

  function renderIdle() {
    if (stream) return;
    const can = Boolean(inferCap?.available) && Boolean(settings);
    $("live-cannot").hidden = can;
    setText($("live-cannot"), !settings ? "The server has not said what the dog's eye expects, so the camera cannot be used yet."
      : "Its model is not loaded on this server, so it cannot react to your camera yet.");
    $("cam-toggle").disabled = !settings || starting;
  }

  // ---- camera ----------------------------------------------------------------------------------------------------
  async function start() {
    if (stream || starting || !settings) return;
    const blocker = cameraProblem({ secure: window.isSecureContext, hasApi: Boolean(navigator.mediaDevices?.getUserMedia) });
    if (blocker) return showProblem(blocker);
    starting = true;
    $("cam-toggle").disabled = true;
    setText($("cam-toggle"), "Asking for the camera…");
    let s;
    try {
      // The rate the model expects, asked of the camera (it may give less in dim light).
      s = await navigator.mediaDevices.getUserMedia({ video: { frameRate: { ideal: settings.hz }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    } catch (error) {
      starting = false;
      setText($("cam-toggle"), "Start camera");
      $("cam-toggle").disabled = false;
      return showProblem(cameraProblem({ error }));
    }
    starting = false;
    if (!active) { s.getTracks().forEach((t) => t.stop()); return renderIdle(); }
    $("live-problem").hidden = true;
    stream = s;
    session++;
    video.srcObject = stream;
    try { await video.play(); } catch { /* muted inline video plays; a refusal shows as no frames */ }
    const n = settings.height * settings.width;
    run = { settings, frameWindow: createFrameWindow(settings.frames, n), sampler: createSampler(settings.hz), retryAt: 0, sessionStart: true,
      looked: 0, dodged: 0, sender: null };
    const r = run;
    r.sender = createSender(async (snap) => {
      const sentAt = performance.now();
      ageAtSend.add(sentAt, sentAt - snap.capturedAt);
      sent.add(sentAt);
      const first = r.sessionStart; // until the server has answered one window of this session
      const result = await api.infer(snap.window, r.settings.frames, toEpochMs(snap.capturedAt, performance.now(), Date.now()), first);
      if (first && result.ok) r.sessionStart = false;
      const now = performance.now();
      rtt.add(now, now - sentAt);
      return { result, rttMs: now - sentAt };
    }, (snap, answer) => showDecision(r, answer));
    pace = createPace(settings.hz);
    document.body.classList.add("cam-on");
    $("eye").hidden = false;
    $("mind").hidden = false;
    setText($("state-line"), "Watching…");
    $("state-line").dataset.tone = "calm";
    setText($("why-line"), "It needs a moment of camera frames before its first look.");
    setDanger(null, null);
    setText($("n-looked"), "0");
    setText($("n-dodged"), "0");
    $("honesty").hidden = true;
    setText($("cam-toggle"), "Stop camera");
    $("cam-toggle").disabled = false;
    $("cam-toggle").dataset.on = "true";
    schedule(session);
    ctx.drawerChanged();
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
    for (const id of ["eye-grey", "eye-events"]) { const c = $(id); c.width = c.height = 1; }
    for (const m of [cam, sent]) m.clear();
    for (const m of [ageAtSend, rtt, serverAge]) m.clear();
    pace?.reset();
    document.body.classList.remove("cam-on");
    $("eye").hidden = true;
    $("mind").hidden = true;
    $("slow-hint").hidden = true;
    $("cam-crop").hidden = true;
    setText($("cam-toggle"), "Start camera");
    $("cam-toggle").dataset.on = "false";
    renderIdle();
    ctx.drawerChanged();
  }

  function showProblem(text) {
    $("live-problem").hidden = false;
    setText($("live-problem"), text);
    notes.report("danger", text, "Camera");
    renderIdle();
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
    const captured = typeof meta?.captureTime === "number" ? meta.captureTime : now;
    cam.add(now);
    const p = pace.update(now, cam.rate(now));
    $("slow-hint").hidden = !p.warn;
    if (run && run.sampler.take(now, frame)) take(run, captured);
    schedule(mine);
  }

  function take(r, t) {
    const W = video.videoWidth, H = video.videoHeight;
    if (!W || !H) return;
    if (capture.width !== W || capture.height !== H) { capture.width = W; capture.height = H; }
    captureCtx.drawImage(video, 0, 0); // same size as the frame: no scaling, no mirroring
    let cells;
    try {
      cells = preprocess(captureCtx.getImageData(0, 0, W, H).data, W, H, r.settings);
    } catch (e) {
      notes.report("danger", e.message, "Camera");
      return;
    }
    r.frameWindow.push(cells, t);
    const s = r.settings;
    drawCells($("eye-grey"), cells, s.width, s.height);
    const frames = r.frameWindow.list();
    if (frames.length >= 2) {
      try { drawCells($("eye-events"), changePicture(frames.at(-2), frames.at(-1), s.windowEncoding, s.diffFloor), s.width, s.height); }
      catch (e) { notes.set("events", "warn", `Event view: ${e.message}`, "Live"); }
    }
    placeCrop(W, H);
    const snap = r.frameWindow.snapshot();
    if (snap && performance.now() >= r.retryAt) r.sender.offer(snap);
  }

  // Values in 0..1 into a canvas of w x h pixels; CSS scales it up with its pixels kept.
  function drawCells(canvas, cells, w, h) {
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const img = new ImageData(w, h);
    for (let k = 0; k < cells.length; k++) {
      const g = Math.max(0, Math.min(255, Math.round(cells[k] * 255)));
      img.data[4 * k] = img.data[4 * k + 1] = img.data[4 * k + 2] = g;
      img.data[4 * k + 3] = 255;
    }
    canvas.getContext("2d").putImageData(img, 0, 0);
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

  function showDecision(r, { result, rttMs }) {
    if (r !== run) return;
    const view = readout(result);
    lastView = view;
    lastAnswer = { at: performance.now(), rttMs, data: result.ok ? result.data : null };
    if (result.ok && typeof result.data.frame_age_ms === "number") serverAge.add(performance.now(), result.data.frame_age_ms);
    const st = plainState(view);
    setText($("state-line"), st.text);
    $("state-line").dataset.tone = st.tone;
    setText($("why-line"), plainWhy(view));
    if (view.state !== "decision") {
      r.retryAt = performance.now() + RETRY_MS; // unavailable or failed: pause instead of repeating the request at camera rate
      if (view.state === "unavailable") notes.set("infer-answer", "warn", `infer: ${view.message}`, "Live");
      else notes.report("danger", `infer failed: ${view.message} (${view.code})`, "Live");
      ctx.drawerChanged();
      return;
    }
    notes.clear("infer-answer");
    r.looked++;
    if (view.issued === true) r.dodged++;
    setText($("n-looked"), String(r.looked));
    setText($("n-dodged"), String(r.dodged));
    setDanger(view.values.threat, view.thresholds?.threat ?? null);
    const honest = plainHonesty(view);
    setText($("honesty"), honest.join(" "));
    $("honesty").hidden = !honest.length;
    ctx.drawerChanged();
  }

  $("cam-toggle").addEventListener("click", () => (stream ? stop() : start()));
  addEventListener("pagehide", () => { if (stream) stop(); });

  return {
    setState,
    enter() { active = true; renderIdle(); },
    leave() { active = false; if (stream) stop(); },
    stop: () => { if (stream) stop(); },
    // For the engineers' drawer: the pipeline as measured here, and the last answer exactly as sent.
    stats() {
      const now = performance.now();
      return { on: Boolean(stream), settings, inferCap, cameraFps: cam.rate(now), sendRate: sent.rate(now), ageAtSend: ageAtSend.summary(now),
        rtt: rtt.summary(now), serverAge: serverAge.summary(now), dropped: run?.sender.dropped() ?? 0, looked: run?.looked ?? 0, dodged: run?.dodged ?? 0,
        view: lastView, answer: lastAnswer, video: stream ? { width: video.videoWidth, height: video.videoHeight } : null };
    },
  };
}
