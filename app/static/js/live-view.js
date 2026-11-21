// The Live area: camera on an explicit click, preprocessing, the rolling window, inference requests and readouts.
// Frames live only in memory and are released on Stop; nothing is stored.
import { cameraProblem, createFrameWindow, createRateMeter, createSampler, createSender } from "./live.js";
import { preprocess, readWindowSettings } from "./vision.js";
import { readout } from "./readout.js";
import { createStage } from "./stage.js";
import { cropRect } from "./drive.js";

const RETRY_MS = 3000; // after an unavailable or failed answer, wait this long before offering the next window

export function initLiveView({ api, el }) {
  const $ = (id) => document.getElementById(id);
  const video = $("cam-video");
  const grey = $("cam-grey");
  const capture = document.createElement("canvas"); // never in the page, never exported
  const captureCtx = capture.getContext("2d", { willReadFrequently: true });

  let settings = null;          // the latest readable image_window
  let headCapability = null;
  let visible = false;
  let stream = null, session = 0, timer = null;
  let run = null;               // {settings, frameWindow, sampler, sender, rate, decisions, issued, retryAt, lastAction}

  const status = (text) => { $("cam-status").textContent = text; };

  // ---- server state -------------------------------------------------------------------------------------------
  function setState(state, capabilities) {
    const r = readWindowSettings(state?.image_window);
    settings = r.ok ? r.settings : null;
    $("window-note").hidden = r.ok;
    $("window-note").textContent = r.ok ? "" : `The window settings are unavailable: ${r.problems.join(" ")} The preview works; nothing is sent to the model.`;
    $("window-text").hidden = !r.ok;
    if (r.ok) $("window-text").textContent = `Window: ${settings.frames} frames of ${settings.height} x ${settings.width} grey cells at ${settings.hz} Hz, oldest first.`;
    headCapability = capabilities.find((c) => c.name === "head_camera") ?? null;
    stage.setAspect(state?.frame?.width, state?.frame?.height);
    updateStage();
  }

  function setVisible(v) {
    visible = v;
    updateStage();
  }

  // ---- simulator stage: streams load only while this view is shown --------------------------------------------
  const stage = createStage({ root: $("stage"), trackingImg: $("stage-tracking"), trackingNote: $("trk-note"), headImg: $("stage-head"), headNote: $("head-note") });
  const updateStage = () => stage.update(visible, headCapability);

  // ---- camera ----------------------------------------------------------------------------------------------------
  async function start() {
    const blocker = cameraProblem({ secure: window.isSecureContext, hasApi: Boolean(navigator.mediaDevices?.getUserMedia) });
    if (blocker) return status(blocker);
    $("cam-start").disabled = true;
    status("Asking the browser for the camera.");
    let s;
    try {
      s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (error) {
      $("cam-start").disabled = false;
      return status(cameraProblem({ error }));
    }
    if (!visible) { s.getTracks().forEach((t) => t.stop()); $("cam-start").disabled = false; return; }
    stream = s;
    session++;
    video.srcObject = stream;
    try { await video.play(); } catch { /* muted inline video plays; a refusal shows as no frames */ }
    run = settings ? newRun(settings) : null;
    $("cam-stop").disabled = false;
    $("cam-stop").focus();
    status(settings ? "Camera on. Frames are taken, turned into grey cells and sent to the model." : "Camera on, preview only: the window settings are unavailable.");
    resetReadouts();
    schedule(session, run?.settings.hz ?? 30); // 30: only the timer fallback's pace in a preview-only session
  }

  function newRun(s) {
    const r = { settings: s, frameWindow: createFrameWindow(s.frames, s.height * s.width), sampler: createSampler(s.hz),
      decisions: 0, issued: 0, retryAt: 0, lastAction: null, sessionStart: true };
    buildStrip(s);
    r.sender = createSender(async (snap) => {
      const sentAt = performance.now();
      const first = r.sessionStart; // until the server has answered one window of this session
      const result = await api.infer(snap.window, s.frames, performance.timeOrigin + snap.capturedAt, first);
      if (first && result.ok) r.sessionStart = false;
      const now = performance.now();
      return { result, rttMs: now - sentAt, ageMs: now - snap.capturedAt };
    }, (snap, answer) => showDecision(r, answer));
    return r;
  }

  function stop(message) {
    session++;
    clearTimeout(timer);
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
    if (run) { run.frameWindow.clear(); run.sender.reset(); }
    run = null;
    capture.width = capture.height = 0; // drops the last frame's pixels
    meter.clear();
    clearDecision("Camera off: no decision.");
    grey.width = grey.height = 1;
    $("cam-greyfull").width = $("cam-greyfull").height = 1;
    $("cam-strip").replaceChildren();
    $("cam-crop").hidden = true;
    $("grey-text").textContent = "Nothing yet.";
    $("st-fps").textContent = "not measured";
    $("st-fill").textContent = "not started";
    $("cam-start").disabled = false;
    $("cam-stop").disabled = true;
    status(message);
  }

  // ---- frames ----------------------------------------------------------------------------------------------------
  const meter = createRateMeter();
  function schedule(mine, hz) {
    if (mine !== session) return;
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback((now, meta) => onFrame(mine, hz, now, meta));
    else timer = setTimeout(() => onFrame(mine, hz, performance.now(), null), 1000 / hz);
  }

  function onFrame(mine, hz, now, meta) {
    if (mine !== session) return;
    const t = typeof meta?.captureTime === "number" ? meta.captureTime : now;
    meter.add(now);
    const fps = meter.rate(now);
    $("st-fps").textContent = fps === null ? "not measured" : fps.toFixed(1);
    if (run && run.sampler.take(t)) take(run, t);
    schedule(mine, hz);
  }

  function take(r, t) {
    const W = video.videoWidth, H = video.videoHeight;
    if (!W || !H) return;
    if (capture.width !== W || capture.height !== H) { capture.width = W; capture.height = H; }
    captureCtx.drawImage(video, 0, 0); // same size as the frame: no scaling, no mirroring
    let cells, square;
    try {
      const px = captureCtx.getImageData(0, 0, W, H).data;
      cells = preprocess(px, W, H, r.settings);
      // For people only: the same centre square in grey at up to 96 x 96, by the same area averaging (levels 0-255).
      const g = Math.min(96, W, H);
      square = preprocess(px, W, H, { ...r.settings, height: g, width: g, divisor: 1 });
      drawCells($("cam-greyfull"), square, g, g, 1);
    } catch (e) {
      status(e.message);
      return;
    }
    drawCells(grey, cells, r.settings.width, r.settings.height, 255);
    $("grey-text").textContent = `${r.settings.height} x ${r.settings.width} grey cells of the newest frame, as sent to the model.`;
    placeCrop(W, H);
    r.frameWindow.push(cells, t);
    r.frameWindow.list().forEach((v, k) => { const c = $("cam-strip").children[k]?.querySelector("canvas"); if (c) drawCells(c, v, r.settings.width, r.settings.height, 255); });
    const snap = r.frameWindow.snapshot();
    $("st-fill").textContent = snap ? `${r.settings.frames} of ${r.settings.frames} frames` : "filling";
    if (snap && performance.now() >= r.retryAt) r.sender.offer(snap);
    $("st-dropped").textContent = String(r.sender.dropped());
  }

  // Grey values (times toByte) into a canvas of w x h pixels; CSS scales it up with its pixels kept.
  function drawCells(canvas, cells, w, h, toByte) {
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const img = new ImageData(w, h);
    cells.forEach((v, k) => { const g = Math.max(0, Math.min(255, Math.round(v * toByte))); img.data.set([g, g, g, 255], 4 * k); });
    canvas.getContext("2d").putImageData(img, 0, 0);
  }

  // The centre square the model uses, outlined on the preview.
  function placeCrop(W, H) {
    const c = cropRect(W, H, video.clientWidth, video.clientHeight), box = $("cam-crop");
    box.hidden = !c;
    if (c) Object.assign(box.style, { left: `${c.left}px`, top: `${c.top}px`, width: `${c.size}px`, height: `${c.size}px` });
  }

  // One small canvas per frame of the window, oldest first.
  function buildStrip(s) {
    $("cam-strip").replaceChildren(...Array.from({ length: s.frames }, (_, k) => el("li", {},
      el("canvas", { width: s.width, height: s.height, role: "img", "aria-label": `Frame ${k + 1} of ${s.frames}` }),
      el("span", { class: "hint" }, k === 0 ? "oldest" : k === s.frames - 1 ? "newest" : String(k + 1)))));
  }

  // ---- decisions -------------------------------------------------------------------------------------------------
  function clearDecision(text) {
    $("dec-action").textContent = text;
    $("dec-issued").textContent = "";
    $("dec-rows").replaceChildren();
    $("dec-gate").replaceChildren(el("li", { class: "empty" }, "No gate state yet."));
  }
  function resetReadouts() {
    clearDecision("No decision yet.");
    for (const id of ["st-dropped", "st-decisions", "st-issued"]) $(id).textContent = "0";
  }

  function showDecision(r, { result, rttMs, ageMs }) {
    if (r !== run) return;
    const view = readout(result);
    if (view.state !== "decision") {
      r.retryAt = performance.now() + RETRY_MS; // unavailable or failed: pause instead of repeating the request 30 times a second
      setAction(r, `${view.state}:${view.message}`, view.state === "unavailable"
        ? `Inference is unavailable: ${view.message}`
        : `Inference failed: ${view.message} (${view.code})`);
      $("dec-issued").textContent = "";
      return;
    }
    r.decisions++;
    if (view.issued === true) r.issued++;
    $("dev-flag").hidden = !view.development;
    $("st-decisions").textContent = String(r.decisions);
    $("st-issued").textContent = String(r.issued);
    setAction(r, view.action.code, `${view.action.label} (${view.action.code})`);
    $("dec-issued").textContent = view.issued === null ? "Command issued: not sent" : view.issued ? "Command issued to the simulated dog now." : "No command issued.";
    const rows = [...view.rows, ["Round trip measured here", `${rttMs.toFixed(1)} ms`], ["Frame age when the answer arrived", `${ageMs.toFixed(1)} ms`]];
    $("dec-rows").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));
    $("dec-gate").replaceChildren(...(view.gate.length ? view.gate.map((g) => el("li", { class: g.ok ? "is-good" : "is-bad" },
      el("span", { class: "mark", "aria-hidden": "true" }, g.ok ? "✓" : "✕"), `${g.ok ? "Pass" : "Fail"}: ${g.label}`))
      : [el("li", { class: "empty" }, "The server sent no gate state.")]));
  }

  // The action line is a live region: it changes only when the action does, so it is announced once per change.
  function setAction(r, key, text) {
    if (r.lastAction === key) return;
    r.lastAction = key;
    $("dec-action").textContent = text;
  }

  $("cam-start").addEventListener("click", start);
  $("cam-stop").addEventListener("click", () => stop("The camera is off and its frames are cleared."));
  addEventListener("pagehide", () => { if (stream) stop("The camera is off."); });

  return { setState, setVisible };
}
