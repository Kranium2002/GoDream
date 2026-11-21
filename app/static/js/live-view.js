// The Live area: camera on an explicit click, preprocessing, the rolling window, inference requests and readouts.
// Frames live only in memory and are released on Stop; nothing is stored.
import { cameraProblem, createFrameWindow, createRateMeter, createSampler, createSender } from "./live.js";
import { preprocess, readWindowSettings } from "./vision.js";
import { readout } from "./readout.js";

const RETRY_MS = 3000; // after an unavailable answer, wait this long before offering the next window

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
    updateStage();
  }

  function setVisible(v) {
    visible = v;
    if (!v && stream) stop("The camera stopped because you left the Live view.");
    updateStage();
  }

  // ---- simulator stage: streams load only while this view is shown --------------------------------------------
  function stage(img, note, url, unavailableReason) {
    const want = visible && !unavailableReason ? url : null;
    note.hidden = !unavailableReason;
    note.textContent = unavailableReason ? `Unavailable: ${unavailableReason}` : "";
    if (want && img.getAttribute("src") !== want) {
      img.onerror = () => { img.hidden = true; note.hidden = false; note.textContent = "The simulator stream could not be loaded."; };
      img.setAttribute("src", want);
    } else if (!want) {
      img.removeAttribute("src"); // closes the stream
    }
    img.hidden = !want;
  }
  function updateStage() {
    stage($("stage-tracking"), $("trk-note"), "/api/stream?camera=tracking", null);
    const head = headCapability?.available ? null : headCapability?.reason || "The server does not report the head camera.";
    stage($("stage-head"), $("head-note"), "/api/stream?camera=head", head);
  }

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
      decisions: 0, issued: 0, retryAt: 0, lastAction: null };
    r.sender = createSender(async (snap) => {
      const sentAt = performance.now();
      const result = await api.infer(snap.window, s.frames, performance.timeOrigin + snap.capturedAt);
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
    let cells;
    try {
      cells = preprocess(captureCtx.getImageData(0, 0, W, H).data, W, H, r.settings);
    } catch (e) {
      status(e.message);
      return;
    }
    drawGrey(cells, r.settings);
    r.frameWindow.push(cells, t);
    const snap = r.frameWindow.snapshot();
    $("st-fill").textContent = snap ? `${r.settings.frames} of ${r.settings.frames} frames` : "filling";
    if (snap && performance.now() >= r.retryAt) r.sender.offer(snap);
    $("st-dropped").textContent = String(r.sender.dropped());
  }

  function drawGrey(cells, s) {
    if (grey.width !== s.width || grey.height !== s.height) { grey.width = s.width; grey.height = s.height; }
    const img = new ImageData(s.width, s.height);
    cells.forEach((v, k) => { const g = Math.max(0, Math.min(255, Math.round(v * 255))); img.data.set([g, g, g, 255], 4 * k); });
    grey.getContext("2d").putImageData(img, 0, 0);
    $("grey-text").textContent = `${s.height} x ${s.width} grey cells of the newest frame, centre square, as sent to the model.`;
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
      if (view.state === "unavailable") r.retryAt = performance.now() + RETRY_MS;
      setAction(r, `${view.state}:${view.message}`, view.state === "unavailable"
        ? `Inference is unavailable: ${view.message}`
        : `Inference failed: ${view.message} (${view.code})`);
      $("dec-issued").textContent = "";
      return;
    }
    r.decisions++;
    if (view.issued === true) r.issued++;
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
