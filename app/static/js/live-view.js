// The Live area: camera on an explicit click, preprocessing, the rolling window, inference requests and readouts, and
// the simulated dog (the browser's 3D view by default, the simulator's own render on request). Frames live only in
// memory and are released on Stop; nothing is stored. The readouts are built once and updated in place.
import { cameraProblem, createFrameWindow, createRateMeter, createSampler, createSender } from "./live.js";
import { preprocess, readWindowSettings } from "./vision.js";
import { commandText, readout } from "./readout.js";
import { createStage } from "./stage.js";
import { commandFields, cropRect } from "./drive.js";

const RETRY_MS = 3000; // after an unavailable or failed answer, wait this long before offering the next window
const FPS_TEXT_MS = 500; // the camera rate line changes at most this often

// What the student computes from the window the page sends (student.json input.window_encoding, via image_window);
// the server's runtime does the encoding, the page only names it.
const ENCODINGS = {
  raw: "the grey cells of every frame as they are",
  abs_diff: "how much each cell changed between adjacent frames",
  signed_diff: "the signed change of each cell between adjacent frames (brighter or darker)",
  signed_diff_first: "the signed change of each cell against the oldest frame of the window",
  events: "event-camera ON and OFF events: cells that got brighter or darker than a threshold between adjacent frames",
};
function encodingText(s) {
  if (!s.windowEncoding) return "";
  const what = ENCODINGS[s.windowEncoding] ?? `the "${s.windowEncoding}" encoding`;
  const from = s.encodingFrom === "student" ? "the loaded student's student.json" : "config.yaml vision (no student is loaded)";
  return `The network reads ${what} of the ${s.frames}-frame window, computed on the server (${s.windowEncoding}, from ${from}).`;
}

const setText = (node, text) => { if (node && node.textContent !== text) node.textContent = text; };

export function initLiveView({ api, el, notes, robotStage }) {
  const $ = (id) => document.getElementById(id);
  const video = $("cam-video");
  const grey = $("cam-grey");
  const capture = document.createElement("canvas"); // never in the page, never exported
  const captureCtx = capture.getContext("2d", { willReadFrequently: true });

  let settings = null;          // the latest readable image_window
  let fields = null;            // /api/state commands: units for the command line
  let headCapability = null;
  let visible = false, canvasOnScreen = false;
  let stream = null, session = 0, timer = null, fpsShownAt = 0;
  let run = null;               // {settings, frameWindow, sampler, sender, decisions, issued, retryAt, lastAction}

  const status = (text) => setText($("cam-status"), text);

  // ---- server state -------------------------------------------------------------------------------------------
  function setState(state, capabilities) {
    const r = readWindowSettings(state?.image_window);
    settings = r.ok ? r.settings : null;
    fields = commandFields(state);
    if (r.ok) notes.clear("window");
    else notes.set("window", "warn", `The window settings are unavailable: ${r.problems.join(" ")} The preview works; nothing is sent to the model.`, "Unavailable");
    $("window-text").hidden = !r.ok;
    if (r.ok) setText($("window-text"), `Window: ${settings.frames} frames of ${settings.height} x ${settings.width} grey cells at ${settings.hz} Hz, oldest first.`);
    // Whether a student answers: plainly when none is configured, and labelled when it runs on development thresholds.
    const inferCap = capabilities.find((c) => c.name === "infer");
    if (!inferCap) notes.clear("infer");
    else if (!inferCap.available) notes.set("infer", "warn", `No decisions: ${inferCap.reason} The camera preview works; no model answers and the dog gets no command.`, "Unavailable");
    else if (/development thresholds/i.test(inferCap.reason)) notes.set("infer", "warn", inferCap.reason, "Development thresholds");
    else if (inferCap.reason) notes.set("infer", "info", inferCap.reason, "Server");
    else notes.clear("infer");
    setText($("encoding-text"), r.ok ? encodingText(r.settings) : "");
    if (!run && (r.ok ? $("cam-strip").children.length !== settings.frames : true)) r.ok ? buildStrip(settings) : $("cam-strip").replaceChildren(); // empty squares until the camera runs
    headCapability = capabilities.find((c) => c.name === "head_camera") ?? null;
    updateStage();
  }

  function setVisible(v) {
    visible = v;
    updateStage();
  }

  // ---- the simulated dog: the browser's 3D view, or the simulator's render on request ---------------------------
  const stage = createStage({ trackingImg: $("stage-tracking"), trackingNote: $("trk-note"), headImg: $("stage-head"), headNote: $("head-note") });
  const liveCanvas = $("live-scene");
  const say3d = (text) => setText($("live-scene-status"), text);
  const view3d = robotStage.attach(liveCanvas, (text) => { if (showing3d()) say3d(text); });
  if (!view3d.ok) { $("live-scene-note").hidden = false; setText($("live-scene-note"), `The 3D view is unavailable: ${view3d.reason} Pick "Simulator render" to see the simulator's own frames.`); }
  const showing3d = () => (document.querySelector('input[name="dog-view"]:checked')?.value ?? "3d") === "3d";
  new IntersectionObserver(([e]) => { canvasOnScreen = e.isIntersecting; updateStage(); }, { threshold: 0.01 }).observe(liveCanvas.parentElement);
  function updateStage() {
    const in3d = showing3d();
    liveCanvas.hidden = !in3d;
    $("live-scene-note").hidden = !(in3d && !view3d.ok);
    view3d.setActive(in3d && visible && canvasOnScreen);
    stage.update(visible, headCapability, !in3d);
    setText($("trk-label"), in3d ? "3D view, drawn here from the simulator's poses" : "Simulator render: frames rendered by MuJoCo on the server, a few per second");
    if (!in3d) say3d("Showing the simulator's own render, streamed from the server.");
  }
  for (const r of document.querySelectorAll('input[name="dog-view"]')) r.addEventListener("change", updateStage);

  // ---- camera ----------------------------------------------------------------------------------------------------
  async function start() {
    const blocker = cameraProblem({ secure: window.isSecureContext, hasApi: Boolean(navigator.mediaDevices?.getUserMedia) });
    if (blocker) { notes.set("camera", "danger", blocker, "Camera"); return status("The camera did not start."); }
    $("cam-start").disabled = true;
    status("Asking the browser for the camera.");
    let s;
    try {
      s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (error) {
      $("cam-start").disabled = false;
      notes.set("camera", "danger", cameraProblem({ error }), "Camera");
      return status("The camera did not start.");
    }
    notes.clear("camera");
    if (!visible) { s.getTracks().forEach((t) => t.stop()); $("cam-start").disabled = false; return; }
    stream = s;
    session++;
    video.srcObject = stream;
    try { await video.play(); } catch { /* muted inline video plays; a refusal shows as no frames */ }
    run = settings ? newRun(settings) : null;
    document.body.classList.add("cam-on");
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
    clearDecision("Camera off: no decision");
    grey.width = grey.height = 1;
    $("cam-greyfull").width = $("cam-greyfull").height = 1;
    if (settings) buildStrip(settings); else $("cam-strip").replaceChildren();
    $("cam-crop").hidden = true;
    document.body.classList.remove("cam-on");
    setText($("grey-text"), "Nothing yet.");
    setText($("st-fps"), "not measured");
    setText($("st-fill"), "not started");
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
    if (now - fpsShownAt >= FPS_TEXT_MS) {
      fpsShownAt = now;
      const fps = meter.rate(now);
      setText($("st-fps"), fps === null ? "not measured" : fps.toFixed(1));
    }
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
      notes.report("danger", e.message, "Camera");
      return;
    }
    drawCells(grey, cells, r.settings.width, r.settings.height, 255);
    setText($("grey-text"), `${r.settings.height} x ${r.settings.width} grey cells of the newest frame, as sent to the model.`);
    placeCrop(W, H);
    r.frameWindow.push(cells, t);
    r.frameWindow.list().forEach((v, k) => { const c = $("cam-strip").children[k]?.querySelector("canvas"); if (c) drawCells(c, v, r.settings.width, r.settings.height, 255); });
    const snap = r.frameWindow.snapshot();
    setText($("st-fill"), snap ? `${r.settings.frames} of ${r.settings.frames} frames` : "filling");
    if (snap && performance.now() >= r.retryAt) r.sender.offer(snap);
    setText($("st-dropped"), String(r.sender.dropped()));
  }

  // Grey values (times toByte) into a canvas of w x h pixels; CSS scales it up with its pixels kept.
  function drawCells(canvas, cells, w, h, toByte) {
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const img = new ImageData(w, h);
    cells.forEach((v, k) => { const g = Math.max(0, Math.min(255, Math.round(v * toByte))); img.data.set([g, g, g, 255], 4 * k); });
    canvas.getContext("2d").putImageData(img, 0, 0);
  }

  // The centre square the model uses, outlined on the preview.
  let cropKey = "";
  function placeCrop(W, H) {
    const c = cropRect(W, H, video.clientWidth, video.clientHeight), box = $("cam-crop");
    box.hidden = !c;
    const key = c ? `${c.left},${c.top},${c.size}` : "";
    if (c && key !== cropKey) Object.assign(box.style, { left: `${c.left}px`, top: `${c.top}px`, width: `${c.size}px`, height: `${c.size}px` });
    cropKey = key;
  }

  // One small canvas per frame of the window, oldest first.
  function buildStrip(s) {
    $("cam-strip").replaceChildren(...Array.from({ length: s.frames }, (_, k) => el("li", {},
      el("canvas", { width: s.width, height: s.height, role: "img", "aria-label": `Frame ${k + 1} of ${s.frames}` }),
      el("span", {}, k === 0 ? "oldest" : k === s.frames - 1 ? "newest" : String(k + 1)))));
  }

  // ---- decisions: one fixed layout, values changed in place --------------------------------------------------------
  const readoutRow = (id) => { const root = $(id); return { val: root.querySelector("[data-v]"), fill: root.querySelector("[data-fill]"), tick: root.querySelector("[data-tick]"), thr: root.querySelector("[data-thr]") }; };
  const rd = { threat: readoutRow("rd-threat"), best: readoutRow("rd-best"), action: readoutRow("rd-action"), margin: readoutRow("rd-margin") };
  const checks = Object.fromEntries([...$("dec-gate").children].map((li) => [li.dataset.check, li]));
  const dash = "—";

  function setChip(tone, text) {
    const chip = $("dec-action");
    const cls = `chip chip--lg chip--${tone}`;
    if (chip.className !== cls) chip.className = cls;
    setText($("dec-action-text"), text);
  }
  function setMeter(row, value, threshold, thresholdText) {
    setText(row.val, value === null ? "not sent" : value.toFixed(2));
    if (row.fill) row.fill.style.transform = `scaleX(${value === null ? 0 : Math.min(1, Math.max(0, value))})`;
    if (row.tick) {
      row.tick.hidden = threshold === null;
      if (threshold !== null) row.tick.style.left = `${Math.min(100, Math.max(0, threshold * 100))}%`;
    }
    setText(row.thr, thresholdText);
  }
  function setCheck(li, ok, text) {
    li.classList.toggle("is-met", ok === true);
    const dot = li.querySelector(".dot");
    dot.classList.toggle("dot--hollow", ok !== true);
    setText(li.querySelector(".checklist__state"), text);
  }
  function setBadges(list) {
    const key = list.map((b) => `${b.tone}:${b.text}`).join("|");
    const box = $("dec-badges");
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    box.replaceChildren(...list.map((b) => el("span", { class: `badge badge--${b.tone}`, ...(b.title ? { title: b.title } : {}) }, b.text)));
  }
  function setVotes(g) {
    const n = g?.ofLast !== null && g?.ofLast !== undefined ? Math.min(12, Math.max(0, Math.round(g.ofLast))) : 0;
    const box = $("dec-votes");
    const key = `${n}:${g?.votes}:${g?.votesNeeded}`;
    if (box.dataset.key !== key) {
      box.dataset.key = key;
      const dots = Array.from({ length: n }, (_, k) => el("span", { class: `dot${k < (g?.votes ?? 0) ? "" : " dot--hollow"}` }));
      const need = g?.votesNeeded;
      if (Number.isFinite(need) && need > 0 && need < n) dots.splice(need, 0, el("span", { class: "votes__need" }));
      box.replaceChildren(...dots);
    }
    setText($("dec-votes-text"), !g || g.votes === null ? "Votes: not sent"
      : `Votes: ${g.votes} in the last ${g.ofLast ?? "?"}, ${g.votesNeeded ?? "?"} needed`);
  }
  function setCountdown(id, v) {
    const root = $(id);
    setText(root.querySelector("[data-v]"), v === null ? "not sent" : `${v.toFixed(2)} s`);
    root.classList.toggle("is-active", v !== null && v > 0);
  }

  function clearDecision(text) {
    setChip("neutral", text);
    setText($("dec-issued"), "Commands appear here when the model answers a window.");
    setBadges([]);
    for (const k of ["threat", "action"]) { rd[k].fill.style.transform = "scaleX(0)"; setText(rd[k].val, dash); }
    setText(rd.best.val, dash);
    setText(rd.margin.val, dash);
    for (const li of Object.values(checks)) setCheck(li, null, "no data");
    setVotes(null);
    setText($("dec-votes-text"), "Votes: no data");
    const gs = $("dec-gate-state");
    gs.classList.remove("is-open");
    gs.querySelector(".dot").classList.add("dot--hollow");
    setText(gs.lastElementChild, "Gate: no data");
    for (const id of ["dec-cooldown", "dec-hold"]) { setText($(id).querySelector("[data-v]"), dash); $(id).classList.remove("is-active"); }
    setText($("dec-flags"), "Window valid: no data · Cooldown conflict: no data");
    for (const id of ["tm-inference", "tm-latency", "tm-age", "tm-rtt", "tm-age-answer"]) setText($(id), dash);
  }
  function resetReadouts() {
    clearDecision("No decision yet");
    for (const id of ["st-dropped", "st-decisions", "st-issued"]) setText($(id), "0");
  }

  function showDecision(r, { result, rttMs, ageMs }) {
    if (r !== run) return;
    const view = readout(result);
    if (view.state !== "decision") {
      r.retryAt = performance.now() + RETRY_MS; // unavailable or failed: pause instead of repeating the request 30 times a second
      r.lastAction = null;
      setChip("neutral", view.state === "unavailable" ? "No decision: inference unavailable" : "No decision: inference failed");
      setText($("dec-issued"), "No command issued.");
      if (view.state === "unavailable") notes.report("warn", `Inference is unavailable: ${view.message}`, "Unavailable");
      else notes.report("danger", `Inference failed: ${view.message} (${view.code})`, "Error");
      return;
    }
    r.decisions++;
    if (view.issued === true) r.issued++;
    if (view.development) notes.set("dev", "warn", "Development thresholds: the server says this model's decision thresholds are not calibrated yet.", "Development thresholds");
    else notes.clear("dev");
    setText($("st-decisions"), String(r.decisions));
    setText($("st-issued"), String(r.issued));
    // The chip is a live region: it changes only when the action does, so it is announced once per change.
    const key = `${view.action.code}:${view.development}`;
    if (r.lastAction !== key) {
      r.lastAction = key;
      setChip(view.action.tone, `${view.action.label} (${view.action.code})${view.development ? ", development thresholds" : ""}`);
    }
    setText($("dec-issued"), commandText(view, fields));
    setBadges(view.badges);
    const v = view.values, th = view.thresholds;
    const thrText = (x) => (x === null || x === undefined ? "Threshold not sent by the server" : `Threshold ${x.toFixed(2)}`);
    setMeter(rd.threat, v.threat, th?.threat ?? null, thrText(th?.threat));
    setText(rd.best.val, v.best ?? "not sent");
    setMeter(rd.action, v.action, th?.action ?? null, thrText(th?.action));
    setText(rd.margin.val, v.margin === null ? "not sent" : v.margin.toFixed(2));
    setText(rd.margin.thr, th?.margin === null || th?.margin === undefined ? "Threshold not sent by the server" : `Needs at least ${th.margin.toFixed(2)}`);
    const g = view.gateState;
    const state = (ok) => (ok === true ? "met" : ok === false ? "not met" : "not sent");
    setCheck(checks.threat_ok, g?.threat ?? null, state(g?.threat ?? null));
    setCheck(checks.action_ok, g?.action ?? null, state(g?.action ?? null));
    setCheck(checks.margin_ok, g?.margin ?? null, state(g?.margin ?? null));
    setCheck(checks.votes, g?.votesOk ?? null, state(g?.votesOk ?? null));
    setVotes(g);
    const gs = $("dec-gate-state"), open = g?.passed === true;
    gs.classList.toggle("is-open", open);
    gs.querySelector(".dot").classList.toggle("dot--hollow", !open);
    setText(gs.lastElementChild, !g || g.passed === null ? "Gate: not sent" : open ? "Gate open" : "Gate closed");
    setCountdown("dec-cooldown", v.cooldown);
    setCountdown("dec-hold", v.hold);
    const yn = (x) => (x === true ? "yes" : x === false ? "no" : "not sent");
    setText($("dec-flags"), `Window valid: ${yn(v.valid)} · Cooldown conflict: ${yn(v.cooldownConflict)}`);
    setText($("tm-inference"), view.timing[0][1]);
    setText($("tm-latency"), view.timing[1][1]);
    setText($("tm-age"), view.timing[2][1]);
    setText($("tm-rtt"), `${rttMs.toFixed(1)} ms`);
    setText($("tm-age-answer"), `${ageMs.toFixed(1)} ms`);
  }

  $("cam-start").addEventListener("click", start);
  $("cam-stop").addEventListener("click", () => stop("The camera is off and its frames are cleared."));
  addEventListener("pagehide", () => { if (stream) stop("The camera is off."); });
  clearDecision("No decision yet");

  return { setState, setVisible };
}
