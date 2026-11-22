// The real simulated Go2 in 3D from /api/robot and the SSE pose and scene events: the landing stage, plus any other
// view attached to it (the Live section's dog), all fed by the same events. When the stage's 3D view cannot work it
// shows the simulator's own camera stream instead; the status line says which, and why.
import { createViewer } from "./viewer/viewer.js";

const FRESH_MS = 1500; // a pose newer than this counts as "live"

export function poseStatus(poseCount, sinceLastMs) {
  if (!poseCount) return "Waiting for the simulator's first pose.";
  return sinceLastMs < FRESH_MS
    ? "Live from the MuJoCo simulation. Drag to look around; zoom with the buttons, Ctrl and the wheel, or a pinch."
    : "The simulator has stopped sending poses; the dog is shown where it last was.";
}

// The stage's sky and floor from the page's tokens (style.css --stage-sky, --stage-floor), as #rrggbb.
function stageColours() {
  const css = getComputedStyle(document.documentElement);
  const pick = (name) => { const v = css.getPropertyValue(name).trim(); return /^#[0-9a-f]{6}$/i.test(v) ? v : undefined; };
  return Object.fromEntries(Object.entries({ background: pick("--stage-sky"), floor: pick("--stage-floor") }).filter(([, v]) => v));
}

export function initRobotStage({ api }) {
  const $ = (id) => document.getElementById(id);
  const canvas = $("scene"), fallback = $("scene-fallback"), statusEl = $("scene-status");
  let viewer = null, robot = null, lastPoseAt = 0, problem = null, loading = false, visible = true, poseCount = 0, shift = [0, 0, 1];
  const attached = []; // {viewer, onStatus}

  const say = (text) => { if (statusEl.textContent !== text) statusEl.textContent = text; };

  function useFallback(reason) {
    problem = reason;
    canvas.hidden = true;
    fallback.hidden = false;
    viewer?.setActive(false);
    if (visible && fallback.getAttribute("src") !== "/api/stream?camera=tracking") fallback.setAttribute("src", "/api/stream?camera=tracking");
    say(`The 3D view is unavailable: ${reason} Showing the simulator's tracking camera instead.`);
  }

  async function loadRobot() {
    if (loading) return;
    loading = true;
    const r = await api.robot();
    loading = false;
    if (!r.ok) {
      for (const a of attached) a.onStatus(`The 3D view is unavailable: ${r.message}`);
      return useFallback(r.message);
    }
    robot = r.data;
    for (const a of attached) {
      try { a.viewer.setRobot(robot); } catch (e) { a.onStatus(`The 3D view is unavailable: ${e.message}`); }
    }
    try {
      if (!viewer) { viewer = createViewer(canvas, stageColours()); viewer.setShift(...shift); }
      viewer.setRobot(robot);
    } catch (e) {
      return useFallback(e.message);
    }
    problem = null;
    canvas.hidden = false;
    fallback.hidden = true;
    fallback.removeAttribute("src");
    viewer.setActive(visible);
    say(poseStatus(poseCount, performance.now() - lastPoseAt));
  }

  const bodyCount = () => viewer?.bodyCount() || attached.find((a) => a.viewer.bodyCount())?.viewer.bodyCount() || 0;

  function onPose(pose) {
    if (!robot) return;
    if (Array.isArray(pose?.pos) && pose.pos.length !== bodyCount()) return loadRobot(); // the world changed
    if (viewer && !problem) viewer.pushPose(pose);
    for (const a of attached) a.viewer.pushPose(pose);
    lastPoseAt = performance.now();
    poseCount++;
  }

  function onScene(scene) {
    if (!scene || typeof scene !== "object") return;
    if (viewer && !problem) viewer.setScene(scene);
    for (const a of attached) a.viewer.setScene(scene);
  }

  // The status follows the poses on this page's clock: live while they arrive, stale when they stop.
  setInterval(() => {
    const text = poseStatus(poseCount, performance.now() - lastPoseAt);
    if (!problem && viewer) say(text);
    for (const a of attached) if (a.ok) a.onStatus(text);
  }, 500);

  function setVisible(v) {
    visible = v;
    if (problem) {
      if (v) fallback.setAttribute("src", "/api/stream?camera=tracking");
      else fallback.removeAttribute("src"); // closes the stream
    } else viewer?.setActive(v);
  }

  // Another 3D view of the same dog, fed by the same events. -> {ok, reason?, setActive, resetView}
  function attach(otherCanvas, onStatus = () => {}) {
    let v;
    try { v = createViewer(otherCanvas, { distance: 1.35 }); } catch (e) { onStatus(`The 3D view is unavailable: ${e.message}`); return { ok: false, reason: e.message, setActive() {}, resetView() {} }; }
    v.setActive(false);
    const a = { viewer: v, onStatus, ok: true };
    attached.push(a);
    if (robot) { try { v.setRobot(robot); } catch (e) { a.ok = false; onStatus(`The 3D view is unavailable: ${e.message}`); } }
    return { ok: true, setActive: (x) => v.setActive(x), resetView: () => v.resetView() };
  }

  loadRobot();
  return {
    onPose,
    onScene,
    onRobotChanged: loadRobot,
    setVisible,
    attach,
    setShift: (x, y, k) => { shift = [x, y, k]; viewer?.setShift(x, y, k); },
    resetView: () => viewer?.resetView(),
    zoomBy: (f) => viewer?.zoomBy(f),
  };
}
