// The landing stage: the real simulated Go2 in 3D from /api/robot and SSE pose events, or, when the 3D view cannot
// work, the simulator's own camera stream. The status line says which, and why.
import { createViewer } from "./viewer/viewer.js";

const FRESH_MS = 1500; // a pose newer than this counts as "live"

export function initRobotStage({ api }) {
  const $ = (id) => document.getElementById(id);
  const canvas = $("scene"), fallback = $("scene-fallback"), statusEl = $("scene-status");
  let viewer = null, lastPoseAt = 0, problem = null, loading = false, visible = true, poseCount = 0, shift = [0, 0];

  const say = (text) => { if (statusEl.textContent !== text) statusEl.textContent = text; };

  function useFallback(reason) {
    problem = reason;
    canvas.hidden = true;
    fallback.hidden = false;
    if (visible && fallback.getAttribute("src") !== "/api/stream?camera=tracking") fallback.setAttribute("src", "/api/stream?camera=tracking");
    say(`The 3D view is unavailable: ${reason} Showing the simulator's tracking camera instead.`);
  }

  async function loadRobot() {
    if (loading) return;
    loading = true;
    const r = await api.robot();
    loading = false;
    if (!r.ok) return useFallback(r.message);
    try {
      if (!viewer) { viewer = createViewer(canvas); viewer.setShift(...shift); }
      viewer.setRobot(r.data);
    } catch (e) {
      return useFallback(e.message);
    }
    problem = null;
    canvas.hidden = false;
    fallback.hidden = true;
    fallback.removeAttribute("src");
    say("Waiting for the simulator's first pose.");
  }

  function onPose(pose) {
    if (!viewer || problem) return;
    if (Array.isArray(pose?.pos) && pose.pos.length !== viewer.bodyCount()) return loadRobot(); // the world changed
    viewer.pushPose(pose);
    lastPoseAt = performance.now();
    poseCount++;
  }

  // The status follows the poses on this page's clock: live while they arrive, stale when they stop.
  setInterval(() => {
    if (problem || !viewer) return;
    if (!poseCount) return say("Waiting for the simulator's first pose.");
    say(performance.now() - lastPoseAt < FRESH_MS
      ? "Live: the real MuJoCo simulation of the Go2. Drag to look around, scroll or pinch to zoom."
      : "The simulator has stopped sending poses; the dog is shown where it last was.");
  }, 500);

  function setVisible(v) {
    visible = v;
    if (problem) {
      if (v) fallback.setAttribute("src", "/api/stream?camera=tracking");
      else fallback.removeAttribute("src"); // closes the stream
    }
  }

  loadRobot();
  return {
    onPose,
    onRobotChanged: loadRobot,
    setVisible,
    setShift: (x, y) => { shift = [x, y]; viewer?.setShift(x, y); },
    resetView: () => viewer?.resetView(),
  };
}
