// The real simulated Go2 in 3D, from /api/robot placed by the SSE pose and scene events: one robot description shared
// by every view on the page. Each view listens to one lane: "before" and "after" (a pose or scene event that names that
// side, during a side-by-side run) or null (the single stage: events without a side, as the simulator and the episode
// player send them). When WebGL cannot work, the single stage shows the simulator's own tracking camera instead.
import { createViewer } from "./viewer/viewer.js";
import { sideOf } from "./duel.js";

const FRESH_MS = 1500; // a pose newer than this counts as live

export function poseStatus(poseCount, sinceLastMs) {
  if (!poseCount) return "Waiting for the simulator's first pose.";
  return sinceLastMs < FRESH_MS ? "Live from the simulator. Drag to look around." : "The simulator has paused; the dog is shown where it last was.";
}

// The stage's sky and floor from the page's tokens (style.css --sky-before etc.), as #rrggbb.
function colours(prefix) {
  const css = getComputedStyle(document.documentElement);
  const pick = (name) => { const v = css.getPropertyValue(name).trim(); return /^#[0-9a-f]{6}$/i.test(v) ? v : undefined; };
  return Object.fromEntries(Object.entries({ background: pick(`--sky-${prefix}`), floor: pick(`--floor-${prefix}`) }).filter(([, v]) => v));
}

export function createRobotViews({ api, onProblem = () => {} }) {
  let robot = null, loading = false;
  const views = []; // {side, viewer, fallback, poses, lastPoseAt}

  async function loadRobot() {
    if (loading) return;
    loading = true;
    const r = await api.robot();
    loading = false;
    if (!r.ok) { onProblem(`The robot description did not load: ${r.message}`); for (const v of views) useFallback(v, r.message); return; }
    robot = r.data;
    for (const v of views) {
      if (!v.viewer) continue;
      try { v.viewer.setRobot(robot); } catch (e) { useFallback(v, e.message); }
    }
  }

  function useFallback(v, reason) {
    v.problem = reason;
    if (v.fallback && v.side === null) {
      v.canvas.hidden = true;
      v.fallback.hidden = false;
      if (v.active && v.fallback.getAttribute("src") !== "/api/stream?camera=tracking") v.fallback.setAttribute("src", "/api/stream?camera=tracking");
    }
    onProblem(`The 3D view cannot draw here (${reason}).${v.fallback ? " The simulator's own camera is shown instead." : ""}`);
  }

  // A view of one lane in `canvas`. options: viewer options (distance, yaw, pitch, lockHeading, frameScene) and the
  // colour set ("before", "after" or "single"). -> {setActive, setShift, clear, resetView, status(), ok}
  function attach(canvas, { side = null, palette = "single", fallback = null, ...options } = {}) {
    const v = { side, canvas, fallback, viewer: null, poses: 0, lastPoseAt: 0, problem: null, active: false };
    try {
      v.viewer = createViewer(canvas, { ...colours(palette), trained: side === "after", ...options });
      v.viewer.setActive(false);
      if (robot) v.viewer.setRobot(robot);
    } catch (e) {
      v.viewer = null;
      useFallback(v, e.message);
    }
    views.push(v);
    return {
      ok: () => Boolean(v.viewer) && !v.problem,
      setActive(on) {
        v.active = on;
        if (v.viewer) v.viewer.setActive(on);
        else if (v.fallback && v.side === null) { if (on) v.fallback.setAttribute("src", "/api/stream?camera=tracking"); else v.fallback.removeAttribute("src"); }
      },
      setShift: (x, y, k) => v.viewer?.setShift(x, y, k),
      clear() { v.viewer?.clear(); v.poses = 0; },
      resetView: () => v.viewer?.resetView(),
      setYaw: (yaw, fit) => v.viewer?.setYaw(yaw, fit),
      status: () => (v.problem ? `The 3D view cannot draw here: ${v.problem}` : poseStatus(v.poses, performance.now() - v.lastPoseAt)),
      hasPose: () => v.poses > 0,
    };
  }

  const bodyCount = () => views.find((v) => v.viewer?.bodyCount())?.viewer.bodyCount() ?? 0;

  // A pose with fewer bodies than the description, or of another robot version, belongs to another world: the
  // description is read again (at most every 2 s). More bodies than described is a world with objects added after the
  // robot (a hill, a ball), whose robot bodies keep their indices: the robot is drawn from the first ones.
  let lastLoad = 0;
  function onPose(pose) {
    if (!robot) return;
    const n = bodyCount();
    const otherVersion = pose?.version !== undefined && robot?.version !== undefined && pose.version !== robot.version;
    if (Array.isArray(pose?.pos) && n && (pose.pos.length < n || otherVersion)) {
      if (performance.now() - lastLoad > 2000) { lastLoad = performance.now(); loadRobot(); }
      return;
    }
    const side = sideOf(pose);
    for (const v of views) {
      if (v.side !== side || !v.viewer) continue;
      v.viewer.pushPose(pose);
      v.poses++;
      v.lastPoseAt = performance.now();
    }
  }

  function onScene(scene) {
    if (!scene || typeof scene !== "object") return;
    const side = sideOf(scene);
    for (const v of views) if (v.side === side) v.viewer?.setScene(scene);
  }

  loadRobot();
  return { attach, onPose, onScene, onRobotChanged: loadRobot };
}
