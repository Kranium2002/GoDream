// The stage: which lanes show (layout), the 3D view in each lane, each lane's result, the shared count-in, the single
// stage's caption and the landing page's live dog. Layouts: "home" (the landing page: only its hero view draws),
// "duel" (before | after), "single" (the dog as the simulator streams it) and "camera" (your camera | the dog
// reacting).
import { createRobotViews } from "./robot-stage.js";
import { outcomeLine } from "./demos.js";
import { laneResult } from "./showcase.js";
import { $, setText } from "./dom.js";

const COUNT_STEP_MS = 520; // one beat of the count-in before both lanes start

export function initStageView({ api, notes }) {
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const views = createRobotViews({ api, onProblem: (text) => notes.set("3d", "warn", text, "3D view") });
  // The two lanes look at their dogs from the same angle: behind and to the left, the heading of the first pose kept,
  // the frame widened to hold the hill or the incoming object.
  const laneOptions = { distance: 2.2, yaw: 2.6, pitch: 0.45, lockHeading: true, frameScene: true };
  // A demo seen by eye (an object comes at the dog's head) is watched from the dog's left side instead, a little from
  // behind: its head and what comes at it both show, the object entering from the left. From behind, the head is hidden.
  const SIDE_YAW = 1.75;
  const SIDE_FIT = 2.1; // wider than the hills' 1.45: a punch that glances off or is dodged keeps flying past the dog
  const lanes = {
    before: views.attach($("view-before"), { side: "before", palette: "before", ...laneOptions }),
    after: views.attach($("view-after"), { side: "after", palette: "after", ...laneOptions }),
    single: views.attach($("view-single"), { side: null, palette: "single", fallback: $("view-single-fallback"), distance: 1.7 }),
    // The landing page's dog: the same side-less poses as the single stage. No stand-in picture when it cannot draw.
    hero: views.attach($("view-hero"), { side: null, palette: "hero", distance: 1.9, yaw: 0.9, pitch: 0.28 }),
  };
  let layout = null;

  function setLayout(next) {
    if (next === layout) return;
    layout = next;
    document.body.dataset.layout = next;
    lanes.before.setActive(next === "duel");
    lanes.after.setActive(next === "duel");
    lanes.single.setActive(next === "single" || next === "camera");
    lanes.hero.setActive(next === "home");
    heroStatus();
    place();
  }

  // ---- framing: each dog sits in the middle of the part of its lane that no caption or strip covers -------------------
  function place() {
    for (const [key, id] of [["before", "lane-before"], ["after", "lane-after"], ["single", "lane-single"]]) {
      const lane = $(id).getBoundingClientRect();
      if (!lane.height || !lane.width) continue;
      // What covers the bottom of the lane: the sense strip of a duel lane, the dog's state under the camera's dog.
      const strip = key === "single" ? ($("react").hidden ? null : $("react").getBoundingClientRect()) : $(`sense-${key}`).getBoundingClientRect();
      // What covers the top: the lane's label and result, and the learned lane's brain readout.
      const heads = [...$(id).querySelectorAll(".lane__head, .brain")].filter((n) => !n.hidden).map((n) => n.getBoundingClientRect().bottom);
      const top = Math.min(lane.top + lane.height * 0.45, Math.max(lane.top + 48, ...heads) + 8);
      const bottom = Math.min(lane.bottom, strip && strip.height ? strip.top : lane.bottom);
      const left = lane.left, right = lane.right;
      const freeW = right - left, freeH = Math.max(80, bottom - top);
      const xNdc = (2 * ((left + right) / 2 - lane.left)) / lane.width - 1;
      const yNdc = 1 - (2 * ((top + bottom) / 2 - lane.top)) / lane.height;
      const k = Math.max(0.45, Math.min(1, 1.15 * Math.min(freeW / lane.width, freeH / lane.height)));
      lanes[key].setShift(xNdc, yNdc, k);
    }
  }
  addEventListener("resize", place);
  const ro = new ResizeObserver(() => place());
  for (const id of ["stage", "react", "head-before", "head-after", "brain"]) ro.observe($(id));

  // ---- results -------------------------------------------------------------------------------------------------------
  function setOutcome(side, fields, lane) {
    const big = $(`result-${side}`), detail = $(`detail-${side}`);
    // Nothing while the lane runs: the dog and its strip show that. The result once the server says the lane is done.
    if (!lane || lane.status !== "done") { big.hidden = true; detail.hidden = true; return; }
    let r = laneResult(lane.outcome);
    if (!r) {
      const line = outcomeLine(fields, lane.outcome); // a demo whose outcome this page has no plain words for
      r = line ? { big: line.text, detail: line.detail, tone: line.tone } : null;
    }
    big.hidden = !r;
    detail.hidden = !r?.detail;
    if (!r) return;
    big.dataset.tone = r.tone;
    big.dataset.long = String(r.big.length > 12); // a sentence ("Did not reach the finish line") is set smaller than a word
    setText(big, r.big);
    setText(detail, r.detail ?? "");
  }

  // ---- the one orchestrated moment: a shared count-in, then both lanes start together ----------------------------
  function countIn() {
    if (reduceMotion.matches) return Promise.resolve();
    const box = $("countin");
    return new Promise((resolve) => {
      let n = 3;
      const step = () => {
        if (n === 0) { box.hidden = true; resolve(); return; }
        box.hidden = false;
        box.textContent = String(n);
        box.classList.remove("tick"); void box.offsetWidth; box.classList.add("tick");
        n--;
        setTimeout(step, COUNT_STEP_MS);
      };
      step();
    });
  }

  // ---- captions: the single stage's and the landing page's live dog -------------------------------------------------
  function heroStatus() {
    // The landing page's frame shows only while its view can draw; a view that cannot start leaves nothing there.
    $("hero-visual").hidden = !lanes.hero.ok();
    setText($("hero-status"), layout === "home" && lanes.hero.ok() && lanes.hero.hasPose() ? "Live from the simulator" : "");
  }
  setInterval(() => {
    if (layout === "single" || layout === "camera") setText($("status-single"), lanes.single.status());
    else setText($("status-single"), "");
    if (layout === "home") heroStatus();
  }, 500);

  return {
    setLayout,
    layout: () => layout,
    place,
    setOutcome,
    countIn,
    setSideView(on) { for (const key of ["before", "after"]) lanes[key].setYaw(on ? SIDE_YAW : laneOptions.yaw, on ? SIDE_FIT : undefined); },
    setTally(side, text) { const p = $(`tally-${side}`); p.hidden = !text; setText(p, text ? `This session: ${text}` : ""); },
    // The lanes' names as the demo declares them; a name it leaves out is not shown.
    setLaneNames(names) {
      for (const side of ["before", "after"]) { const n = $(`name-${side}`); n.hidden = !names?.[side]; setText(n, names?.[side] ?? ""); }
    },
    clearLanes() { lanes.before.clear(); lanes.after.clear(); },
    onPose: (d) => views.onPose(d),
    onScene: (d) => views.onScene(d),
    onRobotChanged: () => views.onRobotChanged(),
    hasPose: (key) => lanes[key].hasPose(),
  };
}
