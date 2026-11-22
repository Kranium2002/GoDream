// The stage: which lanes show (layout), the 3D view in each lane, each lane's outcome line, the shared count-in, and the
// single stage's caption. Layouts: "duel" (before | after), "single" (the dog as the simulator streams it), "camera"
// (your camera | the dog reacting) and "class" (one wide lane playing a lesson's real tries).
import { createRobotViews } from "./robot-stage.js";
import { outcomeLine } from "./demos.js";
import { $, el, setText } from "./dom.js";

const COUNT_STEP_MS = 520; // one beat of the count-in before both lanes start
const RESULT_TONE = { Hit: "bad", Fell: "bad", Dodged: "good" };

export function initStageView({ api, notes }) {
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const views = createRobotViews({ api, onProblem: (text) => notes.set("3d", "warn", text, "3D view") });
  // The two lanes look at their dogs from the same angle: behind and to the left, the heading of the first pose kept,
  // the frame widened to hold the hill or the incoming object.
  const laneOptions = { distance: 2.2, yaw: 2.6, pitch: 0.45, lockHeading: true, frameScene: true };
  const lanes = {
    before: views.attach($("view-before"), { side: "before", palette: "before", ...laneOptions }),
    after: views.attach($("view-after"), { side: "after", palette: "after", ...laneOptions }),
    single: views.attach($("view-single"), { side: null, palette: "single", fallback: $("view-single-fallback"), distance: 1.7 }),
  };
  let layout = null;

  function setLayout(next) {
    if (next === layout) return;
    layout = next;
    document.body.dataset.layout = next;
    lanes.before.setActive(next === "duel");
    lanes.after.setActive(next === "duel");
    lanes.single.setActive(next === "single" || next === "camera" || next === "class");
    place();
  }

  // ---- framing: each dog sits in the middle of the part of its lane that no caption, strip or bar covers ------------
  function place() {
    const root = document.documentElement;
    root.style.setProperty("--top-h", `${Math.round(document.querySelector(".top").getBoundingClientRect().height)}px`);
    const bar = $("bar").getBoundingClientRect();
    root.style.setProperty("--bar-h", `${Math.round(bar.height)}px`);
    const sheet = $("sheet").hidden ? null : $("sheet").getBoundingClientRect();
    for (const [key, id] of [["before", "lane-before"], ["after", "lane-after"], ["single", "lane-single"]]) {
      const lane = $(id).getBoundingClientRect();
      if (!lane.height || !lane.width) continue;
      // What covers the bottom of the lane: the sense strip of a duel lane, the dog's state under the camera's dog.
      const strip = key === "single" ? ($("react").hidden ? null : $("react").getBoundingClientRect()) : $(`sense-${key}`).getBoundingClientRect();
      const top = lane.top + 40;
      const bottom = Math.min(lane.bottom, strip && strip.height ? strip.top : lane.bottom, bar.top > lane.top + 120 ? bar.top - 8 : lane.bottom);
      let left = lane.left, right = lane.right;
      // The sheet covers the right of the stage: the dog moves into what is left, when enough is left to see it.
      if (sheet && sheet.left < right && sheet.left - 16 - left >= 260) right = sheet.left - 16;
      const freeW = right - left, freeH = Math.max(80, bottom - top);
      const xNdc = (2 * ((left + right) / 2 - lane.left)) / lane.width - 1;
      const yNdc = 1 - (2 * ((top + bottom) / 2 - lane.top)) / lane.height;
      const k = Math.max(0.45, Math.min(1, 1.15 * Math.min(freeW / lane.width, freeH / lane.height)));
      lanes[key].setShift(xNdc, yNdc, k);
    }
  }
  addEventListener("resize", place);
  const ro = new ResizeObserver(() => place());
  for (const id of ["bar", "react", "sheet"]) ro.observe($(id));

  // ---- outcome lines ------------------------------------------------------------------------------------------------
  function setOutcome(side, fields, lane) {
    const p = $(`outcome-${side}`);
    // Nothing while the lane runs: the dog and its strip show that. One line once the server says the lane is done.
    if (!lane || lane.status !== "done") { p.hidden = true; return; }
    // A lane whose outcome carries the server's result word (the dodge duel) shows that word alone.
    const word = typeof lane.outcome?.result === "string" ? lane.outcome.result : null;
    const line = word ? { text: word, detail: null, tone: RESULT_TONE[word] ?? "neutral" } : outcomeLine(fields, lane.outcome);
    p.hidden = false; p.dataset.pending = "false";
    p.dataset.tone = line?.tone ?? "neutral";
    const text = line ? line.text : "Finished. The server sent no outcome this page can put into words.";
    if (p.dataset.text !== text + (line?.detail ?? "")) {
      p.dataset.text = text + (line?.detail ?? "");
      p.replaceChildren(text, ...(line?.detail ? [el("span", { class: "lane__detail" }, line.detail)] : []));
    }
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

  // ---- the single stage's caption -------------------------------------------------------------------------------
  setInterval(() => {
    if (layout === "single" || layout === "camera") setText($("status-single"), lanes.single.status());
    else setText($("status-single"), "");
  }, 500);

  return {
    setLayout,
    layout: () => layout,
    place,
    setOutcome,
    countIn,
    setTally(side, text) { const p = $(`tally-${side}`); p.hidden = !text; setText(p, text ? `So far: ${text}` : ""); },
    clearLanes() { lanes.before.clear(); lanes.after.clear(); },
    onPose: (d) => views.onPose(d),
    onScene: (d) => views.onScene(d),
    onRobotChanged: () => views.onRobotChanged(),
    hasPose: (key) => lanes[key].hasPose(),
  };
}
