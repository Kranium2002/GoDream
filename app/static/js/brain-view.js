// The learned lane's live readout of its spiking brain, from the server's "brain" events: the median time of its
// latest decisions as measured on this laptop, the choice it is making in plain words, and, when the events carry
// them, its spikes per decision as small bars. Hidden until the first decision arrives. Redrawn once per frame at most.
import { BRAIN_KEEP, fmtMs } from "./showcase.js";
import { $, setText, svgEl } from "./dom.js";

const W = 160, H = 30, GAP = 1; // the bars, in CSS pixels
const MOVE_WORDS = new Set(["Step left", "Step right", "Step back"]); // dodge moves, drawn in the move colour

export function initBrainView() {
  let view = null, running = false, queued = false;

  function draw() {
    queued = false;
    const box = $("brain");
    const show = Boolean(view && view.ms !== null);
    if (box.hidden === show) box.hidden = !show;
    if (!show) return;
    setText($("brain-ms"), `${fmtMs(view.ms)} ms`);
    const choice = $("brain-choice");
    choice.hidden = !running || !view.choice;
    setText(choice, view.choice ?? "");
    choice.dataset.move = String(MOVE_WORDS.has(view.choice));
    const spark = $("brain-spikes");
    spark.hidden = !view.spikes;
    if (view.spikes) bars($("brain-bars"), view.spikes);
  }

  function bars(svg, spikes) {
    const max = Math.max(1, ...spikes.filter((s) => s !== null));
    const w = W / BRAIN_KEEP;
    const offset = BRAIN_KEEP - spikes.length; // newest at the right edge
    while (svg.childNodes.length < spikes.length) svg.append(svgEl("rect", { class: "spark__bar" }));
    while (svg.childNodes.length > spikes.length) svg.lastChild.remove();
    spikes.forEach((s, k) => {
      const h = s === null ? 0 : Math.max(1, (s / max) * H);
      const r = svg.childNodes[k];
      r.setAttribute("x", ((offset + k) * w).toFixed(1));
      r.setAttribute("y", (H - h).toFixed(1));
      r.setAttribute("width", Math.max(1, w - GAP).toFixed(1));
      r.setAttribute("height", h.toFixed(1));
      r.dataset.new = String(k === spikes.length - 1);
    });
  }

  $("brain-bars").setAttribute("viewBox", `0 0 ${W} ${H}`);

  return {
    // view: brainView() of showcase.js, or null to hide; running: whether the round is still going (the choice shows only then).
    show(v, isRunning) {
      view = v; running = Boolean(isRunning);
      if (!queued) { queued = true; requestAnimationFrame(draw); }
    },
  };
}
