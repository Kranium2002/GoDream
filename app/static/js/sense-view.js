// The sense strips under the two lanes: what each dog feels (nose-up tilt, roll, turning rate and the four feet, like a
// seismograph) or what its eye notices (the 12 x 12 event view, newest large, a few before it small). Drawn from the
// server's "feel" and "eye" events only; both lanes share one scale per channel so a slope that bends one trace more
// than the other is a real difference. Redrawn at most once per animation frame.
import { contactRuns, createTrace, readEye, readFeel, sharedScale, tracePath } from "./feel.js";
import { changePicture } from "./live.js";
import { el, svgEl } from "./dom.js";

const SPAN_S = 6; // seconds of feel shown across a strip
const PAST = 6;   // earlier eye pictures shown beside the newest
const ROWS = [
  { key: "nose", label: "Nose up", get: (s) => s.tilt[0], unit: "°" },
  { key: "roll", label: "Roll", get: (s) => s.tilt[1], unit: "°" },
  { key: "turn", label: "Turning", get: (s) => s.turn[2], unit: "" },
];
const FEET = ["Front left", "Front right", "Rear left", "Rear right"];

export function createSenses({ before, after }, { eyeSettings = () => null } = {}) {
  const boxes = { before, after };
  const traces = { before: createTrace(SPAN_S), after: createTrace(SPAN_S) };
  const eyes = { before: [], after: [] }; // newest last: Float32Array pictures
  const lastCells = { before: null, after: null };
  let kind = null, queued = false;
  const parts = { before: null, after: null };

  function build(side) {
    const box = boxes[side];
    if (kind === "feel") {
      const svg = svgEl("svg", { role: "img", "aria-label": `What the dog ${side === "before" ? "with its walking brain alone" : "with the learned reflex"} feels: tilt, turning and its feet touching the ground.` });
      const p = { svg, rows: [], feet: [], empty: null, scales: [] };
      ROWS.forEach((r) => {
        const label = svgEl("text", { class: "sense__label" }); label.textContent = r.label;
        const base = svgEl("line", { class: "sense__base" });
        const line = svgEl("polyline", { class: "sense__trace" });
        const scale = svgEl("text", { class: "sense__scale", "text-anchor": "end" });
        svg.append(label, base, line, scale);
        p.rows.push({ label, base, line, scale });
      });
      const feetLabel = svgEl("text", { class: "sense__label" }); feetLabel.textContent = "Feet";
      svg.append(feetLabel);
      p.feetLabel = feetLabel;
      FEET.forEach((name) => { const g = svgEl("g", { "aria-label": name }); svg.append(g); p.feet.push(g); });
      p.empty = svgEl("text", { class: "sense__empty" });
      p.empty.textContent = "What it feels draws here as it walks.";
      svg.append(p.empty);
      box.replaceChildren(svg);
      parts[side] = p;
    } else if (kind === "eye") {
      const now = el("canvas", { class: "eyes__now pixelated", width: 1, height: 1, role: "img", "aria-label": "What its eye notices now: white got brighter, black got darker, grey did not change." });
      const past = Array.from({ length: PAST }, () => el("canvas", { class: "eyes__past pixelated", width: 1, height: 1, "aria-hidden": "true" }));
      box.replaceChildren(el("div", { class: "eyes" }, el("span", { class: "eyes__label" }, "What its eye notices"), ...past, now));
      parts[side] = { now, past };
    } else {
      box.replaceChildren();
      parts[side] = null;
    }
  }

  function draw() {
    queued = false;
    if (kind === "feel") drawFeel();
    else if (kind === "eye") for (const side of ["before", "after"]) drawEyes(side);
  }

  function drawFeel() {
    const lists = [traces.before.samples(), traces.after.samples()];
    const tEnd = Math.max(...lists.map((l) => l.at(-1)?.t ?? -Infinity));
    const scales = ROWS.map((r) => sharedScale(lists, r.get));
    for (const side of ["before", "after"]) {
      const p = parts[side];
      if (!p) continue;
      const W = boxes[side].clientWidth - 36, H = boxes[side].clientHeight - 14;
      if (W <= 0 || H <= 0) continue;
      p.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const x0 = 70, x1 = W - 34, rowH = (H - 34) / ROWS.length, half = rowH * 0.42;
      const samples = traces[side].samples();
      ROWS.forEach((r, k) => {
        const yMid = rowH * k + rowH / 2;
        const row = p.rows[k];
        set(row.label, { x: 0, y: yMid + 4 });
        set(row.base, { x1: x0, x2: x1, y1: yMid, y2: yMid });
        row.line.setAttribute("points", Number.isFinite(tEnd) ? tracePath(samples, r.get, { x0, x1, yMid, half, tEnd, spanS: SPAN_S, scale: scales[k] }) : "");
        set(row.scale, { x: W, y: yMid + 4 });
        const sc = scales[k];
        row.scale.textContent = sc && r.unit ? `±${sc >= 10 ? Math.round(sc) : sc.toFixed(1)}${r.unit}` : "";
      });
      const fy = rowH * ROWS.length + 4;
      set(p.feetLabel, { x: 0, y: fy + 14 });
      FEET.forEach((_, f) => {
        const y = fy + 3 + f * 5;
        const runs = Number.isFinite(tEnd) ? contactRuns(samples, f, { x0, x1, tEnd, spanS: SPAN_S }) : [];
        const g = p.feet[f];
        while (g.childNodes.length > runs.length) g.lastChild.remove();
        while (g.childNodes.length < runs.length) g.append(svgEl("line", { class: "sense__foot" }));
        runs.forEach(([a, b], i) => set(g.childNodes[i], { x1: a, x2: b, y1: y, y2: y }));
      });
      p.empty.hidden = samples.length > 0;
      p.empty.style.display = samples.length ? "none" : "";
      set(p.empty, { x: x0, y: rowH + 4 });
    }
  }

  function drawEyes(side) {
    const p = parts[side];
    if (!p) return;
    const list = eyes[side];
    const n = Math.round(Math.sqrt(list.at(-1)?.length ?? 0));
    if (!n) return;
    paint(p.now, list.at(-1), n);
    p.past.forEach((c, k) => { const pic = list[list.length - 1 - PAST + k]; if (pic) paint(c, pic, n); });
  }

  function schedule() { if (!queued) { queued = true; requestAnimationFrame(draw); } }

  return {
    // "feel" | "eye" | null: what the lanes' strips show (the demo's declared sense).
    setKind(k) {
      if (k === kind) return;
      kind = k;
      for (const side of ["before", "after"]) build(side);
      schedule();
    },
    clear() {
      traces.before.clear(); traces.after.clear();
      eyes.before = []; eyes.after = []; lastCells.before = lastCells.after = null;
      for (const side of ["before", "after"]) build(side);
      schedule();
    },
    onFeel(ev) {
      const s = readFeel(ev);
      if (!s || !s.side || kind !== "feel") return;
      traces[s.side].push(s);
      schedule();
    },
    onEye(ev) {
      if (kind !== "eye") return;
      const w = eyeSettings();
      const e = readEye(ev, w ? w.height * w.width : null);
      if (!e || !e.side) return;
      // The server sends the eye's grey cells; the picture is the change since the previous cells, with the encoding
      // and threshold the student uses (/api/state image_window). Without them the grey cells are shown as they are.
      const prev = lastCells[e.side];
      lastCells[e.side] = e.cells;
      let pic = e.cells;
      if (w?.windowEncoding && prev && prev.length === e.cells.length) {
        try { pic = changePicture(prev, e.cells, w.windowEncoding, w.diffFloor); } catch { pic = e.cells; }
      } else if (w?.windowEncoding) return; // the first cells of a lane: no change to show yet
      eyes[e.side].push(pic);
      if (eyes[e.side].length > PAST + 1) eyes[e.side].shift();
      schedule();
    },
    redraw: schedule,
    counts: () => ({ before: traces.before.samples().length, after: traces.after.samples().length, eyesBefore: eyes.before.length, eyesAfter: eyes.after.length }),
  };
}

function set(node, attrs) { for (const [k, v] of Object.entries(attrs)) { const s = String(Math.round(v * 10) / 10); if (node.getAttribute(k) !== s) node.setAttribute(k, s); } }

// Values in 0..1 into a canvas of n x n pixels; CSS scales it up with its pixels kept.
export function paint(canvas, cells, n) {
  if (canvas.width !== n || canvas.height !== n) { canvas.width = n; canvas.height = n; }
  const img = new ImageData(n, n);
  for (let k = 0; k < cells.length; k++) {
    const g = Math.max(0, Math.min(255, Math.round(cells[k] * 255)));
    img.data[4 * k] = img.data[4 * k + 1] = img.data[4 * k + 2] = g;
    img.data[4 * k + 3] = 255;
  }
  canvas.getContext("2d").putImageData(img, 0, 0);
}
