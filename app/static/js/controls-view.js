// The top bar's demo tabs and the control bar: one tab per demo the server lists, the demo's handles drawn from their
// declared kind, the two-way switch of a demo that can use the camera, the main button and one line under it. Every
// range, step, default and label is the demo's own; nothing here knows a demo by name.
import { angleDeg, capProfile, fraction, handleText, snap } from "./demos.js";
import { $, el, setText, svgEl } from "./dom.js";

const W = 196, H = 54; // a handle's drawing, in CSS pixels

export function initControls({ onTab, onParam, onMode, onGo, onCamera }) {
  let handlesKey = "";

  // ---- tabs ---------------------------------------------------------------------------------------------------------
  function setTabs(demos, currentId) {
    const nav = $("tabs");
    nav.hidden = !demos.length;
    const key = demos.map((d) => d.id).join("|");
    if (nav.dataset.key !== key) {
      nav.dataset.key = key;
      nav.setAttribute("role", "tablist");
      nav.replaceChildren(...demos.map((d) => el("button", { class: "tab", type: "button", role: "tab", "data-demo": d.id, "aria-selected": "false", title: d.title,
        on: { click: () => onTab(d.id) } }, d.title)));
    }
    for (const b of nav.querySelectorAll(".tab")) b.setAttribute("aria-selected", String(b.dataset.demo === currentId));
  }

  // ---- handles --------------------------------------------------------------------------------------------------------
  function setHandles(demo, params, disabled) {
    const box = $("handles");
    const key = demo ? `${demo.id}:${demo.handles.map((h) => `${h.name}/${h.kind}/${h.min}/${h.max}/${h.step}`).join(",")}` : "";
    if (key !== handlesKey) {
      handlesKey = key;
      box.replaceChildren(...(demo ? demo.handles.map((h) => handleNode(h)) : []));
    }
    for (const node of box.querySelectorAll(".handle")) {
      const h = demo.handles.find((x) => x.name === node.dataset.name);
      const v = params[h.name] ?? h.default;
      const input = node.querySelector("input");
      if (Number(input.value) !== v) input.value = String(v);
      input.disabled = disabled;
      input.setAttribute("aria-valuetext", handleText(h, v));
      setText(node.querySelector(".handle__value"), handleText(h, v));
      drawHandle(node.querySelector("svg"), h, v);
    }
  }

  function handleNode(h) {
    const id = `h-${h.name}`;
    const input = el("input", { type: "range", id, min: h.min, max: h.max, step: h.step, value: h.default });
    input.addEventListener("input", () => onParam(h.name, snap(h, Number(input.value))));
    return el("div", { class: "handle", "data-name": h.name, "data-kind": h.kind },
      el("div", { class: "handle__head" }, el("label", { class: "handle__label", for: id }, h.label), el("span", { class: "handle__value" })),
      el("div", { class: "handle__draw" }, svgEl("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }), input),
      el("div", { class: "handle__ends", "aria-hidden": "true" }, el("span", {}, handleText(h, h.min)), el("span", {}, handleText(h, h.max))));
  }

  // Each kind draws what it changes: a slope the hill's cross-section, an offset where the obstacle sits beside the
  // path (seen from above), a direction the way something comes in, an amount a filled bar.
  function drawHandle(svg, h, v) {
    const f = fraction(h, v);
    const parts = [];
    if (h.kind === "slope") {
      const ground = H - 6, cx = W / 2, a = 64;
      const deg = angleDeg(h, v);
      // An angle unit draws the true angle; any other unit draws its place in the range between gentle and steep.
      const shown = deg !== null ? Math.min(85, Math.max(1, deg)) : 6 + f * 54;
      const cap = capProfile(shown, { cx, ground, a });
      parts.push(svgEl("line", { class: "hd-ground", x1: 0, x2: W, y1: ground, y2: ground }));
      if (cap) parts.push(svgEl("path", { class: "hd-hill", d: cap.path }));
      parts.push(svgEl("rect", { class: "hd-dog", x: 6, y: ground - 7, width: 16, height: 6, rx: 2 }));
    } else if (h.kind === "offset") {
      const mid = H / 2, x = 20 + f * (W - 40);
      parts.push(svgEl("line", { class: "hd-path", x1: x < W / 2 ? 0 : 0, x2: W, y1: mid, y2: mid }));
      parts.push(svgEl("rect", { class: "hd-dog", x: 4, y: mid - 4, width: 14, height: 8, rx: 2 }));
      // the obstacle seen from above, shifted across the path in proportion to the offset
      const y = H - 6 - f * (H - 12);
      parts.push(svgEl("circle", { class: "hd-hill", cx: W * 0.62, cy: y, r: 13 }));
      parts.push(svgEl("circle", { class: "hd-knob", cx: W * 0.62, cy: y, r: 3 }));
    } else if (h.kind === "direction") {
      // Seen from above: the dog at the bottom facing up, the arc of every direction the range allows, and the way
      // this value comes in. An angle unit draws the true angle; any other unit spreads the range over a half turn.
      const cx = W / 2, cy = H - 4, R = H - 12;
      const toAng = (x) => { const d = angleDeg(h, x); return ((d !== null ? d : -90 + fraction(h, x) * 180) * Math.PI) / 180; };
      const pt = (a, r) => [cx + Math.sin(a) * r * 1.6, cy - Math.cos(a) * r];
      const [a0, a1, a] = [toAng(h.min), toAng(h.max), toAng(v)];
      const p0 = pt(a0, R), p1 = pt(a1, R), p = pt(a, R), q = pt(a, 10);
      parts.push(svgEl("path", { class: "hd-path", d: `M${p0[0].toFixed(1)} ${p0[1].toFixed(1)} A ${(R * 1.6).toFixed(1)} ${R} 0 0 1 ${p1[0].toFixed(1)} ${p1[1].toFixed(1)}` }));
      parts.push(svgEl("rect", { class: "hd-dog", x: cx - 4, y: cy - 8, width: 8, height: 12, rx: 2 }));
      parts.push(svgEl("path", { class: "hd-arrow", d: `M${p[0].toFixed(1)} ${p[1].toFixed(1)} L${q[0].toFixed(1)} ${q[1].toFixed(1)}` }));
      parts.push(svgEl("circle", { class: "hd-knob", cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: 5 }));
    } else {
      const y = H / 2 - 4;
      parts.push(svgEl("rect", { class: "hd-track", x: 0, y, width: W, height: 8, rx: 4 }));
      parts.push(svgEl("rect", { class: "hd-fill", x: 0, y, width: Math.max(8, f * W), height: 8, rx: 4 }));
      parts.push(svgEl("circle", { class: "hd-knob", cx: Math.max(6, f * W), cy: y + 4, r: 7 }));
    }
    svg.replaceChildren(...parts);
  }

  // ---- the switch, the main button and the line under the handles -------------------------------------------------
  function setSwitch(show, mode, simLabel) {
    $("mode-switch").hidden = !show;
    setText($("mode-sim"), simLabel);
    for (const b of document.querySelectorAll(".switch__opt")) b.setAttribute("aria-checked", String(b.dataset.mode === mode));
  }
  for (const b of document.querySelectorAll(".switch__opt")) b.addEventListener("click", () => onMode(b.dataset.mode));
  $("mode-switch").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    e.preventDefault();
    const other = document.querySelector('.switch__opt[aria-checked="false"]');
    other?.click(); other?.focus();
  });

  function setGo({ show, label, disabled, running }) {
    const b = $("go");
    b.hidden = !show;
    setText(b, label);
    b.disabled = disabled;
    b.dataset.running = String(Boolean(running));
  }
  $("go").addEventListener("click", () => onGo());
  $("cam-toggle").addEventListener("click", () => onCamera());

  const setLine = (text) => setText($("bar-line"), text ?? "");

  return { setTabs, setHandles, setSwitch, setGo, setLine };
}
