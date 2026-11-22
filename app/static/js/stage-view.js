// The full-window stage behind every screen: the real simulated dog (robot-stage.js), framed in the part of the window
// no panel covers; the caption of the episode on stage; and the small "Drive it yourself" pad.
import { initRobotStage } from "./robot-stage.js";
import { commandFields, driveCommand } from "./drive.js";
import { episodeCaption } from "./plain.js";
import { $, setText } from "./dom.js";

const KEYS = { w: "forward", s: "back", a: "left", d: "right", q: "turn_left", e: "turn_right", " ": "stop" };
const DOING = { forward: "Walking forward", back: "Walking back", left: "Stepping left", right: "Stepping right", turn_left: "Turning left",
  turn_right: "Turning right", stop: "Standing still" };
const SPEED = 0.5; // the pad's share of each published range

export function initStageView({ api, catalogue, notes, onEpisode }) {
  const stage = initRobotStage({ api });
  const narrow = matchMedia("(max-width: 759px)");
  const root = document.documentElement;

  // ---- framing: the dog sits in the middle of the free region, as a lens shift -------------------------------------
  function place() {
    const W = innerWidth, H = innerHeight;
    const bar = $("topbar").getBoundingClientRect();
    const panel = document.querySelector(".panel:not([hidden])")?.getBoundingClientRect() ?? null;
    const drawer = $("drawer").hidden ? null : $("drawer").getBoundingClientRect();
    let left = 0, right = W, top = bar.bottom, bottom = H;
    if (panel && panel.width) {
      if (narrow.matches) bottom = Math.min(bottom, panel.top);
      else left = Math.max(left, panel.right);
    }
    if (drawer && drawer.width < W) right = Math.min(right, drawer.left);
    if (right - left < 40 || bottom - top < 40) { left = 0; right = W; top = bar.bottom; bottom = H; } // covered: frame the whole window
    for (const [k, v] of Object.entries({ "--free-l": left, "--free-r": W - right, "--free-t": top, "--free-b": H - bottom })) root.style.setProperty(k, `${Math.round(v)}px`);
    // At full scale the dog spans about 0.68 x 0.57 of the window's height; a smaller free region draws it smaller.
    const k = Math.max(0.3, Math.min(1, (0.8 * (right - left)) / (0.68 * H), (0.8 * (bottom - top)) / (0.57 * H)));
    stage.setShift((2 * (left + right) / 2) / W - 1, 1 - (2 * (top + bottom) / 2) / H, k);
  }
  addEventListener("resize", place);
  narrow.addEventListener("change", place);
  const ro = new ResizeObserver(() => place());
  ro.observe($("topbar"));
  for (const p of document.querySelectorAll(".panel, #drawer")) ro.observe(p);

  // ---- the episode on stage ---------------------------------------------------------------------------------------
  let hideTimer = null, last = null;
  function onEpisodeEvent(ev) {
    const v = episodeCaption(ev, catalogue());
    const box = $("ep-caption");
    if (v.hidden) {
      clearTimeout(hideTimer);
      hideTimer = setTimeout(() => { box.hidden = true; last = null; onEpisode?.(null); }, ev?.state === "idle" ? 2500 : 0);
      return;
    }
    clearTimeout(hideTimer);
    last = v;
    box.hidden = false;
    box.dataset.phase = v.phase;
    setText($("ep-phase"), v.phaseText);
    setText($("ep-title"), v.title);
    setText($("ep-outcome"), v.outcome ? v.outcome : "");
    $("ep-outcome").hidden = !v.outcome;
    setText($("ep-detail"), v.detail);
    $("ep-info").hidden = !v.detail;
    onEpisode?.(v);
  }
  $("ep-info").addEventListener("click", () => {
    const open = $("ep-detail").hidden;
    $("ep-detail").hidden = !open;
    $("ep-info").setAttribute("aria-expanded", String(open));
  });
  function hideCaption() { clearTimeout(hideTimer); $("ep-caption").hidden = true; last = null; onEpisode?.(null); }

  // ---- drive it yourself ------------------------------------------------------------------------------------------
  let fields = null;
  const say = (text) => setText($("drive-status"), text);
  async function go(dir) {
    if (!fields) return say("The simulator has not said which moves it accepts yet.");
    const cmd = driveCommand(dir, fields, SPEED);
    if (!cmd) return say("The dog cannot move that way.");
    const r = dir === "stop" ? await api.stop() : await api.drive(cmd);
    if (!r.ok) { notes.set("drive", "danger", `Drive command refused: ${r.message} (${r.code})`, "Drive"); return say("The simulator did not take that move."); }
    notes.clear("drive");
    say(`${DOING[dir]}. It keeps going until you press another move.`);
  }
  for (const b of document.querySelectorAll("[data-dir]")) b.addEventListener("click", () => go(b.dataset.dir));
  $("drive-reset").addEventListener("click", async () => {
    const r = await api.reset();
    say(r.ok ? "A fresh start." : "The simulator did not restart.");
    if (!r.ok) notes.set("drive", "danger", `Reset refused: ${r.message} (${r.code})`, "Drive");
  });
  function setPad(open) {
    $("drive-pad").hidden = !open;
    $("drive-toggle").setAttribute("aria-expanded", String(open));
    setText($("drive-toggle"), open ? "Close the pad" : "Drive it yourself");
    if (open) $("drive-pad").querySelector('[data-dir="forward"]').focus({ preventScroll: true });
  }
  $("drive-toggle").addEventListener("click", () => setPad($("drive-pad").hidden));
  addEventListener("keydown", (e) => {
    if ($("drive-pad").hidden || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest?.("input, textarea, select, [contenteditable]")) return;
    if (e.key === "Escape") return setPad(false);
    const dir = KEYS[e.key.toLowerCase()];
    if (!dir) return;
    e.preventDefault();
    go(dir);
  });

  function setState(state, capabilities) {
    fields = commandFields(state);
    const drive = capabilities.find((c) => c.name === "drive");
    $("drive").hidden = !drive?.available || !fields; // driving that cannot work is left out
  }

  place();
  return {
    place,
    setState,
    onPose: (d) => stage.onPose(d),
    onScene: (d) => stage.onScene(d),
    onRobotChanged: () => stage.onRobotChanged(),
    onEpisode: onEpisodeEvent,
    hideCaption,
    caption: () => last,
    resetView: () => stage.resetView(),
  };
}
