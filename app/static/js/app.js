// The page: one screen that opens on the live comparison of the demo on show (the walking brain alone | with the learned
// reflex), what each dog senses under its lane, the glass control bar, "How it learned" and the engineers' drawer.
// Wires the pure modules to the DOM and the server; every server string goes in as text, and what the server cannot
// give yet is left out or said plainly.
import { createApi } from "./api.js";
import { normaliseCapabilities } from "./capabilities.js";
import { cardsFrom } from "./flow.js";
import { mountNotes } from "./notes.js";
import { defaultParams, headlineSentence, readDemos } from "./demos.js";
import { applyDuel, duelShows, emptyDuel } from "./duel.js";
import { readWindowSettings } from "./vision.js";
import { $, el, setText } from "./dom.js";
import { initStageView } from "./stage-view.js";
import { initControls } from "./controls-view.js";
import { createSenses } from "./sense-view.js";
import { initLiveView } from "./live-view.js";
import { initSheetView } from "./sheet-view.js";
import { initDrawerView } from "./drawer-view.js";

const api = createApi((url, init) => fetch(url, init));
const notes = mountNotes($("drawer-notes"), el, { visible: 8 });
const DEMOS_RETRY_MS = 20000; // while the server has no demo list, ask again this often

// ---- shared state ----------------------------------------------------------------------------------------------------
let state = null, capabilities = [], cards = [], eyeSettings = null;
let demosInfo = { ok: false, raw: null, demos: [], problems: [], message: null, unavailable: false };
let current = null, mode = "sim", duel = emptyDuel(), starting = false;
const params = new Map(); // demo id -> {handle name: value}
const bodies = new Map(), cached = new Map(), runLines = new Map();

const ctx = {
  api, notes,
  state: () => state,
  capabilities: () => capabilities,
  cards: () => cards,
  demo: () => demoOnShow(),
  duel: () => duel,
  demosInfo: () => demosInfo,
  senseCounts: () => senses.counts(),
  runBody(id) {
    if (!bodies.has(id)) {
      bodies.set(id, api.run(id).then((r) => {
        if (r.ok) { cached.set(id, r.data); return r.data; }
        notes.set(`run:${id}`, "danger", `GET /api/runs/${id}: ${r.message} (${r.code})`, "Runs");
        bodies.delete(id);
        return null;
      }));
    }
    return bodies.get(id);
  },
  cachedRunBody: (id) => cached.get(id) ?? null,
  runLog: () => runLines.get(sheet?.stats().lesson) ?? [],
  placeStage: () => stage.place(),
  changed: () => drawer?.changed(),
  enterClass() { live.leave(); apply(); },
  leaveClass() { loadDemos(); loadCards(); apply(); },
};
let drawer = null, sheet = null;

const demoOnShow = () => (demosInfo.ok ? demosInfo.demos.find((d) => d.id === current) ?? null : null);
const paramsOf = (demo) => { if (!params.has(demo.id)) params.set(demo.id, defaultParams(demo)); return params.get(demo.id); };

// ---- the views -----------------------------------------------------------------------------------------------------
const stage = initStageView({ api, notes });
const senses = createSenses({ before: $("sense-before"), after: $("sense-after") }, { eyeSettings: () => eyeSettings });
const live = initLiveView({ api, notes, setLine: (t) => controls.setLine(t), changed: () => drawer?.changed() });
ctx.live = live;
const controls = initControls({
  onTab(id) { if (id !== current) { location.hash = id; } },
  onParam(name, value) { const d = demoOnShow(); if (!d) return; paramsOf(d)[name] = value; apply(); },
  onMode(m) { if (m !== mode) { mode = m; if (m !== "camera") live.leave(); apply(); } },
  onGo: runBoth,
  onCamera: () => live.toggle(),
});
sheet = initSheetView(ctx);
ctx.sheet = sheet;
drawer = initDrawerView(ctx);

// ---- what the screen shows ------------------------------------------------------------------------------------------
function apply() {
  const demo = demoOnShow();
  const inferOk = capabilities.find((c) => c.name === "infer")?.available === true;
  let layout;
  if (sheet.inClass()) layout = "class";
  else if (demo) layout = mode === "camera" && demo.liveCamera ? "camera" : "duel";
  else layout = mode === "camera" && inferOk ? "camera" : "single";
  stage.setLayout(layout);
  document.body.dataset.mode = layout === "camera" ? "camera" : "sim";
  controls.setTabs(demosInfo.ok ? demosInfo.demos : [], current);
  senses.setKind(layout === "duel" ? demo?.sense ?? null : null);

  const camSwitch = layout !== "class" && (demo ? demo.liveCamera : inferOk);
  controls.setSwitch(camSwitch, layout === "camera" ? "camera" : "sim", demo ? "Simulated punch" : "The simulator");
  if (layout === "camera") live.enter(); else live.leave();

  if (layout === "duel") {
    const mine = duelShows(duel, demo.id), running = mine && duel.state === "start";
    const otherRunning = duel.state === "start" && !mine;
    controls.setHandles(demo, paramsOf(demo), running || starting);
    controls.setGo({ show: true, label: starting ? "Get ready" : running ? "Stop" : "Run both", disabled: !demo.available || starting || otherRunning, running });
    for (const side of ["before", "after"]) stage.setOutcome(side, demo.outcomeFields, mine ? duel.sides[side] : null);
    const invite = $("invite");
    invite.hidden = mine || starting || !demo.tagline;
    setText(invite, demo.tagline ?? "");
    if (!demo.available) controls.setLine(`This demo is not ready on this server yet${demo.missing.length ? `: ${demo.missing.join("; ")}` : "."}`);
    else if (otherRunning) controls.setLine("Another demo is running both lanes now; this one can run when it ends.");
    else controls.setLine(headlineSentence(demo.headline) ?? "");
  } else {
    controls.setHandles(null, {}, true);
    controls.setGo({ show: false, label: "Run both", disabled: true });
    $("invite").hidden = true;
    if (layout === "single") {
      controls.setLine(demosInfo.unavailable
        ? "The side-by-side comparison is being connected. Until then, this is the simulator as it runs now."
        : demosInfo.message ?? "");
    }
    if (layout === "class") controls.setLine("The lesson's real tries, as they happen in the simulator.");
  }
  setText($("name-single"), layout === "class" ? "The lesson's tries" : layout === "camera" ? "The dog, reacting to your camera" : "The simulator, live");
  stage.place();
  sheet.render();
  drawer?.changed();
}

async function runBoth() {
  const demo = demoOnShow();
  if (!demo) return;
  if (duelShows(duel, demo.id) && duel.state === "start") {
    const r = await api.duelStop();
    if (!r.ok) controls.setLine(`The run did not stop: ${r.message}`);
    return;
  }
  starting = true;
  senses.clear();
  stage.clearLanes();
  apply();
  await stage.countIn();
  const r = await api.duelStart(demo.id, { ...paramsOf(demo) }, null);
  starting = false;
  if (!r.ok) {
    const why = r.status === 409 ? r.message : r.unavailable ? "Running both lanes is not connected on this server yet." : `Both lanes did not start: ${r.message}`;
    notes.set("duel", r.unavailable ? "warn" : "danger", `POST /api/duel/start: ${r.message} (${r.code})`, "Comparison");
    apply();
    controls.setLine(why);
    return;
  }
  notes.clear("duel");
  if (Number.isInteger(r.data?.seed)) duel = { ...duel, seed: r.data.seed };
  apply();
}

// ---- server state and data ------------------------------------------------------------------------------------------
async function loadState() {
  const r = await api.state();
  if (!r.ok) { notes.set("state", "danger", `GET /api/state: ${r.message} (${r.code})`, "Server"); return; }
  notes.clear("state");
  state = r.data;
  capabilities = normaliseCapabilities(state.capabilities);
  const w = readWindowSettings(state.image_window);
  eyeSettings = w.ok ? w.settings : null;
  live.setState(state, capabilities);
  apply();
}

let demosTimer = null;
async function loadDemos() {
  clearTimeout(demosTimer);
  const r = await api.demos();
  if (r.ok) {
    const d = readDemos(r.data);
    demosInfo = { ok: d.ok, raw: r.data, demos: d.demos, problems: d.problems, message: d.ok ? null : d.problems.join(" "), unavailable: false };
    for (const p of d.problems) notes.set(`demos:${p}`, "warn", p, "Demos");
    notes.clear("demos");
  } else {
    demosInfo = { ok: false, raw: null, demos: [], problems: [], message: r.unavailable ? null : `The demo list did not load: ${r.message}`, unavailable: r.unavailable };
    notes.set("demos", r.unavailable ? "info" : "danger", `GET /api/demos: ${r.message} (${r.code})`, "Demos");
    demosTimer = setTimeout(loadDemos, DEMOS_RETRY_MS);
  }
  pickDemo();
  apply();
  sheet.refresh();
}

function pickDemo() {
  const want = decodeURIComponent(location.hash.replace(/^#/, ""));
  const list = demosInfo.ok ? demosInfo.demos : [];
  current = list.find((d) => d.id === want)?.id ?? list.find((d) => d.id === current)?.id ?? list[0]?.id ?? null;
}
addEventListener("hashchange", () => { const before = current; pickDemo(); if (current !== before) { mode = "sim"; senses.clear(); } apply(); sheet.refresh(); });

let cardsLoading = null;
function loadCards() {
  cardsLoading ??= (async () => {
    const r = await api.runCards();
    cardsLoading = null;
    if (!r.ok) { notes.set("cards", "danger", `GET /api/runs?summary=1: ${r.message} (${r.code})`, "Runs"); return; }
    notes.clear("cards");
    cards = cardsFrom(r.data);
    sheet.refresh();
    apply();
  })();
  return cardsLoading;
}

// ---- events from the server ------------------------------------------------------------------------------------------
let reconnecting = false, stateTimer = null;
const stateSoon = () => { clearTimeout(stateTimer); stateTimer = setTimeout(loadState, 300); };
function connectEvents() {
  const source = new EventSource("/api/events");
  source.onopen = () => {
    $("conn").hidden = true;
    if (reconnecting) stage.onRobotChanged(); // the world may have been rebuilt while the connection was down
    reconnecting = true;
    loadState();
  };
  source.onerror = () => {
    $("conn").hidden = false;
    setText($("conn"), source.readyState === EventSource.CLOSED ? "Lost the connection to the simulator. Reload the page." : "Lost the connection to the simulator. Trying again.");
  };
  const on = (name, handle, keep = true) => source.addEventListener(name, (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; } // an unreadable event shows nothing
    handle(data);
    if (keep) drawer.raw(name, data);
  });
  on("pose", (d) => stage.onPose(d), false);
  on("scene", (d) => stage.onScene(d));
  on("robot", () => stage.onRobotChanged());
  on("feel", (d) => senses.onFeel(d), false);
  on("eye", (d) => senses.onEye(d), false);
  on("duel", (d) => {
    const was = duel;
    duel = applyDuel(duel, d);
    if (d?.state === "start" && was.state !== "start" && !starting) { senses.clear(); stage.clearLanes(); }
    apply();
  });
  on("run", (d) => {
    if (typeof d?.run_id === "string") { const l = runLines.get(d.run_id) ?? []; l.push(d); runLines.set(d.run_id, l); }
    sheet.onRunEvent(d);
    drawer.changed();
  });
  on("episode", (d) => {
    sheet.onEpisode(d);
    if (d?.state === "idle" || (d?.state === "start" && d?.index === 0)) stateSoon();
  });
  on("infer", () => {});
  on("notice", (d) => { if (typeof d?.message === "string") notes.set("notice", "info", d.message, "Simulator"); });
}

// ---- frosted glass only while the stage stays smooth ---------------------------------------------------------------
// The blur behind the glass is recomposited on every 3D frame. If the page cannot hold 40 frames per second for two
// seconds, the glass turns solid navy (and stays so); the drawer says why.
function watchFrameRate() {
  if (!CSS.supports("backdrop-filter", "blur(2px)") && !CSS.supports("-webkit-backdrop-filter", "blur(2px)")) { document.body.classList.add("glass-solid"); return; }
  let frames = 0, since = performance.now(), slow = 0;
  const tick = (t) => {
    frames++;
    if (t - since >= 1000) {
      const fps = (frames * 1000) / (t - since);
      slow = !document.hidden && fps < 40 ? slow + 1 : 0;
      frames = 0; since = t;
      if (slow >= 2) {
        document.body.classList.add("glass-solid");
        notes.set("glass", "info", `Frosted glass turned solid: the page drew ${fps.toFixed(0)} frames per second with it.`, "Display");
        return;
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ---- start ---------------------------------------------------------------------------------------------------------
connectEvents();
loadDemos();
loadCards();
apply();
watchFrameRate();
