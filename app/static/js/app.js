// The page. Two views on one page, chosen by the address: the landing page at "/" (no hash), and the demo screen at
// #<demo id>: the live comparison of the demo on show (before | after, the same situation), what each dog senses, the
// learned lane's live brain readout, the controls and the story band of how it learned, all in view at once. Wires the
// pure modules to the DOM and the server; every server string goes in as text, and what the server does not send is
// left out.
import { createApi } from "./api.js";
import { normaliseCapabilities } from "./capabilities.js";
import { defaultParams, readDemos } from "./demos.js";
import { applyDuel, countRound, duelShows, emptyDuel, tallyText } from "./duel.js";
import { applyBrain, brainView, emptyBrain, laneResult } from "./showcase.js";
import { readWindowSettings } from "./vision.js";
import { $, setText } from "./dom.js";
import { initStageView } from "./stage-view.js";
import { initControls } from "./controls-view.js";
import { createSenses } from "./sense-view.js";
import { initLiveView } from "./live-view.js";
import { initHomeView } from "./home-view.js";
import { initBandView } from "./band-view.js";
import { initBrainView } from "./brain-view.js";

const api = createApi((url, init) => fetch(url, init));
const DEMOS_RETRY_MS = 20000; // while the server has no demo list, ask again this often

// Problems the audience does not need to read: to the browser console only, once per change of a standing condition.
// The page itself shows only the connection toast; a part that cannot show is left out.
const standing = new Map();
const say = (severity, text) => (severity === "info" ? console.info(text) : console.warn(text));
const notes = {
  set(key, severity, text) { if (standing.get(key) !== text) { standing.set(key, text); say(severity, text); } },
  clear(key) { standing.delete(key); },
  report: say,
};

// ---- shared state ----------------------------------------------------------------------------------------------------
let capabilities = [], eyeSettings = null;
let demosInfo = { ok: false, demos: [], unavailable: false };
let view = "home", current = null, mode = "sim", duel = emptyDuel(), starting = false;
let tally = {}; // this page session's rounds per demo and lane (duel.js countRound); a reload starts it again
let brain = emptyBrain(); // the learned lane's decisions of the round on show ("brain" events)
let warming = false; // a round has started and neither lane has drawn its dog yet
const params = new Map(); // demo id -> {handle name: value}
const wordOf = (outcome) => laneResult(outcome)?.word ?? null; // what the session tally counts for a lane

const demoOnShow = () => (demosInfo.ok ? demosInfo.demos.find((d) => d.id === current) ?? null : null);
const paramsOf = (demo) => { if (!params.has(demo.id)) params.set(demo.id, defaultParams(demo)); return params.get(demo.id); };

// ---- the views -----------------------------------------------------------------------------------------------------
const stage = initStageView({ api, notes });
const senses = createSenses({ before: $("sense-before"), after: $("sense-after") }, { eyeSettings: () => eyeSettings });
const live = initLiveView({ api, notes, setLine: (t) => controls.setLine(t) });
const controls = initControls({
  onParam(name, value) { const d = demoOnShow(); if (!d) return; paramsOf(d)[name] = value; apply(); },
  onMode(m) { if (m !== mode) { mode = m; if (m !== "camera") live.leave(); apply(); } },
  onGo: runBoth,
  onCamera: () => live.toggle(),
});
const home = initHomeView();
const band = initBandView();
const brainUi = initBrainView();

// ---- where the page is: the landing page, or a demo by its id in the hash -------------------------------------------
const hashId = () => { try { return decodeURIComponent(location.hash.replace(/^#/, "")); } catch { return ""; } };

function pickDemo() {
  const want = hashId();
  const list = demosInfo.ok ? demosInfo.demos : [];
  current = list.find((d) => d.id === want)?.id ?? list.find((d) => d.id === current)?.id ?? list[0]?.id ?? null;
}

function route() {
  const before = current, was = view;
  view = hashId() ? "demo" : "home";
  pickDemo();
  if (current !== before) { mode = "sim"; senses.clear(); brain = emptyBrain(); }
  apply();
  if (view !== was) { scrollTo(0, 0); stage.place(); }
}
addEventListener("hashchange", route);
addEventListener("popstate", route);

function goHome(e) {
  e?.preventDefault();
  if (location.hash) history.pushState(null, "", location.pathname + location.search);
  route();
  $("home-main").focus({ preventScroll: true });
}
for (const a of document.querySelectorAll("[data-home]")) a.addEventListener("click", goHome);
$("skip").addEventListener("click", (e) => { e.preventDefault(); (view === "home" ? $("home-main") : $("stage")).focus(); });

// ---- what the screen shows ------------------------------------------------------------------------------------------
function apply() {
  document.body.dataset.view = view;
  home.render(demosInfo.ok ? demosInfo.demos : []);
  if (view === "home") {
    stage.setLayout("home");
    live.leave();
    band.render(null);
    brainUi.show(null, false);
    return;
  }

  const demo = demoOnShow();
  const inferOk = capabilities.find((c) => c.name === "infer")?.available === true;
  let layout;
  if (demo) layout = mode === "camera" && demo.liveCamera ? "camera" : "duel";
  else layout = mode === "camera" && inferOk ? "camera" : "single";
  stage.setLayout(layout);
  document.body.dataset.mode = layout === "camera" ? "camera" : "sim";
  controls.setTabs(demosInfo.ok ? demosInfo.demos : [], current);
  senses.setKind(layout === "duel" ? demo?.sense ?? null : null);

  const camSwitch = demo ? demo.liveCamera : inferOk;
  controls.setSwitch(camSwitch, layout === "camera" ? "camera" : "sim", demo ? "Simulated punch" : "The simulator");
  if (layout === "camera") live.enter(); else live.leave();

  let running = false;
  if (layout === "duel") {
    const mine = duelShows(duel, demo.id);
    running = mine && duel.state === "start";
    const otherRunning = duel.state === "start" && !mine;
    stage.setSideView(demo.sense === "eye");
    stage.setLaneNames(demo.lanes);
    controls.setHandles(demo, paramsOf(demo), running || starting);
    controls.setGo({ show: true, label: starting ? "Get ready" : running ? "Stop" : "Run both", disabled: !demo.available || starting || otherRunning, running });
    for (const side of ["before", "after"]) {
      stage.setOutcome(side, demo.outcomeFields, mine ? duel.sides[side] : null);
      stage.setTally(side, tallyText(tally[demo.id]?.[side]));
    }
    // Before a round: the demo's title and sentence. After Run both, until the first pose of either dog arrives (the
    // server builds both worlds first), a short line instead of two empty floors.
    const invite = $("invite");
    warming = running && !stage.hasPose("before") && !stage.hasPose("after");
    invite.hidden = warming ? false : mine || starting || !(demo.tagline || demo.title);
    invite.dataset.state = warming ? "warming" : "idle";
    setText($("invite-title"), warming ? "" : demo.title ?? "");
    setText($("invite-text"), warming ? "Getting both dogs ready" : demo.tagline ?? "");
    $("invite-title").hidden = warming || !demo.title;
    $("invite-text").hidden = !warming && !demo.tagline;
    if (!demo.available) {
      notes.set(`demo:${demo.id}`, "warn", `${demo.title}: ${demo.missing.join("; ") || "not available"}`);
      controls.setLine("This demo is not ready on this server yet.");
    } else if (otherRunning) controls.setLine("The other demo is running now. This one can run when it ends.");
    else controls.setLine("");
    brainUi.show(mine ? brainView(brain) : null, running);
  } else {
    controls.setHandles(null, {}, true);
    controls.setGo({ show: false, label: "Run both", disabled: true });
    $("invite").hidden = true;
    warming = false;
    brainUi.show(null, false);
    if (layout === "single") {
      controls.setLine(demosInfo.unavailable
        ? "The side-by-side comparison is being connected. Until then, this is the simulator as it runs now."
        : demosInfo.ok ? "" : "The demos did not load. This is the simulator as it runs now.");
    }
  }
  setText($("name-single"), layout === "camera" ? "The dog, reacting to your camera" : "The simulator, live");
  band.render(demo, { running, counts: demo ? tally[demo.id] : null });
  stage.place();
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
  brain = emptyBrain();
  apply();
  await stage.countIn();
  const r = await api.duelStart(demo.id, { ...paramsOf(demo) }, null);
  starting = false;
  if (!r.ok) {
    const why = r.status === 409 ? r.message : r.unavailable ? "Running both lanes is not connected on this server yet." : `Both lanes did not start: ${r.message}`;
    notes.set("duel", r.unavailable ? "warn" : "danger", `POST /api/duel/start: ${r.message} (${r.code})`);
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
  if (!r.ok) { notes.set("state", "danger", `GET /api/state: ${r.message} (${r.code})`); return; }
  notes.clear("state");
  const state = r.data;
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
    demosInfo = { ok: d.ok, demos: d.demos, unavailable: false };
    for (const p of d.problems) notes.set(`demos:${p}`, "warn", p);
    notes.clear("demos");
  } else {
    demosInfo = { ok: false, demos: [], unavailable: r.unavailable };
    notes.set("demos", r.unavailable ? "info" : "danger", `GET /api/demos: ${r.message} (${r.code})`);
    demosTimer = setTimeout(loadDemos, DEMOS_RETRY_MS);
  }
  pickDemo();
  apply();
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
  const on = (name, handle) => source.addEventListener(name, (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; } // an unreadable event shows nothing
    handle(data);
  });
  on("pose", (d) => { stage.onPose(d); if (warming && (stage.hasPose("before") || stage.hasPose("after"))) apply(); });
  on("scene", (d) => stage.onScene(d));
  on("robot", () => stage.onRobotChanged());
  on("feel", (d) => senses.onFeel(d));
  on("eye", (d) => senses.onEye(d));
  on("duel", (d) => {
    const was = duel;
    duel = applyDuel(duel, d);
    tally = countRound(tally, was, duel, wordOf);
    if (d?.state === "start" && was.state !== "start" && !starting) { senses.clear(); stage.clearLanes(); brain = emptyBrain(); }
    apply();
  });
  // The learned lane's decisions: only the readout redraws, at most once a frame.
  on("brain", (d) => {
    brain = applyBrain(brain, d);
    if (view === "demo" && stage.layout() === "duel" && duelShows(duel, current)) brainUi.show(brainView(brain), duel.state === "start");
  });
  // The episode player (started and ended elsewhere) may change what the simulator offers.
  on("episode", (d) => { if (d?.state === "idle" || (d?.state === "start" && d?.index === 0)) stateSoon(); });
  on("notice", (d) => { if (typeof d?.message === "string") notes.set("notice", "info", d.message); });
}

// ---- frosted glass only while the stage stays smooth ---------------------------------------------------------------
// The blur behind the glass is recomposited on every 3D frame. If the page cannot hold 40 frames per second for two
// seconds, the glass turns solid (and stays so).
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
        notes.set("glass", "info", `Frosted glass turned solid: the page drew ${fps.toFixed(0)} frames per second with it.`);
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
route();
watchFrameRate();
