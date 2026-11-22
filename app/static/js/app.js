// The page: four screens (Teach, Learn, Results, Try it live) over one full-window stage with the real simulated dog,
// and the engineers' drawer. Wires the pure modules to the DOM and the server; every server string goes in as text.
import { createApi } from "./api.js";
import { normaliseCapabilities } from "./capabilities.js";
import { cardsFrom } from "./flow.js";
import { mountNotes } from "./notes.js";
import { parseRoute, routeHash, SCREENS } from "./plain.js";
import { $, el } from "./dom.js";
import { initStageView } from "./stage-view.js";
import { initTeachView } from "./teach-view.js";
import { initLearnView } from "./learn-view.js";
import { initResultsView } from "./results-view.js";
import { initLiveView } from "./live-view.js";
import { initDrawerView } from "./drawer-view.js";

const api = createApi((url, init) => fetch(url, init));
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const notes = mountNotes($("drawer-notes"), el, { visible: 8 });

// ---- shared state ------------------------------------------------------------------------------------------------
let state = null, capabilities = [], catalogue = null, cards = [], openRun = null;
const bodies = new Map(); // run id -> Promise of GET /api/runs/{id} data (null when unreadable); read once per session

const ctx = {
  api, notes,
  state: () => state,
  capabilities: () => capabilities,
  catalogue: () => catalogue,
  cards: () => cards,
  cardOf: (id) => cards.find((c) => c.runId === id) ?? null,
  openRun: () => openRun,
  setOpenRun(id) { openRun = id; renderProgress(); },
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
  forgetRunBody(id) { bodies.delete(id); cached.delete(id); },
  go(screen, runId = null) { location.hash = routeHash(screen, runId); },
  runStarted(id) { openRun = id; loadCards(); },
  cardsMayHaveChanged: () => loadCards(),
  reloadState: () => loadState(),
  drawerChanged: () => drawer?.changed(),
  placeStage: () => stage.place(),
  stageCaption: () => stage.caption(),
  timeline: () => learn.timeline(),
  runLog: () => (learn.runId() === openRun ? learn.log() : []),
};
const cached = new Map();
let drawer = null; // made below, after the views it reads

// ---- the views ---------------------------------------------------------------------------------------------------
const stage = initStageView({ api, notes, catalogue: () => catalogue, onEpisode: (v) => { results.onEpisode(v); drawer?.changed(); } });
const teach = initTeachView(ctx);
const learn = initLearnView(ctx);
const results = initResultsView(ctx);
const live = initLiveView(ctx);
ctx.live = live;
drawer = initDrawerView(ctx);
const views = { teach, learn, results, live };

// ---- screens -----------------------------------------------------------------------------------------------------
let current = null, currentRun = null;
function route() {
  const r = parseRoute(location.hash);
  if (r.screen === current && r.runId === currentRun) return;
  const before = current;
  if (current) views[current].leave();
  current = r.screen;
  currentRun = r.runId;
  for (const s of SCREENS) {
    const panel = $(`screen-${s}`);
    const on = s === current;
    panel.hidden = !on;
    if (on && before && before !== s && !reduceMotion.matches) {
      panel.classList.remove("is-entering");
      void panel.offsetWidth; // restarts the entrance animation
      panel.classList.add("is-entering");
    }
  }
  document.body.dataset.screen = current;
  renderProgress();
  views[current].enter(r);
  stage.place();
  if (before) $(`screen-${current}`).querySelector("h1")?.focus({ preventScroll: true });
}
$("main").addEventListener("animationend", (e) => e.target.classList?.remove("is-entering"));
addEventListener("hashchange", route);

// The four-step progress control: the current step highlighted; Learn and Results carry the open run.
function renderProgress() {
  for (const a of document.querySelectorAll(".progress__step")) {
    const s = a.dataset.screen;
    a.setAttribute("href", routeHash(s, s === "learn" || s === "results" ? openRun : null));
    if (s === current) a.setAttribute("aria-current", "step");
    else a.removeAttribute("aria-current");
    a.dataset.done = String(SCREENS.indexOf(s) < SCREENS.indexOf(current));
  }
}

// ---- server state and data ---------------------------------------------------------------------------------------
async function loadState() {
  const r = await api.state();
  if (!r.ok) { notes.set("state", "danger", `GET /api/state: ${r.message} (${r.code})`, "Server"); return; }
  notes.clear("state");
  state = r.data;
  capabilities = normaliseCapabilities(state.capabilities);
  for (const c of capabilities) {
    if (c.available || !["chat", "runs", "infer", "drive"].includes(c.name)) notes.clear(`cap:${c.name}`);
    else notes.set(`cap:${c.name}`, "warn", `${c.name} unavailable: ${c.reason}`, "Capability");
  }
  stage.setState(state, capabilities);
  live.setState(state, capabilities);
  teach.stateChanged();
  results.stateChanged();
  drawer?.changed();
}

async function loadCatalogue() {
  const r = await api.skills();
  catalogue = r.ok && Array.isArray(r.data) ? r.data : null;
  if (catalogue) notes.clear("catalogue");
  else notes.set("catalogue", "danger", `GET /api/skills: ${r.ok ? "not a list" : `${r.message} (${r.code})`}`, "Server");
  teach.cardsChanged();
}

let cardsLoading = null;
function loadCards() {
  cardsLoading ??= (async () => {
    const r = await api.runCards();
    cardsLoading = null;
    if (!r.ok) { notes.set("cards", "danger", `GET /api/runs?summary=1: ${r.message} (${r.code})`, "Runs"); return; }
    notes.clear("cards");
    cards = cardsFrom(r.data);
    teach.cardsChanged();
  })();
  return cardsLoading;
}

// ---- events from the server --------------------------------------------------------------------------------------
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
    $("conn").textContent = source.readyState === EventSource.CLOSED ? "Lost the connection to the simulator. Reload the page." : "Lost the connection to the simulator. Trying again…";
  };
  const on = (name, handle) => source.addEventListener(name, (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; } // an unreadable event shows nothing
    handle(data);
  });
  on("pose", (d) => stage.onPose(d));
  on("scene", (d) => stage.onScene(d));
  on("robot", () => stage.onRobotChanged());
  on("run", (d) => { learn.onRunEvent(d); drawer?.changed(); });
  on("episode", (d) => {
    if (typeof d?.state === "string" && "run_id" in d) stage.onEpisode(d);
    if (d?.state === "idle" || d?.state === "start" && d?.index === 0) stateSoon();
  });
  on("notice", (d) => { if (typeof d?.message === "string") notes.set("notice", "info", d.message, "Simulator"); });
}

// ---- frosted glass only while the stage stays smooth -------------------------------------------------------------
// Blur behind the panels is recomposited on every 3D frame. If the page cannot hold 40 frames per second for two
// seconds while the stage is moving, the panels turn solid (and stay so); the drawer says why.
function watchFrameRate() {
  if (!CSS.supports("backdrop-filter", "blur(2px)") && !CSS.supports("-webkit-backdrop-filter", "blur(2px)")) { document.body.classList.add("glass-solid"); return; }
  let frames = 0, since = performance.now(), slow = 0;
  const tick = (t) => {
    frames++;
    if (t - since >= 1000) {
      const rate = (frames * 1000) / (t - since);
      slow = !document.hidden && rate < 40 ? slow + 1 : 0;
      frames = 0; since = t;
      if (slow >= 2) {
        document.body.classList.add("glass-solid");
        notes.set("glass", "info", `Frosted glass turned off: the page drew ${rate.toFixed(0)} frames per second with it.`, "Display");
        return;
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ---- start -------------------------------------------------------------------------------------------------------
connectEvents();
loadCatalogue();
loadCards();
route();
watchFrameRate();
