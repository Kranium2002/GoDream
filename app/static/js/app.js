// The page: wires the pure modules to the DOM. Every server string goes in as text (el() never parses HTML).
import { createApi } from "./api.js";
import { normaliseCapabilities } from "./capabilities.js";
import { attachProblems, catalogueRows, findSkill, isValid, specRows } from "./spec.js";
import { canStart, INITIAL_RUN, runStep } from "./run.js";
import { initLiveView } from "./live-view.js";
import { initLearnView } from "./learn-view.js";
import { initRobotStage } from "./robot-stage.js";
import { initDriveView } from "./drive-view.js";

const api = createApi((url, init) => fetch(url, init));
const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c !== null && c !== undefined) node.append(c); // strings become text nodes
  return node;
}

const CAP_LABELS = { drive: "Drive the dog", head_camera: "Head camera", scenarios: "Scenarios", chat: "Chat",
  runs: "Learning runs", results: "Results", infer: "Live inference" };

const live = initLiveView({ api, el });
const learn = initLearnView({ api, el, catalogue: () => catalogue });

// ---- the robot stage and the sections below it --------------------------------------------------------------
const stage = initRobotStage({ api });
const wide = matchMedia("(min-width: 760px)"); // copy beside the dog; below this the dog sits above the copy
// The stage fills the screen under the header, whatever height the header wraps to.
new ResizeObserver(() => document.documentElement.style.setProperty("--bar-h", `${document.querySelector(".bar").offsetHeight}px`)).observe(document.querySelector(".bar"));
const drive = initDriveView({ api, onOpen: () => {
  scrollTo({ top: 0, behavior: "auto" }); // instant: a smooth scroll is cancelled by the focus that follows
  placeDog();
}, onClose: () => placeDog() });

// Streams load only while their section is on screen; the camera itself is the person's to start and stop.
const watched = { robot: (v) => stage.setVisible(v), learn: (v) => learn.setVisible(v), live: (v) => live.setVisible(v) };
const io = new IntersectionObserver((entries) => {
  for (const e of entries) watched[e.target.id]?.(e.isIntersecting);
}, { threshold: 0.05 });
for (const id of Object.keys(watched)) io.observe($(id));

// "Teach it anything" fades the chat in over the scene; the dog moves aside so both stay visible.
function setDock(open) {
  const dock = $("dock");
  $("teach-open").setAttribute("aria-expanded", String(open));
  document.body.classList.toggle("dock-open", open);
  if (open) {
    dock.hidden = false;
    requestAnimationFrame(() => dock.classList.add("is-in"));
    $("msg").focus({ preventScroll: true }); // the page stays where it is; the dock scrolls on its own
  } else {
    dock.classList.remove("is-in");
    dock.hidden = true;
    $("teach-open").focus({ preventScroll: true });
  }
  placeDog();
}
// The dog is centred in the part of the stage nothing covers: between the copy (or the drive pad, or the dock) and the
// edge, above the bottom bar and the track; on narrow screens below the copy. Measured from the layout, as a lens shift.
function placeDog() {
  const s = $("robot").getBoundingClientRect(); // the canvas covers the whole stage
  if (!s.width || !s.height) return;
  const main = document.querySelector(".stage-main").getBoundingClientRect();
  const bar = document.querySelector(".stage-bar").getBoundingClientRect();
  const dockOpen = !$("dock").hidden, side = $(document.body.classList.contains("drive-open") ? "drive" : "hero-copy").getBoundingClientRect();
  let left = s.left, right = s.right, top = main.top, bottom = bar.top;
  if (wide.matches) {
    if (dockOpen) right = $("dock").getBoundingClientRect().left;
    else left = side.right;
  } else if (!dockOpen) {
    if (document.body.classList.contains("drive-open")) top = side.bottom;
    else bottom = side.top; // the copy sits at the bottom on narrow screens
  }
  // At full scale the dog spans about 0.85 x 0.57 of the stage's height (its three-quarter view, measured by eye); a smaller free region
  // scales it down so it fits with a margin, in width and in height.
  const k = Math.max(0.3, Math.min(1, 0.85 * (right - left) / (0.85 * s.height), 0.85 * (bottom - top) / (0.57 * s.height)));
  stage.setShift(2 * ((left + right) / 2 - s.left) / s.width - 1, 1 - 2 * ((top + bottom) / 2 - s.top) / s.height, k);
}
addEventListener("resize", placeDog);
wide.addEventListener("change", placeDog);
new ResizeObserver(placeDog).observe(document.querySelector(".stage-main")); // the track opening, the copy collapsing
placeDog();
$("teach-open").addEventListener("click", () => setDock(true));
$("zoom-in").addEventListener("click", () => stage.zoomBy(0.85));
$("zoom-out").addEventListener("click", () => stage.zoomBy(1 / 0.85));
$("view-reset").addEventListener("click", () => stage.resetView());
$("teach-close").addEventListener("click", () => setDock(false));
$("dock").addEventListener("keydown", (e) => { if (e.key === "Escape") setDock(false); });

// ---- connection, state, capabilities ---------------------------------------------------------------------------
let capabilities = [];

function setConnection(good, text) {
  const mark = $("conn-mark");
  mark.textContent = good ? "●" : "○";
  mark.className = `mark ${good ? "is-good" : "is-bad"}`;
  $("conn-text").textContent = text;
}

async function loadState() {
  const r = await api.state();
  if (!r.ok) {
    setConnection(false, `Server state not received: ${r.message}`);
    return;
  }
  const s = r.data;
  const parts = [];
  if (typeof s.status === "string") parts.push(`Simulator: ${s.status}`);
  if (typeof s.preset === "string") parts.push(`World: ${s.preset}`);
  $("sim-text").textContent = parts.join(" · ");
  capabilities = normaliseCapabilities(s.capabilities);
  renderCapabilities();
  live.setState(s, capabilities);
  drive.setState(s, capabilities);
  learn.setState(s, capabilities);
}

function renderCapabilities() {
  $("caps-empty").hidden = true;
  $("caps-count").textContent = `${capabilities.filter((c) => c.available).length} of ${capabilities.length} available`;
  $("caps").replaceChildren(...capabilities.map((c) => el("li", { class: c.available ? "is-good" : "is-bad" },
    el("span", { class: "mark", "aria-hidden": "true" }, c.available ? "✓" : "✕"),
    el("span", {}, el("span", { class: "caps__name" }, CAP_LABELS[c.name] ?? c.name), " ",
      el("span", { class: "caps__state" }, c.available ? "Available" : "Unavailable")),
    c.reason ? el("span", { class: "caps__reason" }, c.reason) : null)));
  for (const [name, id] of [["chat", "chat-cap"], ["runs", "runs-cap"]]) {
    const c = capabilities.find((x) => x.name === name);
    $(id).hidden = !c || c.available;
    if (c && !c.available) $(id).textContent = name === "runs" ? `Starting a new run is not available yet: ${c.reason}` : `${CAP_LABELS[name]} unavailable: ${c.reason}`;
  }
}

let reconnecting = false;
function connectEvents() {
  const source = new EventSource("/api/events");
  source.onopen = () => {
    setConnection(true, "Server connected");
    if (reconnecting) stage.onRobotChanged(); // the world may have been rebuilt while the connection was down
    reconnecting = true;
    loadState(); // after a reconnect too: anything may have changed while the connection was down
  };
  source.onerror = () => {
    setConnection(false, source.readyState === EventSource.CLOSED
      ? "Server connection lost. Reload the page to reconnect."
      : "Server connection lost. Trying to reconnect.");
  };
  source.addEventListener("episode", (e) => {
    try { if (JSON.parse(e.data)?.phase === "start") $("notice").hidden = true; } catch { /* keep the notice */ }
    loadState();
  });
  const forward = (name, handle) => source.addEventListener(name, (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; } // an unreadable event shows nothing
    handle(data);
  });
  forward("run", (d) => learn.onRunEvent(d));
  forward("pose", (d) => stage.onPose(d));
  forward("robot", () => stage.onRobotChanged());
  forward("observation", (d) => learn.onObservation(d));
  forward("infer", (d) => learn.onInfer(d));
  source.addEventListener("notice", (e) => {
    let data = null;
    try { data = JSON.parse(e.data); } catch { /* not shown: nothing readable came */ }
    if (typeof data?.message === "string") {
      $("notice").hidden = false;
      $("notice").textContent = data.message;
    }
  });
}

// ---- skill catalogue -------------------------------------------------------------------------------------------
let catalogue = null;
async function loadCatalogue() {
  const r = await api.skills();
  catalogue = r.ok && Array.isArray(r.data) ? r.data : null;
  const problem = !r.ok ? r.message : catalogue ? "" : "The server's answer was not a list of skills.";
  $("catalogue-note").hidden = !problem;
  $("catalogue-note").textContent = problem
    ? `${r.unavailable ? "The skill catalogue is unavailable" : "The skill catalogue could not be read"}: ${problem} Supported ranges cannot be shown.`
    : "";
  learn.catalogueChanged();
  const rows = catalogueRows(catalogue);
  $("skills-empty").hidden = rows.length > 0;
  $("skills-empty").textContent = catalogue ? "The catalogue lists no skills." : "Not known: the skill catalogue could not be read.";
  $("skills").replaceChildren(...rows.map((s) => el("li", { class: s.available ? "is-good" : "is-bad" },
    el("span", { class: "mark", "aria-hidden": "true" }, s.available ? "✓" : "✕"),
    el("span", {}, el("span", { class: "caps__name" }, s.skill), " ",
      el("span", { class: "caps__state" }, s.available ? "Supported" : "Not supported yet")),
    s.description ? el("span", { class: "caps__reason" }, s.description) : null,
    s.missing.length ? el("span", { class: "caps__reason" }, `Missing: ${s.missing.join("; ")}`) : null)));
}

// ---- chat ------------------------------------------------------------------------------------------------------
let messages = []; // the running list sent to /api/chat: only exchanges the server answered
let lastChat = null;

function logEntry(who, text, cls) {
  $("log-empty").hidden = true;
  const item = el("li", { class: cls }, el("span", { class: "log__who" }, who), el("span", { class: "log__text" }, text));
  $("log").append(item);
  $("dock-scroll").scrollTop = item.offsetTop - 8; // the newest message at the top of the dock's scrolling part
}

async function send() {
  const box = $("msg");
  const text = box.value.trim();
  if (!text || run.phase === "starting") return;
  const user = { role: "user", content: text };
  logEntry("You", text, "from-user");
  box.value = "";
  setBusy(true);
  const r = await api.chat([...messages, user]);
  setBusy(false);
  if (!r.ok) {
    logEntry(r.unavailable ? "Chat unavailable" : "Not answered", `${r.message} Your message was not kept; it is back in the box.`, "is-error");
    box.value = text;
    box.focus();
    return;
  }
  const reply = typeof r.data.reply === "string" ? r.data.reply : "";
  messages = [...messages, user, { role: "assistant", content: reply }];
  logEntry("Model", reply || "The server sent no reply text.", "from-model");
  lastChat = r.data;
  if (r.data.declared_spec && !catalogue) await loadCatalogue();
  renderSpec();
  dispatch({ type: "spec", valid: isValid(r.data, catalogue) });
}

function setBusy(busy) {
  $("send").disabled = busy;
  $("ask").setAttribute("aria-busy", String(busy));
  $("send").textContent = busy ? "Sending" : "Send";
}

$("ask").addEventListener("submit", (e) => { e.preventDefault(); send(); });
$("msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});

// ---- declared skill --------------------------------------------------------------------------------------------
function setVerdict(good, text) {
  const v = $("verdict");
  v.className = `verdict ${good === null ? "" : good ? "is-good" : "is-bad"}`;
  v.replaceChildren(el("span", { class: "mark", "aria-hidden": "true" }, good === null ? "○" : good ? "✓" : "✕"), " ", el("span", {}, text));
}

function renderSpec() {
  const c = lastChat;
  const problems = Array.isArray(c.problems) ? c.problems : [];
  const missing = Array.isArray(c.missing) ? c.missing : [];
  const decl = c.declared_spec && typeof c.declared_spec === "object" ? c.declared_spec : null;
  const entry = decl ? findSkill(catalogue, decl.skill) : null;
  const unsupported = entry?.available === false ? catalogueRows([entry])[0] : null;
  const attached = decl ? attachProblems(specRows(decl, catalogue), problems) : { rows: [], general: problems };
  const body = [];

  if (attached.general.length) { // problems whose field names no row go first
    body.push(el("div", {}, el("p", { class: "block-title" }, "Problems"),
      el("ul", { class: "problems" }, ...attached.general.map((p) => el("li", { class: "problem" }, problemText(p))))));
  }
  if (missing.length) {
    body.push(el("div", {}, el("p", { class: "block-title" }, "Missing capabilities"),
      el("ul", { class: "missing" }, ...missing.map((m) => el("li", {}, typeof m === "string" ? m : JSON.stringify(m))))));
  }
  if (unsupported) {
    body.push(el("div", {}, el("p", { class: "block-title" }, `${unsupported.skill} is not supported yet. Missing:`),
      el("ul", { class: "missing" }, ...unsupported.missing.map((m) => el("li", {}, m)))));
  }
  if (decl) body.push(specTable(attached.rows));
  else if (!missing.length) body.push(el("p", { class: "empty" }, "The server did not declare a skill for this request."));
  $("spec-body").replaceChildren(...body);

  if (isValid(c, catalogue)) setVerdict(true, "Valid: ready to start a learning run");
  else if (missing.length) setVerdict(false, `Not supported: ${missing.length} missing ${missing.length === 1 ? "capability" : "capabilities"}`);
  else if (unsupported) setVerdict(false, `Not supported yet: ${unsupported.skill}`);
  else if (problems.length) setVerdict(false, `Not valid: ${problems.length} ${problems.length === 1 ? "problem" : "problems"}`);
  else setVerdict(null, "No skill declared");
}

function problemText(p) {
  const field = typeof p?.field === "string" && p.field ? `${p.field}: ` : "";
  return el("span", {}, `Problem: ${field}${typeof p?.message === "string" ? p.message : JSON.stringify(p)} `,
    typeof p?.code === "string" ? el("code", {}, `(${p.code})`) : null);
}

function specTable(rows) {
  return el("table", { class: "specs" },
    el("caption", {}, "As parsed by the server; ranges from the skill catalogue"),
    el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Item"), el("th", { scope: "col" }, "Value"),
      el("th", { scope: "col", class: "col-range" }, "Supported range"))),
    el("tbody", {}, ...rows.map((r) => el("tr", { class: r.problems.length ? "has-problem" : "" },
      el("th", { scope: "row" }, el("span", { class: r.field.startsWith("params.") ? "field" : "" }, r.label),
        r.note ? el("span", { class: "note" }, r.note) : null),
      el("td", {}, r.value === null ? el("em", { class: "unset" }, "not set") : r.value,
        r.isDefault ? el("span", { class: "note" }, "catalogue default, not declared") : null,
        ...r.problems.map((p) => el("p", { class: "problem" }, problemText(p)))),
      el("td", { class: "col-range" }, r.range ?? "")))));
}

// ---- learning run ----------------------------------------------------------------------------------------------
let run = INITIAL_RUN;

function dispatch(ev) {
  const before = run;
  run = runStep(run, ev);
  if (run !== before) renderRun(before);
}

async function requestRun(ev) {
  dispatch(ev); // "start" or "confirm"
  if (run.phase !== "starting") return;
  dispatch({ type: "answer", result: await api.startRun(lastChat.declared_spec, run.confirmCost) });
}

function renderRun(before) {
  $("start").disabled = !canStart(run);
  $("confirm").hidden = run.phase !== "confirm";
  $("start-row").hidden = run.phase === "confirm";
  const status = $("run-status");
  switch (run.phase) {
    case "blocked": status.textContent = "Available once the declared skill is valid."; break;
    case "ready": status.textContent = "Ready. Starting runs real simulator episodes on the server."; break;
    case "starting": status.textContent = run.confirmCost ? "Starting the run with the cost confirmed." : "Asking the server to start the run."; break;
    case "confirm":
      status.textContent = "The server asks you to confirm the quoted usage before the run starts.";
      $("confirm-msg").textContent = run.message;
      $("confirm-quote").replaceChildren(...run.quote.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));
      if (before.phase !== "confirm") $("confirm-h").focus();
      break;
    case "started":
      learn.addRun(run.runId, true);
      status.replaceChildren(`Run ${run.runId} started. `, el("a", { href: "#learn", class: "link" }, "Follow it in Learn"), ".");
      break;
    case "unavailable": status.textContent = `Learning runs are unavailable: ${run.message}`; break;
    case "failed": status.textContent = `The run did not start: ${run.message} (${run.code})`; break;
  }
  if (before.phase === "confirm" && run.phase !== "confirm") ($("start").disabled ? status : $("start")).focus();
}

$("start").addEventListener("click", () => requestRun({ type: "start" }));
$("confirm-yes").addEventListener("click", () => requestRun({ type: "confirm" }));
$("confirm-no").addEventListener("click", () => dispatch({ type: "cancel" }));

// ---- start -----------------------------------------------------------------------------------------------------
loadState();
loadCatalogue();
connectEvents();

// Example requests fill the message box; the person still decides to send.
for (const chip of document.querySelectorAll("[data-example]")) {
  chip.addEventListener("click", () => { $("msg").value = chip.dataset.example; $("msg").focus({ preventScroll: true }); });
}
