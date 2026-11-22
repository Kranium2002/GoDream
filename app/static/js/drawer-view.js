// "Details for engineers": everything technical, exactly as the server sent it or as this page measured it: the camera
// pipeline's rates and frame ages, the last decision with its thresholds and gate, the server's exact status sentence,
// the side-by-side run, the demos' declarations, the lesson on show (run id, seeds, methods, timings, results), the
// run's event log, capabilities and the latest raw events. Rendered only while open, a few times a second at most.
import { fromRunResponse, resultView, timelineRows, timingView } from "./learn.js";
import { commandFields, driveCommand } from "./drive.js";
import { $, el, kv, setText } from "./dom.js";

const ms = (s) => (s ? `${s.p50.toFixed(1)} ms median, ${s.p95.toFixed(1)} ms p95 (n = ${s.n})` : "not measured");
const rate = (v) => (typeof v === "number" ? `${v.toFixed(1)} per second` : "not measured");
const MAX_LOG = 400, MAX_RAW = 60;
const KEYS = { w: "forward", s: "back", a: "left", d: "right", q: "turn_left", e: "turn_right", " ": "stop" };
const PAD_SHARE = 0.5; // the drive pad moves at this share of each published range

export function initDrawerView(ctx) {
  let open = false, queued = false, lastLogKey = "";
  const raw = []; // newest last: [time, name, text]

  function setOpen(v) {
    open = v;
    $("drawer").hidden = !v;
    $("drawer-open").setAttribute("aria-expanded", String(v));
    if (v) { render(); $("drawer-h").focus({ preventScroll: true }); }
    else $("drawer-open").focus({ preventScroll: true });
  }
  $("drawer-open").addEventListener("click", () => setOpen($("drawer").hidden));
  $("drawer-close").addEventListener("click", () => setOpen(false));
  $("drawer").addEventListener("keydown", (e) => { if (e.key === "Escape") setOpen(false); });

  function changed() {
    if (!open || queued) return;
    queued = true;
    setTimeout(() => { queued = false; if (open) render(); }, 300);
  }

  function section(id, rows) { $(id).replaceChildren(...rows.filter(Boolean)); }

  function render() {
    renderLive();
    renderDuel();
    renderRun();
    renderServer();
    $("dr-raw").replaceChildren(...raw.slice().reverse().map(([t, name, text]) => el("li", {}, `${t} ${name} ${text}`)));
  }

  function renderLive() {
    const s = ctx.live.stats();
    const w = s.settings;
    const rows = [el("h4", {}, "Camera pipeline, measured in this browser"), kv([
      ["Camera", s.on ? `on${s.video ? `, ${s.video.width} x ${s.video.height}` : ""}` : "off"],
      ["Camera frames", rate(s.cameraFps)],
      ["Model expects (image_window.hz)", w ? `${w.hz} per second` : "not published"],
      ["Windows sent", rate(s.sendRate)],
      ["Frame age at send", ms(s.ageAtSend)],
      ["Round trip", ms(s.rtt)],
      ["Frame age at the server (frame_age_ms)", ms(s.serverAge)],
      ["Windows replaced while one was in flight", String(s.dropped)],
      ["Answers / issued commands", `${s.looked} / ${s.moved}`],
    ])];
    const v = s.view;
    if (v?.state === "decision") {
      rows.push(el("h4", {}, `Last decision: ${v.action.code}${v.issued === true ? " (issued)" : ""}`), kv(v.rows));
      if (v.gate.length) rows.push(el("h4", {}, "Gate"), el("ul", { class: "checks" }, ...v.gate.map((g) => el("li", { "data-ok": String(g.ok) }, `${g.ok ? "met" : "not met"}: ${g.label}`))));
      if (v.thresholds) rows.push(kv([["threshold_values.threat", fmt(v.thresholds.threat)], ["threshold_values.action", fmt(v.thresholds.action)], ["threshold_values.margin", fmt(v.thresholds.margin)]]));
      if (v.badges.length) rows.push(el("ul", { class: "checks" }, ...v.badges.map((b) => el("li", {}, b.text))));
      if (v.studentStatus) rows.push(el("h4", {}, "student_status, the server's exact sentence"), el("p", { class: "quote" }, v.studentStatus));
      if (s.answer?.data?.command) rows.push(kv([["command", JSON.stringify(s.answer.data.command)]]));
    } else if (v) rows.push(el("p", {}, `Last answer: ${v.state}: ${v.message}${v.code ? ` (${v.code})` : ""}`));
    if (s.inferCap) rows.push(kv([["infer capability", `${s.inferCap.available ? "available" : "unavailable"}${s.inferCap.reason ? `: ${s.inferCap.reason}` : ""}`]]));
    if (w) rows.push(el("h4", {}, "Image window contract (/api/state image_window)"), kv([
      ["frames x height x width", `${w.frames} x ${w.height} x ${w.width}`], ["hz", String(w.hz)],
      ["window_encoding", w.windowEncoding ? `${w.windowEncoding} (from ${w.encodingFrom})` : "not sent"],
      ["diff_floor", w.diffFloor !== undefined ? String(w.diffFloor) : "not sent"],
      ["luma", w.luma.join(", ")], ["normalization", String(w.divisor)], ["crop, order, mirrored", "centre_square, oldest_first, false"],
    ]));
    section("dr-live", rows);
  }

  function renderDuel() {
    const d = ctx.duel(), demos = ctx.demosInfo();
    const rows = [kv([["duel.state", d.state], ["demo", String(d.demo ?? "none")], ["seed", String(d.seed ?? "none")], ["params", JSON.stringify(d.params ?? null)],
      ["before", JSON.stringify(d.sides.before)], ["after", JSON.stringify(d.sides.after)], ["sense samples", JSON.stringify(ctx.senseCounts())]])];
    rows.push(el("h4", {}, "GET /api/demos"), demos.ok ? el("p", { class: "quote" }, JSON.stringify(demos.raw)) : el("p", {}, demos.message ?? "not read yet"));
    if (demos.problems?.length) rows.push(el("ul", { class: "checks" }, ...demos.problems.map((p) => el("li", {}, p))));
    section("dr-duel", rows);
  }

  function renderRun() {
    const id = ctx.sheet.stats().lesson;
    const body = id ? ctx.cachedRunBody(id) : null;
    const rows = [];
    if (id) {
      const card = ctx.cards().find((c) => c.runId === id);
      rows.push(kv([["run_id", id], ["skill", String(body?.results?.skill ?? card?.skill ?? "not known")], ["state", String(body?.state ?? card?.state ?? "not read")],
        ...(card?.updated ? [["updated_utc", card.updated]] : []), ["story source", String(ctx.sheet.stats().story?.source ?? "not read")]]));
      const t = timingView(body?.results);
      if (t) rows.push(el("h4", {}, "Timing (run_skill, wall clock on this machine)"), kv([
        ["mode", String(t.mode)], ["attempts", String(t.attempts)], ["learn_s", t.learnS.toFixed(1)], ["evaluate_s", t.evaluateS === null ? "none" : t.evaluateS.toFixed(1)],
        ...t.steps.map(([k, s]) => [`steps.${k}`, `${s.toFixed(1)} s`]), ["machine", [t.machine, t.cpuCount !== null && `${t.cpuCount} logical CPUs`].filter(Boolean).join(", ")]]));
      const r = resultView(body?.results ?? null);
      if (r.kind === "table") {
        rows.push(el("h4", {}, `Results (${r.methods.map((m) => m.key).join(" vs ")})`), el("div", { class: "scroll-x" }, el("table", { class: "dtable" },
          el("thead", {}, el("tr", {}, el("th", {}, "metric"), ...r.methods.map((m) => el("th", {}, m.key)))),
          el("tbody", {}, ...r.rows.map((row) => el("tr", {}, el("th", {}, row.name), ...row.cells.map((c) => el("td", {}, c))))))));
        if (r.paired.length) rows.push(kv(r.paired.map((p) => [`paired ${p.name} (${p.comparison})`, p.diff])));
        rows.push(kv([["seeds", JSON.stringify(body.results.seeds ?? null)], ["scenarios", String(r.scenarioCount ?? "not listed")]]));
      }
      if (Array.isArray(body?.steps)) rows.push(el("h4", {}, "Pipeline steps, as logged"),
        kv(timelineRows(fromRunResponse(id, body)).map((s) => [s.step, [s.state ?? "no line", s.progress, s.time, s.reason].filter(Boolean).join(", ")])));
    } else rows.push(el("p", {}, "No lesson is on show."));
    const st = ctx.sheet.stats();
    if (st.klass) rows.push(kv([["classroom", JSON.stringify(st.klass)]]));
    if (st.teach?.phase && st.teach.phase !== "blocked") rows.push(kv([["teach", JSON.stringify(st.teach)]]));
    section("dr-run", rows);
    const log = ctx.runLog();
    const key = `${id}:${log.length}`;
    if (key !== lastLogKey) {
      lastLogKey = key;
      const lines = log.slice(-MAX_LOG);
      $("dr-log-body").replaceChildren(...lines.map((l) => el("li", {}, `${String(l.time_utc ?? "").slice(11, 19)} ${l.step} ${l.state}${Number.isFinite(l.done) ? ` ${l.done}/${l.total}` : ""}${l.data ? ` ${JSON.stringify(l.data).slice(0, 240)}` : ""}`)));
      $("dr-log").hidden = !lines.length;
      setText($("dr-log-count"), log.length > MAX_LOG ? `(last ${MAX_LOG} of ${log.length} lines)` : `(${log.length} lines)`);
    }
  }

  function renderServer() {
    const s = ctx.state();
    const rows = [];
    if (s) {
      rows.push(kv([["simulator", `${s.status ?? "?"}, preset ${s.preset ?? "?"}`], ["episodes", JSON.stringify(s.episodes ?? null)], ["glass", document.body.classList.contains("glass-solid") ? "solid (see messages)" : "frosted"]]));
      rows.push(el("h4", {}, "Capabilities"), el("ul", { class: "checks" }, ...ctx.capabilities().map((c) => el("li", { "data-ok": String(c.available) }, `${c.name}: ${c.available ? "available" : "unavailable"}${c.reason ? `. ${c.reason}` : ""}`))));
      if (s.run_modes) rows.push(el("h4", {}, "run_modes"), kv(Object.entries(s.run_modes).flatMap(([sk, m]) => Object.entries(m ?? {}).map(([k, why]) => [`${sk}.${k}`, why === "" ? "offered" : `refused: ${why}`]))));
    }
    section("dr-server", rows);
  }

  // ---- the drive pad: commands from the ranges /api/state publishes ------------------------------------------------
  const say = (text) => setText($("drive-status"), text);
  async function go(dir) {
    const fields = commandFields(ctx.state());
    if (!fields) return say("The simulator has not published its command ranges yet.");
    const cmd = driveCommand(dir, fields, PAD_SHARE);
    if (!cmd) return say("The published ranges do not allow that direction.");
    const r = dir === "stop" ? await ctx.api.stop() : await ctx.api.drive(cmd);
    say(r.ok ? `Sent: ${Object.entries(r.data?.command ?? cmd).map(([k, v]) => `${k} ${Number(v).toFixed(2)}`).join(", ")}. It keeps this until the next command.` : `Refused: ${r.message}`);
  }
  for (const b of document.querySelectorAll("[data-dir]")) b.addEventListener("click", () => go(b.dataset.dir));
  $("drive-reset").addEventListener("click", async () => { const r = await ctx.api.reset(); say(r.ok ? "A fresh episode started." : `The reset was refused: ${r.message}`); });
  addEventListener("keydown", (e) => {
    const pad = $("drive-status").closest("details");
    if (!open || !pad.open || e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.("input, textarea, select")) return;
    const dir = KEYS[e.key.toLowerCase()];
    if (!dir) return;
    e.preventDefault();
    go(dir);
  });

  const fmt = (v) => (typeof v === "number" ? String(v) : "not sent");
  return {
    changed, setOpen, isOpen: () => open,
    raw(name, data) {
      raw.push([new Date().toISOString().slice(11, 23), name, JSON.stringify(data).slice(0, 300)]);
      if (raw.length > MAX_RAW) raw.shift();
      changed();
    },
  };
}
