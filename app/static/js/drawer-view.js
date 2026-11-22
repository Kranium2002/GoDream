// "Details for engineers": everything technical, exactly as the server sent it or as this page measured it. Decision
// numbers and thresholds, votes, timings, frame ages, the image window contract, run ids, seeds, method names, the
// capability list and the open run's raw event log. Rendered only while open, at most a few times a second.
import { resultView, timelineRows, timingView } from "./learn.js";
import { $, el, kv, setText } from "./dom.js";

const ms = (s) => (s ? `${s.p50.toFixed(1)} ms median, ${s.p95.toFixed(1)} ms p95 (n = ${s.n})` : "not measured");
const fps = (v) => (typeof v === "number" ? `${v.toFixed(1)} per second` : "not measured");
const MAX_LOG = 400;

export function initDrawerView(ctx) {
  let open = false, queued = false, lastLogKey = "";

  function setOpen(v) {
    open = v;
    $("drawer").hidden = !v;
    $("drawer-open").setAttribute("aria-expanded", String(v));
    document.body.classList.toggle("drawer-open", v);
    if (v) { render(); $("drawer-close").focus({ preventScroll: true }); }
    else $("drawer-open").focus({ preventScroll: true });
    ctx.placeStage();
  }
  $("drawer-open").addEventListener("click", () => setOpen($("drawer").hidden));
  $("drawer-close").addEventListener("click", () => setOpen(false));
  $("drawer").addEventListener("keydown", (e) => { if (e.key === "Escape") setOpen(false); });

  function changed() {
    if (!open || queued) return;
    queued = true;
    setTimeout(() => { queued = false; if (open) render(); }, 250);
  }

  function section(id, rows) {
    const body = $(id);
    body.replaceChildren(...rows.filter(Boolean));
    body.closest("details").hidden = !rows.filter(Boolean).length;
  }

  function render() {
    renderLive();
    renderRun();
    renderServer();
  }

  function renderLive() {
    const s = ctx.live.stats();
    const w = s.settings;
    const rows = [];
    rows.push(el("h4", {}, "Camera pipeline (measured in this browser)"), kv([
      ["Camera", s.on ? `on${s.video ? `, ${s.video.width} x ${s.video.height}` : ""}` : "off"],
      ["Camera frames", fps(s.cameraFps)],
      ["Model expects (image_window.hz)", w ? `${w.hz} per second` : "not published"],
      ["Windows sent", fps(s.sendRate)],
      ["Frame age at send", ms(s.ageAtSend)],
      ["Round trip of /api/infer", ms(s.rtt)],
      ["Frame age at the server (frame_age_ms)", ms(s.serverAge)],
      ["Windows replaced while one was in flight", String(s.dropped)],
      ["Answers / issued commands", `${s.looked} / ${s.dodged}`],
    ]));
    const v = s.view;
    if (v?.state === "decision") {
      rows.push(el("h4", {}, `Last decision: ${v.action.code}${v.issued === true ? " (issued)" : ""}`), kv(v.rows));
      if (v.gate.length) rows.push(el("h4", {}, "Gate"), el("ul", { class: "checks" }, ...v.gate.map((g) => el("li", { "data-ok": String(g.ok) }, `${g.ok ? "met" : "not met"}: ${g.label}`))));
      if (v.thresholds) rows.push(kv([["threshold_values.threat", fmt(v.thresholds.threat)], ["threshold_values.action", fmt(v.thresholds.action)], ["threshold_values.margin", fmt(v.thresholds.margin)]]));
      if (v.badges.length) rows.push(el("p", { class: "tags" }, ...v.badges.map((b) => el("span", { class: "tag" }, b.text))));
      if (v.studentStatus) rows.push(el("h4", {}, "student_status (the server's exact sentence)"), el("p", { class: "quote" }, v.studentStatus));
      if (s.answer?.data?.command) rows.push(kv([["command", JSON.stringify(s.answer.data.command)]]));
    } else if (v) rows.push(el("p", {}, `Last answer: ${v.state}: ${v.message}${v.code ? ` (${v.code})` : ""}`));
    if (s.inferCap) rows.push(kv([["infer capability", `${s.inferCap.available ? "available" : "unavailable"}${s.inferCap.reason ? `: ${s.inferCap.reason}` : ""}`]]));
    if (w) {
      rows.push(el("h4", {}, "Image window contract (/api/state.image_window)"), kv([
        ["frames x height x width", `${w.frames} x ${w.height} x ${w.width}`], ["hz", String(w.hz)],
        ["window_encoding", w.windowEncoding ? `${w.windowEncoding} (from ${w.encodingFrom})` : "not sent"],
        ["diff_floor (event threshold for events)", w.diffFloor !== undefined ? String(w.diffFloor) : "not sent"],
        ["luma", w.luma.join(", ")], ["normalization", String(w.divisor)], ["crop, order, mirrored", "centre_square, oldest_first, false"],
      ]));
    }
    section("dr-live-body", rows);
  }

  function renderRun() {
    const id = ctx.openRun();
    const body = id ? ctx.cachedRunBody(id) : null;
    const rows = [];
    if (id) {
      const card = ctx.cardOf(id);
      rows.push(kv([["run_id", id], ["skill (spec)", String(body?.results?.skill ?? card?.skill ?? "not known")], ["state", String(body?.state ?? card?.state ?? "not read")],
        ...(card?.updated ? [["updated_utc", card.updated]] : [])]));
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
      const tl = ctx.timeline();
      if (tl?.runId === id) rows.push(el("h4", {}, "Pipeline steps (as logged)"), kv(timelineRows(tl).map((s) => [s.step, [s.state ?? "no line", s.progress, s.time, s.reason].filter(Boolean).join(" · ")])));
    }
    section("dr-run-body", rows);
    const log = id ? ctx.runLog() : [];
    const key = `${id}:${log.length}`;
    if (key !== lastLogKey) {
      lastLogKey = key;
      const lines = log.slice(-MAX_LOG);
      $("dr-log-body").replaceChildren(...lines.map((l) => el("li", {}, `${String(l.time_utc ?? "").slice(11, 19)} ${l.step} ${l.state}${Number.isFinite(l.done) ? ` ${l.done}/${l.total}` : ""}${l.data ? ` ${JSON.stringify(l.data).slice(0, 240)}` : ""}`)));
      $("dr-log").hidden = !lines.length;
      setText($("dr-log-count"), log.length > MAX_LOG ? `last ${MAX_LOG} of ${log.length} lines` : `${log.length} lines`);
    }
  }

  function renderServer() {
    const s = ctx.state();
    const rows = [];
    if (s) {
      rows.push(kv([["simulator", `${s.status ?? "?"}, preset ${s.preset ?? "?"}`], ["episodes", JSON.stringify(s.episodes ?? null)]]));
      rows.push(el("h4", {}, "Capabilities"), el("ul", { class: "checks" }, ...ctx.capabilities().map((c) => el("li", { "data-ok": String(c.available) }, `${c.name}: ${c.available ? "available" : "unavailable"}${c.reason ? `. ${c.reason}` : ""}`))));
      if (s.run_modes) rows.push(el("h4", {}, "run_modes"), kv(Object.entries(s.run_modes).flatMap(([sk, m]) => Object.entries(m ?? {}).map(([k, why]) => [`${sk}.${k}`, why === "" ? "offered" : `refused: ${why}`]))));
    }
    const cap = ctx.stageCaption();
    if (cap?.detail) rows.push(el("h4", {}, "Episode on stage (the server's caption)"), el("p", { class: "quote" }, cap.detail));
    section("dr-server-body", rows);
  }

  const fmt = (v) => (typeof v === "number" ? String(v) : "not sent");
  return { changed, setOpen, isOpen: () => open };
}
