// "How it learned": a glass sheet with the lesson as four beats, each with a picture drawn from the run's own files,
// then "Teach it again, live" where the server offers a quick lesson (its quote stated, a confirm step), or the time a
// full lesson took and "Replay the lesson we recorded". While a lesson runs or replays, the stage becomes the
// classroom: one wide lane with the real tries, a tally, and the beats filling in from the run's events.
import { beatsOf, choiceAxis, curvePaths, endings, readStory, storyFromRun, topView } from "./story.js";
import { costSentence, liveBeats, quickLesson, recordedDuration } from "./lesson.js";
import { canStart, INITIAL_RUN, runStep } from "./run.js";
import { applyLine, emptyTimeline } from "./learn.js";
import { valueText } from "./demos.js";
import { episodesControl } from "./episodes.js";
import { $, el, setText, svgEl } from "./dom.js";

const fmt = (n) => (Number.isInteger(n) ? n.toLocaleString("en-US") : String(Number(n.toPrecision(3))));

export function initSheetView(ctx) {
  const { api, notes } = ctx;
  let open = false, lesson = null, story = null, beat = 1, loadingFor = null;
  let teach = INITIAL_RUN, quote = null;
  let klass = null; // {kind: "live" | "replay", runId, tl, state, tally}
  const stories = new Map(); // run id -> story (read once per page)

  // ---- open and close -------------------------------------------------------------------------------------------
  function setOpen(v, focus = true) {
    open = v;
    $("sheet").hidden = !v;
    $("learn-open").setAttribute("aria-expanded", String(v));
    document.body.classList.toggle("sheet-open", v);
    ctx.placeStage();
    if (v) { load(); if (focus) $("sheet-h").focus({ preventScroll: true }); }
    else if (focus) $("learn-open").focus({ preventScroll: true });
  }
  $("learn-open").addEventListener("click", () => setOpen(!open));
  $("sheet-close").addEventListener("click", () => setOpen(false));
  $("sheet").addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").hidden) setOpen(false); });

  // ---- which lesson: the demo's recorded run, or (without demos) one picked from the recorded runs ----------------
  // The newest finished run of each skill, newest first.
  function lessons() {
    const seen = new Set();
    return ctx.cards().filter((c) => c.state === "done" && c.hasResult !== false && c.skill)
      .sort((a, b) => String(b.updated ?? "").localeCompare(String(a.updated ?? "")))
      .filter((c) => !seen.has(c.skill) && seen.add(c.skill));
  }

  function currentLesson() {
    const demo = ctx.demo();
    if (klass) return klass.runId;
    if (demo) return demo.runId;
    if (lesson && lessons().some((c) => c.runId === lesson)) return lesson;
    return lessons()[0]?.runId ?? null;
  }

  async function load() {
    const id = currentLesson();
    if (!id) { story = null; render(); return; }
    if (stories.has(id)) { story = stories.get(id); render(); return; }
    if (loadingFor === id) return;
    loadingFor = id;
    story = null;
    render();
    let s = null;
    const r = await api.story(id);
    if (!r.ok && !r.unavailable) notes.set("story", "danger", `GET /api/story/${id}: ${r.message} (${r.code})`, "Lesson");
    const body = await ctx.runBody(id);
    const fromRun = body ? storyFromRun(body, id) : null;
    if (r.ok) {
      // The story's own parts first; a part it leaves out is filled from the run's files when they hold it.
      s = readStory(r.data, id);
      if (fromRun) for (const k of ["tries", "brain", "test", "imagined"]) s[k] ??= fromRun[k];
    } else s = fromRun;
    loadingFor = null;
    if (s) stories.set(id, s);
    if (currentLesson() === id) { story = s; render(); }
    ctx.changed();
  }

  // ---- rendering ---------------------------------------------------------------------------------------------------
  function render() {
    if (!open) return;
    renderLessons();
    const id = currentLesson();
    const live = klass ? liveBeats(klass.tl, klass.state) : null;
    const beats = story ? beatsOf(story) : [1, 2, 3, 4].map((n) => ({ n, has: false }));
    const titles = ["A few real tries", "TabPFN fills in the what-ifs", "A tiny spiking brain learns the choices", "Tested on cases it never saw"];
    $("beat-tabs").replaceChildren(...beats.map((b, i) => {
      const lb = live?.[i];
      const btn = el("button", { type: "button", "aria-current": beat === b.n ? "step" : null, "data-state": lb?.state ?? "", "data-empty": String(!b.has && !lb),
        on: { click: () => { beat = b.n; render(); } } }, el("span", { class: "num" }, String(b.n)), el("span", {}, titles[i]));
      return el("li", {}, btn);
    }));
    const box = $("beat");
    if (!id) box.replaceChildren(el("p", { class: "quiet-p" }, "No finished lesson is recorded on this server yet. When one is, its four steps show here."));
    else if (!story && loadingFor) box.replaceChildren(el("p", { class: "quiet-p" }, "Reading the lesson's files."));
    else if (!story && !klass) box.replaceChildren(el("p", { class: "quiet-p" }, "This lesson's files could not be read. The engineers' details say why."));
    else box.replaceChildren(...beatBody(beat, titles[beat - 1], live?.[beat - 1] ?? null));
    renderTeach();
  }

  function renderLessons() {
    const box = $("lessons");
    if (ctx.demo() || klass) { box.hidden = true; return; }
    const list = lessons();
    box.hidden = list.length < 2;
    const id = currentLesson();
    box.replaceChildren(...list.slice(0, 8).map((c) => el("button", { type: "button", "aria-pressed": String(c.runId === id),
      on: { click: () => { lesson = c.runId; beat = 1; load(); } } }, `${c.skill}, ${when(c.updated)}`)));
  }

  function beatBody(n, title, live) {
    const out = [el("h3", {}, title)];
    if (live) out.push(el("p", { class: "note" }, liveText(live)));
    const s = story;
    if (!s) return out;
    if (n === 1) out.push(...beatTries(s.tries));
    if (n === 2) out.push(...beatWhatIf(s));
    if (n === 3) out.push(...beatBrain(s.brain));
    if (n === 4) out.push(...beatTest(s.test));
    if (n < 4) out.push(el("button", { class: "next", type: "button", on: { click: () => { beat = n + 1; render(); } } }, "Next step"));
    return out;
  }

  function liveText(b) {
    const p = b.progress ? ` ${fmt(b.progress.done)} of ${fmt(b.progress.total)}.` : "";
    return { waiting: "Waiting for the steps before it.", running: `Happening now.${p}`, done: `Done.${p}`, stopped: "The lesson stopped here." }[b.state] ?? "";
  }

  function facts(rows) {
    const shown = rows.filter(([v]) => v !== null && v !== undefined);
    return shown.length ? [el("dl", { class: "facts" }, ...shown.map(([v, label]) => el("div", {}, el("dt", {}, label), el("dd", {}, typeof v === "number" ? fmt(v) : v))))] : [];
  }

  // 1: the real tries, seen from above (Hills) or as a grid of outcomes (Dodge), coloured by how they ended.
  function beatTries(t) {
    const out = [el("p", {}, "The dog tried for real in the simulator. Real tries are slow, so there are only a handful.")];
    if (!t) return [...out, el("p", { class: "quiet-p" }, "How many tries this lesson made is not in its files.")];
    out.push(...facts([[t.nTries, "real tries"], [t.nDecisions, "decision moments"], [t.contextRows, "examples TabPFN learned from"]]));
    const groups = endings(t.paths ?? t.outcomes ?? []);
    const look = (ended) => ["k-plain", "k-before", "k-val"][groups.findIndex((g) => g.ended === ended) % 3];
    if (t.paths) {
      const w = 500, h = 230, v = topView(t, { w, h, pad: 12 });
      const svg = svgEl("svg", { class: "pic", viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": `The paths of ${t.paths.length} tries seen from above.` });
      for (const o of v.obstacles) svg.append(svgEl("circle", { class: "obstacle", cx: o.cx, cy: o.cy, r: o.r }));
      for (const p of v.paths) {
        const k = look(p.ended);
        svg.append(svgEl("path", { class: "path", d: p.d, stroke: k === "k-before" ? "var(--slate)" : "var(--bone)", "stroke-dasharray": k === "k-val" ? "5 4" : "none" }));
      }
      out.push(svg);
    } else if (t.outcomes) {
      const cols = Math.min(20, t.outcomes.length), size = 22, w = cols * size, h = Math.ceil(t.outcomes.length / cols) * size;
      const svg = svgEl("svg", { class: "pic", viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": `${t.outcomes.length} tries and how each ended.` });
      t.outcomes.forEach((o, k) => {
        const g = look(o.ended);
        svg.append(svgEl("rect", { x: (k % cols) * size + 2, y: Math.floor(k / cols) * size + 2, width: size - 4, height: size - 4, rx: 3,
          fill: g === "k-plain" ? "var(--bone)" : g === "k-before" ? "var(--slate)" : "none", stroke: "var(--bone)", "stroke-width": g === "k-val" ? 1.5 : 0 }));
      });
      out.push(svg);
    } else out.push(el("p", { class: "quiet-p" }, "A picture of these tries is not published yet."));
    if (groups.length) out.push(el("p", { class: "legend" }, ...groups.map((g) => el("span", { class: look(g.ended) }, `${g.ended || "not said"}: ${g.count}`))));
    return out;
  }

  // 2: one real decision moment: the choice made (solid) beside TabPFN's predictions for the others (ghosted).
  function beatWhatIf(s) {
    const out = [el("p", {}, "For the choices the dog did not make, TabPFN predicts what would have happened. That turns a handful of tries into many.")];
    const w = s.whatIf;
    if (w) {
      const ax = choiceAxis(w.choices), W = 500, rowH = 34, left = 178, right = W - 16, H = w.choices.length * rowH + 30;
      const X = (v) => left + ax.x(v) * (right - left);
      const svg = svgEl("svg", { class: "pic", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Each choice with TabPFN's prediction and its range." });
      svg.append(svgEl("line", { class: "axis", x1: left, x2: right, y1: H - 22, y2: H - 22 }));
      const unit = w.choices.find((c) => c.predicted?.unit)?.predicted.unit ?? "";
      for (const v of [ax.lo, ax.hi]) { const t = svgEl("text", { class: "lbl", x: X(v), y: H - 6, "text-anchor": v === ax.lo ? "start" : "end" }); t.textContent = valueText(Number(v.toPrecision(2)), unit); svg.append(t); }
      w.choices.forEach((c, k) => {
        const y = k * rowH + 16;
        const name = svgEl("text", { class: c.tried ? "lbl-strong" : "lbl", x: 0, y: y + 4 }); name.textContent = c.tried ? `${c.label} (chosen)` : c.label;
        svg.append(name);
        const p = c.predicted;
        if (p) {
          if (p.low !== null && p.high !== null) svg.append(svgEl("line", { class: c.tried ? "tried" : "ghost", x1: X(p.low), x2: X(p.high), y1: y, y2: y, "stroke-width": 2 }));
          svg.append(svgEl("circle", { class: c.tried ? "tried" : "ghost-dot", cx: X(p.value), cy: y, r: 6 }));
        }
        if (c.measured) svg.append(svgEl("circle", { class: "measured", cx: X(c.measured.value), cy: y, r: 9 }));
      });
      out.push(svg);
      const label = w.choices.find((c) => c.predicted?.label)?.predicted.label;
      out.push(el("p", { class: "legend" }, el("span", { class: "k-after dot" }, "the choice it made"), el("span", { class: "k-ghost ring" }, "TabPFN's prediction for a choice it did not make"),
        w.choices.some((c) => c.measured) ? el("span", { class: "k-val ring" }, "what really happened") : null));
      const when = [w.moment.tryNo !== null ? `try ${w.moment.tryNo}` : null, w.moment.t !== null ? `${w.moment.t.toFixed(1)} s in` : null].filter(Boolean).join(", ");
      out.push(el("p", { class: "quiet-p" }, `${when ? `One moment from ${when}. ` : ""}${label ? `Predicted: ${label.toLowerCase()}.` : ""}`));
    }
    if (s.imagination) {
      const W = 500, rowH = 40, H = s.imagination.length * rowH + 4, left = 132, right = W - 70;
      const max = Math.max(...s.imagination.flatMap((r) => [r.tabpfn, r.formula]));
      const svg = svgEl("svg", { class: "pic", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "How far TabPFN's guesses and the formula's guesses were from real tries held back." });
      s.imagination.forEach((r, k) => {
        const y = k * rowH + 4;
        const t = svgEl("text", { class: "lbl", x: 0, y: y + 14 }); t.textContent = r.label; svg.append(t);
        for (const [cls, v, dy] of [["tabpfn", r.tabpfn, 0], ["formula", r.formula, 15]]) {
          svg.append(svgEl("rect", { class: cls, x: left, y: y + dy, width: Math.max(2, (v / max) * (right - left)), height: 11, rx: 2 }));
          const vt = svgEl("text", { class: "lbl", x: left + (v / max) * (right - left) + 6, y: y + dy + 10 }); vt.textContent = valueText(Number(v.toPrecision(2)), r.unit); svg.append(vt);
        }
      });
      out.push(el("p", {}, "How far off each guess was, on real tries kept back from both:"), svg,
        el("p", { class: "legend" }, el("span", { class: "k-ghost" }, "TabPFN"), el("span", { class: "k-before" }, "the plain physics formula")));
    }
    if (!w && !s.imagination) {
      if (s.imagined) out.push(...facts([[s.imagined.situations, "imagined situations"], [s.imagined.choices, "choices predicted in each"]]));
      out.push(el("p", { class: "quiet-p" }, "A picture of one moment's what-ifs is not published for this lesson yet."));
    }
    return out;
  }

  // 3: the student: its size, its training curve, spikes per decision and decision time.
  function beatBrain(b) {
    const out = [el("p", {}, "Inside TabPFN's imagined situations, a tiny spiking network learns which choice to make. It learns the choices, not TabPFN's numbers.")];
    if (!b) return [...out, el("p", { class: "quiet-p" }, "This lesson's files say nothing about its spiking network.")];
    out.push(...facts([[b.neurons, "spiking neurons"], [b.inputs, "numbers it reads each time"], [b.choices, "choices it picks from"], [b.spikesPerDecision !== null ? Math.round(b.spikesPerDecision) : null, "spikes per decision"],
      [b.decisionMs !== null ? `${b.decisionMs.toFixed(b.decisionMs < 10 ? 2 : 0)} ms` : null, "to decide, on this laptop"], [b.stepsRun, "training rounds"]]));
    const c = b.curve.length ? curvePaths(b.curve, { w: 500, h: 150, pad: 10 }) : null;
    if (c) {
      const svg = svgEl("svg", { class: "pic", viewBox: "0 0 500 170", role: "img", "aria-label": "The training curve: how far its choices were from the imagined best, round by round." });
      svg.append(svgEl("line", { class: "axis", x1: 10, x2: 490, y1: 150, y2: 150 }));
      if (c.train) svg.append(svgEl("polyline", { class: "curve-train", points: c.train }));
      if (c.validation) svg.append(svgEl("polyline", { class: "curve-val", points: c.validation }));
      for (const [x, s, a] of [[10, c.steps[0], "start"], [490, c.steps[1], "end"]]) { const t = svgEl("text", { class: "lbl", x, y: 166, "text-anchor": a }); t.textContent = `round ${fmt(s)}`; svg.append(t); }
      out.push(svg, el("p", { class: "legend" }, c.train ? el("span", { class: "k-after" }, "on its practice situations") : null, c.validation ? el("span", { class: "k-val" }, "on situations kept back to check") : null),
        el("p", { class: "quiet-p" }, "Lower means its choices are closer to the best choices in the imagined situations."));
    }
    return out;
  }

  // 4: before and after on situations it never saw, with counts and ranges.
  function beatTest(t) {
    const out = [el("p", {}, "Then both brains face the same new situations: the walking brain alone, and with the learned reflex.")];
    if (!t) return [...out, el("p", { class: "quiet-p" }, "This lesson has no test results in its files.")];
    for (const r of t.rows) {
      const fr = r.unit === "fraction";
      const vals = [r.before, r.after].filter(Boolean).flatMap((c) => [c.value, c.low, c.high]).filter((v) => typeof v === "number");
      // Shares on 0..1; other measures from 0 (or their lowest value when below 0) to a little past the largest, so a
      // small change is drawn small.
      const lo = fr ? 0 : Math.min(0, ...vals), hi = fr ? 1 : Math.max(...vals) * 1.08 || 1;
      const W = 500, left = 8, right = W - 8, X = (v) => (hi === lo ? W / 2 : left + ((v - lo) / (hi - lo)) * (right - left));
      const svg = svgEl("svg", { class: "pic", viewBox: `0 0 ${W} 44`, role: "img", "aria-label": `${r.label}: ${r.before ? `${r.before.text} before` : "nothing measured before"}, ${r.after.text} after.` });
      svg.append(svgEl("line", { class: "axis", x1: 0, x2: W, y1: 40, y2: 40 }));
      [[r.before, "before", 12], [r.after, "after", 28]].forEach(([c, cls, y]) => {
        if (!c) return;
        if (c.low !== null && c.high !== null) svg.append(svgEl("line", { class: cls, x1: X(c.low), x2: X(c.high), y1: y, y2: y, "stroke-width": 2, opacity: 0.6 }));
        svg.append(svgEl("circle", { class: cls, cx: X(c.value), cy: y, r: 5.5 }));
      });
      out.push(el("div", { class: "testrow" },
        el("p", { class: "testrow__head" }, el("strong", {}, r.label), " ",
          el("span", { class: "testrow__nums" }, r.before ? `${r.before.text} before, ` : "", el("span", { class: "testrow__after" }, `${r.after.text} after`))), svg,
        r.after.range || r.before?.range ? el("p", { class: "quiet-p testrow__range" }, [r.before?.range && `before ${r.before.range}`, r.after.range && `after ${r.after.range}`].filter(Boolean).join("; ")) : null));
    }
    out.push(el("p", { class: "legend" }, el("span", { class: "k-before dot" }, "walking brain alone"), el("span", { class: "k-after dot" }, "with the learned reflex"),
      t.rows.some((r) => r.after.low !== null) ? el("span", { class: "k-plain" }, "the likely range") : null));
    for (const n of t.notes) out.push(el("p", { class: "note" }, n.text));
    return out;
  }

  // ---- teach it again --------------------------------------------------------------------------------------------
  // The lesson's skill as the server's catalogue names it (run_modes and run declarations use that name; a demo or a
  // run card may carry the SkillSpec's module name instead, such as dodge_vision for DODGE).
  function skillOfLesson() {
    const demo = ctx.demo();
    const id = demo ? demo.skill : ctx.cards().find((c) => c.runId === currentLesson())?.skill ?? null;
    const entry = Array.isArray(ctx.state()?.skills) ? ctx.state().skills.find((s) => s && (s.skill === id || s.spec === id)) : null;
    return entry?.skill ?? id;
  }

  function renderTeach() {
    const box = $("teach");
    const skill = skillOfLesson();
    const state = ctx.state();
    if (klass) {
      box.replaceChildren(el("div", { class: "teach__row" }, el("button", { class: "quiet", type: "button", on: { click: endClass } },
        klass.kind === "replay" ? "Stop the replay" : "Back to the comparison")));
      return;
    }
    if (!skill || !state) { box.replaceChildren(); return; }
    const q = quickLesson(state.run_modes, skill);
    const runId = currentLesson();
    if (teach.phase === "confirm") {
      box.replaceChildren(el("p", { class: "teach__cost" }, quote ?? "The server asked to confirm the cost but sent no estimate this page can read; the engineers' details show its quote."),
        el("div", { class: "teach__row" }, el("button", { class: "go", type: "button", on: { click: () => step({ type: "confirm" }) } }, "Start the lesson"),
          el("button", { class: "quiet", type: "button", on: { click: () => { teach = runStep(teach, { type: "cancel" }); renderTeach(); } } }, "Not now")));
      return;
    }
    const failed = teach.phase === "failed" || teach.phase === "unavailable" ? el("p", {}, `The lesson did not start: ${teach.message}`) : null;
    if (q.quick) {
      box.replaceChildren(...[failed, el("p", {}, "Teach it again from scratch, live: a handful of real tries, TabPFN's what-ifs, a new spiking brain, then the test."),
        el("button", { class: "go", type: "button", disabled: teach.phase === "starting" ? true : null, on: { click: () => step({ type: "start", mode: "fast" }) } },
          teach.phase === "starting" ? "Asking the server" : "Teach it again, live")].filter(Boolean));
      return;
    }
    const card = ctx.cards().find((c) => c.runId === runId);
    const d = recordedDuration(card?.timing);
    const words = d ? `A full lesson of this skill took ${d.learn} to learn${d.test ? ` and ${d.test} to test` : ""} on this laptop, so it cannot be run live here.` : "This skill has no quick lesson on this server, so it cannot be taught live here.";
    const canReplay = runId && episodesControl(state).available;
    box.replaceChildren(el("p", {}, words), canReplay ? el("button", { class: "quiet", type: "button", on: { click: () => startReplay(runId) } }, "Replay the lesson we recorded") : null);
  }

  async function step(ev) {
    if (ev.type === "start" && !canStart(teach)) teach = runStep(teach, { type: "spec", valid: true });
    teach = runStep(teach, ev);
    renderTeach();
    if (teach.phase !== "starting") return;
    const skill = skillOfLesson();
    const r = await api.startRun({ skill }, teach.confirmCost, teach.mode);
    teach = runStep(teach, { type: "answer", result: r });
    if (teach.phase === "confirm") { quote = costSentence(r.data); requestAnimationFrame(() => $("teach").scrollIntoView({ block: "nearest" })); }
    if (teach.phase === "failed" || teach.phase === "unavailable") notes.set("teach", "danger", `POST /api/runs: ${teach.message} (${teach.code})`, "Lesson");
    if (teach.phase === "started") startClass(teach.runId);
    renderTeach();
    ctx.changed();
  }

  // ---- the classroom -------------------------------------------------------------------------------------------------
  async function startClass(runId) {
    teach = INITIAL_RUN;
    klass = { kind: "live", runId, tl: { ...emptyTimeline(), runId }, state: "running", tally: null };
    ctx.enterClass();
    beat = 1;
    setOpen(true, false);
    const r = await api.episodesStart(runId);
    if (!r.ok) notes.set("class", "warn", `The run's tries cannot be shown as they happen: ${r.message}`, "Lesson");
    render();
  }

  async function startReplay(runId) {
    const r = await api.episodesStart(runId);
    if (!r.ok) { notes.set("class", "danger", `The recorded lesson did not replay: ${r.message} (${r.code})`, "Lesson"); setText($("bar-line"), `The recorded lesson did not replay: ${r.message}`); return; }
    klass = { kind: "replay", runId, tl: emptyTimeline(), state: null, tally: null };
    ctx.enterClass();
    render();
  }

  async function endClass() {
    const was = klass;
    klass = null;
    $("tally").hidden = true;
    if (was?.kind === "replay") await api.episodesStop();
    ctx.leaveClass(was);
    render();
  }

  // A live run's event line: the beats fill in; a finished run ends the classroom with the new lesson on show.
  function onRunEvent(line) {
    if (!klass || klass.kind !== "live" || line?.run_id !== klass.runId) return;
    klass.tl = applyLine(klass.tl, line);
    const st = line?.data?.run_state;
    if (typeof st === "string") klass.state = st;
    if (line.step === "evaluate" && ["done", "failed", "stopped", "refused"].includes(line.state)) klass.state = line.state === "done" ? "done" : line.state;
    const b = liveBeats(klass.tl, klass.state);
    const now = b.find((x) => x.state === "running");
    if (now && now.n !== beat) beat = now.n;
    render();
    if (["done", "failed", "stopped", "refused"].includes(klass.state)) {
      const id = klass.runId;
      stories.delete(id);
      setTimeout(() => { if (klass?.runId === id) endClass(); }, 1500);
    }
  }

  // An "episode" event of the player: the tally of tries on the wide lane.
  function onEpisode(ev) {
    if (!klass || !ev || ev.run_id !== klass.runId) return;
    const t = $("tally");
    if (ev.state === "idle") {
      t.hidden = false;
      t.replaceChildren("All tries shown", el("small", {}, typeof ev.caption === "string" ? ev.caption.split(".")[0] + "." : ""));
      if (klass.kind === "replay") setTimeout(() => { if (klass?.kind === "replay") endClass(); }, 4000);
      return;
    }
    if (!Number.isInteger(ev.index)) return;
    const word = ev.step === "attempts" ? "Try" : "Test";
    t.hidden = false;
    t.replaceChildren(`${word} ${ev.index + 1}${Number.isInteger(ev.total) ? ` of ${ev.total}` : ""}`,
      el("small", {}, ev.step === "attempts" ? "a real try in the simulator" : "a situation it never saw while learning"));
  }

  function when(iso) {
    const t = Date.parse(iso ?? "");
    return Number.isNaN(t) ? "date not recorded" : new Date(t).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  }

  return {
    setOpen, isOpen: () => open, render, refresh() { if (open) load(); }, onRunEvent, onEpisode, inClass: () => Boolean(klass),
    stats: () => ({ lesson: currentLesson(), story, beat, teach, klass: klass && { kind: klass.kind, runId: klass.runId, state: klass.state } }),
  };
}
