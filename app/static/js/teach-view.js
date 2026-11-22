// Screen 1, Teach: one input to the real chat (Claude on the server), the answer as a friendly confirmation, the lesson
// choice the server offers, the cost question, and what the dog already knows from finished recorded runs.
import { canStart, INITIAL_RUN, runStep } from "./run.js";
import { cardLine, costQuestion, latestDone, lessonChoices, skillKey, skillTitle, understood } from "./plain.js";
import { $, el, setText } from "./dom.js";

export function initTeachView(ctx) {
  const { api, notes } = ctx;
  let messages = [];        // the running conversation sent to /api/chat: only exchanges the server answered
  let chat = null;          // the last answer
  let view = null;          // understood(chat)
  let mode = null;          // the chosen lesson mode
  let run = INITIAL_RUN;    // run.js state machine
  let busy = false;
  let knownKey = "", lastModes = "";

  // ---- asking ----------------------------------------------------------------------------------------------------
  async function send(text) {
    if (!text || busy) return;
    busy = true;
    $("send").disabled = true;
    setText($("send"), "Thinking…");
    $("ask").setAttribute("aria-busy", "true");
    const user = { role: "user", content: text };
    const r = await api.chat([...messages, user]);
    busy = false;
    $("send").disabled = false;
    setText($("send"), "Teach");
    $("ask").setAttribute("aria-busy", "false");
    if (!r.ok) {
      notes.set("chat", "danger", `Chat: ${r.message} (${r.code})`, "Chat");
      chat = null; view = null;
      return renderAnswer(r.unavailable ? "The teacher chat is not available on this server right now." : "The teacher chat did not answer. Try again in a moment.");
    }
    notes.clear("chat");
    messages = [...messages, user, { role: "assistant", content: typeof r.data.reply === "string" ? r.data.reply : "" }];
    chat = r.data;
    view = understood(chat, ctx.catalogue());
    run = runStep(INITIAL_RUN, { type: "spec", valid: view.ok });
    mode = null;
    $("msg").value = "";
    renderAnswer();
  }

  $("ask").addEventListener("submit", (e) => { e.preventDefault(); send($("msg").value.trim()); });
  for (const chip of document.querySelectorAll("[data-example]")) {
    chip.addEventListener("click", () => { $("msg").value = chip.dataset.example; send(chip.dataset.example); });
  }

  // ---- the answer ------------------------------------------------------------------------------------------------
  function renderAnswer(problem = null) {
    const box = $("answer");
    const parts = [];
    if (problem) parts.push(el("p", { class: "answer__lead" }, problem));
    else if (view) {
      if (view.heard) parts.push(el("p", { class: "answer__lead" }, view.ok ? view.heard : view.heard.replace(/^Got it: /, "You asked to ").replace(/\.$/, ".")));
      else parts.push(el("p", { class: "answer__lead" }, "That is not something the dog can learn here yet."));
      if (view.ok && view.settings.length) parts.push(el("ul", { class: "settings" }, ...view.settings.map((s) => el("li", {}, el("span", { class: "settings__k" }, s.label), el("span", { class: "settings__v" }, s.value)))));
      if (view.ok && view.usual) parts.push(el("p", { class: "muted" }, "It will use the usual settings."));
      const cannot = [...view.refused, ...view.unsupported];
      if (cannot.length) parts.push(el("div", { class: "cannot" }, el("p", { class: "cannot__lead" }, "It cannot learn this part yet:"), el("ul", {}, ...cannot.map((t) => el("li", {}, t)))));
      if (view.problems.length) parts.push(el("div", { class: "cannot" }, el("p", { class: "cannot__lead" }, "This needs a change first:"), el("ul", {}, ...view.problems.map((t) => el("li", {}, t)))));
      if (!view.ok && view.reply) parts.push(el("p", { class: "muted small" }, view.reply));
      if (view.ok) parts.push(...lessonPart());
    }
    box.replaceChildren(...parts);
    box.hidden = !parts.length;
    $("send").className = `btn ${view?.ok ? "btn--quiet" : "btn--accent"} ask__send`;
    renderFoot();
  }

  // The lesson choice: only the modes the server accepts, with durations measured by finished recorded runs.
  function lessonPart() {
    const key = view.skill;
    const { choices, quickRefused } = lessonChoices(ctx.state()?.run_modes, key, ctx.cards(), ctx.catalogue());
    const out = [];
    if (!choices.length) {
      notes.set("modes", "warn", `/api/state run_modes has no entry for ${key}.`, "Lesson");
      return [el("p", { class: "muted" }, "The server does not offer a lesson for this skill right now.")];
    }
    notes.clear("modes");
    if (!choices.some((c) => c.mode === mode)) mode = choices[0].mode;
    if (choices.length > 1 || quickRefused) {
      out.push(el("fieldset", { class: "choice" }, el("legend", { class: "choice__legend" }, "How long a lesson?"),
        ...choices.map((c) => {
          const input = el("input", { type: "radio", name: "lesson-mode", value: c.mode, id: `mode-${c.mode}` });
          input.checked = c.mode === mode;
          input.disabled = run.phase === "starting" || run.phase === "confirm";
          input.addEventListener("change", () => { mode = c.mode; });
          const time = c.learn ? `Learns in ${c.learn}${c.test ? `, then tests for ${c.test}` : ""}.` : "Not timed on this computer yet.";
          return el("label", { class: "choice__item", for: `mode-${c.mode}` }, input,
            el("span", { class: "choice__text" }, el("span", { class: "choice__name" }, c.label), el("span", { class: "choice__time" }, time)));
        })));
    } else {
      const c = choices[0];
      out.push(el("p", { class: "muted" }, `${c.label}. ${c.learn ? `Learns in ${c.learn}${c.test ? `, then tests for ${c.test}` : ""}.` : "Not timed on this computer yet."}`));
    }
    if (quickRefused) {
      notes.set("quick", "info", `Quick mode refused for ${key}: ${quickRefused}`, "Lesson");
      const recorded = latestDone(ctx.cards(), key, null, ctx.catalogue());
      out.push(el("div", { class: "long" },
        el("p", {}, "This skill has no quick lesson: a full one takes a long time."),
        recorded ? el("button", { class: "btn btn--quiet", type: "button", on: { click: () => ctx.go("learn", recorded.runId) } }, "Watch the lesson we recorded earlier") : null));
    } else notes.clear("quick");
    return out;
  }

  // ---- the footer: one main button -------------------------------------------------------------------------------
  function renderFoot() {
    const foot = $("teach-foot");
    const items = [];
    if (view?.ok) {
      if (run.phase === "confirm") {
        items.push(el("p", { class: "foot__ask", id: "cost-q" }, run.message));
        items.push(el("div", { class: "foot__row" },
          el("button", { class: "btn btn--accent btn--lg", type: "button", id: "confirm-yes", on: { click: () => request({ type: "confirm" }) } }, "Start the lesson"),
          el("button", { class: "btn btn--quiet", type: "button", on: { click: () => { run = runStep(run, { type: "cancel" }); renderAnswer(); } } }, "Not now")));
      } else if (run.phase === "starting") {
        items.push(el("button", { class: "btn btn--accent btn--lg", type: "button", disabled: true }, "Starting…"));
      } else {
        if (run.phase === "failed" || run.phase === "unavailable") items.push(el("p", { class: "foot__problem" }, run.plain));
        items.push(el("button", { class: "btn btn--accent btn--lg", type: "button", disabled: !canStart(run), on: { click: () => request({ type: "start" }) } }, "Start the lesson"));
        if (run.busyRun) items.push(el("button", { class: "btn btn--quiet", type: "button", on: { click: () => ctx.go("learn", run.busyRun) } }, "Watch the running lesson"));
      }
    }
    foot.replaceChildren(...items);
    foot.hidden = !items.length;
    if (run.phase === "confirm") $("confirm-yes")?.focus({ preventScroll: true });
  }

  async function request(ev) {
    run = runStep(run, ev.type === "start" ? { ...ev, mode } : ev);
    renderAnswer();
    if (run.phase !== "starting") return;
    const r = await api.startRun(chat.declared_spec, run.confirmCost, run.mode);
    const quote = !r.ok && r.status === 409 && r.code === "cost_confirmation_required";
    run = runStep(run, { type: "answer", result: r });
    if (quote) run = { ...run, message: costQuestion(r.data) };
    if (run.phase === "started") {
      notes.clear("run");
      ctx.runStarted(run.runId);
      run = runStep(INITIAL_RUN, { type: "spec", valid: true });
      renderAnswer();
      return ctx.go("learn", r.data.run_id);
    }
    if (run.phase === "failed" || run.phase === "unavailable") {
      notes.set("run", "danger", `Start run: ${r.message} (${r.code})`, "Lesson");
      const busyRun = r.code === "run_in_progress" && typeof r.data?.run_id === "string" ? r.data.run_id : null;
      run = { ...run, busyRun, plain: busyRun ? "Another lesson is still running. Let it finish first."
        : r.code === "mode_not_supported" ? "That kind of lesson is not offered for this skill."
          : run.phase === "unavailable" ? "Lessons cannot start on this server right now." : "The lesson did not start. Details are in the engineers' panel." };
    }
    renderAnswer();
  }

  // ---- what it already knows -------------------------------------------------------------------------------------
  // One card per skill: its newest finished run that has test results, with one headline number from them.
  async function renderKnown() {
    const cards = ctx.cards(), cat = ctx.catalogue();
    const skills = [...new Set(cards.filter((c) => c.state === "done" && c.hasResult).map((c) => skillKey(c.skill, cat)).filter(Boolean))];
    const picked = [];
    for (const skill of skills) {
      const runs = cards.filter((c) => c.state === "done" && c.hasResult && skillKey(c.skill, cat) === skill)
        .sort((a, b) => String(b.updated ?? "").localeCompare(String(a.updated ?? "")));
      for (const c of runs) {
        const body = await ctx.runBody(c.runId);
        const line = body ? cardLine(body.results) : null;
        if (line) { picked.push({ skill, runId: c.runId, line }); break; }
      }
    }
    const key = picked.map((p) => `${p.runId}:${p.line}`).join("|");
    if (key === knownKey) return;
    knownKey = key;
    $("known-list").replaceChildren(...picked.map((p) => el("li", { class: "known__item" },
      el("div", { class: "known__text" }, el("h3", { class: "known__title" }, skillTitle(p.skill)), el("p", { class: "known__line" }, p.line)),
      el("button", { class: "btn btn--quiet btn--sm", type: "button", "aria-label": `See it: ${skillTitle(p.skill)}`, on: { click: () => ctx.go("results", p.runId) } }, "See it"))));
    $("known").hidden = !picked.length;
  }

  return {
    enter() { renderAnswer(); renderKnown(); },
    leave() {},
    cardsChanged() { renderKnown(); if (view?.ok && run.phase !== "confirm" && run.phase !== "starting") renderAnswer(); },
    stateChanged() {
      const modes = JSON.stringify(ctx.state()?.run_modes ?? null);
      if (modes === lastModes) return;
      lastModes = modes;
      if (view?.ok && run.phase !== "confirm" && run.phase !== "starting") renderAnswer();
    },
  };
}
