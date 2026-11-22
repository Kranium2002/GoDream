// "Show this run's episodes": the buttons on the run cards and on the run strip (any element with data-episodes set
// to a run id) call POST /api/episodes/start or /stop, and the caption bar on the stage follows the "episode" events.
// The buttons show as unavailable while /api/state has no "episodes" key.
import { captionView, episodesButton, episodesControl } from "./episodes.js";

export function initEpisodesView({ api, el, notes, reloadState, onStarted }) {
  const $ = (id) => document.getElementById(id);
  let ctrl = episodesControl(null);
  let busy = null; // the run id a request is in flight for

  function refresh() {
    for (const b of document.querySelectorAll("[data-episodes]")) {
      const id = b.dataset.episodes;
      if (!id) continue;
      const view = episodesButton(ctrl, id, busy === id);
      if (b.textContent !== view.label) b.textContent = view.label;
      b.disabled = view.disabled;
      b.setAttribute("aria-pressed", String(view.pressed));
      b.dataset.action = view.action ?? "";
      if (!ctrl.available) b.title = "The server does not offer episode playback yet.";
      else b.removeAttribute("title");
    }
  }

  function setState(state) {
    ctrl = episodesControl(state);
    if (!ctrl.playing) hideCaptionLater();
    refresh();
  }

  async function act(runId, action) {
    busy = runId;
    refresh();
    const r = action === "stop" ? await api.episodesStop() : await api.episodesStart(runId);
    busy = null;
    if (r.ok) {
      notes.clear("episodes");
      if (action === "start") onStarted?.();
    } else {
      notes.set("episodes", r.unavailable ? "warn" : "danger",
        `${action === "stop" ? "The episodes did not stop" : "The run's episodes did not start"}: ${r.message}`, r.unavailable ? "Unavailable" : "Error");
    }
    await reloadState();
    refresh();
  }

  document.addEventListener("click", (e) => {
    const b = e.target.closest?.("[data-episodes]");
    if (!b || !b.dataset.episodes || b.disabled || !b.dataset.action) return;
    act(b.dataset.episodes, b.dataset.action);
  });

  // ---- the caption bar ------------------------------------------------------------------------------------------
  let hideTimer = null;
  function hideCaptionLater() { clearTimeout(hideTimer); hideTimer = setTimeout(() => { $("episode-caption").hidden = true; }, 4000); }

  function onEvent(ev) {
    const v = captionView(ev);
    const bar = $("episode-caption");
    if (v.hidden) { if (ev?.state === "idle") hideCaptionLater(); return; }
    clearTimeout(hideTimer);
    bar.hidden = false;
    $("ep-chip").className = `chip chip--${v.chip.tone}`;
    $("ep-chip").textContent = v.chip.text;
    $("ep-title").textContent = v.title;
    $("ep-text").textContent = v.text;
    $("ep-text").hidden = !v.text;
    const out = $("ep-outcome");
    out.hidden = !v.outcome.length && !v.noOutcome;
    out.replaceChildren(...(v.outcome.length ? v.outcome.map(([k, val]) => el("div", {}, el("dt", {}, k), el("dd", {}, val)))
      : v.noOutcome ? [el("div", {}, el("dd", {}, v.noOutcome))] : []));
    $("ep-note").textContent = v.note;
    $("ep-note").hidden = !v.note;
  }

  return { setState, refresh, onEvent };
}
