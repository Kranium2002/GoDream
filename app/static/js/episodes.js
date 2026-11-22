// A run's own episodes replayed on the robot stage: the control's state from /api/state "episodes", and the caption
// bar from the SSE "episode" events. Pure; the page shows exactly what these return.

const STEPS = { attempts: "real attempt", baseline: "baseline scene", evaluate: "held-out evaluation scene" };
const fmt = (v) => (typeof v === "number" && Number.isFinite(v) ? String(Number(v.toPrecision(4))) : v === true ? "yes" : v === false ? "no" : v === null ? "none" : typeof v === "string" ? v : JSON.stringify(v));

// /api/state -> {available, playing, runId, note}. Without an "episodes" key the server does not offer playback.
export function episodesControl(state) {
  const e = state && typeof state === "object" ? state.episodes : undefined;
  if (!e || typeof e !== "object") return { available: false, playing: false, runId: null, note: "" };
  return { available: true, playing: e.playing === true, runId: typeof e.run_id === "string" ? e.run_id : null, note: typeof e.note === "string" ? e.note : "" };
}

// The button for one run: {label, disabled, action: "start" | "stop" | null, pressed}.
export function episodesButton(ctrl, runId, busy = false) {
  if (!ctrl.available) return { label: "Episodes: unavailable", disabled: true, action: null, pressed: false };
  if (busy) return { label: "Asking the server", disabled: true, action: null, pressed: false };
  const mine = ctrl.playing && ctrl.runId === runId;
  return mine
    ? { label: "Stop showing episodes", disabled: false, action: "stop", pressed: true }
    : { label: "Show this run's episodes", disabled: false, action: "start", pressed: false };
}

// An "episode" event -> the caption bar, or {hidden: true} for "idle". Every field is shown as the event sends it.
export function captionView(ev) {
  if (!ev || typeof ev !== "object" || ev.state === "idle" || !["start", "end"].includes(ev.state)) return { hidden: true };
  const step = STEPS[ev.step] ?? (typeof ev.step === "string" ? ev.step : null);
  // index counts from 0 (app/player.py numbers the plan's episodes with enumerate); people count from 1.
  const count = Number.isInteger(ev.index) ? `${ev.index + 1}${Number.isInteger(ev.total) ? ` of ${ev.total}` : ""}` : null;
  const title = [typeof ev.skill === "string" ? ev.skill : null, [step, count].filter(Boolean).join(" ") || null,
    typeof ev.kind === "string" && ev.kind ? ev.kind : null].filter(Boolean).join(" · ");
  const outcome = ev.state === "end" && ev.outcome && typeof ev.outcome === "object" && !Array.isArray(ev.outcome)
    ? Object.entries(ev.outcome).map(([k, v]) => [k, fmt(v)]) : [];
  return {
    hidden: false,
    chip: ev.state === "start" ? { text: "Playing", tone: "accent" } : { text: "Ended", tone: "neutral" },
    title: title || "Episode",
    text: [typeof ev.action === "string" && ev.action ? `Action: ${ev.action}` : null, typeof ev.caption === "string" && ev.caption ? ev.caption : null].filter(Boolean).join(". "),
    outcome,
    noOutcome: ev.state === "end" && !outcome.length ? "The event carried no outcome." : null,
    note: typeof ev.note === "string" ? ev.note : "",
  };
}
