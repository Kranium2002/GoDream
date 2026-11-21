// The start-run state machine, pure. Phases:
//   blocked      no valid spec                     ready     a valid spec, start allowed
//   starting     POST /api/runs in flight          confirm   409 cost_confirmation_required: {message, quote}
//   started      202 {run_id}: {runId}             failed / unavailable   {code, message}; start allowed again
// Events: {type: "spec", valid}, {type: "start"}, {type: "confirm"}, {type: "cancel"}, {type: "answer", result}
// where result is an api.js result.

export const INITIAL_RUN = Object.freeze({ phase: "blocked" });

export const canStart = (s) => s.phase === "ready" || s.phase === "failed" || s.phase === "unavailable";

export function runStep(s, ev) {
  switch (ev.type) {
    case "spec":
      return s.phase === "starting" ? s : { phase: ev.valid ? "ready" : "blocked" };
    case "start":
      return canStart(s) ? { phase: "starting", confirmCost: false } : s;
    case "confirm":
      return s.phase === "confirm" ? { phase: "starting", confirmCost: true } : s;
    case "cancel":
      return s.phase === "confirm" ? { phase: "ready" } : s;
    case "answer":
      return s.phase === "starting" ? answer(ev.result) : s;
    default:
      return s;
  }
}

function answer(r) {
  if (r.ok) {
    const id = r.data?.run_id;
    return typeof id === "string" && id
      ? { phase: "started", runId: id }
      : { phase: "failed", code: "invalid_response", message: "The server accepted the run but sent no run id." };
  }
  if (r.status === 409 && r.code === "cost_confirmation_required") {
    return { phase: "confirm", message: r.message, quote: quoteRows(r.data) };
  }
  return { phase: r.unavailable ? "unavailable" : "failed", code: r.code, message: r.message };
}

// The quoted usage, exactly as the server sent it: every field but code and message, nested objects as dotted keys.
export function quoteRows(body, prefix = "") {
  if (body === null || typeof body !== "object") return [];
  return Object.entries(body).flatMap(([key, value]) => {
    if (!prefix && (key === "code" || key === "message")) return [];
    const name = prefix + key;
    if (value !== null && typeof value === "object" && !Array.isArray(value)) return quoteRows(value, `${name}.`);
    return [[name, typeof value === "string" ? value : JSON.stringify(value)]];
  });
}
