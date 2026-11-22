// Teaching it again, pure: whether the server offers a quick lesson for a skill, how long a lesson took when it was
// recorded, the cost question from the server's quote, and the four beats filling in from a live run's event lines.
import { BEATS } from "./story.js";

const finite = (v) => typeof v === "number" && Number.isFinite(v);

// /api/state run_modes for one skill -> {quick: true} when mode "fast" is offered (""), else {quick: false, reason}.
export function quickLesson(runModes, skill) {
  const modes = runModes && typeof runModes === "object" ? runModes[skill] : null;
  if (!modes || typeof modes !== "object" || !("fast" in modes)) return { quick: false, reason: null };
  return modes.fast === "" ? { quick: true, reason: null } : { quick: false, reason: typeof modes.fast === "string" ? modes.fast : null };
}

export function aboutTime(seconds) {
  if (!finite(seconds) || seconds < 0) return null;
  if (seconds < 90) return "about 1 minute";
  if (seconds < 90 * 60) return `about ${Math.round(seconds / 60)} minutes`;
  return `about ${Math.round(seconds / 360) / 10} hours`;
}

// How long a recorded lesson took on this computer, from its run's timing: {learn, test} plain durations, or null.
export function recordedDuration(timing) {
  if (!timing || typeof timing !== "object" || !finite(timing.learn_s)) return null;
  return { learn: aboutTime(timing.learn_s), test: aboutTime(timing.evaluate_s), mode: typeof timing.mode === "string" ? timing.mode : null };
}

const thousands = (n) => Math.round(n).toLocaleString("en-US");
// The 409 cost_confirmation_required body -> one plain sentence from its estimate, or null when it holds none.
export function costSentence(body) {
  const e = body?.estimate;
  const tries = finite(e?.episodes) ? `about ${thousands(e.episodes)} simulated tries` : null;
  const credits = finite(e?.tabpfn_tokens?.value) ? `about ${thousands(e.tabpfn_tokens.value)} TabPFN credits` : null;
  if (tries && credits) return `This lesson runs ${tries} and uses ${credits}.`;
  if (tries) return `This lesson runs ${tries}; its TabPFN credits were not estimated.`;
  if (credits) return `This lesson uses ${credits}.`;
  return null;
}

// The run's pipeline steps behind each beat.
const STEPS_OF = { tries: ["baseline", "attempts"], whatIf: ["teacher_fit", "dream", "teach"], brain: ["train"], test: ["evaluate", "run_test"] };
const ENDED = ["done", "failed", "refused", "stopped"];

// A learn.js timeline -> the four beats as they fill in: [{n, key, title, state, progress: {done, total} | null}].
// state: waiting | running | done | stopped. A step whose last line says running in a run that has ended finished with
// the run when a later beat began or the run is done, and stopped there otherwise.
export function liveBeats(tl, runState = null) {
  const ended = ENDED.includes(runState);
  const logged = BEATS.map((b) => STEPS_OF[b.key].some((k) => tl?.steps?.[k]));
  return BEATS.map((b, i) => {
    const entries = STEPS_OF[b.key].map((k) => tl?.steps?.[k]).filter(Boolean);
    const last = entries.reduce((a, e) => (a === null || (e.t ?? -Infinity) >= (a.t ?? -Infinity) ? e : a), null);
    let state = last ? last.state : "waiting";
    if (state === "running" && ended) state = runState === "done" || logged.slice(i + 1).some(Boolean) ? "done" : "stopped";
    if (["failed", "refused", "stopped"].includes(state)) state = "stopped";
    if (!["waiting", "running", "done", "stopped"].includes(state)) state = "running";
    const counted = [...entries].reverse().find((e) => finite(e.done) && finite(e.total) && e.total > 0);
    return { n: b.n, key: b.key, title: b.title, state, progress: counted ? { done: counted.done, total: counted.total } : null };
  });
}
