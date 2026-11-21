// The learning flow's pure parts: the pipeline track under the robot, the replay schedule of a recorded run, and the
// skill cards of runs on disk.

const EXPLAIN = {
  baseline: "Measures the do-nothing and physics-only rules on simulator scenes, before any learning.",
  attempts: "Runs real MuJoCo episodes with the walking policy to collect outcomes.",
  teacher_fit: "Fits the PFN teacher to the measured attempts: it learns where the physics guess is wrong.",
  dream: "Asks the teacher to score many simulated situations: the dreams the student learns in.",
  train: "Trains the spiking network student on the teacher's dreams.",
  evaluate: "Checks the teacher and the student on held-out simulator scenes.",
};

// Timeline rows (learn.js timelineRows) -> track stages. fraction only from a "done of total" the run reported.
// runState: the run's state when known. Once the run has ended, a step whose last line still says "running" never
// wrote an end line (run_skill logs a step's start; some commands log nothing after it): it is shown as such, neither
// as running nor as done.
export function trackStages(rows, runState = null) {
  const ended = ["done", "failed", "refused", "stopped"].includes(runState);
  return rows.map((r) => {
    const m = typeof r.progress === "string" ? r.progress.match(/^(\d+(?:\.\d+)?) of (\d+(?:\.\d+)?)$/) : null;
    const unended = ended && r.state === "running";
    return {
      step: r.step,
      label: r.label,
      explain: EXPLAIN[r.step] ?? "",
      state: r.state,
      stateText: unended ? "started; no end line recorded" : r.state ?? "no event yet",
      progress: r.progress,
      fraction: m && Number(m[2]) > 0 ? Math.min(1, Number(m[1]) / Number(m[2])) : null,
      running: r.state === "running" && !unended,
    };
  });
}

// Recorded lines -> [{line, atMs}]: each line at its time_utc gap from the first, divided by speed, in file order.
// Time never runs backwards in the schedule; a line without a readable time comes at the previous line's moment.
export function replayPlan(lines, speed) {
  if (!(typeof speed === "number" && speed > 0)) throw new Error("The replay speed must be a number above 0.");
  let start = null, last = 0;
  return lines.map((line) => {
    const t = typeof line?.time_utc === "string" ? Date.parse(line.time_utc) : NaN;
    if (!Number.isNaN(t)) {
      start ??= t;
      last = Math.max(last, (t - start) / speed);
    }
    return { line, atMs: last };
  });
}

// GET /api/runs?summary=1 -> cards; the plain list of ids -> bare cards; anything else -> none. A card's skill is the
// catalogue's display name when the server gives one (MOVE), else the run's spec (move); timing is the summary's
// timing (Dev B's measured step times) as sent, or null.
export function cardsFrom(data) {
  if (!Array.isArray(data)) return [];
  const seen = new Set();
  const cards = [];
  for (const item of data) {
    const c = typeof item === "string"
      ? { runId: item, skill: null, state: null, updated: null, stepsDone: null, hasResult: null, hasStudent: null, timing: null }
      : item && typeof item.run_id === "string"
        ? { runId: item.run_id, skill: item.display_name ?? item.skill ?? null, state: item.state ?? null, updated: item.updated_utc ?? null,
            stepsDone: Number.isFinite(item.steps_done) ? item.steps_done : null, hasResult: item.has_result ?? null, hasStudent: item.has_student ?? null,
            timing: item.timing && typeof item.timing === "object" ? item.timing : null }
        : null;
    if (c && c.runId && !seen.has(c.runId)) { seen.add(c.runId); cards.push(c); }
  }
  return cards;
}

// A run's skill id (a catalogue name or a SkillSpec module such as dodge_vision) -> the catalogue's name and description.
export function skillLabel(catalogue, id) {
  const entry = Array.isArray(catalogue) ? catalogue.find((s) => s && (s.skill === id || s.spec === id)) : null;
  return entry ? { name: entry.skill, description: typeof entry.description === "string" ? entry.description : "" } : null;
}
