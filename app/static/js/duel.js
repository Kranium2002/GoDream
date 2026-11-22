// The side-by-side run of one demo ("duel", server contract v1), pure: which lane an event belongs to, and the state of
// both lanes from the "duel" events. A lane's outcome is shown only once the server says that lane is done.

export const SIDES = ["before", "after"];

// "before" | "after" for an event of a lane; null for the single stage (no side, as before duels existed).
export const sideOf = (ev) => (SIDES.includes(ev?.side) ? ev.side : null);

const lane = (x) => ({
  status: x?.status === "running" || x?.status === "done" ? x.status : null,
  outcome: x?.status === "done" && x?.outcome && typeof x.outcome === "object" && !Array.isArray(x.outcome) ? x.outcome : null,
});

export const emptyDuel = () => ({ state: "idle", demo: null, params: null, seed: null, sides: { before: lane(null), after: lane(null) } });

// One "duel" event -> the next state. An unreadable event changes nothing.
export function applyDuel(prev, ev) {
  if (!ev || typeof ev !== "object" || !["start", "end", "idle"].includes(ev.state)) return prev;
  if (ev.state === "idle") return { ...emptyDuel(), last: prev.state === "idle" ? prev.last ?? null : { ...prev, state: "end" } };
  const sides = ev.sides && typeof ev.sides === "object" ? ev.sides : {};
  return {
    state: ev.state,
    demo: typeof ev.demo === "string" ? ev.demo : prev.demo,
    params: ev.params && typeof ev.params === "object" ? ev.params : prev.params,
    seed: Number.isInteger(ev.seed) ? ev.seed : prev.seed,
    sides: { before: lane(sides.before), after: lane(sides.after) },
  };
}

// Whether the comparison of `demo` is on stage now (running, or ended and still showing its outcome).
export const duelShows = (duel, demoId) => duel.state !== "idle" && duel.demo === demoId;
