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

// The running tally of this page session, {demo: {before: {word: n}, after: {word: n}}}: when a round goes from
// running (prev) to ended (next), each lane whose real outcome has a result word adds one to it. wordOf(outcome) gives
// that word; by default the server's own result word. Anything else (the same end seen again, a page opened on an
// ended round, a stopped lane) changes nothing.
const serverWord = (outcome) => outcome?.result;
export function countRound(tally, prev, next, wordOf = serverWord) {
  if (prev?.state !== "start" || next?.state !== "end" || !next.demo || next.demo !== prev.demo) return tally;
  const old = tally[next.demo] ?? { before: {}, after: {} };
  const mine = { before: { ...old.before }, after: { ...old.after } };
  for (const side of SIDES) {
    const outcome = next.sides?.[side]?.outcome;
    const word = outcome ? wordOf(outcome) : null;
    if (typeof word === "string" && word) mine[side][word] = (mine[side][word] ?? 0) + 1;
  }
  return { ...tally, [next.demo]: mine };
}

const WORD_ORDER = ["Hit", "Dodged", "Not hit", "Fell"];
// One lane's counts in words, "3 hit, 1 dodged"; a finish-line lane (words Reached, Not reached) as one yes-or-no
// count, "reached 2 of 3"; "" before its first counted round.
export function tallyText(counts) {
  const shown = Object.entries(counts ?? {}).filter(([, n]) => n > 0);
  if (shown.some(([word]) => word === "Reached" || word === "Not reached")) {
    return `reached ${counts.Reached ?? 0} of ${shown.reduce((a, [, n]) => a + n, 0)}`;
  }
  return shown.sort(([a], [b]) => (WORD_ORDER.indexOf(a) + 1 || 99) - (WORD_ORDER.indexOf(b) + 1 || 99))
    .map(([word, n]) => `${n} ${word.toLowerCase()}`).join(", ");
}
