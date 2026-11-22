import { test } from "node:test";
import assert from "node:assert/strict";
import { applyBrain, brainView, choiceWords, emptyBrain, fmtClock, fmtInt, fmtMetres, fmtMinutes, fmtMs, fmtSeconds, heroSub, homeSteps,
  laneResult, learnedIn, median, readLanes, readShowcase, rowValue, sessionRows, statsBand, storySteps } from "../../app/static/js/showcase.js";
import { readDemos } from "../../app/static/js/demos.js";

// Shapes as the /api/demos contract writes them; the numbers are made up for the test, not results.
const showcase = {
  results: { n: 30, split: "test", rows: [
    { field: "crossed_goal_line", label: "Reached the finish line", kind: "count", before: 12, after: 28, of: 30 },
    { field: "some_amount_m", label: "A measured amount", kind: "metres", before: 4.105, after: 0.678, better: "lower" }] },
  learning: { real_tries: 48, imagined: 1385, neurons: 64, train_s: 14.4, learn_s: 393.2, decision_ms: 0.13, spikes_per_decision: 98.6 },
};
const raw = (id, extra = {}) => ({ id, title: id === "hills" ? "Feel the slope" : "See the punch", name: id === "hills" ? "Hills" : "Dodge", tagline: "One sentence.",
  lanes: id === "hills" ? { before: "Walking alone", after: "With the learned reflex" } : { before: "Untrained", after: "Trained" },
  handles: [], outcome_fields: [], sense: "feel", available: true, missing: [], ...extra });

test("the showcase is read as sent, and a missing or unreadable part is null, never filled in", () => {
  const s = readShowcase(showcase);
  assert.deepEqual(s.learning, { realTries: 48, imagined: 1385, neurons: 64, trainS: 14.4, learnS: 393.2, decisionMs: 0.13, spikesPerDecision: 98.6 });
  assert.equal(s.results.n, 30);
  assert.equal(s.results.rows.length, 2);
  assert.equal(s.results.rows[1].better, "lower");
  const empty = readShowcase({ results: null, learning: { real_tries: null, imagined: "many", neurons: -3, train_s: Number.NaN, learn_s: null, decision_ms: undefined } });
  assert.equal(empty.results, null);
  assert.deepEqual(Object.values(empty.learning), [null, null, null, null, null, null, null]);
  assert.equal(readShowcase(undefined).results, null);
  // a count row without "of", or a row without numbers, is left out
  const bad = readShowcase({ results: { n: 3, rows: [{ field: "a", label: "A", kind: "count", before: 1, after: 2 }, { field: "b", label: "B", kind: "metres", before: null, after: 1 }] } });
  assert.equal(bad.results, null);
  assert.deepEqual(readLanes({ before: " Untrained ", after: "" }), { before: "Untrained", after: null });
});

test("readDemos carries the short name, the lanes and the showcase", () => {
  const d = readDemos([raw("hills", { showcase }), raw("dodge")]).demos;
  assert.equal(d[0].name, "Hills");
  assert.deepEqual(d[0].lanes, { before: "Walking alone", after: "With the learned reflex" });
  assert.equal(d[0].showcase.learning.imagined, 1385);
  assert.equal(d[1].showcase.results, null);
  assert.equal(d[1].showcase.learning.decisionMs, null);
  assert.equal(readDemos([{ ...raw("x"), name: undefined, lanes: undefined }]).demos[0].name, null);
});

test("numbers read as people read them", () => {
  assert.equal(fmtInt(1385), "1,385");
  assert.equal(fmtMs(0.1312), "0.13");
  assert.equal(fmtMs(0.0123), "0.012");
  assert.equal(fmtMs(1.54), "1.5");
  assert.equal(fmtMs(1.02), "1.0");
  assert.equal(fmtMs(12.3), "12");
  assert.equal(fmtSeconds(14.4), "14 s");
  assert.equal(fmtSeconds(4.25), "4.3 s");
  assert.equal(fmtClock(393.2), "6:33");
  assert.equal(fmtClock(59.7), "1:00");
  assert.equal(fmtMinutes(393.2), "6.6");
  assert.equal(fmtMinutes(1500), "25");
  assert.equal(fmtMetres(0.678), "0.68 m");
  assert.equal(fmtMetres(4.105), "4.1 m");
  assert.equal(fmtMetres(-0.4), "−0.4 m");
  assert.equal(fmtMs(null), "");
  const rows = readShowcase(showcase).results.rows;
  assert.equal(rowValue(rows[0], "after"), "28 of 30");
  assert.equal(rowValue(rows[1], "before"), "4.1 m");
});

test("a choice name becomes plain words; the yaw sign follows the command schema (positive turns left)", () => {
  assert.equal(choiceWords("yaw+0.70_vx0.60"), "Steer left");
  assert.equal(choiceWords("yaw-0.70_vx0.40"), "Steer right");
  assert.equal(choiceWords("yaw+0.00_vx0.60"), "Straight on");
  assert.equal(choiceWords("DODGE_LEFT"), "Step left");
  assert.equal(choiceWords("DODGE_RIGHT"), "Step right");
  assert.equal(choiceWords("NO_OP"), "Watching");
  assert.equal(choiceWords("RETREAT"), "Step back");
  assert.equal(choiceWords("UNCERTAIN"), "Not sure yet");
  assert.equal(choiceWords("SIT_DOWN"), "Sit down");
  assert.equal(choiceWords(""), null);
  assert.equal(choiceWords(7), null);
});

test("a lane's result is one plain result: the server's word first, else reached the finish line or not, a fall first", () => {
  assert.deepEqual(laneResult({ collision: 1, fall: 0, min_separation_m: 0, result: "Hit" }), { word: "Hit", big: "Hit", detail: null, tone: "bad" });
  // no distance line under the word
  assert.deepEqual(laneResult({ collision: 0, fall: 0, min_separation_m: 0.142, result: "Dodged" }), { word: "Dodged", big: "Dodged", detail: null, tone: "good" });
  assert.equal(laneResult({ result: "Not hit" }).tone, "neutral");
  // a finish-line round: one yes or no, whatever else the outcome holds
  assert.deepEqual(laneResult({ crossed_goal_line: 1, fall: 0, end_s: 11.2 }), { word: "Reached", big: "Reached the finish line", detail: null, tone: "good" });
  assert.deepEqual(laneResult({ crossed_goal_line: 0, fall: 0 }), { word: "Not reached", big: "Did not reach the finish line", detail: null, tone: "neutral" });
  assert.deepEqual(laneResult({ crossed_goal_line: 1, fall: 1 }), { word: "Not reached", big: "Fell", detail: null, tone: "bad" }); // a fall wins
  assert.equal(laneResult({ fall: 1 }).word, "Fell");
  assert.equal(laneResult({ something: 1 }), null);
  assert.equal(laneResult(null), null);
});

test("the brain readout keeps the learned lane's last decisions and shows the median of the newest", () => {
  let b = emptyBrain();
  assert.equal(brainView(b), null);
  b = applyBrain(b, { side: "before", decision_ms: 9, choice: "NO_OP" }); // the other lane: ignored
  b = applyBrain(b, { side: "after", decision_ms: "fast" });              // unreadable: ignored
  assert.equal(b.list.length, 0);
  for (let k = 0; k < 50; k++) b = applyBrain(b, { side: "after", t_s: k * 0.1, decision_ms: k < 30 ? 5 : 0.1 + (k % 3) * 0.01, choice: k === 49 ? "DODGE_LEFT" : "NO_OP", issued: k === 49 });
  assert.equal(b.list.length, 40);
  const v = brainView(b);
  assert.equal(v.choice, "Step left");
  assert.equal(v.ms, 0.11); // the last 20 are all from k >= 30
  assert.equal(v.spikes, null); // no spike counts were sent
  b = applyBrain(b, { side: "after", decision_ms: 0.1, choice: "NO_OP", spikes: 97 });
  assert.equal(brainView(b).spikes.at(-1), 97);
  // dodge: an opinion the gate did not act on is not shown; an issued move stays MOVE_SHOW_S, then "Watching"
  let d = emptyBrain();
  d = applyBrain(d, { side: "after", t_s: 1.0, decision_ms: 1, choice: "UNCERTAIN", issued: false });
  assert.equal(brainView(d).choice, "Watching");
  d = applyBrain(d, { side: "after", t_s: 1.1, decision_ms: 1, choice: "DODGE_RIGHT", issued: true });
  d = applyBrain(d, { side: "after", t_s: 1.6, decision_ms: 1, choice: "UNCERTAIN", issued: false });
  assert.equal(brainView(d).choice, "Step right");
  d = applyBrain(d, { side: "after", t_s: 2.2, decision_ms: 1, choice: "DODGE_LEFT", issued: false });
  assert.equal(brainView(d).choice, "Watching");
  // hills: every decision is a command
  let h = applyBrain(emptyBrain(), { side: "after", t_s: 0.5, decision_ms: 1, choice: "yaw+0.70_vx0.60" });
  assert.equal(brainView(h).choice, "Steer left");
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(median([]), null);
});

test("the story steps use the run's numbers and keep their words when a number is missing", () => {
  const s = storySteps(readShowcase(showcase).learning);
  assert.deepEqual(s.map((x) => `${x.strong ?? ""}${x.rest}`), ["48 real tries", "1,385 situations imagined by TabPFN", "64 neurons in a spiking brain, trained in 14 s", "0.13 ms for the brain alone to decide"]);
  const none = storySteps(readShowcase({}).learning);
  assert.deepEqual(none.map((x) => x.strong), [null, null, null]);
  assert.deepEqual(none.map((x) => x.rest), ["Real tries in the simulator", "Situations imagined by TabPFN", "A spiking brain"]);
  assert.equal(learnedIn(readShowcase(showcase).learning), "Learned in 6.6 minutes, start to finish");
  assert.equal(learnedIn(readShowcase({ learning: { learn_s: 42 } }).learning), "Learned in 42 s, start to finish");
  assert.equal(learnedIn(readShowcase({}).learning), null);
});

test("the landing page shows only numbers that exist", () => {
  const demos = readDemos([raw("hills", { showcase }), raw("dodge", { showcase: { results: null, learning: { real_tries: 40, decision_ms: 0.2 } } })]).demos;
  const steps = homeSteps(demos);
  assert.deepEqual(steps[0].values.map((v) => [v.demo, v.value]), [["Hills", "48"], ["Dodge", "40"]]);
  assert.deepEqual(steps[1].values.map((v) => v.demo), ["Hills"]);
  const stats = statsBand(demos);
  assert.deepEqual(stats.map((s) => s.value), ["0.13", "6.6", "28 of 30"]);
  assert.equal(stats[0].label, "milliseconds for the spiking brain to decide, on this laptop");
  assert.equal(stats[1].label, "minutes to learn the hills, start to finish");
  assert.equal(stats[2].label, "reached the finish line with the learned reflex");
  assert.equal(stats[2].sub, "12 of 30 walking alone");
  assert.deepEqual(statsBand(readDemos([raw("dodge")]).demos), []);
  assert.match(heroSub(demos), /fraction of a millisecond\.$/);
  assert.match(heroSub(demos), /imagines many times more situations/); // hills 48 vs 1,385; dodge has no imagined count
  assert.match(heroSub(readDemos([raw("hills", { showcase: { learning: { real_tries: 48, imagined: 300 } } })]).demos), /imagines more situations/);
  assert.doesNotMatch(heroSub([]), /millisecond/);
  assert.doesNotMatch(heroSub(readDemos([raw("hills", { showcase: { learning: { decision_ms: 2.5 } } })]).demos), /millisecond/);
});

test("this session's rows count real rounds only", () => {
  assert.deepEqual(sessionRows(undefined), { rounds: 0, rows: [] });
  const s = sessionRows({ before: { Hit: 3, "Not hit": 1 }, after: { Dodged: 3, Hit: 1 } });
  assert.equal(s.rounds, 4);
  assert.deepEqual(s.rows, [{ label: "Dodged", before: "0 of 4", after: "3 of 4", good: true }, { label: "Hit", before: "3 of 4", after: "1 of 4", good: false }, { label: "Not hit", before: "1 of 4", after: "0 of 4", good: false }]);
  // both dogs hit: the dodged row still shows, at 0
  assert.deepEqual(sessionRows({ before: { Hit: 1 }, after: { Hit: 1 } }).rows.map((r) => [r.label, r.after]), [["Dodged", "0 of 1"], ["Hit", "1 of 1"]]);
  // a finish-line round: one row, reached the finish line or not
  assert.deepEqual(sessionRows({ before: { "Not reached": 2 }, after: { Reached: 2 } }).rows, [{ label: "Reached the finish line", before: "0 of 2", after: "2 of 2", good: true }]);
});
