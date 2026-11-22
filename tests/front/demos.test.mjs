import { test } from "node:test";
import assert from "node:assert/strict";
import { capProfile, defaultParams, fraction, handleText, headlineSentence, outcomeLine, readDemos, readHandle, snap, valueText, wordsOf } from "../../app/static/js/demos.js";
import { applyDuel, duelShows, emptyDuel, sideOf } from "../../app/static/js/duel.js";

// Shapes as the v1 contract writes them; the numbers are made up for the test, not results.
const slope = { name: "slope_deg", kind: "slope", label: "Hill steepness", unit: "deg", min: 4, max: 24, step: 2, default: 10 };
const demo = {
  id: "hills", title: "Hills", tagline: "Blind, by feel.", skill: "RAMP", run_id: "r1",
  headline: { metric_label: "Reached the spot", unit: "fraction", before: 0.5, after: 0.75, n: 40 },
  handles: [slope, { name: "offset_m", kind: "offset", label: "Sideways", unit: "m", min: -1, max: 1, step: 0.1, default: 0 }],
  outcome_fields: [{ name: "reached", kind: "flag", label: "Reached the spot", tone: "good" }, { name: "t", kind: "seconds", label: "in" }],
  sense: "feel", live_camera: false, available: true, missing: [],
};

test("a demo list is read with every declared part and nothing added", () => {
  const r = readDemos([demo]);
  assert.equal(r.ok, true);
  assert.deepEqual(r.problems, []);
  const d = r.demos[0];
  assert.equal(d.id, "hills");
  assert.equal(d.runId, "r1");
  assert.equal(d.sense, "feel");
  assert.equal(d.handles.length, 2);
  assert.deepEqual(d.headline, { metricLabel: "Reached the spot", unit: "fraction", before: 0.5, after: 0.75, n: 40 });
  assert.deepEqual(defaultParams(d), { slope_deg: 10, offset_m: 0 });
});

test("an answer that is not a list is refused; unreadable demos and handles are left out with a problem", () => {
  assert.equal(readDemos({ demos: [] }).ok, false);
  const r = readDemos([{ id: "x" }, demo, demo, { ...demo, id: "y", handles: [{ ...slope, kind: "wheel" }, { ...slope, min: 5, max: 5 }] }]);
  assert.deepEqual(r.demos.map((d) => d.id), ["hills", "y"]);
  assert.equal(r.demos[1].handles.length, 0);
  assert.equal(r.problems.length, 4);
  assert.match(r.problems.join(" "), /wheel/);
});

test("a handle needs a known kind, a range, a step and a default inside the range", () => {
  assert.equal(readHandle(slope).ok, true);
  assert.equal(readHandle({ ...slope, step: 0 }).ok, false);
  assert.equal(readHandle({ ...slope, default: 30 }).ok, false);
  assert.equal(readHandle({ ...slope, label: "" }).ok, false);
  assert.match(readHandle({ ...slope, kind: "dial" }).problem, /slope, offset, direction, amount/);
});

test("values snap to the handle's grid and stay in range; text carries the unit", () => {
  const h = readHandle(slope).handle;
  assert.equal(snap(h, 11.2), 12);
  assert.equal(snap(h, 99), 24);
  assert.equal(snap(h, -3), 4);
  assert.equal(fraction(h, 14), 0.5);
  assert.equal(handleText(h, 12), "12°");
  const o = readHandle(demo.handles[1]).handle;
  assert.equal(snap(o, 0.33), 0.3);
  assert.equal(handleText(o, -0.3), "−0.3 m");
  assert.equal(valueText(1.5, "m/s"), "1.5 m/s");
});

test("the hill's cross-section has the declared rim slope", () => {
  const p = capProfile(30, { cx: 100, ground: 60, a: 40 });
  // R = a / sin 30 = 80, h = R (1 - cos 30) = 10.72
  assert.equal(Math.round(p.height * 100) / 100, 10.72);
  assert.match(p.path, /A 80 80/);
  assert.equal(capProfile(0, { cx: 0, ground: 0, a: 1 }), null);
  assert.ok(capProfile(20, { cx: 0, ground: 50, a: 40 }).height < capProfile(40, { cx: 0, ground: 50, a: 40 }).height);
});

test("an outcome line is built from the typed fields only", () => {
  const fields = readDemos([demo]).demos[0].outcomeFields;
  assert.deepEqual(outcomeLine(fields, { reached: true, t: 9.24 }), { text: "Reached the spot", detail: "in 9.2\u00a0s", tone: "good" });
  assert.deepEqual(outcomeLine(fields, { reached: false, t: 12 }), { text: "in 12\u00a0s", detail: null, tone: "neutral" });
  assert.equal(outcomeLine(fields, { other: 1 }), null);
  const dodge = [{ name: "hit", kind: "flag", label: "Got hit", tone: "bad" }, { name: "aside", kind: "flag", label: "Stepped aside in time", tone: "good" }];
  assert.deepEqual(outcomeLine(dodge, { hit: 1, aside: 0 }), { text: "Got hit", detail: null, tone: "bad" });
  assert.deepEqual(outcomeLine(dodge, { hit: false, aside: true }), { text: "Stepped aside in time", detail: null, tone: "good" });
  // labels as the server sends them today: names, no tone
  const named = [{ name: "reached", kind: "flag", label: "Reached" }, { name: "end_s", kind: "seconds", label: "Time" }, { name: "side", kind: "metres", label: "Side offset" }];
  assert.deepEqual(outcomeLine(named, { reached: 1, end_s: 9.21, side: 0.13 }), { text: "Reached", detail: "Time 9.2\u00a0s, side offset 0.1\u00a0m", tone: "neutral" });
});

test("a label sent as an identifier is shown in words", () => {
  assert.equal(wordsOf("collision_rate"), "Collision rate");
  assert.equal(wordsOf("Reached the spot"), "Reached the spot");
  assert.equal(readDemos([{ ...demo, headline: { ...demo.headline, metric_label: "reached_rate" } }]).demos[0].headline.metricLabel, "Reached rate");
});
test("the result sentence states counts out of n for a fraction", () => {
  const d = readDemos([demo]).demos[0];
  assert.equal(headlineSentence(d.headline), "Tested on 40 situations it never saw while learning. Reached the spot: 20 of 40 before, 30 of 40 with the learned reflex.");
  assert.equal(headlineSentence(null), null);
});

test("duel events give both lanes' state; outcomes only for a lane that is done", () => {
  let s = emptyDuel();
  s = applyDuel(s, { state: "start", demo: "hills", params: { slope_deg: 10 }, seed: 3, sides: { before: { status: "running", outcome: { reached: 1 } }, after: { status: "running" } } });
  assert.equal(s.state, "start");
  assert.equal(s.sides.before.outcome, null);
  assert.equal(duelShows(s, "hills"), true);
  s = applyDuel(s, { state: "end", demo: "hills", sides: { before: { status: "done", outcome: { reached: 0 } }, after: { status: "done", outcome: { reached: 1 } } } });
  assert.deepEqual(s.sides.after.outcome, { reached: 1 });
  assert.equal(s.seed, 3);
  assert.equal(applyDuel(s, { state: "bogus" }), s);
  s = applyDuel(s, { state: "idle" });
  assert.equal(s.state, "idle");
  assert.equal(duelShows(s, "hills"), false);
  assert.equal(sideOf({ side: "after" }), "after");
  assert.equal(sideOf({ side: "left" }), null);
  assert.equal(sideOf({}), null);
});
