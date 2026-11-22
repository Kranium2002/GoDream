import { test } from "node:test";
import assert from "node:assert/strict";
import { beatsOf, choiceAxis, curvePaths, endings, readStory, storyFromRun, testNotes, testRows, topView } from "../../app/static/js/story.js";
import { aboutTime, costSentence, liveBeats, quickLesson, recordedDuration } from "../../app/static/js/lesson.js";
import { applyLine, emptyTimeline } from "../../app/static/js/learn.js";

// Shapes as the v1 contract and the recorded result files write them; the numbers are made up for the test.
const metric = (name, unit, hib, before, after, n, extra = {}) => ({ name, unit, higher_is_better: hib, ...extra,
  values: { no_op: { value: before, ci_low: null, ci_high: null, n }, snn: { value: after, ci_low: after - 0.1, ci_high: after + 0.1, n } } });

test("a story is mapped beat by beat, keeping only readable parts", () => {
  const s = readStory({
    attempts: { n_tries: 40, n_decisions: 28, paths: [{ points: [[0, 0], [1, 0.2]], ended: "reached" }, { points: [[0, 0]], ended: "x" }], outcomes: null, obstacles: [{ x: 1, y: 0, r: 0.5 }, { x: "a" }] },
    what_if: [{ moment: { try: 3, t_s: 1.2 }, choices: [{ label: "Slow", tried: true, predicted: { value: 0.6, low: 0.4, high: 0.8, unit: "", label: "Chance to reach" }, measured: { value: 1 } }, { label: "Fast", tried: false, predicted: { value: 0.3 } }, { label: "Bad" }] }],
    imagination: { rows: [{ label: "Progress", unit: "m", tabpfn_error: 0.2, formula_error: 0.9, n: 12 }, { label: "x" }] },
    brain: { inputs: 12, neurons: 32, choices: 4, curve: [{ step: 0, train: 1, validation: 1.1 }, { step: 100, train: 0.5 }], spikes_per_decision: 66.9, decision_ms: 0.7 },
    test: { metrics: [metric("reached_rate", "fraction", true, 0.5, 0.75, 40, { label: "Reached the spot" })], paired: [] },
  }, "r1");
  assert.equal(s.tries.nTries, 40);
  assert.equal(s.tries.paths.length, 1);
  assert.equal(s.tries.obstacles.length, 1);
  assert.equal(s.whatIf.choices.length, 2);
  assert.equal(s.whatIf.moment.tryNo, 3);
  assert.equal(s.imagination.length, 1);
  assert.equal(s.brain.neurons, 32);
  assert.equal(s.test.rows[0].label, "Reached the spot");
  assert.equal(s.test.rows[0].after.text, "30 of 40");
  assert.deepEqual(beatsOf(s).map((b) => b.has), [true, true, true, true]);
  assert.deepEqual(beatsOf(readStory(null)).map((b) => b.has), [false, false, false, false]);
});

test("a recorded run fills beats 3 and 4 from its own files and keeps the counts it recorded", () => {
  const body = { results: {
    timing: { attempts: 40 }, teach: { teacher: { n_context_rows: 28 }, scores: { shape: [1000, 4] } },
    train: { hidden_sizes: [32], parity: { action: { n_inputs_checked: 1000, total_spikes_per_layer_numpy: [66943] } },
      training: { steps_run: 1200, train_loss_curve: [[100, 0.13], [200, 0.07]], validation_loss_curve: [[0, { action: 0.45 }], [100, { action: 0.1, threat: 0.3 }]] } },
    metrics: [metric("ascent_success_rate", "fraction", true, 25 / 41, 31 / 41, 41)],
    paired: [{ name: "ascent_success_rate", diff: 0.14, ci_low: 0.05, ci_high: 0.27, n: 41 }],
  } };
  const s = storyFromRun(body, "ramp1");
  assert.equal(s.source, "run");
  assert.deepEqual([s.tries.nTries, s.tries.contextRows], [40, 28]);
  assert.deepEqual(s.imagined, { situations: 1000, choices: 4 });
  assert.equal(s.brain.neurons, 32);
  assert.equal(Math.round(s.brain.spikesPerDecision * 10) / 10, 66.9);
  assert.deepEqual(s.brain.curve.map((p) => [p.step, p.train, p.validation]), [[0, null, 0.45], [100, 0.13, 0.2], [200, 0.07, null]]);
  assert.equal(s.test.rows[0].before.text, "25 of 41");
  assert.equal(s.test.rows[0].after.text, "31 of 41");
  assert.equal(s.test.rows[0].change, "better");
  assert.equal(s.test.notes.length, 0);
  assert.equal(s.whatIf, null);
});

test("test rows read counts, intervals and direction; a missed goal and a chance-level change get calm notes", () => {
  const result = {
    metrics: [metric("collision_rate", "fraction", false, 0.5, 0.4, 240), metric("active_precision", "fraction", true, null, 0.952, 42), metric("mean_gap_m", "m", true, 0.31, 0.32, 240)],
    paired: [{ name: "collision_rate", diff: -0.1, ci_low: -0.2, ci_high: 0.01 }],
    sweep: { calibration: { precision_target: 0.99 } },
  };
  const rows = testRows(result.metrics);
  assert.deepEqual(rows.map((r) => [r.label, r.before?.text ?? null, r.after.text, r.change]), [
    ["Collision rate", "120 of 240", "96 of 240", "better"], ["Active precision", null, "40 of 42", null], ["Mean gap", "0.31 m", "0.32 m", "better"]]);
  assert.equal(rows[0].after.range, "likely 72 to 120");
  const notes = testNotes(result, rows);
  assert.deepEqual(notes.map((n) => n.kind), ["goal", "chance"]);
  assert.match(notes[0].text, /99%.*95% \(40 of 42\)/);
});

test("pictures: paths fit their box at one scale; the what-if axis covers every value; curves draw lower loss lower", () => {
  const v = topView({ paths: [{ points: [[0, 0], [2, 1]], ended: "a" }], obstacles: [] }, { w: 220, h: 120, pad: 10 });
  assert.equal(v.paths[0].d, "M10 110 L210 10");
  assert.deepEqual(endings([{ ended: "a" }, { ended: "b" }, { ended: "a" }]), [{ ended: "a", count: 2 }, { ended: "b", count: 1 }]);
  const ax = choiceAxis([{ predicted: { value: 0.5, low: 0.2, high: 0.9 } }, { measured: { value: 1 } }]);
  assert.ok(ax.x(0.2) > 0 && ax.lo >= 0 && ax.hi === 1); // shares stay inside 0..1
  const ax2 = choiceAxis([{ predicted: { value: 2, low: 1, high: 4 } }]);
  assert.ok(ax2.x(1) > 0 && ax2.x(4) < 1); // other measures get a margin on both sides
  const c = curvePaths([{ step: 0, train: 1, validation: null }, { step: 10, train: 0.5, validation: 0.75 }], { w: 100, h: 50, pad: 5 });
  assert.equal(c.train, "5,5 95,45");
  assert.equal(c.validation, null);
});

test("a quick lesson is offered only when the server's run_modes offer fast; durations come from recorded timing", () => {
  const modes = { RAMP: { full: "", fast: "" }, DODGE: { full: "", fast: "mode 'fast' is defined for move and ramp only" } };
  assert.deepEqual(quickLesson(modes, "RAMP"), { quick: true, reason: null });
  assert.equal(quickLesson(modes, "DODGE").quick, false);
  assert.match(quickLesson(modes, "DODGE").reason, /move and ramp only/);
  assert.deepEqual(quickLesson(null, "RAMP"), { quick: false, reason: null });
  assert.deepEqual(recordedDuration({ mode: "full", learn_s: 411, evaluate_s: 5068 }), { learn: "about 7 minutes", test: "about 84 minutes", mode: "full" });
  assert.equal(recordedDuration(null), null);
  assert.equal(aboutTime(30), "about 1 minute");
});

test("the cost sentence uses the quote's estimate only", () => {
  assert.equal(costSentence({ estimate: { episodes: 1200, tabpfn_tokens: { value: 10000 } } }), "This lesson runs about 1,200 simulated tries and uses about 10,000 TabPFN credits.");
  assert.equal(costSentence({ estimate: {} }), null);
});

test("the four beats fill in from a live run's event lines", () => {
  let tl = emptyTimeline();
  tl = applyLine(tl, { step: "attempts", state: "done", done: 40, total: 40, time_utc: "2026-01-01T00:00:01Z" });
  tl = applyLine(tl, { step: "teacher_fit", state: "running", time_utc: "2026-01-01T00:00:02Z" });
  assert.deepEqual(liveBeats(tl).map((b) => b.state), ["done", "running", "waiting", "waiting"]);
  assert.deepEqual(liveBeats(tl)[0].progress, { done: 40, total: 40 });
  tl = applyLine(tl, { step: "train", state: "running", done: 300, total: 1200, time_utc: "2026-01-01T00:00:03Z" });
  assert.deepEqual(liveBeats(tl, "failed").map((b) => b.state), ["done", "done", "stopped", "waiting"]);
});

test("a method that beat the learned one in the same test is said, not hidden", () => {
  const m = { name: "reached_rate", unit: "fraction", higher_is_better: true, values: { no_op: { value: 1 / 30, n: 30 }, physics_only: { value: 1, n: 30 }, snn: { value: 13 / 30, n: 30 } } };
  const rows = testRows([m]);
  assert.deepEqual([rows[0].before.text, rows[0].after.text], ["1 of 30", "13 of 30"]);
  assert.deepEqual(testNotes({ metrics: [m] }, rows).map((n) => n.text), ["In the same test, physics only did better: 30 of 30."]);
});
test("the brain part accepts layer sizes and choice names, as the server sends them", () => {
  const s = readStory({ brain: { inputs: 81, neurons: [64], choices: ["a", "b", "c"], curve: null, spikes_per_decision: null, decision_ms: null } });
  assert.deepEqual([s.brain.inputs, s.brain.neurons, s.brain.choices, s.brain.layers], [81, 64, 3, 1]);
  assert.equal(readStory({ brain: { inputs: null, neurons: null, choices: null, curve: null } }).brain, null);
});