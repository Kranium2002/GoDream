// Plain words for visitors (app/static/js/plain.js). Inputs are shaped like the server's answers; pure-function fixtures
// only, never results.
import { test } from "node:test";
import assert from "node:assert/strict";
import { aboutTime, cardLine, costQuestion, episodeCaption, lessonChoices, lessonSteps, latestDone, parseRoute, replaySpeed,
  resultHeadline, resultNotes, resultTiles, routeHash, skillKey, understood } from "../../app/static/js/plain.js";
import { applyLine, emptyTimeline } from "../../app/static/js/learn.js";

const catalogue = [
  { skill: "RAMP", spec: "ramp", available: true, params: { incline_deg: { unit: "deg" }, nominal_speed_m_s: { unit: "m/s" } } },
  { skill: "DODGE", spec: "dodge_vision", available: true, params: {} },
  { skill: "KICK", spec: "kick", available: false, missing: ["a ball model"], params: {} },
];
const frac = (name, before, after, n, extra = {}) => ({ name, unit: "fraction", higher_is_better: true, ...extra,
  values: { no_op: { value: before / n, n }, snn: { value: after / n, n } } });

test("skill ids from runs, events and the catalogue all name the same skill", () => {
  assert.equal(skillKey("ramp", catalogue), "RAMP");
  assert.equal(skillKey("dodge_vision", catalogue), "DODGE");
  assert.equal(skillKey("move", null), "MOVE");
  assert.equal(skillKey("brake", null), "BRAKE");
  assert.equal(skillKey(null, catalogue), null);
});

test("a chat answer reads as the person's own words, the settings it declared, and what it cannot do", () => {
  const u = understood({ reply: "RAMP at 15 degrees.", declared_spec: { skill: "RAMP", params: { incline_deg: 15 }, source_text: "Climb a steep ramp" }, problems: [], missing: [] }, catalogue);
  assert.equal(u.ok, true);
  assert.equal(u.heard, "Got it: climb a steep ramp.");
  assert.deepEqual(u.settings, [{ label: "Ramp angle", value: "15°" }]);
  assert.equal(u.usual, false);
  const none = understood({ reply: "No skill fits.", declared_spec: null, problems: [], missing: ["juggling"] }, catalogue);
  assert.equal(none.ok, false);
  assert.equal(none.heard, null);
  assert.deepEqual(none.refused, ["juggling"]);
  const bad = understood({ declared_spec: { skill: "KICK", params: {} }, problems: [{ field: "", message: "Too far." }], missing: [] }, catalogue);
  assert.equal(bad.ok, false);
  assert.deepEqual(bad.problems, ["Too far."]);
  assert.deepEqual(bad.unsupported, ["a ball model"]);
  assert.equal(understood({ declared_spec: { skill: "RAMP", params: {} }, problems: [], missing: [] }, catalogue).usual, true);
});

test("lesson choices offer only the modes the server accepts, timed only by finished runs of that mode", () => {
  const cards = [
    { runId: "r-fast", skill: "RAMP", state: "done", updated: "2026-11-22T03:26:19Z", timing: { mode: "fast", learn_s: 129.3, evaluate_s: 227.6 } },
    { runId: "r-old", skill: "RAMP", state: "done", updated: "2026-11-21T21:59:18Z", timing: null },
    { runId: "d-failed", skill: "DODGE", state: "failed", updated: "2026-11-22T01:27:29Z", timing: { mode: "full", learn_s: 411 } },
  ];
  const ramp = lessonChoices({ RAMP: { full: "", fast: "" } }, "RAMP", cards);
  assert.deepEqual(ramp.choices.map((c) => [c.mode, c.label, c.learn, c.test, c.timedBy]),
    [["fast", "Quick lesson", "about 2 minutes", "about 4 minutes", "r-fast"], ["full", "Full lesson", null, null, null]]);
  assert.equal(ramp.quickRefused, null);
  const dodge = lessonChoices({ DODGE: { full: "", fast: "mode 'fast' is defined for move and ramp only" } }, "DODGE", cards);
  assert.deepEqual(dodge.choices.map((c) => [c.mode, c.learn]), [["full", null]]); // a failed run times nothing
  assert.match(dodge.quickRefused, /move and ramp only/);
  assert.deepEqual(lessonChoices({}, "RAMP", cards), { choices: [], quickRefused: null });
  assert.equal(latestDone(cards, "RAMP").runId, "r-fast");
  assert.equal(latestDone(cards, "DODGE"), null);
});

test("durations are rounded to whole minutes from the measured seconds", () => {
  assert.equal(aboutTime(40), "about 1 minute");
  assert.equal(aboutTime(129.3), "about 2 minutes");
  assert.equal(aboutTime(5068), "about 84 minutes");
  assert.equal(aboutTime(9000), "about 2.5 hours");
  assert.equal(aboutTime(null), null);
});

test("the cost question uses the quote's own counts, and says so when one is missing", () => {
  assert.equal(costQuestion({ estimate: { episodes: 1240, tabpfn_tokens: { value: 437652 } } }),
    "This runs about 1,240 simulated tries and uses about 437,652 teacher credits. Start?");
  assert.match(costQuestion({ estimate: { episodes: 80, tabpfn_tokens: { value: null } } }), /about 80 simulated tries.*could not be estimated/);
  assert.equal(costQuestion({}), "This lesson uses simulator time and teacher credits. Start?");
});

test("the five steps follow the run's lines, and a finished run closes a step whose end was not logged", () => {
  const lines = [
    { step: "attempts", state: "running", time_utc: "2026-11-22T03:20:21Z" },
    { step: "teacher_fit", state: "done", time_utc: "2026-11-22T03:21:34Z" },
    { step: "dream", state: "done", time_utc: "2026-11-22T03:21:50Z" },
    { step: "train", state: "running", done: 600, total: 1200, time_utc: "2026-11-22T03:22:11Z", data: { train_loss: 0.03, validation_loss: 0.04 } },
  ];
  const tl = lines.reduce(applyLine, emptyTimeline());
  const live = lessonSteps(tl, "running");
  assert.deepEqual(live.map((s) => s.state), ["running", "done", "done", "running", "waiting"]);
  assert.deepEqual(live[3].progress, { done: 600, total: 1200, unit: "rounds" });
  assert.equal(live.filter((s) => s.current).length, 1);
  const done = lessonSteps(applyLine(applyLine(tl, { step: "train", state: "done", done: 1200, total: 1200, time_utc: "2026-11-22T03:22:29Z" }),
    { step: "evaluate", state: "done", done: 120, total: 120, time_utc: "2026-11-22T03:26:18Z" }), "done");
  assert.deepEqual(done.map((s) => s.state), ["done", "done", "done", "done", "done"]);
  assert.equal(done.some((s) => s.current), false);
  const failed = lessonSteps(applyLine(tl, { step: "train", state: "failed", time_utc: "2026-11-22T03:23:00Z", data: { reason: "boom" } }), "failed");
  assert.equal(failed[3].state, "stopped");
  assert.equal(failed[3].current, true);
  assert.equal(failed[4].state, "unlogged");
  assert.deepEqual(lessonSteps(emptyTimeline(), null).map((s) => s.state), ["waiting", "waiting", "waiting", "waiting", "waiting"]);
});

const ramp = {
  metrics: [frac("ascent_success_rate", 25, 31, 41), { name: "fall_rate", unit: "fraction", higher_is_better: false, values: { no_op: { value: 0, n: 60 }, snn: { value: 0, n: 60 } } },
    { name: "mean_progress_m", unit: "m", higher_is_better: true, values: { no_op: { value: 3.186, n: 41 }, snn: { value: 4.4746, n: 41 } } },
    { name: "mean_slip_m", unit: "m", higher_is_better: false, values: { no_op: { value: 6.408, n: 41 }, snn: { value: 7.1047, n: 41 } } },
    { name: "flat_false_activations", unit: "count", higher_is_better: false, values: { no_op: { value: 0, n: 19 }, snn: { value: 0, n: 19 } } }],
  paired: [{ name: "ascent_success_rate", a: "snn", b: "no_op", diff: 0.146, ci_low: 0.049, ci_high: 0.268, n: 41 }],
};

test("a result's headline and tiles are the counts it was measured from", () => {
  assert.deepEqual(resultHeadline(ramp, 40), { metric: "ascent_success_rate", text: "After 40 tries, the dog reaches the top of the ramp 31 times out of 41. Before: 25." });
  assert.equal(resultHeadline(ramp).text.startsWith("After learning,"), true);
  const tiles = resultTiles(ramp);
  assert.equal(tiles.length, 4);
  assert.deepEqual(tiles.map((t) => [t.label, t.before, t.after, t.change, t.basis]), [
    ["Reaches the top of the ramp", "25", "31", "better", "out of 41 tests"],
    ["Falls", "0", "0", "same", "out of 60 tests"],
    ["Distance up the ramp, on average", "3.19 m", "4.47 m", "better", "over 41 tests"],
    ["Foot slip, on average", "6.41 m", "7.10 m", "worse", "over 41 tests"],
  ]);
  assert.equal(cardLine(ramp), "Reaches the top of the ramp: 31 of 41, up from 25");
  assert.deepEqual(resultNotes(ramp), []);
});

test("a mean error reads with its unit, and lower is better", () => {
  const move = { metrics: [{ name: "mean_error_forward_cm", unit: "cm", higher_is_better: false, values: { physics_only: { value: 9.8614, n: 14 }, snn: { value: 5.1715, n: 14 } } }] };
  assert.equal(resultHeadline(move).text, "After learning, the dog misses its mark by 5.2 cm on average when walking forward or back. Before: 9.9 cm.");
  assert.equal(cardLine(move), "Miss when walking forward or back: 5.2 cm, down from 9.9 cm");
  assert.equal(resultTiles(move)[0].change, "better");
});

test("a goal named in the result and not reached, and a change that could be chance, become calm notes", () => {
  const dodge = {
    closed_loop_test: { calibration: { precision_target: 0.99 } }, sweep: { calibration: { precision_target: 0.99 } },
    metrics: [{ name: "collision_rate", unit: "fraction", higher_is_better: false, values: { no_op: { value: 121 / 240, n: 240 }, snn: { value: 98 / 240, n: 240 } } },
      { name: "active_precision", unit: "fraction", higher_is_better: true, values: { no_op: { value: null, n: 0 }, snn: { value: 40 / 42, n: 42 } } }],
    paired: [{ name: "collision_rate", ci_low: -0.1375, ci_high: -0.0583 }],
  };
  assert.deepEqual(resultNotes(dodge), [{ kind: "goal", text: "Not there yet. Goal for moves that were needed: 99%. In testing: 95% (40 of 42)." }]);
  assert.equal(resultTiles(dodge)[1].before, "—");
  assert.equal(resultTiles(dodge)[1].change, null);
  const chance = { ...ramp, paired: [{ name: "ascent_success_rate", ci_low: 0, ci_high: 0.24 }] };
  assert.deepEqual(resultNotes(chance).map((n) => n.kind), ["chance"]);
  assert.equal(resultHeadline({ results: "no metrics" }), null);
});

test("episode events read as plain captions: which test, before or after learning, and how it went", () => {
  const start = { state: "start", skill: "RAMP", step: "evaluate", index: 11, total: 120, kind: "slope", action: "nominal", caption: "RAMP, test scenario 5", note: "showing this run's own episodes" };
  const c = episodeCaption(start);
  assert.deepEqual([c.title, c.phase, c.phaseText, c.outcome], ["Test 12 of 120: a ramp", "before", "Before learning", null]);
  assert.equal(c.detail, "RAMP, test scenario 5 showing this run's own episodes");
  const end = episodeCaption({ ...start, state: "end", action: "student", outcome: { reached_top: 1, fall: 0 } });
  assert.deepEqual([end.phase, end.outcome], ["after", "reached the top"]);
  assert.equal(episodeCaption({ ...start, state: "end", outcome: { reached_top: 0, fall: 1 } }).outcome, "did not reach the top, fell");
  assert.equal(episodeCaption({ state: "end", skill: "MOVE", step: "attempts", index: 0, total: 40, kind: "forward", action: "attempt", outcome: { error_units: 0.052, unit: "m", fall: false } }).title, "Try 1 of 40: walking forward or back");
  assert.equal(episodeCaption({ state: "end", skill: "MOVE", step: "evaluate", kind: "turn", action: "student", outcome: { error_units: 0.1, unit: "rad", fall: false } }).outcome, "missed by 6°");
  assert.equal(episodeCaption({ state: "end", skill: "DODGE", step: "evaluate", kind: "incoming_left", action: "NO_OP", outcome: { collision: true, fall: false } }).outcome, "got hit");
  assert.deepEqual(episodeCaption({ state: "idle" }), { hidden: true });
});

test("a recorded log replays in about forty seconds, never slower than it ran", () => {
  const at = (s) => ({ time_utc: new Date(Date.UTC(2026, 10, 22, 3, 20, 0) + s * 1000).toISOString() });
  assert.equal(replaySpeed([at(0), at(360)]), 9);
  assert.equal(replaySpeed([at(0), at(10)]), 1);
  assert.equal(replaySpeed([]), 1);
});

test("routes name the screen and, for Learn and Results, the run", () => {
  assert.deepEqual(parseRoute("#results/ramp-fast1"), { screen: "results", runId: "ramp-fast1" });
  assert.deepEqual(parseRoute("#live"), { screen: "live", runId: null });
  assert.deepEqual(parseRoute(""), { screen: "teach", runId: null });
  assert.deepEqual(parseRoute("#nowhere"), { screen: "teach", runId: null });
  assert.equal(routeHash("learn", "a b"), "#learn/a%20b");
  assert.equal(routeHash("live", "x"), "#live");
});
