import { test } from "node:test";
import assert from "node:assert/strict";
import { cardsFrom, replayPlan, skillLabel, trackStages } from "../../app/static/js/flow.js";

const L = (s, step = "dream", state = "running") => ({ run_id: "v0", step, state, time_utc: `2026-11-21T21:25:${String(s).padStart(2, "0")}Z`, data: null });

test("replayPlan schedules recorded lines by their time_utc gaps divided by the speed, in file order", () => {
  const plan = replayPlan([L(0), L(4), L(34)], 10);
  assert.deepEqual(plan.map((p) => p.atMs), [0, 400, 3400]);
  assert.equal(plan[2].line.time_utc, "2026-11-21T21:25:34Z");
});

test("replayPlan never schedules backwards; an unreadable time keeps the previous moment", () => {
  const plan = replayPlan([L(10), L(5), { ...L(0), time_utc: "nonsense" }, L(12)], 1);
  assert.deepEqual(plan.map((p) => p.atMs), [0, 0, 0, 2000]);
  assert.deepEqual(replayPlan([], 5), []);
  assert.throws(() => replayPlan([L(0)], 0), /speed/);
});

test("trackStages gives each pipeline stage its state, a fraction only from real numbers, and one line on what it does", () => {
  const rows = [
    { step: "baseline", label: "Baseline measured", state: "done", progress: null },
    { step: "attempts", label: "Real simulator attempts", state: null, progress: null },
    { step: "train", label: "SNN training", state: "running", progress: "300 of 3000" },
    { step: "calibrate", label: "calibrate", state: "done", progress: null },
  ];
  const t = trackStages(rows);
  assert.deepEqual(t.map((s) => s.step), ["baseline", "attempts", "train", "calibrate"]);
  assert.equal(t[0].fraction, null);
  assert.equal(t[2].fraction, 0.1);
  assert.equal(t[2].running, true);
  assert.equal(t[1].stateText, "no event yet");
  assert.ok(t[2].explain.length > 10);
  assert.equal(t[3].explain, "");
});

test("cardsFrom reads the summary list, or makes a bare card per id of the plain list", () => {
  const cards = cardsFrom([{ run_id: "v0", skill: "dodge_vision", state: "running", updated_utc: "2026-11-21T21:26:43Z", steps_done: 4, has_result: true, has_student: true }]);
  assert.deepEqual(cards, [{ runId: "v0", skill: "dodge_vision", state: "running", updated: "2026-11-21T21:26:43Z", stepsDone: 4, hasResult: true, hasStudent: true }]);
  assert.deepEqual(cardsFrom(["a", "b", "a", 3]), [
    { runId: "a", skill: null, state: null, updated: null, stepsDone: null, hasResult: null, hasStudent: null },
    { runId: "b", skill: null, state: null, updated: null, stepsDone: null, hasResult: null, hasStudent: null },
  ]);
  assert.deepEqual(cardsFrom({ nope: 1 }), []);
});

test("skillLabel names a run's skill from the catalogue by its name or its SkillSpec module", () => {
  const catalogue = [{ skill: "DODGE", spec: "dodge_vision", description: "Dodge an object." }, { skill: "RAMP", spec: "ramp", description: "Cross an incline." }];
  assert.deepEqual(skillLabel(catalogue, "dodge_vision"), { name: "DODGE", description: "Dodge an object." });
  assert.deepEqual(skillLabel(catalogue, "RAMP"), { name: "RAMP", description: "Cross an incline." });
  assert.equal(skillLabel(catalogue, "kick"), null);
  assert.equal(skillLabel(null, "ramp"), null);
});
