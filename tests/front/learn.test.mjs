import { test } from "node:test";
import assert from "node:assert/strict";
import {
  STEPS, applyLine, emptyTimeline, flatRows, fromRunResponse, lossCurve, rampMode, resultView, timelineRows, trainSeries,
} from "../../app/static/js/learn.js";

// Pure-function fixtures in the agreed event-line shape; never shown on the page.
const line = (step, state, time, extra = {}) => ({ run_id: "r1", skill: "RAMP", step, state, done: 0, total: 0, time_utc: time, data: {}, ...extra });
const T = (s) => `2026-11-21T21:00:${String(s).padStart(2, "0")}Z`;
const build = (lines) => lines.reduce(applyLine, emptyTimeline());
const rowOf = (tl, step) => timelineRows(tl).find((r) => r.step === step);

test("the six steps are fixed and in order", () => {
  assert.deepEqual(STEPS, ["baseline", "attempts", "teacher_fit", "dream", "train", "evaluate"]);
});

test("an empty timeline lists all six steps as not started", () => {
  const rows = timelineRows(emptyTimeline());
  assert.deepEqual(rows.map((r) => r.step), STEPS);
  assert.ok(rows.every((r) => r.state === null && r.progress === null));
});

test("lines in order set each step's state, progress and time", () => {
  const tl = build([line("baseline", "running", T(1), { done: 3, total: 10 }), line("baseline", "done", T(2), { done: 10, total: 10 })]);
  assert.equal(tl.runId, "r1");
  assert.equal(tl.skill, "RAMP");
  const r = rowOf(tl, "baseline");
  assert.equal(r.state, "done");
  assert.equal(r.progress, "10 of 10");
  assert.equal(r.time, T(2));
});

test("an older line arriving late does not overwrite a newer one", () => {
  const tl = build([line("attempts", "done", T(9), { done: 40, total: 40 }), line("attempts", "running", T(3), { done: 5, total: 40 })]);
  assert.equal(rowOf(tl, "attempts").state, "done");
  assert.equal(rowOf(tl, "attempts").progress, "40 of 40");
});

test("a repeated line changes nothing", () => {
  const l = line("dream", "running", T(4), { done: 100, total: 1000 });
  assert.deepEqual(timelineRows(build([l, l, l])), timelineRows(build([l])));
});

test("at the same time the later arrival wins; an untimed line never overrides a timed one", () => {
  let tl = build([line("dream", "running", T(4), { done: 1, total: 9 }), line("dream", "running", T(4), { done: 2, total: 9 })]);
  assert.equal(rowOf(tl, "dream").progress, "2 of 9");
  tl = applyLine(tl, line("dream", "failed", undefined));
  assert.equal(rowOf(tl, "dream").state, "running");
});

test("an unknown step is kept and listed after the six; an unreadable line is counted", () => {
  const tl = build([line("calibrate", "done", T(1)), { nonsense: true }, null]);
  const rows = timelineRows(tl);
  assert.equal(rows.length, 7);
  assert.equal(rows[6].step, "calibrate");
  assert.equal(rows[6].label, "calibrate");
  assert.equal(tl.unreadable, 2);
});

test("refused, failed and stopped show the reason text the data carries; other data stays as key and value", () => {
  const tl = build([line("teacher_fit", "refused", T(5), { data: { reason: "Quote above the limit.", quote_usd: 2.5 } })]);
  const r = rowOf(tl, "teacher_fit");
  assert.equal(r.reason, "Quote above the limit.");
  assert.deepEqual(r.dataRows, [["reason", "Quote above the limit."], ["quote_usd", "2.5"]]);
  const m = rowOf(build([line("train", "failed", T(5), { data: { message: "NaN loss." } })]), "train");
  assert.equal(m.reason, "NaN loss.");
  assert.equal(rowOf(build([line("train", "running", T(5), { data: { reason: "x" } })]), "train").reason, null);
});

test("train observations are collected by their own step, deduplicated and sorted", () => {
  const obs = (step, v) => line("train", "running", T(step), { data: { step, max_steps: 100, train_loss: v + 0.1, validation_loss: v } });
  const tl = build([obs(20, 0.5), obs(10, 0.8), obs(20, 0.5), obs(30, 0.4), line("train", "running", T(40), { data: { note: "no step" } })]);
  assert.deepEqual(trainSeries(tl).map((d) => d.step), [10, 20, 30]);
  assert.equal(rowOf(tl, "train").isTrainObservation, false); // the latest data has no training fields
});

test("fromRunResponse reads the run's steps as lines and keeps baseline and results", () => {
  const tl = fromRunResponse("r9", { state: "running", steps: [{ step: "baseline", state: "done", done: 5, total: 5, data: {} }], baseline: { a: 1 }, results: null });
  assert.equal(tl.runId, "r9");
  assert.equal(tl.state, "running");
  assert.equal(rowOf(tl, "baseline").state, "done");
  assert.deepEqual(tl.baseline, { a: 1 });
  assert.equal(tl.results, null);
  assert.equal(rowOf(fromRunResponse("r9", { steps: "bad" }), "baseline").state, null);
});

test("flatRows lists nested data as dotted keys, lists as JSON text", () => {
  assert.deepEqual(flatRows({ a: 1, b: { c: "x", d: { e: null } }, f: [1, 2], g: true }),
    [["a", "1"], ["b.c", "x"], ["b.d.e", "null"], ["f", "[1,2]"], ["g", "true"]]);
  assert.deepEqual(flatRows(null), []);
  assert.deepEqual(flatRows("text"), [["value", "text"]]);
});

test("lossCurve maps steps to x and losses to y inside the padded box, y growing downwards", () => {
  const series = [{ step: 0, train_loss: 1.0, validation_loss: 0.8 }, { step: 100, train_loss: 0.2, validation_loss: 0.4 }];
  const c = lossCurve(series, { width: 220, height: 120, pad: 10 });
  assert.deepEqual(c.x, [0, 100]);
  assert.deepEqual(c.y, [0.2, 1.0]);
  assert.equal(c.train, "10,10 210,110");      // loss 1.0 at the top, 0.2 at the bottom
  assert.equal(c.validation, "10,35 210,85");
});

test("lossCurve skips missing values, centres a flat range, and is null without points", () => {
  const c = lossCurve([{ step: 5, validation_loss: 0.3 }], { width: 120, height: 120, pad: 10 });
  assert.equal(c.train, null);
  assert.equal(c.validation, "60,60");
  assert.equal(lossCurve([{ step: 1 }], { width: 100, height: 100, pad: 0 }), null);
});

test("resultView maps the proposed result shape to one column per method", () => {
  const v = resultView({
    run_id: "r1", skill: "RAMP", declaration: {}, seeds: [1, 2, 3], scenario_ids: ["a", "b"],
    metrics: [
      { name: "completed ascents", unit: "%", higher_is_better: true, values: { no_op: { value: 40, ci_low: 30, ci_high: 50, n: 60 }, snn: { value: 72.25 } } },
      { name: "slip", unit: "m", higher_is_better: false, values: { physics: { value: 0.123456, n: 60 } } },
    ],
    paired: [{ name: "completed ascents", a: "snn", b: "no_op", diff: 32, ci_low: 20, ci_high: 44, n: 60 }],
  });
  assert.equal(v.kind, "table");
  assert.equal(v.seeds, "1, 2, 3");
  assert.equal(v.scenarioCount, 2);
  assert.deepEqual(v.methods, [{ key: "no_op", label: "No-op" }, { key: "snn", label: "SNN" }, { key: "physics", label: "Physics only" }]);
  assert.deepEqual(v.rows[0], { name: "completed ascents", unit: "%", better: "higher is better", cells: ["40 % (30 to 50), n = 60", "72.25 %", "not measured"] });
  assert.deepEqual(v.rows[1].cells, ["not measured", "not measured", "0.1235 m, n = 60"]);
  assert.equal(v.rows[1].better, "lower is better");
  assert.deepEqual(v.paired, [{ name: "completed ascents", comparison: "snn minus no_op", diff: "32 (20 to 44), n = 60" }]);
});

test("a result that does not match is shown as key and value, never guessed", () => {
  assert.deepEqual(resultView({ accuracy: 0.9, metrics: "x" }), { kind: "raw", rows: [["accuracy", "0.9"], ["metrics", "x"]] });
  assert.deepEqual(resultView({ metrics: [{ name: "a" }] }).kind, "raw");
  assert.deepEqual(resultView(null), { kind: "none" });
});

test("rampMode is read only when an observation actually carries NORMAL or SLOPE", () => {
  assert.equal(rampMode({ ramp_mode: "SLOPE" }), "SLOPE");
  assert.equal(rampMode({ mode: "NORMAL" }), "NORMAL");
  assert.equal(rampMode({ mode: "something" }), null);
  assert.equal(rampMode({ t_s: 1 }), null);
  assert.equal(rampMode(null), null);
});
