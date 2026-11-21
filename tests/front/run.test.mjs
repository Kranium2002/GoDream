import { test } from "node:test";
import assert from "node:assert/strict";
import { canStart, INITIAL_RUN, quoteRows, runStep } from "../../app/static/js/run.js";

const ok202 = { ok: true, status: 202, data: { run_id: "run-7" } };
const quote = { code: "cost_confirmation_required", message: "This run will use hosted TabPFN.", usage: { requests: 3, estimated_cost_usd: 0.4 }, limit: 100 };
const ask409 = { ok: false, status: 409, code: quote.code, message: quote.message, unavailable: false, data: quote };

test("start is blocked until a valid spec arrives", () => {
  assert.equal(INITIAL_RUN.phase, "blocked");
  assert.equal(canStart(INITIAL_RUN), false);
  assert.equal(runStep(INITIAL_RUN, { type: "start" }), INITIAL_RUN);
  const ready = runStep(INITIAL_RUN, { type: "spec", valid: true });
  assert.equal(ready.phase, "ready");
  assert.equal(canStart(ready), true);
  assert.equal(runStep(ready, { type: "spec", valid: false }).phase, "blocked");
});

test("start sends without cost confirmation first", () => {
  const s = runStep({ phase: "ready" }, { type: "start" });
  assert.deepEqual(s, { phase: "starting", confirmCost: false });
  assert.equal(canStart(s), false);
});

test("a 202 with a run id means started", () => {
  const s = runStep({ phase: "starting", confirmCost: false }, { type: "answer", result: ok202 });
  assert.deepEqual(s, { phase: "started", runId: "run-7" });
  assert.equal(canStart(s), false);
});

test("409 cost_confirmation_required asks with the quoted usage, then retries confirmed", () => {
  const ask = runStep({ phase: "starting", confirmCost: false }, { type: "answer", result: ask409 });
  assert.equal(ask.phase, "confirm");
  assert.equal(ask.message, quote.message);
  assert.deepEqual(ask.quote, [["usage.requests", "3"], ["usage.estimated_cost_usd", "0.4"], ["limit", "100"]]);
  assert.equal(canStart(ask), false);
  assert.deepEqual(runStep(ask, { type: "confirm" }), { phase: "starting", confirmCost: true });
  assert.equal(runStep(ask, { type: "cancel" }).phase, "ready");
});

test("confirm and cancel do nothing outside the confirm phase", () => {
  const ready = { phase: "ready" };
  assert.equal(runStep(ready, { type: "confirm" }), ready);
  assert.equal(runStep(ready, { type: "cancel" }), ready);
});

test("an unavailable answer and any other error are kept apart and both allow a retry", () => {
  const starting = { phase: "starting", confirmCost: false };
  const u = runStep(starting, { type: "answer", result: { ok: false, status: 503, code: "runs_unavailable", message: "No pipeline entry point.", unavailable: true, data: {} } });
  assert.deepEqual(u, { phase: "unavailable", code: "runs_unavailable", message: "No pipeline entry point." });
  assert.equal(canStart(u), true);
  const f = runStep(starting, { type: "answer", result: { ok: false, status: 400, code: "invalid_spec", message: "Bad.", unavailable: false, data: {} } });
  assert.deepEqual(f, { phase: "failed", code: "invalid_spec", message: "Bad." });
  assert.equal(canStart(f), true);
});

test("a success answer without a run id is a failure, not a started run", () => {
  const s = runStep({ phase: "starting", confirmCost: true }, { type: "answer", result: { ok: true, status: 200, data: {} } });
  assert.equal(s.phase, "failed");
  assert.equal(s.code, "invalid_response");
});

test("a new spec does not interrupt a start in flight; answers outside starting are ignored", () => {
  const starting = { phase: "starting", confirmCost: false };
  assert.equal(runStep(starting, { type: "spec", valid: false }), starting);
  const ready = { phase: "ready" };
  assert.equal(runStep(ready, { type: "answer", result: ok202 }), ready);
});

test("a new valid spec after a started run allows another start", () => {
  assert.equal(runStep({ phase: "started", runId: "r" }, { type: "spec", valid: true }).phase, "ready");
});

test("quoteRows flattens nested values and skips code and message", () => {
  assert.deepEqual(quoteRows({ code: "c", message: "m", a: { b: { c: 1 } }, list: [1, 2], flag: false, none: null }), [["a.b.c", "1"], ["list", "[1,2]"], ["flag", "false"], ["none", "null"]]);
  assert.deepEqual(quoteRows(null), []);
});
