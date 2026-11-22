import { test } from "node:test";
import assert from "node:assert/strict";
import { applyDuel, countRound, emptyDuel, tallyText } from "../../app/static/js/duel.js";

// Event shapes as app/duel.py sends them; the outcomes are made up for the test, not results.
const start = { state: "start", demo: "dodge", params: {}, seed: 5, sides: { before: { status: "running", outcome: null }, after: { status: "running", outcome: null } } };
const end = (before, after) => ({ state: "end", demo: "dodge", params: {}, seed: 5, sides: {
  before: { status: "done", outcome: before && { collision: 1, result: before } },
  after: { status: "done", outcome: after && { collision: 0, result: after } } } });

// One round through the reducer, counted as the page counts it.
function round(tally, endEvent) {
  const running = applyDuel(emptyDuel(), start);
  return countRound(tally, running, applyDuel(running, endEvent));
}

test("a finished round adds each lane's result word once", () => {
  let t = round({}, end("Hit", "Dodged"));
  assert.deepEqual(t, { dodge: { before: { Hit: 1 }, after: { Dodged: 1 } } });
  t = round(t, end("Hit", "Hit"));
  t = round(t, end("Fell", "Not hit"));
  assert.deepEqual(t.dodge.before, { Hit: 2, Fell: 1 });
  assert.deepEqual(t.dodge.after, { Dodged: 1, Hit: 1, "Not hit": 1 });
});

test("the same ended state seen again, or an end without a running round, counts nothing", () => {
  const running = applyDuel(emptyDuel(), start), ended = applyDuel(running, end("Hit", "Dodged"));
  const once = countRound({}, running, ended);
  assert.equal(countRound(once, ended, ended), once);
  assert.equal(countRound(once, emptyDuel(), ended), once); // a page that opens on an ended round
  assert.equal(countRound(once, ended, applyDuel(ended, { state: "idle" })), once);
});

test("a lane without a real outcome (stopped or failed) adds nothing; the other lane still counts", () => {
  const t = round({}, end(null, "Dodged"));
  assert.deepEqual(t, { dodge: { before: {}, after: { Dodged: 1 } } });
  assert.deepEqual(round({}, end(null, null)), { dodge: { before: {}, after: {} } });
});

test("an outcome without a result word (the hills demo) is not counted", () => {
  const hills = { ...start, demo: "hills" };
  const running = applyDuel(emptyDuel(), hills);
  const ended = applyDuel(running, { ...hills, state: "end", sides: { before: { status: "done", outcome: { reached: true } }, after: { status: "done", outcome: { reached: false } } } });
  assert.deepEqual(countRound({}, running, ended), { hills: { before: {}, after: {} } });
});

test("the tally reads in plain words in a fixed order, nothing before the first round", () => {
  assert.equal(tallyText(undefined), "");
  assert.equal(tallyText({}), "");
  assert.equal(tallyText({ Dodged: 1, Hit: 3 }), "Hit 3 · Dodged 1");
  assert.equal(tallyText({ Fell: 1, "Not hit": 2, Dodged: 0, Hit: 1 }), "Hit 1 · Not hit 2 · Fell 1");
});
