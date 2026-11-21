import { test } from "node:test";
import assert from "node:assert/strict";
import { CONTRACT_CAPABILITIES, normaliseCapabilities } from "../../app/static/js/capabilities.js";

const byName = (list) => Object.fromEntries(list.map((c) => [c.name, c]));

test("the contract's capability names come first, in contract order", () => {
  assert.deepEqual(CONTRACT_CAPABILITIES, ["drive", "head_camera", "scenarios", "chat", "runs", "results", "infer"]);
  const list = normaliseCapabilities({ extra: { available: true, reason: "" }, infer: { available: false, reason: "x" } });
  assert.deepEqual(list.map((c) => c.name), [...CONTRACT_CAPABILITIES, "extra"]);
});

test("the contract shape {available, reason} is kept as sent", () => {
  const c = byName(normaliseCapabilities({ chat: { available: false, reason: "No ANTHROPIC_API_KEY in .env." }, drive: { available: true, reason: "" } }));
  assert.deepEqual(c.chat, { name: "chat", available: false, reason: "No ANTHROPIC_API_KEY in .env." });
  assert.deepEqual(c.drive, { name: "drive", available: true, reason: "" });
});

test("the older boolean shape is accepted; false gets an honest no-reason text", () => {
  const c = byName(normaliseCapabilities({ drive: true, throw: false }));
  assert.deepEqual(c.drive, { name: "drive", available: true, reason: "" });
  assert.deepEqual(c.throw, { name: "throw", available: false, reason: "The server gave no reason." });
});

test("a contract capability the server does not report is unavailable", () => {
  const c = byName(normaliseCapabilities({ drive: true }));
  assert.equal(c.runs.available, false);
  assert.equal(c.runs.reason, "The server does not report this capability.");
});

test("no capabilities at all makes every contract capability unavailable", () => {
  for (const raw of [undefined, null, "nonsense", []]) {
    const list = normaliseCapabilities(raw);
    assert.equal(list.length, CONTRACT_CAPABILITIES.length);
    assert.ok(list.every((c) => c.available === false));
  }
});

test("an unreadable entry is unavailable, and only available === true counts", () => {
  const c = byName(normaliseCapabilities({ chat: "yes", runs: { available: "true", reason: "r" }, infer: { available: false } }));
  assert.deepEqual(c.chat, { name: "chat", available: false, reason: "The server sent an unreadable entry." });
  assert.equal(c.runs.available, false);
  assert.equal(c.runs.reason, "r");
  assert.equal(c.infer.reason, "The server gave no reason.");
});
