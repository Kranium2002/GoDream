import { test } from "node:test";
import assert from "node:assert/strict";
import { readout } from "../../app/static/js/readout.js";

// Shape of Dev B's infer() result plus the server's latency_ms (plan, "Live inference"); pure-function input only.
const answer = (over = {}) => ({
  ok: true, status: 200,
  data: {
    action: "DODGE_LEFT", command: { vel_x: 0, vel_y: 0.5, yaw_rate: 0 },
    threat_confidence: 0.934, best_action: "DODGE_LEFT", action_confidence: 0.8812, margin: 0.41,
    gate: { threat_ok: true, action_ok: true, margin_ok: true, votes: 3, votes_needed: 3, of_last: 4, passed: true },
    cooldown_remaining_s: 0.95, hold_remaining_s: 0.3, issued: true, inference_ms: 1.234, latency_ms: 4.56,
    ...over,
  },
});
const row = (r, label) => r.rows.find(([l]) => l === label)?.[1];

test("a decision maps every field to a labelled readout", () => {
  const r = readout(answer());
  assert.equal(r.state, "decision");
  assert.deepEqual(r.action, { code: "DODGE_LEFT", label: "Dodge left" });
  assert.equal(r.issued, true);
  assert.equal(row(r, "Threat confidence"), "0.93");
  assert.equal(row(r, "Best action"), "DODGE_LEFT");
  assert.equal(row(r, "Action confidence"), "0.88");
  assert.equal(row(r, "Margin"), "0.41");
  assert.equal(row(r, "Cooldown remaining"), "0.95 s");
  assert.equal(row(r, "Hold remaining"), "0.30 s");
  assert.equal(row(r, "Inference on the server"), "1.2 ms");
  assert.equal(row(r, "Server latency"), "4.6 ms");
  assert.deepEqual(r.gate, [
    { label: "Threat confidence high enough", ok: true },
    { label: "Action confidence high enough", ok: true },
    { label: "Margin wide enough", ok: true },
    { label: "Votes: 3 of 3 needed in the last 4", ok: true },
    { label: "Gate passed", ok: true },
  ]);
});

test("UNCERTAIN reads as uncertain, with no command issued", () => {
  const r = readout(answer({ action: "UNCERTAIN", issued: false }));
  assert.deepEqual(r.action, { code: "UNCERTAIN", label: "Uncertain: stays still" });
  assert.equal(r.issued, false);
});

test("a failed gate shows which checks failed and that the gate is closed", () => {
  const r = readout(answer({ action: "NO_OP", issued: false, gate: { threat_ok: true, action_ok: false, margin_ok: false, votes: 1, votes_needed: 3, of_last: 4, passed: false } }));
  assert.deepEqual(r.action, { code: "NO_OP", label: "No action" });
  assert.deepEqual(r.gate.map((g) => g.ok), [true, false, false, false, false]);
  assert.equal(r.gate[3].label, "Votes: 1 of 3 needed in the last 4");
  assert.equal(r.gate[4].label, "Gate closed");
});

test("a field the server did not send says so, never a made-up value", () => {
  const r = readout(answer({ margin: undefined, gate: undefined, issued: undefined, action: "SOMETHING_NEW" }));
  assert.equal(row(r, "Margin"), "not sent");
  assert.deepEqual(r.gate, []);
  assert.equal(r.issued, null);
  assert.deepEqual(r.action, { code: "SOMETHING_NEW", label: "SOMETHING_NEW" });
});

test("unavailable and other errors keep the server's message", () => {
  assert.deepEqual(readout({ ok: false, status: 503, code: "infer_unavailable", message: "No frozen model.", unavailable: true, data: {} }),
    { state: "unavailable", message: "No frozen model." });
  assert.deepEqual(readout({ ok: false, status: 400, code: "bad_window", message: "Wrong size.", unavailable: false, data: {} }),
    { state: "error", code: "bad_window", message: "Wrong size." });
});
