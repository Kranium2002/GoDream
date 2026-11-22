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
  assert.deepEqual(r.action, { code: "DODGE_LEFT", label: "Dodge left", tone: "accent" });
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
  assert.deepEqual(r.action, { code: "UNCERTAIN", label: "Uncertain: stays still", tone: "warn" });
  assert.equal(r.issued, false);
});

test("a failed gate shows which checks failed and that the gate is closed", () => {
  const r = readout(answer({ action: "NO_OP", issued: false, gate: { threat_ok: true, action_ok: false, margin_ok: false, votes: 1, votes_needed: 3, of_last: 4, passed: false } }));
  assert.deepEqual(r.action, { code: "NO_OP", label: "No action", tone: "neutral" });
  assert.deepEqual(r.gate.map((g) => g.ok), [true, false, false, false, false]);
  assert.equal(r.gate[3].label, "Votes: 1 of 3 needed in the last 4");
  assert.equal(r.gate[4].label, "Gate closed");
});

test("a field the server did not send says so, never a made-up value", () => {
  const r = readout(answer({ margin: undefined, gate: undefined, issued: undefined, action: "SOMETHING_NEW" }));
  assert.equal(row(r, "Margin"), "not sent");
  assert.deepEqual(r.gate, []);
  assert.equal(r.issued, null);
  assert.deepEqual(r.action, { code: "SOMETHING_NEW", label: "SOMETHING_NEW", tone: "neutral" });
});

test("unavailable and other errors keep the server's message", () => {
  assert.deepEqual(readout({ ok: false, status: 503, code: "infer_unavailable", message: "No frozen model.", unavailable: true, data: {} }),
    { state: "unavailable", message: "No frozen model." });
  assert.deepEqual(readout({ ok: false, status: 400, code: "bad_window", message: "Wrong size.", unavailable: false, data: {} }),
    { state: "error", code: "bad_window", message: "Wrong size." });
});

test("the structured readout keeps numbers, gate state and badges exactly as sent", async () => {
  const { commandText } = await import("../../app/static/js/readout.js");
  const r = readout(answer({ thresholds: "frozen", gate_passed: true, valid: true, cooldown_conflict: false }));
  assert.deepEqual(r.values, { threat: 0.934, action: 0.8812, margin: 0.41, best: "DODGE_LEFT", cooldown: 0.95, hold: 0.3, valid: true, cooldownConflict: false });
  assert.equal(r.thresholds, null); // the server sent no threshold values: no tick is drawn
  assert.deepEqual(r.gateState, { threat: true, action: true, margin: true, votes: 3, votesNeeded: 3, ofLast: 4, passed: true, votesOk: true });
  assert.deepEqual(r.badges.map((b) => b.text), ["Thresholds frozen by calibration", "Calibration gate passed"]);
  assert.deepEqual(r.command, { vel_x: 0, vel_y: 0.5, yaw_rate: 0 });
  assert.equal(commandText(r, { vel_y: { unit: "m/s" } }), "Command issued to the simulated dog now: vel_y 0.50 m/s.");
  assert.equal(commandText(readout(answer({ issued: false })), null), "No command issued.");
  assert.equal(commandText(readout(answer({ issued: undefined })), null), "Command issued: not sent");
});

test("threshold values and a provisional student show only when the server sends them", () => {
  const r = readout(answer({ threshold_values: { threat: 0.95, action: 0.4, margin: "x" }, student_status: "PROVISIONAL development student", thresholds: "development", gate_passed: false }));
  assert.deepEqual(r.thresholds, { threat: 0.95, action: 0.4, margin: null });
  assert.deepEqual(r.badges.map((b) => [b.text, b.tone]), [["Provisional student", "warn"], ["Development thresholds, not calibrated", "warn"], ["Calibration gate not passed", "warn"]]);
  assert.deepEqual(readout(answer({ gate: undefined })).gateState, null);
  assert.equal(readout(answer({ margin: "0.4" })).values.margin, null);
});

test("the plain state line names what the dog does, in the visitor's words", async () => {
  const { plainState } = await import("../../app/static/js/readout.js");
  const s = (over) => plainState(readout(answer(over)));
  assert.equal(s({ action: "NO_OP", issued: false, cooldown_remaining_s: 0 }).text, "Calm. Nothing is coming.");
  assert.equal(s({ action: "UNCERTAIN", issued: false, cooldown_remaining_s: 0 }).text, "Something moved. Not sure yet.");
  assert.equal(s({ action: "DODGE_LEFT" }).text, "Dodging left!");
  assert.equal(s({ action: "DODGE_RIGHT" }).text, "Dodging right!");
  assert.equal(s({ action: "RETREAT" }).text, "Backing away!");
  assert.equal(s({ action: "NO_OP", issued: false, cooldown_remaining_s: 0.4 }).text, "Catching its breath");
  assert.equal(plainState(readout({ ok: false, status: 503, code: "infer_unavailable", message: "x", unavailable: true })).kind, "unavailable");
});

test("the why line comes from the gate's own fields", async () => {
  const { plainWhy } = await import("../../app/static/js/readout.js");
  const w = (over) => plainWhy(readout(answer(over)));
  assert.equal(w({}), "It was sure in 3 of its last 4 looks, so it moves.");
  assert.equal(w({ action: "NO_OP", issued: false, cooldown_remaining_s: 0, gate: { threat_ok: true, action_ok: true, margin_ok: true, votes: 1, votes_needed: 3, of_last: 4, passed: false } }),
    "Something seems to be coming. It wants to be sure 3 times before it moves (1 so far).");
  assert.equal(w({ action: "NO_OP", issued: false, cooldown_remaining_s: 0, gate: { threat_ok: false, votes: 0, votes_needed: 3, of_last: 4 } }),
    "Nothing in view looks like it is coming at it.");
  assert.equal(w({ action: "NO_OP", issued: false, cooldown_remaining_s: 0, gate: { threat_ok: true, action_ok: false, margin_ok: true } }),
    "Something seems to be coming, but it cannot yet tell which way to go.");
});

test("the honesty lines come from the student's status, its gate and its thresholds only", async () => {
  const { plainHonesty } = await import("../../app/static/js/readout.js");
  const h = (over) => plainHonesty(readout(answer(over)));
  assert.deepEqual(h({ student_status: "Thresholds frozen from 240 simulation dev trajectories. NOT been evaluated on held-out data yet." }),
    ["Trained only in simulation.", "This model has not yet reached our strictest accuracy goal in testing."]);
  assert.deepEqual(h({ student_status: "Held-out: the 0.99 target NOT met, in simulation." }),
    ["Trained only in simulation.", "This model has not yet reached our strictest accuracy goal in testing."]);
  assert.deepEqual(h({ student_status: "Calibrated in simulation; target met." }), ["Trained only in simulation."]);
  assert.deepEqual(h({}), []);
  assert.deepEqual(h({ thresholds: "development" }), ["Its reaction settings are still a first guess."]);
});

test("the server's frame age, window validity and cooldown conflict are shown, and development thresholds are flagged", () => {
  const r = readout(answer({ frame_age_ms: 41.26, frame_age_clamped: true, valid: true, cooldown_conflict: false, thresholds: "development" }));
  assert.equal(row(r, "Frame age at the server"), "41.3 ms (clamped)");
  assert.equal(row(r, "Window valid"), "yes");
  assert.equal(row(r, "Cooldown conflict"), "no");
  assert.equal(r.development, true);
  const f = readout(answer({ frame_age_ms: 10, valid: false, thresholds: "frozen" }));
  assert.equal(row(f, "Frame age at the server"), "10.0 ms");
  assert.equal(row(f, "Window valid"), "no");
  assert.equal(row(f, "Cooldown conflict"), "not sent");
  assert.equal(f.development, false);
});
