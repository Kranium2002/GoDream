// An api.infer() result -> what the Live readouts show. Values are the server's, only rounded.

const ACTIONS = { NO_OP: "No action", UNCERTAIN: "Uncertain: stays still", DODGE_LEFT: "Dodge left", DODGE_RIGHT: "Dodge right", RETREAT: "Retreat" };
const NOT_SENT = "not sent";

const finiteNum = (v) => typeof v === "number" && Number.isFinite(v);
const yesNo = (v) => (v === true ? "yes" : v === false ? "no" : NOT_SENT);
const num = (v, digits, unit = "") => (typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(digits)}${unit}` : NOT_SENT);

export function readout(result) {
  if (!result.ok) {
    return result.unavailable ? { state: "unavailable", message: result.message } : { state: "error", code: result.code, message: result.message };
  }
  const d = result.data;
  const code = typeof d.action === "string" ? d.action : NOT_SENT;
  const g = d.gate && typeof d.gate === "object" ? d.gate : null;
  const check = (label, v) => ({ label, ok: v === true });
  return {
    state: "decision",
    development: d.thresholds === "development", // shown prominently: the thresholds are not calibrated
    action: { code, label: ACTIONS[code] ?? code },
    issued: typeof d.issued === "boolean" ? d.issued : null,
    rows: [
      ["Threat confidence", num(d.threat_confidence, 2)],
      ["Best action", typeof d.best_action === "string" ? d.best_action : NOT_SENT],
      ["Action confidence", num(d.action_confidence, 2)],
      ["Margin", num(d.margin, 2)],
      ["Cooldown remaining", num(d.cooldown_remaining_s, 2, " s")],
      ["Hold remaining", num(d.hold_remaining_s, 2, " s")],
      ["Inference on the server", num(d.inference_ms, 1, " ms")],
      ["Server latency", num(d.latency_ms, 1, " ms")],
      ["Frame age at the server", num(d.frame_age_ms, 1, " ms") + (d.frame_age_clamped === true && finiteNum(d.frame_age_ms) ? " (clamped)" : "")],
      ["Window valid", yesNo(d.valid)],
      ["Cooldown conflict", yesNo(d.cooldown_conflict)],
    ],
    gate: g ? [
      check("Threat confidence high enough", g.threat_ok),
      check("Action confidence high enough", g.action_ok),
      check("Margin wide enough", g.margin_ok),
      check(`Votes: ${num(g.votes, 0)} of ${num(g.votes_needed, 0)} needed in the last ${num(g.of_last, 0)}`,
        typeof g.votes === "number" && typeof g.votes_needed === "number" && g.votes >= g.votes_needed),
      check(g.passed === true ? "Gate passed" : "Gate closed", g.passed),
    ] : [],
  };
}
