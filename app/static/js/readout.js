// An api.infer() result -> what the Live readouts show. Values are the server's, only rounded; a field the server did
// not send is shown as not sent, never filled in.

const ACTIONS = { NO_OP: "No action", UNCERTAIN: "Uncertain: stays still", DODGE_LEFT: "Dodge left", DODGE_RIGHT: "Dodge right", RETREAT: "Retreat" };
// The action chip's tone: the quiet answer is neutral, the unsure one amber, a move the accent.
const TONES = { NO_OP: "neutral", UNCERTAIN: "warn", DODGE_LEFT: "accent", DODGE_RIGHT: "accent", RETREAT: "accent" };
const NOT_SENT = "not sent";

const finiteNum = (v) => typeof v === "number" && Number.isFinite(v);
const orNull = (v) => (finiteNum(v) ? v : null);
const boolOrNull = (v) => (typeof v === "boolean" ? v : null);
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
  const tv = d.threshold_values && typeof d.threshold_values === "object" ? d.threshold_values : null;
  return {
    state: "decision",
    development: d.thresholds === "development", // shown prominently: the thresholds are not calibrated
    studentStatus: typeof d.student_status === "string" && d.student_status ? d.student_status : null,
    gatePassed: typeof d.gate_passed === "boolean" ? d.gate_passed : null,
    action: { code, label: ACTIONS[code] ?? code, tone: TONES[code] ?? "neutral" },
    issued: typeof d.issued === "boolean" ? d.issued : null,
    command: commandOf(d.command),
    badges: badges(d),
    values: {
      threat: orNull(d.threat_confidence), action: orNull(d.action_confidence), margin: orNull(d.margin),
      best: typeof d.best_action === "string" ? d.best_action : null,
      cooldown: orNull(d.cooldown_remaining_s), hold: orNull(d.hold_remaining_s),
      valid: boolOrNull(d.valid), cooldownConflict: boolOrNull(d.cooldown_conflict),
    },
    // The gate's own thresholds, only when the server sends them (threshold_values: {threat, action, margin}).
    thresholds: tv ? { threat: orNull(tv.threat), action: orNull(tv.action), margin: orNull(tv.margin) } : null,
    gateState: g ? {
      threat: boolOrNull(g.threat_ok), action: boolOrNull(g.action_ok), margin: boolOrNull(g.margin_ok),
      votes: orNull(g.votes), votesNeeded: orNull(g.votes_needed), ofLast: orNull(g.of_last), passed: boolOrNull(g.passed),
      votesOk: finiteNum(g.votes) && finiteNum(g.votes_needed) ? g.votes >= g.votes_needed : null,
    } : null,
    timing: [
      ["Inference on the server", num(d.inference_ms, 1, " ms")],
      ["Server latency", num(d.latency_ms, 1, " ms")],
      ["Frame age at the server", num(d.frame_age_ms, 1, " ms") + (d.frame_age_clamped === true && finiteNum(d.frame_age_ms) ? " (clamped)" : "")],
    ],
    rows: [
      ["Gate thresholds", d.thresholds === "development" ? "DEVELOPMENT thresholds, not calibrated"
        : d.thresholds === "frozen" ? "frozen by the student's calibration" : NOT_SENT],
      ["Calibration precision gate", d.gate_passed === true ? "passed" : d.gate_passed === false ? "not passed" : NOT_SENT],
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

// Badges exactly as the answer says: which thresholds, whether the calibration passed its gate, and a provisional
// student when the server names its status.
function badges(d) {
  const out = [];
  if (typeof d.student_status === "string" && /provisional/i.test(d.student_status)) out.push({ text: "Provisional student", tone: "warn", title: d.student_status });
  if (d.thresholds === "development") out.push({ text: "Development thresholds, not calibrated", tone: "warn" });
  else if (d.thresholds === "frozen") out.push({ text: "Thresholds frozen by calibration", tone: "neutral" });
  if (d.gate_passed === true) out.push({ text: "Calibration gate passed", tone: "ok" });
  else if (d.gate_passed === false) out.push({ text: "Calibration gate not passed", tone: "warn" });
  return out;
}

const commandOf = (c) => (c && typeof c === "object" && !Array.isArray(c) && Object.values(c).every(finiteNum) ? { ...c } : null);

// ---- plain words for visitors, built only from the fields of a readout() view -----------------------------------
const MOVES = { DODGE_LEFT: "Dodging left!", DODGE_RIGHT: "Dodging right!", RETREAT: "Backing away!" };

// What the dog is doing now, in one short line: {text, tone, kind}. A move (issued now or still held) first, then the
// pause after a move, then the quiet answers. tone: calm | warn | accent | neutral | danger.
export function plainState(view) {
  if (view.state === "unavailable") return { text: "The dog cannot react to the camera right now.", tone: "neutral", kind: "unavailable" };
  if (view.state !== "decision") return { text: "The dog's answer did not arrive.", tone: "danger", kind: "error" };
  const code = view.action.code, v = view.values;
  if (MOVES[code]) return { text: MOVES[code], tone: "accent", kind: "move" };
  if (v.cooldown !== null && v.cooldown > 0) return { text: "Catching its breath", tone: "neutral", kind: "cooldown" };
  if (code === "UNCERTAIN") return { text: "Something moved. Not sure yet.", tone: "warn", kind: "unsure" };
  if (code === "NO_OP") return { text: "Calm. Nothing is coming.", tone: "calm", kind: "calm" };
  return { text: "The dog gave an answer this page does not know.", tone: "neutral", kind: "unknown" };
}

// One sentence on why it moves or stays still, from the gate's own fields.
export function plainWhy(view) {
  if (view.state === "unavailable") return "Its model is not loaded on this server, so the camera gives it no orders.";
  if (view.state !== "decision") return "The server could not answer this time; the page tries again in a few seconds.";
  const g = view.gateState, v = view.values, code = view.action.code;
  if (MOVES[code] && view.issued === true) {
    return g && g.votes !== null && g.ofLast !== null
      ? `It was sure in ${g.votes} of its last ${g.ofLast} looks, so it moves.` : "It is sure something is coming, so it moves.";
  }
  if (MOVES[code]) return "It is still getting out of the way.";
  if (v.cooldown !== null && v.cooldown > 0) return "It just moved, so it waits a moment before it can move again.";
  if (v.valid === false) return "It could not read this moment, so it waits for the next one.";
  if (!g || g.threat !== true) return code === "UNCERTAIN" ? "Something changed, but not enough to look like a threat." : "Nothing in view looks like it is coming at it.";
  if (g.action === false || g.margin === false) return "Something seems to be coming, but it cannot yet tell which way to go.";
  if (g.votesOk === false && g.votesNeeded !== null) {
    return `Something seems to be coming. It wants to be sure ${g.votesNeeded} times before it moves (${g.votes ?? 0} so far).`;
  }
  return "It is watching.";
}

// The honesty lines shown to visitors, from student_status, gate_passed and thresholds only. The exact server sentence
// belongs in the engineers' drawer.
export function plainHonesty(view) {
  if (view.state !== "decision") return [];
  const s = view.studentStatus ?? "", out = [];
  if (/simulat/i.test(s)) out.push("Trained only in simulation.");
  if (/\bNOT\b[^.]*\b(met|evaluated)\b/.test(s) || view.gatePassed === false) out.push("This model has not yet reached our strictest accuracy goal in testing.");
  if (view.development) out.push("Its reaction settings are still a first guess.");
  return out;
}

// The command line under the action: what was sent to the simulated dog, with the units /api/state publishes.
export function commandText(view, fields) {
  if (view.issued === null) return "Command issued: not sent";
  if (!view.issued) return "No command issued.";
  const parts = view.command ? Object.entries(view.command).filter(([, v]) => v !== 0)
    .map(([k, v]) => `${k} ${v.toFixed(2)}${fields?.[k]?.unit ? ` ${fields[k].unit}` : ""}`) : [];
  return `Command issued to the simulated dog now${parts.length ? `: ${parts.join(", ")}` : ""}.`;
}
