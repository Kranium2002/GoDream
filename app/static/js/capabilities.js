// /api/state capabilities -> [{name, available, reason}]. Accepts the contract shape {name: {available, reason}} and the
// older {name: bool}; every contract name is listed, a missing one as unavailable.

export const CONTRACT_CAPABILITIES = ["drive", "head_camera", "scenarios", "chat", "runs", "results", "infer"];

const NO_REASON = "The server gave no reason.";

function entry(name, value) {
  if (value === undefined) return { name, available: false, reason: "The server does not report this capability." };
  if (typeof value === "boolean") return { name, available: value, reason: value ? "" : NO_REASON };
  if (value === null || typeof value !== "object") return { name, available: false, reason: "The server sent an unreadable entry." };
  const available = value.available === true;
  const reason = typeof value.reason === "string" && value.reason ? value.reason : available ? "" : NO_REASON;
  return { name, available, reason };
}

export function normaliseCapabilities(raw) {
  const given = raw !== null && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
  const names = [...CONTRACT_CAPABILITIES, ...Object.keys(given).filter((n) => !CONTRACT_CAPABILITIES.includes(n))];
  return names.map((name) => entry(name, given[name]));
}
