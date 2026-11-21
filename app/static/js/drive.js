// Driving the simulated dog by hand, and where the model's centre square sits on the camera preview.

// /api/state's command ranges: {name: {unit, min, max}} under commands (or command_fields on the older server).
export const commandFields = (state) => state?.commands ?? state?.command_fields ?? null;

const MOVES = { forward: ["vel_x", 1], back: ["vel_x", -1], left: ["vel_y", 1], right: ["vel_y", -1], turn_left: ["yaw_rate", 1], turn_right: ["yaw_rate", -1] };

// A full command (every field the server publishes) moving along one field by `fraction` of its published range in
// that direction; stop is all zeros. null when the direction is unknown or the ranges do not allow it.
export function driveCommand(direction, fields, fraction) {
  if (!fields || typeof fields !== "object") return null;
  const cmd = Object.fromEntries(Object.keys(fields).map((k) => [k, 0]));
  if (direction === "stop") return cmd;
  const move = MOVES[direction];
  if (!move || !fields[move[0]]) return null;
  const [name, sign] = move;
  const limit = sign > 0 ? fields[name].max : fields[name].min;
  if (!(typeof limit === "number" && limit * sign > 0)) return null;
  cmd[name] = fraction * limit;
  return cmd;
}

// The centre square of a W x H frame, in the pixels of a box that shows the frame with object-fit: contain.
export function cropRect(W, H, boxW, boxH) {
  if (!(W > 0 && H > 0 && boxW > 0 && boxH > 0)) return null;
  const scale = Math.min(boxW / W, boxH / H);
  const ox = (boxW - W * scale) / 2, oy = (boxH - H * scale) / 2, S = Math.min(W, H);
  return { left: ox + Math.floor((W - S) / 2) * scale, top: oy + Math.floor((H - S) / 2) * scale, size: S * scale };
}
