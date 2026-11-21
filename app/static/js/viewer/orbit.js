// An orbit camera around a target, z up: yaw about z, pitch above the ground plane, distance from the target.
const PITCH_MIN = 0.05, PITCH_MAX = Math.PI / 2 - 0.03; // always above the target's height, never straight down

export function createOrbit({ target, distance, yaw, pitch, minDistance = 0.4, maxDistance = 12 }) {
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const o = {
    target: [...target], distance, yaw, pitch: clamp(pitch, PITCH_MIN, PITCH_MAX),
    eye(baseYaw = 0) { // baseYaw: the follow camera's heading; o.yaw is then the person's angle around the dog
      const c = Math.cos(o.pitch), yaw = baseYaw + o.yaw;
      return [o.target[0] + o.distance * c * Math.cos(yaw), o.target[1] + o.distance * c * Math.sin(yaw), o.target[2] + o.distance * Math.sin(o.pitch)];
    },
    rotate(dYaw, dPitch) { o.yaw += dYaw; o.pitch = clamp(o.pitch + dPitch, PITCH_MIN, PITCH_MAX); },
    zoom(factor) { o.distance = clamp(o.distance * factor, minDistance, maxDistance); },
    setTarget(t) { o.target = [...t]; },
  };
  return o;
}

// The heading (yaw about z) of a [w, x, y, z] quaternion, in radians.
export const headingOf = ([w, x, y, z]) => Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));

// Exponential smoothing for the follow camera: after dtMs the value has moved 1 - exp(-dt / tau) of the way to goal.
export const smooth = (value, goal, dtMs, tauMs) => value + (goal - value) * (1 - Math.exp(-dtMs / tauMs));
export function smoothAngle(value, goal, dtMs, tauMs) {
  const d = Math.atan2(Math.sin(goal - value), Math.cos(goal - value)); // the short way round
  return value + d * (1 - Math.exp(-dtMs / tauMs));
}
