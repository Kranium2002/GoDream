// An orbit camera around a target, z up: yaw about z, pitch above the ground plane, distance from the target.
const PITCH_MIN = 0, PITCH_MAX = Math.PI / 2 - 0.03; // never below the target's height, never straight down

export function createOrbit({ target, distance, yaw, pitch, minDistance = 0.4, maxDistance = 12 }) {
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const o = {
    target: [...target], distance, yaw, pitch: clamp(pitch, PITCH_MIN, PITCH_MAX),
    eye() {
      const c = Math.cos(o.pitch);
      return [o.target[0] + o.distance * c * Math.cos(o.yaw), o.target[1] + o.distance * c * Math.sin(o.yaw), o.target[2] + o.distance * Math.sin(o.pitch)];
    },
    rotate(dYaw, dPitch) { o.yaw += dYaw; o.pitch = clamp(o.pitch + dPitch, PITCH_MIN, PITCH_MAX); },
    zoom(factor) { o.distance = clamp(o.distance * factor, minDistance, maxDistance); },
    setTarget(t) { o.target = [...t]; },
  };
  return o;
}
