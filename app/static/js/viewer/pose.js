// The last two simulator poses and where to draw the dog between them. The drawing runs one arrival interval behind:
// after a pose arrives it moves from the previous pose to that one over the interval the two arrived apart, then holds.
// It never extrapolates past the newest pose and never invents one.
export function createPoseBuffer() {
  let prev = null, latest = null;
  const sameShape = (a, b) => a.pos.length === b.pos.length;
  return {
    push(pose, arrivedMs) {
      const entry = { pose, at: arrivedMs };
      prev = latest && sameShape(latest.pose, pose) ? latest : null;
      latest = entry;
    },
    sample(nowMs) {
      if (!latest) return null;
      if (!prev) return latest.pose;
      const span = latest.at - prev.at;
      const a = span > 0 ? Math.min(1, Math.max(0, (nowMs - latest.at) / span)) : 1;
      return blend(prev.pose, latest.pose, a);
    },
    clear() { prev = latest = null; },
  };
}

function blend(p, q, a) {
  return {
    pos: q.pos.map((v, i) => v.map((x, k) => p.pos[i][k] + (x - p.pos[i][k]) * a)),
    quat: q.quat.map((v, i) => {
      const u = p.quat[i];
      const sign = u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3] < 0 ? -1 : 1; // the short way round
      const r = u.map((x, k) => x + (sign * v[k] - x) * a);
      const l = Math.hypot(...r) || 1;
      return r.map((x) => x / l);
    }),
  };
}
