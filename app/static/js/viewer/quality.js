// Keeps the viewer's frame time in budget on integrated graphics by changing the internal resolution scale.
// Rule: the time between drawn frames is smoothed (exponential average, weight 0.1). At most once every settleMs:
// above slowMs (under 25 frames per second at 40 ms) the scale drops by 15%, never below min; under fastMs (over about
// 45 frames per second at 22 ms) it rises by 8%, never above max. A gap over 250 ms is a pause in drawing (nothing
// changed, the tab was hidden), not a slow frame: it is skipped and the average starts again.
export function createQuality({ min = 0.5, max = 1, slowMs = 40, fastMs = 22, settleMs = 500 } = {}) {
  let last = null, avg = null, changedAt = null;
  const q = {
    scale: max,
    frame(now) {
      const gap = last === null ? null : now - last;
      last = now;
      if (gap === null || gap > 250) { avg = null; return q.scale; }
      avg = avg === null ? gap : avg + (gap - avg) * 0.1;
      changedAt ??= now;
      if (now - changedAt < settleMs) return q.scale;
      if (avg > slowMs && q.scale > min) { q.scale = Math.max(min, q.scale * 0.85); changedAt = now; }
      else if (avg < fastMs && q.scale < max) { q.scale = Math.min(max, q.scale * 1.08); changedAt = now; }
      return q.scale;
    },
  };
  return q;
}
