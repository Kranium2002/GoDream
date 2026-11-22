"""R05: soft_targets / score_spread / dream_regret numerics, and the KL loss on exact-zero targets.

  (a) large score ranges and small temperatures: finite, rows sum to 1, exact zeros appear;
  (b) torch kl_div(log_softmax(logits), targets, "batchmean") on targets with exact zeros: finite
      value and finite gradient (xlogy convention), and equal to the hand KL;
  (c) all-equal rows -> uniform target; ties -> equal mass on the tied best candidates;
      dream_regret under ties;
  (d) one global spread for rows of very different score scale (a DODGE-like objective where some
      situations contain a collision-penalised candidate and others do not): how much the target
      on the small-scale rows distinguishes their best candidate;
  (e) temperature x spread underflowing to 0 although each factor passes the > 0 check.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import numpy as np  # noqa: E402

import _review_common  # noqa: E402,F401
from learned_judgment.students.targets import dream_regret, score_spread, soft_targets  # noqa: E402

rng = np.random.default_rng(0)
print("== (a) large ranges / small temperature ==")
scores = np.column_stack([np.zeros(4), -rng.uniform(1, 1e6, size=(4, 6))])
for temperature in (0.5, 1e-3, 1e-12):
    q = soft_targets(scores, temperature=temperature, spread=1.0)
    print(f"  T={temperature:g}: finite={bool(np.all(np.isfinite(q)))} max|rowsum-1|={float(np.max(np.abs(q.sum(1) - 1))):.1e} exact zeros={int(np.sum(q == 0.0))}/{q.size}")

print("== (b) torch KL with exact-zero targets ==")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

q = soft_targets(scores, temperature=1e-3, spread=1.0)
logits = torch.tensor(rng.normal(size=q.shape), dtype=torch.float64, requires_grad=True)
target = torch.as_tensor(q, dtype=torch.float64)
loss = F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")
loss.backward()
log_p = F.log_softmax(logits.detach(), dim=1).numpy()
hand = float(np.mean(np.sum(np.where(q > 0, q * (np.log(np.where(q > 0, q, 1.0)) - log_p), 0.0), axis=1)))
print(f"  loss={float(loss.detach()):.6f} hand KL={hand:.6f} finite grad={bool(torch.all(torch.isfinite(logits.grad)))}")

print("== (c) ties and flat rows ==")
tied = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 5.0, 5.0, 0.0], [3.0, 3.0, 1.0, 3.0]])
q = soft_targets(tied, temperature=0.5, spread=1.0)
print(f"  targets: {q.round(4).tolist()}")
print(f"  argmax (first max): {np.argmax(q, axis=1).tolist()}; dream_regret of choices [3,2,3]: {dream_regret(tied, np.array([3, 2, 3])).tolist()}")

print("== (d) one global spread for rows of different score scale ==")
C = 35
small = -0.05 * rng.uniform(0, 1, size=(500, C))           # no collision risk: only motion cost differs
large = -0.05 * rng.uniform(0, 1, size=(600, C))
large[:, : C // 2] -= 10.0 * rng.uniform(0.5, 1.0, size=(600, C // 2))  # half the candidates risk a collision
mixed = np.vstack([small, large])
spread = score_spread(mixed)
q = soft_targets(mixed, temperature=0.5, spread=spread)
per_row = soft_targets(small, temperature=0.5, spread=score_spread(small))
print(f"  global spread (median row std) = {spread:.4f}; std of the small-scale rows ~ {float(np.median(small.std(1))):.4f}")
print(f"  small-scale rows: mean max target probability {float(q[:500].max(1).mean()):.4f} (uniform = {1 / C:.4f}); "
      f"with their own spread it would be {float(per_row.max(1).mean()):.4f}")
print(f"  small-scale rows: KL(target || uniform) = {float(np.mean(np.sum(q[:500] * np.log(q[:500] * C), axis=1))):.2e} nats")
print(f"  large-scale rows: mean max target probability {float(q[500:].max(1).mean()):.4f}")

print("== (e) temperature x spread underflow ==")
try:
    q = soft_targets(np.array([[0.0, 0.0, -1.0]]), temperature=1e-200, spread=1e-200)
    print(f"  accepted; targets = {q.tolist()} finite={bool(np.all(np.isfinite(q)))}")
except ValueError as error:
    print(f"  refused: {error}")
