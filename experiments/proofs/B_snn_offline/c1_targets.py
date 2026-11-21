"""Check 1: learned_judgment.students.targets on a fixture dream built by the real engine.

FIXTURE CHECK - mechanics only. Soft targets sum to 1, sharpen as the temperature falls, and
their argmax is the dream's choice; dream_regret is 0 exactly for the dream's best candidates
and positive otherwise; flat scores and invalid arguments are refused. No file is written.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c1_targets.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any trained student")

import sys  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.students.targets import dream_regret, score_spread, soft_targets  # noqa: E402

checks = Checks()
config = load_config()
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config)
scores = dreams.scores
n, c = scores.shape
print(f"\nfixture dream: {n} rows x {c} candidates, grid {FIXTURE_POINTS_PER_DIM}, method {dreams.method!r}, "
      f"request_hashes {list(dreams.request_hashes)}")

print("\n== score_spread ==")
spread = score_spread(scores)
checks.expect(spread == float(np.median(np.std(scores, axis=1))), f"spread {spread:.6f} = median over rows of the row std")
checks.raises("all rows flat", ValueError, lambda: score_spread(np.ones((5, 4))), "score spread is 0")
mostly_flat = np.vstack([np.ones((3, 4)), np.arange(8.0).reshape(2, 4)])
checks.raises("3 of 5 rows flat (median std 0)", ValueError, lambda: score_spread(mostly_flat), "3 of 5 rows")
mostly_varied = np.vstack([np.ones((2, 4)), np.arange(12.0).reshape(3, 4)])
checks.expect(score_spread(mostly_varied) > 0.0, "2 flat rows of 5: spread > 0, accepted")
checks.raises("non-finite scores", ValueError, lambda: score_spread(np.array([[0.0, np.nan]])), "non-finite")
checks.raises("1-D scores", ValueError, lambda: score_spread(np.zeros(4)), "shape")

print("\n== soft_targets ==")
temperature = float(require(config, "students.snn.target_temperature"))
targets = soft_targets(scores, temperature=temperature, spread=spread)
checks.expect(targets.shape == (n, c), f"shape {targets.shape}")
row_error = float(np.max(np.abs(targets.sum(axis=1) - 1.0)))
checks.expect(row_error <= 1e-12, f"rows sum to 1 (max |sum - 1| = {row_error:.2e}) at config temperature {temperature}")
checks.expect(bool(np.all(targets >= 0.0)), "targets are non-negative")
checks.expect(bool(np.array_equal(np.argmax(targets, axis=1), dreams.chosen)), "argmax of every row = the dream's choice (DreamSet.chosen)")
checks.expect(bool(np.array_equal(np.argmax(targets, axis=1), np.argmax(scores, axis=1))), "argmax of every row = argmax of the scores")

temperatures = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1]
max_probs = [soft_targets(scores, temperature=t, spread=spread).max(axis=1) for t in temperatures]
entropies = []
for t in temperatures:
    q = soft_targets(scores, temperature=t, spread=spread)
    entropies.append(float(np.mean(-np.sum(np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0), axis=1))))
means = [float(m.mean()) for m in max_probs]
print("  mechanics numbers (fixture): temperature -> mean max target probability, mean entropy (nats)")
for t, m, e in zip(temperatures, means, entropies):
    print(f"    {t:>5} -> {m:.4f}, {e:.4f}")
checks.expect(all(b > a for a, b in zip(means, means[1:])), "mean max probability rises strictly as the temperature falls")
checks.expect(all(bool(np.all(b >= a)) for a, b in zip(max_probs, max_probs[1:])), "every row's max probability is non-decreasing as the temperature falls")
checks.expect(all(b < a for a, b in zip(entropies, entropies[1:])), "mean entropy falls strictly as the temperature falls")
for bad in (0.0, -1.0, float("nan"), float("inf")):
    checks.raises(f"temperature {bad}", ValueError, lambda bad=bad: soft_targets(scores, temperature=bad, spread=spread), "temperature")
checks.raises("spread 0", ValueError, lambda: soft_targets(scores, temperature=1.0, spread=0.0), "spread")
checks.raises("non-finite scores", ValueError, lambda: soft_targets(np.array([[0.0, np.inf]]), temperature=1.0, spread=1.0), "non-finite")

print("\n== dream_regret ==")
regret_chosen = dream_regret(scores, dreams.chosen)
checks.expect(bool(np.all(regret_chosen == 0.0)), "regret of the dream's choice is exactly 0 for every row")
row_max = scores.max(axis=1)
consistent = True
for candidate in range(c):
    regret = dream_regret(scores, np.full(n, candidate))
    consistent &= bool(np.all(regret >= 0.0))
    consistent &= bool(np.array_equal(regret > 0.0, scores[:, candidate] < row_max))
    consistent &= bool(np.array_equal(regret == 0.0, scores[:, candidate] == row_max))
checks.expect(consistent, f"for all {c} candidates: regret >= 0, = 0 exactly when the candidate is a best one, > 0 otherwise")
ties = int(np.sum(np.sum(scores == row_max[:, None], axis=1) > 1))
print(f"  mechanics numbers (fixture): {ties} of {n} rows have more than one best candidate (e.g. no side-step: every delay ties)")
checks.raises("index out of range", ValueError, lambda: dream_regret(scores, np.full(n, c)), "outside")
checks.raises("float indices", ValueError, lambda: dream_regret(scores, np.zeros(n)), "integer")
checks.raises("wrong length", ValueError, lambda: dream_regret(scores, np.zeros(n - 1, dtype=np.int64)), "shape")

checks.expect("torch" not in sys.modules, "torch was not imported by targets or the engine dream step")
checks.finish()
