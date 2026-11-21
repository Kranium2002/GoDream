"""Check 4: the DODGE objective on hand-made outcome distributions, and physics-only decisions.

FIXTURE CHECK - mechanics only. Outcome distributions here are written by hand (point masses
and one hand-made spread); decide_batch runs with model=None (physics only, zero residual by
definition of that method) on the fixture config. Nothing here is a prediction of any model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_offline\\c4_objective_decide.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import numpy as np  # noqa: E402

from fixture_dodge import FIXTURE_DECLARED, Checks, fixture_config, fixture_throw_sampler  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.types import OutcomeDistribution, ResidualDistribution  # noqa: E402
from learned_judgment.skills import load_skill  # noqa: E402

checks = Checks()
config = fixture_config(load_config())
spec = load_skill("dodge", config, throw_sampler=fixture_throw_sampler(config), **FIXTURE_DECLARED)
levels = np.asarray(require(config, "engine.quantile_levels"), dtype=np.float64)
spread_levels = tuple(require(config, "engine.spread_levels"))
w = spec.weights
print(f"\nweights (config skills.dodge.objective_weights): {w}")
print(f"quantile levels (config engine.quantile_levels): {levels.tolist()}")


def point(rows):  # noqa: ANN001, ANN201
    """rows: [min_separation_m, collision, fall, recovery_time_s, lateral_command_m]"""
    return OutcomeDistribution.point(spec.outcome_names, np.asarray(rows, dtype=np.float64), levels)


def score(rows):  # noqa: ANN001, ANN201
    return spec.objective(point(rows))


print("\n== exact value of one point row ==")
row = [0.3, 0.0, 0.0, 0.5, 0.4]
hand = w.separation * min(0.3, w.separation_cap_m) - w.motion * 0.4 - w.recovery * 0.5
value = float(score([row])[0])
print(f"  objective {value} (hand {hand})")
checks.check("point row scores the hand-computed value", abs(value - hand) < 1e-12)

print("\n== collision, fall, contact margin ==")
s = score([[0.3, 0.0, 0.0, 0.0, 0.2], [0.3, 1.0, 0.0, 0.0, 0.2], [0.3, 0.0, 1.0, 0.0, 0.2]])
print(f"  no collision {s[0]:.4f}, collision {s[1]:.4f}, fall {s[2]:.4f}")
checks.check("a collision scores below the same outcome without one", s[1] < s[0])
checks.check("a fall scores below the same outcome without one", s[2] < s[0])
checks.check("collision penalty equals the collision weight", abs((s[0] - s[1]) - w.collision) < 1e-12)
margin = score([[w.contact_margin_m * 0.5, 0.0, 0.0, 0.0, 0.2], [w.contact_margin_m * 2.0, 0.0, 0.0, 0.0, 0.2]])
print(f"  separation inside the contact margin {margin[0]:.4f}, outside {margin[1]:.4f}")
checks.check("a separation inside the contact margin is penalised as contact", margin[1] - margin[0] > 0.9 * w.collision)

print("\n== separation reward up to the cap ==")
separations = [0.05, 0.1, 0.2, 0.4, w.separation_cap_m, w.separation_cap_m + 0.3]
s = score([[sep, 0.0, 0.0, 0.0, 0.2] for sep in separations])
print(f"  separations {separations}: {np.round(s, 4).tolist()}")
checks.check("more separation scores higher below the cap", bool(np.all(np.diff(s[:5]) > 0.0)))
checks.check("separation beyond the cap scores the same as the cap", s[5] == s[4])

print("\n== commanded motion and recovery ==")
s = score([[0.3, 0.0, 0.0, 0.0, 0.2], [0.3, 0.0, 0.0, 0.0, 0.6], [0.3, 0.0, 0.0, 1.0, 0.2]])
checks.check("more commanded sideways travel scores lower", s[1] < s[0])
checks.check("a longer recovery scores lower", s[2] < s[0])

print("\n== predictive uncertainty lowers the score ==")
physics = np.array([[0.3, 0.0, 0.0, 0.0, 0.2]] * 2)
narrow = np.zeros((2, 5, levels.size))
narrow[1, 0, :] = np.linspace(-0.2, 0.2, levels.size)  # row 1: same median, wide separation spread (hand-made)
residual = ResidualDistribution.from_quantiles(
    mean=np.zeros((2, 5)), median=np.zeros((2, 5)), quantile_levels=levels, quantiles=narrow, spread_levels=spread_levels
)
predicted = OutcomeDistribution.from_physics_and_residual(spec.outcome_names, physics, residual, spec.outcome_low, spec.outcome_high)
s = spec.objective(predicted)
print(f"  point {s[0]:.4f}, wide spread with the same median {s[1]:.4f}")
checks.check("a wider separation distribution scores lower than a point with the same median", s[1] < s[0])
other = OutcomeDistribution.point(("a", "b", "c", "d", "e"), physics, levels)
checks.raises("a distribution of other outcomes is refused", ValueError, lambda: spec.objective(other), "are not the DODGE outcomes")

print("\n== decide_batch, physics only ==")
layout = build_layout(spec)
points = spec.candidate_points_per_dim
situations = spec.sample_situations(stream_rng(3, "fixture/dodge/decide"), 40)
decisions = decide_batch(
    spec, situations, model=None, layout=layout, points_per_dim=points,
    quantile_levels=levels, spread_levels=spread_levels, method=PHYSICS_ONLY,
)
chosen = np.stack([d.action for d in decisions])
checks.check("40 decisions with 108 candidates each", len(decisions) == 40 and all(d.candidates.shape == (108, 3) for d in decisions))
checks.check("every chosen action is inside the declared bounds", bool(np.all(spec.action_in_bounds(chosen))))
checks.check("physics only: chosen index equals the physics index", all(d.chosen_index == d.physics_index for d in decisions))
checks.check("no provider request behind physics-only decisions", all(d.request_hash is None and d.request_hashes == () for d in decisions))

gravity = spec.physics.gravity_m_s2
aimed = np.array([[3.0, 0.0, 0.0, -5.0, 0.0, 0.5 * gravity * 0.6, 0.0, FIXTURE_DECLARED["declared_payload_kg"], 0.0]])
decision = decide_batch(
    spec, aimed, model=None, layout=layout, points_per_dim=points,
    quantile_levels=levels, spread_levels=spread_levels, method=PHYSICS_ONLY,
)[0]
best = decision.action
chosen_outcome = decision.predicted.mean[decision.chosen_index]
print(f"  ball aimed at the trunk: chosen side {best[0]} m/s, delay {best[1]:.2f} s, hold {best[2]:.2f} s; "
      f"guessed separation {chosen_outcome[0]:.4f} m, collision {chosen_outcome[1]:.4f}")
checks.check("ball aimed at the trunk: the choice is not 'no dodge'", best[0] != 0.0)
no_dodge_scores = decision.scores[decision.candidates[:, 0] == 0.0]
checks.check("ball aimed at the trunk: every no-dodge candidate scores below the choice", bool(np.all(no_dodge_scores < decision.scores[decision.chosen_index])))

checks.finish()
