"""Check 6: DODGE loss and success on small hand-written outcome arrays, and engine.dream with
model=None (physics only) on the fixture config.

FIXTURE CHECK - mechanics only. The outcome arrays are hand-written, not measured; the dreams
are physics-only decisions on fixture throws (no teacher, no provider request). Nothing is
written to disk.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_offline\\c6_loss_success_dream.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import numpy as np  # noqa: E402

from fixture_dodge import FIXTURE_DECLARED, Checks, fixture_config, fixture_throw_sampler  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.dream import DreamSet, dream  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.skills import load_skill  # noqa: E402

checks = Checks()
config = fixture_config(load_config())
spec = load_skill("dodge", config, throw_sampler=fixture_throw_sampler(config), **FIXTURE_DECLARED)

print("\n== loss and success (preregistered DODGE definitions) ==")
# columns: min_separation_m, collision, fall, recovery_time_s, lateral_command_m (hand-written)
measured = np.array(
    [
        [0.30, 0.0, 0.0, 0.0, 0.20],  # clean dodge
        [0.00, 1.0, 0.0, 0.5, 0.20],  # hit
        [0.10, 0.0, 1.0, 2.0, 0.40],  # missed the ball but fell
        [0.00, 1.0, 1.0, 1.6, 0.00],  # hit and fell
        [0.01, 0.0, 0.0, 0.3, 0.60],  # very close, no contact, no fall
    ]
)
loss = spec.loss(measured)
success = spec.success(measured)
print(f"  loss {loss.tolist()}\n  success {success.tolist()}")
checks.check("loss = 1 exactly where collision or fall is 1", loss.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0])
checks.check("loss is float64 of shape (n,)", loss.dtype == np.float64 and loss.shape == (5,))
checks.check("success = no collision and no fall", success.tolist() == [True, False, False, False, True])
checks.check("success is bool of shape (n,)", success.dtype == np.bool_ and success.shape == (5,))
checks.check("success is the complement of loss", np.array_equal(success, loss == 0.0))
checks.check("empty input gives empty output", spec.loss(np.empty((0, 5))).shape == (0,) and spec.success(np.empty((0, 5))).shape == (0,))
checks.raises("four columns instead of five", ValueError, lambda: spec.loss(measured[:, :4]), "measured must have shape (n, 5)")
bad = measured.copy()
bad[0, 0] = np.inf
checks.raises("non-finite measured value", ValueError, lambda: spec.success(bad), "non-finite")

print("\n== engine.dream with model=None ==")
levels = require(config, "engine.quantile_levels")
spread_levels = tuple(require(config, "engine.spread_levels"))
layout = build_layout(spec)
n_dreams = 60
dreams = dream(
    spec,
    None,
    layout,
    n_dreams=n_dreams,
    rng=stream_rng(int(require(config, "runtime.seed")), "fixture/dodge/dreams/situations"),
    points_per_dim=spec.candidate_points_per_dim,
    quantile_levels=levels,
    spread_levels=spread_levels,
    method=PHYSICS_ONLY,
)
print(f"  DreamSet: {dreams.n_rows} rows x {dreams.n_candidates} candidates, method {dreams.method}")
checks.check("a DreamSet", isinstance(dreams, DreamSet))
checks.check("108 candidates per row", dreams.n_candidates == 108 and dreams.candidates.shape == (n_dreams, 108, 3))
checks.check("one row per dreamed situation", dreams.n_rows == n_dreams and dreams.situations.shape == (n_dreams, 9))
checks.check("names follow the spec", dreams.situation_names == spec.situation_names and dreams.action_names == spec.action_names and dreams.outcome_names == spec.outcome_names)
checks.check("primary outcome min_separation_m", dreams.primary_outcome == "min_separation_m")
checks.check("dreamed situations carry the declared values", bool(np.all(dreams.situations[:, 7] == 1.0) and np.all(dreams.situations[:, 8] == 0.0)))
checks.check("chosen actions inside the bounds", bool(np.all(spec.action_in_bounds(dreams.chosen_actions))))
checks.check("physics only: chosen equals the physics choice", np.array_equal(dreams.chosen, dreams.physics_choice))
checks.check("no provider request behind the dreams", dreams.request_hashes == ())
checks.check("physics only: zero predictive spread", bool(np.all(dreams.chosen_outcome_spread == 0.0)))
checks.check(
    "chosen physics equals physics_guess at the chosen actions",
    np.array_equal(dreams.chosen_physics, spec.physics_guess(dreams.situations, dreams.chosen_actions)),
)
sides = dreams.chosen_actions[:, 0]
print(f"  chosen side velocities: {np.unique(sides, return_counts=True)}")

checks.finish()
