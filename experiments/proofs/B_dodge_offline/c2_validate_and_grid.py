"""Check 2: the DODGE SkillSpec builds and validates with fixture values; action ranges, grid,
build-time refusals and situation sampling.

FIXTURE CHECK - mechanics only. The config is a copy of the real config.yaml whose null
skills.dodge keys (only those) are filled with labelled fixture values; the throw sampler is
a fixture defined in fixture_dodge.py, not Person A's throw model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_offline\\c2_validate_and_grid.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import numpy as np  # noqa: E402

from fixture_dodge import (  # noqa: E402
    FIXTURE_DECLARED,
    FIXTURE_VALUES,
    Checks,
    changed_leaves,
    fixture_config,
    fixture_throw_sampler,
    with_value,
)
from learned_judgment.config import ConfigError, load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.skills import available_skills, load_skill  # noqa: E402
from learned_judgment.skills.base import CommandSchema, SkillSpecError, validate_spec  # noqa: E402
from learned_judgment.skills.dodge import BALL_DOG_KEYS, DodgeSpec  # noqa: E402

checks = Checks()
real = load_config()
config = fixture_config(real)
schema = CommandSchema.from_config(config)
sampler = fixture_throw_sampler(config)

print("\n== fixture config: only null skills.dodge keys were filled ==")
for key, value in FIXTURE_VALUES.items():
    print(f"  FIXTURE {key} = {value}")
changed = changed_leaves(real, config)
checks.check("changed keys are exactly the fixture keys", sorted(changed) == sorted(FIXTURE_VALUES), f"{len(changed)} keys")

print("\n== load_skill runs validate_spec ==")
spec = load_skill("dodge", config, throw_sampler=sampler, **FIXTURE_DECLARED)
checks.check("load_skill('dodge', fixture config) returns a DodgeSpec", isinstance(spec, DodgeSpec), spec.name)
settings = dict(
    hidden_value_names=tuple(require(config, "world.hidden_effects")),
    quantile_levels=require(config, "engine.quantile_levels"),
    probe_seed=require(config, "runtime.seed"),
    probe_situations=require(config, "engine.spec_validation.probe_situations"),
    probe_points_per_dim=require(config, "engine.spec_validation.probe_points_per_dim"),
)
try:
    validate_spec(spec, schema, **settings)
    checks.check("validate_spec passes with config-read settings", True, f"hidden={settings['hidden_value_names']}")
except SkillSpecError as error:
    checks.check("validate_spec passes with config-read settings", False, str(error))

print("\n== declared order of the agreed interface ==")
checks.check(
    "situation order",
    spec.situation_names == (*BALL_DOG_KEYS, "declared_payload_kg", "declared_surface"),
    f"{spec.situation_names}",
)
checks.check("action order", spec.action_names == ("side_velocity_m_s", "start_delay_s", "hold_duration_s"))
checks.check(
    "outcome order",
    spec.outcome_names == ("min_separation_m", "collision", "fall", "recovery_time_s", "lateral_command_m"),
)
checks.check("primary outcome", spec.primary_outcome == "min_separation_m")
checks.check("outcome kinds", [o.kind for o in spec.outcomes] == ["regression", "binary", "binary", "regression", "regression"])
checks.check("regression outcomes bounded below by 0", all(o.low == 0.0 for o in spec.outcomes if o.kind == "regression"))
checks.check("declared_surface levels from config", spec.situation[-1].levels == tuple(require(config, "skills.dodge.declared_surfaces")))
checks.check("hidden effects documented", len(spec.hidden_effects) >= 10, f"{len(spec.hidden_effects)} entries")

print("\n== action ranges ==")
vel_y = schema.field(require(config, "skills.dodge.action.side_velocity_field"))
side = spec.actions[0]
print(f"  side_velocity_m_s: field={side.command_field} unit={side.unit} [{side.low}, {side.high}]; verified [{vel_y.low}, {vel_y.high}]")
checks.check("side velocity is bound to vel_y", side.role == "command" and side.command_field == "vel_y")
checks.check("side velocity range equals the verified vel_y range", (side.low, side.high) == (vel_y.low, vel_y.high))
checks.check("side velocity unit equals the verified unit", side.unit == vel_y.unit)
for dim, key in ((spec.actions[1], "start_delay_s"), (spec.actions[2], "hold_duration_s")):
    expected = tuple(float(v) for v in require(config, f"skills.dodge.action.{key}"))
    checks.check(f"{key} is timing with the config range", dim.role == "timing" and (dim.low, dim.high) == expected, f"{expected}")

print("\n== candidate grid ==")
situation = spec.sample_situations(stream_rng(1, "fixture/dodge/grid"), 1)[0]
grid = spec.candidate_actions(situation, spec.candidate_points_per_dim)
print(f"  points per dim (config): {spec.candidate_points_per_dim}; grid shape {grid.shape}")
checks.check("grid has 9 x 4 x 3 = 108 candidates", grid.shape == (108, 3))
checks.check("grid inside the declared bounds", bool(np.all(spec.action_in_bounds(grid))))
sides, delays, holds = (np.unique(grid[:, i]) for i in range(3))
print(f"  side values {sides.tolist()}\n  delays {delays.tolist()}\n  holds {holds.tolist()}")
checks.check("9 side velocities in steps of 0.25, including no dodge (0.0)", sides.size == 9 and 0.0 in sides and np.allclose(np.diff(sides), 0.25))
checks.check("4 start delays from 0.0 to 0.6", delays.size == 4 and delays[0] == 0.0 and delays[-1] == 0.6)
checks.check("3 holds from 0.2 to 1.0", holds.size == 3 and holds[0] == 0.2 and holds[-1] == 1.0)

print("\n== build-time refusals ==")
build = lambda cfg, **kw: DodgeSpec(cfg, CommandSchema.from_config(cfg), **kw)  # noqa: E731
checks.raises(
    "delay max + hold max longer than the episode",
    ConfigError,
    lambda: build(with_value(config, "skills.dodge.episode_duration_s", 1.5)),
    "exceeds 'skills.dodge.episode_duration_s'",
)
checks.raises(
    "side field not in the verified schema",
    ConfigError,
    lambda: build(with_value(config, "skills.dodge.action.side_velocity_field", "vel_z")),
    "'skills.dodge.action.side_velocity_field' names 'vel_z', which is not a verified command field",
)
checks.raises(
    "forward field not in the verified schema",
    ConfigError,
    lambda: build(with_value(config, "skills.dodge.action.forward_velocity_field", "forward")),
    "'skills.dodge.action.forward_velocity_field' names 'forward'",
)
checks.raises(
    "two action fields naming the same command field",
    ConfigError,
    lambda: build(with_value(config, "skills.dodge.action.forward_velocity_field", "vel_y")),
    "three different command fields",
)
checks.raises(
    "forward speed range beyond the verified vel_x range",
    ConfigError,
    lambda: build(with_value(config, "skills.dodge.situation_ranges.dog_forward_speed_m_s", [0.0, 1.5])),
    "exceeds its verified range",
)
checks.raises(
    "candidate_points_per_dim with two entries",
    ConfigError,
    lambda: build(with_value(config, "skills.dodge.candidate_points_per_dim", [9, 4])),
    "candidate_points_per_dim",
)
checks.raises(
    "malformed situation range",
    ConfigError,
    lambda: build(with_value(config, "skills.dodge.situation_ranges.ball_rel_y_m", [2.0, -2.0])),
    "'skills.dodge.situation_ranges.ball_rel_y_m' must be [low, high]",
)
checks.raises(
    "non-positive response time",
    ConfigError,
    lambda: build(with_value(config, "skills.dodge.physics.response_time_s", 0.0)),
    "'skills.dodge.physics.response_time_s' must be > 0",
)
checks.raises("unknown skill name", KeyError, lambda: load_skill("dodgeball", config), "available skills")
checks.raises(
    "load_skill without a throw sampler (validation probe needs one)",
    RuntimeError,
    lambda: load_skill("dodge", config),
    "needs a throw_sampler",
)
checks.raises(
    "throw sampler without declared values",
    ValueError,
    lambda: build(config, throw_sampler=sampler),
    "needs the current world's declared_payload_kg and declared_surface",
)
checks.raises(
    "declared values without a throw sampler",
    ValueError,
    lambda: build(config, **FIXTURE_DECLARED),
    "pass them together with a throw_sampler",
)
checks.raises(
    "declared surface that is not a level",
    ValueError,
    lambda: build(config, throw_sampler=sampler, declared_payload_kg=1.0, declared_surface="ice"),
    "declared_surface='ice' is not one of",
)
checks.raises(
    "declared payload outside its range",
    ValueError,
    lambda: build(config, throw_sampler=sampler, declared_payload_kg=9.0, declared_surface="fixture_surface_a"),
    "declared_payload_kg=9.0 is outside",
)

print("\n== sample_situations ==")
bare = build(config)
checks.raises("no throw sampler", RuntimeError, lambda: bare.sample_situations(stream_rng(1, "x"), 3), "needs a throw_sampler")
rows = spec.sample_situations(stream_rng(7, "fixture/dodge/situations"), 50)
checks.check("shape (50, 9), finite", rows.shape == (50, 9) and bool(np.all(np.isfinite(rows))))
checks.check("declared payload appended to every row", bool(np.all(rows[:, 7] == FIXTURE_DECLARED["declared_payload_kg"])))
checks.check("declared surface index appended to every row", bool(np.all(rows[:, 8] == 0.0)))
checks.check("the first 7 columns are the throw sampler's rows", np.array_equal(rows[:, :7], sampler(stream_rng(7, "fixture/dodge/situations"), 50)))
prefix = spec.sample_situations(stream_rng(7, "fixture/dodge/situations"), 10)
checks.check("situation i does not depend on how many are drawn", np.array_equal(prefix, rows[:10]))
checks.check("zero rows", spec.sample_situations(stream_rng(7, "x"), 0).shape == (0, 9))
outside = lambda rng, n: np.column_stack([np.full(n, 9.0), sampler(rng, n)[:, 1:]])  # noqa: E731
wrong_shape = lambda rng, n: sampler(rng, n)[:, :6]  # noqa: E731
with_nan = lambda rng, n: np.where(np.arange(7) == 2, np.nan, sampler(rng, n))  # noqa: E731
for label, bad, fragment in (
    ("throw outside the declared range", outside, "ball_rel_x_m outside its declared range"),
    ("throw sampler with 6 columns", wrong_shape, "must return shape (4, 7)"),
    ("throw sampler with NaN", with_nan, "non-finite"),
):
    bad_spec = build(config, throw_sampler=bad, **FIXTURE_DECLARED)
    checks.raises(label, ValueError, lambda bad_spec=bad_spec: bad_spec.sample_situations(stream_rng(3, "x"), 4), fragment)
checks.raises("negative n", ValueError, lambda: spec.sample_situations(stream_rng(3, "x"), -1), "n must be an int >= 0")
checks.check("registry lists exactly the built skills", available_skills() == ("dodge",), f"{available_skills()}")

checks.finish()
