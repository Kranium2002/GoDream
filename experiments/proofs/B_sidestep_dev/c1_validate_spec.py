"""Check 1: SidestepReachSpec built for the real Go2World passes validate_spec with the project's
validation settings and the command-binding check; the formula can reach every dev target.

DEV RUN on the real Go2 simulator: Person A's Go2World(config, seed, "default") is built and its
declared values are read from it (attribute if present, else the observation of world.reset()).
No episode is scored here. Not an evaluation result.

Validation settings are the ones load_skill uses (learned_judgment.skills._validation_settings):
world.hidden_effects names, engine.quantile_levels, runtime.seed, engine.spec_validation.*;
the command schema is CommandSchema.from_config(config).

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_sidestep_dev\\c1_validate_spec.py
"""

from __future__ import annotations

import time

import numpy as np

from _common import DEV_RUN_LABEL, WORLD_SEED_STREAM, Checks, world_config

print(DEV_RUN_LABEL)

from learned_judgment.config import config_digest, load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_seed  # noqa: E402
from learned_judgment.env import Go2World  # noqa: E402
from learned_judgment.eval.splits import split_situations  # noqa: E402
from learned_judgment.skills import _validation_settings  # noqa: E402
from learned_judgment.skills.base import (  # noqa: E402
    CommandSchema,
    SkillSpecError,
    check_command_binding,
    command_binding_problems,
    exposes_command_sequence,
    validate_spec,
)
from sidestep_reach_spec import ROOT, SidestepReachSpec, for_world  # noqa: E402

checks = Checks()
config = load_config()
schema = CommandSchema.from_config(config)
settings = _validation_settings(config)
base_seed = int(require(config, "runtime.seed"))
preset = str(require(config, f"{ROOT}.world_preset"))
world_seed = stream_seed(base_seed, WORLD_SEED_STREAM)
print(f"config digest {config_digest(config)[:12]} | runtime.seed {base_seed} | world seed {world_seed} "
      f"(stream '{WORLD_SEED_STREAM}') | preset {preset}")
print(f"validation settings: {settings}")

started = time.perf_counter()
world = Go2World(world_config(config), world_seed, preset)
has_attributes = hasattr(world, "declared_payload_kg") and hasattr(world, "declared_surface")
print(f"world built in {time.perf_counter() - started:.2f} s; declared values read from "
      f"{'world attributes' if has_attributes else 'the observation of world.reset() (attributes not present yet)'}")

try:
    spec = for_world(config, world)
    built = True
    detail = f"declared payload {spec._declared_row[0]} kg, surface level {spec.situation[3].levels[int(spec._declared_row[1])]!r}"
except (SkillSpecError, ValueError) as error:
    built, detail = False, str(error)
checks.check("for_world(config, world) builds the spec and validate_spec passes inside it", built, detail)
if not built:
    checks.finish()

checks.check("spec is a SidestepReachSpec named 'sidestep_reach'", isinstance(spec, SidestepReachSpec) and spec.name == "sidestep_reach", "")
checks.check(
    "the spec exposes its reset request and command sequence (reset_situation + commands, checked by validate_spec)",
    exposes_command_sequence(spec),
    "",
)
try:
    validate_spec(spec, schema, **settings)
    checks.check("validate_spec passes again, called directly with the project's settings", True, "")
except SkillSpecError as error:
    checks.check("validate_spec passes again, called directly with the project's settings", False, str(error))
problems = command_binding_problems(spec, schema)
try:
    check_command_binding(spec, schema)
    binding_ok = not problems
except SkillSpecError as error:
    binding_ok = False
    problems = [str(error)]
checks.check("command binding: every command dimension is a verified field inside its verified range", binding_ok, f"problems {problems}")
side = spec.actions[0]
field = schema.field(side.command_field)
checks.check(
    "the side-velocity dimension spans exactly the verified range of DODGE's side field",
    (side.command_field, side.low, side.high, side.unit) == (require(config, "skills.dodge.action.side_velocity_field"), field.low, field.high, field.unit),
    f"{side.command_field} [{side.low}, {side.high}] {side.unit}",
)

print("\n== the spec ==")
for variable in spec.situation:
    print(f"  situation {variable.name:24s} {variable.kind:11s} {variable.unit:5s} "
          f"{[variable.low, variable.high] if variable.kind == 'float' else list(variable.levels)}")
for dim in spec.actions:
    print(f"  action    {dim.name:24s} role {dim.role:7s} field {dim.command_field!s:8s} [{dim.low}, {dim.high}] {dim.unit}")
for outcome in spec.outcomes:
    print(f"  outcome   {outcome.name:24s} {outcome.kind:10s} {outcome.unit}")
print(f"  episode {spec.episode_duration_s} s = {spec.episode_periods} control periods of {spec.control_period_s} s; "
      f"candidate grid {spec.candidate_points_per_dim}")

print("\n== can the formula reach every dev target? (physics-only decisions, no simulator) ==")
n_dev = int(require(config, f"{ROOT}.dev_situations"))
dev = split_situations(spec, base_seed=base_seed, split="dev", n=n_dev, world_preset=preset)
layout = build_layout(spec)
decisions = decide_batch(
    spec,
    dev,
    model=None,
    layout=layout,
    points_per_dim=spec.candidate_points_per_dim,
    quantile_levels=require(config, "engine.quantile_levels"),
    spread_levels=tuple(require(config, "engine.spread_levels")),
    method=PHYSICS_ONLY,
)
errors = np.array([abs(d.predicted.physics[d.chosen_index, spec.outcome_index("sideways_error_m")]) for d in decisions])
travel = np.array([d.predicted.physics[:, 0] + dev[i, 1] for i, d in enumerate(decisions)])[0]
print(f"  formula travel over the {travel.size} candidates: {travel.min():.4f} to {travel.max():.4f} m")
print(f"  dev targets: {dev[:, 1].min():.4f} to {dev[:, 1].max():.4f} m; forward speeds {dev[:, 0].min():.3f} to {dev[:, 0].max():.3f} m/s")
print(f"  formula error of the physics-only choice: mean {errors.mean():.4f} m, largest {errors.max():.4f} m")
checks.check(
    "the formula reaches every dev target within success_tolerance_m (the targets are reachable by the formula)",
    float(errors.max()) <= spec.success_tolerance_m,
    f"largest {errors.max():.4f} m <= {spec.success_tolerance_m} m",
)

checks.finish()
