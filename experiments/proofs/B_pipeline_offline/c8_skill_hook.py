"""c8 PLUMBING CHECK: skills.load_skill_for_world and pipeline.open_run through the DODGE module's for_world.

Since bcf4ad3 (Person A's ball, throw model and DODGE settings merged into person-b) the DODGE hook
runs on the real project pieces, so this check uses no stand-in: the real config.yaml values of
skills.dodge, Person A's throw model learned_judgment.env.sample_dodge_situations, and the declared
values of the default preset. Only the output directories are redirected into a temporary directory
(fixture_pipeline.make_config). The earlier version injected a fixture throw model and aborted once
the real one existed; it was rewritten by the integrator on 2026-11-21.

1. load_skill_for_world("dodge", config, world) calls the module's for_world, which reads the
   world's declared values; the result is a validated spec named "dodge" whose sampled situations
   carry those declared values. A hook that returns something else is refused (TypeError); an
   unknown skill raises KeyError.
2. open_run(config, "dodge", ...) builds ONE real Go2World (go2_world_factory) to read its declared
   values, builds the spec through the hook and opens a SkillRun whose manifest names the Go2World
   factory. No episode is run, so nothing here is a simulator result.

PLUMBING CHECK: mechanics of the hook and of opening a run only; nothing here is evidence about the
robot, the simulator, TabPFN or any model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c8_skill_hook.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import Checks, FixturePipelineSpec, make_config  # noqa: E402

print("PLUMBING CHECK - the DODGE for_world hook and open_run on the real config and throw model; one real Go2World is "
      "built and no episode is run; not evidence about the robot, the simulator, TabPFN or any model")

import numpy as np  # noqa: E402

import learned_judgment.env as env  # noqa: E402
import learned_judgment.skills.dodge as dodge_module  # noqa: E402
from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.skills import has_for_world, load_skill_for_world  # noqa: E402

checks = Checks()
expect = checks.expect
SKILL = dodge_module.SPEC_CLASS.skill_name
PRESET = "default"

expect(callable(getattr(env, "sample_dodge_situations", None)), "Person A's throw model learned_judgment.env.sample_dodge_situations exists")

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"
    config = make_config(tmp)
    declared = config["world"]["presets"][PRESET]

    class DeclaringWorld:
        """Stand-in that only declares the default preset's values (no reset, no step): what for_world reads."""

        declared_payload_kg = declared["declared_payload_kg"]
        declared_surface = declared["declared_surface"]

    print("\n== 1. load_skill_for_world through the module's for_world ==")
    expect(has_for_world(SKILL), "the DODGE module defines for_world")
    spec = load_skill_for_world(SKILL, config, DeclaringWorld())
    expect(type(spec) is dodge_module.SPEC_CLASS and spec.name == SKILL, f"a {type(spec).__name__} named {spec.name!r}")
    sampled = spec.sample_situations(np.random.default_rng(3), 4)
    decoded = [spec.decode_situation(row) for row in sampled]
    expect(all(row["declared_payload_kg"] == declared["declared_payload_kg"]
               and row["declared_surface"] == declared["declared_surface"] for row in decoded),
           f"sampled situations carry the world's declared values ({declared['declared_payload_kg']}, {declared['declared_surface']!r})")
    original = dodge_module.for_world
    try:
        dodge_module.for_world = lambda config, world: {"not": "a spec"}
        checks.raises("a hook returning a mapping", TypeError, lambda: load_skill_for_world(SKILL, config, DeclaringWorld()),
                      "must return a SkillSpec")
        dodge_module.for_world = lambda config, world: FixturePipelineSpec()
        checks.raises("a hook returning another skill's spec", TypeError,
                      lambda: load_skill_for_world(SKILL, config, DeclaringWorld()), "built a spec named")
    finally:
        dodge_module.for_world = original
    checks.raises("an unknown skill", KeyError, lambda: load_skill_for_world("no_such_skill", config, DeclaringWorld()), "unknown skill")

    print("\n== 2. open_run on one real Go2World (no episode) ==")
    run = P.open_run(config, SKILL, run_id="hook", world_preset=PRESET, base_seed=20261121, clock=P.utc_now)
    manifest = P.read_run_manifest(run.paths.manifest)
    expect(isinstance(run, P.SkillRun) and run.spec.name == SKILL, "open_run returned a SkillRun of the registered skill")
    expect("Go2World" in manifest["world"], f"manifest names the world factory: {manifest['world']}")
    expect(all(manifest["steps"][step]["state"] == "not_run" for step in P.STEPS), "no step was run")
    expect(str(run.paths.manifest).startswith(tmp_name), "the manifest is under the temporary directory")
    expect(run.paths.attempts.parent == tmp / "attempts" and not run.paths.attempts.exists(), "no attempt file was written")

checks.finish()
