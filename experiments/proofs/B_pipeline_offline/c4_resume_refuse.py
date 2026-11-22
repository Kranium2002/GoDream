"""c4 PLUMBING CHECK: a second SkillRun on the same run_id continues from the manifest, and a
changed skill, preset, base seed, world or config is refused.

1. Run 1 collects 4 attempts and fits constant_bias. A second SkillRun object on the same run_id
   reads the manifest (states, history) and continues: collect_attempts(6) runs attempts 4 and 5.
2. Opening the same run_id with another skill name, another world preset, another base seed, a
   config with one changed value, or another world factory raises RunMismatchError naming the
   field, and the manifest file is left byte-for-byte unchanged.

PLUMBING CHECK on a fixture world (fixture_pipeline.py): mechanics only; outputs in a temporary
directory; nothing here is evidence about the robot, the simulator, TabPFN or any model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c4_resume_refuse.py
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import (  # noqa: E402
    BANNER,
    BASE_SEED,
    POINTS_PER_DIM,
    Checks,
    FixturePipelineSpec,
    FixturePlumbingWorld,
    fixed_clock,
    make_config,
    open_fixture_run,
)

print(BANNER)

from learned_judgment import pipeline as P  # noqa: E402

checks = Checks()
expect = checks.expect
OTHER_SKILL = "fixture_pipeline_other"

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"
    config = make_config(tmp)
    config["skills"][OTHER_SKILL] = {"candidate_points_per_dim": list(POINTS_PER_DIM)}  # set before the run starts

    print("\n== 1. continue from the manifest ==")
    first = open_fixture_run(config, run_id="resume", clock=fixed_clock)
    first.collect_attempts(4)
    first.residual_model("constant_bias")
    second = open_fixture_run(config, run_id="resume", clock=fixed_clock)
    expect(second.step_state("collect") == "done" and second.step_state("teacher") == "done", "states read from the manifest")
    expect([entry["step"] for entry in second.manifest["history"]] == ["collect", "teacher"], "history read from the manifest")
    events: list[dict] = []
    third = open_fixture_run(config, run_id="resume", clock=fixed_clock, progress=events.append)
    rows = third.collect_attempts(6)
    expect(len(rows) == 6, "collect_attempts(6) on the reopened run gives 6 rows")
    expect([event["done"] for event in events if event["event"] == "progress"] == [5, 6], "only attempts 4 and 5 ran")
    expect([entry["counts"]["new_rows"] for entry in third.manifest["history"] if entry["step"] == "collect"] == [4, 2],
           "history: 4 new rows, then 2")

    print("\n== 2. a changed identity is refused ==")
    before = third.paths.manifest.read_bytes()
    checks.raises("another skill", P.RunMismatchError,
                  lambda: open_fixture_run(config, run_id="resume", spec=FixturePipelineSpec(name=OTHER_SKILL), clock=fixed_clock), "skill")
    checks.raises("another world preset", P.RunMismatchError,
                  lambda: open_fixture_run(config, run_id="resume", world_preset="heavy_payload", clock=fixed_clock), "world_preset")
    checks.raises("another base seed", P.RunMismatchError,
                  lambda: open_fixture_run(config, run_id="resume", base_seed=BASE_SEED + 1, clock=fixed_clock), "base_seed")
    changed = copy.deepcopy(config)
    changed["engine"]["attempt_exploration_fraction"] = 0.25
    checks.raises("config with one changed value", P.RunMismatchError,
                  lambda: open_fixture_run(changed, run_id="resume", clock=fixed_clock), "config_digest")

    def other_factory(world_seed: int, world_preset: str) -> FixturePlumbingWorld:
        return FixturePlumbingWorld(world_seed, world_preset)

    checks.raises("another world factory", P.RunMismatchError,
                  lambda: P.SkillRun(config, FixturePipelineSpec(), world_factory=other_factory, run_id="resume",
                                     world_preset="default", base_seed=BASE_SEED, clock=fixed_clock), "world")
    expect(third.paths.manifest.read_bytes() == before, "the manifest file is unchanged after every refusal")
    reopened = open_fixture_run(config, run_id="resume", clock=fixed_clock)
    expect(len(reopened.attempts()) == 6, "the original identity still opens the run")
    checks.raises("an invalid run id", ValueError, lambda: open_fixture_run(config, run_id="bad.id", clock=fixed_clock), "run_id")
    checks.raises("an unknown preset", ValueError,
                  lambda: open_fixture_run(config, run_id="fresh", world_preset="no_such_preset", clock=fixed_clock), "world.presets")

checks.finish()
