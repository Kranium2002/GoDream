"""c2 PLUMBING CHECK: incremental collection equals one-shot collection; the observer changes nothing.

1. Run A collects 10 attempts in one call. Run B (same run_id, base seed and preset, its own
   temporary directory) collects 5, then 10. The two attempt files must be byte-identical (fixed
   clock), and B's second call must run attempts 5..9 only. Further calls with 10 or 7 run nothing.
2. Run C does NOT use the pipeline: it builds the same plan from the documented streams
   (eval.splits context situations and attempt seeds, engine.collect.exploration_actions on
   "<skill>/context/<preset>/exploration") and calls engine.collect.collect on the bare fixture
   world, without the observer. Its attempt file must be byte-identical to A's.
3. ObservedWorld forwards reset / step / finish_episode and other attributes unchanged, counts
   finished episodes, and raises Stopped from reset (before the wrapped reset runs) when
   should_stop() is true.

PLUMBING CHECK on a fixture world (fixture_pipeline.py): mechanics only; outputs in a temporary
directory; nothing here is evidence about the robot, the simulator, TabPFN or any model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c2_incremental_observer.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import (  # noqa: E402
    BANNER,
    BASE_SEED,
    PRESET,
    SKILL,
    Checks,
    FixturePipelineSpec,
    FixturePlumbingWorld,
    fixed_clock,
    make_config,
    open_fixture_run,
)

print(BANNER)

import numpy as np  # noqa: E402

from learned_judgment import pipeline as P  # noqa: E402
from learned_judgment.config import require  # noqa: E402
from learned_judgment.engine.collect import collect, exploration_actions  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng, stream_seed  # noqa: E402
from learned_judgment.eval.splits import noise_seed, split_situations  # noqa: E402
from learned_judgment.skills.base import CommandSchema  # noqa: E402

checks = Checks()
expect = checks.expect
RUN_ID = "incremental"

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"

    print("\n== 1. one call with 10 vs. 5 then 10 ==")
    run_a = open_fixture_run(make_config(tmp / "A"), run_id=RUN_ID, clock=fixed_clock)
    rows_a = run_a.collect_attempts(10)
    events_b: list[dict] = []
    run_b = open_fixture_run(make_config(tmp / "B"), run_id=RUN_ID, clock=fixed_clock, progress=events_b.append)
    rows_b5 = run_b.collect_attempts(5)
    expect(len(rows_b5) == 5, "B: 5 rows after the first call")
    events_b.clear()
    rows_b = run_b.collect_attempts(10)
    progress = [event["done"] for event in events_b if event["event"] == "progress"]
    expect(progress == [6, 7, 8, 9, 10], f"B: the second call ran attempts 5..9 only (progress done {progress})")
    bytes_a = run_a.paths.attempts.read_bytes()
    bytes_b = run_b.paths.attempts.read_bytes()
    expect(bytes_a == bytes_b, f"attempt files byte-identical ({len(bytes_a)} bytes, 10 rows each)")
    expect([r.seed for r in rows_a] == [r.seed for r in rows_b], "same attempt seeds in the same order")
    history = [entry for entry in run_b.manifest["history"] if entry["step"] == "collect"]
    expect([entry["counts"]["new_rows"] for entry in history] == [5, 5], "B manifest history: 5 new rows, then 5 new rows")
    run_b.collect_attempts(10)
    run_b.collect_attempts(7)
    expect(run_b.paths.attempts.read_bytes() == bytes_b, "calls with 10 and 7 afterwards run nothing and change nothing")
    expect([entry["counts"]["new_rows"] for entry in run_b.manifest["history"] if entry["step"] == "collect"][-2:] == [0, 0],
           "those calls are recorded with 0 new rows")

    print("\n== 2. the pipeline's rows equal engine.collect.collect on the bare world ==")
    config_c = make_config(tmp / "C")
    spec = FixturePipelineSpec()
    situations = split_situations(spec, base_seed=BASE_SEED, split="context", n=10, world_preset=PRESET)
    actions = exploration_actions(
        spec,
        situations,
        rng=stream_rng(BASE_SEED, f"{SKILL}/context/{PRESET}/exploration"),
        points_per_dim=require(config_c, f"skills.{SKILL}.candidate_points_per_dim"),
        exploration_fraction=require(config_c, "engine.attempt_exploration_fraction"),
        quantile_levels=require(config_c, "engine.quantile_levels"),
        spread_levels=tuple(require(config_c, "engine.spread_levels")),
        layout=build_layout(spec),
    )
    seeds = np.array([noise_seed(spec, base_seed=BASE_SEED, split="context", world_preset=PRESET, situation_index=i, repeat=0)
                      for i in range(10)], dtype=np.int64)
    world_seed = stream_seed(BASE_SEED, f"pipeline/world/{PRESET}")
    expect(world_seed == run_a.world_seed, f"world seed is stream_seed(base, 'pipeline/world/{PRESET}') = {world_seed}")
    bare_world = FixturePlumbingWorld(world_seed, PRESET)
    path_c = tmp / "C" / "attempts" / f"{RUN_ID}.jsonl"
    collect(spec, bare_world, situations, actions, attempt_seeds=seeds, command_schema=CommandSchema.from_config(config_c),
            run_id=RUN_ID, world_preset=PRESET, out_path=path_c, clock=fixed_clock)
    expect(path_c.read_bytes() == bytes_a, "rows collected without the observer are byte-identical to the pipeline's rows")
    expect(int(np.sum(actions[:, 0] != 0.0)) > 0, "the plan is not trivial (some non-zero side speeds)")

    print("\n== 3. ObservedWorld ==")
    inner = FixturePlumbingWorld(world_seed, PRESET)
    twin = FixturePlumbingWorld(world_seed, PRESET)
    finished: list[int] = []
    observed = P.ObservedWorld(inner, on_episode=finished.append)
    situation = spec.decode_situation(situations[0])
    command = {"vel_x": 0.0, "vel_y": 0.3, "yaw_rate": 0.0}
    expect(observed.reset(situation, 7) == twin.reset(situation, 7), "reset returns the wrapped world's value")
    expect(observed.step(command, 0.5) == twin.step(command, 0.5), "step returns the wrapped world's value")
    expect(observed.finish_episode() == twin.finish_episode(), "finish_episode returns the wrapped world's value")
    expect(observed.finished == 1 and finished == [1], "one finished episode counted and reported")
    expect(observed.hidden_gain == inner.hidden_gain and observed.wrapped is inner, "other attributes are read from the wrapped world")
    stopping = P.ObservedWorld(inner, should_stop=lambda: True)
    started = inner.episodes_started
    checks.raises("should_stop() true -> Stopped from reset", P.Stopped, lambda: stopping.reset(situation, 8), "should_stop")
    expect(inner.episodes_started == started, "the wrapped reset was not called")

checks.finish()
