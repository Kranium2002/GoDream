"""Check 3: physics-only baseline and local residual baselines of sidestep_reach on the real simulator.

DEV RUN on the real Go2 simulator (Person A's Go2World, preset default, no ball). Not an
evaluation result: nothing here is tuned on these numbers, and they go into no report or table.
It is the first measurement of whether real attempts close the formula's gap.

    1. Physics only: decide_batch(model=None) picks the physics action for 40 context situations
       and 60 dev situations (learned_judgment.eval.splits.split_situations), run through
       engine.collect.collect.
    2. Context attempts: engine.collect.exploration_actions (engine.attempt_exploration_fraction)
       on the same 40 context situations, run through collect.
    3. Residual baselines of learned_judgment/eval/baselines.py named in
       dev_checks.sidestep_reach.residual_baselines, each fitted on the 40 context attempts with
       engine.learn.fit_residual_model; decide_batch picks an action for each dev situation; those
       actions are run on the simulator through collect.

Every method meets each dev situation with the same attempt seed (the dev noise seed of that
situation, repeat 0). Every attempt row is appended to out/c3_<UTC time>/attempts.jsonl as it is
measured; the metrics are computed from the rows read back from that file. Seeds, streams,
versions and wall times are in manifest.json beside it; the printed table is table.txt.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_sidestep_dev\\c3_baselines_real.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import OrderedDict

import numpy as np

from _common import DEV_RUN_LABEL, OUT_DIR, WORLD_SEED_STREAM, Checks, utc_now, utc_stamp, versions, world_config

print(DEV_RUN_LABEL, flush=True)

from learned_judgment.config import config_digest, load_config, require  # noqa: E402
from learned_judgment.engine.collect import collect, exploration_actions  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.learn import fit_residual_model  # noqa: E402
from learned_judgment.engine.rows import read_rows, rows_to_arrays  # noqa: E402
from learned_judgment.engine.streams import stream_rng, stream_seed  # noqa: E402
from learned_judgment.env import Go2World  # noqa: E402
from learned_judgment.eval.baselines import (  # noqa: E402
    ConstantBiasResidual,
    GaussianProcessResidual,
    GradientBoostingResidual,
    MLPAttemptsResidual,
)
from learned_judgment.eval.splits import check_disjoint, noise_seeds, split_situations  # noqa: E402
from learned_judgment.skills.base import CommandSchema  # noqa: E402
from sidestep_reach_spec import ROOT, for_world  # noqa: E402

BASELINES = {cls.method: cls for cls in (ConstantBiasResidual, GaussianProcessResidual, GradientBoostingResidual, MLPAttemptsResidual)}

checks = Checks()
started_utc = utc_now()
stamp = utc_stamp()
config = load_config()
schema = CommandSchema.from_config(config)
base_seed = int(require(config, "runtime.seed"))
preset = str(require(config, f"{ROOT}.world_preset"))
n_context = int(require(config, f"{ROOT}.context_attempts"))
n_dev = int(require(config, f"{ROOT}.dev_situations"))
baseline_names = list(require(config, f"{ROOT}.residual_baselines"))
unknown = [name for name in baseline_names if name not in BASELINES]
if unknown:
    raise ValueError(f"'{ROOT}.residual_baselines' names {unknown}; known: {sorted(BASELINES)}")
exploration_fraction = require(config, "engine.attempt_exploration_fraction")
quantile_levels = tuple(require(config, "engine.quantile_levels"))
spread_levels = tuple(require(config, "engine.spread_levels"))
world_seed = stream_seed(base_seed, WORLD_SEED_STREAM)

out_dir = OUT_DIR / f"c3_{stamp}"
out_dir.mkdir(parents=True, exist_ok=False)
attempts_path = out_dir / "attempts.jsonl"
print(f"config digest {config_digest(config)[:12]} | runtime.seed {base_seed} | world seed {world_seed} "
      f"(stream '{WORLD_SEED_STREAM}') | preset {preset}")
print(f"output directory {out_dir.relative_to(OUT_DIR.parents[3]).as_posix()}")

world = Go2World(world_config(config), world_seed, preset)
spec = for_world(config, world)
layout = build_layout(spec)
decide_settings = dict(
    layout=layout, points_per_dim=spec.candidate_points_per_dim, quantile_levels=quantile_levels, spread_levels=spread_levels
)

context = split_situations(spec, base_seed=base_seed, split="context", n=n_context, world_preset=preset)
dev = split_situations(spec, base_seed=base_seed, split="dev", n=n_dev, world_preset=preset)
check_disjoint({"context": context, "dev": dev})
context_seeds = noise_seeds(spec, base_seed=base_seed, split="context", world_preset=preset, n_situations=n_context, n_repeats=1)[:, 0]
dev_seeds = noise_seeds(spec, base_seed=base_seed, split="dev", world_preset=preset, n_situations=n_dev, n_repeats=1)[:, 0]
exploration_stream = f"{spec.name}/context/{preset}/exploration"
print(f"context situations {n_context} (stream '{spec.name}/context/{preset}/situations'), dev situations {n_dev} "
      f"(stream '{spec.name}/dev/{preset}/situations'); attempt seeds from '{spec.name}/<split>/{preset}/noise/0'")

runs: "OrderedDict[str, dict]" = OrderedDict()


def run(label: str, situations: np.ndarray, actions: np.ndarray, seeds: np.ndarray) -> list:
    """Run one episode per row through engine.collect and record the wall time."""
    run_id = f"{spec.name}/c3/{stamp}/{label}"
    began = time.perf_counter()
    rows = collect(
        spec,
        world,
        situations,
        actions,
        attempt_seeds=seeds,
        command_schema=schema,
        run_id=run_id,
        world_preset=preset,
        out_path=attempts_path,
        clock=utc_now,
    )
    wall = time.perf_counter() - began
    runs[label] = {"run_id": run_id, "episodes": len(rows), "wall_s": wall}
    print(f"  ran {label:28s} {len(rows):3d} episodes in {wall:6.1f} s ({wall / max(len(rows), 1):.2f} s per episode)", flush=True)
    return rows


def physics_actions(situations: np.ndarray) -> np.ndarray:
    decisions = decide_batch(spec, situations, model=None, method=PHYSICS_ONLY, **decide_settings)
    return np.stack([decision.action for decision in decisions])


print("\n== episodes on the simulator ==")
context_physics = physics_actions(context)
dev_physics = physics_actions(dev)
run("context/physics_only", context, context_physics, context_seeds)
run("dev/physics_only", dev, dev_physics, dev_seeds)
explore = exploration_actions(
    spec,
    context,
    rng=stream_rng(base_seed, exploration_stream),
    exploration_fraction=exploration_fraction,
    **{key: decide_settings[key] for key in ("points_per_dim", "quantile_levels", "spread_levels", "layout")},
)
context_rows = run("context/explore", context, explore, context_seeds)

fits: dict[str, dict] = {}
for name in baseline_names:
    seed_stream = f"{spec.name}/c3/baselines/{name}"
    model_seed = stream_seed(base_seed, seed_stream)
    model = BASELINES[name].from_config(config, layout, seed=model_seed)
    fit_residual_model(spec, model, context_rows, layout, on_physics_mismatch="error")
    decisions = decide_batch(spec, dev, model=model, method=model.method, **decide_settings)
    actions = np.stack([decision.action for decision in decisions])
    changed = int(np.count_nonzero(np.any(actions != dev_physics, axis=1)))
    fits[name] = {"seed": model_seed, "seed_stream": seed_stream, "decisions_unlike_physics": changed,
                  "fit_warnings": len(model.fit_warnings), "describe": model.describe()}
    print(f"  fitted {name} on {model.n_train} context attempts (seed {model_seed}); {changed} of {n_dev} dev decisions "
          f"differ from the physics action; {len(model.fit_warnings)} fit warnings", flush=True)
    run(f"dev/{name}", dev, actions, dev_seeds)

# ----- metrics from the saved rows ---------------------------------------------------------------
rows = read_rows(attempts_path, spec=spec)
checks.check(
    "every episode is in the attempt file, read back and validated against the spec",
    len(rows) == sum(record["episodes"] for record in runs.values()),
    f"{len(rows)} rows in {attempts_path.name}",
)
by_run = {label: [row for row in rows if row.run_id == record["run_id"]] for label, record in runs.items()}
checks.check(
    "each run's rows are in the file in input order with the run's attempt seeds",
    all([row.seed for row in by_run[label]] == list(map(int, context_seeds if label.startswith("context") else dev_seeds))
        for label in runs),
    "",
)
arrays = {label: rows_to_arrays(spec, run_rows, on_physics_mismatch="error") for label, run_rows in by_run.items()}
checks.check(
    "stored physics guesses equal the current spec's, and residual = measured - physics (rows_to_arrays, policy 'error')",
    all(np.array_equal(a.residuals, a.measured - a.physics) for a in arrays.values()),
    "",
)
same_action = np.all(arrays["context/explore"].actions == arrays["context/physics_only"].actions, axis=1)
same_outcome = np.all(arrays["context/explore"].measured == arrays["context/physics_only"].measured, axis=1)
checks.check(
    "context attempts that took the physics action reproduce the physics-only run's outcome bit for bit",
    bool(np.all(same_outcome[same_action])),
    f"{int(same_action.sum())} of {n_context} context attempts took the physics action",
)

error_column = spec.outcome_index("sideways_error_m")
fall_column = spec.outcome_index("fall")
table = ["Sideways error = measured sideways travel at the end of the episode - target, metres (+ is left).",
         "DEV RUN, real Go2World, preset default; not an evaluation result.",
         "",
         f"{'run':28s} {'n':>3s} {'mean |error|':>12s} {'mean error':>10s} {'median |e|':>10s} {'max |e|':>8s} "
         f"{'falls':>5s} {'success':>7s} {'s/episode':>9s}"]
summary: dict[str, dict] = {}
for label, a in arrays.items():
    error = a.measured[:, error_column]
    fall = a.measured[:, fall_column]
    success = spec.success(a.measured)
    record = runs[label]
    summary[label] = {
        "n": int(error.size),
        "mean_abs_error_m": float(np.mean(np.abs(error))),
        "mean_error_m": float(np.mean(error)),
        "median_abs_error_m": float(np.median(np.abs(error))),
        "max_abs_error_m": float(np.max(np.abs(error))),
        "fall_rate": float(np.mean(fall)),
        "success_rate": float(np.mean(success)),
        "wall_s_per_episode": record["wall_s"] / record["episodes"],
    }
    s = summary[label]
    table.append(
        f"{label:28s} {s['n']:3d} {s['mean_abs_error_m']:12.4f} {s['mean_error_m']:10.4f} {s['median_abs_error_m']:10.4f} "
        f"{s['max_abs_error_m']:8.4f} {s['fall_rate']:5.2f} {s['success_rate']:7.2f} {s['wall_s_per_episode']:9.2f}"
    )
table.append(f"(success: |error| <= {spec.success_tolerance_m} m and no fall; falls and success are rates)")
table.append("")
table.append("Paired on the 60 dev situations (same situation, same attempt seed): mean of |error(method)| - |error(physics only)|,")
table.append("negative = the method is closer to the target; and how many situations it is closer / farther / equal.")
reference = np.abs(arrays["dev/physics_only"].measured[:, error_column])
for name in baseline_names:
    delta = np.abs(arrays[f"dev/{name}"].measured[:, error_column]) - reference
    summary[f"dev/{name}"]["paired_mean_abs_error_change_m"] = float(delta.mean())
    table.append(f"  {name:20s} {delta.mean():+.4f} m   closer {int((delta < 0).sum()):2d}, farther {int((delta > 0).sum()):2d}, "
                 f"equal {int((delta == 0).sum()):2d}")
residual = arrays["context/explore"].residuals[:, error_column]
table.append("")
table.append(f"Context attempts (exploration, n={residual.size}): residual of sideways error (measured - formula) "
             f"mean {residual.mean():+.4f} m, std {residual.std(ddof=1):.4f} m, min {residual.min():+.4f}, max {residual.max():+.4f}")
for side_name, mask in (("left (side > 0)", arrays["context/explore"].actions[:, 0] > 0),
                        ("right (side < 0)", arrays["context/explore"].actions[:, 0] < 0),
                        ("zero command", arrays["context/explore"].actions[:, 0] == 0)):
    if mask.any():
        table.append(f"  {side_name:17s} n={int(mask.sum()):2d}: residual mean {residual[mask].mean():+.4f} m")
total_episodes = sum(record["episodes"] for record in runs.values())
total_wall = sum(record["wall_s"] for record in runs.values())
table.append("")
table.append(f"Simulator episodes {total_episodes}, wall time in collect {total_wall:.1f} s, "
             f"{total_wall / total_episodes:.2f} s per episode (reset, steps and finish, one process).")
print("\n" + "\n".join(table))
(out_dir / "table.txt").write_text("\n".join(table) + "\n", encoding="utf-8")

manifest = {
    "label": "DEV RUN on the real Go2 simulator; not an evaluation result",
    "command": f"{sys.executable} experiments/proofs/B_sidestep_dev/c3_baselines_real.py",
    "started_utc": started_utc,
    "finished_utc": utc_now(),
    "config_digest": config_digest(config),
    "versions": versions(),
    "runtime_seed": base_seed,
    "world_seed": world_seed,
    "world_seed_stream": WORLD_SEED_STREAM,
    "world_preset": preset,
    "declared_values": {key: spec.decode_situation(context[0])[key] for key in ("declared_payload_kg", "declared_surface")},
    "situation_streams": {split: f"{spec.name}/{split}/{preset}/situations" for split in ("context", "dev")},
    "attempt_seed_streams": {split: f"{spec.name}/{split}/{preset}/noise/0, index = situation index" for split in ("context", "dev")},
    "context_attempt_seeds": [int(seed) for seed in context_seeds],
    "dev_attempt_seeds": [int(seed) for seed in dev_seeds],
    "exploration_stream": exploration_stream,
    "exploration_fraction": exploration_fraction,
    "baselines": fits,
    "runs": runs,
    "summary": summary,
    "attempts_file": "attempts.jsonl",
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
print(f"\nsaved {attempts_path.name}, table.txt and manifest.json in {out_dir.relative_to(OUT_DIR.parents[3]).as_posix()}")

checks.finish()
