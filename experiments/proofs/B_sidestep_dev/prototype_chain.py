"""Prototype: the whole learning chain on the REAL Go2 simulator with the sidestep_reach dev skill (no ball).

DEV RUN. One command runs, in order, on Person A's Go2World:

  1. attempts   40 real episodes with exploration actions (engine.collect);
  2. TabPFN     the residual teacher is fitted on those attempts (real, billed requests through
                the project's teacher: usage and quote are read before every request);
  3. dream      the teacher decides many sampled situations (engine.dream);
  4. SNN        a spiking network is trained inside that dream (students.snn);
  5. measure    physics only, the teacher and the SNN each decide the same dev situations, and
                every decision is run on the simulator with the same attempt seeds;
  6. watch      with --watch, the live MuJoCo window (app.view_episode) plays one dev situation
                twice: the physics-only decision, then the SNN's decision.

sidestep_reach is a development skill: "reach a given sideways displacement". It isolates the
dog-motion half of the DODGE physics guess, so the chain can run before the simulated ball exists.
Every count and setting comes from config.yaml (dev_checks.sidestep_reach, engine, tabpfn,
students.snn). The numbers are development measurements on one world and one seed: they are not
evaluation results and enter no report or table.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_sidestep_dev\\prototype_chain.py [--watch]

To open the live window again for a finished run, without running anything else:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_sidestep_dev\\prototype_chain.py --replay <output directory>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _common import DEV_RUN_LABEL, OUT_DIR, ROOT_DIR, WORLD_SEED_STREAM, utc_now, utc_stamp, versions, world_config

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--watch", action="store_true", help="afterwards play one dev situation in the live MuJoCo window")
parser.add_argument("--replay", metavar="DIRECTORY", help="only open the live window for a finished run's output directory")
arguments = parser.parse_args()


def play(watch: dict) -> None:
    """Open the live MuJoCo window once per recorded decision (the viewer runs the commands on Go2World)."""
    for name, record in watch.items():
        print(f"\nopening the MuJoCo window: {name}, action {record['action']}, measured sideways error "
              f"{record['measured_sideways_error_m']:+.4f} m in the recorded run (close the window or wait for the episode to end)",
              flush=True)
        subprocess.run([sys.executable, *record["viewer_command"][1:]], cwd=ROOT_DIR, check=False)


if arguments.replay:
    replayed = json.loads((Path(arguments.replay) / "manifest.json").read_text(encoding="utf-8"))
    print(replayed["label"])
    print(f"replaying dev situation {replayed['watch_index']} of the run started {replayed['started_utc']}")
    play(replayed["watch"])
    sys.exit(0)

print(DEV_RUN_LABEL, flush=True)

from learned_judgment.config import config_digest, load_config, require  # noqa: E402
from learned_judgment.engine.collect import collect, exploration_actions  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.dream import dream, keep_confident, save_dreams  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.learn import fit_residual_model  # noqa: E402
from learned_judgment.engine.rows import read_rows, rows_to_arrays  # noqa: E402
from learned_judgment.engine.streams import stream_rng, stream_seed  # noqa: E402
from learned_judgment.env import Go2World  # noqa: E402
from learned_judgment.eval.splits import check_disjoint, noise_seeds, split_situations  # noqa: E402
from learned_judgment.skills.base import CommandSchema  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings, TrainingError  # noqa: E402
from learned_judgment.teacher import TabPFNProvider, TabPFNTeacher, TeacherSettings  # noqa: E402
from sidestep_reach_spec import ROOT, for_world  # noqa: E402

TEACHER = "teacher"  # method name of the residual teacher in evaluation.methods
STUDENT = "snn_per_skill"  # method name of the per-skill spiking student in evaluation.methods
VIEWER_FIELDS = ("vel_x", "vel_y", "yaw_rate")  # the order app.view_episode's --command takes

started_utc = utc_now()
stamp = utc_stamp()
config = load_config()
for name in (TEACHER, STUDENT, PHYSICS_ONLY):
    if name not in require(config, "evaluation.methods"):
        raise ValueError(f"method {name!r} is not listed in config.yaml evaluation.methods")
schema = CommandSchema.from_config(config)
base_seed = int(require(config, "runtime.seed"))
preset = str(require(config, f"{ROOT}.world_preset"))
n_context = int(require(config, f"{ROOT}.context_attempts"))
n_dev = int(require(config, f"{ROOT}.dev_situations"))
n_dreams = int(require(config, "engine.dreams_per_skill_initial"))
view_tail_s = float(require(config, f"{ROOT}.view_tail_s"))
exploration_fraction = require(config, "engine.attempt_exploration_fraction")
quantile_levels = tuple(require(config, "engine.quantile_levels"))
spread_levels = tuple(require(config, "engine.spread_levels"))
thresholds = require(config, "engine").get("confidence_thresholds_by_skill")
world_seed = stream_seed(base_seed, WORLD_SEED_STREAM)

out_dir = OUT_DIR / f"prototype_{stamp}"
out_dir.mkdir(parents=True, exist_ok=False)
attempts_path = out_dir / "attempts.jsonl"
print(f"config digest {config_digest(config)[:12]} | runtime.seed {base_seed} | world seed {world_seed} "
      f"(stream '{WORLD_SEED_STREAM}') | preset {preset}")
print(f"output directory {out_dir.relative_to(ROOT_DIR).as_posix()}", flush=True)

world = Go2World(world_config(config), world_seed, preset)
spec = for_world(config, world)
layout = build_layout(spec)
points = spec.candidate_points_per_dim
decide_settings = dict(layout=layout, points_per_dim=points, quantile_levels=quantile_levels, spread_levels=spread_levels)

context = split_situations(spec, base_seed=base_seed, split="context", n=n_context, world_preset=preset)
dev = split_situations(spec, base_seed=base_seed, split="dev", n=n_dev, world_preset=preset)
check_disjoint({"context": context, "dev": dev})
context_seeds = noise_seeds(spec, base_seed=base_seed, split="context", world_preset=preset, n_situations=n_context, n_repeats=1)[:, 0]
dev_seeds = noise_seeds(spec, base_seed=base_seed, split="dev", world_preset=preset, n_situations=n_dev, n_repeats=1)[:, 0]

runs: dict[str, dict] = {}
timings: dict[str, float] = {}


def run(label: str, situations: np.ndarray, actions: np.ndarray, seeds: np.ndarray) -> list:
    """One real episode per row through engine.collect; rows go to the attempt file."""
    run_id = f"{spec.name}/prototype/{stamp}/{label}"
    began = time.perf_counter()
    rows = collect(spec, world, situations, actions, attempt_seeds=seeds, command_schema=schema, run_id=run_id,
                   world_preset=preset, out_path=attempts_path, clock=utc_now)
    wall = time.perf_counter() - began
    runs[label] = {"run_id": run_id, "episodes": len(rows), "wall_s": wall}
    print(f"  ran {label:22s} {len(rows):3d} episodes in {wall:6.1f} s ({wall / max(len(rows), 1):.2f} s per episode)", flush=True)
    return rows


def chosen_actions(decisions: list) -> np.ndarray:
    return np.stack([decision.action for decision in decisions])


# ----- 1. attempts --------------------------------------------------------------------------------
print(f"\n== 1. attempts: {n_context} real episodes with exploration actions ==", flush=True)
exploration_stream = f"{spec.name}/context/{preset}/exploration"
explore = exploration_actions(
    spec, context, rng=stream_rng(base_seed, exploration_stream), exploration_fraction=exploration_fraction,
    points_per_dim=points, quantile_levels=quantile_levels, spread_levels=spread_levels, layout=layout,
)
context_rows = run("context/explore", context, explore, context_seeds)
context_arrays = rows_to_arrays(spec, context_rows, on_physics_mismatch="error")
error_column, fall_column = spec.outcome_index("sideways_error_m"), spec.outcome_index("fall")
residual = context_arrays.residuals[:, error_column]
print(f"  residual of the sideways error (measured - formula): mean {residual.mean():+.4f} m, std {residual.std():.4f} m")

# ----- 2. TabPFN teacher --------------------------------------------------------------------------
print("\n== 2. TabPFN: the residual teacher on those attempts (real requests) ==", flush=True)
settings = TeacherSettings.from_config(config)
provider = TabPFNProvider(settings)
provider.connect()
usage_before = provider.usage()
print(f"  account usage before: {usage_before.get('current_usage')} of {usage_before.get('usage_limit')} tokens; "
      f"at most {settings.max_tokens_per_request} tokens per request; project budget {settings.project_token_budget}")
teacher = TabPFNTeacher.from_settings(layout, settings, provider=provider, clock=lambda: datetime.now(timezone.utc))
log_before = len(teacher.call_log.read()) if settings.call_log.exists() else 0
fit_residual_model(spec, teacher, context_rows, layout, on_physics_mismatch="error")

began = time.perf_counter()
physics_dev = chosen_actions(decide_batch(spec, dev, model=None, method=PHYSICS_ONLY, **decide_settings))
teacher_decisions = decide_batch(spec, dev, model=teacher, method=TEACHER, **decide_settings)
teacher_dev = chosen_actions(teacher_decisions)
timings["teacher_dev_decisions_s"] = time.perf_counter() - began
changed = int(np.count_nonzero(np.any(teacher_dev != physics_dev, axis=1)))
print(f"  the teacher decided {n_dev} dev situations x {teacher_decisions[0].candidates.shape[0]} candidates in "
      f"{timings['teacher_dev_decisions_s']:.1f} s; {changed} of {n_dev} decisions differ from the physics action", flush=True)

# ----- 3. dream -----------------------------------------------------------------------------------
print(f"\n== 3. dream: the teacher decides {n_dreams} sampled situations ==", flush=True)
began = time.perf_counter()
dreams = dream(spec, teacher, layout, n_dreams=n_dreams, rng=stream_rng(base_seed, f"{spec.name}/dreams/situations"),
               points_per_dim=points, quantile_levels=quantile_levels, spread_levels=spread_levels, method=TEACHER)
timings["dream_s"] = time.perf_counter() - began
threshold = thresholds.get(spec.name) if isinstance(thresholds, dict) else None
if threshold is None:
    kept = dreams
    print(f"  {dreams.n_rows} dreams in {timings['dream_s']:.1f} s; no confidence threshold is set for this skill in "
          f"engine.confidence_thresholds_by_skill, so every dream is kept")
else:
    kept = keep_confident(dreams, float(threshold))
    print(f"  {dreams.n_rows} dreams in {timings['dream_s']:.1f} s; {kept.n_rows} kept at chosen spread <= {threshold}")
differ = int(np.count_nonzero(kept.chosen != kept.physics_choice))
print(f"  in {differ} of {kept.n_rows} kept dreams the teacher's choice differs from the physics choice", flush=True)
save_dreams(out_dir / "dreams.npz", kept, {"run": f"{spec.name}/prototype/{stamp}", "label": "DEV RUN", "base_seed": base_seed,
                                          "n_context": n_context, "confidence_threshold": threshold})
log_records = teacher.call_log.read()[log_before:]
billed = [record for record in log_records if record.get("billable_dispatch") is True]
cache_hits = [record for record in log_records if record.get("cache_hit") is True]
tokens_quoted = sum(int(record.get("quote_tokens", 0)) for record in billed)
usage_after = provider.usage()
print(f"  TabPFN in this run: {len(billed)} billed requests, {tokens_quoted} tokens quoted, {len(cache_hits)} cache hits; "
      f"account usage now {usage_after.get('current_usage')} of {usage_after.get('usage_limit')}", flush=True)

# ----- 4. SNN trained in the dream ----------------------------------------------------------------
print(f"\n== 4. SNN: a spiking network trained inside the {kept.n_rows} dreams ==", flush=True)
snn_settings = SNNSettings.from_config(config)
began = time.perf_counter()
student = None
training_error = None
try:
    student = SNNStudent.train_in_dream(spec, kept, settings=snn_settings, points_per_dim=points, base_seed=base_seed)
except TrainingError as error:
    training_error = f"{type(error).__name__}: {error}"
timings["train_s"] = time.perf_counter() - began
if student is None:
    print(f"  TRAINING REFUSED after {timings['train_s']:.1f} s: {training_error}")
else:
    digest = student.save(out_dir / "student")
    training = dict(student.meta.get("training", {}))
    shown = {key: training[key] for key in ("steps_run", "best_step", "stopped_early", "initial_validation_loss",
                                            "best_validation_loss", "firing_rate_per_layer") if key in training}
    print(f"  trained in {timings['train_s']:.1f} s; network digest {digest[:12]}; hidden sizes {list(snn_settings.hidden_sizes)}, "
          f"{snn_settings.time_steps} time steps; training report: {shown}", flush=True)

# ----- 5. measure on the simulator ----------------------------------------------------------------
print(f"\n== 5. measure: every method decides the same {n_dev} dev situations; each decision is run on the simulator ==", flush=True)
methods: dict[str, np.ndarray] = {PHYSICS_ONLY: physics_dev, TEACHER: teacher_dev}
if student is not None:
    began = time.perf_counter()
    methods[STUDENT] = student.decide(spec, dev)
    timings["snn_dev_decisions_s"] = time.perf_counter() - began
    same = int(np.count_nonzero(np.all(methods[STUDENT] == teacher_dev, axis=1)))
    print(f"  the SNN decided {n_dev} dev situations in {timings['snn_dev_decisions_s'] * 1000:.1f} ms; "
          f"{same} of {n_dev} decisions equal the teacher's", flush=True)
for name, actions in methods.items():
    run(f"dev/{name}", dev, actions, dev_seeds)

rows = read_rows(attempts_path, spec=spec)
measured = {
    name: rows_to_arrays(spec, [row for row in rows if row.run_id == runs[f"dev/{name}"]["run_id"]], on_physics_mismatch="error").measured
    for name in methods
}
table = ["Sideways error = measured sideways travel at the end of the episode - target, metres (+ is left).",
         f"DEV RUN, real Go2World, preset {preset}, world seed {world_seed}, {n_context} context attempts; not an evaluation result.",
         "",
         f"{'method':16s} {'n':>3s} {'mean |error|':>12s} {'median |e|':>10s} {'max |e|':>8s} {'falls':>5s} {'success':>7s}"]
summary: dict[str, dict] = {}
for name, values in measured.items():
    error, fall, success = values[:, error_column], values[:, fall_column], spec.success(values)
    summary[name] = {"n": int(error.size), "mean_abs_error_m": float(np.mean(np.abs(error))), "mean_error_m": float(np.mean(error)),
                     "median_abs_error_m": float(np.median(np.abs(error))), "max_abs_error_m": float(np.max(np.abs(error))),
                     "fall_rate": float(np.mean(fall)), "success_rate": float(np.mean(success))}
    s = summary[name]
    table.append(f"{name:16s} {s['n']:3d} {s['mean_abs_error_m']:12.4f} {s['median_abs_error_m']:10.4f} {s['max_abs_error_m']:8.4f} "
                 f"{s['fall_rate']:5.2f} {s['success_rate']:7.2f}")
table.append("(success: |error| <= dev_checks.sidestep_reach.success_tolerance_m and no fall; falls and success are rates)")
table.append("")
table.append("Paired on the dev situations (same situation, same attempt seed): mean of |error(method)| - |error(physics only)|;")
table.append("negative = the method is closer to the target; and how many situations it is closer / farther / equal.")
reference = np.abs(measured[PHYSICS_ONLY][:, error_column])
for name, values in measured.items():
    if name == PHYSICS_ONLY:
        continue
    difference = np.abs(values[:, error_column]) - reference
    summary[name]["paired_mean_abs_error_minus_physics_m"] = float(difference.mean())
    table.append(f"  {name:16s} {difference.mean():+.4f} m   closer {int((difference < 0).sum()):2d}, "
                 f"farther {int((difference > 0).sum()):2d}, equal {int((difference == 0).sum()):2d}")
print("\n" + "\n".join(table), flush=True)
(out_dir / "table.txt").write_text("\n".join(table) + "\n", encoding="utf-8")

# ----- 6. one situation to watch ------------------------------------------------------------------
shown_method = STUDENT if student is not None else TEACHER
watch_index = int(np.argmax(reference))  # the dev situation where the physics-only decision missed by the most
situation = spec.decode_situation(dev[watch_index])
reset_request = spec.reset_situation(situation)
situation_path = out_dir / "watch_situation.json"
situation_path.write_text(json.dumps(reset_request) + "\n", encoding="utf-8")
watch: dict[str, dict] = {}
for name in (PHYSICS_ONLY, shown_method):
    action = spec.decode_action(methods[name][watch_index])
    sequence = spec.commands(situation, action)
    for command, _ in sequence:
        if set(command) != set(VIEWER_FIELDS):
            raise ValueError(f"the viewer takes exactly the fields {VIEWER_FIELDS}; the spec sent {sorted(command)}")
    nominal = dict(sequence[-1][0])
    viewing = list(sequence) + ([(nominal, view_tail_s)] if view_tail_s > 0 else [])
    command_arguments = [
        argument
        for command, duration in viewing
        for argument in ("--command", ",".join(repr(float(value)) for value in (*(command[field] for field in VIEWER_FIELDS), duration)))
    ]
    viewer = [sys.executable, "-m", "app.view_episode", "--seed", str(world_seed), "--preset", preset,
              "--situation", str(situation_path), *command_arguments]
    watch[name] = {"action": action, "commands": [[command, duration] for command, duration in sequence],
                   "measured_sideways_error_m": float(measured[name][watch_index, error_column]),
                   "viewer_command": viewer}
print(f"\n== 6. watch: dev situation {watch_index}, where the physics-only decision missed by the most ==")
print(f"  situation: {situation}")
for name, record in watch.items():
    print(f"  {name:16s} action {record['action']} -> measured sideways error {record['measured_sideways_error_m']:+.4f} m")
    print("    " + subprocess.list2cmdline(["python", *record["viewer_command"][1:]]))
if view_tail_s > 0:
    print(f"  (the window plays the episode's commands, then {view_tail_s} s more of the trot for viewing; that tail is not "
          "part of the measured episode. The window's episode is a separate run with the world's default attempt seed.)")

manifest = {
    "label": DEV_RUN_LABEL, "started_utc": started_utc, "finished_utc": utc_now(), "command": " ".join(sys.argv),
    "config_digest": config_digest(config), "base_seed": base_seed, "world_seed": world_seed, "world_seed_stream": WORLD_SEED_STREAM,
    "world_preset": preset, "skill": spec.name, "n_context": n_context, "n_dev": n_dev, "n_dreams": n_dreams,
    "dreams_kept": int(kept.n_rows), "confidence_threshold": threshold, "candidate_points_per_dim": [int(p) for p in points],
    "exploration_stream": exploration_stream, "runs": runs, "timings_s": timings,
    "tabpfn": {"billed_requests": len(billed), "tokens_quoted": tokens_quoted, "cache_hits": len(cache_hits),
               "usage_before": usage_before, "usage_after": usage_after,
               "request_hashes": sorted({record["key"] for record in billed if "key" in record})},
    "student": None if student is None else {"digest": student.digest, "path": "student.npz"},
    "training_error": training_error, "summary": summary, "watch_index": watch_index, "watch": watch, "versions": versions(),
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
print(f"\nwritten: {out_dir.relative_to(ROOT_DIR).as_posix()}/ (attempts.jsonl, dreams.npz, student, table.txt, manifest.json)", flush=True)

if arguments.watch:
    play(watch)

sys.exit(0 if student is not None else 1)
