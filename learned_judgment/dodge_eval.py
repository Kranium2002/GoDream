"""Closed-loop DODGE from the head camera (Person B): threat-gate calibration on simulation dev trajectories, then
the held-out evaluation of the frozen student, with the result file the UI reads.

Lane contract (Person B, dodge_eval)
------------------------------------
Scenarios of a split are learned_judgment.skills.dodge_vision's (its _split_plan: the rows command line's rng for
that split), the first N; attempt_seed = scenario_id. Every episode of a scenario gets a fresh
np.random.default_rng(augmentation seed), the seed being the second child of
np.random.SeedSequence([runtime.seed, split code], spawn_key=(scenario_id,)).spawn(2) (the rows command line's
augmentation child), so the episodes of a scenario see the same image variation and the methods are paired.

    NO_OP episode   dodge_vision.closed_loop with the zero command at every frame. Truth of the scorer: threat =
                    its collision is 1; t_hit = the world's first_collision_time_s (closed_loop drops it, so each
                    worker wraps its own Go2World.finish_episode to keep the raw outcome; nothing else changes).
                    calibrate also records the readout at every frame: runtime.readout of the newest F frames
                    (vision.take_frames; F = student.json input.frames) of the vision.frames window.
    replay          for one thresholds triple, learned_judgment.gate.ThreatGate over a NO_OP readout trace; the
                    first issued output is the scenario's issue (action, t_issue). Until that instant the dog has
                    not moved, so the replay equals the real closed loop up to it.
    calibrate       every triple of inference.gate.calibration_grid on the dev split: emitted (scenarios with an
                    issue), precision = threats among them / emitted, recall = threats with an issue at or before
                    t_hit / threats, false activations = emitted non-threats. Chosen: among triples with emitted >= 1
                    and precision >= inference.gate.precision_target the highest recall, then fewer false
                    activations, then higher threat, action, margin. None reaching the target: gate_passed false,
                    the highest precision (then recall) is chosen and the target is never lowered. The triple and
                    the calibration are written into student.json (the freeze); existing thresholds are kept unless
                    --refreeze.
    evaluate        per scenario the NO_OP episode and the frozen DodgeRuntime episode (runtime.reset() first;
                    policy = runtime.infer(newest F frames, t_s)), same seeds. Metrics from the saved records only.
    trace           the NO_OP episodes of a split exactly as calibrate runs them (same scenarios, seeds,
                    augmentation), saving the RAW camera frames instead of one student's readouts: the episode is
                    the same for every student (the dog does not move before the first issued command), so it is
                    run once and reused. The policy callback sees full vision.frames windows: the first window
                    gives frames 0 .. vision.frames - 1, every later window its newest frame; each window is checked
                    to equal the stored frames (exact equality) while the episode runs.
    calibrate --trace   offline calibration of one student on a saved trace: at every decision frame (the frames
                    calibrate reads out: vision.frames - 1 of the trace onwards) the student's F-frame window is
                    rebuilt from the stored frames, all readouts come from one batched observed.student_readout
                    (student.json "heads" and "input"; checked against DodgeRuntime.readout on 50 random frames,
                    seed runtime.seed, max |difference| <= 1e-9, else it fails), then the SAME gate replay, grid,
                    choice rule and precision target as calibrate. Dev traces only. --no-freeze writes
                    calibration_dev_offline.* and leaves student.json alone; otherwise calibration_dev.* and the
                    freeze, as calibrate.

Files under <observed.runs_dir>/<RUN>/:

    calibration_dev.json   grid table, chosen triple, per-scenario records (truth, NO_OP outcome, chosen issue)
    calibration_dev.npz    scenario_id (n,), t_s (T,), threat_confidence (n, T), action_scores (n, T, C)
    trace_<split>.npz      frames (n, T, N, N) float32 (every camera frame of the episode, oldest first), t_s (T,)
                           frame times, scenario_id (n,), threat (n,) bool (NO_OP collision), t_hit (n,) float (NaN:
                           none), kind (n,) str, window_frames (vision.frames), first_decision_index (window_frames - 1)
    trace_<split>.json     trace metadata (vision settings, seeds, config digest, rebuild check) and per-scenario
                           records (truth, NO_OP outcome)
    closed_loop_<split>.json  per-scenario records: scenario_id, kind, threat, t_hit, outcomes {no_op, snn},
                           issues {no_op, snn} as [t_s, action], cooldown_conflicts, inference_ms
    result.json            sections "calibration_dev", "closed_loop_<split>", "runtime_evidence", and the UI keys
                           run_id, skill, seeds, scenario_ids, metrics, paired (written by evaluate)

Command line (repository root)::

    python -m learned_judgment.dodge_eval calibrate --run RUN --student DIR [--n N] [--refreeze]
    python -m learned_judgment.dodge_eval calibrate --run RUN --student DIR --trace FILE [--no-freeze] [--refreeze]
    python -m learned_judgment.dodge_eval trace --run RUN [--split dev] [--n N]
    python -m learned_judgment.dodge_eval evaluate --run RUN --student DIR [--split test] [--n N]
    python -m learned_judgment.dodge_eval evidence --run RUN --student DIR

Exit codes: 0 done, 1 failed (traceback), 3 refused (thresholds already frozen without --refreeze, or none frozen
for evaluate), 4 calibrate finished but no triple reached the precision target (gate_passed false).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import io
import json
import math
import multiprocessing
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment.config import ConfigError, config_digest, load_config, require
from learned_judgment.fileio import atomic_write_bytes, atomic_write_text
from learned_judgment.gate import GateSettings, ThreatGate
from learned_judgment.runtime import DodgeRuntime, measure_latency
from learned_judgment.skills.dodge_vision import KINDS, SKILL, SPLIT_CODES, HeadView, _split_plan, closed_loop, frame_times
from learned_judgment.vision import VisionSettings, preprocess_frame, take_frames

__all__ = [
    "EXIT_GATE_FAILED",
    "EXIT_REFUSED",
    "METHODS",
    "calibrate",
    "choose_triple",
    "closed_loop_metrics",
    "evaluate",
    "latency_and_parity",
    "main",
    "offline_calibration",
    "paired_bootstrap",
    "replay_first_issue",
    "trace",
    "wilson_interval",
]

EXIT_REFUSED = 3
EXIT_GATE_FAILED = 4
METHODS = ("no_op", "snn")
_TOLERANCE = 1e-9  # s: issue times are sums of 0.02 s periods, hit times of 0.002 s physics steps
_THRESHOLD_KEYS = ("threat", "action", "margin")
_READOUT_CHECK_FRAMES = 50  # offline calibration: random frames whose batched readout is compared with DodgeRuntime.readout
_READOUT_TOLERANCE = 1e-9
_BATCH_ROWS = 4096  # rows per batched student_readout call (bounds memory, not the result)
_INFO_PRECISIONS = (0.95, 0.90)  # reported for information only: best recall at these dev precisions
_TRACE_ARRAYS = ("frames", "t_s", "scenario_id", "threat", "t_hit", "kind", "window_frames", "first_decision_index")


class _Refused(Exception):
    """A command refused to run (frozen thresholds without --refreeze, or none frozen for evaluate)."""


# ----- config ----------------------------------------------------------------------------------------------------


def _count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an int >= 1; got {value!r}")
    return int(value)


def _scenario_count(config: Mapping[str, Any], split: str, n: int | None) -> int:
    key = f"observed.dodge_vision.{split}_scenarios"
    return _count(key if n is None else "n", require(config, key) if n is None else n)


def _fraction(config: Mapping[str, Any], key: str) -> float:
    value = require(config, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
        raise ConfigError(f"config.yaml key {key!r} must be a number in (0, 1]; got {value!r}")
    return float(value)


def _level(config: Mapping[str, Any]) -> float:
    level = _fraction(config, "evaluation.bootstrap_confidence_level")
    if level >= 1.0:
        raise ConfigError(f"config.yaml key 'evaluation.bootstrap_confidence_level' must be below 1; got {level}")
    return level


def _grid(config: Mapping[str, Any]) -> list[dict[str, float]]:
    """Every triple of inference.gate.calibration_grid (threat x action x margin), ascending."""
    grid = require(config, "inference.gate.calibration_grid")
    if not isinstance(grid, Mapping):
        raise ConfigError(f"config.yaml key 'inference.gate.calibration_grid' must be a mapping; got {grid!r}")
    axes = {}
    for key in _THRESHOLD_KEYS:
        values = grid.get(key)
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in values
        ):
            raise ConfigError(f"config.yaml key 'inference.gate.calibration_grid.{key}' must be a non-empty list of numbers")
        axes[key] = sorted({float(v) for v in values})
    return [{"threat": t, "action": a, "margin": m} for t in axes["threat"] for a in axes["action"] for m in axes["margin"]]


def _seeds(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "runtime_seed": require(config, "runtime.seed"),
        "split": split,
        "split_code": SPLIT_CODES[split],
        "scenarios": "dodge_vision._split_plan: np.random.default_rng(np.random.SeedSequence([runtime_seed, split_code])), "
        "first n of sample_threat_scenarios",
        "augmentation": "np.random.default_rng(np.random.SeedSequence([runtime_seed, split_code], spawn_key=(scenario_id,))"
        ".spawn(2)[1]), fresh for every episode of the scenario (paired)",
        "attempt_seed": "scenario_id",
        "world_seed": require(config, "observed.world_seed"),
        "world_preset": require(config, "observed.world_preset"),
    }


def _plan(config: Mapping[str, Any], split: str, n: int) -> list[tuple[dict[str, Any], np.random.SeedSequence]]:
    scenarios, seeds = _split_plan(config, split, n)
    return [(scenario, seed.spawn(2)[1]) for scenario, seed in zip(scenarios, seeds)]


# ----- statistics ------------------------------------------------------------------------------------------------


def wilson_interval(k: int, n: int, level: float) -> tuple[float | None, float | None]:
    """Wilson score interval of k successes in n at confidence ``level``; (None, None) for n = 0."""
    if n == 0:
        return None, None
    z = statistics.NormalDist().inv_cdf(0.5 + level / 2.0)
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap(differences: Sequence[float], resamples: int, level: float, rng: np.random.Generator) -> dict[str, Any]:
    """Mean of per-scenario differences and its percentile bootstrap interval (resampling scenarios)."""
    d = np.asarray(differences, dtype=np.float64)
    if d.size == 0:
        return {"diff": None, "ci_low": None, "ci_high": None, "n": 0}
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):  # chunks keep the index array small
        stop = min(start + 1000, resamples)
        means[start:stop] = d[rng.integers(0, d.size, size=(stop - start, d.size))].mean(axis=1)
    low, high = np.quantile(means, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    return {"diff": float(d.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(d.size)}


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "min": None}
    return {"n": len(values), "median": float(np.median(values)), "min": float(np.min(values))}


# ----- workers ---------------------------------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(config: Mapping[str, Any], student_dir: str | None, thresholds: Mapping[str, float] | None) -> None:
    """Per process: its own Go2World (observed.world_seed / world_preset, no trace files), HeadView, the
    vision.frames window settings and DodgeRuntime (none for student_dir None: the trace runs no student);
    finish_episode is wrapped to keep the raw outcome."""
    from learned_judgment.env import Go2World  # on use: loads the walking policy's runtime

    config = copy.deepcopy(dict(config))
    config["world"]["save_traces"] = False
    world = Go2World(config, require(config, "observed.world_seed"), require(config, "observed.world_preset"))
    finish = world.finish_episode

    def finish_and_keep() -> dict:
        outcome = finish()
        _WORKER["outcome"] = outcome
        return outcome

    world.finish_episode = finish_and_keep
    if student_dir is None:
        runtime = None
        no_op = GateSettings.from_config(config, _grid(config)[0]).no_op  # the runtime's own source of NO_OP
    else:
        runtime = DodgeRuntime.load(student_dir, config, thresholds)
        no_op = runtime.settings.no_op
    zero ={key: float(value) for key, value in require(config, "skills.dodge_vision.actions")[no_op].items()}
    _WORKER.update(world=world, view=HeadView(config), settings=VisionSettings.from_config(config), runtime=runtime,
                   no_op=no_op, zero=zero, outcome=None)


def _episode(policy: Callable[[np.ndarray, float], Mapping[str, Any]], scenario: Mapping[str, Any],
             augment_seed: np.random.SeedSequence) -> tuple[dict[str, Any], float | None]:
    """closed_loop of one scenario and the world's first_collision_time_s of that episode."""
    w = _WORKER
    w["outcome"] = None
    out = closed_loop(w["world"], w["view"], scenario, scenario["scenario_id"], policy, w["settings"],
                      np.random.default_rng(augment_seed))
    if w["outcome"] is None:
        raise RuntimeError("the world's raw outcome was not captured: finish_episode is not the wrapped one")
    hit = w["outcome"]["first_collision_time_s"]
    return out, None if hit is None else float(hit)


def _calibrate_task(task: tuple) -> dict[str, Any]:
    scenario, augment_seed = task
    w = _WORKER
    runtime, frames = w["runtime"], w["runtime"].frames

    def readout_policy(window: np.ndarray, t_s: float) -> dict[str, Any]:
        threat, scores = runtime.readout(take_frames(window, frames))
        return {"command": w["zero"], "action": w["no_op"], "threat_confidence": float(threat),
                "action_scores": [float(s) for s in scores]}

    out, hit = _episode(readout_policy, scenario, augment_seed)
    trace = out["trace"]
    return {
        "scenario_id": scenario["scenario_id"],
        "measured": out["measured"],
        "first_collision_time_s": hit,
        "t_s": [float(e["t_s"]) for e in trace],
        "threat_confidence": [e["threat_confidence"] for e in trace],
        "action_scores": [e["action_scores"] for e in trace],
    }


def _evaluate_task(task: tuple) -> dict[str, Any]:
    scenario, augment_seed = task
    w = _WORKER
    runtime, frames = w["runtime"], w["runtime"].frames

    def zero_policy(window: np.ndarray, t_s: float) -> dict[str, Any]:
        return {"command": w["zero"], "action": w["no_op"]}

    def snn_policy(window: np.ndarray, t_s: float) -> dict[str, Any]:
        return runtime.infer(take_frames(window, frames), t_s)

    runtime.reset()
    base, base_hit = _episode(zero_policy, scenario, augment_seed)
    runtime.reset()
    snn, snn_hit = _episode(snn_policy, scenario, augment_seed)
    conflict = [bool(e.get("cooldown_conflict", False)) for e in snn["trace"]]
    rising = sum(1 for i, c in enumerate(conflict) if c and (i == 0 or not conflict[i - 1]))
    return {
        "scenario_id": scenario["scenario_id"],
        "outcomes": {"no_op": {**base["measured"], "first_collision_time_s": base_hit},
                     "snn": {**snn["measured"], "first_collision_time_s": snn_hit}},
        "issues": {"no_op": [[float(t), a] for t, a in base["issued"]], "snn": [[float(t), a] for t, a in snn["issued"]]},
        "cooldown_conflicts": {"no_op": 0, "snn": rising},
        "inference_ms": {"snn": [float(e["inference_ms"]) for e in snn["trace"]]},
        "n_decisions": len(snn["trace"]),
    }


def _trace_task(task: tuple) -> dict[str, Any]:
    """The NO_OP episode of one scenario with its raw camera frames: the first full window gives frames
    0 .. F - 1, every later window its newest frame; each window must equal the stored frames it overlaps."""
    scenario, augment_seed = task
    w = _WORKER
    frames: list[np.ndarray] = []
    last: list[np.ndarray] = []

    def record_policy(window: np.ndarray, t_s: float) -> dict[str, Any]:
        overlap = window.shape[0] - 1
        if not frames:
            frames.extend(np.array(frame, dtype=np.float32, copy=True) for frame in window)
        else:
            if overlap and not np.array_equal(window[:overlap], np.stack(frames[len(frames) - overlap:])):
                raise RuntimeError(f"scenario {scenario['scenario_id']}: the window at t {t_s} is not the stored frames "
                                   "plus one new frame; the trace cannot rebuild it")
            frames.append(np.array(window[-1], dtype=np.float32, copy=True))
        last[:] = [np.array(window, dtype=np.float32, copy=True)]
        return {"command": w["zero"], "action": w["no_op"]}

    out, hit = _episode(record_policy, scenario, augment_seed)
    stacked = np.stack(frames)
    rebuilt = stacked[stacked.shape[0] - last[0].shape[0]:]
    if not np.array_equal(rebuilt, last[0]):  # one window rebuilt from the stored frames, compared exactly
        raise RuntimeError(f"scenario {scenario['scenario_id']}: the last window differs from its rebuild")
    return {
        "scenario_id": scenario["scenario_id"],
        "measured": out["measured"],
        "first_collision_time_s": hit,
        "t_s": [float(e["t_s"]) for e in out["trace"]],
        "frames": stacked,
        "windows_checked": len(out["trace"]),
    }


def _run(config: Mapping[str, Any], student_dir: Path | None, thresholds: Mapping[str, float] | None,
         function: Callable[[tuple], Any], tasks: list[tuple], on_result: Callable[[int], None] | None = None) -> list[Any]:
    """function over tasks in observed.workers spawned processes, results in task order, sparse progress lines."""
    workers = require(config, "observed.workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ConfigError(f"config.yaml key 'observed.workers' must be an int >= 1; got {workers!r}")
    every = max(1, len(tasks) // 10)
    results: list[Any] = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(workers, len(tasks)), initializer=_worker_init,
                      initargs=(config, None if student_dir is None else str(student_dir),
                                None if thresholds is None else dict(thresholds))) as pool:
        for result in pool.imap(function, tasks):
            results.append(result)
            if on_result is not None:
                on_result(len(results))
            if len(results) % every == 0 or len(results) == len(tasks):
                print(f"{len(results)}/{len(tasks)}", flush=True)
    return results


# ----- gate replay and calibration -------------------------------------------------------------------------------


def replay_first_issue(settings: GateSettings, t_s: Sequence[float], threat: Sequence[float],
                       scores: Sequence[Sequence[float]]) -> tuple[str, float] | None:
    """(action, t_issue) of the first issued output of a fresh ThreatGate over one readout trace, or None."""
    if not threat or max(threat) < settings.threat_threshold:
        return None  # no frame can vote: threat_ok is false everywhere
    gate = ThreatGate(settings)
    for t, confidence, row in zip(t_s, threat, scores):
        out = gate.update(confidence, row, t)
        if out.issued:
            return str(out.action), float(t)
    return None


def _on_time(t_issue: float | None, t_hit: float | None) -> bool:
    """An issue (at t_issue; None: no issue) at or before t_hit; any issue counts when t_hit is unknown."""
    return t_issue is not None and (t_hit is None or float(t_issue) <= t_hit + _TOLERANCE)


def _grid_row(triple: Mapping[str, float], issues: Sequence[tuple[str, float] | None], threat: Sequence[bool],
              t_hit: Sequence[float | None]) -> dict[str, Any]:
    """Metrics of one triple from the replayed first issues (action, t_issue) or None per scenario."""
    emitted = sum(1 for issue in issues if issue is not None)
    correct = sum(1 for issue, is_threat in zip(issues, threat) if issue is not None and is_threat)
    n_threats = sum(1 for is_threat in threat if is_threat)
    recalled = sum(1 for issue, is_threat, hit in zip(issues, threat, t_hit)
                   if is_threat and _on_time(None if issue is None else issue[1], hit))
    return {
        **{key: float(triple[key]) for key in _THRESHOLD_KEYS},
        "emitted": emitted,
        "correct": correct,
        "precision": correct / emitted if emitted else None,
        "recalled": recalled,
        "recall": recalled / n_threats if n_threats else None,
        "false_activations": emitted - correct,
    }


def choose_triple(rows: Sequence[Mapping[str, Any]], target: float) -> tuple[Mapping[str, Any], bool, str]:
    """(chosen grid row, gate_passed, rule used) by the calibration rule of the module docstring."""
    def thresholds(row: Mapping[str, Any]) -> tuple[float, float, float]:
        return row["threat"], row["action"], row["margin"]

    def recall(row: Mapping[str, Any]) -> float:
        return -1.0 if row["recall"] is None else row["recall"]

    passing = [row for row in rows if row["emitted"] >= 1 and row["precision"] >= target]
    if passing:
        rule = "precision >= target: highest recall, then fewer false activations, then higher threat, action, margin"
        return max(passing, key=lambda r: (recall(r), -r["false_activations"], *thresholds(r))), True, rule
    emitting = [row for row in rows if row["emitted"] >= 1]
    if emitting:
        rule = "NO triple reached the target: highest precision, then recall, then fewer false activations, then higher thresholds"
        return max(emitting, key=lambda r: (r["precision"], recall(r), -r["false_activations"], *thresholds(r))), False, rule
    rule = "NO triple emitted any command: the highest thresholds"
    return max(rows, key=thresholds), False, rule


def _replay_grid(config: Mapping[str, Any], grid: Sequence[Mapping[str, float]], t_s: Sequence[float],
                 traces: Sequence[tuple[Sequence[float], Sequence[Sequence[float]]]], threat: Sequence[bool],
                 t_hit: Sequence[float | None]) -> tuple[list[dict[str, Any]], Mapping[str, Any], bool, str, list]:
    """Every triple replayed over the NO_OP readout traces (threat_confidence, action_scores) per scenario:
    (grid rows, chosen row, gate_passed, rule, the chosen triple's first issue per scenario). calibrate and the
    offline calibration both run this."""
    target = _fraction(config, "inference.gate.precision_target")
    rows, issues_by_triple = [], {}
    for triple in grid:
        settings = GateSettings.from_config(config, triple)
        issues = [replay_first_issue(settings, t_s, c, s) for c, s in traces]
        rows.append(_grid_row(triple, issues, threat, t_hit))
        issues_by_triple[tuple(triple[k] for k in _THRESHOLD_KEYS)] = issues
    chosen, passed, rule = choose_triple(rows, target)
    return rows, chosen, passed, rule, issues_by_triple[tuple(chosen[k] for k in _THRESHOLD_KEYS)]


def _freeze(student_dir: Path, thresholds: Mapping[str, float], calibration: Mapping[str, Any], refreeze: bool) -> None:
    """Write thresholds and calibration into student.json; refused if another process froze it meanwhile."""
    student = _read_student(student_dir)  # re-read: refuse if another process froze meanwhile
    if student.get("thresholds") is not None and not refreeze:
        raise _Refused(f"{student_dir / 'student.json'} was frozen while calibrating; pass --refreeze")
    student["thresholds"], student["calibration"] = dict(thresholds), dict(calibration)
    _write_json(student_dir / "student.json", student)


def _read_student(student_dir: Path) -> dict[str, Any]:
    student = json.loads((student_dir / "student.json").read_text(encoding="utf-8"))
    if not isinstance(student, dict):
        raise ValueError(f"{student_dir / 'student.json'} is not a JSON object")
    return student


def _merge_result(run_dir: Path, run_id: str, updates: Mapping[str, Any]) -> Path:
    """Merge top-level keys into <run_dir>/result.json (observed's file: other keys are kept)."""
    from learned_judgment.observed import _dumps, _jsonable, _utc_now

    path = run_dir / "result.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"run_id": run_id, "skill": SKILL}
    if result.get("run_id") != run_id or result.get("skill") != SKILL:
        raise ValueError(f"{path} belongs to run {result.get('run_id')!r} of skill {result.get('skill')!r}")
    result.update(_jsonable(dict(updates)))
    result["updated_utc"] = _utc_now().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, _dumps(result, indent=1) + "\n")
    return path


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    from learned_judgment.observed import _dumps

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, _dumps(record, indent=1) + "\n")


def calibrate(config: Mapping[str, Any], run_id: str, student_dir: str | Path, n: int | None = None,
              refreeze: bool = False, argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Choose and freeze the gate thresholds on the dev split; returns the calibration written into student.json.

    Raises _Refused when student.json already holds thresholds and refreeze is false; whatever the episodes raise.
    """
    from learned_judgment.observed import _run_dir, _utc_now

    student_dir = Path(student_dir).resolve()
    if _read_student(student_dir).get("thresholds") is not None and not refreeze:
        raise _Refused(f"{student_dir / 'student.json'} already holds frozen thresholds; pass --refreeze to replace them")
    grid = _grid(config)
    target = _fraction(config, "inference.gate.precision_target")
    level = _level(config)
    n = _scenario_count(config, "dev", n)
    probe = DodgeRuntime.load(student_dir, config, grid[0])  # fails before any episode; readout ignores thresholds
    run_dir = _run_dir(config, run_id)
    section = "calibration_dev"
    _merge_result(run_dir, run_id, {section: {"state": "running", "argv": list(argv or []), "student": student_dir}})
    tasks = _plan(config, "dev", n)
    print(f"calibrate: dev, {n} NO_OP episodes with readouts (F {probe.frames}), grid {len(grid)} triples", flush=True)
    try:
        started = time.perf_counter()
        results = _run(config, student_dir, grid[0], _calibrate_task, tasks)
        episodes_s = time.perf_counter() - started
        t_s = results[0]["t_s"]
        if any(r["t_s"] != t_s for r in results):
            raise RuntimeError("the readout traces of the dev scenarios have different frame times")
        kinds = {s["scenario_id"]: s["kind"] for s, _ in tasks}
        threat = [r["measured"]["collision"] == 1.0 for r in results]
        t_hit = [r["first_collision_time_s"] for r in results]
        traces = [(r["threat_confidence"], r["action_scores"]) for r in results]
        rows, chosen, passed, rule, issues = _replay_grid(config, grid, t_s, traces, threat, t_hit)
        thresholds = {key: chosen[key] for key in _THRESHOLD_KEYS}
        low, high = wilson_interval(chosen["correct"], chosen["emitted"], level)

        sid = np.array([r["scenario_id"] for r in results], dtype=np.int64)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, scenario_id=sid, t_s=np.asarray(t_s, dtype=np.float64),
                            threat_confidence=np.array([c for c, _ in traces], dtype=np.float64),
                            action_scores=np.array([s for _, s in traces], dtype=np.float64))
        atomic_write_bytes(run_dir / f"{section}.npz", buffer.getvalue())
        records = [
            {"scenario_id": r["scenario_id"], "kind": kinds[r["scenario_id"]], "threat": is_threat, "t_hit": hit,
             "no_op": r["measured"], "n_readouts": len(r["t_s"]),
             "chosen_issue": None if issue is None else {"action": issue[0], "t_s": issue[1]}}
            for r, is_threat, hit, issue in zip(results, threat, t_hit, issues)
        ]
        calibration = {
            "split": "dev",
            "n_scenarios": n,
            "n_threats": int(sum(threat)),
            "grid_size": len(grid),
            "chosen": {**chosen, "precision_ci": [low, high], "ci_level": level, "ci_method": "Wilson"},
            "precision_target": target,
            "gate_passed": passed,
            "rule": rule,
            "seeds": _seeds(config, "dev"),
            "t_hit": "Go2World.finish_episode first_collision_time_s of the NO_OP episode",
            "run_id": run_id,
            "records": run_dir / f"{section}.json",
            "traces": run_dir / f"{section}.npz",
            "frozen_utc": _utc_now().isoformat(),
            "config_digest": config_digest(config),
        }
        _write_json(run_dir / f"{section}.json", {"skill": SKILL, "student": student_dir, "calibration": calibration,
                                                  "grid": rows, "records": records})
        _freeze(student_dir, thresholds, calibration, refreeze)
    except Exception as error:
        _merge_result(run_dir, run_id, {section: {"state": "failed", "error": {"type": type(error).__name__,
                                                                                 "message": str(error)}}})
        raise
    _merge_result(run_dir, run_id, {section: {"state": "done", "argv": list(argv or []), "student": student_dir,
                                              "thresholds": thresholds, "episodes_wall_s": episodes_s, **calibration}})
    print(f"thresholds {thresholds}: emitted {chosen['emitted']}, precision {chosen['precision']} "
          f"(Wilson {low}..{high}), recall {chosen['recall']}, false activations {chosen['false_activations']}; "
          f"{calibration['n_threats']} threats of {n}", flush=True)
    if not passed:
        print(f"GATE NOT PASSED: no triple of the grid reaches precision_target {target} on dev ({rule}). "
              "The target was not lowered.", flush=True)
    print(f"froze {student_dir / 'student.json'}", flush=True)
    return calibration


# ----- saved NO_OP trace and offline calibration -----------------------------------------------------------------


def trace(config: Mapping[str, Any], run_id: str, split: str = "dev", n: int | None = None,
          argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the NO_OP closed-loop episodes of ``split`` as calibrate does and save their raw frames to
    <run>/trace_<split>.npz and .json (module docstring); returns the metadata written to the .json."""
    from learned_judgment.observed import _run_dir, _utc_now

    if split not in ("dev", "test"):
        raise ValueError(f"split must be dev or test; got {split!r}")
    n = _scenario_count(config, split, n)
    settings = VisionSettings.from_config(config)
    times = np.asarray(frame_times(config), dtype=np.float64)
    first = settings.frames - 1
    run_dir = _run_dir(config, run_id)
    section = f"trace_{split}"
    tasks = _plan(config, split, n)
    kinds = {s["scenario_id"]: s["kind"] for s, _ in tasks}
    _merge_result(run_dir, run_id, {section: {"state": "running", "argv": list(argv or [])}})
    print(f"trace: {split}, {n} NO_OP episodes, {times.size} camera frames each (window {settings.frames})", flush=True)
    try:
        started = time.perf_counter()
        results = _run(config, None, None, _trace_task, tasks)
        wall_s = time.perf_counter() - started
        shape = (times.size, settings.frame_size, settings.frame_size)
        for r in results:
            if r["frames"].shape != shape:
                raise RuntimeError(f"scenario {r['scenario_id']}: {r['frames'].shape} frames stored, frame_times gives {shape}")
            if r["t_s"] != times[first:].tolist():
                raise RuntimeError(f"scenario {r['scenario_id']}: the policy's frame times differ from frame_times[{first}:]")
        frames = np.stack([r["frames"] for r in results]).astype(np.float32, copy=False)
        sid = np.array([r["scenario_id"] for r in results], dtype=np.int64)
        threat = np.array([r["measured"]["collision"] == 1.0 for r in results], dtype=bool)
        t_hit = np.array([np.nan if r["first_collision_time_s"] is None else r["first_collision_time_s"] for r in results],
                         dtype=np.float64)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, frames=frames, t_s=times, scenario_id=sid, threat=threat, t_hit=t_hit,
                            kind=np.array([str(kinds[int(s)]) for s in sid]),
                            window_frames=np.int64(settings.frames), first_decision_index=np.int64(first))
        npz_path = run_dir / f"{section}.npz"
        atomic_write_bytes(npz_path, buffer.getvalue())
        records = [
            {"scenario_id": r["scenario_id"], "kind": kinds[r["scenario_id"]], "threat": bool(is_threat),
             "t_hit": r["first_collision_time_s"], "no_op": r["measured"], "n_frames": int(r["frames"].shape[0]),
             "windows_checked": r["windows_checked"]}
            for r, is_threat in zip(results, threat)
        ]
        meta = {
            "skill": SKILL,
            "run_id": run_id,
            "split": split,
            "n_scenarios": n,
            "n_threats": int(threat.sum()),
            "frames_shape": list(frames.shape),
            "window_frames": settings.frames,
            "first_decision_index": first,
            "decision_frames": int(times.size - first),
            "vision": dataclasses.asdict(settings),
            "seeds": _seeds(config, split),
            "t_hit": "Go2World.finish_episode first_collision_time_s of the NO_OP episode (NaN in the npz: none)",
            "window_rebuild_check": {
                "windows_checked": int(sum(r["windows_checked"] for r in results)),
                "method": "every policy window equal (np.array_equal) to the stored frames it overlaps plus its newest "
                          "frame; the last window of each episode rebuilt from the stored stack and compared exactly",
                "passed": True,
            },
            "npz": npz_path,
            "wall_s": wall_s,
            "created_utc": _utc_now().isoformat(),
            "config_digest": config_digest(config),
        }
        _write_json(run_dir / f"{section}.json", {**meta, "records": records})
    except Exception as error:
        _merge_result(run_dir, run_id, {section: {"state": "failed", "error": {"type": type(error).__name__,
                                                                                 "message": str(error)}}})
        raise
    _merge_result(run_dir, run_id, {section: {"state": "done", "argv": list(argv or []), **meta}})
    print(f"trace: {npz_path}, frames {list(frames.shape)}, {meta['n_threats']} threats of {n}, "
          f"{meta['window_rebuild_check']['windows_checked']} windows checked, {wall_s:.1f} s", flush=True)
    return meta


def _load_trace(path: str | Path) -> dict[str, Any]:
    """The arrays of trace_<split>.npz and the metadata of the .json beside it ("meta"), checked for shape."""
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in _TRACE_ARRAYS if name not in archive.files]
        if missing:
            raise ValueError(f"{path} lacks {missing}: not a dodge_eval trace")
        data: dict[str, Any] = {name: np.array(archive[name]) for name in _TRACE_ARRAYS}
    meta_path = path.with_suffix(".json")
    data["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    data["path"] = path
    frames = data["frames"]
    n = frames.shape[0]
    if frames.ndim != 4 or frames.dtype != np.float32 or frames.shape[2] != frames.shape[3]:
        raise ValueError(f"{path} frames must be float32 (n, T, N, N); got {frames.dtype} {frames.shape}")
    if data["t_s"].shape != (frames.shape[1],) or any(data[k].shape != (n,) for k in ("scenario_id", "threat", "t_hit", "kind")):
        raise ValueError(f"{path} arrays disagree in shape with frames {frames.shape}")
    return data


def offline_calibration(student_dir: str | Path, trace_path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Calibrate one student on a saved dev trace without any episode (module docstring, "calibrate --trace").

    Returns {"student", "network_sha256", "frames", "window_encoding", "thresholds", "emitted", "correct",
    "precision", "precision_ci", "recalled", "recall", "false_activations", "gate_passed", "n_scenarios",
    "n_threats", "precision_target", "best_recall_at_precision" {"0.95", "0.90"} (information only),
    "readout_check", "wall_s", "calibration" (the block calibrate freezes, without run paths), "grid" (rows),
    "records" (per scenario), "readouts" {"scenario_id", "t_s", "threat_confidence", "action_scores"} (arrays)}.
    Writes nothing.

    Raises ValueError for a trace that is not a dev trace or does not fit the student (frame size, camera settings,
    window longer than the trace's); RuntimeError when the batched readouts differ from DodgeRuntime.readout by more
    than 1e-9; whatever loading the student raises.
    """
    from learned_judgment.observed import student_readout
    from learned_judgment.students.snn_core import load_network

    started = time.perf_counter()
    student_dir = Path(student_dir).resolve()
    student = _read_student(student_dir)
    grid = _grid(config)
    target = _fraction(config, "inference.gate.precision_target")
    level = _level(config)
    runtime = DodgeRuntime.load(student_dir, config, grid[0])  # checks the student against config; readout ignores thresholds
    data = _load_trace(trace_path)
    meta = data["meta"]
    if meta.get("split") != "dev":
        raise ValueError(f"{data['path']} is a {meta.get('split')!r} trace; thresholds are chosen on dev only")
    frames = data["frames"]
    n, total, size = frames.shape[0], frames.shape[1], frames.shape[2]
    f, first = runtime.frames, int(data["first_decision_index"])
    if size != runtime.frame_size:
        raise ValueError(f"the trace's frames are {size} x {size}; the student takes {runtime.frame_size}")
    camera = {key: value for key, value in (student.get("vision") or {}).items() if key != "frames"}
    traced = {key: value for key, value in meta["vision"].items() if key != "frames"}
    if camera and camera != traced:
        raise ValueError(f"the student's camera settings {camera} differ from the trace's {traced}")
    if not 1 <= f <= first + 1:
        raise ValueError(f"the student's window of {f} frames is longer than the trace's {first + 1}")

    # (n, T - F + 1, N, N, F): window j = frames j .. j + F - 1; decision frame d ends window d - F + 1.
    view = np.lib.stride_tricks.sliding_window_view(frames, f, axis=1)
    windows = np.moveaxis(view, -1, 2)[:, first - f + 1:]  # (n, Td, F, N, N), oldest frame first
    td = windows.shape[1]
    network, _ = load_network(student_dir / "network")
    threat = np.empty((n, td), dtype=np.float64)
    scores = np.empty((n, td, len(runtime.action_names)), dtype=np.float64)
    per_chunk = max(1, _BATCH_ROWS // td)
    for start in range(0, n, per_chunk):
        stop = min(n, start + per_chunk)
        block = np.ascontiguousarray(windows[start:stop]).reshape((stop - start) * td, f * size * size)
        t_block, s_block = student_readout(network, block, student["heads"], student["input"])
        threat[start:stop] = t_block.reshape(stop - start, td)
        scores[start:stop] = s_block.reshape(stop - start, td, -1)

    seed = int(require(config, "runtime.seed"))
    picks = np.random.default_rng(seed).choice(n * td, size=min(_READOUT_CHECK_FRAMES, n * td), replace=False)
    largest = 0.0
    for pick in picks:
        i, j = divmod(int(pick), td)
        one_threat, one_scores = runtime.readout(np.array(windows[i, j], dtype=np.float32))
        largest = max(largest, abs(one_threat - threat[i, j]), float(np.max(np.abs(one_scores - scores[i, j]))))
    check = {"n_frames": int(picks.size), "max_abs_difference": largest, "tolerance": _READOUT_TOLERANCE,
             "passed": largest <= _READOUT_TOLERANCE, "seed": seed,
             "method": "np.random.default_rng(runtime.seed).choice of (scenario, decision frame) without replacement; "
                       "DodgeRuntime.readout of that window against the batched observed.student_readout"}
    if not check["passed"]:
        raise RuntimeError(f"batched readouts differ from DodgeRuntime.readout: {check}")

    t_s = data["t_s"][first:].tolist()
    truth = [bool(x) for x in data["threat"]]
    t_hit = [None if math.isnan(float(h)) else float(h) for h in data["t_hit"]]
    traces = [(threat[i].tolist(), scores[i].tolist()) for i in range(n)]
    rows, chosen, passed, rule, issues = _replay_grid(config, grid, t_s, traces, truth, t_hit)
    thresholds = {key: chosen[key] for key in _THRESHOLD_KEYS}
    low, high = wilson_interval(chosen["correct"], chosen["emitted"], level)

    def best_at(precision: float) -> dict[str, Any] | None:
        rows_ok = [r for r in rows if r["emitted"] >= 1 and r["precision"] >= precision]
        if not rows_ok:
            return None
        top = max(rows_ok, key=lambda r: (-1.0 if r["recall"] is None else r["recall"], -r["false_activations"],
                                          *(r[k] for k in _THRESHOLD_KEYS)))
        return {key: top[key] for key in (*_THRESHOLD_KEYS, "emitted", "precision", "recall", "false_activations")}

    sid = data["scenario_id"]
    kinds = [str(k) for k in data["kind"]]
    outcomes = {r["scenario_id"]: r.get("no_op") for r in meta.get("records", [])}
    records = [
        {"scenario_id": int(s), "kind": kind, "threat": is_threat, "t_hit": hit, "no_op": outcomes.get(int(s)),
         "n_readouts": td, "chosen_issue": None if issue is None else {"action": issue[0], "t_s": issue[1]}}
        for s, kind, is_threat, hit, issue in zip(sid, kinds, truth, t_hit, issues)
    ]
    calibration = {
        "split": "dev",
        "n_scenarios": n,
        "n_threats": int(sum(truth)),
        "grid_size": len(grid),
        "chosen": {**chosen, "precision_ci": [low, high], "ci_level": level, "ci_method": "Wilson"},
        "precision_target": target,
        "gate_passed": passed,
        "rule": rule,
        "seeds": meta["seeds"],
        "t_hit": "Go2World.finish_episode first_collision_time_s of the NO_OP episode",
        "source": "offline: the saved NO_OP trace (dodge_eval trace), windows rebuilt from its frames, readouts batched "
                  "through observed.student_readout, the same gate replay as calibrate",
        "trace": data["path"],
        "trace_config_digest": meta.get("config_digest"),
        "readout_check": check,
        "config_digest": config_digest(config),
    }
    return {
        "student": student_dir,
        "network_sha256": student.get("network", {}).get("sha256"),
        "frames": f,
        "window_encoding": student["input"].get("window_encoding", "raw"),
        "thresholds": thresholds,
        **{key: chosen[key] for key in ("emitted", "correct", "precision", "recalled", "recall", "false_activations")},
        "precision_ci": [low, high],
        "gate_passed": passed,
        "n_scenarios": n,
        "n_threats": int(sum(truth)),
        "precision_target": target,
        "best_recall_at_precision": {f"{p:.2f}": best_at(p) for p in _INFO_PRECISIONS},
        "readout_check": check,
        "wall_s": time.perf_counter() - started,
        "calibration": calibration,
        "grid": rows,
        "records": records,
        "readouts": {"scenario_id": sid, "t_s": np.asarray(t_s, dtype=np.float64), "threat_confidence": threat,
                     "action_scores": scores},
    }


def calibrate_from_trace(config: Mapping[str, Any], run_id: str, student_dir: str | Path, trace_path: str | Path,
                         freeze: bool = True, refreeze: bool = False, argv: Sequence[str] | None = None) -> dict[str, Any]:
    """offline_calibration plus its files: calibration_dev.* and the freeze (freeze True), or calibration_dev_offline.*
    and student.json untouched (freeze False); the result.json section of the same name. Returns the report
    without "grid", "records" and "readouts"."""
    from learned_judgment.observed import _run_dir, _utc_now

    student_dir = Path(student_dir).resolve()
    if freeze and _read_student(student_dir).get("thresholds") is not None and not refreeze:
        raise _Refused(f"{student_dir / 'student.json'} already holds frozen thresholds; pass --refreeze to replace them")
    run_dir = _run_dir(config, run_id)
    section = "calibration_dev" if freeze else "calibration_dev_offline"
    _merge_result(run_dir, run_id, {section: {"state": "running", "argv": list(argv or []), "student": student_dir,
                                              "trace": Path(trace_path).resolve(), "frozen": freeze}})
    try:
        report = offline_calibration(student_dir, trace_path, config)
        readouts, rows, records = report.pop("readouts"), report.pop("grid"), report.pop("records")
        report["calibration"] = {**report["calibration"], "run_id": run_id, "records": run_dir / f"{section}.json",
                                 "traces": run_dir / f"{section}.npz",
                                 "frozen_utc": _utc_now().isoformat() if freeze else None}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **readouts)
        atomic_write_bytes(run_dir / f"{section}.npz", buffer.getvalue())
        _write_json(run_dir / f"{section}.json", {"skill": SKILL, "student": student_dir, "calibration": report["calibration"],
                                                  "grid": rows, "records": records})
        if freeze:
            _freeze(student_dir, report["thresholds"], report["calibration"], refreeze)
    except Exception as error:
        _merge_result(run_dir, run_id, {section: {"state": "failed", "error": {"type": type(error).__name__,
                                                                                 "message": str(error)}}})
        raise
    _merge_result(run_dir, run_id, {section: {"state": "done", "argv": list(argv or []), "frozen": freeze, **report}})
    ci = report["precision_ci"]
    print(f"calibrate (trace): thresholds {report['thresholds']}: emitted {report['emitted']}, precision "
          f"{report['precision']} (Wilson {ci[0]}..{ci[1]}), recall {report['recall']}, false activations "
          f"{report['false_activations']}; {report['n_threats']} threats of {report['n_scenarios']}; best recall at "
          f"precision {json.dumps(report['best_recall_at_precision'])}; readout check {report['readout_check']['max_abs_difference']}"
          f" <= {_READOUT_TOLERANCE}; {report['wall_s']:.1f} s", flush=True)
    if not report["gate_passed"]:
        print(f"GATE NOT PASSED: no triple of the grid reaches precision_target {report['precision_target']} on dev. "
              "The target was not lowered.", flush=True)
    print(f"froze {student_dir / 'student.json'}" if freeze else "not frozen (--no-freeze)", flush=True)
    return report


# ----- evaluation ------------------------------------------------------------------------------------------------


def _method_metrics(records: Sequence[Mapping[str, Any]], method: str, action_names: Sequence[str],
                    duration_s: float, t_first_s: float, level: float) -> dict[str, Any]:
    n = len(records)
    threats = [r for r in records if r["threat"]]
    safe_minutes = sum(1 for r in records if not r["threat"]) * duration_s / 60.0
    issues = [(r, t, a) for r in records for t, a in r["issues"][method]]
    correct = sum(1 for r, _, _ in issues if r["threat"])
    false_activations = len(issues) - correct
    recalled = abstentions = 0
    from_first_frame, time_left = [], []
    for r in threats:
        listed = r["issues"][method]
        if not listed:
            abstentions += 1
            continue
        t_issue = float(listed[0][0])
        from_first_frame.append(t_issue - t_first_s)
        if r["t_hit"] is not None:
            time_left.append(float(r["t_hit"]) - t_issue)
        recalled += int(_on_time(t_issue, r["t_hit"]))
    outcomes = [r["outcomes"][method] for r in records]
    collisions = int(sum(o["collision"] for o in outcomes))
    falls = int(sum(o["fall"] for o in outcomes))
    separation = [float(o["min_separation_m"]) for o in outcomes]
    inference = [ms for r in records for ms in r.get("inference_ms", {}).get(method, [])]
    counts = Counter(a for _, _, a in issues)
    return {
        "n_scenarios": n,
        "issued": len(issues),
        "issued_correct": correct,
        "precision": correct / len(issues) if issues else None,
        "precision_ci": list(wilson_interval(correct, len(issues), level)),
        "false_activations": false_activations,
        "safe_minutes": safe_minutes,
        "false_activations_per_safe_minute": false_activations / safe_minutes if safe_minutes > 0 else None,
        "n_threats": len(threats),
        "recalled": recalled,
        "recall": recalled / len(threats) if threats else None,
        "recall_ci": list(wilson_interval(recalled, len(threats), level)),
        "misses": len(threats) - recalled,
        "late": len(threats) - recalled - abstentions,
        "abstentions": abstentions,
        "collisions": collisions,
        "collision_rate": collisions / n if n else None,
        "collision_ci": list(wilson_interval(collisions, n, level)),
        "falls": falls,
        "fall_rate": falls / n if n else None,
        "fall_ci": list(wilson_interval(falls, n, level)),
        "mean_min_separation_m": float(np.mean(separation)) if separation else None,
        "min_min_separation_m": float(np.min(separation)) if separation else None,
        "mean_command_travel_m": float(np.mean([o["command_travel_m"] for o in outcomes])) if outcomes else None,
        "cooldown_conflicts": int(sum(r["cooldown_conflicts"][method] for r in records)),
        "latency": {"t_first_frame_s": t_first_s, "issue_after_first_frame_s": _stats(from_first_frame),
                    "time_left_to_hit_s": _stats(time_left)},
        "inference_ms": {"n": len(inference), "median": float(np.median(inference)) if inference else None,
                         "p95": float(np.percentile(inference, 95)) if inference else None},
        "action_counts": {name: int(counts.get(name, 0)) for name in action_names},
    }


def closed_loop_metrics(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], split: str,
                        action_names: Sequence[str]) -> dict[str, Any]:
    """Every metric of the evaluation from the per-scenario records, overall and by scenario kind."""
    from learned_judgment.engine.streams import stream_rng

    level = _level(config)
    resamples = _count("evaluation.bootstrap_resamples", require(config, "evaluation.bootstrap_resamples"))
    duration_s = float(require(config, "skills.dodge_vision.episode_duration_s"))
    t_first_s = float(frame_times(config)[0])
    seed = int(require(config, "runtime.seed"))
    stream = f"{SKILL}/closed_loop/{split}/paired_bootstrap"
    names = list(action_names)

    def summary(subset: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
        d = [r["outcomes"]["snn"]["collision"] - r["outcomes"]["no_op"]["collision"] for r in subset]
        return {
            "n_scenarios": len(subset),
            "n_threats": sum(1 for r in subset if r["threat"]),
            "methods": {m: _method_metrics(subset, m, names, duration_s, t_first_s, level) for m in METHODS},
            "paired_collision_rate": {"a": "snn", "b": "no_op", **paired_bootstrap(d, resamples, level, stream_rng(seed, stream, index)),
                                      "method": "percentile bootstrap over scenarios", "resamples": resamples,
                                      "level": level, "stream": f"{stream}[{index}]"},
        }

    out = summary(records, 0)
    out["by_kind"] = {kind: summary([r for r in records if r["kind"] == kind], 1 + i)
                      for i, kind in enumerate(KINDS) if any(r["kind"] == kind for r in records)}
    return out


def _ui_metrics(metrics: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The result.json UI keys "metrics" and "paired" from closed_loop_metrics."""
    methods = metrics["methods"]

    def value(method: str, key: str, ci: str | None, n: int) -> dict[str, Any]:
        m = methods[method]
        low, high = (None, None) if ci is None else m[ci]
        return {"value": m[key], "ci_low": low, "ci_high": high, "n": n}

    rows = [
        ("collision_rate", "fraction", False, "collision_rate", "collision_ci", "n_scenarios"),
        ("fall_rate", "fraction", False, "fall_rate", "fall_ci", "n_scenarios"),
        ("mean_min_separation_m", "m", True, "mean_min_separation_m", None, "n_scenarios"),
        ("active_precision", "fraction", True, "precision", "precision_ci", "issued"),
        ("recall", "fraction", True, "recall", "recall_ci", "n_threats"),
        ("false_activations_per_safe_minute", "1/min", False, "false_activations_per_safe_minute", None, None),
    ]
    ui = []
    for name, unit, higher, key, ci, n_key in rows:
        values = {}
        for method in METHODS:
            n = methods[method][n_key] if n_key else metrics["n_scenarios"] - metrics["n_threats"]
            values[method] = value(method, key, ci, int(n))
        ui.append({"name": name, "unit": unit, "higher_is_better": higher, "values": values})
    p = metrics["paired_collision_rate"]
    paired = [{"name": "collision_rate", "a": "snn", "b": "no_op", "diff": p["diff"], "ci_low": p["ci_low"],
               "ci_high": p["ci_high"], "n": p["n"]}]
    return ui, paired


def evaluate(config: Mapping[str, Any], run_id: str, student_dir: str | Path, split: str = "test",
             n: int | None = None, argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Closed-loop evaluation of the frozen student on ``split``; returns the metrics written to result.json.

    Raises _Refused when student.json holds no frozen thresholds; whatever the episodes raise.
    """
    from learned_judgment.observed import EventLog, _run_dir

    if split not in ("dev", "test"):
        raise ValueError(f"split must be dev or test; got {split!r}")
    student_dir = Path(student_dir).resolve()
    student = _read_student(student_dir)
    if student.get("thresholds") is None:
        raise _Refused(f"{student_dir / 'student.json'} has no frozen thresholds: run calibrate first")
    runtime = DodgeRuntime.load(student_dir, config)  # fails before any episode
    n = _scenario_count(config, split, n)
    run_dir = _run_dir(config, run_id)
    section = f"closed_loop_{split}"
    events = EventLog(run_dir / "events.jsonl", run_id, SKILL)
    tasks = _plan(config, split, n)
    kinds = {s["scenario_id"]: s["kind"] for s, _ in tasks}
    common = {"argv": list(argv or []), "student": student_dir, "network_sha256": student.get("network", {}).get("sha256"),
              "thresholds": runtime.thresholds, "calibration": student.get("calibration"), "frames": runtime.frames}
    _merge_result(run_dir, run_id, {section: {"state": "running", **common}})
    events.emit("evaluate", "running", done=0, total=n, data={"split": split, "student": student_dir})
    print(f"evaluate: {split}, {n} scenarios x (NO_OP, snn) closed loop, thresholds {runtime.thresholds}", flush=True)
    try:
        started = time.perf_counter()
        results = _run(config, student_dir, None, _evaluate_task, tasks,
                       on_result=lambda done: events.emit("evaluate", "running", done=done, total=n))
        wall_s = time.perf_counter() - started
        records = [
            {"scenario_id": r["scenario_id"], "kind": kinds[r["scenario_id"]],
             "threat": r["outcomes"]["no_op"]["collision"] == 1.0, "t_hit": r["outcomes"]["no_op"]["first_collision_time_s"],
             **{key: r[key] for key in ("outcomes", "issues", "cooldown_conflicts", "inference_ms", "n_decisions")}}
            for r in results
        ]
        seeds = _seeds(config, split)
        records_path = run_dir / f"{section}.json"
        _write_json(records_path, {"skill": SKILL, "split": split, "seeds": seeds, "scenario_ids": [r["scenario_id"] for r in records],
                                   "config_digest": config_digest(config), **common, "records": records})
        metrics = closed_loop_metrics(records, config, split, runtime.action_names)
        ui, paired = _ui_metrics(metrics)
        seeds["bootstrap"] = metrics["paired_collision_rate"]["stream"]
        _merge_result(run_dir, run_id, {
            section: {"state": "done", **common, "records": records_path, "wall_s": wall_s,
                      "config_digest": config_digest(config), "seeds": seeds, "metrics": metrics,
                      "where": "simulation (MuJoCo Go2World); student inference on this computer's CPU (numpy)"},
            "seeds": seeds, "scenario_ids": [r["scenario_id"] for r in records], "metrics": ui, "paired": paired,
        })
    except Exception as error:
        events.emit("evaluate", "failed", data={"type": type(error).__name__, "message": str(error)})
        _merge_result(run_dir, run_id, {section: {"state": "failed", **common,
                                                  "error": {"type": type(error).__name__, "message": str(error)}}})
        raise
    snn, base = metrics["methods"]["snn"], metrics["methods"]["no_op"]
    headline = {
        "split": split, "n_scenarios": n, "n_threats": metrics["n_threats"],
        "collision_rate": {"no_op": base["collision_rate"], "snn": snn["collision_rate"]},
        "paired_collision_diff": paired[0], "active_precision": snn["precision"], "recall": snn["recall"],
        "false_activations_per_safe_minute": snn["false_activations_per_safe_minute"], "records": records_path,
    }
    events.emit("evaluate", "done", done=n, total=n, data=headline)
    print(f"evaluate: {json.dumps(headline, default=str)}", flush=True)
    return metrics


# ----- runtime evidence ------------------------------------------------------------------------------------------


def _preprocess_brute_force(rgb: np.ndarray, settings: VisionSettings) -> np.ndarray:
    """vision.preprocess_frame's contract by loops over output cells and pixels (float64)."""
    n = settings.frame_size
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    w0, w1, w2 = settings.luma_weights
    out = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        r0, r1 = (i * side) // n, ((i + 1) * side) // n
        for j in range(n):
            c0, c1 = (j * side) // n, ((j + 1) * side) // n
            total = 0.0
            for y in range(top + r0, top + r1):
                for x in range(left + c0, left + c1):
                    total += w0 * float(rgb[y, x, 0]) + w1 * float(rgb[y, x, 1]) + w2 * float(rgb[y, x, 2])
            out[i, j] = total / ((r1 - r0) * (c1 - c0)) / settings.pixel_scale
    return out


def latency_and_parity(student_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """runtime.measure_latency of the student and a preprocessing-parity check on one real head-camera frame.

    Latency: evaluation.latency_decisions random windows, seed runtime.seed; the frozen thresholds (or, before the
    freeze, the first calibration-grid triple, recorded). Parity: the first dev scenario with a ball, reset with
    attempt_seed = scenario_id, the HeadView frame at the launch, vision.preprocess_frame against a loop
    implementation of the same contract: max |difference|.
    """
    import mujoco

    from learned_judgment.env import Go2World

    student_dir = Path(student_dir).resolve()
    student = _read_student(student_dir)
    frozen = student.get("thresholds") is not None
    runtime = DodgeRuntime.load(student_dir, config, None if frozen else _grid(config)[0])
    latency = measure_latency(runtime, n=_count("evaluation.latency_decisions", require(config, "evaluation.latency_decisions")),
                              seed=int(require(config, "runtime.seed")))

    world_config = copy.deepcopy(dict(config))
    world_config["world"]["save_traces"] = False
    world = Go2World(world_config, require(config, "observed.world_seed"), require(config, "observed.world_preset"))
    scenarios, _ = _split_plan(config, "dev", _scenario_count(config, "dev", None))
    scenario = next((s for s in scenarios if s["kind"] != "empty"), scenarios[0])
    world.reset(dict(scenario["situation"]), attempt_seed=int(scenario["scenario_id"]))
    view = HeadView(world_config)
    try:
        rgb = view.frame(world)
    finally:
        view.close()
    settings = VisionSettings.from_config(config)
    fast = preprocess_frame(rgb, settings)
    reference = _preprocess_brute_force(rgb, settings)
    return {
        "student": student_dir,
        "network_sha256": student.get("network", {}).get("sha256"),
        "thresholds_used": runtime.thresholds,
        "thresholds_source": "student.json (frozen)" if frozen else "first calibration-grid triple (student not frozen)",
        "latency": latency,
        "latency_method": "learned_judgment.runtime.measure_latency: readout + gate per random valid window",
        "preprocess_parity": {
            "split": "dev", "scenario_id": scenario["scenario_id"], "kind": scenario["kind"], "t_s": 0.0,
            "image_shape": list(rgb.shape), "frame_size": settings.frame_size,
            "max_abs_difference": float(np.max(np.abs(fast.astype(np.float64) - reference))),
            "max_abs_difference_float32": float(np.max(np.abs(fast - reference.astype(np.float32)))),
            "reference": "loops over output cells and crop pixels (dodge_eval._preprocess_brute_force), float64",
        },
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__},
        "where": "this computer's CPU (numpy); no network",
    }


# ----- command line ----------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.dodge_eval", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--run", required=True, help="run id: the directory under observed.runs_dir")
        sub.add_argument("--student", required=True, help="exported student directory (student.json, network.*)")

    cal = commands.add_parser("calibrate", help="choose and freeze the gate thresholds on dev trajectories")
    common(cal)
    cal.add_argument("--n", type=int, help="dev scenarios (default observed.dodge_vision.dev_scenarios)")
    cal.add_argument("--refreeze", action="store_true", help="replace thresholds already frozen in student.json")
    cal.add_argument("--trace", help="trace_dev.npz written by the trace command: calibrate offline on it (no episode)")
    cal.add_argument("--no-freeze", action="store_true", help="with --trace: report only, student.json is not written")
    tr = commands.add_parser("trace", help="run the NO_OP episodes of a split once and save their raw frames")
    tr.add_argument("--run", required=True, help="run id: the directory under observed.runs_dir")
    tr.add_argument("--split", default="dev", choices=("dev", "test"))
    tr.add_argument("--n", type=int, help="scenarios (default observed.dodge_vision.<split>_scenarios)")
    ev = commands.add_parser("evaluate", help="closed-loop NO_OP vs frozen student on a split")
    common(ev)
    ev.add_argument("--split", default="test", choices=("dev", "test"))
    ev.add_argument("--n", type=int, help="scenarios (default observed.dodge_vision.<split>_scenarios)")
    evidence = commands.add_parser("evidence", help="runtime latency and preprocessing parity")
    common(evidence)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (see the module docstring). Returns the exit code."""
    args = _parser().parse_args(argv)
    words = list(sys.argv[1:] if argv is None else argv)
    config = load_config()
    try:
        if args.command == "calibrate" and args.trace is not None:
            if args.n is not None:
                raise SystemExit("--n does not apply with --trace: the trace holds its scenarios")
            report = calibrate_from_trace(config, args.run, args.student, args.trace, freeze=not args.no_freeze,
                                          refreeze=args.refreeze, argv=words)
            return 0 if report["gate_passed"] else EXIT_GATE_FAILED
        if args.command == "calibrate":
            if args.no_freeze:
                raise SystemExit("--no-freeze applies with --trace only")
            calibration = calibrate(config, args.run, args.student, n=args.n, refreeze=args.refreeze, argv=words)
            return 0 if calibration["gate_passed"] else EXIT_GATE_FAILED
        if args.command == "trace":
            trace(config, args.run, split=args.split, n=args.n, argv=words)
            return 0
        if args.command == "evaluate":
            evaluate(config, args.run, args.student, split=args.split, n=args.n, argv=words)
            return 0
        from learned_judgment.observed import _run_dir

        evidence = latency_and_parity(args.student, config)
        path = _merge_result(_run_dir(config, args.run), args.run, {"runtime_evidence": evidence})
        print(f"evidence: latency {evidence['latency']}, preprocess max |diff| "
              f"{evidence['preprocess_parity']['max_abs_difference']}; {path}", flush=True)
        return 0
    except _Refused as refusal:
        print(f"refused: {refusal}", flush=True)
        return EXIT_REFUSED


if __name__ == "__main__":
    # Through the importable module, so spawned workers find _worker_init and the tasks by their module name.
    from learned_judgment import dodge_eval as _module

    raise SystemExit(_module.main())
