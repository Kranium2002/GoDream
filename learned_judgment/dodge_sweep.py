"""DODGE students trained side by side with different settings on one dream, each calibrated offline on ONE saved dev
trace, and one rule, fixed before any result, picks the winner (Person B, DODGE from the head camera).

Decision rule (fixed before any sweep result was seen; do not change it after looking at a table)
-------------------------------------------------------------------------------------------------
A variant QUALIFIES when its offline dev calibration (learned_judgment.dodge_eval.offline_calibration: the same gate
replay, inference.gate.calibration_grid, choice rule and inference.gate.precision_target as dodge_eval calibrate)
chose a thresholds triple whose dev precision is >= inference.gate.precision_target with at least
observed.dodge_vision.sweep.min_emitted issued commands. Among the qualifying variants the highest dev recall wins;
ties -> fewer frames, then the smaller hidden size (fewer hidden units in total, then fewer layers, then the sizes
compared in order), then the encoding listed first in observed.dodge_vision.sweep.encodings. No qualifying variant:
NO winner; the table is reported and the gate target was not met. Nothing is relaxed (precision target, min_emitted,
grid). A variant whose training or calibration fails (for example snn_train.TrainingError: a head not below the
constant predictor) is recorded as failed with its message and never qualifies.

Variants
--------
config.yaml observed.dodge_vision.sweep {frames, encodings, hidden_sizes, steps, jobs, min_emitted}: the full grid
frames x encodings x hidden_sizes, every variant trained with `steps` max optimizer steps. Variant <name> =
f<F>_<encoding>_h<H1>[-<H2>...], under <observed.runs_dir>/<RUN>/sweep/<name>/:

    config.yaml         config.yaml with vision.encoding = the variant's, observed.runs_dir = <runs_dir>/<RUN>/sweep
                        (so the variant is run <name> there) and the --set overrides
    train.log           python -m learned_judgment.dodge_sweep _exec --config <that> learned_judgment.observed train
                        --run <name> --skill dodge_vision --dream DIR --scores FILE --frames F --hidden H.. --steps N
    calibrate.log       ... _exec --config <that> learned_judgment.dodge_eval calibrate --run <name> --student
                        <name>/student --trace FILE --no-freeze
    student/, result.json (sections train, calibration_dev_offline), calibration_dev_offline.json/.npz, events.jsonl

Up to J variants run at once, each as its own subprocesses (training is single-threaded: students.snn.torch_threads;
the BLAS thread variables are set to 1 in the children). _exec is run_skill's helper: MODULE.main(ARGS) with
load_config() reading the given file.

Files under <runs_dir>/<RUN>/: sweep.json (every variant: settings, training summary with the validation loss per head
against the constant predictor, calibration summary, wall times; the rule; the ranking; the winner or null),
events.jsonl ("train" events: running/done/failed per variant, source dodge_sweep). With --freeze-best and a winner:
the winner's student directory copied to <RUN>/student (an existing one is first moved to
<RUN>/student_before_sweep_<UTC>) and frozen there (student.json thresholds + calibration block, as calibrate), and
a "sweep" section merged into <RUN>/result.json (also written, with winner null, when nothing qualified).

Command line (repository root)::

    python -m learned_judgment.dodge_sweep --run RUN --dream DIR --scores FILE --trace FILE [--jobs J]
        [--freeze-best] [--variants-json JSON] [--set KEY=VALUE ...]

--variants-json  a JSON object replacing some of frames, encodings, hidden_sizes, steps for this invocation only
                 (smaller mechanics checks); min_emitted and the rule cannot be overridden. Recorded in sweep.json.
--set            dotted config key = YAML value, applied to every variant's config copy (the key must exist).
                 Recorded in sweep.json.

Exit codes: 0 a winner, 4 no variant qualified, 1 the sweep itself failed (traceback).
"""

from __future__ import annotations

import argparse
import copy
import importlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from learned_judgment.config import CONFIG_PATH, ConfigError, config_digest, load_config, require
from learned_judgment.fileio import atomic_write_text

__all__ = ["EXIT_NO_WINNER", "RULE", "choose_winner", "main", "sweep", "sweep_settings", "variants"]

SKILL = "dodge_vision"
EXIT_NO_WINNER = 4
ROOT = CONFIG_PATH.parent
RULE = ("qualifies: offline dev calibration's chosen triple has precision >= inference.gate.precision_target and "
        "emitted >= observed.dodge_vision.sweep.min_emitted; among qualifying: highest dev recall, ties -> fewer frames, "
        "then smaller hidden size (total units, then layers, then sizes in order), then encoding order as listed; "
        "none qualifying: no winner, nothing relaxed")
_SWEEP_KEY = "observed.dodge_vision.sweep"
_OVERRIDABLE = ("frames", "encodings", "hidden_sizes", "steps")
_SELF = "learned_judgment.dodge_sweep"
_CALIBRATION_SECTION = "calibration_dev_offline"


# ----- settings --------------------------------------------------------------------------------------------------


def _int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be an int >= 1; got {value!r}")
    return value


def _list(name: str, value: Any) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ConfigError(f"{name} must be a non-empty list; got {value!r}")
    return list(value)


def sweep_settings(config: Mapping[str, Any], override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """observed.dodge_vision.sweep, checked, with ``override`` replacing some of frames/encodings/hidden_sizes/steps."""
    from learned_judgment.observed import WINDOW_ENCODINGS

    block = require(config, _SWEEP_KEY)
    if not isinstance(block, Mapping):
        raise ConfigError(f"config.yaml key {_SWEEP_KEY!r} must be a mapping; got {block!r}")
    settings = dict(block)
    if override:
        unknown = set(override) - set(_OVERRIDABLE)
        if unknown:
            raise ValueError(f"--variants-json may set only {_OVERRIDABLE}; got {sorted(unknown)}")
        settings.update(override)
    frames = [_int(f"{_SWEEP_KEY}.frames[]", f) for f in _list(f"{_SWEEP_KEY}.frames", settings.get("frames"))]
    variants_allowed = {require(config, "vision.frames"), *require(config, "vision.frame_variants")}
    if set(frames) - variants_allowed:
        raise ConfigError(f"{_SWEEP_KEY}.frames {frames} must be vision.frames or vision.frame_variants {sorted(variants_allowed)}")
    encodings = _list(f"{_SWEEP_KEY}.encodings", settings.get("encodings"))
    if any(e not in WINDOW_ENCODINGS for e in encodings) or len(set(encodings)) != len(encodings):
        raise ConfigError(f"{_SWEEP_KEY}.encodings must be distinct members of {WINDOW_ENCODINGS}; got {encodings}")
    hidden = [[_int(f"{_SWEEP_KEY}.hidden_sizes[][]", h) for h in _list(f"{_SWEEP_KEY}.hidden_sizes[]", sizes)]
              for sizes in _list(f"{_SWEEP_KEY}.hidden_sizes", settings.get("hidden_sizes"))]
    return {
        "frames": frames,
        "encodings": encodings,
        "hidden_sizes": hidden,
        "steps": _int(f"{_SWEEP_KEY}.steps", settings.get("steps")),
        "jobs": _int(f"{_SWEEP_KEY}.jobs", settings.get("jobs")),
        "min_emitted": _int(f"{_SWEEP_KEY}.min_emitted", settings.get("min_emitted")),
    }


def variants(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The full grid frames x encodings x hidden_sizes, each {"name", "frames", "encoding", "hidden_sizes", "steps"}."""
    out = []
    for frames, encoding, hidden in itertools.product(settings["frames"], settings["encodings"], settings["hidden_sizes"]):
        name = f"f{frames}_{encoding}_h{'-'.join(str(h) for h in hidden)}"
        out.append({"name": name, "frames": frames, "encoding": encoding, "hidden_sizes": list(hidden),
                    "steps": settings["steps"]})
    if len({v["name"] for v in out}) != len(out):
        raise ConfigError(f"{_SWEEP_KEY} gives repeated variants: {[v['name'] for v in out]}")
    return out


def _parse_sets(pairs: Sequence[str]) -> list[tuple[str, Any]]:
    out = []
    for pair in pairs:
        key, sep, text = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--set takes KEY=VALUE; got {pair!r}")
        out.append((key, yaml.safe_load(text)))
    return out


def _assign(config: dict[str, Any], dotted: str, value: Any) -> None:
    node: Any = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"--set {dotted}: config.yaml has no key {dotted!r}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ConfigError(f"--set {dotted}: config.yaml has no key {dotted!r}")
    node[parts[-1]] = value


def _variant_config(config: Mapping[str, Any], variant: Mapping[str, Any], sweep_dir: Path,
                    sets: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    out = copy.deepcopy(dict(config))
    out["vision"]["encoding"] = variant["encoding"]
    out["observed"]["runs_dir"] = sweep_dir.as_posix()
    for key, value in sets:
        _assign(out, key, value)
    return out


# ----- decision --------------------------------------------------------------------------------------------------


def _hidden_key(hidden: Sequence[int]) -> tuple:
    return sum(hidden), len(hidden), tuple(hidden)


def _qualifies(entry: Mapping[str, Any], target: float, min_emitted: int) -> bool:
    c = entry.get("calibration")
    return (entry.get("status") == "done" and c is not None and c.get("precision") is not None
            and c["precision"] >= target and c["emitted"] >= min_emitted)


def choose_winner(entries: Sequence[dict[str, Any]], target: float, min_emitted: int,
                  encodings: Sequence[str]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """(winner or None, entries ranked by the rule); sets "qualifies" and "rank" on every entry. Order: qualifying,
    then precision target met with too few commands, then precision target not met, then failed; within each the
    rule's own order (recall, frames, hidden size, encoding)."""
    for entry in entries:
        entry["qualifies"] = _qualifies(entry, target, min_emitted)

    def key(entry: Mapping[str, Any]) -> tuple:
        s, c = entry["settings"], entry.get("calibration") or {}
        if entry["qualifies"]:
            group = 0
        elif entry.get("status") == "done":
            group = 1 if c.get("precision") is not None and c["precision"] >= target else 2
        else:
            group = 3
        recall = c.get("recall")
        return (group, -(-1.0 if recall is None else recall), s["frames"], _hidden_key(s["hidden_sizes"]),
                list(encodings).index(s["encoding"]))

    ranked = sorted(entries, key=key)
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank
    return (ranked[0] if ranked and ranked[0]["qualifies"] else None), ranked


# ----- one variant -----------------------------------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"})
    return env


def _call(command: Sequence[str], log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + subprocess.list2cmdline(list(command)) + "\n")
        log.flush()
        return subprocess.call(list(command), cwd=ROOT, env=_child_env(), stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _log_tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _training_summary(section: Mapping[str, Any]) -> dict[str, Any]:
    """The train section of a variant's result.json: steps, losses per head against the constant predictor."""
    report = section.get("training") if isinstance(section.get("training"), Mapping) else section.get("report")
    report = report if isinstance(report, Mapping) else {}
    best = report.get("best_validation_loss_per_head") or {}
    constant = report.get("constant_predictor_validation_loss") or {}
    heads = {}
    for head in sorted(set(best) | set(constant)):
        b, c = best.get(head), constant.get(head)
        numbers = isinstance(b, (int, float)) and isinstance(c, (int, float))
        heads[head] = {"validation_loss": b, "constant_predictor": c, "below_constant": (b < c) if numbers else None}
    return {
        "state": section.get("state"),
        "steps_run": report.get("steps_run"),
        "best_step": report.get("best_step"),
        "stopped_early": report.get("stopped_early"),
        "initial_validation_loss": report.get("initial_validation_loss"),
        "best_validation_loss": report.get("best_validation_loss"),
        "validation_vs_constant": heads,
        "split": section.get("split"),
        "train_wall_s": section.get("wall_s"),
        "network_sha256": section.get("network_sha256"),
    }


def _calibration_summary(section: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("thresholds", "emitted", "correct", "precision", "precision_ci", "recalled", "recall", "false_activations",
            "gate_passed", "n_scenarios", "n_threats", "precision_target", "best_recall_at_precision", "wall_s")
    out = {key: section.get(key) for key in keys}
    check = section.get("readout_check") or {}
    out["readout_check"] = {k: check.get(k) for k in ("n_frames", "max_abs_difference", "tolerance", "passed")}
    return out


def _run_variant(variant: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    name, vdir = variant["name"], Path(context["sweep_dir"]) / variant["name"]
    vdir.mkdir(parents=True, exist_ok=True)
    config_path = vdir / "config.yaml"
    atomic_write_text(config_path, yaml.safe_dump(context["configs"][name], sort_keys=False, allow_unicode=True))
    entry: dict[str, Any] = {
        "name": name,
        "settings": {key: variant[key] for key in ("frames", "encoding", "hidden_sizes", "steps")},
        "dir": vdir, "config": config_path, "student": vdir / "student",
        "status": None, "failed_step": None, "error": None, "training": None, "calibration": None,
        "wall_s": {}, "logs": {"train": vdir / "train.log", "calibrate": vdir / "calibrate.log"},
    }
    context["emit"]("running", {"variant": name, "settings": entry["settings"]})
    exec_prefix = [sys.executable, "-m", _SELF, "_exec", "--config", str(config_path)]
    started = time.perf_counter()
    code = _call([*exec_prefix, "learned_judgment.observed", "train", "--run", name, "--skill", SKILL,
                  "--dream", str(context["dream"]), "--scores", str(context["scores"]), "--frames", str(variant["frames"]),
                  "--hidden", *[str(h) for h in variant["hidden_sizes"]], "--steps", str(variant["steps"])],
                 entry["logs"]["train"])
    entry["wall_s"]["train"] = time.perf_counter() - started
    result = _read_json(vdir / "result.json")
    entry["training"] = _training_summary(result.get("train", {}))
    if code != 0:
        error = result.get("train", {}).get("error") or {"type": f"exit {code}", "message": _log_tail(entry["logs"]["train"])}
        entry.update(status="failed", failed_step="train", error=error)
    else:
        clock = time.perf_counter()
        code = _call([*exec_prefix, "learned_judgment.dodge_eval", "calibrate", "--run", name, "--student",
                      str(entry["student"]), "--trace", str(context["trace"]), "--no-freeze"], entry["logs"]["calibrate"])
        entry["wall_s"]["calibrate"] = time.perf_counter() - clock
        section = _read_json(vdir / "result.json").get(_CALIBRATION_SECTION, {})
        if code in (0, 4) and section.get("state") == "done":  # 4: no triple reached the target, a valid result
            entry.update(status="done", calibration=_calibration_summary(section))
        else:
            error = section.get("error") or {"type": f"exit {code}", "message": _log_tail(entry["logs"]["calibrate"])}
            entry.update(status="failed", failed_step="calibrate", error=error)
    entry["wall_s"]["total"] = time.perf_counter() - started
    return entry


# ----- the sweep -------------------------------------------------------------------------------------------------


def _finite(value: Any) -> Any:
    """JSON values with non-finite floats as None (sweep.json is strict JSON)."""
    if isinstance(value, Mapping):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(ranked: Sequence[Mapping[str, Any]]) -> str:
    """The ranked variants as a fixed-width text table."""
    header = ("rank", "variant", "status", "Q", "thr val/const", "act val/const", "thresholds t/a/m", "gate", "emit",
              "prec", "Wilson", "recall", "FA", "r@.95", "r@.90", "train s", "cal s")
    lines = []
    for e in ranked:
        t, c = e.get("training") or {}, e.get("calibration") or {}
        heads = t.get("validation_vs_constant") or {}

        def head(name: str) -> str:
            h = heads.get(name) or {}
            return f"{_fmt(h.get('validation_loss'))}/{_fmt(h.get('constant_predictor'))}"

        th = c.get("thresholds") or {}
        ci = c.get("precision_ci") or [None, None]
        best = c.get("best_recall_at_precision") or {}
        lines.append((
            str(e.get("rank")), e["name"], e["status"] + ("" if e["status"] == "done" else f" ({e['failed_step']})"),
            "Q" if e.get("qualifies") else "", head("threat"), head("action"),
            "/".join(_fmt(th.get(k), 2) for k in ("threat", "action", "margin")) if th else "-",
            _fmt(c.get("gate_passed")), _fmt(c.get("emitted")),_fmt(c.get("precision")), f"{_fmt(ci[0], 2)}..{_fmt(ci[1], 2)}",
            _fmt(c.get("recall")), _fmt(c.get("false_activations")),
            _fmt((best.get("0.95") or {}).get("recall")), _fmt((best.get("0.90") or {}).get("recall")),
            _fmt(e["wall_s"].get("train"), 0), _fmt(e["wall_s"].get("calibrate"), 0),
        ))
    widths = [max(len(row[i]) for row in [header, *lines]) for i in range(len(header))]
    out = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in [header, *lines]]
    for e in ranked:
        if e["status"] != "done":
            out.append(f"  {e['name']} failed in {e['failed_step']}: {(e.get('error') or {}).get('type')}: "
                       f"{str((e.get('error') or {}).get('message'))[:300]}")
    return "\n".join(out)


def _freeze_best(config: Mapping[str, Any], run_dir: Path, run_id: str, winner: Mapping[str, Any], sweep_path: Path,
                 argv: Sequence[str]) -> dict[str, Any]:
    from learned_judgment.dodge_eval import _freeze, _merge_result
    from learned_judgment.observed import _utc_now
    from learned_judgment.runtime import DodgeRuntime

    now = _utc_now()
    section = _read_json(Path(winner["dir"]) / "result.json")[_CALIBRATION_SECTION]
    destination, backup = run_dir / "student", None
    if destination.exists():
        backup = run_dir / f"student_before_sweep_{now.strftime('%Y%m%dT%H%M%SZ')}"
        shutil.move(str(destination), str(backup))
    shutil.copytree(winner["student"], destination)
    calibration = {**section["calibration"], "frozen_utc": now.isoformat(),
                   "sweep": {"sweep_json": sweep_path, "variant": winner["name"], "copied_from": winner["student"],
                             "rule": RULE}}
    _freeze(destination, section["thresholds"], calibration, refreeze=False)
    runtime = DodgeRuntime.load(destination, config)  # the frozen copy loads: digest, contract, thresholds
    record = {
        "state": "done", "argv": list(argv), "sweep_json": sweep_path, "rule": RULE, "winner": winner["name"],
        "settings": winner["settings"], "student": destination, "replaced_student": backup,
        "thresholds": runtime.thresholds, "calibration": winner["calibration"], "training": winner["training"],
        "frozen_utc": now.isoformat(),
    }
    _merge_result(run_dir, run_id, {"sweep": _finite(json.loads(json.dumps(record, default=str)))})
    return record


def sweep(config: Mapping[str, Any], run_id: str, dream: str | Path, scores: str | Path, trace: str | Path,
          jobs: int | None = None, freeze_best: bool = False, override: Mapping[str, Any] | None = None,
          sets: Sequence[str] = (), argv: Sequence[str] = ()) -> dict[str, Any]:
    """Run the sweep (module docstring); returns the sweep.json record."""
    from learned_judgment.dodge_eval import _fraction, _merge_result
    from learned_judgment.observed import EventLog, _jsonable, _run_dir, _utc_now

    started, created = time.perf_counter(), _utc_now()
    settings = sweep_settings(config, override)
    jobs = settings["jobs"] if jobs is None else _int("--jobs", jobs)
    target = _fraction(config, "inference.gate.precision_target")
    pairs = _parse_sets(sets)
    run_dir = _run_dir(config, run_id)
    sweep_dir = run_dir / "sweep"
    dream, scores, trace = (Path(p).resolve() for p in (dream, scores, trace))
    for path in (dream / "dream.npz", scores, trace, trace.with_suffix(".json")):
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
    trace_meta = _read_json(trace.with_suffix(".json"))
    if trace_meta.get("split") != "dev":
        raise ValueError(f"{trace} is a {trace_meta.get('split')!r} trace; the sweep chooses on dev only")
    grid = variants(settings)
    if max(v["frames"] for v in grid) > int(trace_meta["window_frames"]):
        raise ValueError(f"the trace's windows hold {trace_meta['window_frames']} frames; the sweep asks for {settings['frames']}")
    configs = {v["name"]: _variant_config(config, v, sweep_dir, pairs) for v in grid}  # fails before any training
    events = EventLog(run_dir / "events.jsonl", run_id, SKILL)
    lock = threading.Lock()
    finished = [0]

    def emit(state: str, data: Mapping[str, Any]) -> None:
        with lock:
            events.emit("train", state, done=finished[0], total=len(grid), data={"source": "dodge_sweep", **data})

    context = {"sweep_dir": sweep_dir, "configs": configs, "dream": dream, "scores": scores, "trace": trace, "emit": emit}
    print(f"sweep {run_id}: {len(grid)} variants, {jobs} at once; dream {dream}, scores {scores}, trace {trace}", flush=True)
    entries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(jobs, len(grid))) as pool:
        futures = {pool.submit(_run_variant, v, context): v for v in grid}
        for future in as_completed(futures):
            entry = future.result()
            entries.append(entry)
            with lock:
                finished[0] += 1
            c = entry["calibration"] or {}
            emit("done" if entry["status"] == "done" else "failed", {
                "variant": entry["name"], "status": entry["status"], "failed_step": entry["failed_step"],
                "error": entry["error"], "precision": c.get("precision"), "recall": c.get("recall"),
                "emitted": c.get("emitted"), "wall_s": entry["wall_s"]})
            print(f"[{finished[0]}/{len(grid)}] {entry['name']}: {entry['status']}"
                  + (f" precision {c.get('precision')} recall {c.get('recall')} emitted {c.get('emitted')}" if c else
                     f" in {entry['failed_step']}: {(entry['error'] or {}).get('type')}")
                  + f" ({entry['wall_s']['total']:.0f} s)", flush=True)
    order = {v["name"]: i for i, v in enumerate(grid)}
    entries.sort(key=lambda e: order[e["name"]])
    winner, ranked = choose_winner(entries, target, settings["min_emitted"], settings["encodings"])
    sweep_path = run_dir / "sweep.json"
    record = {
        "run_id": run_id,
        "skill": SKILL,
        "argv": list(argv),
        "created_utc": created.isoformat(),
        "rule": RULE,
        "rule_settings": {"precision_target": target, "min_emitted": settings["min_emitted"],
                          "tie_breaks": ["fewer frames", "smaller hidden size", "encoding order"],
                          "encodings_order": settings["encodings"]},
        "settings": settings,
        "variants_override": dict(override or {}),
        "config_overrides": [{"key": k, "value": v} for k, v in pairs],
        "dream": dream, "scores": scores, "trace": trace, "trace_n_scenarios": trace_meta.get("n_scenarios"),
        "trace_n_threats": trace_meta.get("n_threats"), "jobs": jobs, "config_digest": config_digest(config),
        "variants": entries,
        "ranking": [e["name"] for e in ranked],
        "winner": None if winner is None else winner["name"],
        "result": (f"winner {winner['name']}" if winner is not None else
                   f"NO WINNER: no variant reached precision_target {target} with >= {settings['min_emitted']} issued "
                   "commands on dev; the gate target was not met and nothing was relaxed"),
        "wall_s": time.perf_counter() - started,
        "where": "simulation dev trace (MuJoCo Go2World, saved once); students trained and read out on this computer's CPU",
    }
    clean = _finite(_jsonable(record))
    atomic_write_text(sweep_path, json.dumps(clean, indent=1, allow_nan=False, ensure_ascii=False) + "\n")
    print(table(ranked), flush=True)
    print(f"rule: {RULE}", flush=True)
    print(f"{record['result']}; {sweep_path}; {record['wall_s']:.0f} s", flush=True)
    if freeze_best:
        if winner is None:
            _merge_result(run_dir, run_id, {"sweep": {"state": "no_winner", "argv": list(argv), "sweep_json": sweep_path,
                                                      "rule": RULE, "winner": None, "result": record["result"]}})
            print("--freeze-best: nothing frozen (no winner)", flush=True)
        else:
            frozen = _freeze_best(config, run_dir, run_id, winner, sweep_path, argv)
            print(f"--freeze-best: {winner['name']} -> {frozen['student']} thresholds {frozen['thresholds']}"
                  + (f"; the previous student moved to {frozen['replaced_student']}" if frozen["replaced_student"] else ""),
                  flush=True)
    with lock:
        events.emit("train", "done", done=len(grid), total=len(grid),
                    data={"source": "dodge_sweep", "sweep": sweep_path, "winner": record["winner"]})
    return clean


# ----- command line ----------------------------------------------------------------------------------------------


def _exec(config_path: str, module: str, args: Sequence[str]) -> int:
    """MODULE.main(args) with load_config() reading ``config_path`` in this process (run_skill's _exec)."""
    if not module.startswith("learned_judgment."):
        raise ValueError(f"_exec runs learned_judgment modules only; got {module!r}")
    import learned_judgment.config as config_module

    config_module.load_config.__defaults__ = (Path(config_path),)
    code = importlib.import_module(module).main(list(args))
    return int(code or 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m {_SELF}", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run id: the directory under observed.runs_dir")
    parser.add_argument("--dream", required=True, help="dream file directory (dream.npz)")
    parser.add_argument("--scores", required=True, help="scores.npz written by teach")
    parser.add_argument("--trace", required=True, help="trace_dev.npz written by dodge_eval trace")
    parser.add_argument("--jobs", type=int, help="variants at once (default observed.dodge_vision.sweep.jobs)")
    parser.add_argument("--freeze-best", action="store_true", help="copy the winner to <RUN>/student and freeze it")
    parser.add_argument("--variants-json", help="JSON object replacing frames/encodings/hidden_sizes/steps for this run")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                        help="config key (dotted) = YAML value in every variant's config copy")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (module docstring). Returns the exit code."""
    words = list(sys.argv[1:] if argv is None else argv)
    if words[:1] == ["_exec"]:
        if len(words) < 4 or words[1] != "--config":
            raise SystemExit(f"usage: python -m {_SELF} _exec --config PATH MODULE ARGS...")
        return _exec(words[2], words[3], words[4:])
    args = _parser().parse_args(words)
    override = None if args.variants_json is None else json.loads(args.variants_json)
    if override is not None and not isinstance(override, dict):
        raise SystemExit("--variants-json must be a JSON object")
    record = sweep(load_config(), args.run, args.dream, args.scores, args.trace, jobs=args.jobs,
                   freeze_best=args.freeze_best, override=override, sets=args.sets, argv=words)
    return 0 if record["winner"] is not None else EXIT_NO_WINNER


if __name__ == "__main__":
    raise SystemExit(main())
