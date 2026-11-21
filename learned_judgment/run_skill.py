"""One call for a declared skill's learning run (Person B): plan it, quote it, start it in the background, read its
status, stop it.

Lane contract (Person B, run_skill)
-----------------------------------
A chat declaration (learned_judgment.declared) names one declared skill and its parameters. This module turns a valid
declaration into the fixed command chain of that skill's SkillSpec and runs the chain as ONE detached background
process, so a web server can start a run and return at once. Progress is read from
<observed.runs_dir>/<RUN>/events.jsonl and the result from result.json; both are written by the commands themselves
(observed.EventLog and the result.json merge helpers). Nothing here measures, scores or trains: every number in a run
comes from the commands.

    plan_run    resolve_declaration (DeclarationError with .problems when invalid), then the chain: steps
                {"name", "argv"}, each a ``python -m`` line run with the project interpreter (sys.executable) from
                the repository root. No side effect.
    quote_run   validity, the simulator episodes and an estimate of the TabPFN tokens from the configured sizes.
                No network call and nothing billed.
    start_run   makes <runs_dir>/<RUN>/, writes config.yaml (the config the run uses), declaration.json and
                run_status.json, starts ``python -m learned_judgment.run_skill _worker`` detached (stdout and stderr
                to run.log; its pid in worker.json) and returns at once.
    run_status  run_status.json plus {"alive"} (the worker pid still runs; best effort).
    stop_run    terminates the worker's process tree (Windows: taskkill /T /F) and writes state "stopped".

Chains (R = <observed.runs_dir>/<RUN>, M = learned_judgment.skills.<spec>, O = learned_judgment.observed)::

    dodge_vision  rows_context  M rows --split context --out R/context
                  rows_dev      M rows --split dev --out R/dev
                  dream         M dream --out R/dream
                  teach         O teach --run RUN --skill dodge_vision --context R/context --dream R/dream --dev R/dev
                                [--confirm-billing]
                  train         O train --run RUN --skill dodge_vision --dream R/dream --scores R/scores.npz
                  calibrate     learned_judgment.dodge_eval calibrate --run RUN --student R/student
                  evaluate      learned_judgment.dodge_eval evaluate --run RUN --student R/student --split test
                  evidence      learned_judgment.dodge_eval evidence --run RUN --student R/student
    ramp          rows_context, rows_dev, dream, teach (with --dev), train as above with M = skills.ramp, then
                  run_test      M run --student R/student --split test --out R/test_run
                                (no --run: ramp's run takes RUN from the student's provenance.run_id, the same id)
    move          as ramp with M = skills.move, run_test with --run RUN.

Mode "fast" (move and ramp only; MISSION.md's "about 10-40 attempts per skill"; S = the spec)::

    attempts   M attempts --out R/context                      (observed.S.fast_attempts episodes, ONE per scenario)
    dream      M dream --out R/dream --n observed.S.fast_dream_situations
    teach      O teach --run RUN --skill S --context R/context --dream R/dream [--confirm-billing]   (no --dev)
    train      O train --run RUN --skill S --dream R/dream --scores R/scores.npz --steps observed.S.fast_snn_max_steps
    run_test   M run --student R/student --split test --out R/test_run --run RUN

dodge_vision has no fast mode (ValueError): its student learns from camera pixels and needs thousands of rendered
windows. The full chain reads its sizes from the config (no --n is passed).

Timing. The worker records each step's wall-clock seconds ("wall_s") in run_status.json history and, when the chain
ends, merges into result.json a top-level "timing": {"mode", "learn_s" (attempts or rows_context + dream + teach +
train), "evaluate_s" (every other step), "learn_steps", "steps" {name: s}, "attempts" (rows.json n_attempts of the
context; None for the full chain), "machine_note", "cpu_count", "clock"}.

Declared parameters are RECORDED ONLY (declaration.json "declaration" and "resolved", plan "recorded_only"): no command
of any chain has an option for them, so the run trains on the configured distributions, not on the declared value:

    DODGE threats             the run uses the configured skills.dodge_vision.object_kinds
    RAMP incline_deg          the run trains over the configured incline band skills.ramp.scenarios.incline_deg
    RAMP nominal_speed_m_s    the run uses skills.ramp.scenarios.nominal_speed_m_s
    MOVE forward_m, sideways_m, turn_deg
                              the run trains over skills.move.targets with skills.move.kind_weights

Worker. Runs the steps in order. Each step runs as ``python -m learned_judgment.run_skill _exec --config
R/config.yaml MODULE ARGS...``, which points load_config() at the run's config snapshot and calls MODULE.main(ARGS):
the commands have no --config option, and the snapshot keeps a run on the config it started with. The worker stops at
the first non-zero exit code and rewrites R/run_status.json at every step start and end: {"run_id", "skill", "spec",
"state" (running | done | failed | refused | stopped), "mode", "step", "returncode", "steps_done", "updated_utc",
"history" (one entry per finished step: step, returncode, started_utc, finished_utc, wall_s)} plus "reason" for a known exit code. A
teach step exiting with observed.EXIT_REFUSED (no --confirm-billing, or the teacher's budget gate refused) ends the run
"refused"; any other non-zero exit ends it "failed" (dodge_eval calibrate exits 4 when no threshold triple reached
inference.gate.precision_target). The commands emit their own events; the worker adds one EventLog line per step start
(state "running") and one at the very end (the final state), with the step names the UI knows: rows_context and
attempts -> attempts, rows_dev -> baseline, dream -> dream, teach -> teacher_fit, train -> train, every later step -> evaluate.

Billing. Only teach with --confirm-billing sends billable TabPFN requests, and each one passes the teacher's own gate
(usage read and exact quote first). start_run adds --confirm-billing only when confirm_billing is True.

Command line (repository root)::

    python -m learned_judgment.run_skill [--config PATH] plan  (--declaration JSON | --declaration-file PATH) [--run RUN]
        [--mode {full,fast}]
    python -m learned_judgment.run_skill [--config PATH] quote (--declaration JSON | --declaration-file PATH)
        [--mode {full,fast}]
    python -m learned_judgment.run_skill [--config PATH] start (--declaration JSON | --declaration-file PATH)
        [--run RUN] [--confirm-billing] [--mode {full,fast}]
    python -m learned_judgment.run_skill [--config PATH] status --run RUN
    python -m learned_judgment.run_skill [--config PATH] stop --run RUN

Exit codes: 0 done, 1 failed (traceback), 2 invalid declaration (the problems are printed). Internal: ``_worker --run
RUN --runs-dir DIR`` and ``_exec --config PATH MODULE ARGS...``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from learned_judgment.config import CONFIG_PATH, ConfigError, config_digest, load_config, optional, require
from learned_judgment.declared import DeclarationError, check_declaration, resolve_declaration
from learned_judgment.fileio import atomic_write_text

__all__ = ["EXIT_INVALID", "MODES", "RUN_STATES", "main", "plan_run", "quote_run", "run_status", "start_run", "stop_run"]

ROOT = CONFIG_PATH.parent
RUN_STATES = ("running", "done", "failed", "refused", "stopped")
EXIT_INVALID = 2

_SELF = "learned_judgment.run_skill"
_OBSERVED = "learned_judgment.observed"
_DODGE_EVAL = "learned_judgment.dodge_eval"
_CHAINS = ("dodge_vision", "ramp", "move")
MODES = ("full", "fast")
_FAST_CHAINS = ("ramp", "move")
_FAST_REFUSAL = ("mode 'fast' is defined for move and ramp only: dodge_vision's student learns from camera pixels and "
                 "needs thousands of rendered windows (observed.dodge_vision.dream_episodes x "
                 "skills.dodge_vision.windows_per_dream_episode), which a 40-attempt run cannot give")
_EVENT_STEP = {"rows_context": "attempts", "attempts": "attempts", "rows_dev": "baseline", "dream": "dream",
               "teach": "teacher_fit", "train": "train"}  # every later step: "evaluate"
# timing.learn_s sums these steps; every other step (rows_dev, run_test, calibrate, evaluate, evidence) is evaluate_s.
_LEARN_STEPS = ("rows_context", "attempts", "dream", "teach", "train")
_EXIT_REASONS = {
    ("teach", 3): "refused: no --confirm-billing, or the teacher's budget gate refused (observed.EXIT_REFUSED)",
    ("calibrate", 3): "refused by dodge_eval (thresholds already frozen without --refreeze)",
    ("calibrate", 4): "no threshold triple reached inference.gate.precision_target (gate_passed false)",
    ("evaluate", 3): "refused by dodge_eval (no thresholds frozen in student.json)",
}
# Config keys each declared parameter does NOT override (what the run uses instead).
_RECORDED_ONLY = {
    ("dodge_vision", "threats"): ("skills.dodge_vision.object_kinds",),
    ("ramp", "incline_deg"): ("skills.ramp.scenarios.incline_deg",),
    ("ramp", "nominal_speed_m_s"): ("skills.ramp.scenarios.nominal_speed_m_s",),
    ("move", "forward_m"): ("skills.move.targets", "skills.move.kind_weights"),
    ("move", "sideways_m"): ("skills.move.targets", "skills.move.kind_weights"),
    ("move", "turn_deg"): ("skills.move.targets", "skills.move.kind_weights"),
}

# Token quote of one request: experiments/proofs/B2_research/teacher_limits.out.txt section 5 (least squares on 85
# free quotes, model v3.5, 8 estimators, max |diff| 1 token over 88 grid quotes), T = train + query rows as sent,
# F = columns as sent.
_QUOTE_SOURCE = "experiments/proofs/B2_research/teacher_limits.out.txt"
_QUOTE_FORMULA = "max(10000, round(5418.62 + 1.943981e-6*T^2 + 1.295988e-3*T*min(F,768))) per request"
_QUOTE_LABEL = ("estimated from configured sizes; the exact quote is taken and checked against the budget gate "
                "before the billed step")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quote_tokens(n_train: int, n_query: int, width: int) -> int:
    t = n_train + n_query
    return max(10000, round(5418.62 + 1.943981e-6 * t * t + 1.295988e-3 * t * min(width, 768)))


def _count(config: Mapping[str, Any], key: str) -> int:
    value = require(config, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"config.yaml key {key!r} must be an integer >= 1; got {value!r}")
    return value


def _run_dir(config: Mapping[str, Any], run_id: str) -> Path:
    from learned_judgment.observed import _run_dir as observed_run_dir  # the one rule for run ids and directories

    return observed_run_dir(config, run_id)


# --------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------


def _mode(mode: Any, spec: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}; got {mode!r}")
    if mode == "fast" and spec not in _FAST_CHAINS:
        raise ValueError(_FAST_REFUSAL)
    return mode


def _steps(
    spec: str, run_id: str, run_dir: Path, confirm_billing: bool, mode: str = "full",
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    def step(name: str, module: str, *args: Any) -> dict[str, Any]:
        return {"name": name, "argv": [sys.executable, "-m", module, *(str(arg) for arg in args)]}

    module = f"learned_judgment.skills.{spec}"
    common = ("--run", run_id, "--skill", spec)
    student = run_dir / "student"
    billing = ["--confirm-billing"] if confirm_billing else []
    if mode == "fast":
        root = f"observed.{spec}"
        return [
            step("attempts", module, "attempts", "--out", run_dir / "context"),
            step("dream", module, "dream", "--out", run_dir / "dream", "--n",
                 _count(config, f"{root}.fast_dream_situations")),
            step("teach", _OBSERVED, "teach", *common, "--context", run_dir / "context", "--dream", run_dir / "dream",
                 *billing),
            step("train", _OBSERVED, "train", *common, "--dream", run_dir / "dream", "--scores", run_dir / "scores.npz",
                 "--steps", _count(config, f"{root}.fast_snn_max_steps")),
            step("run_test", module, "run", "--student", student, "--split", "test", "--out", run_dir / "test_run",
                 "--run", run_id),
        ]
    teach = [*common, "--context", run_dir / "context", "--dream", run_dir / "dream", "--dev", run_dir / "dev"]
    steps = [
        step("rows_context", module, "rows", "--split", "context", "--out", run_dir / "context"),
        step("rows_dev", module, "rows", "--split", "dev", "--out", run_dir / "dev"),
        step("dream", module, "dream", "--out", run_dir / "dream"),
        step("teach", _OBSERVED, "teach", *teach, *billing),
        step("train", _OBSERVED, "train", *common, "--dream", run_dir / "dream", "--scores", run_dir / "scores.npz"),
    ]
    if spec == "dodge_vision":
        steps += [
            step("calibrate", _DODGE_EVAL, "calibrate", "--run", run_id, "--student", student),
            step("evaluate", _DODGE_EVAL, "evaluate", "--run", run_id, "--student", student, "--split", "test"),
            step("evidence", _DODGE_EVAL, "evidence", "--run", run_id, "--student", student),
        ]
    else:
        run = ["run", "--student", student, "--split", "test", "--out", run_dir / "test_run"]
        steps.append(step("run_test", module, *run, *(["--run", run_id] if spec == "move" else [])))
    return steps


def _recorded_only(spec: str, params: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, str]:
    out = {}
    for name in params:
        keys = _RECORDED_ONLY.get((spec, name), ())
        used = "; ".join(f"{key} = {json.dumps(optional(config, key))}" for key in keys)
        out[name] = "recorded only: no command takes it" + (f"; the run uses {used}" if used else "")
    return out


def _plan(
    declaration: Any, config: Mapping[str, Any], run_id: str | None, confirm_billing: bool, mode: str = "full"
) -> tuple[dict, dict]:
    resolved = resolve_declaration(declaration, config)
    spec = resolved["spec"]
    if spec not in _CHAINS:
        raise ValueError(f"SkillSpec {spec!r} has no run chain here; chains: {list(_CHAINS)}")
    mode = _mode(mode, spec)
    if run_id is None:
        run_id = f"{spec}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    run_dir = _run_dir(config, run_id)
    plan = {
        "skill": resolved["skill"],
        "spec": spec,
        "mode": mode,
        "params": resolved["params"],
        "steps": _steps(spec, run_id, run_dir, confirm_billing, mode, config),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "recorded_only": _recorded_only(spec, resolved["params"], config),
    }
    return plan, resolved


def plan_run(
    declaration: Any, config: Mapping[str, Any], *, run_id: str | None = None, confirm_billing: bool = False,
    mode: str = "full",
) -> dict[str, Any]:
    """The command chain of a declaration: {"skill", "spec", "mode", "params", "steps": [{"name", "argv"}], "run_id",
    "run_dir", "recorded_only": {param: what the run uses instead}}. No side effect.

    run_id None: "<spec>-<UTC yyyymmddThhmmss>" now. confirm_billing adds --confirm-billing to teach. mode "full"
    (default) or "fast" (move and ramp only; see the module docstring).

    Raises:
        DeclarationError: the declaration is invalid (``.problems``).
        ValueError:       the spec has no chain here, run_id is not a valid run id, or mode is unknown or "fast" for
                          dodge_vision.
    """
    return _plan(declaration, config, run_id, confirm_billing is True, mode)[0]


# --------------------------------------------------------------------------------------
# Quote
# --------------------------------------------------------------------------------------


def _sizes(spec_name: str, spec: Any, config: Mapping[str, Any], mode: str = "full") -> dict[str, Any]:
    """Configured sizes of a chain: rows, dream situations, window width D and simulator episodes per step."""
    c = len(spec.action_names)
    root = f"observed.{spec_name}"
    if mode == "fast":
        attempts, test = _count(config, f"{root}.fast_attempts"), _count(config, f"{root}.test_scenarios")
        note = "attempts: ONE episode per context scenario; dream: no simulator; run_test: two methods per test scenario"
        if spec_name == "ramp":
            note += "; ramp context rows are an upper bound (a scenario without a detected slope entry gives no row)"
        return {"n_actions": c, "context_rows": attempts, "dev_rows": 0,
                "dream_situations": _count(config, f"{root}.fast_dream_situations"),
                "window_width": len(spec.feature_names),
                "episodes": {"attempts": attempts, "dream": 0, "run_test": 2 * test}, "episodes_note": note}
    context, dev, test = (_count(config, f"{root}.{split}_scenarios") for split in ("context", "dev", "test"))
    if spec_name == "dodge_vision":
        from learned_judgment.vision import VisionSettings

        dream_episodes = _count(config, f"{root}.dream_episodes")
        situations = dream_episodes * _count(config, "skills.dodge_vision.windows_per_dream_episode")
        width = VisionSettings.from_config(config).n_features
        episodes = {"rows_context": context * c, "rows_dev": dev * c, "dream": dream_episodes, "calibrate": dev,
                    "evaluate": 2 * test, "evidence": 0}
        note = ("rows: scenarios x C actions; dream: rendered zero-command episodes; calibrate: one NO_OP episode per "
                "dev scenario; evaluate: NO_OP and student per test scenario")
    else:
        situations = _count(config, f"{root}.dream_situations")
        width = len(spec.feature_names)
        episodes = {"rows_context": context * c, "rows_dev": dev * c, "dream": 0, "run_test": 2 * test}
        note = "rows: scenarios x C candidates; dream: no simulator; run_test: two methods per test scenario"
        if spec_name == "ramp":
            note += ("; ramp rows are an upper bound (the other C - 1 candidates run only where candidate 0 reached "
                     "a decision)")
    return {"n_actions": c, "context_rows": context * c, "dev_rows": dev * c, "dream_situations": situations,
            "window_width": width, "episodes": episodes, "episodes_note": note}


def _requests(purpose: str, n_train: int, n_query: int, width: int, k: int, settings: Any) -> list[dict[str, Any]]:
    """The requests one prediction sends, by the teacher's rules (observed.dream_request_plan) with every outcome
    active; a constant outcome would be left out and make the real requests smaller."""
    if settings.multi_outcome_mode == "stacked":
        several = k > 1
        groups = [(n_train * (k if several else 1), width + int(several), (k if several else 1) * n_query)]
    else:
        groups = [(n_train, width, n_query)] * k
    cap, levels = settings.max_query_rows_per_request, len(settings.quantile_levels)
    out = []
    for train_rows, columns, total in groups:
        for start in range(0, total, cap):
            rows = min(start + cap, total) - start
            rows += int(rows == levels)  # a chunk of exactly Q rows is padded with one row
            out.append({"purpose": purpose, "n_train": train_rows, "n_query": rows, "n_features": columns,
                        "tokens": _quote_tokens(train_rows, rows, columns)})
    return out


def quote_run(declaration: Any, config: Mapping[str, Any], *, mode: str = "full") -> dict[str, Any]:
    """Validity and cost of a run before any data exists. No network call; nothing billed.

    mode "fast" estimates from observed.<spec>.fast_attempts and fast_dream_situations: no dev rows, so only the dream
    prediction is billed. ValueError for an unknown mode or "fast" with dodge_vision.

    Returns {"ok", "problems" (check_declaration), "skill", "spec", "mode", "estimate"} with estimate None when not ok, else
    {"episodes": total simulator episodes from the config sizes, "episodes_by_step", "episodes_note",
    "tabpfn_tokens": {"basis", "value" (int, or None when the formula does not apply to the configured model),
    "formula", "source", "assumptions", "requests"}, "max_tokens_per_request", "project_token_budget", "usage",
    "usage_note"}.

    The token value sums the teach step's two billed predictions (dev rows, then every dream candidate), each chunked
    at tabpfn.max_query_rows_per_request, with every outcome assumed active. usage is None: observed has no stand-alone
    free usage read (quote_dream needs a fitted teacher); the teach step reads usage and the exact quote first.
    """
    problems = check_declaration(declaration, config)
    skill = declaration.get("skill") if isinstance(declaration, dict) else None
    if problems:
        return {"ok": False, "problems": problems, "skill": skill, "spec": None, "mode": mode, "estimate": None}
    from learned_judgment.observed import load_spec
    from learned_judgment.teacher import TeacherSettings

    resolved = resolve_declaration(declaration, config)
    spec_name = resolved["spec"]
    if spec_name not in _CHAINS:
        raise ValueError(f"SkillSpec {spec_name!r} has no run chain here; chains: {list(_CHAINS)}")
    mode = _mode(mode, spec_name)
    spec = load_spec(spec_name, config)
    settings = TeacherSettings.from_config(config)
    sizes = _sizes(spec_name, spec, config, mode)
    blocks = list(require(config, f"skills.{spec_name}.teacher_features"))
    k = len(spec.outcome_names)
    width = (sizes["window_width"] if "window" in blocks else 0) + len(spec.action_feature_names) + k
    requests = (_requests("dev", sizes["context_rows"], sizes["dev_rows"], width, k, settings)  # none at 0 dev rows
                + _requests("dream", sizes["context_rows"], sizes["dream_situations"] * sizes["n_actions"], width, k,
                            settings))
    applies = settings.n_estimators == 8 and settings.model_path.startswith("v3.5")
    basis = _QUOTE_LABEL if applies else (
        f"{_QUOTE_LABEL}; no value: the formula was fitted for model v3.5 with 8 estimators, the config has "
        f"{settings.model_path!r} with {settings.n_estimators}"
    )
    tokens = {
        "basis": basis,
        "value": int(sum(r["tokens"] for r in requests)) if applies else None,
        "formula": _QUOTE_FORMULA + ", T = train + query rows, F = columns as sent",
        "source": _QUOTE_SOURCE,
        "assumptions": (
            f"context rows {sizes['context_rows']}, dev rows {sizes['dev_rows']}, dream rows "
            f"{sizes['dream_situations']} x {sizes['n_actions']} candidates; {width} teacher columns ({blocks}); "
            f"mode {settings.multi_outcome_mode}; all {k} outcomes active"
        ),
        "requests": requests,
    }
    episodes = sizes["episodes"]
    return {
        "ok": True,
        "problems": [],
        "skill": resolved["skill"],
        "spec": spec_name,
        "mode": mode,
        "estimate": {
            "episodes": int(sum(episodes.values())),
            "episodes_by_step": episodes,
            "episodes_note": sizes["episodes_note"],
            "tabpfn_tokens": tokens,
            "max_tokens_per_request": settings.max_tokens_per_request,
            "project_token_budget": settings.project_token_budget,
            "usage": None,
            "usage_note": "not read here (no free stand-alone usage read in observed); teach reads it before billing",
        },
    }


# --------------------------------------------------------------------------------------
# Start, status, stop
# --------------------------------------------------------------------------------------


def _dumps(value: Any) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False, default=str) + "\n"


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_status(run_dir: Path, status: dict[str, Any], **changes: Any) -> dict[str, Any]:
    status.update(changes, updated_utc=_now())
    atomic_write_text(run_dir / "run_status.json", _dumps(status))
    return status


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def start_run(
    declaration: Any, config: Mapping[str, Any], *, confirm_billing: bool, run_id: str | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    """Validate, make the run directory, write its files and start the worker detached; returns at once.

    mode "full" (default) or "fast" (move and ramp only: attempts, dream, teach without dev, train, run_test).

    Returns {"run_id", "mode", "pid" (the worker process), "runs_dir" (observed.runs_dir), "run_dir", "events",
    "result", "status", "log"} (paths as str).

    Raises:
        DeclarationError: the declaration is invalid.
        ValueError:       run_id invalid, or its directory exists already; the spec has no chain; mode unknown or
                          "fast" for dodge_vision.
        TypeError:        confirm_billing is not a bool.
    """
    if not isinstance(confirm_billing, bool):
        raise TypeError(f"confirm_billing must be a bool; got {confirm_billing!r}")
    plan, resolved = _plan(declaration, config, run_id, confirm_billing, mode)
    run_id, run_dir = plan["run_id"], Path(plan["run_dir"])
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError(f"run {run_id!r} exists already: {run_dir}") from None
    snapshot = run_dir / "config.yaml"
    atomic_write_text(snapshot, yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True))
    digest = config_digest(config)
    if config_digest(load_config(snapshot)) != digest:
        raise RuntimeError(f"{snapshot} does not reload to the given config")
    started = _now()
    atomic_write_text(run_dir / "declaration.json", _dumps({
        "declaration": declaration,
        "resolved": resolved,
        "confirm_billing": confirm_billing,
        "mode": plan["mode"],
        "started_utc": started,
        "steps": plan["steps"],
        "recorded_only": plan["recorded_only"],
        "config": str(snapshot),
        "config_digest": digest,
    }))
    status = _write_status(run_dir, {
        "run_id": run_id, "skill": plan["skill"], "spec": plan["spec"], "mode": plan["mode"], "state": "running",
        "step": None, "returncode": None, "steps_done": [], "history": [],
    })
    argv = [sys.executable, "-m", _SELF, "_worker", "--run", run_id, "--runs-dir", str(run_dir.parent)]
    if os.name == "nt":
        detach: dict[str, Any] = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    else:
        detach = {"start_new_session": True}
    log_path = run_dir / "run.log"
    try:
        with open(log_path, "ab") as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env=_child_env(), close_fds=True, **detach)
    except BaseException as error:
        _write_status(run_dir, status, state="failed", reason=f"the worker did not start: {error!r}")
        raise
    atomic_write_text(run_dir / "worker.json", _dumps({"pid": process.pid, "argv": argv, "started_utc": started}))
    return {
        "run_id": run_id,
        "mode": plan["mode"],
        "pid": process.pid,
        "runs_dir": str(run_dir.parent),
        "run_dir": str(run_dir),
        "events": str(run_dir / "events.jsonl"),
        "result": str(run_dir / "result.json"),
        "status": str(run_dir / "run_status.json"),
        "log": str(log_path),
    }


def _alive(pid: Any) -> bool:
    """Whether process ``pid`` still runs (best effort; a reused pid reads as alive)."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:  # our own child that has ended: reaped now
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _existing(config: Mapping[str, Any], run_id: str) -> Path:
    run_dir = _run_dir(config, run_id)
    if not (run_dir / "run_status.json").exists():
        raise ValueError(f"no run {run_id!r}: {run_dir / 'run_status.json'} does not exist")
    return run_dir


def run_status(run_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """run_status.json of a run plus {"pid", "alive"}. ValueError when the run does not exist."""
    run_dir = _existing(config, run_id)
    status = _read_json(run_dir / "run_status.json")
    pid = (_read_json(run_dir / "worker.json") or {}).get("pid")
    return {**status, "pid": pid, "alive": _alive(pid)}


def stop_run(run_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Terminate the worker's process tree (best effort) and write state "stopped"; returns run_status.

    A run whose worker no longer runs and whose state is final is left as it is.
    """
    from learned_judgment.observed import EventLog

    run_dir = _existing(config, run_id)
    status = _read_json(run_dir / "run_status.json")
    pid = (_read_json(run_dir / "worker.json") or {}).get("pid")
    if not _alive(pid) and status.get("state") in ("done", "failed", "refused", "stopped"):
        return run_status(run_id, config)
    if pid is not None:
        if os.name == "nt":
            killed = subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, text=True)
            output = (killed.stdout + killed.stderr).strip()
        else:
            import signal

            try:
                os.killpg(pid, signal.SIGTERM)
                output = f"SIGTERM sent to process group {pid}"
            except ProcessLookupError as error:
                output = repr(error)
        status["stop_output"] = output
    _write_status(run_dir, status, state="stopped")
    EventLog(run_dir / "events.jsonl", run_id, status["spec"]).emit(
        _EVENT_STEP.get(status.get("step"), "evaluate"), "stopped",
        data={"source": "run_skill", "command": status.get("step"), "reason": "stop_run"},
    )
    return run_status(run_id, config)


# --------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------


def _exec_argv(argv: Sequence[str], config_path: Path) -> list[str]:
    if len(argv) < 3 or argv[1] != "-m":
        raise ValueError(f"a step must be a python -m command line; got {list(argv)}")
    return [sys.executable, "-m", _SELF, "_exec", "--config", str(config_path), argv[2], *argv[3:]]


def _write_timing(run_dir: Path, run_id: str, spec: str, mode: str, history: Sequence[Mapping[str, Any]]) -> None:
    """Merge the top-level "timing" of the finished steps into <run_dir>/result.json (every other key is kept)."""
    steps = {entry["step"]: entry["wall_s"] for entry in history}
    context = _read_json(run_dir / "context" / "rows.json") or {}
    timing = {
        "mode": mode,
        "learn_s": round(sum(s for name, s in steps.items() if name in _LEARN_STEPS), 3),
        "evaluate_s": round(sum(s for name, s in steps.items() if name not in _LEARN_STEPS), 3),
        "learn_steps": [name for name in steps if name in _LEARN_STEPS],
        "steps": steps,
        "attempts": context.get("n_attempts"),
        "context_rows_note": None if "n_attempts" in context else "full chain: every candidate of every context scenario",
        "machine_note": "one laptop, CPU only",
        "cpu_count": os.cpu_count(),
        "clock": "wall clock of each step's process (interpreter start-up, imports and worker pools included)",
    }
    path = run_dir / "result.json"
    result = _read_json(path) or {"run_id": run_id, "skill": spec}
    result["timing"] = timing
    result["updated_utc"] = _now()
    atomic_write_text(path, _dumps(result))


def _worker(run_id: str, runs_dir: str) -> int:
    from learned_judgment.observed import EXIT_REFUSED, EventLog

    run_dir = Path(runs_dir) / run_id
    record = _read_json(run_dir / "declaration.json")
    steps = record["steps"]
    events = EventLog(run_dir / "events.jsonl", run_id, record["resolved"]["spec"])
    status = _read_json(run_dir / "run_status.json")
    state, last, code = "done", None, 0
    try:
        for index, step in enumerate(steps):
            last, started = step["name"], _now()
            events.emit(_EVENT_STEP.get(last, "evaluate"), "running",
                        data={"source": "run_skill", "command": last, "chain_index": index, "chain_length": len(steps)})
            _write_status(run_dir, status, state="running", step=last, returncode=None)
            print(f"[run_skill {started}] step {index + 1}/{len(steps)} {last}: {subprocess.list2cmdline(step['argv'])}",
                  flush=True)
            clock = time.perf_counter()
            # Explicit std handles: the step writes into run.log like the worker (no console handle to inherit).
            code = subprocess.call(_exec_argv(step["argv"], run_dir / "config.yaml"), cwd=ROOT, env=_child_env(),
                                   stdin=subprocess.DEVNULL, stdout=sys.stdout.fileno(), stderr=sys.stderr.fileno())
            wall_s = round(time.perf_counter() - clock, 3)
            print(f"[run_skill {_now()}] step {last} exit {code} after {wall_s} s", flush=True)
            status["history"].append({"step": last, "returncode": code, "started_utc": started, "finished_utc": _now(),
                                      "wall_s": wall_s})
            if code != 0:
                state = "refused" if last == "teach" and code == EXIT_REFUSED else "failed"
                if (last, code) in _EXIT_REASONS:
                    status["reason"] = _EXIT_REASONS[(last, code)]
                break
            status["steps_done"].append(last)
    except BaseException as error:
        state = "failed"
        status["reason"] = f"the worker failed: {error!r}"
        raise
    finally:
        try:
            _write_timing(run_dir, run_id, record["resolved"]["spec"], record.get("mode", "full"), status["history"])
        except Exception as error:  # the timing must never hide the run's own state
            status["timing_error"] = repr(error)
        _write_status(run_dir, status, state=state, step=last, returncode=code)
        events.emit(_EVENT_STEP.get(last, "evaluate"), state, data={
            "source": "run_skill", "run_state": state, "command": last, "returncode": code,
            "steps_done": status["steps_done"], "reason": status.get("reason"),
        })
    return 0 if state == "done" else (EXIT_REFUSED if state == "refused" else 1)


def _exec(config_path: str, module: str, args: Sequence[str]) -> int:
    """MODULE.main(args) with load_config() reading ``config_path`` (the run's snapshot) in this process."""
    if not module.startswith("learned_judgment."):
        raise ValueError(f"_exec runs learned_judgment modules only; got {module!r}")
    import learned_judgment.config as config_module

    config_module.load_config.__defaults__ = (Path(config_path),)
    code = importlib.import_module(module).main(list(args))
    return int(code or 0)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def _declaration(args: argparse.Namespace) -> Any:
    if args.declaration_file is not None:
        return json.loads(Path(args.declaration_file).read_text(encoding="utf-8"))
    return json.loads(args.declaration)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m {_SELF}", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="config.yaml to use (default: the repository's)")
    commands = parser.add_subparsers(dest="command", required=True)

    def declared(sub: argparse.ArgumentParser) -> None:
        group = sub.add_mutually_exclusive_group(required=True)
        group.add_argument("--declaration", help="the declaration as JSON text")
        group.add_argument("--declaration-file", help="a file holding the declaration as JSON")
        sub.add_argument("--mode", choices=MODES, default="full", help="full (default) or fast (move and ramp only)")

    plan = commands.add_parser("plan", help="the command chain of a declaration (no side effect)")
    declared(plan)
    plan.add_argument("--run", help="run id (default <spec>-<UTC time>)")
    quote = commands.add_parser("quote", help="episodes and estimated TabPFN tokens from the config sizes")
    declared(quote)
    start = commands.add_parser("start", help="start the run in the background and return")
    declared(start)
    start.add_argument("--run", help="run id (default <spec>-<UTC time>)")
    start.add_argument("--confirm-billing", action="store_true", help="allow the billed teach step")
    for name, text in (("status", "run_status.json and whether the worker runs"), ("stop", "stop the worker")):
        commands.add_parser(name, help=text).add_argument("--run", required=True)
    worker = commands.add_parser("_worker", help="internal: run the chain of a started run")
    worker.add_argument("--run", required=True)
    worker.add_argument("--runs-dir", required=True)
    execute = commands.add_parser("_exec", help="internal: MODULE.main(ARGS) with the given config")
    execute.add_argument("--config", dest="exec_config", required=True)
    execute.add_argument("module")
    execute.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point; returns the exit code."""
    args = _parser().parse_args(argv)
    if args.command == "_exec":
        return _exec(args.exec_config, args.module, args.args)
    if args.command == "_worker":
        return _worker(args.run, args.runs_dir)
    config = load_config(args.config) if args.config else load_config()
    try:
        if args.command == "plan":
            out = plan_run(_declaration(args), config, run_id=args.run, mode=args.mode)
        elif args.command == "quote":
            out = quote_run(_declaration(args), config, mode=args.mode)
        elif args.command == "start":
            out = start_run(_declaration(args), config, confirm_billing=args.confirm_billing, run_id=args.run,
                            mode=args.mode)
        elif args.command == "status":
            out = run_status(args.run, config)
        else:
            out = stop_run(args.run, config)
    except DeclarationError as error:
        print(_dumps({"ok": False, "problems": error.problems}), end="", flush=True)
        return EXIT_INVALID
    print(_dumps(out), end="", flush=True)
    return 0 if out.get("ok", True) else EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
