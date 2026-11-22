"""What limits the punch dodge's reaction (Dev A, measurement only; experimental line wip-dev-a). Nothing here
changes an episode, the world or a student: it measures with dodge_eval's workers and closed loop.

    speed       A. How fast a dodge moves the body. Per active action of skills.dodge_vision.actions and attempt
                seed 0 .. N-1: the dodge scene's stand (dog_forward_speed_m_s 0, settle_s = skills.dodge_vision.settle_s,
                no object), the action issued at t_s 0 and held hold_s, then the zero command to
                episode_duration_s. Time from the issue to each distance of trunk travel along the commanded
                direction (sideways for the sidesteps, backwards for the retreat; the decision instant's heading
                frame) and of planar travel, at the 0.02 s control periods; the attempt's command latency (whole
                periods, nominal plus jitter) read from the world, so the table says how much is latency.
    eye         C. From config alone: for window lengths vision.frame_variants at the given camera rates, the first
                full window (F - 1) / rate after the first frame at the launch, and the earliest command when the
                gate (inference.gate.votes_needed) votes on every window from then on: (F - 1 + votes - 1) / rate.
    deadline    B and D, from the ceiling records of the same config (dodge_eval ceiling, <run>/ceiling_<split>.json,
                config digest must equal): B, the NO_OP first contact times of the kinds aimed at the dog
                (incoming_*), with their straight-line arrivals (dodge_vision.arrival_time_s); D, for every NO_OP
                hit avoidable at the first full window, its best action (the cleanly avoiding one with the largest
                separation, readout order on a tie) issued at later decision frames: a grid of every s-th decision
                frame from the first full window to the first frame at or after contact (that one fails without
                running: the hit has happened), searched by bisection; s is the smallest step whose worst-case
                episode count fits --budget, fixed from the contact times before any grid episode.
    compare     E. This config's ceiling beside another config's ceiling of the same scenarios (e.g. a punch of twice
                the radius): ceiling counts and falls of each, B of each, and by kind the scenarios whose NO_OP episode
                changed between no contact and contact.

Output: one JSON file (default artifacts/reaction/reaction.json), one section per command.

    python -m learned_judgment.reaction speed [--seeds 10] [--out PATH]
    python -m learned_judgment.reaction eye [--rates 15 30] [--out PATH]
    python -m learned_judgment.reaction deadline --run RUN [--split test] [--budget 500] [--rates 15 30] [--out PATH]
    python -m learned_judgment.reaction compare --run RUN --other RUN [--split test] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment import dodge_eval
from learned_judgment.config import config_digest, load_config, require
from learned_judgment.skills.dodge_vision import KINDS, arrival_time_s, frame_times

_OUT = Path("artifacts/reaction/reaction.json")
_DISTANCES_M = (0.05, 0.10, 0.20)
_AIMED = tuple(kind for kind in KINDS if kind.startswith("incoming_"))


# ----- pure parts --------------------------------------------------------------------------------------------------


def eye_times(frame_rates_hz: Sequence[float], window_frames: Sequence[int], votes_needed: int) -> list[dict[str, Any]]:
    """Per (rate, F): first full window (F - 1) / rate and earliest gated command (F - 1 + votes_needed - 1) / rate."""
    return [{"frame_rate_hz": float(rate), "frames": int(f), "first_full_window_s": (f - 1) / rate,
             "earliest_command_s": (f - 1 + votes_needed - 1) / rate}
            for rate in frame_rates_hz for f in window_frames]


def reach_times(t_s: Sequence[float], values: Sequence[float], distances: Sequence[float]) -> list[float | None]:
    """Per distance the first t_s whose value is at or beyond it, None if never."""
    out = []
    for d in distances:
        out.append(next((float(t) for t, v in zip(t_s, values) if v >= d), None))
    return out


def spread(values: Sequence[float]) -> dict[str, Any]:
    """Count, minimum, quartiles (numpy linear) and maximum; None for an empty list."""
    if not len(values):
        return {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None}
    q = np.quantile(np.asarray(values, dtype=np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"n": len(values), "min": float(q[0]), "q1": float(q[1]), "median": float(q[2]), "q3": float(q[3]), "max": float(q[4])}


# shortcut: bisection assumes a later decision never avoids what an earlier one could not; a full grid would show exceptions, at about 6x the episodes.
def last_ok(ok: Callable[[int], bool], lo: int, hi: int) -> int:
    """The largest index in [lo, hi) with ok, by bisection, given ok(lo) true and ok(hi) false (neither probed)."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _probes(points: int) -> int:
    return math.ceil(math.log2(points)) if points > 1 else 0


def choose_step(gaps: Sequence[int], budget: int) -> int:
    """Smallest grid step s whose worst-case bisection count, the sum of ceil(log2(ceil(gap / s))), is <= budget;
    gap = decision frames from the first full window to the first frame at or after contact."""
    step = 1
    while sum(_probes(math.ceil(g / step)) for g in gaps) > budget:
        step += 1
    return step


def best_action(outcomes: Mapping[str, Mapping[str, Any]], active: Sequence[str]) -> str | None:
    """The active action that neither collides nor falls with the largest min_separation_m (readout order on a
    tie), None if none does."""
    clean = [name for name in active if dodge_eval._avoids(outcomes[name])]
    return max(clean, key=lambda name: (outcomes[name]["min_separation_m"], -active.index(name))) if clean else None


def compare_ceilings(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], kinds: Mapping[int, str],
                     action_names: Sequence[str], no_op: str) -> dict[str, Any]:
    """Two ceilings of the same scenarios side by side: each one's ceiling_summary and falls per action, and by
    kind the scenarios whose NO_OP episode went from no contact (a) to contact (b) and back. Raises ValueError
    when the scenario ids differ."""
    if sorted(r["scenario_id"] for r in a) != sorted(r["scenario_id"] for r in b):
        raise ValueError("the two ceilings hold different scenario ids")

    def side(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        falls = {name: sum(int(float(r["outcomes"][name]["fall"]) == 1.0) for r in records if name in r["outcomes"])
                 for name in action_names}
        return {**dodge_eval.ceiling_summary(records, action_names, no_op), "falls": falls}

    hit_b = {r["scenario_id"]: float(r["outcomes"][no_op]["collision"]) == 1.0 for r in b}
    to_hit: dict[str, int] = {}
    to_miss: dict[str, int] = {}
    for r in a:
        sid, was = r["scenario_id"], float(r["outcomes"][no_op]["collision"]) == 1.0
        if was != hit_b[sid]:
            change = to_hit if hit_b[sid] else to_miss
            change[kinds[sid]] = change.get(kinds[sid], 0) + 1
    return {"a": side(a), "b": side(b), "missed_to_hit_by_kind": to_hit, "hit_to_missed_by_kind": to_miss}


def issue_at_policy(t_issue: float, action: str, command: Mapping[str, float], zero: Mapping[str, float], no_op: str,
                    hold_s: float) -> Callable[[np.ndarray, float], dict[str, Any]]:
    """The zero command until t_issue, then dodge_eval.first_issue_policy (issue once, hold hold_s, zero)."""
    inner = dodge_eval.first_issue_policy(action, command, zero, no_op, hold_s)

    def policy(window: np.ndarray, t_s: float) -> dict[str, Any]:
        if t_s < t_issue - dodge_eval._TOLERANCE:
            return {"command": dict(zero), "action": no_op}
        return inner(window, t_s)

    return policy


# ----- simulator tasks (in dodge_eval's workers) ---------------------------------------------------------------------


def _yaw(q: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _speed_task(task: tuple) -> dict[str, Any]:
    action, command, seed = task
    w = dodge_eval._WORKER
    world, config = w["world"], w["world"].config
    period = float(require(config, "robot.walking_policy.command_schema.update_period_s"))
    hold_s = float(require(config, "skills.dodge_vision.hold_s"))
    duration = float(require(config, "skills.dodge_vision.episode_duration_s"))
    obs = world.reset({"dog_forward_speed_m_s": 0.0, "settle_s": require(config, "skills.dodge_vision.settle_s")},
                      attempt_seed=int(seed))
    latency_s = world._delay * world.policy.control_dt  # hidden per-attempt value, read for this measurement only
    start, yaw = np.asarray(obs["base_pos_m"][:2], dtype=np.float64), _yaw(obs["base_quat_wxyz"])
    heading = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])  # rows: forward, left
    direction = np.array([command["vel_x"], command["vel_y"]], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    t_s, along, planar = [], [], []
    hold_periods = round(hold_s / period)
    for k in range(round(duration / period)):
        obs = world.step(command if k < hold_periods else w["zero"], period)
        travel = heading @ (np.asarray(obs["base_pos_m"][:2], dtype=np.float64) - start)
        t_s.append(float(obs["t_s"]))
        along.append(float(travel @ direction))
        planar.append(float(np.hypot(*travel)))
    outcome = world.finish_episode()
    end_of_hold = hold_periods - 1
    return {"action": action, "attempt_seed": int(seed), "latency_s": latency_s, "fall": int(outcome["fall"]),
            "reach_along_s": reach_times(t_s, along, _DISTANCES_M), "reach_planar_s": reach_times(t_s, planar, _DISTANCES_M),
            "along_at_end_of_hold_m": along[end_of_hold], "along_max_m": max(along), "planar_max_m": max(planar)}


def _deadline_task(task: tuple) -> dict[str, Any]:
    scenario, augment_seed, action, command, grid_t, hi = task
    w = dodge_eval._WORKER
    hold_s = float(require(w["world"].config, "skills.dodge_vision.hold_s"))
    probes: list[dict[str, Any]] = []

    def ok(i: int) -> bool:
        out, hit = dodge_eval._episode(issue_at_policy(grid_t[i], action, command, w["zero"], w["no_op"], hold_s),
                                       scenario, augment_seed)
        if len(out["issued"]) != 1 or out["issued"][0][1] != action or abs(out["issued"][0][0] - grid_t[i]) > 1e-6:
            raise RuntimeError(f"scenario {scenario['scenario_id']}: {action} at {grid_t[i]} was issued {out['issued']}")
        m = out["measured"]
        probes.append({"index": i, "t_issue_s": grid_t[i], "collision": m["collision"], "fall": m["fall"],
                       "min_separation_m": m["min_separation_m"], "first_collision_time_s": hit})
        return dodge_eval._avoids(m)

    lo = last_ok(ok, 0, hi)
    return {"scenario_id": scenario["scenario_id"], "action": action, "deadline_index": lo, "probes": probes}


# ----- commands ----------------------------------------------------------------------------------------------------


def _merge(path: Path, key: str, value: Mapping[str, Any]) -> None:
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record[key] = value
    dodge_eval._write_json(path, record)


def _eye(config: Mapping[str, Any], rates: Sequence[float]) -> dict[str, Any]:
    votes = int(require(config, "inference.gate.votes_needed"))
    return {"keys": ["vision.frame_variants", "vision.frame_rate_hz", "vision.frames", "inference.gate.votes_needed",
                     "inference.gate.of_last"],
            "frame_rates_hz": list(rates), "config_frame_rate_hz": require(config, "vision.frame_rate_hz"),
            "votes_needed": votes, "of_last": require(config, "inference.gate.of_last"),
            "closed_loop_window_frames": require(config, "vision.frames"),
            "rule": "first frame at the launch (t 0); first full window (F - 1) / rate; the gate votes on every window "
                    "from then on, so the earliest command is votes_needed - 1 frames later",
            "table": eye_times(rates, require(config, "vision.frame_variants"), votes)}


def speed(config: Mapping[str, Any], seeds: int, out: Path, argv: Sequence[str]) -> dict[str, Any]:
    actions = require(config, "skills.dodge_vision.actions")
    no_op = dodge_eval.GateSettings.from_config(config, dodge_eval._grid(config)[0]).no_op
    tasks = [(name, {k: float(v) for k, v in command.items()}, seed)
             for name, command in actions.items() if name != no_op for seed in range(seeds)]
    started = time.perf_counter()
    rows = dodge_eval._run(config, None, None, _speed_task, tasks)
    table = {}
    for name in dict.fromkeys(r["action"] for r in rows):
        mine = [r for r in rows if r["action"] == name]
        entry = {"n": len(mine), "falls": sum(r["fall"] for r in mine), "latency_s": spread([r["latency_s"] for r in mine])}
        for kind in ("along", "planar"):
            for j, d in enumerate(_DISTANCES_M):
                times = [r[f"reach_{kind}_s"][j] for r in mine if r[f"reach_{kind}_s"][j] is not None]
                entry[f"{kind}_{round(d * 100)}cm"] = {"reached": len(times), **spread(times),
                                                       "after_latency": spread([r[f"reach_{kind}_s"][j] - r["latency_s"] for r in mine
                                                                                if r[f"reach_{kind}_s"][j] is not None])}
        entry["along_at_end_of_hold_m"] = spread([r["along_at_end_of_hold_m"] for r in mine])
        table[name] = entry
    section = {"argv": list(argv), "config_digest": config_digest(config), "wall_s": time.perf_counter() - started,
               "where": "simulation (MuJoCo Go2World, frozen walking policy)",
               "world_seed": require(config, "observed.world_seed"), "world_preset": require(config, "observed.world_preset"),
               "attempt_seeds": list(range(seeds)), "distances_m": list(_DISTANCES_M),
               "along": "trunk travel along the commanded (vel_x, vel_y) direction in the decision instant's heading frame",
               "time": "t_s of the first 0.02 s control period at or beyond the distance; the command is issued at t_s 0",
               "table": table, "rows": rows}
    _merge(out, "A_dodge_speed", section)
    return section


def _load_ceiling(config: Mapping[str, Any], run_id: str, split: str, same_config: bool) -> tuple[Path, dict, dict, list]:
    """(path, ceiling file, records by scenario id, dodge_eval._plan of those scenarios); the ceiling must hold the
    first n scenarios of the split and, when same_config, carry this config's digest."""
    from learned_judgment.observed import _run_dir

    path = _run_dir(config, run_id) / f"ceiling_{split}.json"
    ceiling = json.loads(path.read_text(encoding="utf-8"))
    if same_config and ceiling["config_digest"] != config_digest(config):
        raise ValueError(f"{path} was run with config {ceiling['config_digest']}, this config is {config_digest(config)}")
    records = {r["scenario_id"]: r for r in ceiling["records"]}
    tasks_all = dodge_eval._plan(config, split, len(records))
    if sorted(s["scenario_id"] for s, _ in tasks_all) != sorted(records):
        raise ValueError(f"{path} does not hold the first {len(records)} scenarios of split {split}")
    return path, ceiling, records, tasks_all


def _arrival(config: Mapping[str, Any], path: Path, ceiling: Mapping[str, Any], records: Mapping[int, Any],
             tasks_all: Sequence[tuple]) -> dict[str, Any]:
    """B: NO_OP first contact times of the aimed kinds (standing dog), their straight-line arrivals, all contacts."""
    no_op = ceiling["no_op"]
    aimed = [(s, records[s["scenario_id"]]["outcomes"][no_op]) for s, _ in tasks_all if s["kind"] in _AIMED]
    contact = [o["first_collision_time_s"] for _, o in aimed if o["first_collision_time_s"] is not None]
    hits_all = [r["outcomes"][no_op]["first_collision_time_s"] for r in records.values()
                if r["outcomes"][no_op]["first_collision_time_s"] is not None]
    return {"kinds": list(_AIMED), "n_scenarios": len(aimed), "n_contact": len(contact), "contact_s": spread(contact),
            "straight_line_arrival_s": spread([t for t in (arrival_time_s(config, s) for s, _ in aimed) if t is not None]),
            "all_kinds_n_contact": len(hits_all), "all_kinds_contact_s": spread(hits_all), "ceiling_records": path,
            "config_digest": ceiling["config_digest"]}


def compare(config: Mapping[str, Any], run_a: str, run_b: str, split: str, out: Path, argv: Sequence[str]) -> dict[str, Any]:
    """E: the ceiling of run_a (this config) beside run_b's (another config, same scenarios)."""
    path_a, ceil_a, rec_a, tasks_all = _load_ceiling(config, run_a, split, same_config=True)
    path_b, ceil_b, rec_b, _ = _load_ceiling(config, run_b, split, same_config=False)
    names = list(require(config, "skills.dodge_vision.actions"))
    kinds = {s["scenario_id"]: s["kind"] for s, _ in tasks_all}
    section = {"argv": list(argv), "a": {"run": run_a, "config_digest": ceil_a["config_digest"]},
               "b": {"run": run_b, "config_digest": ceil_b["config_digest"]},
               **compare_ceilings(list(rec_a.values()), list(rec_b.values()), kinds, names, ceil_a["no_op"]),
               "arrival_a": _arrival(config, path_a, ceil_a, rec_a, tasks_all),
               "arrival_b": _arrival(config, path_b, ceil_b, rec_b, tasks_all)}
    _merge(out, "E_twice_radius", section)
    return section


def deadline(config: Mapping[str, Any], run_id: str, split: str, budget: int, rates: Sequence[float], out: Path,
             argv: Sequence[str]) -> dict[str, Any]:
    path, ceiling, records, tasks_all = _load_ceiling(config, run_id, split, same_config=True)
    digest = ceiling["config_digest"]
    no_op = ceiling["no_op"]
    actions = require(config, "skills.dodge_vision.actions")
    active = [name for name in actions if name != no_op]
    b = _arrival(config, path, ceiling, records, tasks_all)

    # D: the latest decision frame that still avoids, on the grid
    eye = _eye(config, rates)
    times = frame_times(config)
    first = int(require(config, "vision.frames")) - 1  # closed_loop's first full window
    decision_t = times[first:]
    plan = []
    for scenario, seed in tasks_all:
        outcomes = records[scenario["scenario_id"]]["outcomes"]
        hit = outcomes[no_op]["first_collision_time_s"]
        if outcomes[no_op]["collision"] != 1.0 or (action := best_action(outcomes, active)) is None:
            continue
        gap = int(np.searchsorted(decision_t, hit - dodge_eval._TOLERANCE))  # first decision frame at or after contact
        plan.append((scenario, seed, action, gap))
    step = choose_step([gap for *_, gap in plan], budget)
    tasks = []
    for scenario, seed, action, gap in plan:
        grid_t = [float(t) for t in decision_t[0:gap:step]]
        command = {k: float(v) for k, v in actions[action].items()}
        tasks.append((scenario, seed, action, command, grid_t, len(grid_t)))
    print(f"deadline: {len(tasks)} avoidable hits, grid step {step} decision frames, worst case "
          f"{sum(_probes(t[5]) for t in tasks)} episodes", flush=True)
    started = time.perf_counter()
    results = dodge_eval._run(config, None, None, _deadline_task, tasks) if tasks else []
    _merge(out, "D_deadline", {"state": "episodes done, summary pending", "grid_step_decision_frames": step, "results": results})
    rows = []
    for (scenario, _, action, _, grid_t, _), r in zip(tasks, results):
        frames_late = r["deadline_index"] * step
        rows.append({**r, "kind": scenario["kind"], "deadline_s": grid_t[r["deadline_index"]],
                     "contact_s": records[scenario["scenario_id"]]["outcomes"][no_op]["first_collision_time_s"],
                     "deadline_minus_first_window_s": grid_t[r["deadline_index"]] - float(decision_t[0]),
                     "deadline_minus_first_window_frames": frames_late})
    leave = [{"frame_rate_hz": e["frame_rate_hz"], "frames": e["frames"], "earliest_command_s": e["earliest_command_s"],
              "punches_deadline_at_or_after": sum(1 for r in rows if r["deadline_s"] >= e["earliest_command_s"] - dodge_eval._TOLERANCE)}
             for e in eye["table"]]
    d = {"argv": list(argv), "config_digest": digest, "grid_step_decision_frames": step, "budget": budget,
         "first_full_window_s": float(decision_t[0]), "n_avoidable": len(rows),
         "n_hit_no_op": sum(1 for r in records.values() if r["outcomes"][no_op]["collision"] == 1.0),
         "episodes": sum(len(r["probes"]) for r in rows), "wall_s": time.perf_counter() - started,
         "deadline_minus_first_window_s": spread([r["deadline_minus_first_window_s"] for r in rows]),
         "deadline_minus_first_window_frames": spread([r["deadline_minus_first_window_frames"] for r in rows]),
         "deadline_to_contact_s": spread([r["contact_s"] - r["deadline_s"] for r in rows]),
         "leave_eye_time": leave,
         "resolution": f"the deadline is a grid point: the true latest moment lies before the next grid point ({step} decision frames)",
         "rule": "best action at the first full window (clean avoidance, largest separation, readout order on a tie) issued at "
                 "grid decision frames; bisection between the first full window (avoids, from the ceiling) and the first "
                 "frame at or after contact (fails)",
         "where": "simulation (MuJoCo Go2World, frozen walking policy)", "rows": rows}
    _merge(out, "B_punch_arrival", b)
    _merge(out, "C_eye_time", eye)
    _merge(out, "D_deadline", d)
    return {"B": b, "D": d}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.reaction", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sp = commands.add_parser("speed", help="A: time from a dodge command to 5/10/20 cm of trunk travel")
    sp.add_argument("--seeds", type=int, default=10, help="attempt seeds 0 .. N-1 per action (latency jitter differs)")
    ey = commands.add_parser("eye", help="C: time to the first full window and the earliest gated command")
    de = commands.add_parser("deadline", help="B and D from the ceiling records: contact times and decision deadlines")
    de.add_argument("--run", required=True, help="run id whose ceiling_<split>.json is read")
    de.add_argument("--split", default="test", choices=("dev", "test"))
    de.add_argument("--budget", type=int, default=500, help="worst-case grid episodes; sets the grid step")
    for sub in (ey, de):
        sub.add_argument("--rates", type=float, nargs="+", help="camera rates in Hz (default 15 and vision.frame_rate_hz)")
    co = commands.add_parser("compare", help="E: this config's ceiling beside another config's on the same scenarios")
    co.add_argument("--run", required=True, help="run id of the ceiling made with this config")
    co.add_argument("--other", required=True, help="run id of the ceiling made with the other config")
    co.add_argument("--split", default="test", choices=("dev", "test"))
    for sub in (sp, ey, de, co):
        sub.add_argument("--out", type=Path, default=_OUT, help=f"JSON file of the sections (default {_OUT})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    words = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(words)
    config = load_config()
    rates = None if args.command == "speed" else (args.rates or [15.0, float(require(config, "vision.frame_rate_hz"))])
    if args.command == "speed":
        section = speed(config, args.seeds, args.out, words)
        print(json.dumps({name: {k: v for k, v in e.items() if k.startswith(("along", "planar", "latency"))}
                          for name, e in section["table"].items()}, default=str), flush=True)
    elif args.command == "eye":
        section = _eye(config, rates)
        _merge(args.out, "C_eye_time", section)
        print(json.dumps(section["table"]), flush=True)
    elif args.command == "compare":
        section = compare(config, args.run, args.other, args.split, args.out, words)
        print(json.dumps({k: v for k, v in section.items() if k not in ("arrival_a", "arrival_b")}, default=str), flush=True)
    else:
        report = deadline(config, args.run, args.split, args.budget, rates, args.out, words)
        print(json.dumps({"B": {k: report["B"][k] for k in ("n_scenarios", "n_contact", "contact_s")},
                          "D": {k: v for k, v in report["D"].items() if k != "rows"}}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    # Through the importable module, so spawned workers find the tasks by their module name.
    from learned_judgment import reaction as _module

    raise SystemExit(_module.main())
