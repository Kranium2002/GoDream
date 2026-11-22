"""Run-backed story artifacts. All records are derived from the selected run's saved files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from learned_judgment.config import load_config
from learned_judgment import observed

ROOT = Path(__file__).resolve().parents[1]


def _read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _run(run_id):
    folder = (ROOT / "results" / "runs" / run_id).resolve()
    if folder.parent != (ROOT / "results" / "runs").resolve() or not folder.is_dir():
        raise ValueError(f"unknown run: {run_id}")
    return folder


def _context(folder):
    for name in ("context", "dev"):
        path = folder / name
        if (path / "rows.npz").is_file() and (path / "rows.json").is_file():
            return path
    # Multi-round skills (for example reflex2) store each finished run under round_N.
    # Select from complete on-disk artifact sets, not a preferred run/round name.
    candidates = []
    for path in folder.glob("round_*/context"):
        base = path.parent
        required = (path / "rows.npz", path / "rows.json", base / "dream" / "dream.npz",
                    base / "dream" / "dream.json", base / "scores.npz", base / "student" / "student.json")
        if all(item.is_file() for item in required):
            candidates.append((max(item.stat().st_mtime_ns for item in required), path))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    raise FileNotFoundError("run has no complete context/rows.npz and rows.json")


def _artifact_root(folder):
    try:
        return _context(folder).parent
    except FileNotFoundError:
        return folder


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _outline(saved):
    """Turn saved mound circle parameters into drawable XY vertices; keep unknown geometry verbatim."""
    if isinstance(saved, dict) and all(k in saved for k in ("mound_x_m", "mound_y_m", "mound_radius_m")):
        x, y, radius = (float(saved[k]) for k in ("mound_x_m", "mound_y_m", "mound_radius_m"))
        return {"kind": "circle", "center_xy_m": [x, y], "radius_m": radius,
                "outline_xy_m": [[x + radius * math.cos(2 * math.pi * i / 48),
                                   y + radius * math.sin(2 * math.pi * i / 48)] for i in range(48)],
                "source": "saved mound_world center and radius"}
    return saved


def _skill(folder):
    cfg = load_config(folder / "config.yaml")
    result = _read(folder / "result.json", {}) or {}
    skill = result.get("skill") or _read(_context(folder) / "rows.json", {}).get("skill")
    if skill not in {"reflex", "dodge_vision", "ramp", "move"}:
        raise ValueError(f"unsupported or missing observed-engine skill: {skill}")
    return cfg, skill, observed.load_spec(skill, cfg)


def _finished(folder):
    status = _read(folder / "run_status.json", {}) or {}
    if status.get("state"):
        return status["state"] == "done"
    result = _read(folder / "result.json", {}) or {}
    terminal = ("closed_loop_test", "run_test", "test", "loop")
    return any(isinstance(result.get(key), dict) and result[key].get("state") == "done" for key in terminal)


def paths(folder):
    try:
        ctx = _context(folder)
    except FileNotFoundError:
        record = {"n_tries": None, "n_decisions": None, "tries": None, "paths": None,
                  "outcomes": None, "obstacles": [], "obstacle_outlines": None,
                  "source": "run contains no saved context/development rows; no paths or outcomes inferred"}
        _json(folder / "story" / "attempts.json", record)
        return record
    rows = observed.load_rows(ctx)
    meta = rows["meta"]
    scenarios = meta.get("scenarios") or []
    episodes = meta.get("episodes") or []
    by_id = {int(x.get("scenario_id", i)): x for i, x in enumerate(episodes) if isinstance(x, dict)}
    scenario_by_id = {int(x.get("scenario_id", i)): x for i, x in enumerate(scenarios) if isinstance(x, dict)}
    action_names = meta.get("action_names") or []
    outcome_names = meta.get("outcome_names") or []
    attempts = []
    ids = np.unique(rows["scenario_index"])
    artifact_root = ctx.parent
    track_index = _read(artifact_root / "test_run" / "tracks" / "index.json", []) or []
    saved_tracks = [entry for entry in track_index if entry.get("method") in ("snn", "student")]
    if saved_tracks:
        # These are real evaluated attempts; use their own outcome and track metadata.
        ids = np.asarray(sorted({int(x["scenario_id"]) for x in saved_tracks}), dtype=np.int64)
        for sid in ids:
            item = next((x for x in saved_tracks if int(x["scenario_id"]) == int(sid)), {})
            outcome = item.get("outcome") or {}
            attempts.append({"try": int(sid), "n_decisions": outcome.get("n_decisions"), "outcome": outcome or None,
                            "scenario": {"kind": item.get("kind")} if item.get("kind") else scenario_by_id.get(int(sid)),
                            "path": None, "obstacles": _outline(item.get("mound_world") or item.get("obstacles"))})
    else:
        for sid in ids:
            mask = rows["scenario_index"] == sid
            item = by_id.get(int(sid), {})
            outcome = item.get("outcome") or {}
            measured = [{"action": action_names[int(rows["action"][i])] if int(rows["action"][i]) < len(action_names) else int(rows["action"][i]),
                         "outcomes": {name: float(value) for name, value in zip(outcome_names, rows["measured"][i])}}
                        for i in np.flatnonzero(mask)]
            attempts.append({"try": int(sid), "n_decisions": int(mask.sum()), "outcome": outcome or None,
                            "scenario": scenario_by_id.get(int(sid)), "measured_outcomes": measured,
                            "path": None, "obstacles": _outline(item.get("mound_world") or item.get("obstacles"))})
    for entry in track_index:
        sid = entry.get("scenario_id")
        if sid is None or entry.get("method") not in ("snn", "student"):
            continue
        dest = next((a for a in attempts if a["try"] == int(sid)), None)
        p = artifact_root / "test_run" / "tracks" / entry.get("file", "")
        if dest is None or not p.is_file():
            continue
        try:
            with np.load(p, allow_pickle=False) as archive:
                qpos = np.asarray(archive["qpos"])
                dest["path"] = [[float(v[0]), float(v[1])] for v in qpos[:, :2]]
        except (OSError, KeyError, ValueError):
            pass
    decisions = (sum(int(a.get("n_decisions") or 0) for a in attempts) if saved_tracks
                 else int(len(rows["action"])))
    paths_for_ui = [{"points": a["path"], "ended": (a.get("outcome") or {}).get("end_reason")}
                    for a in attempts if a.get("path")]
    outcomes_for_ui = [{"kind": (a.get("scenario") or {}).get("kind", ""),
                        "ended": (a.get("outcome") or {}).get("end_reason")}
                       for a in attempts]
    obstacles_for_ui = []
    for a in attempts:
        obstacle = a.get("obstacles")
        if isinstance(obstacle, dict) and "center_xy_m" in obstacle and "radius_m" in obstacle:
            obstacles_for_ui.append({"x": obstacle["center_xy_m"][0], "y": obstacle["center_xy_m"][1], "r": obstacle["radius_m"]})
    record = {"n_tries": int(len(ids)), "n_decisions": decisions, "tries": attempts,
              "paths": paths_for_ui or None, "outcomes": outcomes_for_ui, "obstacles": obstacles_for_ui,
              "obstacle_outlines": [x.get("obstacles") for x in attempts if x.get("obstacles") is not None],
              "source": "context rows and saved test_run/tracks only; paths null where no matching track exists"}
    _json(folder / "story" / "attempts.json", record)
    return record


def brain(folder):
    artifact_root = _artifact_root(folder)
    student_meta = _read(artifact_root / "student" / "student.json")
    network_path = artifact_root / "student" / "network.npz"
    record = None
    if student_meta and network_path.is_file():
        from learned_judgment.students.snn_core import count_operations, load_network
        from learned_judgment.observed import _student_features, encode_inputs
        net, net_meta = load_network(network_path)
        try:
            dream = observed.load_dream(artifact_root / "dream")
            features = np.asarray(dream["features"])
        except (FileNotFoundError, ValueError):
            features = np.empty((0, 0))
        if features.ndim == 2 and len(features) and _student_features(features, student_meta.get("input")).shape[1] == net.n_inputs:
            student_features = _student_features(features, student_meta.get("input"))
            encoded = encode_inputs(student_features)
            counts = count_operations(net, encoded)
            spikes = counts.spikes_per_decision
            spike_summary = {"n_decisions": int(len(spikes)), "mean_per_layer": spikes.mean(axis=0).tolist(),
                             "per_decision_per_layer": spikes.tolist(), "source": "snn_core.count_operations on saved dream features"}
        else:
            spike_summary = None
        curve = []
        event_file = artifact_root / "events.jsonl"
        if not event_file.is_file():
            event_file = folder / "events.jsonl"
        active_dream = None
        if event_file.is_file():
            for line in event_file.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                    data = event.get("data")
                    if event.get("step") == "train" and isinstance(data, dict) and data.get("dream"):
                        active_dream = str(data["dream"]).replace("\\", "/").lower()
                    if event.get("step") != "train" or not isinstance(data, dict) or not isinstance(event.get("done"), (int, float)):
                        continue
                    if "train_loss" not in data and "validation_loss" not in data:
                        continue
                    if artifact_root != folder:
                        if active_dream is None or artifact_root.name.lower() not in active_dream:
                            continue
                    curve.append({"step": int(event["done"]), "train": data.get("train_loss"),
                                  "validation": data.get("validation_loss")})
                except ValueError:
                    continue
        curve.sort(key=lambda point: point["step"])
        decision_time = None
        mean_spikes = None
        if spike_summary is not None:
            mean_spikes = float(np.sum(spikes, axis=1).mean())
            from learned_judgment.eval.harness import measure_latency
            timing = measure_latency(net.decide_one, list(encoded), min(1000, max(1, len(encoded))))
            decision_time = {"median_ms": timing["median_s"] * 1000, "p95_ms": timing["p95_s"] * 1000,
                             "n_decisions": timing["n_decisions"], "n_inputs": timing["n_inputs"],
                             "method": "eval.harness.measure_latency of network.decide_one on encoded saved dream windows"}
        record = {"inputs": int(net.n_inputs), "neurons": list(net.layer_sizes[1:-1]),
                  "choices": student_meta.get("action_names"), "curve": curve,
                  "training_curve": curve or None,
                  "spikes": spike_summary, "decision_time": decision_time,
                  "spikes_per_decision": mean_spikes,
                  "decision_ms": decision_time["median_ms"] if decision_time else None,
                  "decision_time_note": None if decision_time else "run has no compatible saved dream windows for latency measurement",
                  "network_sha256": net_meta.get("network_sha256"),
                  "dream_rows": int(len(features)), "source": "student/, dream/, events.jsonl"}
    _json(folder / "story" / "brain.json", record)
    return record


def _physics(skill, cfg, rows, scenario_meta, row_ids, actions):
    f = rows["features"][row_ids]
    if skill == "reflex":
        from learned_judgment.skills.reflex import physics_guess
        return physics_guess(cfg, rows["state"][row_ids], actions)
    if skill == "move":
        from learned_judgment.skills.move import physics_guess
        return physics_guess(cfg, f, actions)
    if skill == "ramp":
        from learned_judgment.skills.ramp import physics_guess
        episode = float(cfg["skills"][skill].get("episode_duration_s", 0))
        return physics_guess(cfg, f, actions, np.maximum(0, episode - rows["decision_time_s"][row_ids]))
    from learned_judgment.skills.dodge_vision import physics_guess
    lookup = {int(value.get("scenario_id", index)): value for index, value in enumerate(scenario_meta)}
    scenarios = [lookup[int(rows["scenario_index"][i])] for i in row_ids]
    return physics_guess(cfg, scenarios, rows["decision_time_s"][row_ids], actions)


def holdout(folder, share=0.25, confirm=False):
    if confirm and not _finished(folder):
        raise RuntimeError("refusing billed holdout: this run has no finished status in run_status.json/result.json")
    cfg, skill, spec = _skill(folder)
    ctx = _context(folder)
    rows = observed.load_rows(ctx)
    meta = rows["meta"]
    ids = np.unique(rows["scenario_index"])
    if len(ids) < 2:
        raise ValueError("holdout needs at least two distinct tries/episodes")
    seed = int(cfg["runtime"]["seed"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5702]))
    n_test = min(len(ids)-1, max(1, int(round(len(ids)*share))))
    held_ids = np.sort(rng.choice(ids, size=n_test, replace=False))
    held_mask = np.isin(rows["scenario_index"], held_ids)
    train_idx, test_idx = np.flatnonzero(~held_mask), np.flatnonzero(held_mask)
    feature_set = tuple(cfg["skills"][skill].get("teacher_features", ("window", "action", "physics")))
    train = {key: np.asarray(value)[train_idx] for key, value in rows.items()
             if key in {"features", "action", "physics", "measured"}}
    teacher = observed.make_teacher(spec, train, cfg, feature_set)
    quote = observed.quote_dream(teacher, len(test_idx))
    print(json.dumps({"run": folder.name, "skill": skill, "n_tries_held_out": len(held_ids),
                      "n_decisions_held_out": len(test_idx), "quote": quote}, indent=2, default=str))
    if not confirm:
        return 3
    c = spec.n_action
    actions = np.tile(np.arange(c, dtype=np.int64), len(test_idx))
    repeated_rows = np.repeat(test_idx, c)
    scenario_meta = meta.get("scenarios") or []
    physical = _physics(skill, cfg, rows, scenario_meta, repeated_rows, actions)
    x, _layout = observed.teacher_matrix(spec, rows["features"][repeated_rows], actions, physical, feature_set)
    dist = observed.predict_outcomes(teacher, spec, x, physical, allow_billing=True)
    mean = dist.mean.reshape(len(test_idx), c, spec.n_outcome)
    quantiles = dist.quantiles.reshape(len(test_idx), c, spec.n_outcome, len(dist.quantile_levels))
    qi_low = int(np.argmin(np.abs(dist.quantile_levels - 0.1)))
    qi_high = int(np.argmin(np.abs(dist.quantile_levels - 0.9)))
    actual_actions = rows["action"][test_idx]
    measured = rows["measured"][test_idx]
    actual_idx = np.arange(len(test_idx))
    abs_tab = np.abs(mean[actual_idx, actual_actions] - measured)
    abs_formula = np.abs(rows["physics"][test_idx] - measured)
    per_try_tab = np.stack([abs_tab[rows["scenario_index"][test_idx] == sid].mean(axis=0) for sid in held_ids])
    per_try_formula = np.stack([abs_formula[rows["scenario_index"][test_idx] == sid].mean(axis=0) for sid in held_ids])
    tab_err = per_try_tab.mean(axis=0)
    formula_err = per_try_formula.mean(axis=0)
    units = [outcome.unit for outcome in spec.outcomes]
    per_outcome = [{"label": n, "unit": units[i], "n": int(len(held_ids)),
                    "tabpfn_error": float(tab_err[i]), "formula_error": float(formula_err[i])}
                   for i, n in enumerate(spec.outcome_names)]
    imagination = {"computed": "For each held-out try, average absolute error across its saved decision rows for the action actually taken; then average those per-try means equally across tries. TabPFN uses the teacher mean; formula uses the run's saved physics guess.",
                   "seed": seed, "split_seed": [seed, 0x5702], "held_out_try_ids": held_ids.tolist(),
                   "split_rule": "SeedSequence([runtime.seed, 0x5702]); choose round(unique_tries * share) unique scenario_index values without replacement, clamped to [1, n_tries-1].",
                   "n_tries_held_out": int(len(held_ids)), "n_decisions_held_out": int(len(test_idx)),
                   "request_keys": list(getattr(teacher, "last_request_hashes", ())),
                   "quote": quote, "outcomes": per_outcome, "rows": per_outcome}
    _json(folder / "story" / "imagination.json", imagination)
    moments_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5703]))
    moments = []
    for sid in moments_rng.permutation(held_ids)[:min(6, len(held_ids))]:
        possible = np.flatnonzero(rows["scenario_index"][test_idx] == sid)
        row = int(moments_rng.choice(possible))
        tried = int(actual_actions[row]); candidates=[]
        for a, name in enumerate(spec.action_names):
            outcome = {label: {"value": float(mean[row,a,k]), "low": float(quantiles[row,a,k,qi_low]),
                               "high": float(quantiles[row,a,k,qi_high]), "unit": units[k], "label": label}
                       for k, label in enumerate(spec.outcome_names)}
            measured_all = ({label: {"value": float(measured[row,k]), "unit": units[k], "label": label}
                             for k, label in enumerate(spec.outcome_names)} if a == tried else None)
            primary = spec.outcome_names[0]
            candidates.append({"label": name, "tried": a == tried, "predicted": outcome[primary],
                               "predicted_outcomes": outcome,
                               "measured": measured_all[primary] if measured_all else None,
                               "measured_outcomes": measured_all})
        moments.append({"selection_rule": "Seeded RNG [runtime.seed, 0x5703]: shuffle held-out tries; select up to six, then choose one held-out decision uniformly within each selected try.",
                        "seed": seed,
                        "moment": {"try": int(sid), "t_s": float(rows["decision_time_s"][test_idx[row]])},
                        "choices": candidates})
    _json(folder / "story" / "what_if.json", moments)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.story")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("paths", "brain"):
        p = sub.add_parser(command); p.add_argument("--run", required=True)
    p = sub.add_parser("holdout"); p.add_argument("--run", required=True); p.add_argument("--share", type=float, default=.25); p.add_argument("--confirm-billing", action="store_true")
    args = parser.parse_args(argv); folder = _run(args.run)
    if args.command == "paths":
        paths(folder); return 0
    if args.command == "brain":
        brain(folder); return 0
    if not 0 < args.share < 1:
        parser.error("--share must be between 0 and 1")
    return holdout(folder, args.share, args.confirm_billing)


if __name__ == "__main__":
    raise SystemExit(main())
