"""Phase 1 locomotion gate: run the borrowed policy on the flat scene and measure what it does.

    python -m learned_judgment.env.acceptance --out DIR

run_locomotion and run_stop only measure. The CLI judges each condition against robot.acceptance in config.yaml,
writes DIR/locomotion.json and one DIR/<condition>.npz trace per condition, and exits 1 if any condition fails.
prove_commands runs every command field at both ends of its schema range and saves the signed responses.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from learned_judgment.env import load_config
from learned_judgment.env.policy import WalkingPolicy
from learned_judgment.env.scene import build_model, is_ground


def run_locomotion(config: dict, vx: float, yaw_rate: float, duration_s: float, seed: int,
                   trace_path: str | Path | None = None) -> dict:
    """One episode at the constant command [vx, 0, yaw_rate]."""
    return _episode(config, [(duration_s, [vx, 0.0, yaw_rate])], seed, trace_path)


def run_stop(config: dict, vx: float, trot_s: float, stop_s: float, seed: int,
             trace_path: str | Path | None = None) -> dict:
    """One episode: trot_s at [vx, 0, 0], then stop_s at zero command."""
    return _episode(config, [(trot_s, [vx, 0.0, 0.0]), (stop_s, [0.0, 0.0, 0.0])], seed, trace_path)


def _episode(config, schedule, seed, trace_path) -> dict:
    # seed is recorded only: nothing in the simulator, policy or this episode draws random numbers.
    acc = config["robot"]["acceptance"]
    model = build_model(config)
    wp = WalkingPolicy(config, model)
    base = model.body("base_link").id
    worst = {"cos_tilt": 1.0, "trunk_floor": False}

    def watch():  # after every physics step of the commanded run
        worst["trunk_floor"] |= trunk_on_floor(model, wp.data)
        worst["cos_tilt"] = min(worst["cos_tilt"], wp.data.xmat[base][8])  # world z . trunk z

    wall0, cpu0 = time.perf_counter(), time.process_time()
    wp.reset()
    commands = [np.zeros(3)]
    qpos, qvel = [wp.data.qpos.copy()], [wp.data.qvel.copy()]
    for duration_s, command in schedule:
        for _ in range(round(duration_s / wp.control_dt)):
            wp.step(np.array(command), on_substep=watch)
            commands.append(command)
            qpos.append(wp.data.qpos.copy())
            qvel.append(wp.data.qvel.copy())
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    sim_s = wp.data.time  # settle + soft start + commanded run; mj_resetData in reset() zeroed it

    qpos, qvel, commands = np.array(qpos), np.array(qvel), np.array(commands)
    t = np.arange(len(qpos)) * wp.control_dt  # time since the commanded run started
    w, x, y, z = qpos[:, 3:7].T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    v_xy = qvel[:, :2]  # free-joint linear velocity is in the world frame
    forward = v_xy[:, 0] * np.cos(yaw) + v_xy[:, 1] * np.sin(yaw)  # along the dog's own heading
    left = -v_xy[:, 0] * np.sin(yaw) + v_xy[:, 1] * np.cos(yaw)  # along the heading frame's y (left)
    late = t > acc["exclude_first_s"]
    max_tilt_deg = float(np.degrees(np.arccos(np.clip(worst["cos_tilt"], -1.0, 1.0))))
    if trace_path is not None:
        np.savez(trace_path, t=t, qpos=qpos, qvel=qvel, command=commands, seed=seed, sim_s=sim_s, wall_s=wall)
    return {
        "fallen": worst["trunk_floor"] or max_tilt_deg > config["robot"]["fall"]["max_tilt_deg"],
        "max_tilt_deg": max_tilt_deg,
        "mean_speed_m_s": float(forward[late].mean()),
        "mean_lateral_speed_m_s": float(left[late].mean()),
        "mean_yaw_rate_rad_s": float((np.diff(yaw) / wp.control_dt)[late[1:]].mean()),
        "final_speed_m_s": float(np.linalg.norm(v_xy, axis=1)[t > t[-1] - acc["final_window_s"]].mean()),
        "rtf": sim_s / wall,
        "cpu_fraction": cpu / wall,
        "trace_path": None if trace_path is None else Path(trace_path).as_posix(),
    }


# The signed response to each native command index of [vel_x, vel_y, yaw_rate]: forward speed along the heading,
# leftward speed along the heading frame's y, yaw rate.
_RESPONSE = ("mean_speed_m_s", "mean_lateral_speed_m_s", "mean_yaw_rate_rad_s")


def prove_commands(config: dict, out_dir: str | Path) -> Path:
    """Run each command_schema field at its min and at its max for 10 s, the other fields zero, and write
    out_dir/commands.json: per field and end the commanded value, the signed measured response (mean after the first
    exclude_first_s), fallen, proven (not fallen and the response has the commanded sign) and the trace path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = config["robot"]["walking_policy"]["command_schema"]["fields"]
    proof = {}
    for name, f in fields.items():
        for end in ("min", "max"):
            command = [0.0] * len(fields)
            command[f["native_index"]] = f[end]
            m = _episode(config, [(10, command)], config["runtime"]["seed"], out / f"{name}_{end}.npz")
            measured = m[_RESPONSE[f["native_index"]]]
            proof.setdefault(name, {})[f"at_{end}"] = {
                "commanded": f[end], "measured": measured, "fallen": m["fallen"],
                "proven": not m["fallen"] and measured * f[end] > 0, "trace_path": m["trace_path"]}
            print(f"{name} at_{end}: {json.dumps(proof[name][f'at_{end}'])}", flush=True)
    path = out / "commands.json"
    path.write_text(json.dumps(proof, indent=2))
    return path


def trunk_on_floor(model, data) -> bool:
    """True if a current contact pairs the ground (scene.is_ground: the floor or the incline) with a geom of the trunk
    (body base_link)."""
    g = data.contact.geom  # (ncon, 2) geom ids
    floor = is_ground(model)[g]
    trunk = model.geom_bodyid[g] == model.body("base_link").id
    return bool(np.any((floor[:, 0] & trunk[:, 1]) | (floor[:, 1] & trunk[:, 0])))


def judge(config: dict, check: str, command: dict, m: dict) -> tuple[bool, str]:
    """Pass or fail one condition on its measurements, and the rule applied."""
    acc = config["robot"]["acceptance"]
    if check == "speed":
        vx = command["vx"]
        tol = acc["speed_tolerance_fraction"] * vx
        ok, rule = abs(m["mean_speed_m_s"] - vx) <= tol, f"|mean_speed_m_s - {vx}| <= {tol:.4g}"
    elif check == "yaw":
        yaw = command["yaw_rate"]
        tol = acc["yaw_tolerance_fraction"] * abs(yaw)
        ok, rule = abs(m["mean_yaw_rate_rad_s"] - yaw) <= tol, f"|mean_yaw_rate_rad_s - {yaw}| <= {tol:.4g}"
    elif check == "stop":
        ok, rule = m["final_speed_m_s"] < acc["stop_speed_m_s"], f"final_speed_m_s < {acc['stop_speed_m_s']}"
    elif check == "stand":
        ok, rule = True, None
    else:
        raise ValueError(f"unknown check {check!r}")
    return ok and not m["fallen"], " and ".join(r for r in ("not fallen", rule) if r)


def conditions(config: dict) -> list[tuple[str, str, object, dict]]:
    """The seven conditions of tests/env/test_locomotion.py (plan Task 2, Step 1): name, check, run, command."""
    acc = config["robot"]["acceptance"]
    yaw = 0.5 * max(acc["yaw_rates_supported_by_policy"])
    trot = [(f"trot_{vx}", "speed", run_locomotion, {"vx": vx, "yaw_rate": 0.0, "duration_s": acc["trot_duration_s"]})
            for vx in acc["commanded_speeds_m_s"]]
    return trot + [
        ("turn_left", "yaw", run_locomotion, {"vx": 0.3, "yaw_rate": yaw, "duration_s": 10}),
        ("turn_right", "yaw", run_locomotion, {"vx": 0.3, "yaw_rate": -yaw, "duration_s": 10}),
        ("stop", "stop", run_stop, {"vx": 0.6, "trot_s": 5, "stop_s": 5}),
        ("stand", "stand", run_locomotion, {"vx": 0.0, "yaw_rate": 0.0, "duration_s": acc["trot_duration_s"]}),
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="acceptance")
    parser.add_argument("--out", required=True)
    out = Path(parser.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config()
    seed = config["runtime"]["seed"]
    entries = []
    for name, check, run, command in conditions(config):
        measured = run(config, **command, seed=seed, trace_path=out / f"{name}.npz")
        passed, rule = judge(config, check, command, measured)
        entries.append({"condition": name, "command": command, "seed": seed, "measured": measured,
                        "criterion": rule, "passed": passed})
        print(f"{name}: {'PASS' if passed else 'FAIL'} ({rule}) {json.dumps(measured)}", flush=True)
    (out / "locomotion.json").write_text(json.dumps(entries, indent=2))
    failed = [e["condition"] for e in entries if not e["passed"]]
    print("LOCOMOTION GATE:", "PASS" if not failed else f"FAIL {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
