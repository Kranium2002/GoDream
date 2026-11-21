"""DODGE measurements run through Go2World: throw calibration, the published situation ranges, the sideways step
response while trotting and a saved example episode.

calibrate_throws measures what the throw protocol does to a dog that ignores the ball; situation_ranges derives the
`skills.dodge.situation_ranges` published in config.yaml; step_response and save_dodge_example are runs for Person B.
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np

from learned_judgment.env.throws import BALL_KEYS, sample_dodge_situations
from learned_judgment.env.world import Go2World

OUTCOME = ("collision", "first_collision_time_s", "min_separation_m", "fall", "recovery_time_s", "recovered",
           "sim_unstable", "trace_path")


def calibrate_throws(config: dict, seed: int, n: int, out_dir: str | Path) -> Path:
    """n throws from Go2World(config, seed).sample_dodge_situation(), each against a dog that holds its initial
    command (vel_x at the throw's dog_forward_speed_m_s, the other fields 0) for skills.dodge.episode_duration_s.
    Writes and returns out_dir/throws.json: per throw its attempt_seed, situation and measured outcome."""
    world = Go2World(config, seed)
    duration = config["skills"]["dodge"]["episode_duration_s"]
    throws = []
    for i in range(n):
        situation = world.sample_dodge_situation()
        world.reset(situation, attempt_seed=i)
        world.step({name: 0.0 for name in world.command_schema["fields"]} | {
            "vel_x": situation["dog_forward_speed_m_s"]}, duration)
        outcome = world.finish_episode()
        throws.append({"attempt_seed": i, "situation": situation, **{k: outcome[k] for k in OUTCOME}})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "throws.json"
    path.write_text(json.dumps(throws, indent=2))
    return path


def situation_ranges(config: dict, seed: int, n: int) -> dict[str, list[float]]:
    """[min, max] of each situation key over n sampled throws, widened by 2% of the span on each side and rounded
    outward to 0.01: the published skills.dodge.situation_ranges."""
    throws = sample_dodge_situations(config, np.random.default_rng(seed), n)
    ranges = {}
    for key in (*BALL_KEYS, "dog_forward_speed_m_s"):
        lo, hi = min(t[key] for t in throws), max(t[key] for t in throws)
        pad = 0.02 * (hi - lo)
        ranges[key] = [math.floor((lo - pad) * 100) / 100, math.ceil((hi + pad) * 100) / 100]
    return ranges


def step_response(config: dict, forward_speeds: list[float], side_speeds: list[float], duration_s: float,
                  out_dir: str | Path) -> Path:
    """A sideways step while trotting, through Go2World. Per forward speed vx and side speed vy: reset trotting at vx
    (world.settle_s), then command vel_x vx, vel_y vy, yaw_rate 0 for duration_s, observed every control period.
    Saves out_dir/vx<vx>_vy<vy>.npz (t_s; side_displacement_m along the decision instant's heading y, the ball's frame;
    side_speed_m_s and forward_speed_m_s along the dog's current heading, what the policy tracks; yaw_rate_rad_s) and
    out_dir/summary.json: per run the fall and recovery outcome, the mean yaw rate and, at each of 0.2, 0.4, 0.6 and
    1.0 s, the displacement and the side speed and yaw rate averaged over the five samples within 0.04 s."""
    world = Go2World(config, config["runtime"]["seed"])
    period = world.command_schema["update_period_s"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for vx in forward_speeds:
        for vy in side_speeds:
            obs = [world.reset({"dog_forward_speed_m_s": vx}, attempt_seed=0)]
            obs += [world.step({"vel_x": vx, "vel_y": vy, "yaw_rate": 0.0}, period)
                    for _ in range(round(duration_s / period))]
            outcome = world.finish_episode()
            t = np.array([o["t_s"] for o in obs])
            pos, vel = (np.array([o[k][:2] for o in obs]) for k in ("base_pos_m", "base_lin_vel_m_s"))
            w, x, y, z = np.array([o["base_quat_wxyz"] for o in obs]).T
            yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            moved = pos - pos[0]
            run = {"t_s": t,
                   "side_displacement_m": moved[:, 1] * np.cos(yaw[0]) - moved[:, 0] * np.sin(yaw[0]),
                   "side_speed_m_s": vel[:, 1] * np.cos(yaw) - vel[:, 0] * np.sin(yaw),
                   "forward_speed_m_s": vel[:, 0] * np.cos(yaw) + vel[:, 1] * np.sin(yaw),
                   "yaw_rate_rad_s": np.array([o["base_ang_vel_rad_s"][2] for o in obs])}
            np.savez(out / f"vx{vx}_vy{vy}.npz", **run)

            def at(key, s, window_s):  # mean of the samples within window_s of s
                return round(float(run[key][np.abs(t - s) <= window_s + 1e-9].mean()), 4)
            summary.append({
                "vel_x": vx, "vel_y": vy, "fall": outcome["fall"], "recovery_time_s": outcome["recovery_time_s"],
                "recovered": outcome["recovered"], "mean_yaw_rate_rad_s": round(float(run["yaw_rate_rad_s"].mean()), 4),
                "at_s": {str(s): {"side_displacement_m": at("side_displacement_m", s, 0.0),
                                  "side_speed_m_s": at("side_speed_m_s", s, 0.04),
                                  "yaw_rate_rad_s": at("yaw_rate_rad_s", s, 0.04)} for s in (0.2, 0.4, 0.6, 1.0)}})
            print(json.dumps(summary[-1]), flush=True)
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def save_dodge_example(config: dict, seed: int, side_speed: float, start_delay_s: float, hold_s: float,
                       out_dir: str | Path) -> Path:
    """One DODGE episode run as Person B's SkillSpec runs it (HANDOFF_PERSON_B.md): the first throw of
    Go2World(config, seed), attempt seed 0, then the nominal command (vel_x at the throw's forward speed) for the start
    delay, vel_y side_speed for hold_s, and the nominal command to skills.dodge.episode_duration_s. Copies the trace to
    out_dir/episode.npz and writes out_dir/episode.json with the situation, the commands and the outcome."""
    world = Go2World(config, seed)
    situation = world.sample_dodge_situation()
    world.reset(situation, attempt_seed=0)
    nominal = {"vel_x": situation["dog_forward_speed_m_s"], "vel_y": 0.0, "yaw_rate": 0.0}
    rest = config["skills"]["dodge"]["episode_duration_s"] - start_delay_s - hold_s
    steps = [(c, d) for c, d in ((nominal, start_delay_s), (nominal | {"vel_y": side_speed}, hold_s), (nominal, rest))
             if d > 0]
    for command, duration in steps:
        world.step(command, duration)
    outcome = world.finish_episode()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(outcome["trace_path"], out / "episode.npz")
    episode = {"seed": seed, "preset": world.preset, "attempt_seed": 0, "situation": situation,
               "action": {"side_speed_m_s": side_speed, "start_delay_s": start_delay_s, "hold_duration_s": hold_s},
               "steps": [{"command": c, "duration_s": d} for c, d in steps],
               "outcome": outcome | {"trace_path": "episode.npz"}}
    path = out / "episode.json"
    path.write_text(json.dumps(episode, indent=2))
    return path
