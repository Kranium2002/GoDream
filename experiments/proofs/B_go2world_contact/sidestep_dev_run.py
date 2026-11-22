"""First contact between Person B's code and Person A's real Go2World (no ball).

DEV RUN on the real Go2 simulator. It sends the command sequence the DODGE SkillSpec sends
(trot, a sideways command for the hold time, trot) and compares the trunk's measured
sideways travel with the dog-motion part of the DODGE physics guess. It is a development
check of the formula's constants and of the interface. It is not an evaluation result and
goes into no report or table.

Run from the repository root:

    .venv\\Scripts\\python.exe experiments\\proofs\\B_go2world_contact\\sidestep_dev_run.py
"""

from __future__ import annotations

import copy
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learned_judgment.config import config_digest, load_config, require  # noqa: E402
from learned_judgment.engine.streams import stream_seed  # noqa: E402
from learned_judgment.skills.base import CommandSchema  # noqa: E402

print("DEV RUN on the real Go2 simulator (Person A's Go2World, no ball) - development check of the "
      "DODGE dog-motion formula and of the interface; not an evaluation result, not in any report")

config = load_config()
schema = CommandSchema.from_config(config)
action = require(config, "skills.dodge.action")
forward_field, side_field, yaw_field = (
    action["forward_velocity_field"], action["side_velocity_field"], action["yaw_rate_field"])
latency = float(require(config, "skills.dodge.physics.assumed_latency_s"))
tau = float(require(config, "skills.dodge.physics.response_time_s"))
delay_range = require(config, "skills.dodge.action.start_delay_s")
hold_range = require(config, "skills.dodge.action.hold_duration_s")
base_seed = int(require(config, "runtime.seed"))
world_seed = stream_seed(base_seed, "dodge/dev/sidestep_contact/world")

from learned_judgment.env import Go2World  # noqa: E402  (Person A's public interface)

# The cases are the corners of the DODGE action space at three forward speeds. Durations are
# members of the configured action ranges; the tail lets the sideways speed decay.
delay_s = float(delay_range[0]) + 0.5 * (float(delay_range[1]) - float(delay_range[0])) / 1.5  # inside the range
delay_s = round(delay_s / schema.update_period_s) * schema.update_period_s
hold_s = float(hold_range[0]) + 0.5 * (float(hold_range[1]) - float(hold_range[0]))
hold_s = round(hold_s / schema.update_period_s) * schema.update_period_s
tail_s = 1.0
side_low, side_high = schema.field(side_field).low, schema.field(side_field).high
cases = [(forward, side) for forward in (0.3, 0.6, 0.9) for side in (side_high, side_low, 0.5 * side_high, 0.5 * side_low)]
cases.append((0.6, 0.0))  # no dodge: the drift a zero sideways command produces

print(f"config digest {config_digest(config)[:12]} | world seed {world_seed} | preset default")
print(f"command fields: forward={forward_field} side={side_field} yaw={yaw_field} | update period {schema.update_period_s} s")
print(f"sequence per case: trot {delay_s:.2f} s, sideways command {hold_s:.2f} s, trot {tail_s:.2f} s")
print(f"formula constants from config: assumed_latency_s {latency}, response_time_s {tau}")


def formula_travel(side: float, t: np.ndarray) -> np.ndarray:
    """Sideways travel the DODGE physics guess assumes (same closed form as skills/dodge.py)."""
    start, stop = delay_s + latency, delay_s + latency + hold_s
    rise = np.clip(t - start, 0.0, hold_s)
    decay = np.maximum(t - stop, 0.0)
    rise_left = np.exp(-rise / tau)
    return side * (rise - tau * (1.0 - rise_left)) + side * (1.0 - rise_left) * tau * (1.0 - np.exp(-decay / tau))


def yaw_of(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (quat_wxyz[..., i] for i in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


run_config = copy.deepcopy(config)
rows = []
with tempfile.TemporaryDirectory() as trace_dir:
    run_config["world"]["trace_dir"] = trace_dir
    run_config["world"]["save_traces"] = True
    started = time.perf_counter()
    world = Go2World(run_config, world_seed, "default")
    print(f"world built in {time.perf_counter() - started:.2f} s")
    for index, (forward, side) in enumerate(cases):
        nominal = {forward_field: float(forward), side_field: 0.0, yaw_field: 0.0}
        dodge = {forward_field: float(forward), side_field: float(side), yaw_field: 0.0}
        for command in (nominal, dodge):
            schema.check_command(command)
        wall = time.perf_counter()
        observation = world.reset({"dog_forward_speed_m_s": float(forward)}, attempt_seed=index)
        declared = (observation["declared_payload_kg"], observation["declared_surface"])
        world.step(nominal, delay_s)
        world.step(dodge, hold_s)
        world.step(nominal, tail_s)
        outcome = world.finish_episode()
        wall = time.perf_counter() - wall
        with np.load(outcome["trace_path"]) as trace:
            t = trace["t_s"]
            position = trace["base_pos_m"]
            yaw = yaw_of(trace["base_quat_wxyz"])
        heading = yaw[0]
        dx, dy = position[:, 0] - position[0, 0], position[:, 1] - position[0, 1]
        sideways = -math.sin(heading) * dx + math.cos(heading) * dy  # + is to the dog's left at the decision instant
        forward_travel = math.cos(heading) * dx + math.sin(heading) * dy
        predicted = formula_travel(side, t)
        at_hold_end = int(np.argmin(np.abs(t - (delay_s + hold_s))))
        rows.append({
            "forward": forward, "side": side, "fall": outcome["fall"], "unstable": outcome["sim_unstable"],
            "elapsed": outcome["elapsed_s"], "commands": len(outcome["commands_received"]),
            "measured_hold_end": float(sideways[at_hold_end]), "formula_hold_end": float(predicted[at_hold_end]),
            "measured_final": float(sideways[-1]), "formula_final": float(predicted[-1]),
            "yaw_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
            "forward_speed": float(forward_travel[-1] / t[-1]) if t[-1] > 0 else float("nan"),
            "declared": declared, "wall": wall,
        })

print("\nMeasured on the simulator against the formula. Sideways travel in metres, + is to the dog's left.")
print(f"{'forward':>8} {'side':>6} | {'end of hold':^23} | {'end of episode':^23} | {'yaw change':>10} {'fwd speed':>9} {'fall':>4} {'wall s':>6}")
print(f"{'cmd m/s':>8} {'cmd':>6} | {'measured':>10} {'formula':>10}   | {'measured':>10} {'formula':>10}   | {'deg':>10} {'m/s':>9}")
for r in rows:
    print(f"{r['forward']:8.2f} {r['side']:6.2f} | {r['measured_hold_end']:10.3f} {r['formula_hold_end']:10.3f}   | "
          f"{r['measured_final']:10.3f} {r['formula_final']:10.3f}   | {r['yaw_change_deg']:10.1f} {r['forward_speed']:9.3f} "
          f"{r['fall']:4d} {r['wall']:6.2f}")

interface_ok = all(r["commands"] == 3 and r["unstable"] == 0 for r in rows)
print(f"\ninterface: every episode took the three commands and stayed stable: {interface_ok}")
print(f"declared values reported by the world: {sorted({r['declared'] for r in rows})}")
print(f"episodes with a fall: {sum(r['fall'] for r in rows)} of {len(rows)}")
dodges = [r for r in rows if r["side"] != 0.0]
ratio = [r["measured_final"] / r["formula_final"] for r in dodges if r["formula_final"] != 0.0]
print(f"measured / formula sideways travel at the end of the episode: min {min(ratio):.2f}, median {float(np.median(ratio)):.2f}, max {max(ratio):.2f}")
print("The gap between the two columns is what the residual model has to learn; it is not an error of the simulator.")
sys.exit(0 if interface_ok else 1)
