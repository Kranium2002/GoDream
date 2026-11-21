"""What the ball costs in real-time factor (Task 5, decision 6): step() of a 3 s DODGE episode with a thrown ball against
the same episode without one (same seed, attempt seed and forward speed), alternating so that machine load hits both.
Simulated seconds per wall second, and per CPU second of this process (time.process_time), which other processes on a
busy machine disturb far less.
Run from the repository root: uv run python -X utf8 artifacts/env/research/ball_cost.py
"""
import statistics
import sys
import tempfile
import time

sys.path.insert(0, ".")
from learned_judgment.env import Go2World, load_config  # noqa: E402

config = load_config()
config["world"]["trace_dir"] = tempfile.mkdtemp()  # traces of this measurement are not kept
world = Go2World(config, seed=11)
rtf = {"ball": [], "no ball": []}
cpu = {"ball": [], "no ball": []}
for k in range(8):
    throw = world.sample_dodge_situation()
    trot = {"vel_x": throw["dog_forward_speed_m_s"], "vel_y": 0.0, "yaw_rate": 0.0}
    for label, situation in (("ball", throw), ("no ball", {"dog_forward_speed_m_s": throw["dog_forward_speed_m_s"]})):
        world.reset(situation, attempt_seed=k)
        t0, c0 = time.perf_counter(), time.process_time()
        obs = world.step(trot, 3.0)
        rtf[label].append(obs["t_s"] / (time.perf_counter() - t0))
        cpu[label].append(obs["t_s"] / (time.process_time() - c0))
        world.finish_episode()
for name, measured in (("wall", rtf), ("CPU", cpu)):
    for label, values in measured.items():
        print(f"step(command, 3.0) simulated s per {name} s, {label}: median {statistics.median(values):.3f}; "
              f"runs {[round(v, 3) for v in values]}")
    ratio = [b / n for b, n in zip(measured["ball"], measured["no ball"])]
    print(f"{name}: ball / no ball, per pair: median {statistics.median(ratio):.3f}; runs {[round(v, 3) for v in ratio]}")
