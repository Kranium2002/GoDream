"""Wall-time cost of Go2World on this machine, for planning run times (decision 11 of Task 4).
Run from the repository root: uv run python -X utf8 artifacts/env/research/world_cost.py
Prints: construction time, reset() time (default situation: 1 s spawn settle + 3 s soft start + world.settle_s trot),
step() real-time factor for long and for one-period calls, and finish_episode() time with and without a trace.
"""
import platform
import statistics
import sys
import tempfile
import time

sys.path.insert(0, ".")
from learned_judgment.env import Go2World, load_config  # noqa: E402

config = load_config()
config["world"]["trace_dir"] = tempfile.mkdtemp()  # traces of this measurement are not kept
seed = config["runtime"]["seed"]
forward = {"vel_x": 0.6, "vel_y": 0.0, "yaw_rate": 0.0}
print("python", platform.python_version(), platform.platform(), "| world.settle_s", config["world"]["settle_s"])


def timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def show(label, values, unit="s"):
    print(f"{label}: median {statistics.median(values):.4f} {unit}; runs {[round(v, 4) for v in values]}")


construct = [timed(lambda: Go2World(config, seed))[0] for _ in range(5)]
show("construct Go2World", construct)

world = Go2World(config, seed)
resets = []
for k in range(5):
    resets.append(timed(lambda: world.reset(attempt_seed=k))[0])
    world.finish_episode()
show("reset() default situation", resets)

long_rtf, short_rtf, finish_trace, finish_none = [], [], [], []
for k in range(3):
    world.reset(attempt_seed=k)
    wall, obs = timed(lambda: world.step(forward, 10.0))
    long_rtf.append(obs["t_s"] / wall)
    t_before = obs["t_s"]
    wall, _ = timed(lambda: [world.step(forward, 0.02) for _ in range(500)])
    short_rtf.append((world.step(forward, 0.02)["t_s"] - 0.02 - t_before) / wall)
    finish_trace.append(timed(world.finish_episode)[0])  # 20.02 s of trace written
show("step(command, 10.0) real-time factor", long_rtf, "sim s per wall s")
show("step(command, 0.02) x 500 real-time factor", short_rtf, "sim s per wall s")
show("finish_episode() writing a 20 s trace", finish_trace)

config["world"]["save_traces"] = False
quiet = Go2World(config, seed)
for k in range(3):
    quiet.reset(attempt_seed=k)
    quiet.step(forward, 3.0)
    finish_none.append(timed(quiet.finish_episode)[0])
show("finish_episode() with save_traces false", finish_none)
