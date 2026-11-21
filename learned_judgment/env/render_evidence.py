"""Save a head-camera and a tracking frame of the same state, and measure what one frame of each costs.

    python -m learned_judgment.env.render_evidence [--out artifacts/env/head-view]

The state: a punch thrown from 2 m ahead at head height at 1.5 m/s at the dog standing on the zero command, 0.6 s
into its flight (seed 3, default preset). Writes head.png (96 x 96), head_large.png (480 x 480, the same camera),
tracking.png (640 x 480) and render_cost.json, and prints the costs. Simulator frames only; no camera image.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from app.simulation import encode_png
from learned_judgment.env import Go2World, load_config


def cost_ms(render, n: int) -> dict:
    render()  # the first call makes the renderer
    times = []
    for _ in range(n):
        t = time.perf_counter()
        render()
        times.append((time.perf_counter() - t) * 1e3)
    return {"n": n, "median_ms": statistics.median(times), "mean_ms": statistics.fmean(times), "max_ms": max(times)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/env/head-view"))
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    config = load_config()
    world = Go2World(config, seed=3)
    world.reset({"ball_rel_x_m": 2.34, "ball_rel_y_m": 0.0, "ball_rel_z_m": 0.017, "ball_vel_x_m_s": -1.5,
                 "ball_vel_y_m_s": 0.0, "ball_vel_z_m_s": 0.0, "object_kind": "punch"})
    world.step({"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}, 0.6)
    frames = {"head.png": world.render_frame(camera="head"),
              "head_large.png": world.render_frame(camera="head", size=(480, 480)),
              "tracking.png": world.render_frame()}
    for name, rgb in frames.items():
        (out / name).write_bytes(encode_png(rgb, 6))
    costs = {"head_96x96": cost_ms(lambda: world.render_frame(camera="head"), 300),
             "tracking_640x480": cost_ms(world.render_frame, 30)}
    world.finish_episode()
    (out / "render_cost.json").write_text(json.dumps(costs, indent=2))
    print(json.dumps({"frames": {k: list(v.shape) for k, v in frames.items()}, "cost": costs}, indent=2))


if __name__ == "__main__":
    main()
