"""Watch one Go2 episode live in MuJoCo's interactive window, at real-time pace.

    uv run python -m app.view_episode [--seed N] [--preset NAME] [--situation FILE.json]
                                      [--command vel_x,vel_y,yaw_rate,duration_s ...] [--max-seconds S]

The episode runs through Go2World exactly as the engine runs it (reset with the situation, then the commands in turn);
the window only shows it, synced after every control period. The outcome is printed at the end.
"""
import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from learned_judgment.env import Go2World, load_config, validate_command


class StopViewing(Exception):
    """Raised from step's on_period callback when the window is closed or --max-seconds is reached."""


def parse_command(text: str) -> tuple[dict, float]:
    vel_x, vel_y, yaw_rate, duration_s = map(float, text.split(","))
    if not 0.0 < duration_s < math.inf:
        raise ValueError(f"duration_s must be finite and > 0, got {duration_s}")
    return {"vel_x": vel_x, "vel_y": vel_y, "yaw_rate": yaw_rate}, duration_s


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(prog="python -m app.view_episode", description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=config["runtime"]["seed"])
    parser.add_argument("--preset", default="default")
    parser.add_argument("--situation", type=Path, help="JSON file holding a situation dict, as Go2World.reset takes it")
    parser.add_argument("--command", type=parse_command, action="append",
                        help="vel_x,vel_y,yaw_rate,duration_s (m/s, m/s, rad/s, s); repeat for a sequence. "
                             "Default: 0.6,0,0,5")
    parser.add_argument("--max-seconds", type=float, help="close the window after this many wall-clock seconds")
    args = parser.parse_args()
    commands = args.command or [parse_command("0.6,0,0,5")]

    world = Go2World(config, args.seed, args.preset)
    for command, _ in commands:
        validate_command(command, world.command_schema)  # before the window opens
    world.reset(json.loads(args.situation.read_text(encoding="utf-8")) if args.situation else None)

    with mujoco.viewer.launch_passive(world.model, world.policy.data) as viewer:
        with viewer.lock():  # follow the trunk, as render_frame does
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = world.model.body("base_link").id
        start, periods = time.perf_counter(), 0

        def on_period(_obs: dict) -> None:
            """Show the period just simulated, keep real-time pace, and stop when the window closes or time is up."""
            nonlocal periods
            periods += 1
            viewer.sync()
            time.sleep(max(0.0, start + periods * world.policy.control_dt - time.perf_counter()))
            if not viewer.is_running() or (args.max_seconds is not None and time.perf_counter() - start > args.max_seconds):
                raise StopViewing

        try:
            for command, duration_s in commands:
                world.step(command, duration_s, on_period=on_period)
        except StopViewing:  # step stopped after a whole period; the episode is finished below as it stands
            pass

    print(json.dumps(world.finish_episode(), indent=2))


if __name__ == "__main__":
    main()
