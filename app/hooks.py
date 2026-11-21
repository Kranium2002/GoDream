"""Hooks the dashboard calls to show real Go2 episodes."""
import math
import os

import numpy as np

from learned_judgment.env import Go2World


def record_episode(config: dict, seed: int, situation: dict | None, commands: list[tuple[dict, float]],
                   fps: int = 25) -> dict:
    """Run one episode through Go2World(config, seed) (default preset) and film it.

    reset(situation), then each (command, duration_s) in turn, held for duration_s rounded to whole control periods
    as Go2World.step does. Returns:
        frames        (T, H, W, 3) uint8, world.render_frame(); frame i shows episode time i / fps, from the decision
                      instant up to the last frame period before the end (2 s at 25 fps: 50 frames).
        observations  The observation at the end of each command, one per command.
        outcome       world.finish_episode(). Each command is sent in one step() call per frame period, so
                      commands_received has an entry per call.
    fps must divide the control rate (50 Hz) into whole control periods, else ValueError.
    """
    world = Go2World(config, seed)
    dt = world.policy.control_dt
    every = round(1 / (fps * dt)) if fps > 0 else 0  # control periods per frame
    if every < 1 or not math.isclose(every * fps * dt, 1.0):
        raise ValueError(f"fps must divide the {1 / dt:g} Hz control rate into whole control periods, got {fps}")
    world.reset(situation)
    frames, observations, n = [], [], 0  # n: control periods since the decision instant
    for command, duration_s in commands:
        if not 0.0 < duration_s < math.inf:  # checked here: the split below would hide a bad duration from step()
            raise ValueError(f"duration_s must be finite and > 0, got {duration_s}")
        end = n + max(1, round(duration_s / dt))
        while n < end:
            if n % every == 0:
                try:
                    frames.append(world.render_frame())
                except Exception as e:  # never substitute an image: stop, naming the GL backend
                    backend = os.environ.get("MUJOCO_GL") or "glfw (MUJOCO_GL unset)"
                    raise RuntimeError(f"offscreen rendering failed on MuJoCo's OpenGL backend {backend}") from e
            k = min(every - n % every, end - n)  # up to the next frame or the end of the command
            obs = world.step(command, k * dt)
            n += k
        observations.append(obs)
    return {"frames": np.stack(frames), "observations": observations, "outcome": world.finish_episode()}
