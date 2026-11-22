"""Hooks the dashboard calls to show real Go2 episodes."""
import math
import os

import numpy as np

from learned_judgment.env import Go2World


def record_episode(config: dict, seed: int, situation: dict | None, commands: list[tuple[dict, float]],
                   fps: int = 25) -> dict:
    """Run one episode through Go2World(config, seed) (default preset) and film it.

    reset(situation), then world.step(command, duration_s) for each pair in turn, exactly as the engine runs it; the
    frames are taken by step's on_period callback. Returns:
        frames        (T, H, W, 3) uint8, world.render_frame(); frame i shows episode time i / fps, from the decision
                      instant up to the last frame time before the end (2 s at 25 fps: 50 frames).
        observations  The observation at the end of each command, one per command.
        outcome       world.finish_episode().
    fps must divide the control rate (50 Hz) into whole control periods, else ValueError.
    """
    world = Go2World(config, seed)
    dt = world.policy.control_dt
    every = round(1 / (fps * dt)) if fps > 0 else 0  # control periods per frame
    if every < 1 or not math.isclose(every * fps * dt, 1.0):
        raise ValueError(f"fps must divide the {1 / dt:g} Hz control rate into whole control periods, got {fps}")

    def render() -> np.ndarray:
        try:
            return world.render_frame()
        except Exception as e:  # never substitute an image: stop, naming the GL backend
            backend = os.environ.get("MUJOCO_GL") or "glfw (MUJOCO_GL unset)"
            raise RuntimeError(f"offscreen rendering failed on MuJoCo's OpenGL backend {backend}") from e

    frames, observations, periods = [], [], 0  # periods: control periods since the decision instant
    due = None  # the frame of the latest frame time, kept once the episode goes on past that time

    def on_period(_obs: dict) -> None:
        nonlocal periods, due
        if due is not None:
            frames.append(due)
        periods += 1
        due = render() if periods % every == 0 else None

    world.reset(situation)
    due = render()
    for command, duration_s in commands:
        observations.append(world.step(command, duration_s, on_period=on_period))
    return {"frames": np.stack(frames), "observations": observations, "outcome": world.finish_episode()}
