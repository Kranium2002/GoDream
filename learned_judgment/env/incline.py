"""RAMP's traversable incline: a slab rising from the floor and a level plateau after it (config.yaml world.incline).

Both are boxes, each on its own mocap body: a mocap body is welded to the world, so it is ground (scene.is_ground)
and never moves in the simulation, but data.mocap_pos/quat place it anew each episode without a recompile. Moving a
box compiled on the world body instead is unsafe: MuJoCo's collision tree of the world body is built at compile time,
and a ball dropped on a box moved that way fell through it (proof: scratchpad poc-w3/reposition.py).
"""
import math

import mujoco
import numpy as np

BODIES = ("incline_slope", "incline_plateau")


def add_incline(spec: mujoco.MjSpec, inc: dict) -> None:
    """The two boxes, parked park_height_m up until place() puts them in front of the dog. Added after every other
    body so the robot's (and the ball's) body and geom indices are the same as without them."""
    sizes = ([inc["length_m"] / 2, inc["width_m"] / 2, inc["thickness_m"] / 2],
             [inc["plateau_length_m"] / 2, inc["width_m"] / 2, inc["thickness_m"] / 2])
    for name, size in zip(BODIES, sizes):
        spec.worldbody.add_body(name=name, mocap=True, pos=[0.0, 0.0, inc["park_height_m"]]).add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, rgba=inc["rgba"])


def place(model: mujoco.MjModel, data: mujoco.MjData, inc: dict, deg: float, xy, yaw: float) -> tuple[float, float]:
    """Put the slope's foot start_x_m ahead of floor point xy along heading yaw (rad), rising at deg towards it, and
    the plateau on top of its far end. Returns (ramp_start_m, ramp_end_m), the floor distances from xy to where the
    slope begins and ends."""
    a, length, half_t = math.radians(deg), inc["length_m"], inc["thickness_m"] / 2
    ahead, up = np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([0.0, 0.0, 1.0])
    origin = np.array([xy[0], xy[1], 0.0])
    start, end, rise = inc["start_x_m"], inc["start_x_m"] + length * math.cos(a), length * math.sin(a)
    normal = -math.sin(a) * ahead + math.cos(a) * up  # of the slope's top face
    turn = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])  # about world z
    tilt, slope_quat = np.array([math.cos(a / 2), 0.0, -math.sin(a / 2), 0.0]), np.empty(4)  # nose up about y
    mujoco.mju_mulQuat(slope_quat, turn, tilt)
    poses = ((origin + (start + length / 2 * math.cos(a)) * ahead + length / 2 * math.sin(a) * up - half_t * normal,
              slope_quat),
             (origin + (end + inc["plateau_length_m"] / 2) * ahead + (rise - half_t) * up, turn))
    for name, (pos, quat) in zip(BODIES, poses):
        mocap = model.body_mocapid[model.body(name).id]
        data.mocap_pos[mocap], data.mocap_quat[mocap] = pos, quat
    return start, end


def sample_ramp_situations(config: dict, rng: np.random.Generator, n: int) -> list[dict]:
    """n RAMP situations {"incline_deg": float}, uniform over config.yaml world.incline.range_deg, from rng."""
    lo, hi = config["world"]["incline"]["range_deg"]
    return [{"incline_deg": float(v)} for v in rng.uniform(lo, hi, n)]
