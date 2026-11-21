"""The simulated world: the pinned Unitree Go2 on a flat, infinite floor, with an optional incline."""
import math

import mujoco
import numpy as np

from learned_judgment.env.incline import add_incline


def build_model(config: dict, hidden: dict | None = None, ball: bool = False, incline: bool = False) -> mujoco.MjModel:
    """Unitree's go2.xml plus a floor plane and a light, with ball=True the thrown ball and with incline=True the
    parked ramp and plateau (incline.add_incline), in that order, after the robot. Unitree's scene.xml is not
    used: it puts bars and a staircase on the +x path. `hidden` (Go2World's hidden effects) adds a payload box on the
    trunk, the floor's friction and a slope; with none of them the model is the plain scene."""
    hidden = hidden or {}
    # from_file resolves go2.xml's meshdir="assets" against the file's own directory.
    spec = mujoco.MjSpec.from_file(config["robot"]["model_asset"]["path"])
    # Same floor and light as Unitree's scene.xml (lines 20-21), without its material.
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    if hidden.get("payload", {}).get("mass_kg", 0) > 0:
        # A box resting on the top face of the trunk's collision box, offset in x, y; it adds mass and touches nothing.
        trunk = spec.body("base_link")
        top = max(g.pos[2] + g.size[2] for g in trunk.geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX)
        half = config["world"]["hidden_effects"]["payload_box_half_size_m"]
        x, y = hidden["payload"]["offset_m"]
        trunk.add_body(name="payload", pos=[x, y, top + half[2]]).add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=half, mass=hidden["payload"]["mass_kg"], contype=0, conaffinity=0)
    if "friction" in hidden:
        # The foot geoms' contact priority 1 makes their friction, not the floor's, that of every foot contact (E5).
        for foot in ("FL", "FR", "RL", "RR"):
            spec.geom(foot).friction[0] = hidden["friction"]
    if "slope" in hidden:
        # shortcut: exact only for an infinite flat floor, replace with ramp geometry when RAMP needs a visible incline
        # Positive is uphill along +x, the dog's heading at the spawn: gravity gets a -x component.
        g, slope = -spec.option.gravity[2], math.radians(hidden["slope"])
        spec.option.gravity = [-g * math.sin(slope), 0.0, -g * math.cos(slope)]
    if ball:
        _add_ball(spec, config["ball"])
    if incline:
        add_incline(spec, config["world"]["incline"])
    return spec.compile()


def is_ground(model: mujoco.MjModel) -> np.ndarray:
    """Per geom: True if its body is welded to the world (the floor, and the incline's mocap boxes)."""
    return model.body_weldid[model.geom_bodyid] == 0


def _add_ball(spec: mujoco.MjSpec, ball: dict) -> None:
    """A free body after the robot, so the robot's free joint stays the first one (Efferent takes the first free
    joint as the base) and its qpos, qvel, body and geom indices do not change. It starts 2 m up: Efferent's spawn
    lowers the robot onto its lowest non-world geom, which must stay a robot geom."""
    body = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, 2.0])
    body.add_freejoint()
    body.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[ball["radius_m"], 0, 0], mass=ball["mass_kg"],
                  typeinertia=getattr(mujoco.mjtGeomInertia, f"mjINERTIA_{ball['inertia'].upper()}"),  # shell, volume
                  friction=ball["friction"], condim=ball["condim"], priority=ball["priority"], solref=ball["solref"],
                  contype=ball["collision_masks"]["contype"], conaffinity=ball["collision_masks"]["conaffinity"])
