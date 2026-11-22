"""A smooth mound on the floor: a spherical cap (Go2World situation keys MOUND_KEYS; config.yaml world.mound).

The cap of footprint radius a and top height h is the part above the floor of a sphere of radius R = (a^2 + h^2) / (2 h)
whose centre lies h - R below the floor (R > h, so below it). Its surface is smooth everywhere; the slope is 0 on top
and grows towards the rim, where it is steepest: asin(a / R). The sphere is ONE geom on a mocap body: welded to the
world, so it is ground (scene.is_ground) like the floor and the incline, never moves in the simulation, and
data.mocap_pos places it anew each episode without a recompile (as incline.py). Its radius changes per episode too: the
geom's size, bounding radius, bounding box and its body's single bounding-volume node are set together, which are the
only compiled arrays a sphere's radius changes besides the body's mass (the geom is massless, a mocap body needs none)
and the visual statistics (model.stat). The part below the floor is hidden by the floor; MuJoCo does not collide two
bodies welded to the world, so the sphere and the floor plane never touch each other.
"""
import math

import mujoco
import numpy as np

BODY = "mound"
MOUND_KEYS = ("mound_x_m", "mound_y_m", "mound_radius_m", "mound_height_m")


def sphere_radius(radius_m: float, height_m: float) -> float:
    """R of the cap of footprint radius a and height h: (a^2 + h^2) / (2 h)."""
    return (radius_m * radius_m + height_m * height_m) / (2.0 * height_m)


def rim_slope_deg(radius_m: float, height_m: float) -> float:
    """The cap's steepest slope, at its rim: asin(a / R), degrees."""
    return math.degrees(math.asin(min(1.0, radius_m / sphere_radius(radius_m, height_m))))


def height_for_slope(radius_m: float, slope_deg: float) -> float:
    """The cap height h whose rim slope is slope_deg for footprint radius a: R = a / sin(slope), h = R (1 - cos(slope))."""
    s = math.radians(slope_deg)
    return radius_m / math.sin(s) * (1.0 - math.cos(s))


def check(mound: dict, radius_m: float, height_m: float, x_m: float, y_m: float) -> None:
    """ValueError naming the limit of config.yaml world.mound that the mound breaks, if any."""
    lo, hi = mound["radius_m"]
    if not lo <= radius_m <= hi:
        raise ValueError(f"mound_radius_m must be from {lo} to {hi} (world.mound.radius_m), got {radius_m}")
    lo, hi = mound["height_m"]
    if not lo <= height_m <= hi:
        raise ValueError(f"mound_height_m must be from {lo} to {hi} (world.mound.height_m), got {height_m}")
    if height_m >= radius_m:
        raise ValueError(f"mound_height_m ({height_m}) must be below mound_radius_m ({radius_m}): a cap under a half sphere")
    slope = rim_slope_deg(radius_m, height_m)
    if slope > mound["max_rim_slope_deg"]:
        raise ValueError(f"the mound's rim slope {slope:.2f} deg (radius {radius_m} m, height {height_m} m) is above "
                         f"world.mound.max_rim_slope_deg ({mound['max_rim_slope_deg']})")
    gap = math.hypot(x_m, y_m) - radius_m
    if gap < mound["min_gap_m"]:
        raise ValueError(f"the mound's rim is {gap:.3f} m from the trunk origin at the decision instant; world.mound.min_gap_m "
                         f"is {mound['min_gap_m']} (the mound must not appear under the dog)")


def add_mound(spec: mujoco.MjSpec, mound: dict) -> None:
    """The massless sphere on its mocap body, parked park_height_m up with radius park_radius_m until place() puts it
    in front of the dog. Added after every other body, so the robot's (and the ball's) indices do not change."""
    spec.worldbody.add_body(name=BODY, mocap=True, pos=[0.0, 0.0, mound["park_height_m"]]).add_geom(
        name=BODY, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[mound["park_radius_m"], 0, 0], mass=0.0, rgba=mound["rgba"])


def _set_radius(model: mujoco.MjModel, sphere_r: float) -> None:
    g, b = model.geom(BODY).id, model.body(BODY).id
    model.geom_size[g, 0] = model.geom_rbound[g] = sphere_r
    model.geom_aabb[g, 3:6] = sphere_r
    model.bvh_aabb[model.body_bvhadr[b], 3:6] = sphere_r  # its only node (body_bvhnum 1): the sphere's box, body frame


def park(model: mujoco.MjModel, mound: dict) -> None:
    """Back to the compiled radius (the parked pose is restored by mj_resetData, the compiled body position)."""
    _set_radius(model, mound["park_radius_m"])


def place(model: mujoco.MjModel, data: mujoco.MjData, x_m: float, y_m: float, radius_m: float, height_m: float, xy,
          yaw: float) -> dict:
    """Put the cap's centre x_m ahead of and y_m to the left of floor point xy along heading yaw (rad). Returns the
    placed mound in the world frame: {"centre_xy_m", "sphere_radius_m", "rim_slope_deg"}."""
    r = sphere_radius(radius_m, height_m)
    ahead, left = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    centre = np.asarray(xy, dtype=np.float64)[:2] + x_m * ahead + y_m * left
    _set_radius(model, r)
    mocap = model.body_mocapid[model.body(BODY).id]
    data.mocap_pos[mocap] = [centre[0], centre[1], height_m - r]
    data.mocap_quat[mocap] = [1.0, 0.0, 0.0, 0.0]
    return {"centre_xy_m": centre.tolist(), "sphere_radius_m": r, "rim_slope_deg": rim_slope_deg(radius_m, height_m)}
