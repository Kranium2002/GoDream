"""The simulated world: the pinned Unitree Go2 on a flat, infinite floor."""
import mujoco


def build_model(config: dict, hidden: dict | None = None, ball: bool = False) -> mujoco.MjModel:
    """Unitree's go2.xml plus a floor plane and a light, and with ball=True the thrown ball. Unitree's scene.xml is not
    used: it puts bars and a staircase on the +x path. `hidden` is not used yet."""
    # from_file resolves go2.xml's meshdir="assets" against the file's own directory.
    spec = mujoco.MjSpec.from_file(config["robot"]["model_asset"]["path"])
    # Same floor and light as Unitree's scene.xml (lines 20-21), without its material.
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    if ball:
        _add_ball(spec, config["ball"])
    return spec.compile()


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
