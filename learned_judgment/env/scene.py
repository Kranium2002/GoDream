"""The simulated world: the pinned Unitree Go2 on a flat, infinite floor."""
import mujoco


def build_model(config: dict, hidden: dict | None = None) -> mujoco.MjModel:
    """Unitree's go2.xml plus a floor plane and a light. Unitree's scene.xml is not used: it puts bars and a
    staircase on the +x path. `hidden` is not used yet."""
    # from_file resolves go2.xml's meshdir="assets" against the file's own directory.
    spec = mujoco.MjSpec.from_file(config["robot"]["model_asset"]["path"])
    # Same floor and light as Unitree's scene.xml (lines 20-21), without its material.
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    return spec.compile()
