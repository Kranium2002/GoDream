"""The simulated world is the pinned Go2 on a flat floor, without Unitree's obstacle scene."""
import mujoco

from learned_judgment.env.scene import build_model


def test_flat_scene_holds_the_go2_and_only_a_floor(config):
    model = build_model(config)
    assert model.nu == 12
    world_geoms = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                   for g in range(model.ngeom) if model.geom_bodyid[g] == 0]
    # scene.xml adds 8 unnamed boxes (bars, stairs) to the world body; only the floor plane may be there.
    assert world_geoms == ["floor"]
    assert model.geom_type[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")] == mujoco.mjtGeom.mjGEOM_PLANE
    assert model.nlight == 1
