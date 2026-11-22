"""The robot description (app/robot.py) built from the real compiled Go2 MuJoCo model, and the body poses."""
import base64

import mujoco
import numpy as np
import pytest

from app import robot
from learned_judgment.env import Go2World, load_config


@pytest.fixture(scope="module")
def world():
    return Go2World(load_config(), 1)


@pytest.fixture(scope="module")
def described(world):
    return robot.describe(world.model)


def test_bodies_are_the_models_in_id_order_with_their_parents(world, described):
    m = world.model
    assert [b["id"] for b in described["bodies"]] == list(range(m.nbody))
    for b in described["bodies"]:
        assert b["name"] == mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b["id"]) or b["name"] == ""
        assert b["parent"] == int(m.body_parentid[b["id"]])
    assert described["up"] == "z" and described["units"] == "m"


def test_only_visual_robot_geoms_and_the_world_are_drawn(world, described):
    m = world.model
    robot_root = m.body_rootid[m.body("base_link").id]
    for g in described["geoms"]:
        if m.body_rootid[g["body"]] == robot_root:
            assert m.geom_group[g["id"]] == 2 and m.geom_contype[g["id"]] == 0  # go2.xml class "visual"
    drawn = {g["id"] for g in described["geoms"]}
    collision = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == robot_root and m.geom_group[g] == 3}
    assert collision and not drawn & collision
    assert m.geom("floor").id in drawn
    assert sum(1 for g in described["geoms"] if g["type"] == "mesh") == sum(
        1 for g in range(m.ngeom) if m.geom_group[g] == 2)


def test_geoms_carry_their_pose_in_the_body_frame_size_and_colour(world, described):
    m = world.model
    for g in described["geoms"]:
        i = g["id"]
        assert g["body"] == int(m.geom_bodyid[i]) and g["group"] == int(m.geom_group[i])
        assert np.allclose(g["pos"], m.geom_pos[i]) and np.allclose(g["quat"], m.geom_quat[i])
        assert np.allclose(g["size"], m.geom_size[i]) and len(g["rgba"]) == 4
        assert g["type"] in ("mesh", "sphere", "box", "capsule", "cylinder", "ellipsoid", "plane")
        expected = m.mat_rgba[m.geom_matid[i]] if m.geom_matid[i] >= 0 else m.geom_rgba[i]
        assert np.allclose(g["rgba"], expected)


def test_trained_rgba_recolours_only_the_named_materials(world, described):
    m, pink = world.model, [1.0, 0.31, 0.64, 1.0]
    trained = robot.describe(m, {"black": pink})["geoms"]
    black = m.material("black").id
    assert any(m.geom_matid[g["id"]] == black for g in trained)
    for g, plain in zip(trained, described["geoms"]):
        assert g["rgba"] == plain["rgba"] and plain["trained_rgba"] == plain["rgba"]  # the model's colours stay
        assert g["trained_rgba"] == (pink if m.geom_matid[g["id"]] == black else g["rgba"])


def test_meshes_decode_to_the_models_vertices_and_faces(world, described):
    m = world.model
    used = {g["mesh"] for g in described["geoms"] if g["mesh"] is not None}
    assert used == {mesh["id"] for mesh in described["meshes"]}
    for mesh in described["meshes"]:
        i = mesh["id"]
        vertices = np.frombuffer(base64.b64decode(mesh["vertices"]), "<f4").reshape(-1, 3)
        faces = np.frombuffer(base64.b64decode(mesh["faces"]), "<u4").reshape(-1, 3)
        start, n = m.mesh_vertadr[i], m.mesh_vertnum[i]
        assert np.array_equal(vertices, m.mesh_vert[start:start + n])
        f0, nf = m.mesh_faceadr[i], m.mesh_facenum[i]
        assert np.array_equal(faces, m.mesh_face[f0:f0 + nf]) and faces.max() < n


def test_a_thrown_ball_is_drawn_once_its_scene_is_used(world):
    world.reset(world.sample_dodge_situation())
    m = world.model
    described = robot.describe(m)
    assert m.geom("ball").id in {g["id"] for g in described["geoms"]}
    world.finish_episode()


def test_poses_are_every_bodys_world_pose_from_the_data(world):
    world.reset()
    world.step({"vel_x": 0.6, "vel_y": 0.0, "yaw_rate": 0.0}, 0.2)
    data = world.policy.data
    pose = robot.pose(data, t_s=0.2)
    assert pose["t_s"] == 0.2 and len(pose["pos"]) == len(pose["quat"]) == world.model.nbody
    assert np.allclose(pose["pos"], data.xpos) and np.allclose(pose["quat"], data.xquat)
    world.finish_episode()


def test_the_trained_label_is_passed_on_only_when_configured(world, described):
    label = {"text": "tiramisu", "rgba": [0.24, 0.13, 0.07, 1.0], "centre_m": [0.0, 0.099, 0.034], "size_m": [0.2, 0.05]}
    assert robot.describe(world.model, None, label)["trained_label"] == label
    assert "trained_label" not in described  # none configured: the answer is as before
