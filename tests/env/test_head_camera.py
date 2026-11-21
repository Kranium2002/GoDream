"""The head camera, the frames after reset and on_physics_step, on the real simulator and walking policy."""
import math

import mujoco
import numpy as np
import pytest

from learned_judgment.env import Go2World

pytestmark = pytest.mark.system


def ahead(x, y, z, vx=0.0, kind=None):
    """Six ball keys: the object at (x, y, z) in the decision-instant frame, moving at vx along it."""
    s = {"ball_rel_x_m": x, "ball_rel_y_m": y, "ball_rel_z_m": z, "ball_vel_x_m_s": vx, "ball_vel_y_m_s": 0.0,
         "ball_vel_z_m_s": 0.0}
    return s | ({"object_kind": kind} if kind else {})


def test_the_head_camera_sits_on_the_trunk_and_looks_along_its_x(config, world):
    head = config["world"]["render"]["head_camera"]
    world.reset({"dog_forward_speed_m_s": 0.6})  # trotted: the trunk has turned away from the world axes
    m, d = world.model, mujoco.MjData(world.model)
    d.qpos[:] = world.policy.data.qpos
    mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    cam, base = m.camera("head").id, m.body("base_link").id
    assert m.cam_bodyid[cam] == base and m.cam_fovy[cam] == head["fovy_deg"]
    trunk = d.xmat[base].reshape(3, 3)
    assert trunk.T @ (d.cam_xpos[cam] - d.xpos[base]) == pytest.approx(head["pos_m"], abs=1e-9)
    assert -d.cam_xmat[cam].reshape(3, 3)[:, 2] == pytest.approx(trunk[:, 0], abs=1e-9)  # a camera looks along -z


def test_head_frames_have_their_size_and_differ_from_the_tracking_view(config, world):
    head = config["world"]["render"]["head_camera"]
    world.reset()
    frame = world.render_frame(camera="head")
    assert frame.shape == (head["height"], head["width"], 3) and frame.dtype == np.uint8
    tracking = world.render_frame()
    large = world.render_frame(camera="head", size=tracking.shape[:2])
    assert large.shape == tracking.shape and np.any(large != tracking)
    assert world.render_frame(camera="tracking").shape == tracking.shape
    with pytest.raises(ValueError, match="nose"):
        world.render_frame(camera="nose")


def test_the_head_view_shows_the_backdrop_and_not_the_dog(config, world):
    world.reset()
    frame = world.render_frame(camera="head")
    grey = round(config["world"]["render"]["backdrop_grey"] * 255)
    assert np.abs(frame[0].astype(int) - grey).max() <= 2  # the top row is the plain backdrop
    # Raise both front legs forward: a camera that saw the dog's own meshes would show them; this one does not.
    d = world.policy.data
    for leg in ("FL", "FR"):
        d.qpos[world.model.jnt_qposadr[world.model.joint(f"{leg}_thigh_joint").id]] -= 1.2
    assert np.array_equal(world.render_frame(camera="head"), frame)


def test_the_near_plane_is_fixed_in_metres(config):
    znear = config["world"]["render"]["znear_m"]
    w = Go2World(config, seed=3)
    for situation in ({}, {"incline_deg": 10.0}):  # the incline raises model.stat.extent
        w.reset(situation)
        assert w.model.vis.map.znear * w.model.stat.extent == pytest.approx(znear)


@pytest.mark.parametrize("camera", ["head", "tracking"])
def test_the_first_frame_draws_the_object_where_it_is(config, camera):
    # Two objects at different places, both in view: frames at the decision instant must differ. Drawn at the
    # object's parking spot (stale positions after reset) they would be the same.
    frames = []
    for y in (-0.25, 0.25):
        w = Go2World(config, seed=3)
        w.reset(ahead(0.9, y, 0.1))
        frames.append(w.render_frame(camera=camera))
    assert np.any(frames[0] != frames[1])


@pytest.mark.parametrize("kind", ["ball", "punch"])
def test_an_object_thrown_from_the_front_is_visible_in_the_head_view_from_release(config, kind):
    def frame(situation):
        w = Go2World(config, seed=3)
        w.reset(situation)
        return w.render_frame(camera="head")
    empty, thrown = frame({}), frame(ahead(3.3, 0.0, 0.017, vx=-2.0, kind=kind))
    changed = np.any(empty != thrown, axis=2)
    assert changed.sum() >= 4, changed.sum()  # about 3 degrees across at 3 m: several of the 96 x 96 pixels


def test_on_physics_step_sees_every_physics_step_and_rendering_in_it_changes_nothing(config, zero):
    def episode(render):
        w = Go2World(config, seed=3)
        w.reset(ahead(3.3, 0.0, 0.017, vx=-2.0, kind="punch"), attempt_seed=1)
        times = []

        def each(t_s):
            times.append(t_s)
            if render and len(times) % 17 == 0:  # about 1/30 s
                w.render_frame(camera="head")
        w.step(zero | {"vel_y": 0.5}, 0.5, on_physics_step=each)
        w.step(zero, 1.0, on_physics_step=each)
        out = w.finish_episode()
        return times, dict(np.load(out.pop("trace_path"))), out
    (times, trace_a, out_a), (_, trace_b, out_b) = episode(False), episode(True)
    dt = config["robot"]["walking_policy"]["command_schema"]["update_period_s"] / 10  # 10 physics steps a period
    assert times == pytest.approx([(i + 1) * dt for i in range(round(1.5 / dt))])
    assert out_a == out_b and all(np.array_equal(trace_a[k], trace_b[k]) for k in trace_a)


def test_the_ball_on_the_incline_is_seen_and_measured(config, zero):
    # The ball and the incline in one scene (NOT RUN before W2): a throw from the front over the slope.
    w = Go2World(config, seed=3)
    obs = w.reset(ahead(3.3, 0.0, 0.3, vx=-3.0) | {"incline_deg": 10.0})
    frame = w.render_frame(camera="head")
    w.step(zero, 2.0)
    out = w.finish_episode()
    assert obs["ball"] is not None and frame.shape[:2] == (96, 96)
    assert out["min_separation_m"] is not None and out["ramp_start_m"] == config["world"]["incline"]["start_x_m"]
    assert math.isfinite(out["min_separation_m"]) and out["sim_unstable"] == 0
