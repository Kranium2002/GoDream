"""The acceptance rules: the CLI's pass rule on hand-written measurements (a pure-function test, not simulator
evidence), and the fall detector on real MuJoCo contacts of the Go2 model."""
import mujoco
import pytest

from learned_judgment.env.acceptance import judge, trunk_on_floor
from learned_judgment.env.scene import build_model

OK = {"fallen": False, "mean_speed_m_s": 0.6, "mean_yaw_rate_rad_s": 0.5, "final_speed_m_s": 0.01}


@pytest.mark.parametrize("check, command, measured, passed", [
    ("speed", {"vx": 0.6}, OK, True),
    ("speed", {"vx": 0.6}, {**OK, "mean_speed_m_s": 0.47}, False),  # tolerance 0.2 * 0.6 = 0.12
    ("speed", {"vx": 0.6}, {**OK, "fallen": True}, False),
    ("yaw", {"yaw_rate": -0.5}, {**OK, "mean_yaw_rate_rad_s": -0.36}, True),  # tolerance 0.3 * 0.5 = 0.15
    ("yaw", {"yaw_rate": -0.5}, OK, False),
    ("stop", {}, {**OK, "final_speed_m_s": 0.05}, False),  # must be below 0.05
    ("stand", {}, OK, True),
    ("stand", {}, {**OK, "fallen": True}, False),
])
def test_judge_applies_config_tolerances(config, check, command, measured, passed):
    assert judge(config, check, command, measured)[0] is passed


def test_trunk_on_floor_sees_only_trunk_floor_contacts(config):
    model = build_model(config)
    data = mujoco.MjData(model)
    data.qpos[2] = 0.30  # joints at zero, legs straight down into the floor; trunk box bottom at 0.243 m
    mujoco.mj_forward(model, data)
    assert data.ncon > 0 and not trunk_on_floor(model, data)
    data.qpos[2] = 0.03  # trunk collision box (half-height 0.057 m) sunk into the floor
    mujoco.mj_forward(model, data)
    assert trunk_on_floor(model, data)
