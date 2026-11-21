"""learned_judgment.skills.dodge_vision: the sampler, physics guess and objective (real config.yaml), and the head view
and episodes on the real Go2 world with the borrowed walking policy."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from learned_judgment.config import load_config, require
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.env import Go2World
from learned_judgment.env.throws import BALL_KEYS
from learned_judgment.skills.dodge_vision import (
    KINDS,
    OUTCOME_NAMES,
    HeadView,
    _EpisodeAugmentation,
    build_observed_spec,
    closed_loop,
    frame_times,
    objective,
    observe_and_act,
    physics_guess,
    render_windows,
    sample_threat_scenarios,
)
from learned_judgment.vision import VisionSettings, augment_window, frame_control_periods

CONFIG = load_config()
SETTINGS = VisionSettings.from_config(CONFIG)
SPEC = build_observed_spec(CONFIG)
ROOT = "skills.dodge_vision"
SCENARIOS = f"{ROOT}.scenarios"
FLOOR_Z = require(CONFIG, "ball.radius_m") - require(CONFIG, f"{ROOT}.physics.base_height_m")
CAMERA = require(CONFIG, "vision.head_camera.pos_m")
TOL = 1e-9


def sample(n: int, seed: int = 7) -> list[dict]:
    return sample_threat_scenarios(CONFIG, np.random.default_rng(seed), n)


SAMPLE = sample(3000)


def is_roll(scenario: dict) -> bool:
    situation = scenario["situation"]
    return situation["ball_vel_z_m_s"] == 0.0 and situation["ball_rel_z_m"] == FLOOR_Z


def first(kind: str, roll: bool | None = None) -> dict:
    return next(s for s in SAMPLE if s["kind"] == kind and (roll is None or is_roll(s) == roll))


def inside(value: float, key: str) -> bool:
    low, high = require(CONFIG, key)
    return low - TOL <= value <= high + TOL


def crossing(situation: dict) -> tuple[float, float, float]:
    """(time, y, z) of the ball's centre at the plane x = 0, flying ballistically (no floor)."""
    t = -situation["ball_rel_x_m"] / situation["ball_vel_x_m_s"]
    g = require(CONFIG, "ball.gravity_m_s2")
    y = situation["ball_rel_y_m"] + situation["ball_vel_y_m_s"] * t
    z = situation["ball_rel_z_m"] + situation["ball_vel_z_m_s"] * t - 0.5 * g * t * t
    return t, y, z


@pytest.fixture(scope="module")
def world_and_view():
    config = copy.deepcopy(CONFIG)
    config["world"]["save_traces"] = False
    world = Go2World(config, require(config, "observed.world_seed"), require(config, "observed.world_preset"))
    view = HeadView(config)
    yield world, view
    view.close()


# ----- sampler ---------------------------------------------------------------------------------------------------


def test_sampler_is_deterministic_and_prefix_independent() -> None:
    assert sample(200) == sample(200)
    assert sample(60) == sample(200)[:60]
    assert sample(1) == sample(200)[:1]
    assert sample(200, seed=8) != sample(200)


def test_scenarios_have_valid_kinds_keys_and_ranges() -> None:
    assert set(require(CONFIG, f"{ROOT}.kind_weights")) == set(KINDS)
    assert {s["kind"] for s in SAMPLE} == set(KINDS)
    assert {is_roll(s) for s in SAMPLE if s["kind"].startswith("incoming")} == {True, False}
    half_view = math.tan(math.radians(require(CONFIG, "vision.head_camera.fovy_deg")) / 2.0)
    for index, scenario in enumerate(SAMPLE):
        assert set(scenario) == {"scenario_id", "kind", "object_kind", "situation", "visual"}
        assert scenario["scenario_id"] == index and scenario["object_kind"] == "ball" and scenario["visual"] == {}
        s = scenario["situation"]
        assert all(type(value) is float for value in s.values())
        assert s["dog_forward_speed_m_s"] == 0.0 and s["settle_s"] == require(CONFIG, f"{ROOT}.settle_s")
        kind = scenario["kind"]
        if kind == "empty":
            assert set(s) == {"dog_forward_speed_m_s", "settle_s"}
            continue
        assert set(s) == {"dog_forward_speed_m_s", "settle_s", *BALL_KEYS}
        x0, y0, z0 = s["ball_rel_x_m"], s["ball_rel_y_m"], s["ball_rel_z_m"]
        vx, vy, vz = s["ball_vel_x_m_s"], s["ball_vel_y_m_s"], s["ball_vel_z_m_s"]
        distance, bearing = math.hypot(x0, y0), math.degrees(math.atan2(y0, x0))
        if kind.startswith("incoming") or kind in ("ambiguous", "off_path"):
            assert inside(distance, f"{SCENARIOS}.distance_m") and inside(bearing, f"{SCENARIOS}.bearing_deg")
            assert vx < 0.0
            _, aim_y, z_at_dog = crossing(s)
            if is_roll(scenario):
                assert inside(math.hypot(vx, vy), f"{SCENARIOS}.roll_speed_m_s")
            else:
                assert inside(z0, f"{SCENARIOS}.throw_release_height_m")
                assert inside(math.hypot(vx, vy), f"{SCENARIOS}.throw_speed_m_s")
                assert z_at_dog == pytest.approx(CAMERA[2], abs=1e-9)
            key = {"incoming_center": "aim_center_m", "incoming_left": "aim_side_m", "incoming_right": "aim_side_m",
                   "ambiguous": "aim_ambiguous_m", "off_path": "aim_off_path_m"}[kind]
            assert inside(abs(aim_y), f"{SCENARIOS}.{key}")
            if kind == "incoming_left":
                assert aim_y > 0.0
            if kind == "incoming_right":
                assert aim_y < 0.0
            continue
        assert z0 == FLOOR_Z and vz == 0.0
        if kind == "static":
            assert vx == vy == 0.0
            assert inside(distance, f"{SCENARIOS}.static_distance_m") and inside(bearing, f"{SCENARIOS}.bearing_deg")
        elif kind == "receding":
            assert vy == 0.0 and inside(vx, f"{SCENARIOS}.roll_speed_m_s")
            assert inside(distance, f"{SCENARIOS}.receding_distance_m") and inside(bearing, f"{SCENARIOS}.bearing_deg")
        else:  # background_motion: at the edge of the square view, rolling across it
            assert vx == 0.0 and inside(abs(vy), f"{SCENARIOS}.roll_speed_m_s")
            assert inside(x0, f"{SCENARIOS}.background_distance_m")
            assert abs(y0) == pytest.approx((x0 - CAMERA[0]) * half_view, abs=1e-12)
            assert np.sign(vy) == -np.sign(y0)


def test_situations_are_accepted_by_a_real_world(world_and_view) -> None:
    world, _ = world_and_view
    for kind in KINDS:
        situation = first(kind)["situation"]
        obs = world.reset(situation, attempt_seed=0)
        assert obs["t_s"] == 0.0
        if kind == "empty":
            assert obs["ball"] is None
        else:
            offset = obs["ball"]["pos_m"] - obs["base_pos_m"]
            launch = np.array([situation[k] for k in BALL_KEYS[:3]])
            assert np.linalg.norm(offset) == pytest.approx(np.linalg.norm(launch), abs=1e-9)
            assert offset[2] == pytest.approx(launch[2], abs=1e-9)


# ----- physics guess and objective -------------------------------------------------------------------------------


def no_op_guess(scenario: dict, decision_time_s: float = 0.5) -> np.ndarray:
    return physics_guess(CONFIG, [scenario], [decision_time_s], [SPEC.no_op_index])[0]


def test_physics_guess_shapes_and_bounds() -> None:
    scenarios = SAMPLE[:300]
    c = len(SPEC.action_names)
    rows = [s for s in scenarios for _ in range(c)]
    times = np.repeat(np.linspace(0.3, 1.3, len(scenarios)), c)
    actions = np.tile(np.arange(c), len(scenarios))
    guess = physics_guess(CONFIG, rows, times, actions)
    assert guess.shape == (len(rows), len(OUTCOME_NAMES)) and guess.dtype == np.float64
    assert np.all(np.isfinite(guess))
    assert np.all(guess >= SPEC.outcome_bounds[:, 0]) and np.all(guess <= SPEC.outcome_bounds[:, 1])
    assert np.all(guess[:, 2] == 0.0) and np.all(guess[:, 3] == 0.0)
    hold_s = require(CONFIG, f"{ROOT}.hold_s")
    assert np.array_equal(guess[:, 4], np.abs(SPEC.action_features[actions]).max(axis=1) * hold_s)
    assert np.array_equal(physics_guess(CONFIG, rows[:8], times[:8], actions[:8]), guess[:8])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limit of the physics guess, kept visible: its trunk capsule (radius body_radius_m around the trunk "
        "origin's x axis) passes over a ball resting on the floor (centre ball.radius_m - base_height_m below the "
        "trunk origin) with a small positive clearance, so a centre roll gets a collision probability below 0.5; "
        "the Go2's legs are not in the model"
    ),
)
def test_physics_centre_roll_collides_under_no_op() -> None:
    assert no_op_guess(first("incoming_center", roll=True))[1] > 0.5


def test_physics_sanity() -> None:
    assert no_op_guess(first("incoming_center", roll=False))[1] > 0.5
    assert no_op_guess(first("off_path", roll=True))[1] < 0.5
    assert no_op_guess(first("off_path", roll=False))[1] < 0.5
    clip = require(CONFIG, f"{ROOT}.separation_clip_m")
    for action in range(len(SPEC.action_names)):
        guess = physics_guess(CONFIG, [first("empty")], [0.5], [action])[0]
        assert guess[0] == clip and guess[1] == 0.0


def test_objective_prefers_no_op_in_an_empty_scene() -> None:
    c = len(SPEC.action_names)
    guess = physics_guess(CONFIG, [first("empty")] * c, [0.5] * c, np.arange(c))
    predicted = OutcomeDistribution.point(OUTCOME_NAMES, guess, np.array(require(CONFIG, "engine.quantile_levels")))
    scores = objective(CONFIG, predicted)
    assert scores.shape == (c,)
    assert int(np.argmax(scores)) == SPEC.no_op_index
    assert np.array_equal(SPEC.objective(predicted), scores)


def test_spec_and_frame_times() -> None:
    assert SPEC.name == "dodge_vision"
    assert SPEC.action_names == tuple(require(CONFIG, f"{ROOT}.actions"))
    assert SPEC.action_feature_names == ("vel_x", "vel_y", "yaw_rate")
    assert SPEC.action_names[SPEC.no_op_index] == "NO_OP" and not SPEC.action_features[SPEC.no_op_index].any()
    assert SPEC.outcome_bounds.shape == (5, 2)
    times = frame_times(CONFIG)
    dt = require(CONFIG, "robot.walking_policy.command_schema.update_period_s")
    periods = frame_control_periods(times.size, require(CONFIG, "vision.frame_rate_hz"), dt)
    assert np.array_equal(times, periods * dt)
    assert times[0] == 0.0 and np.all(np.diff(times) > 0.0)
    assert times[-1] < require(CONFIG, f"{ROOT}.episode_duration_s")


def test_episode_augmentation_matches_augment_window_on_one_frame() -> None:
    frame = np.random.default_rng(3).uniform(0.0, 1.0, size=(12, 12)).astype(np.float32)
    augmentation = require(CONFIG, "vision.augmentation")
    once = augment_window(frame[None], np.random.default_rng(11), augmentation)[0]
    helper = _EpisodeAugmentation(np.random.default_rng(11), augmentation)
    assert np.array_equal(helper.apply(frame), once)
    second = helper.apply(frame)  # same four values, fresh noise
    assert not np.array_equal(second, once)


# ----- episodes on the real world --------------------------------------------------------------------------------


def test_observe_and_act_on_a_centre_roll_is_repeatable(world_and_view) -> None:
    world, view = world_and_view
    scenario = first("incoming_center", roll=True)
    first_run = observe_and_act(world, view, scenario, 0, 20, SPEC.no_op_index, SETTINGS)
    features, measured = first_run
    assert features.shape == (SETTINGS.n_features,) and features.dtype == np.float32
    assert np.all(features >= 0.0) and np.all(features <= 1.0)
    assert tuple(measured) == OUTCOME_NAMES
    assert 0.0 <= measured["min_separation_m"] <= require(CONFIG, f"{ROOT}.separation_clip_m")
    assert measured["command_travel_m"] == 0.0
    again = observe_and_act(world, view, scenario, 0, 20, SPEC.no_op_index, SETTINGS)
    assert np.array_equal(again[0], features) and again[1] == measured


def test_frames_change_as_a_ball_rolls_in(world_and_view) -> None:
    world, view = world_and_view
    windows = render_windows(world, view, first("incoming_center", roll=True), 0, [9, 24, 39], SETTINGS)
    assert windows.shape == (3, SETTINGS.n_features) and windows.dtype == np.float32
    assert not np.array_equal(windows[0], windows[1]) and not np.array_equal(windows[1], windows[2])
    frames = windows[2].reshape(SETTINGS.frames, SETTINGS.frame_size, SETTINGS.frame_size)
    assert not np.array_equal(frames[0], frames[-1])


def test_closed_loop_on_the_zero_command_matches_observe_and_act(world_and_view) -> None:
    world, view = world_and_view
    scenario = first("incoming_left", roll=True)
    zero = {"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}
    calls = []

    def policy(window: np.ndarray, t_s: float) -> dict:
        calls.append((window.shape, t_s))
        return {"command": zero, "action": "NO_OP", "issued": False, "scores": np.zeros(4)}

    result = closed_loop(world, view, scenario, 0, policy, SETTINGS)
    _, measured = observe_and_act(world, view, scenario, 0, 20, SPEC.no_op_index, SETTINGS)
    assert result["measured"] == measured
    times = frame_times(CONFIG)
    assert [t for _, t in calls] == pytest.approx(times[SETTINGS.frames - 1 :].tolist(), abs=1e-12)
    assert all(shape == (SETTINGS.frames, SETTINGS.frame_size, SETTINGS.frame_size) for shape, _ in calls)
    assert len(result["trace"]) == len(calls) and result["issued"] == []
    assert "scores" not in result["trace"][0] and result["trace"][0]["action"] == "NO_OP"
