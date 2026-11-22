"""The dodge_vision teacher's scene-truth block (Task 4): pure functions on hand-set launch states (fixtures only;
nothing here is a result) with the real config.yaml. No TabPFN request and no simulator run."""

from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest

from learned_judgment import observed
from learned_judgment.config import load_config, require
from learned_judgment.observed import teacher_matrix
from learned_judgment.skills import dodge_vision
from learned_judgment.skills.dodge_vision import TRUTH_NAMES, build_observed_spec, sample_threat_scenarios, truth_features


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def spec(config):
    return build_observed_spec(config)


@pytest.fixture(scope="module")
def scenarios(config):
    return sample_threat_scenarios(config, np.random.default_rng(0), 50)


@pytest.fixture
def features(config):
    d = require(config, "vision.frames") * 12 * 12
    return np.zeros((6, d), np.float32)


@pytest.fixture
def action(spec):
    return np.arange(6) % len(spec.action_names)


@pytest.fixture
def physics():
    return np.zeros((6, 5))


@pytest.fixture
def punch(scenarios):
    def make(x, y, z, vx, vy, vz):
        scenario = copy.deepcopy(next(s for s in scenarios if s["object_kind"] == "punch"))
        scenario["situation"].update(zip(dodge_vision._ball_keys(), (x, y, z, vx, vy, vz)))
        return scenario

    return make


@pytest.fixture
def empty_scenario(scenarios):
    return next(s for s in scenarios if s["kind"] == "empty")


def test_truth_of_a_head_on_punch(config, punch):
    t = truth_features(config, [punch(1.0, 0.1, 0.2, -1.0, 0.0, 0.0)], [0.25])
    assert t.shape == (1, 5) and t.dtype == np.float64
    present, arrival, aim_y, aim_z, speed = t[0]
    assert present == 1.0 and speed == pytest.approx(1.0)
    assert arrival == pytest.approx(0.75)  # 1.0 m at 1.0 m/s, 0.25 s already gone
    assert aim_y == pytest.approx(0.1) and aim_z == pytest.approx(0.2)


def test_truth_is_finite_when_nothing_arrives(config, punch, empty_scenario):
    duration = config["skills"]["dodge_vision"]["episode_duration_s"]
    static = truth_features(config, [punch(1.0, 0.3, 0.2, 0.0, 0.0, 0.0)], [0.5])[0]
    receding = truth_features(config, [punch(1.0, 0.3, 0.2, 1.0, 0.0, 0.0)], [0.5])[0]
    empty = truth_features(config, [empty_scenario], [0.5])[0]
    assert np.all(np.isfinite([static, receding, empty]))
    assert static[1] == duration and receding[1] == duration and static[2] == pytest.approx(0.3)
    assert empty.tolist() == [0.0, duration, 0.0, 0.0, 0.0]


def test_truth_block_replaces_the_window_columns(spec, features, action, physics):
    truth = np.zeros((features.shape[0], 5))
    x, layout = teacher_matrix(spec, features, action, physics, ["truth", "action", "physics"], truth=truth)
    assert layout.n_situation == 5 and layout.names[:5] == tuple(f"truth_{n}" for n in TRUTH_NAMES)
    assert not any(name.startswith("window_") for name in layout.names)
    assert x.shape == (features.shape[0], 5 + layout.n_action + layout.n_physics)


def test_truth_block_without_truth_is_refused_and_the_window_block_is_unchanged(spec, features, action, physics):
    with pytest.raises(ValueError, match="truth"):
        teacher_matrix(spec, features, action, physics, ["truth", "action", "physics"])
    old, _ = teacher_matrix(spec, features, action, physics, ["window", "action", "physics"])
    assert old.shape[1] == features.shape[1] + spec.action_features.shape[1] + physics.shape[1]


# ----- own checks beyond the plan's four --------------------------------------------------------------------------


def test_truth_of_every_sampled_kind_is_finite(config, scenarios):
    t = truth_features(config, scenarios, np.full(len(scenarios), 0.5))
    assert t.shape == (len(scenarios), 5) and np.all(np.isfinite(t))
    duration = require(config, "skills.dodge_vision.episode_duration_s")
    assert np.all((t[:, 1] >= 0.0) & (t[:, 1] <= duration))


def test_truth_never_reaches_the_student_side():
    """The student path (train_student, save_student, encode_inputs, the train command) takes no truth argument and
    never computes it: truth is built only where a teacher table is built."""
    for function in (observed.train_student, observed.save_student, observed.encode_inputs, observed._train_command):
        source = inspect.getsource(function)
        assert "truth" not in source, function.__name__
