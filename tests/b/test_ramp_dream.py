"""learned_judgment.skills.ramp dream sampling: recorded_situations on small hand-made arrays (fixtures only) and the
two modes of collect_dream (no simulator, no teacher)."""

import copy

import numpy as np
import pytest

from learned_judgment.config import load_config
from learned_judgment.skills.ramp import collect_dream, recorded_situations


def _config(sampling: str) -> dict:
    config = copy.deepcopy(load_config())
    config["skills"]["ramp"]["dream_sampling"] = sampling
    config["skills"]["ramp"]["dream_noise"] = 0.02
    return config


@pytest.fixture
def config():
    return _config("uniform")


@pytest.fixture
def config_recorded():
    return _config("recorded")


FEATS = np.array([[0.2, 0.25, 0.5, 0.4], [0.2, 0.25, 0.5, 0.4], [0.7, 0.72, 0.6, 0.4]], np.float32)  # scenario 0 has two rows
TIMES, INDEX = np.array([2.1, 2.1, 2.2]), np.array([0, 0, 1])


def test_recorded_situations_stay_near_real_ones_and_keep_their_time():
    f, t = recorded_situations(FEATS, TIMES, INDEX, n=500, noise=0.02, rng=np.random.default_rng(1))
    assert f.shape == (500, 4) and f.dtype == np.float32 and f.min() >= 0.0 and f.max() <= 1.0
    near0 = np.abs(f - FEATS[0]).max(axis=1) < 0.1
    near1 = np.abs(f - FEATS[2]).max(axis=1) < 0.1
    assert np.all(near0 | near1)
    assert 0.35 < near0.mean() < 0.65  # scenarios are drawn evenly, not rows
    assert np.all(np.isclose(t[near0], 2.1)) and np.all(np.isclose(t[near1], 2.2))


def test_noise_zero_returns_real_vectors_and_noise_is_clipped():
    f, _ = recorded_situations(FEATS, TIMES, INDEX, n=50, noise=0.0, rng=np.random.default_rng(2))
    assert all(any(np.array_equal(row, real) for real in FEATS) for row in f)
    edge = np.array([[0.0, 1.0, 0.0, 1.0]], np.float32)
    g, _ = recorded_situations(edge, np.array([2.0]), np.array([0]), n=200, noise=0.5, rng=np.random.default_rng(3))
    assert g.min() >= 0.0 and g.max() <= 1.0


def test_uniform_mode_is_the_old_dream(tmp_path, config):  # config: the repository config with dream_sampling "uniform"
    collect_dream(config, tmp_path, n=64)
    rng = np.random.default_rng(np.random.SeedSequence([config["runtime"]["seed"], 4]))
    with np.load(tmp_path / "dream.npz") as d:
        assert np.array_equal(d["features"], rng.random((64, 4)).astype(np.float32))


def test_recorded_mode_needs_the_rows(tmp_path, config_recorded):
    with pytest.raises(ValueError, match="--from-rows"):
        collect_dream(config_recorded, tmp_path, n=8)
