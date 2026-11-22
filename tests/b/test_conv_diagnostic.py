"""Tests of learned_judgment.conv_diagnostic: pure-function fixtures (a toy separable window set)."""

from __future__ import annotations

import numpy as np

from learned_judgment.conv_diagnostic import conv_readouts, trace_windows, train_conv


def toy(m=400, f=4, n=12, c=4, seed=0):
    rng = np.random.default_rng(seed)
    windows = (rng.random((m, f, n, n)) * 0.1).astype(np.float32)
    label = rng.integers(0, 2, m)
    windows[label == 1, -1, 5:7, 5:7] += 0.8  # a bright blob in the newest frame is the threat
    threat = np.stack([1 - label, label], axis=1).astype(np.float64)
    action = np.zeros((m, c)); action[np.arange(m), label] = 1.0
    return windows, {"threat": threat, "action": action}, label


def test_conv_learns_a_separable_toy_and_readouts_fit_the_replay():
    windows, targets, _ = toy()
    out = train_conv(windows, targets, np.arange(400), steps=300, seed=0, validation_fraction=0.2)
    assert out["validation"]["threat_accuracy"] >= 0.95
    threat, scores = conv_readouts(out["model"], windows[:6].reshape(2, 3, 4, 12, 12))
    assert threat.shape == (2, 3) and scores.shape == (2, 3, 4)
    assert np.all((threat >= 0) & (threat <= 1)) and np.allclose(scores.sum(axis=2), 1.0)


def test_one_scenario_never_sits_on_both_sides_of_the_split():
    windows, targets, _ = toy()
    out = train_conv(windows, targets, np.repeat(np.arange(100), 4), steps=5, seed=0, validation_fraction=0.2)
    assert not set(out["split"]["train_groups"]) & set(out["split"]["validation_groups"])
    assert len(out["split"]["validation_groups"]) == 20


def test_same_seed_same_result():
    windows, targets, _ = toy()
    a = train_conv(windows, targets, np.arange(400), steps=50, seed=3, validation_fraction=0.2)
    b = train_conv(windows, targets, np.arange(400), steps=50, seed=3, validation_fraction=0.2)
    assert a["validation"]["loss"] == b["validation"]["loss"]


def test_trace_windows_end_at_the_decision_frame():
    frames = np.arange(20, dtype=np.float32).reshape(2, 10, 1, 1)
    w = trace_windows(frames, f=4, first=7)
    assert w.shape == (2, 3, 4, 1, 1)
    assert w[0, 0, :, 0, 0].tolist() == [4.0, 5.0, 6.0, 7.0] and w[0, 2, :, 0, 0].tolist() == [6.0, 7.0, 8.0, 9.0]
