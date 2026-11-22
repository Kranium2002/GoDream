"""Shared fixtures for the server tests.

write_student builds a student directory in Dev B's exported format (learned_judgment.runtime: student.json,
network.npz, network.json) around a HAND-BUILT, UNTRAINED spiking network, the way Dev B's tests/b/test_runtime.py
does. It lets this server's adapter run against Dev B's real DodgeRuntime code. It is not a trained student: what it
answers is no evidence that anything dodges.
"""
import json

import numpy as np
import pytest

import learned_judgment.run_skill as run_skill
from learned_judgment.runtime import ENCODING, STUDENT_FORMAT
from learned_judgment.students.snn_core import SpikingNetwork, save_network

THRESHOLDS = {"threat": 0.9, "action": 0.6, "margin": 0.3}


@pytest.fixture(autouse=True)
def no_real_runs(monkeypatch):
    """No server test may start or stop a real learning run: Dev B's start_run spawns a detached simulator chain and,
    with confirm_billing, bills hosted TabPFN; stop_run kills a process tree. Route tests use a stand-in runner."""
    def refuse(*args, **kwargs):
        raise AssertionError("a test reached Dev B's real start_run or stop_run")

    monkeypatch.setattr(run_skill, "start_run", refuse)
    monkeypatch.setattr(run_skill, "stop_run", refuse)


@pytest.fixture(scope="session", autouse=True)
def no_new_run_folders():
    """Fails the session if any test made a folder in the real runs directory (observed.runs_dir): a real run."""
    from app import backend
    from learned_judgment.env import load_config
    root, _ = backend.runs_dir(load_config())

    def folders():
        return {p.name for p in root.iterdir() if p.is_dir()} if root and root.is_dir() else set()

    before = folders()
    yield
    assert folders() - before == set(), "a test created a run folder in the real runs directory"


def write_student(directory, config, frames=None, thresholds=THRESHOLDS):
    """A student directory for config.yaml's dodge_vision actions and vision frame size; its network never spikes and
    its readout bias favours no threat and NO_OP, so every window gives NO_OP."""
    names = list(config["skills"]["dodge_vision"]["actions"])
    frames = config["vision"]["frames"] if frames is None else frames
    size = config["vision"]["frame_size"]
    network = SpikingNetwork(
        hidden_weights=(np.zeros((1, frames * size * size)),), hidden_biases=(np.array([-1.0]),), betas=(0.5,),
        thresholds=(0.25,), readout_weight=np.zeros((2 + len(names), 1)),
        readout_bias=np.array([3.0, 0.0] + [2.0] + [0.0] * (len(names) - 1)), readout_beta=0.0, time_steps=2)
    directory.mkdir(parents=True, exist_ok=True)
    save_network(directory / "network", network, {"note": "hand-built server test fixture, untrained"})
    student = {"format": STUDENT_FORMAT, "skill": "dodge_vision", "action_names": names,
               "heads": {"threat": [0, 2], "action": [2, 2 + len(names)]}, "no_op_index": names.index("NO_OP"),
               "input": {"frames": frames, "frame_size": size, "encoding": ENCODING},
               "vision": {"frames": frames, "frame_size": size}, "thresholds": thresholds,
               "provenance": {"note": "hand-built server test fixture, not a trained student"}}
    (directory / "student.json").write_text(json.dumps(student), encoding="utf-8")
    return directory
