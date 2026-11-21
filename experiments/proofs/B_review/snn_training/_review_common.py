"""Shared helpers for the snn_training review reproductions (REVIEW REPRODUCTION - fixture, not evidence).

Puts the repository root and the B_snn_offline fixture directory on sys.path so the scripts can
use the real learned_judgment code and the existing FIXTURE spec (fixture_snn_spec.py). Nothing
computed here is evidence about the robot, the simulator, TabPFN or a project student.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
FIXTURE_DIR = REPO_ROOT / "experiments" / "proofs" / "B_snn_offline"
for entry in (str(REPO_ROOT), str(FIXTURE_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402

from learned_judgment.students.snn_core import SpikingNetwork  # noqa: E402


def random_network(rng: np.random.Generator, sizes, *, time_steps: int, beta: float = 0.9, threshold: float = 1.0,
                   readout_beta: float = 0.9) -> SpikingNetwork:
    """A numpy SpikingNetwork with torch-Linear-like uniform init (+-1/sqrt(fan_in)). Fixture only."""
    sizes = tuple(sizes)
    weights, biases = [], []
    for fan_in, fan_out in zip(sizes[:-2], sizes[1:-1]):
        bound = 1.0 / np.sqrt(fan_in)
        weights.append(rng.uniform(-bound, bound, size=(fan_out, fan_in)))
        biases.append(rng.uniform(-bound, bound, size=fan_out))
    bound = 1.0 / np.sqrt(sizes[-2])
    return SpikingNetwork(
        hidden_weights=tuple(weights),
        hidden_biases=tuple(biases),
        betas=tuple(beta for _ in weights),
        thresholds=tuple(threshold for _ in weights),
        readout_weight=rng.uniform(-bound, bound, size=(sizes[-1], sizes[-2])),
        readout_bias=rng.uniform(-bound, bound, size=sizes[-1]),
        readout_beta=readout_beta,
        time_steps=time_steps,
    )
