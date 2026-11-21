"""Shared helpers of the snn_shared_measure review reproductions (REVIEW REPRODUCTION - fixture, not evidence).

Reuses the FIXTURE SkillSpecs and DreamSets of experiments/proofs/B_snn_shared_offline (made-up specs,
physics-only dreams built by the real engine). Nothing here is evidence about the robot, the simulator,
TabPFN or a project student.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_FIXTURES = REPO_ROOT / "experiments" / "proofs" / "B_snn_shared_offline"
for entry in (str(REPO_ROOT), str(SHARED_FIXTURES)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from fixture_shared_specs import (  # noqa: E402,F401
    FIXTURE_FEATURE_WIDTH,
    FIXTURE_SHARED_SKILLS,
    POINTS,
    FixtureSharedSpec,
    fixture_config,
    fixture_dreams,
    fixture_spec,
)
from learned_judgment.engine.dream import DreamSet  # noqa: E402

ROW_ARRAYS = (
    "situations", "candidates", "scores", "chosen_spread", "chosen_physics", "chosen_mean",
    "chosen_median", "chosen_outcome_spread", "chosen", "physics_choice", "request_index",
)


def subset(dreams: DreamSet, rows: np.ndarray) -> DreamSet:
    """The given rows of a DreamSet (same names and provenance)."""
    return dataclasses.replace(dreams, **{name: getattr(dreams, name)[rows] for name in ROW_ARRAYS})


def concat(first: DreamSet, second: DreamSet) -> DreamSet:
    """Rows of ``first`` followed by rows of ``second`` (both physics-only, no request hashes)."""
    assert not first.request_hashes and not second.request_hashes
    return dataclasses.replace(
        first, **{name: np.concatenate([getattr(first, name), getattr(second, name)]) for name in ROW_ARRAYS}
    )
