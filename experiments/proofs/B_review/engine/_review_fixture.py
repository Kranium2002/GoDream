"""REVIEW REPRODUCTION fixture (engine lens). Not a project skill, not evidence.

A toy SkillSpec and a toy command schema used only to reproduce suspected defects in the
engine lane. No world, no provider, no measured data. Nothing here may enter a report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learned_judgment.engine.types import OutcomeDistribution  # noqa: E402
from learned_judgment.skills.base import ActionDim, CommandField, CommandSchema, Outcome, SkillSpec, Variable  # noqa: E402

BANNER = "REVIEW REPRODUCTION - fixture, not evidence"
LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
SPREAD = (0.1, 0.9)
TS = "2026-11-21T12:00:00+00:00"

# Toy verified schema: ONE field "rv_vy" in [-1, 1] m/s (not the Go2 schema).
REVIEW_SCHEMA = CommandSchema(
    fields={"rv_vy": CommandField(name="rv_vy", unit="m/s", low=-1.0, high=1.0)},
    update_period_s=0.02,
    source="review fixture",
)


class ReviewSpec(SkillSpec):
    """Toy spec: 2 situation features (one categorical), 2 action dims, 2 outcomes (one binary)."""

    def __init__(self, *, gain: float = 1.0, command_field: str = "rv_vy", vy_range=(-1.0, 1.0)) -> None:
        self.gain = gain
        self.name = "review_toy"
        self.description = "review fixture; not a robot skill"
        self.situation = (
            Variable("rv_dist", "float", "m", "toy distance", low=0.0, high=2.0),
            Variable("rv_surface", "categorical", "label", "toy surface", levels=("a", "b")),
        )
        self.actions = (
            ActionDim("rv_side", "float", "m/s", "toy side speed", low=vy_range[0], high=vy_range[1],
                      role="command", command_field=command_field),
            ActionDim("rv_delay", "float", "s", "toy delay", low=0.0, high=0.4, role="timing"),
        )
        self.outcomes = (
            Outcome("rv_sep", "regression", "m", "toy separation", low=0.0),
            Outcome("rv_hit", "binary", "", "toy hit flag"),
        )
        self.primary_outcome = "rv_sep"
        self.hidden_effects = ("review fixture: none",)

    def physics_guess(self, situations, actions):
        situations = np.asarray(situations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        sep = np.abs(0.2 * situations[:, 0] + self.gain * actions[:, 0] * (0.5 - actions[:, 1]))
        hit = np.clip(1.0 - sep / 0.3, 0.0, 1.0)
        return np.column_stack([sep, hit])

    def objective(self, predicted: OutcomeDistribution):
        return predicted.quantile_of("rv_sep", 0.1) - 2.0 * predicted.mean_of("rv_hit")

    def episode(self, world, situation, action):
        raise RuntimeError("review fixture has no episode")

    def loss(self, measured):
        return np.asarray(measured)[:, 1]

    def success(self, measured):
        return np.asarray(measured)[:, 1] == 0.0
