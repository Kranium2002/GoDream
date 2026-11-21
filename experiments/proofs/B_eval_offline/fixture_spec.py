"""FIXTURE SkillSpec for the B_eval_offline utility checks only.

Not a project skill, not the Go2, not DODGE: a tiny made-up spec whose pure functions
(physics_guess, objective, loss, success) exercise array shapes in the eval lane. It has no
world and its episode() refuses to run. Nothing computed with it is evidence about the
robot, the simulator, TabPFN or any model, and none of it may enter a report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learned_judgment.engine.types import OutcomeDistribution  # noqa: E402
from learned_judgment.skills.base import (  # noqa: E402
    ActionDim,
    CommandField,
    CommandSchema,
    Outcome,
    SkillSpec,
    Variable,
)

FIXTURE_BANNER = "FIXTURE CHECK - utility test only, not evidence about the robot, the simulator, TabPFN or any model"

# Fixture command schema for validate_spec only (NOT the Go2 walking policy's schema).
FIXTURE_COMMAND_SCHEMA = CommandSchema(
    fields={"fixture_vy": CommandField(name="fixture_vy", unit="m/s", low=-1.0, high=1.0)},
    update_period_s=0.02,
    source="fixture (B_eval_offline checks)",
)
FIXTURE_POINTS_PER_DIM = (5, 3)


class FixtureSpec(SkillSpec):
    """Made-up 3-feature / 2-action / 2-outcome spec (one categorical feature, one binary outcome)."""

    def __init__(self) -> None:
        self.name = "fixture"
        self.description = "fixture spec for eval-lane utility checks; not a project skill"
        self.situation = (
            Variable("approach_speed", "float", "m/s", "fixture feature", low=1.0, high=5.0),
            Variable("lateral_offset", "float", "m", "fixture feature", low=-1.0, high=1.0),
            Variable("surface", "categorical", "", "fixture categorical feature", levels=("a", "b", "c")),
        )
        self.actions = (
            ActionDim("side_speed", "float", "m/s", "fixture command", low=-0.5, high=0.5, role="command", command_field="fixture_vy"),
            ActionDim("start_delay", "float", "s", "fixture timing", low=0.0, high=0.4, role="timing"),
        )
        self.outcomes = (
            Outcome("separation", "regression", "m", "fixture regression outcome", low=0.0),
            Outcome("contact", "binary", "", "fixture binary outcome"),
        )
        self.primary_outcome = "separation"
        self.hidden_effects = ("fixture: none; this spec never runs an episode",)

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        situations = np.asarray(situations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        time_left = 2.0 / situations[:, 0] - actions[:, 1]
        separation = np.abs(situations[:, 1] + actions[:, 0] * np.clip(time_left, 0.0, None))
        contact = np.clip(1.0 - separation / 0.3, 0.0, 1.0)
        return np.column_stack([separation, contact])

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        low_level = float(predicted.quantile_levels[0])
        return predicted.quantile_of("separation", low_level) - 2.0 * predicted.mean_of("contact")

    def episode(self, world: Any, situation: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, float]:
        raise RuntimeError("the fixture spec has no world; checks never run episodes")

    def loss(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return measured[:, 1] + np.clip(0.3 - measured[:, 0], 0.0, None)

    def success(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return (measured[:, 0] >= 0.3) & (measured[:, 1] == 0.0)


class Checks:
    """Counts executed assertions of a check script and sets its exit code."""

    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def expect(self, condition: bool, label: str) -> None:
        print(("OK   " if condition else "FAIL ") + label)
        if condition:
            self.passed += 1
        else:
            self.failures.append(label)

    def finish(self) -> None:
        print(f"\nCHECK RESULT: {self.passed} assertions held, {len(self.failures)} failed {self.failures or ''}")
        raise SystemExit(0 if not self.failures else 1)


def fixture_rows(spec: FixtureSpec, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixture arrays for shape tests: situations (n,S), actions (n,A), physics (n,K), residuals (n,K).

    The residuals are arbitrary seeded numbers (a feature-dependent pattern plus noise) so the
    learners have something non-constant to fit. They are NOT measured outcomes.
    """
    rng = np.random.default_rng(seed)
    situations = spec.sample_situations(rng, n)
    actions = np.column_stack([rng.uniform(-0.5, 0.5, n), rng.uniform(0.0, 0.4, n)])
    physics = spec.physics_guess(situations, actions)
    residual_separation = 0.2 * situations[:, 2] - 0.3 * actions[:, 0] + 0.05 * rng.normal(size=n)
    contact = (rng.uniform(size=n) < 0.5).astype(np.float64)
    residuals = np.column_stack([residual_separation, contact - physics[:, 1]])
    return situations, actions, physics, residuals
