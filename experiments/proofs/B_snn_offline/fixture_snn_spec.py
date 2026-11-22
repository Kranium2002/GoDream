"""FIXTURE SkillSpecs and helpers for the B_snn_offline checks only.

Not project skills, not the Go2, not DODGE: made-up specs (2 float situation features and one
categorical, a 2-D action grid, 3 outcomes) whose physics_guess and objective make the best
candidate depend on the situation, so a dream built from them has something to learn. Their
dreams are built with the REAL engine (engine.dream.dream with model=None, method=PHYSICS_ONLY),
so the scores come from the real decide step. There is no world; episode() refuses to run.

Variants used by the candidate-grid checks: FixtureSnnSpec with other side-speed bounds or another
command field; FixtureReversedGridSpec (same action dimensions, candidates in reverse order); and
FixtureSideSpec, whose side command is categorical (levels with command values).

Nothing computed with these fixtures is evidence about the robot, the simulator, TabPFN or any
model, and none of it may enter a report or results/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learned_judgment.config import require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.dream import DreamSet, dream  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.engine.types import OutcomeDistribution  # noqa: E402
from learned_judgment.skills.base import ActionDim, Outcome, SkillSpec, Variable  # noqa: E402

BANNER = "FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model"
FIXTURE_POINTS_PER_DIM = (7, 5)  # fixture grid: 7 side speeds x 5 delays = 35 candidates
FIXTURE_SIDE_POINTS_PER_DIM = (1, 5)  # categorical side (all 3 levels) x 5 delays = 15 candidates
FIXTURE_DREAM_SEED = 20261121  # fixture seed of the dreamed situations
FIXTURE_N_DREAMS = 1000


def _fixture_situation(speed_high: float) -> tuple[Variable, ...]:
    return (
        Variable("fx_lateral", "float", "m", "fixture: lateral offset of the approaching object", low=-1.0, high=1.0),
        Variable("fx_speed", "float", "m/s", "fixture: approach speed", low=2.0, high=speed_high),
        Variable("fx_surface", "categorical", "label", "fixture: surface", levels=("grip", "slip")),
    )


def _fixture_outcomes(situations: np.ndarray, side_speed: np.ndarray, delay: np.ndarray) -> np.ndarray:
    lateral, speed, slip = situations[:, 0], situations[:, 1], situations[:, 2]
    moving_time = np.clip(3.0 / speed - delay, 0.0, None)
    grip = np.where(slip == 1.0, 0.6, 1.0)
    displacement = side_speed * moving_time * grip
    separation = np.abs(lateral - displacement)
    fall = np.clip(0.02 + np.where(slip == 1.0, 1.2, 0.3) * side_speed**2, 0.0, 1.0)
    effort = np.abs(side_speed) * moving_time
    return np.column_stack([separation, fall, effort])


class _FixtureBase(SkillSpec):
    """Outcomes, objective, loss and success shared by the fixture specs."""

    def _set_common(self, name: str, speed_high: float) -> None:
        self.name = name
        self.description = "FIXTURE spec for spiking-student mechanics checks; not a project skill"
        self.situation = _fixture_situation(speed_high)
        self.outcomes = (
            Outcome("fx_separation", "regression", "m", "fixture separation", low=0.0),
            Outcome("fx_fall", "binary", "", "fixture fall"),
            Outcome("fx_effort", "regression", "m", "fixture side-step distance", low=0.0),
        )
        self.primary_outcome = "fx_separation"
        self.hidden_effects = ("fixture: none; this spec never runs an episode",)

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        return (
            np.minimum(predicted.median_of("fx_separation"), 0.5)
            - predicted.mean_of("fx_fall")
            - 0.1 * predicted.median_of("fx_effort")
        )

    def episode(
        self, world: Any, situation: Mapping[str, Any], action: Mapping[str, Any], attempt_seed: int
    ) -> Mapping[str, float]:
        raise RuntimeError("the fixture spec has no world; these checks never run episodes")

    def loss(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return measured[:, 1] + np.clip(0.5 - measured[:, 0], 0.0, None)

    def success(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return (measured[:, 0] >= 0.5) & (measured[:, 1] == 0.0)


class FixtureSnnSpec(_FixtureBase):
    """FIXTURE: an object approaches with a lateral offset; side-step now or later, slowly or fast.

    Best candidate: the smallest side-step that reaches the separation cap, away from the
    object, fewer falls on the slippery surface; stand still when the offset is already large.
    """

    def __init__(
        self,
        name: str = "fixture_snn",
        speed_high: float = 6.0,
        side_bounds: tuple[float, float] = (-0.6, 0.6),
        command_field: str = "fixture_vy",
    ) -> None:
        self._set_common(name, speed_high)
        self.actions = (
            ActionDim(
                "fx_side_speed", "float", "m/s", "fixture side-step command", low=side_bounds[0], high=side_bounds[1],
                role="command", command_field=command_field,
            ),
            ActionDim("fx_delay", "float", "s", "fixture delay before the side-step", low=0.0, high=0.4, role="timing"),
        )

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        situations = np.asarray(situations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        return _fixture_outcomes(situations, actions[:, 0], actions[:, 1])


class FixtureReversedGridSpec(FixtureSnnSpec):
    """FIXTURE: FixtureSnnSpec with identical action dimensions whose candidate_actions returns
    the same candidates in reverse order (a skill's own candidate_actions changed)."""

    def candidate_actions(self, situation: np.ndarray, points_per_dim: Sequence[int]) -> np.ndarray:
        return super().candidate_actions(situation, points_per_dim)[::-1].copy()


class FixtureSideSpec(_FixtureBase):
    """FIXTURE: the side command is categorical: each level sends its command value."""

    def __init__(
        self,
        name: str = "fixture_side",
        levels: tuple[str, ...] = ("left", "stay", "right"),
        command_values: tuple[float, ...] = (0.5, 0.0, -0.5),
    ) -> None:
        self._set_common(name, 6.0)
        self.actions = (
            ActionDim(
                "fx_side", "categorical", "m/s", "fixture side-step direction", levels=levels,
                role="command", command_field="fixture_vy", command_values=command_values,
            ),
            ActionDim("fx_delay", "float", "s", "fixture delay before the side-step", low=0.0, high=0.4, role="timing"),
        )

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        situations = np.asarray(situations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        side_speed = np.asarray(self.actions[0].command_values, dtype=np.float64)[actions[:, 0].astype(np.int64)]
        return _fixture_outcomes(situations, side_speed, actions[:, 1])


def fixture_dreams(
    spec: SkillSpec,
    config: Mapping[str, Any],
    n: int = FIXTURE_N_DREAMS,
    *,
    seed: int = FIXTURE_DREAM_SEED,
    points_per_dim: tuple[int, ...] = FIXTURE_POINTS_PER_DIM,
    stream: str | None = None,
) -> DreamSet:
    """Physics-only DreamSet of a fixture spec, built by the real engine.dream.dream."""
    return dream(
        spec,
        None,
        build_layout(spec),
        n_dreams=n,
        rng=stream_rng(seed, stream or f"{spec.name}/dreams/situations"),
        points_per_dim=points_per_dim,
        quantile_levels=require(config, "engine.quantile_levels"),
        spread_levels=tuple(require(config, "engine.spread_levels")),
        method=PHYSICS_ONLY,
    )


class Checks:
    """Counts executed assertions of a check script and sets its exit code."""

    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []
        self.redactions: dict[str, str] = {}  # e.g. {temporary directory: "<tmp>"} for printed labels

    def expect(self, condition: bool, label: str) -> None:
        for text, replacement in self.redactions.items():
            label = label.replace(text, replacement)
        print(("  [ok] " if condition else "  [FAILED] ") + label)
        if condition:
            self.passed += 1
        else:
            self.failures.append(label)

    def raises(self, label: str, error: type[BaseException], function: Callable[[], Any], needle: str) -> BaseException | None:
        try:
            function()
        except error as caught:
            message = str(caught).replace("\n", " ")
            self.expect(needle in message, f"{label} -> {type(caught).__name__}: {message[:220]}")
            return caught
        except Exception as other:  # noqa: BLE001
            self.expect(False, f"{label} raised {type(other).__name__} instead of {error.__name__}: {other}")
            return None
        self.expect(False, f"{label} did not raise {error.__name__}")
        return None

    def finish(self) -> None:
        print(f"\nCHECK RESULT: {self.passed} assertions held, {len(self.failures)} failed {self.failures or ''}")
        raise SystemExit(0 if not self.failures else 1)
