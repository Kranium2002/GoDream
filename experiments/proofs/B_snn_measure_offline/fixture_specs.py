"""FIXTURE specs, hand-built deployed students and dream-row helpers for the measurement checks.

FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model.
Nothing here is a project skill, the Go2 or DODGE. The specs never run an episode. The hand-built
students have weights written by hand and were not trained in any dream; their records say so.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import numpy as np

from _checks import BANNER  # noqa: F401  (also puts the repository root on sys.path)
from learned_judgment.engine.decide import PHYSICS_ONLY
from learned_judgment.engine.dream import DreamSet, dream
from learned_judgment.engine.features import build_layout
from learned_judgment.engine.types import OutcomeDistribution
from learned_judgment.skills.base import ActionDim, Outcome, SkillSpec, Variable
from learned_judgment.students.encoding import InputEncoder
from learned_judgment.students.snn import (
    DeployedStudent,
    action_record,
    candidate_grid_sha256,
    situation_digests,
)
from learned_judgment.students.snn_core import SpikingNetwork

ROOT3 = math.sqrt(3.0)  # low = -sqrt(3), high = sqrt(3): midpoint 0 and scale 2 sqrt(3) / sqrt(12) = 1, so the encoding is the identity
HAND_META_NOTE = "FIXTURE: hand-built weights, not trained in any dream; mechanics only"


class FixtureMeasureSpec(SkillSpec):
    """FIXTURE: an object approaches with a lateral offset; side-step at some speed after some wait.

    Best candidate: the smallest side-step (away from the object, on a floor where falls are
    likelier when wet) whose lower-quantile gap reaches the 0.4 m cap; stand still when the
    offset is already large. There is no world: episode() refuses to run.
    """

    def __init__(self, name: str = "fixture_measure") -> None:
        self.name = name
        self.description = "FIXTURE spec for measurement mechanics checks; not a project skill"
        self.situation = (
            Variable("fx_offset", "float", "m", "fixture: lateral offset of the approaching object", low=-1.0, high=1.0),
            Variable("fx_speed", "float", "m/s", "fixture: approach speed", low=2.0, high=6.0),
            Variable("fx_floor", "categorical", "label", "fixture: floor", levels=("dry", "wet")),
        )
        self.actions = (
            ActionDim("fx_vy", "float", "m/s", "fixture side-step speed", low=-0.6, high=0.6, role="command", command_field="fixture_vy"),
            ActionDim("fx_wait", "float", "s", "fixture wait before the side-step", low=0.0, high=0.4, role="timing"),
        )
        self.outcomes = (
            Outcome("fx_gap", "regression", "m", "fixture closest gap", low=0.0),
            Outcome("fx_fall", "binary", "", "fixture fall"),
            Outcome("fx_effort", "regression", "m", "fixture side-step distance", low=0.0),
        )
        self.primary_outcome = "fx_gap"
        self.hidden_effects = ("fixture: none; this spec never runs an episode",)

    @staticmethod
    def terms(situations: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, ...]:
        situations, actions = np.asarray(situations, dtype=np.float64), np.asarray(actions, dtype=np.float64)
        offset, speed, wet = situations[:, 0], situations[:, 1], situations[:, 2]
        vy, wait = actions[:, 0], actions[:, 1]
        moving = np.clip(2.5 / speed - wait, 0.0, None)
        grip = np.where(wet == 1.0, 0.6, 1.0)
        return offset, wet, vy, moving, grip

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        offset, wet, vy, moving, grip = self.terms(situations, actions)
        gap = np.abs(offset - vy * moving * grip)
        fall = np.clip(0.03 + np.where(wet == 1.0, 1.0, 0.25) * vy**2, 0.0, 1.0)
        return np.column_stack([gap, fall, np.abs(vy) * moving])

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        lowest = float(predicted.quantile_levels[0])  # risk-aware: the lowest predictive quantile of the gap
        return (
            np.minimum(predicted.quantile_of("fx_gap", lowest), 0.4)
            - predicted.mean_of("fx_fall")
            - 0.1 * predicted.median_of("fx_effort")
        )

    def episode(self, world: Any, situation: Mapping[str, Any], action: Mapping[str, Any], attempt_seed: int) -> Mapping[str, float]:
        raise RuntimeError("the fixture spec has no world; these checks never run episodes")

    def loss(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return measured[:, 1] + np.clip(0.4 - measured[:, 0], 0.0, None)

    def success(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return (measured[:, 0] >= 0.4) & (measured[:, 1] == 0.0)


def fixture_attempt_outcomes(spec: FixtureMeasureSpec, situations: np.ndarray, actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """FIXTURE attempt outcomes (made up, NOT simulator episodes): in this fixture 'world' a side-step
    moves 70% as far as the physics guess says, falls are 1.5x likelier, and the gap is noisy."""
    offset, wet, vy, moving, grip = spec.terms(situations, actions)
    n = situations.shape[0]
    gap = np.clip(np.abs(offset - 0.7 * vy * moving * grip) + rng.normal(0.0, 0.03, n), 0.0, None)
    fall_probability = np.clip(1.5 * (0.03 + np.where(wet == 1.0, 1.0, 0.25) * vy**2), 0.0, 1.0)
    fall = (rng.uniform(size=n) < fall_probability).astype(np.float64)
    return np.column_stack([gap, fall, 0.7 * np.abs(vy) * moving])


class FixtureHandSpec(SkillSpec):
    """FIXTURE spec of the hand-built networks: 1 or 2 float situation features on [-sqrt 3, sqrt 3]
    (so the student's encoding is the identity) and one float action; the physics guess says the
    gap is |0.5 fx_a - fx_choice|. Counts its candidate_actions calls (``grid_calls``)."""

    def __init__(self, name: str, n_inputs: int, *, choice_low: float = -0.5, choice_high: float = 0.5) -> None:
        self.name = name
        self.description = "FIXTURE spec of a hand-built spiking network; not a project skill"
        variables = (
            Variable("fx_a", "float", "1", "fixture hand input a", low=-ROOT3, high=ROOT3),
            Variable("fx_b", "float", "1", "fixture hand input b", low=-ROOT3, high=ROOT3),
        )
        self.situation = variables[:n_inputs]
        self.actions = (
            ActionDim("fx_choice", "float", "m/s", "fixture choice", low=choice_low, high=choice_high, role="command", command_field="fixture_vy"),
        )
        self.outcomes = (Outcome("fx_gap", "regression", "m", "fixture gap", low=0.0),)
        self.primary_outcome = "fx_gap"
        self.hidden_effects = ("fixture: none; this spec never runs an episode",)
        self.grid_calls = 0

    def candidate_actions(self, situation: np.ndarray, points_per_dim: Sequence[int]) -> np.ndarray:
        self.grid_calls += 1
        return super().candidate_actions(situation, points_per_dim)

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        situations, actions = np.asarray(situations, dtype=np.float64), np.asarray(actions, dtype=np.float64)
        return np.abs(0.5 * situations[:, :1] - actions[:, :1])

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        return -predicted.mean_of("fx_gap")

    def episode(self, world: Any, situation: Mapping[str, Any], action: Mapping[str, Any], attempt_seed: int) -> Mapping[str, float]:
        raise RuntimeError("the fixture spec has no world; these checks never run episodes")

    def loss(self, measured: np.ndarray) -> np.ndarray:
        return np.asarray(measured, dtype=np.float64)[:, 0]

    def success(self, measured: np.ndarray) -> np.ndarray:
        return np.asarray(measured, dtype=np.float64)[:, 0] <= 0.1


def hand_deployed(
    network: SpikingNetwork,
    spec: SkillSpec,
    points_per_dim: Sequence[int],
    training_situations: np.ndarray,
    *,
    out_of_range: str = "error",
    meta: Mapping[str, Any] | None = None,
) -> DeployedStudent:
    """A DeployedStudent built by hand around ``network`` for ``spec`` (encoding = the spec's own
    InputEncoder). ``training_situations`` (k, S) are FIXTURE situations standing for what training
    saw: their digests become training_situation_digests and the first 16 distinct ones the grid probe."""
    encoder = InputEncoder.from_variables(spec.situation)
    training = np.asarray(training_situations, dtype=np.float64)
    probe = np.unique(training, axis=0)[:16]
    grid = candidate_grid_sha256(spec, probe, points_per_dim)
    return DeployedStudent(
        skill=spec.name,
        network=network,
        encode=encoder.encode,
        encode_one=encoder.encode_one,
        situation_variables=encoder.variables,
        points_per_dim=tuple(points_per_dim),
        n_candidates=network.n_outputs,
        action_records=tuple(action_record(dim) for dim in spec.actions),
        grid_probe=probe,
        grid_sha256=grid,
        training_situation_digests=frozenset(situation_digests(training)),
        out_of_range=out_of_range,
        meta=dict(meta) if meta is not None else {
            "kind": "fixture_hand_built_network",
            "label": HAND_META_NOTE,
            "dream": {"method": PHYSICS_ONLY, "request_hashes": [], "note": HAND_META_NOTE},
        },
    )


def make_dreams(
    spec: SkillSpec, model: Any, method: str, rng: np.random.Generator, n: int, points_per_dim: Sequence[int], config: Mapping[str, Any]
) -> DreamSet:
    """engine.dream.dream with the engine quantile and spread levels of ``config``."""
    from learned_judgment.config import require

    return dream(
        spec,
        model,
        build_layout(spec),
        n_dreams=n,
        rng=rng,
        points_per_dim=points_per_dim,
        quantile_levels=require(config, "engine.quantile_levels"),
        spread_levels=tuple(require(config, "engine.spread_levels")),
        method=method,
    )


def _row_fields(dreams: DreamSet) -> list[str]:
    return [
        field.name
        for field in dataclasses.fields(dreams)
        if isinstance(getattr(dreams, field.name), np.ndarray) and getattr(dreams, field.name).shape[:1] == (dreams.n_rows,)
    ]


def dream_rows(dreams: DreamSet, rows: Sequence[int] | np.ndarray) -> DreamSet:
    """A DreamSet holding the given rows of ``dreams`` (every per-row array indexed alike)."""
    index = np.asarray(rows, dtype=np.int64)
    return dataclasses.replace(dreams, **{name: getattr(dreams, name)[index] for name in _row_fields(dreams)})


def concat_dreams(first: DreamSet, second: DreamSet) -> DreamSet:
    """The rows of ``first`` followed by those of ``second`` (both without provider requests)."""
    if first.request_groups or second.request_groups:
        raise ValueError("concat_dreams joins dreams without provider requests only")
    names = _row_fields(first)
    return dataclasses.replace(first, **{name: np.concatenate([getattr(first, name), getattr(second, name)]) for name in names})
