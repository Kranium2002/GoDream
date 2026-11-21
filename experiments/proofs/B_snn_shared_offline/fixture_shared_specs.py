"""FIXTURE SkillSpecs and helpers for the B_snn_shared_offline checks only.

Not project skills, not the Go2, not DODGE/BRAKE/TURN/RAMP: made-up specs with different
situation widths and different candidate counts. In each, the dream's best candidate is the grid
action nearest a smooth target that depends on the situation, minus a small fall risk that grows
with the first action's magnitude, so a dream built from it has something to learn. Their dreams
are built with the REAL engine (engine.dream.dream with model=None, method=PHYSICS_ONLY), so the
scores come from the real decide step. There is no world; episode() refuses to run.

    fixture   S  encoded D  actions                                  candidates  dream rows
    fx_alpha  3  4          vy float x delay timing     (7, 5)        35          1000
    fx_beta   5  5          speed float                 (15,)         15          800
    fx_gamma  2  3          side categorical x yaw float (3, 6)       18          600
    fx_delta  4  4          speed float                 (11,)         11          600   (added later)
    fx_wide   7  7          speed float                 (5,)          5           50    (too wide: refused)

Spec variants (for the grid guard): fx_alpha with other side-speed bounds (its action records
change) and fx_alpha with its candidate order reversed (only the grid digest can see it).

The check config is config.yaml loaded in memory with students.snn.shared_feature_width filled in
(6, fixture value; config.yaml leaves it null until the project specs exist). Every other value,
including students.snn.shared_skill_slots, is config.yaml's. config.yaml itself is never modified.

Nothing computed with these fixtures is evidence about the robot, the simulator, TabPFN or any
model, and none of it may enter a report or results/.
"""

from __future__ import annotations

import copy
import dataclasses
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
FIXTURE_FEATURE_WIDTH = 6  # fixture value of students.snn.shared_feature_width (widest fixture skill encodes 5)
FIXTURE_DREAM_SEED = 20261121  # fixture seed of the dreamed situations
SEQUENCE = ("fx_alpha", "fx_beta", "fx_gamma")  # the three fixture skills added in order
PER_ROW_FIELDS = (
    "situations",
    "candidates",
    "scores",
    "chosen",
    "physics_choice",
    "chosen_spread",
    "request_group",
    "chosen_physics",
    "chosen_mean",
    "chosen_median",
    "chosen_outcome_spread",
)


def _float(name: str, unit: str, low: float, high: float) -> Variable:
    return Variable(name, "float", unit, f"fixture: {name}", low=low, high=high)


class FixtureSharedSpec(SkillSpec):
    """FIXTURE: choose the grid action nearest a situation-dependent target, with a fall risk."""

    def __init__(
        self,
        name: str,
        situation: tuple[Variable, ...],
        actions: tuple[ActionDim, ...],
        mixing_seed: int,
        reverse_grid: bool = False,
    ) -> None:
        self.name = name
        self.description = "FIXTURE spec for shared spiking-student mechanics checks; not a project skill"
        self.situation = situation
        self.actions = actions
        self.outcomes = (
            Outcome("fx_miss", "regression", "1", "fixture: normalised distance to the situation's target action", low=0.0),
            Outcome("fx_fall", "binary", "", "fixture: fall"),
        )
        self.primary_outcome = "fx_miss"
        self.hidden_effects = ("fixture: none; this spec never runs an episode",)
        self._mixing = np.random.default_rng(mixing_seed).normal(0.0, 1.0, size=(len(situation), len(actions)))
        self._reverse_grid = reverse_grid

    @staticmethod
    def _unit(variables: tuple[Variable, ...], values: np.ndarray) -> np.ndarray:
        columns = []
        for index, variable in enumerate(variables):
            column = values[:, index]
            if variable.kind == "categorical":
                columns.append(2.0 * column / (len(variable.levels) - 1) - 1.0)
            else:
                columns.append(2.0 * (column - variable.low) / (variable.high - variable.low) - 1.0)
        return np.column_stack(columns)

    def candidate_actions(self, situation: np.ndarray, points_per_dim: Sequence[int]) -> np.ndarray:
        grid = super().candidate_actions(situation, points_per_dim)
        return grid[::-1].copy() if self._reverse_grid else grid

    def physics_guess(self, situations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        situations = np.asarray(situations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        target = np.tanh(self._unit(self.situation, situations) @ self._mixing)
        unit_actions = self._unit(self.actions, actions)
        miss = np.sqrt(np.sum((unit_actions - target) ** 2, axis=1))
        fall = np.clip(0.05 + 0.2 * unit_actions[:, 0] ** 2, 0.0, 1.0)
        return np.column_stack([miss, fall])

    def objective(self, predicted: OutcomeDistribution) -> np.ndarray:
        return -predicted.median_of("fx_miss") - 0.5 * predicted.mean_of("fx_fall")

    def episode(
        self, world: Any, situation: Mapping[str, Any], action: Mapping[str, Any], attempt_seed: int
    ) -> Mapping[str, float]:
        raise RuntimeError("the fixture spec has no world; these checks never run episodes")

    def loss(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return measured[:, 0] + measured[:, 1]

    def success(self, measured: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured, dtype=np.float64)
        return (measured[:, 0] < 0.2) & (measured[:, 1] == 0.0)


def _speed(name: str, low: float, high: float) -> ActionDim:
    return ActionDim(name, "float", "m/s", f"fixture: {name}", low=low, high=high, role="command", command_field="fixture_vx")


def fixture_spec(
    name: str,
    *,
    alpha_speed_high: float = 2.0,
    alpha_vy_bounds: tuple[float, float] = (-0.6, 0.6),
    reverse_grid: bool = False,
) -> FixtureSharedSpec:
    """The named fixture spec (see the module table). ``alpha_speed_high`` changes fx_alpha's
    declared speed range (another situation encoding); ``alpha_vy_bounds`` changes its side-speed
    action bounds (other action records, other grid); ``reverse_grid`` reverses the candidate
    order of any fixture spec (same action records, other grid digest)."""
    if name == "fx_alpha":
        situation = (
            _float("fx_a_offset", "m", -1.0, 1.0),
            _float("fx_a_speed", "m/s", 0.5, alpha_speed_high),
            Variable("fx_a_surface", "categorical", "label", "fixture: surface", levels=("grip", "slip")),
        )
        actions = (
            ActionDim(
                "fx_a_vy", "float", "m/s", "fixture: side speed", low=alpha_vy_bounds[0], high=alpha_vy_bounds[1],
                role="command", command_field="fixture_vy",
            ),
            ActionDim("fx_a_delay", "float", "s", "fixture: delay", low=0.0, high=0.4, role="timing"),
        )
        return FixtureSharedSpec(name, situation, actions, mixing_seed=101, reverse_grid=reverse_grid)
    if name == "fx_beta":
        situation = (
            _float("fx_b_x", "m", -2.0, 2.0),
            _float("fx_b_y", "m", -2.0, 2.0),
            _float("fx_b_vx", "m/s", -1.0, 1.0),
            _float("fx_b_vy", "m/s", -1.0, 1.0),
            _float("fx_b_slope", "deg", -5.0, 5.0),
        )
        return FixtureSharedSpec(name, situation, (_speed("fx_b_speed", 0.0, 0.9),), mixing_seed=102, reverse_grid=reverse_grid)
    if name == "fx_gamma":
        situation = (
            _float("fx_g_gap", "m", 0.2, 1.2),
            Variable("fx_g_floor", "categorical", "label", "fixture: floor", levels=("tile", "carpet")),
        )
        actions = (
            ActionDim(
                "fx_g_side", "categorical", "label", "fixture: side", levels=("left", "stay", "right"),
                role="command", command_field="fixture_side", command_values=(0.3, 0.0, -0.3),
            ),
            ActionDim("fx_g_yaw", "float", "rad/s", "fixture: yaw rate", low=-0.5, high=0.5, role="command", command_field="fixture_yaw"),
        )
        return FixtureSharedSpec(name, situation, actions, mixing_seed=103, reverse_grid=reverse_grid)
    if name == "fx_delta":
        situation = tuple(_float(f"fx_d_{letter}", "1", -1.0, 1.0) for letter in "pqrs")
        return FixtureSharedSpec(name, situation, (_speed("fx_d_speed", -0.3, 0.9),), mixing_seed=104, reverse_grid=reverse_grid)
    if name == "fx_wide":
        situation = tuple(_float(f"fx_w_{index}", "1", -1.0, 1.0) for index in range(7))
        return FixtureSharedSpec(name, situation, (_speed("fx_w_speed", 0.0, 0.9),), mixing_seed=105, reverse_grid=reverse_grid)
    raise ValueError(f"unknown fixture spec {name!r}")


POINTS = {
    "fx_alpha": (7, 5),
    "fx_beta": (15,),
    "fx_gamma": (3, 6),
    "fx_delta": (11,),
    "fx_wide": (5,),
}
N_DREAMS = {"fx_alpha": 1000, "fx_beta": 800, "fx_gamma": 600, "fx_delta": 600, "fx_wide": 50}


def fixture_dreams(spec: SkillSpec, config: Mapping[str, Any], n: int | None = None) -> DreamSet:
    """Physics-only DreamSet of a fixture spec, built by the real engine.dream.dream."""
    return dream(
        spec,
        None,
        build_layout(spec),
        n_dreams=N_DREAMS[spec.name] if n is None else n,
        rng=stream_rng(FIXTURE_DREAM_SEED, f"{spec.name}/dreams/situations"),
        points_per_dim=POINTS[spec.name],
        quantile_levels=require(config, "engine.quantile_levels"),
        spread_levels=tuple(require(config, "engine.spread_levels")),
        method=PHYSICS_ONLY,
    )


def repeated_dreams(dreams: DreamSet, times: int) -> DreamSet:
    """FIXTURE: the dream with every row repeated ``times`` times (rows i, i + n, i + 2n, ... are
    the same situation), as a dream assembled from overlapping batches would be."""
    return dataclasses.replace(dreams, **{name: np.concatenate([getattr(dreams, name)] * times) for name in PER_ROW_FIELDS})


def fixture_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """config.yaml (already loaded) with the fixture students.snn.shared_feature_width."""
    result = copy.deepcopy(dict(config))
    result["students"]["snn"]["shared_feature_width"] = FIXTURE_FEATURE_WIDTH
    return result


class Checks:
    """Counts executed assertions of a check script and sets its exit code."""

    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []
        self.redactions: dict[str, str] = {}

    def clean(self, text: str) -> str:
        """``text`` with every redaction applied (e.g. the temporary directory -> "<tmp>")."""
        for needle, replacement in self.redactions.items():
            text = text.replace(needle, replacement)
        return text

    def expect(self, condition: bool, label: str) -> None:
        label = self.clean(label)
        print(("  [ok] " if condition else "  [FAILED] ") + label)
        if condition:
            self.passed += 1
        else:
            self.failures.append(label)

    def raises(self, label: str, error: type[BaseException], function: Callable[[], Any], needle: str) -> None:
        try:
            function()
        except error as caught:
            message = self.clean(str(caught)).replace("\n", " ")
            self.expect(needle in message, f"{label} -> {type(caught).__name__}: {message[:190]}")
            return
        except Exception as other:  # noqa: BLE001
            self.expect(False, f"{label} raised {type(other).__name__} instead of {error.__name__}: {other}")
            return
        self.expect(False, f"{label} did not raise {error.__name__}")

    def finish(self) -> None:
        print(f"\nCHECK RESULT: {self.passed} assertions held, {len(self.failures)} failed {self.failures or ''}")
        raise SystemExit(0 if not self.failures else 1)
