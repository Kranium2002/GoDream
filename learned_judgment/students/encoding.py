"""Input encoding of the spiking student: situation features -> constant input current.

The encoding is fixed by the SkillSpec alone (no statistics are fitted to data), so it is
identical for training, deployment and every rehearsal set:

    float feature        (x - midpoint) / (range / sqrt(12))
                         a situation drawn uniformly over the declared range has mean 0 and
                         standard deviation 1
    categorical feature  one column per level, 1.0 for the active level and 0.0 elsewhere

The shared multi-skill network adds zero padding up to a fixed width and a one-hot skill ID.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from learned_judgment.skills.base import Variable

__all__ = ["InputEncoder", "SharedInputEncoder"]

_UNIFORM_STD_DIVISOR = math.sqrt(12.0)  # standard deviation of a uniform variable = range / sqrt(12)


@dataclass(frozen=True)
class InputEncoder:
    """Encoder of one skill's situation vectors.

    Attributes:
        variables: the skill's situation variables, in SkillSpec order.
        names:     D encoded column names.
    """

    variables: tuple[Variable, ...]
    names: tuple[str, ...]

    @classmethod
    def from_variables(cls, variables: Sequence[Variable]) -> "InputEncoder":
        variables = tuple(variables)
        if not variables:
            raise ValueError("an encoder needs at least one situation variable")
        names: list[str] = []
        for variable in variables:
            problems = variable.problems()
            if problems:
                raise ValueError("; ".join(problems))
            if variable.kind == "categorical":
                names.extend(f"{variable.name}={level}" for level in variable.levels)
            else:
                names.append(variable.name)
        return cls(variables=variables, names=tuple(names))

    @property
    def width(self) -> int:
        """D, the number of encoded columns."""
        return len(self.names)

    def encode(self, situations: np.ndarray) -> np.ndarray:
        """Encode situations (n, S) into float64 (n, D)."""
        situations = np.asarray(situations, dtype=np.float64)
        if situations.ndim != 2 or situations.shape[1] != len(self.variables):
            raise ValueError(f"situations must have shape (n, {len(self.variables)}); got {situations.shape}")
        if not np.all(np.isfinite(situations)):
            raise ValueError("situations contain non-finite values")
        columns: list[np.ndarray] = []
        for index, variable in enumerate(self.variables):
            values = situations[:, index]
            if variable.kind == "categorical":
                levels = np.arange(len(variable.levels), dtype=np.float64)
                if not np.all(np.isin(values, levels)):
                    raise ValueError(f"{variable.name}: values are not valid level indices")
                columns.extend((values == level).astype(np.float64) for level in levels)
            else:
                midpoint = 0.5 * (variable.low + variable.high)
                scale = (variable.high - variable.low) / _UNIFORM_STD_DIVISOR
                columns.append((values - midpoint) / scale)
        return np.column_stack(columns)

    def encode_one(self, situation: np.ndarray) -> np.ndarray:
        """Encode one situation (S,) into float64 (D,): the per-decision path.

        Uses the same expressions as :meth:`encode`, so the result is bitwise equal to the
        corresponding row of ``encode(situation[None, :])``.
        """
        if len(situation) != len(self.variables):
            raise ValueError(f"situation must have {len(self.variables)} values; got {len(situation)}")
        encoded = np.zeros(len(self.names))
        column = 0
        for index, variable in enumerate(self.variables):
            value = float(situation[index])
            if variable.kind == "categorical":
                n_levels = len(variable.levels)
                if not value.is_integer() or not 0 <= int(value) < n_levels:
                    raise ValueError(f"{variable.name}: {value} is not a valid level index")
                encoded[column + int(value)] = 1.0
                column += n_levels
            else:
                if not math.isfinite(value):
                    raise ValueError(f"{variable.name}: value is not finite")
                midpoint = 0.5 * (variable.low + variable.high)
                scale = (variable.high - variable.low) / _UNIFORM_STD_DIVISOR
                encoded[column] = (value - midpoint) / scale
                column += 1
        return encoded

    def describe(self) -> dict[str, Any]:
        """JSON-serializable description stored with every trained network."""
        return {
            "kind": "range_standardized_one_hot",
            "names": list(self.names),
            "variables": [
                {
                    "name": variable.name,
                    "kind": variable.kind,
                    "unit": variable.unit,
                    "low": variable.low,
                    "high": variable.high,
                    "levels": list(variable.levels),
                }
                for variable in self.variables
            ],
        }

    def matches(self, description: Mapping[str, Any]) -> bool:
        """True when ``description`` (from a saved network) is this exact encoding."""
        return dict(description) == self.describe()


@dataclass(frozen=True)
class SharedInputEncoder:
    """Encoder of the shared multi-skill network.

    Layout of one encoded row (width = ``feature_width`` + ``n_slots``):

        [ the skill's own encoding, zero-padded to feature_width | one-hot skill slot ]

    ``n_slots`` and ``feature_width`` are fixed when the network is created (config.yaml),
    because the input width of a trained network cannot change. A skill takes the next free
    slot when it is added, and keeps it: ``skills`` lists the skills assigned so far, and a
    skill's slot is its position in that tuple. Unassigned slots are inputs that stay zero.
    """

    n_slots: int
    feature_width: int
    skills: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        skills = tuple(self.skills)
        if int(self.n_slots) < 1:
            raise ValueError(f"n_slots must be >= 1; got {self.n_slots}")
        if int(self.feature_width) < 1:
            raise ValueError(f"feature_width must be >= 1; got {self.feature_width}")
        if len(set(skills)) != len(skills) or not all(isinstance(skill, str) and skill for skill in skills):
            raise ValueError(f"skills must be unique non-empty names; got {skills}")
        if len(skills) > int(self.n_slots):
            raise ValueError(f"{len(skills)} skills do not fit into {self.n_slots} slots")
        object.__setattr__(self, "n_slots", int(self.n_slots))
        object.__setattr__(self, "feature_width", int(self.feature_width))
        object.__setattr__(self, "skills", skills)

    @property
    def width(self) -> int:
        return self.feature_width + self.n_slots

    @property
    def free_slots(self) -> int:
        return self.n_slots - len(self.skills)

    def slot_of(self, skill: str) -> int:
        try:
            return self.skills.index(skill)
        except ValueError:
            raise ValueError(f"skill {skill!r} has no slot; assigned skills: {self.skills}") from None

    def with_skill(self, skill: str) -> "SharedInputEncoder":
        """The encoder after ``skill`` took the next free slot. Existing slots never move."""
        if skill in self.skills:
            raise ValueError(f"skill {skill!r} already has slot {self.skills.index(skill)}")
        if self.free_slots == 0:
            raise ValueError(f"all {self.n_slots} skill slots are taken by {self.skills}")
        return SharedInputEncoder(n_slots=self.n_slots, feature_width=self.feature_width, skills=(*self.skills, skill))

    def _check_width(self, skill: str, encoder: InputEncoder) -> None:
        if encoder.width > self.feature_width:
            raise ValueError(
                f"skill {skill!r} needs {encoder.width} encoded columns but the shared feature width is "
                f"{self.feature_width}"
            )

    def encode(self, skill: str, encoder: InputEncoder, situations: np.ndarray) -> np.ndarray:
        """Encode one skill's situations (n, S) into the shared layout (n, width)."""
        slot = self.slot_of(skill)
        self._check_width(skill, encoder)
        own = encoder.encode(situations)
        rows = np.zeros((own.shape[0], self.width))
        rows[:, : own.shape[1]] = own
        rows[:, self.feature_width + slot] = 1.0
        return rows

    def encode_one(self, skill: str, encoder: InputEncoder, situation: np.ndarray) -> np.ndarray:
        """Encode one situation (S,) into the shared layout (width,): the per-decision path."""
        slot = self.slot_of(skill)
        self._check_width(skill, encoder)
        own = encoder.encode_one(situation)
        row = np.zeros(self.width)
        row[: own.shape[0]] = own
        row[self.feature_width + slot] = 1.0
        return row

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "shared_padded_slot_one_hot",
            "n_slots": self.n_slots,
            "feature_width": self.feature_width,
            "skills": list(self.skills),
        }
