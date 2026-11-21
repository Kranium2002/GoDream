"""Feature matrix of a residual model: one fixed column order per skill.

    [ situation features | candidate action | physics guess | extra observed context ]

    situation  S columns, ``spec.situation`` order
    action     A columns, ``spec.actions`` order
    physics    K columns named ``physics_<outcome>``, ``spec.outcomes`` order
               (dropped for the "direct, no physics feature" baseline)
    extra      optional observed context declared by the caller, e.g. a declared world id
               used by the adaptation experiment -- never a hidden simulator value

Categorical columns hold the float index of the level; :attr:`FeatureLayout.categorical`
tells a model which columns those are and what the levels mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from learned_judgment.skills.base import SkillSpec, Variable

__all__ = ["FeatureLayout", "build_features", "build_layout"]


@dataclass(frozen=True)
class FeatureLayout:
    """Column layout of a feature matrix.

    Attributes:
        skill:        SkillSpec name the layout belongs to.
        names:        F column names, in order.
        categorical:  ((column index, levels), ...) for every categorical column.
        n_situation, n_action, n_physics, n_extra: block widths; they sum to F.
        outcome_names: K outcome names, the order of the residual targets.
        outcome_kinds: K outcome kinds ("regression" | "binary"), same order.
    """

    skill: str
    names: tuple[str, ...]
    categorical: tuple[tuple[int, tuple[str, ...]], ...]
    n_situation: int
    n_action: int
    n_physics: int
    n_extra: int
    outcome_names: tuple[str, ...]
    outcome_kinds: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return len(self.names)

    @property
    def n_outcomes(self) -> int:
        return len(self.outcome_names)

    @property
    def categorical_indices(self) -> tuple[int, ...]:
        return tuple(index for index, _ in self.categorical)

    def describe(self) -> dict[str, Any]:
        """JSON-serializable description; part of every request-cache key and of the handoff."""
        return {
            "skill": self.skill,
            "names": list(self.names),
            "categorical": [[index, list(levels)] for index, levels in self.categorical],
            "blocks": {
                "situation": self.n_situation,
                "action": self.n_action,
                "physics": self.n_physics,
                "extra": self.n_extra,
            },
            "outcome_names": list(self.outcome_names),
            "outcome_kinds": list(self.outcome_kinds),
        }


def build_layout(
    spec: SkillSpec, *, include_physics: bool = True, extra: Sequence[Variable] = ()
) -> FeatureLayout:
    """Layout for ``spec``. ``include_physics=False`` gives the direct (no physics feature) layout."""
    names: list[str] = []
    categorical: list[tuple[int, tuple[str, ...]]] = []

    def add(variable: Variable) -> None:
        if variable.kind == "categorical":
            categorical.append((len(names), tuple(variable.levels)))
        names.append(variable.name)

    for variable in spec.situation:
        add(variable)
    for dim in spec.actions:
        add(dim)
    n_physics = 0
    if include_physics:
        for outcome in spec.outcomes:
            names.append(f"physics_{outcome.name}")
        n_physics = spec.n_outcome
    for variable in extra:
        problems = variable.problems()
        if problems:
            raise ValueError("; ".join(problems))
        add(variable)
    if len(set(names)) != len(names):
        raise ValueError(f"feature names must be unique; got {names}")
    return FeatureLayout(
        skill=spec.name,
        names=tuple(names),
        categorical=tuple(categorical),
        n_situation=spec.n_situation,
        n_action=spec.n_action,
        n_physics=n_physics,
        n_extra=len(extra),
        outcome_names=spec.outcome_names,
        outcome_kinds=tuple(outcome.kind for outcome in spec.outcomes),
    )


def build_features(
    layout: FeatureLayout,
    situations: np.ndarray,
    actions: np.ndarray,
    *,
    physics: np.ndarray | None = None,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the float64 (n, F) feature matrix in ``layout`` order.

    ``physics`` (n, K) is required exactly when the layout has a physics block and ``extra``
    (n, E) exactly when it has an extra block. Shapes are checked; nothing is broadcast.
    """
    situations = np.asarray(situations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if situations.ndim != 2 or situations.shape[1] != layout.n_situation:
        raise ValueError(f"situations must have shape (n, {layout.n_situation}); got {situations.shape}")
    n = situations.shape[0]
    if actions.shape != (n, layout.n_action):
        raise ValueError(f"actions must have shape ({n}, {layout.n_action}); got {actions.shape}")
    blocks = [situations, actions]
    for label, block, width in (("physics", physics, layout.n_physics), ("extra", extra, layout.n_extra)):
        if width == 0:
            if block is not None:
                raise ValueError(f"the layout has no {label} block but {label} was given")
            continue
        if block is None:
            raise ValueError(f"the layout needs a {label} block of shape ({n}, {width})")
        block = np.asarray(block, dtype=np.float64)
        if block.shape != (n, width):
            raise ValueError(f"{label} must have shape ({n}, {width}); got {block.shape}")
        blocks.append(block)
    features = np.concatenate(blocks, axis=1)
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain non-finite values")
    return features


def layout_from_description(description: Mapping[str, Any]) -> FeatureLayout:
    """Inverse of :meth:`FeatureLayout.describe` (used when reloading saved artifacts)."""
    blocks = description["blocks"]
    return FeatureLayout(
        skill=str(description["skill"]),
        names=tuple(description["names"]),
        categorical=tuple((int(index), tuple(levels)) for index, levels in description["categorical"]),
        n_situation=int(blocks["situation"]),
        n_action=int(blocks["action"]),
        n_physics=int(blocks["physics"]),
        n_extra=int(blocks["extra"]),
        outcome_names=tuple(description["outcome_names"]),
        outcome_kinds=tuple(description["outcome_kinds"]),
    )
