"""Context selection and recovery after a world change (engine lane, Person B).

Lane contract
-------------
After the judge switches the world preset, the residual model is refitted on a context
chosen by one of the policies compared on dev data (03_METRICS.md, "Adaptation"):

    "all"     every real attempt, in file order
    "recent"  the last ``recent_window`` real attempts, in file order
    + world ID: either policy, with the declared preset name as an extra categorical
              feature (``world_id_variable`` / ``world_id_column``), in a layout built by
              :func:`world_id_layout`, which requires the hidden simulator value names so the
              extra block is checked against them. The preset name is an observed label of the
              experiment, never a hidden simulator value.

Dream rows are never context: they are teacher decisions, not measurements.

:func:`recovery_attempts` reads off how many new attempts were needed to come back within
``tolerance_fraction`` of the pre-shift error (config.yaml
``evaluation.recovery_tolerance_fraction``): the first count that reaches the threshold and
the count from which the error stays there; ``None`` means not reached.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from learned_judgment.engine.features import FeatureLayout, build_layout
from learned_judgment.engine.rows import AttemptRow
from learned_judgment.skills.base import SkillSpec, Variable

__all__ = [
    "CONTEXT_POLICIES",
    "WORLD_ID",
    "recovery_attempts",
    "select_context",
    "world_id_column",
    "world_id_layout",
    "world_id_variable",
]

CONTEXT_POLICIES = ("all", "recent")
WORLD_ID = "world_id"


def select_context(
    rows: Sequence[AttemptRow], *, policy: str, recent_window: int | None
) -> list[AttemptRow]:
    """Real rows of ``rows`` selected by ``policy``, in file order.

    "all" takes every real row and requires ``recent_window`` to be None; "recent" takes
    the last ``recent_window`` (int >= 1) real rows, or all of them when there are fewer.
    An unknown policy raises ValueError.
    """
    if policy not in CONTEXT_POLICIES:
        raise ValueError(f"unknown context policy {policy!r}; known policies: {CONTEXT_POLICIES}")
    real = []
    for position, row in enumerate(rows):
        if not isinstance(row, AttemptRow):
            raise TypeError(f"rows[{position}] is {type(row).__name__}, not AttemptRow")
        if row.source == "real":
            real.append(row)
    if policy == "all":
        if recent_window is not None:
            raise ValueError("recent_window is used only by policy 'recent'; pass None with policy 'all'")
        return real
    if isinstance(recent_window, bool) or not isinstance(recent_window, (int, np.integer)) or recent_window < 1:
        raise ValueError(f"policy 'recent' needs recent_window as an int >= 1; got {recent_window!r}")
    return real[-int(recent_window) :]


def world_id_variable(world_presets: Sequence[str]) -> Variable:
    """Categorical feature ``world_id`` whose levels are the DECLARED preset names, in the
    given order (pass the keys of config.yaml ``world.presets``)."""
    if isinstance(world_presets, str):
        raise TypeError("world_presets must be a sequence of preset names, not one string")
    variable = Variable(
        name=WORLD_ID,
        kind="categorical",
        unit="label",
        description="declared world preset of the run (an observed experiment label, not a hidden value)",
        levels=tuple(world_presets),
    )
    problems = variable.problems()
    if problems:
        raise ValueError("; ".join(problems))
    return variable


def world_id_layout(
    spec: SkillSpec,
    world_presets: Sequence[str],
    *,
    hidden_value_names: Iterable[str],
    include_physics: bool,
) -> FeatureLayout:
    """Layout of ``spec`` with the world-id feature as its extra block (one categorical column).

    Built through ``build_layout(spec, include_physics=..., extra=(world_id_variable(world_presets),),
    hidden_value_names=...)``, so the extra block is refused if it carries the name of a hidden
    simulator value.

    Args:
        world_presets:      the DECLARED preset names (keys of config.yaml ``world.presets``),
                            which become the levels of ``world_id`` in this order.
        hidden_value_names: names of the hidden simulator values (keys of config.yaml
                            ``world.hidden_effects``); required and non-empty. A single string
                            is refused (it would be read as its characters).
        include_physics:    True for the residual layout with a physics block, False for the
                            direct (no physics feature) layout.

    Raises TypeError / ValueError naming the offending argument.
    """
    if isinstance(hidden_value_names, (str, bytes)):
        raise TypeError("hidden_value_names must be a collection of names, not one string")
    hidden = tuple(hidden_value_names)
    if not hidden or not all(isinstance(name, str) and name for name in hidden):
        raise ValueError(
            "hidden_value_names must list the hidden simulator values by name (keys of config.yaml "
            f"world.hidden_effects); got {hidden!r}"
        )
    if not isinstance(include_physics, bool):
        raise TypeError(f"include_physics must be True or False; got {include_physics!r}")
    return build_layout(
        spec,
        include_physics=include_physics,
        extra=(world_id_variable(world_presets),),
        hidden_value_names=hidden,
    )


def world_id_column(rows_or_presets: Sequence[AttemptRow] | Sequence[str], variable: Variable) -> np.ndarray:
    """(n, 1) float64 level index of each row's ``world_preset`` (or of each preset name).

    Aligned with the given order, for the ``extra`` block of a layout built with
    ``extra=(variable,)``. A preset that is not a level of ``variable`` raises ValueError.
    """
    if variable.kind != "categorical":
        raise ValueError(f"variable {variable.name!r} must be categorical; got {variable.kind!r}")
    if isinstance(rows_or_presets, str):
        raise TypeError("rows_or_presets must be a sequence, not one string")
    indices = []
    for position, item in enumerate(rows_or_presets):
        preset = item.world_preset if isinstance(item, AttemptRow) else item
        if not isinstance(preset, str) or preset not in variable.levels:
            raise ValueError(f"item {position}: world preset {preset!r} is not one of {variable.levels}")
        indices.append(float(variable.levels.index(preset)))
    return np.array(indices, dtype=np.float64).reshape(-1, 1)


def recovery_attempts(
    errors_by_new_attempts: Mapping[int, float], pre_shift_error: float, tolerance_fraction: float
) -> dict[str, Any]:
    """How many new attempts after a world change bring the error back to the pre-shift level.

    Args:
        errors_by_new_attempts: {new attempts after the shift (int >= 0): measured error (finite)}.
        pre_shift_error:        measured error before the shift (finite, >= 0).
        tolerance_fraction:     config.yaml ``evaluation.recovery_tolerance_fraction`` (>= 0).

    Returns a dict (comparisons are exact, ``error <= threshold``)::

        threshold                 pre_shift_error * (1 + tolerance_fraction)
        first                     smallest measured count whose error is at or below the
                                  threshold; None when no count reaches it
        sustained                 smallest measured count from which the error of EVERY larger
                                  measured count stays at or below the threshold (so a single
                                  dip that is lost again does not count); None when the largest
                                  measured count is above the threshold
        pre_shift_error_is_zero   True when pre_shift_error == 0: the threshold is then 0 and
                                  any non-zero error counts as not recovered, so a None above
                                  says nothing about adaptation and the caller must report it
                                  separately (e.g. as undefined) instead of as "not recovered"
    """
    for label, value in (("pre_shift_error", pre_shift_error), ("tolerance_fraction", tolerance_fraction)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise ValueError(f"{label} must be a number; got {value!r}")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} must be finite and >= 0; got {value}")
    if not errors_by_new_attempts:
        raise ValueError("errors_by_new_attempts is empty")
    threshold = float(pre_shift_error) * (1.0 + float(tolerance_fraction))
    within: dict[int, bool] = {}
    for count, error in errors_by_new_attempts.items():
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError(f"attempt count {count!r} must be an int >= 0")
        if isinstance(error, bool) or not isinstance(error, (int, float, np.floating)) or not math.isfinite(error):
            raise ValueError(f"error at {count} attempts must be a finite number; got {error!r}")
        within[int(count)] = float(error) <= threshold
    counts = sorted(within)
    reached = [count for count in counts if within[count]]
    sustained = None
    for count in reversed(counts):
        if not within[count]:
            break
        sustained = count
    return {
        "first": reached[0] if reached else None,
        "sustained": sustained,
        "threshold": threshold,
        "pre_shift_error_is_zero": float(pre_shift_error) == 0.0,
    }
