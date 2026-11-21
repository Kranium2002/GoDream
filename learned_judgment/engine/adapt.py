"""Context selection and recovery after a world change (engine lane, Person B).

Lane contract
-------------
After the judge switches the world preset, the residual model is refitted on a context
chosen by one of the policies compared on dev data (03_METRICS.md, "Adaptation"):

    "all"     every real attempt, in file order
    "recent"  the last ``recent_window`` real attempts, in file order
    + world ID: either policy, with the declared preset name as an extra categorical
              feature (``world_id_variable`` / ``world_id_column``). The preset name is an
              observed label of the experiment, never a hidden simulator value.

Dream rows are never context: they are teacher decisions, not measurements.

:func:`recovery_attempts` reads off how many new attempts were needed to come back within
``tolerance_fraction`` of the pre-shift error (config.yaml
``evaluation.recovery_tolerance_fraction``); ``None`` means not reached.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from learned_judgment.engine.rows import AttemptRow
from learned_judgment.skills.base import Variable

__all__ = [
    "CONTEXT_POLICIES",
    "WORLD_ID",
    "recovery_attempts",
    "select_context",
    "world_id_column",
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
) -> int | None:
    """Smallest new-attempt count whose error <= ``pre_shift_error * (1 + tolerance_fraction)``.

    Args:
        errors_by_new_attempts: {new attempts after the shift (int >= 0): measured error (finite)}.
        pre_shift_error:        measured error before the shift (finite, >= 0).
        tolerance_fraction:     config.yaml ``evaluation.recovery_tolerance_fraction`` (>= 0).

    Returns the count, or ``None`` when no measured count reaches the threshold. The
    comparison is exact.
    """
    for label, value in (("pre_shift_error", pre_shift_error), ("tolerance_fraction", tolerance_fraction)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise ValueError(f"{label} must be a number; got {value!r}")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} must be finite and >= 0; got {value}")
    if not errors_by_new_attempts:
        raise ValueError("errors_by_new_attempts is empty")
    threshold = float(pre_shift_error) * (1.0 + float(tolerance_fraction))
    reached = []
    for count, error in errors_by_new_attempts.items():
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError(f"attempt count {count!r} must be an int >= 0")
        if isinstance(error, bool) or not isinstance(error, (int, float, np.floating)) or not math.isfinite(error):
            raise ValueError(f"error at {count} attempts must be a finite number; got {error!r}")
        if float(error) <= threshold:
            reached.append(int(count))
    return min(reached) if reached else None
