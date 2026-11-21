"""Fitting a residual model on actual attempts (engine lane, Person B).

Lane contract
-------------
:func:`fit_residual_model` turns real attempt rows into the model's feature matrix and
residual targets and calls ``model.fit`` once. Nothing is recomputed from the simulator:
situation, action and measured outcome come from the stored rows. The physics guess is
recomputed with the current spec and compared with the stored one
(:func:`~learned_judgment.engine.rows.rows_to_arrays`); the caller's required
``on_physics_mismatch`` says whether a difference is an error or is recomputed, so the
targets are always ``measured_outcome - physics_guess`` of the physics the model will be
queried with. Zero attempts is not a fit: it is the physics-only method, so an empty row set
is rejected.

    features   (n, F) in ``layout`` order: situation | action | physics (if the layout has
               a physics block) | extra (if it has an extra block)
    residuals  (n, K) in the skill's fixed outcome order
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from learned_judgment.engine.features import FeatureLayout, build_features
from learned_judgment.engine.rows import AttemptRow, rows_to_arrays
from learned_judgment.engine.types import ResidualModel
from learned_judgment.skills.base import SkillSpec

__all__ = ["fit_residual_model"]


def _check_layout(spec: SkillSpec, layout: FeatureLayout) -> None:
    problems = []
    if layout.skill != spec.name:
        problems.append(f"layout skill {layout.skill!r} != {spec.name!r}")
    if layout.n_situation != spec.n_situation or layout.n_action != spec.n_action:
        problems.append(
            f"layout blocks (S={layout.n_situation}, A={layout.n_action}) != spec "
            f"(S={spec.n_situation}, A={spec.n_action})"
        )
    if layout.outcome_names != spec.outcome_names:
        problems.append(f"layout outcomes {layout.outcome_names} != {spec.outcome_names}")
    if layout.n_physics not in (0, spec.n_outcome):
        problems.append(f"layout physics block has {layout.n_physics} columns; expected 0 or {spec.n_outcome}")
    if problems:
        raise ValueError("; ".join(problems))


def fit_residual_model(
    spec: SkillSpec,
    model: ResidualModel,
    rows: Sequence[AttemptRow],
    layout: FeatureLayout,
    *,
    on_physics_mismatch: str,
    extra: np.ndarray | None = None,
) -> ResidualModel:
    """Fit ``model`` on the real ``rows`` (n >= 1) and return the fitted model.

    Args:
        rows:                real attempt rows of ``spec`` (a dream row, or two rows with the
                             same ``(run_id, seed)``, raises ValueError).
        layout:              the feature layout the model will be queried with.
        on_physics_mismatch: "error" or "recompute", passed to ``rows_to_arrays``: what to do
                             with a row whose stored physics guess differs from the current spec
                             ("error" raises ``StalePhysicsError``).
        extra:               (n, E) observed context aligned with ``rows`` when the layout has an
                             extra block (e.g. ``adapt.world_id_column(rows, variable)``).
    """
    _check_layout(spec, layout)
    arrays = rows_to_arrays(spec, rows, on_physics_mismatch=on_physics_mismatch)
    if arrays.n_rows == 0:
        raise ValueError("no attempts to fit; zero attempts is the physics-only method")
    features = build_features(
        layout,
        arrays.situations,
        arrays.actions,
        physics=arrays.physics if layout.n_physics else None,
        extra=extra,
    )
    fitted = model.fit(features, arrays.residuals)
    if not isinstance(fitted, ResidualModel):
        raise TypeError(f"{type(model).__name__}.fit returned {type(fitted).__name__}, not a ResidualModel")
    return fitted
