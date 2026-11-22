"""R01: follow one attempt measured-outcome mapping -> build_real_row -> JSONL -> read_rows ->
rows_to_arrays -> fit_residual_model -> decide_batch -> OutcomeDistribution, and check that
residual = measured - physics survives in spec outcome order. Then: rows whose stored
physics_guess came from a different physics formula are accepted silently.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import LEVELS, SPREAD, TS, ReviewSpec  # noqa: E402

from learned_judgment.engine.decide import decide_batch  # noqa: E402
from learned_judgment.engine.features import build_features, build_layout  # noqa: E402
from learned_judgment.engine.learn import fit_residual_model  # noqa: E402
from learned_judgment.engine.rows import append_rows, build_real_row, read_rows, rows_to_arrays  # noqa: E402
from learned_judgment.eval.baselines import ConstantBiasResidual  # noqa: E402

spec = ReviewSpec(gain=1.0)
layout = build_layout(spec)
sit = np.array([[1.0, 1.0], [0.5, 0.0], [1.5, 1.0]])
act = np.array([[0.5, 0.1], [-0.5, 0.2], [1.0, 0.0]])
# mapping keys deliberately in reverse order of spec.outcomes
measured_maps = [{"rv_hit": 1.0, "rv_sep": 0.05}, {"rv_hit": 0.0, "rv_sep": 0.9}, {"rv_hit": 0.0, "rv_sep": 0.7}]

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "a.jsonl"
    rows = [build_real_row(spec, run_id="rv", seed=1, world_preset="default", situation=s, action=a,
                           measured_outcome=m, timestamp_utc=TS) for s, a, m in zip(sit, act, measured_maps)]
    append_rows(path, rows)
    back = read_rows(path, spec=spec)
    arrays = rows_to_arrays(spec, back)
    physics = spec.physics_guess(sit, act)
    measured = np.array([[m["rv_sep"], m["rv_hit"]] for m in measured_maps])
    print("A. residual columns in spec order after JSONL round trip:",
          np.array_equal(arrays.residuals, measured - physics))
    print("   residual sign (measured - physics), row 0:", arrays.residuals[0].tolist(),
          "expected", (measured[0] - physics[0]).tolist())

    model = fit_residual_model(spec, ConstantBiasResidual(layout, LEVELS, SPREAD, seed=0), back, layout)
    d = decide_batch(spec, sit[:1], model=model, layout=layout, points_per_dim=[3, 3],
                     quantile_levels=LEVELS, spread_levels=SPREAD, method="constant_bias")[0]
    expected_mean = np.clip(d.predicted.physics + arrays.residuals.mean(axis=0), spec.outcome_low, spec.outcome_high)
    print("B. OutcomeDistribution.mean == clip(physics + mean residual):", np.array_equal(d.predicted.mean, expected_mean))

    # ---- stored physics drift ------------------------------------------------------------
    spec_b = ReviewSpec(gain=2.0)  # same name and schema, different physics constant
    back_b = read_rows(path, spec=spec_b)  # accepted
    arrays_b = rows_to_arrays(spec_b, back_b)  # accepted
    now_physics = spec_b.physics_guess(arrays_b.situations, arrays_b.actions)
    print("C. read_rows(spec=gain 2.0) accepted rows written with gain 1.0:", len(back_b), "rows")
    print("   stored physics == current spec.physics_guess ?", np.array_equal(arrays_b.physics, now_physics))
    print("   stored physics  rv_sep:", arrays_b.physics[:, 0].round(4).tolist())
    print("   current physics rv_sep:", now_physics[:, 0].round(4).tolist())
    model_b = fit_residual_model(spec_b, ConstantBiasResidual(build_layout(spec_b), LEVELS, SPREAD, seed=0), back_b,
                                 build_layout(spec_b))
    d_b = decide_batch(spec_b, sit[:1], model=model_b, layout=build_layout(spec_b), points_per_dim=[3, 3],
                       quantile_levels=LEVELS, spread_levels=SPREAD, method="constant_bias")[0]
    # The residual learnt is relative to the OLD physics but is added to the NEW physics.
    print("   residual targets used by fit (relative to old physics):", arrays_b.residuals[:, 0].round(4).tolist())
    print("   residual vs current physics would be               :", (arrays_b.measured - now_physics)[:, 0].round(4).tolist())
