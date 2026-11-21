"""R02: does measure.student_record detect "held-out" dreams that contain the training rows?

student_record refuses dreams_heldout only when the sha256 over ALL its rows equals the training dream's
sha256 (measure.py lines 697-699). This trains a FIXTURE per-skill student on a small physics-only fixture
dream (fx_beta, 150 rows, 300 steps) and passes four "held-out" sets:
    fresh        500 rows from an independent stream (the honest reference)
    minus_one    the training dream without its last row
    train_only   exactly the 135 rows the optimiser trained on (validation rows dropped)
    superset     the training dream followed by the 500 fresh rows
    exact        the training dream itself (the only case the check is built for)

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\snn_shared_measure\\r02_heldout_overlap.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

from _review_common import POINTS, concat, fixture_config, fixture_dreams, fixture_spec, subset  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY  # noqa: E402
from learned_judgment.engine.dream import dream  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.measure import student_record  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

BASE_SEED = 21
config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(fixture_config(config)), max_steps=300)
spec = fixture_spec("fx_beta")
train = fixture_dreams(spec, config, 150)
student = SNNStudent.train_in_dream(spec, train, settings=settings, points_per_dim=POINTS["fx_beta"], base_seed=BASE_SEED)
n = train.n_rows
n_val = max(1, int(round(settings.validation_fraction * n)))
order = stream_rng(BASE_SEED, "fx_beta/snn/validation_split").permutation(n)
train_rows = np.sort(order[n_val:])
fresh = dream(
    spec, None, build_layout(spec), n_dreams=500, rng=stream_rng(99, "fx_beta/review/heldout_situations"),
    points_per_dim=POINTS["fx_beta"], quantile_levels=require(config, "engine.quantile_levels"),
    spread_levels=tuple(require(config, "engine.spread_levels")), method=PHYSICS_ONLY,
)
situations = spec.sample_situations(stream_rng(BASE_SEED, "fx_beta/review/measure_situations"), 50)
print(f"student trained on {n} dream rows ({train_rows.size} optimiser rows, {n_val} validation rows), steps {student.meta['training']['steps_run']}")
print(f"student's own validation_in_dream agreement: {student.meta['validation_in_dream']['agreement_with_dream_choice']:.3f}")

cases = {
    "fresh": fresh,
    "minus_one": subset(train, np.arange(n - 1)),
    "train_only": subset(train, train_rows),
    "superset": concat(train, fresh),
    "exact": train,
}
print(f"\n{'held-out set':12s} {'rows':>5s} {'train rows in it':>17s}  outcome")
for label, heldout in cases.items():
    overlap = {"fresh": 0, "minus_one": int(np.sum(train_rows < n - 1)), "train_only": train_rows.size, "superset": train_rows.size, "exact": train_rows.size}[label]
    try:
        record = student_record(student, spec, heldout, situations, config, seed=BASE_SEED)
        agreement = record["agreement_with_heldout_dream"]["student"]
        print(
            f"{label:12s} {heldout.n_rows:5d} {overlap:17d}  ACCEPTED: agreement_with_dream_choice "
            f"{agreement['agreement_with_dream_choice']:.3f}, fraction_zero regret {agreement['dream_regret']['fraction_zero']:.3f}, "
            f"mean regret/spread {agreement['dream_regret']['over_spread']['mean']:.4f}, "
            f"paired student-physics {record['dream_regret_difference_student_minus_physics']['estimate']:.4f}"
        )
    except ValueError as error:
        print(f"{label:12s} {heldout.n_rows:5d} {overlap:17d}  REFUSED: {str(error)[:90]}")
