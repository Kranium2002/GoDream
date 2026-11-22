"""Check 13 (review finding E7): ratio metrics refuse negative or sign-changing losses.

Reviewer reproductions r06 / r07(c): a student better everywhere on a loss of -utility got
student_teacher_ratio 2.0, and a denominator near 0 that changes sign gave a BCa "interval"
(-16.8, 2113.8) with no warning. Both ratio functions must now raise ValueError; non-negative
inputs keep working and a zero denominator stays "undefined". Fixture arrays only. On the old
code these calls returned numbers, so this check fails there.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import numpy as np  # noqa: E402

from fixture_spec import Checks  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.eval.bootstrap import paired_difference_ci, paired_ratio_ci  # noqa: E402
from learned_judgment.eval.metrics import student_teacher_ratio  # noqa: E402

checks = Checks()
expect = checks.expect
config = load_config()  # read only
settings = dict(n_resamples=int(require(config, "evaluation.bootstrap_resamples")),
                confidence_level=float(require(config, "evaluation.bootstrap_confidence_level")),
                method=require(config, "evaluation.bootstrap_method"), seed=1)

teacher = np.array([-0.2, -0.1, -0.3, -0.2])  # a loss defined as -utility
student = np.array([-0.4, -0.2, -0.6, -0.4])  # lower is better: the student is better everywhere
T = np.tile(teacher[:, None], (1, 3)) + np.random.default_rng(1).normal(0, 0.01, (4, 3))
S = np.tile(student[:, None], (1, 3)) + np.random.default_rng(2).normal(0, 0.01, (4, 3))
crossing = np.random.default_rng(4).normal(0.02, 0.2, (25, 3))  # denominator mean near 0, changes sign
positive = np.abs(np.random.default_rng(5).normal(0.3, 0.1, (25, 3)))

for label, call in (
    ("student_teacher_ratio with negative losses", lambda: student_teacher_ratio(student, teacher)),
    ("student_teacher_ratio with a negative student loss only", lambda: student_teacher_ratio([-0.1, 0.2], [0.3, 0.3])),
    ("paired_ratio_ci with negative losses", lambda: paired_ratio_ci(T, S, **settings)),
    ("paired_ratio_ci with a sign-changing denominator", lambda: paired_ratio_ci(crossing, positive, **settings)),
):
    try:
        result = call()
        expect(False, f"refuses {label} (got {result})")
    except ValueError as error:
        expect(True, f"refuses {label}: {error}")

ok = student_teacher_ratio([1.0, 3.0], [1.0, 1.0])
expect(ok["ratio"] == 2.0, f"non-negative errors still give the ratio: {ok['ratio']}")
zero = student_teacher_ratio([1.0, 2.0], [0.0, 0.0])
expect(zero["ratio"] == "undefined", "a zero teacher mean is still 'undefined'")
ratio = paired_ratio_ci(np.abs(crossing), positive, **settings)
expect(isinstance(ratio["estimate"], float) and ratio["low"] <= ratio["estimate"] <= ratio["high"],
       f"non-negative losses still give a ratio CI: {ratio['estimate']:.4f} ({ratio['low']:.4f}, {ratio['high']:.4f})")
difference = paired_difference_ci(T, S, **settings)
expect(isinstance(difference["estimate"], float), "the paired DIFFERENCE still accepts negative losses (no ratio involved)")

checks.finish()
