"""Check 6: TabPFNTeacher.fit / predict input validation (no network).

FIXTURE arrays and a fixture FeatureLayout only. The provider object is the real TabPFNProvider,
never connected; fit() and the validation paths make no provider call. The cache and call log
live in a temporary directory.
Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c6_fit_validation.py
"""

import dataclasses
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

print("FIXTURE CHECK - utility test only, not evidence about the robot, the simulator, TabPFN or any model")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.features import FeatureLayout  # noqa: E402
from learned_judgment.engine.types import ResidualModel  # noqa: E402
from learned_judgment.teacher import CallLog, RequestCache, TabPFNProvider, TabPFNTeacher, TeacherSettings  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAILED'}] {label}")
    if not ok:
        FAILURES.append(label)


def raises(label: str, exc_type, call) -> None:
    try:
        call()
        check(f"{label} raises {exc_type.__name__}", False)
    except exc_type as exc:
        check(f"{label} raises {exc_type.__name__}: {exc}", True)


LAYOUT = FeatureLayout(
    skill="fixture_check",
    names=("sit_x", "sit_side", "act_t", "physics_r0", "physics_r1"),
    categorical=((1, ("left", "right")),),
    n_situation=2,
    n_action=1,
    n_physics=2,
    n_extra=0,
    outcome_names=("r0", "r1"),
    outcome_kinds=("regression", "regression"),
)
rng = np.random.default_rng(6)
FEATURES = rng.normal(size=(6, 5))
FEATURES[:, 1] = [0, 1, 0, 1, 1, 0]
RESIDUALS = rng.normal(size=(6, 2))

with tempfile.TemporaryDirectory() as tmp:
    settings = dataclasses.replace(
        TeacherSettings.from_config(load_config()), cache_dir=Path(tmp) / "cache", call_log=Path(tmp) / "calls.jsonl"
    )
    clock = lambda: datetime.now(timezone.utc)  # noqa: E731
    provider = TabPFNProvider(settings)
    teacher = TabPFNTeacher(
        LAYOUT,
        settings,
        provider=provider,
        cache=RequestCache(settings.cache_dir),
        call_log=CallLog(settings.call_log, clock=clock),
        clock=clock,
    )
    check("TabPFNTeacher satisfies the ResidualModel protocol", isinstance(teacher, ResidualModel))

    print("== fit() validation ==")
    raises("features with 4 columns (layout has 5)", ValueError, lambda: teacher.fit(FEATURES[:, :4], RESIDUALS))
    raises("residuals with 3 outcomes (layout has 2)", ValueError, lambda: teacher.fit(FEATURES, rng.normal(size=(6, 3))))
    raises("residual rows != feature rows", ValueError, lambda: teacher.fit(FEATURES, RESIDUALS[:5]))
    raises("1-D residuals", ValueError, lambda: teacher.fit(FEATURES, RESIDUALS[:, 0]))
    raises("1 context row", ValueError, lambda: teacher.fit(FEATURES[:1], RESIDUALS[:1]))
    raises("0 context rows", ValueError, lambda: teacher.fit(FEATURES[:0], RESIDUALS[:0]))
    nan_features = FEATURES.copy()
    nan_features[2, 3] = np.nan
    raises("NaN in features", ValueError, lambda: teacher.fit(nan_features, RESIDUALS))
    inf_residuals = RESIDUALS.copy()
    inf_residuals[4, 1] = np.inf
    raises("inf in residuals", ValueError, lambda: teacher.fit(FEATURES, inf_residuals))
    bad_level = FEATURES.copy()
    bad_level[0, 1] = 2.0
    raises("categorical level index 2 for a 2-level column", ValueError, lambda: teacher.fit(bad_level, RESIDUALS))
    half_level = FEATURES.copy()
    half_level[0, 1] = 0.5
    raises("categorical level index 0.5", ValueError, lambda: teacher.fit(half_level, RESIDUALS))
    raises("predict before any successful fit", RuntimeError, lambda: teacher.predict_residual_distribution(FEATURES))

    print("\n== a valid fit stores the context without any provider call ==")
    returned = teacher.fit(FEATURES, RESIDUALS)
    check("fit returns the teacher", returned is teacher)
    check("2 rows is accepted (minimum context)", teacher.fit(FEATURES[:2], RESIDUALS[:2]) is teacher)
    check("no prediction call was started", provider.prediction_calls_started == 0)
    check("no call-log file was written", not settings.call_log.exists())

    print("\n== predict() query validation happens before any provider call ==")
    teacher.fit(FEATURES, RESIDUALS)
    raises("query with 6 columns", ValueError, lambda: teacher.predict_residual_distribution(rng.normal(size=(3, 6))))
    raises("query with 0 rows", ValueError, lambda: teacher.predict_residual_distribution(FEATURES[:0]))
    raises("NaN in query", ValueError, lambda: teacher.predict_residual_distribution(nan_features))

    print("\n== all-constant residuals need no provider call ==")
    constant = np.column_stack([np.full(6, 0.25), np.full(6, -1.5)])
    distribution = teacher.fit(FEATURES, constant).predict_residual_distribution(FEATURES[:3])
    check("median is the constant per outcome", np.array_equal(distribution.median, np.tile([0.25, -1.5], (3, 1))))
    check("spread is exactly 0", np.array_equal(distribution.spread, np.zeros((3, 2))))
    check("no request hash", teacher.last_request_hash is None and teacher.last_request_hashes == ())
    check("still no prediction call and no call log", provider.prediction_calls_started == 0 and not settings.call_log.exists())

    print("\n== an unconnected provider is refused before any network call ==")
    teacher.fit(FEATURES, RESIDUALS)
    raises("uncached request with an unconnected provider", RuntimeError, lambda: teacher.predict_residual_distribution(FEATURES[:2]))

    print("\n== construction rules ==")
    raises(
        "provider settings differing in a keyed field",
        ValueError,
        lambda: TabPFNTeacher(
            LAYOUT, settings, provider=TabPFNProvider(dataclasses.replace(settings, n_estimators=4)),
            cache=RequestCache(settings.cache_dir), call_log=CallLog(settings.call_log, clock=clock), clock=clock,
        ),
    )
    reserved = dataclasses.replace(LAYOUT, names=("sit_x", "sit_side", "act_t", "physics_r0", "stacked_outcome_index"))
    raises(
        "a feature named like the reserved stacked outcome column",
        ValueError,
        lambda: TabPFNTeacher(
            reserved, settings, provider=provider, cache=RequestCache(settings.cache_dir),
            call_log=CallLog(settings.call_log, clock=clock), clock=clock,
        ),
    )

print(f"\nRESULT: {len(FAILURES)} failed check(s)" + (f": {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
