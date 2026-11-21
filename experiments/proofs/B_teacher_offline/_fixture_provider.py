"""FIXTURE provider for the teacher-lane checks (check scripts only; never production, never evidence).

FixtureProvider subclasses the real learned_judgment.teacher.TabPFNProvider and replaces ONLY the
network-touching parts: usage(), quote() and the estimator factory _build_regressor(). Everything
else is the real code: the real predict_raw phases and request checks, parse_main_output, the
teacher's lock, gate, unvalidated store, cache and call log. FakeRegressor counts every predict()
as one "billable" call and moves a fake account counter by the documented minimum charge; its
answers are arithmetic on the sent rows, not a model.

Nothing here touches the network, reads .env or depends on config.yaml to stop a paid request:
there is no paid request in any fixture check.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.features import FeatureLayout  # noqa: E402
from learned_judgment.teacher import (  # noqa: E402
    TabPFNProvider,
    TabPFNTeacher,
    TeacherSettings,
    _tabpfn_config,
)

print("FIXTURE provider in use: usage, quote and the estimator are stand-ins (no network); outputs are not evidence")

QUOTE = 10_000  # the documented minimum charge (E7 p3); a fixture number here, not a measurement
FIXTURE_BUDGET = 35_000_000  # explicit in every fixture check; never read from config.yaml


class Account:
    """Fake account: tokens used, billed predicts, and every query matrix a predict received."""

    def __init__(self) -> None:
        self.used = 40_000
        self.billed_predicts = 0
        self.sent_queries: list[np.ndarray] = []
        self.sent_training: list[tuple[np.ndarray, np.ndarray]] = []


class FakeRegressor:
    """Stands in for tabpfn_client.TabPFNRegressor. predict() is the "billable" step.

    Answer for row i of x: center_i = mean(y_train) + 0.01 * x[i, 0]; mean = median = mode =
    center; quantile at level l = center + (l - 0.5); quantiles in the documented (Q, n) layout.
    """

    def __init__(self, account: Account, response_hook: Callable[[dict], Any] | None) -> None:
        self.account = account
        self.response_hook = response_hook
        self.model_id_ = "00000000-0000-0000-0000-000000000000"
        self._last_meta: dict[str, Any] = {}

    def fit(self, x, y):
        self._x = np.array(x, copy=True)
        self._y = np.asarray(y, dtype=np.float64).copy()
        return self

    def predict(self, x, output_type, quantiles):
        self.account.used += QUOTE
        self.account.billed_predicts += 1
        self.account.sent_queries.append(np.array(x, copy=True))
        self.account.sent_training.append((self._x, self._y))
        n = x.shape[0]
        center = np.full(n, float(self._y.mean())) + 0.01 * x[:, 0]
        offsets = np.array([(level - 0.5) for level in quantiles])
        raw = {
            "mean": center.copy(),
            "median": center.copy(),
            "mode": center.copy(),
            "quantiles": center[None, :] + offsets[:, None],  # documented server layout (Q, n)
        }
        self._last_meta = {"package_version": "fixture", "test_set_num_rows": n}
        if self.response_hook is not None:
            raw = self.response_hook(raw)
        return raw

    def get_timings(self):
        return {"fit": {"elapsed_s": 0.0}, "predict": {"elapsed_s": 0.0}}


class FixtureProvider(TabPFNProvider):
    """The real TabPFNProvider with usage(), quote() and _build_regressor() replaced by fixtures."""

    def __init__(self, settings: TeacherSettings, account: Account, response_hook=None) -> None:
        super().__init__(settings)
        self.account = account
        self.response_hook = response_hook
        self.usage_reads = 0
        self.quote_reads = 0
        self._connected = True  # no prior_labs.connect(): nothing here touches the network

    def usage(self):
        self.usage_reads += 1
        used = self.account.used
        return {"daily_token_limit": 35_000_000, "daily_tokens_used": used,
                "monthly_token_limit": 35_000_000, "monthly_tokens_used": used,
                "usage_limit": 35_000_000, "current_usage": used}

    def quote(self, n_train, n_query, n_features):
        self.quote_reads += 1
        return {"estimated_cost": QUOTE, "pricing_version": "fixture",
                "inputs": {"train_rows": n_train, "test_rows": n_query, "raw_columns": n_features,
                           "model_version": "v3.5", "n_estimators": self.settings.n_estimators}}

    def _build_regressor(self, categorical_indices):
        return FakeRegressor(self.account, self.response_hook), _tabpfn_config(self.settings, categorical_indices)


LAYOUT = FeatureLayout(
    skill="fixture_teacher_check",
    names=("sit_x", "sit_side", "act_t", "physics_r0", "physics_r1", "physics_r2"),
    categorical=((1, ("left", "right")),),
    n_situation=2,
    n_action=1,
    n_physics=3,
    n_extra=0,
    outcome_names=("r0", "r1", "r2"),
    outcome_kinds=("regression", "regression", "regression"),
)


def arrays(seed: int = 5, n: int = 8, m: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FIXTURE features (n, 6), residuals (n, 3) and query rows (m, 6)."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 6))
    features[:, 1] = rng.integers(0, 2, size=n)
    residuals = rng.normal(size=(n, 3))
    query = rng.normal(size=(m, 6))
    query[:, 1] = rng.integers(0, 2, size=m)
    return features, residuals, query


def make_clock(start: datetime | None = None) -> Callable[[], datetime]:
    """A UTC clock that advances one second per call (deterministic log stamps)."""
    origin = start or datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
    state = {"i": 0}

    def clock() -> datetime:
        state["i"] += 1
        return origin + timedelta(seconds=state["i"])

    return clock


def settings_for(tmp: Path, **overrides: Any) -> TeacherSettings:
    """Config settings with the cache and call log in ``tmp`` and an explicit fixture budget."""
    base = TeacherSettings.from_config(load_config())
    values = {"cache_dir": tmp / "cache", "call_log": tmp / "calls.jsonl", "project_token_budget": FIXTURE_BUDGET}
    values.update(overrides)
    return dataclasses.replace(base, **values)


def make_teacher(tmp: Path, *, account: Account | None = None, response_hook=None, clock=None,
                 provider_class: type = FixtureProvider, **overrides: Any):
    """(teacher, provider, account, settings) with a FixtureProvider; overrides go to TeacherSettings."""
    settings = settings_for(tmp, **overrides)
    account = account or Account()
    provider = provider_class(settings, account, response_hook)
    teacher = TabPFNTeacher.from_settings(LAYOUT, settings, provider=provider, clock=clock or make_clock())
    return teacher, provider, account, settings


def mask_tmp(text: str) -> str:
    """``text`` with the system temporary directory shown as <tmp> (no local paths in proof output)."""
    temp = tempfile.gettempdir()
    return text.replace(temp, "<tmp>").replace(str(Path(temp).resolve()), "<tmp>")


class Checks:
    """[ok] / [FAILED] lines and the exit code of a check script."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def __call__(self, label: str, ok: bool) -> bool:
        label = mask_tmp(label).encode("ascii", "backslashreplace").decode("ascii")  # any console encoding
        print(f"  [{'ok' if ok else 'FAILED'}] {label}")
        if not ok:
            self.failures.append(label)
        return ok

    def raises(self, label: str, exc_type: type, call: Callable[[], Any]) -> BaseException | None:
        try:
            call()
        except exc_type as exc:
            self(f"{label} raises {exc_type.__name__}: {str(exc)[:160]}", True)
            return exc
        except Exception as exc:  # the wrong exception type is a failure, reported with its text
            self(f"{label} raises {exc_type.__name__} (got {type(exc).__name__}: {str(exc)[:160]})", False)
            return None
        self(f"{label} raises {exc_type.__name__} (nothing raised)", False)
        return None

    def finish(self) -> None:
        print(f"\nRESULT: {len(self.failures)} failed check(s)" + (f": {self.failures}" if self.failures else ""))
        sys.exit(1 if self.failures else 0)


def events(teacher: TabPFNTeacher) -> list[str]:
    return [record["event"] for record in teacher.call_log.read()]


def dispatch_count(teacher: TabPFNTeacher) -> int:
    return sum(1 for record in teacher.call_log.read() if record.get("billable_dispatch") is True)
