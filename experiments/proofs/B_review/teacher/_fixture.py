"""Shared FIXTURE for the teacher review reproductions (never production, never evidence).

FixtureProvider subclasses the real TabPFNProvider and replaces ONLY the network-touching parts:
usage(), quote() and the estimator built by _build_regressor(). The real predict_main (phases,
request checks, parse_main_output, metadata) and the real TabPFNTeacher gate, cache and call log
run unchanged. FakeRegressor counts every predict() as one "billable" call and moves a fake
account counter by 10,000 tokens, like the documented minimum charge.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.features import FeatureLayout  # noqa: E402
from learned_judgment.teacher import (  # noqa: E402
    CallLog,
    RequestCache,
    TabPFNProvider,
    TabPFNTeacher,
    TeacherSettings,
    _tabpfn_config,
)

QUOTE = 10_000


class Account:
    def __init__(self) -> None:
        self.used = 40_000
        self.billed_predicts = 0


class FakeRegressor:
    """Stands in for tabpfn_client.TabPFNRegressor. predict() is the billable step."""

    def __init__(self, account: Account, response_hook=None) -> None:
        self.account = account
        self.response_hook = response_hook
        self.model_id_ = "00000000-0000-0000-0000-000000000000"
        self._last_meta = {}

    def fit(self, x, y):
        self._y = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, x, output_type, quantiles):
        self.account.used += QUOTE
        self.account.billed_predicts += 1
        n = x.shape[0]
        center = np.full(n, float(self._y.mean())) + 0.01 * x[:, 0]
        offsets = np.array([(level - 0.5) for level in quantiles])
        raw = {
            "mean": center.copy(),
            "median": center.copy(),
            "mode": center.copy(),
            "quantiles": center[None, :] + offsets[:, None],  # server layout (Q, n)
        }
        self._last_meta = {"package_version": "fixture", "test_set_num_rows": n}
        if self.response_hook is not None:
            raw = self.response_hook(raw)
        return raw

    def get_timings(self):
        return {"fit": {"elapsed_s": 0.0}, "predict": {"elapsed_s": 0.0}}


class FixtureProvider(TabPFNProvider):
    def __init__(self, settings, account: Account, response_hook=None) -> None:
        super().__init__(settings)
        self.account = account
        self.response_hook = response_hook
        self._connected = True  # no prior_labs.connect(): nothing here touches the network

    def usage(self):
        return {"daily_token_limit": 35_000_000, "daily_tokens_used": self.account.used,
                "monthly_token_limit": 35_000_000, "monthly_tokens_used": self.account.used,
                "usage_limit": 35_000_000, "current_usage": self.account.used}

    def quote(self, n_train, n_query, n_features):
        return {"estimated_cost": QUOTE, "pricing_version": "quota_v3",
                "inputs": {"train_rows": n_train, "test_rows": n_query, "raw_columns": n_features,
                           "model_version": "v3.5", "n_estimators": self.settings.n_estimators}}

    def _build_regressor(self, categorical_indices):
        return FakeRegressor(self.account, self.response_hook), _tabpfn_config(self.settings, categorical_indices)


LAYOUT = FeatureLayout(
    skill="fixture_review",
    names=("sit_x", "sit_side", "act_t", "physics_r0", "physics_r1", "physics_r2"),
    categorical=((1, ("left", "right")),),
    n_situation=2,
    n_action=1,
    n_physics=3,
    n_extra=0,
    outcome_names=("r0", "r1", "r2"),
    outcome_kinds=("regression", "regression", "regression"),
)


def arrays(seed: int = 5, n: int = 8, m: int = 4):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 6))
    features[:, 1] = rng.integers(0, 2, size=n)
    residuals = rng.normal(size=(n, 3))
    query = rng.normal(size=(m, 6))
    query[:, 1] = rng.integers(0, 2, size=m)
    return features, residuals, query


def make_clock():
    start = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
    state = {"i": 0}

    def clock():
        state["i"] += 1
        return start + timedelta(seconds=state["i"])

    return clock


def make_teacher(tmp: Path, *, mode: str = "stacked", account: Account | None = None, response_hook=None,
                 budget: int | None = 35_000_000):
    import dataclasses

    base = TeacherSettings.from_config(load_config())
    settings = dataclasses.replace(base, cache_dir=tmp / "cache", call_log=tmp / "calls.jsonl",
                                   multi_outcome_mode=mode, project_token_budget=budget)
    account = account or Account()
    provider = FixtureProvider(settings, account, response_hook)
    clock = make_clock()
    log = CallLog(settings.call_log, clock=clock)
    teacher = TabPFNTeacher(LAYOUT, settings, provider=provider, cache=RequestCache(settings.cache_dir),
                            call_log=log, clock=clock)
    return teacher, provider, account, log, settings
