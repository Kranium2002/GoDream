"""Tests of learned_judgment.compare_runs: pure-function fixtures (hand-built result.json metrics)."""

from __future__ import annotations

import pytest

from learned_judgment.compare_runs import side_by_side


def run(value, n):
    return {"metrics": [{"name": "ascent_success_rate", "values": {"snn": {"value": value, "n": n}, "no_op": {"value": 0.61, "n": n}}},
                        {"name": "only_here", "values": {"snn": {"value": 1.0, "n": n}}}]}


def test_side_by_side_pairs_metrics_by_name_and_reports_the_difference():
    rows = side_by_side(run(0.76, 41), {"metrics": run(0.80, 41)["metrics"][:1]}, "snn")
    assert rows == [{"name": "ascent_success_rate", "a": 0.76, "b": 0.80, "difference": pytest.approx(0.04), "n_a": 41, "n_b": 41}]
