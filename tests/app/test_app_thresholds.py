"""app.backend.app_thresholds: the threat cutoff the app runs the dodge student with (config app.infer.threat_threshold)."""
import copy
import json

import pytest

from app.backend import app_thresholds
from learned_judgment.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


def _student(tmp_path, thresholds):
    (tmp_path / "student.json").write_text(json.dumps({"thresholds": thresholds}), encoding="utf-8")
    return tmp_path


def test_replaces_only_the_threat_cutoff(tmp_path, config):
    cfg = copy.deepcopy(config)
    cfg["app"]["infer"]["threat_threshold"] = 0.95
    folder = _student(tmp_path, {"threat": 0.98, "action": 0.4, "margin": 0.1})
    assert app_thresholds(folder, cfg) == {"threat": 0.95, "action": 0.4, "margin": 0.1}


def test_null_keeps_the_frozen_thresholds(tmp_path, config):
    cfg = copy.deepcopy(config)
    cfg["app"]["infer"]["threat_threshold"] = None
    assert app_thresholds(_student(tmp_path, {"threat": 0.98, "action": 0.4, "margin": 0.1}), cfg) is None


def test_a_student_without_frozen_thresholds_is_left_alone(tmp_path, config):
    cfg = copy.deepcopy(config)
    cfg["app"]["infer"]["threat_threshold"] = 0.95
    assert app_thresholds(_student(tmp_path, None), cfg) is None


def test_out_of_range_cutoff_is_refused(tmp_path, config):
    cfg = copy.deepcopy(config)
    cfg["app"]["infer"]["threat_threshold"] = 1.5
    with pytest.raises(ValueError, match="threat_threshold"):
        app_thresholds(_student(tmp_path, {"threat": 0.98, "action": 0.4, "margin": 0.1}), cfg)
