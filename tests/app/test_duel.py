"""The dodge duel: which punch a round shows, and the plain result word of each lane (app/duel.py)."""
import copy

import pytest

from app.duel import Duel, dodge_duel_scenario, result_word
from learned_judgment.env import load_config
from learned_judgment.skills import dodge_vision

AIMED_AT_DOG = {"incoming_center", "incoming_left", "incoming_right"}


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_config_names_only_kinds_aimed_at_the_dog_from_the_held_out_split(config):
    duel = config["app"]["duel"]["dodge"]
    assert set(duel["kinds"]) <= AIMED_AT_DOG and duel["kinds"]
    assert duel["split"] == "test"


def test_every_seed_draws_a_held_out_scenario_aimed_at_the_dog(config):
    duel = config["app"]["duel"]["dodge"]
    split, _ = dodge_vision._split_plan(config, "test", config["observed"]["dodge_vision"]["test_scenarios"])
    pool = [s for s in split if s["kind"] in duel["kinds"]]
    seen = set()
    for seed in range(1000):
        s = dodge_duel_scenario(config, duel, seed)
        assert s["kind"] in AIMED_AT_DOG
        assert s["situation"]["ball_vel_x_m_s"] < 0  # flies toward the dog (x is ahead of it)
        assert s == pool[seed % len(pool)]  # by the seed alone, nothing else
        seen.add(s["scenario_id"])
    assert len(seen) == len(pool)  # consecutive seeds walk through all of them
    assert dodge_duel_scenario(config, duel, 7) == dodge_duel_scenario(config, duel, 7)
    s = dodge_duel_scenario(config, duel, 7)
    s["situation"]["ball_rel_x_m"] = 99.0  # a round's changes never leak into the next round
    assert dodge_duel_scenario(config, duel, 7)["situation"]["ball_rel_x_m"] != 99.0


def _crossing(situation, aim_x):
    t = (aim_x - situation["ball_rel_x_m"]) / situation["ball_vel_x_m_s"]
    return (situation["ball_rel_y_m"] + situation["ball_vel_y_m_s"] * t,
            situation["ball_rel_z_m"] + situation["ball_vel_z_m_s"] * t)


@pytest.mark.parametrize("distance", [0.75, 1.075, 1.4])
def test_the_distance_control_keeps_the_punch_aimed_where_it_was(config, distance):
    aim_x = config["skills"]["dodge_vision"]["punch"]["aim_x_m"]
    duel = config["app"]["duel"]["dodge"]
    for seed in range(40):
        s = dodge_duel_scenario(config, duel, seed)
        before = _crossing(s["situation"], aim_x)
        moved = copy.deepcopy(s)
        Duel._apply_dodge_params(moved, {"distance_m": distance, "speed_m_s": 0.8}, aim_x)
        st = moved["situation"]
        assert _crossing(st, aim_x) == pytest.approx(before, abs=1e-9)
        assert (st["ball_rel_x_m"] ** 2 + st["ball_rel_y_m"] ** 2) ** 0.5 == pytest.approx(distance)
        assert sum(st[k] ** 2 for k in ("ball_vel_x_m_s", "ball_vel_y_m_s", "ball_vel_z_m_s")) ** 0.5 == pytest.approx(0.8)


@pytest.mark.parametrize("measured, issued, word", [
    ({"collision": 1.0, "fall": 0.0}, [], "Hit"),
    ({"collision": 1.0, "fall": 0.0}, [(0.4, "SIDESTEP_LEFT")], "Hit"),
    ({"collision": 0.0, "fall": 0.0}, [(0.4, "SIDESTEP_LEFT")], "Dodged"),
    ({"collision": 0.0, "fall": 0.0}, [], "Not hit"),  # standing still is no dodge
    ({"collision": 0.0, "fall": 1.0}, [(0.4, "RETREAT")], "Fell"),
    ({"collision": 1.0, "fall": 1.0}, [], "Fell"),
])
def test_result_word_comes_from_the_measured_outcome(measured, issued, word):
    assert result_word(measured, issued) == word
