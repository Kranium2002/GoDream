"""The dodge duel: which punch a round shows, and the plain result word of each lane (app/duel.py); the brain events
of the learned lanes."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app import backend
from app.duel import Duel, _demo, _read_yaml, _student_for, dodge_duel_scenario, result_word
from learned_judgment.env import load_config
from learned_judgment.skills import dodge_vision, reflex

AIMED_AT_DOG = {"incoming_center", "incoming_left", "incoming_right"}


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_config_names_only_kinds_aimed_at_the_dog_from_the_held_out_split(config):
    duel = config["app"]["duel"]["dodge"]
    assert set(duel["kinds"]) <= AIMED_AT_DOG and duel["kinds"]
    assert duel["split"] == "test"


def test_config_sets_the_hills_walking_alone_yaw_trim(config):
    trim = config["app"]["duel"]["hills"]["baseline_yaw_trim_rad_s"]
    assert isinstance(trim, float) and -1.0 <= trim <= 1.0  # inside the yaw_rate command range it is added to


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


# ---- brain events: the learned lane's own decisions, timed --------------------------------------------------------
# The saved students of the served demos decide on seeded random windows; only the episode loop (reflex.run_episode,
# dodge_vision.closed_loop) and the head camera are stand-ins, so no simulator runs. Plumbing only: no evidence that
# anything walks or dodges.

class Announcements:
    def __init__(self):
        self.events = []

    def announce(self, name, data, latest_only=False):
        self.events.append((name, data, latest_only))


def _lane(config, demo):
    root, _ = backend.runs_dir(config)
    card = _demo(config, demo)
    if not card["available"]:
        pytest.skip(f"the {demo} demo has no run and student in this tree")
    folder = root / card["run_id"]
    student_folder = _student_for(root, card["skill"], config, folder)
    cfg = _read_yaml(folder / "config.yaml") or config
    cfg = _read_yaml(student_folder.parent / "config.yaml") or cfg  # as start_duel picks it
    return folder, student_folder, cfg, json.loads((student_folder / "student.json").read_text(encoding="utf-8"))


def _stage_worlds(cfg):
    return {side: SimpleNamespace(config=cfg, _t=lambda t=t: t) for side, t in (("before", 1.0), ("after", 2.5))}


def _brains(sim):
    found = [(data, latest) for name, data, latest in sim.events if name == "brain"]
    assert all(latest for _, latest in found)  # newest only
    return [data for data, _ in found]


def test_hills_brain_events_time_the_students_own_choice(config, monkeypatch):
    folder, student_folder, cfg, student = _lane(config, "hills")
    calls, trims = [], {}

    def episode(world, scenario, seed, policy):
        trims[world.side] = world.yaw_trim_rad_s
        rng = np.random.default_rng([seed, len(calls)])
        for _ in range(6):
            window = rng.random(student["input"]["features"]).astype(np.float32)
            calls.append((world.side, window, policy(window.copy(), {})))
        return {"outcome": {"crossed_goal_line": 1, "fall": 0, "end_s": 9.0}}

    monkeypatch.setattr(reflex, "run_episode", episode)
    sim = Announcements()
    manager = Duel(SimpleNamespace(sim=sim))
    worlds = _stage_worlds(cfg)
    duel = config["app"]["duel"]["hills"]
    manager._run_hills({"folder": folder, "student_folder": student_folder, "config": cfg, "params": {}, "duel": duel},
                       worlds["before"], worlds["after"], 7)
    # the configured trim reaches the walking-alone lane only; the learned lane's commands are the student's own
    assert trims == {"before": float(duel["baseline_yaw_trim_rad_s"]), "after": 0.0}
    after = [c for c in calls if c[0] == "after"]
    assert len(after) == 6 and [c[2] for c in calls if c[0] == "before"] == [0] * 6
    direct = reflex._StudentPolicy(student_folder, cfg)
    assert [action for _, _, action in after] == [direct(window) for _, window, _ in after]  # unchanged decisions
    brains = _brains(sim)
    assert len(brains) == len(after)
    for (_, _, action), event in zip(after, brains):
        assert set(event) == {"side", "t_s", "decision_ms", "choice"}
        assert event["side"] == "after" and event["t_s"] == 2.5 and event["decision_ms"] > 0
        assert event["choice"] == student["action_names"][action]
    assert manager.outcomes["after"]["crossed_goal_line"] == 1


def test_dodge_brain_events_time_the_runtimes_own_answer(config, monkeypatch):
    from learned_judgment.runtime import DodgeRuntime
    folder, student_folder, cfg, student = _lane(config, "dodge")
    frames, size = student["input"]["frames"], student["input"]["frame_size"]
    calls = []

    def closed_loop(world, relay, scenario, seed, policy, settings, augment_rng=None):
        rng = np.random.default_rng([seed, len(calls)])
        for i in range(8):
            window, t_s = rng.random((frames, size, size)).astype(np.float32), 0.1 * (i + 1)
            calls.append((world.side, window, t_s, policy(window.copy(), t_s)))
        return {"measured": {"collision": 0.0, "fall": 0.0, "min_separation_m": 0.3}, "issued": []}

    monkeypatch.setattr(dodge_vision, "closed_loop", closed_loop)
    monkeypatch.setattr(dodge_vision, "HeadView", lambda _cfg: SimpleNamespace(close=lambda: None))
    actions = dodge_vision._actions(cfg)
    plan = {"folder": folder, "student_folder": student_folder, "config": cfg, "params": {}, "student": student,
            "runtime": DodgeRuntime.load(student_folder, cfg), "actions": actions.commands, "no_op": actions.no_op,
            "duel": config["app"]["duel"]["dodge"]}
    sim = Announcements()
    manager = Duel(SimpleNamespace(sim=sim))
    worlds = _stage_worlds(cfg)
    manager._run_dodge(plan, worlds["before"], worlds["after"], 3)
    after = [c for c in calls if c[0] == "after"]
    assert len(after) == 8 and len([c for c in calls if c[0] == "before"]) == 8
    reference = DodgeRuntime.load(student_folder, cfg)  # a fresh runtime fed the same windows: the same answers
    for _, window, t_s, out in after:
        expected = reference.infer(window, t_s)
        assert (out["action"], out["issued"], out["command"]) == (expected["action"], expected["issued"],
                                                                  expected["command"])
    brains = _brains(sim)
    assert len(brains) == len(after)
    for (_, _, t_s, out), event in zip(after, brains):
        assert set(event) == {"side", "t_s", "decision_ms", "choice", "issued"}
        assert event["side"] == "after" and event["t_s"] == t_s and event["decision_ms"] > 0
        assert event["choice"] == out["action"] and event["issued"] is out["issued"]
    assert manager.outcomes["after"]["result"] == "Not hit"
