"""The demo words and the showcase (app/showcase.py): config.yaml app.demos checked at load, and the held-out results and
learning record read from saved run files. The tiny dicts below are fixtures for the pure functions only; the last
tests read the real saved runs that GET /api/demos serves."""
import copy
import json

import pytest

from app import backend, duel, showcase
from learned_judgment.env import load_config

SHA = "a" * 64


@pytest.fixture(scope="module")
def config():
    return load_config()


# ---- config.yaml app.demos -----------------------------------------------------------------------------------------

def test_every_demo_has_its_words_in_config(config):
    for demo in duel.DEMOS:
        entry = config["app"]["demos"][demo]
        words = showcase.demo_settings(config, demo)
        assert words == {"name": entry["name"], "title": entry["title"], "tagline": entry["tagline"],
                         "lanes": entry["lanes"], "results": entry["results"]}


@pytest.mark.parametrize("change, key", [
    (lambda d: d.pop("name"), "hills.name"),
    (lambda d: d.pop("title"), "hills.title"),
    (lambda d: d.__setitem__("tagline", ""), "hills.tagline"),
    (lambda d: d.pop("lanes"), "hills.lanes"),
    (lambda d: d["lanes"].pop("after"), "hills.lanes"),
    (lambda d: d.pop("results"), "hills.results"),
    (lambda d: d["results"].append({"field": "x", "label": "X", "kind": "percent"}), "hills.results[{n}].kind"),
    (lambda d: d["results"].append({"field": "x", "label": "X", "kind": "metres", "better": "up"}),
     "hills.results[{n}].better"),
    (lambda d: d["results"].append({"field": "x", "label": "X", "kind": "count", "unit": "m"}), "hills.results[{n}]"),
], ids=["name", "title", "empty_tagline", "lanes", "one_lane", "results", "kind", "better", "extra_key"])
def test_a_missing_or_bad_demo_setting_is_an_error_naming_its_key(config, change, key):
    bad = copy.deepcopy(config)
    key = key.format(n=len(bad["app"]["demos"]["hills"]["results"]))  # an appended row's index
    change(bad["app"]["demos"]["hills"])
    with pytest.raises(ValueError, match=f"app.demos.{key}".replace("[", r"\[").replace("]", r"\]")):
        showcase.demo_settings(bad, "hills")


def test_a_missing_demo_is_an_error(config):
    bad = copy.deepcopy(config)
    del bad["app"]["demos"]["dodge"]
    with pytest.raises(ValueError, match="app.demos.dodge"):
        showcase.demo_settings(bad, "dodge")


# ---- held-out results (fixture dicts) -----------------------------------------------------------------------------

ROWS = [{"field": "crossed", "label": "Crossed", "kind": "count"},
        {"field": "miss_m", "label": "Miss", "kind": "metres", "better": "lower"}]
STUDENT = {"skill": "reflex", "network": {"sha256": SHA}, "provenance": {"hidden_sizes": [16, 8], "dream": "d/r1"}}


def _test(**changes):
    test = {"state": "done", "split": "test", "network_sha256": SHA,
            "summary": {"no_op": {"all": {"n": 4, "crossed": 1, "miss_m": 2.5}},
                        "snn": {"all": {"n": 4, "crossed": 3, "miss_m": 0.5}}}}
    test.update(changes)
    return {"closed_loop_test": test}


def test_results_pair_no_op_before_and_snn_after():
    assert showcase.held_out_results(_test(), ROWS, STUDENT) == {"n": 4, "split": "test", "rows": [
        {"field": "crossed", "label": "Crossed", "kind": "count", "before": 1, "after": 3, "of": 4},
        {"field": "miss_m", "label": "Miss", "kind": "metres", "before": 2.5, "after": 0.5, "better": "lower"}]}


def test_results_are_null_unless_the_test_ran_on_this_students_network():
    assert showcase.held_out_results(_test(network_sha256="b" * 64), ROWS, STUDENT) is None
    assert showcase.held_out_results(_test(state="running"), ROWS, STUDENT) is None
    assert showcase.held_out_results(_test(), ROWS, {"skill": "reflex"}) is None
    assert showcase.held_out_results({}, ROWS, STUDENT) is None
    assert showcase.held_out_results(_test(), [], STUDENT) is None  # nothing configured: nothing shown


def test_results_leave_out_a_missing_field_and_refuse_unequal_counts():
    result = _test()
    del result["closed_loop_test"]["summary"]["snn"]["all"]["miss_m"]
    assert [r["field"] for r in showcase.held_out_results(result, ROWS, STUDENT)["rows"]] == ["crossed"]
    result["closed_loop_test"]["summary"]["snn"]["all"]["crossed"] = None
    assert showcase.held_out_results(result, ROWS, STUDENT) is None  # no row left
    result = _test()
    result["closed_loop_test"]["summary"]["snn"]["all"]["n"] = 5
    assert showcase.held_out_results(result, ROWS, STUDENT) is None
    result = _test()
    result["closed_loop_test"]["summary"]["snn"]["all"]["crossed"] = 9  # more than n
    assert [r["field"] for r in showcase.held_out_results(result, ROWS, STUDENT)["rows"]] == ["miss_m"]


# ---- learning record (fixture dicts) ------------------------------------------------------------------------------

def _reflex_result():
    return {"rounds": [{"attempt_episodes": 3, "dream_windows": 50, "dream": "d/r0", "network_sha256": "c" * 64},
                       {"attempt_episodes": 4, "dream_windows": 90, "dream": "d/r1", "network_sha256": SHA}],
            "train": {"network_sha256": SHA, "split": {"n_train_rows": 80, "n_validation_rows": 10},
                      "training": {"wall_seconds": 2.5}},
            "loop": {"steps": [{"step": "attempts", "wall_s": 1.0}, {"step": "dream", "wall_s": 2.0},
                               {"step": "test", "wall_s": 100.0}, {"step": "train", "wall_s": 3.0}]}}


BRAIN = {"network_sha256": SHA, "decision_time": {"median_ms": 0.2}, "spikes_per_decision": 40.5}


def test_reflex_learning_reads_every_item_from_its_source():
    assert showcase.reflex_learning(_reflex_result(), STUDENT, BRAIN) == {
        "real_tries": 7, "imagined": 90, "neurons": 24, "train_s": 2.5, "learn_s": 6.0, "decision_ms": 0.2,
        "spikes_per_decision": 40.5}


def test_reflex_learning_leaves_out_what_does_not_belong_to_this_student():
    result = _reflex_result()
    result["loop"]["steps"][1].pop("wall_s")  # one lesson step without a wall time: no total
    result["train"]["split"]["n_validation_rows"] = 11  # the split does not add up to the dream
    learned = showcase.reflex_learning(result, STUDENT, BRAIN | {"network_sha256": "d" * 64})
    assert learned["learn_s"] is None and learned["imagined"] is None
    assert learned["decision_ms"] is None and learned["spikes_per_decision"] is None
    assert learned["real_tries"] == 7 and learned["train_s"] == 2.5
    other = showcase.reflex_learning(_reflex_result() | {"train": {"network_sha256": "e" * 64}}, STUDENT, None)
    assert other == dict.fromkeys(showcase.LEARNING) | {"neurons": 24}  # another network's run: only the student's size


DODGE_STUDENT = {"skill": "dodge_vision", "network": {"sha256": SHA},
                 "provenance": {"hidden_sizes": [32], "dream": "x/runs/v9/dream", "scores_sha256": "s1",
                                "dream_config_digest": "g1"}}


def _dodge():
    result = {"sweep": {"training": {"network_sha256": SHA, "train_wall_s": 5.0,
                                     "split": {"n_train_rows": 90, "n_validation_rows": 10}}}}
    lesson = {"result": {"run_id": "v9", "teach": {"scores": {"sha256": "s1"}, "teacher": {"n_context_rows": 12}}},
              "rows": {"n_situations": 3, "action_names": ["A", "B", "C", "D"]},
              "dream": {"n_situations": 100, "config_digest": "g1"}}
    return result, lesson


def test_dodge_learning_reads_the_lesson_run_that_trained_the_student():
    result, lesson = _dodge()
    assert showcase.dodge_learning(result, DODGE_STUDENT, None, lesson) == {
        "real_tries": 12, "imagined": 100, "neurons": 32, "train_s": 5.0, "learn_s": None, "decision_ms": None,
        "spikes_per_decision": None}
    lesson["rows"]["wall_s"], lesson["dream"]["wall_s"], lesson["result"]["teach"]["wall_s"] = 1.0, 2.0, 3.0
    assert showcase.dodge_learning(result, DODGE_STUDENT, None, lesson)["learn_s"] == 11.0  # every step timed


def test_dodge_learning_refuses_another_lesson_and_disagreeing_counts():
    result, lesson = _dodge()
    lesson["result"]["teach"]["scores"]["sha256"] = "other"  # another run's teacher scores
    learned = showcase.dodge_learning(result, DODGE_STUDENT, None, lesson)
    assert learned["real_tries"] is None and learned["imagined"] == 100  # the split alone still counts
    result, lesson = _dodge()
    lesson["dream"]["n_situations"] = 101
    lesson["result"]["teach"]["teacher"]["n_context_rows"] = 13
    learned = showcase.dodge_learning(result, DODGE_STUDENT, None, lesson)
    assert learned["imagined"] is None and learned["real_tries"] is None
    result, lesson = _dodge()
    result["sweep"]["training"]["network_sha256"] = "f" * 64
    assert showcase.dodge_learning(result, DODGE_STUDENT, None, lesson) == dict.fromkeys(showcase.LEARNING) | {
        "neurons": 32}


# ---- the real saved runs ------------------------------------------------------------------------------------------

def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_served_showcase_comes_from_the_saved_runs(config):
    root, _ = backend.runs_dir(config)
    for demo in duel.DEMOS:
        card = duel._demo(config, demo)
        assert set(card["showcase"]) == {"results", "learning"}
        assert set(card["showcase"]["learning"]) == set(showcase.LEARNING)
        for value in card["showcase"]["learning"].values():
            assert value is None or (isinstance(value, (int, float)) and value >= 0)
    hills = duel._demo(config, "hills")
    if hills["run_id"] is None:
        pytest.skip("no finished reflex run in this tree")
    result = _json(root / hills["run_id"] / "result.json")
    learning = hills["showcase"]["learning"]
    if learning["real_tries"] is not None:
        assert learning["real_tries"] == sum(r["attempt_episodes"] for r in result["rounds"])
    if learning["train_s"] is not None:
        assert learning["train_s"] == result["train"]["training"]["wall_seconds"]
    results = hills["showcase"]["results"]
    if results is not None:
        summary = result["closed_loop_test"]["summary"]
        for row in results["rows"]:
            assert row["before"] == summary["no_op"]["all"][row["field"]]
            assert row["after"] == summary["snn"]["all"][row["field"]]


def test_hills_outcome_fields_are_keys_of_the_saved_episode_outcomes(config):
    """reflex.run_episode's outcome is what a hills lane sends; its saved test episodes show which keys it has."""
    root, _ = backend.runs_dir(config)
    hills = duel._demo(config, "hills")
    run = root / str(hills["run_id"]) / "test_run" / "run.json"
    if not run.is_file():
        pytest.skip("the hills run has no saved test episodes in this tree")
    names = {f["name"] for f in hills["outcome_fields"]}
    assert names == {"crossed_goal_line", "fall", "end_s"}
    for record in _json(run)["records"]:
        for method in ("no_op", "snn"):
            assert names <= set(record[method]["outcome"])
