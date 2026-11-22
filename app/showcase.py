"""What GET /api/demos shows about each demo beyond its lanes: its words from config.yaml app.demos, and the saved
evidence of how its learned lane's student came to be (demo["showcase"]).

    showcase = {"results": null | {"n", "split", "rows": [{"field", "label", "kind", "before", "after", "of"?,
                                                           "better"?}]},
                "learning": {"real_tries", "imagined", "neurons", "train_s", "learn_s", "decision_ms",
                             "spikes_per_decision"}}

Every number is read from a saved run file and checked to belong to the student that runs in the lane (its network
sha256); a number whose source is missing, malformed or belongs to another student is null, never a stand-in. The
functions below are pure over the parsed files; build() only reads them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SETTINGS = ("name", "title", "tagline", "lanes", "results")
LANES = ("before", "after")
RESULT_KINDS = ("count", "metres")
BETTER = ("lower", "higher")
LEARNING = ("real_tries", "imagined", "neurons", "train_s", "learn_s", "decision_ms", "spikes_per_decision")
# The small metadata files of the run whose dream trained a student, copied next to that student's run when that run
# lives in another worktree (outside observed.runs_dir): result.json, context/rows.json, dream/dream.json.
LESSON_SOURCE = "lesson_source"


def demo_settings(config: dict, demo: str) -> dict:
    """config.yaml app.demos.<demo> checked: {name, title, tagline, lanes {before, after}, results [{field, label,
    kind, better?}]}. ValueError naming the key otherwise; nothing has a default."""
    where = f"config.yaml app.demos.{demo}"
    entry = ((config.get("app") or {}).get("demos") or {}).get(demo)
    if not isinstance(entry, dict):
        raise ValueError(f"{where} is missing; each demo needs {', '.join(SETTINGS)}")
    for key in ("name", "title", "tagline"):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            raise ValueError(f"{where}.{key} must be a non-empty string")
    lanes = entry.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != set(LANES) or \
            not all(isinstance(lanes[k], str) and lanes[k].strip() for k in LANES):
        raise ValueError(f"{where}.lanes must name exactly before and after, each a non-empty string")
    rows = entry.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"{where}.results must be a list (empty for none)")
    for i, row in enumerate(rows):
        at = f"{where}.results[{i}]"
        if not isinstance(row, dict) or set(row) - {"field", "label", "kind", "better"}:
            raise ValueError(f"{at} must have field, label, kind and optionally better")
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("field", "label")):
            raise ValueError(f"{at}.field and .label must be non-empty strings")
        if row.get("kind") not in RESULT_KINDS:
            raise ValueError(f"{at}.kind must be one of {', '.join(RESULT_KINDS)}")
        if "better" in row and row["better"] not in BETTER:
            raise ValueError(f"{at}.better must be one of {', '.join(BETTER)}")
    return {"name": entry["name"], "title": entry["title"], "tagline": entry["tagline"],
            "lanes": {k: lanes[k] for k in LANES}, "results": [dict(row) for row in rows]}


def _number(value):
    """value if it is a finite int or float (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _sha(student) -> str | None:
    sha = ((student or {}).get("network") or {}).get("sha256") if isinstance(student, dict) else None
    return sha if isinstance(sha, str) and sha else None


def held_out_results(result, rows: list, student) -> dict | None:
    """The configured rows from a run's held-out closed-loop test (result.json closed_loop_test): before = method
    no_op, after = method snn, over all scenarios. None without rows, without a finished test, when the test's network
    is not this student's, or when the two methods ran different numbers of scenarios. A row whose field either
    method lacks is left out; None if none is left."""
    if not rows or not isinstance(result, dict):
        return None
    test = result.get("closed_loop_test")
    sha = _sha(student)
    if not isinstance(test, dict) or test.get("state") != "done" or sha is None or test.get("network_sha256") != sha:
        return None
    summary = test.get("summary") or {}
    before = ((summary.get("no_op") or {}).get("all")) if isinstance(summary, dict) else None
    after = ((summary.get("snn") or {}).get("all")) if isinstance(summary, dict) else None
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    n = _count(before.get("n"))
    if not n or _count(after.get("n")) != n:
        return None
    out = []
    for row in rows:
        b, a = before.get(row["field"]), after.get(row["field"])
        if row["kind"] == "count":
            b, a = _count(b), _count(a)
            if b is None or a is None or b > n or a > n:
                continue
            out.append({"field": row["field"], "label": row["label"], "kind": "count", "before": b, "after": a,
                        "of": n})
        else:
            b, a = _number(b), _number(a)
            if b is None or a is None:
                continue
            item = {"field": row["field"], "label": row["label"], "kind": row["kind"], "before": b, "after": a}
            if "better" in row:
                item["better"] = row["better"]
            out.append(item)
    split = test.get("split")
    return {"n": n, "split": split if isinstance(split, str) else None, "rows": out} if out else None


def _neurons(student):
    sizes = (((student or {}).get("provenance") or {}).get("hidden_sizes")) if isinstance(student, dict) else None
    if not isinstance(sizes, list) or not sizes or any(_count(s) is None or s == 0 for s in sizes):
        return None
    return sum(sizes)


def _brain(brain, sha):
    """(decision_ms, spikes_per_decision) from a run's story/brain.json measured on this student's network."""
    if not isinstance(brain, dict) or sha is None or brain.get("network_sha256") != sha:
        return None, None
    timing = brain.get("decision_time") if isinstance(brain.get("decision_time"), dict) else {}
    ms = _number(timing.get("median_ms"))
    spikes = _number(brain.get("spikes_per_decision"))
    return (ms if ms is not None and ms > 0 else None), (spikes if spikes is not None and spikes >= 0 else None)


def reflex_learning(result, student, brain) -> dict:
    """How a reflex student learned, from its run's result.json (the attempt loop), its student.json and story/brain.json.

    real_tries  sum of rounds[*].attempt_episodes (real simulator episodes)
    imagined    rounds[-1].dream_windows, the dream the final student was trained on; only if the student's provenance
                names that dream and the training split's rows add up to it
    neurons     sum of the student's hidden layer sizes
    train_s     train.training.wall_seconds (the spiking network's training)
    learn_s     sum of loop.steps wall_s of every step but "test"; null unless each of them has one
    decision_ms, spikes_per_decision   story/brain.json, measured on this network
    Each is null when its source is missing or belongs to another network."""
    out = dict.fromkeys(LEARNING)
    sha = _sha(student)
    out["neurons"] = _neurons(student)
    out["decision_ms"], out["spikes_per_decision"] = _brain(brain, sha)
    if not isinstance(result, dict) or sha is None:
        return out
    train = result.get("train") if isinstance(result.get("train"), dict) else {}
    if train.get("network_sha256") != sha:
        return out
    rounds = result.get("rounds")
    if isinstance(rounds, list) and rounds and all(isinstance(r, dict) for r in rounds):
        tries = [_count(r.get("attempt_episodes")) for r in rounds]
        out["real_tries"] = sum(tries) if None not in tries else None
        last = rounds[-1]
        windows = _count(last.get("dream_windows"))
        split = train.get("split") if isinstance(train.get("split"), dict) else {}
        rows = [_count(split.get("n_train_rows")), _count(split.get("n_validation_rows"))]
        provenance = student.get("provenance") or {}
        if windows and last.get("network_sha256") == sha and provenance.get("dream") == last.get("dream") \
                and (None in rows or sum(rows) == windows):
            out["imagined"] = windows
    training = train.get("training") if isinstance(train.get("training"), dict) else {}
    seconds = _number(training.get("wall_seconds"))
    out["train_s"] = seconds if seconds is not None and seconds > 0 else None
    steps = (result.get("loop") or {}).get("steps") if isinstance(result.get("loop"), dict) else None
    if isinstance(steps, list) and all(isinstance(s, dict) for s in steps):
        walls = [_number(s.get("wall_s")) for s in steps if s.get("step") != "test"]
        if walls and None not in walls:
            out["learn_s"] = sum(walls)
    return out


def dodge_learning(result, student, brain, lesson) -> dict:
    """How a dodge_vision student learned, from its sweep run's result.json (sweep), its student.json, story/brain.json
    if any, and lesson = {"result", "rows", "dream"}: result.json, context/rows.json and dream/dream.json of the run
    whose dream and teacher scores trained it (None when not found), accepted only if that run's teacher scores are
    the ones in the student's provenance (scores_sha256).

    real_tries  the lesson run's real context episodes: n_situations x actions of rows.json, equal to the teacher's
                n_context_rows
    imagined    training + validation rows of sweep.training.split; null if the lesson's dream.json disagrees
    neurons     sum of the student's hidden layer sizes
    train_s     sweep.training.train_wall_s
    learn_s     context, dream, teach and train wall times added; null unless every one is recorded
    decision_ms, spikes_per_decision   story/brain.json, measured on this network (none saved for a sweep run)"""
    out = dict.fromkeys(LEARNING)
    sha = _sha(student)
    out["neurons"] = _neurons(student)
    out["decision_ms"], out["spikes_per_decision"] = _brain(brain, sha)
    if not isinstance(result, dict) or sha is None:
        return out
    sweep = result.get("sweep") if isinstance(result.get("sweep"), dict) else {}
    training = sweep.get("training") if isinstance(sweep.get("training"), dict) else {}
    if training.get("network_sha256") != sha:
        return out
    split = training.get("split") if isinstance(training.get("split"), dict) else {}
    rows = [_count(split.get("n_train_rows")), _count(split.get("n_validation_rows"))]
    imagined = sum(rows) if None not in rows and sum(rows) > 0 else None
    seconds = _number(training.get("train_wall_s"))
    out["train_s"] = seconds if seconds is not None and seconds > 0 else None
    lesson = lesson if isinstance(lesson, dict) else {}
    source, context, dream = (lesson.get(k) if isinstance(lesson.get(k), dict) else None
                              for k in ("result", "rows", "dream"))
    provenance = student.get("provenance") or {}
    teach = (source or {}).get("teach") if isinstance((source or {}).get("teach"), dict) else {}
    scores = teach.get("scores") if isinstance(teach.get("scores"), dict) else {}
    owned = source is not None and scores.get("sha256") is not None \
        and scores.get("sha256") == provenance.get("scores_sha256")
    if owned and dream is not None and dream.get("config_digest") == provenance.get("dream_config_digest"):
        counted = _count(dream.get("n_situations"))
        if counted is not None and imagined is not None and counted != imagined:
            imagined = None  # the sweep's split and the dream disagree: neither is shown
    out["imagined"] = imagined
    if owned and context is not None:
        situations, actions = _count(context.get("n_situations")), context.get("action_names")
        teacher = teach.get("teacher") if isinstance(teach.get("teacher"), dict) else {}
        if situations and isinstance(actions, list) and actions \
                and situations * len(actions) == _count(teacher.get("n_context_rows")):
            out["real_tries"] = situations * len(actions)
    if owned and context is not None and dream is not None:
        walls = [_number(context.get("wall_s")), _number(dream.get("wall_s")), _number(teach.get("wall_s")),
                 out["train_s"]]
        if None not in walls:
            out["learn_s"] = sum(walls)
    return out


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _lesson(student, student_run: Path, runs_root: Path | None):
    """The parsed metadata of the run whose dream trained a dodge student: <runs_root>/<that run> if it is here, else
    the copy in <the student's run>/lesson_source; the first whose result.json names that run."""
    dream = ((student or {}).get("provenance") or {}).get("dream")
    if not isinstance(dream, str) or not dream:
        return None
    name = Path(dream.replace("\\", "/")).parent.name
    places = ([runs_root / name] if runs_root is not None and name else []) + [student_run / LESSON_SOURCE]
    for place in places:
        result = _read(place / "result.json")
        if isinstance(result, dict) and result.get("run_id") == name:
            return {"result": result, "rows": _read(place / "context" / "rows.json"),
                    "dream": _read(place / "dream" / "dream.json")}
    return None


def build(skill: str, settings: dict, run_folder: Path | None, student_folder: Path | None,
          runs_root: Path | None) -> dict:
    """demo["showcase"] from the saved files: test results of the demo's run when its student is the lane's (the
    student folder is that run's own), and the learning record of the lane's student from its own run."""
    learning = dict.fromkeys(LEARNING)
    if student_folder is None:
        return {"results": None, "learning": learning}
    student = _read(student_folder / "student.json")
    if not isinstance(student, dict) or student.get("skill") != skill:
        return {"results": None, "learning": learning}
    own = student_folder.parent
    result = _read(own / "result.json")
    brain = _read(own / "story" / "brain.json")
    results = None
    if run_folder is not None and own.resolve() == run_folder.resolve():
        results = held_out_results(result, settings["results"], student)
    if skill == "reflex":
        learning = reflex_learning(result, student, brain)
    elif skill == "dodge_vision":
        learning = dodge_learning(result, student, brain, _lesson(student, own, runs_root))
    return {"results": results, "learning": learning}
