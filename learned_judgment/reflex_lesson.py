"""The REFLEX lesson, designed by a language model AS DATA from a menu (config.yaml skills.reflex.menu).

A lesson says what the blind dog may feel (sense groups of the menu), how long a stretch of feeling it keeps, how often
it decides, which steering commands it may pick, what reaching the spot means and which mounds it practises on. It is a
JSON object of numbers, lists of numbers, fixed choices and one short title: nothing in it is code, nothing in it is
evaluated or executed, and every value is checked against the menu and the verified command schema before a skill
uses it (learned_judgment/skills/reflex.py reads it from skills.reflex.declaration).

    lesson_schema(config)        the JSON schema given to the model, every menu choice and limit described in words
    check_lesson(lesson, config) problems [{code, field, message}] in plain sentences; [] when the lesson is usable
    resolve_lesson(lesson, cfg)  the cleaned lesson (numbers as floats, senses in menu order, choices sorted) plus
                                 "summary": 3-6 plain sentences for a non-technical reader
    design_lesson(request, cfg)  ONE call to Claude (config.yaml app.chat: model, effort, max_tokens, timeout, retries)
                                 with the schema as structured output: {"lesson", "problems", "model", "usage", ...}

The API key is ANTHROPIC_API_KEY from the environment, else from app.chat.env_file, loaded the way app/chat.py loads it
(dotenv_values, in this process only); it is never printed, logged or written.

Command line (from the repository root):

    python -m learned_judgment.reflex_lesson design --out results/runs/reflex1/declaration.json [--request TEXT]
        [--repair] [--extra-context FILE]
    python -m learned_judgment.reflex_lesson config --declaration results/runs/reflex1/declaration.json
        --out results/runs/reflex1/config.yaml

design appends one call to the declaration file (request, raw answer, parsed lesson, problems, model, usage); with
--repair the previous call's answer and its problems are passed back to the model once; --extra-context adds the text
of a file (for example a measured table) to the request. config writes the repository config plus
skills.reflex.declaration = the accepted (cleaned) lesson.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from learned_judgment.config import CONFIG_PATH, ConfigError, load_config, require

__all__ = ["OWNER_REQUEST", "check_lesson", "design_lesson", "lesson_schema", "main", "menu_text", "resolve_lesson"]

ROOT = CONFIG_PATH.parent
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # as app/chat.py: a refusal is retried server-side
OWNER_REQUEST = ("There is a small steep hill somewhere ahead. Using only what the dog can feel in its body and legs, "
                 "teach it to keep heading for the spot straight ahead: if it feels a slope it should steer left or "
                 "right to get off it, go on, and still reach the spot.")
LESSON_KEYS = ("title", "senses", "sense", "decide_every_s", "horizon_s", "time_limit_s", "start", "nominal_speed_m_s",
               "steer", "goal", "practice_world")
SENSE_KEYS = ("history_s", "sample_every_s")
STEER_KEYS = ("yaw_rate", "vel_x")
GOAL_KEYS = ("forward_m", "tolerance_m", "weights")
WEIGHT_KEYS = ("progress", "offset", "tilt", "fall", "spread")
WORLD_KEYS = ("flat_share", "radius_m", "height_m", "centre_ahead_m", "centre_side_m")
RANGE_KEYS = ("min", "max")
TITLE_MAX_CHARS = 80
_PERIOD_TOLERANCE = 1e-6

SYSTEM = """You design a lesson for a blind simulated robot dog (a Unitree Go2 in the MuJoCo simulator).

The dog walks with a frozen, borrowed walking policy that takes only three commands: forward speed, sideways speed and
turn rate. Your lesson does not change how it walks. It decides what a small spiking neural network may FEEL (raw body
sensors only: no camera, no map, no position) and which walking commands it may PICK, several times per second. The
network then learns from practice episodes in the simulator which pick is best for what it feels.

Answer with data only, in the JSON format you are given: one lesson object. Rules:
- Use only the choices and ranges of the menu below. Every number must lie inside its stated range; every sense group,
  start and field must be one the menu names. Do not invent sensors, commands, keys or values.
- The lesson is data. Write no code, no formulas and no instructions for the server; the server only checks it.
- Keep it learnable from about a dozen practice episodes: few candidate commands, a short window of feeling.

Menu (JSON):
"""


# ----- menu ------------------------------------------------------------------------------------------------------


def _menu(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return require(config, "skills.reflex.menu")


def _period(config: Mapping[str, Any]) -> float:
    return float(require(config, "robot.walking_policy.command_schema.update_period_s"))


def _verified(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return require(config, "robot.walking_policy.command_schema")["fields"]


def menu_text(config: Mapping[str, Any]) -> dict[str, Any]:
    """The menu as the model reads it: sense groups with their channels, every limit, the world's mound limits."""
    menu, mound = _menu(config), require(config, "world.mound")
    fields = _verified(config)
    return {
        "sense_groups": {name: {"what": g["description"], "channels": [c[0] for c in g["channels"]]}
                         for name, g in menu["sense_groups"].items()},
        "never_available": "position, speed over the ground, a camera, a map, or anything about where or how big the "
                           "hill is",
        "window": {"sample_every_s": menu["sample_every_s"], "history_s": menu["history_s"],
                   "max_samples": menu["max_history_samples"], "max_numbers": menu["max_window_width"],
                   "note": "the window holds samples every sample_every_s over the last history_s, oldest first"},
        "timing": {"decide_every_s": menu["decide_every_s"], "horizon_s": menu["horizon_s"],
                   "time_limit_s": menu["time_limit_s"], "control_period_s": _period(config),
                   "note": "every time is a whole number of control periods"},
        "starts": menu["starts"],
        "nominal_speed_m_s": menu["nominal_speed_m_s"],
        "steer": {"turn_rate_rad_s": menu["steer"]["yaw_rate_rad_s"], "forward_speed_m_s": menu["steer"]["vel_x_m_s"],
                  "max_choices_per_field": menu["steer"]["max_choices"], "max_candidates": menu["max_candidates"],
                  "sideways_speed": "always 0",
                  "verified_command_ranges": {k: [fields[k]["min"], fields[k]["max"]] for k in ("vel_x", "yaw_rate")}},
        "goal": menu["goal"],
        "practice_world": {**menu["world"], "max_rim_slope_deg": mound["max_rim_slope_deg"],
                           "min_gap_rim_to_dog_m": mound["min_gap_m"]},
    }


def _range_text(pair) -> str:
    return f"from {pair[0]} to {pair[1]}"


def _number(description: str) -> dict[str, Any]:
    return {"type": "number", "description": description}


def _span(description: str) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": list(RANGE_KEYS), "description": description,
            "properties": {"min": {"type": "number"}, "max": {"type": "number"}}}


def lesson_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    """The lesson's JSON schema with every menu choice and limit in its descriptions (structured outputs accept no
    numeric bounds, so check_lesson enforces them)."""
    menu, mound = _menu(config), require(config, "world.mound")
    groups = list(menu["sense_groups"])
    group_text = "; ".join(f"{name}: {g['description']}" for name, g in menu["sense_groups"].items())
    steer, goal, world = menu["steer"], menu["goal"], menu["world"]
    return {
        "type": "object", "additionalProperties": False, "required": list(LESSON_KEYS),
        "properties": {
            "title": {"type": "string", "description": f"A short name for the lesson, at most {TITLE_MAX_CHARS} characters."},
            "senses": {"type": "array", "items": {"type": "string", "enum": groups},
                       "description": f"The sense groups the dog may feel, at least one, each at most once. {group_text}. "
                                      "Position, speed over the ground and anything about the hill are never available."},
            "sense": {"type": "object", "additionalProperties": False, "required": list(SENSE_KEYS),
                      "description": "How much recent feeling the dog keeps: samples every sample_every_s covering the "
                                     f"last history_s, oldest first. At most {menu['max_history_samples']} samples "
                                     f"(history_s / sample_every_s + 1) and at most {menu['max_window_width']} numbers "
                                     "(samples x channels of the chosen groups).",
                      "properties": {
                          "history_s": _number(f"Seconds, {_range_text(menu['history_s'])}; a whole number of sample_every_s."),
                          "sample_every_s": _number(f"Seconds, {_range_text(menu['sample_every_s'])}; a whole number of "
                                                    f"{_period(config)} s control periods.")}},
            "decide_every_s": _number(f"Seconds between two decisions, {_range_text(menu['decide_every_s'])}; whole control "
                                      "periods. The picked command is held until the next decision."),
            "horizon_s": _number(f"Seconds after a decision over which its result is measured in practice (progress, "
                                 f"offset, lean, fall), {_range_text(menu['horizon_s'])}; whole control periods."),
            "time_limit_s": _number(f"Longest episode in seconds, {_range_text(menu['time_limit_s'])}."),
            "start": {"type": "string", "enum": list(menu["starts"]),
                      "description": "standing: the dog starts from a stand; trotting: it is already walking at the "
                                     "nominal speed when the lesson starts."},
            "nominal_speed_m_s": _number(f"Forward speed of the plain walk, m/s, {_range_text(menu['nominal_speed_m_s'])}. "
                                         "It is sent until the first decision and is the 'straight' choice."),
            "steer": {"type": "object", "additionalProperties": False, "required": list(STEER_KEYS),
                      "description": f"The commands the dog may pick: every combination of one turn rate and one forward "
                                     f"speed (sideways speed is always 0), at most {menu['max_candidates']} combinations.",
                      "properties": {
                          "yaw_rate": {"type": "array", "items": {"type": "number"},
                                       "description": f"1 to {steer['max_choices']} distinct turn rates, rad/s, "
                                                      f"{_range_text(steer['yaw_rate_rad_s'])}; + turns left, - right. "
                                                      "Must include 0 (no turn)."},
                          "vel_x": {"type": "array", "items": {"type": "number"},
                                    "description": f"1 to {steer['max_choices']} distinct forward speeds, m/s, "
                                                   f"{_range_text(steer['vel_x_m_s'])}. Must include nominal_speed_m_s."}}},
            "goal": {"type": "object", "additionalProperties": False, "required": list(GOAL_KEYS),
                     "description": "What reaching the spot means and how each decision is judged.",
                     "properties": {
                         "forward_m": _number(f"Distance to the spot straight ahead, m, {_range_text(goal['forward_m'])}, "
                                              "along the direction the dog faces when the lesson starts."),
                         "tolerance_m": _number(f"Largest sideways miss at the spot that still counts as reaching it, m, "
                                                f"{_range_text(goal['tolerance_m'])}."),
                         "weights": {"type": "object", "additionalProperties": False, "required": list(WEIGHT_KEYS),
                                     "description": f"How a decision is judged over its horizon, each weight from 0 to "
                                                    f"{goal['weight_max']}: + progress (per metre gained toward the "
                                                    "spot), - offset (per metre away from the straight line to the "
                                                    "spot), - tilt (per degree of mean trunk lean), - fall (per fall), "
                                                    "- spread (per metre of uncertainty about the progress).",
                                     "properties": {k: {"type": "number"} for k in WEIGHT_KEYS}}}},
            "practice_world": {
                "type": "object", "additionalProperties": False, "required": list(WORLD_KEYS),
                "description": "The practice courses. A course is flat or has one smooth mound (a spherical cap: flat on "
                               "top, steepest at its rim, the rim slope being asin(radius / sphere radius) with sphere "
                               "radius = (radius^2 + height^2) / (2 height)). Its centre lies centre_ahead_m ahead of the "
                               "dog at the start and centre_side_m to its left (- right). A drawn height whose rim slope "
                               f"would exceed {mound['max_rim_slope_deg']} degrees is lowered to that slope. The rim must "
                               f"stay at least {mound['min_gap_m']} m from the dog at the start: centre_ahead_m.min - "
                               f"radius_m.max >= {mound['min_gap_m']}.",
                "properties": {
                    "flat_share": _number(f"Share of flat courses, {_range_text(world['flat_share'])}."),
                    "radius_m": _span(f"Mound footprint radius range, m, within {_range_text(world['radius_m'])}."),
                    "height_m": _span(f"Mound top height range, m, within {_range_text(world['height_m'])}."),
                    "centre_ahead_m": _span(f"Distance ahead to the mound's centre, m, within "
                                            f"{_range_text(world['centre_ahead_m'])}."),
                    "centre_side_m": _span(f"Sideways offset of the mound's centre, m, + left, within "
                                           f"{_range_text(world['centre_side_m'])}.")}},
        },
    }


# ----- checking --------------------------------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _whole(value: float, unit: float) -> bool:
    count = round(value / unit)
    return count >= 1 and abs(count * unit - value) <= _PERIOD_TOLERANCE


class _Checker:
    def __init__(self) -> None:
        self.problems: list[dict[str, str]] = []

    def add(self, code: str, field: str, message: str) -> None:
        self.problems.append({"code": code, "field": field, "message": message})

    def keys(self, value: Any, field: str, expected: tuple[str, ...]) -> bool:
        """True when value is an object with exactly the expected keys (problems named otherwise)."""
        label = field or "the lesson"
        if not isinstance(value, Mapping):
            self.add("wrong_type", field, f"{label} must be an object with the keys {', '.join(expected)}.")
            return False
        for key in value:
            if key not in expected:
                self.add("unknown_key", f"{field}.{key}".strip("."), f"{label} has a key '{key}' the menu does not "
                                                                       f"know; it allows only {', '.join(expected)}.")
        for key in expected:
            if key not in value:
                self.add("missing_key", f"{field}.{key}".strip("."), f"{label} lacks '{key}'.")
        return all(key in value for key in expected)

    def number(self, value: Any, field: str, low: float, high: float, unit: str = "") -> float | None:
        if not _is_number(value):
            self.add("wrong_type", field, f"{field} must be a number; got {json.dumps(value)}.")
            return None
        if not low <= value <= high:
            self.add("out_of_range", field, f"{field} is {value}{unit}; the menu allows {low} to {high}{unit}.")
            return None
        return float(value)

    def span(self, value: Any, field: str, low: float, high: float, unit: str = "") -> tuple[float, float] | None:
        if not self.keys(value, field, RANGE_KEYS):
            return None
        a = self.number(value["min"], f"{field}.min", low, high, unit)
        b = self.number(value["max"], f"{field}.max", low, high, unit)
        if a is None or b is None:
            return None
        if a > b:
            self.add("bad_range", field, f"{field}.min ({a}) is above {field}.max ({b}).")
            return None
        return a, b

    def choices(self, value: Any, field: str, low: float, high: float, most: int, unit: str) -> list[float] | None:
        if not isinstance(value, list) or not value:
            self.add("wrong_type", field, f"{field} must be a non-empty list of numbers.")
            return None
        if len(value) > most:
            self.add("too_many_choices", field, f"{field} has {len(value)} choices; the menu allows at most {most}.")
        out = [self.number(v, f"{field}[{i}]", low, high, unit) for i, v in enumerate(value)]
        if any(v is None for v in out):
            return None
        if len(set(out)) != len(out):
            self.add("repeated_value", field, f"{field} repeats a value: {out}.")
            return None
        return out


def _check(lesson: Any, config: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """(problems, the values that passed, as floats); nothing in the lesson is ever evaluated."""
    from learned_judgment.env.commands import validate_command  # on use: learned_judgment.env loads the policy runtime

    menu, mound = _menu(config), require(config, "world.mound")
    period = _period(config)
    c = _Checker()
    ok: dict[str, Any] = {}
    if not c.keys(lesson, "", LESSON_KEYS):
        if not isinstance(lesson, Mapping):
            return c.problems, ok
    get = lesson.get

    title = get("title")
    if "title" in lesson:
        if not isinstance(title, str) or not title.strip():
            c.add("wrong_type", "title", "title must be a non-empty text.")
        elif len(title) > TITLE_MAX_CHARS:
            c.add("out_of_range", "title", f"title has {len(title)} characters; at most {TITLE_MAX_CHARS}.")
        else:
            ok["title"] = title.strip()

    groups = list(menu["sense_groups"])
    senses = get("senses")
    if "senses" in lesson:
        if not isinstance(senses, list) or not senses:
            c.add("empty_senses", "senses", "senses must list at least one sense group of the menu.")
        elif any(not isinstance(s, str) or s not in groups for s in senses):
            bad = [s for s in senses if not isinstance(s, str) or s not in groups]
            c.add("unknown_sense", "senses", f"senses names {bad}, not sense groups of the menu ({', '.join(groups)}).")
        elif len(set(senses)) != len(senses):
            c.add("repeated_value", "senses", f"senses lists a group twice: {senses}.")
        else:
            ok["senses"] = [g for g in groups if g in senses]  # menu order: the window's channel order

    sense = get("sense")
    if "sense" in lesson and c.keys(sense, "sense", SENSE_KEYS):
        every = c.number(sense["sample_every_s"], "sense.sample_every_s", *menu["sample_every_s"], " s")
        history = c.number(sense["history_s"], "sense.history_s", *menu["history_s"], " s")
        if every is not None and not _whole(every, period):
            c.add("not_whole_periods", "sense.sample_every_s", f"sense.sample_every_s ({every} s) must be a whole number "
                                                               f"of {period} s control periods.")
            every = None
        if every is not None and history is not None:
            if not _whole(history, every):
                c.add("not_whole_periods", "sense.history_s", f"sense.history_s ({history} s) must be a whole number of "
                                                              f"sense.sample_every_s ({every} s).")
            else:
                samples = round(history / every) + 1
                if samples > menu["max_history_samples"]:
                    c.add("history_too_long", "sense", f"the window would hold {samples} samples; the menu allows at "
                                                       f"most {menu['max_history_samples']}.")
                else:
                    ok["sense"] = {"history_s": history, "sample_every_s": every, "samples": samples}
    if "senses" in ok and "sense" in ok:
        width = ok["sense"]["samples"] * sum(len(menu["sense_groups"][g]["channels"]) for g in ok["senses"])
        if width > menu["max_window_width"]:
            c.add("window_too_wide", "senses", f"the window would hold {width} numbers (samples x channels); the menu "
                                               f"allows at most {menu['max_window_width']}.")
            del ok["sense"]

    for key, unit in (("decide_every_s", " s"), ("horizon_s", " s"), ("time_limit_s", " s")):
        if key in lesson:
            value = c.number(lesson[key], key, *menu[key], unit)
            if value is not None and not _whole(value, period):
                c.add("not_whole_periods", key, f"{key} ({value} s) must be a whole number of {period} s control periods.")
            elif value is not None:
                ok[key] = value

    start = get("start")
    if "start" in lesson:
        if start not in menu["starts"]:
            c.add("unknown_choice", "start", f"start is {json.dumps(start)}; the menu offers {menu['starts']}.")
        else:
            ok["start"] = start
    if "nominal_speed_m_s" in lesson:
        nominal = c.number(lesson["nominal_speed_m_s"], "nominal_speed_m_s", *menu["nominal_speed_m_s"], " m/s")
        if nominal is not None:
            ok["nominal_speed_m_s"] = nominal

    steer = get("steer")
    if "steer" in lesson and c.keys(steer, "steer", STEER_KEYS):
        most = int(menu["steer"]["max_choices"])
        yaw = c.choices(steer["yaw_rate"], "steer.yaw_rate", *menu["steer"]["yaw_rate_rad_s"], most, " rad/s")
        vel = c.choices(steer["vel_x"], "steer.vel_x", *menu["steer"]["vel_x_m_s"], most, " m/s")
        if yaw is not None and 0.0 not in yaw:
            c.add("no_straight", "steer.yaw_rate", "steer.yaw_rate must include 0 (walking on without turning).")
            yaw = None
        if vel is not None and "nominal_speed_m_s" in ok and ok["nominal_speed_m_s"] not in vel:
            c.add("nominal_not_a_choice", "steer.vel_x", f"steer.vel_x must include nominal_speed_m_s "
                                                         f"({ok['nominal_speed_m_s']}).")
            vel = None
        if yaw is not None and vel is not None:
            if len(yaw) * len(vel) > menu["max_candidates"]:
                c.add("too_many_candidates", "steer", f"steer gives {len(yaw)} x {len(vel)} = {len(yaw) * len(vel)} "
                                                      f"commands; the menu allows at most {menu['max_candidates']}.")
            else:
                schema = require(config, "robot.walking_policy.command_schema")
                bad = []
                for v in vel:
                    for w in yaw:
                        try:
                            validate_command({"vel_x": v, "vel_y": 0.0, "yaw_rate": w}, schema)
                        except ValueError as error:
                            bad.append(f"vel_x {v}, yaw_rate {w}: {error}")
                if bad:
                    c.add("unverified_command", "steer", "these commands are outside the verified command ranges: "
                                                         + "; ".join(bad))
                else:
                    ok["steer"] = {"yaw_rate": sorted(yaw), "vel_x": sorted(vel)}

    goal = get("goal")
    if "goal" in lesson and c.keys(goal, "goal", GOAL_KEYS):
        forward = c.number(goal["forward_m"], "goal.forward_m", *menu["goal"]["forward_m"], " m")
        tolerance = c.number(goal["tolerance_m"], "goal.tolerance_m", *menu["goal"]["tolerance_m"], " m")
        weights = None
        if c.keys(goal["weights"], "goal.weights", WEIGHT_KEYS):
            values = {k: c.number(goal["weights"][k], f"goal.weights.{k}", 0.0, float(menu["goal"]["weight_max"]))
                      for k in WEIGHT_KEYS}
            if all(v is not None for v in values.values()):
                if values["progress"] <= 0.0 and values["offset"] <= 0.0:
                    c.add("no_goal_weight", "goal.weights", "goal.weights needs progress or offset above 0, or no "
                                                            "decision is better than another.")
                else:
                    weights = values
        if forward is not None and tolerance is not None and weights is not None:
            ok["goal"] = {"forward_m": forward, "tolerance_m": tolerance, "weights": weights}

    world = get("practice_world")
    if "practice_world" in lesson and c.keys(world, "practice_world", WORLD_KEYS):
        limits = menu["world"]
        share = c.number(world["flat_share"], "practice_world.flat_share", *limits["flat_share"])
        spans = {k: c.span(world[k], f"practice_world.{k}", *limits[k], " m") for k in WORLD_KEYS[1:]}
        if spans["radius_m"] and spans["centre_ahead_m"]:
            gap = spans["centre_ahead_m"][0] - spans["radius_m"][1]
            if gap < mound["min_gap_m"] - 1e-9:
                c.add("mound_too_close", "practice_world.centre_ahead_m",
                      f"a mound could start {gap:.2f} m ahead of the dog (centre_ahead_m.min - radius_m.max); the "
                      f"world needs at least {mound['min_gap_m']} m.")
                spans["centre_ahead_m"] = None
        if spans["radius_m"] and spans["height_m"] and spans["height_m"][0] >= spans["radius_m"][0]:
            c.add("mound_too_tall", "practice_world.height_m", "practice_world.height_m.min must be below "
                                                               "practice_world.radius_m.min (a cap under a half sphere).")
            spans["height_m"] = None
        if share is not None and all(spans.values()):
            ok["practice_world"] = {"flat_share": share, **{k: {"min": v[0], "max": v[1]} for k, v in spans.items()}}
    if "time_limit_s" in ok and "goal" in ok and "nominal_speed_m_s" in ok:
        fastest = max(ok.get("steer", {}).get("vel_x", [ok["nominal_speed_m_s"]]))
        if ok["goal"]["forward_m"] > fastest * ok["time_limit_s"]:
            c.add("goal_out_of_reach", "goal.forward_m", f"goal.forward_m ({ok['goal']['forward_m']} m) is beyond the "
                                                         f"fastest choice ({fastest} m/s) times time_limit_s "
                                                         f"({ok['time_limit_s']} s).")
            del ok["goal"]
    return c.problems, ok


def check_lesson(lesson: Any, config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Problems [{code, field, message}] of a lesson against the menu; [] when it is usable. Never evaluates anything."""
    return _check(lesson, config)[0]


def _plain(values: list[float], unit: str) -> str:
    return ", ".join(f"{v:g}" for v in values) + f" {unit}"


def _summary(lesson: Mapping[str, Any]) -> str:
    words ={"tilt": "how its body leans", "turn_rates": "how fast it is rotating", "heading":
             "which way it faces compared with the start", "joint_angles": "the angles of its leg joints",
             "joint_speeds": "how fast its leg joints move", "feet": "which feet touch the ground",
             "last_command": "the command it sent last"}
    felt = [words.get(g, g) for g in lesson["senses"]]
    felt_text = felt[0] if len(felt) == 1 else ", ".join(felt[:-1]) + " and " + felt[-1]
    goal, world, steer = lesson["goal"], lesson["practice_world"], lesson["steer"]
    turns = [w for w in steer["yaw_rate"] if w != 0.0]
    turn_text = ("go straight or turn at " + _plain([abs(w) for w in sorted(set(map(abs, turns)))], "rad/s")
                 + (" to either side" if any(w > 0 for w in turns) and any(w < 0 for w in turns) else "")) \
        if turns else "go straight"
    n = len(steer["yaw_rate"]) * len(steer["vel_x"])
    weights = goal["weights"]
    against = [name for key, name in (("offset", "drifting off the line"), ("tilt", "leaning"), ("fall", "falling"),
                                      ("spread", "uncertain progress")) if weights[key] > 0]
    sentences = [
        f"The dog learns to walk to a spot {goal['forward_m']:g} m straight ahead and to arrive within "
        f"{goal['tolerance_m']:g} m of it, in at most {lesson['time_limit_s']:g} s, starting "
        f"{'from a stand' if lesson['start'] == 'standing' else 'at a trot'} at {lesson['nominal_speed_m_s']:g} m/s.",
        f"It cannot see: it only feels {felt_text}, over the last {lesson['sense']['history_s']:g} s.",
        f"Every {lesson['decide_every_s']:g} s it picks one of {n} ways to walk on: {turn_text}, at "
        f"{_plain(steer['vel_x'], 'm/s')}.",
        f"It practises on courses of which {round(100 * (1 - world['flat_share']))} in 100 have a smooth mound "
        f"{world['radius_m']['min']:g}-{world['radius_m']['max']:g} m in radius and {world['height_m']['min']:g}-"
        f"{world['height_m']['max']:g} m high, {world['centre_ahead_m']['min']:g}-{world['centre_ahead_m']['max']:g} m "
        f"ahead; the rest are flat.",
        f"Each pick is judged on the following {lesson['horizon_s']:g} s: getting closer to the spot counts for it"
        + (f"; {', '.join(against)} count against it." if against else "."),
    ]
    return " ".join(sentences)


def resolve_lesson(lesson: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """The cleaned lesson (numbers as floats, senses in menu order, steering choices sorted, ranges as [min, max])
    plus "summary". Raises ValueError listing the problems when check_lesson finds any."""
    problems, ok = _check(lesson, config)
    if problems:
        raise ValueError("the lesson has problems: " + " ".join(p["message"] for p in problems))
    cleaned = {k: ok[k] for k in LESSON_KEYS}
    cleaned["sense"] = {k: ok["sense"][k] for k in SENSE_KEYS}
    cleaned["summary"] = _summary(cleaned)
    return cleaned


# ----- the language-model call -----------------------------------------------------------------------------------


def _api_key(settings: Mapping[str, Any]) -> str | None:
    """ANTHROPIC_API_KEY from the environment, else from app.chat.env_file (app/chat.py's api_key, the same way)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        from dotenv import dotenv_values

        file = ROOT / settings["env_file"]
        key = dotenv_values(file).get("ANTHROPIC_API_KEY") if file.is_file() else None
    return key or None


def build_request(messages: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """The keyword arguments of client.beta.messages.create (everything but the key)."""
    settings = require(config, "app.chat")
    return {
        "model": settings["model"],
        "max_tokens": settings["max_tokens"],
        "system": SYSTEM + json.dumps(menu_text(config), indent=1, sort_keys=True),
        "messages": messages,
        "output_config": {"effort": settings["effort"], "format": {"type": "json_schema", "schema": lesson_schema(config)}},
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }


def design_lesson(request_text: str, config: Mapping[str, Any], *, previous: Mapping[str, Any] | None = None,
                  extra_context: str | None = None, call=None) -> dict[str, Any]:
    """ONE call to Claude: the owner's request (plus extra_context, and with `previous` = an earlier design_lesson
    result, its answer and problems for one repair) -> {"lesson" (parsed JSON or None), "problems", "model", "usage",
    "raw_answer", "stop_reason", "request_id"}. The answer is only parsed with json.loads and checked."""
    import anthropic

    settings = require(config, "app.chat")
    text = request_text if not extra_context else f"{request_text}\n\nContext measured in the simulator:\n{extra_context}"
    messages = [{"role": "user", "content": text}]
    if previous is not None:
        messages += [{"role": "assistant", "content": previous["raw_answer"] or "(no answer)"},
                     {"role": "user", "content": "The server's check found these problems with that lesson. Answer with "
                                                 "a corrected lesson that keeps to the menu:\n"
                                                 + "\n".join(f"- {p['field']}: {p['message']}" for p in previous["problems"])}]
    request = build_request(messages, config)
    if call is None:
        key = _api_key(settings)
        if key is None:
            raise RuntimeError("no ANTHROPIC_API_KEY in the environment or in app.chat.env_file")
        client = anthropic.Anthropic(api_key=key, timeout=settings["timeout_s"], max_retries=settings["max_retries"])
        response = client.beta.messages.create(**request)
    else:
        response = call(request)
    answer = next((block.text for block in response.content if block.type == "text"), None)
    try:
        lesson = json.loads(answer) if answer is not None and response.stop_reason != "refusal" else None
    except (TypeError, ValueError):
        lesson = None
    problems = check_lesson(lesson, config) if lesson is not None else [
        {"code": "no_lesson", "field": "", "message": f"the model gave no readable lesson (stop {response.stop_reason})."}]
    usage = response.usage
    return {"lesson": lesson, "problems": problems, "model": response.model,
            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
            "raw_answer": answer, "stop_reason": response.stop_reason, "request_id": getattr(response, "_request_id", None),
            "messages_sent": len(messages), "extra_context": extra_context}


# ----- command line ----------------------------------------------------------------------------------------------


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _design_command(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    path = Path(args.out)
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "request": args.request, "calls": [], "lesson": None, "summary": None, "problems": None}
    if record["request"] != args.request:
        raise ValueError(f"{path} holds another request; use a new file")
    previous = None
    if args.repair:
        if not record["calls"]:
            raise ValueError("--repair needs an earlier call in the declaration file")
        previous = record["calls"][-1]
    extra = Path(args.extra_context).read_text(encoding="utf-8") if args.extra_context else None
    result = design_lesson(args.request, config, previous=previous, extra_context=extra)
    record["calls"].append({"utc": _utc(), "repair": bool(args.repair), **result})
    record["schema"] = lesson_schema(config)
    record["menu"] = menu_text(config)
    if result["lesson"] is not None and not result["problems"]:
        resolved = resolve_lesson(result["lesson"], config)
        record["summary"] = resolved.pop("summary")
        record["lesson"] = resolved
        record["problems"] = []
        record["accepted_call"] = len(record["calls"]) - 1
    else:
        record["problems"] = result["problems"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"model": result["model"], "usage": result["usage"], "stop_reason": result["stop_reason"],
                      "problems": result["problems"], "lesson": result["lesson"], "summary": record["summary"]}, indent=1))
    return 0 if not result["problems"] else 2


def _config_command(args: argparse.Namespace) -> int:
    import yaml

    record = json.loads(Path(args.declaration).read_text(encoding="utf-8"))
    if record.get("lesson") is None:
        raise ValueError(f"{args.declaration} holds no accepted lesson")
    config = load_config()
    problems = check_lesson(record["lesson"], config)
    if problems:
        raise ValueError(f"the stored lesson no longer passes the menu: {problems}")
    config["skills"]["reflex"]["declaration"] = record["lesson"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {out}: the repository config plus skills.reflex.declaration")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m learned_judgment.reflex_lesson", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    design = commands.add_parser("design", help="ONE language-model call; appends it to the declaration file")
    design.add_argument("--out", required=True, help="declaration file (JSON)")
    design.add_argument("--request", default=OWNER_REQUEST, help="the owner's request (default: the reflex request)")
    design.add_argument("--repair", action="store_true", help="pass the previous call's answer and problems back once")
    design.add_argument("--extra-context", help="a text file added to the request (for example a measured table)")
    write = commands.add_parser("config", help="write a run config: the repository config plus the accepted lesson")
    write.add_argument("--declaration", required=True)
    write.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "config":
        return _config_command(args)
    try:
        config = load_config()
    except ConfigError as error:
        print(f"config: {error}", file=sys.stderr)
        return 1
    return _design_command(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
