"""Declared skills: the gate between a chat request and a registered SkillSpec (plain data).

Lane contract (Person B)
------------------------
A chat request becomes a *declaration*: plain data naming one declared skill and values for its
supported parameters, for example::

    {"skill": "RAMP", "params": {"incline_deg": 10.0}, "source_text": "cross this 10 degree slope"}

config.yaml ``declared`` is the single source of what may be declared. Per skill (``declared.skills``,
in config order): the SkillSpec that runs it (``spec``), a description, request ``keywords``, the
simulator capabilities it needs (``requires``), its ``actions`` and ``outcomes``, and its ``params``:
``number`` (min, max, default, unit) or ``choices`` (choices, a default subset and an optional
``requires`` map {choice: capability}). ``declared.planned`` names the MISSION.md skills that have no
SkillSpec yet, with their keywords.

    world_capabilities   what the simulator offers now, read from Go2World's situation keys
    skill_catalog        every declared skill, its parameters and whether it can run now
    declaration_schema   JSON Schema (draft 2020-12 vocabulary) of a declaration
    check_declaration    plain-language problems of a declaration; [] means valid
    resolve_declaration  a valid declaration with every parameter filled in
    parse_request        keyword rules from request text to a declaration (no model call)

Nothing in a declaration is evaluated or executed, and no command is built here: a resolved
declaration only names a skill, its SkillSpec and parameter values.

A problem is {"code", "field", "message"}. ``field`` is "" (the whole declaration or request),
"skill", "source_text", "params" or "params.<name>". Declaration codes: not_an_object,
unknown_field, missing_field, unknown_skill, wrong_type, too_long, unknown_param, out_of_range,
empty_choice, not_a_choice, duplicate_choice, capability_missing, spec_missing. Request codes:
empty_request, unsupported_request, ambiguous_request, planned_skill.

Functions taking ``capabilities`` and ``implemented`` use, when None, :func:`world_capabilities` and
the names from :func:`learned_judgment.skills.available_skills` plus the names of the modules of
learned_judgment.skills that define a callable ``build_observed_spec`` (a module that fails to import
is skipped). A malformed ``declared`` block
raises ConfigError naming the key; a bad declaration never raises (except in resolve_declaration).
"""

from __future__ import annotations

import copy
import importlib
import math
import pkgutil
import re
from collections.abc import Collection, Mapping
from typing import Any

from learned_judgment.config import ConfigError, optional, require

__all__ = [
    "DeclarationError",
    "check_declaration",
    "declaration_schema",
    "parse_request",
    "resolve_declaration",
    "skill_catalog",
    "world_capabilities",
]

_TOP_LEVEL = ("skill", "params", "source_text")
_CATALOG_PARAM_KEYS = ("type", "min", "max", "choices", "default", "unit", "description")

# Each simulator capability a skill or choice may require, and the sentence shown when it is absent.
_MISSING_CAPABILITY = {
    "ball": "the simulator has no ball",
    "incline": "the simulator has no traversable incline yet (Go2World.reset does not accept incline_deg)",
    "punch_proxy": "the simulator has no punch proxy yet",
}

_WORDS = re.compile(r"[a-z]+")
# Number parameters read from request text, by unit; a parameter with another unit is never read from text.
_UNIT_PATTERNS = {
    "deg": re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?\b|deg\b)"),
    "m/s": re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:m/s|met(?:er|re)s? per second)"),
}


class DeclarationError(ValueError):
    """A declaration failed :func:`check_declaration`. ``problems`` is that list; str() joins the messages."""

    def __init__(self, problems: list[dict[str, str]]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(problem["message"] for problem in self.problems))


# --- config.yaml `declared` -------------------------------------------------------------------------


def _text(config: Mapping[str, Any], key: str) -> str:
    value = require(config, key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"config.yaml key {key!r} must be a non-empty string; got {value!r}")
    return value


def _str_list(config: Mapping[str, Any], key: str) -> list[str]:
    value = require(config, key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"config.yaml key {key!r} must be a list of strings; got {value!r}")
    return list(value)


def _bound(config: Mapping[str, Any], key: str) -> float:
    value = require(config, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"config.yaml key {key!r} must be a finite number; got {value!r}")
    return float(value)


def _known_capability(capability: Any, key: str) -> str:
    if not isinstance(capability, str) or capability not in _MISSING_CAPABILITY:
        known = list(_MISSING_CAPABILITY)
        raise ConfigError(f"config.yaml key {key!r} names {capability!r}; known capabilities: {known}")
    return capability


def _skills(config: Mapping[str, Any]) -> Mapping[str, Any]:
    skills = require(config, "declared.skills")
    if not isinstance(skills, Mapping) or not skills or not all(isinstance(name, str) for name in skills):
        raise ConfigError("config.yaml key 'declared.skills' must be a non-empty mapping of skill names")
    return skills


def _planned(config: Mapping[str, Any]) -> dict[str, list[str]]:
    planned = require(config, "declared.planned")
    if not isinstance(planned, Mapping) or not all(isinstance(name, str) for name in planned):
        raise ConfigError("config.yaml key 'declared.planned' must be a mapping of skill name -> keywords")
    return {name: _str_list(config, f"declared.planned.{name}") for name in planned}


def _max_chars(config: Mapping[str, Any]) -> int:
    value = require(config, "declared.max_source_text_chars")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"config.yaml key 'declared.max_source_text_chars' must be an integer >= 1; got {value!r}")
    return value


def _params(config: Mapping[str, Any], skill: str) -> dict[str, dict[str, Any]]:
    """name -> validated parameter: {"type": "number", min, max, default (floats), unit, description} or
    {"type": "choices", choices, default (lists), requires ({choice: capability}), description}."""
    key = f"declared.skills.{skill}.params"
    raw = require(config, key)
    if not isinstance(raw, Mapping) or not all(isinstance(name, str) for name in raw):
        raise ConfigError(f"config.yaml key {key!r} must be a mapping of parameter names")
    params: dict[str, dict[str, Any]] = {}
    for name in raw:
        where = f"{key}.{name}"
        kind = require(config, f"{where}.type")
        param: dict[str, Any] = {"type": kind, "description": _text(config, f"{where}.description")}
        if kind == "number":
            param |= {bound: _bound(config, f"{where}.{bound}") for bound in ("min", "max", "default")}
            param["unit"] = _text(config, f"{where}.unit")
            if not param["min"] <= param["default"] <= param["max"]:
                raise ConfigError(f"config.yaml key {where!r} needs min <= default <= max")
        elif kind == "choices":
            choices = _str_list(config, f"{where}.choices")
            default = _str_list(config, f"{where}.default")
            requires = optional(config, f"{where}.requires")  # absent: no choice needs a capability
            requires = {} if requires is None else requires
            if not choices or len(set(choices)) != len(choices):
                raise ConfigError(f"config.yaml key '{where}.choices' must be a non-empty list without repeats")
            if not default or len(set(default)) != len(default) or not set(default) <= set(choices):
                raise ConfigError(f"config.yaml key '{where}.default' must be a non-empty subset of its choices")
            if not isinstance(requires, Mapping) or not set(requires) <= set(choices):
                raise ConfigError(f"config.yaml key '{where}.requires' must map some of its choices to capabilities")
            for capability in requires.values():
                _known_capability(capability, f"{where}.requires")
            param |= {"choices": choices, "default": default, "requires": dict(requires)}
        else:
            raise ConfigError(f"config.yaml key '{where}.type' must be 'number' or 'choices'; got {kind!r}")
        params[name] = param
    return params


def _skill(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """The validated entry of declared.skills.<name>."""
    where = f"declared.skills.{name}"
    requires = _str_list(config, f"{where}.requires")
    for capability in requires:
        _known_capability(capability, f"{where}.requires")
    return {
        "spec": _text(config, f"{where}.spec"),
        "description": _text(config, f"{where}.description"),
        "keywords": _str_list(config, f"{where}.keywords"),
        "requires": requires,
        "actions": _str_list(config, f"{where}.actions"),
        "outcomes": _str_list(config, f"{where}.outcomes"),
        "params": _params(config, name),
    }


# --- capabilities and implemented SkillSpecs ---------------------------------------------------------


def world_capabilities() -> frozenset[str]:
    """Capabilities of the simulator now, from the situation keys Go2World.reset accepts.

    "ball" when every BALL_KEYS entry is in SITUATION_KEYS; "incline" when "incline_deg" is;
    "punch_proxy" when "object_kind" is. Imports learned_judgment.env.world (MuJoCo) on first call;
    builds no world.
    """
    from learned_judgment.env.throws import BALL_KEYS
    from learned_judgment.env.world import SITUATION_KEYS

    keys = set(SITUATION_KEYS)
    found = set()
    if all(key in keys for key in BALL_KEYS):
        found.add("ball")
    if "incline_deg" in keys:
        found.add("incline")
    if "object_kind" in keys:
        found.add("punch_proxy")
    return frozenset(found)


def _names(value: Any, argument: str) -> frozenset[str]:
    names = not isinstance(value, (str, bytes)) and isinstance(value, Collection)
    if not names or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{argument} must be a collection of names (str), not a single string; got {value!r}")
    return frozenset(value)


def _capabilities(capabilities: Collection[str] | None) -> frozenset[str]:
    return world_capabilities() if capabilities is None else _names(capabilities, "capabilities")


def _observed_specs() -> frozenset[str]:
    """Names of the modules of learned_judgment.skills that define a callable ``build_observed_spec``.
    A module that fails to import is skipped."""
    import learned_judgment.skills as skills

    names = set()
    for module_info in pkgutil.iter_modules(skills.__path__):
        try:
            module = importlib.import_module(f"{skills.__name__}.{module_info.name}")
        except Exception:  # noqa: BLE001 - a module that does not import declares no SkillSpec
            continue
        if callable(getattr(module, "build_observed_spec", None)):
            names.add(module_info.name)
    return frozenset(names)


def _implemented(implemented: Collection[str] | None) -> frozenset[str]:
    if implemented is None:
        from learned_judgment.skills import available_skills

        return frozenset(available_skills()) | _observed_specs()
    return _names(implemented, "implemented")


def _spec_sentence(spec: str) -> str:
    return f"SkillSpec '{spec}' is not implemented yet (learned_judgment/skills/{spec}.py)"


def _missing(entry: Mapping[str, Any], capabilities: frozenset[str], implemented: frozenset[str]) -> list[str]:
    missing = [_MISSING_CAPABILITY[capability] for capability in entry["requires"] if capability not in capabilities]
    if entry["spec"] not in implemented:
        missing.append(_spec_sentence(entry["spec"]))
    return missing


# --- catalog and schema ----------------------------------------------------------------------------


def skill_catalog(
    config: Mapping[str, Any],
    *,
    capabilities: Collection[str] | None = None,
    implemented: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """One entry per declared skill, in config order.

    Each entry: {"skill", "spec", "description", "params": {name: the keys present in config among
    type, min, max, choices, default, unit, description}, "actions", "outcomes", "available", "missing"}.
    ``missing`` holds one sentence per capability of the skill's ``requires`` absent from
    ``capabilities`` and one when its spec is not in ``implemented``; available = not missing. A
    choice's own ``requires`` does not affect availability.

    Raises:
        ConfigError: the ``declared`` block is missing a key or malformed.
        TypeError:   capabilities or implemented is not a collection of str.
    """
    capability_set, implemented_set = _capabilities(capabilities), _implemented(implemented)
    catalog = []
    for name in _skills(config):
        entry = _skill(config, name)
        raw = require(config, f"declared.skills.{name}.params")
        missing = _missing(entry, capability_set, implemented_set)
        catalog.append({
            "skill": name,
            "spec": entry["spec"],
            "description": entry["description"],
            "params": {
                param: {key: copy.deepcopy(raw[param][key]) for key in _CATALOG_PARAM_KEYS if key in raw[param]}
                for param in raw
            },
            "actions": list(entry["actions"]),
            "outcomes": list(entry["outcomes"]),
            "available": not missing,
            "missing": missing,
        })
    return catalog


def declaration_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    """JSON Schema (draft 2020-12 vocabulary) of a declaration: {"oneOf": [one object schema per skill]}.

    A number parameter is {"type": "number", "minimum", "maximum", "description"}; a choices parameter is
    {"type": "array", "items": {"enum": choices}, "minItems": 1, "uniqueItems": true, "description"}.
    Capabilities and implemented SkillSpecs are not part of the schema: :func:`check_declaration` checks them.

    Raises:
        ConfigError: the ``declared`` block is missing a key or malformed.
    """
    max_chars = _max_chars(config)
    variants = []
    for name in _skills(config):
        properties: dict[str, Any] = {}
        for param, spec in _skill(config, name)["params"].items():
            if spec["type"] == "number":
                properties[param] = {
                    "type": "number",
                    "minimum": spec["min"],
                    "maximum": spec["max"],
                    "description": spec["description"],
                }
            else:
                properties[param] = {
                    "type": "array",
                    "items": {"enum": list(spec["choices"])},
                    "minItems": 1,
                    "uniqueItems": True,
                    "description": spec["description"],
                }
        variants.append({
            "type": "object",
            "properties": {
                "skill": {"const": name},
                "params": {"type": "object", "properties": properties, "additionalProperties": False},
                "source_text": {"type": "string", "maxLength": max_chars},
            },
            "required": ["skill"],
            "additionalProperties": False,
        })
    return {"oneOf": variants}


# --- checking and resolving ------------------------------------------------------------------------


def _problem(code: str, field: str, message: str) -> dict[str, str]:
    return {"code": code, "field": field, "message": message}


def _shown(value: Any) -> str:
    """repr of a value taken from a declaration, for a message. Never raises (Python refuses to print an
    integer of more than 4300 digits, and a deeply nested value can exceed the recursion limit)."""
    try:
        return repr(value)
    except Exception:  # noqa: BLE001 - a message must not fail on what it describes
        return f"<{type(value).__name__} that cannot be printed>"


def _kind(value: Any) -> str:
    return f"a value of type {type(value).__name__}"


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _check_number(name: str, spec: Mapping[str, Any], value: Any) -> list[dict[str, str]]:
    field, unit = f"params.{name}", spec["unit"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [_problem("wrong_type", field, f"{name} must be a finite number in {unit}; got {_kind(value)}")]
    if isinstance(value, float) and not math.isfinite(value):  # an int is finite (never convert: a huge one overflows)
        return [_problem("wrong_type", field, f"{name} must be a finite number in {unit}; got {value!r}")]
    if not spec["min"] <= value <= spec["max"]:
        return [_problem(
            "out_of_range", field,
            f"{name} = {_shown(value)} {unit} is outside the supported range {spec['min']!r} to {spec['max']!r} {unit}",
        )]
    return []


def _check_choices(name: str, spec: Mapping[str, Any], value: Any) -> list[dict[str, str]]:
    field, choices = f"params.{name}", spec["choices"]
    listed = ", ".join(choices)
    if not _is_str_list(value):
        return [_problem("wrong_type", field, f"{name} must be a list of choices from: {listed}; got {_kind(value)}")]
    if not value:
        return [_problem("empty_choice", field, f"{name} must hold at least one of: {listed}")]
    problems = [
        _problem("not_a_choice", field, f"{item!r} is not a choice of {name}; choices: {listed}")
        for item in dict.fromkeys(item for item in value if item not in choices)
    ]
    seen: set[str] = set()
    repeated: dict[str, None] = {}
    for item in value:
        if item in seen:
            repeated[item] = None
        seen.add(item)
    problems += [_problem("duplicate_choice", field, f"{name} lists {item!r} more than once") for item in repeated]
    return problems


def _check_choice_capabilities(
    name: str, spec: Mapping[str, Any], params: Mapping[Any, Any], capabilities: frozenset[str]
) -> list[dict[str, str]]:
    requires = spec["requires"]
    given = name in params
    if given:
        if not _is_str_list(params[name]):
            return []  # already reported as wrong_type: what was chosen is unknown
        chosen = [choice for choice in spec["choices"] if choice in params[name]]
    else:
        chosen = spec["default"]
    available = [
        choice for choice in spec["choices"] if requires.get(choice) is None or requires[choice] in capabilities
    ]
    problems = []
    for choice in chosen:
        capability = requires.get(choice)
        if capability is None or capability in capabilities:
            continue
        if given:
            lead = f"{name} includes {choice!r}"
        else:
            lead = f"{name} is not given, so its default ({', '.join(spec['default'])}) applies and includes {choice!r}"
        problems.append(_problem(
            "capability_missing", f"params.{name}",
            f"{lead}, but {_MISSING_CAPABILITY[capability]}; choices available now: {', '.join(available) or 'none'}",
        ))
    return problems


def check_declaration(
    declaration: Any,
    config: Mapping[str, Any],
    *,
    capabilities: Collection[str] | None = None,
    implemented: Collection[str] | None = None,
) -> list[dict[str, str]]:
    """Problems of ``declaration``, each {"code", "field", "message"}; [] means it is valid.

    Checks in this order: an object (else only not_an_object); unknown top-level fields; skill present
    (else stop after missing_field); skill declared (else stop after unknown_skill, saying when it is
    only planned); source_text a str within declared.max_source_text_chars; params a dict; unknown
    params; each declared param in config order (number: int/float, not bool, finite, inside
    [min, max]; choices: a non-empty list of str from the choices, without repeats); the skill's
    required capabilities; each choice's required capability (the default's choices when the param is
    omitted); the skill's spec in ``implemented``.

    Never raises for a bad declaration and never evaluates a string. Raises ConfigError for a malformed
    ``declared`` block and TypeError when capabilities or implemented is not a collection of str.
    """
    if not isinstance(declaration, dict):
        return [_problem("not_an_object", "", f"a declaration must be an object (dict); got {_kind(declaration)}")]
    problems = []
    for key in declaration:
        if key not in _TOP_LEVEL:
            field = key if isinstance(key, str) else _shown(key)
            message = f"unknown field {_shown(key)}; a declaration holds only skill, params and source_text"
            problems.append(_problem("unknown_field", field, message))
    skills = _skills(config)
    supported = ", ".join(skills)
    if "skill" not in declaration:
        message = f"the declaration names no skill; supported skills: {supported}"
        problems.append(_problem("missing_field", "skill", message))
        return problems
    name = declaration["skill"]
    if not isinstance(name, str) or name not in skills:
        if not isinstance(name, str):
            message = f"skill must be the name of a supported skill ({supported}); got {_kind(name)}"
        elif name in _planned(config):
            message = f"{name} is planned in MISSION.md but has no SkillSpec yet; supported skills: {supported}"
        else:
            message = f"unknown skill {name!r}; supported skills: {supported}"
        problems.append(_problem("unknown_skill", "skill", message))
        return problems
    entry = _skill(config, name)

    if "source_text" in declaration:
        text, limit = declaration["source_text"], _max_chars(config)
        if not isinstance(text, str):
            problems.append(_problem("wrong_type", "source_text", f"source_text must be text; got {_kind(text)}"))
        elif len(text) > limit:
            message = f"source_text has {len(text)} characters; at most {limit} are accepted"
            problems.append(_problem("too_long", "source_text", message))

    params = declaration.get("params", {})
    if not isinstance(params, dict):
        problems.append(_problem("wrong_type", "params", f"params must be an object (dict); got {_kind(params)}"))
    else:
        listed = ", ".join(entry["params"]) or "none"
        for key in params:
            if key not in entry["params"]:
                field = f"params.{key if isinstance(key, str) else _shown(key)}"
                message = f"{name} has no parameter {_shown(key)}; its parameters: {listed}"
                problems.append(_problem("unknown_param", field, message))
        for param, spec in entry["params"].items():
            if param in params:
                check = _check_number if spec["type"] == "number" else _check_choices
                problems += check(param, spec, params[param])

    capability_set = _capabilities(capabilities)
    for capability in entry["requires"]:
        if capability not in capability_set:
            problems.append(_problem("capability_missing", "skill", _MISSING_CAPABILITY[capability]))
    if isinstance(params, dict):
        for param, spec in entry["params"].items():
            if spec["type"] == "choices":
                problems += _check_choice_capabilities(param, spec, params, capability_set)

    if entry["spec"] not in _implemented(implemented):
        problems.append(_problem("spec_missing", "skill", _spec_sentence(entry["spec"])))
    return problems


def resolve_declaration(
    declaration: Any,
    config: Mapping[str, Any],
    *,
    capabilities: Collection[str] | None = None,
    implemented: Collection[str] | None = None,
) -> dict[str, Any]:
    """A valid declaration with every parameter filled in.

    Returns {"skill", "spec", "params": {every declared param: the given value or its default; numbers
    as float, choices as a list in the config's choices order}, "actions", "outcomes", "source_text":
    str or None}.

    Raises:
        DeclarationError: :func:`check_declaration` found problems (they are in ``.problems``).
        ConfigError, TypeError: as :func:`check_declaration`.
    """
    problems = check_declaration(declaration, config, capabilities=capabilities, implemented=implemented)
    if problems:
        raise DeclarationError(problems)
    name = declaration["skill"]
    entry = _skill(config, name)
    given = declaration.get("params", {})
    params: dict[str, Any] = {}
    for param, spec in entry["params"].items():
        value = given.get(param, spec["default"])
        if spec["type"] == "number":
            params[param] = float(value)
        else:
            params[param] = [choice for choice in spec["choices"] if choice in value]
    return {
        "skill": name,
        "spec": entry["spec"],
        "params": params,
        "actions": list(entry["actions"]),
        "outcomes": list(entry["outcomes"]),
        "source_text": declaration.get("source_text"),
    }


# --- request text ----------------------------------------------------------------------------------


def _matches(keywords: list[str], words: set[str]) -> bool:
    """True when a keyword k, k + "s" or k + "es" is one of ``words``."""
    return any(keyword in words or keyword + "s" in words or keyword + "es" in words for keyword in keywords)


def _refused(code: str, message: str) -> dict[str, Any]:
    return {"declaration": None, "problems": [_problem(code, "", message)]}


def parse_request(
    text: Any,
    config: Mapping[str, Any],
    *,
    capabilities: Collection[str] | None = None,
    implemented: Collection[str] | None = None,
) -> dict[str, Any]:
    """Build a declaration from request text with keyword rules (no model call).

    Returns {"declaration": dict or None, "problems": [...]}. Blank or non-str text: empty_request.
    The words are the runs of a-z in the lower-cased text. A skill (declared or planned) matches when one
    of its keywords k, k + "s" or k + "es" is a word. No match: unsupported_request; several skills:
    ambiguous_request; one planned skill: planned_skill; all with declaration None. One declared skill:
    a number param with unit "deg" or "m/s" takes the first number written with that unit in the text,
    a choices param the choices named as words (config order); a param not found is omitted, so its
    default applies. The declaration {"skill", "params", "source_text" (stripped, cut to
    declared.max_source_text_chars)} is returned with the problems of :func:`check_declaration`, also
    when there are some.

    Raises ConfigError for a malformed ``declared`` block and TypeError for bad capabilities/implemented.
    """
    skills = _skills(config)
    supported = ", ".join(skills)
    if not isinstance(text, str) or not text.strip():
        return _refused("empty_request", f"the request holds no text; ask for one of the supported skills: {supported}")

    words = set(_WORDS.findall(text.lower()))
    entries = {name: _skill(config, name) for name in skills}
    planned = _planned(config)
    matched = [name for name, entry in entries.items() if _matches(entry["keywords"], words)]
    matched += [name for name, keywords in planned.items() if _matches(keywords, words)]
    names = list(dict.fromkeys(matched))

    if not names:
        described = "; ".join(f"{name} ({entry['description']})" for name, entry in entries.items())
        return _refused("unsupported_request", f"the request names no supported skill; supported skills: {described}")
    if len(names) > 1:
        only_planned = [name for name in names if name not in skills]
        message = f"the request matches several skills ({', '.join(names)}); ask for one skill at a time"
        if only_planned:
            message += f"; only planned in MISSION.md, with no SkillSpec yet: {', '.join(only_planned)}"
        return _refused("ambiguous_request", message)
    name = names[0]
    if name not in skills:
        return _refused(
            "planned_skill", f"{name} is planned in MISSION.md but has no SkillSpec yet; supported skills: {supported}"
        )

    params: dict[str, Any] = {}
    for param, spec in entries[name]["params"].items():
        if spec["type"] == "number":
            pattern = _UNIT_PATTERNS.get(spec["unit"])
            found = pattern.search(text) if pattern is not None else None
            if found is not None:
                params[param] = float(found.group(1))
        else:
            picked = [choice for choice in spec["choices"] if _matches([choice], words)]
            if picked:
                params[param] = picked
    declaration = {"skill": name, "params": params, "source_text": text.strip()[: _max_chars(config)]}
    return {
        "declaration": declaration,
        "problems": check_declaration(declaration, config, capabilities=capabilities, implemented=implemented),
    }
