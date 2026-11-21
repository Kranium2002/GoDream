"""The skill catalogue adapter (app/backend.py) and the chat's pure parts (app/chat.py): message checks, prompt
assembly and reply parsing. The catalogues and replies written here are fixtures for these pure functions only; the
live Claude call is the last test and runs only with a key and Dev B's declaration module."""
import json
import logging
import types

import pytest

from app import backend, chat
from app.routes import Refusal
from learned_judgment.env import load_config

SENTINEL_KEY = "sk-ant-test-SENTINEL-0123456789"  # not a real key: only to see whether it leaks anywhere


@pytest.fixture(scope="module")
def config():
    return load_config()


CATALOGUE = [  # fixture, in skill_catalog's agreed shape
    {"skill": "RAMP", "description": "Cross an incline.", "available": True, "missing": [],
     "params": {"incline_deg": {"type": "float", "min": 0.0, "max": 15.0, "choices": None, "default": 5.0,
                                "unit": "deg"}},
     "actions": ["approach_speed"], "outcomes": ["fall"]},
    {"skill": "KICK", "description": "Kick a ball.", "available": False, "missing": ["KICK SkillSpec (Dev B)"],
     "params": {}, "actions": [], "outcomes": []},
]
SCHEMA = {"type": "object", "properties": {"skill": {"type": "string"}, "params": {"type": "object"},
                                           "source_text": {"type": "string"}},
          "required": ["skill", "params", "source_text"], "additionalProperties": False}


def reply(declaration, text="Here is the skill.", unsupported=()):
    return json.dumps({"reply": text, "declaration": declaration, "unsupported": list(unsupported)})


RAMP = {"skill": "RAMP", "params": {"incline_deg": 5.0}, "source_text": "cross a 5 degree slope"}


# ---- the catalogue -----------------------------------------------------------------------------------------------

def test_catalogue_comes_from_dev_b_or_marks_every_skill_unavailable(config):
    declared, missing = backend.declared()
    catalogue = backend.skill_catalog(config)
    if declared is not None:
        assert catalogue == declared.skill_catalog(config)  # served unchanged
        return
    assert missing == ["learned_judgment.declared (Dev B, not pushed yet)"]
    assert {entry["skill"] for entry in catalogue} == {name.upper() for name in config["skills"]}
    for entry in catalogue:
        assert set(entry) == {"skill", "description", "params", "actions", "outcomes", "available", "missing"}
        assert entry["available"] is False and missing[0] in entry["missing"]
        assert entry["params"] == {} and entry["actions"] == [] and entry["outcomes"] == []  # nothing invented


def test_catalogue_names_skills_without_a_registered_skillspec(config):
    if backend.declared()[0] is not None:
        pytest.skip("Dev B's catalogue is served unchanged")
    by_name = {entry["skill"]: entry for entry in backend.skill_catalog(config)}
    assert not any("SkillSpec" in m for m in by_name["DODGE"]["missing"])  # learned_judgment.skills has dodge
    assert any("SkillSpec" in m for m in by_name["RAMP"]["missing"])  # and no ramp


# ---- messages from the browser -------------------------------------------------------------------------------------

@pytest.mark.parametrize("body", [
    {}, {"messages": "hi"}, {"messages": []}, {"messages": [{"role": "system", "content": "x"}]},
    {"messages": [{"role": "user", "content": 3}]}, {"messages": [{"role": "user", "content": "  "}]},
    {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
    {"messages": [{"role": "user", "content": "a", "extra": 1}]}, {"messages": [{"role": "user", "content": "a"}],
                                                                  "model": "x"},
], ids=["none", "not_a_list", "empty", "system_role", "not_text", "blank", "last_not_user", "extra_key", "extra_field"])
def test_invalid_chat_messages_are_refused(config, body):
    with pytest.raises(Refusal) as refused:
        chat.check_messages(body, config["app"]["chat"])
    assert refused.value.status == 400 and refused.value.code == "invalid_messages"


def test_chat_message_limits_come_from_config(config):
    limits = config["app"]["chat"]
    long = [{"role": "user", "content": "x" * (limits["max_message_chars"] + 1)}]
    many = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x"} for i in range(limits["max_messages"] + 1)]
    for messages in [long, many]:
        with pytest.raises(Refusal):
            chat.check_messages({"messages": messages}, limits)
    ok = [{"role": "user", "content": "teach it to cross this slope safely"}]
    assert chat.check_messages({"messages": ok}, limits) == ok


# ---- prompt assembly -----------------------------------------------------------------------------------------------

def test_request_carries_the_catalogue_the_schema_and_the_config(config):
    settings = config["app"]["chat"]
    messages = [{"role": "user", "content": "teach it to cross this slope safely"}]
    request = chat.build_request(messages, CATALOGUE, SCHEMA, settings)
    assert request["model"] == settings["model"] and request["max_tokens"] == settings["max_tokens"]
    assert request["messages"] == messages
    assert request["output_config"]["effort"] == settings["effort"]
    out = request["output_config"]["format"]
    assert out["type"] == "json_schema"
    assert out["schema"]["properties"]["declaration"] == {"anyOf": [SCHEMA, {"type": "null"}]}
    assert set(out["schema"]["required"]) == {"reply", "declaration", "unsupported"}
    system = request["system"]
    for entry in CATALOGUE:  # every skill, its parameters, and what an unavailable one lacks
        assert entry["skill"] in system and entry["description"] in system
    assert "incline_deg" in system and "KICK SkillSpec (Dev B)" in system
    assert "data" in system.lower()  # the model is told it returns data only


UNSUPPORTED = {"oneOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "uniqueItems",
               "maxItems", "minLength", "maxLength"}  # structured outputs reject these with a 400 (docs, 2026-11-21)


def keys_in(schema):
    if isinstance(schema, dict):
        return set(schema) | set().union(*(keys_in(v) for v in schema.values()))
    if isinstance(schema, list):
        return set().union(*(keys_in(v) for v in schema)) if schema else set()
    return set()


def test_the_declaration_schema_is_made_acceptable_to_structured_outputs(config):
    declared, _ = backend.declared()
    if declared is None:
        pytest.skip("needs Dev B's declaration_schema")
    original = declared.declaration_schema(config)
    sent = chat.model_schema(original)
    assert not keys_in(sent) & UNSUPPORTED and "anyOf" in sent
    assert all(item.get("minItems", 0) <= 1 for item in [sent] if isinstance(item, dict))
    text = json.dumps(sent)
    assert "5.0" in text and "15.0" in text  # the RAMP range survives as description text for the model
    assert declared.declaration_schema(config) == original  # Dev B's schema object is not changed


def test_model_schema_moves_constraints_into_descriptions():
    sent = chat.model_schema({"oneOf": [{"type": "object", "properties": {
        "a": {"type": "number", "minimum": 1, "maximum": 2, "description": "A."},
        "b": {"type": "array", "items": {"enum": ["x", "y"]}, "minItems": 2, "uniqueItems": True}}}]})
    a, b = sent["anyOf"][0]["properties"]["a"], sent["anyOf"][0]["properties"]["b"]
    assert a == {"type": "number", "description": "A. (minimum 1, maximum 2)"}
    assert b == {"type": "array", "items": {"enum": ["x", "y"]}, "minItems": 1,
                 "description": "(minItems 2, uniqueItems true)"}


# ---- reply parsing -------------------------------------------------------------------------------------------------

def no_problems(declaration, config):
    return []


def test_a_valid_declaration_of_an_available_skill_is_returned(config):
    result = chat.parse_reply(reply(RAMP), CATALOGUE, no_problems, config)
    assert result == {"reply": "Here is the skill.", "declared_spec": RAMP, "problems": [], "missing": []}


def test_problems_from_check_declaration_withhold_the_spec(config):
    problem = {"code": "out_of_range", "field": "params.incline_deg", "message": "Too steep."}
    result = chat.parse_reply(reply(RAMP), CATALOGUE, lambda d, c: [problem], config)
    assert result["declared_spec"] is None and result["problems"] == [problem]


def test_an_unavailable_skill_names_what_it_lacks(config):
    kick = {"skill": "KICK", "params": {}, "source_text": "kick the ball"}
    result = chat.parse_reply(reply(kick), CATALOGUE, no_problems, config)
    assert result["declared_spec"] is None
    assert [p["code"] for p in result["problems"]] == ["skill_unavailable"]
    assert result["missing"] == ["KICK SkillSpec (Dev B)"]


def test_a_skill_outside_the_catalogue_is_refused(config):
    result = chat.parse_reply(reply({"skill": "BACKFLIP", "params": {}, "source_text": "x"}), CATALOGUE,
                              no_problems, config)
    assert result["declared_spec"] is None and [p["code"] for p in result["problems"]] == ["unknown_skill"]


def test_no_declaration_reports_the_unsupported_request(config):
    result = chat.parse_reply(reply(None, "No skill does that.", ["a backflip"]), CATALOGUE, no_problems, config)
    assert result["declared_spec"] is None and result["problems"] == []
    assert result["missing"] == ["no registered skill for: a backflip"]


@pytest.mark.parametrize("text", ["not json", "[]", json.dumps({"reply": 3, "declaration": None, "unsupported": []}),
                                  json.dumps({"reply": "x", "declaration": "RAMP", "unsupported": []}),
                                  json.dumps({"reply": "x", "declaration": None})])
def test_a_malformed_model_reply_is_a_problem_not_a_crash(config, text):
    result = chat.parse_reply(text, CATALOGUE, no_problems, config)
    assert result["declared_spec"] is None and [p["code"] for p in result["problems"]] == ["invalid_model_reply"]


# ---- the key -------------------------------------------------------------------------------------------------------

def test_the_key_comes_from_the_environment_or_the_env_file(config, tmp_path, monkeypatch):
    settings = dict(config["app"]["chat"])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    assert chat.api_key(settings) is None  # no file, no variable
    (tmp_path / settings["env_file"]).write_text(f"ANTHROPIC_API_KEY={SENTINEL_KEY}\n")
    assert chat.api_key(settings) == SENTINEL_KEY
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY + "-env")
    assert chat.api_key(settings) == SENTINEL_KEY + "-env"


def test_a_failed_model_call_leaks_the_key_nowhere(config, caplog):
    """Even an error whose own text carries the key gives a plain refusal and log lines without it."""
    stand_in = types.SimpleNamespace(skill_catalog=lambda c: CATALOGUE, declaration_schema=lambda c: SCHEMA,
                                     check_declaration=no_problems)  # Dev B's module, for this unit test only

    def failing_call(key, request, settings):
        raise RuntimeError(f"connection refused for key {key}")

    messages = [{"role": "user", "content": "teach it to cross this slope safely"}]
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Refusal) as refused:
            chat.run_chat(messages, config, SENTINEL_KEY, stand_in, failing_call)
    assert refused.value.status == 502 and refused.value.code == "chat_failed"
    assert SENTINEL_KEY not in refused.value.message
    assert caplog.records and not any(SENTINEL_KEY in r.getMessage() or SENTINEL_KEY in (r.exc_text or "")
                                      for r in caplog.records)
    assert all(r.exc_info is None for r in caplog.records)  # no traceback, whose frames could show the key


# ---- the live call -------------------------------------------------------------------------------------------------

def test_live_chat_turns_a_slope_request_into_a_ramp_declaration(config):
    key = chat.api_key(config["app"]["chat"])
    declared, missing = backend.declared()
    if not key:
        pytest.skip("NOT RUN: no ANTHROPIC_API_KEY in the environment or .env at the repository root")
    if declared is None:
        pytest.skip(f"NOT RUN: missing {missing}")
    result = chat.run_chat([{"role": "user", "content": "Teach the dog to cross this 5 degree slope safely."}],
                           config, key, declared)
    assert result["reply"]
    assert result["declared_spec"] is None or result["declared_spec"]["skill"] == "RAMP"
