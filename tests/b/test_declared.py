"""learned_judgment.declared against the real config.yaml, with explicit capabilities and SkillSpec names."""

import copy
import importlib
import math
import sys

import pytest

import learned_judgment.skills as skills_package
from learned_judgment.config import load_config
from learned_judgment.declared import (
    DeclarationError,
    check_declaration,
    declaration_schema,
    parse_request,
    resolve_declaration,
    skill_catalog,
    world_capabilities,
)

ALL = frozenset({"ball", "incline", "punch_proxy"})
SPECS = frozenset({"ramp", "dodge_vision", "move"})
NO_INCLINE = "the simulator has no traversable incline yet (Go2World.reset does not accept incline_deg)"
NO_BALL = "the simulator has no ball"
NO_PUNCH = "the simulator has no punch proxy yet"


@pytest.fixture(scope="module")
def config():
    return load_config()


def check(declaration, config, capabilities=ALL, implemented=SPECS):
    return check_declaration(declaration, config, capabilities=capabilities, implemented=implemented)


def codes(problems):
    return [(problem["code"], problem["field"]) for problem in problems]


def parse(text, config, capabilities=ALL, implemented=SPECS):
    return parse_request(text, config, capabilities=capabilities, implemented=implemented)


# --- catalog and schema ---


def test_catalog_lists_ramp_then_dodge_from_config(config):
    catalog = skill_catalog(config, capabilities=ALL, implemented=SPECS)
    assert [entry["skill"] for entry in catalog] == list(config["declared"]["skills"])  # every declared skill, in config order
    ramp, dodge = catalog[:2]
    declared = config["declared"]["skills"]
    assert ramp["spec"] == "ramp" and dodge["spec"] == "dodge_vision"
    assert ramp["actions"] == declared["RAMP"]["actions"] and dodge["outcomes"] == declared["DODGE"]["outcomes"]
    assert ramp["params"]["incline_deg"] == {
        key: declared["RAMP"]["params"]["incline_deg"][key]
        for key in ("type", "min", "max", "default", "unit", "description")
    }
    assert "requires" not in dodge["params"]["threats"]
    assert dodge["params"]["threats"]["choices"] == ["ball", "punch"]
    assert all(entry["available"] and entry["missing"] == [] for entry in catalog)


def test_catalog_availability_follows_capabilities_and_specs(config):
    ramp, dodge = skill_catalog(config, capabilities={"ball"}, implemented={"dodge_vision"})[:2]
    assert ramp["available"] is False
    assert ramp["missing"] == [NO_INCLINE, "SkillSpec 'ramp' is not implemented yet (learned_judgment/skills/ramp.py)"]
    assert dodge["available"] is True and dodge["missing"] == []  # the punch choice's requirement is not the skill's

    ramp, dodge = skill_catalog(config, capabilities=set(), implemented=SPECS)
    assert ramp["missing"] == [NO_INCLINE] and dodge["missing"] == [NO_BALL]
    assert not ramp["available"] and not dodge["available"]


def test_catalog_does_not_alias_config(config):
    catalog = skill_catalog(config, capabilities=ALL, implemented=SPECS)
    catalog[1]["params"]["threats"]["choices"].append("rock")
    assert config["declared"]["skills"]["DODGE"]["params"]["threats"]["choices"] == ["ball", "punch"]


def test_capability_and_implemented_arguments_must_be_name_collections(config):
    with pytest.raises(TypeError, match="capabilities"):
        skill_catalog(config, capabilities="ball", implemented=SPECS)
    with pytest.raises(TypeError, match="implemented"):
        skill_catalog(config, capabilities=ALL, implemented=[1])


def test_schema(config):
    schema = declaration_schema(config)
    assert list(schema) == ["oneOf"] and len(schema["oneOf"]) == len(config["declared"]["skills"])
    ramp, dodge = schema["oneOf"][:2]
    for variant, name in ((ramp, "RAMP"), (dodge, "DODGE")):
        assert variant["type"] == "object"
        assert variant["additionalProperties"] is False
        assert variant["required"] == ["skill"]
        assert variant["properties"]["skill"] == {"const": name}
        assert variant["properties"]["params"]["additionalProperties"] is False
        assert variant["properties"]["source_text"] == {
            "type": "string", "maxLength": config["declared"]["max_source_text_chars"]}
    incline = ramp["properties"]["params"]["properties"]["incline_deg"]
    assert incline["type"] == "number" and incline["minimum"] == 5.0 and incline["maximum"] == 15.0
    assert incline["description"] == config["declared"]["skills"]["RAMP"]["params"]["incline_deg"]["description"]
    assert set(ramp["properties"]["params"]["properties"]) == {"incline_deg", "nominal_speed_m_s"}
    threats = dodge["properties"]["params"]["properties"]["threats"]
    assert threats["type"] == "array" and threats["items"] == {"enum": ["ball", "punch"]}
    assert threats["minItems"] == 1 and threats["uniqueItems"] is True and "description" in threats


# --- check_declaration ---


def test_valid_declarations(config):
    ramp = {"skill": "RAMP", "params": {"incline_deg": 10.0, "nominal_speed_m_s": 0.6}, "source_text": "cross it"}
    assert check(ramp, config) == []
    assert check({"skill": "RAMP"}, config) == []
    assert check({"skill": "RAMP", "params": {"incline_deg": 5}}, config) == []  # an int is a number; bounds included
    assert check({"skill": "DODGE", "params": {"threats": ["ball"]}}, config) == []
    assert check({"skill": "DODGE", "params": {"threats": ["ball", "punch"]}}, config) == []


@pytest.mark.parametrize("declaration", [None, [], "RAMP", 3])
def test_not_an_object(config, declaration):
    assert codes(check(declaration, config)) == [("not_an_object", "")]


def test_unknown_field(config):
    problems = check({"skill": "RAMP", "command": "go", 3: 1, ("a",): 2}, config)
    assert codes(problems) == [("unknown_field", "command"), ("unknown_field", "3"), ("unknown_field", "('a',)")]


def test_unknown_field_with_an_unprintable_key(config):
    problems = check({"skill": "RAMP", 10**5000: 1}, config)  # Python refuses to print an int this long
    assert codes(problems) == [("unknown_field", "<int that cannot be printed>")]


def test_missing_field_returns_what_was_found(config):
    problems = check({"extra": 1, "params": {"incline_deg": 99}}, config)
    assert codes(problems) == [("unknown_field", "extra"), ("missing_field", "skill")]


def test_unknown_skill(config):
    (problem,) = check({"skill": "SWIM", "params": "not checked"}, config)
    assert (problem["code"], problem["field"]) == ("unknown_skill", "skill")
    assert "RAMP" in problem["message"] and "DODGE" in problem["message"]
    assert "planned" not in problem["message"]

    (problem,) = check({"skill": 5}, config)
    assert problem["code"] == "unknown_skill" and "RAMP, DODGE" in problem["message"]

    assert codes(check({"skill": ["RAMP"]}, config)) == [("unknown_skill", "skill")]


def test_unknown_skill_that_is_planned(config):
    (problem,) = check({"skill": "BRAKE"}, config)
    assert problem["code"] == "unknown_skill"
    assert "planned in MISSION.md" in problem["message"] and "no SkillSpec yet" in problem["message"]
    (problem,) = check({"skill": "brake"}, config)  # compared case-sensitively
    assert problem["code"] == "unknown_skill" and "planned" not in problem["message"]


def test_source_text_wrong_type_and_too_long(config):
    assert codes(check({"skill": "RAMP", "source_text": 7}, config)) == [("wrong_type", "source_text")]
    limit = config["declared"]["max_source_text_chars"]
    assert check({"skill": "RAMP", "source_text": "x" * limit}, config) == []
    assert codes(check({"skill": "RAMP", "source_text": "x" * (limit + 1)}, config)) == [("too_long", "source_text")]


def test_params_wrong_type_skips_param_checks(config):
    assert codes(check({"skill": "RAMP", "params": ["incline_deg", 99]}, config)) == [("wrong_type", "params")]
    assert codes(check({"skill": "RAMP", "params": None}, config)) == [("wrong_type", "params")]


def test_unknown_param(config):
    (problem,) = check({"skill": "RAMP", "params": {"speed": 1.0}}, config)
    assert (problem["code"], problem["field"]) == ("unknown_param", "params.speed")
    assert "incline_deg" in problem["message"] and "nominal_speed_m_s" in problem["message"]
    assert codes(check({"skill": "RAMP", "params": {1: 1.0}}, config)) == [("unknown_param", "params.1")]


@pytest.mark.parametrize("value", ["12", None, [12.0], {"v": 12}, True, False, math.nan, math.inf, -math.inf])
def test_number_wrong_type(config, value):
    problems = check({"skill": "RAMP", "params": {"incline_deg": value}}, config)
    assert codes(problems) == [("wrong_type", "params.incline_deg")]


def test_code_looking_string_is_only_wrong_type(config, capsys):
    declaration = {"skill": "RAMP", "params": {"incline_deg": "__import__('os').system('echo hi')"}}
    assert codes(check(declaration, config)) == [("wrong_type", "params.incline_deg")]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", [40, 4.99, 15.01, -12.0, 10**400, 10**5000],
                         ids=["40", "4.99", "15.01", "-12.0", "1e400", "1e5000"])
def test_number_out_of_range(config, value):
    (problem,) = check({"skill": "RAMP", "params": {"incline_deg": value}}, config)
    assert (problem["code"], problem["field"]) == ("out_of_range", "params.incline_deg")
    assert "5.0 to 15.0 deg" in problem["message"]


def test_out_of_range_names_the_unit(config):
    (problem,) = check({"skill": "RAMP", "params": {"nominal_speed_m_s": 1.2}}, config)
    assert problem["code"] == "out_of_range" and "0.4 to 0.8 m/s" in problem["message"]


@pytest.mark.parametrize("value", ["ball", ("ball",), [1], ["ball", None], None, {"ball": True}])
def test_choices_wrong_type(config, value):
    assert codes(check({"skill": "DODGE", "params": {"threats": value}}, config)) == [("wrong_type", "params.threats")]


def test_choices_empty(config):
    assert codes(check({"skill": "DODGE", "params": {"threats": []}}, config)) == [("empty_choice", "params.threats")]


def test_choices_not_a_choice_once_per_bad_element(config):
    problems = check({"skill": "DODGE", "params": {"threats": ["ball", "kick", "rock"]}}, config)
    assert codes(problems) == [("not_a_choice", "params.threats")] * 2
    assert "'kick'" in problems[0]["message"] and "'rock'" in problems[1]["message"]


def test_choices_duplicate(config):
    problems = check({"skill": "DODGE", "params": {"threats": ["ball", "ball"]}}, config)
    assert codes(problems) == [("duplicate_choice", "params.threats")]


def test_skill_capability_missing(config):
    (problem,) = check({"skill": "RAMP"}, config, capabilities={"ball", "punch_proxy"})
    assert (problem["code"], problem["field"], problem["message"]) == ("capability_missing", "skill", NO_INCLINE)
    (problem,) = check({"skill": "DODGE", "params": {"threats": ["ball"]}}, config, capabilities=set())
    assert (problem["code"], problem["field"], problem["message"]) == ("capability_missing", "skill", NO_BALL)


def test_choice_capability_missing(config):
    (problem,) = check({"skill": "DODGE"}, config, capabilities={"ball"})  # the default holds punch
    assert (problem["code"], problem["field"]) == ("capability_missing", "params.threats")
    assert "'punch'" in problem["message"] and NO_PUNCH in problem["message"]
    assert "choices available now: ball" in problem["message"]

    (problem,) = check({"skill": "DODGE", "params": {"threats": ["ball", "punch"]}}, config, capabilities={"ball"})
    assert (problem["code"], problem["field"]) == ("capability_missing", "params.threats")
    assert check({"skill": "DODGE", "params": {"threats": ["ball"]}}, config, capabilities={"ball"}) == []
    assert check({"skill": "DODGE"}, config, capabilities=ALL) == []


def test_spec_missing(config):
    (problem,) = check({"skill": "RAMP"}, config, implemented={"dodge_vision"})
    assert (problem["code"], problem["field"]) == ("spec_missing", "skill")
    assert problem["message"] == "SkillSpec 'ramp' is not implemented yet (learned_judgment/skills/ramp.py)"


def test_problem_order(config):
    declaration = {"skill": "RAMP", "extra": 1, "source_text": 5, "params": {"bogus": 1, "incline_deg": 40}}
    assert codes(check(declaration, config, capabilities=set(), implemented=set())) == [
        ("unknown_field", "extra"),
        ("wrong_type", "source_text"),
        ("unknown_param", "params.bogus"),
        ("out_of_range", "params.incline_deg"),
        ("capability_missing", "skill"),
        ("spec_missing", "skill"),
    ]


# --- resolve_declaration ---


def test_resolve_fills_defaults(config):
    resolved = resolve_declaration({"skill": "RAMP", "params": {"incline_deg": 10}}, config,
                                   capabilities=ALL, implemented=SPECS)
    declared = config["declared"]["skills"]["RAMP"]
    assert resolved == {
        "skill": "RAMP",
        "spec": "ramp",
        "params": {"incline_deg": 10.0, "nominal_speed_m_s": 0.6},
        "actions": declared["actions"],
        "outcomes": declared["outcomes"],
        "source_text": None,
    }
    assert type(resolved["params"]["incline_deg"]) is float

    resolved = resolve_declaration({"skill": "DODGE", "source_text": "dodge"}, config,
                                   capabilities=ALL, implemented=SPECS)
    assert resolved["params"] == {"threats": ["ball", "punch"]} and resolved["source_text"] == "dodge"
    resolved = resolve_declaration({"skill": "DODGE", "params": {"threats": ["punch", "ball"]}}, config,
                                   capabilities=ALL, implemented=SPECS)
    assert resolved["params"]["threats"] == ["ball", "punch"]  # the config's choices order


def test_resolve_raises_with_problems(config):
    with pytest.raises(DeclarationError) as raised:
        resolve_declaration({"skill": "RAMP", "params": {"incline_deg": 40}}, config,
                            capabilities=set(), implemented=SPECS)
    error = raised.value
    assert isinstance(error, ValueError)
    assert codes(error.problems) == [("out_of_range", "params.incline_deg"), ("capability_missing", "skill")]
    assert str(error) == "; ".join(problem["message"] for problem in error.problems)


# --- parse_request ---


def test_parse_slope_without_numbers(config):
    text = "teach the dog to cross this slope safely"
    result = parse(text, config)
    assert result["declaration"] == {"skill": "RAMP", "params": {}, "source_text": text}
    assert result["problems"] == []


def test_parse_degree_sign(config):
    result = parse("learn to cross this 5° slope without slipping", config)
    assert result["declaration"]["skill"] == "RAMP" and result["declaration"]["params"] == {"incline_deg": 5.0}
    assert result["problems"] == []


def test_parse_degrees_and_speed(config):
    result = parse("walk up a 12 degree hill at 0.5 m/s", config)
    assert result["declaration"]["params"] == {"incline_deg": 12.0, "nominal_speed_m_s": 0.5}
    assert result["problems"] == []


def test_parse_dodge_with_both_threats(config):
    result = parse("dodge an incoming punch or ball only when it is actually coming at the dog", config)
    assert result["declaration"]["skill"] == "DODGE"
    assert result["declaration"]["params"] == {"threats": ["ball", "punch"]}
    assert result["problems"] == []


def test_parse_dodge_the_ball(config):
    result = parse("  dodge the ball  ", config, capabilities={"ball"})
    assert result["declaration"] == {"skill": "DODGE", "params": {"threats": ["ball"]}, "source_text": "dodge the ball"}
    assert result["problems"] == []


def test_parse_ambiguous(config):
    result = parse("kick the ball to the target", config)
    assert result["declaration"] is None
    (problem,) = result["problems"]
    assert problem["code"] == "ambiguous_request"
    for part in ("DODGE", "KICK", "one skill at a time", "only planned"):
        assert part in problem["message"]


def test_parse_planned(config):
    result = parse("brake on the mark", config)
    assert result["declaration"] is None
    (problem,) = result["problems"]
    assert problem["code"] == "planned_skill" and "BRAKE" in problem["message"] and "MISSION.md" in problem["message"]


def test_parse_unsupported(config):
    result = parse("do a backflip", config)
    assert result["declaration"] is None
    (problem,) = result["problems"]
    assert problem["code"] == "unsupported_request"
    for name, entry in config["declared"]["skills"].items():
        assert name in problem["message"] and entry["description"] in problem["message"]


@pytest.mark.parametrize("text", ["", "   ", None, 42])
def test_parse_empty(config, text):
    result = parse(text, config)
    assert result["declaration"] is None and codes(result["problems"]) == [("empty_request", "")]


def test_parse_returns_declaration_with_problems(config):
    text = "cross a 40 degree slope"
    result = parse(text, config)
    assert result["declaration"] == {"skill": "RAMP", "params": {"incline_deg": 40.0}, "source_text": text}
    assert codes(result["problems"]) == [("out_of_range", "params.incline_deg")]


def test_parse_cuts_source_text(config):
    limit = config["declared"]["max_source_text_chars"]
    result = parse("cross the slope " + "x" * (2 * limit), config)
    assert len(result["declaration"]["source_text"]) == limit and result["problems"] == []


# --- the default set of implemented SkillSpecs ---

_FAKE_MODULES = {
    "zz_test_observed": "def build_observed_spec(config):\n    return None\n",
    "zz_test_not_callable": "build_observed_spec = 3\n",
    "zz_test_broken": "raise ImportError('this test module fails to import on purpose')\n",
}


@pytest.fixture
def fake_skill_modules(tmp_path, monkeypatch):
    """Temporary modules on the learned_judgment.skills path, and a registry that names only 'dodge'; all undone after."""
    for name, text in _FAKE_MODULES.items():
        (tmp_path / f"{name}.py").write_text(text, encoding="utf-8")
    monkeypatch.setattr(skills_package, "__path__", [*skills_package.__path__, str(tmp_path)])
    monkeypatch.setattr(skills_package, "available_skills", lambda: ("dodge",))
    importlib.invalidate_caches()
    yield
    for name in _FAKE_MODULES:
        sys.modules.pop(f"{skills_package.__name__}.{name}", None)
        if hasattr(skills_package, name):
            delattr(skills_package, name)


def test_default_implemented_adds_modules_with_build_observed_spec(config, fake_skill_modules):
    changed = copy.deepcopy(config)
    declared = changed["declared"]["skills"]
    declared["RAMP"]["spec"], declared["DODGE"]["spec"] = "zz_test_observed", "zz_test_not_callable"
    ramp, dodge = skill_catalog(changed, capabilities=ALL)[:2]
    assert ramp["available"] is True and ramp["missing"] == []
    assert dodge["missing"] == [
        "SkillSpec 'zz_test_not_callable' is not implemented yet (learned_judgment/skills/zz_test_not_callable.py)"
    ]

    declared["DODGE"]["spec"] = "dodge"  # the registry's names stay in the default set
    assert check_declaration({"skill": "DODGE"}, changed, capabilities=ALL) == []
    declared["RAMP"]["spec"] = "zz_test_broken"  # skipped, not raised
    assert codes(check_declaration({"skill": "RAMP"}, changed, capabilities=ALL)) == [("spec_missing", "skill")]


# --- the real simulator's capabilities ---


def test_world_capabilities_has_the_ball():
    capabilities = world_capabilities()
    assert isinstance(capabilities, frozenset) and "ball" in capabilities
