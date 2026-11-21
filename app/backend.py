"""The one place the viewer server touches Dev B's code, so a change in Dev B's contract changes this file only.

Agreed with Dev B (message log, 2026-11-21 20:55 UTC): one module learned_judgment.declared with
    skill_catalog(config)        -> [{skill, description, params, actions, outcomes, available, missing}]
    declaration_schema(config)   -> JSON Schema of a declaration {skill, params, source_text}
    check_declaration(d, config) -> [{code, field, message}], empty when valid
    resolve_declaration(d, config), parse_request(text, config): not used by the server yet.
Until it imports, the catalogue lists config.yaml's skills, each unavailable, naming what is missing; no parameter,
action or outcome is made up here.
"""
import importlib
import logging

from learned_judgment.skills import available_skills

MODULE = "learned_judgment.declared"
NEEDED = ("skill_catalog", "declaration_schema", "check_declaration")
log = logging.getLogger(__name__)


def declared():
    """(Dev B's declaration module, []) once it imports with everything the server needs, else (None, [what is
    missing, one plain phrase each])."""
    try:
        module = importlib.import_module(MODULE)
    except ImportError as e:
        if e.name == MODULE:
            return None, [f"{MODULE} (Dev B, not pushed yet)"]
        log.warning("%s does not import: %s", MODULE, e)
        return None, [f"{MODULE} (Dev B) does not import: it needs {e.name}"]
    absent = [name for name in NEEDED if not callable(getattr(module, name, None))]
    if absent:
        return None, [f"{MODULE}.{name} (Dev B)" for name in absent]
    return module, []


def skill_catalog(config: dict) -> list:
    """Dev B's catalogue unchanged, or every skill named in config.yaml skills marked unavailable with the reason."""
    module, missing = declared()
    if module is not None:
        return module.skill_catalog(config)
    registered = set(available_skills())  # the existing SkillSpec registry (learned_judgment.skills)
    return [{"skill": name.upper(), "description": "", "params": {}, "actions": [], "outcomes": [],
             "available": False,
             "missing": missing + ([] if name in registered else [f"{name.upper()} SkillSpec (Dev B)"])}
            for name in config["skills"]]
