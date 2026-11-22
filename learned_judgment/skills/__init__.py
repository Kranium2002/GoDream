"""SkillSpec registry: build any skill by name, bound to the verified command schema and validated.

A skill is one module in this package that defines ``SPEC_CLASS``: a :class:`SkillSpec`
subclass with a class attribute ``skill_name`` and the constructor
``SPEC_CLASS(config, command_schema, **skill_arguments)``. Adding a skill means adding its
module; nothing here names a skill.

:func:`load_skill` builds the spec with :meth:`CommandSchema.from_config` and runs
:func:`validate_spec` with values read from config.yaml:

    hidden_value_names    keys of world.hidden_effects
    quantile_levels       engine.quantile_levels
    probe_seed            runtime.seed
    probe_situations      engine.spec_validation.probe_situations
    probe_points_per_dim  engine.spec_validation.probe_points_per_dim

The validation probe calls ``spec.sample_situations``, so a skill whose sampler needs
arguments (for example a throw model) must receive them through ``skill_arguments``.

:func:`load_skill_for_world` builds a skill for one world. A skill whose constructor needs
values from the world (declared values, a situation sampler) defines, in its module, the
optional function::

    for_world(config, world) -> SkillSpec

which builds the spec for that world, normally with ``load_skill(name, config,
**skill_arguments)``. :func:`load_skill_for_world` calls it, checks that it returned a
:class:`SkillSpec` of that name and runs :func:`validate_spec` on it with the settings above.
A skill module without ``for_world`` is built with ``load_skill(name, config)``.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Mapping

from learned_judgment.config import ConfigError, require
from learned_judgment.skills.base import CommandSchema, SkillSpec, validate_spec

__all__ = ["available_skills", "has_for_world", "load_skill", "load_skill_for_world"]

_FOR_WORLD = "for_world"  # name of the optional module-level hook of a skill module

_NOT_SKILL_MODULES = frozenset({"base"})


def _skill_classes() -> dict[str, type[SkillSpec]]:
    """skill name -> SkillSpec class, from every module of this package that defines SPEC_CLASS."""
    found: dict[str, type[SkillSpec]] = {}
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_") or module_info.name in _NOT_SKILL_MODULES:
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        spec_class = getattr(module, "SPEC_CLASS", None)
        if spec_class is None:
            continue
        if not (isinstance(spec_class, type) and issubclass(spec_class, SkillSpec)):
            raise TypeError(f"{module.__name__}.SPEC_CLASS must be a SkillSpec subclass; got {spec_class!r}")
        skill_name = getattr(spec_class, "skill_name", None)
        if not isinstance(skill_name, str) or not skill_name:
            raise TypeError(f"{module.__name__}.SPEC_CLASS needs a non-empty class attribute skill_name")
        if skill_name in found:
            raise TypeError(f"two modules register the skill {skill_name!r}")
        found[skill_name] = spec_class
    return found


def available_skills() -> tuple[str, ...]:
    """Sorted names of every registered skill."""
    return tuple(sorted(_skill_classes()))


def _validation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    hidden = require(config, "world.hidden_effects")
    if not isinstance(hidden, Mapping) or not hidden:
        raise ConfigError("config.yaml key 'world.hidden_effects' must be a non-empty mapping of hidden effect names")
    settings: dict[str, Any] = {
        "hidden_value_names": tuple(str(name) for name in hidden),
        "quantile_levels": tuple(float(level) for level in require(config, "engine.quantile_levels")),
    }
    for argument, key in (
        ("probe_seed", "runtime.seed"),
        ("probe_situations", "engine.spec_validation.probe_situations"),
        ("probe_points_per_dim", "engine.spec_validation.probe_points_per_dim"),
    ):
        value = require(config, key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"config.yaml key {key!r} must be an integer; got {value!r}")
        settings[argument] = value
    return settings


def load_skill(name: str, config: Mapping[str, Any], **skill_arguments: Any) -> SkillSpec:
    """Build the skill ``name`` from ``config`` and validate it against the SkillSpec contract.

    Args:
        name:            a member of :func:`available_skills`.
        config:          the parsed config.yaml.
        skill_arguments: passed to the skill's constructor (e.g. a situation sampler and the
                         world's declared values).

    Raises:
        KeyError:                     unknown skill name.
        UnverifiedCommandSchemaError: the walking policy's command schema is not verified.
        ConfigError:                  a required key is missing or null (the message names it).
        SkillSpecError:               the built spec violates the contract.
    """
    classes = _skill_classes()
    if name not in classes:
        raise KeyError(f"unknown skill {name!r}; available skills: {sorted(classes)}")
    settings = _validation_settings(config)
    command_schema = CommandSchema.from_config(config)
    spec = classes[name](config, command_schema, **skill_arguments)
    if spec.name != name:
        raise TypeError(f"{type(spec).__name__} built a spec named {spec.name!r}, not {name!r}")
    validate_spec(spec, command_schema, **settings)
    return spec


def _for_world_hook(name: str) -> Any | None:
    classes = _skill_classes()
    if name not in classes:
        raise KeyError(f"unknown skill {name!r}; available skills: {sorted(classes)}")
    hook = getattr(importlib.import_module(classes[name].__module__), _FOR_WORLD, None)
    if hook is not None and not callable(hook):
        raise TypeError(f"{classes[name].__module__}.{_FOR_WORLD} must be a function (config, world) -> SkillSpec")
    return hook


def has_for_world(name: str) -> bool:
    """True when the module of skill ``name`` defines the optional ``for_world`` hook."""
    return _for_world_hook(name) is not None


def load_skill_for_world(name: str, config: Mapping[str, Any], world: Any) -> SkillSpec:
    """Build the skill ``name`` for ``world`` and validate it.

    When the skill's module defines ``for_world(config, world)``, the spec it returns is
    checked (a :class:`SkillSpec` named ``name``) and validated with :func:`validate_spec`
    against the verified command schema and the config settings of :func:`load_skill`;
    otherwise this is ``load_skill(name, config)``. Raises what :func:`load_skill` and the hook
    raise, KeyError for an unknown skill, and TypeError when the hook returns something else.
    """
    hook = _for_world_hook(name)
    if hook is None:
        return load_skill(name, config)
    spec = hook(config, world)
    if not isinstance(spec, SkillSpec):
        raise TypeError(f"{_FOR_WORLD} of skill {name!r} must return a SkillSpec; got {type(spec).__name__}")
    if spec.name != name:
        raise TypeError(f"{_FOR_WORLD} of skill {name!r} built a spec named {spec.name!r}")
    validate_spec(spec, CommandSchema.from_config(config), **_validation_settings(config))
    return spec
