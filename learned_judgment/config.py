"""Access to config.yaml, the single home of every tunable.

Code never falls back to a built-in default: :func:`require` raises :class:`ConfigError`
when a key is missing or still ``null``, naming the key, so an unset tunable stops the run
instead of being silently replaced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["CONFIG_PATH", "ConfigError", "config_digest", "load_config", "optional", "require"]

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_MISSING = object()


class ConfigError(LookupError):
    """A required config.yaml key is missing, null, or has the wrong shape."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses a mapping with a repeated key (plain YAML keeps the last)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConfigError(f"config.yaml repeats the key {key!r} at line {key_node.start_mark.line + 1}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load config.yaml as a plain dict. A repeated key in any mapping is an error."""
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=_UniqueKeyLoader)  # noqa: S506 - a SafeLoader subclass
    if not isinstance(config, dict):
        raise ConfigError(f"{path} does not contain a mapping at the top level")
    return config


def _lookup(config: Mapping[str, Any], dotted_key: str) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Return ``config[a][b][c]`` for ``"a.b.c"``; raise if it is missing or null."""
    value = _lookup(config, dotted_key)
    if value is _MISSING:
        raise ConfigError(f"config.yaml has no key {dotted_key!r}")
    if value is None:
        raise ConfigError(f"config.yaml key {dotted_key!r} is null: it must be set before this step can run")
    return value


def optional(config: Mapping[str, Any], dotted_key: str) -> Any | None:
    """Return the value, or ``None`` when the key is missing or null.

    Only for keys whose absence is itself meaningful (for example "no world-id feature");
    never use it to hide a missing tunable behind a default.
    """
    value = _lookup(config, dotted_key)
    return None if value is _MISSING else value


def config_digest(config: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``config``: the config ID recorded with every run."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
