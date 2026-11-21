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


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load config.yaml as a plain dict."""
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
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
