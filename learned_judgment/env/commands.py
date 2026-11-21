"""Strict validation of a high-level command against the verified command schema (config.yaml,
robot.walking_policy.command_schema)."""
import math

import numpy as np


class CommandError(ValueError):
    """A command the verified schema does not accept; the message names the offending field."""


def validate_command(command: dict, schema: dict) -> np.ndarray:
    """The command as the policy's native vector (each field at its native_index). Every field is required and an
    unknown, non-numeric, non-finite or out-of-range value raises CommandError. Nothing is clipped."""
    fields = schema["fields"]
    unknown = sorted(set(command) - set(fields))
    if unknown:
        raise CommandError(f"unknown command field(s) {unknown}; the schema has {list(fields)}")
    native = np.zeros(len(fields))
    for name, f in fields.items():
        if name not in command:
            raise CommandError(f"missing command field {name!r}")
        try:
            value = float(command[name])
        except (TypeError, ValueError):
            raise CommandError(f"command field {name!r} is not a number: {command[name]!r}") from None
        if not math.isfinite(value):
            raise CommandError(f"command field {name!r} is not finite: {value}")
        if not f["min"] <= value <= f["max"]:
            raise CommandError(f"command field {name!r} = {value} is outside [{f['min']}, {f['max']}]")
        native[f["native_index"]] = value
    return native
