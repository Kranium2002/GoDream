"""The command schema: strict validation (pure-function tests on the published schema) and the proof that every
published range end moves the real simulated dog the commanded way without a fall (system test)."""
import json

import pytest

from learned_judgment.env import CommandError, validate_command
from learned_judgment.env.acceptance import prove_commands


def test_edges_are_accepted_unchanged(schema):
    for name, f in schema["fields"].items():
        cmd = {n: 0.0 for n in schema["fields"]} | {name: f["max"]}
        assert validate_command(cmd, schema)[f["native_index"]] == f["max"]

@pytest.mark.parametrize("mutate", [
    lambda c, s: c | {"jump": 1.0},
    lambda c, s: {k: v for k, v in list(c.items())[1:]},
    lambda c, s: c | {next(iter(c)): float("nan")},
    lambda c, s: c | {next(iter(c)): s["fields"][next(iter(c))]["max"] + 1e-6},
])
def test_rejects_instead_of_clipping(schema, mutate):
    zero = {n: 0.0 for n in schema["fields"]}
    with pytest.raises(CommandError):
        validate_command(mutate(zero, schema), schema)

@pytest.mark.system
def test_every_field_moves_the_dog_the_signed_way(config, tmp_path):
    proof = json.loads(prove_commands(config, tmp_path).read_text())
    for name, f in config["robot"]["walking_policy"]["command_schema"]["fields"].items():
        assert proof[name]["at_max"]["measured"] > 0 > proof[name]["at_min"]["measured"]
        assert not proof[name]["at_max"]["fallen"] and not proof[name]["at_min"]["fallen"]
