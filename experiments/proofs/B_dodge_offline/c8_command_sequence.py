"""Check 8: DODGE reset_situation() and commands() are what episode() sends.

FIXTURE CHECK - mechanics only. CallRecorder below is NOT a simulator, NOT a world and NOT
Go2World: it only records the reset / step / finish_episode calls it receives and returns
fixed dictionaries written in this script. The check proves that an episode replayed from
``spec.reset_situation`` and ``spec.commands`` (the dashboard's recorder, the live viewer)
sends the world exactly what ``spec.episode`` sends, and that ``validate_spec`` checks the
exposed command sequence. No outcome printed here is a measurement.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_offline\\c8_command_sequence.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import numpy as np  # noqa: E402

from fixture_dodge import FIXTURE_DECLARED, Checks, fixture_config, fixture_throw_sampler  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.skills import load_skill  # noqa: E402
from learned_judgment.skills.base import (  # noqa: E402
    CommandSchema,
    SkillSpec,
    SkillSpecError,
    exposes_command_sequence,
    validate_spec,
)
from learned_judgment.skills.dodge import BALL_DOG_KEYS  # noqa: E402


class CallRecorder:
    """RECORDER, not a simulator: records calls and returns the fixed dictionaries it was given."""

    def __init__(self, observation: dict, outcome: dict) -> None:
        self.observation = observation
        self.outcome = outcome
        self.calls: list[tuple] = []

    def reset(self, situation, attempt_seed):  # noqa: ANN001, ANN201
        self.calls.append(("reset", dict(situation), attempt_seed))
        return dict(self.observation)

    def step(self, command, duration_s):  # noqa: ANN001, ANN201
        self.calls.append(("step", dict(command), duration_s))
        return {"recorder": True}

    def finish_episode(self):  # noqa: ANN201
        self.calls.append(("finish_episode",))
        return dict(self.outcome)


print("CallRecorder: a call recorder that returns fixed dictionaries; it is not a simulator")

checks = Checks()
real_config = load_config()
config = fixture_config(real_config)
schema = CommandSchema.from_config(config)
spec = load_skill("dodge", config, throw_sampler=fixture_throw_sampler(config), **FIXTURE_DECLARED)
OUTCOME = {"min_separation_m": 0.4, "collision": 0, "fall": 0, "recovery_time_s": 0.3}  # recorder's fixed reply

checks.check("the DODGE spec exposes its reset request and command sequence", exposes_command_sequence(spec))

rng = np.random.default_rng(8)
situations = spec.sample_situations(rng, 6)
grid_points = require(config, "skills.dodge.candidate_points_per_dim")
n_compared = 0
all_equal = True
all_fields = True
total_ok = True
for row in situations:
    situation = spec.decode_situation(row)
    for action_row in spec.candidate_actions(row, grid_points):
        action = spec.decode_action(action_row)
        recorder = CallRecorder(dict(FIXTURE_DECLARED), dict(OUTCOME))
        spec.episode(recorder, situation, action, 3)
        sent_reset = recorder.calls[0]
        sent_steps = [(call[1], call[2]) for call in recorder.calls if call[0] == "step"]
        request = spec.reset_situation(situation)
        sequence = spec.commands(situation, action)
        n_compared += 1
        all_equal &= sent_reset == ("reset", request, 3) and sent_steps == sequence
        all_fields &= all(set(command) == set(schema.fields) for command, _ in sequence)
        total_ok &= abs(sum(duration for _, duration in sequence) - spec.episode_duration_s) < 1e-12
checks.check(
    f"episode() sent exactly reset_situation() and commands() on {n_compared} situation x candidate pairs", all_equal
)
checks.check("every command names exactly the verified fields", all_fields)
checks.check("the durations of every sequence add up to the episode duration", total_ok)

situation = spec.decode_situation(situations[0])
request = spec.reset_situation(situation)
checks.check("the reset request holds the seven ball and dog keys and no declared key", tuple(request) == BALL_DOG_KEYS)

first = spec.commands(situation, spec.decode_action(spec.candidate_actions(situations[0], grid_points)[0]))
first[0][0]["vel_x"] = 123.0
again = spec.commands(situation, spec.decode_action(spec.candidate_actions(situations[0], grid_points)[0]))
checks.check("the returned commands are copies: changing one does not change the next call", again[0][0]["vel_x"] != 123.0)

settings = {
    "hidden_value_names": tuple(require(real_config, "world.hidden_effects")),
    "quantile_levels": require(real_config, "engine.quantile_levels"),
    "probe_seed": int(require(real_config, "runtime.seed")),
    "probe_situations": int(require(real_config, "engine.spec_validation.probe_situations")),
    "probe_points_per_dim": int(require(real_config, "engine.spec_validation.probe_points_per_dim")),
}
validate_spec(spec, schema, **settings)
checks.check("validate_spec passes with the exposed command sequence", True)


def broken(kind: type, **overrides):  # noqa: ANN001, ANN201
    """A DODGE spec whose class overrides one of the two methods (for the refusal cases below)."""
    built = load_skill("dodge", config, throw_sampler=fixture_throw_sampler(config), **FIXTURE_DECLARED)
    built.__class__ = type("Broken" + kind.__name__, (kind,), overrides)
    return built


def refused(label: str, fragment: str, **overrides) -> None:  # noqa: ANN001
    checks.raises(
        label, SkillSpecError, lambda: validate_spec(broken(type(spec), **overrides), schema, **settings), fragment
    )


refused(
    "validate_spec refuses a command outside the verified range",
    "outside the verified range",
    commands=lambda self, situation, action: [({"vel_x": 0.0, "vel_y": 5.0, "yaw_rate": 0.0}, 0.2)],
)
refused(
    "validate_spec refuses a command that leaves out a verified field",
    "must name exactly the verified fields",
    commands=lambda self, situation, action: [({"vel_x": 0.0, "vel_y": 0.5}, 0.2)],
)
refused(
    "validate_spec refuses a duration of zero",
    "duration_s must be a finite number > 0",
    commands=lambda self, situation, action: [({"vel_x": 0.0, "vel_y": 0.5, "yaw_rate": 0.0}, 0.0)],
)
refused(
    "validate_spec refuses an empty sequence",
    "non-empty list",
    commands=lambda self, situation, action: [],
)
refused(
    "validate_spec refuses a spec that exposes commands without the reset request",
    "must be implemented together",
    reset_situation=SkillSpec.reset_situation,
)

checks.finish()
