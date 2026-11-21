"""Check 5: DODGE episode() against a call recorder.

FIXTURE CHECK - mechanics only. CallRecorder below is NOT a simulator, NOT a world and NOT
Go2World: it only records the reset / step / finish_episode calls it receives and returns
fixed dictionaries written in this script. The check proves the order, durations and content
of the commands the SkillSpec sends, and that bad input is refused before the world is
stepped. No outcome printed here is a measurement.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_offline\\c5_episode_recorder.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402
import math  # noqa: E402

from fixture_dodge import FIXTURE_DECLARED, Checks, fixture_config, fixture_throw_sampler  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.skills import load_skill  # noqa: E402
from learned_judgment.skills.base import CommandField, SkillSpecError  # noqa: E402
from learned_judgment.skills.dodge import BALL_DOG_KEYS  # noqa: E402


class CallRecorder:
    """RECORDER, not a simulator: records calls and returns the fixed dictionaries it was given."""

    def __init__(self, observation: dict, outcome: dict) -> None:
        self.observation = observation
        self.outcome = outcome
        self.calls: list[tuple] = []

    def reset(self, situation, attempt_seed):  # noqa: ANN001, ANN201
        self.calls.append(("reset", situation, attempt_seed))
        return dict(self.observation)

    def step(self, command, duration_s):  # noqa: ANN001, ANN201
        self.calls.append(("step", command, duration_s))
        return {"recorder": True}

    def finish_episode(self):  # noqa: ANN201
        self.calls.append(("finish_episode",))
        return dict(self.outcome)


print("CallRecorder: a call recorder that returns fixed dictionaries; it is not a simulator")

checks = Checks()
config = fixture_config(load_config())
spec = load_skill("dodge", config, throw_sampler=fixture_throw_sampler(config), **FIXTURE_DECLARED)
DURATION = spec.episode_duration_s

SITUATION = {
    "ball_rel_x_m": 4.0,
    "ball_rel_y_m": 0.2,
    "ball_rel_z_m": 0.3,
    "ball_vel_x_m_s": -5.0,
    "ball_vel_y_m_s": -0.2,
    "ball_vel_z_m_s": 3.0,
    "dog_forward_speed_m_s": 0.4,
    **FIXTURE_DECLARED,
}
FIXED_OUTCOME = {  # fixed recorder output, not a measurement
    "min_separation_m": 0.12,
    "collision": 0,
    "fall": False,
    "recovery_time_s": 0.0,
    "recovered": 1,
    "fall_time_s": None,
    "trace_path": "not-a-trace",
    "commands_received": 80,
}
OBSERVATION = {"t_s": 0.0, **FIXTURE_DECLARED}


def run(action, *, situation=SITUATION, seed=17, observation=OBSERVATION, outcome=FIXED_OUTCOME, spec_=spec):  # noqa: ANN001, ANN201
    recorder = CallRecorder(observation, outcome)
    result = spec_.episode(recorder, dict(situation), dict(action), seed)
    return recorder, result


def nominal(forward):  # noqa: ANN001, ANN201
    return {"vel_x": forward, "vel_y": 0.0, "yaw_rate": 0.0}


print("\n== the command sequence ==")
action = {"side_velocity_m_s": 0.5, "start_delay_s": 0.2, "hold_duration_s": 0.6}
recorder, result = run(action)
for call in recorder.calls:
    print(f"  {call}")
names = [call[0] for call in recorder.calls]
checks.check("calls: reset, step, step, step, finish_episode", names == ["reset", "step", "step", "step", "finish_episode"])
_, request, seed = recorder.calls[0]
checks.check("reset gets exactly the seven ball/dog keys, in order", tuple(request) == BALL_DOG_KEYS, f"{tuple(request)}")
checks.check("reset gets the situation's values as floats", all(type(request[k]) is float and request[k] == SITUATION[k] for k in BALL_DOG_KEYS))
checks.check("declared values are not requested from the world", not set(request) & {"declared_payload_kg", "declared_surface"})
checks.check("attempt_seed passed to reset", seed == 17 and type(seed) is int)
steps = [call for call in recorder.calls if call[0] == "step"]
checks.check("first step: nominal command for the start delay", steps[0][1] == nominal(0.4) and steps[0][2] == 0.2)
checks.check(
    "second step: dodge command for the hold",
    steps[1][1] == {"vel_x": 0.4, "vel_y": 0.5, "yaw_rate": 0.0} and steps[1][2] == 0.6,
)
checks.check("third step: nominal command for the rest", steps[2][1] == nominal(0.4) and math.isclose(steps[2][2], DURATION - 0.8, abs_tol=1e-12))
checks.check("durations add up to the episode duration", math.isclose(sum(s[2] for s in steps), DURATION, abs_tol=1e-12))
checks.check(
    "every command has the three verified fields, each a Python float",
    all(set(s[1]) == {"vel_x", "vel_y", "yaw_rate"} and all(type(v) is float for v in s[1].values()) for s in steps),
)
checks.check("every duration is a Python float", all(type(s[2]) is float for s in steps))

print("\n== the returned outcome ==")
print(f"  {result}")
checks.check("exactly the five outcome keys, in outcome order", tuple(result) == spec.outcome_names)
checks.check("the four world keys are read from finish_episode", [result[k] for k in spec.outcome_names[:4]] == [0.12, 0.0, 0.0, 0.0])
checks.check("lateral_command_m = |side| x hold of the commands issued", result["lateral_command_m"] == abs(0.5) * 0.6)
checks.check("every value is a Python float", all(type(v) is float for v in result.values()))

print("\n== zero start delay, and nothing left after the hold ==")
recorder, result = run({"side_velocity_m_s": -1.0, "start_delay_s": 0.0, "hold_duration_s": 1.0})
steps = [call for call in recorder.calls if call[0] == "step"]
checks.check("zero delay skips the first step", len(steps) == 2 and steps[0][1]["vel_y"] == -1.0 and steps[0][2] == 1.0)
checks.check("then nominal for the rest", steps[1][1] == nominal(0.4) and math.isclose(steps[1][2], DURATION - 1.0, abs_tol=1e-12))
checks.check("negative side: lateral_command_m uses |side|", result["lateral_command_m"] == 1.0)
recorder, _ = run({"side_velocity_m_s": 1.0, "start_delay_s": 0.6, "hold_duration_s": 1.0})
steps = [call for call in recorder.calls if call[0] == "step"]
checks.check("delay 0.6 + hold 1.0 = the episode: no zero-length third step", len(steps) == 2, f"durations {[s[2] for s in steps]}")
grid = spec.candidate_actions(spec.encode_situation(SITUATION), spec.candidate_points_per_dim)
for row in grid:
    recorder, _ = run(spec.decode_action(row))
    durations = [call[2] for call in recorder.calls if call[0] == "step"]
    if not (math.isclose(sum(durations), DURATION, abs_tol=1e-12) and all(d > 0 for d in durations)):
        break
checks.check("all 108 grid actions: positive durations adding up to the episode", math.isclose(sum(durations), DURATION, abs_tol=1e-12) and all(d > 0 for d in durations))

print("\n== refused before the world is stepped ==")
recorder = CallRecorder(OBSERVATION, FIXED_OUTCOME)
checks.raises(
    "side velocity 1.5 (outside the verified vel_y range)",
    ValueError,
    lambda: spec.episode(recorder, dict(SITUATION), {**action, "side_velocity_m_s": 1.5}, 1),
    "side_velocity_m_s=1.5 is outside [-1.0, 1.0]",
)
checks.check("  ... no world call", recorder.calls == [])

narrow_y = dataclasses.replace(spec.command_schema, fields={**spec.command_schema.fields, "vel_y": CommandField("vel_y", "m/s", -0.5, 0.5)})
narrowed = load_skill("dodge", config, throw_sampler=fixture_throw_sampler(config), **FIXTURE_DECLARED)
narrowed.command_schema = narrow_y  # the schema changed after the spec was built
recorder = CallRecorder(OBSERVATION, FIXED_OUTCOME)
checks.raises(
    "schema narrowed after build: action range exceeds the verified vel_y range",
    SkillSpecError,
    lambda: narrowed.episode(recorder, dict(SITUATION), action, 1),
    "exceeds the verified range",
)
checks.check("  ... no world call", recorder.calls == [])

narrow_x = dataclasses.replace(spec.command_schema, fields={**spec.command_schema.fields, "vel_x": CommandField("vel_x", "m/s", -0.3, 0.3)})
narrowed.command_schema = narrow_x
recorder = CallRecorder(OBSERVATION, FIXED_OUTCOME)
checks.raises(
    "forward command 0.4 outside a verified vel_x range of [-0.3, 0.3]",
    ValueError,
    lambda: narrowed.episode(recorder, dict(SITUATION), action, 1),
    "command vel_x=0.4 m/s is outside the verified range",
)
checks.check("  ... no world call", recorder.calls == [])

for label, situation, fragment in (
    ("a hidden value in the situation", {**SITUATION, "payload_mass_kg": 2.5}, "extra=['payload_mass_kg']"),
    ("a missing situation key", {k: v for k, v in SITUATION.items() if k != "ball_rel_z_m"}, "missing=['ball_rel_z_m']"),
    ("forward speed outside its declared range", {**SITUATION, "dog_forward_speed_m_s": 0.9}, "dog_forward_speed_m_s=0.9 is outside"),
):
    recorder = CallRecorder(OBSERVATION, FIXED_OUTCOME)
    checks.raises(label, ValueError, lambda situation=situation, recorder=recorder: spec.episode(recorder, situation, action, 1), fragment)
    checks.check("  ... no world call", recorder.calls == [])
for bad_seed in (-1, True, 1.5):
    recorder = CallRecorder(OBSERVATION, FIXED_OUTCOME)
    checks.raises(f"attempt_seed {bad_seed!r}", ValueError, lambda s=bad_seed, r=recorder: spec.episode(r, dict(SITUATION), action, s), "attempt_seed must be an int >= 0")
    checks.check("  ... no world call", recorder.calls == [])

print("\n== declared values come from the world ==")
for label, observation, fragment in (
    ("the world declares another payload", {**OBSERVATION, "declared_payload_kg": 2.0}, "declares declared_payload_kg=2.0"),
    ("the world declares another surface", {**OBSERVATION, "declared_surface": "fixture_surface_b"}, "declares declared_surface='fixture_surface_b'"),
):
    recorder = CallRecorder(observation, FIXED_OUTCOME)
    checks.raises(label, ValueError, lambda o=observation, r=recorder: spec.episode(r, dict(SITUATION), action, 1), fragment)
    checks.check("  ... only reset was called, no step", [c[0] for c in recorder.calls] == ["reset"])
recorder, _ = run(action, observation={"t_s": 0.0})
checks.check("an observation without declared keys is accepted", [c[0] for c in recorder.calls][-1] == "finish_episode")

print("\n== the world's outcome is checked ==")
for label, outcome, fragment in (
    ("outcome without recovery_time_s", {k: v for k, v in FIXED_OUTCOME.items() if k != "recovery_time_s"}, "lacks ['recovery_time_s']"),
    ("collision 0.5", {**FIXED_OUTCOME, "collision": 0.5}, "collision=0.5 must be 0 or 1"),
    ("negative separation", {**FIXED_OUTCOME, "min_separation_m": -0.01}, "min_separation_m=-0.01, outside"),
    ("non-finite recovery time", {**FIXED_OUTCOME, "recovery_time_s": float("nan")}, "is not finite"),
):
    recorder = CallRecorder(OBSERVATION, outcome)
    checks.raises(label, ValueError, lambda o=outcome, r=recorder: spec.episode(r, dict(SITUATION), action, 1), fragment)

checks.finish()
