"""Check 7: harness input validation and the latency-record shape. NO episode runs here.

evaluate_actions / compare_methods need Person A's Go2World and stay NOT RUN. This check
only shows that invalid inputs are refused BEFORE any world is built, before any episode,
before any policy is asked for actions and before any manifest or attempt file is written:
the world factory passed below raises if it is ever called, the "world" handed to
evaluate_actions is a refusing token, and the fixture spec's episode raises and records the
call. Engine review G1: a spec whose command dimension is not inside the verified command
schema is refused first (on the old code the harness took no command schema at all).
Engine review G-seed / G7: attempt seeds must be distinct within a run and the world seed
must be a non-negative integer. measure_latency is called on a no-op function only to check
the keys of its record; the timings of a no-op are printed nowhere and mean nothing.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import tempfile  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_spec import FIXTURE_COMMAND_SCHEMA, Checks, FixtureSpec, fixture_rows  # noqa: E402
from learned_judgment.config import config_digest, load_config  # noqa: E402
from learned_judgment.eval.harness import compare_methods, evaluate_actions, measure_latency  # noqa: E402
from learned_judgment.skills.base import CommandField, CommandSchema, SkillSpecError  # noqa: E402

checks = Checks()
expect = checks.expect
built = []
asked = []
episodes = []
DIGEST = config_digest(load_config())


class RecordingSpec(FixtureSpec):
    """FIXTURE: the episode records that it was reached and raises; it never measures anything."""

    def episode(self, world, situation, action, attempt_seed):  # noqa: ANN001, ANN201
        episodes.append(attempt_seed)
        raise RuntimeError("check 7 never runs an episode")


spec = RecordingSpec()


def refusing_factory(seed: int):
    built.append(seed)
    raise RuntimeError("check 7 never builds a world")


class RefusingWorld:
    """Not a world: any attribute access fails, so a call that touched it would show up."""

    def __getattr__(self, name):  # noqa: ANN001, ANN204
        built.append(name)
        raise RuntimeError("check 7 never steps a world")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Fixture command schemas that do NOT cover the fixture spec's command dimension (side_speed is
# bound to "fixture_vy" with range [-0.5, 0.5] m/s).
SCHEMA_WITHOUT_FIELD = CommandSchema(
    fields={"fixture_vx": CommandField(name="fixture_vx", unit="m/s", low=-1.0, high=1.0)},
    update_period_s=0.02, source="fixture: field missing",
)
SCHEMA_NARROWER = CommandSchema(
    fields={"fixture_vy": CommandField(name="fixture_vy", unit="m/s", low=-0.2, high=0.2)},
    update_period_s=0.02, source="fixture: range narrower than the spec's action range",
)
SCHEMA_OTHER_UNIT = CommandSchema(
    fields={"fixture_vy": CommandField(name="fixture_vy", unit="rad/s", low=-1.0, high=1.0)},
    update_period_s=0.02, source="fixture: other unit",
)

RUN = dict(run_id="check7", method="physics_only", world_preset="default", split="dev", base_seed=7, repeat=0,
           world_seed=11, config_digest=DIGEST, clock=utc_now, command_schema=FIXTURE_COMMAND_SCHEMA)
situations, actions, _, _ = fixture_rows(spec, 3, seed=31)
with tempfile.TemporaryDirectory() as directory:
    out = Path(directory) / "rows.jsonl"
    cases = (
        ("action outside the declared bounds", ValueError, dict(actions=actions + np.array([[1.0, 0.0]]), attempt_seeds=np.arange(3))),
        ("attempt seeds of the wrong shape", ValueError, dict(actions=actions, attempt_seeds=np.arange(2))),
        ("float attempt seeds", ValueError, dict(actions=actions, attempt_seeds=np.arange(3, dtype=float))),
        ("repeated attempt seeds", ValueError, dict(actions=actions, attempt_seeds=np.array([4, 5, 4]))),
        ("negative attempt seeds", ValueError, dict(actions=actions, attempt_seeds=np.array([4, -5, 6]))),
        ("a negative world seed", ValueError, dict(actions=actions, attempt_seeds=np.arange(3), world_seed=-1)),
        ("a float world seed", ValueError, dict(actions=actions, attempt_seeds=np.arange(3), world_seed=11.0)),
        ("actions of the wrong width", ValueError, dict(actions=actions[:, :1], attempt_seeds=np.arange(3))),
        ("an unknown split", ValueError, dict(actions=actions, attempt_seeds=np.arange(3), split="validation")),
        ("a config digest that is not sha256 hex", ValueError, dict(actions=actions, attempt_seeds=np.arange(3), config_digest="abc")),
        ("a negative base seed", ValueError, dict(actions=actions, attempt_seeds=np.arange(3), base_seed=-1)),
        ("a clock without a UTC offset", ValueError,
         dict(actions=actions, attempt_seeds=np.arange(3), clock=lambda: "2026-11-21T12:00:00")),
        ("an attempt file that is not .jsonl", ValueError,
         dict(actions=actions, attempt_seeds=np.arange(3), out_path=Path(directory) / "rows.csv")),
        ("a command schema without the spec's command field (G1)", SkillSpecError,
         dict(actions=actions, attempt_seeds=np.arange(3), command_schema=SCHEMA_WITHOUT_FIELD)),
        ("a command schema narrower than the spec's action range (G1)", SkillSpecError,
         dict(actions=actions, attempt_seeds=np.arange(3), command_schema=SCHEMA_NARROWER)),
        ("a command schema with another unit (G1)", SkillSpecError,
         dict(actions=actions, attempt_seeds=np.arange(3), command_schema=SCHEMA_OTHER_UNIT)),
        ("a command schema that is not a CommandSchema (G1)", TypeError,
         dict(actions=actions, attempt_seeds=np.arange(3), command_schema={"fixture_vy": (-1.0, 1.0)})),
    )
    for label, error_type, arguments in cases:
        arguments = {**RUN, "out_path": out, **arguments}
        try:
            evaluate_actions(spec, RefusingWorld(), situations, **arguments)
            expect(False, f"evaluate_actions refuses {label}")
        except error_type as error:
            expect(True, f"evaluate_actions refuses {label}: {type(error).__name__}: {str(error)[:150]}")
        except Exception as error:  # noqa: BLE001
            expect(False, f"evaluate_actions refuses {label} with {error_type.__name__}; got {type(error).__name__}: {error}")

    def compare(**overrides):  # noqa: ANN003, ANN202
        def policy(s):  # noqa: ANN001, ANN202
            asked.append(s.shape[0])
            return actions

        arguments = dict(
            world_factory=refusing_factory, world_seed=11, situations=situations,
            attempt_seeds=np.arange(6).reshape(3, 2), command_schema=FIXTURE_COMMAND_SCHEMA, run_id="check7",
            world_preset="default", split="dev", base_seed=7, config_digest=DIGEST, out_dir=directory, clock=utc_now,
        )
        arguments.update(overrides)
        policies = arguments.pop("policies", {"ok": policy})
        return compare_methods(spec, policies, **arguments)

    try:
        compare(policies={"ok": lambda s: actions, "bad": lambda s: actions + np.array([[1.0, 0.0]])})
        expect(False, "compare_methods validates every method's actions before the first world is built")
    except ValueError as error:
        expect(True, f"compare_methods validates every method's actions before the first world is built: {error}")
    compare_cases = (
        ("a command schema without the spec's command field (G1)", SkillSpecError, dict(command_schema=SCHEMA_WITHOUT_FIELD)),
        ("a command schema narrower than the spec's action range (G1)", SkillSpecError, dict(command_schema=SCHEMA_NARROWER)),
        ("an attempt seed used twice in the (n, R) matrix", ValueError, dict(attempt_seeds=np.array([[0, 1], [2, 3], [4, 0]]))),
        ("attempt seeds of shape (n,) instead of (n, R)", ValueError, dict(attempt_seeds=np.arange(3))),
        ("a negative world seed", ValueError, dict(world_seed=-1)),
        ("a clock without a UTC offset", ValueError, dict(clock=lambda: "2026-11-21T12:00:00")),
    )
    for label, error_type, overrides in compare_cases:
        before = len(asked)
        try:
            compare(**overrides)
            expect(False, f"compare_methods refuses {label}")
        except error_type as error:
            expect(len(asked) == before, f"compare_methods refuses {label} before asking any policy: {type(error).__name__}: {str(error)[:120]}")
        except Exception as error:  # noqa: BLE001
            expect(False, f"compare_methods refuses {label} with {error_type.__name__}; got {type(error).__name__}: {error}")
    expect(built == [], "no world was built and no world was touched")
    expect(episodes == [], "no episode was reached")
    expect(not out.exists() and not any(Path(directory).iterdir()), "no attempt file and no manifest was written")

record = measure_latency(lambda x: x, [1, 2, 3], 50)
expect(set(record) == {"median_s", "p95_s", "n_decisions", "n_inputs", "timer", "platform", "processor", "cpu_count"},
       f"latency record keys {sorted(record)}")
expect(record["p95_s"] >= record["median_s"] >= 0.0 and record["n_decisions"] == 50, "p95 >= median >= 0, n_decisions recorded")

checks.finish()
