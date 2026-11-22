"""Check 1: the DODGE SkillSpec against the real config.yaml as it is.

FIXTURE CHECK - mechanics only. The real config.yaml still has null keys under skills.dodge
(Person A publishes them with the throw model and the Go2 measurements), so building DODGE
must raise ConfigError naming a null key, while the verified command schema itself loads.
Then each null key is tested alone: every OTHER null key is filled with a labelled fixture
value (fixture_dodge.FIXTURE_VALUES) and the build must name exactly the one left null.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_dodge_offline\\c1_real_config.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import re  # noqa: E402

from fixture_dodge import FIXTURE_DECLARED, FIXTURE_VALUES, Checks, fixture_config, fixture_throw_sampler  # noqa: E402
from learned_judgment.config import ConfigError, load_config  # noqa: E402
from learned_judgment.skills import available_skills, load_skill  # noqa: E402
from learned_judgment.skills.base import CommandSchema  # noqa: E402
from learned_judgment.skills.dodge import DodgeSpec  # noqa: E402

checks = Checks()
real = load_config()

print("\n== registry ==")
checks.check("available_skills() lists dodge", "dodge" in available_skills(), f"{available_skills()}")

print("\n== the verified command schema loads from the real config ==")
schema = CommandSchema.from_config(real)
for name, field in schema.fields.items():
    print(f"  {name}: unit={field.unit} range=[{field.low}, {field.high}]")
print(f"  update_period_s={schema.update_period_s}")
checks.check(
    "agreed command fields vel_x, vel_y, yaw_rate are verified",
    {"vel_x", "vel_y", "yaw_rate"} <= set(schema.fields),
    f"{sorted(schema.fields)}",
)


def null_key_named(message: str) -> str | None:
    match = re.search(r"key '([^']+)' is null", message)
    return match.group(1) if match else None


def lookup(config: dict, dotted: str) -> object:
    node: object = config
    for part in dotted.split("."):
        node = node[part]  # type: ignore[index]
    return node


print("\n== load_skill('dodge', real config) ==")
sampler = fixture_throw_sampler(real)
for label, build in (
    ("load_skill without arguments", lambda: load_skill("dodge", real)),
    (
        "load_skill with a fixture throw sampler",
        lambda: load_skill("dodge", real, throw_sampler=sampler, **FIXTURE_DECLARED),
    ),
    ("DodgeSpec(real config, verified schema)", lambda: DodgeSpec(real, schema)),
):
    message = checks.raises(f"{label} raises ConfigError naming a null key", ConfigError, build, "is null")
    named = null_key_named(message)
    print(f"  null key named: {named}")
    checks.check(f"{label}: the named key is null in the real config", named is not None and lookup(real, named) is None)

print("\n== each null key alone stops the build ==")
for key in FIXTURE_VALUES:
    config = fixture_config(real, skip=(key,))
    checks.raises(
        f"only {key} null",
        ConfigError,
        lambda config=config: load_skill(
            "dodge", config, throw_sampler=fixture_throw_sampler(config), **FIXTURE_DECLARED
        ),
        f"'{key}' is null",
    )

checks.finish()
