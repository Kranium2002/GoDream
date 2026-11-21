"""R06: config.load_config with duplicate keys; CommandSchema.from_config on malformed schemas;
hidden-name gaps (validate_spec checks situation names only; build_layout extra is unchecked)."""

print("REVIEW REPRODUCTION - fixture, not evidence")

import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import LEVELS, REVIEW_SCHEMA, ReviewSpec  # noqa: E402

from learned_judgment.config import config_digest, load_config, require  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.skills.base import CommandSchema, SkillSpecError, Variable, validate_spec  # noqa: E402

# (1) duplicate keys in config.yaml
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "c.yaml"
    path.write_text(
        "engine:\n  quantile_levels: [0.1, 0.25, 0.5, 0.75, 0.9]\n  spread_levels: [0.1, 0.9]\n"
        "  spread_levels: [0.25, 0.75]\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    print("1. duplicate 'engine.spread_levels' in YAML -> load_config returns", require(cfg, "engine.spread_levels"),
          "(no error)")


def schema(fields, period=0.02, status="verified"):
    return {"robot": {"walking_policy": {"status": status, "command_schema": {"update_period_s": period, "fields": fields}}}}


cases = {
    "update_period_s = inf": schema({"vy": {"unit": "m/s", "min": -1.0, "max": 1.0}}, period=float("inf")),
    "min/max as quoted strings": schema({"vy": {"unit": "m/s", "min": "-1.0", "max": "1.0"}}),
    "min/max as booleans": schema({"vy": {"unit": "m/s", "min": False, "max": True}}),
    "empty unit string": schema({"vy": {"unit": "", "min": -1.0, "max": 1.0}}),
    "unit given as a number": schema({"vy": {"unit": 7, "min": -1.0, "max": 1.0}}),
    "field name not a string (YAML key 1)": schema({1: {"unit": "m/s", "min": -1.0, "max": 1.0}}),
    "walking_policy is a string": {"robot": {"walking_policy": "verified"}},
    "min is a non-numeric string": schema({"vy": {"unit": "m/s", "min": "fast", "max": 1.0}}),
}
print("2. CommandSchema.from_config on malformed schemas:")
for label, cfg in cases.items():
    try:
        s = CommandSchema.from_config(cfg)
        print(f"   {label:38s}: ACCEPTED -> {[(f.name, f.unit, f.low, f.high) for f in s.fields.values()]} period={s.update_period_s}")
    except Exception as error:  # noqa: BLE001
        print(f"   {label:38s}: {type(error).__name__}: {str(error)[:80]}")

# (3) hidden-value names: only situation names are compared, exactly
spec = ReviewSpec()
hidden = ("payload_mass_kg", "slope_deg", "motor_strength_fraction", "command_latency_s")
layout = build_layout(spec, extra=(Variable("motor_strength_fraction", "float", "1", "TRUE motor strength", low=0.8, high=1.0),))
print("3a. build_layout(extra=(motor_strength_fraction,)) accepted; feature names:", layout.names)
leaky = ReviewSpec()
leaky.situation = (*leaky.situation, Variable("true_slope_deg", "float", "deg", "simulator slope", low=-5.0, high=5.0))
leaky.physics_guess = lambda s, a: ReviewSpec.physics_guess(leaky, s[:, :2], a)
try:
    validate_spec(leaky, REVIEW_SCHEMA, hidden_value_names=hidden, quantile_levels=LEVELS, probe_seed=0, probe_situations=4,
                  probe_points_per_dim=3)
    print("3b. validate_spec accepts a situation feature 'true_slope_deg' with hidden names", hidden)
except SkillSpecError as error:
    print("3b. validate_spec rejects:", error)
