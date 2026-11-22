"""Go2/MuJoCo environment package. Implementation belongs to Person A."""
import yaml

from learned_judgment.env.commands import CommandError, validate_command


def load_config(path="config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# After load_config: world imports acceptance, which imports load_config from this package.
from learned_judgment.env.incline import sample_ramp_situations  # noqa: E402
from learned_judgment.env.throws import BALL_KEYS, sample_dodge_situations  # noqa: E402
from learned_judgment.env.world import Go2World, make_streams  # noqa: E402
