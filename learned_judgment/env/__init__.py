"""Go2/MuJoCo environment package. Implementation belongs to Person A."""
import yaml


def load_config(path="config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
