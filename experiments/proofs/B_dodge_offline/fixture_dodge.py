"""Shared FIXTURE helpers of the DODGE offline checks (mechanics only).

FIXTURE ONLY. Everything here exists to exercise the DODGE SkillSpec's mechanics without a
world: a copy of the real config.yaml whose null ``skills.dodge`` keys are filled with
labelled fixture values, a fixture throw sampler standing in for Person A's throw model,
and a small pass/fail harness. None of it is a robot value, a measured throw or evidence
about the Go2, the simulator, TabPFN or any model, and none of it may enter results/.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learned_judgment.config import require  # noqa: E402

# Fixture values for exactly the skills.dodge keys that are null in the real config.yaml and
# that the DODGE SkillSpec reads. Chosen only so the hand-computed cases of the checks lie
# inside the ranges; NOT robot, ball or throw-model values.
FIXTURE_VALUES: dict[str, Any] = {
    "skills.dodge.situation_ranges.ball_rel_x_m": [0.0, 6.0],
    "skills.dodge.situation_ranges.ball_rel_y_m": [-2.0, 2.0],
    "skills.dodge.situation_ranges.ball_rel_z_m": [-0.25, 1.0],
    "skills.dodge.situation_ranges.ball_vel_x_m_s": [-10.0, 1.0],
    "skills.dodge.situation_ranges.ball_vel_y_m_s": [-3.0, 3.0],
    "skills.dodge.situation_ranges.ball_vel_z_m_s": [-1.0, 8.0],
    "skills.dodge.situation_ranges.dog_forward_speed_m_s": [0.0, 0.8],
    "skills.dodge.declared_surfaces": ["fixture_surface_a", "fixture_surface_b"],
    "skills.dodge.episode_duration_s": 1.6,
    "skills.dodge.physics.body_half_length_m": 0.19,
    "skills.dodge.physics.body_radius_m": 0.10,
    "skills.dodge.physics.base_height_m": 0.30,
}

# The fixture world's declared values (labels a world would declare); fixture only.
FIXTURE_DECLARED: dict[str, Any] = {"declared_payload_kg": 1.0, "declared_surface": "fixture_surface_a"}


def _set(config: dict, dotted_key: str, value: Any) -> None:
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = copy.deepcopy(value)


def _get(config: Mapping[str, Any], dotted_key: str) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        node = node[part]
    return node


def fixture_config(real_config: Mapping[str, Any], *, skip: tuple[str, ...] = ()) -> dict:
    """Deep copy of ``real_config`` with every FIXTURE_VALUES key filled, except ``skip``.

    Refuses to overwrite a key that is not null in the real config: only null keys are filled.
    """
    config = copy.deepcopy(dict(real_config))
    for key, value in FIXTURE_VALUES.items():
        if _get(real_config, key) is not None:
            raise RuntimeError(f"fixture would overwrite the non-null real key {key!r}; refusing")
        if key not in skip:
            _set(config, key, value)
    return config


def changed_leaves(before: Any, after: Any, prefix: str = "") -> list[str]:
    """Dotted keys whose leaf value differs between two nested mappings."""
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        keys = sorted(set(before) | set(after), key=str)
        found: list[str] = []
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            found.extend(changed_leaves(before.get(key), after.get(key), path))
        return found
    return [] if before == after else [prefix]


def with_value(config: Mapping[str, Any], dotted_key: str, value: Any) -> dict:
    """Deep copy of ``config`` with one key replaced (for refusal checks)."""
    changed = copy.deepcopy(dict(config))
    _set(changed, dotted_key, value)
    return changed


def fixture_throw_sampler(config: Mapping[str, Any]) -> Callable[[np.random.Generator, int], np.ndarray]:
    """FIXTURE stand-in for Person A's throw model (check scripts only): throws aimed near the
    dog's future position, drawn row by row (one ``rng.random((n, 7))`` block, row-major).

    Columns: ball_rel_x/y/z_m, ball_vel_x/y/z_m_s, dog_forward_speed_m_s. All values fall
    inside FIXTURE_VALUES' ranges (flight 0.6-1.2 s, start 3-5 m ahead).
    """
    gravity = float(require(config, "skills.dodge.physics.gravity_m_s2"))

    def sample(rng: np.random.Generator, n: int) -> np.ndarray:
        u = rng.random((n, 7))
        flight = 0.6 + 0.6 * u[:, 0]
        forward = 0.3 + 0.3 * u[:, 1]
        start = np.column_stack([3.0 + 2.0 * u[:, 2], -1.0 + 2.0 * u[:, 3], 0.6 * u[:, 4]])
        target = np.column_stack(
            [forward * flight + (-0.2 + 0.4 * u[:, 5]), -0.3 + 0.6 * u[:, 6], np.zeros(n)]
        )
        velocity = (target - start) / flight[:, None]
        velocity[:, 2] += 0.5 * gravity * flight
        return np.column_stack([start, velocity, forward])

    return sample


class Checks:
    """Tiny pass/fail harness; ``finish`` exits 0 only when every check held."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"PASS  {label}" + (f"  [{detail}]" if detail else ""))
        else:
            self.failed.append(label)
            print(f"FAIL  {label}" + (f"  [{detail}]" if detail else ""))

    def raises(self, label: str, error: type[BaseException], function: Callable[[], Any], fragment: str) -> str:
        try:
            function()
        except error as caught:  # noqa: PERF203
            message = str(caught)
            self.check(label, fragment in message, f"{type(caught).__name__}: {message}")
            return message
        except Exception as other:  # noqa: BLE001
            self.check(label, False, f"wrong error {type(other).__name__}: {other}")
            return ""
        self.check(label, False, f"no {error.__name__} raised")
        return ""

    def finish(self) -> None:
        print(f"\n{self.passed} passed, {len(self.failed)} failed")
        for label in self.failed:
            print(f"  failed: {label}")
        sys.exit(1 if self.failed else 0)
