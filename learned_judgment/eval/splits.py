"""Fixed, independent, seeded evaluation splits per skill.

Lane contract (Person B, evaluation)
------------------------------------
Each split of a skill is drawn from its own named random stream
(:mod:`learned_judgment.engine.streams`), so splits are independent of each other and of
call order, and identical across processes:

    situations   stream  f"{skill}/{split}/{world_preset}/situations", index 0
    episode noise        f"{skill}/{split}/{world_preset}/noise/{repeat}", index = situation index

    split in {"dev", "test", "shift", "context"}
        dev      model, feature and threshold choices
        test     sealed: only :func:`test_situations` with ``final_evaluation=True`` returns
                 it, and every access is appended to a caller-given log
        shift    independent situations per world preset
        context  situations of the context attempts (never a test situation)

The per-episode noise seed does not depend on the method, so every method meets the same
situation with the same noise draw (paired comparison). Situations come from
``spec.sample_situations``; nothing is drawn here from a hidden world value. Sizes, seeds
and the log path are arguments (read from config.yaml by the caller).

Arrays: situations float64 (n, S) in ``spec.situation`` order; seeds int64.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from learned_judgment.engine.streams import stream_rng, stream_seed
from learned_judgment.skills.base import SkillSpec

__all__ = [
    "SPLITS",
    "SealedSplitError",
    "check_disjoint",
    "noise_seed",
    "noise_seeds",
    "split_situations",
    "test_situations",
]

SPLITS = ("dev", "test", "shift", "context")
_SEPARATOR = "/"


class SealedSplitError(PermissionError):
    """The test split was requested outside the final evaluation."""


def _check_identity(spec: SkillSpec, split: str, world_preset: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
    for label, value in (("spec.name", spec.name), ("world_preset", world_preset)):
        if not isinstance(value, str) or not value or _SEPARATOR in value:
            raise ValueError(f"{label} must be a non-empty string without {_SEPARATOR!r}; got {value!r}")


def _check_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return int(value)


def _draw(spec: SkillSpec, *, base_seed: int, split: str, n: int, world_preset: str) -> np.ndarray:
    _check_identity(spec, split, world_preset)
    base_seed = _check_non_negative_int("base_seed", base_seed)
    if _check_non_negative_int("n", n) < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    rng = stream_rng(base_seed, f"{spec.name}/{split}/{world_preset}/situations", 0)
    situations = np.asarray(spec.sample_situations(rng, n), dtype=np.float64)
    if situations.shape != (n, spec.n_situation) or not np.all(np.isfinite(situations)):
        raise ValueError(
            f"{spec.name}.sample_situations must return finite ({n}, {spec.n_situation}); got {situations.shape}"
        )
    return situations


def split_situations(spec: SkillSpec, *, base_seed: int, split: str, n: int, world_preset: str) -> np.ndarray:
    """Situations (n, S) of the ``dev``, ``shift`` or ``context`` split.

    The ``test`` split is sealed: this function raises :class:`SealedSplitError` for it so
    the only way to read test situations is :func:`test_situations`, which logs the access.
    """
    if split == "test":
        raise SealedSplitError(
            "the test split is sealed; use test_situations(..., final_evaluation=True) at the final evaluation"
        )
    return _draw(spec, base_seed=base_seed, split=split, n=n, world_preset=world_preset)


def test_situations(
    spec: SkillSpec,
    *,
    base_seed: int,
    n: int,
    world_preset: str,
    final_evaluation: bool,
    access_log_path: str | Path,
    clock: Callable[[], str],
) -> np.ndarray:
    """Situations (n, S) of the sealed test split, for the final evaluation only.

    Raises :class:`SealedSplitError` unless ``final_evaluation is True``. On success one JSON
    line ``{"event", "timestamp_utc", "skill", "world_preset", "base_seed", "n"}`` is appended
    to ``access_log_path`` (parent directories are created). ``clock()`` returns the current
    time as an ISO 8601 string with a zero UTC offset (the engine's clock convention); it is
    injected so the log time is never invented.
    """
    if final_evaluation is not True:
        raise SealedSplitError(
            f"test situations of {spec.name!r} are sealed until the final evaluation (final_evaluation=True)"
        )
    situations = _draw(spec, base_seed=base_seed, split="test", n=n, world_preset=world_preset)
    now = clock()
    try:
        moment = datetime.fromisoformat(now) if isinstance(now, str) else None
    except ValueError:
        moment = None
    if moment is None or moment.utcoffset() != timedelta(0):
        raise ValueError(f"clock() must return an ISO 8601 string with a zero UTC offset; got {now!r}")
    line = json.dumps(
        {
            "event": "test_split_access",
            "timestamp_utc": now,
            "skill": spec.name,
            "world_preset": world_preset,
            "base_seed": int(base_seed),
            "n": int(n),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    path = Path(access_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return situations


def noise_seed(
    spec: SkillSpec, *, base_seed: int, split: str, world_preset: str, situation_index: int, repeat: int
) -> int:
    """32-bit per-episode seed for situation ``situation_index`` and noise ``repeat``.

    It depends only on (base_seed, skill, split, world preset, situation index, repeat) --
    never on the method -- so all methods share it and comparisons stay paired.
    """
    _check_identity(spec, split, world_preset)
    base_seed = _check_non_negative_int("base_seed", base_seed)
    situation_index = _check_non_negative_int("situation_index", situation_index)
    repeat = _check_non_negative_int("repeat", repeat)
    return stream_seed(base_seed, f"{spec.name}/{split}/{world_preset}/noise/{repeat}", situation_index)


def noise_seeds(
    spec: SkillSpec, *, base_seed: int, split: str, world_preset: str, n_situations: int, n_repeats: int
) -> np.ndarray:
    """int64 (n_situations, n_repeats): ``noise_seed`` for every situation index and repeat."""
    if _check_non_negative_int("n_situations", n_situations) < 1 or _check_non_negative_int("n_repeats", n_repeats) < 1:
        raise ValueError(f"n_situations and n_repeats must be >= 1; got {n_situations}, {n_repeats}")
    seeds = np.empty((n_situations, n_repeats), dtype=np.int64)
    for index in range(n_situations):
        for repeat in range(n_repeats):
            seeds[index, repeat] = noise_seed(
                spec,
                base_seed=base_seed,
                split=split,
                world_preset=world_preset,
                situation_index=index,
                repeat=repeat,
            )
    return seeds


def check_disjoint(splits: Mapping[str, np.ndarray]) -> None:
    """Raise ValueError if any two named situation sets (each (n_i, S)) share an identical row.

    Rows are compared exactly (bitwise float64 values). Duplicates inside one set are allowed.
    """
    rows: dict[str, set[bytes]] = {}
    width: int | None = None
    for name, situations in splits.items():
        array = np.ascontiguousarray(np.asarray(situations, dtype=np.float64) + 0.0)
        if array.ndim != 2:
            raise ValueError(f"split {name!r} must have shape (n, S); got {array.shape}")
        if width is not None and array.shape[1] != width:
            raise ValueError(f"split {name!r} has {array.shape[1]} columns; expected {width}")
        width = array.shape[1]
        rows[name] = {row.tobytes() for row in array}
    names = list(rows)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = len(rows[first] & rows[second])
            if shared:
                raise ValueError(f"splits {first!r} and {second!r} share {shared} identical situation rows")
