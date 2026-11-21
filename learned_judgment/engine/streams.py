"""Named, independent, reproducible random streams.

A stream is identified by ``(base_seed, name, index)`` and never by call order, so adding
a draw somewhere cannot shift the numbers of another stream. Names keep the mission's
streams apart, for example::

    "dodge/dev/situations"      situation sampling of a split
    "dodge/context/situations"  situations of the context attempts
    "dodge/dev/noise"           per-attempt noise, one index per episode
    "dodge/dreams/situations"   dreamed situations

Recipe verified across processes with different PYTHONHASHSEED in
experiments/proofs/B_students_eval_contract/p08_streams.py (no builtin ``hash()``).
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["stream_rng", "stream_seed"]

_NAME_WORDS = 4  # 4 little-endian uint32 words = the first 128 bits of sha256(name)


def _spawn_key(name: str, index: int) -> tuple[int, ...]:
    if not name:
        raise ValueError("a stream needs a non-empty name")
    if index < 0:
        raise ValueError(f"stream index must be >= 0; got {index}")
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    words = tuple(int.from_bytes(digest[4 * i : 4 * i + 4], "little") for i in range(_NAME_WORDS))
    return (*words, int(index))


def stream_rng(base_seed: int, name: str, index: int = 0) -> np.random.Generator:
    """Generator for stream ``name`` / ``index`` under ``base_seed`` (non-negative int)."""
    sequence = np.random.SeedSequence(int(base_seed), spawn_key=_spawn_key(name, index))
    return np.random.Generator(np.random.PCG64(sequence))


def stream_seed(base_seed: int, name: str, index: int = 0) -> int:
    """A 32-bit integer seed derived from the same stream identity, for libraries that take
    an integer seed (scikit-learn ``random_state``, ``torch.manual_seed``)."""
    sequence = np.random.SeedSequence(int(base_seed), spawn_key=_spawn_key(name, index))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
