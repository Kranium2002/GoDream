"""Training targets of the spiking student, taken from the dream (pure numpy).

Lane contract (Person B, per-skill spiking student)
---------------------------------------------------
The physics guess corrected by the PFN is a dream world. For each dreamed situation it gives
the predicted outcome, and therefore the objective score, of every candidate command:
``DreamSet.scores`` (n, C), higher is better. The spiking student is trained inside that dream
to choose the best candidate. It never sees or learns the PFN's outcome predictions; its only
target is which candidates the dream scores well.

    score_spread   typical spread of the dreamed scores across candidates (the score scale)
    soft_targets   softmax((scores - row max) / (temperature * spread)): a distribution over
                   candidates that puts most weight on the dream's best candidate and keeps the
                   ranking of near-best candidates
    dream_regret   dreamed objective a decision gives up relative to the dream's best candidate

Arrays are float64; n rows (dreamed situations), C candidates per situation.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["dream_regret", "score_spread", "soft_targets"]


def _scores(scores: np.ndarray) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"scores must have shape (n >= 1, C >= 1); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("scores contain non-finite values")
    return array


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number; got {value!r}")
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be finite and > 0; got {number}")
    return number


def score_spread(scores: np.ndarray) -> float:
    """Typical spread of dreamed scores across candidates.

    Args:
        scores: (n, C) dreamed objective per candidate.

    Returns:
        The median over rows of the per-row standard deviation (population, ddof = 0).

    Raises:
        ValueError when the spread is 0: at least half of the rows score every candidate
        identically, so the dream gives the student nothing to learn.
    """
    array = _scores(scores)
    spread = float(np.median(np.std(array, axis=1)))
    if spread == 0.0:
        flat = int(np.sum(np.all(array == array[:, :1], axis=1)))
        raise ValueError(
            f"score spread is 0: {flat} of {array.shape[0]} rows score all {array.shape[1]} candidates "
            "identically; the dream gives nothing to learn"
        )
    return spread


def soft_targets(scores: np.ndarray, *, temperature: float, spread: float) -> np.ndarray:
    """Soft training targets (n, C), each row summing to 1.

    ``softmax((scores - row max) / (temperature * spread))`` row by row. The argmax of every
    row equals the argmax of its scores (the first maximum on ties). A lower temperature
    sharpens the targets towards the dream's best candidate.

    Args:
        scores:      (n, C) dreamed objective per candidate.
        temperature: > 0 (config: students.snn.target_temperature).
        spread:      > 0, the score scale, normally :func:`score_spread` of the training rows.
    """
    array = _scores(scores)
    scale = _positive("temperature", temperature) * _positive("spread", spread)
    logits = (array - array.max(axis=1, keepdims=True)) / scale
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def dream_regret(scores: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Dreamed objective each decision gives up: (n,) row max of ``scores`` minus the score of
    the chosen candidate. 0 means the decision is the dream's best candidate.

    Args:
        scores: (n, C) dreamed objective per candidate.
        chosen: (n,) integer candidate indices in [0, C).
    """
    array = _scores(scores)
    index = np.asarray(chosen)
    if index.shape != (array.shape[0],):
        raise ValueError(f"chosen must have shape ({array.shape[0]},); got {index.shape}")
    if index.size and index.dtype.kind not in "iu":
        raise ValueError(f"chosen must hold integer indices; got dtype {index.dtype}")
    index = index.astype(np.int64)
    if np.any(index < 0) or np.any(index >= array.shape[1]):
        raise ValueError(f"chosen holds an index outside [0, {array.shape[1]})")
    return array.max(axis=1) - array[np.arange(array.shape[0]), index]
