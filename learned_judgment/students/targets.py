"""Training targets of the spiking student, taken from the dream (pure numpy).

Lane contract (Person B, per-skill spiking student)
---------------------------------------------------
The physics guess corrected by the PFN is a dream world. For each dreamed situation it gives
the predicted outcome, and therefore the objective score, of every candidate command:
``DreamSet.scores`` (n, C), higher is better. The spiking student is trained inside that dream
to choose the best candidate. It never sees or learns the PFN's outcome predictions; its only
target is which candidates the dream scores well.

    row_spreads    spread of each row's dreamed scores across its candidates (population std)
    score_spread   typical spread over rows: the median of row_spreads (one global score scale)
    soft_targets   softmax((scores - row max) / (temperature * scale)): a distribution over
                   candidates that puts most weight on the dream's best candidate and keeps the
                   ranking of near-best candidates. The scale is chosen by ``target_spread``
                   (config.yaml students.snn.target_spread):
                     "global_median"  one scale for every row, score_spread of the training rows
                     "per_row"        each row's own spread; a flat row (every candidate
                                      scored exactly the same) gets the uniform target
    dream_regret   dreamed objective a decision gives up relative to the dream's best candidate

With "global_median", rows whose scores vary on a much smaller scale than the median row get
nearly uniform targets (for example rows with no collision risk next to rows with a large
collision penalty); "per_row" keeps the ranking inside every row, but also sharpens rows whose
small differences are noise. Which one is used is chosen on dev data.

Arrays are float64; n rows (dreamed situations), C candidates per situation.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["TARGET_SPREADS", "dream_regret", "flat_rows", "row_spreads", "score_spread", "soft_targets"]

TARGET_SPREADS = ("global_median", "per_row")


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


def row_spreads(scores: np.ndarray) -> np.ndarray:
    """(n,) spread of each row of ``scores`` (n, C): the population standard deviation (ddof = 0)
    of the row's dreamed scores across its candidates. A row that scores every candidate the
    same is :func:`flat_rows`; its spread is 0 up to rounding of the mean."""
    return np.std(_scores(scores), axis=1)


def flat_rows(scores: np.ndarray) -> np.ndarray:
    """bool (n,): True where a row of ``scores`` (n, C) scores every candidate exactly the same."""
    array = _scores(scores)
    return np.all(array == array[:, :1], axis=1)


def score_spread(scores: np.ndarray) -> float:
    """Typical spread of dreamed scores across candidates.

    Args:
        scores: (n, C) dreamed objective per candidate.

    Returns:
        The median over rows of :func:`row_spreads`.

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


def soft_targets(scores: np.ndarray, *, temperature: float, target_spread: str, spread: float | None) -> np.ndarray:
    """Soft training targets (n, C), each row summing to 1.

    ``softmax((scores - row max) / (temperature * scale))`` row by row. The argmax of every
    row equals the argmax of its scores (the first maximum on ties). A lower temperature
    sharpens the targets towards the dream's best candidate.

    Args:
        scores:        (n, C) dreamed objective per candidate.
        temperature:   > 0 (config: students.snn.target_temperature), in units of the scale.
        target_spread: "global_median" or "per_row" (config: students.snn.target_spread).
        spread:        "global_median": the one scale for every row, > 0, normally
                       :func:`score_spread` of the training rows. "per_row": must be None; each
                       row uses its own :func:`row_spreads` value, and a :func:`flat_rows` row
                       (every candidate scored the same) gets the uniform target 1 / C.

    Raises:
        ValueError for non-finite scores, an unknown ``target_spread``, a ``spread`` that does
        not fit the mode, or a temperature x scale product that is not a finite number > 0.
    """
    array = _scores(scores)
    temperature = _positive("temperature", temperature)
    shifted = array - array.max(axis=1, keepdims=True)
    if target_spread == "global_median":
        if spread is None:
            raise ValueError("target_spread 'global_median' needs spread (the score_spread of the training rows)")
        scale = temperature * _positive("spread", spread)
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(f"temperature x spread = {scale} is not a finite number > 0")
        logits = shifted / scale
    elif target_spread == "per_row":
        if spread is not None:
            raise ValueError(f"target_spread 'per_row' takes each row's own spread; spread must be None, got {spread!r}")
        own = np.std(array, axis=1)
        flat = np.all(array == array[:, :1], axis=1)
        scales = temperature * own
        bad = ~flat & ~((scales > 0.0) & np.isfinite(scales))
        if np.any(bad):
            row = int(np.argwhere(bad)[0][0])
            raise ValueError(
                f"row {row}: temperature x row spread = {scales[row]} is not a finite number > 0 "
                f"(row spread {own[row]}, temperature {temperature})"
            )
        logits = shifted / np.where(flat, 1.0, scales)[:, None]
        logits[flat] = 0.0
    else:
        raise ValueError(f"target_spread must be one of {TARGET_SPREADS}; got {target_spread!r}")
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
