"""Measurement of one deployed spiking student (Person B, students lane: measurement).

Lane contract
-------------
The physics guess corrected by the PFN is a dream world: for every dreamed situation it gives
the dreamed objective score of every candidate command (:class:`~learned_judgment.engine.dream.DreamSet`).
A spiking student is trained inside that dream to choose the best candidate. What decides is a
:class:`~learned_judgment.students.snn.DeployedStudent`: one skill's numpy network, its input
encoding, the records of its candidate grid and the digests of every situation its training
saw. Every function here takes one, so a per-skill student and one skill's view of the shared
spiking network are measured by the same code:

    per-skill SNNStudent        pass it as it is (anything with a ``deployed()`` method taking
                                no arguments is converted by calling it), or pass
                                ``student.deployed()``
    shared spiking network      pass ``shared.deployed(skill)``, the view of one skill

The functions return JSON-serializable records that name their inputs (network digest, digests
of the situations and dream rows, seeds, the config values read). Every number comes from
actual runs of the DEPLOYED numpy network (:class:`~learned_judgment.students.snn_core.SpikingNetwork`);
importing this module or calling any function in it never imports torch.

    measure_decision_latency  wall-clock latency of single decisions on THIS machine, through a
                              student bound to its spec once (DeployedStudent.bind): the full
                              path (situation in, command values out; the one for H3) and the
                              index-only path, each median and p95
    measure_operations        spikes, synaptic operations and per-neuron firing statistics per
                              decision; BOTH energy ESTIMATES of learned_judgment.students.energy
                              with intervals, the dense reference, both ratios, the break-even
                              firing rate of each estimate and the H4 energy verdict rule
    agreement_with_dream      decision quality INSIDE dreams the student did not see: agreement
                              with the dream's choice and dream regret, next to the physics-only
                              choice stored in the same DreamSet (the "no learning" reference)
    student_record            the three measurements plus provenance; every tunable is read from
                              config.yaml (evaluation.*, students.snn.energy.*)
    h4_energy_criterion       the rule of PREREGISTRATION.md's amendment "how H4's energy
                              criterion is judged", applied to the two estimates
    write_record              canonical JSON written atomically
    dream_source              which kind of scoring produced a dream (provider-backed teacher,
                              local residual model without a provider request, physics only)

Held out means held out. A dream row whose situation (sha256 of its values,
:func:`learned_judgment.students.snn.situation_row_digests`) is one the student's optimiser or
early stopping saw (``DeployedStudent.training_situation_digests``) is not held out.
:func:`student_record` refuses any such row (:class:`HeldOutOverlapError`, stating how many rows
overlap); :func:`agreement_with_dream` refuses them unless ``allow_seen=True``, and then labels
its record "includes situations seen in training".

Distinct situations. Records state the number of distinct situations next to the row count,
and every bootstrap interval resamples DISTINCT situations (each with all of its rows), not
rows: the statistic of a resample is the mean over the rows it contains, so the point estimate
is the row mean and an exact duplicate of a row set does not narrow the interval.

Out of scope: decision quality on actual Go2 episodes is measured by
:mod:`learned_judgment.eval.harness` on the simulator, not here. Quality numbers in these
records are statements about dreamed objective scores; energy numbers are estimates.

Shapes (float64 unless noted)::

    n situations or dream rows   u distinct situations among them   S situation features
    D encoded inputs (the network's input width)   C candidates   A action dimensions
    L hidden layers   H_l hidden widths   fo_l fan-out of hidden layer l   T time steps
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy import stats

from learned_judgment.config import config_digest, require
from learned_judgment.engine.decide import PHYSICS_ONLY
from learned_judgment.engine.dream import DreamSet
from learned_judgment.engine.streams import stream_seed
from learned_judgment.eval.bootstrap import METHODS
from learned_judgment.eval.harness import measure_latency
from learned_judgment.eval.metrics import UNDEFINED, regression_summary
from learned_judgment.fileio import atomic_write_text
from learned_judgment.skills.base import SkillSpec
from learned_judgment.students.encoding import InputEncoder
from learned_judgment.students.energy import CAVEATS, EnergyConstants, estimate_energy
from learned_judgment.students.snn import DeployedStudent, check_dreams, dream_content_sha256, situation_row_digests
from learned_judgment.students.targets import dream_regret, score_spread

__all__ = [
    "DREAM_SOURCES",
    "H4_ENERGY_RULE",
    "H4_ENERGY_VERDICTS",
    "RECORD_FORMAT",
    "SEEN_IN_TRAINING_LABEL",
    "HeldOutOverlapError",
    "agreement_with_dream",
    "dream_source",
    "h4_energy_criterion",
    "measure_decision_latency",
    "measure_operations",
    "student_record",
    "write_record",
]

RECORD_FORMAT = "learned-judgment/snn-student-measurement/v2"

DREAM_SOURCES = {
    "provider_backed_teacher": (
        "scores = objective of the physics guess plus the residual predicted by a provider-backed model; "
        "request_hashes name the cached provider requests behind them"
    ),
    "local_residual_model": (
        "scores = objective of the physics guess plus the residual predicted by a residual model that made "
        "no provider request (for example a local baseline); this is not the TabPFN teacher"
    ),
    "physics_only": (
        "scores = objective of the physics guess alone; no residual model and no provider request, so a "
        "student trained in this dream learns the physics guess's choices"
    ),
}

SEEN_IN_TRAINING_LABEL = "includes situations seen in training"

H4_ENERGY_VERDICTS = ("supported", "not supported", "inconclusive")
H4_ENERGY_RULE = (
    "PREREGISTRATION.md, amendment 2026-11-21 \"how H4's energy criterion is judged\": two estimates are reported "
    "for the SNN, each with an interval: the convention common in the SNN literature (real-valued first layer plus "
    "one accumulate per synaptic event), and the same plus every neuron's leak, input and reset update. The reference "
    "is the non-spiking dense network of the same layer sizes. H4's energy criterion is supported only if both "
    "estimates are below the reference, not supported if both are above it, and inconclusive if they disagree. "
    "Applied here to each estimate's mean per decision over the measured situations; a mean equal to the reference "
    "is neither below nor above it, which makes the verdict inconclusive."
)
_H4_SCOPE = (
    "H4's energy criterion only, for this network on these situations. H4's error criterion (within 25% of the "
    "teacher's error) is judged separately on simulator episodes. Every energy figure is an estimate from operation "
    "counts and cited per-operation energies, not a measurement."
)


class HeldOutOverlapError(ValueError):
    """Dreams passed as held out contain situations the student's optimiser or early stopping
    saw; the message states how many rows (and distinct situations) overlap."""


# Structural rules (not tunables): the fallback when a BCa interval is not finite, the same
# rule as learned_judgment.eval.bootstrap; and the formats of the digests recorded here.
_FALLBACK_METHOD = "percentile"
_SITUATIONS_DIGEST_TAG = b"learned-judgment/situations/v1"
_RESAMPLED_UNIT = (
    "distinct situation (sha256 of its values) with all of its rows; the statistic of a resample is the mean over "
    "the rows it contains (sum over the drawn situations / their row count), so the estimate is the row mean"
)
_FULL_PATH = (
    "BoundStudent.decide_one(situation (S,)) -> command values (A,): input checks and the out_of_range policy -> "
    "encode_one -> SpikingNetwork.decide_one (numpy) -> spec.candidate_actions(situation, points_per_dim)[index] -> "
    "spec.action_in_bounds; situation in, command values out"
)
_INDEX_PATH = (
    "BoundStudent.decide_index_one(situation (S,)) -> candidate index: input checks and the out_of_range policy -> "
    "encode_one -> SpikingNetwork.decide_one (numpy); the candidate lookup and the bounds check are not included"
)
_GRID_CHECK = (
    "DeployedStudent.bind(spec) ran check_spec(spec) once, before timing: the spec's action records and the digest "
    "of its candidate grid over the student's stored probe situations. It is inside no timed call; each timed call "
    "only checks that spec.actions and spec.situation are still the objects checked at binding"
)
_H3_NOTE = (
    "For hypothesis H3 (at least 1,000x lower decision latency than the teacher) use full_path: situation in, command "
    "values out. index_only leaves out the candidate lookup and is reported for comparison only"
)
_LATENCY_PROTOCOL = (
    "learned_judgment.eval.harness.measure_latency, once for full_path and then once for index_only over the same "
    "situations: call i decides situation i mod n; each call is timed alone with time.perf_counter_ns; no warm-up "
    "calls; median and p95 (numpy 'linear' quantile) over all calls"
)


# --------------------------------------------------------------------------------------
# Input checks and digests
# --------------------------------------------------------------------------------------


def _deployed(student: Any) -> DeployedStudent:
    """The :class:`DeployedStudent` to measure: ``student`` itself, or ``student.deployed()``."""
    if isinstance(student, DeployedStudent):
        return student
    method = getattr(student, "deployed", None)
    if not callable(method):
        raise TypeError(
            f"student must be a DeployedStudent or have a deployed() method returning one (a per-skill SNNStudent); "
            f"got {type(student).__name__}. For the shared spiking network pass shared.deployed(skill)"
        )
    try:
        deployed = method()
    except TypeError as error:
        raise TypeError(
            f"{type(student).__name__}.deployed() cannot be called without arguments ({error}); for the shared spiking "
            "network pass shared.deployed(skill)"
        ) from error
    if not isinstance(deployed, DeployedStudent):
        raise TypeError(f"{type(student).__name__}.deployed() returned {type(deployed).__name__}, not a DeployedStudent")
    return deployed


def _check_count(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return int(value)


def _check_bootstrap_settings(n_resamples: Any, confidence_level: Any, method: Any, seed: Any) -> tuple[int, float, str, int]:
    n_resamples = _check_count("n_resamples", n_resamples, 1)
    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(
        confidence_level, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"confidence_level must be a number; got {confidence_level!r}")
    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence_level must lie inside (0, 1); got {level}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    return n_resamples, level, str(method), _check_count("seed", seed, 0)


def _situation_arrays(deployed: DeployedStudent, situations: Any, *, minimum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate situations (n >= minimum, S) for ``deployed``.

    Returns:
        raw:     (n, S) float64, as given (what a deployed call receives).
        checked: (n, S) float64 after the student's input checks and out_of_range policy.
        clipped: bool (n, S), True where a value was clipped (all False under "error").

    Raises ValueError for a wrong shape, a non-finite value or an invalid level index, and
    :class:`~learned_judgment.students.snn.OutOfRangeError` under the "error" policy.
    """
    raw = np.asarray(situations, dtype=np.float64)
    width = len(deployed.situation_variables)
    if raw.ndim != 2 or raw.shape[1] != width or raw.shape[0] < minimum:
        raise ValueError(f"situations must have shape (n >= {minimum}, S={width}); got {raw.shape}")
    raw = np.ascontiguousarray(raw)
    checked, clipped = deployed.check_situations(raw)
    return raw, np.ascontiguousarray(checked), clipped


def _distinct(situations: np.ndarray) -> tuple[np.ndarray, int]:
    """int64 (n,) index of each row's distinct situation, in order of first appearance, and u."""
    groups: dict[str, int] = {}
    labels = np.array([groups.setdefault(digest, len(groups)) for digest in situation_row_digests(situations)], dtype=np.int64)
    return labels, len(groups)


def _situations_sha256(situations: np.ndarray) -> str:
    """sha256 over dtype, shape and bytes of the float64 situations (n, S)."""
    array = np.ascontiguousarray(situations, dtype=np.float64)
    hasher = hashlib.sha256(_SITUATIONS_DIGEST_TAG)
    hasher.update(f"{array.dtype.str}{array.shape}".encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def _situations_record(raw: np.ndarray, n_distinct: int, clipped: np.ndarray, policy: str) -> dict[str, Any]:
    return {
        "n_rows": int(raw.shape[0]),
        "n_distinct_situations": int(n_distinct),
        "sha256": _situations_sha256(raw),
        "out_of_range_policy": policy,
        "rows_with_clipped_values": int(np.sum(clipped.any(axis=1))),
        "values_clipped": int(clipped.sum()),
    }


def _check_dream_for_student(deployed: DeployedStudent, dreams: Any) -> DreamSet:
    if not isinstance(dreams, DreamSet):
        raise TypeError(f"dreams must be a DreamSet; got {type(dreams).__name__}")
    situation_names = tuple(variable.name for variable in deployed.situation_variables)
    for label, have, want in (
        ("skill", dreams.skill, deployed.skill),
        ("situation_names", dreams.situation_names, situation_names),
        ("action_names", dreams.action_names, deployed.action_names),
        ("n_candidates", dreams.n_candidates, deployed.n_candidates),
    ):
        if have != want:
            raise ValueError(f"DreamSet.{label} {have!r} does not match the student's {want!r}")
    if dreams.n_rows < 1:
        raise ValueError("the DreamSet has no rows; there is nothing to measure")
    return dreams


def _overlap(deployed: DeployedStudent, dreams: DreamSet, labels: np.ndarray, n_distinct: int) -> dict[str, Any]:
    """How many dream rows have a situation the student's optimiser or early stopping saw."""
    seen = deployed.seen_in_training(dreams.situations)
    return {
        "definition": (
            "dream rows whose situation digest (learned_judgment.students.snn.situation_row_digests) is one of the "
            "student's training_situation_digests: every situation its optimiser trained on or its early stopping "
            "evaluated"
        ),
        "n_rows": int(dreams.n_rows),
        "n_distinct_situations": int(n_distinct),
        "rows_seen_in_training": int(np.sum(seen)),
        "distinct_situations_seen_in_training": int(np.unique(labels[seen]).size),
        "n_training_situations": len(deployed.training_situation_digests),
    }


def _overlap_message(overlap: Mapping[str, Any]) -> str:
    return (
        f"{overlap['rows_seen_in_training']} of {overlap['n_rows']} dream rows "
        f"({overlap['distinct_situations_seen_in_training']} of {overlap['n_distinct_situations']} distinct situations) "
        "have a situation this student saw in training (optimiser or early-stopping rows), so they are not held out"
    )


def _plain(value: Any) -> Any:
    """``value`` with mappings as dicts, sequences as lists and numpy scalars as Python numbers."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_copy(label: str, value: Any) -> Any:
    try:
        return json.loads(json.dumps(_plain(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not JSON-serializable with finite numbers: {error}") from error


# --------------------------------------------------------------------------------------
# Bootstrap over distinct situations
# --------------------------------------------------------------------------------------


def _cluster_interval(
    values: np.ndarray,
    labels: np.ndarray,
    n_distinct: int,
    *,
    statistic: str,
    n_resamples: int,
    confidence_level: float,
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap interval of the row mean of ``values`` (n,), resampling distinct situations.

    ``labels`` (n,) int64 gives each row's distinct situation (0..u-1). The u situations are
    resampled with replacement, each with all of its rows; a resample's statistic is
    (sum of its rows' values) / (its row count). ``scipy.stats.bootstrap(paired=True,
    vectorized=True, rng=np.random.default_rng(seed))`` with the conventions of
    :mod:`learned_judgment.eval.bootstrap`: a BCa interval that is not finite (degenerate data,
    e.g. identical values) is recomputed with the percentile method from the same seed, and the
    fallback and every warning are recorded. Intervals computed with the same seed on the same
    labels resample the same situations.

    Raises ValueError when there are fewer than 2 distinct situations.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or labels.shape != values.shape:
        raise ValueError(f"values (n,) and labels (n,) must match; got {values.shape} and {labels.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{statistic}: values must be finite")
    if n_distinct < 2:
        raise ValueError(
            f"{statistic}: the interval resamples distinct situations and needs at least 2; got {n_distinct} distinct "
            f"situation(s) in {values.size} rows"
        )
    sums = np.bincount(labels, weights=values, minlength=n_distinct)
    sizes = np.bincount(labels, minlength=n_distinct).astype(np.float64)

    def row_mean(cluster_sums: np.ndarray, cluster_sizes: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.sum(cluster_sums, axis=axis) / np.sum(cluster_sizes, axis=axis)

    def run(chosen: str) -> tuple[Any, list[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stats.bootstrap(
                (sums, sizes),
                row_mean,
                paired=True,
                vectorized=True,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                method=chosen,
                rng=np.random.default_rng(seed),
            )
        return result, [f"{item.category.__name__}: {item.message}" for item in caught]

    result, caught = run(method)
    low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    used, reason = method, None
    if method == "BCa" and not (np.isfinite(low) and np.isfinite(high)):
        reason = (
            "BCa interval was not finite (degenerate bootstrap distribution); recomputed with "
            f"method={_FALLBACK_METHOD!r} from the same seed"
        )
        result, more = run(_FALLBACK_METHOD)
        caught = caught + more
        used = _FALLBACK_METHOD
        low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    return {
        "statistic": statistic,
        "estimate": float(row_mean(sums, sizes)),
        "low": low,
        "high": high,
        "n_rows": int(values.size),
        "n_distinct_situations": int(n_distinct),
        "resampled_unit": _RESAMPLED_UNIT,
        "n_resamples": int(n_resamples),
        "confidence_level": float(confidence_level),
        "method_requested": method,
        "method_used": used,
        "fallback_reason": reason,
        "warnings": caught,
        "seed": int(seed),
    }


# --------------------------------------------------------------------------------------
# Dream provenance
# --------------------------------------------------------------------------------------


def dream_source(method: str, request_hashes: Sequence[str]) -> str:
    """Which kind of scoring produced a dream; one key of :data:`DREAM_SOURCES`.

    Non-empty ``request_hashes`` -> ``"provider_backed_teacher"``; otherwise ``method ==
    PHYSICS_ONLY`` -> ``"physics_only"``; otherwise ``"local_residual_model"``. A physics-only
    dream that names provider requests is inconsistent and raises ValueError.
    """
    if not isinstance(method, str) or not method:
        raise ValueError(f"method must be a non-empty string; got {method!r}")
    hashes = tuple(request_hashes)
    if hashes:
        if method == PHYSICS_ONLY:
            raise ValueError(f"a {PHYSICS_ONLY!r} dream cannot name provider requests; got {len(hashes)} request hashes")
        return "provider_backed_teacher"
    return "physics_only" if method == PHYSICS_ONLY else "local_residual_model"


def _dream_record(method: str, request_hashes: Sequence[str]) -> dict[str, Any]:
    source = dream_source(method, request_hashes)
    return {
        "method": method,
        "request_hashes": list(request_hashes),
        "source": source,
        "source_description": DREAM_SOURCES[source],
        "provider_backed": source == "provider_backed_teacher",
    }


# --------------------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------------------


def _blas_info() -> list[dict[str, Any]]:
    from threadpoolctl import threadpool_info

    keys = ("user_api", "internal_api", "prefix", "version", "num_threads", "threading_layer", "architecture")
    return [
        {key: info.get(key) for key in keys}
        for info in threadpool_info()
        if info.get("user_api") == "blas"
    ]


def _timing(result: Mapping[str, Any], includes: str, use_for_h3: bool) -> dict[str, Any]:
    return {
        "includes": includes,
        "use_for_h3": use_for_h3,
        "median_s": result["median_s"],
        "p95_s": result["p95_s"],
        "n_decisions": result["n_decisions"],
        "n_inputs": result["n_inputs"],
    }


def measure_decision_latency(student: Any, spec: SkillSpec, situations: np.ndarray, *, n_decisions: int) -> dict[str, Any]:
    """Wall-clock latency of single decisions on the deployed numpy path, on THIS machine.

    The student is bound to ``spec`` once (``DeployedStudent.bind``: the grid check runs there,
    outside every timed call). Then ``n_decisions`` calls cycling through ``situations`` (n, S)
    are timed one by one with :func:`learned_judgment.eval.harness.measure_latency`
    (time.perf_counter_ns, median and p95), for two paths:

        full_path   ``bound.decide_one(situation)`` -> (A,) command values: input checks ->
                    encode_one -> SpikingNetwork.decide_one -> candidate lookup -> bounds check.
                    This is the timing for hypothesis H3.
        index_only  ``bound.decide_index_one(situation)`` -> int: the same without the lookup.

    Before timing, the batched decisions (n,) are computed once; after timing, every situation
    is decided alone on both paths and compared: ``single_row_vs_batch_decisions_differing``
    and ``full_path_actions_not_the_looked_up_candidate`` (the full path's command against
    ``spec.candidate_actions(situation, points_per_dim)[index]``).

    Args:
        student:     a DeployedStudent (or an object whose ``deployed()`` returns one).
        spec:        the student's SkillSpec (checked once at binding; GridMismatchError when it
                     differs from the one the student was trained with).
        situations:  (n >= 1, S) situations, passed to each call as given.
        n_decisions: calls per path (config.yaml evaluation.latency_decisions).

    Returns a JSON-serializable dict: the grid check record, both timings (each with what it
    includes and whether it is the H3 timing), the machine (timer, platform, processor,
    cpu_count, python and numpy versions, BLAS libraries), whether torch was imported in this
    process before and after the timing, the network digest and the situations (rows, distinct
    situations, digest, clipped values). Raises RuntimeError when the timed paths imported torch.
    """
    deployed = _deployed(student)
    n_decisions = _check_count("n_decisions", n_decisions, 1)
    raw, checked, clipped = _situation_arrays(deployed, situations, minimum=1)
    _, n_distinct = _distinct(raw)
    torch_before = "torch" in sys.modules
    started = time.perf_counter_ns()
    bound = deployed.bind(spec)
    bind_seconds = (time.perf_counter_ns() - started) / 1e9
    batch = deployed.decide_index(raw)

    full = measure_latency(bound.decide_one, raw, n_decisions)
    index_only = measure_latency(bound.decide_index_one, raw, n_decisions)

    single = np.array([bound.decide_index_one(row) for row in raw], dtype=np.int64)
    actions = np.stack([bound.decide_one(row) for row in raw])
    looked_up = np.stack(
        [
            np.asarray(spec.candidate_actions(row, deployed.points_per_dim), dtype=np.float64)[index]
            for row, index in zip(checked, single)
        ]
    )
    torch_after = "torch" in sys.modules
    if torch_after and not torch_before:
        raise RuntimeError("the deployed decision path imported torch; it must run on numpy only")
    return {
        "label": "wall-clock latency of the deployed numpy decision path on this machine (a measurement)",
        "grid_check": {
            "ran": "once, at binding, before timing",
            "definition": _GRID_CHECK,
            "bind_seconds": float(bind_seconds),
            "bind_seconds_note": "one call to DeployedStudent.bind; not part of any per-decision timing",
        },
        "full_path": _timing(full, _FULL_PATH, True),
        "index_only": _timing(index_only, _INDEX_PATH, False),
        "h3_timing": "full_path",
        "h3_note": _H3_NOTE,
        "protocol": _LATENCY_PROTOCOL,
        "timer": full["timer"],
        "platform": full["platform"],
        "processor": full["processor"],
        "cpu_count": full["cpu_count"],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "blas": _blas_info(),
        "torch_in_sys_modules_before": torch_before,
        "torch_in_sys_modules_after": torch_after,
        "torch_free_process": not torch_before and not torch_after,
        "skill": deployed.skill,
        "network_sha256": deployed.digest,
        "layer_sizes": list(deployed.network.layer_sizes),
        "time_steps": deployed.network.time_steps,
        "situations": _situations_record(raw, n_distinct, clipped, deployed.out_of_range),
        "single_row_vs_batch_decisions_differing": int(np.sum(single != batch)),
        "full_path_actions_not_the_looked_up_candidate": int(np.sum(np.any(actions != looked_up, axis=1))),
    }


# --------------------------------------------------------------------------------------
# Operations, firing statistics and the two energy estimates
# --------------------------------------------------------------------------------------


def h4_energy_criterion(
    main_estimate_pj: float, with_neuron_updates_estimate_pj: float, dense_reference_estimate_pj: float
) -> dict[str, Any]:
    """The verdict rule of PREREGISTRATION.md's amendment "how H4's energy criterion is judged".

    Args:
        main_estimate_pj:                mean main estimate per decision (pJ).
        with_neuron_updates_estimate_pj: mean estimate including neuron updates per decision (pJ).
        dense_reference_estimate_pj:     estimate of the dense reference per decision (pJ), > 0.

    Returns a JSON-serializable dict: ``verdict_from_estimates`` ("supported" when both
    estimates are below the reference, "not supported" when both are above it, otherwise
    "inconclusive"; a value equal to the reference is neither), where each estimate lies
    against the reference, the values used, the rule text and its scope.
    """
    values = {
        "main_estimate_pj": main_estimate_pj,
        "with_neuron_updates_estimate_pj": with_neuron_updates_estimate_pj,
        "dense_reference_estimate_pj": dense_reference_estimate_pj,
    }
    for name, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a number; got {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite; got {value!r}")
    dense = float(dense_reference_estimate_pj)
    if not dense > 0.0:
        raise ValueError(f"dense_reference_estimate_pj must be > 0; got {dense}")

    def position(value: float) -> str:
        return "below" if value < dense else ("above" if value > dense else "equal")

    main = position(float(main_estimate_pj))
    updates = position(float(with_neuron_updates_estimate_pj))
    if main == updates == "below":
        verdict = "supported"
    elif main == updates == "above":
        verdict = "not supported"
    else:
        verdict = "inconclusive"
    return {
        "verdict_from_estimates": verdict,
        "rule": H4_ENERGY_RULE,
        "scope": _H4_SCOPE,
        "main_estimate_vs_dense_reference_estimate": main,
        "with_neuron_updates_estimate_vs_dense_reference_estimate": updates,
        "main_estimate_pj_per_decision": float(main_estimate_pj),
        "with_neuron_updates_estimate_pj_per_decision": float(with_neuron_updates_estimate_pj),
        "dense_reference_estimate_pj_per_decision": dense,
    }


def _interval_position(interval: Mapping[str, Any], reference: float) -> str:
    if interval["high"] < reference:
        return "entirely below the dense reference estimate"
    if interval["low"] > reference:
        return "entirely above the dense reference estimate"
    return "contains the dense reference estimate"


def _variant(
    definition: str,
    each: np.ndarray,
    mean: float,
    ratio: float,
    interval: dict[str, Any],
    dense: float,
    break_even: dict[str, Any],
) -> dict[str, Any]:
    return {
        "definition": definition,
        "mean_estimate_pj_per_decision": float(mean),
        "estimate_pj_interval": interval,
        "ratio_estimate_to_dense_reference_estimate": float(ratio),
        "ratio_interval": {
            "low": interval["low"] / dense,
            "high": interval["high"] / dense,
            "definition": "estimate_pj_interval divided by the dense reference estimate (a constant per decision)",
        },
        "mean_vs_dense_reference_estimate": "below" if mean < dense else ("above" if mean > dense else "equal"),
        "interval_vs_dense_reference_estimate": _interval_position(interval, dense),
        "break_even": break_even,
        "per_decision_estimate_pj": [float(value) for value in each],
    }


def measure_operations(
    student: Any,
    situations: np.ndarray,
    constants: EnergyConstants,
    *,
    n_resamples: int,
    confidence_level: float,
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Operations of the deployed network per decision on ``situations`` (n, S; at least 2
    distinct situations), and the two energy ESTIMATES built from them.

    Counts come from ``DeployedStudent.operation_counts`` (snn_core.count_operations) on the
    situations after the student's input checks and out_of_range policy. Per-neuron statistics
    come from ``SpikingNetwork.forward_with_spike_trains`` on the same encoded inputs; its spike
    trains must sum to the counted spikes per (decision, layer) (RuntimeError otherwise).

    Energy (learned_judgment.students.energy, constants from config.yaml): the main estimate,
    the estimate including neuron updates, and the dense reference. Each SNN estimate gets its
    mean per decision, a bootstrap interval, its ratio to the dense reference (with the interval
    divided by the reference), whether the mean and the interval lie below or above the
    reference, and its break-even firing rate per step next to the measured rates:

        main estimate          E - E_dense = e_ac T sum_l r_l H_l fo_l - e_mac sum_l H_l fo_l, so the
                               estimate is below the reference exactly when the rate weighted by
                               H_l fo_l is below e_mac / (e_ac T); per layer, connection l's spikes
                               cost less than its dense synapses when r_l < e_mac / (e_ac T)
        with neuron updates    E - E_dense = e_ac T sum_l r_l H_l (fo_l + 1)
                               + (e_mac + e_ac) (sum_l H_l + C) T - e_mac sum_l H_l fo_l, so the
                               estimate is below the reference exactly when the rate weighted by
                               H_l (fo_l + 1) is below
                               (e_mac sum_l H_l fo_l - (e_mac + e_ac)(sum_l H_l + C) T)
                               / (e_ac T sum_l H_l (fo_l + 1)); no rate >= 0 reaches it when that is <= 0

    (r_l: firing rate of hidden layer l per step; fo_l: width of the layer it feeds.) The H4
    energy verdict (:func:`h4_energy_criterion`) is stored with its rule text.

    Args:
        constants: per-operation energies (EnergyConstants.from_config).
        n_resamples, confidence_level, method, seed: bootstrap settings (config.yaml
                   evaluation.bootstrap_*). The four intervals (total spikes, synaptic
                   operations, both energy estimates per decision) resample distinct
                   situations, and the same ones, because each uses ``np.random.default_rng(seed)``.

    Returns a JSON-serializable dict; every energy figure sits under ``energy_estimate`` with
    a name that says it is an estimate, together with energy.CAVEATS.
    """
    deployed = _deployed(student)
    if not isinstance(constants, EnergyConstants):
        raise TypeError(f"constants must be EnergyConstants; got {type(constants).__name__}")
    n_resamples, confidence_level, method, seed = _check_bootstrap_settings(n_resamples, confidence_level, method, seed)
    raw, checked, clipped = _situation_arrays(deployed, situations, minimum=2)
    labels, n_distinct = _distinct(raw)
    if n_distinct < 2:
        raise ValueError(
            f"situations hold {n_distinct} distinct situation(s) in {raw.shape[0]} rows; the intervals resample "
            "distinct situations and need at least 2"
        )
    network = deployed.network
    counts = deployed.operation_counts(checked)
    _, trains = network.forward_with_spike_trains(deployed.encode(checked))
    from_trains = np.stack([train.sum(axis=(0, 2), dtype=np.int64) for train in trains], axis=1).astype(np.float64)
    if not np.array_equal(from_trains, counts.spikes_per_decision):
        raise RuntimeError(
            "the spike trains of SpikingNetwork.forward_with_spike_trains do not sum to the counted spikes per decision; "
            "the firing statistics would not describe the counted operations"
        )

    n = raw.shape[0]
    steps = network.time_steps
    widths = network.hidden_sizes
    per_neuron = [train.sum(axis=(0, 1), dtype=np.int64) for train in trains]  # (H_l,) spikes over all decisions and steps
    never = [int(np.sum(total == 0)) for total in per_neuron]
    every = [int(np.sum(total == n * steps)) for total in per_neuron]
    rates = np.asarray(counts.firing_rate_per_layer, dtype=np.float64)

    energy = estimate_energy(counts, constants)
    e_mac, e_ac = constants.e_mac_pj, constants.e_ac_pj
    spikes_total = counts.spikes_per_decision.sum(axis=1)
    synaptic_ops = counts.synaptic_ops_per_decision
    main_each = np.asarray(energy["snn_main_pj_each"], dtype=np.float64)
    updates_each = main_each + (e_mac + e_ac) * counts.neuron_updates + e_ac * spikes_total
    main_mean = float(energy["snn_main_pj_per_decision"])
    updates_mean = float(energy["snn_with_neuron_updates_pj_per_decision"])
    dense = float(energy["dense_reference_pj_per_decision"])
    if not math.isclose(float(np.mean(updates_each)), updates_mean, rel_tol=1e-12, abs_tol=0.0):
        raise RuntimeError(
            "energy.estimate_energy's with-neuron-updates estimate is not the mean of the per-decision values computed "
            "here from the same counts; the two formulas differ"
        )
    settings = {"n_resamples": n_resamples, "confidence_level": confidence_level, "method": method, "seed": seed}

    width_array = np.asarray(widths, dtype=np.float64)
    fan_out = np.asarray(counts.fan_out, dtype=np.float64)
    main_weights = width_array * fan_out
    updates_weights = width_array * (fan_out + 1.0)
    main_break_even = e_mac / (e_ac * steps)
    updates_break_even = (e_mac * float(np.sum(main_weights)) - (e_mac + e_ac) * counts.neuron_updates) / (
        e_ac * steps * float(np.sum(updates_weights))
    )
    main_weighted = float(np.sum(rates * main_weights) / np.sum(main_weights))
    updates_weighted = float(np.sum(rates * updates_weights) / np.sum(updates_weights))
    rate_list = [float(value) for value in rates]
    main_break = {
        "rate_per_step": float(main_break_even),
        "formula": "e_mac / (e_ac * T)",
        "measured_rate_per_layer": rate_list,
        "weights_per_layer": [float(value) for value in main_weights],
        "weights_definition": "H_l x fo_l (synapses from hidden layer l to the layer it feeds)",
        "measured_weighted_rate_per_step": main_weighted,
        "weighted_rate_below_break_even": bool(main_weighted < main_break_even),
        "per_layer": [
            {"layer": layer, "firing_rate_per_step": rate, "below_break_even": bool(rate < main_break_even)}
            for layer, rate in enumerate(rate_list)
        ],
        "definition": (
            "under the main estimate hidden layer l's spikes cost e_ac per synapse per spike and its dense synapses cost "
            "e_mac once per decision, so connection l costs less than its dense counterpart when its firing rate per step "
            "is below e_mac / (e_ac T); the estimate is below the dense reference exactly when the H_l x fo_l weighted "
            "rate is"
        ),
    }
    updates_break = {
        "rate_per_step": float(updates_break_even),
        "formula": "(e_mac * sum_l H_l fo_l - (e_mac + e_ac) * (sum_l H_l + C) * T) / (e_ac * T * sum_l H_l (fo_l + 1))",
        "measured_rate_per_layer": rate_list,
        "weights_per_layer": [float(value) for value in updates_weights],
        "weights_definition": "H_l x (fo_l + 1) (synapses from hidden layer l plus one reset subtraction per spike)",
        "measured_weighted_rate_per_step": updates_weighted,
        "weighted_rate_below_break_even": bool(updates_weighted < updates_break_even),
        "reachable": bool(updates_break_even > 0.0),
        "definition": (
            "the estimate including neuron updates is below the dense reference exactly when the H_l x (fo_l + 1) "
            "weighted firing rate per step is below this rate; when the rate is <= 0 the neuron updates alone cost at "
            "least as much as the dense reference's synapses after the first layer, so no firing rate reaches it"
        ),
    }

    variants = {
        "main_estimate": _variant(
            "e_mac x first-layer MACs (D x H_1, once per decision) + e_ac x synaptic operations (spikes x fan-out)",
            main_each,
            main_mean,
            float(energy["ratio_snn_main_to_dense"]),
            _cluster_interval(main_each, labels, n_distinct, statistic="mean main energy estimate per decision (pJ)", **settings),
            dense,
            main_break,
        ),
        "with_neuron_updates_estimate": _variant(
            "main estimate + (e_mac + e_ac) x neuron updates ((sum_l H_l + C) x T) + e_ac x spikes (one reset "
            "subtraction each)",
            updates_each,
            updates_mean,
            float(energy["ratio_snn_with_neuron_updates_to_dense"]),
            _cluster_interval(
                updates_each, labels, n_distinct, statistic="mean energy estimate with neuron updates per decision (pJ)", **settings
            ),
            dense,
            updates_break,
        ),
    }
    return {
        "label": (
            "operation counts measured by running the deployed numpy network on the given situations; every energy "
            "figure is an estimate from those counts, not a measurement"
        ),
        "skill": deployed.skill,
        "network_sha256": deployed.digest,
        "situations": _situations_record(raw, n_distinct, clipped, deployed.out_of_range),
        "layer_sizes": list(counts.layer_sizes),
        "time_steps": int(steps),
        "fan_out": list(counts.fan_out),
        "first_layer_macs": int(counts.first_layer_macs),
        "neuron_updates": int(counts.neuron_updates),
        "dense_reference_macs": int(counts.dense_reference_macs),
        "spikes_per_decision": {
            "mean_per_layer": [float(value) for value in counts.mean_spikes_per_layer],
            "mean_total": float(np.mean(spikes_total)),
            "interval_total": _cluster_interval(
                spikes_total, labels, n_distinct, statistic="mean spikes per decision (all hidden layers)", **settings
            ),
        },
        "synaptic_ops_per_decision": {
            "definition": "sum over hidden layers of spikes emitted x fan-out (the width of the layer they feed)",
            "mean": float(np.mean(synaptic_ops)),
            "interval": _cluster_interval(
                synaptic_ops, labels, n_distinct, statistic="mean synaptic operations per decision", **settings
            ),
        },
        "firing": {
            "source": "per-neuron spike trains of SpikingNetwork.forward_with_spike_trains on the counted inputs",
            "neurons_per_layer": [int(width) for width in widths],
            "rate_per_layer": rate_list,
            "rate_definition": "mean fraction of a hidden layer's neurons firing per time step, over the situations",
            "never_fire_count_per_layer": never,
            "never_fire_fraction_per_layer": [count / float(width) for count, width in zip(never, widths)],
            "fire_every_step_count_per_layer": every,
            "fire_every_step_fraction_per_layer": [count / float(width) for count, width in zip(every, widths)],
            "neuron_definitions": (
                "never fire: no spike at any step on any of these situations; fire every step: a spike at every "
                "step on every one of these situations"
            ),
        },
        "energy_estimate": {
            "label": "estimate, 45 nm CMOS, arithmetic only; not a measurement",
            "constants": {"e_mac_pj": e_mac, "e_ac_pj": e_ac, "precision": constants.precision, "source": constants.source},
            "first_layer_counted": "once per decision",
            "dense_reference_estimate": {
                "definition": (
                    "e_mac x sum over layers of fan_in x fan_out: one pass through a non-spiking dense network with the "
                    "same layer sizes (no such network was trained)"
                ),
                "macs": int(counts.dense_reference_macs),
                "estimate_pj_per_decision": dense,
            },
            **variants,
            "h4_energy_criterion": h4_energy_criterion(main_mean, updates_mean, dense),
            "caveats": list(CAVEATS),
        },
        "per_decision": {
            "spikes_per_layer": [[float(value) for value in row] for row in counts.spikes_per_decision],
            "synaptic_ops": [float(value) for value in synaptic_ops],
            "situation_group": [int(value) for value in labels],
        },
        "spike_train_check": {
            "spike_trains_sum_to_counted_spikes_per_decision_and_layer": True,
            "source": "SpikingNetwork.forward_with_spike_trains against snn_core.count_operations on the same inputs",
        },
        "caveats": list(CAVEATS),
    }


# --------------------------------------------------------------------------------------
# Agreement with dreams the student did not see
# --------------------------------------------------------------------------------------


def _regret_summary(regret: np.ndarray, spread: float | None) -> dict[str, Any]:
    summary = regression_summary(regret)  # mean, median, p90 (numpy 'linear'), n
    record: dict[str, Any] = {
        "mean": summary["mean"],
        "median": summary["median"],
        "p90": summary["p90"],
        "n": summary["n"],
        "fraction_zero": float(np.mean(regret == 0.0)),
    }
    if spread is None:
        record["over_spread"] = UNDEFINED
    else:
        record["over_spread"] = {key: summary[key] / spread for key in ("mean", "median", "p90")}
    return record


def agreement_with_dream(student: Any, dreams: DreamSet, *, allow_seen: bool = False) -> dict[str, Any]:
    """Decision quality of a deployed student INSIDE ``dreams``.

    First the overlap with training: rows whose situation the student's optimiser or early
    stopping saw (``DeployedStudent.seen_in_training``). With any such row this raises
    :class:`HeldOutOverlapError` (stating the counts) unless ``allow_seen`` is True; then the
    record's label starts with :data:`SEEN_IN_TRAINING_LABEL` and ``held_out`` is False.

    For the n dream rows: the student's decisions (n,) = ``deployed.decide_index(dreams.situations)``;
    agreement with the dream's choice ``dreams.chosen`` (the first best-scored candidate, so
    ties make agreement understate quality; ``fraction_zero`` regret counts every best
    candidate); dream regret (targets.dream_regret: row max of ``dreams.scores`` minus the
    score of the decision) as mean / median / p90 in objective units and divided by the score
    spread of these dreams (targets.score_spread; ``"undefined"`` when that spread is 0). The
    same regret is reported for ``dreams.physics_choice``, the "no learning" reference. The
    number of distinct situations is reported next to the row count, and ``per_row`` carries
    each row's distinct-situation index for intervals that resample situations.

    The dream's candidate grid cannot be compared with the spec here (no spec is given);
    :func:`student_record` does that row by row.

    Returns a JSON-serializable dict including the per-row decisions and regrets.
    """
    deployed = _deployed(student)
    dreams = _check_dream_for_student(deployed, dreams)
    if not isinstance(allow_seen, bool):
        raise TypeError(f"allow_seen must be True or False; got {allow_seen!r}")
    labels, n_distinct = _distinct(dreams.situations)
    overlap = _overlap(deployed, dreams, labels, n_distinct)
    if overlap["rows_seen_in_training"] and not allow_seen:
        raise HeldOutOverlapError(
            f"{_overlap_message(overlap)}. Pass dreams made from situations the student never saw, or allow_seen=True to "
            f"measure anyway with the record labelled {SEEN_IN_TRAINING_LABEL!r}"
        )
    held_out = overlap["rows_seen_in_training"] == 0
    _, clipped = deployed.check_situations(dreams.situations)
    decisions = deployed.decide_index(dreams.situations)
    student_regret = dream_regret(dreams.scores, decisions)
    physics_regret = dream_regret(dreams.scores, dreams.physics_choice)
    try:
        spread: float | None = score_spread(dreams.scores)
        spread_reason = None
    except ValueError as error:
        spread, spread_reason = None, str(error)
    n = dreams.n_rows
    inside = (
        "measured inside the dream: dreamed objective scores of the dream's candidates, not outcomes in the simulator "
        "or on the robot"
    )
    label = (
        f"held out: no row's situation was seen in training; {inside}"
        if held_out
        else f"{SEEN_IN_TRAINING_LABEL}: {_overlap_message(overlap)}; {inside}"
    )
    return {
        "label": label,
        "held_out": held_out,
        "includes_situations_seen_in_training": not held_out,
        "overlap_with_training": {**overlap, "allow_seen": allow_seen},
        "dream": {
            "skill": dreams.skill,
            **_dream_record(dreams.method, dreams.request_hashes),
            "n_rows": int(n),
            "n_distinct_situations": int(n_distinct),
            "n_candidates": int(dreams.n_candidates),
            "rows_sha256": dream_content_sha256(dreams, np.arange(n)),
            "out_of_range_policy": deployed.out_of_range,
            "rows_with_clipped_values": int(np.sum(clipped.any(axis=1))),
        },
        "network_sha256": deployed.digest,
        "score_spread": UNDEFINED if spread is None else float(spread),
        "score_spread_definition": "targets.score_spread of these dreams' scores: median over rows of the row std",
        "score_spread_undefined_reason": spread_reason,
        "student": {
            "agreement_with_dream_choice": float(np.mean(decisions == dreams.chosen)),
            "agreement_with_physics_choice": float(np.mean(decisions == dreams.physics_choice)),
            "dream_regret": _regret_summary(student_regret, spread),
        },
        "physics_only_choice": {
            "agreement_with_dream_choice": float(np.mean(dreams.physics_choice == dreams.chosen)),
            "dream_regret": _regret_summary(physics_regret, spread),
        },
        "paired_rows": {
            "student_regret_lower": float(np.mean(student_regret < physics_regret)),
            "equal": float(np.mean(student_regret == physics_regret)),
            "student_regret_higher": float(np.mean(student_regret > physics_regret)),
        },
        "per_row": {
            "student_choice": [int(value) for value in decisions],
            "student_regret": [float(value) for value in student_regret],
            "physics_regret": [float(value) for value in physics_regret],
            "situation_group": [int(value) for value in labels],
        },
    }


# --------------------------------------------------------------------------------------
# One record per deployed student
# --------------------------------------------------------------------------------------


def student_record(
    student: Any,
    spec: SkillSpec,
    dreams_heldout: DreamSet,
    situations: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Everything a report needs about one deployed spiking student, in one JSON record.

    Args:
        student:        a DeployedStudent: ``snn_student.deployed()`` (or the per-skill
                        SNNStudent itself) or ``shared.deployed(skill)``. Its ``meta`` must hold
                        ``"dream"`` with the training dream's ``"method"`` and ``"request_hashes"``
                        (``"rows_sha256"`` is recorded when present).
        spec:           the student's SkillSpec; ``DeployedStudent.check_spec`` must accept it
                        (name, situation encoding, action records, candidate-grid digest).
        dreams_heldout: dreams of ``spec`` made on the student's candidate grid (checked row by
                        row) whose situations the student never saw: any row whose situation is
                        one of the student's training or early-stopping situations raises
                        :class:`HeldOutOverlapError`, which states how many rows overlap.
        situations:     (n, S) situations, at least 2 distinct, for latency and operations.
        config:         loaded config.yaml; read: evaluation.latency_decisions,
                        evaluation.bootstrap_resamples, evaluation.bootstrap_confidence_level,
                        evaluation.bootstrap_method, students.snn.energy.*.
        seed:           run seed; the bootstrap seeds come from the named streams
                        "<skill>/measure/regret_bootstrap" and "<skill>/measure/operations_bootstrap".

    The record holds: the network (digest, layer sizes, T, neuron constants), the student's
    grid and encoding records, settings and provenance from ``meta``; whether the training and
    held-out dreams came from a provider-backed teacher, a local residual model or physics only;
    the latency, operations (both energy estimates and the H4 energy verdict) and in-dream
    agreement measurements; and a bootstrap interval, resampling distinct held-out situations,
    of the mean dream regret of the student minus that of the physics-only choice. Decision
    quality on actual Go2 episodes is not in it (learned_judgment.eval.harness measures that).
    """
    deployed = _deployed(student)
    seed = _check_count("seed", seed, 0)
    deployed.check_spec(spec)
    dreams = _check_dream_for_student(deployed, dreams_heldout)
    try:
        check_dreams(spec, dreams, deployed.points_per_dim)
    except ValueError as error:
        raise ValueError(f"dreams_heldout do not fit the student's spec and candidate grid: {error}") from error
    labels, n_distinct = _distinct(dreams.situations)
    overlap = _overlap(deployed, dreams, labels, n_distinct)
    if overlap["rows_seen_in_training"]:
        raise HeldOutOverlapError(f"dreams_heldout: {_overlap_message(overlap)}; pass dreams made from situations the student never saw")
    training_dream = deployed.meta.get("dream")
    if not isinstance(training_dream, Mapping) or "method" not in training_dream or "request_hashes" not in training_dream:
        raise ValueError(
            "the student's meta['dream'] with 'method' and 'request_hashes' is missing: the dream it was trained in is "
            "unknown, so it cannot be recorded as a dream-trained student"
        )

    values = {
        "evaluation.latency_decisions": require(config, "evaluation.latency_decisions"),
        "evaluation.bootstrap_resamples": require(config, "evaluation.bootstrap_resamples"),
        "evaluation.bootstrap_confidence_level": require(config, "evaluation.bootstrap_confidence_level"),
        "evaluation.bootstrap_method": require(config, "evaluation.bootstrap_method"),
    }
    constants = EnergyConstants.from_config(config)
    n_resamples, confidence_level, method, _ = _check_bootstrap_settings(
        values["evaluation.bootstrap_resamples"],
        values["evaluation.bootstrap_confidence_level"],
        values["evaluation.bootstrap_method"],
        0,
    )
    bootstrap = {"n_resamples": n_resamples, "confidence_level": confidence_level, "method": method}
    regret_stream = f"{deployed.skill}/measure/regret_bootstrap"
    operations_stream = f"{deployed.skill}/measure/operations_bootstrap"
    regret_seed = stream_seed(seed, regret_stream)
    operations_seed = stream_seed(seed, operations_stream)

    agreement = agreement_with_dream(deployed, dreams)
    latency = measure_decision_latency(deployed, spec, situations, n_decisions=values["evaluation.latency_decisions"])
    student_regret = np.asarray(agreement["per_row"]["student_regret"], dtype=np.float64)
    physics_regret = np.asarray(agreement["per_row"]["physics_regret"], dtype=np.float64)
    regret_difference = _cluster_interval(
        student_regret - physics_regret,
        labels,
        n_distinct,
        statistic="mean dream regret of the student minus that of the physics-only choice",
        seed=regret_seed,
        **bootstrap,
    )
    operations = measure_operations(deployed, situations, constants, seed=operations_seed, **bootstrap)

    training_source = _dream_record(str(training_dream["method"]), tuple(training_dream["request_hashes"]))
    heldout_source = {
        key: agreement["dream"][key]
        for key in ("method", "request_hashes", "source", "source_description", "provider_backed", "n_rows", "n_distinct_situations", "rows_sha256")
    }
    network = deployed.network
    meta = deployed.meta
    record = {
        "format": RECORD_FORMAT,
        "label": (
            "Measurements of one deployed dream-trained spiking student from runs of its numpy network. Decision "
            "quality is measured inside held-out dreams (dreamed objective scores), not on the robot or in the "
            "simulator; energy figures are estimates; latency is wall-clock time on this machine."
        ),
        "skill": deployed.skill,
        "student": {
            "kind": _json_copy("the student's meta['kind']", meta.get("kind")),
            "network_sha256": deployed.digest,
            "layer_sizes": list(network.layer_sizes),
            "hidden_sizes": list(network.hidden_sizes),
            "n_inputs": network.n_inputs,
            "n_inputs_note": (
                "the network's input width D; larger than the situation encoding's width when the network is one skill's "
                "view of a shared network (padding and skill-slot inputs)"
            ),
            "n_candidates": deployed.n_candidates,
            "time_steps": network.time_steps,
            "betas": list(network.betas),
            "thresholds": list(network.thresholds),
            "readout_beta": network.readout_beta,
            "points_per_dim": list(deployed.points_per_dim),
            "action_names": list(deployed.action_names),
            "action_records": _json_copy("the student's action records", deployed.action_records),
            "situation_encoding": InputEncoder.from_variables(deployed.situation_variables).describe(),
            "grid_sha256": deployed.grid_sha256,
            "n_grid_probe_situations": int(deployed.grid_probe.shape[0]),
            "n_training_situations": len(deployed.training_situation_digests),
            "out_of_range_policy": deployed.out_of_range,
            "settings": _json_copy("the student's meta['settings']", meta.get("settings")),
        },
        "dream_provenance": {
            "training_dream": {**training_source, "rows_sha256": _json_copy("rows_sha256", training_dream.get("rows_sha256"))},
            "heldout_dream": heldout_source,
            "heldout_overlap_with_training": overlap,
            "trained_in_provider_backed_dream": training_source["provider_backed"],
            "heldout_same_method_as_training": heldout_source["method"] == training_source["method"],
        },
        "agreement_with_heldout_dream": agreement,
        "dream_regret_difference_student_minus_physics": {
            "label": (
                "over held-out dream rows, resampling distinct held-out situations: mean dream regret of the student minus "
                "that of the physics-only choice; negative = the student gives up less dreamed objective. Inside the "
                "dream, not the simulator"
            ),
            "stream": regret_stream,
            **regret_difference,
        },
        "latency": latency,
        "operations": operations,
        "seeds": {
            "seed": seed,
            "regret_bootstrap_stream": regret_stream,
            "regret_bootstrap_seed": int(regret_seed),
            "operations_bootstrap_stream": operations_stream,
            "operations_bootstrap_seed": int(operations_seed),
        },
        "config": {
            "config_sha256": config_digest(config),
            "values_read": {
                **values,
                "students.snn.energy.e_mac_pj": constants.e_mac_pj,
                "students.snn.energy.e_ac_pj": constants.e_ac_pj,
                "students.snn.energy.precision": constants.precision,
                "students.snn.energy.source": constants.source,
            },
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "student_meta": _json_copy("the student's meta", meta),
        "out_of_scope": (
            "decision quality on actual Go2 episodes is measured by learned_judgment.eval.harness on the simulator, "
            "not in this record"
        ),
    }
    return _json_copy("the student record", record)


def _canonical_json(record: Any) -> str:
    try:
        text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"record is not JSON-serializable with finite numbers: {error}") from error
    return text + "\n"


def write_record(path: str | Path, record: dict) -> None:
    """Write ``record`` as canonical JSON (sorted keys, no whitespace, UTF-8, finite numbers,
    one trailing newline) to ``path`` atomically (learned_judgment.fileio.atomic_write_text)."""
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict; got {type(record).__name__}")
    atomic_write_text(path, _canonical_json(record))
