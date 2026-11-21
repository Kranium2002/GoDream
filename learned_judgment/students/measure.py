"""Measurement of one dream-trained spiking student (Person B, students lane C).

Lane contract
-------------
The physics guess corrected by the PFN is a dream world: for every dreamed situation it gives
the dreamed objective score of every candidate command (:class:`~learned_judgment.engine.dream.DreamSet`).
A spiking student is trained inside that dream to choose the best candidate
(:mod:`learned_judgment.students.snn`). This module measures such a student and returns
JSON-serializable records that name their inputs (network digest, digests of the situations
and dream rows, seeds, the config values read). Every number comes from actual runs of the
DEPLOYED numpy network (:class:`~learned_judgment.students.snn_core.SpikingNetwork`); importing
this module or calling any function in it never imports torch.

    measure_decision_latency  wall-clock latency of single decisions on the deployed path,
                              timed with :func:`learned_judgment.eval.harness.measure_latency`
    measure_operations        spikes, synaptic operations and firing statistics per decision,
                              the energy ESTIMATE (:mod:`learned_judgment.students.energy`) and
                              bootstrap intervals that resample decisions
    agreement_with_dream      decision quality INSIDE held-out dreams: agreement with the dream's
                              choice and dream regret, next to the physics-only choice stored in
                              the same DreamSet (the "no learning" reference)
    student_record            the three measurements plus provenance; every tunable is read from
                              config.yaml (evaluation.*, students.snn.energy.*)
    write_record              canonical JSON written atomically
    dream_source              which kind of scoring produced a dream (provider-backed teacher,
                              local residual model without a provider request, physics only)

Out of scope: decision quality on actual Go2 episodes is measured by
:mod:`learned_judgment.eval.harness` on the simulator, not here. Quality numbers in these
records are statements about dreamed objective scores; energy numbers are estimates.

Shapes (float64 unless noted)::

    n situations or dream rows   S situation features   D encoded inputs   C candidates
    L hidden layers              H_l hidden widths       T time steps
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
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
from learned_judgment.eval.bootstrap import METHODS, paired_difference_ci
from learned_judgment.eval.harness import measure_latency
from learned_judgment.eval.metrics import UNDEFINED, regression_summary
from learned_judgment.fileio import atomic_write_text
from learned_judgment.skills.base import SkillSpec
from learned_judgment.students.encoding import InputEncoder
from learned_judgment.students.energy import CAVEATS, EnergyConstants, estimate_energy
from learned_judgment.students.snn import SNNStudent, _dream_content_sha256
from learned_judgment.students.snn_core import SpikingNetwork
from learned_judgment.students.targets import dream_regret, score_spread

__all__ = [
    "DREAM_SOURCES",
    "RECORD_FORMAT",
    "agreement_with_dream",
    "dream_source",
    "measure_decision_latency",
    "measure_operations",
    "student_record",
    "write_record",
]

RECORD_FORMAT = "learned-judgment/snn-student-measurement/v1"

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

# Structural rules (not tunables): the fallback when a BCa interval is not finite, the same
# rule as learned_judgment.eval.bootstrap; and the formats of the digests recorded here.
_FALLBACK_METHOD = "percentile"
_SITUATIONS_DIGEST_TAG = b"learned-judgment/situations/v1"
_DECISION_PATH = (
    "SNNStudent.decide_index on one situation reshaped to (1, S): finite check -> InputEncoder.encode -> "
    "SpikingNetwork.decide (numpy) -> candidate index; the action lookup spec.candidate_actions is not timed"
)
_LATENCY_PROTOCOL = (
    "learned_judgment.eval.harness.measure_latency: call i decides situation i mod n; each call is timed "
    "alone with time.perf_counter_ns; no warm-up calls; median and p95 (numpy 'linear' quantile) over all calls"
)


# --------------------------------------------------------------------------------------
# Input checks and digests
# --------------------------------------------------------------------------------------


def _check_student(student: Any) -> SNNStudent:
    if not isinstance(student, SNNStudent):
        raise TypeError(f"student must be an SNNStudent; got {type(student).__name__}")
    return student


def _check_situations(student: SNNStudent, situations: Any, *, minimum: int) -> np.ndarray:
    array = np.asarray(situations, dtype=np.float64)
    width = len(student.encoder.variables)
    if array.ndim != 2 or array.shape[1] != width or array.shape[0] < minimum:
        raise ValueError(f"situations must have shape (n >= {minimum}, S={width}); got {array.shape}")
    if not np.all(np.isfinite(array)):
        row, column = np.argwhere(~np.isfinite(array))[0]
        raise ValueError(
            f"situations contain non-finite values (first at row {row}, feature "
            f"{student.encoder.variables[column].name!r})"
        )
    return np.ascontiguousarray(array)


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


def _situations_sha256(situations: np.ndarray) -> str:
    """sha256 over dtype, shape and bytes of the float64 situations (n, S)."""
    array = np.ascontiguousarray(situations, dtype=np.float64)
    hasher = hashlib.sha256(_SITUATIONS_DIGEST_TAG)
    hasher.update(f"{array.dtype.str}{array.shape}".encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def _trained_action_names(student: SNNStudent) -> tuple[str, ...]:
    record = student.meta.get("student")
    if not isinstance(record, Mapping) or "action_names" not in record:
        raise ValueError("student.meta['student']['action_names'] is missing: the student's actions are unknown")
    return tuple(str(name) for name in record["action_names"])


def _check_spec(student: SNNStudent, spec: Any) -> SkillSpec:
    """Refuse a spec whose name, situation encoding or action names differ from the student's."""
    if not isinstance(spec, SkillSpec):
        raise TypeError(f"spec must be a SkillSpec; got {type(spec).__name__}")
    if spec.name != student.skill:
        raise ValueError(f"the student was trained for skill {student.skill!r}, not {spec.name!r}")
    if InputEncoder.from_variables(spec.situation).describe() != student.encoder.describe():
        raise ValueError(f"spec {spec.name!r} encodes its situation differently from the student's encoder")
    trained = _trained_action_names(student)
    if spec.action_names != trained:
        raise ValueError(f"spec actions {spec.action_names} differ from the student's trained actions {trained}")
    return spec


def _check_dream_for_student(student: SNNStudent, dreams: Any) -> DreamSet:
    if not isinstance(dreams, DreamSet):
        raise TypeError(f"dreams must be a DreamSet; got {type(dreams).__name__}")
    situation_names = tuple(variable.name for variable in student.encoder.variables)
    for label, have, want in (
        ("skill", dreams.skill, student.skill),
        ("situation_names", dreams.situation_names, situation_names),
        ("action_names", dreams.action_names, _trained_action_names(student)),
        ("n_candidates", dreams.n_candidates, student.n_candidates),
    ):
        if have != want:
            raise ValueError(f"DreamSet.{label} {have!r} does not match the student's {want!r}")
    if dreams.n_rows < 1:
        raise ValueError("the DreamSet has no rows; there is nothing to measure")
    return dreams


def _check_dream_grid(spec: SkillSpec, student: SNNStudent, dreams: DreamSet) -> None:
    """Every dream row's candidates must be the grid the student decides over."""
    for label, have, want in (
        ("outcome_names", dreams.outcome_names, spec.outcome_names),
        ("primary_outcome", dreams.primary_outcome, spec.primary_outcome),
    ):
        if have != want:
            raise ValueError(f"DreamSet.{label} {have!r} does not match the spec's {want!r}")
    for row in range(dreams.n_rows):
        grid = np.asarray(spec.candidate_actions(dreams.situations[row], student.points_per_dim), dtype=np.float64)
        if grid.shape != dreams.candidates.shape[1:] or not np.array_equal(grid, dreams.candidates[row]):
            raise ValueError(
                f"dream row {row}: the dream's candidates are not spec.candidate_actions(situation, "
                f"points_per_dim={list(student.points_per_dim)}), the grid the student decides over"
            )


def _json_copy(label: str, value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not JSON-serializable with finite numbers: {error}") from error


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


def measure_decision_latency(student: SNNStudent, situations: np.ndarray, *, n_decisions: int) -> dict[str, Any]:
    """Wall-clock latency of single decisions on the deployed numpy path, on THIS machine.

    One decision: situation (S,) -> ``student.decide_index(situation[None, :])`` (finite check,
    InputEncoder.encode, SpikingNetwork.decide) -> int candidate index. ``n_decisions`` calls
    cycle through ``situations`` (n, S) and are timed one by one by
    :func:`learned_judgment.eval.harness.measure_latency` (time.perf_counter_ns, median and p95).

    Before timing, the batched decisions (n,) are computed once; after timing, every situation
    is decided alone and compared with them (``single_row_vs_batch_decisions_differing``).

    Returns a JSON-serializable dict: the harness result (median_s, p95_s, n_decisions,
    n_inputs, timer, platform, processor, cpu_count) plus python and numpy versions, BLAS
    libraries, the network digest, the situations digest and whether torch was imported in
    this process before and after the timing. Raises RuntimeError when the timed path itself
    imported torch.
    """
    student = _check_student(student)
    n_decisions = _check_count("n_decisions", n_decisions, 1)
    array = _check_situations(student, situations, minimum=1)
    torch_before = "torch" in sys.modules
    batch = student.decide_index(array)

    def decide_one(situation: np.ndarray) -> int:
        return int(student.decide_index(situation[None, :])[0])

    timing = measure_latency(decide_one, array, n_decisions)
    single = np.array([decide_one(situation) for situation in array], dtype=np.int64)
    torch_after = "torch" in sys.modules
    if torch_after and not torch_before:
        raise RuntimeError("the deployed decision path imported torch; it must run on numpy only")
    return {
        "label": "wall-clock latency of the deployed numpy decision path on this machine (a measurement)",
        "decision_path": _DECISION_PATH,
        "protocol": _LATENCY_PROTOCOL,
        **timing,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "blas": _blas_info(),
        "torch_in_sys_modules_before": torch_before,
        "torch_in_sys_modules_after": torch_after,
        "torch_free_process": not torch_before and not torch_after,
        "network_sha256": student.digest,
        "layer_sizes": list(student.network.layer_sizes),
        "time_steps": student.network.time_steps,
        "situations": {"n": int(array.shape[0]), "sha256": _situations_sha256(array)},
        "single_row_vs_batch_decisions_differing": int(np.sum(single != batch)),
    }


# --------------------------------------------------------------------------------------
# Operations, firing statistics and the energy estimate
# --------------------------------------------------------------------------------------


def _spikes_per_neuron(network: SpikingNetwork, encoded: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """The forward pass of :class:`SpikingNetwork` (same expressions, same order) keeping each
    neuron's spikes, which snn_core does not expose.

    Args:
        encoded: (n, D) encoded inputs.

    Returns:
        per_neuron:  L arrays (n, H_l): spikes of each neuron summed over the T steps.
        step_counts: int64 (L, T, n): spikes of each layer at each step (snn_core's layout).
        logits:      (n, C).
    """
    n = encoded.shape[0]
    layers = len(network.hidden_weights)
    first_current = encoded @ network.hidden_weights[0].T + network.hidden_biases[0]
    membranes = [np.zeros((n, size)) for size in network.hidden_sizes]
    per_neuron = [np.zeros((n, size)) for size in network.hidden_sizes]
    step_counts = np.zeros((layers, network.time_steps, n), dtype=np.int64)
    readout = np.zeros((n, network.n_outputs))
    total = np.zeros((n, network.n_outputs))
    for step in range(network.time_steps):
        current = first_current
        spikes = None
        for layer in range(layers):
            if layer > 0:
                current = spikes @ network.hidden_weights[layer].T + network.hidden_biases[layer]
            threshold = network.thresholds[layer]
            reset = (membranes[layer] > threshold).astype(np.float64)
            membranes[layer] = (network.betas[layer] * membranes[layer] + current) - reset * threshold
            spikes = (membranes[layer] > threshold).astype(np.float64)
            per_neuron[layer] += spikes
            step_counts[layer, step] = spikes.sum(axis=1).astype(np.int64)
        readout = network.readout_beta * readout + (spikes @ network.readout_weight.T + network.readout_bias)
        total = total + readout
    return per_neuron, step_counts, total / float(network.time_steps)


def _mean_interval(
    values: np.ndarray, *, n_resamples: int, confidence_level: float, method: str, seed: int
) -> dict[str, Any]:
    """One-sample bootstrap interval of the mean of ``values`` (n,), resampling its entries.

    ``scipy.stats.bootstrap(..., vectorized=True, rng=np.random.default_rng(seed))`` with the
    conventions of :mod:`learned_judgment.eval.bootstrap`: a BCa interval that is not finite
    (degenerate data, e.g. identical values) is recomputed with the percentile method from the
    same seed, and the fallback and every warning are recorded.
    """
    data = (np.asarray(values, dtype=np.float64),)

    def run(chosen: str) -> tuple[Any, list[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stats.bootstrap(
                data,
                np.mean,
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
        "statistic": "mean over decisions",
        "estimate": float(np.mean(data[0])),
        "low": low,
        "high": high,
        "n": int(data[0].size),
        "n_resamples": int(n_resamples),
        "confidence_level": float(confidence_level),
        "method_requested": method,
        "method_used": used,
        "fallback_reason": reason,
        "warnings": caught,
        "seed": int(seed),
    }


def measure_operations(
    student: SNNStudent,
    situations: np.ndarray,
    constants: EnergyConstants,
    *,
    n_resamples: int,
    confidence_level: float,
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Operations of the deployed network per decision on ``situations`` (n >= 2, S), and the
    energy ESTIMATE built from them.

    Counts come from ``student.operation_counts`` (snn_core.count_operations). Per-neuron
    statistics come from a replica of the forward pass that is checked, on every call, to
    reproduce snn_core's spike counts per (layer, step, decision) exactly (RuntimeError if not).

    Args:
        constants:        per-operation energies (EnergyConstants.from_config).
        n_resamples, confidence_level, method, seed: bootstrap settings (config.yaml
                          evaluation.bootstrap_*); the three intervals (total spikes, synaptic
                          operations, main energy estimate per decision) resample the same
                          decisions because each uses ``np.random.default_rng(seed)``.

    Returns a JSON-serializable dict labelled "estimate" carrying energy.CAVEATS: layer sizes,
    T, fan-out, spikes and synaptic operations per decision (means, intervals, per-decision
    values), firing rate per hidden layer, the fraction of hidden neurons that never fire and
    that fire at every step on these situations, the break-even firing rate per step
    e_mac / (e_ac * T) next to the measured rates, and :func:`estimate_energy`'s result.
    """
    student = _check_student(student)
    if not isinstance(constants, EnergyConstants):
        raise TypeError(f"constants must be EnergyConstants; got {type(constants).__name__}")
    n_resamples, confidence_level, method, seed = _check_bootstrap_settings(n_resamples, confidence_level, method, seed)
    array = _check_situations(student, situations, minimum=2)
    network = student.network
    encoded = student.encoder.encode(array)

    counts = student.operation_counts(array)
    per_neuron, step_counts, replica_logits = _spikes_per_neuron(network, encoded)
    reference_logits, reference_counts = network.forward_with_spikes(encoded)
    replica_totals = np.stack([layer.sum(axis=1) for layer in per_neuron], axis=1)
    if not np.array_equal(step_counts, reference_counts) or not np.array_equal(
        replica_totals, counts.spikes_per_decision
    ):
        raise RuntimeError(
            "the per-neuron replica of the forward pass does not reproduce snn_core's spike counts; "
            "the firing statistics would not describe the deployed network"
        )

    energy = estimate_energy(counts, constants)
    spikes_total = counts.spikes_per_decision.sum(axis=1)
    synaptic_ops = counts.synaptic_ops_per_decision
    main_each = np.asarray(energy["snn_main_pj_each"], dtype=np.float64)
    settings = {"n_resamples": n_resamples, "confidence_level": confidence_level, "method": method, "seed": seed}

    n = array.shape[0]
    steps = network.time_steps
    widths = network.hidden_sizes
    fired = [layer.sum(axis=0) for layer in per_neuron]  # (H_l,) spikes of each neuron over all decisions and steps
    never = [int(np.sum(total == 0.0)) for total in fired]
    every = [int(np.sum(total == float(n * steps))) for total in fired]
    rates = [float(value) for value in counts.firing_rate_per_layer]
    break_even = constants.e_mac_pj / (constants.e_ac_pj * steps)

    return {
        "label": "estimate",
        "what": (
            "operation counts of the deployed numpy network on the given situations (measured by running it); "
            "the energy figures are analytical estimates from those counts, not measurements"
        ),
        "network_sha256": student.digest,
        "situations": {"n": int(n), "sha256": _situations_sha256(array)},
        "layer_sizes": list(counts.layer_sizes),
        "time_steps": int(steps),
        "fan_out": list(counts.fan_out),
        "first_layer_macs": int(counts.first_layer_macs),
        "neuron_updates": int(counts.neuron_updates),
        "dense_reference_macs": int(counts.dense_reference_macs),
        "spikes_per_decision": {
            "mean_per_layer": [float(value) for value in counts.mean_spikes_per_layer],
            "mean_total": float(np.mean(spikes_total)),
            "interval_total": _mean_interval(spikes_total, **settings),
        },
        "synaptic_ops_per_decision": {
            "definition": "sum over hidden layers of spikes emitted x fan-out (the width of the layer they feed)",
            "mean": float(np.mean(synaptic_ops)),
            "interval": _mean_interval(synaptic_ops, **settings),
        },
        "firing": {
            "neurons_per_layer": [int(width) for width in widths],
            "rate_per_layer": rates,
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
        "break_even": {
            "rate_per_step": float(break_even),
            "formula": "e_mac / (e_ac * T)",
            "definition": (
                "under the main estimate a hidden layer's spikes cost e_ac per synapse per spike and the dense "
                "reference costs e_mac per synapse once per decision, so the spiking connection costs less than "
                "the dense one when the layer's firing rate per step is below e_mac / (e_ac * T)"
            ),
            "per_layer": [
                {"layer": layer, "firing_rate_per_step": rate, "below_break_even": bool(rate < break_even)}
                for layer, rate in enumerate(rates)
            ],
        },
        "energy_estimate": energy,
        "energy_main_pj_interval": _mean_interval(main_each, **settings),
        "per_decision": {
            "spikes_per_layer": [[float(value) for value in row] for row in counts.spikes_per_decision],
            "synaptic_ops": [float(value) for value in synaptic_ops],
        },
        "replica_check": {
            "spike_counts_identical_per_layer_step_decision": True,
            "max_abs_logit_difference": float(np.max(np.abs(replica_logits - reference_logits))),
        },
        "caveats": list(CAVEATS),
    }


# --------------------------------------------------------------------------------------
# Agreement with held-out dreams
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


def agreement_with_dream(student: SNNStudent, dreams: DreamSet) -> dict[str, Any]:
    """Decision quality of ``student`` INSIDE ``dreams`` (callers pass held-out dreams).

    For the n dream rows: the student's decisions (n,) = ``student.decide_index(dreams.situations)``;
    agreement with the dream's choice ``dreams.chosen`` (the first best-scored candidate, so
    ties make agreement understate quality; ``fraction_zero`` regret counts every best
    candidate); dream regret (targets.dream_regret: row max of ``dreams.scores`` minus the
    score of the decision) as mean / median / p90 in objective units and divided by the score
    spread of these dreams (targets.score_spread; ``"undefined"`` when that spread is 0). The
    same regret is reported for ``dreams.physics_choice``, the "no learning" reference.

    Returns a JSON-serializable dict including the per-row decisions and regrets.
    """
    student = _check_student(student)
    dreams = _check_dream_for_student(student, dreams)
    decisions = student.decide_index(dreams.situations)
    student_regret = dream_regret(dreams.scores, decisions)
    physics_regret = dream_regret(dreams.scores, dreams.physics_choice)
    try:
        spread: float | None = score_spread(dreams.scores)
        spread_reason = None
    except ValueError as error:
        spread, spread_reason = None, str(error)
    n = dreams.n_rows
    return {
        "label": (
            "measured inside the dream: dreamed objective scores of the dream's candidates, not outcomes in the "
            "simulator or on the robot"
        ),
        "dream": {
            "skill": dreams.skill,
            **_dream_record(dreams.method, dreams.request_hashes),
            "n_rows": int(n),
            "n_candidates": int(dreams.n_candidates),
            "rows_sha256": _dream_content_sha256(dreams, np.arange(n)),
        },
        "network_sha256": student.digest,
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
        },
    }


# --------------------------------------------------------------------------------------
# One record per student
# --------------------------------------------------------------------------------------


def student_record(
    student: SNNStudent,
    spec: SkillSpec,
    dreams_heldout: DreamSet,
    situations: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Everything a report needs about one dream-trained spiking student, in one JSON record.

    Args:
        spec:           the student's SkillSpec (name, situation encoding and actions must match).
        dreams_heldout: dreams the student was NOT trained on, made on the student's candidate
                        grid (checked row by row); refused when it is exactly the training dream.
        situations:     (n >= 2, S) situations for the latency and operation measurements.
        config:         loaded config.yaml; read: evaluation.latency_decisions,
                        evaluation.bootstrap_resamples, evaluation.bootstrap_confidence_level,
                        evaluation.bootstrap_method, students.snn.energy.*.
        seed:           run seed; the bootstrap seeds come from the named streams
                        "<skill>/measure/regret_bootstrap" and "<skill>/measure/operations_bootstrap".

    The record holds: the network (digest, layer sizes, T, neuron constants), the settings and
    the full provenance from ``student.meta``; whether the training and held-out dreams came
    from a provider-backed teacher, a local residual model or physics only; the latency,
    operations and in-dream agreement measurements; and a paired bootstrap interval of the
    mean dream regret of the student minus that of the physics-only choice. Decision quality on
    actual Go2 episodes is not in it (learned_judgment.eval.harness measures that).
    """
    student = _check_student(student)
    spec = _check_spec(student, spec)
    seed = _check_count("seed", seed, 0)
    if not isinstance(dreams_heldout, DreamSet):
        raise TypeError(f"dreams_heldout must be a DreamSet; got {type(dreams_heldout).__name__}")
    _check_dream_for_student(student, dreams_heldout)
    _check_dream_grid(spec, student, dreams_heldout)
    training_dream = student.meta.get("dream")
    if not isinstance(training_dream, Mapping) or "method" not in training_dream or "request_hashes" not in training_dream:
        raise ValueError("student.meta['dream'] with 'method' and 'request_hashes' is missing: not a dream-trained student")

    values = {
        "evaluation.latency_decisions": require(config, "evaluation.latency_decisions"),
        "evaluation.bootstrap_resamples": require(config, "evaluation.bootstrap_resamples"),
        "evaluation.bootstrap_confidence_level": require(config, "evaluation.bootstrap_confidence_level"),
        "evaluation.bootstrap_method": require(config, "evaluation.bootstrap_method"),
    }
    constants = EnergyConstants.from_config(config)
    bootstrap = {
        "n_resamples": values["evaluation.bootstrap_resamples"],
        "confidence_level": values["evaluation.bootstrap_confidence_level"],
        "method": values["evaluation.bootstrap_method"],
    }
    regret_stream = f"{student.skill}/measure/regret_bootstrap"
    operations_stream = f"{student.skill}/measure/operations_bootstrap"
    regret_seed = stream_seed(seed, regret_stream)
    operations_seed = stream_seed(seed, operations_stream)

    agreement = agreement_with_dream(student, dreams_heldout)
    training_rows = training_dream.get("rows_sha256")
    if training_rows is not None and agreement["dream"]["rows_sha256"] == training_rows:
        raise ValueError("dreams_heldout are exactly the rows the student was trained on; pass held-out dreams")
    latency = measure_decision_latency(student, situations, n_decisions=values["evaluation.latency_decisions"])
    student_regret = np.asarray(agreement["per_row"]["student_regret"], dtype=np.float64)
    physics_regret = np.asarray(agreement["per_row"]["physics_regret"], dtype=np.float64)
    regret_difference = paired_difference_ci(
        physics_regret[:, None], student_regret[:, None], seed=regret_seed, **bootstrap
    )
    operations = measure_operations(student, situations, constants, seed=operations_seed, **bootstrap)

    training_source = _dream_record(str(training_dream["method"]), tuple(training_dream["request_hashes"]))
    heldout_source = {key: agreement["dream"][key] for key in ("method", "request_hashes", "source", "source_description", "provider_backed")}
    network = student.network
    record = {
        "format": RECORD_FORMAT,
        "label": (
            "Measurements of one dream-trained spiking student from runs of its deployed numpy network. Decision "
            "quality is measured inside held-out dreams (dreamed objective scores), not on the robot or in the "
            "simulator; energy figures are estimates; latency is wall-clock time on this machine."
        ),
        "skill": student.skill,
        "student": {
            "kind": student.meta.get("kind"),
            "network_sha256": student.digest,
            "layer_sizes": list(network.layer_sizes),
            "hidden_sizes": list(network.hidden_sizes),
            "n_inputs": network.n_inputs,
            "n_candidates": student.n_candidates,
            "time_steps": network.time_steps,
            "betas": list(network.betas),
            "thresholds": list(network.thresholds),
            "readout_beta": network.readout_beta,
            "points_per_dim": list(student.points_per_dim),
            "action_names": list(_trained_action_names(student)),
            "encoder": student.encoder.describe(),
            "settings": _json_copy("student.meta['settings']", student.meta.get("settings")),
        },
        "dream_provenance": {
            "training_dream": {**training_source, "rows_sha256": training_rows},
            "heldout_dream": {**heldout_source, "rows_sha256": agreement["dream"]["rows_sha256"]},
            "trained_in_provider_backed_dream": training_source["provider_backed"],
            "heldout_same_method_as_training": heldout_source["method"] == training_source["method"],
        },
        "agreement_with_heldout_dream": agreement,
        "dream_regret_difference_student_minus_physics": {
            "label": (
                "paired over held-out dream rows: mean dream regret of the student minus that of the physics-only "
                "choice; negative = the student gives up less dreamed objective. Inside the dream, not the simulator"
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
        "student_meta": _json_copy("student.meta", student.meta),
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
