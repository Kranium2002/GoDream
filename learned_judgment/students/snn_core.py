"""The deployed spiking student: inference and operation counting in plain numpy.

Training uses PyTorch + snnTorch (``snn_train.py``); decisions never import them. The
dynamics below reproduce snnTorch 1.0.0 ``Leaky``. This numpy network is the student: every
reported decision, spike count and latency comes from it, not from the torch model.

Agreement with the trained torch model in float64, as observed: spike counts and decisions
were identical in every run (experiments/proofs/E15_snn_contract/p05b and
experiments/proofs/B_snn_offline). Readout logits were bitwise equal in the E15 proof and
differed by about 2e-15 in a later run, because the two libraries sum a matrix product in a
different order. Training checks spike counts and decisions after export and stops with an
error if either differs.

Network (fully connected, T time steps, encoded input x of width D):

    first layer current   I   = W_1 x + b_1          constant over the T steps ("direct"
                                                      encoding); computed once per decision
    hidden layer l, step t:
        reset_t = 1 if U_{t-1} > threshold_l else 0   taken from the PREVIOUS membrane
        U_t     = beta_l * U_{t-1} + I_t - reset_t * threshold_l
        S_t     = 1 if U_t > threshold_l else 0       strict comparison
    current into layer l > 1:  I_t = W_l S^{l-1}_t + b_l
    readout (non-spiking leaky integrator, no reset):
        M_t = beta_o * M_{t-1} + W_o S^L_t + b_o
    logits = mean over t of M_t ;  decision = argmax over the C outputs

Each output is one candidate command of the skill's candidate grid, so a decision is always
a command the grid contains.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from learned_judgment.fileio import atomic_write_bytes

__all__ = ["NETWORK_FORMAT", "OperationCounts", "SpikingNetwork", "count_operations", "load_network", "save_network"]

NETWORK_FORMAT = "learned-judgment/spiking-network/v1"


def _matrix(name: str, value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2-D float64 array; got shape {array.shape}")
    return array


def _vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite float64 vector of length {size}; got shape {array.shape}")
    return array


def _leak(name: str, value: Any) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; got {number}")
    return number


@dataclass(frozen=True)
class SpikingNetwork:
    """Weights and neuron constants of one trained spiking network (all float64).

    Attributes:
        hidden_weights: L matrices; layer l has shape (H_l, fan_in) with fan_in = D for l = 1.
        hidden_biases:  L vectors (H_l,).
        betas:          L membrane leak factors in [0, 1].
        thresholds:     L firing thresholds.
        readout_weight: (C, H_L).
        readout_bias:   (C,).
        readout_beta:   leak factor of the non-spiking readout, in [0, 1].
        time_steps:     T >= 1.
    """

    hidden_weights: tuple[np.ndarray, ...]
    hidden_biases: tuple[np.ndarray, ...]
    betas: tuple[float, ...]
    thresholds: tuple[float, ...]
    readout_weight: np.ndarray
    readout_bias: np.ndarray
    readout_beta: float
    time_steps: int

    def __post_init__(self) -> None:
        layers = len(self.hidden_weights)
        if layers < 1 or not (len(self.hidden_biases) == len(self.betas) == len(self.thresholds) == layers):
            raise ValueError("a spiking network needs >= 1 hidden layer with one bias, beta and threshold each")
        weights, biases = [], []
        fan_in = None
        for index, (weight, bias) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            weight = _matrix(f"hidden_weights[{index}]", weight)
            if fan_in is not None and weight.shape[1] != fan_in:
                raise ValueError(f"hidden_weights[{index}] expects fan-in {fan_in}; got {weight.shape[1]}")
            weights.append(weight)
            biases.append(_vector(f"hidden_biases[{index}]", bias, weight.shape[0]))
            fan_in = weight.shape[0]
        readout_weight = _matrix("readout_weight", self.readout_weight)
        if readout_weight.shape[1] != fan_in:
            raise ValueError(f"readout_weight expects fan-in {fan_in}; got {readout_weight.shape[1]}")
        thresholds = tuple(float(value) for value in self.thresholds)
        if not all(np.isfinite(thresholds)):
            raise ValueError("thresholds must be finite")
        steps = int(self.time_steps)
        if steps < 1:
            raise ValueError(f"time_steps must be >= 1; got {self.time_steps}")
        set_ = object.__setattr__
        set_(self, "hidden_weights", tuple(weights))
        set_(self, "hidden_biases", tuple(biases))
        set_(self, "betas", tuple(_leak(f"betas[{i}]", value) for i, value in enumerate(self.betas)))
        set_(self, "thresholds", thresholds)
        set_(self, "readout_weight", readout_weight)
        set_(self, "readout_bias", _vector("readout_bias", self.readout_bias, readout_weight.shape[0]))
        set_(self, "readout_beta", _leak("readout_beta", self.readout_beta))
        set_(self, "time_steps", steps)

    # ----- structure -------------------------------------------------------------------

    @property
    def n_inputs(self) -> int:
        return int(self.hidden_weights[0].shape[1])

    @property
    def n_outputs(self) -> int:
        return int(self.readout_weight.shape[0])

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        return tuple(int(weight.shape[0]) for weight in self.hidden_weights)

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """(D, H_1, ..., H_L, C)."""
        return (self.n_inputs, *self.hidden_sizes, self.n_outputs)

    # ----- inference -------------------------------------------------------------------

    def _simulate(
        self, inputs: np.ndarray, counts: np.ndarray | None, trains: list[np.ndarray] | None
    ) -> np.ndarray:
        """The one implementation of the dynamics: logits (n, C) for float64 inputs (n, D).

        Fills ``counts`` (L, T, n) and ``trains`` (one uint8 array (T, n, H_l) per hidden layer)
        when given. Every public method runs through here, so there is one statement of the
        dynamics. A row computed alone and the same row inside a batch can still differ in the
        last bit, because the matrix library sums a product differently for different batch
        sizes; in rare cases that flips a spike. Training therefore checks the single-decision
        path against the batched path on its rows, and decisions on the robot always come from
        the single-decision path.
        """
        n = inputs.shape[0]
        weights, biases = self.hidden_weights, self.hidden_biases
        layers = len(weights)
        first_current = inputs @ weights[0].T + biases[0]
        membranes = [np.zeros((n, weight.shape[0])) for weight in weights]
        readout = np.zeros((n, self.readout_weight.shape[0]))
        total = np.zeros_like(readout)
        for step in range(self.time_steps):
            current = first_current
            spikes = None
            for layer in range(layers):
                if layer > 0:
                    current = spikes @ weights[layer].T + biases[layer]
                threshold = self.thresholds[layer]
                reset = (membranes[layer] > threshold).astype(np.float64)
                membranes[layer] = (self.betas[layer] * membranes[layer] + current) - reset * threshold
                spikes = (membranes[layer] > threshold).astype(np.float64)
                if counts is not None:
                    counts[layer, step] = spikes.sum(axis=1).astype(np.int64)
                if trains is not None:
                    trains[layer][step] = spikes
            readout = self.readout_beta * readout + (spikes @ self.readout_weight.T + self.readout_bias)
            total = total + readout
        return total / float(self.time_steps)

    def _checked(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.n_inputs:
            raise ValueError(f"inputs must have shape (n, {self.n_inputs}); got {inputs.shape}")
        if not np.all(np.isfinite(inputs)):
            raise ValueError("inputs contain non-finite values")
        return inputs

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Logits (n, C) for encoded inputs (n, D)."""
        return self._simulate(self._checked(inputs), None, None)

    def forward_with_spikes(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Logits (n, C) and spike counts (L, T, n): spikes each hidden layer emitted at each
        step for each input row."""
        inputs = self._checked(inputs)
        counts = np.zeros((len(self.hidden_weights), self.time_steps, inputs.shape[0]), dtype=np.int64)
        return self._simulate(inputs, counts, None), counts

    def forward_with_spike_trains(self, inputs: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Logits (n, C) and, per hidden layer, the spike train as uint8 (T, n, H_l):
        1 where that neuron fired at that step for that input row."""
        inputs = self._checked(inputs)
        trains = [np.zeros((self.time_steps, inputs.shape[0], size), dtype=np.uint8) for size in self.hidden_sizes]
        return self._simulate(inputs, None, trains), tuple(trains)

    def decide(self, inputs: np.ndarray) -> np.ndarray:
        """int (n,): index of the chosen output for each encoded input row."""
        return np.argmax(self.forward(inputs), axis=1)

    def decide_one(self, encoded: np.ndarray) -> int:
        """Index of the chosen output for one encoded input (D,): the per-decision path.

        Runs the same dynamics as :meth:`forward` on a batch of one, with only the shape and
        finiteness checks a single vector needs. This is the path a deployed student uses.
        """
        encoded = np.asarray(encoded, dtype=np.float64)
        if encoded.shape != (self.n_inputs,):
            raise ValueError(f"encoded input must have shape ({self.n_inputs},); got {encoded.shape}")
        if not np.isfinite(encoded).all():
            raise ValueError("encoded input contains non-finite values")
        return int(np.argmax(self._simulate(encoded[None, :], None, None)[0]))

    # ----- serialization ---------------------------------------------------------------

    def to_arrays(self) -> dict[str, np.ndarray]:
        """All weights and constants as named float64 / int64 arrays (the on-disk layout)."""
        arrays: dict[str, np.ndarray] = {
            "n_hidden_layers": np.array(len(self.hidden_weights), dtype=np.int64),
            "time_steps": np.array(self.time_steps, dtype=np.int64),
            "betas": np.array(self.betas, dtype=np.float64),
            "thresholds": np.array(self.thresholds, dtype=np.float64),
            "readout_weight": self.readout_weight,
            "readout_bias": self.readout_bias,
            "readout_beta": np.array(self.readout_beta, dtype=np.float64),
        }
        for index, (weight, bias) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            arrays[f"hidden_weight_{index}"] = weight
            arrays[f"hidden_bias_{index}"] = bias
        return arrays

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "SpikingNetwork":
        layers = int(arrays["n_hidden_layers"])
        return cls(
            hidden_weights=tuple(arrays[f"hidden_weight_{index}"] for index in range(layers)),
            hidden_biases=tuple(arrays[f"hidden_bias_{index}"] for index in range(layers)),
            betas=tuple(float(value) for value in np.asarray(arrays["betas"]).reshape(-1)),
            thresholds=tuple(float(value) for value in np.asarray(arrays["thresholds"]).reshape(-1)),
            readout_weight=arrays["readout_weight"],
            readout_bias=arrays["readout_bias"],
            readout_beta=float(arrays["readout_beta"]),
            time_steps=int(arrays["time_steps"]),
        )

    def digest(self) -> str:
        """sha256 over every array in a fixed order: the identity of the trained network."""
        hasher = hashlib.sha256(NETWORK_FORMAT.encode("utf-8"))
        arrays = self.to_arrays()
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            hasher.update(name.encode("utf-8"))
            hasher.update(str(array.dtype.str).encode("utf-8"))
            hasher.update(str(array.shape).encode("utf-8"))
            hasher.update(array.tobytes())
        return hasher.hexdigest()


# --------------------------------------------------------------------------------------
# Operation counts (inputs to the energy ESTIMATE; nothing here is an energy measurement)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OperationCounts:
    """Operations of the network per decision, measured on the given inputs.

    Attributes:
        layer_sizes:          (D, H_1, ..., H_L, C).
        time_steps:           T.
        spikes_per_decision:  (n, L) spikes each hidden layer emitted for each decision,
                              summed over the T steps. Kept per decision so that reports can
                              bootstrap an interval.
        first_layer_macs:     D * H_1 multiply-accumulates of the real-valued input layer,
                              executed once per decision because the input current is constant.
        neuron_updates:       (H_1 + ... + H_L + C) * T membrane updates per decision.
        dense_reference_macs: multiply-accumulates of one pass through a non-spiking dense
                              network with the same layer sizes (sum of fan_in * fan_out).
    """

    layer_sizes: tuple[int, ...]
    time_steps: int
    spikes_per_decision: np.ndarray
    first_layer_macs: int
    neuron_updates: int
    dense_reference_macs: int

    @property
    def n_decisions(self) -> int:
        return int(self.spikes_per_decision.shape[0])

    @property
    def fan_out(self) -> tuple[int, ...]:
        """Fan-out of each hidden layer: the width of the layer its spikes feed."""
        return tuple(self.layer_sizes[2:])

    @property
    def mean_spikes_per_layer(self) -> np.ndarray:
        """(L,) mean spikes per decision of each hidden layer."""
        return self.spikes_per_decision.mean(axis=0)

    @property
    def synaptic_ops_per_decision(self) -> np.ndarray:
        """(n,) synaptic operations per decision: sum over layers of spikes * fan-out."""
        return self.spikes_per_decision @ np.asarray(self.fan_out, dtype=np.float64)

    @property
    def firing_rate_per_layer(self) -> np.ndarray:
        """(L,) mean fraction of neurons of each hidden layer that fire per time step."""
        widths = np.asarray(self.layer_sizes[1:-1], dtype=np.float64)
        return self.mean_spikes_per_layer / (widths * float(self.time_steps))


def count_operations(network: SpikingNetwork, inputs: np.ndarray) -> OperationCounts:
    """Run ``network`` on encoded ``inputs`` (n, D) and count its operations per decision."""
    _, counts = network.forward_with_spikes(inputs)
    sizes = network.layer_sizes
    hidden = sizes[1:-1]
    return OperationCounts(
        layer_sizes=sizes,
        time_steps=network.time_steps,
        spikes_per_decision=counts.sum(axis=1).T.astype(np.float64),
        first_layer_macs=int(sizes[0] * sizes[1]),
        neuron_updates=int((sum(hidden) + sizes[-1]) * network.time_steps),
        dense_reference_macs=int(sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))),
    )


# --------------------------------------------------------------------------------------
# Persistence: <stem>.npz (arrays) + <stem>.json (metadata), no pickle
# --------------------------------------------------------------------------------------


def _stem(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix("") if path.suffix in (".npz", ".json") else path


def save_network(path: str | Path, network: SpikingNetwork, meta: dict[str, Any]) -> str:
    """Save ``network`` and its JSON-serializable ``meta``; return the network digest.

    ``meta`` describes what the arrays cannot: input encoder, candidate grid, training
    report, software versions. The digest and format tag are added to it.
    """
    stem = _stem(path)
    buffer = io.BytesIO()
    np.savez(buffer, **network.to_arrays())
    digest = network.digest()
    record = {**meta, "format": NETWORK_FORMAT, "network_sha256": digest, "layer_sizes": list(network.layer_sizes)}
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    atomic_write_bytes(stem.with_suffix(".npz"), buffer.getvalue())
    atomic_write_bytes(stem.with_suffix(".json"), (text + "\n").encode("utf-8"))
    return digest


def load_network(path: str | Path) -> tuple[SpikingNetwork, dict[str, Any]]:
    """Load a network saved by :func:`save_network` and verify its digest."""
    stem = _stem(path)
    with open(stem.with_suffix(".json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("format") != NETWORK_FORMAT:
        raise ValueError(f"{stem}.json is not a {NETWORK_FORMAT} file")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        network = SpikingNetwork.from_arrays({name: archive[name] for name in archive.files})
    if network.digest() != meta.get("network_sha256"):
        raise ValueError(f"{stem}.npz does not match the digest recorded in {stem}.json")
    return network, meta
