"""Training of spiking networks inside the dream (PyTorch + snnTorch).

Lane contract (Person B, per-skill spiking student; reused by the shared multi-skill network)
---------------------------------------------------------------------------------------------
The dream (physics guess corrected by the PFN) scores every candidate command of every dreamed
situation. A spiking network is trained inside that dream to choose the best candidate: its
targets are soft distributions over candidates built from the dreamed scores
(:mod:`learned_judgment.students.targets`). It never learns the PFN's outcome predictions.

This is the only module that imports ``torch`` and ``snntorch``, and it imports them lazily
inside functions: importing this module, loading a saved network or deciding never imports
torch. Decisions run in plain numpy (:mod:`learned_judgment.students.snn_core`), and
:func:`check_parity` verifies, on the inputs it is given (the per-skill student passes every
training and early-stopping row), that the numpy network emits the same spike train for every
neuron at every step, the same decisions, and logits within
``students.snn.parity_logit_tolerance`` of the torch model, and that the deployed per-decision
path (``SpikingNetwork.decide_one``, one row at a time) makes the same decisions as the batched
path. Logits agree to float64 rounding, not bitwise: torch (MKL) and numpy (OpenBLAS) sum the
readout matmul in different orders (experiments/proofs/B_snn_offline/
p01_matmul_rounding_probe.out.txt, c2_end_to_end.out.txt).

:func:`fit` refuses (``TrainingError``) a network whose restored validation loss is not below
its loss at initialisation, not below the loss of the best constant prediction (the head's mean
training target), or that has a hidden layer emitting no spike on the validation rows: such a
network decides without looking at the situation.

Model (float64 throughout; ``D`` encoded inputs, ``H_l`` hidden widths, ``C_head`` candidates)::

    x (B, D) -> first-layer current W_1 x + b_1, computed once and held for T steps
             -> [Linear -> snn.Leaky(reset_mechanism="subtract")] per hidden size   (shared trunk)
             -> per named head: Linear -> snn.Leaky(reset_mechanism="none")          (readout)
    logits (B, C_head) = mean over the T steps of the readout membrane

Verified snnTorch 1.0.0 facts this module relies on (experiments/proofs/E15_snn_contract and
experiments/proofs/B_snn_offline/p00_reset_dtype_probe.out.txt):

* the membrane state is passed explicitly on every call and starts at zeros; the module's
  internal buffer, ``init_hidden`` and ``utils.reset`` are never used;
* beta and threshold are passed as float64 tensors (a float beta is stored as float32(0.9));
* the surrogate's forward returns float32 spikes. snnTorch multiplies the detached float32
  reset by the threshold, which keeps that product in float32; for a threshold that float32
  cannot represent exactly (e.g. 0.3) the float64 membrane then drifts from a float64 numpy
  network by about 1e-7. Every surrogate here is therefore wrapped so that its output has the
  membrane's dtype (p00: zero difference), while the snnTorch surrogate gradient is unchanged;
* ``copy.deepcopy`` of a module fails after a forward and a whole module cannot be
  ``torch.save``-d: weights are snapshotted as cloned state-dict tensors;
* training is bitwise reproducible across processes with ``torch.manual_seed`` and 1 or 4
  threads, not at 16 (p04_determinism).
"""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from learned_judgment.config import require
from learned_judgment.students.snn_core import SpikingNetwork
from learned_judgment.students.targets import TARGET_SPREADS

__all__ = [
    "OUT_OF_RANGE_POLICIES",
    "SURROGATES",
    "ParityError",
    "SNNSettings",
    "TrainingError",
    "TrainingReport",
    "build_model",
    "check_parity",
    "constant_predictor_loss",
    "export_network",
    "fit",
    "is_spiking_model",
    "spike_statistics",
    "validation_losses",
]

SURROGATES = ("atan", "fast_sigmoid")
# What a deployed student does with a situation feature outside its declared range:
# "error" refuses to decide and names the feature; "clip" clips it into the range and counts it.
OUT_OF_RANGE_POLICIES = ("error", "clip")
_ROW_SUM_TOLERANCE = 1e-9  # structural: a soft-target row must be a probability distribution
_SAMPLING_SUM_TOLERANCE = 1e-9  # structural: sampling weights must form a distribution


class TrainingError(RuntimeError):
    """Training did not produce a network better than its initialisation (or diverged).

    ``report`` holds the :class:`TrainingReport` when one was completed, else ``None``.
    """

    def __init__(self, message: str, report: "TrainingReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


class ParityError(RuntimeError):
    """The numpy network does not reproduce the torch model; ``report`` holds the comparison."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


# --------------------------------------------------------------------------------------
# Settings (every value from config.yaml: students.snn.*)
# --------------------------------------------------------------------------------------


def _int(problems: list[str], name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        problems.append(f"students.snn.{name} must be an integer; got {value!r}")
        return minimum
    if int(value) < minimum:
        problems.append(f"students.snn.{name} must be >= {minimum}; got {value}")
    return int(value)


def _float(
    problems: list[str], name: str, value: Any, *, low: float, high: float, low_open: bool, high_open: bool
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        problems.append(f"students.snn.{name} must be a number; got {value!r}")
        return float("nan")
    number = float(value)
    too_low = number <= low if low_open else number < low
    too_high = number >= high if high_open else number > high
    if not math.isfinite(number) or too_low or too_high:
        interval = f"{'(' if low_open else '['}{low}, {high}{')' if high_open else ']'}"
        problems.append(f"students.snn.{name} must lie in {interval}; got {number}")
    return number


def _bool(problems: list[str], name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        problems.append(f"students.snn.{name} must be true or false; got {value!r}")
        return False
    return bool(value)


@dataclass(frozen=True)
class SNNSettings:
    """Hyperparameters of a spiking student (config.yaml ``students.snn``).

    Attributes:
        hidden_sizes:         widths of the shared hidden layers, >= 1 layer.
        time_steps:           T, steps per decision (constant input current held for T steps).
        beta:                 membrane leak of the hidden layers, in [0, 1].
        threshold:            firing threshold of the hidden layers, > 0.
        readout_beta:         leak of the non-spiking readout, in [0, 1] (never learned).
        learn_beta:           learn the hidden layers' beta (one scalar per layer).
        learn_threshold:      learn the hidden layers' threshold (one scalar per layer).
        surrogate:            "atan" or "fast_sigmoid" (snnTorch surrogate gradients).
        surrogate_alpha:      atan's alpha, or fast_sigmoid's slope; > 0.
        target_temperature:   soft-target temperature in units of the score scale; > 0.
        target_spread:        score scale of the soft targets: "global_median" (one scale, the
                              median row spread of the training rows) or "per_row" (each row's
                              own spread); see learned_judgment.students.targets.
        spike_rate_penalty:   weight of the mean hidden firing rate in the loss; >= 0.
        learning_rate:        Adam learning rate; > 0.
        adam_beta1, adam_beta2: Adam's moment decay rates, each in [0, 1).
        adam_eps:             Adam's denominator term; > 0.
        weight_decay:         Adam's L2 weight decay (torch.optim.Adam, added to the gradient); >= 0.
        batch_size:           rows per optimizer step.
        max_steps:            optimizer steps at most.
        validation_fraction:  share of distinct dreamed situations kept for early stopping, in (0, 1).
        evaluate_every_steps: validation-loss evaluation interval in optimizer steps.
        patience_evaluations: evaluations without a better total validation loss before stopping.
        torch_threads:        torch.set_num_threads during training (reproducible at 1 and 4).
        parity_logit_tolerance: largest allowed |logit difference| between the torch model and the
                              exported numpy network in :func:`check_parity`; >= 0.
        out_of_range:         what the deployed student does with a situation feature outside its
                              declared range: "error" (refuse, naming the feature) or "clip"
                              (clip into the range and count the clipped values).
    """

    hidden_sizes: tuple[int, ...]
    time_steps: int
    beta: float
    threshold: float
    readout_beta: float
    learn_beta: bool
    learn_threshold: bool
    surrogate: str
    surrogate_alpha: float
    target_temperature: float
    target_spread: str
    spike_rate_penalty: float
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    weight_decay: float
    batch_size: int
    max_steps: int
    validation_fraction: float
    evaluate_every_steps: int
    patience_evaluations: int
    torch_threads: int
    parity_logit_tolerance: float
    out_of_range: str

    def __post_init__(self) -> None:
        problems: list[str] = []
        sizes = self.hidden_sizes
        if isinstance(sizes, (str, bytes)) or not isinstance(sizes, Sequence) or len(sizes) < 1:
            problems.append(f"students.snn.hidden_sizes must be a non-empty list of integers; got {sizes!r}")
            sizes = ()
        normalized: dict[str, Any] = {
            "hidden_sizes": tuple(_int(problems, f"hidden_sizes[{i}]", size, 1) for i, size in enumerate(sizes)),
            "time_steps": _int(problems, "time_steps", self.time_steps, 1),
            "beta": _float(problems, "beta", self.beta, low=0.0, high=1.0, low_open=False, high_open=False),
            "threshold": _float(
                problems, "threshold", self.threshold, low=0.0, high=math.inf, low_open=True, high_open=True
            ),
            "readout_beta": _float(
                problems, "readout_beta", self.readout_beta, low=0.0, high=1.0, low_open=False, high_open=False
            ),
            "learn_beta": _bool(problems, "learn_beta", self.learn_beta),
            "learn_threshold": _bool(problems, "learn_threshold", self.learn_threshold),
            "surrogate_alpha": _float(
                problems, "surrogate_alpha", self.surrogate_alpha, low=0.0, high=math.inf, low_open=True, high_open=True
            ),
            "target_temperature": _float(
                problems,
                "target_temperature",
                self.target_temperature,
                low=0.0,
                high=math.inf,
                low_open=True,
                high_open=True,
            ),
            "spike_rate_penalty": _float(
                problems,
                "spike_rate_penalty",
                self.spike_rate_penalty,
                low=0.0,
                high=math.inf,
                low_open=False,
                high_open=True,
            ),
            "learning_rate": _float(
                problems, "learning_rate", self.learning_rate, low=0.0, high=math.inf, low_open=True, high_open=True
            ),
            "adam_beta1": _float(
                problems, "adam_beta1", self.adam_beta1, low=0.0, high=1.0, low_open=False, high_open=True
            ),
            "adam_beta2": _float(
                problems, "adam_beta2", self.adam_beta2, low=0.0, high=1.0, low_open=False, high_open=True
            ),
            "adam_eps": _float(problems, "adam_eps", self.adam_eps, low=0.0, high=math.inf, low_open=True, high_open=True),
            "weight_decay": _float(
                problems, "weight_decay", self.weight_decay, low=0.0, high=math.inf, low_open=False, high_open=True
            ),
            "parity_logit_tolerance": _float(
                problems,
                "parity_logit_tolerance",
                self.parity_logit_tolerance,
                low=0.0,
                high=math.inf,
                low_open=False,
                high_open=True,
            ),
            "batch_size": _int(problems, "batch_size", self.batch_size, 1),
            "max_steps": _int(problems, "max_steps", self.max_steps, 1),
            "validation_fraction": _float(
                problems,
                "validation_fraction",
                self.validation_fraction,
                low=0.0,
                high=1.0,
                low_open=True,
                high_open=True,
            ),
            "evaluate_every_steps": _int(problems, "evaluate_every_steps", self.evaluate_every_steps, 1),
            "patience_evaluations": _int(problems, "patience_evaluations", self.patience_evaluations, 1),
            "torch_threads": _int(problems, "torch_threads", self.torch_threads, 1),
        }
        if self.surrogate not in SURROGATES:
            problems.append(f"students.snn.surrogate must be one of {SURROGATES}; got {self.surrogate!r}")
        if self.target_spread not in TARGET_SPREADS:
            problems.append(f"students.snn.target_spread must be one of {TARGET_SPREADS}; got {self.target_spread!r}")
        if self.out_of_range not in OUT_OF_RANGE_POLICIES:
            problems.append(
                f"students.snn.out_of_range must be one of {OUT_OF_RANGE_POLICIES}; got {self.out_of_range!r}"
            )
        if problems:
            raise ValueError("; ".join(problems))
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SNNSettings":
        """Every field from ``require(config, "students.snn.<field>")``; nothing is defaulted."""
        return cls(**{field.name: require(config, f"students.snn.{field.name}") for field in fields(cls)})

    def describe(self) -> dict[str, Any]:
        """JSON-serializable record of every setting, stored with every trained network."""
        record = {field.name: getattr(self, field.name) for field in fields(self)}
        record["hidden_sizes"] = list(self.hidden_sizes)
        return record


# --------------------------------------------------------------------------------------
# Model (the class is defined on first use so that importing this module stays torch-free)
# --------------------------------------------------------------------------------------


class _SpikeInMembraneDtype:
    """An snnTorch surrogate spike function whose forward output has the membrane's dtype.

    snnTorch's surrogates return float32 spikes; see the module docstring for why that breaks
    exact float64 parity of the subtractive reset. The surrogate gradient is snnTorch's own.
    """

    def __init__(self, kind: str, alpha: float) -> None:
        from snntorch import surrogate

        if kind == "atan":
            self._function = surrogate.atan(alpha=alpha)
        elif kind == "fast_sigmoid":
            self._function = surrogate.fast_sigmoid(slope=alpha)
        else:
            raise ValueError(f"surrogate must be one of {SURROGATES}; got {kind!r}")
        self.kind = kind
        self.alpha = alpha

    def __call__(self, membrane_shift: Any) -> Any:
        return self._function(membrane_shift).to(membrane_shift.dtype)


@lru_cache(maxsize=1)
def _model_class() -> type:
    import snntorch as snn
    import torch

    def leaky(beta: float, threshold: float, settings: SNNSettings, *, readout: bool) -> Any:
        return snn.Leaky(
            beta=torch.tensor(beta, dtype=torch.float64),
            threshold=torch.tensor(threshold, dtype=torch.float64),
            spike_grad=_SpikeInMembraneDtype(settings.surrogate, settings.surrogate_alpha),
            learn_beta=False if readout else settings.learn_beta,
            learn_threshold=False if readout else settings.learn_threshold,
            reset_mechanism="none" if readout else "subtract",
        )

    class SpikingTrunkWithHeads(torch.nn.Module):
        """Shared spiking trunk with named non-spiking readout heads (float64)."""

        def __init__(self, n_inputs: int, settings: SNNSettings) -> None:
            super().__init__()
            self.n_inputs = int(n_inputs)
            self.hidden_sizes = tuple(settings.hidden_sizes)
            self.time_steps = int(settings.time_steps)
            self.settings = settings
            sizes = (self.n_inputs, *self.hidden_sizes)
            self.trunk_linears = torch.nn.ModuleList(
                torch.nn.Linear(fan_in, fan_out, dtype=torch.float64) for fan_in, fan_out in zip(sizes[:-1], sizes[1:])
            )
            self.trunk_neurons = torch.nn.ModuleList(
                leaky(settings.beta, settings.threshold, settings, readout=False) for _ in self.hidden_sizes
            )
            self.head_linears = torch.nn.ModuleDict()
            self.head_neurons = torch.nn.ModuleDict()
            self.to(torch.float64)

        @property
        def heads(self) -> dict[str, int]:
            """{head name: number of outputs}, in registration order."""
            return {name: int(linear.out_features) for name, linear in self.head_linears.items()}

        def add_head(self, name: str, n_outputs: int, *, seed: int) -> None:
            """Register a readout head ``name`` with ``n_outputs`` candidates.

            Its weights are initialised from ``seed`` without touching the global torch RNG.
            An optimizer created before this call does not contain the new head.
            """
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"a head name must be a Python identifier (no '.'); got {name!r}")
            if name in self.head_linears:
                raise ValueError(f"head {name!r} already exists; heads: {list(self.head_linears)}")
            if isinstance(n_outputs, bool) or not isinstance(n_outputs, (int, np.integer)) or int(n_outputs) < 1:
                raise ValueError(f"head {name!r} needs an integer number of outputs >= 1; got {n_outputs!r}")
            seed = _seed(seed)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                linear = torch.nn.Linear(self.hidden_sizes[-1], int(n_outputs), dtype=torch.float64)
            # The readout never resets (reset "none") and its spikes are discarded, so its
            # threshold has no effect on the logits; it is set to the hidden threshold only
            # because snn.Leaky requires one.
            neuron = leaky(self.settings.readout_beta, self.settings.threshold, self.settings, readout=True)
            neuron = neuron.to(torch.float64)
            self.head_linears[name] = linear
            self.head_neurons[name] = neuron

        def forward(self, x: Any, head: str, record: bool = False) -> Any:
            """logits (B, C_head) for encoded inputs x (B, D) float64; with ``record`` also the
            list of hidden spike tensors, one (T, B, H_l) per hidden layer."""
            if head not in self.head_linears:
                raise KeyError(f"unknown head {head!r}; heads: {list(self.head_linears)}")
            if x.dtype != torch.float64 or x.ndim != 2 or x.shape[1] != self.n_inputs:
                raise ValueError(f"x must be float64 with shape (B, {self.n_inputs}); got {x.dtype} {tuple(x.shape)}")
            batch = x.shape[0]
            readout_linear = self.head_linears[head]
            readout_neuron = self.head_neurons[head]
            membranes = [x.new_zeros((batch, size)) for size in self.hidden_sizes]
            readout_membrane = x.new_zeros((batch, readout_linear.out_features))
            total = x.new_zeros((batch, readout_linear.out_features))
            records: list[list[Any]] | None = [[] for _ in self.hidden_sizes] if record else None
            first_current = self.trunk_linears[0](x)  # constant input: computed once per forward
            for _ in range(self.time_steps):
                current = first_current
                spikes = None
                for layer, neuron in enumerate(self.trunk_neurons):
                    if layer > 0:
                        current = self.trunk_linears[layer](spikes)
                    spikes, membranes[layer] = neuron(current, membranes[layer])
                    if records is not None:
                        records[layer].append(spikes)
                _, readout_membrane = readout_neuron(readout_linear(spikes), readout_membrane)
                total = total + readout_membrane
            logits = total / self.time_steps
            if records is None:
                return logits
            return logits, [torch.stack(layer_spikes) for layer_spikes in records]

    return SpikingTrunkWithHeads


def _seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**63:
        raise ValueError(f"seed must be an integer in [0, 2**63); got {seed!r}")
    return int(seed)


def is_spiking_model(model: Any) -> bool:
    """True when ``model`` was built by :func:`build_model` (imports torch)."""
    return isinstance(model, _model_class())


def _require_model(model: Any) -> None:
    if not is_spiking_model(model):
        raise TypeError(f"expected a model built by build_model(); got {type(model).__name__}")


def build_model(n_inputs: int, heads: Mapping[str, int], settings: SNNSettings, *, seed: int) -> Any:
    """Build the float64 spiking trunk with one readout head per entry of ``heads``.

    Args:
        n_inputs: D, width of the encoded input rows.
        heads:    {head name: C_head number of candidates}; may be empty (add heads later with
                  ``model.add_head(name, n_outputs, seed=...)``).
        settings: :class:`SNNSettings`.
        seed:     initialises the trunk, then head ``k`` (in mapping order) from ``seed + 1 + k``
                  folded into [0, 2**63); the global torch RNG is left untouched.

    Returns:
        A ``torch.nn.Module`` with ``forward(x (B, D), head, record=False)`` -> logits
        (B, C_head) [, list of (T, B, H_l) spike tensors], ``add_head``, ``heads``.
    """
    import torch

    if not isinstance(settings, SNNSettings):
        raise TypeError(f"settings must be SNNSettings; got {type(settings).__name__}")
    if isinstance(n_inputs, bool) or not isinstance(n_inputs, (int, np.integer)) or int(n_inputs) < 1:
        raise ValueError(f"n_inputs must be an integer >= 1; got {n_inputs!r}")
    seed = _seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _model_class()(int(n_inputs), settings)
    for position, (name, n_outputs) in enumerate(dict(heads).items()):
        model.add_head(name, n_outputs, seed=(seed + 1 + position) % 2**63)
    return model


# --------------------------------------------------------------------------------------
# Export to the deployed numpy network, and parity
# --------------------------------------------------------------------------------------


def _scalar(name: str, tensor: Any) -> float:
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be one scalar per layer; got shape {tuple(tensor.shape)}")
    return float(tensor.detach().reshape(()))


def export_network(model: Any, head: str) -> SpikingNetwork:
    """The trunk plus readout ``head`` as a numpy :class:`SpikingNetwork` (float64 copies).

    Betas are clamped to [0, 1] as snnTorch clamps them in its forward pass; thresholds are
    exported as they are (snnTorch does not clamp them).
    """
    import torch

    _require_model(model)
    if head not in model.head_linears:
        raise KeyError(f"unknown head {head!r}; heads: {list(model.head_linears)}")

    def array(tensor: Any) -> np.ndarray:
        if tensor.dtype != torch.float64:
            raise ValueError(f"the model must be float64 to export exactly; found {tensor.dtype}")
        return tensor.detach().cpu().numpy().astype(np.float64, copy=True)

    with torch.no_grad():
        readout_linear = model.head_linears[head]
        return SpikingNetwork(
            hidden_weights=tuple(array(linear.weight) for linear in model.trunk_linears),
            hidden_biases=tuple(array(linear.bias) for linear in model.trunk_linears),
            betas=tuple(_scalar(f"beta[{i}]", n.beta.clamp(0, 1)) for i, n in enumerate(model.trunk_neurons)),
            thresholds=tuple(_scalar(f"threshold[{i}]", n.threshold) for i, n in enumerate(model.trunk_neurons)),
            readout_weight=array(readout_linear.weight),
            readout_bias=array(readout_linear.bias),
            readout_beta=_scalar("readout_beta", model.head_neurons[head].beta.clamp(0, 1)),
            time_steps=model.time_steps,
        )


def _finite_matrix(name: str, value: Any, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or (width is not None and array.shape[1] != width):
        expected = f"(n >= 1, {width})" if width is not None else "(n >= 1, columns)"
        raise ValueError(f"{name} must have shape {expected}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        bad = np.argwhere(~np.isfinite(array))[0]
        raise ValueError(f"{name} contains non-finite values (first at row {bad[0]}, column {bad[1]})")
    return array


def check_parity(
    model: Any, head: str, network: SpikingNetwork, encoded_inputs: np.ndarray, *, logit_tolerance: float
) -> dict[str, Any]:
    """Run the torch model and the numpy network on the same encoded inputs (n, D).

    Compared, on every row:

    * spike trains: for every hidden layer l, the (T, n, H_l) 0/1 train of every neuron at
      every step (torch spikes against ``SpikingNetwork.forward_with_spike_trains``);
    * logits (n, C): ``max |torch - numpy|`` must not exceed ``logit_tolerance``
      (config: students.snn.parity_logit_tolerance);
    * decisions: argmax of the torch logits against the batched numpy decisions;
    * the deployed per-decision path: ``network.decide_one(row)`` for each row (D,) alone
      against the batched numpy decisions and against torch.

    Args:
        logit_tolerance: >= 0, finite.

    Returns:
        A JSON-serializable report of every comparison (counts of mismatching cells,
        differing decisions, the largest logit difference and the tolerance).

    Raises:
        ParityError if any spike-train cell, any batched decision or any per-decision result
        differs, or the logit difference exceeds the tolerance (``error.report`` holds the report).
        ValueError if the network's structure is not the model's trunk plus ``head``, or for
        non-finite inputs or an invalid tolerance.
    """
    import torch

    _require_model(model)
    if head not in model.head_linears:
        raise KeyError(f"unknown head {head!r}; heads: {list(model.head_linears)}")
    if isinstance(logit_tolerance, bool) or not isinstance(logit_tolerance, (int, float, np.integer, np.floating)):
        raise ValueError(f"logit_tolerance must be a number; got {logit_tolerance!r}")
    tolerance = float(logit_tolerance)
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        raise ValueError(f"logit_tolerance must be finite and >= 0; got {tolerance}")
    inputs = _finite_matrix("encoded_inputs", encoded_inputs, network.n_inputs)
    expected = (model.n_inputs, *model.hidden_sizes, model.heads[head])
    if network.layer_sizes != expected or network.time_steps != model.time_steps:
        raise ValueError(
            f"network layers {network.layer_sizes} x T={network.time_steps} are not the model's trunk plus head "
            f"{head!r}: {expected} x T={model.time_steps}"
        )
    with torch.no_grad():
        torch_logits, torch_spikes = model(torch.as_tensor(inputs, dtype=torch.float64), head, record=True)
    torch_logits = torch_logits.numpy()
    torch_trains = [layer.numpy() for layer in torch_spikes]  # (T, n, H_l) float64 0.0 / 1.0
    numpy_logits, numpy_trains = network.forward_with_spike_trains(inputs)
    layers = range(len(model.hidden_sizes))
    train_mismatches = [int(np.sum(torch_trains[layer] != numpy_trains[layer])) for layer in layers]
    torch_decisions = np.argmax(torch_logits, axis=1)  # first maximum on ties, as torch.argmax
    numpy_decisions = np.argmax(numpy_logits, axis=1)
    single_decisions = np.array([network.decide_one(row) for row in inputs], dtype=np.int64)
    differing = int(np.sum(torch_decisions != numpy_decisions))
    single_vs_batch = int(np.sum(single_decisions != numpy_decisions))
    single_vs_torch = int(np.sum(single_decisions != torch_decisions))
    max_logit_difference = float(np.max(np.abs(torch_logits - numpy_logits)))
    report = {
        "n_inputs_checked": int(inputs.shape[0]),
        "logit_tolerance": tolerance,
        "max_abs_logit_difference": max_logit_difference,
        "spike_train_mismatches_per_layer": train_mismatches,
        "spike_train_cells_per_layer": [int(train.size) for train in numpy_trains],
        "total_spikes_per_layer_torch": [int(np.rint(train.sum())) for train in torch_trains],
        "total_spikes_per_layer_numpy": [int(train.sum(dtype=np.int64)) for train in numpy_trains],
        "decisions_differing": differing,
        "decision_agreement": float(np.mean(torch_decisions == numpy_decisions)),
        "decide_one_vs_batch_differing": single_vs_batch,
        "decide_one_vs_torch_differing": single_vs_torch,
    }
    problems = []
    if any(train_mismatches):
        problems.append(f"spike-train mismatches per layer {train_mismatches}")
    if differing:
        problems.append(f"{differing} of {inputs.shape[0]} batched decisions differ")
    if single_vs_batch or single_vs_torch:
        problems.append(
            f"the per-decision path (decide_one) differs from the batched decisions on {single_vs_batch} rows "
            f"and from torch on {single_vs_torch} rows"
        )
    if not max_logit_difference <= tolerance:  # also catches a NaN difference
        problems.append(f"max |logit difference| {max_logit_difference!r} exceeds the tolerance {tolerance!r}")
    if problems:
        raise ParityError(
            f"the numpy network does not reproduce the torch model on head {head!r} ({inputs.shape[0]} rows): "
            + "; ".join(problems),
            report,
        )
    return report


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingReport:
    """What :func:`fit` did. Everything is measured on the data it was given (dream rows).

    Attributes:
        heads, sampling:              trained heads in order; their batch shares.
        n_train_rows, n_validation_rows: rows per head.
        steps_run:                    optimizer steps taken.
        best_step:                    step whose weights were restored (0 = initialisation).
        stopped_early:                stopped by patience before ``max_steps``.
        initial_validation_loss:      total validation loss at initialisation.
        best_validation_loss:         total validation loss of the restored weights.
        best_validation_loss_per_head: {head: validation KL of the restored weights}.
        constant_predictor_validation_loss: {head: validation KL of the best constant prediction,
                                      the head's mean training target; inf when that mean gives
                                      zero probability to a candidate a validation row targets}.
        train_loss_curve:             ((step, mean training loss since the previous evaluation), ...).
        validation_loss_curve:        ((step, {head: validation KL}), ...), step 0 = initialisation.
        validation_total_curve:       ((step, sampling-weighted total), ...).
        initial_firing_rate_per_layer, firing_rate_per_layer: mean fraction of hidden neurons
                                      firing per step on the pooled validation inputs, at
                                      initialisation and for the restored weights.
        dead_fraction_per_layer:      neurons that never fire on any validation input at any step.
        saturated_fraction_per_layer: neurons that fire at every step on every validation input.
        silent_layers:                hidden layers that emit no spike at all on the validation
                                      inputs (fit raises TrainingError when there is one).
        wall_seconds, seed, torch_threads, versions {torch, snntorch, numpy, python}, optimizer.
    """

    heads: tuple[str, ...]
    sampling: dict[str, float]
    n_train_rows: dict[str, int]
    n_validation_rows: dict[str, int]
    steps_run: int
    best_step: int
    stopped_early: bool
    initial_validation_loss: float
    best_validation_loss: float
    best_validation_loss_per_head: dict[str, float]
    constant_predictor_validation_loss: dict[str, float]
    train_loss_curve: tuple[tuple[int, float], ...]
    validation_loss_curve: tuple[tuple[int, dict[str, float]], ...]
    validation_total_curve: tuple[tuple[int, float], ...]
    initial_firing_rate_per_layer: tuple[float, ...]
    firing_rate_per_layer: tuple[float, ...]
    dead_fraction_per_layer: tuple[float, ...]
    saturated_fraction_per_layer: tuple[float, ...]
    silent_layers: tuple[int, ...]
    wall_seconds: float
    seed: int
    torch_threads: int
    versions: dict[str, str]
    optimizer: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable dict (lists instead of tuples)."""
        return {
            "loss": (
                "per head KL(target || softmax(logits)) averaged over that head's rows, weighted by the head's "
                "share of the batch, plus spike_rate_penalty x mean hidden firing rate; early stopping on the "
                "sampling-weighted validation KL"
            ),
            "acceptance": (
                "the restored weights must have a total validation KL below initialisation, a validation KL per "
                "head below that of the head's mean training target (the best constant prediction), and spikes "
                "in every hidden layer on the validation rows; otherwise fit raises TrainingError"
            ),
            "heads": list(self.heads),
            "sampling": dict(self.sampling),
            "n_train_rows": dict(self.n_train_rows),
            "n_validation_rows": dict(self.n_validation_rows),
            "steps_run": self.steps_run,
            "best_step": self.best_step,
            "stopped_early": self.stopped_early,
            "initial_validation_loss": self.initial_validation_loss,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_loss_per_head": dict(self.best_validation_loss_per_head),
            "constant_predictor_validation_loss": {
                head: (value if math.isfinite(value) else None)
                for head, value in self.constant_predictor_validation_loss.items()
            },
            "constant_predictor_validation_loss_null_means": "infinite",
            "train_loss_curve": [[step, value] for step, value in self.train_loss_curve],
            "validation_loss_curve": [[step, dict(values)] for step, values in self.validation_loss_curve],
            "validation_total_curve": [[step, value] for step, value in self.validation_total_curve],
            "initial_firing_rate_per_layer": list(self.initial_firing_rate_per_layer),
            "firing_rate_per_layer": list(self.firing_rate_per_layer),
            "dead_fraction_per_layer": list(self.dead_fraction_per_layer),
            "saturated_fraction_per_layer": list(self.saturated_fraction_per_layer),
            "silent_layers": list(self.silent_layers),
            "wall_seconds": self.wall_seconds,
            "seed": self.seed,
            "torch_threads": self.torch_threads,
            "versions": dict(self.versions),
            "optimizer": dict(self.optimizer),
        }


def _check_task_data(model: Any, label: str, data: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    checked = {}
    for head, pair in data.items():
        if head not in model.head_linears:
            raise KeyError(f"{label}: unknown head {head!r}; the model's heads are {list(model.head_linears)}")
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{label}[{head!r}] must be a tuple (encoded inputs (n, D), soft targets (n, C))")
        inputs = _finite_matrix(f"{label}[{head!r}] inputs", pair[0], model.n_inputs)
        targets = _finite_matrix(f"{label}[{head!r}] targets", pair[1], model.heads[head])
        if targets.shape[0] != inputs.shape[0]:
            raise ValueError(f"{label}[{head!r}]: {inputs.shape[0]} input rows but {targets.shape[0]} target rows")
        if np.any(targets < 0.0) or np.max(np.abs(targets.sum(axis=1) - 1.0)) > _ROW_SUM_TOLERANCE:
            raise ValueError(f"{label}[{head!r}] targets must be non-negative rows summing to 1")
        checked[head] = (inputs, targets)
    return checked


def _check_sampling(heads: Sequence[str], sampling: Mapping[str, float]) -> np.ndarray:
    if set(sampling) != set(heads):
        raise ValueError(f"sampling must name exactly the trained heads {list(heads)}; got {list(sampling)}")
    weights = []
    for head in heads:
        value = sampling[head]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"sampling[{head!r}] must be a number; got {value!r}")
        if not (math.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"sampling[{head!r}] must be finite and > 0; got {value}")
        weights.append(float(value))
    total = float(np.sum(weights))
    if abs(total - 1.0) > _SAMPLING_SUM_TOLERANCE:
        raise ValueError(f"sampling weights must sum to 1; got {total}")
    return np.asarray(weights, dtype=np.float64) / total


def _batch_counts(weights: np.ndarray, batch_size: int, generator: Any) -> np.ndarray:
    """Rows per head in one batch: floor(batch_size x weight) each, and the remaining rows go
    to heads drawn without replacement with probability proportional to the fractional parts,
    so the expected count of every head is exactly batch_size x weight."""
    import torch

    exact = weights * batch_size
    counts = np.floor(exact).astype(np.int64)
    remainder = int(batch_size - counts.sum())
    if remainder > 0:
        fractions = torch.as_tensor(exact - counts, dtype=torch.float64)
        picks = torch.multinomial(fractions, remainder, replacement=False, generator=generator)
        counts[picks.numpy()] += 1
    return counts


def validation_losses(model: Any, validation: Mapping[str, tuple[Any, Any]]) -> dict[str, float]:
    """{head: KL(target || softmax(logits)) averaged over that head's rows}, without gradients.

    ``validation[head] = (encoded inputs, soft targets)`` as float64 torch tensors or arrays.
    """
    import torch
    import torch.nn.functional as functional

    _require_model(model)
    losses = {}
    with torch.no_grad():
        for head, (inputs, targets) in validation.items():
            inputs = torch.as_tensor(inputs, dtype=torch.float64)
            targets = torch.as_tensor(targets, dtype=torch.float64)
            logits = model(inputs, head)
            losses[head] = float(functional.kl_div(functional.log_softmax(logits, dim=1), targets, reduction="batchmean"))
    return losses


def spike_statistics(model: Any, inputs_by_head: Mapping[str, Any]) -> dict[str, list[float] | list[int]]:
    """Hidden-layer firing statistics pooled over the rows of every head's encoded inputs.

    Returns {"firing_rate_per_layer", "dead_fraction_per_layer", "saturated_fraction_per_layer",
    "silent_layers"} (see :class:`TrainingReport`).
    """
    import torch

    _require_model(model)
    totals = [None] * len(model.hidden_sizes)
    always = [None] * len(model.hidden_sizes)
    cells = 0
    with torch.no_grad():
        for head, inputs in inputs_by_head.items():
            _, spikes = model(torch.as_tensor(inputs, dtype=torch.float64), head, record=True)
            cells += int(spikes[0].shape[0] * spikes[0].shape[1])
            for layer, layer_spikes in enumerate(spikes):
                per_neuron = layer_spikes.sum(dim=(0, 1))
                fires_always = layer_spikes.amin(dim=(0, 1))
                totals[layer] = per_neuron if totals[layer] is None else totals[layer] + per_neuron
                always[layer] = fires_always if always[layer] is None else torch.minimum(always[layer], fires_always)
    if cells == 0:
        raise ValueError("spike statistics need at least one input row")
    rates, dead, saturated, silent = [], [], [], []
    for layer, (total, minimum) in enumerate(zip(totals, always)):
        rates.append(float(total.sum()) / float(cells * total.numel()))
        dead.append(float((total == 0).double().mean()))
        saturated.append(float((minimum == 1).double().mean()))
        if float(total.sum()) == 0.0:
            silent.append(layer)
    return {
        "firing_rate_per_layer": rates,
        "dead_fraction_per_layer": dead,
        "saturated_fraction_per_layer": saturated,
        "silent_layers": silent,
    }


def constant_predictor_loss(train_targets: np.ndarray, validation_targets: np.ndarray) -> float:
    """Validation KL of the best constant prediction learned from the training rows.

    The constant prediction is the mean training target p = mean over rows of
    ``train_targets`` (n_train, C). Returns ``sum_i sum_c q_ic (log q_ic - log p_c) / n_val``
    over ``validation_targets`` q (n_val, C), the same batch-mean KL that
    :func:`validation_losses` reports for a network (terms with q_ic = 0 count 0). Returns inf
    when p gives zero probability to a candidate that a validation row targets.
    """
    train = np.asarray(train_targets, dtype=np.float64)
    validation = np.asarray(validation_targets, dtype=np.float64)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1] or validation.shape[0] < 1:
        raise ValueError(
            f"train_targets (n, C) and validation_targets (m >= 1, C) must share C; got {train.shape} and {validation.shape}"
        )
    mean = train.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(validation > 0.0, validation * (np.log(validation) - np.log(mean)), 0.0)
    return float(terms.sum() / validation.shape[0])


def _versions() -> dict[str, str]:
    import snntorch
    import torch

    return {
        "torch": str(torch.__version__),
        "snntorch": str(snntorch.__version__),
        "numpy": str(np.__version__),
        "python": platform.python_version(),
    }


def fit(
    model: Any,
    tasks: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    validation: Mapping[str, tuple[np.ndarray, np.ndarray]],
    sampling: Mapping[str, float],
    settings: SNNSettings,
    seed: int,
    on_evaluation: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingReport:
    """Train ``model`` inside the dream on one or more heads; restore the best weights.

    Args:
        tasks:      {head: (encoded inputs (n, D), soft targets (n, C_head))} training rows.
        validation: the same layout for every head in ``tasks``: the early-stopping rows (never
                    trained on, but they choose the restored weights, so they are not held out).
        sampling:   {head: share of each batch}, > 0 and summing to 1 (``{head: 1.0}`` for one head).
        settings:   batch size, learning rate, step limits, penalty and threads.
        seed:       batch order (seeded ``torch.Generator``) and ``torch.manual_seed`` inside a
                    forked RNG, so the caller's global torch RNG is unchanged.
        on_evaluation: optional observer, called once before the first step (``step`` 0) and after
                    every validation evaluation with a dict of plain values: ``step``,
                    ``max_steps``, ``train_loss`` (mean training loss since the previous
                    evaluation; None at step 0), ``validation_loss`` (sampling-weighted total),
                    ``validation_loss_per_head``, ``best_validation_loss``, ``best_step`` and
                    ``firing_rate_per_layer`` (hidden firing rates on the validation rows). It
                    only reads: training with and without it gives the same weights. An
                    exception it raises stops the training and propagates to the caller, with
                    the torch settings restored; the model is then left partly trained.

    Each optimizer step draws a batch whose rows come from the heads in the ``sampling``
    proportions (exact in expectation; each head's rows are visited in seeded epoch-wise
    permutations, so a head with fewer rows than its share repeats rows within a batch). The
    loss is, per head present, KL(target || softmax(logits)) averaged over that head's rows,
    weighted by the head's share of the batch, plus ``spike_rate_penalty`` x the mean hidden
    firing rate of the batch; one Adam step follows (``learning_rate``, ``adam_beta1``,
    ``adam_beta2``, ``adam_eps`` and ``weight_decay`` from the settings). Gradients are reset to
    ``None`` so a head absent from a batch is not moved by Adam. Every ``evaluate_every_steps``
    steps (and after the last step) the validation KL per head is evaluated; training stops at
    ``max_steps`` or after ``patience_evaluations`` evaluations without a lower
    sampling-weighted total, and the best weights are restored. Torch threads and the
    deterministic-algorithms flag are set for the run and restored afterwards.

    Raises:
        TrainingError if a loss becomes non-finite, or if the restored network (a) has a total
        validation loss that is not below its loss at initialisation, (b) has, for some head, a
        validation loss that is not below :func:`constant_predictor_loss` of that head (its mean
        training target evaluated on its validation rows), or (c) has a hidden layer that emits
        no spike on the validation rows. ``error.report`` holds the report in cases (a)-(c).
    """
    import torch
    import torch.nn.functional as functional

    _require_model(model)
    if not isinstance(settings, SNNSettings):
        raise TypeError(f"settings must be SNNSettings; got {type(settings).__name__}")
    seed = _seed(seed)
    if not tasks:
        raise ValueError("tasks is empty: name at least one head to train")
    train_data = _check_task_data(model, "tasks", tasks)
    if set(validation) != set(train_data):
        raise ValueError(f"validation must hold exactly the trained heads {list(train_data)}; got {list(validation)}")
    validation_data = _check_task_data(model, "validation", validation)
    heads = tuple(train_data)
    weights = _check_sampling(heads, sampling)
    shares = {head: float(weight) for head, weight in zip(heads, weights)}

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    started = time.perf_counter()
    try:
        torch.set_num_threads(settings.torch_threads)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_tensors = {
                head: (torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(q, dtype=torch.float64))
                for head, (x, q) in train_data.items()
            }
            validation_tensors = {
                head: (torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(q, dtype=torch.float64))
                for head, (x, q) in validation_data.items()
            }
            validation_inputs = {head: x for head, (x, _) in validation_tensors.items()}
            cursors: dict[str, list[Any]] = {head: [torch.empty(0, dtype=torch.int64), 0] for head in heads}

            def draw(head: str, count: int) -> Any:
                pieces = []
                while count > 0:
                    order, position = cursors[head]
                    if position >= order.numel():
                        order = torch.randperm(train_tensors[head][0].shape[0], generator=generator)
                        position = 0
                    take = min(count, order.numel() - position)
                    pieces.append(order[position : position + take])
                    cursors[head] = [order, position + take]
                    count -= take
                return torch.cat(pieces)

            def total_of(losses: Mapping[str, float]) -> float:
                return float(sum(shares[head] * losses[head] for head in heads))

            initial_statistics = spike_statistics(model, validation_inputs)
            initial_losses = validation_losses(model, validation_tensors)
            initial_total = total_of(initial_losses)
            if not math.isfinite(initial_total):
                raise TrainingError(f"the validation loss at initialisation is not finite: {initial_losses}")
            validation_curve: list[tuple[int, dict[str, float]]] = [(0, initial_losses)]
            total_curve: list[tuple[int, float]] = [(0, initial_total)]
            train_curve: list[tuple[int, float]] = []
            best_total, best_step = initial_total, 0
            if on_evaluation is not None:
                on_evaluation({
                    "step": 0,
                    "max_steps": settings.max_steps,
                    "train_loss": None,
                    "validation_loss": initial_total,
                    "validation_loss_per_head": dict(initial_losses),
                    "best_validation_loss": best_total,
                    "best_step": best_step,
                    "firing_rate_per_layer": list(initial_statistics["firing_rate_per_layer"]),
                })
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_evaluations = 0
            stopped_early = False
            interval_losses: list[float] = []
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=settings.learning_rate,
                betas=(settings.adam_beta1, settings.adam_beta2),
                eps=settings.adam_eps,
                weight_decay=settings.weight_decay,
            )
            penalise = settings.spike_rate_penalty > 0.0
            step = 0
            model.train()
            while step < settings.max_steps:
                counts = _batch_counts(weights, settings.batch_size, generator)
                optimizer.zero_grad(set_to_none=True)
                loss = None
                spike_total, neuron_steps = None, 0
                for head, count in zip(heads, counts):
                    if count == 0:
                        continue
                    rows = draw(head, int(count))
                    inputs, targets = train_tensors[head][0][rows], train_tensors[head][1][rows]
                    if penalise:
                        logits, spikes = model(inputs, head, record=True)
                        for layer_spikes in spikes:
                            layer_sum = layer_spikes.sum()
                            spike_total = layer_sum if spike_total is None else spike_total + layer_sum
                            neuron_steps += int(layer_spikes.numel())
                    else:
                        logits = model(inputs, head)
                    head_loss = functional.kl_div(functional.log_softmax(logits, dim=1), targets, reduction="batchmean")
                    term = (float(count) / float(settings.batch_size)) * head_loss
                    loss = term if loss is None else loss + term
                if penalise:
                    loss = loss + settings.spike_rate_penalty * (spike_total / float(neuron_steps))
                if not bool(torch.isfinite(loss)):
                    raise TrainingError(f"the training loss became non-finite at step {step + 1}")
                loss.backward()
                optimizer.step()
                step += 1
                interval_losses.append(float(loss.detach()))
                if step % settings.evaluate_every_steps == 0 or step == settings.max_steps:
                    losses = validation_losses(model, validation_tensors)
                    total = total_of(losses)
                    if not math.isfinite(total):
                        raise TrainingError(f"the validation loss became non-finite at step {step}: {losses}")
                    train_curve.append((step, float(np.mean(interval_losses))))
                    interval_losses = []
                    validation_curve.append((step, losses))
                    total_curve.append((step, total))
                    if total < best_total:
                        best_total, best_step = total, step
                        best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
                        stale_evaluations = 0
                    else:
                        stale_evaluations += 1
                    if on_evaluation is not None:
                        on_evaluation({
                            "step": step,
                            "max_steps": settings.max_steps,
                            "train_loss": train_curve[-1][1],
                            "validation_loss": total,
                            "validation_loss_per_head": dict(losses),
                            "best_validation_loss": best_total,
                            "best_step": best_step,
                            "firing_rate_per_layer": list(
                                spike_statistics(model, validation_inputs)["firing_rate_per_layer"]
                            ),
                        })
                    if stale_evaluations >= settings.patience_evaluations and step < settings.max_steps:
                        stopped_early = True
                        break
            model.load_state_dict(best_weights)
            model.eval()
            final_statistics = spike_statistics(model, validation_inputs)
            optimizer_record = {
                "name": "Adam",
                "lr": float(optimizer.defaults["lr"]),
                "betas": [float(value) for value in optimizer.defaults["betas"]],
                "eps": float(optimizer.defaults["eps"]),
                "weight_decay": float(optimizer.defaults["weight_decay"]),
            }
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)

    best_losses = next(dict(losses) for curve_step, losses in validation_curve if curve_step == best_step)
    constant_losses = {
        head: constant_predictor_loss(train_data[head][1], validation_data[head][1]) for head in heads
    }
    report = TrainingReport(
        heads=heads,
        sampling=shares,
        n_train_rows={head: int(x.shape[0]) for head, (x, _) in train_data.items()},
        n_validation_rows={head: int(x.shape[0]) for head, (x, _) in validation_data.items()},
        steps_run=step,
        best_step=best_step,
        stopped_early=stopped_early,
        initial_validation_loss=initial_total,
        best_validation_loss=best_total,
        best_validation_loss_per_head=best_losses,
        constant_predictor_validation_loss=constant_losses,
        train_loss_curve=tuple(train_curve),
        validation_loss_curve=tuple(validation_curve),
        validation_total_curve=tuple(total_curve),
        initial_firing_rate_per_layer=tuple(initial_statistics["firing_rate_per_layer"]),
        firing_rate_per_layer=tuple(final_statistics["firing_rate_per_layer"]),
        dead_fraction_per_layer=tuple(final_statistics["dead_fraction_per_layer"]),
        saturated_fraction_per_layer=tuple(final_statistics["saturated_fraction_per_layer"]),
        silent_layers=tuple(final_statistics["silent_layers"]),
        wall_seconds=float(time.perf_counter() - started),
        seed=seed,
        torch_threads=settings.torch_threads,
        versions=_versions(),
        optimizer=optimizer_record,
    )
    problems = []
    if not best_total < initial_total:
        problems.append(
            f"the trained network's validation loss {best_total} is not below its loss at initialisation "
            f"{initial_total} (steps run: {step})"
        )
    for head in heads:
        if not best_losses[head] < constant_losses[head]:
            problems.append(
                f"head {head!r}: the restored validation loss {best_losses[head]} is not below "
                f"{constant_losses[head]}, the loss of the best constant prediction (the head's mean training "
                "target): the network does no better than ignoring the situation"
            )
    if report.silent_layers:
        problems.append(
            f"hidden layer(s) {list(report.silent_layers)} emit no spike on the validation rows: the decisions "
            "cannot depend on the situation"
        )
    if problems:
        raise TrainingError("; ".join(problems), report)
    return report
