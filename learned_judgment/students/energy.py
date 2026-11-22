"""Energy ESTIMATE of the spiking student from operation counts.

Nothing here is a measurement. The estimate multiplies counted operations by published
45 nm CMOS per-operation energies and follows the convention of the SNN literature
(sources and the saved excerpts: experiments/proofs/E15_snn_literature/):

    E_MAC  energy of one multiply-accumulate (a real-valued synapse)
    E_AC   energy of one accumulate (a synapse driven by a binary spike)

    main estimate      E_SNN = E_MAC * first_layer_macs + E_AC * synaptic_ops
                       first_layer_macs = D * H_1, counted once per decision because the code
                       computes the constant input current once
                       synaptic_ops = sum over hidden layers of spikes emitted * fan-out
    with neuron updates E_SNN + (E_MAC + E_AC) * neuron_updates + E_AC * total_spikes
                       adds the leak multiply and the input add of every membrane update and
                       one subtraction per reset, which the main convention leaves out
    dense reference    E_ANN = E_MAC * sum over layers of fan_in * fan_out
                       one pass through a non-spiking network with the same layer sizes

The per-operation constants are read from config.yaml (students.snn.energy) together with
their citation. Reports must carry the caveats returned by :func:`estimate_energy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from learned_judgment.config import require
from learned_judgment.students.snn_core import OperationCounts

__all__ = ["CAVEATS", "EnergyConstants", "estimate_energy"]

CAVEATS = (
    "Energy figures are analytical estimates from operation counts and published 45 nm CMOS "
    "per-operation energies. They are not measurements. The networks ran on a CPU laptop with "
    "dense matrix arithmetic, where a spiking network does not save energy.",
    "The estimate counts arithmetic only. Memory accesses for weights and membrane state, data "
    "movement, addressing, static power and spike communication are excluded, although published "
    "analyses find that memory accesses dominate.",
    "The result depends on conventions: arithmetic precision, whether the real-valued first layer "
    "is counted once or at every time step, and whether neuron leak, bias and reset updates are "
    "counted. The variant including neuron updates is reported next to the main estimate.",
    "The dense reference is a non-spiking network with the same layer sizes whose operation count "
    "follows from its architecture. No such network was trained, so the comparison says nothing "
    "about whether it would reach the same decision quality.",
    "Teacher (TabPFN) inference energy is not included. The comparison is per student decision.",
)


@dataclass(frozen=True)
class EnergyConstants:
    """Per-operation energies in picojoules and where they come from."""

    e_mac_pj: float
    e_ac_pj: float
    precision: str
    source: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EnergyConstants":
        e_mac = float(require(config, "students.snn.energy.e_mac_pj"))
        e_ac = float(require(config, "students.snn.energy.e_ac_pj"))
        if not (e_mac > 0.0 and e_ac > 0.0):
            raise ValueError("students.snn.energy.e_mac_pj and e_ac_pj must be positive")
        return cls(
            e_mac_pj=e_mac,
            e_ac_pj=e_ac,
            precision=str(require(config, "students.snn.energy.precision")),
            source=str(require(config, "students.snn.energy.source")),
        )


def estimate_energy(counts: OperationCounts, constants: EnergyConstants) -> dict[str, Any]:
    """Estimated energy per decision in picojoules, from measured spike counts.

    Returns a JSON-serializable dict. ``*_per_decision`` entries are means over the measured
    decisions; ``snn_main_pj_each`` keeps the per-decision values (n,) for a bootstrap interval.
    """
    spikes = counts.spikes_per_decision
    synaptic_ops = counts.synaptic_ops_per_decision
    total_spikes = spikes.sum(axis=1)
    first_layer = constants.e_mac_pj * counts.first_layer_macs
    main_each = first_layer + constants.e_ac_pj * synaptic_ops
    with_updates_each = (
        main_each
        + (constants.e_mac_pj + constants.e_ac_pj) * counts.neuron_updates
        + constants.e_ac_pj * total_spikes
    )
    dense = constants.e_mac_pj * counts.dense_reference_macs
    main = float(np.mean(main_each))
    with_updates = float(np.mean(with_updates_each))
    return {
        "label": "estimate, 45 nm CMOS, arithmetic only",
        "constants": {
            "e_mac_pj": constants.e_mac_pj,
            "e_ac_pj": constants.e_ac_pj,
            "precision": constants.precision,
            "source": constants.source,
        },
        "n_decisions": counts.n_decisions,
        "layer_sizes": list(counts.layer_sizes),
        "time_steps": counts.time_steps,
        "first_layer_macs": counts.first_layer_macs,
        "first_layer_counted": "once per decision",
        "neuron_updates": counts.neuron_updates,
        "dense_reference_macs": counts.dense_reference_macs,
        "mean_spikes_per_layer": [float(value) for value in counts.mean_spikes_per_layer],
        "firing_rate_per_layer": [float(value) for value in counts.firing_rate_per_layer],
        "mean_synaptic_ops": float(np.mean(synaptic_ops)),
        "snn_main_pj_per_decision": main,
        "snn_with_neuron_updates_pj_per_decision": with_updates,
        "dense_reference_pj_per_decision": float(dense),
        "ratio_snn_main_to_dense": main / dense,
        "ratio_snn_with_neuron_updates_to_dense": with_updates / dense,
        "snn_main_pj_each": [float(value) for value in main_each],
        "caveats": list(CAVEATS),
    }
