"""MECHANICS FIXTURE ONLY - not project evidence.

Executes the published operation-count energy formulas on a made-up layer layout and made-up
spike counts, so the lead engineer can check units and the arithmetic of each variant.
Every number printed here comes from FIXTURE inputs (layer sizes, T, spike counts invented
below). None of it is a measurement of the robot, the simulator, TabPFN, or the final
student, and it must never be quoted as a project result.

Per-operation constants: Horowitz, ISSCC 2014, Fig. 1.1.9 "Rough energy costs for various
operations in 45nm 0.9V" (DOI 10.1109/ISSCC.2014.6757323), decoded from the PDF content
stream by decode_horowitz_fig119.py in this folder. Label: estimate, 45 nm CMOS.

Formulas implemented (verbatim sources in excerpts.txt; citations in bibliography.txt, this folder):
  dense reference      E_ANN = E_MAC * sum_l FLOPs_l, FLOPs_l = N_{l-1} * N_l       (Kundu WACV'21 Eq.11; Spikformer Eq.23)
  SNN synaptic ops     SOPs_l = spikes_in_l * N_l  (= fr * T * FLOPs_l)              (Spikformer Eq.21; DIET-SNN Eq.17-18)
  SNN, first layer
    real-valued input  E_MAC * FLOPs_1, counted once  (Spikformer Eq.22)            -> variant A
                       E_MAC * T * FLOPs_1            (SDT Table 1 "First Conv", R_C=1 for analog input) -> variant B
  neuron-update terms  per neuron per step: leak multiply (MAC) + bias add (ACC); reset ACC per output spike
                       (Lemaire et al. 2022, Eq.1-2 and text)                       -> variant C = A + these
Memory accesses are NOT counted in any variant (see caveats).
"""

HOROWITZ_45NM_PJ = {
    # operation: (add_pJ, mult_pJ)  -- Fig. 1.1.9, 45 nm, 0.9 V
    "int8": (0.03, 0.2),
    "int32": (0.1, 3.1),
    "fp16": (0.4, 1.1),
    "fp32": (0.9, 3.7),
}


def constants(precision: str):
    add, mult = HOROWITZ_45NM_PJ[precision]
    return {"E_MAC": mult + add, "E_AC": add}


def dense_energy_pj(sizes, e_mac, e_ac, count_bias=False):
    macs = sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))
    bias_adds = sum(sizes[1:]) if count_bias else 0
    return macs * e_mac + bias_adds * e_ac, macs, bias_adds


def snn_energy_pj(sizes, T, spikes_out_per_layer, e_mac, e_ac, first_layer="once", neuron_updates=False):
    """sizes = [d_in, H1, ..., C]; layer 1 receives the real-valued input (direct encoding).
    spikes_out_per_layer[k] = total spikes emitted by hidden layer k+1 over all T steps
    (one entry per hidden spiking layer; the output layer is a non-spiking membrane readout)."""
    d_in, n1 = sizes[0], sizes[1]
    flops1 = d_in * n1
    first_macs = flops1 * (1 if first_layer == "once" else T)
    sops = 0
    for k, n_next in enumerate(sizes[2:]):
        sops += spikes_out_per_layer[k] * n_next  # each spike -> one AC per fan-out synapse
    energy = first_macs * e_mac + sops * e_ac
    extra = {}
    if neuron_updates:
        n_neurons = sum(sizes[1:])                 # hidden LIF + output leaky integrators
        leak_mults = n_neurons * T                 # beta * U each step (MAC-like, Lemaire)
        bias_adds = n_neurons * T                  # bias added every step (ACC, Lemaire)
        reset_ops = sum(spikes_out_per_layer)      # one ACC per emitted spike (subtractive reset)
        energy += leak_mults * e_mac + (bias_adds + reset_ops) * e_ac
        extra = {"leak_mults": leak_mults, "bias_adds": bias_adds, "reset_ops": reset_ops}
    return energy, first_macs, sops, extra


def main():
    # ---------------- FIXTURE INPUTS (invented) ----------------
    sizes = [10, 64, 64, 60]          # 10 features -> 64 LIF -> 64 LIF -> 60-way membrane readout
    T = 8
    spikes_out_per_layer = [96, 80]   # invented totals over T steps for the two hidden layers
    # ------------------------------------------------------------
    print("FIXTURE ONLY - invented sizes and spike counts; not project evidence")
    print(f"sizes={sizes} T={T} spikes_out_per_layer={spikes_out_per_layer}")
    hidden = sizes[1:-1]
    for k, (n, s) in enumerate(zip(hidden, spikes_out_per_layer)):
        print(f"  hidden layer {k+1}: spikes per neuron per inference={s/n:.3f}  per neuron per step={s/(n*T):.4f}")
    for prec in ("fp32", "int32", "int8"):
        c = constants(prec)
        e_ann, macs, _ = dense_energy_pj(sizes, c["E_MAC"], c["E_AC"])
        eA, m1A, sops, _ = snn_energy_pj(sizes, T, spikes_out_per_layer, c["E_MAC"], c["E_AC"], "once")
        eB, m1B, _, _ = snn_energy_pj(sizes, T, spikes_out_per_layer, c["E_MAC"], c["E_AC"], "per_step")
        eC, _, _, ex = snn_energy_pj(sizes, T, spikes_out_per_layer, c["E_MAC"], c["E_AC"], "once", True)
        print(f"[{prec}] E_MAC={c['E_MAC']:.2f} pJ  E_AC={c['E_AC']:.2f} pJ  (estimate, 45 nm CMOS)")
        print(f"   dense reference: MACs={macs}  E_ANN={e_ann:.1f} pJ")
        print(f"   SNN A (first layer once):     first-layer MACs={m1A} SOPs={sops}  E={eA:.1f} pJ  ratio E_ANN/E_SNN={e_ann/eA:.2f}")
        print(f"   SNN B (first layer x T):      first-layer MACs={m1B} SOPs={sops}  E={eB:.1f} pJ  ratio={e_ann/eB:.2f}")
        print(f"   SNN C (A + neuron updates):   {ex}  E={eC:.1f} pJ  ratio={e_ann/eC:.2f}")
        # SDT supplementary S1: SNN layer cheaper than its ANN layer iff E_AC * T * R < E_MAC
        print(f"   SDT break-even firing rate per neuron per step R* = E_MAC/(E_AC*T) = {c['E_MAC']/(c['E_AC']*T):.3f}")


if __name__ == "__main__":
    main()
