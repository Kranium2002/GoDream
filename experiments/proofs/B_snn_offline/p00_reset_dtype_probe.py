"""p00: does snnTorch's float32 spike/reset tensor make the 'subtract' reset lossy in a float64 model?

FIXTURE CHECK - mechanics only. Random inputs, no trained network, no project data.

Question: snn.Leaky computes reset = spike_grad(mem - threshold).clone().detach(), and the
surrogate's forward returns (x > 0).float(), i.e. float32. In the subtract mechanism
mem = beta*mem + input - reset*threshold; with reset float32 and threshold a 0-dim float64
tensor, torch's promotion rule keeps the product in float32. For a threshold that is not
exactly representable in float32 (e.g. 0.3) the float64 membrane would then differ from a
float64 numpy reference. This probe measures it with the default atan surrogate and with a
wrapper that casts the surrogate output to the input dtype.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\p00_reset_dtype_probe.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any trained student")

import numpy as np  # noqa: E402
import snntorch as snn  # noqa: E402
import torch  # noqa: E402
from snntorch import surrogate  # noqa: E402

torch.set_num_threads(1)


class CastSpike:
    def __init__(self, function):
        self.function = function

    def __call__(self, x):
        return self.function(x).to(x.dtype)


def numpy_reference(currents: np.ndarray, beta: float, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    mem = np.zeros(currents.shape[1:])
    mems, spikes = [], []
    for current in currents:
        reset = (mem > threshold).astype(np.float64)
        mem = (beta * mem + current) - reset * threshold
        spike = (mem > threshold).astype(np.float64)
        mems.append(mem.copy())
        spikes.append(spike)
    return np.stack(mems), np.stack(spikes)


rng = np.random.default_rng(0)
currents = rng.uniform(0.0, 0.8, size=(50, 200, 16))
for threshold in (1.0, 0.3, 0.7):
    for label, spike_grad in (("snntorch atan", surrogate.atan(alpha=2.0)), ("cast wrapper", CastSpike(surrogate.atan(alpha=2.0)))):
        neuron = snn.Leaky(
            beta=torch.tensor(0.9, dtype=torch.float64),
            threshold=torch.tensor(threshold, dtype=torch.float64),
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        ).to(torch.float64)
        mem = torch.zeros(200, 16, dtype=torch.float64)
        torch_mems, torch_spikes = [], []
        with torch.no_grad():
            for current in torch.as_tensor(currents):
                spike, mem = neuron(current, mem)
                torch_mems.append(mem.numpy().copy())
                torch_spikes.append(spike.to(torch.float64).numpy().copy())
        ref_mems, ref_spikes = numpy_reference(currents, 0.9, threshold)
        print(
            f"threshold={threshold:<4} {label:<14} spike dtype={spike.dtype} mem dtype={mem.dtype} "
            f"max|mem diff|={np.abs(np.stack(torch_mems) - ref_mems).max():.3e} "
            f"spike mismatches={int((np.stack(torch_spikes) != ref_spikes).sum())} of {ref_spikes.size}"
        )
