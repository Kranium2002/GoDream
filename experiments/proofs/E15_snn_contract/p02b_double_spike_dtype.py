"""p02b: in float64 (.double()) mode, snnTorch spikes come out float32 (surrogate forward does `.float()`),
so feeding them into the next float64 Linear fails unless cast. FIXTURE tensors only."""
import torch, torch.nn as nn, snntorch as snn
lif = snn.Leaky(beta=0.9).double()
fc2 = nn.Linear(4, 3).double()
cur = torch.randn(2, 4, dtype=torch.float64) * 3
spk, mem = lif(cur, torch.zeros(2, 4, dtype=torch.float64))
print("spk dtype:", spk.dtype, " mem dtype:", mem.dtype, " graded_spikes_factor dtype:", lif.graded_spikes_factor.dtype)
try:
    fc2(spk)
    print("fc2(spk) OK")
except RuntimeError as e:
    print("fc2(spk) -> RuntimeError:", e)
print("fc2(spk.to(torch.float64)) OK ->", fc2(spk.to(torch.float64)).dtype)
