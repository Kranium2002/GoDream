"""Review reproduction: can the sealed test situations be obtained without final_evaluation=True
or without a line in the project's test access log (config evaluation.test_access_log)?

Uses the B_eval_offline FIXTURE spec only. The project's results/test_access_log.jsonl is not
touched; the one sanctioned call below writes to a temporary file on purpose, to show that the
log location is caller-chosen.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "proofs" / "B_eval_offline"))

from fixture_spec import FixtureSpec  # noqa: E402

from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.eval.splits import noise_seeds, split_situations, test_situations  # noqa: E402

print("REVIEW REPRODUCTION - fixture, not evidence")

spec = FixtureSpec()
BASE, N, PRESET = 20261121, 100, "default"
project_log = REPO / "results" / "test_access_log.jsonl"
before = project_log.read_bytes() if project_log.exists() else None

# (1) public stream API, no flag, no log
rng = stream_rng(BASE, f"{spec.name}/test/{PRESET}/situations", 0)
via_streams = np.asarray(spec.sample_situations(rng, N), dtype=np.float64)

# (2) sanctioned API, but the log path is caller-chosen -> the audit line lands in a temp file
with tempfile.TemporaryDirectory() as tmp:
    sealed = test_situations(
        spec,
        base_seed=BASE,
        n=N,
        world_preset=PRESET,
        final_evaluation=True,
        access_log_path=Path(tmp) / "throwaway.jsonl",
        clock=lambda: "2026-11-21T00:00:00+00:00",
    )
    # (3) a second and third 'final' access are allowed; nothing limits the count
    for _ in range(2):
        test_situations(spec, base_seed=BASE, n=N, world_preset=PRESET, final_evaluation=True,
                        access_log_path=Path(tmp) / "throwaway.jsonl", clock=lambda: "2026-11-21T00:00:00+00:00")
    lines = (Path(tmp) / "throwaway.jsonl").read_text(encoding="utf-8").count("\n")

after = project_log.read_bytes() if project_log.exists() else None
print("(1) stream_rng reconstruction == sealed test situations (bitwise):", np.array_equal(via_streams, sealed))
print("(2) project test access log changed:", before != after, "| exists:", project_log.exists())
print("(3) three final-evaluation accesses allowed; lines in caller-chosen log:", lines)
print("    test noise seeds available without the flag:", noise_seeds(spec, base_seed=BASE, split="test", world_preset=PRESET, n_situations=2, n_repeats=1).ravel().tolist())

# (4) prefix stability: the noise seed of situation index i does not depend on n, the situation does
d20 = split_situations(spec, base_seed=BASE, split="dev", n=20, world_preset=PRESET)
d100 = split_situations(spec, base_seed=BASE, split="dev", n=100, world_preset=PRESET)
s20 = noise_seeds(spec, base_seed=BASE, split="dev", world_preset=PRESET, n_situations=20, n_repeats=3)
s100 = noise_seeds(spec, base_seed=BASE, split="dev", world_preset=PRESET, n_situations=100, n_repeats=3)
print("(4) dev n=20 rows == first 20 rows of dev n=100:", np.array_equal(d20, d100[:20]),
      "| identical rows:", int(np.sum(np.all(d20 == d100[:20], axis=1))), "of 20")
print("    noise seeds n=20 == first 20 of n=100:", np.array_equal(s20, s100[:20]))
print("    dev[0] at n=20 :", d20[0].tolist())
print("    dev[0] at n=100:", d100[0].tolist(), "-> same noise seed", int(s20[0, 0]), "==", int(s100[0, 0]))
