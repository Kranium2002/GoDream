"""Review r08: paths that must NOT bill twice: a repeated identical prediction, and per_outcome mode
with two outcomes whose residual columns are identical (same request key within one call).

FixtureProvider (see _fixture.py) replaces only the network; it also counts free usage/quote reads.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\teacher\\r08_no_double_send_paths.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import tempfile
from pathlib import Path

import _fixture as fx


class CountingProvider(fx.FixtureProvider):
    free_reads = 0

    def usage(self):
        CountingProvider.free_reads += 1
        return super().usage()

    def quote(self, *args):
        CountingProvider.free_reads += 1
        return super().quote(*args)


for mode in ("stacked", "per_outcome"):
    with tempfile.TemporaryDirectory() as tmp:
        teacher, provider, account, log, settings = fx.make_teacher(Path(tmp), mode=mode)
        teacher.provider = provider = CountingProvider(settings, account)
        features, residuals, query = fx.arrays()
        residuals[:, 2] = residuals[:, 0]  # two identical outcome columns
        teacher.fit(features, residuals)
        first = teacher.predict_residual_distribution(query)
        reads_after_first = CountingProvider.free_reads
        second = teacher.predict_residual_distribution(query)
        print(f"== {mode} ==")
        print("  billable predicts:", account.billed_predicts, "| events:", [r["event"] for r in log.read()])
        print("  free usage/quote reads on the repeated call:", CountingProvider.free_reads - reads_after_first)
        print("  request keys of the two calls:", [k[:8] for k in first and teacher.last_request_hashes])
        print("  repeated call returns identical arrays:", bool((first.quantiles == second.quantiles).all()))
        CountingProvider.free_reads = 0
