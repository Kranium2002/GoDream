"""R09: how much does check_parity prove when the validation split is small?

train_in_dream checks parity on the validation rows only (n_validation = round(fraction x n),
at least 1; R08 shows 1 row for 3-5 dreams or a tiny fraction). check_parity compares spike COUNTS
per (step, row) and argmax decisions; the logit difference is reported but has no tolerance.
This script takes a FIXTURE model, exports it, and deliberately mis-exports one constant that does
not touch hidden spikes (readout_beta 0.9 -> 0.5), then runs check_parity on 1 row and on 600 rows.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

import _review_common  # noqa: E402,F401
from fixture_snn_spec import FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.students import snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn_train import ParityError, SNNSettings  # noqa: E402

config = load_config()
settings = SNNSettings.from_config(config)
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 600)
x = InputEncoder.from_variables(spec.situation).encode(dreams.situations)
model = snn_train.build_model(x.shape[1], {"h": dreams.n_candidates}, settings, seed=3)
good = snn_train.export_network(model, "h")
bad = dataclasses.replace(good, readout_beta=0.5)
for label, rows in (("1 row", x[:1]), ("10 rows", x[:10]), ("600 rows", x)):
    try:
        report = snn_train.check_parity(model, "h", bad, rows)
        print(f"{label}: mis-exported network ACCEPTED; spike mismatches {report['spike_count_mismatches_per_layer']}, "
              f"decisions differing {report['decisions_differing']}, max |logit diff| {report['max_abs_logit_difference']:.3f}")
    except ParityError as error:
        print(f"{label}: ParityError: {error.report['decisions_differing']} decisions differ, max |logit diff| {error.report['max_abs_logit_difference']:.3f}")
print(f"decisions of the mis-exported vs correct network on 600 rows differ in {int(np.sum(bad.decide(x) != good.decide(x)))} rows")
