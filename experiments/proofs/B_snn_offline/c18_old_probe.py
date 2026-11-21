"""Check 18 worker: the scenarios of c7-c14 run through the OLD API of this lane.

FIXTURE CHECK - mechanics only. Started by c18_old_behaviour.py inside its temporary overlay, where
students/snn.py, snn_train.py, targets.py and engine/train_in_dream.py are the committed versions.
It uses only names and signatures the committed code had, and prints one line per review item:
"OLD <item>: DEFECT <what the old code did>" or "OLD <item>: NO DEFECT <...>". It is not meant to
run against the current code (the signatures changed). Writes no file.
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, FixtureSideSpec, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students import snn, snn_train  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402
from learned_judgment.students.targets import score_spread, soft_targets  # noqa: E402

BASE_SEED = 3  # fixture run seed


def report(item: str, defect: bool, text: str) -> None:
    print(f"OLD {item}: {'DEFECT' if defect else 'NO DEFECT'} {text}")


config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=150, evaluate_every_steps=50)
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 400)
student = SNNStudent.train_in_dream(spec, dreams, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED)
situations = spec.sample_situations(stream_rng(BASE_SEED, "fixture_snn/check/situations"), 50)

# S1: a spec with other side-speed bounds, and a categorical spec with swapped levels.
try:
    narrow = student.decide(FixtureSnnSpec(side_bounds=(-0.3, 0.3)), situations)
    changed = int(np.sum(np.any(narrow != student.decide(spec, situations), axis=1)))
    report("S1", True, f"decide under side-speed bounds [-0.3, 0.3] returned actions with no error ({changed} of 50 commands changed)")
except ValueError as error:
    report("S1", False, f"refused: {error}")
side = FixtureSideSpec()
side_student = SNNStudent.train_in_dream(side, fixture_dreams(side, config, 400, points_per_dim=(1, 5)), settings=settings, points_per_dim=(1, 5), base_seed=BASE_SEED)
try:
    side_student.decide(FixtureSideSpec(levels=("right", "stay", "left"), command_values=(-0.5, 0.0, 0.5)), situations)
    report("S1b", True, "decide with the categorical side levels swapped returned actions with no error")
except ValueError as error:
    report("S1b", False, f"refused: {error}")

# S2: a hidden layer that never spikes.
try:
    silent = SNNStudent.train_in_dream(
        spec, dreams, settings=dataclasses.replace(settings, threshold=1000.0, max_steps=300), points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=9
    )
    distinct = len(set(silent.decide_index(situations).tolist()))
    report("S2", True, f"a student with silent layers {silent.meta['training']['silent_layers']} was returned; {distinct} distinct decision(s) on 50 situations")
except snn_train.TrainingError as error:
    report("S2", False, f"refused: {error}")

# S3: readout_beta mis-exported, checked on 1 row.
encoder = InputEncoder.from_variables(spec.situation)
inputs = encoder.encode(dreams.situations)
model = snn_train.build_model(encoder.width, {"h": dreams.n_candidates}, settings, seed=21)
exported = snn_train.export_network(model, "h")
try:
    parity = snn_train.check_parity(model, "h", dataclasses.replace(exported, readout_beta=0.5), inputs[:1])
    report("S3", True, f"check_parity accepted a mis-exported readout_beta on 1 row (max |logit difference| {parity['max_abs_logit_difference']:.3f})")
except snn_train.ParityError as error:
    report("S3", False, f"refused: {error}")
permutation = np.random.default_rng(0).permutation(settings.hidden_sizes[0])
permuted = dataclasses.replace(
    exported, hidden_weights=(exported.hidden_weights[0][permutation],), hidden_biases=(exported.hidden_biases[0][permutation],),
    readout_weight=exported.readout_weight[:, permutation],
)
try:
    snn_train.check_parity(model, "h", permuted, inputs)
    report("S3b", True, "check_parity accepted a network with permuted hidden neurons (equal spike counts, different spike trains)")
except snn_train.ParityError as error:
    report("S3b", False, f"refused: {error}")
report("S3c", "n_inputs_checked" in student.meta["parity"] and student.meta["parity"]["n_inputs_checked"] < dreams.n_rows,
       f"train_in_dream checked parity on {student.meta['parity']['n_inputs_checked']} of {dreams.n_rows} rows")
report("S4", not hasattr(SNNStudent, "decide_index_one") and "decide_one_vs_batch_differing" not in student.meta["parity"],
       "SNNStudent has no per-decision path and parity never ran decide_one")

# S5: only one global scale.
rng = np.random.default_rng(1)
scores = -0.05 * np.stack([rng.permutation(35) / 34.0 for _ in range(200)])
scores[:110] -= 10.0 * (rng.random((110, 35)) < 0.4)
targets = soft_targets(scores, temperature=0.5, spread=score_spread(scores))
report("S5", float(targets[110:].max(axis=1).mean()) < 1.1 / 35, f"small-scale rows get mean max target {targets[110:].max(axis=1).mean():.4f} (uniform {1 / 35:.4f}); no per-row scale exists")

# S6: Adam settings.
report("S6", not hasattr(settings, "adam_beta1"), f"SNNSettings has no Adam fields; the optimizer ran with {student.meta['training']['optimizer']} (torch defaults)")

# S7: out-of-range situations.
outside = situations.copy()
outside[3, 1] = 9.0
try:
    student.decide_index(outside)
    report("S7", True, "decide_index on fx_speed = 9.0 (declared [2, 6]) returned a decision with no error or count")
except ValueError as error:
    report("S7", False, f"refused: {error}")

# M2/M4: rows split by row, no per-situation record.
doubled = dataclasses.replace(dreams, **{
    item.name: np.concatenate([getattr(dreams, item.name)] * 2) for item in dataclasses.fields(dreams)
    if isinstance(getattr(dreams, item.name), np.ndarray) and getattr(dreams, item.name).shape[:1] == (dreams.n_rows,)
})
order = stream_rng(BASE_SEED, "fixture_snn/snn/validation_split").permutation(doubled.n_rows)
n_validation = max(1, int(round(settings.validation_fraction * doubled.n_rows)))
held, kept = order[:n_validation], order[n_validation:]
keys = [row.tobytes() for row in doubled.situations]
overlap = len({keys[i] for i in held} & {keys[i] for i in kept})
report("M2", overlap > 0, f"the row split of a dream with every situation twice puts {overlap} of {n_validation} early-stopping situations also among the training rows")
report("M4", "situations_seen" not in student.meta.get("student", {}), f"meta['student'] keys {sorted(student.meta['student'])}: no per-situation digests")

# M1 and labels.
report("M1", not hasattr(snn, "DeployedStudent"), "no DeployedStudent: deciding exists only on SNNStudent with its own InputEncoder")
report("labels", "validation_in_dream" in student.meta, "in-dream numbers stored as 'validation_in_dream' (early-stopping rows, not labelled as such)")
