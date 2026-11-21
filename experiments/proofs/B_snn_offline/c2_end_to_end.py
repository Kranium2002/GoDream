"""Checks 2 and 4: train a spiking student inside a fixture dream, parity, deciding, save/load.

FIXTURE CHECK - mechanics only. The dream is built by the real engine (physics only) from the
fixture spec in fixture_snn_spec.py. Settings are config.yaml's students.snn values except
max_steps, which is reduced explicitly (printed below) to keep the check short. Agreement and
regret printed here are MECHANICS numbers on a fixture, measured inside the fixture dream; they
say nothing about the robot, the simulator, TabPFN or a project student.

Check 2: training runs; the restored validation loss is below its initial value; the numpy
network reproduces the torch model (zero spike-count mismatches, identical decisions); decisions
are valid candidate indices; decide() returns in-bounds grid actions. 2b repeats training and
parity with a threshold float32 cannot represent and learned beta/threshold (fast_sigmoid).
Check 4: save/load round trip in-process (same digest and decisions) and in a fresh process
(c4_load_worker.py), which must not import torch. Files go to a temporary directory only.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c2_end_to_end.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any trained student")

import dataclasses  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, REPO_ROOT, Checks, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402
from learned_judgment.students.targets import dream_regret  # noqa: E402

MAX_STEPS = 400  # reduced from config students.snn.max_steps for a short mechanics check
MAX_STEPS_2B = 150
BASE_SEED = 7  # fixture run seed
CHECK_SITUATIONS = 500

checks = Checks()
config = load_config()
config_settings = SNNSettings.from_config(config)
settings = dataclasses.replace(config_settings, max_steps=MAX_STEPS)
print(f"\nsettings: config.yaml students.snn with max_steps={MAX_STEPS} (config value {config_settings.max_steps})")
print(f"  {settings.describe()}")

spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config)
print(f"fixture dream: {dreams.n_rows} rows x {dreams.n_candidates} candidates, grid {FIXTURE_POINTS_PER_DIM}")

print("\n== check 2: train inside the fixture dream ==")
started = time.perf_counter()
student = SNNStudent.train_in_dream(
    spec, dreams, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED
)
print(f"  train_in_dream wall time incl. torch import: {time.perf_counter() - started:.1f} s")
training = student.meta["training"]
print(
    f"  steps_run={training['steps_run']} best_step={training['best_step']} stopped_early={training['stopped_early']} "
    f"wall_seconds={training['wall_seconds']:.1f} threads={training['torch_threads']} versions={training['versions']}"
)
print(f"  validation total curve: {[(s, round(v, 4)) for s, v in training['validation_total_curve']]}")
print(f"  train loss curve: {[(s, round(v, 4)) for s, v in training['train_loss_curve']]}")
print(
    f"  hidden firing rate per layer: initial {[round(v, 4) for v in training['initial_firing_rate_per_layer']]} "
    f"final {[round(v, 4) for v in training['firing_rate_per_layer']]}; dead {training['dead_fraction_per_layer']} "
    f"saturated {training['saturated_fraction_per_layer']} silent layers {training['silent_layers']}"
)
checks.expect(training["steps_run"] >= 1, "training ran")
checks.expect(
    training["best_validation_loss"] < training["initial_validation_loss"],
    f"restored validation loss {training['best_validation_loss']:.4f} < initial {training['initial_validation_loss']:.4f}",
)
parity = student.meta["parity"]
print(f"  parity report: {parity}")
checks.expect(all(m == 0 for m in parity["spike_count_mismatches_per_layer"]), "numpy vs torch: zero spike-count mismatches per layer")
checks.expect(parity["decisions_differing"] == 0, "numpy vs torch: identical decisions on the validation rows")
checks.expect(
    parity["max_abs_logit_difference"] < 1e-12,
    f"max |logit difference| {parity['max_abs_logit_difference']:.2e} is at float64 rounding level (< 1e-12)",
)

validation_numbers = student.meta["validation_in_dream"]
print("  MECHANICS numbers on the fixture dream's validation rows (inside the dream, not the simulator):")
print(f"    {validation_numbers}")

# Independent recomputation of the validation numbers through the public decide path.
split = stream_rng(BASE_SEED, f"{spec.name}/snn/validation_split").permutation(dreams.n_rows)
validation_rows = np.sort(split[: student.meta["dream"]["n_rows_validation"]])
indices = student.decide_index(dreams.situations[validation_rows])
recomputed_agreement = float(np.mean(indices == dreams.chosen[validation_rows]))
recomputed_regret = float(np.mean(dream_regret(dreams.scores[validation_rows], indices)))
checks.expect(
    recomputed_agreement == validation_numbers["agreement_with_dream_choice"]
    and recomputed_regret == validation_numbers["mean_dream_regret"],
    "decide_index on the validation rows reproduces the stored agreement and regret",
)

situations = spec.sample_situations(stream_rng(BASE_SEED, f"{spec.name}/check/situations"), CHECK_SITUATIONS)
indices = student.decide_index(situations)
checks.expect(indices.dtype == np.int64 and indices.shape == (CHECK_SITUATIONS,), f"decide_index -> int64 {indices.shape}")
checks.expect(bool(np.all((indices >= 0) & (indices < student.n_candidates))), f"every index lies in [0, {student.n_candidates})")
actions = student.decide(spec, situations)
grid_actions = np.stack([spec.candidate_actions(s, FIXTURE_POINTS_PER_DIM)[i] for s, i in zip(situations, indices)])
checks.expect(actions.shape == (CHECK_SITUATIONS, spec.n_action), f"decide -> {actions.shape}")
checks.expect(bool(np.all(spec.action_in_bounds(actions))), "every decided action is inside the declared action bounds")
checks.expect(bool(np.array_equal(actions, grid_actions)), "decide = candidate_actions(situation, points_per_dim)[decide_index]")

fresh = decide_batch(
    spec,
    situations,
    model=None,
    layout=build_layout(spec),
    points_per_dim=FIXTURE_POINTS_PER_DIM,
    quantile_levels=config["engine"]["quantile_levels"],
    spread_levels=tuple(config["engine"]["spread_levels"]),
    method=PHYSICS_ONLY,
)
fresh_scores = np.stack([decision.scores for decision in fresh])
fresh_chosen = np.array([decision.chosen_index for decision in fresh])
fresh_regret = dream_regret(fresh_scores, indices)
spread = student.meta["dream"]["score_spread"]
print(
    f"  MECHANICS numbers on {CHECK_SITUATIONS} new fixture situations scored by the same fixture objective: "
    f"agreement {np.mean(indices == fresh_chosen):.3f}, zero-regret share {np.mean(fresh_regret == 0.0):.3f}, "
    f"mean regret {fresh_regret.mean():.4f} ({fresh_regret.mean() / spread:.3f} spreads)"
)
counts = student.operation_counts(situations)
print(
    f"  operation counts (fixture): layer sizes {counts.layer_sizes}, T={counts.time_steps}, mean spikes per layer "
    f"{[round(float(v), 2) for v in counts.mean_spikes_per_layer]}, firing rate {[round(float(v), 4) for v in counts.firing_rate_per_layer]}"
)
checks.expect(counts.n_decisions == CHECK_SITUATIONS, "operation_counts covers every decision")

print("\n== check 2b: threshold 0.3 (not exact in float32), learned beta and threshold, fast_sigmoid ==")
settings_2b = dataclasses.replace(
    settings, threshold=0.3, learn_beta=True, learn_threshold=True, surrogate="fast_sigmoid", max_steps=MAX_STEPS_2B
)
print(f"  settings: as above with threshold=0.3 learn_beta=True learn_threshold=True surrogate=fast_sigmoid max_steps={MAX_STEPS_2B}")
student_2b = SNNStudent.train_in_dream(
    spec, dreams, settings=settings_2b, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=BASE_SEED
)
parity_2b = student_2b.meta["parity"]
print(f"  learned (exported) betas {student_2b.network.betas} thresholds {student_2b.network.thresholds}")
print(f"  parity report: {parity_2b}")
checks.expect(student_2b.network.thresholds[0] != 0.3, "the threshold was learned (moved away from 0.3)")
checks.expect(
    all(m == 0 for m in parity_2b["spike_count_mismatches_per_layer"]) and parity_2b["decisions_differing"] == 0,
    "parity holds with a learned float64 threshold",
)
# Logits are not bitwise equal: the readout matmul sums in a different order in torch (MKL)
# and numpy (OpenBLAS); the first-layer current was bitwise equal (p01_matmul_rounding_probe). The
# contract is identical spikes and decisions; the logit gap must stay at float64 rounding level.
checks.expect(
    parity_2b["max_abs_logit_difference"] < 1e-12,
    f"max |logit difference| {parity_2b['max_abs_logit_difference']:.2e} is at float64 rounding level (< 1e-12)",
)

print("\n== check 4: save / load ==")
scratch = Path(tempfile.mkdtemp(prefix="b_snn_offline_c2_"))
try:
    stem = scratch / "fixture_student"
    digest = student.save(stem)
    checks.expect(digest == student.digest, "save returns the network digest")
    checks.expect(sorted(p.name for p in scratch.iterdir()) == ["fixture_student.json", "fixture_student.npz"], "files: <stem>.json + <stem>.npz")
    loaded = SNNStudent.load(stem)
    checks.expect(loaded.digest == student.digest, "loaded digest = trained digest")
    checks.expect(bool(np.array_equal(loaded.decide_index(situations), indices)), "loaded decide_index = trained decide_index")
    checks.expect(bool(np.array_equal(loaded.decide(spec, situations), actions)), "loaded decide = trained decide")
    checks.expect(
        loaded.points_per_dim == student.points_per_dim and loaded.n_candidates == student.n_candidates
        and loaded.encoder.describe() == student.encoder.describe(),
        "grid and encoder survive the round trip",
    )
    checks.expect(loaded.meta["validation_in_dream"] == validation_numbers, "meta (validation numbers) survives the round trip")
    np.save(scratch / "situations.npy", situations)
    np.save(scratch / "expected.npy", indices)
    worker = Path(__file__).resolve().parent / "c4_load_worker.py"
    command = [sys.executable, str(worker), str(stem), str(scratch / "situations.npy"), str(scratch / "expected.npy"), digest]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    print("  fresh-process worker output:")
    for line in (result.stdout + result.stderr).strip().splitlines():
        print(f"    {line.replace(str(scratch), '<tmp>')}")
    checks.expect(result.returncode == 0, f"fresh process: worker exit code {result.returncode}")
    checks.expect("decisions_identical=True" in result.stdout, "fresh process: identical decisions")
    checks.expect("torch_in_sys_modules=False" in result.stdout, "fresh process: loading and deciding did not import torch")
finally:
    shutil.rmtree(scratch, ignore_errors=True)
checks.expect(not scratch.exists(), "temporary directory removed")
checks.finish()
