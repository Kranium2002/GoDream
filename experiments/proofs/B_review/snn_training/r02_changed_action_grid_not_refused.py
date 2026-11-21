"""R02: a student trained on one candidate grid decides with a spec whose grid changed, without error.

Output k of the network is "candidate k of the grid the dream was made with". SNNStudent.decide
recomputes the grid from the spec it is given and only checks the spec name, the situation
encoding, the action NAMES and the grid SHAPE. This script:
  (a) trains a short FIXTURE student (real train_in_dream, few steps), saves and reloads it;
  (b) decides with a spec whose side-speed action bounds changed from [-0.6, 0.6] to [-0.3, 0.3]
      (same names, same shape) -> no error, every chosen side speed is silently halved;
  (c) builds a spec with a categorical action whose two levels are swapped
      ("left","right") -> ("right","left") and decides with a directly constructed student
      (numpy network, FIXTURE weights) -> no error, the decoded side is the opposite one;
  (d) shows that the stored meta holds no record of the action bounds or levels to compare with.
Temporary files only; removed at the end.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from _review_common import random_network  # noqa: E402
from fixture_snn_spec import FIXTURE_POINTS_PER_DIM, FixtureSnnSpec, fixture_dreams  # noqa: E402
from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.skills.base import ActionDim  # noqa: E402
from learned_judgment.students.encoding import InputEncoder  # noqa: E402
from learned_judgment.students.snn import SNNStudent  # noqa: E402
from learned_judgment.students.snn_train import SNNSettings  # noqa: E402

config = load_config()
settings = dataclasses.replace(SNNSettings.from_config(config), max_steps=100, evaluate_every_steps=50)
spec = FixtureSnnSpec()
dreams = fixture_dreams(spec, config, 400)
student = SNNStudent.train_in_dream(spec, dreams, settings=settings, points_per_dim=FIXTURE_POINTS_PER_DIM, base_seed=3)
scratch = Path(tempfile.mkdtemp(prefix="review_r02_"))
try:
    student.save(scratch / "student")
    loaded = SNNStudent.load(scratch / "student")
finally:
    shutil.rmtree(scratch, ignore_errors=True)

situations = spec.sample_situations(stream_rng(3, "review/r02"), 200)
original = loaded.decide(spec, situations)

narrow = FixtureSnnSpec()
side = narrow.actions[0]
narrow.actions = (dataclasses.replace(side, low=-0.3, high=0.3), narrow.actions[1])
try:
    changed = loaded.decide(narrow, situations)
    print(f"(b) decide with side-speed bounds [-0.3, 0.3] instead of [-0.6, 0.6]: NO ERROR")
    print(f"    rows whose chosen action differs from the trained grid's action: {int(np.sum(np.any(changed != original, axis=1)))} / 200")
    print(f"    distinct side speeds trained grid -> {sorted(set(original[:, 0].round(3)))}")
    print(f"    distinct side speeds changed spec -> {sorted(set(changed[:, 0].round(3)))}")
    nonzero = original[:, 0] != 0.0
    print(f"    changed / original side speed on rows with a side step: {sorted(set((changed[nonzero, 0] / original[nonzero, 0]).round(6)))}")
except ValueError as error:
    print(f"(b) refused: {error}")


class CategoricalSideSpec(FixtureSnnSpec):
    """FIXTURE: the fixture spec with a categorical dodge side instead of a float side speed."""

    def __init__(self, levels):
        super().__init__(name="fixture_cat")
        self.actions = (
            ActionDim("fx_side", "categorical", "label", "fixture dodge side", levels=tuple(levels),
                      role="command", command_field="fixture_side"),
            self.actions[1],
        )


trained_spec = CategoricalSideSpec(("left", "right"))
swapped_spec = CategoricalSideSpec(("right", "left"))
encoder = InputEncoder.from_variables(trained_spec.situation)
points = (2, 5)
n_candidates = trained_spec.candidate_actions(situations[0], points).shape[0]
network = random_network(np.random.default_rng(1), (encoder.width, 32, n_candidates), time_steps=10)
cat_student = SNNStudent(
    skill="fixture_cat", network=network, encoder=encoder, points_per_dim=points, n_candidates=n_candidates,
    meta={"student": {"action_names": list(trained_spec.action_names)}},
)
a_trained = cat_student.decide(trained_spec, situations)
try:
    a_swapped = cat_student.decide(swapped_spec, situations)
    sides_trained = [trained_spec.decode_action(a)["fx_side"] for a in a_trained[:6]]
    sides_swapped = [swapped_spec.decode_action(a)["fx_side"] for a in a_swapped[:6]]
    print(f"(c) decide with the categorical side levels swapped: NO ERROR")
    print(f"    first 6 decoded sides, trained levels: {sides_trained}")
    print(f"    first 6 decoded sides, swapped levels: {sides_swapped}")
    print(f"    rows whose decoded side flips: {sum(trained_spec.decode_action(a)['fx_side'] != swapped_spec.decode_action(b)['fx_side'] for a, b in zip(a_trained, a_swapped))} / 200")
except ValueError as error:
    print(f"(c) refused: {error}")

print(f"(d) meta['student'] keys stored with a trained student: {sorted(student.meta['student'])}")
print(f"    action bounds or levels recorded there: {'action_bounds' in student.meta['student'] or 'actions' in student.meta['student']}")
