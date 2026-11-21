"""R02: can a spec bound to an unverified command field (or beyond the verified range) get
through decide_batch, exploration_actions and collect before the world is stepped?
No world is built: the spec's episode() records the command it was asked to run and raises,
so no measurement is produced. world=None is passed to collect.
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import LEVELS, REVIEW_SCHEMA, SPREAD, ReviewSpec  # noqa: E402

from learned_judgment.engine.collect import collect, exploration_actions  # noqa: E402
from learned_judgment.engine.decide import PHYSICS_ONLY, decide_batch  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.streams import stream_rng  # noqa: E402
from learned_judgment.skills.base import ActionDim, SkillSpecError, validate_spec  # noqa: E402


class Reached(Exception):
    pass


class UnverifiedSpec(ReviewSpec):
    """Bound to 'rv_vx_UNVERIFIED' (not in the schema) with range [-5, 5] (schema has [-1, 1])."""

    def __init__(self):
        super().__init__(command_field="rv_vx_UNVERIFIED", vy_range=(-5.0, 5.0))
        self.sent = []

    def episode(self, world, situation, action):
        self.sent.append(dict(action))
        raise Reached(f"spec.episode reached with action {dict(action)} (world would be stepped now)")


spec = UnverifiedSpec()
try:
    validate_spec(spec, REVIEW_SCHEMA, hidden_value_names=(), quantile_levels=LEVELS, probe_seed=0,
                  probe_situations=4, probe_points_per_dim=3)
    print("A. validate_spec: accepted (unexpected)")
except SkillSpecError as error:
    print("A. validate_spec rejects it:", str(error)[:150])

layout = build_layout(spec)
situations = spec.sample_situations(stream_rng(1, "rv/ctx"), 6)
decisions = decide_batch(spec, situations, model=None, layout=layout, points_per_dim=[3, 3],
                         quantile_levels=LEVELS, spread_levels=SPREAD, method=PHYSICS_ONLY)
print("B. decide_batch accepted it; candidate rv_side values:", sorted(set(decisions[0].candidates[:, 0].tolist())))
actions = exploration_actions(spec, situations, rng=stream_rng(1, "rv/act"), points_per_dim=[3, 3], exploration_fraction=1.0,
                              quantile_levels=LEVELS, spread_levels=SPREAD, layout=layout)
print("C. exploration_actions accepted it; rv_side values:", actions[:, 0].tolist())
with tempfile.TemporaryDirectory() as tmp:
    try:
        collect(spec, None, situations, actions, run_id="rv", seed=0, world_preset="default",
                out_path=Path(tmp) / "a.jsonl", clock=lambda: "2026-11-21T12:00:00+00:00")
    except Reached as reached:
        print("D. collect:", reached)

# Categorical command dimension: validate_spec skips the verified-range and unit checks.
class CategoricalCommandSpec(ReviewSpec):
    def __init__(self):
        super().__init__()
        self.actions = (
            ActionDim("rv_side", "categorical", "m/s", "side as a level -> value mapped in episode()",
                      levels=("-3.0", "3.0"), role="command", command_field="rv_vy"),
            self.actions[1],
        )

    def physics_guess(self, situations, actions):
        a = np.asarray(actions, dtype=np.float64).copy()
        a[:, 0] = np.where(a[:, 0] == 0.0, -3.0, 3.0)
        return super().physics_guess(situations, a)


try:
    validate_spec(CategoricalCommandSpec(), REVIEW_SCHEMA, hidden_value_names=(), quantile_levels=LEVELS, probe_seed=0,
                  probe_situations=4, probe_points_per_dim=3)
    print("E. validate_spec ACCEPTS a categorical command dim whose levels mean +-3.0 m/s on a field verified for [-1, 1]")
except SkillSpecError as error:
    print("E. validate_spec rejects:", error)
