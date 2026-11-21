"""R09: (a) collect() validates run metadata only after the first episode has been started;
(b) keep_confident + save/load + dream_rows keep every per-row array and request hash aligned
when decisions come from two different requests and spreads vary per row."""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import LEVELS, SPREAD, TS, ReviewSpec  # noqa: E402

from learned_judgment.engine.collect import collect  # noqa: E402
from learned_judgment.engine.decide import decide_batch  # noqa: E402
from learned_judgment.engine.dream import dream_rows, dreams_from_decisions, keep_confident, load_dreams, save_dreams  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.rows import AttemptRow  # noqa: E402
from learned_judgment.engine.types import ResidualDistribution  # noqa: E402


class Reached(Exception):
    pass


class RecordingSpec(ReviewSpec):
    def episode(self, world, situation, action):
        raise Reached("spec.episode reached (the world would be stepped)")


spec = RecordingSpec()
sit = np.array([[1.0, 0.0], [0.5, 1.0]])
act = np.array([[0.5, 0.1], [0.0, 0.2]])
with tempfile.TemporaryDirectory() as tmp:
    for label, kwargs in (
        ("run_id=''", dict(run_id="", seed=0, world_preset="default", clock=lambda: TS)),
        ("seed=-1", dict(run_id="r", seed=-1, world_preset="default", clock=lambda: TS)),
        ("world_preset=''", dict(run_id="r", seed=0, world_preset="", clock=lambda: TS)),
        ("clock returns a naive time", dict(run_id="r", seed=0, world_preset="default", clock=lambda: "2026-11-21T12:00:00")),
    ):
        try:
            collect(spec, None, sit, act, out_path=Path(tmp) / "a.jsonl", **kwargs)
        except Reached as error:
            print(f"a. collect with {label:28s}: {error}")
        except ValueError as error:
            print(f"a. collect with {label:28s}: rejected before the episode: {error}")
    try:
        AttemptRow(run_id="", seed=0, skill="x", world_preset="d", source="real", situation={"a": 1.0}, action={"b": 1.0},
                   physics_guess={"c": 1.0}, measured_outcome={"c": 1.0}, residual={"c": 0.0}, timestamp_utc=TS)
    except ValueError as error:
        print("   ... and the row built after that episode is rejected:", error)


layout = build_layout(ReviewSpec())


class VaryingModel:
    """FIXTURE residual model: spread varies with the first feature; hash names the call."""

    def __init__(self, tag):
        self.last_request_hash = None
        self.tag = tag

    def fit(self, f, r):
        return self

    def predict_residuals(self, q):
        d = self.predict_residual_distribution(q)
        return d.median, d.spread

    def predict_residual_distribution(self, q):
        n = q.shape[0]
        width = 0.05 + 0.3 * q[:, 0] / 2.0
        base = np.array([-1.28, -0.67, 0.0, 0.67, 1.28])
        quant = np.zeros((n, 2, 5))
        quant[:, 0, :] = width[:, None] * base[None, :]
        self.last_request_hash = f"fixture-{self.tag}"
        return ResidualDistribution.from_quantiles(np.zeros((n, 2)), np.zeros((n, 2)), np.array(LEVELS), quant, SPREAD)


plain = ReviewSpec()
rng = np.random.default_rng(5)
s1, s2 = plain.sample_situations(rng, 7), plain.sample_situations(rng, 6)
kw = dict(layout=layout, points_per_dim=[3, 3], quantile_levels=LEVELS, spread_levels=SPREAD, method="teacher")
decisions = decide_batch(plain, s1, model=VaryingModel("A"), **kw) + decide_batch(plain, s2, model=VaryingModel("B"), **kw)
ds = dreams_from_decisions(plain, decisions, spread_levels=SPREAD)
thr = float(np.median(ds.chosen_spread))
kept = keep_confident(ds, thr)
mask = ds.chosen_spread <= thr
with tempfile.TemporaryDirectory() as tmp:
    save_dreams(Path(tmp) / "d.npz", kept, {"threshold": thr})
    back, _ = load_dreams(Path(tmp) / "d.npz")
rows = dream_rows(plain, back, run_id="r", seed=0, world_preset="default", timestamp_utc=TS)
expect_hash = [d.request_hash for d, m in zip(decisions, mask) if m]
print("b. kept", back.n_rows, "of", ds.n_rows, "| situations aligned:", np.array_equal(back.situations, ds.situations[mask]),
      "| chosen aligned:", np.array_equal(back.chosen_actions, ds.chosen_actions[mask]),
      "| hashes per row aligned:", [r.teacher["request_hash"] for r in rows] == expect_hash,
      "| request_hashes kept:", back.request_hashes)
