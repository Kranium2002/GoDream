"""Review r06: request_key field by field against what the pinned client sends, and parse_main_output
when the number of query rows equals the number of quantile levels.

Pure functions plus the FixtureProvider teacher (see _fixture.py); no network.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\teacher\\r06_key_and_response.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import dataclasses
import tempfile
from pathlib import Path

import numpy as np

import _fixture as fx
from learned_judgment.config import load_config
from learned_judgment.teacher import TeacherSettings, parse_main_output, request_key

settings = TeacherSettings.from_config(load_config())
rng = np.random.default_rng(0)
xt, yt, xq = rng.normal(size=(6, 3)), rng.normal(size=6), rng.normal(size=(4, 3))
names = ("a", "b", "c")


def key(**overrides):
    args = dict(settings=settings, x_train=xt, y_train=yt, x_query=xq, categorical_indices=(1,), column_names=names)
    args.update(overrides)
    return request_key(**args)


k0 = key()
print("== request_key ==")
print("  -0.0 vs +0.0 in x_query -> same key:", key(x_query=np.where(xq == xq, xq, 0) * 1.0) == k0,
      key(x_query=np.vstack([xq[:3], -np.abs(xq[3:]) * 0.0])) == key(x_query=np.vstack([xq[:3], np.abs(xq[3:]) * 0.0])))
print("  query row order changes key:", key(x_query=xq[::-1].copy()) != k0)
print("  categorical indices change key:", key(categorical_indices=(0,)) != k0)
print("  quantile list changes key:", key(settings=dataclasses.replace(settings, quantile_levels=(0.1, 0.5, 0.9), spread_levels=(0.1, 0.9))) != k0)
print("  random_state changes key:", key(settings=dataclasses.replace(settings, random_state=1)) != k0)
print("  spread_levels / budget / paths do NOT change key:",
      key(settings=dataclasses.replace(settings, spread_levels=(0.25, 0.75), project_token_budget=1, cache_dir=Path("x"))) == k0)
try:
    key(x_train=xt.astype(np.float32))
except TypeError as exc:
    print("  float32 x_train rejected:", exc)

# The provider uploads numpy arrays, so the server sees columns named '0'..'F-1' whatever the layout says.
renamed = key(column_names=("sit_x", "sit_side", "act_t"))
print("  renaming layout columns (bytes sent unchanged) changes key:", renamed != k0)

print("\n== same request after a feature rename: a second billable request for byte-identical data ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, log, s = fx.make_teacher(Path(tmp))
    features, residuals, query = fx.arrays()
    teacher.fit(features, residuals)
    teacher.predict_residual_distribution(query)
    renamed_layout = dataclasses.replace(fx.LAYOUT, names=("sit_x", "sit_side", "act_time", "physics_r0", "physics_r1", "physics_r2"))
    teacher2 = fx.TabPFNTeacher(renamed_layout, s, provider=provider, cache=teacher.cache, call_log=log, clock=fx.make_clock())
    teacher2.fit(features, residuals)
    teacher2.predict_residual_distribution(query)
    print("  billable predicts for identical arrays:", account.billed_predicts, "| events:", [r["event"] for r in log.read()])

print("\n== parse_main_output when n_rows == len(quantile_levels) ==")
levels = settings.quantile_levels
n = len(levels)
# Candidates on an ascending action grid: predictions rise with the candidate index.
truth = np.array([[i + 0.2 * j for j in range(n)] for i in range(n)], dtype=float)  # (n candidates, Q levels)
median = truth[:, levels.index(0.5)]
raw_wrong_layout = {"mean": median, "median": median, "mode": median, "quantiles": truth}  # (n, Q), not (Q, n)
parsed = parse_main_output(raw_wrong_layout, n_rows=n, quantile_levels=levels)
print("  accepted without error; parsed quantiles row 0:", parsed["quantiles"][0].tolist(), "true row 0:", truth[0].tolist())
print("  parsed q0.5 column vs returned median:", parsed["quantiles"][:, levels.index(0.5)].tolist(), "vs", median.tolist())
raw_wrong_layout_n4 = {"mean": median[:4], "median": median[:4], "mode": median[:4], "quantiles": truth[:4]}
try:
    parse_main_output(raw_wrong_layout_n4, n_rows=4, quantile_levels=levels)
except Exception as exc:
    print("  same wrong layout with n=4 is rejected:", type(exc).__name__, exc)
