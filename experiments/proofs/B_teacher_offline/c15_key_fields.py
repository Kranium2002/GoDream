"""Check 15 (review T8): the request key no longer includes the layout column names (never sent)
and now includes the provider base URL; the key schema is v2.

Old behaviour (before the fix): renaming a feature with byte-identical data changed the key and
the same answers were bought again; a run against another deployment URL would have been served
entries from api.priorlabs.ai (review r06).

FixtureProvider (see _fixture_provider.py); OtherDeployment below is a check-only subclass that
reports another base URL. Temporary directories only. Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c15_key_fields.py
"""

import dataclasses
import inspect
import sys
import tempfile
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _fixture_provider as fx  # noqa: E402
from learned_judgment.teacher import TabPFNTeacher, _request_description, request_key  # noqa: E402

check = fx.Checks()
OTHER_URL = "https://other-deployment.invalid:443"


class OtherDeployment(fx.FixtureProvider):
    """FIXTURE: the same provider pointed at another deployment URL (offline)."""

    def base_url(self) -> str:
        return OTHER_URL


with tempfile.TemporaryDirectory() as tmp:
    settings = fx.settings_for(Path(tmp))
    rng = np.random.default_rng(0)
    xt, yt, xq = rng.normal(size=(6, 3)), rng.normal(size=6), rng.normal(size=(4, 3))
    real_url = fx.FixtureProvider(settings, fx.Account()).base_url()
    print(f"== request_key (installed client base URL: {real_url}) ==")
    parameters = list(inspect.signature(request_key).parameters)
    print("  parameters:", parameters)
    check("request_key takes base_url and no column_names", "base_url" in parameters and "column_names" not in parameters)
    key = request_key(settings=settings, x_train=xt, y_train=yt, x_query=xq, categorical_indices=(1,), base_url=real_url)
    other = request_key(settings=settings, x_train=xt, y_train=yt, x_query=xq, categorical_indices=(1,), base_url=OTHER_URL)
    check("another base URL changes the key", key != other)
    description = _request_description(settings, (1,), real_url)
    check("the keyed description is schema v2, holds base_url and no column names",
          description["schema"].endswith("/v2") and description["base_url"] == real_url and "column_names" not in description)
    check.raises("an empty base URL", ValueError,
                 lambda: request_key(settings=settings, x_train=xt, y_train=yt, x_query=xq, categorical_indices=(1,), base_url=""))

print("\n== a renamed feature with identical bytes is a cache hit ==")
with tempfile.TemporaryDirectory() as tmp:
    teacher, provider, account, settings = fx.make_teacher(Path(tmp))
    features, residuals, query = fx.arrays()
    teacher.fit(features, residuals)
    first = teacher.predict_residual_distribution(query)
    entry = teacher.cache.get(teacher.last_request_hash)
    check("the entry keeps the layout column names in its metadata",
          entry.request_meta["column_names"] == list(fx.LAYOUT.names) + ["stacked_outcome_index"])
    renamed = dataclasses.replace(fx.LAYOUT, names=("sit_x", "sit_side", "act_time", "physics_r0", "physics_r1", "physics_r2"))
    teacher2 = TabPFNTeacher.from_settings(renamed, settings, provider=provider, clock=fx.make_clock())
    teacher2.fit(features, residuals)
    second = teacher2.predict_residual_distribution(query)
    check("same key, served from the cache: 1 billable call in total",
          teacher2.last_request_hash == teacher.last_request_hash and account.billed_predicts == 1
          and fx.events(teacher2)[-1] == "cache_hit")
    check("identical answers", np.array_equal(first.quantiles, second.quantiles))

    print("\n== another base URL is never served this cache entry ==")
    elsewhere = OtherDeployment(settings, account)
    teacher3 = TabPFNTeacher.from_settings(fx.LAYOUT, settings, provider=elsewhere, clock=fx.make_clock())
    teacher3.fit(features, residuals)
    teacher3.predict_residual_distribution(query)
    check("a new key and a new dispatch (2 billable calls)",
          teacher3.last_request_hash != teacher.last_request_hash and account.billed_predicts == 2)
    check("its entry records the other base URL", teacher3.cache.get(teacher3.last_request_hash).request_meta["base_url"] == OTHER_URL)

check.finish()
