"""Review r07: config.yaml keys that look like teacher switches but are never read, and persistence.*
validated only at the first cache write (after the billable call).

FixtureProvider (see _fixture.py) replaces only the network. Temporary directory only; config.yaml
is read, never written (a modified copy lives in memory).

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_review\\teacher\\r07_settings.py
"""

print("REVIEW REPRODUCTION - fixture, not evidence")

import copy
import tempfile
from pathlib import Path

import _fixture as fx
import learned_judgment.fileio as fileio
from learned_judgment.config import load_config
from learned_judgment.teacher import TeacherSettings

config = load_config()
print("== 1. tabpfn switches in config.yaml ==")
flipped = copy.deepcopy(config)
for name in ("check_usage_before_uncached_request", "estimate_cost_before_uncached_request", "cache_requests_and_responses"):
    print(f"  config.yaml tabpfn.{name} = {config['tabpfn'][name]!r} -> set to False")
    flipped["tabpfn"][name] = False
flipped["tabpfn"]["cache_key_fields"] = []
print("  TeacherSettings identical after flipping them:", TeacherSettings.from_config(flipped) == TeacherSettings.from_config(config))
root = Path(fx.ROOT) / "learned_judgment"
readers = [p.relative_to(fx.ROOT).as_posix() for p in root.rglob("*.py")
           if any(k in p.read_text(encoding="utf-8") for k in ("check_usage_before", "estimate_cost_before",
                                                                "cache_requests_and_responses", "cache_key_fields",
                                                                "verified_request_limit"))]
print("  modules under learned_judgment/ that mention any of them or verified_request_limit_*:", readers)

print("\n== 2. persistence.* checked only inside cache.put, after the billable call ==")
broken = copy.deepcopy(config)
broken["persistence"]["replace_attempts"] = None
fileio._replace_settings.cache_clear()
real_load = fileio.load_config
fileio.load_config = lambda *a, **k: broken
try:
    with tempfile.TemporaryDirectory() as tmp:
        teacher, provider, account, log, settings = fx.make_teacher(Path(tmp))
        features, residuals, query = fx.arrays()
        teacher.fit(features, residuals)
        try:
            teacher.predict_residual_distribution(query)
        except Exception as exc:
            print("  raised:", type(exc).__name__, exc)
        print("  billable predicts made before the error:", account.billed_predicts,
              "| events:", [r["event"] for r in log.read()])
        print("  cache files:", sorted(p.name for p in settings.cache_dir.glob("*")) if settings.cache_dir.exists() else "(none)")
finally:
    fileio.load_config = real_load
    fileio._replace_settings.cache_clear()
print("  note: fileio reads persistence.* from the global config.yaml, never from the mapping the teacher was built from")
