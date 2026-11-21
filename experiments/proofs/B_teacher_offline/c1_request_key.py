"""Check 1: request_key is process-independent and sensitive to exactly the keyed inputs.

FIXTURE arrays and a fixture base URL only; exercises the pure function
learned_judgment.teacher.request_key (key schema v2: base URL keyed, column names not keyed --
review T8). No network.
Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c1_request_key.py
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.teacher import TeacherSettings, request_key  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAILED'}] {label}")
    if not ok:
        FAILURES.append(label)


def reversed_deep(value):
    if isinstance(value, dict):
        return {key: reversed_deep(value[key]) for key in reversed(list(value))}
    return value


CONFIG = load_config()
SETTINGS = TeacherSettings.from_config(CONFIG)

rng = np.random.default_rng(0)
X_TRAIN = rng.normal(size=(6, 4))
X_TRAIN[:, 1] = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]  # categorical column (level indices)
X_TRAIN[0, 0] = 0.0
Y_TRAIN = rng.normal(size=6)
X_QUERY = rng.normal(size=(5, 4))
X_QUERY[:, 1] = [1.0, 0.0, 1.0, 0.0, 1.0]
CATS = (1,)
URL = "https://api.priorlabs.ai:443"  # fixture string; the teacher passes TabPFNProvider.base_url()


def key(settings=SETTINGS, x_train=X_TRAIN, y_train=Y_TRAIN, x_query=X_QUERY, cats=CATS, url=URL) -> str:
    return request_key(
        settings=settings, x_train=x_train, y_train=y_train, x_query=x_query, categorical_indices=cats, base_url=url
    )


if "--emit-key" in sys.argv:
    print("KEY", key())
    sys.exit(0)

BASE = key()
print("base key:", BASE)

print("\n== identical across processes with different PYTHONHASHSEED ==")
child_keys = []
for seed in ("1", "2"):
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--emit-key"], capture_output=True, text=True, env=env, cwd=ROOT
    )
    line = [l for l in out.stdout.splitlines() if l.startswith("KEY ")]
    child_keys.append(line[0].split()[1] if line else f"<no key; exit {out.returncode}; {out.stderr.strip()[-200:]}>")
    print(f"  PYTHONHASHSEED={seed}: {child_keys[-1]}")
check("child processes agree with each other and with this process", child_keys[0] == child_keys[1] == BASE)


def bumped(array, index):
    copy = array.copy()
    copy[index] = np.nextafter(copy[index], np.inf)
    return copy


print("\n== must CHANGE the key ==")
swap = [2, 1, 0, 3]
changes = {
    "x_train one cell +1 ulp": key(x_train=bumped(X_TRAIN, (3, 2))),
    "y_train one cell +1 ulp": key(y_train=bumped(Y_TRAIN, 4)),
    "x_query one cell +1 ulp": key(x_query=bumped(X_QUERY, (2, 3))),
    "setting n_estimators 8 -> 4": key(settings=dataclasses.replace(SETTINGS, n_estimators=4)),
    "setting random_state 0 -> 1": key(settings=dataclasses.replace(SETTINGS, random_state=1)),
    "setting model_path -> v3_default": key(settings=dataclasses.replace(SETTINGS, model_path="v3_default")),
    "setting client_version -> 0.10.0": key(settings=dataclasses.replace(SETTINGS, client_version="0.10.0")),
    "quantile list (extra level 0.05)": key(
        settings=dataclasses.replace(SETTINGS, quantile_levels=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9))
    ),
    "quantile list (0.25/0.75 dropped)": key(settings=dataclasses.replace(SETTINGS, quantile_levels=(0.1, 0.5, 0.9))),
    "categorical indices (1,) -> ()": key(cats=()),
    "categorical indices (1,) -> (1, 2)": key(cats=(1, 2)),
    "column order swapped (arrays)": key(
        x_train=np.ascontiguousarray(X_TRAIN[:, swap]),
        x_query=np.ascontiguousarray(X_QUERY[:, swap]),
        cats=(1,),
    ),
    "base URL of another deployment": key(url="https://other-deployment.invalid:443"),
    "one query row dropped": key(x_query=X_QUERY[:4].copy()),
}
for label, value in changes.items():
    check(f"{label:42s} {value[:16]}", value != BASE)

print("\n== must NOT change the key ==")
same = {
    "config dict insertion order reversed (deep)": key(settings=TeacherSettings.from_config(reversed_deep(CONFIG))),
    "cache_dir / call_log paths": key(
        settings=dataclasses.replace(SETTINGS, cache_dir=Path("elsewhere"), call_log=Path("other.jsonl"))
    ),
    "max_tokens_per_request / project budget": key(
        settings=dataclasses.replace(SETTINGS, max_tokens_per_request=1, project_token_budget=123)
    ),
    "multi_outcome_mode / spread_levels": key(
        settings=dataclasses.replace(SETTINGS, multi_outcome_mode="per_outcome", spread_levels=(0.25, 0.75))
    ),
    "-0.0 instead of +0.0 in x_train[0, 0]": key(x_train=np.where(np.arange(24).reshape(6, 4) == 0, -0.0, X_TRAIN)),
    "Fortran-order x_train copy": key(x_train=np.asfortranarray(X_TRAIN)),
    "big-endian x_query copy": key(x_query=X_QUERY.astype(">f8")),
    "verified request limits / lock / row cap": key(
        settings=dataclasses.replace(
            SETTINGS, verified_request_limit_per_minute=60, verified_request_limit_per_hour=600,
            lock_timeout_s=5.0, lock_poll_s=0.5, max_query_rows_per_request=100,
        )
    ),
}
for label, value in same.items():
    check(f"{label:42s} {value[:16]}", value == BASE)

print("\n== rejected inputs ==")
for label, kwargs in {
    "float32 x_train": {"x_train": X_TRAIN.astype(np.float32)},
    "unsorted categorical indices": {"cats": (2, 1)},
    "categorical index out of range": {"cats": (4,)},
    "empty base URL": {"url": ""},
    "y_train length mismatch": {"y_train": Y_TRAIN[:5].copy()},
}.items():
    try:
        key(**kwargs)
        check(f"{label} raises", False)
    except (TypeError, ValueError) as exc:
        check(f"{label} raises {type(exc).__name__}: {exc}", True)

print(f"\nRESULT: {len(FAILURES)} failed check(s)" + (f": {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
