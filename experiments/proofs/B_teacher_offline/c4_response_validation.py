"""Check 4: the 'main' response validator, and the tabpfn_config the pinned client would send.

FIXTURE arrays stand in for nothing: they only exercise the pure validator
learned_judgment.teacher.parse_main_output. The second part constructs the pinned
TabPFNRegressor offline (no network call; construction only) and compares the config it would
send with the keyed settings.
Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_teacher_offline\\c4_response_validation.py
"""

import sys
from pathlib import Path

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.teacher import ProviderResponseError, TabPFNProvider, TeacherSettings, parse_main_output  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAILED'}] {label}")
    if not ok:
        FAILURES.append(label)


check("importing learned_judgment.teacher did not import tabpfn_client", "tabpfn_client" not in sys.modules)

LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
N = 7
rng = np.random.default_rng(4)
q_rows = np.sort(rng.normal(size=(N, len(LEVELS))), axis=1)  # (n, Q), non-decreasing per row


def raw(**override):
    base = {
        "mean": rng.normal(size=N),
        "median": q_rows[:, 2].copy(),
        "mode": rng.normal(size=N),
        "quantiles": q_rows.T.copy(),  # the client's documented (Q, n) layout for "main"
    }
    base.update(override)
    return base


print("== accepted: documented (Q, n) layout ==")
good = raw()
parsed = parse_main_output(good, n_rows=N, quantile_levels=LEVELS)
check("quantiles transposed to (n, Q)", parsed["quantiles"].shape == (N, len(LEVELS)))
check("transpose is exact", np.array_equal(parsed["quantiles"], q_rows))
check("mean/median/mode pass through unchanged (n,)", all(np.array_equal(parsed[k], good[k]) for k in ("mean", "median", "mode")))
check("all outputs float64", all(v.dtype == np.float64 for v in parsed.values()))
check("list-of-lists input (as the server JSON decodes) is accepted",
      np.array_equal(parse_main_output({k: v.tolist() for k, v in good.items()}, n_rows=N, quantile_levels=LEVELS)["quantiles"], q_rows))
ties = q_rows.copy()
ties[:, 1] = ties[:, 0]
check("equal neighbouring quantiles (ties) are accepted", parse_main_output(raw(quantiles=ties.T.copy()), n_rows=N, quantile_levels=LEVELS) is not None)

print("\n== rejected with ProviderResponseError ==")
decreasing = q_rows.copy()
decreasing[3, [1, 2]] = decreasing[3, [2, 1]] + np.array([0.5, -0.5])
nan_q = q_rows.T.copy()
nan_q[2, 4] = np.nan
cases = {
    "NaN in quantiles": raw(quantiles=nan_q),
    "inf in mean": raw(mean=np.where(np.arange(N) == 1, np.inf, 0.0)),
    "None inside a list (client keeps it as NaN)": raw(median=[0.0] * (N - 1) + [None]),
    "decreasing quantiles in one row": raw(quantiles=decreasing.T.copy()),
    "quantiles given as (n, Q) instead of (Q, n)": raw(quantiles=q_rows.copy()),
    "wrong number of rows in mode": raw(mode=np.zeros(N - 1)),
    "missing key 'mode'": {k: v for k, v in raw().items() if k != "mode"},
    "extra key 'logits'": raw(logits=np.zeros((N, 3))),
    "not a dict": [1, 2, 3],
}
for label, value in cases.items():
    try:
        parse_main_output(value, n_rows=N, quantile_levels=LEVELS)
        check(f"{label} rejected", False)
    except ProviderResponseError as exc:
        check(f"{label} rejected: {exc}", True)

print("\n== outgoing tabpfn_config of the pinned client (construction only, no network) ==")
settings = TeacherSettings.from_config(load_config())
provider = TabPFNProvider(settings)
for cats in ((), (1,), (1, 5)):
    sent = provider.outgoing_config(cats)
    print(f"  categorical {cats}: {sent}")
    check(
        f"config for categorical {cats} equals the keyed settings",
        sent
        == {
            "categorical_features_indices": list(cats),
            "ignore_pretraining_limits": False,
            "model_path": settings.model_path,
            "n_estimators": settings.n_estimators,
            "random_state": settings.random_state,
        },
    )
import tabpfn_client  # noqa: E402

print("  installed tabpfn_client", tabpfn_client.__version__, "| configured", settings.client_version)
check("installed client version equals tabpfn.client_version", tabpfn_client.__version__ == settings.client_version)
check("no prediction call was started", provider.prediction_calls_started == 0)

print(f"\nRESULT: {len(FAILURES)} failed check(s)" + (f": {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
