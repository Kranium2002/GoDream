"""E7 settle (LIVE, PAID, ONE REQUEST): the first real TabPFN prediction through the project's teacher.

CONTRACT PROOF on FIXTURE data shaped like one DODGE decision: 40 context rows, 17 feature columns
of which one is categorical, 5 outcomes (3 regression, 2 binary), 108 candidate rows. It verifies
three things the offline checks could not:

  1. the provider's real response has the layout the teacher parses (mean, median, quantiles);
  2. the tokens charged, read from the account before and after, against the free quote;
  3. a repeated identical call is answered from the cache and sends no second request.

The predicted numbers are not evidence about the robot, the simulator or the quality of any model.

This is a real billed call, so it uses the configured cache directory and call log (config.yaml,
section tabpfn): the call belongs in the project's ledger. The teacher's own gate applies unchanged:
usage and quote are read first, and the request is refused if the quote is above
tabpfn.max_tokens_per_request or the project budget would be exceeded.

Run from D:\\BerlinPrior:
    .venv\\Scripts\\python.exe experiments\\proofs\\E7_tabpfn_contract\\p6_first_real_request.py
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

print("CONTRACT PROOF - one real billed TabPFN request on FIXTURE data; not evidence about the robot, "
      "the simulator or any model's quality")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tabpfn_client  # noqa: E402

from learned_judgment.config import config_digest, load_config  # noqa: E402
from learned_judgment.engine.features import FeatureLayout  # noqa: E402
from learned_judgment.teacher import TabPFNProvider, TabPFNTeacher, TeacherSettings  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAILED'}] {label}")
    if not ok:
        FAILURES.append(label)


NUMERIC_USAGE = (
    "current_usage", "usage_limit", "daily_tokens_used", "daily_token_limit", "monthly_tokens_used", "monthly_token_limit",
)


def numbers(usage: dict) -> dict:
    return {name: usage.get(name) for name in NUMERIC_USAGE}


# ----- fixture shaped like one DODGE decision ----------------------------------------------------
SITUATION = ("ball_rel_x_m", "ball_rel_y_m", "ball_rel_z_m", "ball_vel_x_m_s", "ball_vel_y_m_s", "ball_vel_z_m_s",
             "dog_forward_speed_m_s", "declared_payload_kg", "declared_surface")
ACTION = ("side_velocity_m_s", "start_delay_s", "hold_duration_s")
OUTCOMES = ("min_separation_m", "collision", "fall", "recovery_time_s", "lateral_command_m")
KINDS = ("regression", "binary", "binary", "regression", "regression")
SURFACE_COLUMN = SITUATION.index("declared_surface")
LAYOUT = FeatureLayout(
    skill="fixture_first_request",
    names=SITUATION + ACTION + tuple(f"physics_{name}" for name in OUTCOMES),
    categorical=((SURFACE_COLUMN, ("default", "wet")),),
    n_situation=len(SITUATION),
    n_action=len(ACTION),
    n_physics=len(OUTCOMES),
    n_extra=0,
    outcome_names=OUTCOMES,
    outcome_kinds=KINDS,
)
N_CONTEXT, N_CANDIDATES = 40, 108
rng = np.random.default_rng(20261121)


def features(n: int) -> np.ndarray:
    block = rng.uniform(-1.0, 1.0, size=(n, LAYOUT.n_features))
    block[:, SURFACE_COLUMN] = rng.integers(0, 2, size=n)
    return block


def true_residual(x: np.ndarray) -> np.ndarray:
    """Noise-free residual of the fixture: a smooth function of two feature columns per outcome."""
    side, delay = x[:, len(SITUATION)], x[:, len(SITUATION) + 1]
    return np.stack([
        0.20 * side + 0.05 * np.sin(3.0 * delay),
        0.30 * (side < -0.3),
        0.10 * (delay > 0.5),
        0.40 * delay * delay,
        -0.15 * side * delay,
    ], axis=1)


CONTEXT = features(N_CONTEXT)
RESIDUALS = true_residual(CONTEXT) + rng.normal(scale=0.02, size=(N_CONTEXT, len(OUTCOMES)))
QUERY = features(N_CANDIDATES)
QUERY[:, : len(SITUATION)] = QUERY[0, : len(SITUATION)]  # one situation, 108 candidate actions
TRUE_QUERY = true_residual(QUERY)

config = load_config()
settings = TeacherSettings.from_config(config)
print("\n== settings (config.yaml, unchanged) ==")
print("  config digest", config_digest(config)[:12], "| client", settings.client_version, "| model_path", settings.model_path,
      "| n_estimators", settings.n_estimators, "| mode", settings.multi_outcome_mode)
print("  max_tokens_per_request", settings.max_tokens_per_request, "| project_token_budget", settings.project_token_budget)
print("  cache_dir", settings.cache_dir, "| call_log", settings.call_log)

clock = lambda: datetime.now(timezone.utc)  # noqa: E731
provider = TabPFNProvider(settings)
provider.connect()
print("\n== before ==")
print("  connected; installed tabpfn_client", tabpfn_client.__version__)
usage_before = numbers(provider.usage())
print("  usage before:", usage_before)

teacher = TabPFNTeacher.from_settings(LAYOUT, settings, provider=provider, clock=clock)
records_before = len(teacher.call_log.read()) if Path(settings.call_log).exists() else 0
teacher.fit(CONTEXT, RESIDUALS)

print("\n== the request ==")
started = time.perf_counter()
first = teacher.predict_residual_distribution(QUERY)
wall_first = time.perf_counter() - started
calls_after_first = provider.prediction_calls_started
hashes_first = teacher.last_request_hashes
print(f"  answered in {wall_first:.2f} s | prediction calls started: {calls_after_first} | request hashes: "
      f"{[h[:12] for h in hashes_first]}")

n, k, q = N_CANDIDATES, len(OUTCOMES), len(first.quantile_levels)
print("\n== response layout as parsed by the teacher ==")
print("  quantile levels", list(first.quantile_levels), "| spread levels", list(first.spread_levels))
check(f"mean has shape ({n}, {k})", first.mean.shape == (n, k))
check(f"median has shape ({n}, {k})", first.median.shape == (n, k))
check(f"quantiles have shape ({n}, {k}, {q})", first.quantiles.shape == (n, k, q))
check(f"spread has shape ({n}, {k})", first.spread.shape == (n, k))
check("every value is finite", all(np.all(np.isfinite(a)) for a in (first.mean, first.median, first.quantiles, first.spread)))
check("quantiles do not decrease with the level", bool(np.all(np.diff(first.quantiles, axis=2) >= 0.0)))
check("spread is not negative", bool(np.all(first.spread >= 0.0)))
levels = [float(level) for level in first.quantile_levels]
if 0.5 in levels:
    gap = float(np.max(np.abs(first.quantiles[:, :, levels.index(0.5)] - first.median)))
    print(f"  largest |quantile(0.5) - median| = {gap:.3e}")
for index, name in enumerate(OUTCOMES):
    predicted, truth = first.mean[:, index], TRUE_QUERY[:, index]
    corr = float(np.corrcoef(predicted, truth)[0, 1]) if np.std(predicted) > 0 and np.std(truth) > 0 else float("nan")
    print(f"  {name:>18}: mean abs error of the predicted mean against the fixture's noise-free residual "
          f"{float(np.mean(np.abs(predicted - truth))):.4f} | correlation {corr:.3f} | mean spread {float(np.mean(first.spread[:, index])):.4f}")

print("\n== charge against the quote ==")
records = teacher.call_log.read()[records_before:]
for record in records:
    shown = {key: value for key, value in record.items() if key not in ("usage_before", "usage_after", "quote")}
    print("  call-log record:", shown)
billed = [record for record in records if record.get("billable_dispatch") is True]
check("the call log holds at least one billed dispatch for this request", len(billed) >= 1)
quoted = sum(int(record.get("quote_tokens", 0)) for record in billed)
usage_after = numbers(provider.usage())
print("  usage after: ", usage_after)
moved = {name: (usage_before[name], usage_after[name]) for name in NUMERIC_USAGE if usage_before[name] != usage_after[name]}
print(f"  quoted for the billed dispatch(es): {quoted} tokens | account counters that moved: {moved}")
if not moved:
    time.sleep(10.0)
    usage_after = numbers(provider.usage())
    moved = {name: (usage_before[name], usage_after[name]) for name in NUMERIC_USAGE if usage_before[name] != usage_after[name]}
    print(f"  re-read after 10 s: account counters that moved: {moved}")
charged = {name: after - before for name, (before, after) in moved.items() if isinstance(before, int) and isinstance(after, int)}
print(f"  charged according to the account: {charged} (the quote was {quoted})")

print("\n== the same call again ==")
started = time.perf_counter()
second = teacher.predict_residual_distribution(QUERY)
wall_second = time.perf_counter() - started
print(f"  answered in {wall_second:.3f} s | prediction calls started: {provider.prediction_calls_started}")
check("no second request was sent", provider.prediction_calls_started == calls_after_first)
check("the cached answer is bit-for-bit the first answer",
      all(np.array_equal(a, b) for a, b in ((first.mean, second.mean), (first.median, second.median),
                                            (first.quantiles, second.quantiles), (first.spread, second.spread))))
check("the request hashes are the same", teacher.last_request_hashes == hashes_first)
usage_final = numbers(provider.usage())
check("the account counters did not move for the cached call", usage_final == usage_after)
print(f"  tokens quoted for billed calls in the project ledger so far: {teacher.call_log.tokens_quoted_for_billed_calls()}")

print(f"\nRESULT: {len(FAILURES)} failed check(s)" + (f": {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
