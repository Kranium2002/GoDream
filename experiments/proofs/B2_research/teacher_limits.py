"""B2 research: what the hosted TabPFN teacher accepts and what it costs (wide image-window tables).

Run from the repository root with the project venv:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe experiments/proofs/B2_research/teacher_limits.py \
        > experiments/proofs/B2_research/teacher_limits.out.txt 2>&1

Network calls made (all FREE, none billable):
    prior_labs.connect() (token check), POST /get_api_usage/, GET /tabpfn/get_settings,
    POST /tabpfn/estimate_cost (dimensions only; "does not upload the data or consume tokens",
    docs.priorlabs.ai/api/metering.md). NO fit, NO predict, NO upload. Account usage is read before
    and after and must be unchanged. No credential is printed.

Sections
    1  ledger (results/tabpfn_calls.jsonl): events, spend, quote == charged, latency
    2  teacher stacking/chunking rules, executed on the teacher's pure functions
    3  offline request validation of wide shapes: tabpfn_client validators + TabPFNProvider.validate_request
    4  account usage and server limits (get_settings)
    5  dense quote grid -> fitted quota_v3 formula (column cap, symmetry, n_estimators)
    6  the requested hypothetical grid N x M x F through the teacher's own quote path
    7  scenario table S1/S2 x F under stacked and per_outcome modes; budget comparison
    8  payload size of a wide test set (client's own parquet serializer, synthetic arrays)
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from learned_judgment.config import load_config  # noqa: E402
from learned_judgment.teacher import (  # noqa: E402
    TabPFNProvider,
    TeacherSettings,
    budget_refusal_reasons,
    remaining_tokens,
    stack_query,
    stack_training,
)
from learned_judgment.teacher_cache import parse_log_text, tokens_quoted  # noqa: E402

FLOOR = 10000  # documented minimum charge of a TabPFN-3/3.5 operation (metering.md)
QUOTE_PAUSE_S = 0.05


def banner(text: str) -> None:
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100, flush=True)


print("utc_now =", datetime.now(timezone.utc).isoformat(timespec="seconds"))
print("python =", sys.version.split()[0], "| numpy =", np.__version__)
config = load_config()
settings = TeacherSettings.from_config(config)
print(
    "teacher settings:",
    json.dumps(
        {
            "client_version": settings.client_version,
            "model_path": settings.model_path,
            "n_estimators": settings.n_estimators,
            "random_state": settings.random_state,
            "quantile_levels": settings.quantile_levels,
            "spread_levels": settings.spread_levels,
            "multi_outcome_mode": settings.multi_outcome_mode,
            "max_tokens_per_request": settings.max_tokens_per_request,
            "max_query_rows_per_request": settings.max_query_rows_per_request,
            "project_token_budget": settings.project_token_budget,
            "verified_request_limit_per_minute": settings.verified_request_limit_per_minute,
            "verified_request_limit_per_hour": settings.verified_request_limit_per_hour,
            "call_log": str(settings.call_log.relative_to(ROOT)),
        }
    ),
)

# ----------------------------------------------------------------------------------------------
banner("1  LEDGER results/tabpfn_calls.jsonl (read-only; parsed with teacher_cache.parse_log_text, no lock taken)")
records = parse_log_text(settings.call_log.read_text(encoding="utf-8"), settings.call_log)
events: dict[str, int] = {}
for record in records:
    events[record.get("event")] = events.get(record.get("event"), 0) + 1
dispatches = [r for r in records if r.get("billable_dispatch") is True]
completed = [r for r in records if r.get("event") == "completed"]
print("records =", len(records), "| events =", events)
print("tokens_quoted over dispatch records (the project gate's spend) =", tokens_quoted(records, settings.call_log))
mismatch = [
    r["key"][:12] for r in completed if r.get("usage_delta", {}).get("current_usage") != r.get("quote_tokens")
]
print(
    "completed records whose usage_delta.current_usage != quote_tokens:",
    mismatch or "none",
    f"(checked {len(completed)})",
)
print("charged per completed request (usage_delta.current_usage):", sorted({r["usage_delta"]["current_usage"] for r in completed}))
by_shape: dict[tuple, list] = {}
for r in completed:
    by_shape.setdefault((r["skill"], r["n_train"], r["n_query"], r["n_features"]), []).append(r)
print("\nper request shape: skill, n_train, n_query, n_features | count | wall fit s | wall predict s | server fit elapsed s | server predict_s | queue waits")
for shape, rs in sorted(by_shape.items(), key=lambda kv: kv[0][2]):
    wf = [r["wall_s"]["fit"] for r in rs]
    wp = [r["wall_s"]["predict"] for r in rs]
    sf = [r["timings"]["fit"]["elapsed_s"] for r in rs]
    sp = [r["timings"]["predict"]["predict_s"] for r in rs]
    qw = [r["timings"]["fit"]["queue_wait_s"] + r["timings"]["predict"]["test_set_transform_queue_wait_s"] for r in rs]
    print(
        f"  {shape} | {len(rs)} | {min(wf):.2f}-{max(wf):.2f} | {min(wp):.2f}-{max(wp):.2f} | "
        f"{min(sf):.2f}-{max(sf):.2f} | {min(sp):.2f}-{max(sp):.2f} | {min(qw):.2f}-{max(qw):.2f}"
    )
gaps = []
for d in dispatches:
    done = next((c for c in completed if c["key"] == d["key"]), None)
    if done:
        gaps.append(
            (datetime.fromisoformat(done["timestamp_utc"]) - datetime.fromisoformat(d["timestamp_utc"])).total_seconds()
        )
print(f"dispatch -> completed seconds (includes usage/quote reads, disk writes): min {min(gaps):.2f} max {max(gaps):.2f} mean {np.mean(gaps):.2f} n {len(gaps)}")

# ----------------------------------------------------------------------------------------------
banner("2  TEACHER STACKING / CHUNKING (teacher.stack_training, teacher.stack_query on small arrays)")
rng = np.random.default_rng(0)
n, F, m, K = 7, 12, 9, 4
x_long, y_long, mean, std = stack_training(rng.normal(size=(n, F)), rng.normal(size=(n, K)), (0, 1, 2, 3))
q_long = stack_query(rng.normal(size=(m, F)), (0, 1, 2, 3))
print(f"n={n} F={F} m={m} A={K}: x_long {x_long.shape} y_long {y_long.shape} q_long {q_long.shape} -> rows A*n, A*m; width F+1")
print("last column of q_long (outcome index, outcome-major blocks):", q_long[:, -1].astype(int).tolist())
one = stack_query(rng.normal(size=(m, F)), (2,))
print(f"A=1: stack_query returns {one.shape} (no extra column)")


def chunk_rows(total: int, cap: int) -> list[int]:
    """Row counts of the chunks teacher._request_rows makes (teacher.py:1336-1338), plus the
    padding row the teacher adds to a chunk of exactly Q rows (teacher.py:1372-1373)."""
    q = len(settings.quantile_levels)
    return [(min(s + cap, total) - s) + (1 if min(s + cap, total) - s == q else 0) for s in range(0, total, cap)]


print("chunks of 32000 stacked rows at cap 32000:", chunk_rows(32000, 32000), "| 32001 rows:", chunk_rows(32001, 32000))

# ----------------------------------------------------------------------------------------------
banner("3  OFFLINE VALIDATION OF WIDE SHAPES (client validators read limits from GET /tabpfn/get_settings)")
provider = TabPFNProvider(settings)
provider.connect()  # checks tabpfn_client.__version__ == tabpfn.client_version, then prior_labs.connect()
import tabpfn_client  # noqa: E402
from tabpfn_client.client import ServiceClient, _serialize_to_parquet  # noqa: E402
from tabpfn_client.estimator import validate_test_set, validate_train_set  # noqa: E402

print("tabpfn_client", tabpfn_client.__version__, "| base_url", provider.base_url())


def client_check(label: str, train: tuple[int, int] | None, test: tuple[int, int] | None, n_train_rows: int | None = None) -> None:
    try:
        if train is not None:
            validate_train_set(np.broadcast_to(0.0, train), np.broadcast_to(0.0, (train[0],)), settings.model_path)
        if test is not None:
            validate_test_set(np.broadcast_to(0.0, test), "main", settings.model_path, train_rows=n_train_rows)
        print(f"  ACCEPTED  {label}")
    except ValueError as exc:
        print(f"  REJECTED  {label}: {exc}")


client_check("S1 stacked train (1600, 1161)", (1600, 1161), None)
client_check("S1 stacked query (32000, 1161) with 1600 train rows", None, (32000, 1161), 1600)
client_check("query (172265, 1161) = 199,999,665 cells", None, (172265, 1161), 1600)
client_check("NEG query (172266, 1161) = 200,000,826 cells", None, (172266, 1161), 1600)
client_check("NEG train (100, 23001) columns", (100, 23001), None)
client_check("train (100, 23000) columns", (100, 23000), None)
client_check("NEG query rows over predict_row_pairs_budget // train_rows (train 1000000 rows, query 250001)", None, (250001, 1), 1000000)
client_check("query (250000, 1) with 1000000 train rows", None, (250000, 1), 1000000)

x_train = np.zeros((1600, 1161))
y_train = np.arange(1600, dtype=np.float64)
for rows in (8000, 32000):
    try:
        provider.validate_request(
            x_train, y_train, np.zeros((rows, 1161)), categorical_indices=(1159, 1160), quantile_levels=settings.quantile_levels
        )
        print(f"  ACCEPTED  TabPFNProvider.validate_request train (1600, 1161) query ({rows}, 1161) categorical (1159, 1160)")
    except Exception as exc:  # noqa: BLE001
        print(f"  REJECTED  TabPFNProvider.validate_request ({rows}, 1161): {type(exc).__name__}: {exc}")

# ----------------------------------------------------------------------------------------------
banner("4  ACCOUNT USAGE (allow-listed fields) AND SERVER LIMITS")
usage_before = provider.usage()
print("usage_before =", json.dumps(usage_before))
print("remaining_tokens(usage) =", remaining_tokens(usage_before))
s = ServiceClient.get_settings()
print("default_model_version =", getattr(s.default_model_version, "value", s.default_model_version))
for version in ("v3.5", "v3.5-fast", "v3"):
    limit = next((v for k, v in s.model_limits.items() if getattr(k, "value", k) == version), None)
    print(f"model_limits[{version}] =", json.dumps(limit.model_dump() if limit else None))
print("dataset_max_size_bytes =", s.dataset_max_size_bytes, "| inline_test_set_max_parquet_bytes =", s.inline_test_set_max_parquet_bytes)
print("async_settings =", json.dumps(s.async_settings.model_dump(mode="json")))


# ----------------------------------------------------------------------------------------------
def raw_quote(n_train: int, n_query: int, n_features: int, n_estimators: int = 8) -> int:
    from tabpfn_client import estimate_cost

    for attempt in range(4):
        try:
            response = estimate_cost(
                np.broadcast_to(0.0, (n_train, n_features)),
                np.broadcast_to(0.0, (n_query, n_features)),
                model_version="v3.5",
                n_estimators=n_estimators,
            )
            time.sleep(QUOTE_PAUSE_S)
            assert response.pricing_version == "quota_v3", response.pricing_version
            return int(response.estimated_cost)
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001 - transient: wait and retry, then give up loudly
            print(f"    quote retry {attempt}: {type(exc).__name__}: {exc}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("estimate_cost failed 4 times")


banner("5  QUOTA_V3 FORMULA FROM A DENSE QUOTE GRID (model_version v3.5, operation predict)")
ref = raw_quote(1600, 254400, 23000)
lo, hi = 1, 23000
while hi - lo > 1:
    mid = (lo + hi) // 2
    if raw_quote(1600, 254400, mid) == ref:
        hi = mid
    else:
        lo = mid
CAP = hi
print(f"column cap: quote(1600, 254400, F) is constant for F >= {CAP} ({ref} tokens); F = {lo} differs ({raw_quote(1600, 254400, lo)})")
print("train/test symmetry (quote depends on N + M):")
for (a, b) in ((400, 64000), (64000, 400), (32200, 32200), (128000, 1600), (1600, 128000)):
    print(f"   N={a:>6} M={b:>6} F=13: {raw_quote(a, b, 13):>7}   F=601: {raw_quote(a, b, 601):>7}")
print("n_estimators linearity at (1600, 128000, 1161):", {ne: raw_quote(1600, 128000, 1161, ne) for ne in (1, 2, 4, 8)})

grid = []
for total, f in itertools.product(
    (40000, 64000, 100000, 160000, 256000, 400000, 640000, 1000000), (1, 5, 13, 34, 61, 161, 301, 601, CAP - 1, CAP, 1161)
):
    grid.append((total, f, raw_quote(400, total - 400, f)))
g = np.array(grid, dtype=float)
above = g[g[:, 2] > FLOOR]
T, Fc, y = above[:, 0], np.minimum(above[:, 1], CAP), above[:, 2]
design = np.c_[np.ones_like(T), T, T**2, T * Fc]
coef, *_ = np.linalg.lstsq(design, y, rcond=None)
resid = y - design @ coef
print(f"least squares on {len(y)} above-floor quotes, cost8 = a + b*T + c*T^2 + d*T*min(F,{CAP}), T = N + M:")
print(f"   a = {coef[0]:.4f}  b = {coef[1]:.4e}  c = {coef[2]:.6e}  d = {coef[3]:.6e}  max|resid| = {abs(resid).max():.3f}")
design2 = np.c_[np.ones_like(T), T**2, T * Fc]
coef2, *_ = np.linalg.lstsq(design2, y, rcond=None)
resid2 = y - design2 @ coef2
A0, C2, D1 = (float(v) for v in coef2)
print(f"without the T term: a = {A0:.4f}  c = {C2:.6e}  d = {D1:.6e}  max|resid| = {abs(resid2).max():.3f}")


def formula(n_train: int, n_query: int, n_features: int, n_estimators: int = 8) -> int:
    t = n_train + n_query
    raw = n_estimators / 8.0 * (A0 + C2 * t * t + D1 * t * min(n_features, CAP))
    return max(FLOOR, int(round(raw)))


errors = [abs(formula(400, int(t) - 400, int(f)) - int(c)) for t, f, c in grid]
print(f"formula vs all {len(grid)} grid quotes (incl. floor): max |diff| = {max(errors)} tokens")

# ----------------------------------------------------------------------------------------------
banner("6  REQUESTED GRID through TabPFNProvider.quote (teacher's own free quote path, pinned model/n_estimators)")
print(f"{'N':>5} {'M':>5} | " + " ".join(f"F={f:>5}" for f in (12, 60, 160, 300, 600, 1160)) + " | formula max|diff|")
worst = 0
for n_ctx, m_q in itertools.product((200, 400, 800), (400, 2000, 8000)):
    cells = []
    for f in (12, 60, 160, 300, 600, 1160):
        quote = provider.quote(n_ctx, m_q, f)
        time.sleep(QUOTE_PAUSE_S)
        cells.append(quote["estimated_cost"])
        worst = max(worst, abs(formula(n_ctx, m_q, f) - quote["estimated_cost"]))
    print(f"{n_ctx:>5} {m_q:>5} | " + " ".join(f"{c:>7}" for c in cells) + f" | {worst}")
print("(N context rows, M query rows, F feature columns exactly as sent; one request each; 8 estimators)")

# ----------------------------------------------------------------------------------------------
banner("7  SCENARIOS: 400 context attempts, K=A=4 active outcomes, dream situations x 4 actions, ONE dream call")
spent_project = tokens_quoted(records, settings.call_log)
remaining_project = settings.project_token_budget - spent_project
remaining_account = remaining_tokens(usage_before)
print(f"project budget {settings.project_token_budget} | quoted for dispatched requests so far {spent_project} | project remaining {remaining_project}")
print(f"account remaining (min over daily/monthly/usage_limit pairs) {remaining_account}")


def plan(mode: str, n_ctx: int, n_situations: int, f: int) -> list[tuple[int, int, int]]:
    """(n_train, n_query, n_features) of every request the teacher makes for one predict call."""
    m_rows = n_situations * 4  # 4 candidate actions per situation
    if mode == "stacked":
        return [(4 * n_ctx, rows, f + 1) for rows in chunk_rows(4 * m_rows, settings.max_query_rows_per_request)]
    return [(n_ctx, rows, f) for _ in range(4) for rows in chunk_rows(m_rows, settings.max_query_rows_per_request)]


quote_memo: dict[tuple[int, int, int], int] = {}


def teacher_quote(shape: tuple[int, int, int]) -> int:
    if shape not in quote_memo:
        quote_memo[shape] = provider.quote(*shape)["estimated_cost"]
        time.sleep(QUOTE_PAUSE_S)
    return quote_memo[shape]


print(f"{'scen':>4} {'mode':>11} {'F':>5} | requests (n_train, n_query, width) x count | tokens/request | total | refused@{settings.max_tokens_per_request} | dream calls in project remainder")
for scen, n_sit in (("S1", 2000), ("S2", 500)):
    for mode in ("stacked", "per_outcome"):
        for f in (12, 60, 160, 1160):
            shapes = plan(mode, 400, n_sit, f)
            costs = [teacher_quote(sh) for sh in shapes]
            total = sum(costs)
            refused = sum(
                1
                for c in costs
                if budget_refusal_reasons(
                    quote_tokens=c,
                    max_tokens_per_request=settings.max_tokens_per_request,
                    project_token_budget=settings.project_token_budget,
                    tokens_already_quoted=spent_project,
                    remaining_account_tokens=remaining_account,
                )
            )
            kinds = sorted(set(shapes))
            text = ", ".join(f"{k} x{shapes.count(k)}" for k in kinds)
            print(f"{scen:>4} {mode:>11} {f:>5} | {text} | {sorted(set(costs))} | {total} | {refused}/{len(shapes)} | {remaining_project // total}")

print("\nkeeping max_tokens_per_request = 10000 instead: largest chunk whose quote stays at the floor (stacked, n_train 1600)")
for f in (61, 161, 1161):
    lo_rows, hi_rows = 1, 32000
    while hi_rows - lo_rows > 1:
        mid = (lo_rows + hi_rows) // 2
        if formula(1600, mid, f) <= FLOOR:
            lo_rows = mid
        else:
            hi_rows = mid
    check = teacher_quote((1600, lo_rows, f)), teacher_quote((1600, lo_rows + 1, f))
    n_req = math.ceil(32000 / lo_rows)
    print(f"  width {f}: chunk {lo_rows} rows -> quote {check[0]} (chunk {lo_rows + 1} -> {check[1]}); S1 = {n_req} requests = {n_req * FLOOR} tokens vs one 32000-row request {teacher_quote((1600, 32000, f))}")

# ----------------------------------------------------------------------------------------------
banner("8  PAYLOAD SIZE OF A WIDE TEST SET (tabpfn_client.client._serialize_to_parquet, zstd; SYNTHETIC arrays, sizing only)")
import pandas as pd  # noqa: E402

rows = 4000
for label, array in (
    ("8-bit pixel levels k/255 uniform", rng.integers(0, 256, size=(rows, 1161)) / 255.0),
    ("continuous uniform float64", rng.random((rows, 1161))),
    ("mostly flat frame: 90% of cells 0.5, rest k/255", np.where(rng.random((rows, 1161)) < 0.9, 0.5, rng.integers(0, 256, size=(rows, 1161)) / 255.0)),
):
    payload, _ = _serialize_to_parquet(pd.DataFrame(array))
    per_row = len(payload) / rows
    print(f"  {label}: {len(payload)} bytes for {rows} rows ({per_row:.0f} B/row) -> 32000 rows ~ {per_row * 32000 / 1e6:.1f} MB; in-memory float64 32000x1161 = {32000 * 1161 * 8 / 1e6:.0f} MB")
print(f"  inline_test_set_max_parquet_bytes = {s.inline_test_set_max_parquet_bytes}: above it the client uploads the test set to a signed URL instead (client.py:2203-2214)")

# ----------------------------------------------------------------------------------------------
banner("USAGE AFTER")
usage_after = provider.usage()
print("usage_after =", json.dumps(usage_after))
print("usage unchanged by this script:", usage_after == usage_before)
print("prediction_calls_started =", provider.prediction_calls_started, "(must be 0)")
print("DONE (no fit, predict or upload called)")
