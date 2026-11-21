"""R05: rows.py file handling edge cases."""

print("REVIEW REPRODUCTION - fixture, not evidence")

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_fixture import LEVELS, SPREAD, TS, ReviewSpec  # noqa: E402

from learned_judgment.engine.adapt import select_context  # noqa: E402
from learned_judgment.engine.features import build_layout  # noqa: E402
from learned_judgment.engine.learn import fit_residual_model  # noqa: E402
from learned_judgment.engine.rows import append_rows, build_real_row, read_rows, rows_to_arrays  # noqa: E402
from learned_judgment.eval.baselines import ConstantBiasResidual  # noqa: E402

spec = ReviewSpec()


def row(i, sep=0.4, hit=0.0):
    return build_real_row(spec, run_id="rv", seed=3, world_preset="default", situation=np.array([0.1 * i, float(i % 2)]),
                          action=np.array([0.5, 0.1]), measured_outcome={"rv_sep": sep, "rv_hit": hit}, timestamp_utc=TS)


def attempt(label, fn):
    try:
        out = fn()
        print(f"   {label}: OK -> {out}")
    except Exception as error:  # noqa: BLE001
        print(f"   {label}: {type(error).__name__}: {str(error)[:140]}")


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # (1) A row mutated after validation is written without re-validation, and poisons the file.
    p = tmp / "mut.jsonl"
    good = [row(i) for i in range(5)]
    append_rows(p, good[:3])
    bad = row(9)
    bad.residual["rv_sep"] = 123.0  # frozen dataclass, but the dict is mutable
    print("1. append_rows with a row whose residual was mutated after construction:")
    attempt("append_rows", lambda: append_rows(p, [bad, *good[3:]]) or "written")
    attempt("read_rows (whole file)", lambda: len(read_rows(p)))

    p2 = tmp / "leak.jsonl"
    leaky = row(1)
    leaky.situation["payload_mass_kg"] = 2.5  # a hidden value added after construction
    print("2. a hidden key added to a built row's situation dict:")
    attempt("append_rows", lambda: append_rows(p2, [leaky]) or "written")
    attempt("read_rows without spec", lambda: read_rows(p2)[0].situation)
    attempt("rows_to_arrays", lambda: rows_to_arrays(spec, read_rows(p2)).situations.shape)

    # (3) Exact duplicate attempts (e.g. collect() re-run after a crash on the same file).
    p3 = tmp / "dup.jsonl"
    first = [row(i, sep=0.3 + 0.05 * i) for i in range(4)]
    append_rows(p3, first)
    append_rows(p3, first[:2])  # the re-run appends the first two attempts again
    back = read_rows(p3, spec=spec)
    ctx = select_context(back, policy="all", recent_window=None)
    layout = build_layout(spec)
    m = fit_residual_model(spec, ConstantBiasResidual(layout, LEVELS, SPREAD, seed=0), ctx, layout)
    print("3. duplicates: read_rows ->", len(back), "rows,", len({r.to_json_line() for r in back}), "distinct;",
          "fit used n_train =", m.n_train)

    # (4) CRLF line ends (e.g. git autocrlf) and a UTF-8 BOM
    p4 = tmp / "crlf.jsonl"
    append_rows(p4, first)
    p4.write_bytes(p4.read_bytes().replace(b"\n", b"\r\n"))
    print("4. CRLF file:")
    attempt("read_rows", lambda: len(read_rows(p4, spec=spec)))
    attempt("append_rows after CRLF then read_rows", lambda: (append_rows(p4, [row(7)]), len(read_rows(p4, spec=spec)))[1])
    p5 = tmp / "bom.jsonl"
    append_rows(p5, first)
    p5.write_bytes(b"\xef\xbb\xbf" + p5.read_bytes())
    attempt("BOM file read_rows", lambda: len(read_rows(p5)))

    # (5) overflow literal and nested duplicate key
    line = first[0].to_json_line().decode()
    p6 = tmp / "inf.jsonl"
    p6.write_text(line.replace('"rv_sep":0.3', '"rv_sep":1e999', 1), encoding="utf-8", newline="\n")
    attempt("5a. 1e999 in measured_outcome", lambda: len(read_rows(p6)))
    # 5b (nested duplicate key) is in r05b_nested_dupkey.py
