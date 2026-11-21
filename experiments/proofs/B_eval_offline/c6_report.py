"""Check 6: final table from records -- a missing method becomes a NOT RUN row.

Artifacts and outputs live only in a temporary directory; every value below is a fixture
number in a fixture file, never a result.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from fixture_spec import Checks  # noqa: E402
from learned_judgment.eval.report import NOT_RUN, build_final_table, write_final_table  # noqa: E402

checks = Checks()
expect = checks.expect
methods = ["physics_only", "constant_bias", "teacher"]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    artifact = root / "fixture_run.json"
    artifact.write_text('{"fixture": true, "note": "not a result"}\n', encoding="utf-8")
    records = [
        {"skill": "fixture", "method": "physics_only",
         "metrics": {"mean_abs_error": {"value": 0.25, "artifact": "fixture_run.json"},
                     "spearman": {"value": "undefined", "artifact": "fixture_run.json"}}},
        {"skill": "fixture", "method": "teacher", "blocker": "fixture blocker: provider request NOT RUN"},
    ]  # constant_bias has no record at all
    table = build_final_table(records, skills=["fixture"], methods=methods, artifact_root=root)
    print(table.to_string(index=False))
    rows = {row["method"]: row for _, row in table.iterrows()}
    expect(list(table["method"]) == methods, "one row per skill x method, in the declared order")
    expect(rows["constant_bias"]["status"] == NOT_RUN and rows["constant_bias"]["mean_abs_error"] == NOT_RUN
           and "no saved result record" in rows["constant_bias"]["blocker"], "method without any record -> NOT RUN row with a blocker")
    expect(rows["teacher"]["status"] == NOT_RUN and rows["teacher"]["blocker"].startswith("fixture blocker"),
           "not-run record keeps its own blocker")
    expect(rows["physics_only"]["mean_abs_error"] == 0.25 and rows["physics_only"]["mean_abs_error__artifact"] == "fixture_run.json",
           "numeric cell carries its artifact path")
    expect(not any(str(value) == "PASS" for value in table.to_numpy().ravel()), "no PASS cell anywhere")

    csv_path, md_path = root / "final_table.csv", root / "final_table.md"
    write_final_table(table, csv_path, md_path)
    back = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    expect(list(back["status"]) == ["RUN", NOT_RUN, NOT_RUN], f"CSV round trip statuses {list(back['status'])}")
    markdown = md_path.read_text(encoding="utf-8").splitlines()
    expect(len(markdown) == 2 + len(methods), f"Markdown has header, divider and {len(methods)} rows")
    print("\n" + "\n".join(markdown))

    for label, bad, error_type in (
        ("a PASS string as a value", {"skill": "fixture", "method": "physics_only",
                                      "metrics": {"m": {"value": "PASS", "artifact": "fixture_run.json"}}}, ValueError),
        ("a missing artifact file", {"skill": "fixture", "method": "physics_only",
                                     "metrics": {"m": {"value": 1.0, "artifact": "missing.json"}}}, FileNotFoundError),
        ("a record with metrics and a blocker", {"skill": "fixture", "method": "physics_only", "blocker": "x",
                                                 "metrics": {"m": {"value": 1.0, "artifact": "fixture_run.json"}}}, ValueError),
        ("an undeclared method", {"skill": "fixture", "method": "oracle", "blocker": "x"}, ValueError),
    ):
        try:
            build_final_table([bad], skills=["fixture"], methods=methods, artifact_root=root)
            expect(False, f"rejects {label}")
        except error_type as error:
            expect(True, f"rejects {label}: {type(error).__name__}: {error}")
    try:
        build_final_table(records + [records[0]], skills=["fixture"], methods=methods, artifact_root=root)
        expect(False, "rejects a duplicated record")
    except ValueError as error:
        expect(True, f"rejects a duplicated record: {error}")

checks.finish()
