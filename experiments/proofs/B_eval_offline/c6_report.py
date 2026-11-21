"""Check 6: final table from run-result files -- every config method gets a row; missing ones are NOT RUN.

The run-result files are written from COMPLETED manifests of throwaway harness runs
(fixture_spec.FileMechanicsSpec: constant rows, file mechanics only) inside a temporary
directory; every metric value below is a fixture number, never a result.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from fixture_spec import Checks, throwaway_compare_run  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.eval.report import (  # noqa: E402
    NOT_REPORTED,
    NOT_RUN,
    build_final_table,
    read_run_result,
    write_final_table,
    write_run_result,
)

checks = Checks()
expect = checks.expect
config = load_config()  # read only: evaluation.methods
methods = list(require(config, "evaluation.methods"))
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    runs = throwaway_compare_run(root / "runs", run_id="fx_dev", methods=("physics_only", "constant_bias"))
    physics_result = root / "results" / "fixture__physics_only.json"
    write_run_result(
        physics_result, skill="fixture", method="physics_only", run_id="fx_dev", world_preset="default", split="dev",
        metrics={"mean_loss": float(runs["physics_only"].losses.mean()), "spearman": "undefined"},
        provenance={"manifests": [runs["physics_only"].manifest_path], "note": "fixture mechanics, not a result"},
    )
    bias_result = root / "results" / "fixture__constant_bias.json"
    write_run_result(
        bias_result, skill="fixture", method="constant_bias", run_id="fx_dev", world_preset="default", split="dev",
        metrics={"mean_loss": float(runs["constant_bias"].losses.mean())},
        provenance={"manifests": [runs["constant_bias"].manifest_path, runs["physics_only"].manifest_path]},
    )
    record = read_run_result(physics_result)
    expect(record["format"] == "learned_judgment.run_result/v1" and record["method"] == "physics_only",
           f"run-result file round trip: format {record['format']!r}, method {record['method']!r}")
    expect(not Path(record["provenance"]["manifests"][0]["path"]).is_absolute(),
           f"manifest stored relative to the result file: {record['provenance']['manifests'][0]['path']}")

    table = build_final_table([physics_result, bias_result], skills=["fixture"], config=config,
                              blockers={("fixture", "teacher"): "fixture blocker: provider request NOT RUN"})
    print(table.to_string(index=False))
    rows = {row["method"]: row for _, row in table.iterrows()}
    expect(list(table["method"]) == methods, f"one row per config method, in config order ({len(methods)} rows)")
    expect(rows["physics_only"]["status"] == "RUN" and rows["physics_only"]["source"] == str(physics_result),
           "a RUN row names its run-result file as source")
    expect(rows["physics_only"]["mean_loss"] == record["metrics"]["mean_loss"], "value copied unchanged from the file")
    expect(rows["constant_bias"]["spearman"] == NOT_REPORTED, "metric absent from a file -> 'not reported'")
    expect(rows["oracle"]["status"] == NOT_RUN and rows["oracle"]["mean_loss"] == NOT_RUN
           and "no saved run-result file" in rows["oracle"]["blocker"], "method without a file -> NOT RUN row with a blocker")
    expect(rows["teacher"]["status"] == NOT_RUN and rows["teacher"]["blocker"].startswith("fixture blocker"),
           "a caller blocker explains a NOT RUN row")
    expect(not any(str(value) == "PASS" for value in table.to_numpy().ravel()), "no PASS cell anywhere")

    csv_path, md_path = root / "final_table.csv", root / "final_table.md"
    write_final_table(table, csv_path, md_path)
    back = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    expect(list(back["status"]) == [("RUN" if m in ("physics_only", "constant_bias") else NOT_RUN) for m in methods],
           "CSV round trip statuses")
    markdown = md_path.read_text(encoding="utf-8").splitlines()
    expect(len(markdown) == 2 + len(methods), f"Markdown has header, divider and {len(methods)} rows")
    print("\n" + "\n".join(markdown))

    for label, call, error_type in (
        ("a PASS metric value", lambda: write_run_result(
            root / "results" / "bad1.json", skill="fixture", method="oracle", run_id="fx_dev", world_preset="default",
            split="dev", metrics={"m": "PASS"}, provenance={"manifests": [runs["physics_only"].manifest_path]}), ValueError),
        ("overwriting a run-result file", lambda: write_run_result(
            physics_result, skill="fixture", method="physics_only", run_id="fx_dev", world_preset="default",
            split="dev", metrics={"m": 1.0}, provenance={"manifests": [runs["physics_only"].manifest_path]}), FileExistsError),
        ("a duplicated skill x method", lambda: build_final_table(
            [physics_result, physics_result], skills=["fixture"], config=config), ValueError),
        ("a skill outside the table", lambda: build_final_table([physics_result], skills=["dodge"], config=config), ValueError),
        ("a blocker for a pair with a file", lambda: build_final_table(
            [physics_result], skills=["fixture"], config=config, blockers={("fixture", "physics_only"): "x"}), ValueError),
    ):
        try:
            call()
            expect(False, f"rejects {label}")
        except error_type as error:
            expect(True, f"rejects {label}: {type(error).__name__}: {error}")
    expect(not (root / "results" / "bad1.json").exists(), "a refused run result writes no file")

checks.finish()
