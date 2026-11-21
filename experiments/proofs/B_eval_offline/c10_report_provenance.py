"""Check 10 (review finding E3): numbers enter the final table ONLY from saved run-result files.

Reviewer reproduction r03: config.yaml, another check's output and the reproduction script
itself were each accepted as the "artifact" behind a hand-typed number, and a method left out
of the caller's list had no row. Here every such input must be refused, every method of
config.yaml evaluation.methods must get a row, and a run-result file must come from a
COMPLETED run manifest whose files have not changed since. Throwaway harness runs
(fixture_spec.FileMechanicsSpec: constant rows, file mechanics only) in a temporary directory.
On the old code build_final_table took hand-built records, so the calls below fail there.
"""

from fixture_spec import FIXTURE_BANNER

print(FIXTURE_BANNER)

import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from fixture_spec import REPO_ROOT, Checks, FileMechanicsSpec, throwaway_compare_run  # noqa: E402
from learned_judgment.config import load_config, require  # noqa: E402
from learned_judgment.eval.report import NOT_RUN, build_final_table, read_run_result, write_run_result  # noqa: E402

checks = Checks()
expect = checks.expect
config = load_config()  # read only: evaluation.methods
methods = list(require(config, "evaluation.methods"))


def refused(label: str, call, error_types=(ValueError,)) -> None:
    try:
        call()
        expect(False, f"refuses {label}")
    except error_types as error:
        expect(True, f"refuses {label}: {type(error).__name__}: {str(error)[:150]}")


print("== the reviewer's files are not run results ==")
for name in ("config.yaml", "experiments/proofs/B_eval_offline/c6_report.out.txt",
             "experiments/proofs/B_review/eval/r03_report_provenance.py"):
    refused(f"{name} as a run-result file",
            lambda name=name: build_final_table([REPO_ROOT / name], skills=["dodge"], config=config))
refused("a hand-built record instead of a file",
        lambda: build_final_table([{"skill": "dodge", "method": "teacher", "metrics": {"m": {"value": 0.0123, "artifact": "config.yaml"}}}],
                                  skills=["dodge"], config=config), (TypeError, ValueError))

print("\n== every config method has a row ==")
empty = build_final_table([], skills=["dodge"], config=config)
expect(list(empty["method"]) == methods and set(empty["status"]) == {NOT_RUN},
       f"no files -> {len(methods)} NOT RUN rows, one per evaluation.methods entry: {list(empty['method'])}")

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    runs = throwaway_compare_run(root / "runs", run_id="fx_dev", methods=("teacher", "my_method"))
    manifest = runs["teacher"].manifest_path
    common = dict(skill="fixture", world_preset="default", split="dev")

    print("\n== write_run_result checks the run manifests ==")
    refused("provenance without manifests", lambda: write_run_result(
        root / "r0.json", method="teacher", run_id="fx_dev", metrics={"m": 1.0}, provenance={}, **common))
    refused("a manifest of another split", lambda: write_run_result(
        root / "r1.json", skill="fixture", method="teacher", run_id="fx_dev", world_preset="default", split="shift",
        metrics={"m": 1.0}, provenance={"manifests": [manifest]}))
    refused("a manifest of another skill", lambda: write_run_result(
        root / "r2.json", skill="dodge", method="teacher", run_id="fx_dev", world_preset="default", split="dev",
        metrics={"m": 1.0}, provenance={"manifests": [manifest]}))
    refused("no manifest of this method and run id", lambda: write_run_result(
        root / "r3.json", method="oracle", run_id="fx_dev", metrics={"m": 1.0}, provenance={"manifests": [manifest]}, **common))
    refused("a file that is not a manifest", lambda: write_run_result(
        root / "r4.json", method="teacher", run_id="fx_dev", metrics={"m": 1.0},
        provenance={"manifests": [REPO_ROOT / "config.yaml"]}, **common))
    crashed_spec = FileMechanicsSpec(crash_at=5)
    try:
        throwaway_compare_run(root / "crashed", run_id="fx_crash", methods=("teacher",), spec=crashed_spec)
    except RuntimeError as error:
        print(f"   throwaway run crashed on purpose: {error}")
    crashed_manifest = root / "crashed" / "fx_crash__teacher.manifest.json"
    expect(json.loads(crashed_manifest.read_text(encoding="utf-8"))["completed"] is False, "the crashed run's manifest says completed: false")
    refused("an incomplete (crashed) run", lambda: write_run_result(
        root / "r5.json", method="teacher", run_id="fx_crash", metrics={"m": 1.0},
        provenance={"manifests": [crashed_manifest]}, **common))
    expect(not any((root / f"r{i}.json").exists() for i in range(6)), "no refused result wrote a file")

    print("\n== a valid result, then its run files change ==")
    result = root / "results" / "teacher.json"
    write_run_result(result, method="teacher", run_id="fx_dev", metrics={"mean_loss": float(runs["teacher"].losses.mean())},
                     provenance={"manifests": [manifest], "command": "fixture check 10"}, **common)
    table = build_final_table([result], skills=["fixture"], config=config)
    row = table[table["method"] == "teacher"].iloc[0]
    expect(row["status"] == "RUN" and row["source"] == str(result) and row["mean_loss"] == runs["teacher"].losses.mean(),
           "a valid run-result file gives a RUN row with its source file")
    other = root / "results" / "my_method.json"
    write_run_result(other, method="my_method", run_id="fx_dev", metrics={"m": 1.0},
                     provenance={"manifests": [runs["my_method"].manifest_path]}, **common)
    refused("a run-result file whose method is not in config evaluation.methods",
            lambda: build_final_table([other], skills=["fixture"], config=config))
    copied = root / "copy"
    shutil.copytree(root / "runs", copied)
    attempt = Path(runs["teacher"].rows_path)
    original = attempt.read_bytes()
    attempt.write_bytes(original.replace(b'"separation":1.0', b'"separation":2.0', 1))
    refused("a result whose attempt file changed afterwards", lambda: read_run_result(result))
    refused("the same in the final table", lambda: build_final_table([result], skills=["fixture"], config=config))
    attempt.write_bytes(original)
    expect(read_run_result(result)["method"] == "teacher", "restoring the attempt file makes the result readable again")
    Path(manifest).unlink()
    refused("a result whose manifest was deleted", lambda: read_run_result(result), (FileNotFoundError,))
    forged = root / "results" / "forged.json"
    record = json.loads((root / "results" / "teacher.json").read_text(encoding="utf-8"))
    record["metrics"]["mean_loss"] = 0.0123
    record["provenance"]["manifests"] = [{"path": "../copy/fx_dev__teacher.manifest.json", "sha256": "0" * 64,
                                          "attempt_file_sha256": "0" * 64}]
    forged.write_text(json.dumps(record), encoding="utf-8")
    refused("a hand-edited result whose digests do not match its run files", lambda: read_run_result(forged))

checks.finish()
