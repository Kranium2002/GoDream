"""c6 PLUMBING CHECK: the command line runs the chain and returns the right exit codes.

Each case runs ``learned_judgment.pipeline.main`` in a separate process through fixture_cli.py,
which only swaps the opener (fixture skill on the fixture world instead of the Go2 simulator)
and tripwires every billable tabpfn-client entry point. The config copy (outputs in a temporary
directory, tabpfn.project_token_budget null) is passed with --config.

    case                                                     expected exit code
    run collect,teacher,dream,train,evaluate (constant_bias) 0, one line per event, summary with paths
    decide --brain physics_only / snn_per_skill              0, JSON answer
    status                                                   0, JSON with every registered skill
    run --steps collect,bogus                                2 (invalid arguments)
    run --steps dream without --method                       2
    run --steps train on a run that already has a student    1 (failed)
    run collect,dream --method teacher (budget null)         3 (refused; LIVE FREE calls only)
    run collect with a stop after 2 episodes                 4 (stopped)
    run on the same run_id with another --preset             1 (the run could not be opened)

PLUMBING CHECK on a fixture world: mechanics only; nothing here is evidence about the robot, the
simulator, TabPFN or any model.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_pipeline_offline\\c6_cli.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_pipeline import BANNER, ROOT, SKILL, Checks, make_config, write_config  # noqa: E402

print(BANNER)

from learned_judgment import pipeline as P  # noqa: E402

checks = Checks()
expect = checks.expect
WRAPPER = Path(__file__).with_name("fixture_cli.py")


def cli(config_path: Path, *arguments: str, stop_after: int | None = None) -> subprocess.CompletedProcess:
    prefix = ["--fixture-stop-after", str(stop_after)] if stop_after is not None else []
    command = [sys.executable, str(WRAPPER), *prefix, "--config", str(config_path), *arguments]
    shown = " ".join(["fixture_cli.py", *prefix, "--config", "<tmp>/config.yaml", *arguments])
    print(f"\n$ {shown}", flush=True)
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=900)
    print(f"  exit code {done.returncode}; last lines: {checks.redact(' | '.join(done.stdout.strip().splitlines()[-3:]))[:400]}")
    return done


def identity(run_id: str, preset: str = "default") -> list[str]:
    return ["--skill", SKILL, "--preset", preset, "--run-id", run_id]


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    checks.redactions[tmp_name] = "<tmp>"
    config = make_config(tmp, max_steps=300, budget_none=True)
    config_path = write_config(config, tmp / "config.yaml")

    full = cli(config_path, "run", *identity("cli"), "--steps", "collect,teacher,dream,train,evaluate", "--method", "constant_bias",
               "--attempts", "10", "--dreams", "800", "--eval-situations", "5", "--eval-repeats", "2")
    expect(full.returncode == P.EXIT_OK, f"full chain exit code {full.returncode} == {P.EXIT_OK} {checks.redact(full.stderr[-300:])}")
    lines = full.stdout.splitlines()
    for step in P.STEPS:
        expect(any(f"] {step:<8} start" in line for line in lines) and any(f"] {step:<8} finish" in line for line in lines),
               f"{step}: start and finish lines printed")
    expect(sum(" collect  progress " in line for line in lines) == 10, "one collect progress line per attempt")
    expect(any(line.strip().startswith("manifest") for line in lines) and any(line.strip().startswith("student") for line in lines)
           and sum(line.strip().startswith("result") for line in lines) == 3, "summary lists the manifest, the student and 3 results")
    manifest = P.read_run_manifest(tmp / "manifests" / "cli.json")
    expect(all(manifest["steps"][step]["state"] == "done" for step in P.STEPS), "manifest: every step done")
    expect(manifest["evaluations"][-1]["methods"] == ["physics_only", "constant_bias", "snn_per_skill"],
           "default evaluation methods: physics_only, --method, snn_per_skill")
    expect(manifest["base_seed"] == config["runtime"]["seed"], "default base seed: runtime.seed")

    for brain in ("physics_only", "snn_per_skill"):
        answer = cli(config_path, "decide", *identity("cli"), "--brain", brain,
                     "--situation", "fx_lateral=0.25", "fx_speed=4", "fx_surface=slip")
        parsed = json.loads(answer.stdout[answer.stdout.index("{"):]) if answer.returncode == 0 else {}
        expect(answer.returncode == P.EXIT_OK and parsed.get("brain") == brain and len(parsed.get("candidates", [])) == 35,
               f"decide --brain {brain}: exit 0, JSON answer with 35 candidates")

    status = cli(config_path, "status")
    parsed = json.loads(status.stdout[status.stdout.index("{"):]) if status.returncode == 0 else {}
    registered = [entry["skill"] for entry in parsed.get("skill_states", {}).get("skills", [])]
    expect(status.returncode == P.EXIT_OK and registered == list(P.available_skills()), f"status: exit 0, registered skills {registered}")
    expect(any(item["run_id"] == "cli" for item in parsed.get("skill_states", {}).get("other_manifests", [])),
           "status lists the fixture run under other_manifests (its skill is not registered)")

    bogus = cli(config_path, "run", *identity("cli"), "--steps", "collect,bogus")
    expect(bogus.returncode == P.EXIT_USAGE, f"unknown step: exit code {bogus.returncode} == {P.EXIT_USAGE}")
    no_method = cli(config_path, "run", *identity("cli"), "--steps", "dream")
    expect(no_method.returncode == P.EXIT_USAGE, f"dream without --method: exit code {no_method.returncode} == {P.EXIT_USAGE}")
    again = cli(config_path, "run", *identity("cli"), "--steps", "train")
    expect(again.returncode == P.EXIT_FAILED and "RunStateError" in again.stdout,
           f"train again on a run with a student: exit code {again.returncode} == {P.EXIT_FAILED}")

    refused = cli(config_path, "run", *identity("cli_teacher"), "--steps", "collect,dream", "--method", "teacher", "--attempts", "6",
                  "--dreams", "10")
    expect(refused.returncode == P.EXIT_REFUSED and "project_token_budget is not set" in refused.stdout,
           f"teacher dream with the budget null: exit code {refused.returncode} == {P.EXIT_REFUSED}")
    teacher_manifest = P.read_run_manifest(tmp / "manifests" / "cli_teacher.json")
    expect(teacher_manifest["steps"]["collect"]["state"] == "done" and teacher_manifest["steps"]["dream"]["state"] == "refused",
           "manifest: collect done, dream refused")
    calls = [json.loads(line) for line in (tmp / "tabpfn_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    expect(calls and all(call.get("event") == "refused" and call.get("billable_dispatch") is False for call in calls),
           f"call log: {len(calls)} refusal(s), no dispatch")

    stopped = cli(config_path, "run", *identity("cli_stop"), "--steps", "collect", "--attempts", "6", stop_after=2)
    expect(stopped.returncode == P.EXIT_STOPPED and "STOPPED collect" in stopped.stdout,
           f"stop after 2 episodes: exit code {stopped.returncode} == {P.EXIT_STOPPED}")
    stop_manifest = P.read_run_manifest(tmp / "manifests" / "cli_stop.json")
    expect(stop_manifest["steps"]["collect"]["state"] == "stopped" and stop_manifest["steps"]["collect"]["counts"]["rows_after"] == 2,
           "manifest: collect stopped with 2 rows")

    mismatch = cli(config_path, "run", *identity("cli", preset="heavy_payload"), "--steps", "collect", "--attempts", "2")
    expect(mismatch.returncode == P.EXIT_FAILED and "RunMismatchError" in mismatch.stdout,
           f"same run_id, other preset: exit code {mismatch.returncode} == {P.EXIT_FAILED}")

checks.finish()
