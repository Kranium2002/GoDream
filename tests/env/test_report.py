"""Pure utility tests for the report generator. Hand-made run.json files check the generator only."""
import json
import subprocess
import sys
from pathlib import Path

import yaml


def save_run(tmp_path, name, **run):
    d = tmp_path / name
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"command": ["x"], "packages": {}, **run}))


def report_lines(tmp_path):
    out = tmp_path / "R.md"
    subprocess.check_call([sys.executable, "-m", "learned_judgment.env.report",
                           "--artifacts", str(tmp_path), "--out", str(out)])
    return out.read_text().splitlines()


def test_empty_artifacts_reports_not_run(tmp_path):
    out = tmp_path / "R.md"
    subprocess.check_call([sys.executable, "-m", "learned_judgment.env.report",
                           "--artifacts", str(tmp_path), "--out", str(out)])
    text = out.read_text()
    assert "NOT RUN" in text and "PASS" not in text


def test_status_comes_from_saved_exit_code(tmp_path):
    for label, code in (("locomotion", 0), ("commands", 1)):
        d = tmp_path / f"20261121T000000-{label}"; d.mkdir()
        (d / "run.json").write_text(json.dumps({"command": ["x"], "exit_code": code, "packages": {}}))
    out = tmp_path / "R.md"
    subprocess.check_call([sys.executable, "-m", "learned_judgment.env.report",
                           "--artifacts", str(tmp_path), "--out", str(out)])
    lines = out.read_text().splitlines()
    assert any("locomotion" in l and "PASS" in l for l in lines)
    assert any("commands" in l and "FAIL" in l for l in lines)


def test_latest_run_of_a_label_wins(tmp_path):
    save_run(tmp_path, "20261121T000000-world", exit_code=0)
    save_run(tmp_path, "20261121T000001-world", exit_code=1)
    lines = report_lines(tmp_path)
    world = [l for l in lines if "| world |" in l and "FAIL" in l]
    assert len(world) == 2  # public API and fixed-seed reproduction both map to `world`
    assert all("20261121T000001-world" in l for l in world)
    assert not any("PASS" in l or "20261121T000000-world" in l for l in lines)


def test_one_oracle_row_per_configured_skill(tmp_path):
    skills = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["skills"]
    save_run(tmp_path, "20261121T000000-oracle-dodge", exit_code=0)
    lines = report_lines(tmp_path)
    status_table = lines[:lines.index("## Provenance")]
    for skill in skills:
        rows = [l for l in status_table if f"| oracle-{skill} |" in l]
        assert len(rows) == 1
        assert ("PASS" if skill == "dodge" else "NOT RUN") in rows[0]


def test_provenance_comes_from_run_json(tmp_path):
    save_run(tmp_path, "20261121T000000-fresh", exit_code=0, python="3.13.9 (test)", platform="TestOS-1",
             git_rev="abc123", packages={"mujoco": "9.8.7", "numpy": "6.5.4", "unlisted-pkg": "1.0"})
    text = "\n".join(report_lines(tmp_path))
    for value in ("3.13.9 (test)", "TestOS-1", "abc123", "9.8.7", "6.5.4"):
        assert value in text
    assert "unlisted-pkg" not in text
