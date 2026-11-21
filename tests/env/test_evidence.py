"""Pure utility tests for the evidence recorder. Nothing here is simulator or model evidence."""
import json
import subprocess
import sys


def record(tmp_path, label, *child):
    return subprocess.call(
        [sys.executable, "-m", "learned_judgment.env.evidence", label, "--out", str(tmp_path), "--", *child]
    )


def test_records_command_and_exit_code(tmp_path):
    rc = record(tmp_path, "t", sys.executable, "-c", "print('hi')")
    run = json.loads(next(tmp_path.glob("*-t/run.json")).read_text())
    assert rc == 0 and run["exit_code"] == 0
    assert run["command"] == [sys.executable, "-c", "print('hi')"]
    assert "hi" in next(tmp_path.glob("*-t/stdout.txt")).read_text()


def test_failing_child_propagates(tmp_path):
    rc = record(tmp_path, "f", sys.executable, "-c", "import sys; sys.stderr.write('boom'); raise SystemExit(3)")
    run = json.loads(next(tmp_path.glob("*-f/run.json")).read_text())
    assert rc == 3 and run["exit_code"] == 3
    assert "boom" in next(tmp_path.glob("*-f/stderr.txt")).read_text()


def test_label_cannot_escape_the_output_directory(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert record(out, "a/../../escaped", sys.executable, "-c", "pass") != 0
    assert not (tmp_path / "escaped").exists()
