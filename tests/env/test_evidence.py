"""Pure utility tests for the evidence recorder. Nothing here is simulator or model evidence."""
import json
import subprocess
import sys
from pathlib import Path

from learned_judgment.env.evidence import redact_home


def record(tmp_path, label, *child):
    return subprocess.call(
        [sys.executable, "-m", "learned_judgment.env.evidence", label, "--out", str(tmp_path), "--", *child]
    )


def test_records_command_and_exit_code(tmp_path):
    rc = record(tmp_path, "t", sys.executable, "-c", "print('hi')")
    run = json.loads(next(tmp_path.glob("*-t/run.json")).read_text())
    assert rc == 0 and run["exit_code"] == 0
    assert run["command"] == [redact_home(sys.executable), "-c", "print('hi')"]  # the venv may sit under home
    assert "hi" in next(tmp_path.glob("*-t/stdout.txt")).read_text()


def test_failing_child_propagates(tmp_path):
    rc = record(tmp_path, "f", sys.executable, "-c", "import sys; sys.stderr.write('boom'); raise SystemExit(3)")
    run = json.loads(next(tmp_path.glob("*-f/run.json")).read_text())
    assert rc == 3 and run["exit_code"] == 3
    assert "boom" in next(tmp_path.glob("*-f/stderr.txt")).read_text()


def test_home_directory_is_redacted(tmp_path):
    home = Path.home()
    spellings = (str(home), home.as_posix(), "/" + home.as_posix().replace(":", "", 1).lower())  # last: Git Bash
    child = "import sys; from pathlib import Path; print(Path.home(), *sys.argv[1:]); sys.stderr.write(' '.join(sys.argv[1:]))"
    rc = record(tmp_path, "h", sys.executable, "-c", child, *spellings)
    run_dir = next(tmp_path.glob("*-h"))
    run = json.loads((run_dir / "run.json").read_text())
    assert rc == 0 and run["home_redacted"] is True
    texts = (" ".join(run["command"]), (run_dir / "stdout.txt").read_text(), (run_dir / "stderr.txt").read_text())
    assert all("~" in t for t in texts)
    assert not [s for s in spellings for t in texts if s.lower() in t.lower()]


def test_label_cannot_escape_the_output_directory(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert record(out, "a/../../escaped", sys.executable, "-c", "pass") != 0
    assert not (tmp_path / "escaped").exists()
