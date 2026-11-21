"""Pure utility tests for the evidence recorder. Nothing here is simulator or model evidence."""
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

from learned_judgment.env.evidence import redact_home


def record(tmp_path, label, *child, **kwargs):
    return subprocess.call(
        [sys.executable, "-m", "learned_judgment.env.evidence", label, "--out", str(tmp_path), "--", *child], **kwargs
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
    spellings = (
        str(home),
        home.as_posix(),
        "/" + home.as_posix().replace(":", "", 1).lower(),  # Git Bash
        str(home).replace("\\", "\\\\"),  # repr and JSON double the backslashes
        str(home).replace("\\", "/", 1),  # mixed separators
    )
    child = "import sys; from pathlib import Path; print(Path.home(), *sys.argv[1:]); sys.stderr.write(' '.join(sys.argv[1:]))"
    rc = record(tmp_path, "h", sys.executable, "-c", child, *spellings)
    run_dir = next(tmp_path.glob("*-h"))
    run = json.loads((run_dir / "run.json").read_text())
    assert rc == 0 and run["home_redacted"] is True
    texts = (" ".join(run["command"]), (run_dir / "stdout.txt").read_text(), (run_dir / "stderr.txt").read_text())
    assert all("~" in t for t in texts)
    assert not [s for s in spellings for t in texts if s.lower() in t.lower()]


def test_longer_name_that_starts_with_the_home_is_kept():
    other = str(Path.home()) + "ey"
    assert redact_home(other) == other


def test_posix_home_is_redacted(monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: PurePosixPath("/home/alice"))
    assert redact_home("/home/alice/x /home/alicea") == "~/x /home/alicea"


def test_undecodable_output_keeps_the_exit_code(tmp_path):
    # UTF-8 for a box-drawing character; 0x81 is undefined in cp1252, the locale codec on Windows.
    rc = record(tmp_path, "u", sys.executable, "-c", r"import sys; sys.stdout.buffer.write(b'\xe2\x94\x81')")
    run_dir = next(tmp_path.glob("*-u"))
    assert rc == 0 and json.loads((run_dir / "run.json").read_text())["exit_code"] == 0
    assert (run_dir / "stdout.txt").read_bytes() == b"\xe2\x94\x81"


def test_git_state_is_read_before_the_run(tmp_path):
    repo, outside = tmp_path / "repo", tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "one"], cwd=repo, check=True)
    # The recorder runs git in its cwd. learned_judgment is not installed (package = false), so PYTHONPATH finds it;
    # the ceiling stops git from finding a repository above tmp_path.
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    assert record(repo / "artifacts", "in", sys.executable, "-c", "pass", cwd=repo, env=env) == 0
    assert record(outside, "out", sys.executable, "-c", "pass", cwd=outside, env=env) == 0
    inside_run = json.loads(next((repo / "artifacts").glob("*-in/run.json")).read_text())
    outside_run = json.loads(next(outside.glob("*-out/run.json")).read_text())
    assert inside_run["git_rev"] and inside_run["git_dirty"] is False  # the run's own new directory does not count
    assert outside_run["git_rev"] is None and outside_run["git_dirty"] is None


def test_label_cannot_escape_the_output_directory(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert record(out, "a/../../escaped", sys.executable, "-c", "pass") != 0
    assert not (tmp_path / "escaped").exists()
