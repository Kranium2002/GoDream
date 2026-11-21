"""Run a command and save what happened.

    python -m learned_judgment.env.evidence <label> [--out DIR] -- <command...>

Writes DIR/<UTC stamp>-<label>/{run.json,stdout.txt,stderr.txt} and exits with the child's exit code.
"""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def _git(*args):
    done = subprocess.run(["git", *args], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def redact_home(text: str) -> str:
    """Replace the user's home directory (and so the user name) with ~, in Windows, /, doubled-\\ and Git Bash spellings."""
    home = Path.home()
    drive = r"(?:[a-z]:|/[a-z])" if home.drive else ""  # C: or Git Bash /c; none on POSIX
    names = "".join(r"[\\/]+" + re.escape(part) for part in home.parts[1:])
    # Windows paths ignore case; the lookahead keeps a longer name that only starts with the home's last part.
    return re.sub(drive + names + r"""(?![^\\/\s'"])""", "~", text, flags=re.IGNORECASE)


def _label(value: str) -> str:
    # The label becomes a directory name, so it must not carry path separators.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(f"label must match [A-Za-z0-9_-]+, got {value!r}")
    return value


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: evidence <label> [--out DIR] -- <command...>")
    split = argv.index("--")
    parser = argparse.ArgumentParser(prog="evidence")
    parser.add_argument("label", type=_label)
    parser.add_argument("--out", default="artifacts/env")
    args = parser.parse_args(argv[:split])
    command = argv[split + 1 :]

    # Before the run: its own new directory must not count as a change, and the checkout may move during it.
    git_rev, git_status = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
    started = datetime.now(timezone.utc)
    run_dir = Path(args.out) / f"{started:%Y%m%dT%H%M%S}-{args.label}"
    run_dir.mkdir(parents=True)

    t0 = time.perf_counter()
    with open(run_dir / "stdout.txt", "w") as out, open(run_dir / "stderr.txt", "w") as err:
        exit_code = subprocess.run(command, stdout=out, stderr=err).returncode
    wall_s = time.perf_counter() - t0
    for name in ("stdout.txt", "stderr.txt"):
        path = run_dir / name
        # latin-1 maps each byte to one character, so only the redaction changes the file.
        # shortcut: a non-ASCII user name printed in another encoding is not matched; decode properly if that occurs.
        path.write_bytes(redact_home(path.read_bytes().decode("latin-1")).encode("latin-1"))

    config = Path("config.yaml")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "command": [redact_home(arg) for arg in command],
                "home_redacted": True,
                "exit_code": exit_code,
                "started_utc": started.isoformat(),
                "wall_s": wall_s,
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
                "git_rev": git_rev,
                "git_dirty": None if git_status is None else bool(git_status),  # None: git failed, state unknown
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None,
            },
            indent=2,
        )
    )
    # Echo after the child ends so the caller still sees the output; the files are the record.
    # Bytes, not text: decoding with the locale codec can fail and would replace the child's exit code.
    sys.stdout.buffer.write((run_dir / "stdout.txt").read_bytes())
    sys.stderr.buffer.write((run_dir / "stderr.txt").read_bytes())
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
