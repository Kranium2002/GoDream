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

    started = datetime.now(timezone.utc)
    run_dir = Path(args.out) / f"{started:%Y%m%dT%H%M%S}-{args.label}"
    run_dir.mkdir(parents=True)

    t0 = time.perf_counter()
    with open(run_dir / "stdout.txt", "w") as out, open(run_dir / "stderr.txt", "w") as err:
        exit_code = subprocess.run(command, stdout=out, stderr=err).returncode
    wall_s = time.perf_counter() - t0

    config = Path("config.yaml")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "command": command,
                "exit_code": exit_code,
                "started_utc": started.isoformat(),
                "wall_s": wall_s,
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
                "git_rev": _git("rev-parse", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None,
            },
            indent=2,
        )
    )
    # Echo after the child ends so the caller still sees the output; the files are the record.
    sys.stdout.write((run_dir / "stdout.txt").read_text())
    sys.stderr.write((run_dir / "stderr.txt").read_text())
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
