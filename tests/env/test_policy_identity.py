"""The files named in config.yaml are the pinned ones, and the model is the 12-motor Go2."""
import hashlib
import subprocess
from pathlib import Path

import mujoco


def test_pinned_files_match_config(config):
    model, policy = config["robot"]["model_asset"], config["robot"]["walking_policy"]
    # The submodule may be checked out with CRLF (core.autocrlf); the pin is the LF file as committed upstream.
    assert hashlib.sha256(Path(model["path"]).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == model["sha256"]
    assert hashlib.sha256(Path(policy["checkpoint"]).read_bytes()).hexdigest() == policy["checkpoint_sha256"]


def test_model_submodule_is_at_pinned_revision(config):
    head = subprocess.run(["git", "-C", "third_party/unitree_mujoco", "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == config["robot"]["model_asset"]["revision"]


def test_model_has_twelve_actuators(config):
    assert mujoco.MjModel.from_xml_path(config["robot"]["model_asset"]["path"]).nu == 12
