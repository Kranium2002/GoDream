# Vendored Go2 assets

Upstream files copied byte-for-byte, unmodified, from the downloads that the E1 proof checked
(`artifacts/env/research/e1_proof.sh`, run `artifacts/env/20261121T144656-e1-proof`). The LICENSE file was
fetched from raw.githubusercontent.com at the same commit. `.gitattributes` (`* -text`) stops Git from
converting line endings, which would change the hashes below.

## efferent@f51e1af

- Source: https://github.com/Eximius-Labs/efferent, `examples/go2_walk/`
- Commit: `f51e1af953afe1016dae9ceeff2b5328fa51781d`
- Licence: Apache-2.0 (`LICENSE`, from the repository root)
- `go2_walk.app` is a zip holding `manifest.yaml` and `policy.onnx`. The ONNX policy is a conversion of rl_sar's
  `policy/go2/robot_lab/policy.pt` (https://github.com/fan-ziqi/rl_sar, commit `376d42c9`, Apache-2.0).

| File | sha256 |
|---|---|
| `go2_walk.app` | `89a5bb7e1ff1cc4cd60f4855bf1dd77bd1b313aeee7c19e984fe7c6e37d74147` |
| `go2.robot.yaml` | `469b610efb7d5de6b2d1f78c0c5a528406a00ed3ec3e6339ec06ce439503a0e6` |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

## Go2 model (not vendored)

The Go2 model is Unitree's `unitree_robots/go2/` from the git submodule `third_party/unitree_mujoco`
(https://github.com/unitreerobotics/unitree_mujoco), pinned at commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`,
BSD-3-Clause under the submodule's own `LICENSE`. These are the same files the E1 proof downloaded and checked.
Initialise it with:

    git submodule update --init --depth 1 third_party/unitree_mujoco

`config.yaml` pins `go2.xml` by the sha256 of its LF form (`2014a3d7…3d4b`), because `core.autocrlf` may check the
submodule out with CRLF line endings.
