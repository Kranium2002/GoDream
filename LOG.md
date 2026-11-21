# Work log

## 2026-11-21 — Mission revamp

- Replaced the earlier project direction with Learned Judgment: simulated Unitree Go2, borrowed verified walking policy, high-level PFN decision engine, and DODGE as the primary skill.
- Rewrote the four numbered work prompts and both A/B ownership prompts to share the Go2 command/observation contract.
- Added MISSION.md and CLAUDE.md with the no-fallback, real-test, and no-fabricated-results rules.
- This change updates project instructions only. No Go2 policy, MuJoCo integration, learned skill, TabPFN call, or test result is claimed as implemented.

## 2026-11-21 — Person A: environment work started

- Local branch `person-a/go2-env`. Python 3.13.18 installed with uv; the declared packages resolve (`uv sync`, exit 0); `uv.lock` added; `pytest` declared as a dev dependency.
- Execution plan: `docs/superpowers/plans/2026-11-21-person-a-go2-env.md`.
- Added `learned_judgment/env/evidence.py`: runs a command and saves its command line, exit code, output and package versions under `artifacts/env/`. Its 3 utility tests pass (`uv run pytest tests/env/test_evidence.py`). This is tooling only, not simulator or policy evidence.
- Research workflow W1 (Efferent, rl_sar, MuJoCo on Windows) launched; no result yet. No Go2 policy is selected or verified. Locomotion gate: NOT RUN.

## 2026-11-21 — Person A: walking stack selected and pinned

- E1 proof run passed (exit 0, `E1 PROOF: PASS`, `artifacts/env/20261121T144656-e1-proof`): the Efferent Go2 policy walked forward headless for 400 steps (final x 1.88 m, max tilt 14.7°). Efferent selected; rl_sar remains the fallback.
- Vendored the checked Efferent policy and robot config with their licence under `learned_judgment/env/assets/` (`* -text` keeps the hashes). The Unitree Go2 model comes from the integrator's submodule `third_party/unitree_mujoco` (merged from `origin/main`) at the commit the proof checked. Added `efferent==0.5.0`, `mujoco==3.11.0`, `onnxruntime==1.32.0`. Filled `runtime.timestep_s`, `robot.model_asset` and `robot.walking_policy` in `config.yaml` from these files.
- The evidence recorder now writes `~` for the home directory in its files (`"home_redacted": true`).
- Added `load_config`, the `config` fixture, the `system` marker and `tests/env/test_policy_identity.py` (pinned hashes, 12 actuators). Locomotion gate (H+3): NOT RUN.
