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

## 2026-11-21 — Person A: locomotion gate (Gate 1) run, FAIL on 0.3 m/s

- Added the flat scene (`learned_judgment/env/scene.py`), the stepwise adapter over Efferent's own components (`policy.py`) and the gate runs and CLI (`acceptance.py`). The adapter reproduces Efferent's `Runner` bit for bit (`artifacts/env/20261121T152519-policy-parity`, exit 0).
- Gate run: exit code 1, `artifacts/env/20261121T152237-locomotion`, results in `artifacts/env/locomotion/locomotion.json`. trot 0.3 m/s FAIL (mean 0.2256, tolerance ±0.06); trot 0.6 PASS (0.5430), trot 0.9 PASS (0.8596), turns PASS (0.4976 and −0.5066 rad/s for ±0.5), stop PASS (final 0.0130 m/s), stand PASS; no falls. Real-time factor 7.80 over all seven episodes. At zero yaw command the dog drifts counter-clockwise by 0.05–0.14 rad/s.
- Nothing was tuned or loosened. Details: `experiments/EXPERIMENTS_LOG.md`, "Gate 1 — locomotion acceptance". `robot.walking_policy.status` unchanged.

## 2026-11-21 — Person A: command schema published (E4), status verified

- Added `prove_commands` (`acceptance.py`) and `validate_command` / `CommandError` (`learned_judgment/env/commands.py`, exported from `learned_judgment.env`): strict, every field required, never clips.
- Proof run: exit code 0, `artifacts/env/20261121T153523-commands-proof`, results in `artifacts/env/commands/commands.json`. `vel_x`, `vel_y` (m/s) and `yaw_rate` (rad/s) each proven at −1 and 1 (no fall, commanded sign): measured −0.8880 / 0.9243, −1.0349 / 0.8939, −0.9555 / 0.9687. Nothing narrowed.
- `config.yaml` `robot.walking_policy.command_schema` published in Person B's layout with `update_period_s` 0.02; `status: verified`, meaning the fields and ranges are proven, not that speeds are tracked (Gate 1 still fails at 0.3 m/s). Tests: `artifacts/env/20261121T153641-commands`, exit 0, 6 passed. Details: `experiments/EXPERIMENTS_LOG.md`, "E4 — command inventory"; handoff: `HANDOFF_PERSON_A.md` item 3.

## 2026-11-21 — Person A: Gate 1 reviewed; 0.3 m/s result accepted as measured

- Independent review of the walking-adapter work: all 21 gate values recomputed from the seven saved traces, each equal to `artifacts/env/locomotion/locomotion.json`. The 0.3 m/s miss holds in every 5 s window (0.2205 to 0.2271 m/s).
- Decision recorded in `DECISIONS.md`, "Gate 1 result at 0.3 m/s accepted as measured". No tolerance changed, nothing tuned.
- Moved to the `Go2World` work from that review: spawn settings go from code into `config.yaml`, and fall detection must cover the settling period.
- `HANDOFF_PERSON_A.md` updated: gate results, real-time factor 7.80, and the two-level seed in Person B's form (their new item 8).
