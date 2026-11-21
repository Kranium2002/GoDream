# Work log

## 2026-11-21 — Mission revamp

- Replaced the earlier project direction with Learned Judgment: simulated Unitree Go2, borrowed verified walking policy, high-level PFN decision engine, and DODGE as the primary skill.
- Rewrote the four numbered work prompts and both A/B ownership prompts to share the Go2 command/observation contract.
- Added MISSION.md and CLAUDE.md with the no-fallback, real-test, and no-fabricated-results rules.
- This change updates project instructions only. No Go2 policy, MuJoCo integration, learned skill, TabPFN call, or test result is claimed as implemented.

## 2026-11-21 — Person B: contracts, first research round, SNN-only student

- Decision by the project lead: the only student is a spiking neural network; no MLP student; KICK stays stretch. Mission and plan documents updated to match; PREREGISTRATION.md carries a dated amendment and keeps the original hypotheses. See DECISIONS.md.
- Research before code (UltraCode workflow, three agents, no production edits): hosted TabPFN client contract (non-billable part of E7), library contracts for baselines and evaluation, and a read-only look-ahead at the candidate walking policies' command interface. Entries are in experiments/EXPERIMENTS_LOG.md; proofs are under experiments/proofs/.
- Written by hand as the fixed contracts: learned_judgment/engine/types.py, engine/features.py, engine/decide.py, engine/streams.py, skills/base.py, config.py.
- config.yaml: added engine, tabpfn, baselines, students, and evaluation tunables in Person B's sections. tabpfn.project_token_budget is null on purpose, so the teacher refuses any billable request until the lead sets a budget.
- Checks executed: a scratch script (outside the repository) exercised the contract modules and the physics-only decide path against the real config.yaml; exit code 0. The real config correctly refuses to build a skill because the walking policy's command schema is not verified.
- NOT RUN: any Go2 episode (Person A's Go2World does not exist yet), any TabPFN prediction request, any teacher dream, any student training.
- Open for the lead: Prior Labs' General Terms section 10.1.1 names "knowledge distillation of the Contract Software" as grounds for blocking access. Training a student on teacher decisions may fall under it; this needs an explicit answer from Prior Labs before real dreams are used for training.

## 2026-11-21 — Person B: teacher, engine, evaluation built; SNN foundations

- Built by three implementation agents against the hand-written contracts, then re-run by the lead: learned_judgment/teacher.py and teacher_cache.py (TabPFN residual teacher, request cache, cost and usage gate, call log), learned_judgment/engine/ (rows, collect, learn, dream, adapt), learned_judgment/eval/ (baselines, splits, metrics, bootstrap, calibration, harness, report).
- Checks: 15 check scripts under experiments/proofs/B_teacher_offline, B_engine_offline, and B_eval_offline; all exit code 0 on the lead's re-run. They use fixtures to test mechanics only. One is live against the provider's free endpoints: it showed the teacher refusing a request before any prediction call, with account usage unchanged at 40,000 tokens.
- Follow-up edits by the lead: one shared atomic file writer (learned_judgment/fileio.py) with its retry bounds in config.yaml; the evaluation harness writes rows through the engine's row builder; the attempt row's seed is the seed of the world the attempt ran in.
- SNN: PyTorch 2.15.0 (CPU) and snnTorch 1.0.0 added to pyproject.toml and installed (experiment E15). Written by hand: learned_judgment/students/snn_core.py (deployed numpy network and operation counts), encoding.py, energy.py. The numpy network reproduced saved snnTorch outputs exactly on the E15 fixture model: zero logit difference and zero spike-count mismatches on 1,000 inputs.
- Decisions recorded in DECISIONS.md: the SNN is trained inside the PFN dream and is not a copy of the PFN; project token budget 35,000,000.
- Note for Person A: mujoco is installed in the venv but not declared in pyproject.toml, so a plain `uv sync` would uninstall it. Declare it or use `uv sync --inexact`.
- NOT RUN: any Go2 episode, any TabPFN prediction request, any dream from a provider-backed teacher, any SNN training on real dreams.

## 2026-11-21 — Person B: SNN student built; independent review

- Research before code: E15 (PyTorch CPU and snnTorch installed and exercised in throwaway environments, exact neuron dynamics, numpy parity, determinism, latency, operation counts), E15-literature (sources for the energy estimate), E7-terms (Prior Labs published terms). Entries in experiments/EXPERIMENTS_LOG.md.
- Built by implementation agents, re-run by the lead: learned_judgment/students/targets.py, snn_train.py, snn.py (per-skill student), shared_snn.py (one trunk, one head per skill, dream rehearsal, retention matrix), measure.py (latency, spikes, synaptic operations, energy estimate, agreement and dream regret), and engine/train_in_dream.py.
- Found during the build: snnTorch ran its reset step in float32 inside a float64 model; the surrogate is now wrapped so the trained model and the deployed numpy network agree on every spike and decision. Readout logits differ by about 2e-15 between the two libraries; training stops with an error if a spike count or decision differs.
- Checks: experiments/proofs/B_snn_offline, B_snn_shared_offline, B_snn_measure_offline; all exit code 0 on the lead's re-run. They run fixture skills through the real dream and train-in-dream code and test mechanics only.
- Added by hand: a lean single-decision path in snn_core.py and encoding.py (same arithmetic as the batched path) and per-neuron spike recording.
- Independent review of the teacher, engine, and evaluation code (three reviewers, every finding reproduced by execution): 1 blocking, 9 important, 22 minor. Reproductions are under experiments/proofs/B_review/. Fixes are applied in two rounds and each gets a regression check.
- NOT RUN: any Go2 episode, any TabPFN prediction request, any SNN training on a real dream, any measurement on the robot.

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

## 2026-11-21 — Person A: Go2World (Task 4), default preset

- Added `learned_judgment/env/world.py`: `Go2World(config, seed, preset="default")` with `reset(situation, attempt_seed)`, `step(command, duration_s)`, `finish_episode()`, `render_frame()` and `command_schema`, and `make_streams(seed, attempt_seed=None)`; both exported from `learned_judgment.env`. Observations and outcome as in Person B's items 1, 5, 7 and 8 (frames and units in the class docstring and `HANDOFF_PERSON_A.md`). Falls (trunk on the floor or tilt above the new `robot.fall.max_tilt_deg`) are checked after every physics step from the spawn on and latched; `sim_unstable` reads MuJoCo's bad-state warnings. One compressed trace per episode under `artifacts/env/traces/` (ignored by git), hidden values only in its `.hidden.json`. Ball, hidden effects and the other presets: NOT RUN (Tasks 5 and 6).
- `config.yaml`: spawn moved from `policy.py` constants to `robot.walking_policy.spawn` (same values); `robot.acceptance.fall_tilt_deg` replaced by `robot.fall.max_tilt_deg` (same 60); `world.settle_s` 2.0 s (from the Gate 1 traces: speed settles within 1.66 s at 0.9 m/s), `world.trace_dir`, `world.save_traces`, `world.render`, and the default preset's declared payload 0.0 kg and surface `default`.
- `WalkingPolicy`: ONNX inference on one intra-op and one inter-op thread; `reset(on_substep)` hook. Gate 1 re-run with it: every trace bit-identical to `artifacts/env/locomotion` and every measured value unchanged; CPU use fell from 1.39–1.80 to 0.94–0.99 cores (`artifacts/env/20261121T161152-single-thread-gate1`). Parity test unchanged and passing.
- Tests: `tests/env/test_world_api.py` (13) and `tests/env/test_determinism.py` (4), on the real simulator and policy: `artifacts/env/20261121T161057-world`, exit code 0, 17 passed. Each test written for a decision was also run against a version of `world.py` with the bug it guards and failed. Full suite: 67 passed, 1 failed (`test_trots_thirty_seconds[0.3]`, the accepted Gate 1 result).
- Simulator facts re-checked on mujoco 3.11.0 with Unitree's model (velocity frames; bad-state warnings survive MuJoCo's auto-reset): `artifacts/env/20261121T155604-m2-world-facts`, `experiments/EXPERIMENTS_LOG.md` entry M2.
- Cost on this machine (`artifacts/env/20261121T160704-world-cost`, medians): building a `Go2World` 0.12 s (0.70 s the first in a process); `reset()` 0.58 s for 6 s simulated; `step()` real-time factor 7.96 (one 10 s call) and 7.79 (0.02 s calls); `finish_episode()` 0.03 s writing a 20 s trace.

## 2026-11-21 — Person A: the ball, the throw protocol and the DODGE measurements (Task 5)

- `Go2World` throws a physical IHF size 3 handball when a situation holds all six ball keys (frame fixed at the decision instant: origin at the trunk origin, x along the heading, y left, z up); partial keys raise; without them the scene has no ball body. New outcome keys `collision`, `first_collision_time_s`, `min_separation_m`, `recovery_time_s`, `recovered`; observation `ball`; trace arrays `ball_pos_m`, `ball_vel_m_s`. `sample_dodge_situations(config, rng, n)` (`learned_judgment/env/throws.py`, exported) and `Go2World.sample_dodge_situation()`. `learned_judgment/env/calibrate.py`: `calibrate_throws`, `situation_ranges`, `step_response`, `save_dodge_example`.
- `config.yaml`: `ball` contact, inertia and throw ranges; `robot.recovery`; `skills.dodge.situation_ranges`, `episode_duration_s` 3.0 (proposed, to be confirmed by the project owner), `physics.body_half_length_m` 0.293, `body_radius_m` 0.107, `base_height_m` 0.3234, `declared_surfaces` [default].
- Evidence: E6 simulator half (`artifacts/env/20261121T170836-e6-sim-facts`), calibration (final hit rate 0.408 over dev seeds 11–13, 120 throws), ball tests 13 passed (`artifacts/env/20261121T171926-dodge`), step response while trotting (no falls; faster than Person B's 0.05 s / 0.3 s model), saved example `artifacts/env/dodge-example/`. Details: `experiments/EXPERIMENTS_LOG.md`, the three entries of this date after "Single-threaded policy inference".
- Task 4 review items folded in: observations returned as copies, stream key `(4, k)` literal, bad `attempt_seed` and non-number `settle_s` rejected before the open episode is abandoned, a test of a fall before the decision instant, test traces under `tmp_path`.
