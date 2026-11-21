# Decisions

## 2026-11-21 — Learned Judgment scope

The project is a simulated Unitree Go2 learning high-level decisions. Dodging a thrown ball is the main behavior.

## 2026-11-21 — Borrowed locomotion

Use one verified pre-trained Go2 walking policy and keep its weights unchanged. PFN/TabPFN learns policies over commands that the selected walking brain actually supports. Person A owns robot and simulator configuration; Person B owns SkillSpecs and the learning engine.

## 2026-11-21 — No substitute completion

There is no alternate robot, controller, simulator, hand-coded behavior, fake provider, or fake result. If a critical gate fails, stop and resolve it. Reports contain only executed, artifact-backed measurements.

## 2026-11-21 — Evidence before implementation

Verify each policy, package, API, config, version, and license with primary sources and a small executable proof before building dependent production code.

## 2026-11-21 — The only student is an SNN

Decided by the project lead: the only student is a spiking neural network. Per-skill SNN first, then a shared SNN with skill ID and dream rehearsal. No MLP student is built. KICK stays a stretch skill. This replaces the earlier plan in which MLP students were required and the SNN was an optional stretch.

MISSION.md, 01_PLAN.md, 03_METRICS.md, 04_EXPERIMENTS.md, and 06_PERSON_B.md were updated to match. PREREGISTRATION.md keeps the original hypotheses and carries a dated amendment. The MLP trained on actual attempts (baseline B5) is a residual baseline, not a student, and stays in the comparison list.

## 2026-11-21 — The SNN is trained inside the PFN dream; token budget

How the project describes the student, set by the project lead: the PFN is not distilled. The physics guess corrected by the PFN is a dream world, and the SNN is trained inside it. The SNN's training signal is the dreamed outcome of every candidate command scored by the skill's objective. It never learns to reproduce the PFN's predictions.

Prior Labs' General Terms (status June 23, 2026), section 10.1.1, list "knowledge distillation of the Contract Software" among violations that allow blocking access. Section 8.2 says the customer may freely use the Service Results. The saved text is in experiments/proofs/E7_provider_terms/. The project lead decided to proceed: Prior Labs sponsors the track and provided the token allowance for it. This is the lead's decision. No explicit statement from Prior Labs on this point has been recorded.

The project token budget is 35,000,000, the allowance Prior Labs gave for the track (config.yaml tabpfn.project_token_budget). Each request stays capped at the 10,000-token minimum charge unless tabpfn.max_tokens_per_request is changed.

## 2026-11-21 — SkillSpec package location

SkillSpecs live in learned_judgment/skills/, one file per skill, because the committed scaffold created that package. MISSION.md and 06_PERSON_B.md call the same thing "skills/".

## 2026-11-21 — No tunable defaults in Person B code

Person B functions take quantile levels, seeds, probe sizes, model settings, and thresholds as required arguments read from config.yaml. Code keeps only structural rules from the mission, such as the 1–3 action-dimension limit and the [0, 1] range of a probability. Float comparisons against configured values are exact.

## 2026-11-21 — Locomotion pass thresholds fixed before any run

Measured speed within 20% of the commanded value, yaw rate within 30%, and "stopped" means mean planar speed under 0.05 m/s over the last 2 s. Set before any locomotion run. A miss is diagnosed; the threshold is not widened.

## 2026-11-21 — Research and execution method for Person A

The "UltraCode" research runs as multi-agent workflows: read-only researchers in parallel, then one verifier per researcher that re-opens every cited source. Research agents do not run repository code or load policy checkpoints. Implementation is done by Opus 5.5 subagents, one implementer and one reviewer per task.

## 2026-11-21 — Evidence recording

Every Person A acceptance claim is backed by a run recorded through `learned_judgment/env/evidence.py` under `artifacts/env/`. `ENV_REPORT.md` is generated from those files only.

## 2026-11-21 — Thrown ball is an IHF size 3 handball

Mass 0.450 kg and radius 0.0939 m, the mid-range of IHF Rule 3:2 (425–475 g, circumference 58–60 cm), verified in experiment E6. Chosen before any throw was simulated. Fallback if dev runs show every hit topples the dog: FIVB volleyball (0.270 kg, radius 0.1050 m). Contact parameters, ball friction and inertia are not set by any rule and remain to be stated by the ball-scene work.

## 2026-11-21 — Walking stack: Efferent go2_walk on Unitree's Go2 model

Efferent is the first path in the mandated order and its proof passed (experiment E1 proof run, `artifacts/env/20261121T144656-e1-proof`). Selected: the `go2_walk.app` policy and `go2.robot.yaml` from `Eximius-Labs/efferent` at `f51e1af953afe1016dae9ceeff2b5328fa51781d` (Apache-2.0; the ONNX is converted from rl_sar `policy/go2/robot_lab/policy.pt` at `376d42c9`), on Unitree's `go2.xml` from `unitreerobotics/unitree_mujoco` at `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` (BSD-3-Clause). Pinned packages: `efferent==0.5.0`, `mujoco==3.11.0`, `onnxruntime==1.32.0`. The Efferent files are vendored under `learned_judgment/env/assets/`; the model comes from the git submodule `third_party/unitree_mujoco` at that commit. Both are pinned by sha256 in `config.yaml`. rl_sar (E2) stays the fallback; its proof is not run.

## 2026-11-21 — Command schema `verified` means proven fields and ranges, not tracking

The integrator's rule for Task 3: `robot.walking_policy.status` becomes `verified` once every command field is proven at both ends of its range by a recorded run (no fall, response with the commanded sign); it makes no claim about speed-tracking accuracy, which stays with Gate 1 (still failing at 0.3 m/s). Applied on E4 (`artifacts/env/commands/commands.json`): `vel_x`, `vel_y` and `yaw_rate` all proven at −1 and 1, nothing narrowed.

## 2026-11-21 — Gate 1 result at 0.3 m/s accepted as measured

The locomotion gate fails on one of its seven conditions: 0.3 m/s commanded gave 0.2256 m/s, outside the 20% tolerance (0.24 to 0.36). The parity test shows our adapter reproduces Efferent's own runner exactly, and the review of that work recomputed every gate value from the saved traces, so the shortfall is the borrowed policy's behaviour and not our wiring. Person A decided to accept the result as measured and continue. The tolerance stays at 20%, nothing is tuned, and the gate row stays FAIL for that condition in `ENV_REPORT.md`.

This sets aside, for this one condition, the rule to stop at a failed gate and to move to the next locomotion path. rl_sar's robot_lab checkpoint is the same policy (experiment E2), so it would not change the result; rl_sar himloco and Quadruped-PyMPC were not tried. The integrator can reverse this.

Consequences for the skills: low forward commands under-deliver by about a quarter, and with zero yaw commanded the dog turns anticlockwise at 0.05 to 0.14 rad/s.

## 2026-11-21 — Two levels of seed, in Person B's proposed form

`Go2World(config, seed, preset)` fixes the hidden world for the life of the object. `reset(situation, attempt_seed)` fixes one episode's noise, drawn from `(seed, attempt_seed)`; leaving `attempt_seed` out uses the episode counter. This replaces the `noise_key` situation key of the plan (HANDOFF_PERSON_B.md item 8).

## 2026-11-21 — Per-attempt random streams on their own SeedSequence branch

`make_streams(seed)` spawns the four streams (`situation`, `world`, `ball`, `noise`) from `numpy.random.SeedSequence(seed)`; the per-attempt streams of `reset(attempt_seed=k)` spawn from `SeedSequence(seed, spawn_key=(4, k))`. Seeding with the plain list `[seed, k]` was rejected: NumPy pads short entropy with zeros, so `[seed, 0]` gives the same streams as `seed` and attempt 0 would repeat the world's own draws (`tests/env/test_determinism.py` checks they differ). Changing this rule changes every recorded attempt's noise, so it is fixed from now on.

## 2026-11-21 — Ranges of the hidden world that are designed, not measured

These are properties of the simulated world we design, not of a real Go2, so each is chosen from something we can point to (Task 6; values in `config.yaml` `world.hidden_effects`):

- **Floor types** `default` and `slippery`, as the sliding friction of every foot-floor contact. `default` is 0.4, the foot geoms' own friction in Unitree's `go2.xml` (class `foot`): their contact priority 1 overrides the floor plane's, so the floor type is applied to the four foot geoms `FL`, `FR`, `RL`, `RR` (proof: `artifacts/env/20261121T165353-e5-effect-facts`). `slippery` is 0.1, from a dev sweep of 0.4 down to 0.02 (`artifacts/env/friction-sweep/sweep.json`): the dog fell at none of them, so the bound is the trot instead: the lowest friction whose 0.6 m/s trot stays within Gate 1's 20% speed tolerance is 0.06 (0.4995 m/s; 0.05 gives 0.4726). 0.1 keeps every run at least 1.5 times that.
- **Per-run friction variation** ±10% of the type's value, drawn once per world: 0.36–0.44 and 0.09–0.11 never overlap, and slippery stays above 0.06.
- **Payload offset range** the top face of the trunk's collision box (`go2.xml` line 74): x within ±0.1881 m, y within ±0.04675 m of `base_link`'s origin. The payload is a 15 × 9 × 8 cm box resting on that face that adds mass and touches nothing.
- **Latency jitter** one control period: each attempt's latency is the nominal one plus −1, 0 or +1 period (0.02 s), uniformly. Every preset's nominal latency is 0.08 s, the shortest whole number of periods whose jittered value stays inside the mission's 0.05–0.3 s.

The preset values (`world.presets`) are a proposal for the integrator and Person B to confirm; changing them is a config edit.

## 2026-11-21 — How the hidden effects are applied

- **Declared values** come from the preset whatever is switched on: `declared_payload_kg` is the preset's payload mass × (1 + an error drawn per world within ±0.2), `declared_surface` its floor type. The `enabled` switches change only what the simulator applies.
- **Motor strength** scales the PD gains and feed-forward torque after Efferent's torque clamp, so every joint torque and the torque limit scale together, from the spawn hold and soft start on: a motor that is tired is tired during the spawn too.
- **Slope** tilts gravity instead of the floor (the plan's shortcut, exact for an infinite flat floor). So that the policy is not blind to it, the trunk orientation it reads (from which Efferent computes projected gravity) is turned to be relative to true gravity, as an IMU on a real slope would sense it. Observations stay in the floor frame. A consequence: `policy_obs[3:6]` shows the slope, as a real robot's sensors would.
- **Latency jitter** is drawn from a child stream of the attempt's `noise` stream (`Generator.spawn`), not from the stream itself, so `attempt_streams` still reach their user untouched (`tests/env/test_determinism.py` checks this unmodified).
