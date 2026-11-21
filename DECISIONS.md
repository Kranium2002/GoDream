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

## 2026-11-21 — Ball scene and DODGE measurements: choices the Task 5 brief left open

Made by the Task 5 implementer; the integrator can reverse any of them.

- **Throw parameterisation.** Six uniform draws per throw from `config.yaml` `ball`: the dog's forward speed, the release point's horizontal distance, its bearing from the heading and its height above the trunk origin, the ball's horizontal speed, and an aim error along the dog's y. The thrower aims at the trunk origin's height where a dog keeping its commanded speed will be (the actual speed is lower at 0.3 m/s, Gate 1); the flight time follows from the horizontal speed. Throws come from in front (bearing within ±60°), where a sideways step moves the dog off the ball's line. The brief's starting distance was 3–5 m; 3.5–5.0 m with a horizontal speed of 3.3–4.8 m/s keeps every flight inside the agreed 0.6–1.6 s (`tests/env/test_dodge_functions.py` checks it).
- **Ball contact and inertia** (no rule sets them): thin-shell inertia, MuJoCo's default friction and `condim`, `priority` 2 so the ball's own contact parameters apply to all its contacts, and `solref` [0.04, 0.1] from a drop test against IHF Rule 3:2's rebound (experiment E6, simulator half).
- **The ball before the decision instant.** The scene with the ball has the ball as a free body after the robot, 2 m up. `Go2World` switches its collisions off during reset's spawn, settle and trot and on when it places the ball, so it cannot meet the dog before the throw however long `settle_s` is.
- **Recovery** is judged at the 50 Hz observations from the decision instant on, on trunk tilt and tilt rate (the horizontal part of the angular velocity, so turning does not count). Limits: the largest value over all 13 saved commanded walking runs plus 25%; `hold_s` 0.6 s, one gait period at the slowest trot. `recovery_time_s` ends at the first sample of the calm period that then lasts `hold_s`.
- **Trunk capsule** for `skills.dodge.physics`: centred on the trunk origin along its x axis (the frame of Person B's physics guess), the smallest radius that can hold the three trunk collision geoms, then the shortest segment. It reaches 0.21 m behind the trunk's back end, because the geoms are not centred on the origin.
- **Outcome keys in every episode.** `collision`, `first_collision_time_s`, `min_separation_m`, `recovery_time_s` and `recovered` are in every outcome; without a ball `collision` is 0 and the other two ball keys are `None`.
- **Separation is exact at every physics step** (23 `mj_geomDistance` calls, distances capped at the smallest so far, none after a contact). The ball's cost in real-time factor is measured (experiment E6, simulator half) and no cheaper bound was added.
- **`skills.dodge.situation_ranges`** come from 100,000 throws of the final protocol (`calibrate.situation_ranges`, seed 0), widened by 2% of each span and rounded outward to 0.01 m or m/s; an independent 20,000-throw sample is tested to stay inside.
- **A dog with and without the ball is not bit-identical before the ball arrives**: MuJoCo groups its sums over all degrees of freedom differently with the ball's six added, and the float32 policy grows that to about 2e-5 by the decision instant. The test allows 1e-4.
- **Separation bound.** After measuring that exact separation against all 23 geoms at every step cost about a third of the CPU real-time factor, `Go2World` measures exactly only geoms whose bounding-sphere lower bound could beat the smallest separation so far or touch; all 40 outcomes of dev seed 11 are equal to the exact-every-step run (experiment E6, simulator half).

## 2026-11-21 — Throw aim error widened from ±0.5 m to ±0.7 m

`ball.throw_aim_error_range_m`, the only throw range changed, from `calibrate_throws` output on dev seeds 11, 12 and 13 (40 throws each, dog holding its initial command for 3.0 s). Before: hit rate 0.725 / 0.550 / 0.550, 73 of 120 (0.608), seed 11 above the 30–70% target. After: 0.425 / 0.375 / 0.425, 49 of 120 (0.408). Records: `artifacts/env/20261121T165323-dodge-calibration-1`, `artifacts/env/20261121T170357-dodge-calibration-2`.

## 2026-11-21 — A no-ball development skill rehearses the chain on the real simulator (Person B)

DODGE cannot run until Person A's simulated ball exists. Half of the DODGE physics guess is the formula for the dog's sideways travel after a sideways command. A development skill, `sidestep_reach` ("reach a given sideways displacement"), isolates that half so the whole chain (attempts, residual model, dreams, SNN trained in the dream, evaluation) runs on the real `Go2World` before the ball lands. It lives under `experiments/proofs/B_sidestep_dev/`, is not registered in `learned_judgment/skills/`, and its settings sit under `dev_checks.sidestep_reach` in `config.yaml`. Its numbers are development measurements: they enter no report, table or claim. It is part of the DODGE work, not one of the later skills, so the rule that no later skill starts before the DODGE gate is resolved is unchanged.

## 2026-11-21 — One pipeline module is the interface to Person B's code (Person B)

`learned_judgment/pipeline.py` composes the engine's steps into one run per skill and world, with a manifest per run. Person A's local web interface and the command line both call it; neither assembles engine calls itself. A step that would send a billed TabPFN request runs only when the caller passes a confirmation for the quoted cost; the default refuses. The held-out test split is reachable only through an explicit final-evaluation argument.

## 2026-11-21 — The first real TabPFN request was a contract proof on fixture data (Person B)

The offline checks could not show that the real response has the layout the teacher parses. One billed request (10,000 tokens, the minimum charge) on fixture data shaped like one DODGE decision settled that before any simulator attempts existed (experiment E7-settle). It is a proof about the service's contract and the project's request path. It is not a result about the robot or about any model's quality.

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
