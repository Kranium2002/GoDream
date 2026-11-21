# Person A — Go2 robot, simulator, and world configuration

You own the real Go2/MuJoCo foundation for Learned Judgment. Read MISSION.md, 01_PLAN.md, and 02_ENVIRONMENT.md before working. The main product behavior is a Go2 dodging a thrown ball. No alternate robot, controller, or project is allowed.

## Read and research first

Read all four numbered prompts. Before each implementation slice, dispatch an UltraCode research agent with a narrow question about the exact current policy, simulator, library, config, or model asset. Require primary sources, exact versions/APIs, a proof-of-concept command and observed output, limitations, and a pass/fail recommendation. Agents research only; they do not write production code. If UltraCode is unavailable, use an independent research agent or primary-source verification and record the substitution.

Do not assume a checkpoint, model file, config, command, or package exists or matches. Verify its exact repository/revision and run a minimal proof before integrating it. Stop on a failed or unsupported gate and resolve the real Go2 path.

## Owned files

- learned_judgment/env/** — MuJoCo Go2 model adapter, verified walking policy adapter, world reset/step/episode outcomes, ball, hidden effects, and trace.
- config.yaml — robot/model/policy identity, native command mapping, simulator settings, presets, and every tunable value.
- app/** — dashboard hooks that use the real Go2 environment.
- ENV_REPORT.md — actual calibration and acceptance evidence.

Person A owns the dog robot configuration. Person B owns SkillSpecs, the generic engine, teacher, students, and evaluation. Do not edit those files. Do not edit the four numbered prompts or shared setup files without the integrator.

## Phase 1 — actual locomotion proof

Follow the order in 02_ENVIRONMENT.md: Efferent Go2 example; rl_sar checkpoints with their exact configs and observation/action mapping; then Quadruped-PyMPC investigation. Respect the timeboxes, but a timebox ending does not count as a pass. Diagnose actual failures and continue on the required Go2 path.

Run the real policy in headless MuJoCo. Record and prove 30-second trots at 0.3, 0.6, and 0.9 m/s, supported yaw turns, stopping, standing, and real-time factor. Save the exact command, exit code, stdout/stderr, dependency versions, simulator/model/policy revision, config, and raw run artifact. Do not hardcode a pass report.

## Stable interface for Person B

Implement the public API in 01_PLAN.md and 02_ENVIRONMENT.md:

    Go2World(config, seed)
    Go2World.reset(situation) -> observation
    Go2World.step(command, duration_s) -> observation
    Go2World.finish_episode() -> measured_outcome

Expose only observations the policy genuinely provides. Document coordinate frame, units, update rate, timestamp, missing fields, contacts, fall state, and ball state. Publish the exact supported command names, native field mapping, units, ranges, and control period. Reject unsupported commands with clear errors rather than silently clipping or ignoring them.

Use deterministic but independent random streams for situation samples, fixed world variation, ball throws, and per-episode noise. A fixed config and seed must reproduce the same measured world outputs.

## Physical Go2 world and DODGE support

Implement MuJoCo ground truth with independently switchable hidden effects: payload 0–3 kg/off-center position and up to 20% label error; floor friction with per-run variation; slope ±5°; motor strength 80–100%; command latency 0.05–0.3 seconds plus jitter.

The DODGE ball is a real simulated MuJoCo object with configured mass/radius and physical collisions. Log its initial state and full trajectory. Use reproducible throw distributions that include both safe misses and collisions. Never move the ball based on the chosen action or desired outcome. Person B owns the DODGE SkillSpec/objective; you own ball physics, observations, and measured collision/fall/separation outcomes.

Add presets default, heavy_payload, slippery_floor, tired_motors, and downhill. Calibration uses dev runs only. ENV_REPORT.md derives all values from saved episode outputs and records simulator/model/policy versions, configs, seeds, commands, and raw artifact paths.

## Acceptance criteria

Mark each item PASS only with executed evidence; otherwise mark OPEN / NOT RUN and name the blocker:

- Go2 policy passes every locomotion condition in Phase 1.
- Public reset/step/finish_episode API accepts verified commands and returns observations/outcomes with documented types, units, and timing.
- Unknown commands fail explicitly. No unsupported command is silently mapped to a different behavior.
- Fixed-seed runs reproduce robot, ball, contact, and outcome data.
- Actual DODGE episodes log ball and robot trajectories, minimum separation, collision, fall, and recovery measurements.
- Each hidden effect can be switched independently and its effect is measured in actual simulator runs.
- All four world presets load from config.yaml.
- ENV_REPORT.md reports measured physics-only error and oracle noise floor by skill, with raw data locations and no test-set tuning.
- A fresh Python 3.13 setup resolves declared packages and executes documented Go2 acceptance commands without undocumented edits.

## No fake tests or results

Run the checks against the actual selected Go2 MuJoCo model and walking policy. A mock policy, hand-authored trace, screenshot, kinematic animation, or fixture cannot pass a system test. Pure utility fixtures may only validate the utility and must not appear in performance reports. Record command, exit code, package/model versions, seed/config, and output path. Unexecuted tests are NOT RUN, never PASS. Generate ENV_REPORT.md from saved data; do not type target values as observations.

## Handoff

Give Person B the public import path, command/observation schema, supported ranges, one actual saved Go2 episode, raw artifact location, policy and simulator identities, test evidence, and current limitations. Do not change a shared field without agreement with the integrator and Person B.
