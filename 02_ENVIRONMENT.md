# Learned Judgment — Go2 environment and robot integration

Person A owns this work. Read MISSION.md and 01_PLAN.md first. This is the only robot and environment path in scope: a Unitree Go2 model in MuJoCo with one verified borrowed walking policy. Do not substitute another robot, controller, or simulator.

## Research before code

Use an UltraCode research agent before each small implementation slice. Verify primary sources for the exact package/version, model/checkpoint files, licensing, Python compatibility, API signatures, observation order, action scaling, default pose, PD gains, control frequency, and CPU behavior. Require a minimal proof-of-concept command and actual output. If UltraCode is unavailable, use an independent agent or primary-source investigation and record the substitution. Research agents must not edit production code.

Do not trust a checkpoint filename or README claim alone. Confirm that the model asset and config actually exist on this machine. Verify observation and action layouts from the policy source/config. An incorrect observation order can make the dog collapse without a clear exception.

## Phase 1: make the Go2 walk

Try in this order and stop at the first that truly passes:

1. Efferent: verify the package/version and its Go2 example; run examples/go2_walk in MuJoCo. Confirm how its ONNX policy is obtained and which Unitree model asset/config it uses. Timebox: 1 hour.
2. rl_sar: verify the repository revision and both candidate artifacts, policy/go2/robot_lab/policy.pt and policy/go2/himloco/himloco.pt. Use PyTorch on CPU with the exact Unitree MuJoCo Go2 model and matching config. Reproduce observations, action scaling, default pose, PD gains, and control rate from the matching source config. Timebox: 3 hours.
3. Quadruped-PyMPC: verify current MuJoCo/acados requirements and CPU behavior before installation. Timebox: 2 hours.

These are ordered investigation paths for the same Go2 requirement, not permission to claim an unverified path works. If an attempt fails, record the exact failure and repair it before continuing. Do not proceed to learned decisions until a Go2 path passes.

### Locomotion acceptance test

Run the selected real policy in headless MuJoCo and save the run data. The dog must:

- trot for 30 seconds at commanded speeds 0.3, 0.6, and 0.9 m/s
- turn at commanded yaw rates supported by the policy
- stop on command
- stay standing without falling for the full run
- report the measured headless real-time factor, CPU usage, policy identity, simulator/model revision, exact command, and config

The observed commands, speeds, turns, and stand/fall outcomes must come from executed runs. Do not type expected performance as measured output.

## Public simulator interface for Person B

Agree and implement this public API before the engine integration. Keep native policy details behind this adapter:

    Go2World(config, seed)
    Go2World.reset(situation) -> observation
    Go2World.step(command, duration_s) -> observation
    Go2World.finish_episode() -> measured_outcome
    Go2World.render_frame() -> frame or renderer handle

An observation contains the verified policy observation, named robot pose/velocity, contact and fall state, and ball state when a ball is present. Document units, coordinate frame, timestamp, update rate, and missing-value rules. A command contains only fields and ranges verified for the selected policy. Publish command metadata in config.yaml, including the native field mapping and source config reference.

An episode returns measured values only: measured outcome fields for the skill, contacts/collisions, fall state, elapsed time, and any required trace. Do not compute a “successful” result from the intended action or expected path.

## True world and hidden effects

Use MuJoCo and the verified Go2 policy for truth. Implement configurable, independently switchable effects:

- payload 0–3 kg and off-center position; declared label can be wrong by up to 20%
- surface-specific floor friction with per-run variation
- slope up to ±5°
- motor strength 80–100%
- command latency 0.05–0.3 seconds plus per-attempt jitter
- a physical simulated ball for DODGE, with logged initial position/velocity, mass/radius, collision state, and trajectory

World presets in config.yaml: default, heavy_payload, slippery_floor, tired_motors, downhill. Every tunable number goes in config.yaml. Use separate seeded streams for situation sampling, fixed world parameters, ball throws, and per-attempt noise. Fixed config and seed must reproduce simulator outputs.

Only expose observations available to the decision engine. Hidden effects are used to generate the true outcome, not as model features. Record the declared and hidden values separately; do not leak hidden values through row builders.

## Ball-dodge environment

DODGE is the primary skill. The ball and Go2 must interact in the actual MuJoCo scene. Record ball and robot body positions over time, minimum separation, contact/collision, fall, recovery time, and the commands the walking policy actually received.

The ball throw protocol must be reproducible: specify throw origin, direction/speed distribution, ball dimensions/mass, episode start time, and random seed. Include varied but safe trajectories that require different choices. Do not use a prerecorded dodge trace or move the ball based on the desired result.

Person B owns the DODGE SkillSpec and objective. Person A owns the physical ball scene, observation fields, and episode-level collision/fall measurements. Coordinate feature names and units before either side freezes the interface.

## Oracle and calibration

Evaluation-only oracle: search candidate actions supported by the walking policy and execute repeated noise draws in the true Go2 world. Its remaining expected error is the noise floor. Keep oracle sets small (about 100 situations) and parallelize independent worlds only after determinism is verified.

For each skill, run the physics-only baseline and oracle on the dev split. ENV_REPORT.md must include the exact simulator/model/policy versions, config, seed, commands, raw output artifact, measured error, and noise floor. Target physics-only error at least 3× the noise floor; a quick residual fit on 40 actual attempts should close at least half the gap. If not, investigate the simulator and report every adjustment. Never tune to the test set or type target values into the report.

## Tests and handoff

Write and execute tests against the actual implementation and actual MuJoCo Go2 policy. Required checks include fixed-seed reproducibility; command bounds and units; actual standing, stop, turn, and speed tests; ball collision and minimum-separation measurements; hidden-effect switches; and correct row/trace output. Record exact commands, exit codes, dependency/model versions, seeds, and artifact paths. An unrun check is NOT RUN, not PASS. A mock walking policy or fake ball episode does not satisfy system acceptance.

Handoff to Person B: public import path, command schema and supported ranges, observation schema, one actual recorded episode row, raw artifact location, policy/model/config identities, limitations, and ENV_REPORT.md. Do not change a shared field without approval from the integrator.
