# Person A handoff and answers to Person B

Branch `person-a/go2-env`. Status on 2026-11-21: the walking stack is selected and pinned and its upstream example ran on the target laptop. `Go2World` does not exist yet. The locomotion gate, the command schema, the ball scene and the hidden effects are NOT RUN. Plan: `docs/superpowers/plans/2026-11-21-person-a-go2-env.md`.

## What is measured so far

- **Walking stack.** Efferent 0.5.0 with its `go2_walk` policy (the rl_sar robot_lab Go2 policy as ONNX) on Unitree's `go2.xml` from the `third_party/unitree_mujoco` submodule, with `mujoco` 3.11.0 and `onnxruntime` 1.32.0. Identities and hashes are in `config.yaml` under `robot`.
- **Proof run.** The upstream example walked forward at 0.4 m/s commanded for 400 policy steps: final x 1.88 m, maximum tilt 14.7°, no estop, exit code 0. Evidence: `artifacts/env/20261121T144656-e1-proof/`, log entry "E1 (proof run)". This is one speed in Unitree's scene, which has bars and stairs on the +x path. It is not the locomotion gate.
- **Not measured.** Speed tracking at 0.3, 0.6 and 0.9 m/s, turning, stopping, standing, lateral motion, command ranges, real-time factor with the policy running.

## Setup

    git submodule update --init --depth 1 third_party/unitree_mujoco
    uv sync
    uv run pytest -q

## Answers to HANDOFF_PERSON_B.md

Names follow your proposals wherever you made one. None of this is implemented yet; the task that delivers each item is named.

1. **World class.** `from learned_judgment.env import Go2World`, with `reset(situation) -> observation`, `step(command, duration_s) -> observation`, `finish_episode() -> measured_outcome`. One fresh world per episode with a per-episode seed works. Task 4.
2. **Preset.** A third constructor argument: `Go2World(config, seed, preset="default")`. Presets: `default`, `heavy_payload`, `slippery_floor`, `tired_motors`, `downhill`. Tasks 4 and 6.
3. **Command schema.** In your loader's layout under `robot.walking_policy.command_schema`: `update_period_s` and `fields.<name>.{unit, min, max, description}`, plus `native_index` and `proven_by` per field. `status` stays `not_verified` until every field is proven at both ends of its range by a recorded run; then it becomes `verified`. Expected fields, from the policy manifest: `vel_x` and `vel_y` in m/s, `yaw_rate` in rad/s, 50 Hz. Ranges are not established: upstream documents only 0.4 m/s forward. Task 3.
4. **DODGE situation.** Your seven keys, accepted by `reset`: `ball_rel_x_m`, `ball_rel_y_m`, `ball_rel_z_m`, `ball_vel_x_m_s`, `ball_vel_y_m_s`, `ball_vel_z_m_s`, `dog_forward_speed_m_s`. Frame: the dog's heading frame (x forward, y left, z up, origin at the base) at the decision instant. `Go2World.sample_dodge_situation()` draws a situation with those keys from a physical throw model, and `config.yaml` will list each key's minimum and maximum. An independent box over the six ball values would mostly miss the dog, so sample through that method or tell us if you need another form. Task 5.
5. **Declared world values.** `declared_payload_kg` and `declared_surface` in every observation and in the outcome. True hidden values are written only to a sidecar file next to the trace and never appear in an observation or outcome. Task 6.
6. **DODGE outcome.** `min_separation_m` (never below 0, 0 on contact), `collision` (0 or 1), `fall` (0 or 1), `recovery_time_s`. `recovery_time_s` is always a number: 0.0 if the dog is never destabilised, otherwise the time until it is stable again, or until the episode ends, with `recovered` (0 or 1) beside it. Other keys: `fall_time_s`, `first_collision_time_s`, `sim_unstable`, `elapsed_s`, `commands_received`, `trace_path`. Task 5.
7. **Episode timing.** The decision instant is the moment `reset` returns (`t_s` 0), after the dog has trotted at `dog_forward_speed_m_s` for the settling time. The ball is placed at that instant, already in flight, and is never touched again. `step` may be called any number of times with different commands; a duration is rounded to whole control periods and the returned `t_s` shows the time actually simulated. The episode ends when you call `finish_episode`. Your nominal, dodge, nominal sequence is supported. Task 4.
8. **Saved episode and real-time factor.** Not available yet. The real-time factor comes from Task 2, the saved DODGE episode from Task 5.
9. **`mujoco` in pyproject.toml.** Declared on this branch as `mujoco==3.11.0`, with `efferent==0.5.0` and `onnxruntime==1.32.0`. Your branch adds `torch`, `snntorch`, `scipy` and `threadpoolctl` in the same block, so the two branches will conflict there and in `config.yaml`, `DECISIONS.md`, `LOG.md` and `experiments/EXPERIMENTS_LOG.md`.

## Ball

IHF size 3 handball: 0.450 kg, radius 0.0939 m (`config.yaml`, `ball`; log entry E6). Contact parameters, friction and inertia are not set by any rule and are stated by the ball-scene work.

## Your look-ahead findings

Confirmed by our research and its second-agent check: the Efferent and rl_sar robot_lab checkpoints are the same policy; forward, lateral and yaw-rate commands at 50 Hz; Unitree's scene has obstacles from about x = 1.1 m; foot geoms have contact priority 1, so the floor-friction preset changes the foot geoms. We will detect ball contact by robot body ids, as you suggest.

## What we need from Person B

- The DODGE SkillSpec (`physics_guess`, `episode`, candidate actions) to run the per-skill oracle and fill the physics-only error and noise-floor rows of `ENV_REPORT.md`. Until then those rows read NOT RUN.
- A reply if any name or frame above does not fit your code.
