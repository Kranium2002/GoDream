# Person A handoff and answers to Person B

Branch `person-a/go2-env`. Status on 2026-11-21, evening: the walking stack is selected and pinned, the locomotion gate has been run (6 of 7 conditions pass, 0.3 m/s fails) and the command schema is published with `status: verified`. `Go2World` is being built now. The ball scene, the hidden effects and the presets are NOT RUN. Plan: `docs/superpowers/plans/2026-11-21-person-a-go2-env.md`.

## What is measured so far

- **Walking stack.** Efferent 0.5.0 with its `go2_walk` policy (the rl_sar robot_lab Go2 policy as ONNX) on Unitree's `go2.xml` from the `third_party/unitree_mujoco` submodule, with `mujoco` 3.11.0 and `onnxruntime` 1.32.0. Identities and hashes are in `config.yaml` under `robot`.
- **Proof run.** The upstream example walked forward at 0.4 m/s commanded for 400 policy steps: final x 1.88 m, maximum tilt 14.7°, no estop, exit code 0. Evidence: `artifacts/env/20261121T144656-e1-proof/`, log entry "E1 (proof run)". This is one speed in Unitree's scene, which has bars and stairs on the +x path. It is not the locomotion gate.
- **Locomotion gate (Gate 1): FAIL on one condition.** Flat floor, 30 s trots: 0.3 / 0.6 / 0.9 m/s commanded gave 0.2256 / 0.5430 / 0.8596 m/s. 0.3 m/s is outside the 20% tolerance (0.24 to 0.36). Turning at ±0.5 rad/s gave +0.4976 / −0.5066 rad/s, stopping ended at 0.0130 m/s, and the dog stood for 30 s. No falls. Evidence: `artifacts/env/20261121T152237-locomotion/`, `artifacts/env/locomotion/locomotion.json`. Person A accepted the result as measured (`DECISIONS.md`): no tolerance was changed and nothing was tuned.
- **Two things to plan around.** Low forward commands under-deliver by about a quarter. With zero yaw commanded the dog still turns anticlockwise at 0.05 to 0.14 rad/s, so at low speed it walks a slow circle.
- **Real-time factor.** 7.80 with the policy running in one process (178 s simulated in 22.8 s of wall time).
- **Command schema.** Item 3 below.
- **Not measured.** Anything with the ball, the hidden effects or the presets.

## Setup

    git submodule update --init --depth 1 third_party/unitree_mujoco
    uv sync
    uv run pytest -q

## Answers to HANDOFF_PERSON_B.md

Names follow your proposals wherever you made one. Item 3 is delivered; for the others the task that delivers each item is named. Numbers follow your current list (ten items).

1. **World class.** `from learned_judgment.env import Go2World`, with `reset(situation, attempt_seed) -> observation`, `step(command, duration_s) -> observation`, `finish_episode() -> measured_outcome`. One fresh world per episode with a per-episode seed works. Task 4.
2. **Preset.** A third constructor argument: `Go2World(config, seed, preset="default")`. Presets: `default`, `heavy_payload`, `slippery_floor`, `tired_motors`, `downhill`. Tasks 4 and 6.
3. **Command schema. Published; `robot.walking_policy.status` is `verified`.** In your loader's layout under `robot.walking_policy.command_schema` in `config.yaml` (your `CommandSchema.from_config` at `origin/person-b` loads it): `update_period_s: 0.02` (one command per 50 Hz policy step), and three fields in the policy's native order, each with `unit`, `min`, `max`, `description`, plus `native_index`, `proven_by`, `measured_at_min` and `measured_at_max`:

    | Field | Unit | Range | Measured at min / max | Meaning |
    |---|---|---|---|---|
    | `vel_x` | m/s | −1.0 to 1.0 | −0.8880 / 0.9243 | forward (+) or backward (−) along the dog's own heading |
    | `vel_y` | m/s | −1.0 to 1.0 | −1.0349 / 0.8939 | left (+) or right (−) in the dog's own heading frame |
    | `yaw_rate` | rad/s | −1.0 to 1.0 | −0.9555 / 0.9687 | anticlockwise (+, turning left) seen from above |

   Each range end was proven by one recorded 10 s run on a flat floor, that field alone, the others 0: the dog did not fall and moved with the commanded sign (log entry "E4 — command inventory", `artifacts/env/commands/commands.json`). `verified` means exactly that and nothing about tracking accuracy. Limits you should plan around: forward tracking on Gate 1 was 0.2256 / 0.5430 / 0.8596 m/s for 0.3 / 0.6 / 0.9 commanded (0.3 m/s misses the gate's 20% tolerance and Gate 1 still fails on it); at zero yaw command the dog turns anticlockwise at 0.05–0.14 rad/s; values between the ends, combined commands and the delay after a command change are not measured; in robot_lab training planar commands with norm ≤ 0.2 m/s were zeroed. `validate_command(command, schema)` (from `learned_judgment.env`) returns the native vector or raises `CommandError` naming the field: every field is required, nothing is clipped.
4. **DODGE situation.** Your seven keys, accepted by `reset`: `ball_rel_x_m`, `ball_rel_y_m`, `ball_rel_z_m`, `ball_vel_x_m_s`, `ball_vel_y_m_s`, `ball_vel_z_m_s`, `dog_forward_speed_m_s`. Frame: the dog's heading frame (x forward, y left, z up, origin at the base) at the decision instant. `Go2World.sample_dodge_situation()` draws a situation with those keys from a physical throw model, and `config.yaml` will list each key's minimum and maximum. An independent box over the six ball values would mostly miss the dog, so sample through that method or tell us if you need another form. Task 5.
5. **Declared world values.** `declared_payload_kg` and `declared_surface` in every observation and in the outcome. True hidden values are written only to a sidecar file next to the trace and never appear in an observation or outcome. Task 6.
6. **DODGE outcome.** `min_separation_m` (never below 0, 0 on contact), `collision` (0 or 1), `fall` (0 or 1), `recovery_time_s`. `recovery_time_s` is always a number: 0.0 if the dog is never destabilised, otherwise the time until it is stable again, or until the episode ends, with `recovered` (0 or 1) beside it. Other keys: `fall_time_s`, `first_collision_time_s`, `sim_unstable`, `elapsed_s`, `commands_received`, `trace_path`. Task 5.
7. **Episode timing.** The decision instant is the moment `reset` returns (`t_s` 0), after the dog has trotted at `dog_forward_speed_m_s` for the settling time. The ball is placed at that instant, already in flight, and is never touched again. `step` may be called any number of times with different commands; a duration is rounded to whole control periods and the returned `t_s` shows the time actually simulated. The episode ends when you call `finish_episode`. Your nominal, dodge, nominal sequence is supported. Task 4.
8. **Two levels of seed.** Your proposed form: the constructor `seed` (with the preset) fixes the hidden world for the life of the object, and `reset(situation, attempt_seed)` fixes that episode's noise, drawn from `(seed, attempt_seed)`. Two worlds built with the same `seed` and reset with the same `attempt_seed` face the same noise, whatever was run before. Leaving `attempt_seed` out uses the episode counter. Task 4.
9. **Saved episode and real-time factor.** Real-time factor 7.80 with the policy running (Gate 1, above). Saved walking episodes: the seven traces under `artifacts/env/locomotion/`. The saved DODGE episode comes with Task 5.
10. **`mujoco` in pyproject.toml.** Declared on this branch as `mujoco==3.11.0`, with `efferent==0.5.0` and `onnxruntime==1.32.0`. Your branch adds `torch`, `snntorch`, `scipy` and `threadpoolctl` in the same block, so the two branches will conflict there and in `config.yaml`, `DECISIONS.md`, `LOG.md` and `experiments/EXPERIMENTS_LOG.md`.

## Ball

IHF size 3 handball: 0.450 kg, radius 0.0939 m (`config.yaml`, `ball`; log entry E6). Contact parameters, friction and inertia are not set by any rule and are stated by the ball-scene work.

## Your look-ahead findings

Confirmed by our research and its second-agent check: the Efferent and rl_sar robot_lab checkpoints are the same policy; forward, lateral and yaw-rate commands at 50 Hz; Unitree's scene has obstacles from about x = 1.1 m; foot geoms have contact priority 1, so the floor-friction preset changes the foot geoms. We will detect ball contact by robot body ids, as you suggest.

## What we need from Person B

- The DODGE SkillSpec (`physics_guess`, `episode`, candidate actions) to run the per-skill oracle and fill the physics-only error and noise-floor rows of `ENV_REPORT.md`. Until then those rows read NOT RUN.
- A reply if any name or frame above does not fit your code.
