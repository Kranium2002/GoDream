# Person A handoff and answers to Person B

Branch `person-a/go2-env`. Status on 2026-11-21, evening: the walking stack is selected and pinned, the locomotion gate has been run (6 of 7 conditions pass, 0.3 m/s fails) and the command schema is published with `status: verified`. `Go2World` exists for the `default` preset: reset, step and finish episodes with two-level seeds, traces and an offscreen frame, tested on the real simulator and policy (`artifacts/env/20261121T161057-world`, 17 passed). The ball, the hidden effects and the other four presets follow (Tasks 5 and 6) and are NOT RUN. Plan: `docs/superpowers/plans/2026-11-21-person-a-go2-env.md`.

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

Names follow your proposals wherever you made one. Items 1, 3, 7 and 8 are delivered, and 2 and 5 for the `default` preset; for the others the task that delivers each item is named. Numbers follow your current list (ten items).

1. **World class. Delivered.**

        from learned_judgment.env import Go2World, CommandError, load_config, make_streams

        world = Go2World(load_config(), seed, preset="default")
        obs = world.reset(situation=None, attempt_seed=None)  # observation at the decision instant, t_s 0.0
        obs = world.step(command, duration_s)                 # any number of times; returns the last observation
        outcome = world.finish_episode()                      # then reset again for the next episode

   `world.command_schema` is `robot.walking_policy.command_schema`. `step` validates the command strictly (`CommandError` names the field). `step` or `finish_episode` without an episode in progress raises `RuntimeError`; a `reset` during an episode abandons it without a trace. `world.render_frame()` returns a 480×640×3 `uint8` image of the current state, camera tracking the trunk; OpenGL is touched only on the first call. The full contract is the `Go2World` docstring (`learned_judgment/env/world.py`).

   Observation, one per 0.02 s control period. World frame: x, y on the floor, z up, origin on the floor under the spawn point, the dog starts facing +x.

    | Key | Value |
    |---|---|
    | `t_s` | float, s; 0.0 at the decision instant |
    | `base_pos_m` | (3,) m, trunk (`base_link`) origin, world frame |
    | `base_quat_wxyz` | (4,) trunk orientation, body to world, w x y z |
    | `base_lin_vel_m_s` | (3,) m/s, world frame |
    | `base_ang_vel_rad_s` | (3,) rad/s, world frame (MuJoCo's body-frame value rotated); z > 0 turns anticlockwise seen from above |
    | `joint_pos_rad`, `joint_vel_rad_s` | (12,) rad and rad/s, model order FL, FR, RL, RR, each hip, thigh, calf |
    | `policy_obs` | (45,) float32, what the walking policy read at its last step (`observation_schema` scaling); NaN before its first step |
    | `foot_contact` | (4,) bool, FL, FR, RL, RR: that foot has a contact with the floor |
    | `fallen` | bool; latched for the episode |
    | `ball` | `None` until the ball is added (Task 5) |
    | `command_active` | dict, the command the policy is receiving now |
    | `declared_payload_kg`, `declared_surface` | item 5 |

   Outcome: `elapsed_s` (s since the decision instant), `fall` (0 or 1: a trunk geom on the floor or tilt above `robot.fall.max_tilt_deg` = 60°, at any physics step from the spawn on), `fall_time_s` (`t_s` of the first such step, negative if before the decision instant, `None` without a fall), `sim_unstable` (0 or 1: MuJoCo flagged a bad qpos, qvel or qacc, or the final state is not finite), `commands_received` (list of `(t_s, command)`, one per `step`), `final_base_pos_m` ([x, y, z] m, world frame), `declared_payload_kg`, `declared_surface`, `trace_path` (`.npz` under `world.trace_dir` with every observation array per control period from the decision instant on, or `None` with `world.save_traces: false`). The DODGE keys of item 6 come with the ball.

   Cost on this machine (`artifacts/env/20261121T160704-world-cost`): building a world 0.12 s (0.70 s the first time in a process: it loads the policy); `reset()` 0.58 s (6 s simulated: 1 s hold, 3 s soft start, 2 s trot); `step()` real-time factor 7.96 for one 10 s call, 7.79 for 0.02 s calls; `finish_episode()` 0.03 s writing a 20 s trace, about 0.1 ms or less without one. One fresh world per episode works.
2. **Preset. Delivered.** A third constructor argument: `Go2World(config, seed, preset="default")`; a name that is not in `config.yaml` `world.presets` raises `ValueError`. Five presets, each holding the hidden world's true values (`payload_mass_kg`, `payload_offset_m`, `floor_type`, `slope_deg` positive uphill along the dog's starting heading, `motor_strength_fraction`, `command_latency_s`): `default` (no payload, floor `default`, flat, full strength), `heavy_payload` (3 kg at the trunk's rear-left corner), `slippery_floor` (floor `slippery`), `tired_motors` (80% strength) and `downhill` (−5°), all with a nominal command latency of 0.08 s plus −1, 0 or +1 control period of jitter drawn per attempt. The seed and preset fix everything but that jitter for the life of the world. Each of the five effects can be switched off in `world.hidden_effects.enabled`. The values are a proposal awaiting confirmation (`DECISIONS.md`); each effect alone is measured in `artifacts/env/effects-data/effects.json` (log entry E5). Floor types, for your `declared_surfaces`: `default`, `slippery`.
3. **Command schema. Published; `robot.walking_policy.status` is `verified`.** In your loader's layout under `robot.walking_policy.command_schema` in `config.yaml` (your `CommandSchema.from_config` at `origin/person-b` loads it): `update_period_s: 0.02` (one command per 50 Hz policy step), and three fields in the policy's native order, each with `unit`, `min`, `max`, `description`, plus `native_index`, `proven_by`, `measured_at_min` and `measured_at_max`:

    | Field | Unit | Range | Measured at min / max | Meaning |
    |---|---|---|---|---|
    | `vel_x` | m/s | −1.0 to 1.0 | −0.8880 / 0.9243 | forward (+) or backward (−) along the dog's own heading |
    | `vel_y` | m/s | −1.0 to 1.0 | −1.0349 / 0.8939 | left (+) or right (−) in the dog's own heading frame |
    | `yaw_rate` | rad/s | −1.0 to 1.0 | −0.9555 / 0.9687 | anticlockwise (+, turning left) seen from above |

   Each range end was proven by one recorded 10 s run on a flat floor, that field alone, the others 0: the dog did not fall and moved with the commanded sign (log entry "E4 — command inventory", `artifacts/env/commands/commands.json`). `verified` means exactly that and nothing about tracking accuracy. Limits you should plan around: forward tracking on Gate 1 was 0.2256 / 0.5430 / 0.8596 m/s for 0.3 / 0.6 / 0.9 commanded (0.3 m/s misses the gate's 20% tolerance and Gate 1 still fails on it); at zero yaw command the dog turns anticlockwise at 0.05–0.14 rad/s; values between the ends, combined commands and the delay after a command change are not measured; in robot_lab training planar commands with norm ≤ 0.2 m/s were zeroed. `validate_command(command, schema)` (from `learned_judgment.env`) returns the native vector or raises `CommandError` naming the field: every field is required, nothing is clipped.
4. **DODGE situation.** Your seven keys, accepted by `reset`: `ball_rel_x_m`, `ball_rel_y_m`, `ball_rel_z_m`, `ball_vel_x_m_s`, `ball_vel_y_m_s`, `ball_vel_z_m_s`, `dog_forward_speed_m_s`. Frame: the dog's heading frame (x forward, y left, z up, origin at the base) at the decision instant. `Go2World.sample_dodge_situation()` draws a situation with those keys from a physical throw model, and `config.yaml` will list each key's minimum and maximum. An independent box over the six ball values would mostly miss the dog, so sample through that method or tell us if you need another form. Task 5.
5. **Declared world values. Delivered.** `declared_payload_kg` (float, kg) and `declared_surface` (str) in every observation and in the outcome, and as the read-only attributes `world.declared_payload_kg` and `world.declared_surface`, all equal and fixed for the life of the world. The payload is the preset's true mass × (1 + an error drawn once per world, uniformly within ±20%), so the true mass cannot be read from it exactly (0.0 for the presets without payload); the surface is the preset's floor type. True hidden values are written only to `<trace>.hidden.json` next to the trace (preset, seed, attempt seed, the effects switched on with their values, and this attempt's latency) and never appear in an observation or outcome (tested). On a slope the policy senses true gravity, as a real robot's IMU would, so `policy_obs[3:6]` (projected gravity) shows the tilt.
6. **DODGE outcome.** `min_separation_m` (never below 0, 0 on contact), `collision` (0 or 1), `fall` (0 or 1), `recovery_time_s`. `recovery_time_s` is always a number: 0.0 if the dog is never destabilised, otherwise the time until it is stable again, or until the episode ends, with `recovered` (0 or 1) beside it. Other keys: `fall_time_s`, `first_collision_time_s`, `sim_unstable`, `elapsed_s`, `commands_received`, `trace_path`. Task 5.
7. **Episode timing. Delivered** (the ball's part with Task 5). `reset` spawns the dog, holds it 1 s, runs the policy's 3 s soft start, then trots at `dog_forward_speed_m_s` (default 0.0; outside `vel_x`'s range it raises `CommandError` naming `vel_x`) for `settle_s` (situation key, default `world.settle_s` = 2.0 s, chosen from the Gate 1 traces; must be ≥ 0), and returns at the decision instant, `t_s` 0.0. The ball will be placed at that instant, already in flight, and never touched again (Task 5). `step` may be called any number of times with different commands; each duration is rounded to whole 0.02 s control periods (at least one; a non-positive or non-finite duration raises `ValueError`) and the returned `t_s` shows the time actually simulated. The episode ends when you call `finish_episode`. Your nominal, dodge, nominal sequence is supported. Situation keys today: `dog_forward_speed_m_s`, `settle_s`; any other key raises `ValueError` naming it, so the six ball keys are refused until Task 5.
8. **Two levels of seed. Delivered.** Your proposed form: the constructor `seed` (with the preset) fixes the hidden world for the life of the object, and `reset(situation, attempt_seed)` fixes that episode's noise: `world.attempt_streams` is `make_streams(seed, attempt_seed)`, generators under `situation`, `world`, `ball` and `noise`, independent of the seed-only streams `make_streams(seed)`. Two worlds built with the same `seed` and reset with the same `attempt_seed` face the same noise, whatever was run before (tested). Leaving `attempt_seed` out uses the episode counter (0 for a world's first reset). Nothing draws noise yet: the simulator and the policy (now on one inference thread) are deterministic, and two runs with the same seed and attempt seed give bit-identical traces (`tests/env/test_determinism.py`).
9. **Saved episode and real-time factor.** Real-time factor 7.80 with the policy running (Gate 1, above). Saved walking episodes: the seven traces under `artifacts/env/locomotion/`. The saved DODGE episode comes with Task 5.
10. **`mujoco` in pyproject.toml.** Declared on this branch as `mujoco==3.11.0`, with `efferent==0.5.0` and `onnxruntime==1.32.0`. Your branch adds `torch`, `snntorch`, `scipy` and `threadpoolctl` in the same block, so the two branches will conflict there and in `config.yaml`, `DECISIONS.md`, `LOG.md` and `experiments/EXPERIMENTS_LOG.md`.

## Ball

IHF size 3 handball: 0.450 kg, radius 0.0939 m (`config.yaml`, `ball`; log entry E6). Contact parameters, friction and inertia are not set by any rule and are stated by the ball-scene work.

## Your look-ahead findings

Confirmed by our research and its second-agent check: the Efferent and rl_sar robot_lab checkpoints are the same policy; forward, lateral and yaw-rate commands at 50 Hz; Unitree's scene has obstacles from about x = 1.1 m; foot geoms have contact priority 1, so the floor-friction preset changes the foot geoms. We will detect ball contact by robot body ids, as you suggest.

## What we need from Person B

- The DODGE SkillSpec (`physics_guess`, `episode`, candidate actions) to run the per-skill oracle and fill the physics-only error and noise-floor rows of `ENV_REPORT.md`. Until then those rows read NOT RUN.
- A reply if any name or frame above does not fit your code.
