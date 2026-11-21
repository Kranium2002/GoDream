# Person B handoff and interface requests

Branch `person-b`. Status on 2026-11-21: Person B's code is written and passes its mechanics checks. No Go2 episode, no TabPFN prediction request, and no training on real dreams has been run, so every acceptance criterion in 06_PERSON_B.md is OPEN. The next step for all of them is Person A's simulator.

## What Person B needs from Person A

These block every measured result. Items marked "proposed" are not frozen; change them and tell Person B.

1. **The world class.** `from learned_judgment.env import Go2World`, with `Go2World(config, seed)`, `reset(situation) -> observation`, `step(command, duration_s) -> observation`, `finish_episode() -> measured_outcome`. Person B builds one fresh world per episode and passes the per-episode seed.
2. **How a world preset is selected** (default, heavy_payload, slippery_floor, tired_motors, downhill). Proposed: a `preset` key inside `config`, or a third constructor argument. Either works; Person B needs to know which.
3. **The verified command schema in config.yaml**, in the layout Person B's loader reads. Extra keys per field (frame, sign, native name and index) are welcome and ignored by the loader.

        robot:
          walking_policy:
            status: verified
            command_schema:
              update_period_s: <seconds between accepted commands>
              fields:
                <field_name>: {unit: <m/s | rad/s>, min: <float>, max: <float>, description: <text>}

   Until `status` is `verified` and the fields are present, Person B's code refuses to build any skill. It never guesses a field or a range.
4. **DODGE situation, proposed.** Values Person B samples and passes to `reset`, and uses as model features. Dog base frame at the decision instant, SI units: `ball_rel_x_m`, `ball_rel_y_m`, `ball_rel_z_m`, `ball_vel_x_m_s`, `ball_vel_y_m_s`, `ball_vel_z_m_s`, `dog_forward_speed_m_s`. Person A's throw distribution in config.yaml (`ball.*`) defines the ranges.
5. **Declared world values.** The declared payload mass and the declared surface type of the current world, as the judge would state them, available from the world or from every observation. Proposed keys: `declared_payload_kg`, `declared_surface`. They are model features. The true hidden values must not appear under any key Person B reads.
6. **DODGE measured outcome, proposed.** Keys in `finish_episode()`: `min_separation_m` (smallest distance between ball surface and robot, 0 on contact), `collision` (0 or 1), `fall` (0 or 1), `recovery_time_s`. Traces and anything else can sit under other keys; the SkillSpec reads only its outcome fields and rejects nothing extra from the world.
7. **Episode timing.** When the ball is released relative to `reset`, how long an episode lasts, and whether `step` may be called several times with different commands before `finish_episode` (Person B plans: nominal command for a delay, dodge command for a hold time, nominal command until the end).
8. **One saved real episode** with its raw artifact, and the measured real-time factor, so run times can be planned.
9. **mujoco in pyproject.toml.** It is installed in the venv but not declared, so a plain `uv sync` uninstalls it together with glfw and pyopengl. Declare it, or use `uv sync --inexact`.

## Look-ahead findings that concern Person A (unverified)

Read-only source inspection, no policy was run. Details and pinned commits: experiments/EXPERIMENTS_LOG.md, entry B-lookahead.

- The Efferent go2_walk checkpoint and the rl_sar robot_lab checkpoint are the same file.
- Both learned candidates take forward velocity, lateral velocity, and yaw rate in the base frame at 50 Hz. Native names and observation scaling differ; trained command ranges are not established for either.
- Upstream HIMLoco is licensed CC BY-NC-SA 4.0 and has no Go2 config.
- Vendored unitree_mujoco Go2 model: timestep 0.002, base body `base_link`, no ball in the scene, static step boxes from x = 1.2 m to 3.4 m.
- The trunk collision geoms have no names and geom ids shift between go2.xml and scene.xml, so contact detection by robot body ids is safer than by geom name.
- Foot geoms have contact priority 1 over the floor, so a floor-friction preset must change the foot geoms or the priority.

## What exists on this branch

All tunables are in config.yaml; code has no built-in defaults.

| Area | Module | Entry points |
|---|---|---|
| Skill contract | `learned_judgment/skills/base.py` | `SkillSpec`, `validate_spec`, `CommandSchema.from_config` |
| Decide | `learned_judgment/engine/decide.py` | `decide`, `decide_batch` (one residual-model call per batch) |
| Attempt rows | `learned_judgment/engine/rows.py` | `build_real_row`, `append_rows`, `read_rows`, `rows_to_arrays` |
| Collect, learn, dream, adapt | `learned_judgment/engine/` | `collect`, `fit_residual_model`, `dream`, `keep_confident`, `select_context` |
| Teacher | `learned_judgment/teacher.py` | `TabPFNTeacher.fit`, `predict_residuals`, `predict_residual_distribution` |
| Baselines and evaluation | `learned_judgment/eval/` | residual baselines, `split_situations`, `test_situations`, `compare_methods`, `paired_difference_ci`, `build_final_table` |
| SNN student | `learned_judgment/students/` | `SpikingNetwork` (numpy inference), `InputEncoder`, `estimate_energy`; training code is being added |

Array order is fixed per skill: situation features, then action dimensions, then one physics-guess column per outcome. Residual = measured outcome − physics guess, per outcome, in the SkillSpec's outcome order.

Attempt rows are JSON Lines, one file per run under `results/attempts/`. The row's `seed` is the seed of the world that attempt ran in.

## Provider use

- Each request is checked against live usage and a quote first, capped at 10,000 tokens, counted against the 35,000,000-token project budget, cached completely on disk, and never sent twice.
- Checked live against the free endpoints: the teacher refused a request before any prediction call; account usage stayed at 40,000 tokens.
- NOT RUN: any real prediction request. The first one must use Go2-derived data (04_EXPERIMENTS.md, E7).

## Limitations

- Every executed check uses fixtures and tests mechanics only. None is evidence about the robot, the simulator, TabPFN, or the SNN student.
- The DODGE SkillSpec is not written; it needs items 3 to 7.
- The teacher's real-response handling (shape and order of the returned quantiles, actual charge against quote) is written from the client's source and is unverified until the first real request.
