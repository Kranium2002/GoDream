# Person B handoff and notes for Person A

Branch `person-b`. Updated 2026-11-21 after merging `person-a/go2-env` (up to `e4ddadd`) into it. Person B's code is written and passes its mechanics checks. No Person B result has been measured on the robot yet: that needs `Go2World`.

## Notes for Person A: what to build next

Ordered by how much each item unblocks. Thank you for adopting the proposed names; they are now in Person B's code and config.

1. **A minimal `Go2World` first, before hidden effects.** `reset`, `step`, `finish_episode` on the default preset with the ball, even with every hidden effect off. With that alone Person B can run real DODGE attempts, the first real TabPFN request and the first SNN trained on real dreams. Hidden effects and the other presets can follow.
2. **Verify `vel_y` first in the command schema.** DODGE is a sideways step, so it depends on the sideways-speed field. Person B's code refuses to build any skill until `robot.walking_policy.status` is `verified` and the field's `min` and `max` are published. `vel_x` and `yaw_rate` can come after.
3. **Two levels of seed.** Evaluation needs the hidden world fixed for a whole run while each attempt has its own noise, and every method must face the same noise on the same episode. Proposed: the constructor seed fixes the hidden world and `reset(situation, attempt_seed)` seeds one attempt. Person B's SkillSpec will call it that way. If you prefer another form, say which.
4. **A throw sampler that needs no running world.** Dreams need about 2,000 situations per skill without stepping physics. Proposed: a function `sample_dodge_situations(config, rng, n)` in `learned_judgment.env` returning the seven agreed keys. `Go2World.sample_dodge_situation()` can call the same function.
5. **Fill these config keys** (Person B added them as `null` under `skills.dodge`; the SkillSpec refuses to build while they are null):
   - `situation_ranges`: minimum and maximum of each of the seven situation keys.
   - `declared_surfaces`: the level names `declared_surface` can take.
   - `episode_duration_s`: total simulated time of one attempt.
   - `physics.body_half_length_m`, `physics.body_radius_m`, `physics.base_height_m`: a capsule around the trunk and the base height while trotting, from your model and runs. They are used only by the physics guess.
6. **Outcome keys as you listed them.** The SkillSpec reads `min_separation_m`, `collision`, `fall` and `recovery_time_s` and ignores the rest. Please keep `commands_received`.
7. **A way to watch an episode live.** The project lead wants to see the dog dodge in MuJoCo: a viewer flag or `render_frame()` is enough.
8. **Gate 1 at 0.3 m/s is yours and the lead's to resolve.** One data point from the look-ahead research: the candidate training config sets planar commands with norm up to 0.2 to zero, and your measured speeds are below the commanded ones at all three speeds (75%, 90% and 96%).

## What changed on `person-b` that affects you

- **Dependencies.** `pyproject.toml` now holds both sets, and `uv.lock` was regenerated. `mujoco` is declared, so `uv sync` no longer removes it. PyTorch comes from the CPU index. The shared venv has `mujoco` 3.11.0, `efferent` 0.5.0, `onnxruntime` 1.32.0 and `pytest`.
- **Your tests in the merged tree:** `tests/env` gives 31 passed and 1 failed, the same as on your branch.
- **config.yaml.** Person B added the sections `engine`, `tabpfn`, `persistence`, `baselines`, `students`, `evaluation`, and keys under `skills.dodge`. Your sections merged without conflict.
- **Mission documents.** The only student is a spiking neural network trained inside the PFN dream; no MLP student is built. `PREREGISTRATION.md` has dated amendments.
- **Logs.** `LOG.md`, `DECISIONS.md` and `experiments/EXPERIMENTS_LOG.md` hold both branches' entries. Appending at the end keeps future merges simple.
- **Branching.** To avoid resolving the same conflicts again, base further work on `person-b`, or merge `person-b` into your branch before the next pull request.

## What Person B delivers to you

- **The DODGE SkillSpec** (`learned_judgment/skills/dodge.py`): `physics_guess`, `episode` and `candidate_actions`, which you need for the oracle and the difficulty rows of `ENV_REPORT.md`. It is being written now against the names in your handoff.
- **How it calls the world:**

        observation = world.reset(situation, attempt_seed)
        world.step({"vel_x": forward_speed, "vel_y": 0.0, "yaw_rate": 0.0}, start_delay_s)
        world.step({"vel_x": forward_speed, "vel_y": side_velocity, "yaw_rate": 0.0}, hold_duration_s)
        world.step({"vel_x": forward_speed, "vel_y": 0.0, "yaw_rate": 0.0}, remaining_s)
        outcome = world.finish_episode()

  The action is three numbers: sideways speed (its sign picks the side), start delay and hold duration. Every command is checked against the verified schema before the world is stepped.

## What exists on this branch

All tunables are in config.yaml; code has no built-in defaults.

| Area | Module | Entry points |
|---|---|---|
| Skill contract | `learned_judgment/skills/base.py` | `SkillSpec`, `validate_spec`, `CommandSchema.from_config` |
| Decide | `learned_judgment/engine/decide.py` | `decide`, `decide_batch` (one residual-model call per batch) |
| Attempt rows | `learned_judgment/engine/rows.py` | `build_real_row`, `append_rows`, `read_rows`, `rows_to_arrays` |
| Collect, learn, dream, adapt | `learned_judgment/engine/` | `collect`, `fit_residual_model`, `dream`, `keep_confident`, `select_context`, `train_in_dream` |
| Teacher | `learned_judgment/teacher.py` | `TabPFNTeacher.fit`, `predict_residuals`, `predict_residual_distribution` |
| Baselines and evaluation | `learned_judgment/eval/` | residual baselines, `split_situations`, `test_situations`, `compare_methods`, `paired_difference_ci`, `build_final_table` |
| SNN student | `learned_judgment/students/` | `SNNStudent`, `SharedSNNStudent`, `SpikingNetwork` (numpy inference), `measure`, `estimate_energy` |

Array order is fixed per skill: situation features, then action dimensions, then one physics-guess column per outcome. Residual = measured outcome − physics guess, per outcome, in the SkillSpec's outcome order.

Attempt rows are JSON Lines, one file per run under `results/attempts/`.

## Provider use

- Each request is checked against live usage and a quote first, capped at 10,000 tokens, counted against the 35,000,000-token project budget, cached completely on disk, and never sent twice.
- Checked live against the free endpoints: the teacher refused a request before any prediction call; account usage stayed at 40,000 tokens.
- NOT RUN: any real prediction request. The first one must use Go2-derived data (04_EXPERIMENTS.md, E7).

## Limitations

- Every executed Person B check uses fixtures and tests mechanics only. None is evidence about the robot, the simulator, TabPFN, or the SNN student.
- The teacher's real-response handling (shape and order of the returned quantiles, actual charge against quote) is written from the client's source and is unverified until the first real request.
