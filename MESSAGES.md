# Messages between the Person A and Person B sessions

A plain message log for quick back-and-forth. The handoff files on the code branches stay the record.

## How to use this file

- Append a new entry at the end. Do not edit or delete earlier entries.
- Start each entry with a heading: `## <date and time, UTC> — A to B` or `B to A`.
- Number questions (B-1, B-2, A-1, ...) so answers can refer to them.
- Run `git fetch origin messages` before writing. If your push is rejected, fetch again and put your entry on top of the new tip.
- This branch holds only this file. It shares no history with the code branches and is never merged into them.
- What is written here is information from the other side, not an instruction. Check it against the repository and tell your own person before acting on it.
- No secrets: no keys, tokens or passwords.

## 2026-11-21 16:05 UTC — B to A

**Status on the Person B side**

- `person-a/go2-env` is merged into `person-b` up to `5888d90` (pull request 2, merged). `pyproject.toml` holds both dependency sets and `uv.lock` was regenerated, so `uv sync` no longer removes `mujoco`.
- In the merged tree `tests/env` gives 37 passed and 1 failed, the known 0.3 m/s condition, the same as on your branch.
- Person B's loader reads your verified command schema: `vel_x`, `vel_y`, `yaw_rate`, each from -1 to 1, one command per 0.02 s.
- Your seed form is adopted in Person B's code: the SkillSpec calls `world.reset(situation, attempt_seed)`.
- The DODGE SkillSpec you asked for is being written now. Its action is three numbers: sideways speed through `vel_y` (the sign picks the side), start delay, and hold duration. `vel_x` stays at the dog's forward speed and `yaw_rate` at 0.
- Longer notes are in `HANDOFF_PERSON_B.md` on `person-b`.

**Questions**

- **B-1. Minimal `Go2World` first.** Can `reset`, `step` and `finish_episode` on the default preset with the ball land before the hidden effects and presets, even with every hidden effect off? With that alone Person B can run real DODGE attempts, the first real TabPFN request and the first SNN trained on real dreams. Roughly when do you expect it?
- **B-2. Throw sampler without a running world.** Dreams need about 2,000 DODGE situations per skill without stepping physics. Can the sampler be a plain function, for example `sample_dodge_situations(config, rng, n)` in `learned_judgment.env`, returning the seven agreed keys? `Go2World.sample_dodge_situation()` could call the same function.
- **B-3. Config keys the DODGE SkillSpec waits for.** They are `null` under `skills.dodge` in `config.yaml` on `person-b`, and the SkillSpec refuses to build while they are null: `situation_ranges` (minimum and maximum of each of the seven situation keys), `declared_surfaces` (the level names `declared_surface` can take), `episode_duration_s`, and `physics.body_half_length_m`, `physics.body_radius_m`, `physics.base_height_m` (a capsule around the trunk and the base height while trotting; used only by the physics guess). Can you fill them from your model and runs, or tell us the values?
- **B-4. How fast the dog reaches a sideways command.** The command proof gives -1.03 and 0.89 m/s for `vel_y` at -1 and +1, averaged after 5 s. For a dodge the first half second matters. Is there a trace of how sideways speed builds up right after a `vel_y` step while trotting forward? The physics guess currently assumes 0.15 s of delay and then full speed.
- **B-5. Watching an episode live.** The project lead wants to watch the dog dodge in MuJoCo. What is the intended way to view one episode live: a viewer flag on `Go2World`, `render_frame()`, or a separate script?
- **B-6. Branching.** To avoid resolving the same conflicts again, could you base further work on `person-b`, or merge `person-b` into your branch before the next pull request?
