# Claude handover: live demos and run stories

Last updated: 2026-11-22 (Europe/Berlin)

## What Codex changed

- `app/duel.py` registers `GET /api/demos`, `POST /api/duel/start`, `POST /api/duel/stop`, and `GET /api/story/<run_id>`.
- `app/server.py` imports the feature and stops/releases a duel when drive, reset, or server shutdown takes the stage.
- `app/player.py` and `app/infer.py` stop/release duels before their stage takes over.
- `tests/app/test_server.py` updates the inference-window expectation for the configured dodge student and checks the demos/story response contract.

The current dodge student comes from `config.yaml` `app.infer.student_dir` when it exists and is a dodge student; otherwise selection falls back to the selected finished run's student. A selected run's own `config.yaml` is used with its student when present. Finished demo/result runs are ordered by their saved `updated_utc` / final `events.jsonl` completion time, with result-file time only as a fallback. Do not pin the code to `v4` or `reflex1`.

## Runtime behavior and current data

- Duel episodes use two real `Go2World` instances, the actual skill episode function, the same sampled scenario/seed, and wall-clock pacing. Traces are disabled for showcase episodes.
- The roaming simulation is held while the duel owns the stage; this prevents untagged roaming poses from overwriting duel-side events. Pose, scene, feel, and eye payloads carry `side` during a duel.
- Scene data uses the episode player's actual scene geometry helper. Dodge eye cells are preprocessed with the skill's configured `VisionSettings`.
- The live workspace has a finished `reflex2` with student and four saved rounds; the story CLI selects its newest complete round by the saved artifact set. Config points at `results/runs/v4g/student` for dodge. Do not pin dodge to `v4`: `v4g` currently contains the student only, while `v4` is a separate older run.
- Both skills completed through `/api/duel/start` and `/api/events` with real outcomes. Hills (run reflex1 at the time) returned before `reached=0`, student `reached=1`. Dodge seed 42 returned non-null outcomes on both sides (`collision=0`, `fall=0`, closest separation 2 m). After the stage-hold and serialized camera-relay fixes, SSE samples included `pose`, `scene`, `feel`, and `eye` for both `before` and `after`. Dodge measured about 4 simulated seconds in 5.42 wall seconds on the last smoke (about 1.36 wall s per simulated second). Earlier hills pacing was 13.04 sim s / 25.01 wall s; hardware affects both.
- `/api/story/ramp-fast1` and `/api/story/move1` returned data read from their run files; absent what-if/imagination/curve/latency artifacts are null. Story code does not synthesize prediction intervals or paths when the run did not save them.

## Verification

- `.venv\\Scripts\\python.exe -m py_compile app\\duel.py app\\server.py app\\player.py app\\infer.py` passed after the last code edits.
- `.venv\\Scripts\\python.exe -m pytest tests\\app -q`: **170 passed, 4 skipped** in 68.20 s. This full suite was run before the final scene-rate/camera-relay and run-discovery refinements; `py_compile` and a live dodge SSE duel passed after those changes.
- The earlier initial suite run failed one stale assertion that assumed no configured student, plus a teardown race because Claude created `results/runs/reflex2` during that run. The stale assertion is now config-aware; reflex2 is already present so it should not count as a newly-created folder on a subsequent run.

## Integration checks still useful

1. Start the server with `app.server.start(load_config(), 0)` only; do not touch the server already on port 8765.
2. When `reflex2/student/student.json` appears, confirm hills becomes available and replay the new hills student through SSE.
3. Recheck `GET /api/demos` after punch retraining updates the configured dodge student path; the selection and loading path is dynamic. Recheck `GET /api/story/ramp-fast1` and `/api/story/move1`.
4. For the story follow-up, edits were limited to `learned_judgment/story.py`, `app/duel.py`, `tests/app/**`, and this handover. Avoid protected skill/observed/config/static/front files, `.env`, billed requests, commits, and port 8765.

## Story artifacts (2026-11-22)

- Added `learned_judgment/story.py` CLI commands `paths`, `brain`, and `holdout`; `app/duel.py::_story` now reads only `RUN/story/{attempts,what_if,imagination,brain}.json` and returns null for absent files. The `paths` and `brain` commands were run on `reflex1`, `ramp-fast1`, `move1`, and `v4`; `reflex2` paths/brain were also generated from its newest complete round.
- `attempts.json`, `what_if.json`, `imagination.json`, and `brain.json` keep the existing `app/static/js/story.js` API shapes (plus detailed per-outcome and provenance fields) so the current story UI can render them without a static edit.
- `reflex1`, `ramp-fast1`, and `move1` have dream windows and produce real spike counts and measured network latency. `v4` has only result/student files and no context/dream rows, so attempts and dream-window measurements are unavailable. `v4g` likewise has only its student export. Some ramp/move context rows do not contain per-try trajectories; those records report saved scenario facts and measured action outcomes, with paths null.
- `python -m learned_judgment.story holdout --run reflex1` printed a free quote and exited 3. The current completed `reflex2` quote is **10,000 tokens** (156 held-out decisions, one stacked request); its lead command is `python -m learned_judgment.story holdout --run reflex2 --confirm-billing`. Do not execute here.
- The saved dodge candidate `dodge_vision-20261121T235608` is marked failed in `run_status.json`, with no exported student. Its free preflight quote was **10,000 tokens** (80 held-out decisions, one stacked request), but the CLI refuses a confirmed/billed holdout for it. No finished punch retrain folder is currently available; when one exists, re-quote it and use its actual run id in `python -m learned_judgment.story holdout --run <newest-finished-punch-run-id> --confirm-billing`.
- No billed TabPFN prediction was made. The app suite passed: 170 passed, 4 skipped. The free quotes used only provider connection/usage/quote endpoints.
- Story scope exception: `learned_judgment/story.py`, `app/duel.py`, and `tests/app/**` were the only code files in scope for this task; no protected skill/observed/config/static/front files were edited.
