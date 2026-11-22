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
- The live workspace has an exported student under `results/runs/reflex1/student`. `reflex2` has a finished result and scenario/trace artifacts but no student at the last check, so it is now the selected hills run and `/api/demos` correctly marks hills unavailable with a plain missing-student reason. Once its student is written, the same discovery path will enable it. Config points at `results/runs/v4g/student` for dodge; the finished result/metrics are in the newest finished dodge run under `results/runs`.
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
4. Keep edits within `app/*.py` and `tests/app/`, except this requested handover document. Avoid `app/static/**`, `tests/front/**`, `learned_judgment/**`, config edits, `.env`, TabPFN, billed runs, commits, and port 8765.
