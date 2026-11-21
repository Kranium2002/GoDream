# Dev A task plan: chat-taught, simulation-trained Go2 demo

**Spec (binding):** `talkhere.md` on branch `messages` at `ec44ca4` (`git show origin/messages:talkhere.md`), section "Dev A prompt" plus the contracts and gates above it. Repo rules: `CLAUDE.md`, `MISSION.md`, `05_PERSON_A.md`.

**Goal:** the product side of the demo. A chat request becomes a validated `DeclaredSpec`; the page shows a measured baseline, real MuJoCo attempts, PFN/SNN progress and paired held-out results for RAMP and DODGE; the finale drives the simulated Go2 from the laptop camera through Dev B's frozen SNN.

**Person A's decisions (2026-11-21):**
1. Code is pushed to `person-a/go2-env` as each milestone passes its tests (the orchestrator merges and pushes; lane agents never push).
2. The chat uses Claude through the Anthropic API from the server. Key: `ANTHROPIC_API_KEY` in `GoDream/.env` (git-ignored), added by Person A. Without it the chat reports `chat_unavailable`.
3. Work builds on the earlier session's hidden-effects branch and viewer server.
4. The UI is built now against the contracts proposed here. Anything Dev B has not delivered is shown as unavailable with the missing piece named. No placeholder numbers, no staged animation.

## Lanes

| Lane | Branch | Worktree (under `GoDream/.claude/worktrees/`) | Owns |
|---|---|---|---|
| world | `person-a/lane-world` | `agent-ab1fb41675a2f3f34` | `learned_judgment/env/**`, `tests/env/**`, `config.yaml` except the `app:` block |
| server | `person-a/lane-server` | `agent-af9ed52c4d3b0b861` | `app/*.py`, `tests/app/**`, the `app:` block of `config.yaml` |
| front | `person-a/lane-front` | `agent-a4c46a3a7c811730a` | `app/static/**`, `tests/front/**` |

A lane edits only what it owns. Nobody edits Dev B's files (`learned_judgment/skills/`, `engine/`, `teacher*.py`, `students/`, `eval/`, `pipeline.py`, `experiments/`), the numbered prompts, or vendored `third_party/`. `LOG.md`, `DECISIONS.md` and `HANDOFF_PERSON_A.md` are written by the orchestrator only: a lane puts the text for them in its report.

## Rules for every task

- Test first. Use the `superpowers:test-driven-development` skill: write the test, run it, watch it fail for the right reason, then write the code. A failing test you did not see fail proves nothing.
- System tests run the real `Go2World`, the real MuJoCo model and the real walking policy. No mock policy, no hand-written trace, no fake episode. A fixture may test a pure function only.
- Before each slice, prove the API you rely on with a small command and keep its output for the report (this replaces a separate research agent; the orchestrator records the substitution). Earlier proofs are in the session scratchpad under `poc-incline/` and `poc-camera/`.
- Every tunable number lives in `config.yaml` with a one-line comment.
- Hidden truth (slope angle, object pose and velocity, scenario kind, latency, payload, friction) never appears in an observation, an outcome, an HTTP response to the policy path, or a browser event. It goes to `<trace>.hidden.json` and to scoring code only.
- Record system claims with `python -m learned_judgment.env.evidence <label> -- <command>`; cite the artifact folder. Anything not executed is NOT RUN.
- Laptop camera frames are never written to disk, logged or sent off the machine.
- Commit small, on your lane branch, message style `env: ...` / `app: ...`, ending with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. Never push, never merge another lane unless your task says so, never use `git stash`.
- The machine has 4 cores shared by all lanes. Run the smallest test selection while iterating and the lane's full suite once at the end of a task.

**Report format (your final message):** task id; status DONE / BLOCKED; commits (`git log --oneline <base>..HEAD`); every command you ran for evidence with exit code and the result line; files changed; rulings you made (what, why, cost if wrong); text for LOG.md and DECISIONS.md; open items and anything another lane or Dev B must know.

## Contracts

### World API (lane world provides; Dev B and lane server consume)

- `Go2World.render_frame(camera="tracking")` -> `(H, W, 3) uint8`. `camera="head"` renders the dog-head camera at its own size. Unknown name raises `ValueError`. Config `world.render.head_camera: {pos_m, quat_wxyz, fovy_deg, width, height}` on `base_link`; pos `[0.34, 0.0, 0.017]`, quat `[0.5, 0.5, -0.5, -0.5]` (looks along trunk +x), `fovy_deg` 45 and a square 96 x 96 image (agreed with Dev B, 20:55 UTC: it matches a laptop camera after a centred square crop, and thousands of training renders stay cheap). An optional `size=(height, width)` argument renders the head view larger for display. The tracking camera gets a fixed distance from config so added geometry does not zoom it out.
- `Go2World.step(command, duration_s, on_period=None, on_physics_step=None)`: `on_period(obs)` is called once per 0.02 s control period (exists in the carried-over work; keep it). `on_physics_step(t_s)` is called after every physics step, so Dev B can render at the step nearest each 1/30 s.
- Observation: `command_active` is the last command issued by the caller (never the latency-delayed one); `policy_obs` is not exposed. `world.declared_payload_kg` and `world.declared_surface` are public attributes.
- Outcome adds, in the heading frame fixed at the decision instant: `final_forward_m`, `final_sideways_m`, `final_yaw_change_rad`, `final_speed_m_s`.
- Incline: situation key `incline_deg` (float, 0 or absent = flat floor). Non-zero builds a visible, physical ramp of that angle ahead of the dog followed by a level plateau (flat -> slope -> flat). Geometry in config `world.incline: {start_x_m, length_m, plateau_length_m, width_m, thickness_m, rgba}`. Ground contact, foot contact and the fall check treat the ramp and plateau as ground. Scoring-only outcome keys asked for by Dev B: `ramp_start_m` and `ramp_end_m` (distance along the heading from the trunk origin at the decision instant) and `foot_slip_m` (summed tangential travel of feet while in contact). `robot.recovery.max_tilt_deg` stays as it is. Sampler `sample_ramp_situations(config, rng, n)`; the supported angle range is published under `world.incline` (`skills.ramp` is Dev B's block).
- Objects: situation key `object_kind` in `{"ball", "punch"}` (default `"ball"`, the existing handball) with the existing six `ball_*` keys. `punch` is a gravity-compensated free body with finite mass that flies straight and collides physically. Outcome keys are unchanged (`collision`, `first_collision_time_s`, `min_separation_m`). In DODGE scenes the dog stands on the zero command.
- Scenarios: Dev B writes `sample_threat_scenarios(config, rng, n)` (ball only) to this signature in their lane: list of `{scenario_id, kind, object_kind, situation, visual}`; `kind` in `empty, static, receding, off_path, background_motion, ambiguous, incoming_left, incoming_center, incoming_right`; `kind` is scoring-only. Lane world provides what it needs from the world: `object_kind` `punch`, and a `visual` situation key (object colour and size, floor and backdrop colour, light direction and strength) applied at reset. Exposure, blur and sensor noise are Dev B's, in image space.

### HTTP API v0 (lane server provides; lane front consumes)

Errors are `{"code": str, "message": str}` with a 4xx/5xx status. A capability that cannot work yet answers 503 with a code ending in `_unavailable` and a message naming the missing piece.

| Method and path | Purpose |
|---|---|
| `GET /api/state` | `{status, preset, presets, commands, frame, capabilities, skills, image_window, actions}`. `capabilities` is `{name: {available: bool, reason: str}}` for `drive, head_camera, scenarios, chat, runs, results, infer`. |
| `GET /api/stream?camera=tracking\|head` | multipart PNG stream of real simulator frames |
| `GET /api/events` | SSE: `observation`, `episode`, `notice`, `run`, `infer` |
| `POST /api/drive`, `/api/stop`, `/api/reset` | as in the carried-over server |
| `GET /api/skills` | the registered skill catalogue with supported parameters and their ranges |
| `POST /api/chat` `{messages:[{role, content}]}` | `{reply, declared_spec\|null, problems:[{code, field, message}], missing:[str]}`; `declared_spec` is Dev B's declaration `{skill, params, source_text}`; the model returns data only |
| `GET /api/scenarios?skill=&seed=&n=` | scenario ids for the dev split only; the scoring-only `kind` is included for display on the Learn screen and is never sent to `/api/infer` |
| `POST /api/episode` `{skill, scenario_id, action\|null}` | runs that real episode on the stage; frames on `/api/stream`, outcome as an `episode` event |
| `POST /api/runs` `{declared_spec, confirm_cost: bool}` | starts a learning run; 202 `{run_id}`. Without `confirm_cost` it answers 409 `cost_confirmation_required` with the quoted usage |
| `GET /api/runs/{run_id}` | `{state, steps:[{step, state, done, total, data}], baseline, results}` built from the run's event file and artifacts |
| `POST /api/infer` `{window:[float], frames, captured_ms}` | Dev B's `infer` result unchanged (`action, command, threat_confidence, best_action, action_confidence, margin, gate, cooldown_remaining_s, hold_remaining_s, issued, inference_ms`) plus `latency_ms` measured by the server; `window` is `frames * 144` floats, oldest frame first; the server applies `command` to the stage |

`image_window` is `{frames, height, width, hz, order, normalization, crop, luma, mirrored}` read from Dev B's `vision` config block when it exists (agreed values: 8, 12, 12, 30, oldest first, divide by 255, centre square, 0.299/0.587/0.114, not mirrored), so the browser and Python cannot drift. `actions` lists `NO_OP, DODGE_LEFT, DODGE_RIGHT, RETREAT` with their command values once Dev B publishes them.

### Dev B contracts (AGREED by Dev B at 20:55 UTC in `talkhere.md`, not pushed yet; code against them behind one adapter module so a change touches one file)

- Declarations, module `learned_judgment.declared`: `skill_catalog(config)` -> `[{skill, description, params: {name: {type, min, max, choices, default, unit}}, actions, outcomes, available, missing}]`; `declaration_schema(config)` -> JSON Schema for the Claude call; `check_declaration(declaration, config)` -> `[{code, field, message}]` (empty = valid); `resolve_declaration(declaration, config)`; `parse_request(text, config)` (keyword rules; not used as a chat fallback, by Person A's decision). A declaration is `{"skill": "RAMP", "params": {"incline_deg": 12.0}, "source_text": "..."}` or `{"skill": "DODGE", "params": {"threats": ["punch", "ball"]}, "source_text": "..."}`.
- Run events: one JSONL file per run, lines `{run_id, skill, step, state, done, total, time_utc, data}` with `step` in `baseline, attempts, teacher_fit, dream, train, evaluate` and `state` in `running, done, failed, refused, stopped`. File location and the `data` fields per step: Dev B posts them when the runner exists.
- Results: per run a JSON with paired before/after metrics on the same held-out scenario ids, with n, seeds and intervals, and the TabPFN quote before a run.
- Live inference, module `learned_judgment.runtime`: `load(path)`; `infer(window, t_capture_s)` -> `{action, command: {vel_x, vel_y, yaw_rate}, threat_confidence, best_action, action_confidence, margin, gate: {threat_ok, action_ok, margin_ok, votes, votes_needed, of_last, passed}, cooldown_remaining_s, hold_remaining_s, issued, inference_ms}` with `action` in `NO_OP, UNCERTAIN, DODGE_LEFT, DODGE_RIGHT, RETREAT`; `window` is float32 `(frames, 12, 12)`, oldest first, in [0, 1].
- Preprocessing, identical in MuJoCo and the browser: RGB frame, not mirrored; centre-crop to a square of side `S = min(W, H)`; grey = `0.299 R + 0.587 G + 0.114 B`; resize to 12 x 12 by area averaging over whole-pixel bins (cell `(i, j)` is the mean over source rows `floor(i*S/12)` to `floor((i+1)*S/12) - 1`, same for columns), computed from pixel data, not canvas scaling; divide by 255; float32 row-major, top-left first; window = the most recent 8 frames at 30 Hz, oldest first (the frame count stays a variable). Dev B ships the Python reference and JSON test vectors with their first push.
- Dev B's config blocks: `vision`, `skills.dodge_vision` (with `actions`: the four commands and hold time), `skills.ramp`, `runtime`.

## Tasks

### World lane

Order: W1, then W3 (the incline is Dev B's one hard dependency), then W2, then W4.

**W1. Verified base.** Run `uv run pytest tests/env -q` on the carried-over work (commit `e2cf9b9`); one failure is known and accepted (the 0.3 m/s locomotion condition). Find the cause of any other failure and fix it. Then close the items Person A promised Dev B, each with a test: public declared attributes; `command_active` and `policy_obs` as in the contract; the four `final_*` measurements; the five presets with the values agreed in `MESSAGES.md` (17:37 UTC entry); `skills.dodge.declared_surfaces`. Done: suite green except the known failure, evidence recorded.

**W2. Head camera.** Contract above. Tests on the real model: the camera exists and looks along trunk +x; `render_frame("head")` has the configured size and differs from the tracking view; a thrown ball in front of the dog is visible in the head view; unknown camera raises. Record one evidence run that saves a head and a tracking PNG of the same state (simulator frames may be saved).

**W3. Traversable incline.** Contract above. Tests: flat when the key is absent or 0; the ramp is visible in a tracking frame; the dog gains the expected height on a 5 degree and a 10 degree ramp and reaches the plateau; feet on the ramp count as ground contact and a standing trunk is not a fall; the angle appears in `hidden.json` and nowhere in observations or outcomes; same seed gives the same trace. Record a sweep (angles from the config range, at least 3 attempt seeds) so the baseline's room to improve is measured, not assumed.

**W4. Punch proxy and visual settings.** Contract above, built on the proof from the objects research (`poc-objects/` in the scratchpad). Tests: a punch flies straight at constant speed and its contact is detected by the existing collision measurement; `object_kind` and `visual` are accepted by `reset` and unknown values raise; visual settings change the rendered frame and are reproducible from the seed; nothing about the object or the scene's truth reaches an observation. The scenario sampler is Dev B's; extend it with `punch` and `visual` only if Dev B asks.

### Server lane

**S1. Land the viewer server.** `tests/app` green on this base: fix the fixture that assumed presets carry `declared_payload_kg`; stream the caller's last issued command, not `command_active`; drop `trace_path` from the episode event. Make routes pluggable so later tasks add a module, not a branch in one long function. Add `capabilities` in the contract's shape. Measure and report PNG encode time and delivered frames per second.

**S2. Skills and chat.** `/api/skills` and `/api/chat`. Load the `claude-api` skill before writing the Claude call. The model is asked for a `DeclaredSpec` as structured data only; the reply is validated through the Dev B adapter (when `check_declaration` is importable) and always against the skill catalogue. The key is read server-side from `.env`, never logged, never sent to the browser. Unsupported requests name the missing capability. Tests: catalogue validation and prompt assembly as pure functions; the live call is NOT RUN without a key and says so.

**S3. Runs, events, results.** `/api/runs` and `/api/runs/{id}` over the proposed event file and results JSON, behind the Dev B adapter; cost confirmation before any run that would call hosted TabPFN; 503 naming the missing pipeline entry point until Dev B publishes it. Tests: the event-file reader as a pure function (malformed lines, partial last line, unknown steps).

**S4. Scenarios, episodes, head stream.** After merging `person-a/lane-world` with W2-W4: `/api/scenarios`, `/api/episode`, `/api/stream?camera=head`. Tests run real episodes.

**S5. Live inference.** `/api/infer` behind the Dev B adapter: shape check against `image_window`, latency measured server-side, command applied to the stage through `validate_command`. 503 `infer_unavailable` until Dev B's adapter is importable.

### Front lane

Plain HTML, CSS and JavaScript modules in `app/static/`, no build step, no CDN, no web fonts. Pure logic lives in modules tested with `node --test tests/front`. Load the `design-taste-frontend` skill before designing.

**F1. Shell and Teach.** App shell with the areas Teach, Learn, Live; state from `/api/state` and `/api/events`; every unavailable capability rendered as unavailable with its reason. Teach: chat, the parsed `DeclaredSpec` with supported parameters and validation state, unsupported requests naming the missing capability, start-run with the cost confirmation.

**F2. Learn: before, during, after.** Measured baseline; the live stage (tracking and head views); the step timeline from run events (attempts, PFN fit, dreams, SNN training curve and spike activity); paired before/after metrics with n, seeds and intervals; RAMP `NORMAL`/`SLOPE` mode and DODGE gate states. Every number comes from an API response.

**F3. Live camera.** Explicit start button, `getUserMedia`, rolling window in memory, preprocessing from `image_window` (pure module, unit-tested, parity-tested against Dev B's vectors when they exist), `POST /api/infer`, the stage beside the camera preview, and the state readouts listed in the spec (action, confidences, margin, persistence gate, cooldown, frame age, latency). Nothing is stored; stopping the camera releases the stream.

### Integration (orchestrator)

Merge lanes into `person-a/go2-env` in the main checkout at each milestone, run `uv run pytest tests/env tests/app -q` and `node --test tests/front`, record evidence, write `LOG.md` / `DECISIONS.md` / `HANDOFF_PERSON_A.md`, push, post progress to `MESSAGES.md`. One fresh whole-branch review at the end.

## Review focus

- Any path by which hidden truth reaches the policy input or the browser's inference request.
- Any number on a screen that does not come from a run artifact or a live API response.
- The camera path: frames or pixel features written to disk, logs or a network destination other than `127.0.0.1`.
- The Claude call: key handling, and the model's reply treated as data and validated, never executed.
- Ground-contact and fall logic on the ramp and plateau.
