# Person A — Go2 MuJoCo Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A verified Go2 walking in headless MuJoCo behind the `Go2World` API, with a physical thrown ball, independently switchable hidden effects, five world presets, and an `ENV_REPORT.md` generated from saved runs.

**Architecture:** One frozen borrowed walking policy is wrapped by `WalkingPolicy`. `scene.py` composes Go2 + floor + payload + ball into one MuJoCo model. `Go2World` owns stepping, command validation and latency, seeded random streams, traces and measured outcomes. Every evidence run goes through one subprocess recorder, so command, exit code, output and versions are saved the same way, and reports are generated only from those files.

**Tech Stack:** Python 3.13, uv, `mujoco` (version pinned in Task 1), the policy runtime Task 1 selects (`onnxruntime` or CPU `torch`), numpy, PyYAML, pytest.

**Spec:** `05_PERSON_A.md` (authority), with `02_ENVIRONMENT.md`, `MISSION.md`, `01_PLAN.md` and `04_EXPERIMENTS.md` (E1–E6).

## Global Constraints

- The only robot is Unitree Go2 in MuJoCo with one verified borrowed walking policy, weights frozen. No alternate robot, controller, simulator, mock policy, hand-authored trace or kinematic animation.
- Try locomotion paths in this order and stop at the first that truly passes: Efferent `examples/go2_walk` (1 h), rl_sar `policy/go2/robot_lab/policy.pt` or `policy/go2/himloco/himloco.pt` (3 h), Quadruped-PyMPC (2 h). A timebox ending is not a pass. All fail: stop and report the blocker.
- Python `>=3.13,<3.14`, CPU only.
- Person A edits only `learned_judgment/env/**`, `config.yaml`, `app/**`, `ENV_REPORT.md`, plus new `tests/env/**` and `artifacts/env/**`. `pyproject.toml`, `LOG.md`, `DECISIONS.md` and `experiments/EXPERIMENTS_LOG.md` are shared: touch them only as the steps below say, and only with integrator approval. **Approval is pending (2026-11-21): do everything else first and stop before the first edit of any of these four files.** Until then, research findings are saved under `artifacts/env/research/` and proofs install their packages with `uv run --with`, which leaves `pyproject.toml` untouched.
- Execution covers all 8 tasks and is subagent-driven: one implementer and one task reviewer per task, then one whole-branch review. Every subagent and workflow agent runs on Opus 5.5 at extra-high effort (`model: 'opus'`, `effort: 'xhigh'`). The session lead coordinates, rules on conflicts and talks to the integrator; it does not write the code.
- Commits stay on the local branch `person-a/go2-env`. Nothing is pushed without asking.
- Research agents read sources and may install well-known PyPI packages in a scratch environment. They do not run code from a cloned repository or load a policy checkpoint: the proof command runs in Task 1 Step 3, after the files it downloads and executes have been named to the integrator.
- Every tunable lives in `config.yaml`.
- Before each slice, one bounded research question answered from primary sources with a proof-of-concept command and its observed output, logged in `experiments/EXPERIMENTS_LOG.md`. Research runs as multi-agent workflows (the integrator's choice for the prompts' "UltraCode" research agents): read-only researchers in parallel, then one verifier per researcher that re-reads every cited source and re-runs the proof command. A claim the verifier cannot confirm is dropped. There are three: W1 in Task 1, W2 in Task 3, W3 in Task 5. Tasks 4, 6 and 8 consume W1's simulator findings; a question those findings do not answer gets its own workflow before the slice. Each log entry names its workflow run ID.
- System tests run the actual Go2 model and the selected policy. Unexecuted checks are `NOT RUN`, never `PASS`. Report values come from saved run files, never typed.
- Hidden world values never appear in observations or in the outcome returned to Person B.
- Calibration uses dev runs only. Every change to a range or threshold after a run gets a dated `DECISIONS.md` entry.

## Values this plan fixes (the spec leaves them open)

These go into `config.yaml` before any run and are not widened to make a run pass.

| Key | Value | Used by |
|---|---|---|
| `robot.acceptance.speed_tolerance_fraction` | 0.2 | Task 2 |
| `robot.acceptance.yaw_tolerance_fraction` | 0.3 | Task 2 |
| `robot.acceptance.stop_speed_m_s` | 0.05 (mean planar speed over the last 2 s) | Task 2 |
| `robot.fall.max_tilt_deg` | 60 (or any trunk geom touching the floor) | Task 4 |
| `robot.recovery` | `max_ang_vel_rad_s: 0.5`, `max_tilt_deg: 10`, `hold_s: 0.5` | Task 5 |
| `world.settle_s` | 2.0 | Task 4 |
| `world.hidden_effects.command_latency_jitter_s` | 0.02 | Task 6 |
| `world.hidden_effects.friction_per_run_variation` | 0.1 (multiplicative, ±) | Task 6 |
| `world.hidden_effects.payload_off_center_range_m` | x ±0.10, y ±0.04, shrunk to the trunk geometry if it does not fit | Task 6 |
| Presets | `default`: 0 kg, `default` floor, 0°, 1.0, 0.05 s. `heavy_payload`: 3.0 kg. `slippery_floor`: `slippery` floor at 0.5 × the model's floor friction. `tired_motors`: 0.8. `downhill`: −5°. Each differs from `default` in that one field. | Task 6 |
| Throw calibration target | 30–70% collisions against a Go2 holding its initial command, on dev throws | Task 5 |

## File Structure

    learned_judgment/env/
      __init__.py     exports Go2World, CommandError, load_config, make_streams
      evidence.py     subprocess recorder: run.json + stdout.txt + stderr.txt
      policy.py       WalkingPolicy: frozen checkpoint -> joint torques
      commands.py     CommandError, validate_command
      scene.py        build_model: Go2 + floor + payload + ball
      world.py        Go2World, make_streams, latency queue, trace, outcomes
      acceptance.py   Phase 1 locomotion runs
      calibrate.py    throw calibration, effect sweeps, oracle
      report.py       ENV_REPORT.md generator
      assets/         pinned model + checkpoint with upstream LICENSE files
      README.md       handoff: schemas, units, limits
    app/hooks.py      record_episode for the dashboard
    tests/env/        one test file per task
    artifacts/env/    raw evidence, one directory per run
    ENV_REPORT.md     generated, never hand-edited

## Review Focus

- A `step` duration that is not a whole number of control periods: it is rounded to the nearest period (minimum one) and the returned `t_s` shows the time actually simulated. Pinned in Task 4.
- A fall or a diverging simulation in the middle of `step`: the episode keeps its measurements, sets `fallen` or `sim_unstable`, and never raises. Pinned in Task 6.
- A ball that bounces off the floor near the dog: only ball-to-robot contact counts as a collision. Pinned in Task 5.
- Calls out of order (`step` before `reset`, `finish_episode` twice) and unknown situation keys: `RuntimeError` / `ValueError` with the offending name. Pinned in Task 4.
- A sign error that still walks (lateral or yaw direction flipped): every command field is proven at both ends of its range with a signed measurement. Pinned in Task 3.

---

### Task 1: Select and pin the walking stack (E1, E2)

**Files:**
- Create: `learned_judgment/env/evidence.py`, `learned_judgment/env/assets/**`, `tests/env/conftest.py`, `tests/env/test_evidence.py`, `tests/env/test_policy_identity.py`
- Modify: `pyproject.toml` (add `mujoco`, the selected runtime, `pytest`), `config.yaml` (`runtime.timestep_s`, `robot.model_asset`, `robot.walking_policy.*`), `experiments/EXPERIMENTS_LOG.md`, `DECISIONS.md`

**Interfaces:**
- Produces: `python -m learned_judgment.env.evidence <label> -- <command...>` writing `artifacts/env/<UTC yyyymmddThhmmss>-<label>/{run.json,stdout.txt,stderr.txt}` and exiting with the child's exit code. `run.json` keys: `command`, `exit_code`, `started_utc`, `wall_s`, `python`, `platform`, `packages` (name -> version, every installed distribution), `git_rev`, `git_dirty`, `config_sha256`. The label must match `[A-Za-z0-9_-]+`. CPU usage is measured in-process by `run_locomotion`, not here.
- Produces: `config["robot"]["walking_policy"]` with `source`, `repository_revision`, `checkpoint`, `checkpoint_sha256`, `matching_config`, `license`, `observation_schema` (ordered list of `{name, dim, scale}`), `action_schema` (`dim: 12`, `joint_order`, `scale`, `default_pose`, `kp`, `kd`, `torque_limit`), `control_rate_hz`; `config["robot"]["model_asset"]` with `path`, `source`, `revision`, `sha256`.
- Produces: pytest fixture `config` in `tests/env/conftest.py` returning `load_config()` output, and marker `system` for tests that run the real simulator.

- [ ] **Step 1: Environment.** `git switch -c person-a/go2-env`, `uv python install 3.13`, `uv sync`.
  Run: `uv run python -V` — Expected: `Python 3.13.x`

- [ ] **Step 2: Research workflow W1.** Three read-only researchers in parallel. E1 and E2 answer "Can `<path>` drive a Unitree Go2 in MuJoCo on Windows, Python 3.13, CPU only?" for Efferent and rl_sar. M answers the simulator questions Tasks 4–8 depend on: which `mujoco` version has a Windows Python 3.13 wheel, the offscreen rendering backend on Windows, how to add a body and scale floor friction at build time, and whether repeated CPU runs are bit-identical. Then one verifier per researcher. Each researcher returns primary-source links, exact revision, license, model asset, checkpoint location, observation order and scales, action scale, default pose, PD gains, control rate, a proof-of-concept command for Step 3 with the output it should produce, every file that command downloads or executes (source, size, sha256, format), limitations, and pass/fail. Append entries E1, E2 and M to `experiments/EXPERIMENTS_LOG.md`.

- [ ] **Step 3: Run the proofs yourself, in spec order.** Selection rule: the first path whose proof loads the real checkpoint and keeps the Go2 upright for 5 s headless on this machine. Both fail: run the same workflow for E3 (note `cmake` is not installed here), and if that fails too, stop and report the blocker. Record the choice in `DECISIONS.md`.

- [ ] **Step 4: Write the failing tests**

```python
# tests/env/test_evidence.py
def test_records_command_and_exit_code(tmp_path):
    rc = subprocess.call([sys.executable, "-m", "learned_judgment.env.evidence", "t",
                          "--out", str(tmp_path), "--", sys.executable, "-c", "print('hi')"])
    run = json.loads(next(tmp_path.glob("*-t/run.json")).read_text())
    assert rc == 0 and run["exit_code"] == 0
    assert "hi" in next(tmp_path.glob("*-t/stdout.txt")).read_text()

def test_failing_child_propagates(tmp_path):
    rc = subprocess.call([sys.executable, "-m", "learned_judgment.env.evidence", "f",
                          "--out", str(tmp_path), "--", sys.executable, "-c", "raise SystemExit(3)"])
    assert rc == 3

# tests/env/test_policy_identity.py
def test_pinned_files_match_config(config):
    for entry, key in ((config["robot"]["model_asset"], "sha256"),
                       (config["robot"]["walking_policy"], "checkpoint_sha256")):
        path = entry.get("path") or entry["checkpoint"]
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == entry[key]

def test_model_has_twelve_actuators(config):
    assert mujoco.MjModel.from_xml_path(config["robot"]["model_asset"]["path"]).nu == 12
```

- [ ] **Step 5: Run them.** `uv run pytest tests/env/test_evidence.py tests/env/test_policy_identity.py -v` — Expected: FAIL (module and config values missing)

- [ ] **Step 6: Implement `evidence.py`; place the pinned assets under `learned_judgment/env/assets/<source>@<revision>/` with upstream LICENSE files untouched if the license allows redistribution (otherwise add `fetch_assets.py` that downloads the pinned revision and checks the sha256); fill the config keys from the verified source config, citing file and revision.**

- [ ] **Step 7: Run again.** Same command — Expected: PASS

- [ ] **Step 8: Commit.** `git add learned_judgment/env tests/env config.yaml pyproject.toml uv.lock experiments/EXPERIMENTS_LOG.md DECISIONS.md` / `git commit -m "env: pin verified Go2 model and walking policy"`

### Task 2: WalkingPolicy and the Phase 1 locomotion gate

**Files:**
- Create: `learned_judgment/env/policy.py`, `learned_judgment/env/acceptance.py`, `tests/env/test_locomotion.py`
- Modify: `config.yaml` (`robot.acceptance.*` from the table above)

**Interfaces:**
- Consumes: `config["robot"]["walking_policy"]`, `config["robot"]["model_asset"]`, the `config` fixture, `evidence`.
- Produces: `WalkingPolicy(policy_cfg: dict)` with `control_dt: float`, `reset() -> None`, `act(model, data, command: np.ndarray) -> np.ndarray` (shape `(12,)`, joint torques in N·m in model actuator order, PD computed from the source config's gains) and `last_obs: np.ndarray` (the exact vector fed to the network).
- Produces: `run_locomotion(config: dict, vx: float, yaw_rate: float, duration_s: float, seed: int) -> dict` with keys `fallen`, `mean_speed_m_s`, `mean_yaw_rate_rad_s`, `final_speed_m_s` (mean planar speed over the last 2 s), `rtf`, `cpu_fraction`, `trace_path`. Means exclude the first 5 s.
- Produces: `python -m learned_judgment.env.acceptance --out <dir>` writing `locomotion.json` (one entry per condition) and one `.npz` trace per condition.

- [ ] **Step 1: Write the failing tests**

```python
# tests/env/test_locomotion.py
pytestmark = pytest.mark.system

@pytest.mark.parametrize("vx", [0.3, 0.6, 0.9])
def test_trots_thirty_seconds(config, vx):
    r = run_locomotion(config, vx=vx, yaw_rate=0.0, duration_s=30, seed=config["runtime"]["seed"])
    assert not r["fallen"]
    assert abs(r["mean_speed_m_s"] - vx) <= config["robot"]["acceptance"]["speed_tolerance_fraction"] * vx

@pytest.mark.parametrize("sign", [1, -1])
def test_turns(config, sign):
    yaw = sign * 0.5 * max(config["robot"]["acceptance"]["yaw_rates_supported_by_policy"])
    r = run_locomotion(config, vx=0.3, yaw_rate=yaw, duration_s=10, seed=config["runtime"]["seed"])
    assert not r["fallen"]
    assert abs(r["mean_yaw_rate_rad_s"] - yaw) <= config["robot"]["acceptance"]["yaw_tolerance_fraction"] * abs(yaw)

def test_stops_and_stands(config):
    # 5 s at 0.6 m/s, then 5 s at zero command, in one episode
    r = run_stop(config, vx=0.6, trot_s=5, stop_s=5, seed=config["runtime"]["seed"])
    assert not r["fallen"]
    assert r["final_speed_m_s"] < config["robot"]["acceptance"]["stop_speed_m_s"]

def test_stands_thirty_seconds(config):
    assert not run_locomotion(config, vx=0.0, yaw_rate=0.0, duration_s=30, seed=config["runtime"]["seed"])["fallen"]
```

`run_stop(config, vx, trot_s, stop_s, seed) -> dict` lives in `acceptance.py` and returns the same keys as `run_locomotion`.

- [ ] **Step 2: Run them.** `uv run pytest tests/env/test_locomotion.py -v` — Expected: FAIL with ImportError

- [ ] **Step 3: Implement `WalkingPolicy` and `acceptance.py`.** Build the observation in exactly the order and scales of `observation_schema`; set `yaw_rates_supported_by_policy` from the source config's command range.

- [ ] **Step 4: Run again.** Same command — Expected: PASS, all seven. **This is the Phase 1 gate.** A fall or a missed tolerance means the adapter is wrong: use superpowers:systematic-debugging on joint order, observation order and scales, default pose, gains and control rate. Do not start Task 3 until it passes, and do not change a tolerance.

- [ ] **Step 5: Save the evidence.** `uv run python -m learned_judgment.env.evidence locomotion -- uv run python -m learned_judgment.env.acceptance --out artifacts/env/locomotion` — Expected: exit 0, `locomotion.json` holds seven entries each with a measured `rtf`.

- [ ] **Step 6: Commit.** `git add learned_judgment/env tests/env config.yaml artifacts/env` / `git commit -m "env: Go2 walking policy passes locomotion gate"`; append the run to `LOG.md`.

### Task 3: Command schema (E4) — first handoff to Person B

**Files:**
- Create: `learned_judgment/env/commands.py`, `tests/env/test_commands.py`
- Modify: `config.yaml` (`robot.walking_policy.command_schema`), `learned_judgment/env/acceptance.py` (add `prove_commands`), `experiments/EXPERIMENTS_LOG.md`

**Interfaces:**
- Consumes: `run_locomotion`, `WalkingPolicy`.
- Produces: `config["robot"]["walking_policy"]["command_schema"] = {control_period_s, source, fields: {<name>: {native_index, unit, min, max, proven_by}}}`, field names and ranges copied from the selected policy's source config.
- Produces: `class CommandError(ValueError)`; `validate_command(command: dict, schema: dict) -> np.ndarray` (native order). Raises `CommandError` naming the field for an unknown field, a missing field, a non-finite value, or a value outside `[min, max]`. It never clips.
- Produces: `prove_commands(config, out_dir) -> Path` running each field at `min` and at `max` for 10 s and saving the signed measured response to `commands.json`.

- [ ] **Step 1: Research workflow W2 (E4).** One researcher extracts every command field's name, unit, legal range, update rate and native index from the selected policy's source config at the pinned revision, citing file and line; one verifier re-reads those lines. Log as E4.

- [ ] **Step 2: Write the failing tests**

```python
# tests/env/test_commands.py
def test_edges_are_accepted_unchanged(schema):
    for name, f in schema["fields"].items():
        cmd = {n: 0.0 for n in schema["fields"]} | {name: f["max"]}
        assert validate_command(cmd, schema)[f["native_index"]] == f["max"]

@pytest.mark.parametrize("mutate", [
    lambda c, s: c | {"jump": 1.0},
    lambda c, s: {k: v for k, v in list(c.items())[1:]},
    lambda c, s: c | {next(iter(c)): float("nan")},
    lambda c, s: c | {next(iter(c)): s["fields"][next(iter(c))]["max"] + 1e-6},
])
def test_rejects_instead_of_clipping(schema, mutate):
    zero = {n: 0.0 for n in schema["fields"]}
    with pytest.raises(CommandError):
        validate_command(mutate(zero, schema), schema)

@pytest.mark.system
def test_every_field_moves_the_dog_the_signed_way(config, tmp_path):
    proof = json.loads(prove_commands(config, tmp_path).read_text())
    for name, f in config["robot"]["walking_policy"]["command_schema"]["fields"].items():
        assert proof[name]["at_max"]["measured"] > 0 > proof[name]["at_min"]["measured"]
        assert not proof[name]["at_max"]["fallen"] and not proof[name]["at_min"]["fallen"]
```

`schema` is a fixture returning `config["robot"]["walking_policy"]["command_schema"]`.

- [ ] **Step 3: Run them.** `uv run pytest tests/env/test_commands.py -v` — Expected: FAIL with ImportError

- [ ] **Step 4: Implement.** A range end where the dog falls is narrowed to the largest value that passes, with a `DECISIONS.md` entry; the schema publishes only proven ranges.

- [ ] **Step 5: Run again.** Same command — Expected: PASS

- [ ] **Step 6: Evidence and handoff.** `uv run python -m learned_judgment.env.evidence commands -- uv run pytest tests/env/test_commands.py -v`; add the proof artifact path to the E4 log entry; send Person B the `command_schema` block and agree the situation, observation and outcome key names listed in Task 4 before freezing them.

- [ ] **Step 7: Commit.** `git commit -m "env: verified command schema and strict validation"`

### Task 4: Go2World API, seeded streams, determinism — second handoff

**Files:**
- Create: `learned_judgment/env/scene.py`, `learned_judgment/env/world.py`, `tests/env/test_world_api.py`, `tests/env/test_determinism.py`
- Modify: `learned_judgment/env/__init__.py`, `config.yaml` (`world.active_preset`, `world.settle_s`, `robot.fall`)

**Interfaces:**
- Consumes: `WalkingPolicy`, `validate_command`, `CommandError`.
- Produces: `load_config(path="config.yaml") -> dict`; `make_streams(seed: int) -> dict[str, np.random.Generator]` with keys `situation`, `world`, `ball`, `noise`; `build_model(config: dict, hidden: dict) -> mujoco.MjModel`.
- Produces: `Go2World(config: dict, seed: int)` with `reset(situation: dict | None = None) -> dict`, `step(command: dict, duration_s: float) -> dict`, `finish_episode() -> dict`, `render_frame() -> np.ndarray` (H×W×3 uint8), `command_schema: dict`.
- Situation keys (all optional): `initial_command` (command dict held during `settle_s`), `settle_s`, `ball_throw` (Task 5), `noise_key: int` (per-episode noise is drawn from `(seed, noise_key)`; absent means the episode counter).
- Observation keys (world frame, z up, SI units, quaternion w-x-y-z, one observation per control period): `t_s` (0 at the end of settling), `base_pos_m`, `base_quat_wxyz`, `base_lin_vel_m_s`, `base_ang_vel_rad_s`, `joint_pos_rad`, `joint_vel_rad_s`, `policy_obs`, `foot_contact` (4 bools, model foot order), `fallen`, `ball` (`None` until Task 5), `command_active` (what the policy is receiving now), `declared_world`.
- Outcome keys: `elapsed_s`, `fallen`, `fall_time_s`, `sim_unstable`, `commands_received` (list of `(t_s, command)`), `final_base_pos_m`, `declared_world`, `trace_path` (`.npz` with every observation array per control period). Hidden values are written only to `<trace>.hidden.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/env/test_world_api.py
pytestmark = pytest.mark.system
OBS_KEYS = {"t_s", "base_pos_m", "base_quat_wxyz", "base_lin_vel_m_s", "base_ang_vel_rad_s", "joint_pos_rad",
            "joint_vel_rad_s", "policy_obs", "foot_contact", "fallen", "ball", "command_active", "declared_world"}

def test_reset_returns_documented_observation(world):
    obs = world.reset()
    assert set(obs) == OBS_KEYS and obs["t_s"] == 0.0 and obs["joint_pos_rad"].shape == (12,)

def test_step_rounds_to_control_periods(world, zero, period):
    world.reset()
    assert world.step(zero, 25 * period)["t_s"] == pytest.approx(25 * period)
    assert world.step(zero, 0.6 * period)["t_s"] == pytest.approx(26 * period)

def test_calls_out_of_order_raise(world, zero):
    with pytest.raises(RuntimeError):
        world.step(zero, 0.1)
    world.reset(); world.finish_episode()
    with pytest.raises(RuntimeError):
        world.finish_episode()

def test_unknown_names_raise(world, zero):
    with pytest.raises(ValueError, match="wind"):
        world.reset({"wind": 1})
    world.reset()
    with pytest.raises(CommandError, match="jump"):
        world.step(zero | {"jump": 1.0}, 0.1)

# tests/env/test_determinism.py
def test_streams_are_independent_and_repeatable():
    a, b = make_streams(7), make_streams(7)
    a["ball"].random(1000)
    assert a["noise"].random() == b["noise"].random()

@pytest.mark.system
def test_fixed_seed_reproduces_the_episode(config, forward):
    def run():
        w = Go2World(config, seed=20261121); w.reset({"noise_key": 3}); w.step(forward, 3.0)
        return dict(np.load(w.finish_episode()["trace_path"]))
    first, second = run(), run()
    assert first.keys() == second.keys() and all(np.array_equal(first[k], second[k]) for k in first)
```

Fixtures in `conftest.py`: `world` (`Go2World(config, config["runtime"]["seed"])`), `zero` (all fields 0.0), `forward` (zero with the forward-speed field at 0.6), `period` (`command_schema["control_period_s"]`).

- [ ] **Step 2: Run them.** `uv run pytest tests/env/test_world_api.py tests/env/test_determinism.py -v` — Expected: FAIL with ImportError

- [ ] **Step 3: Implement.** Single-threaded inference so repeated runs are bit-identical. A fall is a trunk geom touching the floor or tilt above `robot.fall.max_tilt_deg`.

- [ ] **Step 4: Run again.** Same command — Expected: PASS

- [ ] **Step 5: Handoff.** Tell Person B the import path `from learned_judgment.env import Go2World, CommandError, load_config, make_streams` and that ball and hidden effects follow.

- [ ] **Step 6: Commit.** `git commit -m "env: Go2World reset/step/finish_episode with reproducible streams"`

### Task 5: Ball scene and DODGE measurements (E6) — third handoff

**Files:**
- Create: `learned_judgment/env/calibrate.py`, `tests/env/test_ball.py`
- Modify: `learned_judgment/env/scene.py`, `learned_judgment/env/world.py`, `config.yaml` (`ball.*`, `robot.recovery`), `experiments/EXPERIMENTS_LOG.md`

**Interfaces:**
- Consumes: `Go2World`, `make_streams`.
- Produces: situation key `ball_throw = {"pos_m": [x, y, z], "vel_m_s": [vx, vy, vz], "release_time_s": float}` in the world frame; the ball is a free MuJoCo body with `ball.mass_kg` and `ball.radius_m` and is never touched after release.
- Produces: `Go2World.sample_ball_throw() -> dict` drawing from the `ball` stream within `ball.throw_origin_range_m`, `ball.throw_speed_range_m_s`, `ball.throw_direction_range_deg`, aimed relative to the dog's pose at reset.
- Produces: observation `ball = {"pos_m", "vel_m_s", "radius_m", "released"}`; outcome keys `ball_collision` (ball-to-robot contact only), `first_collision_time_s`, `min_separation_m` (surface to surface, minimum over every physics step after release, ≤ 0 on contact), `recovery_time_s` (from first ball contact until angular speed and tilt stay under `robot.recovery` limits for `hold_s`; `None` without contact), `recovered`; trace arrays `ball_pos_m`, `ball_vel_m_s`.
- Produces: `calibrate_throws(config, seed, n, out_dir) -> Path` saving `throws.json` with each throw and its measured outcome against a dog holding its initial command.

- [ ] **Step 1: Research workflow W3 (E6).** One researcher and one verifier: a standardised ball's mass and radius from a primary source; whether the pinned `mujoco` exposes `mj_geomDistance`; the model's collision geom groups. Log as E6.

- [ ] **Step 2: Write the failing tests**

```python
# tests/env/test_ball.py
pytestmark = pytest.mark.system

def throw_at(world, lateral_offset_m):
    """Straight throw from 3 m to the dog's left, aimed lateral_offset_m ahead of the trunk centre."""

def test_centre_throw_hits_standing_dog(world, zero):
    world.reset({"ball_throw": throw_at(world, 0.0)}); world.step(zero, 3.0)
    out = world.finish_episode()
    assert out["ball_collision"] and out["min_separation_m"] <= 0 and out["first_collision_time_s"] > 0

def test_wide_throw_misses_even_after_bouncing(world, zero):
    world.reset({"ball_throw": throw_at(world, 1.5)}); world.step(zero, 3.0)
    out = world.finish_episode()
    assert not out["ball_collision"] and out["min_separation_m"] > 0 and out["recovery_time_s"] is None

def test_ball_ignores_the_chosen_command(config, zero, forward):
    def flight(cmd):
        w = Go2World(config, seed=11); throw = w.sample_ball_throw()
        w.reset({"ball_throw": throw, "noise_key": 0}); w.step(cmd, throw["release_time_s"] + 0.1)
        return throw, np.load(w.finish_episode()["trace_path"])["ball_pos_m"]
    (t1, p1), (t2, p2) = flight(zero), flight(forward)
    assert t1 == t2 and np.array_equal(p1, p2)

def test_throw_distribution_has_hits_and_misses(config, tmp_path):
    throws = json.loads(calibrate_throws(config, seed=11, n=40, out_dir=tmp_path).read_text())
    assert 0.3 <= sum(t["ball_collision"] for t in throws) / 40 <= 0.7

def test_some_supported_command_avoids_a_hit(config, tmp_path):
    # first 10 dev throws that hit a dog holding its initial command; a small grid of schema-legal commands
    assert avoided_count(config, tmp_path) >= 1
```

`throw_at` and `avoided_count` are test helpers in the same file; the grid is the two range ends of each command field.

- [ ] **Step 3: Run them.** `uv run pytest tests/env/test_ball.py -v` — Expected: FAIL (ball keys missing)

- [ ] **Step 4: Implement.** Starting ranges: origin 3–5 m from the dog, speed and direction chosen so flight time is 0.5–1.5 s; adjust only from `calibrate_throws` output on dev seeds until the 30–70% target holds, logging each change in `DECISIONS.md`.

- [ ] **Step 5: Run again.** Same command — Expected: PASS

- [ ] **Step 6: Evidence and handoff.** `uv run python -m learned_judgment.env.evidence dodge -- uv run pytest tests/env/test_ball.py -v`; give Person B one saved DODGE episode (trace path, situation, commands, outcome).

- [ ] **Step 7: Commit.** `git commit -m "env: physical ball, throw protocol and DODGE measurements"`

### Task 6: Hidden effects and presets (E5, first half)

**Files:**
- Create: `tests/env/test_effects.py`
- Modify: `learned_judgment/env/scene.py`, `learned_judgment/env/world.py`, `learned_judgment/env/calibrate.py`, `config.yaml` (`world.hidden_effects.*`, `world.presets.*`)

**Interfaces:**
- Consumes: `Go2World`, `build_model`, `run_locomotion`.
- Produces: preset fields `payload_mass_kg`, `payload_offset_m`, `floor_type`, `slope_deg` (positive is uphill along the dog's heading), `motor_strength_fraction`, `command_latency_s`; `world.hidden_effects.enabled.<effect>: bool` per effect (`payload`, `friction`, `slope`, `motor_strength`, `latency`); `declared_world = {"payload_mass_kg": true mass × (1 + error drawn from the `world` stream within ±0.2), "floor_type": ...}`.
- Produces: `measure_effects(config, seed, out_dir) -> Path` saving `effects.json` with baseline and one-effect-on measurements: steady speed at 0.6 m/s commanded (payload 3 kg, slope +5°, motor 0.8), stopping distance from 0.9 m/s (floor `slippery`), time from command to half the commanded speed (latency 0.3 s).

- [ ] **Step 1: Write the failing tests**

```python
# tests/env/test_effects.py
pytestmark = pytest.mark.system
EFFECTS = ["payload", "friction", "slope", "motor_strength", "latency"]

def test_all_effects_off_equals_baseline(config, tmp_path):
    e = json.loads(measure_effects(config, seed=5, out_dir=tmp_path).read_text())
    assert e["all_off"] == e["baseline"]

@pytest.mark.parametrize("effect", EFFECTS)
def test_one_effect_changes_only_its_measurement(config, tmp_path, effect):
    e = json.loads(measure_effects(config, seed=5, out_dir=tmp_path).read_text())
    assert e[effect]["measured"] != e["baseline"][e[effect]["quantity"]]
    assert e[effect]["hidden"].keys() == {effect}

@pytest.mark.parametrize("preset", ["default", "heavy_payload", "slippery_floor", "tired_motors", "downhill"])
def test_preset_loads_and_stands(config, zero, preset):
    cfg = copy.deepcopy(config); cfg["world"]["active_preset"] = preset
    w = Go2World(cfg, seed=5); w.reset(); w.step(zero, 1.0)
    assert not w.finish_episode()["fallen"]

def test_hidden_values_never_reach_person_b(config, zero):
    cfg = copy.deepcopy(config); cfg["world"]["active_preset"] = "heavy_payload"
    w = Go2World(cfg, seed=5); obs = w.reset(); out = w.finish_episode()
    leaked = {"payload_offset_m", "motor_strength_fraction", "command_latency_s", "friction", "slope_deg"}
    assert not leaked & (set(obs) | set(out) | set(obs["declared_world"]))

def test_dead_motors_fall_is_measured_not_raised(config, zero):
    # 0.0 is outside the 0.8-1.0 experiment range on purpose: presets are not range-checked, so a real fall can be produced
    cfg = copy.deepcopy(config); cfg["world"]["presets"]["default"]["motor_strength_fraction"] = 0.0
    w = Go2World(cfg, seed=5); w.reset({"settle_s": 0.0}); w.step(zero, 3.0)
    out = w.finish_episode()
    assert out["fallen"] and out["fall_time_s"] is not None
```

- [ ] **Step 2: Run them.** `uv run pytest tests/env/test_effects.py -v` — Expected: FAIL (`measure_effects` missing)

- [ ] **Step 3: Implement.** Payload is a box body on the trunk; friction scales the floor geom; latency is a command queue with per-episode jitter from the noise stream, in whole control periods; motor strength scales torque and torque limit. Slope tilts gravity rather than the floor (`# shortcut: exact only for an infinite flat floor, replace with ramp geometry when RAMP needs a visible incline`).

- [ ] **Step 4: Run again.** Same command — Expected: PASS

- [ ] **Step 5: Evidence.** `uv run python -m learned_judgment.env.evidence effects -- uv run python -c "from learned_judgment.env import load_config; from learned_judgment.env.calibrate import measure_effects; measure_effects(load_config(), 5, 'artifacts/env/effects')"`; log as E5.

- [ ] **Step 6: Commit.** `git commit -m "env: switchable hidden effects and five world presets"`

### Task 7: Oracle, calibration and ENV_REPORT.md (E5, second half)

**Files:**
- Create: `learned_judgment/env/report.py`, `tests/env/test_report.py`, `ENV_REPORT.md` (generated)
- Modify: `learned_judgment/env/calibrate.py`

**Interfaces:**
- Consumes: every `artifacts/env/*/run.json`, `locomotion.json`, `commands.json`, `throws.json`, `effects.json`; Person B's SkillSpec (`name`, `physics_guess(situation, action)`, `episode(world, situation, action)`, candidate actions).
- Produces: `oracle(config, skill_spec, situations: list[dict], n_noise: int, out_dir) -> Path` saving `oracle_<skill>.json`: per situation, each candidate action's measured outcomes over `n_noise` noise keys, the best action, its remaining error (the noise floor), and the physics-only action's error. `n_noise` and the situation count come from `evaluation.oracle_situations_per_skill` and are sized from the measured `rtf`.
- Produces: `python -m learned_judgment.env.report [--artifacts artifacts/env] [--out ENV_REPORT.md]` writing one row per acceptance criterion of `05_PERSON_A.md` with `PASS`/`FAIL` taken from a saved `exit_code`, or `NOT RUN` when no artifact exists, plus versions, seeds, commands, artifact paths, per-skill physics-only error, noise floor and their ratio.

- [ ] **Step 1: Write the failing tests** (pure utility: a hand-made `run.json` checks the generator only and never reaches the real report)

```python
# tests/env/test_report.py
def test_empty_artifacts_reports_not_run(tmp_path):
    out = tmp_path / "R.md"
    subprocess.check_call([sys.executable, "-m", "learned_judgment.env.report",
                           "--artifacts", str(tmp_path), "--out", str(out)])
    text = out.read_text()
    assert "NOT RUN" in text and "PASS" not in text

def test_status_comes_from_saved_exit_code(tmp_path):
    for label, code in (("locomotion", 0), ("commands", 1)):
        d = tmp_path / f"20261121T000000-{label}"; d.mkdir()
        (d / "run.json").write_text(json.dumps({"command": ["x"], "exit_code": code, "packages": {}}))
    out = tmp_path / "R.md"
    subprocess.check_call([sys.executable, "-m", "learned_judgment.env.report",
                           "--artifacts", str(tmp_path), "--out", str(out)])
    lines = out.read_text().splitlines()
    assert any("locomotion" in l and "PASS" in l for l in lines)
    assert any("commands" in l and "FAIL" in l for l in lines)
```

- [ ] **Step 2: Run them.** `uv run pytest tests/env/test_report.py -v` — Expected: FAIL (no module)

- [ ] **Step 3: Implement `report.py` and `oracle`.**

- [ ] **Step 4: Run again.** Same command — Expected: PASS

- [ ] **Step 5: Run the DODGE calibration** once Person B's DODGE SkillSpec exists: `uv run python -m learned_judgment.env.evidence oracle-dodge -- uv run python -m learned_judgment.env.calibrate oracle dodge`. Without the SkillSpec the per-skill rows stay `NOT RUN` with the blocker named. If physics-only error is under 3× the noise floor, investigate the simulator and log every adjustment; never edit the report.

- [ ] **Step 6: Generate and commit.** `uv run python -m learned_judgment.env.report` / `git commit -m "env: generated ENV_REPORT from saved runs"`

### Task 8: Dashboard hooks, handoff and fresh-setup check

**Files:**
- Create: `app/hooks.py`, `learned_judgment/env/README.md`, `tests/env/test_hooks.py`
- Modify: `LOG.md`

**Interfaces:**
- Consumes: `Go2World`.
- Produces: `record_episode(config: dict, seed: int, situation: dict | None, commands: list[tuple[dict, float]], fps: int = 25) -> dict` with `frames` (`np.ndarray`, T×H×W×3 uint8), `observations` (one per command) and `outcome`.

- [ ] **Step 1: Write the failing test**

```python
# tests/env/test_hooks.py
@pytest.mark.system
def test_records_real_frames(config, forward):
    r = record_episode(config, seed=5, situation=None, commands=[(forward, 2.0)])
    assert r["frames"].shape[0] == 50 and r["frames"].shape[-1] == 3
    assert not np.array_equal(r["frames"][0], r["frames"][-1])
    assert not r["outcome"]["fallen"]
```

- [ ] **Step 2: Run it.** `uv run pytest tests/env/test_hooks.py -v` — Expected: FAIL with ImportError

- [ ] **Step 3: Implement `record_episode`.** If offscreen rendering has no GL context on this machine, `render_frame` raises `RuntimeError` naming the backend; fix the backend, do not substitute images.

- [ ] **Step 4: Run again.** Same command — Expected: PASS

- [ ] **Step 5: Write `learned_judgment/env/README.md`:** import path, command schema, observation and outcome keys with units, frames and update rate, supported ranges, policy and simulator identities, the saved example episode, limitations.

- [ ] **Step 6: Fresh-setup check.** In a new clone of the branch: `uv sync`, then `uv run python -m learned_judgment.env.evidence fresh -- uv run pytest tests/env -v`. Expected: exit 0 with no manual edits.

- [ ] **Step 7: Regenerate the report and commit.** `uv run python -m learned_judgment.env.report` / `git commit -m "env: dashboard hooks, handoff notes, fresh-setup evidence"`

## Not in this plan

Gate posts for TURN, ramp geometry for RAMP, a kick target, parallel episode execution, video encoding and any dashboard UI. Each is added when Person B reaches the skill that needs it.
