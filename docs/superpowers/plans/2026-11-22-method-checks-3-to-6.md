# Method Checks 3 to 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what limits the punch dodge (a ceiling and a conv-net diagnostic) and try two changes to how the teacher and the dream are fed (recorded RAMP situations, scene truth for the punch teacher), each compared against today's behaviour.

**Architecture:** Everything is experimental and lives on the new branch `wip-dev-a` (Person A, 2026-11-22: "Execute all this on a wip-dev-a branch, this will be experimental, later we decide what we merge to main"). Two tasks are additive measurements that change no existing behaviour. Two change the learning pipeline, each behind a config value whose old setting reproduces today's arrays exactly, so both settings can be run on the same seeds and compared. Nothing reaches the student that it does not get today.

**Tech Stack:** Python 3, NumPy, PyTorch 2.15 (CPU), MuJoCo through `Go2World`, hosted TabPFN through `learned_judgment.teacher`, pytest.

**Spec:** The method review of 2026-11-22 in this session, points 3 to 6, restated under Requirements below, and Person A's branch rule: "if the features are incremental add it to the main dev-a branch we use, if the features change architectural behaviour, we create a separate wip-dev-a branch to cross verify and compare". Project rules: `CLAUDE.md`, `MISSION.md`.

## Requirements

- **R3. RAMP dream from recorded situations.** Today `skills/ramp.py` `collect_dream` draws the four pitch features uniformly and independently. Draw them instead from the feature vectors the simulated dog measured in its attempts, with small noise. "Recorded" means recorded in our MuJoCo runs.
- **R4. Punch teacher on scene truth.** Today the teacher's table for `dodge_vision` is 1,152 pixel columns, the action and the formula's guess. Replace the pixel columns with a few numbers the simulator knows (is there an object, time until it arrives, where it is aimed, how fast). The student's input stays the image window.
- **R5. Punch ceiling.** Count the punches that cannot be dodged: let the simulator try every action at the first moment a decision is possible and see whether any avoids the hit. Then state the student's result against the avoidable ones.
- **R6. Conv-net diagnostic.** Train a plain small conv net on the same dream windows and targets as the spiking student, replay it on the same dev trace through the same gate, and compare recall. It never becomes a student and never runs in the app.

## Global Constraints

- Branch: every task lands on `wip-dev-a`, created from `person-a/go2-env` at `a323da7`, integration worktree `.claude/worktrees/wip-dev-a`. Each task is built by its own agent in its own worktree and merged into `wip-dev-a` by the orchestrator. Nothing is merged into `person-a/go2-env` or `main` until Person A decides.
- The files touched are Person B's (`dodge_eval.py`, `observed.py`, `skills/ramp.py`, `skills/dodge_vision.py`, `run_skill.py`). Tell Person B on `talkhere.md` before the first commit and after each task. New tests go in new files under `tests/b/`; do not edit Person B's test files.
- Every tunable is a key in `config.yaml` with a one-line comment. Values named in this plan are starting values.
- Hidden world values (payload, friction, slope, motor strength, latency) never become a teacher feature. Scene truth of the thrown object is allowed on the teacher side and never becomes a student input.
- Fixtures only in unit tests of pure functions. Every number in a report comes from a run on the real simulator and real artifacts. An unexecuted check is labelled NOT RUN.
- No test may start a learning run or send a billed TabPFN request. A billed step is run by hand, after `observed quote` and Person A's go-ahead. The key is read as `PRIOR_LABS_API_KEY` from the environment or `.env` at the repository root (`prior_labs.py`); never print or log it.
- Thresholds are chosen on dev only. The held-out punch test set (240 trajectories) has been looked at once for `v4`; Tasks 2 and 4 use dev data only.
- Record evidence with `uv run python -m learned_judgment.env.evidence <label> -- <command>`. Write `LOG.md` and `DECISIONS.md` entries at the end of each task.
- Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Data the runs need

`results/runs/v4/` in git holds only `result.json`, `closed_loop_test.json`, `sweep.json`, `events.jsonl` and `student/`. Tasks 2 and 4 also need `results/runs/v4/context/`, `results/runs/v4/dream/`, `results/runs/v4/scores.npz` and `results/runs/v4/trace_dev.npz`, which exist on Person B's machine (`D:/BerlinPrior-eval/`).

Decision: ask Person B for those four (a copy outside git, or Person B runs the two commands of Tasks 2 and 4 there). Regenerating them here means a billed teacher request and is done only on Person A's word; the commands are `python -m learned_judgment.skills.dodge_vision rows --split context --out results/runs/v4/context`, `... dream --out results/runs/v4/dream --render kinematic`, `python -m learned_judgment.dodge_eval trace --run v4`, `python -m learned_judgment.observed teach --run v4 --skill dodge_vision --context results/runs/v4/context --dream results/runs/v4/dream --confirm-billing`. Task 1 needs none of them.

## Setup

- Before the first commit, post to Person B on `talkhere.md` (branch `messages`): which of their files Tasks 1 to 4 touch, that Tasks 3 and 4 live on `wip-dev-a`, and the request for the four `v4` files.
- `person-a/go2-env` is local at Person B's `a323da7`, ahead of `origin/person-a/go2-env`. Pushing it is Person A's call; the tasks only commit.
- `git worktree add .claude/worktrees/wip-dev-a -b wip-dev-a person-a/go2-env` before Task 1. Tasks 1 to 4 run in parallel in agent worktrees; Task 5 runs in the integration worktree after they are merged.

## Review Focus

- A hit where every dodge makes the dog fall: it must count as unavoidable, not as avoided (Task 1, Step 1).
- A ceiling record missing an action because an episode crashed: refuse with the scenario id, never count it silently (Task 1, Step 1).
- Dream rows of one scenario on both sides of the conv net's validation split: the validation number would flatter it (Task 2, Step 1).
- A full RAMP run has one row per candidate per scenario: drawing rows instead of scenarios would weight scenarios by candidate count; noise must not leave [0, 1] (Task 3, Step 1).
- An object that never arrives (static, receding, sideways, or no object): truth values must be finite, with no division by zero (Task 4, Step 1).

---

### Task 1: Punch ceiling

**Files:**
- Modify: `learned_judgment/dodge_eval.py` (new functions after `evaluate`, one parser entry in `_parser`, one branch in `main`)
- Test: `tests/b/test_dodge_eval_ceiling.py` (new)

**Interfaces:**
- Consumes: `dodge_eval._plan`, `_run`, `_episode`, `_WORKER`, `_merge_result`, `_write_json`, `_scenario_count`; `skills.dodge_vision.actions` and `skills.dodge_vision.hold_s` from config.
- Produces:
  - `first_issue_policy(action: str, command: Mapping[str, float], zero: Mapping[str, float], no_op: str, hold_s: float) -> Callable[[np.ndarray, float], dict[str, Any]]`
  - `ceiling_summary(records: Sequence[Mapping[str, Any]], action_names: Sequence[str], no_op: str) -> dict[str, Any]`
  - `student_against_ceiling(ceiling_records: Sequence[Mapping[str, Any]], student_records: Sequence[Mapping[str, Any]], no_op: str) -> dict[str, Any]`
  - `ceiling(config: Mapping[str, Any], run_id: str, split: str = "test", n: int | None = None, argv: Sequence[str] | None = None) -> dict[str, Any]`
  - Command `python -m learned_judgment.dodge_eval ceiling --run RUN [--split {dev,test}] [--n N]`, file `<observed.runs_dir>/<RUN>/ceiling_<split>.json`, section `ceiling_<split>` in `result.json`.
  - A ceiling record: `{"scenario_id": int, "outcomes": {action_name: {"collision": float, "fall": float, "min_separation_m": float, "first_collision_time_s": float | None}}}`. `NO_OP` is always present; the active actions are present exactly when `NO_OP` collided.

- [ ] **Step 1: Write the failing tests**

```python
ZERO = {"vel_x": 0.0, "vel_y": 0.0, "yaw_rate": 0.0}
LEFT = {"vel_x": 0.0, "vel_y": 0.8, "yaw_rate": 0.0}
NAMES = ["NO_OP", "DODGE_LEFT", "DODGE_RIGHT", "RETREAT"]
HIT, MISS, FELL = {"collision": 1.0, "fall": 0.0}, {"collision": 0.0, "fall": 0.0}, {"collision": 0.0, "fall": 1.0}

def test_first_issue_policy_issues_once_holds_then_stops():
    policy = first_issue_policy("DODGE_LEFT", LEFT, ZERO, "NO_OP", hold_s=0.6)
    w = np.zeros((8, 12, 12), np.float32)
    first = policy(w, 0.233)
    assert first["action"] == "DODGE_LEFT" and first["issued"] is True and first["command"] == LEFT
    held = policy(w, 0.80)
    assert held["command"] == LEFT and held.get("issued") is not True
    after = policy(w, 0.84)  # 0.233 + 0.6 = 0.833
    assert after["command"] == ZERO and after["action"] == "NO_OP"

def test_ceiling_summary_counts_avoidable_and_unavoidable():
    records = [
        {"scenario_id": 0, "outcomes": {"NO_OP": MISS}},
        {"scenario_id": 1, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": MISS, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
        {"scenario_id": 2, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": HIT, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
        {"scenario_id": 3, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": FELL, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
    ]
    s = ceiling_summary(records, NAMES, "NO_OP")
    assert (s["n_scenarios"], s["n_hit_no_op"], s["n_avoidable"], s["n_unavoidable"]) == (4, 3, 1, 2)
    assert s["avoidable_by_action"] == {"DODGE_LEFT": 1, "DODGE_RIGHT": 0, "RETREAT": 0}
    assert s["unavoidable_ids"] == [2, 3]  # 3: the only clean miss was a fall

def test_ceiling_summary_refuses_an_incomplete_record():
    with pytest.raises(ValueError, match="scenario 5"):
        ceiling_summary([{"scenario_id": 5, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": MISS}}], NAMES, "NO_OP")
    with pytest.raises(ValueError, match="scenario 6"):
        ceiling_summary([{"scenario_id": 6, "outcomes": {"DODGE_LEFT": MISS}}], NAMES, "NO_OP")

def test_student_against_ceiling_uses_the_avoidable_ones_as_denominator():
    ceiling_records = [
        {"scenario_id": 1, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": MISS, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
        {"scenario_id": 2, "outcomes": {"NO_OP": HIT, "DODGE_LEFT": HIT, "DODGE_RIGHT": HIT, "RETREAT": HIT}},
    ]
    student = [{"scenario_id": 1, "outcomes": {"no_op": HIT, "snn": MISS}}, {"scenario_id": 2, "outcomes": {"no_op": HIT, "snn": HIT}}]
    r = student_against_ceiling(ceiling_records, student, "NO_OP")
    assert (r["n_avoidable"], r["student_avoided_of_avoidable"], r["student_hit_unavoidable"]) == (1, 1, 1)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/b/test_dodge_eval_ceiling.py -q -p no:cacheprovider`
Expected: FAIL at import, `cannot import name 'first_issue_policy'`.

- [ ] **Step 3: Implement the three pure functions in `learned_judgment/dodge_eval.py`**

`first_issue_policy` remembers the time of its first call; it returns the action's command with `"issued": True` on that call, the same command without `issued` while `t_s < first + hold_s`, then `zero` with `no_op`. An action "avoids" a hit when its `collision` is 0 and its `fall` is 0. `student_against_ceiling` matches records by `scenario_id` and reads the student's `outcomes["snn"]` (the shape `_evaluate_task` writes into `closed_loop_test.json` `records`); a scenario present on one side only raises `ValueError`.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/b/test_dodge_eval_ceiling.py -q -p no:cacheprovider`
Expected: `4 passed`.

- [ ] **Step 5: Implement `ceiling` and its command**

`_ceiling_task(task)` runs the `NO_OP` episode of one scenario through `_episode`; when it collided, one more episode per active action with `first_issue_policy` (the first policy call is the first full window). `ceiling` plans the split with `_plan`, runs the tasks with `_run(config, None, None, _ceiling_task, tasks)`, and writes the records, `ceiling_summary`, the config digest, and, when `<run>/closed_loop_<split>.json` exists, `student_against_ceiling` on its records. Add this line above `first_issue_policy`:

```python
# shortcut: the oracle acts at the first full window only; a later decision could avoid more, add a time grid if the ceiling looks too low.
```

- [ ] **Step 6: Run it on the real simulator**

Run: `uv run python -m learned_judgment.env.evidence punch-ceiling -- uv run python -m learned_judgment.dodge_eval ceiling --run v4 --split test`
Expected: exit 0; `results/runs/v4/ceiling_test.json` with `n_scenarios` 240. Compare its `config_digest` with `results/runs/v4/closed_loop_test.json` `config_digest`. When they are equal, `n_hit_no_op` must be 121 (v4's `no_op` collision rate 0.504 of 240); if it is not, stop and report simulator drift between machines. When the digests differ, say so in the report and do not pair the result with v4's student numbers.

- [ ] **Step 7: Commit**

```bash
git add learned_judgment/dodge_eval.py tests/b/test_dodge_eval_ceiling.py results/runs/v4/ceiling_test.json results/runs/v4/result.json artifacts/env
git commit -m "dodge_eval ceiling: how many punches any action avoids at the first decision, and the student against that"
```

---

### Task 2: Conv-net diagnostic

**Files:**
- Create: `learned_judgment/conv_diagnostic.py`
- Modify: `config.yaml` (new block `observed.dodge_vision.conv_diagnostic`)
- Test: `tests/b/test_conv_diagnostic.py` (new)

**Interfaces:**
- Consumes: `observed.load_dream`, `observed._newest_frames`, `observed._predicted_mean`, `observed.dream_targets`, `skills.dodge_vision.build_observed_spec`, `dodge_eval._load_trace`, `_grid`, `_replay_grid`; targets `{"threat": (m, 2), "action": (m, C)}`.
- Produces:
  - `trace_windows(frames: np.ndarray, f: int, first: int) -> np.ndarray` — `(n, T, N, N)` to `(n, T - first, f, N, N)`, oldest frame first, the layout of `offline_calibration`.
  - `train_conv(windows: np.ndarray, targets: Mapping[str, np.ndarray], groups: np.ndarray, *, steps: int, seed: int, validation_fraction: float, channels: Sequence[int] = (16, 32), batch: int = 256, learning_rate: float = 1e-3) -> dict[str, Any]` returning `{"model", "validation": {"loss", "threat_accuracy", "action_accuracy"}, "split": {"train_groups", "validation_groups"}, "wall_s"}`.
  - `conv_readouts(model: torch.nn.Module, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(n, Td, F, N, N)` to threat confidence `(n, Td)` and action scores `(n, Td, C)`, both float64, softmax per head.
  - `diagnose(config, run_id: str, dream: str | Path, scores: str | Path, trace: str | Path) -> dict[str, Any]`
  - Command `python -m learned_judgment.conv_diagnostic --run RUN --dream DIR --scores FILE --trace FILE`, file `<runs_dir>/<RUN>/conv_diagnostic.json`.
- Config block, under `observed.dodge_vision`:

```yaml
    conv_diagnostic:  # A plain conv net on the same dream and dev trace as the spiking student: a measurement, never a student.
      frames: 4  # Newest frames of the window, as the v4 student reads.
      steps: 2000  # Optimizer steps, as the v4 sweep.
      channels: [16, 32]  # Output channels of the two 3 x 3 conv layers.
      batch: 256  # Rows per optimizer step.
      learning_rate: 0.001  # Adam.
      network_limit_recall: 0.60  # Rule fixed before any result: dev recall at the chosen thresholds at or above this means the spiking network or its encoding is the limit.
      input_limit_recall: 0.45  # At or below this, the 12 x 12 input or the dream labels are the limit; between the two the result is inconclusive.
```

- [ ] **Step 1: Write the failing tests**

```python
def toy(m=400, f=4, n=12, c=4, seed=0):
    rng = np.random.default_rng(seed)
    windows = (rng.random((m, f, n, n)) * 0.1).astype(np.float32)
    label = rng.integers(0, 2, m)
    windows[label == 1, -1, 5:7, 5:7] += 0.8  # a bright blob in the newest frame is the threat
    threat = np.stack([1 - label, label], axis=1).astype(np.float64)
    action = np.zeros((m, c)); action[np.arange(m), label] = 1.0
    return windows, {"threat": threat, "action": action}, label

def test_conv_learns_a_separable_toy_and_readouts_fit_the_replay():
    windows, targets, _ = toy()
    out = train_conv(windows, targets, np.arange(400), steps=300, seed=0, validation_fraction=0.2)
    assert out["validation"]["threat_accuracy"] >= 0.95
    threat, scores = conv_readouts(out["model"], windows[:6].reshape(2, 3, 4, 12, 12))
    assert threat.shape == (2, 3) and scores.shape == (2, 3, 4)
    assert np.all((threat >= 0) & (threat <= 1)) and np.allclose(scores.sum(axis=2), 1.0)

def test_one_scenario_never_sits_on_both_sides_of_the_split():
    windows, targets, _ = toy()
    out = train_conv(windows, targets, np.repeat(np.arange(100), 4), steps=5, seed=0, validation_fraction=0.2)
    assert not set(out["split"]["train_groups"]) & set(out["split"]["validation_groups"])
    assert len(out["split"]["validation_groups"]) == 20

def test_same_seed_same_result():
    windows, targets, _ = toy()
    a = train_conv(windows, targets, np.arange(400), steps=50, seed=3, validation_fraction=0.2)
    b = train_conv(windows, targets, np.arange(400), steps=50, seed=3, validation_fraction=0.2)
    assert a["validation"]["loss"] == b["validation"]["loss"]

def test_trace_windows_end_at_the_decision_frame():
    frames = np.arange(20, dtype=np.float32).reshape(2, 10, 1, 1)
    w = trace_windows(frames, f=4, first=7)
    assert w.shape == (2, 3, 4, 1, 1)
    assert w[0, 0, :, 0, 0].tolist() == [4.0, 5.0, 6.0, 7.0] and w[0, 2, :, 0, 0].tolist() == [6.0, 7.0, 8.0, 9.0]
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/b/test_conv_diagnostic.py -q -p no:cacheprovider`
Expected: FAIL, `No module named 'learned_judgment.conv_diagnostic'`.

- [ ] **Step 3: Implement `trace_windows`, `train_conv` and `conv_readouts`**

The net: `Conv2d(F, 16, 3, padding=1)`, ReLU, `Conv2d(16, 32, 3, padding=1)`, ReLU, flatten, two linear heads (2 and C); input is the raw window minus 0.5. Loss: soft-target cross-entropy of both heads, summed. Adam; `torch.manual_seed(seed)` and `torch.set_num_threads(1)`; the validation groups are the first `round(validation_fraction * n_groups)` of `np.random.default_rng(seed).permutation(unique groups)`.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/b/test_conv_diagnostic.py -q -p no:cacheprovider`
Expected: `4 passed`.

- [ ] **Step 5: Implement `diagnose` and the command**

It reads `frames`, `steps`, `channels`, `batch` and `learning_rate` from `observed.dodge_vision.conv_diagnostic`, the seed from `runtime.seed` and the validation share from `students.snn.validation_fraction` (the student's own). It builds the inputs exactly as `observed._train_command` does (`load_dream`, the `scores.npz` checks, `_newest_frames`, `dream_targets(scores, spec, config, predicted_mean=_predicted_mean(scores))`, groups `dream["scenario_index"]`), reshapes to `(m, F, N, N)`, trains, takes `conv_readouts` over `trace_windows(trace["frames"], F, first_decision_index)`, and passes them to `_replay_grid(config, _grid(config), t_s, traces, truth, t_hit)` as `offline_calibration` does. The report holds the chosen row (`emitted`, `precision`, `recall`, `false_activations`), the validation numbers, the spiking student's `sweep.calibration` from `<run>/result.json` as `reference`, and `verdict`: `"network"`, `"input_or_labels"` or `"inconclusive"` by the two config values. It refuses a trace whose `split` is not `dev`.

- [ ] **Step 6: Run it on v4's real dream and dev trace** (needs the files of "Data the runs need")

Run: `uv run python -m learned_judgment.env.evidence conv-diagnostic -- uv run python -m learned_judgment.conv_diagnostic --run v4 --dream results/runs/v4/dream --scores results/runs/v4/scores.npz --trace results/runs/v4/trace_dev.npz`
Expected: exit 0; `results/runs/v4/conv_diagnostic.json` with `reference.recall` 0.336 (the v4 student on dev) and a `verdict`. Without the files: NOT RUN, and say so.

- [ ] **Step 7: Commit**

```bash
git add learned_judgment/conv_diagnostic.py tests/b/test_conv_diagnostic.py config.yaml
git commit -m "conv_diagnostic: a plain conv net on the student's dream and dev trace, to tell input limits from network limits"
```

---

### Task 3: RAMP dream from recorded situations

**Files:**
- Modify: `learned_judgment/skills/ramp.py` (`collect_dream`, the `dream` parser, new `recorded_situations`), `learned_judgment/run_skill.py` (`_steps`, both ramp chains), `config.yaml` (`skills.ramp`)
- Test: `tests/b/test_ramp_dream.py` (new)

**Interfaces:**
- Consumes: a context rows directory (`rows.npz`: `features (r, 4)` in [0, 1], `decision_time_s (r,)`, `scenario_index (r,)`), written by `ramp rows --split context` or `ramp attempts`.
- Produces:
  - `recorded_situations(features: np.ndarray, decision_time_s: np.ndarray, scenario_index: np.ndarray, n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]` — `(n, D)` float32 in [0, 1] and `(n,)` float64.
  - `collect_dream(config, out_dir, n=None, from_rows: str | Path | None = None) -> dict[str, Any]`; command option `dream --from-rows DIR`.
  - Config keys `skills.ramp.dream_sampling` (`uniform` or `recorded`) and `skills.ramp.dream_noise`:

```yaml
    dream_sampling: recorded  # uniform: each feature drawn independently over its range (the behaviour before this key). recorded: drawn from the feature vectors measured in the context attempts, one scenario at a time.
    dream_noise: 0.02  # recorded only: standard deviation of the noise added per feature, in the features' 0-1 scale. Starting value; change only from dev runs.
```

- [ ] **Step 1: Write the failing tests**

Fixtures: `config` is `copy.deepcopy(load_config())` with `skills.ramp.dream_sampling` set to `"uniform"` and `skills.ramp.dream_noise` to `0.02`; `config_recorded` is the same with `"recorded"`.

```python
FEATS = np.array([[0.2, 0.25, 0.5, 0.4], [0.2, 0.25, 0.5, 0.4], [0.7, 0.72, 0.6, 0.4]], np.float32)  # scenario 0 has two rows
TIMES, INDEX = np.array([2.1, 2.1, 2.2]), np.array([0, 0, 1])

def test_recorded_situations_stay_near_real_ones_and_keep_their_time():
    f, t = recorded_situations(FEATS, TIMES, INDEX, n=500, noise=0.02, rng=np.random.default_rng(1))
    assert f.shape == (500, 4) and f.dtype == np.float32 and f.min() >= 0.0 and f.max() <= 1.0
    near0 = np.abs(f - FEATS[0]).max(axis=1) < 0.1
    near1 = np.abs(f - FEATS[2]).max(axis=1) < 0.1
    assert np.all(near0 | near1)
    assert 0.35 < near0.mean() < 0.65  # scenarios are drawn evenly, not rows
    assert np.all(np.isclose(t[near0], 2.1)) and np.all(np.isclose(t[near1], 2.2))

def test_noise_zero_returns_real_vectors_and_noise_is_clipped():
    f, _ = recorded_situations(FEATS, TIMES, INDEX, n=50, noise=0.0, rng=np.random.default_rng(2))
    assert all(any(np.array_equal(row, real) for real in FEATS) for row in f)
    edge = np.array([[0.0, 1.0, 0.0, 1.0]], np.float32)
    g, _ = recorded_situations(edge, np.array([2.0]), np.array([0]), n=200, noise=0.5, rng=np.random.default_rng(3))
    assert g.min() >= 0.0 and g.max() <= 1.0

def test_uniform_mode_is_the_old_dream(tmp_path, config):  # config: the repository config with dream_sampling "uniform"
    collect_dream(config, tmp_path, n=64)
    rng = np.random.default_rng(np.random.SeedSequence([config["runtime"]["seed"], 4]))
    with np.load(tmp_path / "dream.npz") as d:
        assert np.array_equal(d["features"], rng.random((64, 4)).astype(np.float32))

def test_recorded_mode_needs_the_rows(tmp_path, config_recorded):
    with pytest.raises(ValueError, match="--from-rows"):
        collect_dream(config_recorded, tmp_path, n=8)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/b/test_ramp_dream.py -q -p no:cacheprovider`
Expected: FAIL, `cannot import name 'recorded_situations'`.

- [ ] **Step 3: Implement `recorded_situations`, the `recorded` branch of `collect_dream` and `--from-rows`**

One feature vector and decision time per scenario (its first row); `rng.integers` picks scenarios with replacement, `rng.normal(0, noise)` per feature, clip to [0, 1]. The dream seed is unchanged (`SeedSequence([runtime.seed, SPLIT_CODES["dream"]])`). `dream.json` `sampling` says which mode ran, the rows directory, the number of scenarios drawn from and the noise. In `run_skill._steps`, both ramp `dream` steps get `"--from-rows", run_dir / "context"` when `skills.ramp.dream_sampling` is `recorded`.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/b/test_ramp_dream.py tests/b/test_observed.py -q -p no:cacheprovider`
Expected: all pass (`test_observed.py` proves the old path still works).

- [ ] **Step 5: Commit**

```bash
git add learned_judgment/skills/ramp.py learned_judgment/run_skill.py config.yaml tests/b/test_ramp_dream.py
git commit -m "ramp dream: situations drawn from the measured feature vectors of the attempts (skills.ramp.dream_sampling)"
```

---

### Task 4: Punch teacher on scene truth

**Files:**
- Modify: `learned_judgment/skills/dodge_vision.py` (new `TRUTH_NAMES`, `truth_features`, the spec's `truth_names` and `truth`), `learned_judgment/observed.py` (`FEATURE_BLOCKS`, `_feature_set`, `teacher_matrix`, and its callers `make_teacher`, `predict_outcomes`, `dream_scores`, `one_shot_report`), `config.yaml` (`skills.dodge_vision.teacher_features`)
- Test: `tests/b/test_truth_features.py` (new)

**Interfaces:**
- Consumes: a scenario of `sample_threat_scenarios` (its launch state through `dodge_vision._ball_keys()`: x, y, z, vx, vy, vz in the dog's frame at the launch), `rows.json` / `dream.json` `scenarios` with `scenario_index`, and `decision_time_s`.
- Produces:
  - `TRUTH_NAMES = ("present", "time_to_arrival_s", "aim_y_m", "aim_z_m", "speed_m_s")`
  - `truth_features(config: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]], decision_times_s: Sequence[float]) -> np.ndarray` — `(n, 5)` float64.
  - `DodgeVisionSpec.truth_names: tuple[str, ...]` and `DodgeVisionSpec.truth(scenarios, decision_times_s) -> np.ndarray`.
  - `FEATURE_BLOCKS = ("window", "truth", "action", "physics")`; `teacher_matrix(spec, features, action, physics, feature_set, truth: np.ndarray | None = None)`; valid sets: `["window", "action", "physics"]`, `["action", "physics"]`, `["truth", "action", "physics"]`.
  - `config.yaml` on this branch: `teacher_features: [truth, action, physics]`.

- [ ] **Step 1: Write the failing tests**

Fixtures: `config` is `load_config()`; `spec` is `build_observed_spec(config)`; `features` is `np.zeros((6, D), np.float32)` with D = `vision.frames` x 12 x 12; `action` is `np.arange(6) % len(spec.action_names)`; `physics` is `np.zeros((6, 5))`. `punch(x, y, z, vx, vy, vz)` takes the first scenario with `object_kind` `"punch"` from `sample_threat_scenarios(config, np.random.default_rng(0), 50)`, deep-copies it and overwrites the six launch values in its `situation` in `dodge_vision._ball_keys()` order. `empty_scenario` is the first scenario of kind `"empty"` from the same draw.

```python

def test_truth_of_a_head_on_punch(config):
    t = truth_features(config, [punch(1.0, 0.1, 0.2, -1.0, 0.0, 0.0)], [0.25])
    assert t.shape == (1, 5) and t.dtype == np.float64
    present, arrival, aim_y, aim_z, speed = t[0]
    assert present == 1.0 and speed == pytest.approx(1.0)
    assert arrival == pytest.approx(0.75)  # 1.0 m at 1.0 m/s, 0.25 s already gone
    assert aim_y == pytest.approx(0.1) and aim_z == pytest.approx(0.2)

def test_truth_is_finite_when_nothing_arrives(config, empty_scenario):
    duration = config["skills"]["dodge_vision"]["episode_duration_s"]
    static = truth_features(config, [punch(1.0, 0.3, 0.2, 0.0, 0.0, 0.0)], [0.5])[0]
    receding = truth_features(config, [punch(1.0, 0.3, 0.2, 1.0, 0.0, 0.0)], [0.5])[0]
    empty = truth_features(config, [empty_scenario], [0.5])[0]
    assert np.all(np.isfinite([static, receding, empty]))
    assert static[1] == duration and receding[1] == duration and static[2] == pytest.approx(0.3)
    assert empty.tolist() == [0.0, duration, 0.0, 0.0, 0.0]

def test_truth_block_replaces_the_window_columns(spec, features, action, physics):
    truth = np.zeros((features.shape[0], 5))
    x, layout = teacher_matrix(spec, features, action, physics, ["truth", "action", "physics"], truth=truth)
    assert layout.n_situation == 5 and layout.names[:5] == tuple(f"truth_{n}" for n in TRUTH_NAMES)
    assert not any(name.startswith("window_") for name in layout.names)
    assert x.shape == (features.shape[0], 5 + layout.n_action + layout.n_physics)

def test_truth_block_without_truth_is_refused_and_the_window_block_is_unchanged(spec, features, action, physics):
    with pytest.raises(ValueError, match="truth"):
        teacher_matrix(spec, features, action, physics, ["truth", "action", "physics"])
    old, _ = teacher_matrix(spec, features, action, physics, ["window", "action", "physics"])
    assert old.shape[1] == features.shape[1] + spec.action_features.shape[1] + physics.shape[1]
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/b/test_truth_features.py -q -p no:cacheprovider`
Expected: FAIL, `cannot import name 'truth_features'`.

- [ ] **Step 3: Implement `truth_features` and the spec's two members**

Position at the decision time is launch position plus velocity times that time (a punch flies straight). An object arrives when its x velocity is negative and it is still ahead (x > 0): `time_to_arrival_s` is x over the closing speed, clipped to `[0, episode_duration_s]`, and the aim is y and z at that moment. Otherwise `time_to_arrival_s` is `episode_duration_s` and the aim is the current y and z. No object (the scenario's `situation` has none of the six launch keys, as kind `empty`): `[0, episode_duration_s, 0, 0, 0]`. Add above the function:

```python
# shortcut: straight flight only (the punch); a ball under gravity raises ValueError here, give it physics_guess's ballistic path before throwing balls again.
```

- [ ] **Step 4: Implement the `truth` block in `observed.py`**

`teacher_matrix` appends the `truth` columns named `truth_<name>` and sets `n_situation` to their count. `make_teacher`, `predict_outcomes`, `dream_scores` and `one_shot_report` compute `spec.truth([meta["scenarios"][i] for i in scenario_index], decision_time_s)` when `"truth"` is in the feature set (repeated per candidate for the dream) and pass it on; with a set that has no `truth` they behave as today. A spec without `truth` used with that block raises `ValueError` naming the skill.

- [ ] **Step 5: Run the new tests and Person B's**

Run: `uv run pytest tests/b/test_truth_features.py tests/b/test_observed.py tests/b/test_dodge_vision.py -q -p no:cacheprovider`
Expected: all pass, with the same skips `test_dodge_vision.py` has at `a323da7`.

- [ ] **Step 6: Commit**

```bash
git add learned_judgment/skills/dodge_vision.py learned_judgment/observed.py config.yaml tests/b/test_truth_features.py
git commit -m "dodge_vision teacher: scene truth columns instead of pixel columns (teacher_features [truth, action, physics]); the student input is unchanged"
```

---

### Task 5: Run both settings and record the comparison

**Files:**
- Create: `learned_judgment/compare_runs.py`
- Test: `tests/b/test_compare_runs.py` (new)
- Modify: `LOG.md`, `DECISIONS.md`

**Interfaces:**
- Consumes: `result.json` `metrics` (`[{name, unit, higher_is_better, values: {method: {value, ci_low, ci_high, n}}}]`).
- Produces: `side_by_side(a: Mapping[str, Any], b: Mapping[str, Any], method: str) -> list[dict[str, Any]]` with rows `{"name", "a", "b", "difference", "n_a", "n_b"}`; command `python -m learned_judgment.compare_runs RUN_A RUN_B [--method snn]` printing a Markdown table.

- [ ] **Step 1: Write the failing test**

```python
def run(value, n):
    return {"metrics": [{"name": "ascent_success_rate", "values": {"snn": {"value": value, "n": n}, "no_op": {"value": 0.61, "n": n}}},
                        {"name": "only_here", "values": {"snn": {"value": 1.0, "n": n}}}]}

def test_side_by_side_pairs_metrics_by_name_and_reports_the_difference():
    rows = side_by_side(run(0.76, 41), {"metrics": run(0.80, 41)["metrics"][:1]}, "snn")
    assert rows == [{"name": "ascent_success_rate", "a": 0.76, "b": 0.80, "difference": pytest.approx(0.04), "n_a": 41, "n_b": 41}]
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/b/test_compare_runs.py -q -p no:cacheprovider`
Expected: FAIL, `No module named 'learned_judgment.compare_runs'`.

- [ ] **Step 3: Implement `side_by_side` and the command, run the test**

Only metrics present in both runs for that method appear. Run: `uv run pytest tests/b/test_compare_runs.py -q -p no:cacheprovider`. Expected: `1 passed`.

- [ ] **Step 4: Ask for the go-ahead on billed steps**

Run the free quote for each teacher request below (`python -m learned_judgment.observed quote ...`) and give Person A the token totals. Continue only on Person A's yes.

- [ ] **Step 5: RAMP, both dream settings on the same attempts**

First write two copies of `config.yaml`, `results/runs/ramp-cmp/config-uniform.yaml` and `results/runs/ramp-cmp/config-recorded.yaml`, that differ only in `skills.ramp.dream_sampling`. `run_skill _exec --config PATH MODULE ARGS` runs a module on such a copy. The sizes are the fast mode's (`observed.ramp.fast_dream_situations` 1000, `fast_snn_max_steps` 1200).

```bash
RAMP=learned_judgment.skills.ramp; OBS=learned_judgment.observed; CMP=results/runs/ramp-cmp
uv run python -m learned_judgment.run_skill _exec --config $CMP/config-uniform.yaml $RAMP attempts --out $CMP/context
for MODE in uniform recorded; do
  X="uv run python -m learned_judgment.run_skill _exec --config $CMP/config-$MODE.yaml"; R=results/runs/ramp-$MODE
  if [ $MODE = recorded ]; then FROM="--from-rows $CMP/context"; else FROM=""; fi
  $X $RAMP dream --out $R/dream --n 1000 $FROM
  $X $OBS teach --run ramp-$MODE --skill ramp --context $CMP/context --dream $R/dream --confirm-billing
  $X $OBS train --run ramp-$MODE --skill ramp --dream $R/dream --scores $R/scores.npz --steps 1200
  $X $RAMP run --student $R/student --split test --out $R/test_run --run ramp-$MODE
done
uv run python -m learned_judgment.compare_runs ramp-uniform ramp-recorded
```

Expected: both runs end `done`; the table has `ascent_success_rate`, `fall_rate`, `mean_progress_m`, `mean_slip_m`, `flat_false_activations`. The same 41 steep and 19 flat held-out scenarios are used by both (`n` equal). Report the table as measured, whichever way it points.

- [ ] **Step 6: Punch, pixel teacher against truth teacher, on dev only** (needs the files of "Data the runs need")

On `wip-dev-a` the config already says `teacher_features: [truth, action, physics]`; the sweep is run with the one variant that won for `v4`.

```bash
uv run python -m learned_judgment.observed teach --run v5-truth --skill dodge_vision --context results/runs/v4/context --dream results/runs/v4/dream --confirm-billing
uv run python -m learned_judgment.dodge_sweep --run v5-truth --dream results/runs/v4/dream --scores results/runs/v5-truth/scores.npz --trace results/runs/v4/trace_dev.npz --variants-json '{"frames":[4],"encodings":["events"],"hidden_sizes":[[128]],"steps":2000}'
```

Expected: `results/runs/v5-truth/sweep.json` with one variant and its dev calibration. Compare its dev recall and precision with `v4`'s (recall 0.336 at precision 1.0, 39 issued). No held-out test is run for `v5-truth`. Without the files: NOT RUN.

- [ ] **Step 7: Record and commit**

Write the two tables, the commands, exit codes and artifact paths into `LOG.md`; write into `DECISIONS.md` which setting each key keeps and why, and that scene truth is allowed on the teacher side. Post the result to Person B on `talkhere.md`.

```bash
git add learned_judgment/compare_runs.py tests/b/test_compare_runs.py results/runs/ramp-uniform results/runs/ramp-recorded results/runs/v5-truth LOG.md DECISIONS.md
git commit -m "Comparison of dream sampling (ramp) and teacher features (punch) against the old settings"
```
