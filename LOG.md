# Work log

## 2026-11-21 — Mission revamp

- Replaced the earlier project direction with Learned Judgment: simulated Unitree Go2, borrowed verified walking policy, high-level PFN decision engine, and DODGE as the primary skill.
- Rewrote the four numbered work prompts and both A/B ownership prompts to share the Go2 command/observation contract.
- Added MISSION.md and CLAUDE.md with the no-fallback, real-test, and no-fabricated-results rules.
- This change updates project instructions only. No Go2 policy, MuJoCo integration, learned skill, TabPFN call, or test result is claimed as implemented.

## 2026-11-21 — Person B: contracts, first research round, SNN-only student

- Decision by the project lead: the only student is a spiking neural network; no MLP student; KICK stays stretch. Mission and plan documents updated to match; PREREGISTRATION.md carries a dated amendment and keeps the original hypotheses. See DECISIONS.md.
- Research before code (UltraCode workflow, three agents, no production edits): hosted TabPFN client contract (non-billable part of E7), library contracts for baselines and evaluation, and a read-only look-ahead at the candidate walking policies' command interface. Entries are in experiments/EXPERIMENTS_LOG.md; proofs are under experiments/proofs/.
- Written by hand as the fixed contracts: learned_judgment/engine/types.py, engine/features.py, engine/decide.py, engine/streams.py, skills/base.py, config.py.
- config.yaml: added engine, tabpfn, baselines, students, and evaluation tunables in Person B's sections. tabpfn.project_token_budget is null on purpose, so the teacher refuses any billable request until the lead sets a budget.
- Checks executed: a scratch script (outside the repository) exercised the contract modules and the physics-only decide path against the real config.yaml; exit code 0. The real config correctly refuses to build a skill because the walking policy's command schema is not verified.
- NOT RUN: any Go2 episode (Person A's Go2World does not exist yet), any TabPFN prediction request, any teacher dream, any student training.
- Open for the lead: Prior Labs' General Terms section 10.1.1 names "knowledge distillation of the Contract Software" as grounds for blocking access. Training a student on teacher decisions may fall under it; this needs an explicit answer from Prior Labs before real dreams are used for training.

## 2026-11-21 — Person B: teacher, engine, evaluation built; SNN foundations

- Built by three implementation agents against the hand-written contracts, then re-run by the lead: learned_judgment/teacher.py and teacher_cache.py (TabPFN residual teacher, request cache, cost and usage gate, call log), learned_judgment/engine/ (rows, collect, learn, dream, adapt), learned_judgment/eval/ (baselines, splits, metrics, bootstrap, calibration, harness, report).
- Checks: 15 check scripts under experiments/proofs/B_teacher_offline, B_engine_offline, and B_eval_offline; all exit code 0 on the lead's re-run. They use fixtures to test mechanics only. One is live against the provider's free endpoints: it showed the teacher refusing a request before any prediction call, with account usage unchanged at 40,000 tokens.
- Follow-up edits by the lead: one shared atomic file writer (learned_judgment/fileio.py) with its retry bounds in config.yaml; the evaluation harness writes rows through the engine's row builder; the attempt row's seed is the seed of the world the attempt ran in.
- SNN: PyTorch 2.15.0 (CPU) and snnTorch 1.0.0 added to pyproject.toml and installed (experiment E15). Written by hand: learned_judgment/students/snn_core.py (deployed numpy network and operation counts), encoding.py, energy.py. The numpy network reproduced saved snnTorch outputs exactly on the E15 fixture model: zero logit difference and zero spike-count mismatches on 1,000 inputs.
- Decisions recorded in DECISIONS.md: the SNN is trained inside the PFN dream and is not a copy of the PFN; project token budget 35,000,000.
- Note for Person A: mujoco is installed in the venv but not declared in pyproject.toml, so a plain `uv sync` would uninstall it. Declare it or use `uv sync --inexact`.
- NOT RUN: any Go2 episode, any TabPFN prediction request, any dream from a provider-backed teacher, any SNN training on real dreams.

## 2026-11-21 — Person B: SNN student built; independent review

- Research before code: E15 (PyTorch CPU and snnTorch installed and exercised in throwaway environments, exact neuron dynamics, numpy parity, determinism, latency, operation counts), E15-literature (sources for the energy estimate), E7-terms (Prior Labs published terms). Entries in experiments/EXPERIMENTS_LOG.md.
- Built by implementation agents, re-run by the lead: learned_judgment/students/targets.py, snn_train.py, snn.py (per-skill student), shared_snn.py (one trunk, one head per skill, dream rehearsal, retention matrix), measure.py (latency, spikes, synaptic operations, energy estimate, agreement and dream regret), and engine/train_in_dream.py.
- Found during the build: snnTorch ran its reset step in float32 inside a float64 model; the surrogate is now wrapped so the trained model and the deployed numpy network agree on every spike and decision. Readout logits differ by about 2e-15 between the two libraries; training stops with an error if a spike count or decision differs.
- Checks: experiments/proofs/B_snn_offline, B_snn_shared_offline, B_snn_measure_offline; all exit code 0 on the lead's re-run. They run fixture skills through the real dream and train-in-dream code and test mechanics only.
- Added by hand: a lean single-decision path in snn_core.py and encoding.py (same arithmetic as the batched path) and per-neuron spike recording.
- Independent review of the teacher, engine, and evaluation code (three reviewers, every finding reproduced by execution): 1 blocking, 9 important, 22 minor. Reproductions are under experiments/proofs/B_review/. Fixes are applied in two rounds and each gets a regression check.
- NOT RUN: any Go2 episode, any TabPFN prediction request, any SNN training on a real dream, any measurement on the robot.
