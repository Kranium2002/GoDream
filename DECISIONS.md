# Decisions

## 2026-11-21 — Learned Judgment scope

The project is a simulated Unitree Go2 learning high-level decisions. Dodging a thrown ball is the main behavior.

## 2026-11-21 — Borrowed locomotion

Use one verified pre-trained Go2 walking policy and keep its weights unchanged. PFN/TabPFN learns policies over commands that the selected walking brain actually supports. Person A owns robot and simulator configuration; Person B owns SkillSpecs and the learning engine.

## 2026-11-21 — No substitute completion

There is no alternate robot, controller, simulator, hand-coded behavior, fake provider, or fake result. If a critical gate fails, stop and resolve it. Reports contain only executed, artifact-backed measurements.

## 2026-11-21 — Evidence before implementation

Verify each policy, package, API, config, version, and license with primary sources and a small executable proof before building dependent production code.

## 2026-11-21 — Locomotion pass thresholds fixed before any run

Measured speed within 20% of the commanded value, yaw rate within 30%, and "stopped" means mean planar speed under 0.05 m/s over the last 2 s. Set before any locomotion run. A miss is diagnosed; the threshold is not widened.

## 2026-11-21 — Research and execution method for Person A

The "UltraCode" research runs as multi-agent workflows: read-only researchers in parallel, then one verifier per researcher that re-opens every cited source. Research agents do not run repository code or load policy checkpoints. Implementation is done by Opus 5.5 subagents, one implementer and one reviewer per task.

## 2026-11-21 — Evidence recording

Every Person A acceptance claim is backed by a run recorded through `learned_judgment/env/evidence.py` under `artifacts/env/`. `ENV_REPORT.md` is generated from those files only.

## 2026-11-21 — Thrown ball is an IHF size 3 handball

Mass 0.450 kg and radius 0.0939 m, the mid-range of IHF Rule 3:2 (425–475 g, circumference 58–60 cm), verified in experiment E6. Chosen before any throw was simulated. Fallback if dev runs show every hit topples the dog: FIVB volleyball (0.270 kg, radius 0.1050 m). Contact parameters, ball friction and inertia are not set by any rule and remain to be stated by the ball-scene work.
