# Decisions

## 2026-11-21 — Learned Judgment scope

The project is a simulated Unitree Go2 learning high-level decisions. Dodging a thrown ball is the main behavior.

## 2026-11-21 — Borrowed locomotion

Use one verified pre-trained Go2 walking policy and keep its weights unchanged. PFN/TabPFN learns policies over commands that the selected walking brain actually supports. Person A owns robot and simulator configuration; Person B owns SkillSpecs and the learning engine.

## 2026-11-21 — No substitute completion

There is no alternate robot, controller, simulator, hand-coded behavior, fake provider, or fake result. If a critical gate fails, stop and resolve it. Reports contain only executed, artifact-backed measurements.

## 2026-11-21 — Evidence before implementation

Verify each policy, package, API, config, version, and license with primary sources and a small executable proof before building dependent production code.
