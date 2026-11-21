# Decisions

## 2026-11-21 — Learned Judgment scope

The project is a simulated Unitree Go2 learning high-level decisions. Dodging a thrown ball is the main behavior.

## 2026-11-21 — Borrowed locomotion

Use one verified pre-trained Go2 walking policy and keep its weights unchanged. PFN/TabPFN learns policies over commands that the selected walking brain actually supports. Person A owns robot and simulator configuration; Person B owns SkillSpecs and the learning engine.

## 2026-11-21 — No substitute completion

There is no alternate robot, controller, simulator, hand-coded behavior, fake provider, or fake result. If a critical gate fails, stop and resolve it. Reports contain only executed, artifact-backed measurements.

## 2026-11-21 — Evidence before implementation

Verify each policy, package, API, config, version, and license with primary sources and a small executable proof before building dependent production code.

## 2026-11-21 — The only student is an SNN

Decided by the project lead: the only student is a spiking neural network. Per-skill SNN first, then a shared SNN with skill ID and dream rehearsal. No MLP student is built. KICK stays a stretch skill. This replaces the earlier plan in which MLP students were required and the SNN was an optional stretch.

MISSION.md, 01_PLAN.md, 03_METRICS.md, 04_EXPERIMENTS.md, and 06_PERSON_B.md were updated to match. PREREGISTRATION.md keeps the original hypotheses and carries a dated amendment. The MLP trained on actual attempts (baseline B5) is a residual baseline, not a student, and stays in the comparison list.

## 2026-11-21 — The SNN is trained inside the PFN dream; token budget

How the project describes the student, set by the project lead: the PFN is not distilled. The physics guess corrected by the PFN is a dream world, and the SNN is trained inside it. The SNN's training signal is the dreamed outcome of every candidate command scored by the skill's objective. It never learns to reproduce the PFN's predictions.

Prior Labs' General Terms (status June 23, 2026), section 10.1.1, list "knowledge distillation of the Contract Software" among violations that allow blocking access. Section 8.2 says the customer may freely use the Service Results. The saved text is in experiments/proofs/E7_provider_terms/. The project lead decided to proceed: Prior Labs sponsors the track and provided the token allowance for it. This is the lead's decision. No explicit statement from Prior Labs on this point has been recorded.

The project token budget is 35,000,000, the allowance Prior Labs gave for the track (config.yaml tabpfn.project_token_budget). Each request stays capped at the 10,000-token minimum charge unless tabpfn.max_tokens_per_request is changed.

## 2026-11-21 — SkillSpec package location

SkillSpecs live in learned_judgment/skills/, one file per skill, because the committed scaffold created that package. MISSION.md and 06_PERSON_B.md call the same thing "skills/".

## 2026-11-21 — No tunable defaults in Person B code

Person B functions take quantile levels, seeds, probe sizes, model settings, and thresholds as required arguments read from config.yaml. Code keeps only structural rules from the mission, such as the 1–3 action-dimension limit and the [0, 1] range of a probability. Float comparisons against configured values are exact.
