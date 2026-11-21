# Learned Judgment

Learned Judgment is a simulated Unitree Go2 decision-learning project for the HackLab × BRISA Robotics & AI Hackathon, Prior Labs track. The primary learned behavior is dodging a thrown ball. A verified pre-trained walking policy handles locomotion; PFN/TabPFN learns high-level command decisions using only that policy's verified command interface.

Read MISSION.md, then 01_PLAN.md–04_EXPERIMENTS.md. Work is split between [Person A](05_PERSON_A.md) (Go2, MuJoCo world, dog configuration) and [Person B](06_PERSON_B.md) (SkillSpecs, PFN engine, students, evaluation).

This checkout is a scaffold. The real Go2 policy, simulator integration, skills, and reports must be implemented and verified before the project is called runnable. Do not substitute another robot or fabricate test results.

Python 3.13 is declared in pyproject.toml. Once integration code and dependency verification are ready, install with:

    uv sync --project D:\BerlinPrior

The local .env is ignored by Git. .env.example lists the key name without a secret. Check current TabPFN usage and estimated cost before a real provider request.
