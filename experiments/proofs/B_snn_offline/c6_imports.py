"""Import check of every module of this lane; importing them must not import torch.

FIXTURE CHECK - mechanics only (no data at all). Each module is imported in a fresh process so
that one module cannot hide another's torch import. snn_train is the training module: its
training functions import torch lazily, so importing the module itself must not either.
Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_offline\\c6_imports.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any model")

import subprocess  # noqa: E402
import sys  # noqa: E402

from fixture_snn_spec import REPO_ROOT, Checks  # noqa: E402

MODULES = (
    "learned_judgment.students.targets",
    "learned_judgment.students.snn_train",
    "learned_judgment.students.snn",
    "learned_judgment.engine.train_in_dream",
)
PROBE = (
    "import importlib, sys, time; started = time.perf_counter(); importlib.import_module(sys.argv[1]); "
    "print(f'{sys.argv[1]}: imported in {time.perf_counter() - started:.2f} s; '"
    "f'torch_in_sys_modules={\"torch\" in sys.modules} snntorch_in_sys_modules={\"snntorch\" in sys.modules}')"
)

checks = Checks()
for module in MODULES:
    result = subprocess.run([sys.executable, "-c", PROBE, module], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    line = (result.stdout.strip() or result.stderr.strip()[-300:])
    print(f"  {line}")
    checks.expect(result.returncode == 0, f"{module} imports (exit {result.returncode})")
    checks.expect("torch_in_sys_modules=False snntorch_in_sys_modules=False" in result.stdout, f"{module} leaves torch and snntorch unimported")
checks.finish()
