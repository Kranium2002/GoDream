"""Import check of this lane's module; importing it must not import torch.

FIXTURE CHECK - mechanics only (no data at all). The module is imported in a fresh process so that
nothing else imported earlier can hide or cause a torch import. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_shared_offline\\c6_imports.py
"""

print("FIXTURE CHECK - mechanics only, not evidence about the robot, the simulator, TabPFN or any trained student")

import subprocess  # noqa: E402
import sys  # noqa: E402

from fixture_shared_specs import REPO_ROOT, Checks  # noqa: E402

MODULES = ("learned_judgment.students.shared_snn",)
PROBE = (
    "import importlib, sys, time; started = time.perf_counter(); module = importlib.import_module(sys.argv[1]); "
    "print(f'{sys.argv[1]}: imported in {time.perf_counter() - started:.2f} s; exports {module.__all__}; '"
    "f'torch_in_sys_modules={\"torch\" in sys.modules} snntorch_in_sys_modules={\"snntorch\" in sys.modules}')"
)

checks = Checks()
for module in MODULES:
    result = subprocess.run([sys.executable, "-c", PROBE, module], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    line = result.stdout.strip() or result.stderr.strip()[-300:]
    print(f"  {line}")
    checks.expect(result.returncode == 0, f"{module} imports (exit {result.returncode})")
    checks.expect("torch_in_sys_modules=False snntorch_in_sys_modules=False" in result.stdout, f"{module} leaves torch and snntorch unimported")
checks.finish()
