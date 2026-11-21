"""Check 3: importing learned_judgment.students.measure does not import torch or snntorch.

FIXTURE CHECK - mechanics only (no data at all). The module is imported in a fresh process so
nothing imported earlier can hide or cause a torch import. Writes no file.

Run from the repository root:
    .venv\\Scripts\\python.exe experiments\\proofs\\B_snn_measure_offline\\c3_imports.py
"""

from _checks import BANNER, REPO_ROOT, Checks

print(BANNER)

import subprocess  # noqa: E402
import sys  # noqa: E402

MODULES = ("learned_judgment.students.measure",)
PROBE = (
    "import importlib, sys, time; started = time.perf_counter(); module = importlib.import_module(sys.argv[1]); "
    "print(f'{sys.argv[1]}: imported in {time.perf_counter() - started:.2f} s; public: {sorted(module.__all__)}'); "
    "print(f'torch_in_sys_modules={\"torch\" in sys.modules} snntorch_in_sys_modules={\"snntorch\" in sys.modules}')"
)

checks = Checks()
for module in MODULES:
    result = subprocess.run([sys.executable, "-c", PROBE, module], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    for line in (result.stdout.strip() or result.stderr.strip()[-400:]).splitlines():
        print(f"  {line}")
    checks.expect(result.returncode == 0, f"{module} imports (exit {result.returncode})")
    checks.expect(
        "torch_in_sys_modules=False snntorch_in_sys_modules=False" in result.stdout,
        f"{module} leaves torch and snntorch unimported",
    )
checks.finish()
