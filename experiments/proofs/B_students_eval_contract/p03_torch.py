"""p03: confirm torch is not installed in the project venv (no install attempted)."""
import importlib.util, sys
from importlib import metadata
print("python", sys.version.split()[0], "tag cp%d%d" % sys.version_info[:2])
print("find_spec('torch') ->", importlib.util.find_spec("torch"))
try:
    print("torch dist version:", metadata.version("torch"))
except metadata.PackageNotFoundError as e:
    print("metadata.version('torch') -> PackageNotFoundError", e)
try:
    import torch  # noqa: F401
    print("import torch: OK")
except ModuleNotFoundError as e:
    print("import torch ->", type(e).__name__, e)
