"""p01f: environment + import report for a probe venv (torch CPU + snntorch next to project pins).

Run with the probe venv's python, e.g.
  <venv>\\Scripts\\python.exe -X importtime? (no) -W always p01f_env_report.py
Prints versions, CUDA status, import timings, warnings raised on import, heavy modules pulled in,
license strings from installed METADATA, and on-disk sizes.
"""
import sys, time, warnings, importlib, os, pathlib
from importlib import metadata

warnings.simplefilter("always")
caught = []
_orig_showwarning = warnings.showwarning
def _record(message, category, filename, lineno, file=None, line=None):
    caught.append(f"{category.__name__}: {message} ({filename}:{lineno})")
warnings.showwarning = _record

print("python", sys.version)
print("executable", sys.executable)

before = set(sys.modules)
t0 = time.perf_counter(); import numpy; t1 = time.perf_counter()
print(f"import numpy {numpy.__version__}: {t1-t0:.3f}s")
t0 = time.perf_counter(); import torch; t1 = time.perf_counter()
print(f"import torch: {t1-t0:.3f}s")
mods_after_torch = set(sys.modules)
t0 = time.perf_counter(); import snntorch as snn; t1 = time.perf_counter()
print(f"import snntorch: {t1-t0:.3f}s  snntorch.__version__={snn.__version__}")
mods_after_snn = set(sys.modules)
t0 = time.perf_counter(); import snntorch.functional as SF; import snntorch.surrogate as surrogate; import snntorch.utils as sutils; t1 = time.perf_counter()
print(f"import snntorch.functional/surrogate/utils: {t1-t0:.3f}s")
mods_after_all = set(sys.modules)

heavy = ["matplotlib", "pandas", "scipy", "sklearn", "sympy", "networkx", "jinja2", "IPython", "tqdm", "nir", "torchvision", "h5py", "tonic"]
for h in heavy:
    print(f"  module '{h}' loaded after torch? {h in mods_after_torch}; after snntorch? {h in mods_after_snn}; after functional/utils? {h in mods_after_all}")
print("new top-level modules added by `import snntorch` (beyond torch):",
      sorted({m.split('.')[0] for m in mods_after_snn - mods_after_torch}))
print("new top-level modules added by snntorch.functional/surrogate/utils:",
      sorted({m.split('.')[0] for m in mods_after_all - mods_after_snn}))
# spikeplot needs matplotlib?
try:
    import snntorch.spikeplot  # noqa
    print("import snntorch.spikeplot: OK")
except Exception as e:
    print("import snntorch.spikeplot ->", type(e).__name__, e)
try:
    import snntorch.spikegen  # noqa
    print("import snntorch.spikegen: OK")
except Exception as e:
    print("import snntorch.spikegen ->", type(e).__name__, e)
try:
    import snntorch.export_nir  # noqa
    print("import snntorch.export_nir: OK")
except Exception as e:
    print("import snntorch.export_nir ->", type(e).__name__, e)

print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.backends.mkldnn.is_available():", torch.backends.mkldnn.is_available())
print("torch.backends.openmp.is_available():", torch.backends.openmp.is_available())
print("torch.get_num_threads():", torch.get_num_threads(), " os.cpu_count():", os.cpu_count())
pi = torch.__config__.parallel_info().splitlines()
print("parallel_info:", " | ".join(l.strip() for l in pi[:8]))
cfg = torch.__config__.show().splitlines()
print("config.show (first 6 lines):")
for l in cfg[:6]:
    print("   ", l)
for l in cfg:
    if "BLAS_INFO" in l or "LAPACK_INFO" in l or "USE_CUDA" in l or "CXX_FLAGS" in l[:20]:
        print("   ", l.strip()[:200])

print("--- pinned packages ---")
for name in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "snntorch"]:
    print(f"  {name}=={metadata.version(name)}")

print("--- licenses from METADATA (License-Expression / License / license classifiers) ---")
for d in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
    md = d.metadata
    lic_expr = md.get("License-Expression")
    lic = md.get("License")
    if lic and len(lic) > 80:
        lic = lic.splitlines()[0][:80] + " ...(truncated)"
    cls = [c.replace("License :: ", "") for c in (md.get_all("Classifier") or []) if c.startswith("License ::")]
    print(f"  {md['Name']}=={md['Version']}: License-Expression={lic_expr!r} License={lic!r} classifiers={cls}")

print("--- on-disk sizes ---")
def du(p):
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot
sp = pathlib.Path(torch.__file__).parent.parent
for pkg in ["torch", "snntorch", "sympy", "networkx", "numpy", "scipy", "pandas", "sklearn", "functorch", "torchgen"]:
    p = sp / pkg
    if p.exists():
        print(f"  {pkg}: {du(p)/1e6:.1f} MB")
venv_root = pathlib.Path(sys.executable).parent.parent
print(f"  whole venv ({venv_root}): {du(venv_root)/1e6:.1f} MB")

print("--- warnings captured during imports ---")
for w in caught:
    print("  ", w)
print("n_warnings:", len(caught))
