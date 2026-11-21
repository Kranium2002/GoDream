"""p00: environment + dependency provenance (who requires pyarrow / xxhash / scipy). No secrets touched."""
import sys, platform, re
from importlib import metadata

print("python", sys.version)
print("platform", platform.platform(), platform.processor())
for name in ["scikit-learn", "scipy", "numpy", "pandas", "pyarrow", "xxhash", "joblib", "threadpoolctl", "torch", "tabpfn-client"]:
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")

targets = {"pyarrow", "xxhash", "scipy", "joblib"}
req_by = {t: [] for t in targets}
for dist in metadata.distributions():
    dname = dist.metadata["Name"]
    for r in dist.requires or []:
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", r)
        if m and m.group(1).lower().replace("_", "-") in targets:
            req_by[m.group(1).lower().replace("_", "-")].append(f"{dname}: {r}")
for t in sorted(req_by):
    print(f"--- required by -> {t}")
    for line in sorted(req_by[t]):
        print("   ", line)
