# Lists, per mujoco release, whether a cp313 win_amd64 wheel exists (from a saved PyPI JSON file).
import json, sys, re

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("info.version (latest):", d["info"]["version"])

def vkey(v):
    return tuple(int(x) if x.isdigit() else -1 for x in re.split(r"[.]", v))

rows = []
for ver, files in d["releases"].items():
    win313 = [f for f in files if "cp313" in f["filename"] and "win_amd64" in f["filename"] and not f.get("yanked")]
    if win313:
        rows.append((ver, win313[0]))
rows.sort(key=lambda r: vkey(r[0]))
for ver, f in rows[-6:]:
    print(ver, f["filename"], f["size"], f["digests"]["sha256"], f["upload_time_iso_8601"], "requires_python=", f.get("requires_python"))
print("newest with cp313-win_amd64:", rows[-1][0])
