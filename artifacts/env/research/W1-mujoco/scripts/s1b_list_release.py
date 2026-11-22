# Prints every file of the given mujoco releases from saved PyPI JSON.
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for v in sys.argv[2:]:
    print("==", v)
    for f in d["releases"].get(v, []):
        print(" ", f["filename"], f["size"], f.get("requires_python"), "yanked=", f.get("yanked"), f["upload_time_iso_8601"])
