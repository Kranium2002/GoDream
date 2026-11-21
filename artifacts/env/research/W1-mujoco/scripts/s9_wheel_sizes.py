# Prints the cp313 / win_amd64 (or pure-Python) wheel each saved PyPI release JSON offers, with size and sha256.
import json, sys
for path in sys.argv[1:]:
    d = json.load(open(path, encoding="utf-8"))
    i = d["info"]
    picks = [f for f in d["urls"] if f["packagetype"] == "bdist_wheel" and
             (("cp313-cp313-win_amd64" in f["filename"]) or ("py3-none-any" in f["filename"]) or ("abi3" in f["filename"] and "win_amd64" in f["filename"]) or ("py3-none-win_amd64" in f["filename"]))]
    for f in picks:
        print(i["name"], i["version"], f["filename"], f["size"], f["digests"]["sha256"], f["url"])
    if not picks:
        print(i["name"], i["version"], "NO cp313 win_amd64 / pure wheel; files:", [f["filename"] for f in d["urls"]][:8])
    if i["name"] in ("torch", "onnxruntime"):
        print("  requires_dist:", i["requires_dist"])
