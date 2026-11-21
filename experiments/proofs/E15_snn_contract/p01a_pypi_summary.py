"""p01a: summarize PyPI JSON (fetched with curl into fetched/) for torch and snntorch.
Reads only local files. Prints latest version, requires_python, requires_dist, license,
upload time and the cp313 win_amd64 wheel (filename, size, sha256) for torch."""
import json, sys, re, pathlib
d = pathlib.Path(sys.argv[1])
for name in ("torch", "snntorch"):
    j = json.loads((d / f"pypi_{name}.json").read_text(encoding="utf-8"))
    info = j["info"]
    print(f"=== {name} ===")
    print("latest version:", info["version"])
    print("requires_python:", info.get("requires_python"))
    print("license:", repr(info.get("license"))[:200])
    print("license_expression:", info.get("license_expression"))
    print("classifiers(license):", [c for c in info.get("classifiers", []) if "License" in c])
    print("requires_dist:")
    for r in info.get("requires_dist") or []:
        print("   ", r)
    files = j["urls"]
    for f in files:
        fn = f["filename"]
        if name == "snntorch" or ("cp313-cp313-win_amd64" in fn) or fn.endswith("cp313-cp313-win_amd64.whl"):
            print("file:", fn, "size", f["size"], "sha256", f["digests"]["sha256"], "uploaded", f["upload_time_iso_8601"])
    rel = sorted(j["releases"].keys(), key=lambda v: [int(x) if x.isdigit() else x for x in re.split(r"[.]", v)] if all(p.isdigit() for p in v.split(".")) else [0])
    print("last 8 releases:", [v for v in j["releases"] if j["releases"][v]][-8:])
    if name == "torch":
        print("all files for latest:", len(files))
        for f in files:
            if "win" in f["filename"]:
                print("   win file:", f["filename"], f["size"])
