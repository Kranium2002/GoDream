"""p10: persistence-format contract: JSON float round trip, NaN/inf, .npz, parquet (pyarrow), atomic replace on Windows,
concurrent append. FIXTURE values only. Writes scratch files under ./p10_scratch/ (inside the proof directory)."""
import json, math, os, struct, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SCR = HERE / "p10_scratch"
SCR.mkdir(exist_ok=True)

def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]

if len(sys.argv) > 1 and sys.argv[1] == "--append-worker":
    path, wid, n, mode = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    pad = "x" * 900
    if mode == "text-a":
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            for i in range(n):
                f.write(json.dumps({"w": wid, "i": i, "pad": pad}, separators=(",", ":")) + "\n"); f.flush()
    elif mode == "os-append-single-write":
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))
        try:
            for i in range(n):
                os.write(fd, (json.dumps({"w": wid, "i": i, "pad": pad}, separators=(",", ":")) + "\n").encode())
        finally:
            os.close(fd)
    elif mode == "locked":
        import msvcrt
        lock_path = path + ".lock"
        for i in range(n):
            lfd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
            try:
                while True:
                    try:
                        msvcrt.locking(lfd, msvcrt.LK_NBLCK, 1); break
                    except OSError:
                        time.sleep(0.0005)
                with open(path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps({"w": wid, "i": i, "pad": pad}, separators=(",", ":")) + "\n")
                os.lseek(lfd, 0, 0); msvcrt.locking(lfd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(lfd)
    sys.exit(0)

print("python", sys.version.split()[0], "numpy", np.__version__)
rng = np.random.default_rng(0)
vals = np.r_[rng.normal(size=10000) * 10.0 ** rng.integers(-300, 300, 10000), 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, -0.0, 0.1, 1 / 3]
vals_list = [float(v) for v in vals]

print("\n== JSON float round trip (Python float repr) ==")
s = json.dumps(vals_list)
back = json.loads(s)
print("bitwise equal all", len(vals_list), "values incl. subnormal/max/-0.0:", all(bits(a) == bits(b) for a, b in zip(vals_list, back)))
print("json.dumps(-0.0) =", json.dumps(-0.0), "| loads keeps sign:", math.copysign(1, json.loads("-0.0")) == -1)
print("numpy scalar types in json.dumps: float64 ->", end=" ")
try: print(json.dumps(np.float64(0.1)), end=" ")
except TypeError as e: print("TypeError", e, end=" ")
for t in [np.float32(0.1), np.int64(3), np.bool_(True)]:
    try: json.dumps(t); print(f"| {type(t).__name__} OK", end=" ")
    except TypeError as e: print(f"| {type(t).__name__} TypeError: {e}", end=" ")
print()
print("np.float32(0.1) -> float() -> json:", json.dumps(float(np.float32(0.1))), "(float32 widened; repr of the float64 value)")
print("ndarray.tolist() gives Python floats; round trip bitwise:", all(bits(a) == bits(b) for a, b in zip(json.loads(json.dumps(vals.tolist())), vals.tolist())))

print("\n== NaN / inf in json ==")
print("default allow_nan=True:", json.dumps([float("nan"), float("inf"), -float("inf")]), "-> loads:", json.loads(json.dumps([float("nan"), float("inf")])))
try:
    json.dumps([float("nan")], allow_nan=False)
except ValueError as e:
    print("allow_nan=False:", type(e).__name__, e)
try:
    json.loads("[NaN]", parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite constant {c}")))
except ValueError as e:
    print("strict loads via parse_constant:", e)

print("\n== .npz round trip ==")
nanp = np.frombuffer(np.uint64(0xFFF8000000000001).tobytes(), dtype=np.float64)[0]
arr = np.r_[vals, np.nan, nanp, np.inf, -np.inf]
np.savez(SCR / "t.npz", X=arr.reshape(-1, 1), y=arr[:5].astype(np.float32))
with np.load(SCR / "t.npz", allow_pickle=False) as z:
    print("keys", sorted(z.files), "X bitwise equal (incl. NaN payload, -0.0, inf):", z["X"].tobytes() == arr.reshape(-1, 1).tobytes(), "y dtype", z["y"].dtype)
np.savez(SCR / "obj.npz", o=np.array([{"a": 1}], dtype=object))
try:
    with np.load(SCR / "obj.npz", allow_pickle=False) as z:
        z["o"]
except ValueError as e:
    print("object array with allow_pickle=False:", type(e).__name__, e)
import inspect
print("np.load default allow_pickle:", inspect.signature(np.load).parameters["allow_pickle"].default)

print("\n== parquet via pyarrow ==")
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
df = pd.DataFrame({"v": arr})
df.to_parquet(SCR / "t.parquet", engine="pyarrow", index=False)
d2 = pd.read_parquet(SCR / "t.parquet", engine="pyarrow")
a2 = d2["v"].to_numpy()
fin = np.isfinite(arr)
print("finite values bitwise equal (incl -0.0):", a2[fin].tobytes() == arr[fin].tobytes(), "| NaN count before/after:", int(np.isnan(arr).sum()), int(np.isnan(a2).sum()),
      "| NaN payload preserved:", bits(a2[-3]) == bits(nanp), "| inf preserved:", bool(np.isposinf(a2[-2]) and np.isneginf(a2[-1])))
t = pa.Table.from_pandas(df)
print("pa.Table.from_pandas null_count for NaN column:", t.column("v").null_count, "(NaN -> null by default)")
rows = [{"run_id": "r1", "skill": "dodge", "situation": {"ball_vx": -3.0, "dist": 2.0}, "action": {"side": "left", "t": 0.1}},
        {"run_id": "r1", "skill": "brake", "situation": {"speed": 0.6}, "action": {"stop_dist": 0.4}}]
tab = pa.Table.from_pylist(rows)
print("nested dicts with heterogeneous keys -> schema:", tab.schema.field("situation").type, "| missing keys become null:", tab.column("situation").to_pylist()[1])
pq.write_table(tab, SCR / "rows.parquet")
print("parquet append in place: pyarrow.parquet has no append mode; ParquetWriter needs a fixed schema for one open file; new rows => new file/part")

print("\n== atomic replace on Windows ==")
target = SCR / "cache_entry.json"; target.write_text('{"v":1}', encoding="utf-8")
tmp = SCR / "cache_entry.json.tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write('{"v":2}'); f.flush(); os.fsync(f.fileno())
try:
    os.rename(tmp, target); print("os.rename over existing: OK")
except OSError as e:
    print("os.rename over existing:", type(e).__name__, e)
os.replace(tmp, target); print("os.replace over existing: OK ->", target.read_text())
with open(tmp, "w", encoding="utf-8") as f: f.write('{"v":3}')
reader = open(target, "r", encoding="utf-8")
try:
    os.replace(tmp, target); print("os.replace while target open for read in same process: OK ->", target.read_text())
except OSError as e:
    print("os.replace while target open for read:", type(e).__name__, getattr(e, "winerror", None), e)
finally:
    reader.close()
if tmp.exists():
    os.replace(tmp, target); print("retry after close: OK ->", target.read_text())
with open(tmp, "w", encoding="utf-8") as f: f.write('{"v":4}')
try:
    os.replace(tmp, SCR / "nonexistent_dir" / "x.json")
except OSError as e:
    print("os.replace into missing dir:", type(e).__name__)
os.remove(tmp)

print("\n== concurrent append: 8 processes x 400 lines of ~930 bytes ==")
for mode in ["text-a", "os-append-single-write", "locked"]:
    p = SCR / f"append_{mode}.jsonl"
    for f_ in [p, Path(str(p) + ".lock")]:
        if f_.exists(): f_.unlink()
    t0 = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, __file__, "--append-worker", str(p), str(w), "400", mode]) for w in range(8)]
    rcs = [pr.wait() for pr in procs]
    dt = time.perf_counter() - t0
    lines = p.read_bytes().split(b"\n")
    if lines and lines[-1] == b"": lines = lines[:-1]
    ok = bad = 0; seen = set()
    for ln in lines:
        try:
            o = json.loads(ln); ok += 1; seen.add((o["w"], o["i"]))
        except Exception:
            bad += 1
    print(f"  mode={mode:24s} rcs={set(rcs)} lines={len(lines)} parsed_ok={ok} corrupt={bad} unique={len(seen)} expected=3200 elapsed={dt:.2f}s")
