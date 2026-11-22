"""p09: canonical hashing recipe for a persistent request-cache key. FIXTURE arrays only. Run in 2 processes and diff."""
import hashlib, json, math, os, time
import numpy as np
import pandas as pd
import xxhash

print("PYTHONHASHSEED=", os.environ.get("PYTHONHASHSEED"), "numpy", np.__version__, "pandas", pd.__version__, "xxhash", xxhash.VERSION)

CANON_NAN = np.float64("nan")  # positive quiet NaN 0x7ff8000000000000

def canon_array(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"unsupported dtype for hashing: {a.dtype} (convert explicitly; object/str/datetime not allowed)")
    a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
    if a.dtype.kind == "f":
        a = a + a.dtype.type(0.0)                     # -0.0 + 0.0 == +0.0 (IEEE round-to-nearest); NaN stays NaN
        a = np.where(np.isnan(a), a.dtype.type(CANON_NAN), a)  # collapse all NaN payloads/signs to one bit pattern
        a = np.ascontiguousarray(a)
    return a

def update_array(h, a, tag: str):
    a = canon_array(a)
    header = json.dumps({"tag": tag, "dtype": a.dtype.str, "shape": list(a.shape)}, sort_keys=True, separators=(",", ":"))
    h.update(header.encode("utf-8")); h.update(b"\x00"); h.update(a.tobytes(order="C")); h.update(b"\x01")

def update_frame(h, df: pd.DataFrame, tag: str):
    cols = [str(c) for c in df.columns]
    if len(set(cols)) != len(cols):
        raise ValueError("duplicate column names")
    meta = {"tag": tag, "columns": cols, "dtypes": [str(t) for t in df.dtypes], "n_rows": int(len(df))}
    h.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()); h.update(b"\x00")
    for c in df.columns:                                 # column ORDER is significant (kept as given)
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cat_meta = {"categories": [str(x) for x in s.cat.categories], "ordered": bool(s.cat.ordered)}
            h.update(json.dumps(cat_meta, sort_keys=True, separators=(",", ":")).encode())
            update_array(h, s.cat.codes.to_numpy(), f"{tag}/{c}/codes")
        elif s.dtype.kind in "biuf":
            update_array(h, s.to_numpy(), f"{tag}/{c}")
        else:
            raise TypeError(f"column {c!r} dtype {s.dtype} not hashable by this recipe")

def _canon_json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return {"__ndarray__": o.tolist(), "dtype": o.dtype.str}
    raise TypeError(f"not canonical-JSON serialisable: {type(o).__name__}")

def canon_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=_canon_json_default)

def cache_key(X_train, y_train, X_query, output_spec: dict, settings: dict) -> str:
    h = hashlib.sha256()
    h.update(b"learned-judgment-cache-v1\x00")
    for tag, obj in (("X_train", X_train), ("y_train", y_train), ("X_query", X_query)):
        (update_frame if isinstance(obj, pd.DataFrame) else update_array)(h, obj, tag)
    h.update(canon_json({"output_spec": output_spec, "settings": settings}).encode("utf-8"))
    return h.hexdigest()

rng = np.random.default_rng(0)
Xtr = pd.DataFrame(rng.normal(size=(40, 4)), columns=["ball_x", "ball_vx", "dog_vx", "action_t"])
Xtr["surface"] = pd.Categorical(["tile", "carpet"] * 20, categories=["carpet", "tile"])
ytr = rng.normal(size=(40, 2))
Xq = pd.DataFrame(rng.normal(size=(8, 4)), columns=Xtr.columns[:4]); Xq["surface"] = pd.Categorical(["tile"] * 8, categories=["carpet", "tile"])
spec = {"task": "regression", "output_type": "quantiles", "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9]}
settings = {"model": "tabpfn-client", "version": "0.10.1", "n_estimators": 8, "seed": 0}
base = cache_key(Xtr, ytr, Xq, spec, settings)
print("BASE KEY", base)

def changed(label, **kw):
    args = dict(X_train=Xtr, y_train=ytr, X_query=Xq, output_spec=spec, settings=settings); args.update(kw)
    k = cache_key(**args)
    print(f"  {label:42s} differs={k != base}  {k[:16]}")

X2 = Xtr.copy(); X2.iloc[3, 1] = np.nextafter(X2.iloc[3, 1], np.inf); changed("one cell +1ulp", X_train=X2)
X2 = Xtr.copy(); X2["ball_x"] = X2["ball_x"].astype(np.float32); changed("dtype float64->float32 one column", X_train=X2)
X2 = Xtr[["ball_vx", "ball_x", "dog_vx", "action_t", "surface"]]; changed("column order swapped", X_train=X2)
X2 = Xtr.rename(columns={"dog_vx": "dog_v"}); changed("column renamed", X_train=X2)
X2 = Xtr.copy(); X2["surface"] = X2["surface"].cat.set_categories(["carpet", "tile", "ice"]); changed("extra categorical level (same values)", X_train=X2)
X2 = Xtr.copy(); X2["surface"] = X2["surface"].cat.reorder_categories(["tile", "carpet"]); changed("categorical level order", X_train=X2)
changed("quantile list changed", output_spec={**spec, "quantiles": [0.1, 0.5, 0.9]})
changed("setting n_estimators 8->4", settings={**settings, "n_estimators": 4})
changed("setting int 0 -> float 0.0 (seed)", settings={**settings, "seed": 0.0})
changed("settings dict insertion order only", settings=dict(reversed(list(settings.items()))))
changed("y_train as float32", y_train=ytr.astype(np.float32))
changed("y_train transposed copy (2,40)", y_train=np.ascontiguousarray(ytr.T))
changed("y_train Fortran-order same values", y_train=np.asfortranarray(ytr))
yb = ytr.astype(">f8"); changed("y_train big-endian same values", y_train=yb)

print("\n== -0.0 / NaN normalisation ==")
a = np.array([0.0, 1.0]); b = np.array([-0.0, 1.0])
print("raw bytes equal for 0.0 vs -0.0:", a.tobytes() == b.tobytes(), "| canon equal:", canon_array(a).tobytes() == canon_array(b).tobytes())
n1 = np.array([np.nan]); n2 = np.frombuffer(np.uint64(0xFFF8000000000001).tobytes(), dtype=np.float64).copy()
print("NaN payloads raw equal:", n1.tobytes() == n2.tobytes(), "| canon equal:", canon_array(n1).tobytes() == canon_array(n2).tobytes())
print("int64 [1,2] vs float64 [1.,2.] canon differ:", canon_array(np.array([1, 2])).tobytes() + canon_array(np.array([1, 2])).dtype.str.encode() != canon_array(np.array([1., 2.])).tobytes() + canon_array(np.array([1., 2.])).dtype.str.encode())

print("\n== canonical JSON ==")
print(canon_json({"b": 0.1, "a": [1, 2.0, 1e-17, 123456789.123456789], "c": np.float64(0.30000000000000004), "d": np.int64(3), "t": (0.25, 0.75)}))
for bad in [float("nan"), float("inf")]:
    try:
        canon_json({"x": bad})
    except ValueError as e:
        print(f"allow_nan=False rejects {bad}: ValueError: {e}")
print("json repr round-trip 0.1+0.2:", json.loads(json.dumps(0.1 + 0.2)) == 0.1 + 0.2, json.dumps(0.1 + 0.2))

print("\n== pandas.util.hash_pandas_object pitfalls (NOT recommended as the sole key) ==")
hp = pd.util.hash_pandas_object
print("int vs float column same values -> equal row hashes:", np.array_equal(hp(pd.Series([1, 2]), index=False), hp(pd.Series([1.0, 2.0]), index=False)))
c1 = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"])); c2 = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b", "z"]))
print("categorical with extra unused level -> equal row hashes:", np.array_equal(hp(c1, index=False), hp(c2, index=False)))
print("-0.0 vs 0.0 -> equal:", np.array_equal(hp(pd.Series([-0.0]), index=False), hp(pd.Series([0.0]), index=False)))
df1 = pd.DataFrame({"a": [1.0], "b": [2.0]}); df2 = pd.DataFrame({"b": [2.0], "a": [1.0]})
print("DataFrame column order swapped -> equal row hashes:", np.array_equal(hp(df1, index=False), hp(df2, index=False)))

print("\n== speed on (2040, 12) float64 (", 2040 * 12 * 8, "bytes ) median of 2000 ==")
A = rng.normal(size=(2040, 12)); buf = A.tobytes()
def t(fn, n=2000):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns(); fn(); ts.append(time.perf_counter_ns() - t0)
    return np.median(ts) / 1000
print(f"  sha256          : {t(lambda: hashlib.sha256(buf).hexdigest()):8.1f} us")
print(f"  blake2b(32)     : {t(lambda: hashlib.blake2b(buf, digest_size=32).hexdigest()):8.1f} us")
print(f"  xxh3_128        : {t(lambda: xxhash.xxh3_128(buf).hexdigest()):8.1f} us")
print(f"  xxh64           : {t(lambda: xxhash.xxh64(buf).hexdigest()):8.1f} us")
print(f"  canon+sha256 full: {t(lambda: cache_key(A, A[:, 0], A[:64], spec, settings), n=500):8.1f} us")
print("xproc digest of A:", hashlib.sha256(canon_array(A).tobytes()).hexdigest())
