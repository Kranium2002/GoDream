"""p08: seeded independent named random streams without Python's randomized hash(). Prints draws to compare across processes."""
import hashlib, inspect, os, sys
import numpy as np

print("numpy", np.__version__, "PYTHONHASHSEED=", os.environ.get("PYTHONHASHSEED"), "| builtin hash('dodge/dev/situations') =", hash("dodge/dev/situations"))
print("SIGNATURE SeedSequence", inspect.signature(np.random.SeedSequence))

def _name_words(name: str) -> tuple[int, ...]:
    """4 x uint32 words from sha256(name). Fixed length => (base_seed, name, index) -> entropy tuple is injective up to sha256 collisions."""
    d = hashlib.sha256(name.encode("utf-8")).digest()
    return tuple(int.from_bytes(d[i:i + 4], "little") for i in range(0, 16, 4))

def stream_seedseq(base_seed: int, name: str, index: int = 0) -> np.random.SeedSequence:
    if base_seed < 0 or index < 0:
        raise ValueError("base_seed and index must be non-negative ints")
    return np.random.SeedSequence(entropy=int(base_seed), spawn_key=(*_name_words(name), int(index)))

def stream_rng(base_seed: int, name: str, index: int = 0) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stream_seedseq(base_seed, name, index)))

for (b, nm, i) in [(7, "dodge/dev/situations", 0), (7, "dodge/dev/situations", 1), (7, "dodge/dev/noise", 0),
                   (7, "dodge/test/situations", 0), (8, "dodge/dev/situations", 0)]:
    g = stream_rng(b, nm, i)
    print(f"stream base={b} name={nm!r:26} idx={i}: uniform[:3]={np.array2string(g.random(3), precision=17)} ints={g.integers(0, 2**31, 2)}")

print("\n== SeedSequence semantics ==")
ss = np.random.SeedSequence(42)
kids = ss.spawn(3)
print("SeedSequence(42).spawn(3) spawn_keys:", [k.spawn_key for k in kids], "n_children_spawned:", ss.n_children_spawned)
print("child[1] state == SeedSequence(42, spawn_key=(1,)) state:", np.array_equal(kids[1].generate_state(4), np.random.SeedSequence(42, spawn_key=(1,)).generate_state(4)))
kids2 = ss.spawn(2)
print("second spawn() call continues numbering (not repeat):", [k.spawn_key for k in kids2])
print("default_rng([1,2]) == Generator(PCG64(SeedSequence([1,2]))):", np.array_equal(np.random.default_rng([1, 2]).random(4), np.random.Generator(np.random.PCG64(np.random.SeedSequence([1, 2]))).random(4)))
print("default_rng(SeedSequence) accepted:", np.random.default_rng(np.random.SeedSequence(3)).random() == np.random.default_rng(3).random())
print("entropy [7, w...] vs spawn_key: SeedSequence(7, spawn_key=k) != SeedSequence([7, *k]):",
      not np.array_equal(np.random.SeedSequence(7, spawn_key=(1, 2)).generate_state(2), np.random.SeedSequence([7, 1, 2]).generate_state(2)))
for bad in [-1, 1.5]:
    try:
        np.random.SeedSequence(bad); print(f"SeedSequence({bad}) accepted")
    except Exception as e:
        print(f"SeedSequence({bad}) -> {type(e).__name__}: {e}")
gen = stream_rng(7, "x", 0)
gen2 = gen.spawn(2) if hasattr(gen, "spawn") else None
print("Generator.spawn available:", hasattr(gen, "spawn"), "->", [type(x).__name__ for x in gen2] if gen2 else None)

print("\n== independence sanity (fixture): correlation of 1e6 normals between 2 named streams ==")
a = stream_rng(7, "dodge/dev/situations").standard_normal(1_000_000)
b = stream_rng(7, "dodge/dev/noise").standard_normal(1_000_000)
print("pearson r =", float(np.corrcoef(a, b)[0, 1]))
print("digest of 1000 draws for xproc compare:", hashlib.sha256(stream_rng(7, "dodge/dev/situations", 3).random(1000).tobytes()).hexdigest())
