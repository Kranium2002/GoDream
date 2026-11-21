# Item 5: are two identical 2000-step runs bit-identical (compare qpos bytes)?
import sys, hashlib
import mujoco, numpy as np

scene = sys.argv[1]
print("mujoco", mujoco.__version__, "numpy", np.__version__)

def run(m, d, ctrl_seq):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for k in range(2000):
        d.ctrl[:] = ctrl_seq[k]
        mujoco.mj_step(m, d)
    return d.qpos.copy()

def h(a): return hashlib.sha256(a.tobytes()).hexdigest()

m1 = mujoco.MjModel.from_xml_path(scene)
zero = np.zeros((2000, m1.nu))
rng = np.random.default_rng(0)
noisy = rng.uniform(-1, 1, (2000, m1.nu)) * m1.actuator_ctrlrange[:, 1] * 0.5  # excites contacts

for label, seq in (("zero ctrl", zero), ("seeded random ctrl", noisy)):
    a = run(m1, mujoco.MjData(m1), seq)                       # run A: fresh data
    m2 = mujoco.MjModel.from_xml_path(scene)                  # run B: fresh model + data
    b = run(m2, mujoco.MjData(m2), seq)
    dd = mujoco.MjData(m1); run(m1, dd, noisy)                # run C: data reused after another run
    c = run(m1, dd, seq)
    print(f"[{label}] A {h(a)[:32]}  B {h(b)[:32]}  C {h(c)[:32]}  A==B bytes: {a.tobytes()==b.tobytes()}  A==C bytes: {a.tobytes()==c.tobytes()}")
    print(f"[{label}] final base z {a[2]:.6f} time {2000*m1.opt.timestep:.3f}s")
