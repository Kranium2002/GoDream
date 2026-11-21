# Item 8: Go2 menagerie scene facts and single-thread mj_step throughput with zero control.
import sys, time, statistics, os, platform
import mujoco, numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=200)
print("mujoco", mujoco.__version__, "numpy", np.__version__, "python", platform.python_version(), "cpu_count", os.cpu_count())
m = mujoco.MjModel.from_xml_path(sys.argv[1])
o = m.opt
print("timestep", o.timestep, "integrator", mujoco.mjtIntegrator(o.integrator).name, "cone", mujoco.mjtCone(o.cone).name,
      "impratio", o.impratio, "solver", mujoco.mjtSolver(o.solver).name, "iterations", o.iterations, "ls_iterations", o.ls_iterations,
      "gravity", o.gravity, "nativeccd", not (o.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD))
print("nq", m.nq, "nv", m.nv, "nu", m.nu, "na", m.na, "njnt", m.njnt, "nbody", m.nbody, "ngeom", m.ngeom, "nkey", m.nkey, "nsensor", m.nsensor)
print("JOINTS (index name type qposadr dofadr range):")
for j in range(m.njnt):
    print(" ", j, m.joint(j).name or "<free, unnamed>", mujoco.mjtJoint(m.jnt_type[j]).name, m.jnt_qposadr[j], m.jnt_dofadr[j], m.jnt_range[j])
print("ACTUATORS (index name -> joint, trntype dyntype gaintype biastype gainprm[0] biasprm[0:3] ctrlrange forcerange ctrllimited):")
for a in range(m.nu):
    print(" ", a, m.actuator(a).name, "->", m.joint(m.actuator_trnid[a, 0]).name,
          mujoco.mjtTrn(m.actuator_trntype[a]).name, mujoco.mjtDyn(m.actuator_dyntype[a]).name,
          mujoco.mjtGain(m.actuator_gaintype[a]).name, mujoco.mjtBias(m.actuator_biastype[a]).name,
          m.actuator_gainprm[a, 0], m.actuator_biasprm[a, :3], m.actuator_ctrlrange[a], m.actuator_forcerange[a], m.actuator_ctrllimited[a])
for k in range(m.nkey):
    print("KEY", k, repr(m.key(k).name), "qpos", m.key_qpos[k], "ctrl", m.key_ctrl[k])
f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
print("floor geom id", f, "type", mujoco.mjtGeom(m.geom_type[f]).name, "friction", m.geom_friction[f], "condim", m.geom_condim[f], "priority", m.geom_priority[f])
print("foot geoms:", [(n, m.geom_friction[m.geom(n).id], int(m.geom_condim[m.geom(n).id]), int(m.geom_priority[m.geom(n).id]), float(m.geom_size[m.geom(n).id][0])) for n in ("FL","FR","RL","RR")])
print("joint damping/armature/frictionloss (dof 6..17):", m.dof_damping[6:], m.dof_armature[6:], m.dof_frictionloss[6:])
print("total mass", round(float(m.body_subtreemass[1]), 4), "site imu id", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu"))

# Throughput: zero control, from the home keyframe, one MjData, no thread pool (single thread).
d = mujoco.MjData(m)
N = 20000
def bench(c_loop):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = 0
    w0, c0 = time.perf_counter(), time.process_time()
    if c_loop:
        mujoco.mj_step(m, d, nstep=N)
    else:
        for _ in range(N):
            mujoco.mj_step(m, d)
    w, c = time.perf_counter() - w0, time.process_time() - c0
    return N / w, c / w

bench(False)  # warm-up
for label, cl in (("python loop of mj_step(m, d)", False), ("mj_step(m, d, nstep=N) single call", True)):
    res = [bench(cl) for _ in range(5)]
    sps = [r[0] for r in res]
    med = statistics.median(sps)
    print(f"{label}: N={N} steps/s runs={[round(s) for s in sps]} median={med:.0f} min={min(sps):.0f} "
          f"real-time factor (median*dt)={med*m.opt.timestep:.1f}x  cpu/wall={statistics.median(r[1] for r in res):.2f}")
print("ncon at end:", d.ncon, "base z", round(float(d.qpos[2]), 4))

# Extra: holding the home pose with a Python-side PD torque law (what a position-target policy needs on this model).
def bench_pd(kp=30.0, kd=0.5):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    q0 = m.key_qpos[0, 7:].copy()
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    w0 = time.perf_counter()
    for _ in range(N):
        d.ctrl[:] = np.clip(kp * (q0 - d.qpos[7:]) - kd * d.qvel[6:], lo, hi)
        mujoco.mj_step(m, d)
    return N / (time.perf_counter() - w0)
sps = [bench_pd() for _ in range(3)]
med = statistics.median(sps)
print(f"PD hold (kp=30, kd=0.5, python loop): steps/s runs={[round(s) for s in sps]} median={med:.0f} RTF={med*m.opt.timestep:.1f}x  "
      f"ncon {d.ncon} base z {d.qpos[2]:.4f}")
