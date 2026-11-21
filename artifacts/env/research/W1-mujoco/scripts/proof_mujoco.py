import sys, time, numpy as np, mujoco
scene = sys.argv[1]
print("mujoco", mujoco.__version__, "numpy", np.__version__)
m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m)
# 8. Go2 facts
legs = [f"{l}_{p}" for l in ("FL", "FR", "RL", "RR") for p in ("hip", "thigh", "calf")]
assert (m.opt.timestep, m.nq, m.nv, m.nu) == (0.002, 19, 18, 12)
assert [m.actuator(i).name for i in range(m.nu)] == legs
assert [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)] == [n + "_joint" for n in legs]
assert all(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED) and all(m.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
assert np.allclose(m.key_qpos[0], [0, 0, 0.27, 1, 0, 0, 0] + [0, 0.9, -1.8] * 4)
floor, fl = m.geom("floor").id, m.geom("FL").id
assert np.allclose(m.geom_friction[floor], [1, 0.005, 0.0001]) and m.geom_priority[fl] == 1
# 2. MjSpec: add a body with geom and mass before compiling
spec = mujoco.MjSpec.from_file(scene)
b = spec.worldbody.add_body(name="obstacle", pos=[1, 0, 0.1]); b.add_freejoint()
b.add_geom(name="obstacle_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], mass=2.5)
m2 = spec.compile(); assert m2.nbody == m.nbody + 1 and m2.body_mass[m2.body("obstacle").id] == 2.5
# 3. mj_geomDistance on two spheres (radii 0.1 and 0.2, centres 0.5 apart)
ms = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/>'
                                    '<geom type="sphere" size="0.2" pos="0.5 0 0"/></worldbody></mujoco>')
ds = mujoco.MjData(ms); mujoco.mj_forward(ms, ds); ft = np.zeros(6)
dist = mujoco.mj_geomDistance(ms, ds, 0, 1, 10.0, ft); assert abs(dist - 0.2) < 1e-9, dist
print("geomDistance", dist, "fromto", ft)
# 4. offscreen render, no visible window
r = mujoco.Renderer(m, 240, 320); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
r.update_scene(d); img = r.render(); assert img.shape == (240, 320, 3) and img.std() > 5
print("render ok via", mujoco.GLContext.__module__, img.shape, img.dtype); r.close()
# 5. bit-identical qpos after 2000 steps, twice
def run():
    dd = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, dd, 0)
    for _ in range(2000): mujoco.mj_step(m, dd)
    return dd.qpos.tobytes()
assert run() == run(); print("determinism ok")
# 6. friction: foot priority=1 wins over floor; gravity change takes effect immediately
def fl_floor_friction():
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    return [c.friction[0] for c in d.contact[:d.ncon] if {c.geom1, c.geom2} == {floor, fl}][0]
m.geom_friction[floor, 0] = 0.2; assert fl_floor_friction() == 0.8
m.geom_priority[floor] = 2; assert abs(fl_floor_friction() - 0.2) < 1e-12
m.geom_priority[floor] = 0; m.geom_friction[floor, 0] = 1.0
mb = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size="0.1"/></body></worldbody></mujoco>')
mb.opt.gravity[:] = [3.0, 0, -9.0]; db = mujoco.MjData(mb); mujoco.mj_forward(mb, db)
assert np.allclose(db.qacc[:3], [3.0, 0, -9.0]); print("friction/gravity ok")
# 7. contacts between named geoms
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
pairs = d.contact.geom[:d.ncon]
assert any({int(a), int(b)} == {floor, fl} for a, b in pairs); print("contacts", [(m.geom(int(a)).name, m.geom(int(b)).name) for a, b in pairs])
# 8. throughput, zero control, single thread
mujoco.mj_resetDataKeyframe(m, d, 0); d.ctrl[:] = 0; t = time.perf_counter()
for _ in range(10000): mujoco.mj_step(m, d)
sps = 10000 / (time.perf_counter() - t); print(f"steps/s {sps:.0f}  real-time factor {sps * m.opt.timestep:.1f}x")
print("PROOF_OK")
