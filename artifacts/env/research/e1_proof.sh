set -euo pipefail
W="$1"; D="$W/dl"; mkdir -p "$D/go2_model/assets"
E=https://raw.githubusercontent.com/Eximius-Labs/efferent/f51e1af953afe1016dae9ceeff2b5328fa51781d/examples/go2_walk
U=https://raw.githubusercontent.com/unitreerobotics/unitree_mujoco/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/go2
curl -sfL -o "$D/go2_walk.app" "$E/go2_walk.app"
curl -sfL -o "$D/go2.robot.yaml" "$E/go2.robot.yaml"
for f in go2.xml scene.xml; do curl -sfL -o "$D/go2_model/$f" "$U/$f"; done
for m in base_0 base_1 base_2 base_3 base_4 hip_0 hip_1 thigh_0 thigh_1 thigh_mirror_0 thigh_mirror_1 calf_0 calf_1 calf_mirror_0 calf_mirror_1 foot; do curl -sfL -o "$D/go2_model/assets/$m.obj" "$U/assets/$m.obj"; done
(cd "$D" && sha256sum -c --quiet <<'EOF'
89a5bb7e1ff1cc4cd60f4855bf1dd77bd1b313aeee7c19e984fe7c6e37d74147  go2_walk.app
469b610efb7d5de6b2d1f78c0c5a528406a00ed3ec3e6339ec06ce439503a0e6  go2.robot.yaml
2014a3d76e30f17ab9447d8a67bd015291f74fa4d71ae30d005f1a32bd693d4b  go2_model/go2.xml
6c1fda780e7883665d1c84113b9275b6d448f586a8b1c110e438a37417cbccd0  go2_model/scene.xml
8cdc53719002a5ee737d867904e534a6d6fa930b616bdcc1b3131e577d4e46c0  go2_model/assets/base_0.obj
39d2f4724ff64dd5a0640c308fd98902a2946660a8d033ec8dbf368fd5af0dd6  go2_model/assets/base_1.obj
29cae463f95d678e4ede863c36c9b19b0d356b8340412add10670d70bdcc3095  go2_model/assets/base_2.obj
f51ba43bf300e68957a0c1aa83eaf3e4926f4ac7e64a0d6c33ae5d21fcbcf5f6  go2_model/assets/base_3.obj
49ddec0129546e5dd2075d5beacf7f207baa2860aeff7626818e64d2efbf3821  go2_model/assets/base_4.obj
cc29932ab4d251ce3f72389ec4c1f757e44eee138b87e772b648c2bc76b506cd  go2_model/assets/hip_0.obj
3fd5bd9b2ef6abe622c0739d3c67eaa1679b19b47132951972c4892a779ea7cf  go2_model/assets/hip_1.obj
bb3edfe6a7b04f4f4dff1ce7b05d426ddce941e276cc6372fb542055893c71ec  go2_model/assets/thigh_0.obj
3a631822f9f7d2114b1b79dcdff5abf95103630abb7d91d1f3b8b18117132ed2  go2_model/assets/thigh_1.obj
07abaedce761f1dd48647384ac14ba1959276bc78d5a2a62051e8f1c6093933b  go2_model/assets/thigh_mirror_0.obj
db984a3e983eabbc9b38183427cca9ab0bb3bdb049f9eab030e06edda36579d6  go2_model/assets/thigh_mirror_1.obj
4a818cd563c5b7e2a271504442d05fd777d38cbaa028d4a10ad29c9f87bb355e  go2_model/assets/calf_0.obj
51950c3a153ae3208592df86ea11c9d6756f8f645ebca5caf6910c61aeb03817  go2_model/assets/calf_1.obj
6dc0406207a1c08a5953fd5962aca1a270fd06593d30d100205e05c1b32d4af2  go2_model/assets/calf_mirror_0.obj
f84f5af90b914633165beb3a5493ee499169144c5797dcf04c6eb5842f74b31a  go2_model/assets/calf_mirror_1.obj
df9e78a7c0110d02557439010f2a5f11f0ec4fd907b32704145dbd6aa7affc0b  go2_model/assets/foot.obj
EOF
)
cat > "$W/proof.py" <<'EOF'
import sys, time
from pathlib import Path
import numpy as np, mujoco, onnxruntime, efferent
from efferent import Binding, Command, PolicySession, Runner, load_descriptor, load_policy
from efferent.backends.mujoco_backend import MujocoBackend
from efferent.runtime.obs import quat_rotate_inverse
d = Path(sys.argv[1])
m, onnx_path, _ = load_policy(d / "go2_walk.app")
b = Binding(m, load_descriptor(d / "go2.robot.yaml"))
be = MujocoBackend(model_xml=str(d / "go2_model" / "scene.xml"), control_dt=1.0 / m.control.rate_hz,
                   base_pos=[0.0, 0.0, 0.35], start_pose={j.name: j.default for j in m.joints},
                   auto_base_height=True, settle_s=1.0)  # render=False: no window, no GL
cmd = Command(names=["vel_x", "vel_y", "yaw_rate"], values=np.array([0.4, 0.0, 0.0], dtype=np.float32))
t0 = time.perf_counter()
r = Runner(b, PolicySession(m, onnx_path), be, command_source=lambda t: cmd, keep_history=True).run(8.0, soft_start=True)
wall = time.perf_counter() - t0
qa = be._free_qpos_adr
x, z = float(be.data.qpos[qa]), float(be.data.qpos[qa + 2])
tilt = max(float(np.degrees(np.arccos(np.clip(-quat_rotate_inverse(h.state.quat.astype(np.float64), np.array([0.0, 0.0, -1.0]))[2], -1, 1)))) for h in r.history)
print(f"versions efferent={efferent.__version__} mujoco={mujoco.__version__} onnxruntime={onnxruntime.__version__}")
print(f"timestep={be.model.opt.timestep} substeps={be.substeps} rate_hz={m.control.rate_hz} obs_dim={r.history[0].obs.shape[0]} act_dim={r.history[0].action.shape[0]} steps={r.steps}")
print(f"estopped={r.estopped} x={x:.2f}m z={z:.3f}m max_tilt={tilt:.1f}deg wall={wall:.1f}s for 12s sim")
ok = (not r.estopped) and r.steps == 400 and x > 1.6 and z > 0.15 and tilt < 45.0
print("E1 PROOF:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
uv run --no-project --isolated --python 3.13 --no-build --exclude-newer 2026-11-21T00:00:00Z --with efferent==0.5.0 --with mujoco==3.11.0 python -I "$W/proof.py" "$D"