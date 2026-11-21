"""B2 research probe: where do the ~60+ ms of one small offscreen MuJoCo render go on this machine, and which scene
settings remove them? Renders the same state (DODGE situation A of head_view_proof.py right after one control period)
repeatedly with one 96x96 renderer and times make_current, mjr_render (+ glFinish) and mjr_readPixels separately,
for: default flags; shadows off; shadows + reflection off; shadows off with the robot visual meshes (group 2) hidden;
shadows off with vis.quality.offsamples 0 (no multisampling); and with shadows off at 64x64 and 12x12. Also checks
that switching shadows off leaves the ball/floor/sky classes of the image unchanged. Edits no project file.

    .venv/Scripts/python.exe experiments/proofs/B2_research/render_cost_probe.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from OpenGL import GL  # noqa: E402

from learned_judgment.env import Go2World, load_config  # noqa: E402

EYE = np.array([0.33, 0.0, 0.05])
N = 60


def main() -> None:
    config = load_config()
    config["world"]["save_traces"] = False
    world = Go2World(config, seed=0)
    sit = world.sample_dodge_situation()
    world.reset(sit, attempt_seed=0)
    world.step({"vel_x": sit["dog_forward_speed_m_s"], "vel_y": 0.0, "yaw_rate": 0.0}, 0.02)
    model = world.model
    d = mujoco.MjData(model)
    d.qpos[:] = world.policy.data.qpos
    mujoco.mj_kinematics(model, d)
    mujoco.mj_comPos(model, d)
    mujoco.mj_camlight(model, d)
    base = model.body("base_link").id
    rot = d.xmat[base].reshape(3, 3)
    eye = d.xpos[base] + rot @ EYE
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    cam.elevation = math.degrees(math.asin(rot[2, 0]))
    cam.distance = 1.0
    cam.lookat[:] = eye + rot[:, 0]
    print("GL_RENDERER/GL_VERSION:", end=" ")
    probe = mujoco.Renderer(model, 8, 8)
    probe._gl_context.make_current()
    print(GL.glGetString(GL.GL_RENDERER).decode(), "|", GL.glGetString(GL.GL_VERSION).decode())
    probe.close()
    print("vis.quality shadowsize/offsamples/numslices/numstacks/numquads:", model.vis.quality.shadowsize,
          model.vis.quality.offsamples, model.vis.quality.numslices, model.vis.quality.numstacks, model.vis.quality.numquads)
    # (label, size, shadow, reflection, robot visual meshes (geom group 2) drawn, offsamples)
    variants = (("default_96", 96, True, True, True, None),
                ("shadow_off_96", 96, False, True, True, None),
                ("shadow_reflection_off_96", 96, False, False, True, None),
                ("shadow_off_robot_meshes_hidden_96", 96, False, True, False, None),
                ("shadow_off_offsamples0_96", 96, False, True, True, 0),
                ("shadow_off_robot_hidden_offsamples0_96", 96, False, True, False, 0),
                ("shadow_off_64", 64, False, True, True, None),
                ("shadow_off_12", 12, False, True, True, None),
                ("shadow_off_robot_meshes_hidden_12", 12, False, True, False, None))
    med = lambda a: round(float(np.median(a)) * 1e3, 3)  # noqa: E731
    for round_ in range(2):
        results, images = {}, {}
        for label, size, shadow, reflection, robot, offsamples in variants:
            opt = mujoco.MjvOption()
            opt.geomgroup[2] = int(robot)
            samples = int(model.vis.quality.offsamples)
            if offsamples is not None:
                model.vis.quality.offsamples = offsamples
            r = mujoco.Renderer(model, size, size)
            model.vis.quality.offsamples = samples
            out = np.empty((size, size, 3), np.uint8)

            def prepare():
                r.update_scene(d, camera=cam, scene_option=opt)
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
                r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflection

            t_mc, t_render, t_finish, t_read, t_upd = [], [], [], [], []
            for _ in range(N):
                t0 = time.perf_counter()
                prepare()
                t1 = time.perf_counter()
                r._gl_context.make_current()
                t2 = time.perf_counter()
                mujoco.mjr_render(r._rect, r._scene, r._mjr_context)
                t3 = time.perf_counter()
                GL.glFinish()
                t4 = time.perf_counter()
                mujoco.mjr_readPixels(out, None, r._rect, r._mjr_context)
                t5 = time.perf_counter()
                t_upd.append(t1 - t0); t_mc.append(t2 - t1); t_render.append(t3 - t2); t_finish.append(t4 - t3)
                t_read.append(t5 - t4)
            t_pub = []  # the public path: update_scene (+ flags) + Renderer.render()
            for _ in range(N):
                t0 = time.perf_counter()
                prepare()
                img = r.render()
                t_pub.append(time.perf_counter() - t0)
            images[label] = img
            results[label] = {"update_scene": med(t_upd), "make_current": med(t_mc), "mjr_render": med(t_render),
                              "glFinish": med(t_finish), "mjr_readPixels": med(t_read),
                              "public_update_scene_plus_render_median_ms": med(t_pub),
                              "public_p95_ms": round(float(np.quantile(t_pub, 0.95)) * 1e3, 3), "n": N}
            r.close()
        print(f"--- round {round_ + 1} of 2 (ms, medians over {N} renders of the same state)")
        for label, values in results.items():
            print(label, json.dumps(values))
    for other in ("shadow_off_96", "shadow_off_robot_meshes_hidden_96", "shadow_off_robot_hidden_offsamples0_96"):
        a, b = images["default_96"].astype(int), images[other].astype(int)
        print(f"default_96 vs {other}: max |pixel difference| {int(np.abs(a - b).max())}, pixels differing "
              f"{int((np.abs(a - b).sum(axis=2) > 0).sum())} of {96 * 96}")


if __name__ == "__main__":
    main()
