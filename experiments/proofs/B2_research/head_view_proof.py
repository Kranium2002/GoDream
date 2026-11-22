"""B2 research proof: can a forward-looking head view be rendered offscreen from the live Go2 DODGE episode, how fast,
and does the thrown ball have usable contrast at 12x12 grayscale?

Read-only with respect to the project: it imports learned_judgment.env and edits no file there. It builds its OWN
mujoco.Renderer objects on world.model after reset (the ball scene replaces world.model/world.policy at the first ball
episode) and a free MjvCamera whose pose is recomputed every frame from base_link's pose. The pose comes from
world.policy.data: its qpos is copied into a private scratch MjData that gets mj_kinematics + mj_comPos + mj_camlight
before each render, because (a) right after reset() the ball was placed by writing qpos only, so world.policy.data's
geom_xpos still shows the ball at its parking spot, and (b) after mj_step the derived positions lag qpos by one physics
step. world.policy.data itself is never written. finish_episode() is never called, so no trace file is written.
Writes PNGs only into experiments/proofs/B2_research/head_view/.

Run from the repo root:
    .venv/Scripts/python.exe experiments/proofs/B2_research/head_view_proof.py
"""
from __future__ import annotations

import json
import math
import os
import platform
import struct
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from learned_judgment.env import Go2World, load_config  # noqa: E402
from learned_judgment.env.throws import sample_dodge_situations  # noqa: E402

OUT = Path(__file__).resolve().parent / "head_view"
EYE_TRUNK = np.array([0.33, 0.0, 0.05])  # eye offset in the trunk (base_link) frame, m
LOOK_DISTANCE = 1.0  # free-camera distance from eye to lookat; only the direction matters
CONTROL_DT = 0.02
MIN_STEPS = 50  # >= 50 stepped frames (1.0 s) plus the frame at t = 0
MAX_STEPS = 80  # 1.6 s: the longest flight of the throw protocol is 1.59 s
GRAY = np.array([0.299, 0.587, 0.114])  # luminance weights for RGB -> gray


def write_png(path: Path, image: np.ndarray) -> None:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = image.shape[:2]
    ctype = 0 if image.ndim == 2 else 2
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def sync_scratch(model, data, scratch) -> None:
    scratch.qpos[:] = data.qpos
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    mujoco.mj_camlight(model, scratch)


def set_head_camera(cam: mujoco.MjvCamera, d: mujoco.MjData, base: int, pitch_up_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Free camera at base_link pose * EYE_TRUNK looking along trunk +x, tilted up by pitch_up_deg. A free camera has
    no roll: trunk roll is dropped; trunk pitch and yaw are kept. Returns (eye, trunk rotation)."""
    rot = d.xmat[base].reshape(3, 3).copy()  # body -> world; column 0 is trunk +x
    eye = d.xpos[base] + rot @ EYE_TRUNK
    fwd = rot[:, 0] / np.linalg.norm(rot[:, 0])
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
    cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2])))) + pitch_up_deg
    cam.distance = LOOK_DISTANCE
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    cam.lookat[:] = eye + LOOK_DISTANCE * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return eye, rot


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float64) @ GRAY) / 255.0


def block_mean(a: np.ndarray, out: int) -> np.ndarray:
    k = a.shape[0] // out
    return a.reshape(out, k, out, k).mean(axis=(1, 3))


def stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "min": round(float(values.min()), 4), "max": round(float(values.max()), 4),
            "mean": round(float(values.mean()), 4)}


def ms(values) -> dict:
    a = np.array(values) * 1e3
    return {"median_ms": round(float(np.median(a)), 3), "p95_ms": round(float(np.quantile(a, 0.95)), 3), "n": len(a)}


def flight_time(situation: dict, g: float) -> float:
    vz, h = situation["ball_vel_z_m_s"], situation["ball_rel_z_m"]
    return (vz + math.sqrt(vz * vz + 2 * g * h)) / g


def visibility_study(config: dict, n: int, combos: list[tuple[float, float]]) -> dict:
    """Analytic, no simulator: share of throws from the project's own sampler whose ball centre lies inside a square
    head-camera frustum (eye at EYE_TRUNK on a level trunk moving straight at the situation's forward speed, camera
    yaw = heading, tilted up by pitch). Flight sampled every 0.02 s until the ball is back at trunk-origin height."""
    rows = sample_dodge_situations(config, np.random.default_rng(12345), n)
    keys = ["ball_rel_x_m", "ball_rel_y_m", "ball_rel_z_m", "ball_vel_x_m_s", "ball_vel_y_m_s", "ball_vel_z_m_s",
            "dog_forward_speed_m_s"]
    a = np.array([[r[k] for k in keys] for r in rows])
    g = config["ball"]["gravity_m_s2"]
    p0, v, s = a[:, 0:3], a[:, 3:6], a[:, 6]
    flight = (v[:, 2] + np.sqrt(v[:, 2] ** 2 + 2 * g * p0[:, 2])) / g
    t = np.arange(0.0, 1.6 + 1e-9, 0.02)
    rel = p0[:, None, :] + v[:, None, :] * t[None, :, None]
    rel[:, :, 2] -= 0.5 * g * t[None, :] ** 2
    rel[:, :, 0] -= s[:, None] * t[None, :]
    rel -= EYE_TRUNK
    alive = t[None, :] <= flight[:, None]
    late = np.clip(np.round((flight - 0.4) / 0.02).astype(int), 0, len(t) - 1)  # 0.4 s before arrival
    elev = np.degrees(np.arctan2(rel[..., 2], np.hypot(rel[..., 0], rel[..., 1])))
    bearing = np.degrees(np.arctan2(rel[..., 1], rel[..., 0]))
    out = {
        "n_throws": n,
        "flight_s_quantiles_0_50_100": [round(float(q), 3) for q in np.quantile(flight, [0, 0.5, 1])],
        "abs_bearing_from_eye_at_release_deg_q50_q90_max": [round(float(q), 1) for q in np.quantile(np.abs(bearing[:, 0]), [0.5, 0.9, 1.0])],
        "elevation_from_eye_at_release_deg_q10_q50_q90": [round(float(q), 1) for q in np.quantile(elev[:, 0], [0.1, 0.5, 0.9])],
        "max_elevation_during_flight_deg_q10_q50_q90": [round(float(q), 1) for q in np.quantile(np.where(alive, elev, -90).max(axis=1), [0.1, 0.5, 0.9])],
        "abs_bearing_0.4s_before_arrival_deg_q50_q90": [round(float(q), 1) for q in np.quantile(np.abs(bearing[np.arange(n), late]), [0.5, 0.9])],
        "by_fovy_and_pitch_up_deg": {},
    }
    for fovy, pitch in combos:
        tan = math.tan(math.radians(fovy) / 2)
        c, sn = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        xc = rel[..., 0] * c + rel[..., 2] * sn  # camera forward
        zc = -rel[..., 0] * sn + rel[..., 2] * c  # camera up
        inside = (xc > 0) & (np.abs(rel[..., 1]) <= tan * xc) & (np.abs(zc) <= tan * xc)
        frac = (inside & alive).sum(axis=1) / alive.sum(axis=1)
        out["by_fovy_and_pitch_up_deg"][f"fovy{fovy:g}_pitch{pitch:g}"] = {
            "in_view_at_release": round(float(inside[:, 0].mean()), 3),
            "in_view_at_some_time_in_first_0.3s": round(float(inside[:, :16].any(axis=1).mean()), 3),
            "in_view_0.4s_before_arrival": round(float(inside[np.arange(n), late].mean()), 3),
            "mean_fraction_of_flight_in_view": round(float(frac.mean()), 3),
            "ball_diam_px_at_12x12_at_1m_3m_(image_centre)": [round(0.0939 / d / tan * 12, 2) for d in (1.0, 3.0)],
        }
    return out


def timing_pass(world: Go2World, situation: dict, size: int) -> dict:
    """One episode, ONE renderer of size x size (one GL context): update_scene and render timed separately."""
    t0 = time.perf_counter()
    world.reset(situation, attempt_seed=0)
    reset_s = time.perf_counter() - t0
    model, data = world.model, world.policy.data
    base = model.body("base_link").id
    scratch = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, size, size)
    cam = mujoco.MjvCamera()
    command = {"vel_x": situation["dog_forward_speed_m_s"], "vel_y": 0.0, "yaw_rate": 0.0}
    upd, ren, sync, step = [], [], [], []
    for i in range(MIN_STEPS + 1):
        if i > 0:
            ts = time.perf_counter()
            world.step(command, CONTROL_DT)
            step.append(time.perf_counter() - ts)
        ts = time.perf_counter()
        sync_scratch(model, data, scratch)
        set_head_camera(cam, scratch, base, 0.0)
        sync.append(time.perf_counter() - ts)
        ts = time.perf_counter()
        renderer.update_scene(scratch, camera=cam)
        upd.append(time.perf_counter() - ts)
        ts = time.perf_counter()
        renderer.render()
        ren.append(time.perf_counter() - ts)
    static = []
    for _ in range(50):  # same state again: render() alone, no physics in between
        ts = time.perf_counter()
        renderer.render()
        static.append(time.perf_counter() - ts)
    renderer.close()
    return {"size": size, "reset_wall_s": round(reset_s, 3), "step_wall_s_1.0s_(50_control_periods)": round(sum(step), 4),
            "step_ms_per_control_period": ms(step), "scratch_sync_plus_camera_ms": ms(sync), "update_scene_ms": ms(upd),
            "render_ms": ms(ren), "update_scene_plus_render_ms": ms(np.array(upd) + np.array(ren)),
            "render_only_same_state_x50_ms": ms(static)}


def analysis_pass(world: Go2World, situation: dict, label: str, fovy: float, pitch_up_deg: float,
                  ball_rgba: list[float] | None = None) -> dict:
    """One episode with a 96x96 RGB renderer and a 96x96 segmentation renderer (pixel classes floor/sky/ball/robot).
    ball_rgba, if given, recolours the ball geom for this pass only (rendering only: rgba has no effect on physics)."""
    world.reset(situation, attempt_seed=0)
    model, data = world.model, world.policy.data
    base = model.body("base_link").id
    floor, ball_geom = model.geom("floor").id, model.geom("ball").id
    radius = world.config["ball"]["radius_m"]
    scratch = mujoco.MjData(model)
    fovy0 = float(model.vis.global_.fovy)
    model.vis.global_.fovy = fovy  # the free camera's vertical field of view (degrees)
    rgba0 = model.geom_rgba[ball_geom].copy()
    if ball_rgba is not None:
        model.geom_rgba[ball_geom] = ball_rgba
    rgb_r, seg_r = mujoco.Renderer(model, 96, 96), mujoco.Renderer(model, 96, 96)
    seg_r.enable_segmentation_rendering()
    cam = mujoco.MjvCamera()
    command = {"vel_x": situation["dog_forward_speed_m_s"], "vel_y": 0.0, "yaw_rate": 0.0}
    flight = flight_time(situation, world.config["ball"]["gravity_m_s2"])
    n_steps = min(MAX_STEPS, max(MIN_STEPS, int(math.ceil(flight / CONTROL_DT)) + 5))
    tan_half = math.tan(math.radians(fovy) / 2)
    frames, cam_err = [], 0.0
    gray_all = {"floor": [], "sky": [], "ball": [], "robot": []}
    for i in range(n_steps + 1):
        obs = world.step(command, CONTROL_DT) if i > 0 else None
        sync_scratch(model, data, scratch)
        eye, rot = set_head_camera(cam, scratch, base, pitch_up_deg)
        rgb_r.update_scene(scratch, camera=cam)
        rgb = rgb_r.render()
        # mono rendering uses the midpoint of the two stereo eyes (each offset by vis.global_.ipd / 2)
        mid = 0.5 * (np.asarray(rgb_r.scene.camera[0].pos) + np.asarray(rgb_r.scene.camera[1].pos))
        cam_err = max(cam_err, float(np.abs(mid - eye).max()))
        seg_r.update_scene(scratch, camera=cam)
        ids = seg_r.render()
        ball_pos = scratch.xpos[model.body("ball").id].copy()
        rel = rot.T @ (ball_pos - eye)
        gray = to_gray(rgb)
        is_geom = ids[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        masks = {"floor": is_geom & (ids[..., 0] == floor), "ball": is_geom & (ids[..., 0] == ball_geom), "sky": ids[..., 0] == -1}
        masks["robot"] = ~(masks["floor"] | masks["ball"] | masks["sky"])
        for k, m in masks.items():
            gray_all[k].append(gray[m])
        dist = float(np.linalg.norm(ball_pos - eye))
        rec = {"i": i, "t_s": round(i * CONTROL_DT, 2), "eye_ball_dist_m": round(dist, 3),
               "bearing_deg": round(math.degrees(math.atan2(rel[1], rel[0])), 1),
               "elev_deg_vs_trunk_x": round(math.degrees(math.atan2(rel[2], math.hypot(rel[0], rel[1]))), 1),
               "ball_px_96": int(masks["ball"].sum()), "robot_px_96": int(masks["robot"].sum()),
               "expected_diam_px_12_at_centre": round(radius / dist / tan_half * 12, 3)}
        if masks["ball"].any():
            cover12, gray12 = block_mean(masks["ball"].astype(float), 12), block_mean(gray, 12)
            area12 = float(cover12.sum())
            cell = np.unravel_index(int(np.argmax(cover12)), cover12.shape)
            r0, c0 = cell[0] * 8, cell[1] * 8
            ring = [(rr, cc) for rr in range(cell[0] - 1, cell[0] + 2) for cc in range(cell[1] - 1, cell[1] + 2)
                    if 0 <= rr < 12 and 0 <= cc < 12 and cover12[rr, cc] == 0]
            ring_gray = float(np.mean([gray12[p] for p in ring])) if ring else float("nan")
            rec.update({
                "ball_equiv_diam_px_12": round(2 * math.sqrt(area12 / math.pi), 3),
                "cells_touched_12": int((cover12 > 0).sum()), "max_cell_cover_12": round(float(cover12.max()), 3),
                "ball_px_gray_mean": round(float(gray[masks["ball"]].mean()), 3),
                "max_cell_gray_12": round(float(gray12[cell]), 3), "ring_gray_12": round(ring_gray, 3),
                "contrast_12": round(float(gray12[cell]) - ring_gray, 3),
                "max_cell_other_px_sky_floor_robot": [int(masks[k][r0:r0 + 8, c0:c0 + 8].sum()) for k in ("sky", "floor", "robot")],
            })
        frames.append((rec, rgb, gray))
    model.vis.global_.fovy = fovy0
    model.geom_rgba[ball_geom] = rgba0
    rgb_r.close()
    seg_r.close()

    recs = [f[0] for f in frames]
    visible = [r for r in recs if r["ball_px_96"] > 0]
    footprint = {}
    for target in (3.0, 2.0, 1.0, 0.5):
        near = [r for r in visible if abs(r["eye_ball_dist_m"] - target) <= 0.3]
        if near:
            best = min(near, key=lambda r: abs(r["eye_ball_dist_m"] - target))
            footprint[str(target)] = {k: best.get(k) for k in (
                "t_s", "eye_ball_dist_m", "ball_px_96", "ball_equiv_diam_px_12", "expected_diam_px_12_at_centre",
                "cells_touched_12", "max_cell_cover_12", "ball_px_gray_mean", "max_cell_gray_12", "ring_gray_12",
                "contrast_12", "max_cell_other_px_sky_floor_robot", "bearing_deg", "elev_deg_vs_trunk_x")}
        else:
            closest = min(recs, key=lambda r: abs(r["eye_ball_dist_m"] - target))
            footprint[str(target)] = {"in_view": False, "closest_frame": {k: closest[k] for k in (
                "t_s", "eye_ball_dist_m", "bearing_deg", "elev_deg_vs_trunk_x")},
                "expected_diam_px_12_if_at_centre": round(radius / target / tan_half * 12, 3)}
    OUT.mkdir(parents=True, exist_ok=True)
    last = max([r["i"] for r in visible], default=len(frames) - 1)
    first = min([r["i"] for r in visible], default=0)
    picks = np.unique(np.linspace(first, last, 6).round().astype(int))
    saved = []
    for i in picks:
        rec, rgb, gray = frames[i]
        stem = f"{label}_f{i:02d}_t{rec['t_s']:.2f}"
        write_png(OUT / f"{stem}_rgb96.png", rgb)
        g12 = block_mean(gray, 12)
        write_png(OUT / f"{stem}_gray12_x16.png", np.kron(np.round(g12 * 255), np.ones((16, 16))).astype(np.uint8))
        saved.append([stem, rec["eye_ball_dist_m"], rec["ball_px_96"]])
    return {"situation": {k: round(v, 4) for k, v in situation.items()}, "fovy_deg": fovy, "pitch_up_deg": pitch_up_deg,
            "ball_rgba": None if ball_rgba is None else list(ball_rgba),
            "flight_s": round(flight, 3), "n_frames": len(recs), "frames_with_ball_visible": len(visible),
            "visible_t_range_s": [visible[0]["t_s"], visible[-1]["t_s"]] if visible else None,
            "render_eye_vs_requested_max_abs_m": cam_err,
            "gray_levels_96": {k: stats(np.concatenate(v)) for k, v in gray_all.items()},
            "footprint_at_distance": footprint, "saved_pngs_[stem,dist_m,ball_px_96]": saved,
            "per_frame": recs}


def main() -> None:
    config = load_config()
    config["world"]["save_traces"] = False  # in memory only; finish_episode is never called anyway
    report: dict = {"mujoco": mujoco.__version__, "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
                    "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()}
    t0 = time.perf_counter()
    world = Go2World(config, seed=0)
    report["world_init_wall_s"] = round(time.perf_counter() - t0, 3)
    sit_a = world.sample_dodge_situation()
    sit_b, b_note = None, None
    for k in range(200):
        cand = world.sample_dodge_situation()
        if abs(math.degrees(math.atan2(cand["ball_rel_y_m"], cand["ball_rel_x_m"] - EYE_TRUNK[0]))) < 12.0:
            sit_b = cand
            b_note = (f"draw #{k + 2} of Go2World(seed=0).sample_dodge_situation(): the first with |release bearing from "
                      "the eye| < 12 deg, CHOSEN FOR VISIBILITY (a rendering probe, not a sample)")
            break

    r = mujoco.Renderer(world.model, 8, 8)
    report["gl_context_class"] = f"{type(r._gl_context).__module__}.{type(r._gl_context).__name__}"
    r.close()

    report["timing_A"] = [timing_pass(world, sit_a, size) for size in (96, 64, 12)]
    m = world.model  # the ball scene after the first reset
    base = m.body("base_link").id
    report["model"] = {
        "ncam": int(m.ncam), "camera_names": [m.camera(i).name for i in range(m.ncam)],
        "sites": [[m.site(i).name, m.body(int(m.site_bodyid[i])).name, m.site_pos[i].round(4).tolist()] for i in range(m.nsite)],
        "vis.global_.fovy_deg": float(m.vis.global_.fovy), "vis.global_.ipd": float(m.vis.global_.ipd),
        "offwidth_x_offheight": [int(m.vis.global_.offwidth), int(m.vis.global_.offheight)],
        "near_clip_m_(znear*extent)": round(float(m.vis.map.znear * m.stat.extent), 4),
        "headlight_active_ambient_diffuse": [int(m.vis.headlight.active), m.vis.headlight.ambient.round(3).tolist(),
                                             m.vis.headlight.diffuse.round(3).tolist()],
        "ntex": int(m.ntex), "nmat": int(m.nmat), "nlight": int(m.nlight),
        "floor_rgba_matid_group": [m.geom_rgba[m.geom("floor").id].tolist(), int(m.geom_matid[m.geom("floor").id]), int(m.geom_group[m.geom("floor").id])],
        "ball_rgba_matid_group": [m.geom_rgba[m.geom("ball").id].tolist(), int(m.geom_matid[m.geom("ball").id]), int(m.geom_group[m.geom("ball").id])],
        "default_visible_geomgroups": mujoco.MjvOption().geomgroup.tolist(),
    }
    xs_axis = []
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != base or m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[gid]
        verts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[gid])
        v = verts @ rot.reshape(3, 3).T + m.geom_pos[gid]
        near_axis = v[(np.abs(v[:, 1]) < 0.03) & (np.abs(v[:, 2] - EYE_TRUNK[2]) < 0.03)]
        if near_axis.size:
            xs_axis.append(float(near_axis[:, 0].max()))
    report["model"]["trunk_visual_mesh_front_x_near_eye_line_m"] = round(max(xs_axis), 4) if xs_axis else None

    report["A_fovy45"] = analysis_pass(world, sit_a, "A_fovy45", 45.0, 0.0)
    report["B_note"] = b_note
    report["B_fovy45"] = analysis_pass(world, sit_b, "B_fovy45", 45.0, 0.0)
    report["A_fovy90_pitch15"] = analysis_pass(world, sit_a, "A_fovy90_pitch15", 90.0, 15.0)
    report["B_fovy90_pitch15"] = analysis_pass(world, sit_b, "B_fovy90_pitch15", 90.0, 15.0)
    report["B_fovy90_pitch15_white_ball"] = analysis_pass(world, sit_b, "B_fovy90_pitch15_white_ball", 90.0, 15.0,
                                                          ball_rgba=[1.0, 1.0, 1.0, 1.0])
    report["visibility_study"] = visibility_study(config, 20000, [(45, 0), (60, 0), (90, 0), (90, 15), (90, 25), (120, 0), (120, 15)])

    for key, value in report.items():
        if isinstance(value, dict) and "per_frame" in value:
            print(f"\n=== {key}")
            for k, v in value.items():
                if k != "per_frame":
                    print(f"{k}: {json.dumps(v)}")
            print("per_frame:")
            for row in value["per_frame"]:
                print("  " + json.dumps(row))
        else:
            print(f"\n=== {key}: {json.dumps(value, indent=1)}")


if __name__ == "__main__":
    main()
