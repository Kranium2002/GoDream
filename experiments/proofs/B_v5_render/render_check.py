"""v5 render checks on the real Go2 world (Person B), no augmentation unless stated:

1. Exactness: render_windows_kinematic against render_windows (physics episode) on 6 punch scenarios of mixed kinds
   from the dream split x 3 pre-contact decision frames x each camera rate of vision.sim_camera_rates_hz (plus one
   empty scene): max |difference|.
2. PNG strips into results/runs/_v5_smoke/: per scene 8 raw frames (12 x 12, upscaled x16) and below them the 7 event
   frames as the student reads them (observed.encode_window_features "events", vision.event_threshold; ON white, OFF
   black, none mid grey).
3. Still scenes (empty, static): event cells per adjacent frame pair over whole episodes at each camera rate (median,
   max); without augmentation, and for information with the closed-loop episode augmentation (incl. drift).

Run from the repository root: python experiments/proofs/B_v5_render/render_check.py
"""
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from learned_judgment.config import load_config, require
from learned_judgment.observed import encode_window_features
from learned_judgment.skills import dodge_vision as dv
from learned_judgment.vision import VisionSettings

OUT = Path("results/runs/_v5_smoke")
SCALE = 16


def write_png(path: Path, grey: np.ndarray) -> None:
    """8-bit greyscale PNG of a (H, W) array in [0, 1]."""
    data = np.round(np.clip(grey, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = data.shape
    raw = b"".join(b"\x00" + data[r].tobytes() for r in range(h))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def events(window: np.ndarray, threshold: float) -> np.ndarray:
    """The student's "events" encoding of one (F, N, N) window, as (F - 1, N, N)."""
    f, n, _ = window.shape
    return encode_window_features(window[None].astype(np.float32), "events", threshold).reshape(f - 1, n, n)


def strip(window: np.ndarray, threshold: float) -> np.ndarray:
    f, n, _ = window.shape
    cell, gap = n * SCALE, 4
    image = np.full((2 * cell + gap, f * cell + (f - 1) * gap), 0.25, dtype=np.float64)

    def up(x: np.ndarray) -> np.ndarray:
        return np.kron(x, np.ones((SCALE, SCALE)))

    for k in range(f):
        image[:cell, k * (cell + gap): k * (cell + gap) + cell] = up(window[k])
    for k, e in enumerate(events(window, threshold)):
        x0 = k * (cell + gap) + (cell + gap) // 2  # each event frame sits between its two raw frames
        image[cell + gap:, x0: x0 + cell] = up(e)
    return image


def has_person(seed: int, background) -> bool:
    """Replays the first draws of dodge_vision.background_field."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(background.patches[0], background.patches[1] + 1))
    rng.uniform(size=(n, n))
    rng.random()
    return bool(rng.random() < background.person_probability)


def main() -> int:
    print("module", dv.__file__)
    config = load_config()
    settings = VisionSettings.from_config(config)
    threshold = float(require(config, "vision.event_threshold"))
    n_frames = settings.frames
    ep = dv._episode(config)
    rates = ep.timing.camera_rates_hz
    times = dv.frame_times(config)
    eligible = {rate: dv._eligible_frames(config, n_frames, rate) for rate in rates}
    background = dv._background(config)
    world = dv._make_world(config)
    view = dv.HeadView(config)
    t0 = time.perf_counter()
    stand = dv.record_stand(world, config)
    print(f"stand recorded in {time.perf_counter() - t0:.1f} s at {stand['frame_periods'].size} control periods")

    scenarios, _ = dv._split_plan(config, "dream", 300)
    by_kind: dict = {}
    for s in scenarios:
        by_kind.setdefault(s["kind"], []).append(s)

    # 1. exactness, every rate
    wanted = ["incoming_center", "incoming_left", "incoming_right", "ambiguous", "static", "background_motion"]
    print("exactness: kinematic vs physics render, no augmentation")
    chosen = []
    for kind in wanted:
        for s in by_kind[kind]:
            limit = dv._kinematic_limit_s(config, s, stand)
            per_rate = {}
            for rate in rates:
                own = dv._frames_before_arrival(config, s, eligible[rate])
                own = own[times[own] < limit]
                if own.size >= 3:
                    per_rate[rate] = [int(own[0]), int(own[own.size // 2]), int(own[-1])]
            if len(per_rate) == len(rates):
                chosen.append((s, per_rate, limit))
                break
    empty = by_kind["empty"][0]
    chosen.append((empty, {r: [int(eligible[r][0]), int(eligible[r][eligible[r].size // 2]), int(eligible[r][-1])]
                           for r in rates}, np.inf))
    worst = {rate: 0.0 for rate in rates}
    for s, per_rate, limit in chosen:
        for rate, frames in per_rate.items():
            kin = dv.render_windows_kinematic(world, view, s, frames, settings, stand, None, [rate] * 3)
            phys = dv.render_windows(world, view, s, s["scenario_id"], frames, settings, None, [rate] * 3)
            diff = float(np.abs(kin - phys).max())
            if s["kind"] != "empty":
                worst[rate] = max(worst[rate], diff)
            print(f"  scenario {s['scenario_id']:3d} {s['kind']:17s} person "
                  f"{has_person(s['visual']['background_seed'], background)!s:5s} {rate:4.0f} Hz frames {frames} "
                  f"t {[round(float(times[f]), 2) for f in frames]} limit {limit:.3f} s: max |diff| {diff:.6f}")
    for rate in rates:
        print(f"exactness at {rate:g} Hz: max |difference| over the {len(chosen) - 1} punch scenarios x 3 frames: "
              f"{worst[rate]:.6f}")
    print(f"exactness overall: {max(worst.values()):.6f}")

    # 2. PNG strips
    OUT.mkdir(parents=True, exist_ok=True)
    incoming = [s for k in ("incoming_center", "incoming_left", "incoming_right") for s in by_kind[k]]
    aim_x = float(require(config, "skills.dodge_vision.punch.aim_x_m"))
    margin = float(require(config, "skills.dodge_vision.pre_contact_margin_s"))

    def late_frame(s, rate):
        return int(dv._frames_before_arrival(config, s, eligible[rate])[-1])

    def pre_contact(s, rate) -> bool:  # the latest kept frame lies before the arrival less the margin
        arrival = (s["situation"]["ball_rel_x_m"] - aim_x) / -s["situation"]["ball_vel_x_m_s"]
        return times[late_frame(s, rate)] <= arrival - margin + 1e-9

    def late_window(s, rate):
        frame = late_frame(s, rate)  # the latest pre-contact decision frame at that rate (late in the approach)
        window = dv.render_windows(world, view, s, s["scenario_id"], [frame], settings, None, [rate])[0]
        return frame, window.reshape(n_frames, settings.frame_size, settings.frame_size)

    def visible_late(person_wanted: bool, rate: float):
        """The first incoming scene (person blob as wanted, pre-contact at that rate) whose latest pre-contact window
        still shows the punch moving in its newest frame pair (>= 3 event cells); an illustration choice only. Also
        counts the scenes passed over (punch out of view, usually above the head camera, in that newest pair)."""
        skipped = 0
        for s in incoming:
            if has_person(s["visual"]["background_seed"], background) != person_wanted or not pre_contact(s, rate):
                continue
            frame, window = late_window(s, rate)
            if (events(window, threshold)[-1] != 0.5).sum() >= 3:
                return s, skipped
            skipped += 1
        raise RuntimeError("no incoming scene shows the punch in its latest pre-contact window")

    plain, skipped_plain = visible_late(False, 15.0)
    person, skipped_person = visible_late(True, 30.0)
    print(f"strip choice: incoming scenes passed over because the punch had left the view by the newest frame pair of "
          f"the latest pre-contact window: {skipped_plain} (no person, 15 Hz), {skipped_person} (person, 30 Hz)")
    static, still_empty = by_kind["static"][1], by_kind["empty"][1]
    for name, s, rate in (("incoming_patchy_15hz", plain, 15.0), ("incoming_person_30hz", person, 30.0),
                          ("static_30hz", static, 30.0), ("empty_30hz", still_empty, 30.0)):
        frame, window = late_window(s, rate)
        path = OUT / f"strip_{name}.png"
        write_png(path, strip(window, threshold))
        fired = (events(window, threshold) != 0.5).sum(axis=(1, 2))
        periods = dv._window_periods(ep.timing, frame, rate, n_frames)
        print(f"strip {path}: scenario {s['scenario_id']} {s['kind']} object_grey {s['visual']['object_grey']:.2f} "
              f"background_mean {s['visual']['background_mean']:.2f} person "
              f"{has_person(s['visual']['background_seed'], background)} rate {rate:g} Hz decision frame {frame} "
              f"(t {times[frame]:.2f} s) window periods {periods.tolist()}; event cells per pair {fired.tolist()}")
    for old in ("strip_incoming_patchy.png", "strip_incoming_person.png", "strip_static.png", "strip_empty.png"):
        (OUT / old).unlink(missing_ok=True)  # the earlier (15 Hz-only) strips

    # 3. still scenes over whole episodes, every rate
    duration = float(require(config, "skills.dodge_vision.episode_duration_s"))
    for augmented in (False, True):
        label = "with closed-loop episode augmentation (incl. drift)" if augmented else "no augmentation"
        for kind in ("empty", "static"):
            for rate in rates:
                periods = dv._rate_periods(rate, ep.timing.period_s, ep.timing.n_periods, duration)
                counts, peaks = [], []
                for s in by_kind[kind][:5]:
                    dv._reset(world, s, s["scenario_id"])
                    frames = dv._frames_at(world, view, settings, periods.tolist(), ep, s.get("visual"))
                    if augmented:
                        aug = dv._EpisodeAugmentation(np.random.default_rng(s["scenario_id"]), ep.augmentation, n_frames,
                                                      len(frames))
                        frames = [aug.apply(f) for f in frames]
                    frames = np.stack(frames)
                    change = np.abs(np.diff(frames, axis=0))
                    counts.append((change > threshold).sum(axis=(1, 2)))
                    peaks.append(float(change.max()))
                allc = np.concatenate(counts)
                print(f"{kind} {rate:4.0f} Hz, {label}: 5 scenes, {allc.size} frame pairs: event cells per pair median "
                      f"{np.median(allc):g}, max {allc.max()}, pairs with any event {(allc > 0).sum()}; largest "
                      f"|change| {max(peaks):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
