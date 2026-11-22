"""learned_judgment.vision: the image-window contract (pure functions, real config.yaml)."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from learned_judgment.config import load_config, require
from learned_judgment.vision import (
    FrameWindow,
    VisionSettings,
    augment_window,
    flatten_window,
    formula_image,
    frame_control_periods,
    preprocess_frame,
    take_frames,
    unflatten_window,
    write_test_vectors,
)

CONFIG = load_config()
SETTINGS = VisionSettings.from_config(CONFIG)
N = SETTINGS.frame_size


def brute_force(rgb: np.ndarray, settings: VisionSettings) -> np.ndarray:
    height, width = rgb.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    w0, w1, w2 = settings.luma_weights
    n = settings.frame_size
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            total, count = 0.0, 0
            for r in range((i * side) // n, ((i + 1) * side) // n):
                for c in range((j * side) // n, ((j + 1) * side) // n):
                    red, green, blue = (float(v) for v in rgb[top + r, left + c])
                    total += w0 * red + w1 * green + w2 * blue
                    count += 1
            out[i, j] = total / count / settings.pixel_scale
    return out.astype(np.float32)


def test_settings_from_config():
    assert SETTINGS.frame_size == require(CONFIG, "vision.frame_size")
    assert SETTINGS.frames == require(CONFIG, "vision.frames")
    assert SETTINGS.luma_weights == tuple(require(CONFIG, "vision.luma_weights"))
    assert SETTINGS.n_features == SETTINGS.frames * N * N
    for variant in require(CONFIG, "vision.frame_variants"):
        assert VisionSettings.from_config(CONFIG, frames=variant).frames == variant
    with pytest.raises(ValueError):
        VisionSettings.from_config(CONFIG, frames=max(require(CONFIG, "vision.frame_variants")) + 1)


def test_settings_validation():
    with pytest.raises(ValueError):
        VisionSettings(frame_size=0, frame_rate_hz=30.0, frames=8, luma_weights=(0.3, 0.6, 0.1), pixel_scale=255.0)
    with pytest.raises(ValueError):
        VisionSettings(frame_size=12, frame_rate_hz=0.0, frames=8, luma_weights=(0.3, 0.6, 0.1), pixel_scale=255.0)
    with pytest.raises(ValueError):
        VisionSettings(frame_size=12, frame_rate_hz=30.0, frames=0, luma_weights=(0.3, 0.6, 0.1), pixel_scale=255.0)
    with pytest.raises(ValueError):
        VisionSettings(frame_size=12, frame_rate_hz=30.0, frames=8, luma_weights=(0.3, 0.6), pixel_scale=255.0)
    with pytest.raises(ValueError):
        VisionSettings(frame_size=12, frame_rate_hz=30.0, frames=8, luma_weights=(0.3, float("nan"), 0.1), pixel_scale=255.0)
    with pytest.raises(ValueError):
        VisionSettings(frame_size=12, frame_rate_hz=30.0, frames=8, luma_weights=(0.3, 0.6, 0.1), pixel_scale=-1.0)


@pytest.mark.parametrize("height, width", [(96, 96), (480, 640), (37, 53), (12, 12)])
def test_preprocess_matches_brute_force(height, width):
    rng = np.random.default_rng(height * 1000 + width)
    rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    frame = preprocess_frame(rgb, SETTINGS)
    assert frame.shape == (N, N)
    assert frame.dtype == np.float32
    np.testing.assert_allclose(frame, brute_force(rgb, SETTINGS), rtol=0.0, atol=1e-6)
    assert frame.min() >= 0.0 and frame.max() <= 1.0


def test_preprocess_constant_image():
    colour = (200, 100, 50)
    rgb = np.empty((30, 40, 3), dtype=np.uint8)
    rgb[:, :] = colour
    w0, w1, w2 = SETTINGS.luma_weights
    expected = (w0 * colour[0] + w1 * colour[1] + w2 * colour[2]) / SETTINGS.pixel_scale
    np.testing.assert_allclose(preprocess_frame(rgb, SETTINGS), np.full((N, N), expected), rtol=0.0, atol=1e-6)


def test_preprocess_centre_crop():
    # W = 3 * N * 2, H = 2 * N: the left third is white and the rest black; the centre square
    # starts at column (W - H) // 2 = 2 * N, which is exactly where the white third ends.
    height, width = 2 * N, 6 * N
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, : width // 3] = 255
    frame = preprocess_frame(rgb, SETTINGS)
    np.testing.assert_array_equal(frame, np.zeros((N, N), dtype=np.float32))
    # One more white column reaches into the crop's first column bin only.
    rgb[:, : width // 3 + 1] = 255
    frame = preprocess_frame(rgb, SETTINGS)
    assert np.all(frame[:, 0] > 0.0) and np.all(frame[:, 1:] == 0.0)
    np.testing.assert_allclose(frame, brute_force(rgb, SETTINGS), rtol=0.0, atol=1e-6)


def test_preprocess_bad_inputs():
    with pytest.raises(TypeError):
        preprocess_frame([[[0, 0, 0]]], SETTINGS)
    with pytest.raises(TypeError):
        preprocess_frame(np.zeros((20, 20, 3), dtype=np.float32), SETTINGS)
    with pytest.raises(ValueError):
        preprocess_frame(np.zeros((20, 20), dtype=np.uint8), SETTINGS)
    with pytest.raises(ValueError):
        preprocess_frame(np.zeros((20, 20, 4), dtype=np.uint8), SETTINGS)
    with pytest.raises(ValueError):
        preprocess_frame(np.zeros((N - 1, 100, 3), dtype=np.uint8), SETTINGS)


def test_formula_image():
    image = formula_image(5, 3)
    assert image.shape == (3, 5, 3) and image.dtype == np.uint8
    x, y = 4, 2
    assert tuple(image[y, x]) == ((7 * x + 13 * y) % 256, (3 * x + 5 * y + 40) % 256, (x + 11 * y + 90) % 256)


def test_frame_control_periods():
    periods = frame_control_periods(10, 30.0, 0.02)
    assert periods.dtype == np.int64 and periods.shape == (10,)
    assert periods.tolist() == [0, 2, 3, 5, 7, 8, 10, 12, 13, 15]
    assert frame_control_periods(5, 25.0, 0.02).tolist() == [0, 2, 4, 6, 8]
    with pytest.raises(ValueError):
        frame_control_periods(8, 60.0, 0.02)
    with pytest.raises(ValueError):
        frame_control_periods(0, 30.0, 0.02)
    with pytest.raises(ValueError):
        frame_control_periods(8, -30.0, 0.02)
    with pytest.raises(ValueError):
        frame_control_periods(8, 30.0, 0.0)


def _frame(value: float) -> np.ndarray:
    return np.full((N, N), value, dtype=np.float64)


def test_frame_window():
    window = FrameWindow(SETTINGS)
    with pytest.raises(RuntimeError):
        window.array()
    for k in range(SETTINGS.frames + 2):
        window.push(_frame(k))
        assert len(window) == min(k + 1, SETTINGS.frames)
        assert window.full == (k + 1 >= SETTINGS.frames)
    array = window.array()
    assert array.shape == (SETTINGS.frames, N, N) and array.dtype == np.float32
    assert [float(f[0, 0]) for f in array] == [float(k) for k in range(2, SETTINGS.frames + 2)]
    array[0] = -1.0
    assert float(window.array()[0, 0, 0]) == 2.0
    with pytest.raises(ValueError):
        window.push(np.zeros((N, N + 1)))
    window.clear()
    assert len(window) == 0 and not window.full


def test_flatten_unflatten():
    window = np.arange(SETTINGS.n_features, dtype=np.float32).reshape(SETTINGS.frames, N, N)
    vector = flatten_window(window)
    assert vector.shape == (SETTINGS.n_features,) and vector.dtype == np.float32
    assert vector[0] == window[0, 0, 0] and vector[N * N] == window[1, 0, 0] and vector[1] == window[0, 0, 1]
    np.testing.assert_array_equal(unflatten_window(vector, SETTINGS), window)
    with pytest.raises(ValueError):
        unflatten_window(vector[:-1], SETTINGS)


def test_take_frames():
    window = np.stack([_frame(k) for k in range(SETTINGS.frames)]).astype(np.float32)
    newest = take_frames(window, 3)
    assert newest.shape == (3, N, N)
    assert [float(f[0, 0]) for f in newest] == [float(k) for k in range(SETTINGS.frames - 3, SETTINGS.frames)]
    newest[:] = -1.0
    assert float(window[-1, 0, 0]) == SETTINGS.frames - 1
    with pytest.raises(ValueError):
        take_frames(window, 0)
    with pytest.raises(ValueError):
        take_frames(window, SETTINGS.frames + 1)


def _window(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(SETTINGS.frames, N, N)).astype(np.float32)


def test_augment_window():
    augmentation = require(CONFIG, "vision.augmentation")
    window = _window(1)
    before = window.copy()
    first = augment_window(window, np.random.default_rng(7), augmentation)
    second = augment_window(window, np.random.default_rng(7), augmentation)
    other = augment_window(window, np.random.default_rng(8), augmentation)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.shape == window.shape and first.dtype == np.float32
    assert first.min() >= 0.0 and first.max() <= 1.0
    np.testing.assert_array_equal(window, before)


def test_augment_window_identity():
    identity = {"exposure_gain": [1.0, 1.0], "exposure_bias": [0.0, 0.0], "noise_std": [0.0, 0.0], "blur_mix": [0.0, 0.0]}
    window = _window(2)
    np.testing.assert_array_equal(augment_window(window, np.random.default_rng(3), identity), window)


def test_write_test_vectors(tmp_path):
    path = tmp_path / "nested" / "vectors.json"
    returned = write_test_vectors(path, SETTINGS)
    vectors = json.loads(path.read_text(encoding="utf-8"))
    assert vectors == returned
    assert len(vectors["cases"]) == 8
    assert vectors["contract"]["mirrored"] is False
    tolerance = vectors["tolerance"]
    for case in vectors["cases"]:
        width, height = case["width"], case["height"]
        if case["generator"] == "formula":
            rgb = formula_image(width, height)
        else:
            assert case["generator"] == "pixels"
            raw = base64.b64decode(case["rgb_base64"])
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        expected = np.array(case["expected"], dtype=np.float64)
        assert expected.shape == (N * N,)
        frame = preprocess_frame(rgb, SETTINGS)
        np.testing.assert_allclose(frame.reshape(-1), expected, rtol=0.0, atol=tolerance)
        # The JSON floats are the float32 values themselves.
        np.testing.assert_array_equal(expected.astype(np.float32).astype(np.float64), expected)
