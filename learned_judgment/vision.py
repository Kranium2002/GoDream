"""The image-window contract shared by the MuJoCo head camera and the browser camera (numpy).

Lane contract (Person B, vision)
--------------------------------
One reference implementation of everything between a camera image and the student's input,
so that simulation and the browser feed the student the same numbers:

    preprocess_frame       8-bit RGB image (H, W, 3) -> one grey frame (N, N) float32 in [0, 1]:
                           centre square crop (no mirroring), luma, exact box-mean resize to
                           N x N with integer bin edges, divided by pixel_scale
    FrameWindow            rolling window of the last F frames, oldest first
    flatten_window         (F, N, N) -> (F * N * N,): frame-major, oldest first, each row-major
    frame_control_periods  simulation frame schedule: the control period nearest each
                           k / frame_rate_hz (config.yaml vision.sim_timing)
    augment_window         image-space variation of simulation windows only, never live frames
    write_test_vectors     input images and expected frames the JavaScript side is checked against

Every tunable comes from the config.yaml ``vision`` block; N = vision.frame_size,
F = vision.frames (or one of vision.frame_variants). Frames are float32 grey levels,
row-major, top-left first.
"""

from __future__ import annotations

import base64
import json
import math
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from learned_judgment.config import CONFIG_PATH, load_config, require

__all__ = [
    "FrameWindow",
    "VisionSettings",
    "augment_window",
    "flatten_window",
    "formula_image",
    "frame_control_periods",
    "preprocess_frame",
    "take_frames",
    "unflatten_window",
    "write_test_vectors",
]

# Tolerance of the browser's parity check against the test vectors (part of the contract).
_TOLERANCE = 1e-6
# Seed of the random "pixels" test images and their sizes as (width, height).
_PIXELS_SEED = 20261121
_PIXELS_SIZES = ((24, 24), (37, 29))
_FORMULA_SIZES = ((96, 96), (640, 480), (1280, 720), (480, 640), (100, 75), (53, 37))
_AUGMENTATION_KEYS = ("exposure_gain", "exposure_bias", "noise_std", "blur_mix")


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return int(value)


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; got {number}")
    return number


def _positive(name: str, value: Any) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0; got {number}")
    return number


def _window(window: Any) -> np.ndarray:
    if not isinstance(window, np.ndarray):
        raise TypeError(f"window must be a numpy array; got {type(window).__name__}")
    if window.ndim != 3 or window.shape[0] < 1 or window.shape[1] < 1 or window.shape[1] != window.shape[2]:
        raise ValueError(f"window must have shape (F >= 1, N >= 1, N); got {window.shape}")
    return window


@dataclass(frozen=True)
class VisionSettings:
    """The image-window contract (config.yaml ``vision``).

    frame_size     N: a frame is N x N grey pixels, N >= 1
    frame_rate_hz  camera cadence in Hz, > 0
    frames         F: frames per window, >= 1
    luma_weights   (w0, w1, w2): grey = w0 * R + w1 * G + w2 * B, three finite numbers
    pixel_scale    grey levels are divided by this, > 0

    Raises TypeError / ValueError on construction when a field breaks these rules.
    """

    frame_size: int
    frame_rate_hz: float
    frames: int
    luma_weights: tuple[float, float, float]
    pixel_scale: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_size", _integer("frame_size", self.frame_size, 1))
        object.__setattr__(self, "frame_rate_hz", _positive("frame_rate_hz", self.frame_rate_hz))
        object.__setattr__(self, "frames", _integer("frames", self.frames, 1))
        weights = self.luma_weights
        if isinstance(weights, (str, bytes)) or not isinstance(weights, (tuple, list)) or len(weights) != 3:
            raise ValueError(f"luma_weights must be three numbers; got {weights!r}")
        object.__setattr__(self, "luma_weights", tuple(_finite("luma weight", w) for w in weights))
        object.__setattr__(self, "pixel_scale", _positive("pixel_scale", self.pixel_scale))

    @classmethod
    def from_config(cls, config: Mapping[str, Any], frames: int | None = None) -> VisionSettings:
        """Settings from config.yaml ``vision``.

        Args:
            config: the loaded config.yaml mapping.
            frames: window length overriding vision.frames; must equal vision.frames or be a
                    member of vision.frame_variants.

        Raises:
            ConfigError for a missing or null key; ValueError for a ``frames`` that is neither
            vision.frames nor a frame variant; TypeError / ValueError for an invalid value.
        """
        configured = require(config, "vision.frames")
        if frames is not None:
            variants = list(require(config, "vision.frame_variants"))
            if isinstance(frames, bool) or (frames != configured and frames not in variants):
                raise ValueError(
                    f"frames must be vision.frames ({configured}) or one of vision.frame_variants {variants}; got {frames!r}"
                )
            configured = frames
        return cls(
            frame_size=require(config, "vision.frame_size"),
            frame_rate_hz=require(config, "vision.frame_rate_hz"),
            frames=configured,
            luma_weights=require(config, "vision.luma_weights"),
            pixel_scale=require(config, "vision.pixel_scale"),
        )

    @property
    def n_features(self) -> int:
        """Length of a flattened window: frames * frame_size * frame_size."""
        return self.frames * self.frame_size * self.frame_size


def preprocess_frame(rgb: np.ndarray, settings: VisionSettings) -> np.ndarray:
    """One grey frame from an 8-bit RGB image.

    1. S = min(H, W); crop rows top .. top + S - 1 and columns left .. left + S - 1 with
       top = (H - S) // 2, left = (W - S) // 2. No mirroring.
    2. grey = w0 * R + w1 * G + w2 * B in float64 (settings.luma_weights).
    3. Cell (i, j) of the N x N output is the mean of grey over crop rows
       (i * S) // N .. ((i + 1) * S) // N - 1 and crop columns (j * S) // N .. ((j + 1) * S) // N - 1.
    4. Divided by settings.pixel_scale and cast to float32.

    Args:
        rgb:      (H, W, 3) uint8, row-major, top-left first, with min(H, W) >= N.
        settings: the contract; N = settings.frame_size.

    Returns:
        (N, N) float32, row-major, top-left first; in [0, 1] for weights summing to 1 and
        pixel_scale 255.

    Raises:
        TypeError when ``rgb`` is not a uint8 numpy array; ValueError for a wrong shape or an
        image smaller than N in either direction.
    """
    if not isinstance(rgb, np.ndarray):
        raise TypeError(f"rgb must be a numpy array; got {type(rgb).__name__}")
    if rgb.dtype != np.uint8:
        raise TypeError(f"rgb must have dtype uint8; got {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3); got {rgb.shape}")
    n = settings.frame_size
    height, width = rgb.shape[0], rgb.shape[1]
    side = min(height, width)
    if side < n:
        raise ValueError(f"rgb is {width} x {height} (W x H); both sides must be >= frame_size {n}")
    top, left = (height - side) // 2, (width - side) // 2
    crop = rgb[top : top + side, left : left + side].astype(np.float64)
    w0, w1, w2 = settings.luma_weights
    grey = w0 * crop[:, :, 0] + w1 * crop[:, :, 1] + w2 * crop[:, :, 2]
    edges = (np.arange(n + 1, dtype=np.int64) * side) // n
    starts, sizes = edges[:-1], np.diff(edges)
    sums = np.add.reduceat(np.add.reduceat(grey, starts, axis=0), starts, axis=1)
    means = sums / (sizes[:, None] * sizes[None, :])
    return (means / settings.pixel_scale).astype(np.float32)


def formula_image(width: int, height: int) -> np.ndarray:
    """(height, width, 3) uint8 test image: for column x and row y,
    r = (7x + 13y) % 256, g = (3x + 5y + 40) % 256, b = (x + 11y + 90) % 256.

    Raises TypeError / ValueError unless width and height are integers >= 1.
    """
    width = _integer("width", width, 1)
    height = _integer("height", height, 1)
    x = np.arange(width, dtype=np.int64)[None, :]
    y = np.arange(height, dtype=np.int64)[:, None]
    r = (7 * x + 13 * y) % 256
    g = (3 * x + 5 * y + 40) % 256
    b = (x + 11 * y + 90) % 256
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def frame_control_periods(n_frames: int, frame_rate_hz: float, control_period_s: float) -> np.ndarray:
    """Simulation frame schedule: (n_frames,) int64 control-period indices.

    Frame k (k = 0 .. n_frames - 1) is taken at the control period nearest to k / frame_rate_hz,
    the integer nearest to k / (frame_rate_hz * control_period_s) with halves rounding up,
    computed exactly from the decimal strings of the two numbers. 30 Hz and 0.02 s give
    [0, 2, 3, 5, 7, 8, 10, 12, 13, 15, ...]; 25 Hz gives [0, 2, 4, ...].

    Raises:
        TypeError for a non-integer n_frames or a non-number rate or period; ValueError for a
        non-positive or non-finite argument, or when two frames land on the same period (frame
        rate above the control rate).
    """
    n_frames = _integer("n_frames", n_frames, 1)
    rate = Fraction(str(_positive("frame_rate_hz", frame_rate_hz)))
    period = Fraction(str(_positive("control_period_s", control_period_s)))
    step = rate * period
    half = Fraction(1, 2)
    indices = np.array([math.floor(Fraction(k) / step + half) for k in range(n_frames)], dtype=np.int64)
    if np.any(np.diff(indices) == 0):
        raise ValueError(
            f"frame rate {frame_rate_hz} Hz is above the control rate {1.0 / float(control_period_s):g} Hz: "
            f"two of {n_frames} frames land on the same control period"
        )
    return indices


class FrameWindow:
    """Rolling window of the last ``settings.frames`` frames, oldest first."""

    def __init__(self, settings: VisionSettings) -> None:
        if not isinstance(settings, VisionSettings):
            raise TypeError(f"settings must be VisionSettings; got {type(settings).__name__}")
        self._settings = settings
        self._frames: deque[np.ndarray] = deque(maxlen=settings.frames)

    def push(self, frame: np.ndarray) -> None:
        """Append a (frame_size, frame_size) frame, stored as a float32 copy; when the window is
        full the oldest frame drops out. Raises TypeError / ValueError for a wrong input."""
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy array; got {type(frame).__name__}")
        n = self._settings.frame_size
        if frame.shape != (n, n):
            raise ValueError(f"frame must have shape ({n}, {n}); got {frame.shape}")
        self._frames.append(np.array(frame, dtype=np.float32, copy=True))

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def full(self) -> bool:
        """True once the window holds settings.frames frames."""
        return len(self._frames) == self._settings.frames

    def array(self) -> np.ndarray:
        """(frames, frame_size, frame_size) float32 copy, oldest first. RuntimeError if not full."""
        if not self.full:
            raise RuntimeError(f"window holds {len(self._frames)} of {self._settings.frames} frames; it is not full")
        return np.stack(list(self._frames)).astype(np.float32, copy=False)

    def clear(self) -> None:
        """Drop every frame."""
        self._frames.clear()


def flatten_window(window: np.ndarray) -> np.ndarray:
    """(F, N, N) window -> (F * N * N,) float32 copy: frame-major (oldest frame first), each frame
    row-major. Element 0 is the oldest frame's top-left pixel, element N * N the second frame's.
    Raises TypeError / ValueError for a wrong input."""
    return np.array(_window(window), dtype=np.float32, copy=True).reshape(-1)


def unflatten_window(vector: np.ndarray, settings: VisionSettings) -> np.ndarray:
    """(frames * N * N,) vector -> (frames, N, N) float32 copy, the inverse of flatten_window.
    Raises TypeError for a non-array; ValueError unless the vector is 1-D of length
    settings.n_features."""
    if not isinstance(vector, np.ndarray):
        raise TypeError(f"vector must be a numpy array; got {type(vector).__name__}")
    if vector.ndim != 1 or vector.shape[0] != settings.n_features:
        raise ValueError(f"vector must have shape ({settings.n_features},); got {vector.shape}")
    n = settings.frame_size
    return np.array(vector, dtype=np.float32, copy=True).reshape(settings.frames, n, n)


def take_frames(window: np.ndarray, frames: int) -> np.ndarray:
    """The most recent ``frames`` frames of an (F, N, N) window, oldest first, as a copy.
    Raises ValueError unless 1 <= frames <= F."""
    array = _window(window)
    count = _integer("frames", frames, 1)
    if count > array.shape[0]:
        raise ValueError(f"frames must be <= the window's {array.shape[0]} frames; got {count}")
    return array[array.shape[0] - count :].copy()


def _range(augmentation: Mapping[str, Any], key: str) -> tuple[float, float]:
    if key not in augmentation:
        raise ValueError(f"augmentation has no {key!r} range")
    pair = augmentation[key]
    if isinstance(pair, (str, bytes)) or not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError(f"augmentation {key!r} must be [low, high]; got {pair!r}")
    low, high = _finite(f"{key} low", pair[0]), _finite(f"{key} high", pair[1])
    if low > high:
        raise ValueError(f"augmentation {key!r} low {low} is above high {high}")
    return low, high


def _box_mean(frame: np.ndarray) -> np.ndarray:
    padded = np.pad(frame, 1, mode="edge")
    rows, cols = frame.shape
    total = np.zeros(frame.shape, dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            total += padded[di : di + rows, dj : dj + cols]
    return total / 9.0


def augment_window(window: np.ndarray, rng: np.random.Generator, augmentation: Mapping[str, Any]) -> np.ndarray:
    """Image-space variation of one simulation window (never a live window).

    One draw per window, in this order, each rng.uniform(low, high): gain (exposure_gain),
    bias (exposure_bias), noise_std (noise_std), blur (blur_mix). Then, frame by frame in window
    order: x = (1 - blur) * x + blur * box3x3(x) (3 x 3 mean, edge replication);
    x = gain * x + bias; x = x + rng.normal(0, noise_std, size=(N, N)). Finally clipped to [0, 1].

    Args:
        window:       (F, N, N) grey frames on the 0-1 scale; not modified.
        rng:          numpy Generator; the result is deterministic for its state.
        augmentation: config.yaml vision.augmentation, the four [low, high] ranges.

    Returns:
        (F, N, N) float32 in [0, 1].

    Raises:
        TypeError / ValueError for a wrong window, rng or range (missing, not [low, high],
        non-finite, low > high, negative noise_std, blur_mix outside [0, 1]).
    """
    array = _window(window)
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator; got {type(rng).__name__}")
    if not isinstance(augmentation, Mapping):
        raise TypeError(f"augmentation must be a mapping; got {type(augmentation).__name__}")
    ranges = {key: _range(augmentation, key) for key in _AUGMENTATION_KEYS}
    if ranges["noise_std"][0] < 0.0:
        raise ValueError(f"augmentation 'noise_std' must be >= 0; got {list(ranges['noise_std'])}")
    if ranges["blur_mix"][0] < 0.0 or ranges["blur_mix"][1] > 1.0:
        raise ValueError(f"augmentation 'blur_mix' must lie in [0, 1]; got {list(ranges['blur_mix'])}")
    gain = rng.uniform(*ranges["exposure_gain"])
    bias = rng.uniform(*ranges["exposure_bias"])
    noise_std = rng.uniform(*ranges["noise_std"])
    blur = rng.uniform(*ranges["blur_mix"])
    out = np.empty(array.shape, dtype=np.float32)
    for index in range(array.shape[0]):
        x = array[index].astype(np.float64)
        x = (1.0 - blur) * x + blur * _box_mean(x)
        x = gain * x + bias
        x = x + rng.normal(0.0, noise_std, size=x.shape)
        out[index] = np.clip(x, 0.0, 1.0)
    return out


def _expected(frame: np.ndarray) -> list[float]:
    # float(np.float32) is exact, and json writes the shortest repr that reads back to it.
    return [float(value) for value in frame.reshape(-1)]


def write_test_vectors(path: str | Path, settings: VisionSettings) -> dict[str, Any]:
    """Write the browser's parity test vectors as JSON to ``path`` (parent directories created)
    and return them.

    {"contract": {...}, "tolerance": 1e-6, "cases": [...]}: six "formula" cases (the image is
    rebuilt from formula_image(width, height)) and two "pixels" cases (random images from
    np.random.default_rng(20261121), raw row-major RGB bytes in "rgb_base64"), each with
    "expected", the frame_size * frame_size float32 outputs of preprocess_frame, row-major.
    """
    cases: list[dict[str, Any]] = []
    for width, height in _FORMULA_SIZES:
        frame = preprocess_frame(formula_image(width, height), settings)
        cases.append(
            {
                "name": f"formula_{width}x{height}",
                "width": width,
                "height": height,
                "generator": "formula",
                "expected": _expected(frame),
            }
        )
    rng = np.random.default_rng(_PIXELS_SEED)
    for width, height in _PIXELS_SIZES:
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        cases.append(
            {
                "name": f"pixels_{width}x{height}",
                "width": width,
                "height": height,
                "generator": "pixels",
                "rgb_base64": base64.b64encode(image.tobytes()).decode("ascii"),
                "expected": _expected(preprocess_frame(image, settings)),
            }
        )
    vectors = {
        "contract": {
            "frame_size": settings.frame_size,
            "frames": settings.frames,
            "frame_rate_hz": settings.frame_rate_hz,
            "luma_weights": list(settings.luma_weights),
            "pixel_scale": settings.pixel_scale,
            "mirrored": False,
            "crop": "centre square of side S = min(W, H): top = (H - S) // 2, left = (W - S) // 2",
            "resize": (
                "cell (i, j) = mean over crop rows (i*S)//N .. ((i+1)*S)//N - 1 "
                "and columns (j*S)//N .. ((j+1)*S)//N - 1"
            ),
            "order": "row-major, top-left first; window oldest frame first",
            "formula": "r = (7*x + 13*y) % 256, g = (3*x + 5*y + 40) % 256, b = (x + 11*y + 90) % 256",
        },
        "tolerance": _TOLERANCE,
        "cases": cases,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(vectors, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return vectors


if __name__ == "__main__":
    _config = load_config()
    _settings = VisionSettings.from_config(_config)
    _path = Path(require(_config, "vision.test_vectors"))
    if not _path.is_absolute():
        _path = CONFIG_PATH.parent / _path
    _vectors = write_test_vectors(_path, _settings)
    print(_path)
    print(len(_vectors["cases"]))
