import { test } from "node:test";
import assert from "node:assert/strict";
import { preprocess, readWindowSettings } from "../../app/static/js/vision.js";

// The agreed contract values (plan, "image_window"); pure-function input only.
const RAW = { frames: 8, height: 12, width: 12, hz: 30, order: "oldest_first", normalization: 255, crop: "centre_square", luma: [0.299, 0.587, 0.114], mirrored: false };
const S12 = { frames: 8, height: 12, width: 12, hz: 30, luma: [0.299, 0.587, 0.114], divisor: 255 };

// An RGBA image of w x h whose pixel (x, y) is rgb(x, y).
function image(w, h, rgb) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [r, g, b] = rgb(x, y);
    data.set([r, g, b, 255], (y * w + x) * 4);
  }
  return data;
}
const grey = (v) => [v, v, v];
const close = (actual, expected, label) => {
  assert.equal(actual.length, expected.length, `${label}: length`);
  for (let k = 0; k < expected.length; k++) assert.ok(Math.abs(actual[k] - expected[k]) < 1e-6, `${label}[${k}]: ${actual[k]} vs ${expected[k]}`);
};
const cells = (fn) => Array.from({ length: 144 }, (_, k) => fn(Math.floor(k / 12), k % 12));

test("readWindowSettings takes every number from image_window", () => {
  assert.deepEqual(readWindowSettings(RAW), { ok: true, settings: S12 });
  assert.equal(readWindowSettings({ ...RAW, crop: "center_square" }).ok, true);
});

test("a missing image_window is unavailable, with no defaults guessed", () => {
  for (const raw of [undefined, null]) {
    const r = readWindowSettings(raw);
    assert.equal(r.ok, false);
    assert.match(r.problems[0], /does not publish/);
  }
});

test("each unreadable field is named", () => {
  const r = readWindowSettings({ ...RAW, frames: 0, height: 1.5, hz: "30", luma: [1, 2], normalization: "255", order: "newest_first", crop: "stretch", mirrored: true });
  assert.equal(r.ok, false);
  for (const field of ["frames", "height", "hz", "luma", "normalization", "order", "crop", "mirrored"]) {
    assert.ok(r.problems.some((p) => p.startsWith(field)), `no problem names ${field}: ${r.problems}`);
  }
  assert.ok(!r.problems.some((p) => p.startsWith("width")));
});

test("a uniform frame gives the same grey everywhere", () => {
  const out = preprocess(image(12, 12, () => [100, 150, 200]), 12, 12, S12);
  assert.ok(out instanceof Float32Array);
  close(out, new Array(144).fill((0.299 * 100 + 0.587 * 150 + 0.114 * 200) / 255), "uniform");
});

test("a horizontal ramp averages whole-pixel bins of two columns", () => {
  const out = preprocess(image(24, 24, (x) => grey(x * 10)), 24, 24, S12);
  close(out, cells((i, j) => (20 * j + 5) / 255), "horizontal");
});

test("output is row-major, top-left first (vertical ramp)", () => {
  const out = preprocess(image(24, 24, (x, y) => grey(y * 10)), 24, 24, S12);
  close(out, cells((i) => (20 * i + 5) / 255), "vertical");
});

test("a wide frame is centre-cropped to its height", () => {
  // 36 x 24: S = 24, crop columns 6..29; everything outside is white, inside black.
  const out = preprocess(image(36, 24, (x) => grey(x >= 6 && x < 30 ? 0 : 255)), 36, 24, S12);
  close(out, new Array(144).fill(0), "wide");
});

test("a tall frame is centre-cropped to its width", () => {
  const out = preprocess(image(24, 30, (x, y) => grey(y >= 3 && y < 27 ? 0 : 255)), 24, 30, S12);
  close(out, new Array(144).fill(0), "tall");
});

test("a side that is not a multiple of the cell count uses floor bin edges", () => {
  // S = 13: bins are single pixels 0..10, and the last bin holds pixels 11 and 12.
  const out = preprocess(image(13, 13, (x) => grey(x * 10)), 13, 13, S12);
  close(out, cells((i, j) => (j < 11 ? 10 * j : 115) / 255), "13px");
});

test("the luma weights and the divisor come from the settings", () => {
  const out = preprocess(image(12, 12, () => [200, 0, 0]), 12, 12, { ...S12, luma: [1, 0, 0], divisor: 100 });
  close(out, new Array(144).fill(2), "weights");
});

test("a frame smaller than the cell count is refused with a clear error", () => {
  assert.throws(() => preprocess(image(11, 20, () => grey(0)), 11, 20, S12), /11 x 20\) is smaller than the 12 x 12 window/);
});

test("pixel data of the wrong length is refused", () => {
  assert.throws(() => preprocess(new Uint8ClampedArray(10), 12, 12, S12), /pixel data/);
});
