// Parity with Dev B's Python preprocessing: every tests/front/vectors/*.json holds an input image and the expected
// 144 numbers. With no vector files the test is skipped and says NOT RUN.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { preprocess } from "../../app/static/js/vision.js";

const DIR = join(import.meta.dirname, "vectors");
const names = existsSync(DIR) ? readdirSync(DIR).filter((n) => n.endsWith(".json")).sort() : [];
const TOLERANCE = 1e-6; // float32 output; Python may sum in float64

// The contract's values (plan, "Preprocessing"); a vector file may override them.
const CONTRACT = { height: 12, width: 12, luma: [0.299, 0.587, 0.114], divisor: 255 };

// Assumed vector format until Dev B's file lands: {image: [H][W][3] uint8, expected: [144] or [12][12]}, optionally
// {width, height, rgb: flat [H*W*3]} in place of image. Anything else fails with the keys it found.
function load(name) {
  const v = JSON.parse(readFileSync(join(DIR, name), "utf8"));
  let w, h, rgb;
  if (Array.isArray(v.image)) {
    h = v.image.length; w = v.image[0].length; rgb = v.image.flat(2);
  } else if (Array.isArray(v.rgb) && Number.isInteger(v.width) && Number.isInteger(v.height)) {
    w = v.width; h = v.height; rgb = v.rgb;
  } else {
    throw new Error(`${name}: unrecognised vector format; keys: ${Object.keys(v).join(", ")}`);
  }
  if (!Array.isArray(v.expected)) throw new Error(`${name}: no "expected" list; keys: ${Object.keys(v).join(", ")}`);
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let k = 0; k < w * h; k++) rgba.set([rgb[3 * k], rgb[3 * k + 1], rgb[3 * k + 2], 255], 4 * k);
  return { w, h, rgba, expected: v.expected.flat() };
}

if (names.length === 0) {
  test("parity with Dev B's preprocessing vectors", { skip: "NOT RUN: tests/front/vectors holds no *.json vector files yet" }, () => {});
}
for (const name of names) {
  test(`parity with Dev B's vector ${name}`, () => {
    const { w, h, rgba, expected } = load(name);
    const out = preprocess(rgba, w, h, CONTRACT);
    assert.equal(out.length, expected.length);
    for (let k = 0; k < expected.length; k++) {
      assert.ok(Math.abs(out[k] - expected[k]) <= TOLERANCE, `${name}[${k}]: browser ${out[k]} vs Python ${expected[k]}`);
    }
  });
}
