// Parity with Dev B's Python preprocessing (learned_judgment.vision.preprocess_frame): one test per case of the vector
// file that config.yaml names under vision.test_vectors. If the file is missing the test is skipped and says NOT RUN.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { preprocess } from "../../app/static/js/vision.js";

const ROOT = join(import.meta.dirname, "..", "..");
const named = readFileSync(join(ROOT, "config.yaml"), "utf8").match(/^\s+test_vectors:\s*(\S+)/m)?.[1];
const FILE = named ? join(ROOT, named) : null;

// The contract's formula image, as `contract.formula` writes it out (and vision.formula_image computes it).
function formulaRgb(w, h) {
  const rgb = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    rgb.set([(7 * x + 13 * y) % 256, (3 * x + 5 * y + 40) % 256, (x + 11 * y + 90) % 256], (y * w + x) * 3);
  }
  return rgb;
}

function rgba(rgb) {
  const out = new Uint8ClampedArray((rgb.length / 3) * 4);
  for (let k = 0; k < rgb.length / 3; k++) out.set([rgb[3 * k], rgb[3 * k + 1], rgb[3 * k + 2], 255], 4 * k);
  return out;
}

if (!FILE || !existsSync(FILE)) {
  test("parity with Dev B's preprocessing vectors", { skip: `NOT RUN: no vector file (config.yaml vision.test_vectors = ${named ?? "absent"})` }, () => {});
} else {
  const { contract, tolerance, cases } = JSON.parse(readFileSync(FILE, "utf8"));
  const settings = { height: contract.frame_size, width: contract.frame_size, luma: contract.luma_weights, divisor: contract.pixel_scale };

  test("the vector file's contract is the one this page implements", () => {
    assert.equal(contract.mirrored, false);
    assert.match(contract.crop, /top = \(H - S\) \/\/ 2, left = \(W - S\) \/\/ 2/);
    assert.match(contract.formula, /^r = \(7\*x \+ 13\*y\) % 256, g = \(3\*x \+ 5\*y \+ 40\) % 256, b = \(x \+ 11\*y \+ 90\) % 256$/);
    assert.equal(typeof tolerance, "number");
  });

  for (const c of cases) {
    test(`parity with Dev B's vector ${c.name} (${c.width} x ${c.height}, ${c.generator})`, () => {
      let rgb;
      if (c.generator === "formula") rgb = formulaRgb(c.width, c.height);
      else if (c.generator === "pixels") rgb = new Uint8Array(Buffer.from(c.rgb_base64, "base64"));
      else assert.fail(`unknown generator ${c.generator}`);
      assert.equal(rgb.length, c.width * c.height * 3, "input size");
      const out = preprocess(rgba(rgb), c.width, c.height, settings);
      const expected = c.expected.flat();
      assert.equal(out.length, expected.length);
      let worst = 0;
      for (let k = 0; k < expected.length; k++) worst = Math.max(worst, Math.abs(out[k] - expected[k]));
      assert.ok(worst <= tolerance, `largest difference ${worst} exceeds ${tolerance}`);
    });
  }
}
