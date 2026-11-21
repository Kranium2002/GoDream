// Camera frames and features stay in memory and go nowhere but the same-origin /api/infer.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const STATIC = join(import.meta.dirname, "..", "..", "app", "static");
function files(dir) {
  return readdirSync(dir).flatMap((n) => (statSync(join(dir, n)).isDirectory() ? files(join(dir, n)) : [join(dir, n)]));
}
const sources = files(STATIC).filter((p) => /\.(js|html|css)$/.test(p)).map((p) => ({ p, text: readFileSync(p, "utf8") }));

test("the page has sources to check", () => {
  assert.ok(sources.some(({ p }) => p.endsWith("index.html")));
  assert.ok(sources.some(({ p }) => p.endsWith("live.js")));
});

test("no storage, export or recording API appears anywhere in app/static", () => {
  const forbidden = [/localStorage/, /sessionStorage/, /indexedDB/i, /document\.cookie/, /toDataURL/, /toBlob/, /\bdownload\s*=/,
    /setAttribute\(\s*["']download/, /MediaRecorder/, /createObjectURL/, /sendBeacon/, /WebSocket/, /XMLHttpRequest/, /caches\./, /serviceWorker/];
  for (const { p, text } of sources) for (const re of forbidden) assert.ok(!re.test(text), `${p} uses ${re}`);
});

test("no absolute or protocol-relative URL appears in app/static", () => {
  const SVG_NAMESPACE = "http://www.w3.org/2000/svg"; // an XML namespace name for createElementNS, never requested
  for (const { p, text: raw } of sources) {
    const text = raw.split(SVG_NAMESPACE).join("");
    assert.ok(!/[a-z]+:\/\//i.test(text), `${p} holds an absolute URL`);
    assert.ok(!/["'`(]\/\//.test(text), `${p} holds a protocol-relative URL`);
  }
});

test("fetch is called in one place, and only api.js names /api/infer", () => {
  const fetchUses = sources.flatMap(({ p, text }) => (text.match(/\bfetch\(/g) ?? []).map(() => p));
  assert.equal(fetchUses.length, 1, `fetch( found in: ${fetchUses}`);
  const inferUses = sources.filter(({ text }) => text.includes("/api/infer")).map(({ p }) => p);
  assert.deepEqual(inferUses.map((p) => p.split(/[\\/]/).pop()), ["api.js"]);
});
