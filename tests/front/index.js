// `node --test tests/front` hands Node this folder (Node 24 does not search a folder argument), so Node loads
// index.js, which loads every *.test.mjs beside it.
import { readdirSync } from "node:fs";

for (const name of readdirSync(import.meta.dirname).filter((n) => n.endsWith(".test.mjs")).sort()) {
  await import(`./${name}`);
}
