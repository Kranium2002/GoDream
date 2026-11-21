import { test } from "node:test";
import assert from "node:assert/strict";
import { attachProblems, catalogueRows, findSkill, isValid, specRows } from "../../app/static/js/spec.js";

// Pure-function fixtures in the agreed contract shape (Dev B, 20:55 UTC); never shown on the page.
const ramp = {
  skill: "RAMP",
  description: "Climb a ramp by choosing the approach speed.",
  params: {
    incline_deg: { type: "float", min: 0, max: 15, default: 5, unit: "deg" },
    approach: { type: "str", choices: ["slow", "fast"], default: "slow" },
    note: { type: "str" },
  },
  actions: [], outcomes: [], available: true, missing: [],
};
const dodge = {
  skill: "DODGE", description: "Dodge an incoming object.",
  params: { threats: { type: "list", choices: ["punch", "ball"] } },
  actions: [], outcomes: [], available: false, missing: ["head camera", "punch proxy"],
};
const catalogue = [ramp, dodge];
const decl = { skill: "RAMP", params: { incline_deg: 12, extra: 1 }, source_text: "cross this 12 degree slope" };

test("findSkill reads the catalogue list by its skill key", () => {
  assert.equal(findSkill(catalogue, "RAMP"), ramp);
  assert.equal(findSkill(catalogue, "KICK"), null);
  assert.equal(findSkill(null, "RAMP"), null);
  assert.equal(findSkill({ skills: [ramp] }, "RAMP"), null);
});

test("specRows lists the skill, the request, then every catalogue param and any extra one", () => {
  const rows = specRows(decl, catalogue);
  assert.deepEqual(rows.map((r) => r.field), ["skill", "source_text", "params.incline_deg", "params.approach", "params.note", "params.extra"]);
  const f = Object.fromEntries(rows.map((r) => [r.field, r]));
  assert.equal(f.skill.label, "Skill");
  assert.equal(f.skill.value, "RAMP");
  assert.equal(f.skill.note, ramp.description);
  assert.equal(f.source_text.label, "Your request");
  assert.equal(f.source_text.value, "cross this 12 degree slope");
});

test("a param row shows the declared value with its unit and the allowed range", () => {
  const f = Object.fromEntries(specRows(decl, catalogue).map((r) => [r.field, r]));
  assert.deepEqual(f["params.incline_deg"], { field: "params.incline_deg", label: "incline_deg", note: "", value: "12 deg", isDefault: false, range: "0 to 15 deg" });
});

test("an undeclared param shows the catalogue default, marked as default", () => {
  const f = Object.fromEntries(specRows(decl, catalogue).map((r) => [r.field, r]));
  assert.equal(f["params.approach"].value, "slow");
  assert.equal(f["params.approach"].isDefault, true);
  assert.equal(f["params.approach"].range, "one of: slow, fast");
  assert.equal(f["params.note"].value, null);
  assert.equal(f["params.note"].range, "The catalogue gives no range.");
  assert.equal(f["params.extra"].value, "1");
  assert.equal(f["params.extra"].range, "Not a supported parameter of this skill.");
});

test("a list value reads as a plain list", () => {
  const f = Object.fromEntries(specRows({ skill: "DODGE", params: { threats: ["punch", "ball"] } }, catalogue).map((r) => [r.field, r]));
  assert.equal(f["params.threats"].value, "punch, ball");
  assert.equal(f["params.threats"].range, "one of: punch, ball");
});

test("a skill missing from the catalogue still shows the declaration's params", () => {
  const f = Object.fromEntries(specRows({ skill: "KICK", params: { speed: 2 } }, catalogue).map((r) => [r.field, r]));
  assert.equal(f.skill.note, "This skill is not in the catalogue.");
  assert.equal(f["params.speed"].value, "2");
  assert.equal(f.source_text.value, null);
});

test("attachProblems puts each problem on the row its field names; unknown fields go to general", () => {
  const rows = specRows(decl, catalogue);
  const problems = [
    { code: "out_of_range", field: "params.incline_deg", message: "Above 15." },
    { code: "x", field: "skill", message: "Skill msg." },
    { code: "y", field: "nowhere", message: "Lost field." },
    { code: "z", field: null, message: "No field." },
  ];
  const { rows: out, general } = attachProblems(rows, problems);
  const f = Object.fromEntries(out.map((r) => [r.field, r]));
  assert.deepEqual(f["params.incline_deg"].problems, [problems[0]]);
  assert.deepEqual(f.skill.problems, [problems[1]]);
  assert.deepEqual(f.source_text.problems, []);
  assert.deepEqual(general, [problems[2], problems[3]]);
});

test("catalogueRows says which skills are supported and what the others miss", () => {
  assert.deepEqual(catalogueRows(catalogue), [
    { skill: "RAMP", description: ramp.description, available: true, missing: [] },
    { skill: "DODGE", description: dodge.description, available: false, missing: ["head camera", "punch proxy"] },
  ]);
  assert.deepEqual(catalogueRows([{ skill: "X", available: "yes" }]), [{ skill: "X", description: "", available: false, missing: [] }]);
  assert.deepEqual(catalogueRows(null), []);
});

test("isValid needs a declaration, no problems, nothing missing and a skill the catalogue does not mark unavailable", () => {
  const good = { declared_spec: decl, problems: [], missing: [] };
  assert.equal(isValid(good, catalogue), true);
  assert.equal(isValid(good, null), true); // no catalogue: the server's own check decides
  assert.equal(isValid({ ...good, declared_spec: { skill: "DODGE", params: {} } }, catalogue), false);
  assert.equal(isValid({ ...good, declared_spec: null }, catalogue), false);
  assert.equal(isValid({ ...good, problems: [{ code: "c", field: "f", message: "m" }] }, catalogue), false);
  assert.equal(isValid({ ...good, missing: ["punch proxy"] }, catalogue), false);
  assert.equal(isValid({ declared_spec: decl }, catalogue), false);
  assert.equal(isValid(null, catalogue), false);
});
