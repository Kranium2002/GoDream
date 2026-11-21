// A declaration {skill, params, source_text} as display rows, with the params of its skill in the GET /api/skills
// catalogue [{skill, description, params: {name: {type, min, max, choices, default, unit}}, available, missing}]
// (contract agreed with Dev B, 20:55 UTC). Row: {field, label, note, value, isDefault, range}; value null is "not set".
// Every value is the server's, as text.

const SPEC_ONLY = "Not a supported parameter of this skill.";

export function findSkill(catalogue, name) {
  return Array.isArray(catalogue) ? catalogue.find((s) => s?.skill === name) ?? null : null;
}

const text = (v) => (typeof v === "string" ? v : Array.isArray(v) ? v.map(text).join(", ") : JSON.stringify(v));
const withUnit = (v, unit) => (unit ? `${text(v)} ${unit}` : text(v));
const given = (v) => v !== undefined && v !== null;

function rangeText(p) {
  if (Array.isArray(p.choices)) return `one of: ${p.choices.map(text).join(", ")}`;
  if (typeof p.min === "number" && typeof p.max === "number") return `${p.min} to ${withUnit(p.max, p.unit)}`;
  return "The catalogue gives no range.";
}

export function specRows(decl, catalogue) {
  const skill = findSkill(catalogue, decl.skill);
  const rows = [
    { field: "skill", label: "Skill", note: skill ? skill.description ?? "" : "This skill is not in the catalogue.", value: given(decl.skill) ? text(decl.skill) : null },
    { field: "source_text", label: "Your request", note: "", value: given(decl.source_text) ? text(decl.source_text) : null },
  ];
  const supported = skill?.params && typeof skill.params === "object" ? skill.params : {};
  const declared = decl.params && typeof decl.params === "object" ? decl.params : {};
  for (const [name, p = {}] of Object.entries(supported)) {
    const set = given(declared[name]);
    const shown = set ? declared[name] : p.default;
    rows.push({ field: `params.${name}`, label: name, note: "", value: given(shown) ? withUnit(shown, p.unit) : null, isDefault: !set && given(p.default), range: rangeText(p) });
  }
  for (const [name, v] of Object.entries(declared)) {
    if (!(name in supported)) rows.push({ field: `params.${name}`, label: name, note: "", value: text(v), isDefault: false, range: SPEC_ONLY });
  }
  return rows;
}

// A problem's field names a key inside the declaration ("skill", "params.incline_deg"); any other goes to general.
export function attachProblems(rows, problems) {
  const out = rows.map((r) => ({ ...r, problems: [] }));
  const general = [];
  for (const p of problems ?? []) (out.find((r) => r.field === p?.field)?.problems ?? general).push(p);
  return { rows: out, general };
}

export function catalogueRows(catalogue) {
  return (Array.isArray(catalogue) ? catalogue : []).map((s) => ({
    skill: text(s?.skill),
    description: typeof s?.description === "string" ? s.description : "",
    available: s?.available === true,
    missing: Array.isArray(s?.missing) ? s.missing.map(text) : [],
  }));
}

export function isValid(chat, catalogue) {
  const decl = chat?.declared_spec;
  return Boolean(decl) && typeof decl === "object"
    && Array.isArray(chat.problems) && chat.problems.length === 0
    && Array.isArray(chat.missing) && chat.missing.length === 0
    && findSkill(catalogue, decl.skill)?.available !== false;
}
