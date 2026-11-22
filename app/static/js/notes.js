// One status area per section: real errors, "unavailable" notices and the server's notes, newest first. A standing
// condition is kept under a key and replaced when it changes; an event (report) that repeats an identical message
// (same severity and text) merges into it with a count. At most `visible` show until the person asks for all.
// Severities: "danger" (an error), "warn" (unavailable, not calibrated), "info" (a note from the server).

export function createNotes({ visible = 3 } = {}) {
  let items = []; // newest first: {key, severity, kind, text, count}
  return {
    set(key, severity, text, kind = null) {
      const i = items.findIndex((n) => n.key === key);
      if (i >= 0 && items[i].severity === severity && items[i].text === text && items[i].kind === kind) return false;
      if (i >= 0) items.splice(i, 1);
      items.unshift({ key, severity, kind, text, count: 1 });
      return true;
    },
    clear(key) {
      const before = items.length;
      items = items.filter((n) => n.key !== key);
      return items.length !== before;
    },
    report(severity, text, kind = null) {
      const key = `event:${severity}:${kind ?? ""}:${text}`;
      const i = items.findIndex((n) => n.key === key);
      const count = i >= 0 ? items[i].count + 1 : 1;
      if (i >= 0) items.splice(i, 1);
      items.unshift({ key, severity, kind, text, count });
      return true;
    },
    view(showAll = false) {
      return { items: showAll ? [...items] : items.slice(0, visible), hidden: showAll ? 0 : Math.max(0, items.length - visible), total: items.length };
    },
  };
}

const KIND = { danger: "Error", warn: "Warning", info: "Note" };

// The DOM side: renders a createNotes() list into `root` (a div.notes), with a "Show all" button when more wait.
export function mountNotes(root, el, options) {
  const notes = createNotes(options);
  let showAll = false;
  function render() {
    const v = notes.view(showAll);
    root.hidden = v.total === 0;
    const list = el("ul", { class: "notes__list" }, ...v.items.map((n) => el("li", { class: `msg msg--${n.severity}` },
      el("span", { class: "msg__kind" }, n.kind ?? KIND[n.severity] ?? n.severity),
      el("span", { class: "msg__text" }, n.text),
      n.count > 1 ? el("span", { class: "msg__count" }, `${n.count} times`) : el("span", {}))));
    const more = v.total > (options?.visible ?? 3)
      ? el("button", { class: "btn btn--secondary btn--sm notes__more", type: "button", "aria-expanded": String(showAll) },
        showAll ? "Show fewer" : `Show all ${v.total}`)
      : null;
    more?.addEventListener("click", () => { showAll = !showAll; render(); });
    root.replaceChildren(list, ...(more ? [more] : []));
  }
  const wrap = (fn) => (...args) => { if (fn(...args)) render(); };
  render();
  return { set: wrap(notes.set), clear: wrap(notes.clear), report: wrap(notes.report) };
}
