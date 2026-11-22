// The landing page: the hero (headline, sentence, the two demos, the live simulator dog), how it learns in three steps
// with each demo's numbers, the big numbers, the two demo cards and the footer. Its words are fixed; every number is
// from GET /api/demos, and a number the data leaves out is not shown.
import { heroSub, homeSteps, statsBand } from "./showcase.js";
import { $, el, setText } from "./dom.js";

export function initHomeView() {
  let key = "";

  function render(demos) {
    const list = Array.isArray(demos) ? demos : [];
    const k = JSON.stringify(list.map((d) => [d.id, d.name, d.title, d.tagline, d.showcase, d.lanes]));
    if (k === key) return;
    key = k;
    setText($("hero-sub"), heroSub(list));

    $("how-steps").replaceChildren(...homeSteps(list).map((s, i) => el("li", { class: "how__step" },
      el("span", { class: "how__num", "aria-hidden": "true" }, String(i + 1)),
      el("h3", { class: "how__title" }, s.title),
      el("p", { class: "how__text" }, s.sentence),
      s.values.length ? el("div", { class: "how__values" }, ...s.values.map((v) => el("p", { class: "how__value" },
        el("span", { class: "how__big" }, v.value), el("span", { class: "how__unit" }, v.unit),
        s.values.length > 1 && v.demo ? el("span", { class: "how__demo" }, v.demo) : null))) : null)));

    const links = $("home-links");
    links.replaceChildren(...list.map((d) => el("a", { href: `#${encodeURIComponent(d.id)}` }, d.name ?? d.title)));

    const stats = statsBand(list);
    $("stats").hidden = !stats.length;
    $("stats-list").replaceChildren(...stats.map((s) => el("li", { class: "stat" },
      el("span", { class: "stat__big" }, s.value),
      el("span", { class: "stat__label" }, s.label),
      s.sub ? el("span", { class: "stat__sub" }, s.sub) : null)));

    $("cards").hidden = !list.length;
    $("cards-list").replaceChildren(...list.map((d) => el("li", { class: "card" },
      el("h3", { class: "card__name" }, d.name ?? d.title),
      d.name ? el("p", { class: "card__title" }, d.title) : null,
      d.tagline ? el("p", { class: "card__tagline" }, d.tagline) : null,
      el("a", { class: "pill pill--light card__open", href: `#${encodeURIComponent(d.id)}`, "aria-label": `Open ${d.name ?? d.title}` }, "Open"))));
  }

  return { render };
}
