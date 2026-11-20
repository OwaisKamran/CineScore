const $ = (s, el = document) => el.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const picked = { director: [], writer: [], cast: [], franchise: [] };
const regionName = (() => {
  try { const d = new Intl.DisplayNames(["en"], { type: "region" }); return c => d.of(c) || c; }
  catch { return c => c; }
})();

init();

async function init() {
  const opt = await (await fetch("api/options")).json();
  $("#genres").append(...opt.genres.map(g => chip("checkbox", "genres", g, pretty(g))));
  $("#keywords").append(...opt.keywords.map(k => chip("checkbox", "keywords", k.key, k.label)));
  fillSelect($("#language"), opt.languages.map(l => [l.code, l.name]), "en");
  fillSelect($("#cert"), opt.certifications.map(c => [c, c]));
  const countries = opt.countries.map(c => [c, regionName(c)]).sort((a, b) => a[1].localeCompare(b[1]));
  fillSelect($("#country"), countries, "US");
  fillSelect($("#other_countries"), countries);
  $("#modelinfo").textContent =
    `Model: ${opt.model}. On unseen 2022–2026 films its average error is ±${opt.metrics.MAE.toFixed(2)} rating points.`;
  document.querySelectorAll(".ac").forEach(setupAutocomplete);
  const d = new Date(); d.setMonth(d.getMonth() + 6);
  $("[name=release_date]").value = d.toISOString().slice(0, 10);
  $("#genres").addEventListener("change", e => {
    const on = [...document.querySelectorAll("[name=genres]:checked")];
    if (on.length > 3) { e.target.checked = false; showError("Pick at most 3 genres."); }
  });
  $("#form").addEventListener("submit", onSubmit);
}

function pretty(s) { return s.replace(/_/g, "-").replace(/^./, c => c.toUpperCase()); }

function chip(type, name, value, label) {
  const l = document.createElement("label");
  l.className = "chip";
  const i = Object.assign(document.createElement("input"), { type, name, value });
  l.append(i, " " + label);
  return l;
}

function fillSelect(sel, items, selected) {
  for (const [v, t] of items) sel.add(new Option(t, v, false, v === selected));
}

function setupAutocomplete(box) {
  const role = box.dataset.role, max = +box.dataset.max;
  const input = Object.assign(document.createElement("input"), {
    placeholder: role === "franchise" ? "e.g. Toy Story" : "Type a name…", autocomplete: "off" });
  input.setAttribute("role", "combobox");
  input.setAttribute("aria-label", box.querySelector("label").textContent);
  const list = document.createElement("ul"); list.hidden = true; list.setAttribute("role", "listbox");
  const tags = document.createElement("div"); tags.className = "picked";
  box.append(input, list, tags);
  let timer, items = [], active = -1;

  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = input.value.trim();
      if (q.length < 2) { list.hidden = true; return; }
      const r = await fetch(`api/search?role=${role}&q=${encodeURIComponent(q)}`);
      items = r.ok ? await r.json() : [];
      active = -1; render();
    }, 150);
  });
  input.addEventListener("keydown", e => {
    if (list.hidden) return;
    if (e.key === "ArrowDown") { active = Math.min(active + 1, items.length - 1); render(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); render(); e.preventDefault(); }
    else if (e.key === "Enter" && active >= 0) { choose(items[active]); e.preventDefault(); }
    else if (e.key === "Escape") list.hidden = true;
  });
  input.addEventListener("blur", () => setTimeout(() => (list.hidden = true), 150));

  function render() {
    list.replaceChildren(...items.map((p, i) => {
      const li = document.createElement("li");
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", i === active);
      const born = p.birth_year ? ` (b. ${p.birth_year})` : "";
      li.innerHTML = "";
      li.append(p.name + born);
      const s = document.createElement("small");
      s.textContent = `${p.prev_count} film${p.prev_count === 1 ? "" : "s"}, avg ${p.prev_mean.toFixed(1)} · latest: ${p.last_title} (${p.last_year})`;
      li.append(s);
      li.addEventListener("mousedown", e => { e.preventDefault(); choose(p); });
      return li;
    }));
    list.hidden = items.length === 0;
  }

  function choose(p) {
    if (picked[role].some(x => x.id === p.id)) return;
    if (picked[role].length >= max) { showError(`At most ${max} ${role} entr${max === 1 ? "y" : "ies"}.`); return; }
    picked[role].push(p);
    input.value = ""; list.hidden = true; drawTags();
  }

  function drawTags() {
    tags.replaceChildren(...picked[role].map(p => {
      const t = document.createElement("span"); t.className = "tag"; t.textContent = p.name;
      const x = Object.assign(document.createElement("button"), { type: "button", textContent: "×" });
      x.setAttribute("aria-label", `Remove ${p.name}`);
      x.onclick = () => { picked[role] = picked[role].filter(q => q.id !== p.id); drawTags(); };
      t.append(x); return t;
    }));
  }
}

function collect() {
  const f = new FormData($("#form"));
  const main = f.get("main_country");
  const others = [...$("#other_countries").selectedOptions].map(o => o.value).filter(c => c !== main);
  return {
    title: f.get("title"),
    release_date: f.get("release_date"),
    runtime: f.get("runtime"),
    budget_usd: f.get("budget_usd") || null,
    genres: f.getAll("genres"),
    keywords: f.getAll("keywords"),
    original_language: f.get("original_language"),
    us_certification: f.get("us_certification") || null,
    production_countries: [main, ...others],
    n_production_companies: f.get("n_production_companies"),
    n_spoken_languages: f.get("n_spoken_languages"),
    major_studio: f.get("major_studio") === "on",
    female_lead: f.get("female_lead"),
    directors: picked.director.map(p => p.id),
    writers: picked.writer.map(p => p.id),
    cast: picked.cast.map(p => p.id),
    franchise: picked.franchise[0]?.id || null,
  };
}

function clientCheck(p) {
  if (!p.release_date) return "Please choose a release date.";
  const rt = +p.runtime;
  if (!rt || rt < 60 || rt > 240) return "Runtime must be between 60 and 240 minutes.";
  if (p.genres.length < 1) return "Pick at least one genre.";
  if (p.budget_usd && (+p.budget_usd < 10000 || +p.budget_usd > 1e9))
    return "Budget must be between $10,000 and $1,000,000,000 (or leave it empty).";
  return null;
}

async function onSubmit(e) {
  e.preventDefault();
  const payload = collect();
  const problem = clientCheck(payload);
  if (problem) return showError(problem);
  showError(null);
  const btn = $("#submit"); btn.disabled = true; btn.textContent = "Predicting…";
  try {
    const r = await fetch("predict", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) });
    const data = await r.json().catch(() => ({ error: "Unexpected server response." }));
    if (!r.ok) return showError(data.error || "Prediction failed.");
    showResult(data);
  } catch {
    showError("Could not reach the server. Is it running?");
  } finally {
    btn.disabled = false; btn.textContent = "Predict rating";
  }
}

function showError(msg) {
  const el = $("#error");
  el.hidden = !msg; el.textContent = msg || "";
}

function showResult(d) {
  $("#placeholder").hidden = true; $("#output").hidden = false;
  $("#out-title").textContent = d.title ? `Predicted IMDb rating · ${d.title}` : "Predicted IMDb rating";
  $("#score").textContent = d.prediction.toFixed(1);
  $("#lo").textContent = d.interval_80[0].toFixed(1);
  $("#hi").textContent = d.interval_80[1].toFixed(1);
  drawScale(d);
  drawDrivers(d.drivers);
  if (matchMedia("(max-width: 860px)").matches) $("#result").scrollIntoView({ behavior: "smooth" });
}

function el(tag, attrs, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  return n;
}

function drawScale(d) {
  const svg = $("#scale"), x = v => 10 + (v - 1) / 9 * 280;
  const css = getComputedStyle(document.documentElement);
  svg.replaceChildren(
    el("rect", { x: 10, y: 14, width: 280, height: 8, rx: 4, fill: css.getPropertyValue("--line") }),
    el("rect", { x: x(d.interval_80[0]), y: 14, width: x(d.interval_80[1]) - x(d.interval_80[0]),
      height: 8, rx: 4, fill: css.getPropertyValue("--accent"), opacity: .35 }),
    el("circle", { cx: x(d.prediction), cy: 18, r: 7, fill: css.getPropertyValue("--accent") }),
    ...[1, 4, 7, 10].map(v => el("text", { x: x(v), y: 40, "text-anchor": "middle" }, v)),
  );
}

function drawDrivers(drivers) {
  const svg = $("#drivers"), css = getComputedStyle(document.documentElement);
  const rows = drivers.filter(d => Math.abs(d.effect) >= 0.005);
  const H = 22, W = 320, mid = 190, span = 120;
  const maxAbs = Math.max(0.25, ...rows.map(d => Math.abs(d.effect)));
  svg.setAttribute("viewBox", `0 0 ${W} ${rows.length * H + 6}`);
  const kids = [el("line", { x1: mid, x2: mid, y1: 0, y2: rows.length * H, stroke: css.getPropertyValue("--line") })];
  rows.forEach((d, i) => {
    const w = Math.abs(d.effect) / maxAbs * (span - 30), y = i * H + 4;
    const pos = d.effect >= 0;
    kids.push(
      el("text", { x: 0, y: y + 12, class: "lbl" }, d.group),
      el("rect", { x: pos ? mid : mid - w, y, width: Math.max(w, 1), height: 14, rx: 3,
        fill: css.getPropertyValue(pos ? "--good" : "--bad") }),
      el("text", { x: pos ? mid + w + 4 : mid - w - 4, y: y + 11, "text-anchor": pos ? "start" : "end" },
        (pos ? "+" : "−") + Math.abs(d.effect).toFixed(2)),
    );
  });
  if (!rows.length) kids.push(el("text", { x: 0, y: 14 }, "No single input moves the prediction much."));
  svg.replaceChildren(...kids);
}
