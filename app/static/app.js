const $ = (s, el = document) => el.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const picked = { director: [], writer: [], cast: [], franchise: [] };
let otherCountries = [];
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
  fillSelect($("#add_country"), countries);
  $("#modelinfo").textContent =
    `${opt.model} · tested on 4,118 unseen films · average error ±${opt.metrics.MAE.toFixed(2)}`;
  document.querySelectorAll(".ac").forEach(setupAutocomplete);
  const d = new Date(); d.setMonth(d.getMonth() + 6);
  $("[name=release_date]").value = d.toISOString().slice(0, 10);
  $("#genres").addEventListener("change", e => {
    const on = [...document.querySelectorAll("[name=genres]:checked")];
    if (on.length > 3) { e.target.checked = false; showError("Pick at most 3 genres."); }
  });
  $("#add_country").addEventListener("change", e => {
    const c = e.target.value; e.target.value = "";
    if (c && c !== $("#country").value && !otherCountries.includes(c) && otherCountries.length < 15) {
      otherCountries.push(c); drawCountries();
    }
  });
  $("#country").addEventListener("change", () => {
    otherCountries = otherCountries.filter(c => c !== $("#country").value); drawCountries();
  });
  $("#form").addEventListener("submit", onSubmit);
}

function pretty(s) { return s.replace(/_/g, "-").replace(/^./, c => c.toUpperCase()); }

function chip(type, name, value, label) {
  const l = document.createElement("label");
  l.className = "chip";
  const i = Object.assign(document.createElement("input"), { type, name, value });
  l.append(i, label);
  return l;
}

function fillSelect(sel, items, selected) {
  for (const [v, t] of items) sel.add(new Option(t, v, false, v === selected));
}

function tag(label, onRemove, extra) {
  const t = document.createElement("span"); t.className = "tag"; t.append(label);
  if (extra) { const s = document.createElement("small"); s.textContent = extra; t.append(s); }
  const x = Object.assign(document.createElement("button"), { type: "button", textContent: "×" });
  x.setAttribute("aria-label", `Remove ${label}`);
  x.onclick = onRemove;
  t.append(x);
  return t;
}

function drawCountries() {
  $("#other_tags").replaceChildren(...otherCountries.map(c =>
    tag(regionName(c), () => { otherCountries = otherCountries.filter(x => x !== c); drawCountries(); })));
}

function setupAutocomplete(box) {
  const role = box.dataset.role, max = +box.dataset.max;
  const input = Object.assign(document.createElement("input"), {
    placeholder: role === "franchise" ? "e.g. Toy Story" : "Search a name…", autocomplete: "off" });
  input.setAttribute("role", "combobox");
  input.setAttribute("aria-label", box.querySelector(".field__label").textContent);
  const list = document.createElement("ul"); list.hidden = true; list.setAttribute("role", "listbox");
  const tags = document.createElement("div"); tags.className = "tags";
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
      li.append(p.name + born);
      const s = document.createElement("small");
      s.textContent = p.prev_count
        ? `${p.prev_count} earlier film${p.prev_count === 1 ? "" : "s"} · avg ${p.prev_mean.toFixed(1)} · latest: ${p.last_title} (${p.last_year})`
        : "No earlier films";
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
    tags.replaceChildren(...picked[role].map(p => tag(p.name,
      () => { picked[role] = picked[role].filter(q => q.id !== p.id); drawTags(); },
      p.prev_count ? `avg ${p.prev_mean.toFixed(1)}` : "debut")));
  }
}

function collect() {
  const f = new FormData($("#form"));
  const main = f.get("main_country");
  return {
    title: f.get("title"),
    release_date: f.get("release_date"),
    runtime: f.get("runtime"),
    budget_usd: f.get("budget_usd") || null,
    genres: f.getAll("genres"),
    keywords: f.getAll("keywords"),
    original_language: f.get("original_language"),
    us_certification: f.get("us_certification") || null,
    production_countries: [main, ...otherCountries.filter(c => c !== main)],
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
  setLoading(true);
  try {
    const r = await fetch("predict", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) });
    const data = await r.json().catch(() => ({ error: "Unexpected server response." }));
    if (!r.ok) return showError(data.error || "Prediction failed.");
    showResult(data);
  } catch {
    showError("Could not reach the server. Is it running?");
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  const btn = $("#submit");
  btn.disabled = on; btn.classList.toggle("is-loading", on);
  $(".btn__text", btn).textContent = on ? "Predicting…" : "Predict rating";
  $("#result").setAttribute("aria-busy", on);
}

function showError(msg) {
  const el = $("#error");
  el.hidden = !msg; $("#error-text").textContent = msg || "";
  if (msg) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function verdict(v) {
  return v >= 8 ? "Exceptional" : v >= 7 ? "Strong" : v >= 6 ? "Solid" : v >= 5 ? "Mixed" : "Weak";
}

function showResult(d) {
  $("#placeholder").hidden = true; $("#output").hidden = false;
  $("#out-title").textContent = d.title ? `Predicted IMDb rating · ${d.title}` : "Predicted IMDb rating";
  $("#score").textContent = d.prediction.toFixed(1);
  $("#verdict").textContent = verdict(d.prediction);
  const [lo, hi] = d.interval_80, pct = v => `${(v - 1) / 9 * 100}%`;
  $("#lo").textContent = lo.toFixed(1);
  $("#hi").textContent = hi.toFixed(1);
  Object.assign($("#range-band").style, { left: pct(lo), width: `${(hi - lo) / 9 * 100}%` });
  $("#range-mark").style.left = pct(d.prediction);
  drawDrivers(d.drivers);
  drawRecords(d.track_record);
  const top = $("#result").getBoundingClientRect().top;
  if (top < 0 || top > innerHeight * 0.6) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function el(tag, attrs, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  return n;
}

function drawDrivers(drivers) {
  const svg = $("#drivers"), css = getComputedStyle(document.documentElement);
  const color = v => css.getPropertyValue(v).trim();
  const active = drivers.filter(d => Math.abs(d.effect) >= 0.005);
  const idle = drivers.filter(d => Math.abs(d.effect) < 0.005);
  // label column | bar area (zero line placed by the data) | value column
  const H = 28, W = 340, x0 = 138, x1 = 292;
  const maxPos = Math.max(0, ...active.map(d => d.effect));
  const maxNeg = Math.max(0, ...active.map(d => -d.effect));
  const span = Math.max(maxPos + maxNeg, 0.25);
  const zero = x0 + (x1 - x0) * (maxNeg / span), scale = (x1 - x0) / span;
  const h = Math.max(active.length, 1) * H;
  svg.setAttribute("viewBox", `0 0 ${W} ${h + 4}`);
  $("#drivers-idle").textContent = idle.length
    ? `No effect here: ${idle.map(d => d.group.toLowerCase()).join(", ")}.` : "";
  const kids = [];
  active.forEach((d, i) => {
    const w = Math.abs(d.effect) * scale, y = i * H + 6, pos = d.effect >= 0;
    kids.push(
      el("rect", { x: x0, y: y - 2, width: x1 - x0, height: 20, rx: 3, fill: color("--cs-track") }),
      el("text", { x: 0, y: y + 12 }, d.group),
      el("rect", { x: pos ? zero : zero - w, y, width: Math.max(w, 2), height: 16, rx: 2,
        fill: color(pos ? "--cs-up" : "--cs-down") }),
      el("text", { x: W, y: y + 12, class: "val", "text-anchor": "end" },
        (pos ? "+" : "−") + Math.abs(d.effect).toFixed(2)),
    );
  });
  kids.push(el("line", { x1: zero, x2: zero, y1: 0, y2: h, stroke: color("--cs-ink"), "stroke-width": 1 }));
  if (!active.length) kids.push(el("text", { x: 0, y: 16 }, "No single input moves the forecast much."));
  svg.replaceChildren(...kids);
}

function drawRecords(tr) {
  const rows = [["Director", tr.director_prev_mean], ["Writers", tr.writer_prev_mean],
    ["Cast", tr.cast_prev_mean], ["Franchise", tr.franchise_prev_mean]].filter(r => r[1] != null);
  $("#records-block").hidden = !rows.length;
  $("#records").replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = `${v.toFixed(1)} avg`;
    return [dt, dd];
  }));
}
