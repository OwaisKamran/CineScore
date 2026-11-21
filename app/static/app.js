const $ = (s, el = document) => el.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const picked = { director: [], writer: [], cast: [], franchise: [] };
const addPerson = {}, redrawPeople = {};   // per role, set up by setupAutocomplete
// "Try an example film" fills in this film (the user still presses Predict)
const EXAMPLE = { title: "Orbital", release_date: "2027-06-01", runtime: 150, budget_usd: "150000000",
  genres: ["drama", "sci_fi"], original_language: "en", main_country: "US", major_studio: true,
  people: { director: "Christopher Nolan", cast: "Cillian Murphy" } };
let otherCountries = [];
let STATS = null, lastGenres = [];
const regionName = (() => {
  try { const d = new Intl.DisplayNames(["en"], { type: "region" }); return c => d.of(c) || c; }
  catch { return c => c; }
})();

init();

async function init() {
  const [opt, stats] = await Promise.all(["api/options", "api/stats"].map(u => fetch(u).then(r => r.json())));
  STATS = stats;
  $("#genres").append(...opt.genres.map(g => chip("checkbox", "genres", g, pretty(g))));
  $("#keywords").append(...opt.keywords.map(k => chip("checkbox", "keywords", k.key, k.label)));
  fillSelect($("#language"), opt.languages.map(l => [l.code, l.name]));   // main inputs start empty
  fillSelect($("#cert"), opt.certifications.map(c => [c, c]));
  const countries = opt.countries.map(c => [c, regionName(c)]).sort((a, b) => a[1].localeCompare(b[1]));
  fillSelect($("#country"), countries);
  fillSelect($("#add_country"), countries);
  $("#modelinfo").textContent =
    `${opt.model.replace(/\s*\(.*?\)/, "")} · tested on 4,118 unseen films · off by ${opt.metrics.MAE.toFixed(2)} points on average`;
  document.querySelectorAll(".ac").forEach(setupAutocomplete);
  $("#example").addEventListener("click", fillExample);
  $("#clear").addEventListener("click", clearForm);
  $("#genres").addEventListener("change", e => {
    const on = [...document.querySelectorAll("[name=genres]:checked")];
    if (on.length > 3) { e.target.checked = false; showFieldError("genres", "Pick at most 3 genres."); }
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
  $("#form").addEventListener("input", recheckField);
  $("#form").addEventListener("change", recheckField);
  setupEnterKey();
  setupBudget();
  $("#hist-empty-desc").textContent = `IMDb ratings of the ${STATS.n.toLocaleString("en")} films the model ` +
    `learned from and was tested on (${STATS.years[0]}–${STATS.years[1]}). The average is ${STATS.mean.toFixed(1)}.`;
  drawHist($("#hist-empty"), null);
}

// Budget in US dollars: commas appear while typing, a helper line spells the amount out.
function budgetWords(n) {
  if (!n) return "Leave empty if unknown.";
  const say = (v, unit) => `= $${(+v.toFixed(2)).toLocaleString("en")} ${unit}`;
  return n >= 1e9 ? say(n / 1e9, "billion") : n >= 1e6 ? say(n / 1e6, "million")
    : n >= 1e3 ? say(n / 1e3, "thousand") : `= $${n}`;
}

function withCommas(digits) {
  return digits.replace(/^0+(?=\d)/, "").replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function setupBudget() {
  const input = $("#budget");
  input.addEventListener("input", () => {
    const caretDigits = input.value.slice(0, input.selectionStart).replace(/\D/g, "").length;
    const digits = input.value.replace(/\D/g, "");
    input.value = withCommas(digits);
    let pos = 0;                                  // put the caret back after the same digit
    for (let seen = 0; pos < input.value.length && seen < caretDigits; pos++)
      if (/\d/.test(input.value[pos])) seen++;
    input.setSelectionRange(pos, pos);
    $("#budget-help").textContent = budgetWords(+digits);
  });
}

// Enter moves to the next field; only the Predict button runs a prediction.
function setupEnterKey() {
  const form = $("#form");
  form.addEventListener("keydown", e => {
    if (e.key !== "Enter" || e.isComposing || e.defaultPrevented) return;  // autocomplete handled it
    const t = e.target;
    if (t.tagName === "BUTTON") return;                                       // buttons keep their action
    e.preventDefault();
    const fields = [...form.querySelectorAll("input:not([type=checkbox]):not([type=radio]), select")]
      .filter(f => !f.disabled && f.offsetParent !== null);
    const next = fields.find(f => t.compareDocumentPosition(f) & Node.DOCUMENT_POSITION_FOLLOWING);
    (next || $("#submit")).focus();
  });
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
  addPerson[role] = choose; redrawPeople[role] = drawTags;

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
    budget_usd: (f.get("budget_usd") || "").replace(/,/g, "") || null,  // plain number, as before
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

// same checks, same order and same wording as check_inputs() in the Streamlit app
// Returns [field, message] for the first problem (fields in form order), or null.
function clientCheck(p) {
  if (!p.release_date) return ["release_date", "Please choose a release date."];
  if (!p.runtime) return ["runtime", "Please enter the runtime in minutes."];
  const rt = +p.runtime;
  if (!rt || rt < 60 || rt > 240) return ["runtime", "Runtime must be between 60 and 240 minutes."];
  if (p.genres.length < 1) return ["genres", "Pick at least one genre."];
  if (p.genres.length > 3) return ["genres", "Pick at most 3 genres."];
  if (!p.original_language) return ["original_language", "Please choose the original language."];
  if (!p.production_countries[0]) return ["main_country", "Please choose the main production country."];
  if (p.budget_usd && (+p.budget_usd < 10000 || +p.budget_usd > 1e9))
    return ["budget_usd", "Budget must be between $10,000 and $1,000,000,000 (or leave it empty)."];
  return null;
}

// ---------------------------------------------------------------- field-level errors
// the element to highlight + focus for each field
const FIELD = {
  release_date: () => $("[name=release_date]"), runtime: () => $("[name=runtime]"),
  genres: () => $("#genres"), budget_usd: () => $("#budget"),
  original_language: () => $("#language"), main_country: () => $("#country"),
  n_production_companies: () => $("[name=n_production_companies]"),
  n_spoken_languages: () => $("[name=n_spoken_languages]"),
};
// server (/predict) messages -> field, so they are shown at the field too
const SERVER_FIELDS = [[/release date/i, "release_date"], [/runtime/i, "runtime"], [/genre/i, "genres"],
  [/budget/i, "budget_usd"], [/production compan/i, "n_production_companies"],
  [/spoken language/i, "n_spoken_languages"], [/original language|language code/i, "original_language"],
  [/country/i, "main_country"]];
let invalidField = null;

function showFieldError(field, msg) {
  clearFieldError();
  showError(msg, false);                                   // also in the alert next to the button
  const el = FIELD[field](), box = el.closest(".field, .group");
  box.classList.add("is-invalid");
  const p = Object.assign(document.createElement("p"), { className: "field__error", id: "field-error", textContent: msg });
  box.append(p);
  const target = field === "genres" ? $("[name=genres]") : el;
  target.setAttribute("aria-invalid", "true"); target.setAttribute("aria-describedby", "field-error");
  invalidField = field;
  box.scrollIntoView({ behavior: "smooth", block: "center" });
  target.focus({ preventScroll: true });
}

function clearFieldError() {
  document.querySelectorAll(".is-invalid").forEach(b => b.classList.remove("is-invalid"));
  document.querySelectorAll("[aria-invalid]").forEach(e => { e.removeAttribute("aria-invalid"); e.removeAttribute("aria-describedby"); });
  $("#field-error")?.remove();
  invalidField = null;
}

// remove the highlight as soon as the field is fixed (or update the message if it is still wrong)
function recheckField(e) {
  if (!invalidField) return;
  const box = FIELD[invalidField]().closest(".field, .group");
  const problem = clientCheck(collect());
  if (problem && problem[0] === invalidField) {
    $("#field-error").textContent = problem[1]; $("#error-text").textContent = problem[1];
  } else if (!(invalidField in { n_production_companies: 1, n_spoken_languages: 1 }) || box.contains(e.target)) {
    clearFieldError(); showError(null);
  }
}

function clearForm() {
  $("#form").reset();                                   // main inputs empty, minor ones back to defaults
  for (const role in picked) { picked[role] = []; redrawPeople[role](); }
  otherCountries = []; drawCountries();
  $("#budget-help").textContent = budgetWords(0);
  clearFieldError(); showError(null);
  $("#output").hidden = true; $("#placeholder").hidden = false;
}

async function fillExample() {
  clearForm();
  const f = $("#form").elements, x = EXAMPLE;
  f["title"].value = x.title;
  f["release_date"].value = x.release_date;
  f["runtime"].value = x.runtime;
  f["budget_usd"].value = withCommas(x.budget_usd);
  $("#budget-help").textContent = budgetWords(+x.budget_usd);
  x.genres.forEach(g => { $(`[name=genres][value=${g}]`).checked = true; });
  f["original_language"].value = x.original_language;
  f["main_country"].value = x.main_country;
  f["major_studio"].checked = x.major_studio;
  for (const [role, name] of Object.entries(x.people)) {
    const hits = await (await fetch(`api/search?role=${role}&q=${encodeURIComponent(name)}`)).json();
    const p = hits.find(h => h.name === name) || hits[0];
    if (p) addPerson[role](p);
  }
}

async function onSubmit(e) {
  e.preventDefault();
  const payload = collect();
  const problem = clientCheck(payload);
  if (problem) return showFieldError(...problem);
  clearFieldError(); showError(null);
  setLoading(true);
  try {
    const r = await fetch("predict", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) });
    const data = await r.json().catch(() => ({ error: "Unexpected server response." }));
    if (!r.ok) {
      const msg = data.error || "Prediction failed.", hit = SERVER_FIELDS.find(([re]) => re.test(msg));
      return hit ? showFieldError(hit[1], msg) : showError(msg);
    }
    lastGenres = payload.genres;
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

function showError(msg, scroll = true) {
  const el = $("#error");
  el.hidden = !msg; $("#error-text").textContent = msg || "";
  if (msg && scroll) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function verdict(v) {
  return v >= 8 ? "Exceptional" : v >= 7 ? "Strong" : v >= 6 ? "Solid" : v >= 5 ? "Mixed" : "Weak";
}

function showResult(d) {
  $("#placeholder").hidden = true; $("#output").hidden = false;
  $("#out-title").textContent = d.title ? `Predicted IMDb rating · ${d.title}` : "Predicted IMDb rating";
  $("#score").textContent = d.prediction.toFixed(1);
  $("#verdict").textContent = verdict(d.prediction);
  const [lo, hi] = d.interval_80;
  $("#lo").textContent = lo.toFixed(1);
  $("#hi").textContent = hi.toFixed(1);
  drawScale(d.prediction, lo, hi);
  drawCompare(d.prediction, d.track_record, lastGenres);
  drawHist($("#hist"), d.prediction);
  drawDrivers(d.drivers);
  // bring the result into view (desktop and mobile)
  requestAnimationFrame(() => {
    const top = $("#result").getBoundingClientRect().top;
    if (top < 0 || top > 80) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function el(tag, attrs, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  return n;
}

// ---------------------------------------------------------------- charts (same geometry in the Streamlit app)
const W = 340, X0 = 8, X1 = 332;
const xr = v => X0 + (Math.min(10, Math.max(1, v)) - 1) / 9 * (X1 - X0);
const color = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// 1-10 ruler: 80% range as a black band, the forecast as a green mark
function drawScale(pred, lo, hi) {
  const k = [el("line", { x1: X0, x2: X1, y1: 34, y2: 34, stroke: color("--cs-line-strong") })];
  for (let i = 0; i <= 18; i++) {
    const v = 1 + i / 2, major = i % 2 === 0;
    k.push(el("line", { x1: xr(v), x2: xr(v), y1: 34, y2: major ? 42 : 38,
      stroke: color(major ? "--cs-ink" : "--cs-line-strong") }));
    if (major) k.push(el("text", { x: xr(v), y: 56, "text-anchor": "middle", class: "tick" }, String(v)));
  }
  k.push(el("rect", { x: xr(lo), y: 31, width: xr(hi) - xr(lo), height: 6, fill: color("--cs-ink") }),
    el("text", { x: xr(lo), y: 25, "text-anchor": "middle", class: "tick" }, lo.toFixed(1)),
    el("text", { x: xr(hi), y: 25, "text-anchor": "middle", class: "tick" }, hi.toFixed(1)),
    el("rect", { x: xr(pred) - 1.5, y: 20, width: 3, height: 26, fill: color("--cs-accent") }));
  $("#scale").setAttribute("viewBox", `0 0 ${W} 60`);
  $("#scale").replaceChildren(...k);
}

// dot chart: forecast vs the people's earlier films, the genres and all films
function drawCompare(pred, tr, genres) {
  const rows = [["This forecast", pred, true],
    ["Director's earlier films", tr.director_prev_mean], ["Writers' earlier films", tr.writer_prev_mean],
    ["Lead cast's earlier films", tr.cast_prev_mean], ["Franchise's earlier films", tr.franchise_prev_mean],
    ...genres.map(g => [`${pretty(g)} films`, STATS.genres[g][0]]), ["All films", STATS.mean]]
    .filter(r => r[1] != null);
  const vals = rows.map(r => r[1]);
  const lo = Math.max(1, Math.floor(Math.min(...vals) - 0.5));
  const hi = Math.min(10, Math.ceil(Math.max(...vals) + 0.5));
  const H = 26, a = 162, b = 296, px = v => a + (v - lo) / (hi - lo) * (b - a), h = rows.length * H;
  const k = [];
  rows.forEach(([label, v, own], i) => {
    const y = i * H + 13;
    k.push(el("line", { x1: a, x2: b, y1: y, y2: y, stroke: color("--cs-line") }),
      el("text", { x: 0, y: y + 4, class: own ? "own" : "" }, label),
      el("circle", { cx: px(v), cy: y, r: own ? 5.5 : 4.5, fill: color(own ? "--cs-accent" : "--cs-ink") }),
      el("text", { x: W, y: y + 4, "text-anchor": "end", class: own ? "val own" : "val" }, v.toFixed(1)));
  });
  k.push(el("line", { x1: px(pred), x2: px(pred), y1: 2, y2: h, stroke: color("--cs-accent"), "stroke-dasharray": "2 3" }));
  for (let v = lo; v <= hi; v++)
    k.push(el("text", { x: px(v), y: h + 14, "text-anchor": "middle", class: "tick" }, String(v)));
  $("#compare").setAttribute("viewBox", `0 0 ${W} ${h + 18}`);
  $("#compare").replaceChildren(...k);
}

// share of all films rated below a value (IMDb ratings have one decimal)
function shareBelow(v) {
  return STATS.counts.reduce((s, c, i) => s + (1 + i / 10 < v - 1e-9 ? c : 0), 0) / STATS.n;
}

// histogram of all films' ratings (0.2-wide bins), average line, optional forecast marker
function drawHist(svg, pred) {
  const c = STATS.counts, bins = [];
  for (let b = 0; b < 45; b++) bins.push(c[2 * b] + c[2 * b + 1] + (b === 44 ? c[90] : 0));
  const max = Math.max(...bins), top = 30, base = 100, bw = (X1 - X0) / 45;
  const k = bins.map((n, b) => {
    const h = n / max * (base - top), below = pred != null && 1.1 + 0.2 * b < pred;
    return el("rect", { x: X0 + b * bw + 0.5, y: base - h, width: bw - 1, height: h,
      fill: color(pred == null || below ? "--cs-down" : "--cs-line") });
  });
  k.push(el("line", { x1: X0, x2: X1, y1: base, y2: base, stroke: color("--cs-ink") }));
  for (const v of [1, 4, 7, 10])
    k.push(el("text", { x: xr(v), y: base + 14, "text-anchor": "middle", class: "tick" }, String(v)));
  const xm = xr(STATS.mean), right = pred == null || pred < STATS.mean;
  k.push(el("line", { x1: xm, x2: xm, y1: 16, y2: base, stroke: color("--cs-ink"), "stroke-dasharray": "3 2" }),
    el("text", { x: right ? xm + 5 : xm - 5, y: 24, "text-anchor": right ? "start" : "end", class: "tick" },
      `average ${STATS.mean.toFixed(1)}`));
  if (pred != null) {
    const xp = xr(pred), onRight = pred >= STATS.mean && xp < W - 90;
    k.push(el("rect", { x: xp - 1, y: 4, width: 2, height: base - 4, fill: color("--cs-accent") }),
      el("text", { x: onRight ? xp + 5 : xp - 5, y: 12, "text-anchor": onRight ? "start" : "end", class: "own accent" },
        `This film ${pred.toFixed(1)}`));
    $("#hist-desc").textContent = `Rated higher than ${Math.round(shareBelow(pred) * 100)}% of the ` +
      `${STATS.n.toLocaleString("en")} films in the data (${STATS.years[0]}–${STATS.years[1]}).`;
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${base + 18}`);
  svg.replaceChildren(...k);
}

function drawDrivers(drivers) {
  const svg = $("#drivers");
  const active = drivers.filter(d => Math.abs(d.effect) >= 0.005);
  const idle = drivers.filter(d => Math.abs(d.effect) < 0.005);
  // label column | bar area (zero line placed by the data) | value column
  const H = 28, x0 = 138, x1 = 292;
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
      el("rect", { x: x0, y: y - 2, width: x1 - x0, height: 20, fill: color("--cs-track") }),
      el("text", { x: 0, y: y + 12 }, d.group),
      el("rect", { x: pos ? zero : zero - w, y, width: Math.max(w, 2), height: 16,
        fill: color(pos ? "--cs-up" : "--cs-down") }),
      el("text", { x: W, y: y + 12, class: "val", "text-anchor": "end" },
        (pos ? "+" : "−") + Math.abs(d.effect).toFixed(2)),
    );
  });
  kids.push(el("line", { x1: zero, x2: zero, y1: 0, y2: h, stroke: color("--cs-ink"), "stroke-width": 1 }));
  if (!active.length) kids.push(el("text", { x: 0, y: 16 }, "No single input moves the forecast much."));
  svg.replaceChildren(...kids);
}
