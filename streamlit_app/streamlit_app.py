"""CineScore - Streamlit front end.

Local:  streamlit run streamlit_app/streamlit_app.py      (run from the repo root so that
        .streamlit/config.toml, the theme, is picked up)
Cloud:  Streamlit Community Cloud, main file path = streamlit_app/streamlit_app.py

Reuses the saved scikit-learn pipeline (app/model/) and the name lookups (app/data/) through the
same Predictor class as the Flask app, so both front ends validate and predict identically.
Colours and fonts come from app/static/tokens.css and the charts use the same geometry as
app/static/app.js, so both apps look the same.
"""
import json
import math
import re
import sys
from datetime import date
from html import escape
from pathlib import Path

import streamlit as st

# In the browser (stlite on Pyodide, the Hugging Face Space) custom components cannot load their
# frontend, so the app uses a built-in search box + match list there instead of streamlit-searchbox.
IN_BROWSER = sys.platform == "emscripten"
st_searchbox = None
if not IN_BROWSER:
    try:  # one-box autocomplete
        from streamlit_searchbox import st_searchbox
    except ImportError:  # pragma: no cover
        st_searchbox = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from predictor import CERTIFICATIONS, KEYWORDS, LANGUAGE_NAMES, ROLE_LIMITS, Predictor, ValidationError  # noqa: E402

st.set_page_config(page_title="CineScore", page_icon=":material/movie:", layout="wide",
                   initial_sidebar_state="collapsed", menu_items={})


@st.cache_resource(show_spinner="Loading the model…")
def load_predictor() -> Predictor:
    return Predictor()


def load_assets() -> tuple[str, dict, dict]:  # small files; read each run so CSS edits apply live
    css = (ROOT / "app" / "static" / "tokens.css").read_text(encoding="utf-8")
    css += (Path(__file__).parent / "streamlit.css").read_text(encoding="utf-8")
    data = ROOT / "app" / "data"
    names = json.loads((data / "country_names.json").read_text(encoding="utf-8"))
    stats = json.loads((data / "rating_stats.json").read_text(encoding="utf-8"))
    return css, names, stats


P = load_predictor()
OPTS = P.options()
CSS, COUNTRY_NAMES, STATS = load_assets()
st.html(f"<style>{CSS}</style>")


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def section(title: str, desc: str) -> None:
    html(f'<div class="cs-sec"><h2 class="cs-sec__title">{title}</h2><p class="cs-sec__desc">{desc}</p></div>')


def checkbox_grid(key: str, label: str, options: list[str], fmt) -> list[str]:
    """Aligned grid of square checkboxes (4 across, 2 on phones), same as the Flask form."""
    html(f'<p class="cs-label">{label}</p>')
    picked = []
    with st.container(key=key, horizontal=True):
        for o in options:
            if st.checkbox(fmt(o), key=f"{key}_{o}"):
                picked.append(o)
    return picked


def pretty_genre(g: str) -> str:
    return g.replace("_", "-").capitalize()


def budget_words(digits: str) -> str:
    """Helper line under the budget field (same wording as the Flask app)."""
    if not digits:
        return "Leave empty if unknown."
    if not digits.isdigit():
        return "Use digits only, e.g. 40,000,000."
    n = int(digits)
    say = lambda v, unit: f"= ${f'{round(v, 2):,.2f}'.rstrip('0').rstrip('.')} {unit}"  # noqa: E731
    return (say(n / 1e9, "billion") if n >= 1e9 else say(n / 1e6, "million") if n >= 1e6
            else say(n / 1e3, "thousand") if n >= 1e3 else f"= ${n}")


def verdict(v: float) -> str:
    return "Exceptional" if v >= 8 else "Strong" if v >= 7 else "Solid" if v >= 6 else "Mixed" if v >= 5 else "Weak"


# ---------------------------------------------------------------- people: one search box per role
def _suggestion(r: dict) -> str:
    born = f" (b. {r['birth_year']})" if r.get("birth_year") else ""
    if r["prev_count"]:
        n = r["prev_count"]
        return f"{r['name']}{born} · {n} earlier film{'' if n == 1 else 's'} · avg {r['prev_mean']:.1f}"
    return f"{r['name']}{born} · no earlier films"


SEARCHBOX_STYLE = {
    "wrapper": {"backgroundColor": "#ffffff"},
    "clear": {"icon": "cross", "clearable": "never"},
    "dropdown": {"rotate": True, "width": 14, "height": 14, "fill": "#333333"},
    "searchbox": {
        "control": {"borderRadius": 0, "border": "1px solid #8d8d8a", "minHeight": "42px", "boxShadow": "none",
                    "&:hover": {"borderColor": "#111111"},
                    "backgroundColor": "#ffffff", "fontFamily": '"Libre Franklin", "Segoe UI", Arial, sans-serif'},
        "input": {"color": "#111111", "fontSize": "15px"},
        "placeholder": {"color": "#767674", "fontSize": "15px"},
        "menuList": {"backgroundColor": "#ffffff", "border": "1px solid #111111", "borderRadius": 0,
                     "fontFamily": '"Libre Franklin", "Segoe UI", Arial, sans-serif', "fontSize": "14px"},
        "option": {"color": "#111111", "backgroundColor": "#ffffff", "highlightColor": "#f2f2f0"},
        "optionEmpty": "hidden",
    },
}


def _remove(role: str, pid: str) -> None:
    st.session_state[f"picks_{role}"] = [p for p in st.session_state[f"picks_{role}"] if p["id"] != pid]
    if isinstance(st.session_state.get(f"sb_{role}"), dict):  # allow picking the same person again
        st.session_state[f"sb_{role}"]["result"] = None


def people_picker(role: str, title: str) -> list[str]:
    """Type in one box, suggestions appear under it, click (or arrow + Enter) to add a name."""
    picks = st.session_state.setdefault(f"picks_{role}", [])
    limit = ROLE_LIMITS[role]
    html(f'<p class="cs-label cs-label--box">{title}</p>')
    if st_searchbox is None:
        _fallback_box(role, limit)
    else:
        _searchbox(role, limit)
    if picks:
        with st.container(key=f"tags_{role}", horizontal=True):
            for p in picks:
                extra = f"avg {p['prev_mean']:.1f}" if p["prev_count"] else "debut"
                st.button(f"{p['name']} · {extra}  ×", key=f"rm_{role}_{p['id']}", on_click=_remove,
                          args=(role, p["id"]), help=f"Remove {p['name']}")
    return [p["id"] for p in picks]


def _add_person(role: str, limit: int, person: dict) -> None:
    cur = st.session_state[f"picks_{role}"]
    if isinstance(person, dict) and person["id"] not in {p["id"] for p in cur} and len(cur) < limit:
        cur.append(person)


def _searchbox(role: str, limit: int) -> None:
    """streamlit-searchbox: type in one box, suggestions appear under it, pick one."""
    def search(q: str) -> list:
        return [(_suggestion(r), r) for r in P.search(role, q)] if len(q.strip()) >= 2 else []

    with st.container(key=f"sbwrap_{role}"):
        st_searchbox(search, placeholder="e.g. Toy Story" if role == "franchise" else "Search a name…",
                     key=f"sb_{role}", clear_on_submit=True, debounce=150, style_overrides=SEARCHBOX_STYLE,
                     submit_function=lambda r: _add_person(role, limit, r))


def _pick_match(role: str, limit: int) -> None:
    pid = st.session_state.get(f"pick_{role}")
    if pid:
        _add_person(role, limit, st.session_state[f"hits_{role}"][pid])
    st.session_state[f"pick_{role}"] = None
    st.session_state[f"q_{role}"] = ""


def _fallback_box(role: str, limit: int) -> None:
    """Built-in widgets only (used in the browser Space): search box, then a list of matches to pick from."""
    q = st.text_input("Search", key=f"q_{role}", label_visibility="collapsed",
                      placeholder="e.g. Toy Story" if role == "franchise" else "Search a name…")
    hits = P.search(role, q) if q and len(q.strip()) >= 2 else []
    if hits:
        st.session_state[f"hits_{role}"] = {h["id"]: h for h in hits}
        st.selectbox("Matches", list(st.session_state[f"hits_{role}"]), index=None, key=f"pick_{role}",
                     label_visibility="collapsed", placeholder=f"{len(hits)} match{'es' if len(hits) > 1 else ''}: pick one",
                     format_func=lambda i: _suggestion(st.session_state[f"hits_{role}"][i]),
                     on_change=_pick_match, args=(role, limit))
    elif q and len(q.strip()) >= 2:
        html('<p class="cs-hint">No matches. Check the spelling, or leave this empty for a debut.</p>')


# ---------------------------------------------------------------- charts (same geometry as app/static/app.js)
W, X0, X1 = 340, 8, 332


def xr(v: float) -> float:
    return X0 + (min(10.0, max(1.0, v)) - 1) / 9 * (X1 - X0)


def fill(var: str) -> str:
    return f'style="fill:var({var})"'


def stroke(var: str, extra: str = "") -> str:
    return f'style="stroke:var({var}){extra}"'


def svg(h: float, body: str, label: str) -> str:
    return f'<svg class="cs-chart" viewBox="0 0 {W} {h:g}" role="img" aria-label="{label}">{body}</svg>'


def text(x, y, s, cls="", anchor="start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" class="{cls}">{escape(s)}</text>'


def scale_svg(pred: float, lo: float, hi: float) -> str:
    """1-10 ruler: 80% range as a black band, the forecast as a green mark."""
    k = [f'<line x1="{X0}" x2="{X1}" y1="34" y2="34" {stroke("--cs-line-strong")}/>']
    for i in range(19):
        v, major = 1 + i / 2, i % 2 == 0
        k.append(f'<line x1="{xr(v):.2f}" x2="{xr(v):.2f}" y1="34" y2="{42 if major else 38}" '
                 f'{stroke("--cs-ink" if major else "--cs-line-strong")}/>')
        if major:
            k.append(text(xr(v), 56, str(int(v)), "tick", "middle"))
    k += [f'<rect x="{xr(lo):.2f}" y="31" width="{xr(hi) - xr(lo):.2f}" height="6" {fill("--cs-ink")}/>',
          text(xr(lo), 25, f"{lo:.1f}", "tick", "middle"), text(xr(hi), 25, f"{hi:.1f}", "tick", "middle"),
          f'<rect x="{xr(pred) - 1.5:.2f}" y="20" width="3" height="26" {fill("--cs-accent")}/>']
    return svg(60, "".join(k), "Predicted rating and 80% range on a 1 to 10 scale")


def compare_svg(pred: float, tr: dict, genres: list[str]) -> str:
    """Dot chart: forecast vs the people's earlier films, the genres and all films."""
    rows = [("This forecast", pred, True),
            ("Director's earlier films", tr["director_prev_mean"], False),
            ("Writers' earlier films", tr["writer_prev_mean"], False),
            ("Lead cast's earlier films", tr["cast_prev_mean"], False),
            ("Franchise's earlier films", tr["franchise_prev_mean"], False),
            *[(f"{pretty_genre(g)} films", STATS["genres"][g][0], False) for g in genres],
            ("All films", STATS["mean"], False)]
    rows = [r for r in rows if r[1] is not None]
    vals = [r[1] for r in rows]

    lo, hi = max(1, math.floor(min(vals) - 0.5)), min(10, math.ceil(max(vals) + 0.5))
    H, a, b = 26, 162, 296
    px = lambda v: a + (v - lo) / (hi - lo) * (b - a)  # noqa: E731
    h = len(rows) * H
    k = []
    for i, (label, v, own) in enumerate(rows):
        y = i * H + 13
        k += [f'<line x1="{a}" x2="{b}" y1="{y}" y2="{y}" {stroke("--cs-line")}/>',
              text(0, y + 4, label, "own" if own else ""),
              f'<circle cx="{px(v):.2f}" cy="{y}" r="{5.5 if own else 4.5}" {fill("--cs-accent" if own else "--cs-ink")}/>',
              text(W, y + 4, f"{v:.1f}", "val own" if own else "val", "end")]
    k.append(f'<line x1="{px(pred):.2f}" x2="{px(pred):.2f}" y1="2" y2="{h}" '
             f'{stroke("--cs-accent", ";stroke-dasharray:2 3")}/>')
    k += [text(px(v), h + 14, str(v), "tick", "middle") for v in range(lo, hi + 1)]
    return svg(h + 18, "".join(k), "Forecast compared with track records and genre averages")


def share_below(v: float) -> float:
    """Share of all films rated below v (IMDb ratings have one decimal)."""
    return sum(c for i, c in enumerate(STATS["counts"]) if 1 + i / 10 < v - 1e-9) / STATS["n"]


def hist_svg(pred: float | None) -> str:
    """Histogram of all films' ratings (0.2-wide bins), average line, optional forecast marker."""
    c = STATS["counts"]
    bins = [c[2 * b] + c[2 * b + 1] + (c[90] if b == 44 else 0) for b in range(45)]
    top, base, bw = 30, 100, (X1 - X0) / 45
    k = []
    for b, n in enumerate(bins):
        h = n / max(bins) * (base - top)
        dark = pred is None or 1.1 + 0.2 * b < pred
        k.append(f'<rect x="{X0 + b * bw + 0.5:.2f}" y="{base - h:.2f}" width="{bw - 1:.2f}" height="{h:.2f}" '
                 f'{fill("--cs-down" if dark else "--cs-line")}/>')
    k.append(f'<line x1="{X0}" x2="{X1}" y1="{base}" y2="{base}" {stroke("--cs-ink")}/>')
    k += [text(xr(v), base + 14, str(v), "tick", "middle") for v in (1, 4, 7, 10)]
    xm, right = xr(STATS["mean"]), pred is None or pred < STATS["mean"]
    k += [f'<line x1="{xm:.2f}" x2="{xm:.2f}" y1="16" y2="{base}" {stroke("--cs-ink", ";stroke-dasharray:3 2")}/>',
          text(xm + 5 if right else xm - 5, 24, f"average {STATS['mean']:.1f}", "tick", "start" if right else "end")]
    if pred is not None:
        xp = xr(pred)
        on_right = pred >= STATS["mean"] and xp < W - 90
        k += [f'<rect x="{xp - 1:.2f}" y="4" width="2" height="{base - 4}" {fill("--cs-accent")}/>',
              text(xp + 5 if on_right else xp - 5, 12, f"This film {pred:.1f}", "own accent",
                   "start" if on_right else "end")]
    return svg(base + 18, "".join(k), "Distribution of IMDb ratings" if pred is None
               else "Where this forecast sits among all films' ratings")


def years() -> str:
    return f"{STATS['years'][0]}–{STATS['years'][1]}"


# ---------------------------------------------------------------- result rendering
def render_result(res: dict, genres: list[str], stale: bool) -> None:
    lo, hi = res["interval_80"]
    pred = res["prediction"]
    title = f" · {escape(res['title'])}" if res["title"] else ""
    stale_note = ('<p class="cs-stale">Inputs changed since this forecast. '
                  'Press <strong>Predict rating</strong> to update it.</p>') if stale else ""

    active = [d for d in res["drivers"] if abs(d["effect"]) >= 0.005]
    idle = [d["group"].lower() for d in res["drivers"] if abs(d["effect"]) < 0.005]
    max_pos = max([d["effect"] for d in active] + [0])
    max_neg = max([-d["effect"] for d in active] + [0])
    span = max(max_pos + max_neg, 0.25)
    zero = max_neg / span * 100
    rows = []
    for d in active:
        e = d["effect"]
        w = abs(e) / span * 100
        left = zero if e >= 0 else zero - w
        rows.append(
            f'<div class="cs-drv"><span class="cs-drv__label">{escape(d["group"])}</span>'
            f'<span class="cs-drv__track"><span class="cs-drv__bar {"up" if e >= 0 else "down"}" '
            f'style="left:{left:.2f}%;width:{max(w, 0.8):.2f}%"></span>'
            f'<span class="cs-drv__zero" style="left:{zero:.2f}%"></span></span>'
            f'<span class="cs-drv__val">{"+" if e >= 0 else "−"}{abs(e):.2f}</span></div>')
    drivers = "".join(rows) or '<p class="cs-hint">No single input moves the forecast much.</p>'
    idle_note = f'<p class="cs-desc cs-idle">No effect here: {", ".join(idle)}.</p>' if idle else ""
    pct = int(share_below(pred) * 100 + 0.5)

    html(f"""
<div class="cs-result {'is-stale' if stale else ''}">
  {stale_note}
  <p class="cs-eyebrow">Predicted IMDb rating{title}</p>
  <div class="cs-score"><span class="cs-score__num">{pred:.1f}</span><span class="cs-score__of">/ 10</span>
    <span class="cs-verdict">{verdict(pred)}</span></div>
  <div class="cs-range">
    {scale_svg(pred, lo, hi)}
    <p class="cs-range__text"><strong>80% range: {lo:.1f} to {hi:.1f}</strong>
      Comparable films land here about 4 times in 5.</p>
  </div>
  <div class="cs-block">
    <h3>How it compares</h3>
    <p class="cs-desc">This forecast next to the average rating of the people's earlier films
      and of films in the same genres.</p>
    {compare_svg(pred, res["track_record"], genres)}
  </div>
  <div class="cs-block">
    <h3>Among all films</h3>
    <p class="cs-desc">Rated higher than {pct}% of the {STATS['n']:,} films in the data ({years()}).</p>
    {hist_svg(pred)}
  </div>
  <div class="cs-block">
    <h3>What drives this forecast</h3>
    <p class="cs-desc">Points each input adds or removes, compared with a neutral film (unknown people,
      no genre, 103 minutes, no budget, English, US).</p>
    <div class="cs-drivers">{drivers}</div>
    {idle_note}
  </div>
  <p class="cs-note">The model is least accurate for very weak or exceptional films and cannot foresee
    hype or review-bombing.</p>
</div>""")


def render_empty() -> None:
    html(f"""
<div class="cs-result">
  <p class="cs-eyebrow">Your forecast</p>
  <p class="cs-empty__num">—<span>/ 10</span></p>
  <p class="cs-empty__text">Describe the film on the left and press <strong>Predict rating</strong>.
    You will see the expected rating, a likely range, how it compares and what drives it.</p>
  <div class="cs-block">
    <h3>How films are rated</h3>
    <p class="cs-desc">IMDb ratings of the {STATS['n']:,} films the model learned from and was tested on
      ({years()}). The average is {STATS['mean']:.1f}.</p>
    {hist_svg(None)}
  </div>
</div>""")


# ---------------------------------------------------------------- keyboard + scrolling (runs in the page)
# Enter moves to the next field (Streamlit has no form submit on Enter; only Predict runs the model).
# The script is added to the parent page once, so it keeps working across reruns.
PAGE_JS = r"""
(function () {
  const doc = document;
  const main = () => doc.querySelector('[data-testid="stMain"]') || doc.body;
  const isBox = f => f.tagName === "IFRAME";
  function fields() {
    // typed inputs, the first segment (day) of each date field, and the people search boxes
    return [...main().querySelectorAll('input:not([type=checkbox]):not([type=radio]):not([type=date]), ' +
      '[data-testid="stDateInputField"] [role="spinbutton"][data-type="day"], iframe[src*="streamlit_searchbox"]')]
      .filter(f => f.offsetParent !== null && !f.disabled);
  }
  function focusField(f) {
    if (!f) { const b = doc.querySelector('.st-key-predict button'); if (b) b.focus(); return; }
    if (isBox(f)) { try { f.contentDocument.querySelector("input").focus(); return; } catch (e) {} }
    f.focus();
  }
  function next(from) {
    from = from.closest && from.closest('[data-testid="stDateInputField"]') || from;   // a date counts as one field
    return fields().find(f => !from.contains(f) && (from.compareDocumentPosition(f) & Node.DOCUMENT_POSITION_FOLLOWING));
  }
  doc.addEventListener("keydown", e => {
    const t = e.target;
    const typed = (t.tagName === "INPUT" && t.type !== "checkbox") || t.getAttribute("role") === "spinbutton";
    if (e.key !== "Enter" || e.isComposing || !typed) return;
    if (t.getAttribute("aria-expanded") === "true") return;           // a select menu is open
    if (!main().contains(t)) return;
    setTimeout(() => focusField(next(t)), 0);                          // Streamlit still commits the value
  }, true);
  // search boxes live in iframes: Enter with no suggestion list open moves on as well
  function hook(frame) {
    try {
      const d = frame.contentDocument;
      if (!d || d.__csHooked || !d.querySelector("input")) return;
      d.__csHooked = true;
      const f = d.createElement("link");                                // same body font as the page
      f.rel = "stylesheet"; f.href = "https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600&display=swap";
      d.head.appendChild(f);
      d.addEventListener("keydown", e => {
        if (e.key !== "Enter") return;
        if (d.querySelector('[class*="menu"]')) return;                 // suggestion list open: let it pick
        e.preventDefault();
        focusField(next(frame));
      }, true);
    } catch (e) {}
  }
  setInterval(() => doc.querySelectorAll('iframe[src*="streamlit_searchbox"]').forEach(hook), 700);
  // budget: commas appear while typing and the helper line spells the amount out (as in Flask)
  const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
  function words(n) {
    if (!n) return "Leave empty if unknown.";
    const say = (v, unit) => `= $${(+v.toFixed(2)).toLocaleString("en")} ${unit}`;
    return n >= 1e9 ? say(n / 1e9, "billion") : n >= 1e6 ? say(n / 1e6, "million")
      : n >= 1e3 ? say(n / 1e3, "thousand") : `= $${n}`;
  }
  doc.addEventListener("input", e => {
    const t = e.target;
    if (!t.closest || !t.closest(".st-key-budget")) return;
    const caretDigits = t.value.slice(0, t.selectionStart).replace(/\D/g, "").length;
    const digits = t.value.replace(/\D/g, "");
    const v = digits.replace(/^0+(?=\d)/, "").replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    if (v !== t.value) {
      setValue.call(t, v);
      t.dispatchEvent(new Event("input", { bubbles: true }));     // keep Streamlit's copy in step
      let pos = 0;
      for (let seen = 0; pos < v.length && seen < caretDigits; pos++) if (/\d/.test(v[pos])) seen++;
      t.setSelectionRange(pos, pos);
    }
    const help = doc.querySelector(".cs-budget-help");
    if (help) help.textContent = words(+digits);
  }, true);
  window.csFocusField = function (key) {
    const box = doc.querySelector(".st-key-" + key);
    if (!box) return;
    const el = key === "release" ? box.querySelector('[role="spinbutton"][data-type="day"]') : box.querySelector("input");
    (doc.getElementById("cs-field-error") || box).scrollIntoView({ behavior: "smooth", block: "center" });
    if (el) el.focus({ preventScroll: true });
  };
  window.csScrollToResult = function () {
    const r = doc.querySelector(".st-key-result");
    if (!r) return;
    const top = r.getBoundingClientRect().top;
    if (top < 0 || top > 80) r.scrollIntoView({ behavior: "smooth", block: "start" });
  };
})();
"""


def page_script(scroll_id: int, focus: str | None = None) -> None:
    """Adds PAGE_JS to the app page once; after a new prediction scrolls to the result, after a
    rejected Predict scrolls to the first invalid field and puts the cursor in it."""
    js = json.dumps(PAGE_JS)
    scroll = "setTimeout(() => p.csScrollToResult && p.csScrollToResult(), 350);" if scroll_id else ""
    if focus:
        st.session_state.focus_n = st.session_state.get("focus_n", 0) + 1
        scroll = f"setTimeout(() => p.csFocusField && p.csFocusField({json.dumps(focus)}), 350);"
    with st.container(key="cs_js"):
        st.iframe(f"""<script>/* {scroll_id} {st.session_state.get("focus_n", 0)} */
const p = window.parent, d = p.document;
if (!d.getElementById("cs-page-js")) {{
  const s = d.createElement("script"); s.id = "cs-page-js"; s.textContent = {js}; d.head.appendChild(s);
}}
{scroll}
</script>""", height=1)


# ---------------------------------------------------------------- masthead
m = OPTS["metrics"]
html(f"""
<div class="cs-filmstrip" aria-hidden="true"></div>
<header class="cs-masthead">
  <h1 class="cs-brand">CineScore</h1>
  <p class="cs-tagline">Forecast a film's IMDb rating before it opens.</p>
  <p class="cs-meta">{escape(re.sub(r'\s*\(.*?\)', '', OPTS['model']))} · tested on 4,118 unseen films · off by {m['MAE']:.2f} points on average</p>
</header>""")

# ---------------------------------------------------------------- form state, example film, clear
LANGS = [l["code"] for l in OPTS["languages"]]
COUNTRIES = sorted(OPTS["countries"], key=lambda c: COUNTRY_NAMES.get(c, c))
# Main inputs start empty (the user must choose them); minor ones keep a default shown in the label.
DEFAULTS = {"title": "", "release": None, "runtime": None, "budget": "", "cert": "Not rated yet",
            "lang": None, "country": None, "others": [], "companies": 2, "spoken": 1,
            "lead": "Not sure", "major": False,
            **{f"genres_{g}": False for g in OPTS["genres"]}, **{f"keywords_{k}": False for k in KEYWORDS}}
# "Try an example film" fills in this film (same as the Flask app); the user still presses Predict.
EXAMPLE = {"title": "Orbital", "release": date(2027, 6, 1), "runtime": 150, "budget": "150,000,000",
           "lang": "en", "country": "US", "major": True, "genres_drama": True, "genres_sci_fi": True}
EXAMPLE_PEOPLE = {"director": "Christopher Nolan", "cast": "Cillian Murphy"}
BUDGET_MSG = "Budget must be between $10,000 and $1,000,000,000 (or leave it empty)."


def clear_form() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v
    for role in ROLE_LIMITS:
        st.session_state[f"picks_{role}"] = []
        st.session_state[f"q_{role}"] = ""
        if isinstance(st.session_state.get(f"sb_{role}"), dict):
            st.session_state[f"sb_{role}"]["result"] = None
    st.session_state.pop("last", None)
    st.session_state.invalid = None


def fill_example() -> None:
    clear_form()
    st.session_state.update(EXAMPLE)
    for role, name in EXAMPLE_PEOPLE.items():
        hits = P.search(role, name)
        person = next((h for h in hits if h["name"] == name), hits[0] if hits else None)
        if person:
            st.session_state[f"picks_{role}"] = [person]


def check_inputs(release, runtime, genres, lang, country, budget) -> tuple[str, str] | None:
    """Same checks, same order and same wording as clientCheck() in the Flask app.
    Returns (field key, message) for the first problem, or None."""
    if release is None:
        return "release", "Please choose a release date."
    if runtime is None:
        return "runtime", "Please enter the runtime in minutes."
    if not 60 <= runtime <= 240:
        return "runtime", "Runtime must be between 60 and 240 minutes."
    if not genres:
        return "genres", "Pick at least one genre."
    if len(genres) > 3:
        return "genres", "Pick at most 3 genres."
    if not lang:
        return "lang", "Please choose the original language."
    if not country:
        return "country", "Please choose the main production country."
    if budget and (not budget.isdigit() or not 10_000 <= int(budget) <= 1_000_000_000):
        return "budget", BUDGET_MSG
    return None


def current_problem() -> tuple[str, str] | None:
    """check_inputs() on the values in session state, so it can run before the widgets are drawn."""
    s = st.session_state
    genres = [g for g in OPTS["genres"] if s.get(f"genres_{g}")]
    return check_inputs(s.get("release"), s.get("runtime"), genres, s.get("lang"), s.get("country"),
                        re.sub(r"[,\s$]", "", s.get("budget") or ""))


# server (/predict) messages -> field, so they are shown at the field too
SERVER_FIELDS = [("release date", "release"), ("runtime", "runtime"), ("genre", "genres"), ("budget", "budget"),
                 ("production compan", "companies"), ("spoken language", "spoken"), ("language", "lang"),
                 ("country", "country")]
FIELD_BOXES = {  # what gets the red border for each field
    "release": '[data-testid="stDateInputField"]', "runtime": '[data-baseweb="input"]',
    "budget": '[data-baseweb="input"]', "lang": '[data-baseweb="select"] > div',
    "country": '[data-baseweb="select"] > div', "companies": '[data-baseweb="input"]',
    "spoken": '[data-baseweb="input"]'}


def field_error(key: str, where=st) -> None:
    """Message under the field and a red border while this field is the invalid one."""
    bad = st.session_state.get("invalid")
    if not bad or bad[0] != key:
        return
    where.markdown(f'<p class="cs-field-error" id="cs-field-error">{escape(bad[1])}</p>', unsafe_allow_html=True)
    rule = (f".st-key-genres {{ margin-left: -15px; padding-left: 12px !important; border-left: 3px solid var(--cs-danger); }}"
            if key == "genres" else
            f".stApp .st-key-{key} {FIELD_BOXES[key]} {{ border-color: var(--cs-danger) !important;"
            f" box-shadow: 0 0 0 1px var(--cs-danger) !important; }}"
            f".stApp .st-key-{key} [data-testid=\"stWidgetLabel\"] p {{ color: var(--cs-danger) !important; }}")
    st.html(f"<style>{rule}</style>")


for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = list(_v) if isinstance(_v, list) else _v

# Field-level errors. Predict was clicked in this run -> mark the first problem field (and focus it);
# otherwise drop the mark as soon as that field is fixed (Streamlit sees a change on Enter or leaving the field).
focus_field = None
pre_problem = current_problem() if st.session_state.get("predict") else None
if st.session_state.get("predict"):
    st.session_state.invalid = pre_problem
    focus_field = pre_problem[0] if pre_problem else None
elif st.session_state.get("invalid"):
    now, field = current_problem(), st.session_state.invalid[0]
    if field in ("companies", "spoken"):  # server-side checks: clear once the value changes
        if st.session_state.get(field) != st.session_state.get("invalid_value"):
            st.session_state.invalid = None
    elif now and now[0] == field:
        st.session_state.invalid = now   # still wrong: keep the mark, update the message
    else:
        st.session_state.invalid = None
if st.session_state.pop("focus_next", None):  # set when /predict itself rejected a field
    focus_field = st.session_state.invalid[0] if st.session_state.get("invalid") else None

left, right = st.columns([736, 380], gap="small")  # same widths as the Flask grid

with left:
    with st.container(key="formbar", horizontal=True):
        html('<p class="cs-formbar__text">Fill the form with a sample film, or start again.</p>')
        with st.container(key="formbar_btns", horizontal=True):
            st.button("Try an example film", key="example", on_click=fill_example)
            st.button("Clear form", key="clear", on_click=clear_form)

    with st.container(key="sec_film"):
        section("Film details", "What the film is, how long it runs and what it costs.")
        title = st.text_input("Working title *optional*", max_chars=200, key="title",
                              placeholder="e.g. Untitled space drama")
        c1, c2 = st.columns(2)
        release = c1.date_input("Release date", format="DD/MM/YYYY", key="release",
                                min_value=date(1980, 1, 1), max_value=date(2035, 12, 31))
        runtime = c2.number_input("Runtime *minutes*", min_value=60, max_value=240, step=5, key="runtime",
                                  placeholder="e.g. 110")
        field_error("release", c1)
        field_error("runtime", c2)
        c1, c2 = st.columns(2)
        budget_raw = c1.text_input("Budget *US$, optional*", key="budget", placeholder="e.g. 40,000,000")
        budget = re.sub(r"[,\s$]", "", budget_raw or "")  # plain number for /predict, as in Flask
        c1.markdown(f'<p class="cs-help cs-budget-help">{budget_words(budget)}</p>', unsafe_allow_html=True)
        field_error("budget", c1)
        cert = c2.selectbox("US certification", ["Not rated yet"] + CERTIFICATIONS, key="cert")
        genres = checkbox_grid("genres", "Genres <em>pick 1 to 3</em>", OPTS["genres"], pretty_genre)
        field_error("genres")
        keywords = checkbox_grid("keywords", "Story elements <em>optional</em>", list(KEYWORDS), KEYWORDS.get)

    with st.container(key="sec_people"):
        section("People", "Search a name and pick it. The average rating of their earlier films "
                          "feeds the model. Leave a role empty for a debut.")
        c1, c2 = st.columns(2)
        with c1:
            directors = people_picker("director", "Director <em>up to 3</em>")
        with c2:
            writers = people_picker("writer", "Writers <em>up to 5</em>")
        c1, c2 = st.columns(2)
        with c1:
            cast = people_picker("cast", "Lead cast <em>up to 3</em>")
        with c2:
            franchise = people_picker("franchise", "Franchise <em>if a sequel</em>")
        lead = st.segmented_control("Lead role *default: Not sure*", ["Not sure", "Male", "Female"], key="lead")

    with st.container(key="sec_prod"):
        section("Production", "Where it is made, in which language, and by whom.")
        c1, c2 = st.columns(2)
        lang = c1.selectbox("Original language", LANGS, key="lang", placeholder="Choose a language…",
                            format_func=lambda c: LANGUAGE_NAMES.get(c, c))
        main_country = c2.selectbox("Main production country", COUNTRIES, key="country",
                                    placeholder="Choose a country…", format_func=lambda c: COUNTRY_NAMES.get(c, c))
        field_error("lang", c1)
        field_error("country", c2)
        st.session_state.others = [c for c in st.session_state.others if c != main_country]
        other_countries = st.multiselect("Other countries *optional*", [c for c in COUNTRIES if c != main_country],
                                         key="others", format_func=lambda c: COUNTRY_NAMES.get(c, c),
                                         placeholder="Add countries")
        c1, c2 = st.columns(2)
        n_companies = c1.number_input("Production companies *default: 2*", min_value=1, max_value=30, key="companies")
        n_languages = c2.number_input("Spoken languages *default: 1*", min_value=1, max_value=12, key="spoken")
        field_error("companies", c1)
        field_error("spoken", c2)
        major = st.checkbox("A major studio is involved  \n*Disney, Warner, Universal, Sony, Paramount…*",
                            key="major")

    with st.container(key="actions"):
        error_slot = st.empty()
        go = st.button("Predict rating", type="primary", key="predict")

payload = {
    "title": title, "release_date": release.isoformat() if release else None, "runtime": runtime,
    "genres": genres or [], "budget_usd": budget or None,
    "us_certification": None if cert == "Not rated yet" else cert,
    "directors": directors, "writers": writers, "cast": cast,
    "franchise": franchise[0] if franchise else None,
    "original_language": lang, "production_countries": ([main_country] if main_country else []) + other_countries,
    "n_production_companies": n_companies, "n_spoken_languages": n_languages,
    "female_lead": {"Male": "no", "Female": "yes"}.get(lead, "unknown"),
    "major_studio": major, "keywords": keywords or [],
}


def show_error(title: str, msg: str) -> None:
    error_slot.markdown(f'<div class="cs-alert" role="alert"><strong>{title}</strong>'
                        f'<span>{escape(msg)}</span></div>', unsafe_allow_html=True)


scroll_id = 0
with right:
    with st.container(key="result"):
        if go:
            if pre_problem:
                show_error("Please check this input", pre_problem[1])
            else:
                try:
                    with st.spinner("Running the model…"):
                        st.session_state.last = (payload, P.predict(payload))
                    st.session_state.n_pred = st.session_state.get("n_pred", 0) + 1
                    scroll_id = st.session_state.n_pred
                except ValidationError as e:
                    show_error("Please check this input", str(e))
                    field = next((f for text, f in SERVER_FIELDS if text in str(e).lower()), None)
                    if field:  # draw the message at the field too, then focus it
                        st.session_state.invalid = (field, str(e))
                        st.session_state.invalid_value = st.session_state.get(field)
                        st.session_state.focus_next = True
                        st.rerun()
                except Exception:
                    show_error("Something went wrong", "Something went wrong while predicting. Please try again.")
        last = st.session_state.get("last")
        if last:
            render_result(last[1], last[0]["genres"], stale=last[0] != payload)
        else:
            render_empty()

html("""
<footer class="cs-footer">
  <p>Information courtesy of IMDb (https://www.imdb.com). Used with permission, non-commercial.
    This product uses the TMDB API but is not endorsed or certified by TMDB.</p>
  <p>CineScore is a CT-378 Machine Learning course project. Forecasts are estimates, not guarantees.</p>
</footer>""")
page_script(scroll_id, focus_field)
