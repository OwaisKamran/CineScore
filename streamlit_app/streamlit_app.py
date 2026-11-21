"""CineScore - Streamlit front end.

Local:  streamlit run streamlit_app/streamlit_app.py      (run from the repo root so that
        .streamlit/config.toml, the theme, is picked up)
Cloud:  Streamlit Community Cloud, main file path = streamlit_app/streamlit_app.py

Reuses the saved scikit-learn pipeline (app/model/) and the name lookups (app/data/) through the
same Predictor class as the Flask app, so both front ends validate and predict identically.
Colours and fonts come from app/static/tokens.css, shared with the Flask app.
"""
import json
import sys
from datetime import date
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from predictor import CERTIFICATIONS, KEYWORDS, LANGUAGE_NAMES, ROLE_LIMITS, Predictor, ValidationError  # noqa: E402

st.set_page_config(page_title="CineScore", page_icon=":material/movie:", layout="wide",
                   initial_sidebar_state="collapsed", menu_items={})


@st.cache_resource(show_spinner="Loading the model…")
def load_predictor() -> Predictor:
    return Predictor()


def load_assets() -> tuple[str, dict]:  # small files; read each run so CSS edits apply live
    css = (ROOT / "app" / "static" / "tokens.css").read_text(encoding="utf-8")
    css += (Path(__file__).parent / "streamlit.css").read_text(encoding="utf-8")
    names = json.loads((ROOT / "app" / "data" / "country_names.json").read_text(encoding="utf-8"))
    return css, names


P = load_predictor()
OPTS = P.options()
CSS, COUNTRY_NAMES = load_assets()
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


def verdict(v: float) -> str:
    return "Exceptional" if v >= 8 else "Strong" if v >= 7 else "Solid" if v >= 6 else "Mixed" if v >= 5 else "Weak"


# ---------------------------------------------------------------- people search widget
def _label(r: dict) -> str:
    if r["prev_count"]:
        return f"{r['name']} · {r['prev_count']} earlier films · avg {r['prev_mean']:.1f}"
    return f"{r['name']} · no earlier films"


def people_picker(role: str, title: str) -> list[str]:
    """Search box + multiselect. Chosen people stay selected while the user searches again."""
    labels = st.session_state.setdefault(f"labels_{role}", {})
    chosen = st.session_state.get(f"ms_{role}", [])
    q = st.text_input(title, key=f"q_{role}", placeholder="Type at least 2 letters, then pick below")
    results = P.search(role, q) if q else []
    for r in results:
        labels[r["id"]] = _label(r)
    options = list(dict.fromkeys(chosen + [r["id"] for r in results]))
    limit = ROLE_LIMITS[role]
    picked = st.multiselect(title, options, key=f"ms_{role}", format_func=lambda i: labels.get(i, i),
                            max_selections=limit, label_visibility="collapsed",
                            placeholder=f"Pick up to {limit}" if limit > 1 else "Pick one")
    if q and len(q.strip()) >= 2 and not results:
        html('<p class="cs-hint">No matches. Check the spelling, or leave this empty for a debut.</p>')
    return picked


# ---------------------------------------------------------------- result rendering
def render_result(res: dict, stale: bool) -> None:
    lo, hi = res["interval_80"]
    pred = res["prediction"]
    pct = lambda v: (v - 1) / 9 * 100  # noqa: E731
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

    tr = res["track_record"]
    recs = [(k, tr[f]) for k, f in [("Director", "director_prev_mean"), ("Writers", "writer_prev_mean"),
                                    ("Cast", "cast_prev_mean"), ("Franchise", "franchise_prev_mean")]
            if tr[f] is not None]
    records = ("<div class='cs-block'><h3>Track records used</h3><dl class='cs-records'>"
               + "".join(f"<dt>{k}</dt><dd>{v:.1f} avg</dd>" for k, v in recs) + "</dl></div>") if recs else ""

    html(f"""
<div class="cs-result {'is-stale' if stale else ''}">
  {stale_note}
  <p class="cs-eyebrow">Predicted IMDb rating{title}</p>
  <div class="cs-score"><span class="cs-score__num">{pred:.1f}</span><span class="cs-score__of">/ 10</span>
    <span class="cs-verdict">{verdict(pred)}</span></div>
  <div class="cs-range">
    <div class="cs-range__track"><span class="cs-range__band" style="left:{pct(lo):.2f}%;width:{(hi - lo) / 9 * 100:.2f}%"></span>
      <span class="cs-range__mark" style="left:{pct(pred):.2f}%"></span></div>
    <div class="cs-range__ticks"><span>1</span><span>4</span><span>7</span><span>10</span></div>
    <p class="cs-range__text"><strong>80% range: {lo:.1f} to {hi:.1f}</strong>
      Comparable films land here about 4 times in 5.</p>
  </div>
  <div class="cs-block">
    <h3>What drives this forecast</h3>
    <p class="cs-desc">Points each input adds or removes, compared with a neutral film (unknown people,
      no genre, 103 minutes, no budget, English, US).</p>
    <div class="cs-drivers">{drivers}</div>
    {idle_note}
  </div>
  {records}
  <p class="cs-note">The model is least accurate for very weak or exceptional films and cannot foresee
    hype or review-bombing.</p>
</div>""")


def render_empty() -> None:
    html("""
<div class="cs-result">
  <p class="cs-eyebrow">Your forecast</p>
  <p class="cs-empty__num">—<span>/ 10</span></p>
  <p class="cs-empty__text">Describe the film on the left and press <strong>Predict rating</strong>.
    You will see the expected rating, a likely range and what drives it.</p>
  <ol class="cs-steps">
    <li>Pick a release date, runtime and up to three genres</li>
    <li>Add the director and lead cast if they are known</li>
    <li>Press Predict rating</li>
  </ol>
</div>""")


# ---------------------------------------------------------------- masthead
m = OPTS["metrics"]
html(f"""
<header class="cs-masthead">
  <h1 class="cs-brand">CineScore</h1>
  <p class="cs-tagline">Forecast a film's IMDb rating before it opens.</p>
  <p class="cs-meta">{escape(OPTS['model'])} · tested on 4,118 unseen films · average error ±{m['MAE']:.2f}</p>
</header>""")

left, right = st.columns([736, 380], gap="small")  # same widths as the Flask grid

with left:
    with st.container(key="sec_film"):
        section("Film details", "What the film is, how long it runs and what it costs.")
        title = st.text_input("Working title *optional*", max_chars=200, key="title",
                              placeholder="e.g. Untitled space drama")
        c1, c2 = st.columns(2)
        release = c1.date_input("Release date", value=date(2027, 6, 1), format="DD/MM/YYYY", key="release",
                                min_value=date(1980, 1, 1), max_value=date(2035, 12, 31))
        runtime = c2.number_input("Runtime *minutes*", 60, 240, 110, step=5, key="runtime")
        c1, c2 = st.columns(2)
        budget_m = c1.number_input("Budget *US$ millions, 0 if unknown*", 0.0, 1000.0, 0.0, step=5.0, key="budget")
        cert = c2.selectbox("US certification", ["Not rated yet"] + CERTIFICATIONS, key="cert")
        genres = checkbox_grid("genres", "Genres <em>pick 1 to 3</em>", OPTS["genres"], pretty_genre)
        keywords = checkbox_grid("keywords", "Story elements <em>optional</em>", list(KEYWORDS), KEYWORDS.get)

    with st.container(key="sec_people"):
        section("People", "Search a name and pick it. The average rating of their earlier films "
                          "feeds the model. Leave a role empty for a debut.")
        c1, c2 = st.columns(2)
        with c1:
            directors = people_picker("director", "Director *up to 3*")
        with c2:
            writers = people_picker("writer", "Writers *up to 5*")
        c1, c2 = st.columns(2)
        with c1:
            cast = people_picker("cast", "Lead cast *up to 3*")
        with c2:
            franchise = people_picker("franchise", "Franchise *if a sequel*")
        lead = st.segmented_control("Lead role", ["Not sure", "Male", "Female"], default="Not sure",
                                    key="lead")

    with st.container(key="sec_prod"):
        section("Production", "Where it is made, in which language, and by whom.")
        c1, c2 = st.columns(2)
        langs = [l["code"] for l in OPTS["languages"]]
        lang = c1.selectbox("Original language", langs, index=langs.index("en"),
                            format_func=lambda c: LANGUAGE_NAMES.get(c, c))
        countries = sorted(OPTS["countries"], key=lambda c: COUNTRY_NAMES.get(c, c))
        main_country = c2.selectbox("Main production country", countries, index=countries.index("US"),
                                    format_func=lambda c: COUNTRY_NAMES.get(c, c))
        other_countries = st.multiselect("Other countries *optional*",
                                         [c for c in countries if c != main_country],
                                         format_func=lambda c: COUNTRY_NAMES.get(c, c),
                                         placeholder="Add countries")
        c1, c2 = st.columns(2)
        n_companies = c1.number_input("Production companies", 1, 30, 2)
        n_languages = c2.number_input("Spoken languages", 1, 12, 1)
        major = st.checkbox("A major studio is involved  \n*Disney, Warner, Universal, Sony, Paramount…*",
                            key="major")

    with st.container(key="actions"):
        error_slot = st.empty()
        go = st.button("Predict rating", type="primary", key="predict")

payload = {
    "title": title, "release_date": release.isoformat(), "runtime": runtime,
    "genres": genres or [], "budget_usd": budget_m * 1e6 if budget_m else None,
    "us_certification": None if cert == "Not rated yet" else cert,
    "directors": directors, "writers": writers, "cast": cast,
    "franchise": franchise[0] if franchise else None,
    "original_language": lang, "production_countries": [main_country] + other_countries,
    "n_production_companies": n_companies, "n_spoken_languages": n_languages,
    "female_lead": {"Male": "no", "Female": "yes"}.get(lead, "unknown"),
    "major_studio": major, "keywords": keywords or [],
}

with right:
    with st.container(key="result"):
        if go:
            try:
                with st.spinner("Running the model…"):
                    st.session_state.last = (payload, P.predict(payload))
            except ValidationError as e:
                error_slot.markdown('<div class="cs-alert" role="alert"><strong>Please check this input</strong>'
                                    f'<span>{escape(str(e))}</span></div>', unsafe_allow_html=True)
            except Exception:
                error_slot.markdown('<div class="cs-alert" role="alert"><strong>Something went wrong</strong>'
                                    '<span>Something went wrong while predicting. Please try again.</span></div>',
                                    unsafe_allow_html=True)
        last = st.session_state.get("last")
        if last:
            render_result(last[1], stale=last[0] != payload)
        else:
            render_empty()

html("""
<footer class="cs-footer">
  <p>Information courtesy of IMDb (https://www.imdb.com). Used with permission, non-commercial.
    This product uses the TMDB API but is not endorsed or certified by TMDB.</p>
  <p>CineScore is a CT-378 Machine Learning course project. Forecasts are estimates, not guarantees.</p>
</footer>""")
