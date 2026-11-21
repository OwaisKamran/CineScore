"""CineScore - Streamlit front end.

Local:  streamlit run streamlit_app/streamlit_app.py
Cloud:  Streamlit Community Cloud, main file path = streamlit_app/streamlit_app.py

Reuses the saved scikit-learn pipeline (app/model/) and the name lookups (app/data/) through the
same Predictor class as the Flask app, so both front ends validate and predict identically.
"""
import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from predictor import CERTIFICATIONS, KEYWORDS, LANGUAGE_NAMES, ROLE_LIMITS, Predictor, ValidationError  # noqa: E402

st.set_page_config(page_title="CineScore", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_predictor() -> Predictor:
    return Predictor()


P = load_predictor()
OPTS = P.options()
GENRE_LABEL = {g: g.replace("_", "-").title() for g in OPTS["genres"]}


# ---------------------------------------------------------------- people search widget
def _label(r: dict) -> str:
    if r["prev_count"]:
        return f"{r['name']} ({r['prev_count']} earlier films, avg {r['prev_mean']:.1f})"
    return f"{r['name']} (no earlier films)"


def people_picker(role: str, title: str, help_text: str) -> list[str]:
    """Search box + multiselect. Keeps chosen people selected while the user searches again."""
    labels = st.session_state.setdefault(f"labels_{role}", {})
    chosen = st.session_state.get(f"ms_{role}", [])
    q = st.text_input(f"Search {title.lower()}", key=f"q_{role}",
                      placeholder="Type at least 2 letters, then pick below")
    results = P.search(role, q) if q else []
    for r in results:
        labels[r["id"]] = _label(r)
    options = list(dict.fromkeys(chosen + [r["id"] for r in results]))
    limit = ROLE_LIMITS[role]
    if q and len(q.strip()) >= 2 and not results:
        st.caption("No matches. Check the spelling, or leave empty for a debut.")
    return st.multiselect(title, options, key=f"ms_{role}", format_func=lambda i: labels.get(i, i),
                          max_selections=limit, help=help_text,
                          placeholder=f"Pick up to {limit}" if limit > 1 else "Pick one")


# ---------------------------------------------------------------- header
st.title("🎬 CineScore")
st.markdown("Predict a movie's **IMDb rating before release** from information known in advance: "
            "genre, runtime, budget, the track record of the director, writers and cast, and more.")
m = OPTS["metrics"]
st.caption(f"Model: {OPTS['model']} · held-out test set (4,118 films, 2022-2026): "
           f"MAE {m['MAE']:.2f}, RMSE {m['RMSE']:.2f}, R² {m['R2']:.2f}")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("1. The film")
    c1, c2 = st.columns(2)
    title = c1.text_input("Working title (optional)", max_chars=200)
    release = c2.date_input("Release date", value=date(2027, 6, 1),
                            min_value=date(1980, 1, 1), max_value=date(2035, 12, 31))
    c1, c2, c3 = st.columns(3)
    runtime = c1.number_input("Runtime (minutes)", 60, 240, 110, step=5)
    budget_m = c2.number_input("Budget (US$ millions, 0 = unknown)", 0.0, 1000.0, 0.0, step=5.0,
                               help="Leave at 0 if the budget is not public.")
    cert = c3.selectbox("US certification", ["Not rated yet"] + CERTIFICATIONS)
    genres = st.multiselect("Genres (1-3)", OPTS["genres"], format_func=GENRE_LABEL.get,
                            max_selections=3, placeholder="Pick 1 to 3 genres")

    st.subheader("2. People")
    st.caption("Track record = average IMDb rating of their earlier films. "
               "Leave a role empty if unknown or a debut.")
    tabs = st.tabs(["Director", "Writers", "Cast", "Franchise"])
    with tabs[0]:
        directors = people_picker("director", "Director(s)", "Up to 3 directors.")
    with tabs[1]:
        writers = people_picker("writer", "Writer(s)", "Up to 5 writers.")
    with tabs[2]:
        cast = people_picker("cast", "Lead cast", "Up to 3 lead actors.")
    with tabs[3]:
        franchise = people_picker("franchise", "Franchise / collection",
                                  "Only if the film belongs to an existing series.")

    st.subheader("3. Production")
    c1, c2 = st.columns(2)
    langs = [l["code"] for l in OPTS["languages"]]
    lang = c1.selectbox("Original language", langs, index=langs.index("en"),
                        format_func=lambda c: LANGUAGE_NAMES.get(c, c))
    countries = OPTS["countries"]
    main_country = c2.selectbox("Main production country", countries, index=countries.index("US"))
    other_countries = st.multiselect("Other production countries (optional)",
                                     [c for c in countries if c != main_country])
    c1, c2, c3 = st.columns(3)
    n_companies = c1.number_input("Production companies", 1, 30, 2)
    n_languages = c2.number_input("Spoken languages", 1, 12, 1)
    lead = c3.radio("Lead role", ["Not sure", "Male", "Female"], horizontal=True)
    major = st.checkbox("A major studio is involved (Disney, Warner, Universal, Sony, Paramount...)")
    keywords = st.multiselect("Story elements (optional)", list(KEYWORDS), format_func=KEYWORDS.get)

    go = st.button("Predict rating", type="primary", width="stretch")

# ---------------------------------------------------------------- prediction
with right:
    st.subheader("Prediction")
    if not go:
        st.info("Fill in the film on the left and press **Predict rating**.")
    else:
        payload = {
            "title": title, "release_date": release.isoformat(), "runtime": runtime,
            "genres": genres, "budget_usd": budget_m * 1e6 if budget_m else None,
            "us_certification": None if cert == "Not rated yet" else cert,
            "directors": directors, "writers": writers, "cast": cast,
            "franchise": franchise[0] if franchise else None,
            "original_language": lang, "production_countries": [main_country] + other_countries,
            "n_production_companies": n_companies, "n_spoken_languages": n_languages,
            "female_lead": {"Male": "no", "Female": "yes"}.get(lead, "unknown"),
            "major_studio": major, "keywords": keywords,
        }
        try:
            res = P.predict(payload)
        except ValidationError as e:
            st.error(f"Please fix this input: {e}")
            st.stop()
        except Exception:
            st.error("Something went wrong while predicting. Please try again.")
            st.stop()

        lo, hi = res["interval_80"]
        name = res["title"] or "This film"
        st.metric(f"{name}: predicted IMDb rating", f"{res['prediction']:.1f} / 10")
        st.markdown(f"**80% range: {lo:.1f} to {hi:.1f}.** Comparable films land in this range "
                    "about 80% of the time (79% on the test set).")
        band = pd.DataFrame({"lo": [lo], "hi": [hi], "pred": [res["prediction"]]})
        base = alt.Chart(band).encode(y=alt.value(12))
        st.altair_chart(
            (base.mark_rule(strokeWidth=10, opacity=0.35, color="#4C72B0").encode(
                x=alt.X("lo:Q", scale=alt.Scale(domain=[1, 10]), title="IMDb rating"), x2="hi:Q")
             + base.mark_point(size=180, filled=True, color="#4C72B0").encode(x="pred:Q"))
            .properties(height=60), width="stretch")

        st.markdown("##### What drives this prediction")
        st.caption("How far each group of inputs moves the rating compared with a neutral film "
                   "(unknown people, no genre, 103 min, no budget, English, US).")
        d = pd.DataFrame(res["drivers"])
        d["direction"] = d.effect.map(lambda v: "raises" if v > 0 else "lowers" if v < 0 else "none")
        st.altair_chart(
            alt.Chart(d).mark_bar().encode(
                x=alt.X("effect:Q", title="Effect on predicted rating (points)"),
                y=alt.Y("group:N", sort=None, title=None),
                color=alt.Color("direction:N", legend=None, scale=alt.Scale(
                    domain=["raises", "lowers", "none"], range=["#55A868", "#C44E52", "#999999"])),
                tooltip=["group", alt.Tooltip("effect:Q", format="+.2f")])
            .properties(height=280), width="stretch")

        tr = res["track_record"]
        rows = [(k, v) for k, v in [("Director", tr["director_prev_mean"]),
                                    ("Writers", tr["writer_prev_mean"]),
                                    ("Cast", tr["cast_prev_mean"]),
                                    ("Franchise", tr["franchise_prev_mean"])] if v is not None]
        if rows:
            st.markdown("##### Track records used")
            st.dataframe(pd.DataFrame(rows, columns=["Role", "Avg rating of earlier films"]),
                         hide_index=True, width="stretch")
        st.caption("Limitations: the model pulls extreme films towards the average (it is least "
                   "accurate below 4 and above 8) and cannot foresee hype or review-bombing.")

st.divider()
st.caption("Information courtesy of IMDb (https://www.imdb.com). Used with permission. "
           "This product uses the TMDB API but is not endorsed or certified by TMDB. "
           "Course project (CT-378); predictions are estimates, not guarantees.")
