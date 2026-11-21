---
title: CineScore
colorFrom: gray
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: Forecast a film's IMDb rating before it opens
models:
  - {user}/cinescore
---

# CineScore

Describe a planned film (release date, runtime, genres, budget, people, production) and get the
predicted IMDb rating, an 80% range and the inputs that drive the forecast.

- Model and model card: [{user}/cinescore](https://huggingface.co/{user}/cinescore)
- Code: <https://github.com/OwaisKamran/CineScore>

This Space runs the project's Streamlit app (`streamlit_app/`) entirely in your browser with
[stlite](https://github.com/whitphx/stlite) (Streamlit on Pyodide), so it needs no server. It uses the
same saved scikit-learn pipeline and the same `predictor.py` as the project's Flask app and the Streamlit
Community Cloud app, so all three give identical predictions. The first load downloads Python and the
model (about 20 to 40 seconds); after that everything runs locally in the browser.

Information courtesy of IMDb (https://www.imdb.com). Used with permission, non-commercial.
This product uses the TMDB API but is not endorsed or certified by TMDB.
