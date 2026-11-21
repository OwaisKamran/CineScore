---
title: CineScore
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
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

This Space runs the project's Streamlit app (`streamlit_app/`) in Docker, because Hugging Face now
serves Streamlit apps through the Docker SDK. It uses the same saved pipeline and the same `predictor.py`
as the project's Flask app, so both give identical predictions.

Information courtesy of IMDb (https://www.imdb.com). Used with permission, non-commercial.
This product uses the TMDB API but is not endorsed or certified by TMDB.
