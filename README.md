# CineScore: Predicting a Movie's IMDb Rating Before Release

CT-378 Machine Learning semester project. Regression task: predict a film's IMDb average rating
(1-10) from information available **before release** (genre, runtime, budget, cast/crew track
record, certification, studio, language, release timing). Intended users: producers, distributors
and streaming acquisition teams screening projects.

**Metrics:** MAE (primary), RMSE, R², reported once on a held-out, time-ordered test set.

## Data

| Dataset | Source | Collection method | Licence / terms | Accessed |
|---|---|---|---|---|
| IMDb Non-Commercial Datasets (`title.basics`, `title.ratings`, `title.crew`) | https://datasets.imdbws.com/ | Official bulk download (`scripts/01_download_imdb.py`) | Personal & non-commercial use only; not redistributed in this repo | see `data/raw/IMDB_DOWNLOAD_DATE.txt` |
| TMDB API v3 (`/movie/{imdb_id}` + credits, keywords, release_dates) | https://developer.themoviedb.org/ | Official API (`scripts/03_fetch_tmdb.py`) | TMDB API Terms of Use, non-commercial with attribution | 2026-11 |

No website scraping is used. Post-release fields (`numVotes`, revenue, popularity, TMDB votes) are
**never** used as features; `numVotes >= 2000` is used only to select films with a stable rating.
Personal identifiers: only public cast/crew IDs are used, to compute aggregate track-record features.

## Setup

```bash
pip install -r requirements.txt
python scripts/01_download_imdb.py                     # ~310 MB
python scripts/02_build_base.py
TMDB_TOKEN=<your v4 read token> python scripts/03_fetch_tmdb.py   # ~27.6k API calls, resumable
```

## Repository layout

```
data/       raw + processed data (git-ignored; created by scripts)
scripts/    data collection and feature building
notebooks/  EDA, modelling, evaluation
models/     saved joblib pipeline
app/        Streamlit / Gradio / Flask apps
reports/    figures and final report
```

## Deployment links

- Streamlit Community Cloud: _TBD_
- Hugging Face Space: _TBD_
- Local Flask app: `python app/flask_app.py` → http://localhost:5000

## Credits

- Information courtesy of IMDb (https://www.imdb.com). Used with permission (non-commercial).
- This product uses the TMDB API but is not endorsed or certified by TMDB.
