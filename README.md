# CineScore: Predicting a Movie's IMDb Rating Before Release

CT-378 Machine Learning semester project. Regression task: predict a film's IMDb average rating
(1-10) from information available **before release** (genre, runtime, budget, cast/crew track
record, certification, studio, language, release timing). Intended users: producers, distributors
and streaming acquisition teams screening projects.

**Metrics:** MAE (primary), RMSE, R², reported once on a held-out, time-ordered test set.

| Test set (4,118 films, 2022-2026) | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (training mean) | 0.886 | 1.136 | -0.01 |
| Gradient Boosting (HGB), final model | **0.644** | **0.872** | **0.40** |

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
python scripts/04_build_features.py
# notebooks/01_eda.ipynb, notebooks/02_modelling.ipynb  -> models/cinescore_pipeline.joblib
TMDB_TOKEN=<token> python scripts/05_build_app_lookups.py         # app/data/lookups.json.gz
copy models\cinescore_pipeline.joblib models\metadata.json app\model\   # (cp on Linux/macOS)
```

## Run the web app locally (Flask)

```bash
pip install -r app/requirements.txt
python app/index.py        # open http://localhost:5000
```

`POST /predict` takes a JSON description of the film and returns the predicted rating, an 80%
interval and the per-input effects; invalid input returns HTTP 400 with a readable message.

## Deployment

The `app/` folder is self-contained (model, lookups, `requirements.txt`). Planned deployments,
from the options in the course guidelines: Streamlit Community Cloud and a Hugging Face Space
(with a model card), plus the local Flask app.

## Repository layout

```
data/       raw + processed data (git-ignored; created by scripts)
scripts/    data collection and feature building
notebooks/  EDA, modelling, evaluation
models/     saved joblib pipeline
app/        Flask app (index.py, predictor.py, templates/, static/, model/, data/)
reports/    figures and final report
```

## Deployment links

- Local Flask app: `python app/index.py` → http://localhost:5000
- Streamlit Community Cloud: _TBD_
- Hugging Face Space: _TBD_

## Credits

- Information courtesy of IMDb (https://www.imdb.com). Used with permission (non-commercial).
- This product uses the TMDB API but is not endorsed or certified by TMDB.
