---
library_name: sklearn
pipeline_tag: tabular-regression
license: other
license_name: imdb-non-commercial
license_link: https://developer.imdb.com/non-commercial-datasets/
language:
  - en
tags:
  - sklearn
  - tabular-regression
  - movies
  - imdb
  - course-project
metrics:
  - mae
  - rmse
  - r_squared
---

# CineScore: predicting a film's IMDb rating before release

A scikit-learn pipeline that forecasts the IMDb user rating (1 to 10) of a feature film from
information that is known **before it opens**: release date, runtime, genres, budget, certification,
language and countries, production companies, story keywords, and the earlier track record of the
director, writers, lead cast and franchise.

- **Model:** `HistGradientBoostingRegressor` inside a `Pipeline` (imputation, one-hot encoding and
  scaling are fitted on the training years only)
- **Live demo:** [Space: {user}/cinescore](https://huggingface.co/spaces/{user}/cinescore)
- **Code, data scripts and notebooks:** <https://github.com/OwaisKamran/CineScore>
- CT-378 Machine Learning course project.

## Files

| File | What it is |
|---|---|
| `cinescore_pipeline.joblib` | Fitted scikit-learn pipeline (preprocessing + model) |
| `metadata.json` | Feature list, best hyperparameters, 80% interval offsets, validation and test metrics, library versions |

## How to use

The pipeline needs **scikit-learn 1.9.1** exactly (joblib files are tied to the version that saved them).

```python
import json, joblib, pandas as pd
from huggingface_hub import hf_hub_download

repo = "{user}/cinescore"
pipe = joblib.load(hf_hub_download(repo, "cinescore_pipeline.joblib"))
meta = json.load(open(hf_hub_download(repo, "metadata.json")))

# one row with every column in meta["features_numeric"] + meta["features_categorical"]
X = pd.DataFrame([row])[meta["features_numeric"] + meta["features_categorical"]]
pred = pipe.predict(X)[0]
lo, hi = pred + meta["interval_80"][0], pred + meta["interval_80"][1]
```

Building the 57 features by hand is tedious; the GitHub repo's `app/predictor.py` turns a plain film
description (title, date, genres, names, ...) into the feature row, validates it, and also explains the
prediction. The Space runs exactly that code.

## Training data

- **IMDb Non-Commercial Datasets** (title.basics, title.ratings, title.crew, name.basics), downloaded
  2026-11-20. Used for the target (averageRating), runtime, genres, year and people.
- **TMDB API v3** (one call per film: details, credits, keywords, release dates). Used for budget,
  certification, language, countries, companies, keywords and cast.
- Films released 1980-2025, runtime 60-240 minutes, not adult, with at least 2,000 IMDb votes:
  **27,451 films, 57 features** (54 numeric/binary, 3 categorical).
- No scraping. Raw files are not redistributed; the repo contains download scripts.

**Leakage controls.** Vote counts, revenue, popularity and TMDB ratings are never used as features
(the vote count is only a row filter). Track-record features use films from strictly earlier years.

## Evaluation

Time-ordered split, so the model is always tested on films newer than the ones it learned from:
train 70% (to Feb 2018), validation 15% (Feb 2018 to Feb 2022), test 15% (Feb 2022 onwards, 4,118 films,
used once). Hyperparameters chosen with 5-fold cross-validation on the training set; the final model
was refitted on train + validation (23,333 films) before the single test evaluation.

| Model | CV MAE | Validation MAE | Validation R² |
|---|---|---|---|
| Baseline (mean rating) | - | 0.919 | -0.03 |
| Ridge | 0.583 | 0.647 | 0.46 |
| KNN | 0.608 | 0.666 | 0.41 |
| Random Forest | 0.562 | 0.630 | 0.47 |
| **HistGradientBoosting (this model)** | **0.547** | **0.621** | **0.50** |

**Test set (newest 4,118 films):** MAE **0.644**, RMSE 0.872, R² 0.404, against a baseline MAE of 0.886.
The 80% prediction interval (prediction -0.96 / +0.96, from validation residuals) covered 78.7% of
test films.

## Limitations

- Ratings regress toward the middle: errors are about 0.5 for films rated 5-7 but about 1.5 for films
  below 4 or above 8.
- Debut directors are harder to predict (MAE 0.77 vs 0.60 for experienced directors).
- It cannot foresee hype, controversy or review-bombing; these cause the largest misses.
- Average IMDb ratings drift down over time, so forecasts for far-future dates are less certain.
- Only films that reach 2,000 IMDb votes were used for training; small releases are out of scope.

## Intended use

Education and exploration (a course project): estimating how a planned film might be received and
which inputs push the estimate up or down. Not for commercial decisions. Forecasts are estimates, not
guarantees.

## Licence and credits

Information courtesy of IMDb (https://www.imdb.com). Used with permission, non-commercial. Because the
model was trained on IMDb's non-commercial datasets it may only be used non-commercially.
This product uses the TMDB API but is not endorsed or certified by TMDB.
