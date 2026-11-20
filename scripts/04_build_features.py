"""Join IMDb + TMDB data and build pre-release features.

Output: data/processed/features.parquet  (one row per film, target = averageRating)

Leakage rules
-------------
* Never used as features: numVotes, revenue, popularity, TMDB vote_average/vote_count.
* Track-record features (director / writer / cast / franchise) use ONLY films released in a
  strictly earlier year than the target film, so a film's own rating never enters its features.
* Imputation, encoding and scaling are NOT done here - they live in the scikit-learn Pipeline
  and are fitted on the training split only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TMDB = ROOT / "data" / "raw" / "tmdb_movies.jsonl"

MAJOR_STUDIOS = {
    "Warner Bros. Pictures", "Universal Pictures", "Paramount Pictures", "Columbia Pictures",
    "20th Century Fox", "Walt Disney Pictures", "New Line Cinema", "Metro-Goldwyn-Mayer",
    "Lionsgate", "DreamWorks Pictures", "Miramax", "Sony Pictures", "TriStar Pictures",
    "Touchstone Pictures", "Legendary Pictures", "Marvel Studios", "Pixar", "Focus Features",
    "Fox Searchlight Pictures", "A24", "Blumhouse Productions", "Summit Entertainment",
}
KEYWORD_FLAGS = {
    "kw_based_on_novel": "based on novel or book",
    "kw_based_on_true_story": "based on true story",
    "kw_biography": "biography",
    "kw_superhero": "superhero",
    "kw_remake": "remake",
    "kw_sequel": "sequel",
    "kw_based_on_comic": "based on comic",
    "kw_woman_director": "woman director",
    "kw_anime": "anime",
    "kw_slasher": "slasher",
}


def prior_stats(long: pd.DataFrame, key: str) -> pd.DataFrame:
    """long: columns [key, year, rating]. Returns per (key, year): mean & count of ratings
    from strictly earlier years."""
    g = long.groupby([key, "year"]).rating.agg(["sum", "count"]).reset_index()
    g = g.sort_values([key, "year"])
    g["cum_sum"] = g.groupby(key)["sum"].cumsum() - g["sum"]
    g["cum_cnt"] = g.groupby(key)["count"].cumsum() - g["count"]
    return g[[key, "year", "cum_sum", "cum_cnt"]]


def track_record(films: pd.DataFrame, people_col: str, history: pd.DataFrame, prefix: str):
    """films: [film_id, year, people_col(list)]; history: [person, year, rating].
    Returns film-level prior mean rating (pooled over people) and max prior film count."""
    f = films[["film_id", "year", people_col]].explode(people_col).dropna()
    f = f.rename(columns={people_col: "p"})
    f["p"] = f.p.astype(str)
    history = history.assign(person=history.person.astype(str))
    stats = prior_stats(history.rename(columns={"person": "p"}), "p")
    # all (p, year) pairs where p has films in history; merge_asof finds latest prior row
    f = f.sort_values("year")
    stats = stats.sort_values("year")
    # stats rows at year Y already exclude Y itself; for a film in year Y we want stats at
    # the smallest history year >= Y (it excludes Y and later), or the full total if none.
    totals = history.groupby("person").rating.agg(["sum", "count"]).rename(
        columns={"sum": "tot_sum", "count": "tot_cnt"})
    m = pd.merge_asof(f, stats, on="year", by="p", direction="forward")
    m = m.join(totals, on="p")
    m["cum_sum"] = m.cum_sum.fillna(m.tot_sum).fillna(0)
    m["cum_cnt"] = m.cum_cnt.fillna(m.tot_cnt).fillna(0)
    agg = m.groupby("film_id").agg(s=("cum_sum", "sum"), c=("cum_cnt", "sum"),
                                   mx=("cum_cnt", "max"))
    out = pd.DataFrame(index=agg.index)
    out[f"{prefix}_prev_mean"] = np.where(agg.c > 0, agg.s / agg.c.replace(0, np.nan), np.nan)
    out[f"{prefix}_prev_count"] = agg.mx
    return out


def split_ids(s):
    return s.split(",") if isinstance(s, str) else []


def main():
    base = pd.read_parquet(PROC / "imdb_base.parquet")
    hist = pd.read_parquet(PROC / "imdb_history.parquet")
    tmdb = pd.read_json(TMDB, lines=True)
    if "not_found" in tmdb:
        tmdb = tmdb[tmdb.not_found.isna()].drop(columns="not_found")
    tmdb = tmdb.drop_duplicates("imdb_id")

    df = base.merge(tmdb, left_on="tconst", right_on="imdb_id", how="inner")
    df["film_id"] = df.tconst
    df["release_date"] = pd.to_datetime(df.release_date, errors="coerce")
    # TMDB date can be a later re-release; trust IMDb year unless they agree within 1 year
    ok = (df.release_date.dt.year - df.startYear).abs() <= 1
    df.loc[~ok, "release_date"] = pd.NaT
    df["year"] = df.startYear.astype(int)

    feats = pd.DataFrame({"tconst": df.tconst, "title": df.primaryTitle,
                          "year": df.year, "averageRating": df.averageRating})
    feats["sort_date"] = df.release_date.fillna(pd.to_datetime(df.year.astype(str) + "-07-01"))

    # --- basic numeric
    feats["runtime"] = df.runtimeMinutes
    feats["release_month"] = df.release_date.dt.month          # NaN when unknown -> imputed
    feats["release_dow"] = df.release_date.dt.dayofweek
    feats["log_budget"] = np.log10(df.budget.where(df.budget >= 10_000))  # tiny = junk entry
    feats["budget_known"] = feats.log_budget.notna().astype(int)
    feats["is_franchise"] = df.collection_id.notna().astype(int)
    feats["n_genres"] = df.genres.fillna("").str.count(",") + df.genres.notna()
    feats["n_production_companies"] = df.production_companies.str.len()
    feats["n_production_countries"] = df.production_countries.str.len()
    feats["n_spoken_languages"] = df.spoken_languages.str.len()
    feats["major_studio"] = df.production_companies.apply(
        lambda cs: int(any(c in MAJOR_STUDIOS for c in cs)))
    feats["us_production"] = df.production_countries.apply(lambda cs: int("US" in cs))
    feats["female_lead"] = df.cast_top5.apply(
        lambda c: np.nan if not c or c[0]["gender"] == 0 else int(c[0]["gender"] == 1))

    # --- categorical (encoded inside the Pipeline)
    feats["original_language"] = df.original_language
    feats["us_certification"] = df.us_certification
    feats["main_country"] = df.production_countries.apply(lambda cs: cs[0] if cs else None)

    # --- genres multi-hot (IMDb genres)
    g = df.genres.fillna("").str.get_dummies(sep=",")
    g = g.loc[:, g.columns != ""]
    g.columns = ["genre_" + c.lower().replace("-", "_") for c in g.columns]
    feats = pd.concat([feats, g.set_index(feats.index)], axis=1)

    # --- keyword flags
    kws = df.keywords.apply(set)
    for col, kw in KEYWORD_FLAGS.items():
        feats[col] = kws.apply(lambda s, kw=kw: int(kw in s))

    # --- track records: directors & writers from full IMDb history
    hist = hist.copy()
    hist["year"] = hist.startYear.astype(int)
    films = df[["film_id", "year"]].copy()
    for role in ["directors", "writers"]:
        h = hist[["year", "averageRating", role]].copy()
        h["person"] = h[role].apply(split_ids)
        h = h.explode("person").dropna(subset=["person"]).rename(columns={"averageRating": "rating"})
        films[role] = df[f"{role}_x" if f"{role}_x" in df else role].apply(split_ids)
        tr = track_record(films, role, h[["person", "year", "rating"]], role[:-1])
        feats = feats.merge(tr, left_on="tconst", right_index=True, how="left")

    # --- cast (top 3 billed, TMDB IDs) and franchise, history = films in this dataset
    films["cast"] = df.cast_top5.apply(lambda c: [x["id"] for x in c[:3]])
    ch = films[["year", "cast"]].assign(rating=df.averageRating).explode("cast") \
        .dropna(subset=["cast"]).rename(columns={"cast": "person"})
    feats = feats.merge(track_record(films, "cast", ch[["person", "year", "rating"]], "cast"),
                        left_on="tconst", right_index=True, how="left")
    films["coll"] = df.collection_id.apply(lambda x: [x] if pd.notna(x) else [])
    fh = films[["year", "coll"]].assign(rating=df.averageRating).explode("coll") \
        .dropna(subset=["coll"]).rename(columns={"coll": "person"})
    feats = feats.merge(track_record(films, "coll", fh[["person", "year", "rating"]], "franchise"),
                        left_on="tconst", right_index=True, how="left")

    count_cols = [c for c in feats if c.endswith("_prev_count")]
    feats[count_cols] = feats[count_cols].fillna(0)
    feats = feats.sort_values("sort_date").reset_index(drop=True)
    feats.to_parquet(PROC / "features.parquet", index=False)
    print(f"{len(feats):,} rows x {feats.shape[1]} columns")
    print(feats.drop(columns=["tconst", "title"]).describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
