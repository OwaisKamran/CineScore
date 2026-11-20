"""Build the base movie list from the IMDb files.

Filters: titleType == movie, not adult, release year 1980-2025, runtime 60-240 min,
numVotes >= 2000 (so the target rating is stable).

numVotes is used ONLY for this filter - it is a post-release quantity and is never a feature.

Also writes data/processed/imdb_history.parquet: every rated IMDb movie (>=100 votes) with its
year, rating and director/writer IDs. It is used to compute "track record" features from
films released strictly BEFORE each target film.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
MIN_VOTES = 2000


def read_tsv(name, **kw):
    return pd.read_csv(RAW / name, sep="\t", na_values="\\N", quoting=3, **kw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    basics = read_tsv("title.basics.tsv.gz", dtype=str,
                      usecols=["tconst", "titleType", "primaryTitle", "isAdult",
                               "startYear", "runtimeMinutes", "genres"])
    basics = basics[basics.titleType == "movie"].drop(columns="titleType")
    ratings = read_tsv("title.ratings.tsv.gz")
    crew = read_tsv("title.crew.tsv.gz", dtype=str)

    m = basics.merge(ratings, on="tconst").merge(crew, on="tconst", how="left")
    m["startYear"] = pd.to_numeric(m.startYear, errors="coerce")
    m["runtimeMinutes"] = pd.to_numeric(m.runtimeMinutes, errors="coerce")
    m["isAdult"] = m.isAdult.eq("1")

    hist = m[(m.numVotes >= 100) & m.startYear.notna() & ~m.isAdult]
    hist[["tconst", "startYear", "averageRating", "directors", "writers"]] \
        .to_parquet(OUT / "imdb_history.parquet", index=False)

    base = m[(m.numVotes >= MIN_VOTES) & ~m.isAdult
             & m.startYear.between(1980, 2025) & m.runtimeMinutes.between(60, 240)]
    base = base.drop(columns="isAdult").reset_index(drop=True)
    base.to_parquet(OUT / "imdb_base.parquet", index=False)
    print(f"history rows: {len(hist):,}   base rows: {len(base):,}")


if __name__ == "__main__":
    main()
