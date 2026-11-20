"""Build the lookup tables the web app uses to turn names into track-record features.

At prediction time the film is a *future* release, so a person's track record is the mean IMDb
rating and count of ALL their earlier films (no year cut-off). The sources match those used in
training:
  * directors / writers -> IMDb history (movies with >= 100 votes), names from name.basics
  * top-3 billed cast  -> films in the CineScore dataset (TMDB person IDs and names)
  * franchise          -> TMDB collections in the dataset (names fetched from the TMDB API)

Output: app/data/lookups.json.gz
Usage : TMDB_TOKEN=... python scripts/05_build_app_lookups.py
"""
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = ROOT / "app" / "data" / "lookups.json.gz"


def read_tsv(name, **kw):
    return pd.read_csv(RAW / name, sep="\t", na_values="\\N", quoting=3, **kw)


def summarise(long, key):
    """long: [key, rating, year, title] -> per key: mean, count, latest title."""
    long = long.sort_values("year")
    g = long.groupby(key).agg(mean=("rating", "mean"), count=("rating", "size"),
                              last_title=("title", "last"), last_year=("year", "last"))
    g["mean"] = g["mean"].round(3)
    return g


def imdb_people():
    hist = pd.read_parquet(PROC / "imdb_history.parquet")
    titles = read_tsv("title.basics.tsv.gz", usecols=["tconst", "primaryTitle"], dtype=str)
    hist = hist.merge(titles, on="tconst", how="left").rename(
        columns={"averageRating": "rating", "startYear": "year", "primaryTitle": "title"})
    out = {}
    names = None
    for role in ["directors", "writers"]:
        long = hist.assign(person=hist[role].str.split(",")).explode("person").dropna(subset=["person"])
        s = summarise(long, "person")
        out[role] = s
    ids = set(out["directors"].index) | set(out["writers"].index)
    names = read_tsv("name.basics.tsv.gz", usecols=["nconst", "primaryName", "birthYear"], dtype=str)
    names = names[names.nconst.isin(ids)].set_index("nconst")
    result = {}
    for role, s in out.items():
        s = s.join(names, how="inner")
        result[role[:-1]] = [
            [i, r.primaryName, None if pd.isna(r.birthYear) else int(r.birthYear),
             r["mean"], int(r["count"]), r.last_title, int(r.last_year)]
            for i, r in s.iterrows()]
    return result


def dataset_cast_and_franchise():
    feats = pd.read_parquet(PROC / "features.parquet", columns=["tconst", "title", "year",
                                                                "averageRating"])
    tmdb = pd.read_json(RAW / "tmdb_movies.jsonl", lines=True)
    tmdb = tmdb[tmdb.get("not_found").isna()] if "not_found" in tmdb else tmdb
    d = feats.merge(tmdb[["imdb_id", "cast_top5", "collection_id"]], left_on="tconst",
                    right_on="imdb_id").rename(columns={"averageRating": "rating"})

    cast = d.assign(c=d.cast_top5.apply(lambda c: c[:3])).explode("c").dropna(subset=["c"])
    cast["pid"] = cast.c.apply(lambda x: x["id"])
    cast["pname"] = cast.c.apply(lambda x: x["name"])
    s = summarise(cast, "pid").join(cast.groupby("pid").pname.first())
    cast_rows = [[int(i), r.pname, None, r["mean"], int(r["count"]), r.last_title,
                  int(r.last_year)] for i, r in s.iterrows()]

    fr = d.dropna(subset=["collection_id"]).assign(cid=lambda x: x.collection_id.astype(int))
    s = summarise(fr, "cid")
    names = fetch_collection_names(s.index.tolist())
    fr_rows = [[int(i), names.get(int(i)) or r.last_title + " (series)", None, r["mean"],
                int(r["count"]), r.last_title, int(r.last_year)] for i, r in s.iterrows()]
    return cast_rows, fr_rows


def fetch_collection_names(ids):
    sess = requests.Session()
    tok = os.environ.get("TMDB_TOKEN", "")
    if tok:
        sess.headers["Authorization"] = f"Bearer {tok}"

    def one(cid):
        try:
            r = sess.get(f"https://api.themoviedb.org/3/collection/{cid}", timeout=30)
            return cid, r.json().get("name") if r.ok else None
        except requests.RequestException:
            return cid, None

    with ThreadPoolExecutor(12) as ex:
        return dict(ex.map(one, ids))


def main():
    people = imdb_people()
    cast, franchise = dataset_cast_and_franchise()
    data = {"columns": ["id", "name", "birth_year", "prev_mean", "prev_count", "last_title",
                        "last_year"],
            "director": people["director"], "writer": people["writer"],
            "cast": cast, "franchise": franchise}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    for k in ["director", "writer", "cast", "franchise"]:
        print(f"{k:9s} {len(data[k]):7,d}")
    print(f"saved {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
