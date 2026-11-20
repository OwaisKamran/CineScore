"""Fetch pre-release metadata for every base movie from the TMDB API.

Source : The Movie Database (TMDB) API v3 - https://developer.themoviedb.org/
Terms  : https://www.themoviedb.org/api-terms-of-use  (non-commercial use with attribution)
Auth   : set TMDB_TOKEN to your TMDB "API Read Access Token" (v4 bearer token).

Only fields known BEFORE release are kept. revenue, popularity, vote_average and
vote_count are dropped here so they cannot leak into the model.

The script is resumable: results are appended to data/raw/tmdb_movies.jsonl and
already-fetched IMDb IDs are skipped on the next run.

Usage:  TMDB_TOKEN=... python scripts/03_fetch_tmdb.py
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "processed" / "imdb_base.parquet"
OUT = ROOT / "data" / "raw" / "tmdb_movies.jsonl"
URL = "https://api.themoviedb.org/3/movie/{imdb_id}"
PARAMS = {"append_to_response": "credits,keywords,release_dates", "language": "en-US"}
WORKERS = 12

_local = threading.local()
_lock = threading.Lock()


def session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        token = os.environ.get("TMDB_TOKEN", "")
        if token:
            s.headers["Authorization"] = f"Bearer {token}"
        s.headers["accept"] = "application/json"
        _local.s = s
    return _local.s


def us_certification(release_dates):
    for country in release_dates.get("results", []):
        if country["iso_3166_1"] != "US":
            continue
        rels = sorted(country["release_dates"], key=lambda r: (r["type"] != 3, r["release_date"]))
        for rel in rels:
            if rel.get("certification"):
                return rel["certification"]
    return None


def slim(d):
    cast = sorted(d["credits"]["cast"], key=lambda c: c["order"])
    crew = d["credits"]["crew"]
    return {
        "imdb_id": d["imdb_id"],
        "tmdb_id": d["id"],
        "release_date": d.get("release_date") or None,
        "budget": d.get("budget") or 0,
        "collection_id": (d.get("belongs_to_collection") or {}).get("id"),
        "original_language": d.get("original_language"),
        "origin_country": d.get("origin_country", []),
        "production_companies": [c["name"] for c in d.get("production_companies", [])],
        "production_countries": [c["iso_3166_1"] for c in d.get("production_countries", [])],
        "spoken_languages": [c["iso_639_1"] for c in d.get("spoken_languages", [])],
        "overview_len": len(d.get("overview") or ""),
        "has_tagline": bool(d.get("tagline")),
        "has_homepage": bool(d.get("homepage")),
        "tmdb_runtime": d.get("runtime"),
        "cast_top5": [{"id": c["id"], "name": c["name"], "gender": c["gender"]} for c in cast[:5]],
        "cast_size": len(cast),
        "crew_size": len(crew),
        "directors": [c["id"] for c in crew if c["job"] == "Director"],
        "composer": [c["id"] for c in crew if c["job"] == "Original Music Composer"],
        "keywords": [k["name"] for k in d["keywords"]["keywords"]],
        "us_certification": us_certification(d.get("release_dates", {})),
    }


def fetch(imdb_id, retries=5):
    for attempt in range(retries):
        try:
            r = session().get(URL.format(imdb_id=imdb_id), params=PARAMS, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return slim(r.json())
        if r.status_code == 404:
            return {"imdb_id": imdb_id, "not_found": True}
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 2)))
            continue
        time.sleep(2 ** attempt)
    return None


def main():
    ids = pd.read_parquet(BASE, columns=["tconst"]).tconst.tolist()
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {json.loads(line)["imdb_id"] for line in f}
    todo = [i for i in ids if i not in done]
    print(f"{len(ids):,} total, {len(done):,} done, {len(todo):,} to fetch")
    failed = 0
    with open(OUT, "a") as f, ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch, i): i for i in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), mininterval=10):
            rec = fut.result()
            if rec is None:
                failed += 1
                continue
            rec["imdb_id"] = futs[fut]  # key by the IMDb ID we asked for
            with _lock:
                f.write(json.dumps(rec) + "\n")
    print(f"finished; {failed} failed (re-run to retry)")


if __name__ == "__main__":
    main()
