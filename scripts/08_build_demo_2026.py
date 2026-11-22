"""Build app/data/demo_2026.json: five real 2026 films for the apps' "Load a real 2026 film" demo.

Selection rule (fixed before looking at any prediction, so it cannot favour films the model gets right):
  candidates are the 2026 releases in the latest official IMDb ratings file (title.ratings.tsv.gz)
  that are feature films (titleType movie, not adult, runtime 60-240 min, the scope of the training
  data), have at least 20,000 votes and complete TMDB data (release date, budget, US certification,
  original language, production countries and companies, spoken languages, billed cast).
  From these, the most-voted film in each actual-rating band: below 6, 6-7, 7-7.5, 7.5-8, 8 and above
  (a band includes its lower edge). If a band has no candidate, the next most-voted unused candidate
  from the nearest band(s) is taken instead, and the file says so.

Only PRE-RELEASE information is filled in, built exactly like the training features
(scripts/04_build_features.py): IMDb runtime and genres; TMDB release date, budget, certification,
language, countries, companies, spoken languages, top-3 cast, collection and story keywords;
IMDb directors and writers. Track records use only films from strictly earlier years (before
2026), the training rule. The regular app lookups already contain 2026 films, i.e. the film's own
rating, so the directors and writers get demo-only IDs ("<nconst>@pre2026") with their pre-2026
record. Cast and franchise records come from the dataset (films up to 2025) and are used as is.

The actual rating and vote count are stored for comparison only; nothing is retrained and the
model file is not touched.

Usage:  TMDB_TOKEN=... python scripts/08_build_demo_2026.py
        (or put TMDB_TOKEN=... in a .env file in the repo root; .env is git-ignored)
"""
import gzip
import importlib.util
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = ROOT / "app" / "data" / "demo_2026.json"
YEAR, MIN_VOTES = 2026, 20_000
BANDS = [("below 6", None, 6.0), ("6 to 7", 6.0, 7.0), ("7 to 7.5", 7.0, 7.5), ("7.5 to 8", 7.5, 8.0), ("8 and above", 8.0, None)]
NOTE = "Films chosen as the most-voted 2026 release in each rating band"


def band_of(r: float) -> int:
    return next(i for i, (_, lo, hi) in enumerate(BANDS) if (lo is None or r >= lo) and (hi is None or r < hi))
NA = "\\N"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def token_from_dotenv() -> None:
    """Read TMDB_TOKEN from .env if it is not already set (the value is never printed)."""
    env = ROOT / ".env"
    if os.environ.get("TMDB_TOKEN") or not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "TMDB_TOKEN" and v.strip():
            os.environ["TMDB_TOKEN"] = v.strip().strip('"').strip("'")


def candidates() -> pd.DataFrame:
    ratings = pd.read_csv(RAW / "title.ratings.tsv.gz", sep="\t", na_values=NA, quoting=3)
    basics = pd.read_csv(RAW / "title.basics.tsv.gz", sep="\t", na_values=NA, quoting=3, dtype=str,
                         usecols=["tconst", "titleType", "primaryTitle", "isAdult", "startYear",
                                  "runtimeMinutes", "genres"])
    b = basics[(basics.titleType == "movie") & (basics.startYear == str(YEAR)) & (basics.isAdult == "0")].copy()
    b["runtime"] = pd.to_numeric(b.runtimeMinutes, errors="coerce")
    m = b.merge(ratings, on="tconst")
    m = m[(m.numVotes >= MIN_VOTES) & m.runtime.between(60, 240)]
    return m.sort_values(["numVotes", "tconst"], ascending=[False, True]).reset_index(drop=True), basics


def main() -> None:
    token_from_dotenv()
    if not os.environ.get("TMDB_TOKEN"):
        sys.exit("TMDB_TOKEN is not set (environment or .env).")
    for f in ["title.ratings.tsv.gz", "title.basics.tsv.gz"]:
        if not (RAW / f).exists():
            sys.exit(f"data/raw/{f} missing: run scripts/01_download_imdb.py first (it downloads the latest files).")
    checked_on = datetime.fromtimestamp((RAW / "title.ratings.tsv.gz").stat().st_mtime).date().isoformat()

    tmdb = load_script("03_fetch_tmdb.py")      # same API call and field extraction as the training data
    feats = load_script("04_build_features.py")  # same major-studio list and keyword flags
    sys.path.insert(0, str(ROOT / "app"))
    from predictor import CERTIFICATIONS, ROLE_LIMITS, Predictor

    P = Predictor(demo=Path("no-demo"))          # plain lookups, without an older demo file
    opts = P.options()
    lookups = json.load(gzip.open(ROOT / "app" / "data" / "lookups.json.gz", "rt", encoding="utf-8"))
    names = {role: {str(r[0]): r for r in lookups[role]} for role in ("director", "writer", "cast", "franchise")}

    cands, basics = candidates()
    print(f"{len(cands)} feature films from {YEAR} with >= {MIN_VOTES:,} votes (ratings file of {checked_on})")

    # directors / writers: IMDb crew; their record from IMDb history films of strictly earlier years
    hist = pd.read_parquet(PROC / "imdb_history.parquet")
    titles = basics.set_index("tconst").primaryTitle
    crew = hist.set_index("tconst")[["directors", "writers"]]
    before = hist[hist.startYear < YEAR]

    def record(nconst: str, role: str) -> dict:
        col = role + "s"
        films = before[before[col].fillna("").str.split(",").apply(lambda xs: nconst in xs)]
        films = films.sort_values("startYear")
        base = names[role].get(nconst) or names["director" if role == "writer" else "writer"].get(nconst)
        n = len(films)
        return {"id": f"{nconst}@pre{YEAR}", "name": base[1] if base else nconst,
                "birth_year": base[2] if base else None,
                "prev_mean": round(float(films.averageRating.mean()), 3) if n else None, "prev_count": n,
                "last_title": titles.get(films.tconst.iloc[-1]) if n else None,
                "last_year": int(films.startYear.iloc[-1]) if n else None}

    def person(role: str, pid) -> dict | None:
        r = names[role].get(str(pid))
        if r is None:
            return None
        return {"id": str(r[0]), "name": r[1], "birth_year": r[2], "prev_mean": r[3], "prev_count": r[4],
                "last_title": r[5], "last_year": r[6]}

    skipped, built = [], {}

    def build(c):
        """The film's pre-release inputs, or None if its TMDB data is incomplete (memoised)."""
        if c.tconst in built:
            return built[c.tconst]
        d = tmdb.fetch(c.tconst)
        missing = []
        if d is None or d.get("not_found"):
            missing = ["no TMDB entry"]
        else:
            rd = d.get("release_date")
            checks = {
                "release date": rd and abs(int(rd[:4]) - YEAR) <= 1,
                "budget": (d.get("budget") or 0) >= 10_000,
                "US certification": d.get("us_certification") in CERTIFICATIONS,
                "original language": d.get("original_language") in {l["code"] for l in opts["languages"]},
                "production countries": bool(d.get("production_countries")) and d["production_countries"][0] in opts["countries"],
                "production companies": bool(d.get("production_companies")),
                "spoken languages": bool(d.get("spoken_languages")),
                "cast": bool(d.get("cast_top5")),
                "director and writer credits": c.tconst in crew.index and isinstance(crew.loc[c.tconst, "directors"], str),
            }
            missing = [k for k, ok in checks.items() if not ok]
        if missing:
            skipped.append({"id": c.tconst, "title": c.primaryTitle, "votes": int(c.numVotes), "missing": missing})
            print(f"  skip {c.primaryTitle} ({int(c.numVotes):,} votes): missing {', '.join(missing)}")
            built[c.tconst] = None
            return None

        genres = [g.lower().replace("-", "_") for g in str(c.genres).split(",") if g and g != "nan"]
        countries = d["production_countries"]
        others = [x for x in countries[1:] if x in opts["countries"]]
        directors = crew.loc[c.tconst, "directors"].split(",")[:ROLE_LIMITS["director"]]
        writers = [w for w in str(crew.loc[c.tconst, "writers"]).split(",") if w.startswith("nm")][:ROLE_LIMITS["writer"]]
        cast = [p for p in (person("cast", x["id"]) for x in d["cast_top5"][:3]) if p]
        franchise = [p for p in [person("franchise", d["collection_id"])] if p] if d.get("collection_id") else []
        g0 = d["cast_top5"][0]["gender"]
        kws = set(d.get("keywords") or [])
        film = {
            "id": c.tconst, "title": c.primaryTitle, "imdb_url": f"https://www.imdb.com/title/{c.tconst}/",
            "release_date": d["release_date"], "runtime": int(c.runtime),
            "genres": [g for g in genres if g in opts["genres"]][:3],
            "budget_usd": int(d["budget"]), "us_certification": d["us_certification"],
            "original_language": d["original_language"], "production_countries": [countries[0]] + others,
            "n_production_companies": min(30, max(1, len(d["production_companies"]))),
            "n_spoken_languages": min(12, max(1, len(d["spoken_languages"]))),
            "major_studio": any(x in feats.MAJOR_STUDIOS for x in d["production_companies"]),
            "female_lead": "unknown" if g0 == 0 else ("yes" if g0 == 1 else "no"),
            "keywords": [k for k, kw in feats.KEYWORD_FLAGS.items() if kw in kws],
            "people": {"director": [record(n, "director") for n in directors],
                       "writer": [record(n, "writer") for n in writers],
                       "cast": cast, "franchise": franchise},
            "actual": {"rating": float(c.averageRating), "votes": int(c.numVotes)},
            "dropped": {"other countries not in the model": [x for x in countries[1:] if x not in opts["countries"]],
                        "writers beyond the form's 5": max(0, len([w for w in str(crew.loc[c.tconst, 'writers']).split(',') if w.startswith('nm')]) - 5),
                        "cast without earlier films in the data": [x["name"] for x in d["cast_top5"][:3] if not person("cast", x["id"])]},
        }
        built[c.tconst] = film
        return film


    # one film per actual-rating band: the most-voted candidate with complete TMDB data
    cands["band"] = cands.averageRating.apply(band_of)
    picked = {}
    for i, (label, _, _) in enumerate(BANDS):
        for c in cands[cands.band == i].itertuples():
            f = build(c)
            if f:
                picked[i] = dict(f, band=label)
                print(f"  {label:12s} {c.primaryTitle} ({int(c.numVotes):,} votes, actual {c.averageRating})")
                break
    taken = {f["id"] for f in picked.values()}
    substitutes = []
    for i, (label, _, _) in enumerate(BANDS):  # empty band: next most-voted film from the nearest band(s)
        if i in picked:
            continue
        for dist in range(1, len(BANDS)):
            near = [j for j in (i - dist, i + dist) if 0 <= j < len(BANDS)]
            pool = cands[cands.band.isin(near) & ~cands.tconst.isin(taken)]
            f = next((x for x in (build(c) for c in pool.itertuples()) if x), None)
            if f:
                from_band = BANDS[int(cands.loc[cands.tconst == f["id"], "band"].iloc[0])][0]
                picked[i] = dict(f, band=f"{label} (empty; substitute from {from_band})")
                taken.add(f["id"])
                substitutes.append({"band": label, "film": f["title"], "from_band": from_band})
                print(f"  {label:12s} EMPTY -> substitute {f['title']} from band {from_band}")
                break
    films = [picked[i] for i in sorted(picked)]

    data = {
        "checked_on": checked_on,
        "source": ("Actual ratings: IMDb Non-Commercial Datasets, title.ratings.tsv.gz downloaded "
                   f"{checked_on} (https://datasets.imdbws.com/). Pre-release details: IMDb title.basics and "
                   "crew, TMDB API (https://www.themoviedb.org/). This product uses the TMDB API but is not "
                   "endorsed or certified by TMDB."),
        "note": NOTE,
        "rule": (f"Candidates: {YEAR} feature films (runtime 60-240 min, not adult) with at least {MIN_VOTES:,} votes in "
                 "the latest IMDb ratings file and complete TMDB data (release date, budget, US certification, language, "
                 "countries, companies, spoken languages, cast). Pick the most-voted candidate in each actual-rating band: "
                 "below 6, 6-7, 7-7.5, 7.5-8, 8 and above (lower edge included). If a band is empty, take the next "
                 "most-voted unused candidate from the nearest band(s) and say so. Fixed before looking at any "
                 "prediction; not chosen by how well the model does."),
        "bands": [b[0] for b in BANDS],
        "substitutes": substitutes,
        "track_records": (f"Directors and writers: films from before {YEAR} only (demo IDs '<nconst>@pre{YEAR}'), "
                          "because the app's regular lookups include 2026 films. Cast and franchise: dataset films "
                          "up to 2025."),
        "skipped": skipped,
        "films": films,
    }
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {OUT.relative_to(ROOT)} with {len(films)} films")

    # the forecasts these inputs give (printed only, for checking; not stored)
    P = Predictor()
    for f in films:
        payload = {"title": f["title"], "release_date": f["release_date"], "runtime": f["runtime"],
                   "genres": f["genres"], "budget_usd": f["budget_usd"], "us_certification": f["us_certification"],
                   "directors": [p["id"] for p in f["people"]["director"]],
                   "writers": [p["id"] for p in f["people"]["writer"]],
                   "cast": [p["id"] for p in f["people"]["cast"]],
                   "franchise": f["people"]["franchise"][0]["id"] if f["people"]["franchise"] else None,
                   "original_language": f["original_language"], "production_countries": f["production_countries"],
                   "n_production_companies": f["n_production_companies"], "n_spoken_languages": f["n_spoken_languages"],
                   "female_lead": f["female_lead"], "major_studio": f["major_studio"], "keywords": f["keywords"]}
        r = P.predict(payload)
        lo, hi = r["interval_80"]
        inside = lo <= f["actual"]["rating"] <= hi
        print(f"  {f['title']:40s} forecast {r['prediction']:.2f} ({lo:.1f}-{hi:.1f})  actual {f['actual']['rating']:.1f}"
              f"  {'inside' if inside else 'OUTSIDE'}")


if __name__ == "__main__":
    main()
