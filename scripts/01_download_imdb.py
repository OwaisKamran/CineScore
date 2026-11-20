"""Download the official IMDb non-commercial dataset files.

Source : https://datasets.imdbws.com/  (documented at https://developer.imdb.com/non-commercial-datasets/)
Licence: IMDb Non-Commercial Datasets - personal and non-commercial use only.
         Raw files are NOT committed to the repository; run this script instead.

Usage:  python scripts/01_download_imdb.py
"""
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://datasets.imdbws.com/"
FILES = [
    "title.basics.tsv.gz",   # title type, year, runtime, genres, isAdult
    "title.ratings.tsv.gz",  # averageRating (target) and numVotes (used only to filter)
    "title.crew.tsv.gz",     # director and writer IDs
    "name.basics.tsv.gz",    # person names (for the app's director/writer search)
]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(name: str) -> None:
    dest = RAW_DIR / name
    if dest.exists():
        print(f"[skip] {name} already present")
        return
    print(f"[get ] {name}")
    with requests.get(BASE_URL + name, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name)
    (RAW_DIR / "IMDB_DOWNLOAD_DATE.txt").write_text(date.today().isoformat() + "\n")
    print("Done.")
