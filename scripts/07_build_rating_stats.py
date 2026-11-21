"""Build the small summary the web apps use for their context charts.

From data/processed/features.parquet (all 27,451 films, 1980-2025) it stores:
  * counts: number of films at each IMDb rating 1.0, 1.1, ..., 10.0 (histogram + "rated higher than
    X% of films"); IMDb ratings have one decimal, so this is exact
  * mean: average rating of all films
  * genres: average rating and film count per genre

Aggregate counts and averages only; no film-level IMDb data is written.
Output: app/data/rating_stats.json        Usage: python scripts/07_build_rating_stats.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "data" / "rating_stats.json"

df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
idx = np.rint(df["averageRating"].to_numpy() * 10).astype(int) - 10        # 1.0 -> 0, 10.0 -> 90
counts = np.bincount(idx, minlength=91)[:91]
genres = {c.removeprefix("genre_"): [round(float(df.loc[df[c] == 1, "averageRating"].mean()), 3),
                                     int((df[c] == 1).sum())]
          for c in df.columns if c.startswith("genre_")}
stats = {
    "n": int(len(df)),
    "years": [int(df["year"].min()), int(df["year"].max())],
    "mean": round(float(df["averageRating"].mean()), 3),
    "counts": counts.tolist(),
    "genres": genres,
}
OUT.write_text(json.dumps(stats, separators=(",", ":")), encoding="utf-8")
print(f"{OUT.relative_to(ROOT)}: {stats['n']} films {stats['years']}, mean {stats['mean']}, "
      f"{len(genres)} genres, {OUT.stat().st_size} bytes")
