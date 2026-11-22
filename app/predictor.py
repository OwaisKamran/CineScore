"""CineScore prediction logic shared by the web app(s).

Turns a user's description of an upcoming film into the 57 pre-release features the saved
scikit-learn pipeline expects, validates every input, and returns a prediction with an 80%
interval and a simple "what drives this prediction" breakdown.
"""
from __future__ import annotations

import gzip
import json
import math
import unicodedata
from bisect import bisect_right
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"

CERTIFICATIONS = ["G", "PG", "PG-13", "R", "NC-17", "NR"]
KEYWORDS = {
    "kw_based_on_novel": "Based on a novel or book",
    "kw_based_on_true_story": "Based on a true story",
    "kw_biography": "Life story of a real person",
    "kw_superhero": "Superhero",
    "kw_remake": "Remake",
    "kw_sequel": "Sequel",
    "kw_based_on_comic": "Based on a comic",
    "kw_woman_director": "Woman director",
    "kw_anime": "Anime",
    "kw_slasher": "Slasher",
}
LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese (Mandarin)", "cn": "Chinese (Cantonese)",
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "ru": "Russian",
    "pt": "Portuguese", "sv": "Swedish", "da": "Danish", "no": "Norwegian", "tr": "Turkish",
    "fa": "Persian", "nl": "Dutch", "pl": "Polish", "th": "Thai", "kn": "Kannada",
    "fi": "Finnish", "ar": "Arabic", "id": "Indonesian", "bn": "Bengali", "ur": "Urdu",
    "mr": "Marathi", "tl": "Tagalog", "he": "Hebrew", "el": "Greek", "hu": "Hungarian",
    "cs": "Czech", "ro": "Romanian", "pa": "Punjabi",
}
ROLE_LIMITS = {"director": 3, "writer": 5, "cast": 3, "franchise": 1}


class ValidationError(ValueError):
    """Raised for bad user input; message is safe to show to the user."""


def fold(s: str) -> str:
    """Lower case without accents, so 'almodovar' finds 'Almodóvar'."""
    s = s.lower()
    if s.isascii():
        return s
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


class NameIndex:
    """Prefix index over one role's names, built once at start-up.

    Rows are sorted by number of earlier films (most first). For every word of every name the
    index maps its first 2 and first 3 letters to the rows that have such a word, already in that
    order. A query looks up the first letters of its first word and checks only those rows:
    first names that start with the query, then names where a later word does. Only when that
    finds nothing does it fall back to a plain substring scan (old behaviour, e.g. "olan").
    """

    def __init__(self, rows: list):
        self.rows = rows
        self.names = [fold(r[1]) for r in rows]
        index: dict[str, list[int]] = {}
        for k, name in enumerate(self.names):
            for key in {w[:n] for w in name.split() for n in (2, 3) if len(w) >= n}:
                index.setdefault(key, []).append(k)
        self.index = index
        self._text = self._starts = None  # substring fallback, built on first use

    def find(self, q: str, limit: int) -> list[int]:
        first = q.split(" ", 1)[0]
        cands = self.index.get(first[:3], ()) if len(first) >= 2 else None
        if cands is None:  # one-letter first word ("j abrams"): use the fallback
            return self._scan(q, limit)
        names, out = self.names, []
        for k in cands:  # the name starts with the query
            if names[k].startswith(q):
                out.append(k)
                if len(out) == limit:
                    return out
        word, seen = " " + q, set(out)
        for k in cands:  # a later word starts with the query
            if k not in seen and word in names[k]:
                out.append(k)
                if len(out) == limit:
                    break
        return out or self._scan(q, limit)

    def _scan(self, q: str, limit: int) -> list[int]:
        if self._text is None:
            self._starts, pos = [], 1
            for n in self.names:
                self._starts.append(pos)
                pos += len(n) + 1
            self._text = "\n" + "\n".join(self.names) + "\n"
        text, starts, out = self._text, self._starts, []
        i = text.find(q)
        while i != -1 and len(out) < limit:
            k = bisect_right(starts, i) - 1
            out.append(k)
            i = text.find(q, starts[k + 1] - 1 if k + 1 < len(starts) else len(text))
        return out


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR, lookups: Path = HERE / "data" / "lookups.json.gz",
                 demo: Path = HERE / "data" / "demo_2026.json"):
        self.pipe = joblib.load(model_dir / "cinescore_pipeline.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.num = self.meta["features_numeric"]
        self.cat = self.meta["features_categorical"]
        self.lo, self.hi = self.meta["interval_80"]
        self.genres = [c.removeprefix("genre_") for c in self.num if c.startswith("genre_")]

        enc = self.pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        cats = dict(zip(self.cat, enc.categories_))
        self.languages = sorted(c for c in cats["original_language"] if c != "missing")
        self.countries = sorted(c for c in cats["main_country"] if c != "missing")

        with gzip.open(lookups, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        self.people = {role: {str(r[0]): r for r in raw[role]} for role in ROLE_LIMITS}
        # most earlier films first (stable, so ties keep the file order, as before)
        self.search_index = {role: NameIndex(sorted(raw[role], key=lambda r: -r[4])) for role in ROLE_LIMITS}
        self._search = lru_cache(maxsize=4096)(self._search_uncached)

        # Real 2026 films for the demo. Their directors and writers get their own IDs whose track
        # record uses only films from before the film's year (the training rule), because the
        # regular lookups already include 2026 films and so the film's own rating.
        self.demo = None
        if demo.exists():
            self.demo = json.loads(demo.read_text(encoding="utf-8"))
            for film in self.demo["films"]:
                for role, people in film["people"].items():
                    for p in people:
                        if p["id"] not in self.people[role]:
                            self.people[role][p["id"]] = [p["id"], p["name"], p.get("birth_year"),
                                                          p["prev_mean"] or 0.0, p["prev_count"],
                                                          p.get("last_title"), p.get("last_year")]

    # ------------------------------------------------------------------ options / search
    def options(self) -> dict:
        return {
            "genres": self.genres,
            "languages": [{"code": c, "name": LANGUAGE_NAMES.get(c, c)} for c in self.languages],
            "countries": self.countries,
            "certifications": CERTIFICATIONS,
            "keywords": [{"key": k, "label": v} for k, v in KEYWORDS.items()],
            "metrics": self.meta["test_metrics"],
            "model": self.meta["model"],
        }

    def search(self, role: str, q: str, limit: int = 8) -> list[dict]:
        if role not in self.search_index:
            raise ValidationError(f"Unknown role '{role}'.")
        q = " ".join(fold(q or "").split())
        if len(q) < 2:
            return []
        rows = self.search_index[role].rows
        return [self._person(rows[k]) for k in self._search(role, q, limit)]

    def _search_uncached(self, role: str, q: str, limit: int) -> tuple:
        return tuple(self.search_index[role].find(q, limit))

    def demo_films(self) -> list[dict]:
        """The real 2026 films (pre-release inputs + actual rating), or [] if the file is missing."""
        return self.demo["films"] if self.demo else []

    @staticmethod
    def _person(r) -> dict:
        return {"id": str(r[0]), "name": r[1], "birth_year": r[2], "prev_mean": r[3],
                "prev_count": r[4], "last_title": r[5], "last_year": r[6]}

    # ------------------------------------------------------------------ features
    def build_features(self, p: dict) -> tuple[pd.DataFrame, dict]:
        if not isinstance(p, dict):
            raise ValidationError("Request body must be a JSON object.")
        f = {c: 0.0 for c in self.num}

        # release date
        try:
            rd = date.fromisoformat(str(p.get("release_date", "")))
        except ValueError:
            raise ValidationError("Release date is required (YYYY-MM-DD).") from None
        if not 1980 <= rd.year <= 2035:
            raise ValidationError("Release year must be between 1980 and 2035.")
        f.update(year=rd.year, release_month=rd.month, release_dow=rd.weekday())

        runtime = _number(p.get("runtime"), "Runtime", 60, 240, required=True)
        f["runtime"] = runtime

        genres = p.get("genres") or []
        if not isinstance(genres, list) or not 1 <= len(genres) <= 3:
            raise ValidationError("Choose between 1 and 3 genres.")
        for g in genres:
            if g not in self.genres:
                raise ValidationError(f"Unknown genre '{g}'.")
            f[f"genre_{g}"] = 1
        f["n_genres"] = len(genres)

        budget = _number(p.get("budget_usd"), "Budget", 10_000, 1_000_000_000)
        f["log_budget"] = math.log10(budget) if budget else np.nan
        f["budget_known"] = int(budget is not None)

        countries = p.get("production_countries") or []
        if not isinstance(countries, list) or not 1 <= len(countries) <= 16:
            raise ValidationError("Choose at least one production country.")
        f["n_production_countries"] = len(countries)
        f["us_production"] = int("US" in countries)
        f["n_production_companies"] = _number(p.get("n_production_companies", 2),
                                              "Number of production companies", 1, 30, True)
        f["n_spoken_languages"] = _number(p.get("n_spoken_languages", 1),
                                          "Number of spoken languages", 1, 12, True)
        f["major_studio"] = int(bool(p.get("major_studio")))
        lead = p.get("female_lead", "unknown")
        f["female_lead"] = {"yes": 1, "no": 0}.get(lead, np.nan)

        for k in p.get("keywords") or []:
            if k not in KEYWORDS:
                raise ValidationError(f"Unknown keyword '{k}'.")
            f[k] = 1

        # track records
        for role, prefix in [("director", "director"), ("writer", "writer"), ("cast", "cast"),
                             ("franchise", "franchise")]:
            ids = p.get(role + "s" if role != "cast" else "cast") or []
            if role == "franchise":
                ids = [p["franchise"]] if p.get("franchise") else []
            if not isinstance(ids, list) or len(ids) > ROLE_LIMITS[role]:
                raise ValidationError(f"At most {ROLE_LIMITS[role]} {role}(s) allowed.")
            rows = []
            for i in ids:
                r = self.people[role].get(str(i))
                if r is None:
                    raise ValidationError(f"Unknown {role} ID '{i}'. Pick names from the list.")
                rows.append(r)
            total = sum(r[4] for r in rows)
            f[f"{prefix}_prev_mean"] = (sum(r[3] * r[4] for r in rows) / total) if total else np.nan
            f[f"{prefix}_prev_count"] = max((r[4] for r in rows), default=0)
        f["is_franchise"] = int(bool(p.get("franchise")) or bool(p.get("is_franchise")))

        lang = p.get("original_language") or "en"
        cert = p.get("us_certification") or None
        if cert is not None and cert not in CERTIFICATIONS:
            raise ValidationError(f"Unknown certification '{cert}'.")
        row = {**f, "original_language": lang, "us_certification": cert,
               "main_country": countries[0]}
        X = pd.DataFrame([row])[self.num + self.cat]
        return X, row

    # ------------------------------------------------------------------ prediction
    def predict(self, payload: dict) -> dict:
        X, row = self.build_features(payload)
        pred = float(np.clip(self.pipe.predict(X)[0], 1, 10))
        return {
            "title": (payload.get("title") or "").strip()[:200] or None,
            "prediction": round(pred, 2),
            "interval_80": [round(max(1.0, pred + self.lo), 2), round(min(10.0, pred + self.hi), 2)],
            "drivers": self._drivers(X, pred),
            "track_record": {k: (None if pd.isna(row[k]) else round(float(row[k]), 2))
                             for k in ["director_prev_mean", "writer_prev_mean",
                                       "cast_prev_mean", "franchise_prev_mean"]},
        }

    def _drivers(self, X: pd.DataFrame, pred: float) -> list[dict]:
        """How much each input group moves the prediction compared with a neutral value
        (unknown track record, no genre flags, median runtime, unknown budget, ...)."""
        groups = {
            "Director track record": {"director_prev_mean": np.nan, "director_prev_count": 0},
            "Writer track record": {"writer_prev_mean": np.nan, "writer_prev_count": 0},
            "Cast track record": {"cast_prev_mean": np.nan, "cast_prev_count": 0},
            "Franchise history": {"franchise_prev_mean": np.nan, "franchise_prev_count": 0,
                                  "is_franchise": 0},
            "Genres": {c: 0 for c in self.num if c.startswith("genre_")} | {"n_genres": 0},
            "Runtime": {"runtime": 103},
            "Budget": {"log_budget": np.nan, "budget_known": 0},
            "Language & country": {"original_language": "en", "main_country": "US",
                                   "us_production": 1},
            "Story keywords": {k: 0 for k in KEYWORDS},
        }
        variants = []
        for cols in groups.values():
            v = X.copy()
            for c, val in cols.items():
                v[c] = val
            variants.append(v)
        neutral = self.pipe.predict(pd.concat(variants, ignore_index=True))
        out = [{"group": g, "effect": round(pred - float(n), 3)}
               for g, n in zip(groups, neutral)]
        return sorted(out, key=lambda d: -abs(d["effect"]))


def _number(v, label, lo, hi, required=False):
    if v in (None, ""):
        if required:
            raise ValidationError(f"{label} is required.")
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValidationError(f"{label} must be a number.") from None
    if not math.isfinite(x) or not lo <= x <= hi:
        raise ValidationError(f"{label} must be between {lo:,} and {hi:,}.")
    return x
