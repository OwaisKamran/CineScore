"""Export the local processed Parquet files to CSV, e.g. to open them in Excel.

    python scripts/export_csv.py                 # all files in data/processed -> data/csv/
    python scripts/export_csv.py features        # just data/processed/features.parquet

The CSV files are for local use only and are git-ignored: they contain data derived from the IMDb
Non-Commercial Datasets, which may not be republished. Run scripts 01-04 first to create the Parquet files.
Dates are written as ISO text (YYYY-MM-DD) and any list values as "a|b|c"; Excel opens the files with
UTF-8 encoding (utf-8-sig), so accented names display correctly.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data" / "processed", ROOT / "data" / "csv"


def export(path: Path) -> Path:
    df = pd.read_parquet(path)
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda v: isinstance(v, (list, tuple))).any():
            df[col] = df[col].map(lambda v: "|".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
    for col in df.select_dtypes("datetime").columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    out = OUT / f"{path.stem}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{path.relative_to(ROOT)} -> {out.relative_to(ROOT)}  ({len(df):,} rows, {df.shape[1]} columns, "
          f"{out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    names = sys.argv[1:]
    files = [SRC / f"{n.removesuffix('.parquet')}.parquet" for n in names] if names else sorted(SRC.glob("*.parquet"))
    missing = [f for f in files if not f.exists()]
    if not files or missing:
        sys.exit(f"No Parquet files found ({', '.join(map(str, missing)) or SRC}). Run scripts 01-04 first.")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in files:
        export(f)


if __name__ == "__main__":
    main()
