#!/usr/bin/env python3
"""Audit the merged dataset quality.

Usage:
  python3 scripts/audit_dataset.py --file data/processed/full_merged.parquet

This prints a short, human-readable quality report:
- rows, symbols, date range
- duplicates on (symbol, Date)
- basic sanity checks (prices > 0, High >= Low, Volume >= 0)

It is intentionally simple for an academic project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit merged dataset quality")
    ap.add_argument(
        "--file",
        default="data/processed/full_merged.parquet",
        help="Path to merged dataset (.parquet or .csv)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=200000,
        help="Rows to sample for sanity checks (0 = full scan)",
    )
    args = ap.parse_args()

    f = Path(args.file).expanduser().resolve()
    if not f.exists():
        print(f"ERROR: file not found: {f}")
        return 2

    print("=" * 60)
    print("DATASET AUDIT")
    print("=" * 60)
    print(f"File: {f}")

    df = _read_any(f)

    req_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "symbol"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return 1

    rows = len(df)
    symbols = int(df["symbol"].nunique())

    # Date range (robust parse)
    dts = pd.to_datetime(df["Date"], utc=True, errors="coerce", format="mixed")
    unparsed = int(dts.isna().sum())
    min_date = dts.min()
    max_date = dts.max()

    print(f"Rows: {rows:,}")
    print(f"Symbols: {symbols:,}")
    print(f"Date range: {min_date} -> {max_date}")
    print(f"Unparsed Date values: {unparsed:,}")

    dupes = int(df.duplicated(subset=["symbol", "Date"]).sum())
    print(f"Duplicates (symbol, Date): {dupes:,}")

    # Sample for sanity checks to keep it fast.
    sample_n = int(args.sample)
    if sample_n and rows > sample_n:
        chk = df.sample(sample_n, random_state=7)
        mode = f"sample({sample_n:,})"
    else:
        chk = df
        mode = "full"

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        chk[c] = pd.to_numeric(chk[c], errors="coerce")

    bad_price = int(((chk[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)).sum())
    bad_hl = int((chk["High"] < chk["Low"]).sum())
    bad_vol = int((chk["Volume"] < 0).sum())

    print("-" * 60)
    print(f"Sanity checks ({mode}):")
    print(f"Bad price rows (any of O/H/L/C <= 0): {bad_price:,}")
    print(f"High < Low rows: {bad_hl:,}")
    print(f"Negative volume rows: {bad_vol:,}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
