#!/usr/bin/env python3
"""
Merge Kaggle historical data with scraper recent data.
For overlapping stocks (AAPL, MSFT, NVDA), scraper data takes priority.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = Path("/Users/allanodora/Downloads/archive/Stocks")
SCRAPER_DIR = Path("/Users/allanodora/Downloads/stock_scrapper/stock_data_api_hybrid")
OUTPUT_FILE = PROJECT_ROOT / "data/processed/merged_2017_2026.csv"

SCRAPER_STOCKS = {"AAPL", "MSFT", "NVDA"}
STANDARD_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "OpenInt",
    "symbol",
]


def load_scraper_data() -> pd.DataFrame:
    """Load all scraper CSV files."""
    scraper_dfs = []
    for csv_file in SCRAPER_DIR.glob("*.csv"):
        if csv_file.name == "run_report.csv":
            continue
        df = pd.read_csv(csv_file)
        scraper_dfs.append(df)
        print(f"Loaded scraper: {csv_file.stem} ({len(df)} rows)")

    if scraper_dfs:
        return pd.concat(scraper_dfs, ignore_index=True)
    return pd.DataFrame()


def load_kaggle_data(exclude_symbols: Set[str]) -> pd.DataFrame:
    """Load Kaggle data, excluding stocks that have scraper data."""
    kaggle_files = list(KAGGLE_DIR.glob("*.us.txt"))
    kaggle_dfs = []
    loaded = 0

    for f in kaggle_files:
        symbol = f.stem.replace(".us", "").upper()
        if symbol in exclude_symbols:
            continue  # Skip - will use scraper data instead

        try:
            df = pd.read_csv(f)
            df["symbol"] = symbol
            kaggle_dfs.append(df)
            loaded += 1
        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    print(f"Loaded Kaggle: {loaded} files")

    if kaggle_dfs:
        return pd.concat(kaggle_dfs, ignore_index=True)
    return pd.DataFrame()


def merge_data(scraper_df: pd.DataFrame, kaggle_df: pd.DataFrame) -> pd.DataFrame:
    """Merge scraper and Kaggle data, preferring scraper for overlapping stocks."""
    if scraper_df.empty and kaggle_df.empty:
        return pd.DataFrame()

    if scraper_df.empty:
        return kaggle_df

    if kaggle_df.empty:
        return scraper_df

    # For scraper stocks, filter out Kaggle data and use scraper only
    scraper_symbols = set(scraper_df["symbol"].unique())

    # Filter out overlapping stocks from Kaggle data
    kaggle_filtered = kaggle_df[~kaggle_df["symbol"].isin(scraper_symbols)]

    # Combine: scraper data + non-overlapping Kaggle data
    merged = pd.concat([scraper_df, kaggle_filtered], ignore_index=True)

    print(f"Merged: {len(merged)} total rows")
    print(f"  - Scraper stocks: {scraper_symbols}")
    print(f"  - Kaggle stocks (non-overlapping): {kaggle_filtered['symbol'].nunique()}")

    return merged


def save_merged(df: pd.DataFrame, output_path: Path) -> None:
    """Save merged data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by symbol and date
    df = df.sort_values(["symbol", "Date"])

    # Ensure columns are in correct order
    df = df[STANDARD_COLUMNS]

    df.to_csv(output_path, index=False)
    print(f"Saved merged data: {output_path} ({len(df)} rows)")


def main():
    print("=" * 60)
    print("MERGING KAGGLE (2017) + SCRAPER (2026) DATA")
    print("=" * 60)

    # Load scraper data (AAPL, MSFT, NVDA up to 2026)
    print("\n[1] Loading scraper data...")
    scraper_df = load_scraper_data()
    print(f"   Scraper total: {len(scraper_df)} rows")

    if not scraper_df.empty:
        scraper_dates = pd.to_datetime(scraper_df["Date"], errors="coerce")
        print(f"   Scraper date range: {scraper_dates.min()} to {scraper_dates.max()}")

    # Load Kaggle data (excluding scraper stocks)
    print("\n[2] Loading Kaggle data...")
    kaggle_df = load_kaggle_data(exclude_symbols=SCRAPER_STOCKS)
    print(f"   Kaggle total: {len(kaggle_df)} rows")

    if not kaggle_df.empty:
        kaggle_dates = pd.to_datetime(kaggle_df["Date"], errors="coerce")
        print(f"   Kaggle date range: {kaggle_dates.min()} to {kaggle_dates.max()}")

    # Merge
    print("\n[3] Merging data...")
    merged_df = merge_data(scraper_df, kaggle_df)

    # Save
    print("\n[4] Saving...")
    save_merged(merged_df, OUTPUT_FILE)

    # Final stats
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    final_dates = pd.to_datetime(merged_df["Date"], errors="coerce")
    print(f"Total rows: {len(merged_df)}")
    print(f"Unique symbols: {merged_df['symbol'].nunique()}")
    print(f"Date range: {final_dates.min()} to {final_dates.max()}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
