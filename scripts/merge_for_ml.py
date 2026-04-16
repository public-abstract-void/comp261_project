#!/usr/bin/env python3
"""
Merge script for ML pipeline.
Combines Chris's historical data with Allan's current pipeline data.

Usage:
    python scripts/merge_for_ml.py
    python scripts/merge_for_ml.py --dry-run
    python scripts/merge_for_ml.py --chris-data <path> --allan-data <path> --output <path>
"""

import argparse
import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CHRIS_DATA = "data/processed/full_stocks_cleaned.csv"
ALLAN_DATA = "data/processed/training_2017_2026.csv"
OUTPUT = "data/processed/full_cleaned_data.csv"

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "symbol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Chris's data with Allan's data for ML pipeline."
    )
    parser.add_argument(
        "--chris-data",
        default=CHRIS_DATA,
        help=f"Path to Chris's cleaned data (default: {CHRIS_DATA})",
    )
    parser.add_argument(
        "--allan-data",
        default=ALLAN_DATA,
        help=f"Path to Allan's training data (default: {ALLAN_DATA})",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT,
        help=f"Path to output file (default: {OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview merge without writing output file",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing output file before overwriting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def validate_file(path: Path, name: str) -> pd.DataFrame:
    """Validate file exists and has required columns."""
    logger.info(f"[1/6] Validating {name} file: {path}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    if df.empty:
        raise ValueError(f"File is empty: {path}")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"   OK - {len(df):,} rows, columns: {list(df.columns)}")
    return df


def validate_data_quality(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate data quality and clean if needed."""
    logger.info(f"[2/6] Validating {name} data quality...")

    initial_rows = len(df)

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"   Found {invalid_dates:,} invalid dates - removing")
        df = df.dropna(subset=["Date"])

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if col in df.columns:
            invalid = (df[col] <= 0).sum()
            if invalid > 0:
                logger.warning(
                    f"   Found {invalid:,} invalid {col} values (<= 0) - removing"
                )
                df = df[df[col] > 0]

    invalid_volume = (df["Volume"] < 0).sum()
    if invalid_volume > 0:
        logger.warning(
            f"   Found {invalid_volume:,} invalid Volume values (< 0) - removing"
        )
        df = df[df["Volume"] >= 0]

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    removed = initial_rows - len(df)
    if removed > 0:
        logger.info(f"   Cleaned {removed:,} bad rows")
    logger.info(f"   OK - {len(df):,} valid rows remaining")

    return df


def merge_dataframes(chris_df: pd.DataFrame, allan_df: pd.DataFrame) -> pd.DataFrame:
    """Merge two dataframes, removing duplicates."""
    logger.info("[3/6] Merging dataframes...")

    logger.info(f"   Chris's data: {len(chris_df):,} rows")
    logger.info(f"   Allan's data: {len(allan_df):,} rows")

    common_cols = set(chris_df.columns) & set(allan_df.columns)
    all_cols = list(set(chris_df.columns) | set(allan_df.columns))

    logger.info(f"   Common columns: {sorted(common_cols)}")
    logger.info(f"   All columns: {sorted(all_cols)}")

    chris_subset = chris_df[list(common_cols)].copy()
    Allan_subset = allan_df[list(common_cols)].copy()

    combined = pd.concat([chris_subset, Allan_subset], ignore_index=True)
    logger.info(f"   Combined (before dedup): {len(combined):,} rows")

    logger.info("[4/6] Removing duplicates...")
    before_dedup = len(combined)

    combined = combined.drop_duplicates(subset=["Date", "symbol"], keep="first")

    duplicates_removed = before_dedup - len(combined)
    logger.info(f"   Duplicates removed: {duplicates_removed:,}")
    logger.info(f"   Final count: {len(combined):,} rows")

    for col in all_cols:
        if col not in combined.columns:
            combined[col] = pd.NA

    priority_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "symbol"]
    other_cols = [c for c in all_cols if c not in priority_cols]
    combined = combined[priority_cols + other_cols]

    combined = combined.sort_values(["symbol", "Date"]).reset_index(drop=True)

    return combined


def save_output(
    df: pd.DataFrame, output_path: Path, dry_run: bool, backup: bool
) -> None:
    """Save merged dataframe to CSV."""
    logger.info(f"[5/6] Saving output to {output_path}")

    if backup and output_path.exists():
        backup_path = output_path.with_suffix(
            f".backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        output_path.rename(backup_path)
        logger.info(f"   Backed up existing file to: {backup_path}")

    if dry_run:
        logger.info("   DRY RUN - not writing file")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"   OK - {len(df):,} rows written")

    file_hash = hashlib.md5(output_path.read_bytes()).hexdigest()[:12]
    logger.info(f"   File hash: {file_hash}")


def print_summary(
    chris_df: pd.DataFrame,
    allan_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Print final summary."""
    logger.info("[6/6] Summary")
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(
        f"Chris's data:      {len(chris_df):>12,} rows  ({pd.to_datetime(chris_df['Date']).min().date()} to {pd.to_datetime(chris_df['Date']).max().date()})"
    )
    print(
        f"Allan's data:      {len(allan_df):>12,} rows  ({pd.to_datetime(allan_df['Date']).min().date()} to {pd.to_datetime(allan_df['Date']).max().date()})"
    )
    print("-" * 60)
    print(f"Combined:          {len(chris_df) + len(allan_df):>12,} rows")
    print(
        f"Duplicates removed:{len(chris_df) + len(allan_df) - len(merged_df):>12,} rows"
    )
    print(f"Final dataset:     {len(merged_df):>12,} rows")
    print("-" * 60)
    print(f"Unique symbols:    {merged_df['symbol'].nunique():>12,}")
    print(
        f"Date range:        {merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}"
    )
    print("-" * 60)
    print(f"Output file:       {output_path}")
    print("=" * 60)
    print("STATUS: SUCCESS ✓")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("ML DATA MERGE SCRIPT")
    print("=" * 60)

    chris_path = Path(args.chris_data)
    Allan_path = Path(args.allan_data)
    output_path = Path(args.output)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be written ***\n")

    chris_df = validate_file(chris_path, "Chris")
    Allan_df = validate_file(Allan_path, "Allan")

    chris_df = validate_data_quality(chris_df, "Chris")
    Allan_df = validate_data_quality(Allan_df, "Allan")

    merged_df = merge_dataframes(chris_df, Allan_df)

    save_output(merged_df, output_path, args.dry_run, args.backup)

    print_summary(chris_df, Allan_df, merged_df, output_path)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)
