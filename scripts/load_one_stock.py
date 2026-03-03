from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct script execution without package installation during prototyping.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from day_trading_bot.data.loader import (
    build_stock_file_path,
    load_stock_csv,
    missing_value_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Guardrail precheck: load one stock file and validate basic data quality before full runs."
        )
    )
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument(
        "--data-dir",
        default="data/raw/Stocks",
        help="Directory containing Kaggle stock files (default: data/raw/Stocks)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    file_path = build_stock_file_path(data_dir=data_dir, ticker=args.ticker)

    print(f"Loading file: {file_path}")
    try:
        df = load_stock_csv(file_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    print("\n--- Basic Overview ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\n--- Columns ---")
    print(list(df.columns))

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Date Checks ---")
    invalid_dates = df["Date"].isna().sum()
    print(f"Invalid/Unparsed dates: {invalid_dates}")
    if len(df) > 0:
        print(f"Min date: {df['Date'].min()}")
        print(f"Max date: {df['Date'].max()}")

    print("\n--- Missing Values (per column) ---")
    missing = missing_value_report(df)
    print(missing)

    print("\n--- Guardrail Checks ---")
    expected_columns = {"Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"}
    missing_columns = expected_columns - set(df.columns)
    negative_volume = int((df["Volume"] < 0).sum()) if "Volume" in df.columns else -1
    total_missing = int(missing.sum())

    checks_ok = True
    if missing_columns:
        checks_ok = False
        print(f"FAIL: missing expected columns: {sorted(missing_columns)}")
    if invalid_dates > 0:
        checks_ok = False
        print(f"FAIL: invalid dates found: {invalid_dates}")
    if total_missing > 0:
        checks_ok = False
        print(f"FAIL: missing values found: {total_missing}")
    if negative_volume > 0:
        checks_ok = False
        print(f"FAIL: negative volume rows found: {negative_volume}")
    if checks_ok:
        print("PASS: precheck looks good for this ticker.")

    print("\n--- Head (first 5 rows) ---")
    print(df.head())


if __name__ == "__main__":
    main()
