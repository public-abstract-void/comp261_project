from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Kaggle stock files to S&P 500 symbols, while handling empty files automatically."
        )
    )
    parser.add_argument(
        "--stocks-dir",
        default="data/raw/Stocks",
        help="Directory with Kaggle stock files (default: data/raw/Stocks)",
    )
    parser.add_argument(
        "--symbols-file",
        default="data/reference/sp500_symbols_template.csv",
        help="CSV with a Symbol column (default: data/reference/sp500_symbols_template.csv)",
    )
    parser.add_argument(
        "--output-file",
        default="data/processed/sp500_filtered_files.csv",
        help="Output CSV for matched valid files (default: data/processed/sp500_filtered_files.csv)",
    )
    parser.add_argument(
        "--empty-threshold",
        type=int,
        default=50,
        help=(
            "If total empty files are above this, script switches to explicit skip/report mode "
            "(default: 50)."
        ),
    )
    parser.add_argument(
        "--copy-dir",
        default="",
        help="Optional directory to copy filtered valid files into.",
    )
    return parser.parse_args()


def extract_ticker(file_name: str) -> str:
    lower_name = file_name.lower()
    if not lower_name.endswith(".us.txt"):
        return ""
    return file_name[:-7].upper()


def load_symbols(symbols_file: Path) -> set[str]:
    if not symbols_file.exists():
        raise FileNotFoundError(f"Symbols file not found: {symbols_file}")

    df = pd.read_csv(symbols_file)
    if "Symbol" not in df.columns:
        first_col = df.columns[0]
        symbols = df[first_col]
    else:
        symbols = df["Symbol"]

    return {
        str(value).upper().strip()
        for value in symbols.dropna().tolist()
        if str(value).strip()
    }


def main() -> None:
    args = parse_args()

    stocks_dir = Path(args.stocks_dir)
    symbols_file = Path(args.symbols_file)
    output_file = Path(args.output_file)

    if not stocks_dir.exists():
        print(f"Error: stocks directory not found: {stocks_dir}")
        return

    symbols = load_symbols(symbols_file)
    if not symbols:
        print("Error: no symbols found in symbols file.")
        return

    stock_files = sorted(stocks_dir.glob("*.us.txt"))

    records: list[dict[str, str]] = []
    empty_files: list[Path] = []
    matched_empty_files: list[Path] = []

    for file_path in stock_files:
        ticker = extract_ticker(file_path.name)
        if not ticker:
            continue

        if file_path.stat().st_size == 0:
            empty_files.append(file_path)
            if ticker in symbols:
                matched_empty_files.append(file_path)
            continue

        if ticker in symbols:
            records.append(
                {
                    "ticker": ticker,
                    "file_name": file_path.name,
                    "file_path": str(file_path.resolve()),
                }
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records).sort_values("ticker") if records else pd.DataFrame(
        columns=["ticker", "file_name", "file_path"]
    )
    out_df.to_csv(output_file, index=False)

    mode = "ignore" if len(empty_files) <= args.empty_threshold else "skip"

    print("\n--- Filter Summary ---")
    print(f"Stocks directory: {stocks_dir.resolve()}")
    print(f"Symbols loaded: {len(symbols)}")
    print(f"Total stock files found: {len(stock_files)}")
    print(f"Total empty files found: {len(empty_files)}")
    print(f"Empty-file handling mode: {mode}")
    print(f"Matched valid S&P 500 files: {len(out_df)}")
    print(f"Output file: {output_file.resolve()}")

    if mode == "skip" and empty_files:
        skipped_report = output_file.parent / "empty_files_skipped.csv"
        pd.DataFrame(
            {
                "file_name": [p.name for p in empty_files],
                "file_path": [str(p.resolve()) for p in empty_files],
            }
        ).to_csv(skipped_report, index=False)
        print(f"Skipped empty-file report: {skipped_report.resolve()}")

    if matched_empty_files:
        print(f"Matched symbols with empty files: {len(matched_empty_files)}")

    if args.copy_dir:
        copy_dir = Path(args.copy_dir)
        copy_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for row in records:
            source = Path(row["file_path"])
            target = copy_dir / row["file_name"]
            shutil.copy2(source, target)
            copied += 1
        print(f"Copied files: {copied}")
        print(f"Copy directory: {copy_dir.resolve()}")


if __name__ == "__main__":
    main()
