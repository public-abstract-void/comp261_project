#
# This file cleans an input file.
# Usage: python <input file> <-o output file>
#
# @Author(s): Christopher Yalch
# 

import argparse
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

def parse_cli():
    parser = argparse.ArgumentParser(
        description="Data cleaning program for our day trading bot."
    )

    # Required positional argument
    parser.add_argument(
        "input",
        help="Input file path"
    )

    # Optional -o / --output argument
    parser.add_argument(
        "-o", "--output",
        help="Optional output file path",
        default="cleaned_data.csv"
    )

    args = parser.parse_args()
    return args

    # # Example behavior
    # if args.output:
    #     try:
    #         with open(args.output, "w") as f:
    #             f.write(f"Input was: {args.input}\n")
    #         print(f"Written result to {args.output}")

    #     except OSError as e:
    #         print(f"Error writing to file: {e}", file=sys.stderr)
    #         sys.exit(1)
    # else:
    #     print(f"Input was: {args.input}")

def print_stage(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def clean_data(input_file, output_file, drop_zero_volume=False):

    # --------------------------------------------------
    # STAGE 1 — Load Dataset
    # --------------------------------------------------
    print_stage("STAGE 1: LOADING DATA")

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Failed to load file: {e}")
        sys.exit(1)

    print(f"Initial shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # --------------------------------------------------
    # STAGE 2 — Basic Info
    # --------------------------------------------------
    print_stage("STAGE 2: DATASET OVERVIEW")

    print("Columns:")
    print(df.columns.tolist())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    # --------------------------------------------------
    # STAGE 3 — Remove Infinite Values
    # --------------------------------------------------
    print_stage("STAGE 3: REMOVING INFINITE VALUES")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in tqdm(numeric_cols, desc="Checking numeric columns"):
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    print("Infinite values replaced with NaN.")

    # --------------------------------------------------
    # STAGE 4 — Drop Missing Values
    # --------------------------------------------------
    print_stage("STAGE 4: DROPPING MISSING VALUES")

    before_rows = len(df)
    df = df.dropna()
    after_rows = len(df)

    print(f"Rows removed: {before_rows - after_rows}")
    print(f"New shape: {df.shape}")

    # --------------------------------------------------
    # STAGE 5 — Drop Duplicates
    # --------------------------------------------------
    print_stage("STAGE 5: REMOVING DUPLICATES")

    before_rows = len(df)
    df = df.drop_duplicates()
    after_rows = len(df)

    print(f"Duplicate rows removed: {before_rows - after_rows}")
    print(f"New shape: {df.shape}")

    # --------------------------------------------------
    # STAGE 5B — Fix High/Low Inconsistencies
    # --------------------------------------------------
    if "High" in df.columns and "Low" in df.columns:
        print_stage("STAGE 5B: DROPPING HIGH < LOW ROWS")

        before_rows = len(df)
        df = df[df["High"] >= df["Low"]]
        after_rows = len(df)

        print(f"Rows removed (High < Low): {before_rows - after_rows}")
        print(f"New shape: {df.shape}")

    # --------------------------------------------------
    # STAGE 6 — Optional: Remove Zero Volume
    # --------------------------------------------------
    if drop_zero_volume and "Volume" in df.columns:
        print_stage("STAGE 6: REMOVING ZERO VOLUME ROWS")

        before_rows = len(df)
        df = df[df["Volume"] > 0]
        after_rows = len(df)

        print(f"Zero-volume rows removed: {before_rows - after_rows}")
        print(f"New shape: {df.shape}")

    # --------------------------------------------------
    # STAGE 7 — Convert Date Column
    # --------------------------------------------------
    if "Date" in df.columns:
        print_stage("STAGE 7: CONVERTING DATE COLUMN")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        print("Date column converted to datetime.")

    # --------------------------------------------------
    # STAGE 8 — Sort Data
    # --------------------------------------------------
    if "symbol" in df.columns and "Date" in df.columns:
        print_stage("STAGE 8: SORTING DATA")

        df = df.sort_values(["symbol", "Date"])
        print("Data sorted by symbol and Date.")

    # --------------------------------------------------
    # STAGE 9 — Final Report
    # --------------------------------------------------
    print_stage("FINAL DATA SUMMARY")

    print(f"Final shape: {df.shape}")
    print(f"Final memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # --------------------------------------------------
    # STAGE 10 — Save Output
    # --------------------------------------------------
    print_stage("SAVING CLEANED DATA")

    try:
        df.to_csv(output_file, index=False)
        print(f"Cleaned file saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save file: {e}")
        sys.exit(1)


def main():
    args = parse_cli()
    clean_data(args.input, args.output, True)

if __name__ == "__main__":
    main()
