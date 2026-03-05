#
# Loading functions for our datasets.
# Usage: daytrading_loading.py
#
# @Author(s): Christopher Yalch
#

import kagglehub
#!/usr/bin/env python3

import os
import pandas as pd
import argparse
import sys


# Collects desired files
def collect_files(base_path, data_type):
    paths = []

    if data_type in ("stocks", "both"):
        paths.append(("Stock", os.path.join(base_path, "Stocks")))

    if data_type in ("etfs", "both"):
        paths.append(("ETF", os.path.join(base_path, "ETFs")))

    return paths

# Extracts the symbol from file name
def extract_symbol(filename):
    """
    Converts:
        a.us.txt   -> A
        msft.us.txt -> MSFT
    """
    if filename.endswith(".us.txt"):
        return filename.replace(".us.txt", "").upper()
    else:
        return os.path.splitext(filename)[0].upper()

# Merges all files into one beeg file
def merge_data(base_path, data_type, output_file):
    targets = collect_files(base_path, data_type)

    if not targets:
        print("No valid data type selected.")
        sys.exit(1)

    all_dfs = []

    for label, directory in targets:
        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            continue

        files = sorted(f for f in os.listdir(directory) if f.endswith(".txt"))

        for filename in files:
            file_path = os.path.join(directory, filename)

            try:
                df = pd.read_csv(file_path)
                df["symbol"] = extract_symbol(filename)
                df["type"] = label.lower()
                all_dfs.append(df)

            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    if not all_dfs:
        print("No data loaded.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_file, index=False)

    print(f"Saved merged file to: {output_file}")
    print(f"Total rows: {len(combined):,}")

def load_dataset():
    # Download latest version
    path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")

    print("Path to dataset files:", path)

    return path

def main():
    path = load_dataset()

    parser = argparse.ArgumentParser(
    description="Merge Kaggle price-volume TXT files into one CSV."
    )

    parser.add_argument(
        "-t", "--type",
        choices=["stocks", "etfs", "both"],
        default="both",
        help="Choose which data to merge (default: both)"
    )

    parser.add_argument(
        "-o", "--output",
        default="merged_data.csv",
        help="Output CSV filename"
    )

    args = parser.parse_args()

    merge_data(path, args.type, args.output)

if __name__ == "__main__":
    main()