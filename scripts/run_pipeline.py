from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import pandas as pd

# Allow direct script execution without package installation.
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from day_trading_bot.data.contract import validate_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Professional-grade pipeline: merge -> clean -> validate -> (optional) parquet + metadata."
    )
    parser.add_argument(
        "--input-dir",
        default="/Users/allanodora/Downloads/archive/Stocks",
        help="Directory with raw stock .us.txt files",
    )
    parser.add_argument(
        "--merged-csv",
        default="data/processed/full_stocks_merged.csv",
        help="Path for merged raw CSV output",
    )
    parser.add_argument(
        "--cleaned-csv",
        default="data/processed/full_stocks_cleaned.csv",
        help="Path for cleaned CSV output",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Also write cleaned data to parquet",
    )
    parser.add_argument(
        "--cleaned-parquet",
        default="data/processed/full_stocks_cleaned.parquet",
        help="Path for cleaned parquet output",
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/run_metadata.json",
        help="Path for run metadata JSON",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge step if merged CSV already exists",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip clean step if cleaned CSV already exists",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_all(input_dir: Path, merged_csv: Path) -> dict:
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    if merged_csv.exists():
        merged_csv.unlink()

    files = sorted(input_dir.glob("*.us.txt"))
    written_rows = 0
    processed_files = 0
    skipped_empty = 0
    skipped_error = 0
    header_written = False

    for f in files:
        if f.stat().st_size == 0:
            skipped_empty += 1
            continue
        try:
            df = pd.read_csv(f)
            df["symbol"] = f.name[:-7].upper()
            df.to_csv(merged_csv, mode="a", header=not header_written, index=False)
            header_written = True
            written_rows += len(df)
            processed_files += 1
        except Exception:
            skipped_error += 1

    return {
        "total_files": len(files),
        "processed_files": processed_files,
        "skipped_empty": skipped_empty,
        "skipped_error": skipped_error,
        "written_rows": written_rows,
    }


def run_cleaner(merged_csv: Path, cleaned_csv: Path) -> None:
    cmd = [
        ".venv/bin/python",
        "src/cleaning.py",
        str(merged_csv),
        "-o",
        str(cleaned_csv),
    ]
    subprocess.run(cmd, check=True)


def write_parquet(cleaned_csv: Path, cleaned_parquet: Path) -> None:
    # Heavy but straightforward parquet output for professional workflows.
    df = pd.read_csv(cleaned_csv)
    df.to_parquet(cleaned_parquet, index=False)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    merged_csv = Path(args.merged_csv)
    cleaned_csv = Path(args.cleaned_csv)
    cleaned_parquet = Path(args.cleaned_parquet)
    metadata_path = Path(args.metadata)

    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "merged_csv": str(merged_csv),
        "cleaned_csv": str(cleaned_csv),
        "cleaned_parquet": str(cleaned_parquet) if args.write_parquet else "",
    }

    if not args.skip_merge:
        metadata["merge"] = merge_all(input_dir, merged_csv)
    else:
        metadata["merge"] = {"skipped": True}

    if not args.skip_clean:
        run_cleaner(merged_csv, cleaned_csv)
    else:
        metadata["clean"] = {"skipped": True}

    validation = validate_csv(str(cleaned_csv))
    metadata["validation"] = validation.to_dict()

    metadata["cleaned_sha256"] = sha256_file(cleaned_csv)

    if args.write_parquet:
        write_parquet(cleaned_csv, cleaned_parquet)
        metadata["cleaned_parquet_sha256"] = sha256_file(cleaned_parquet)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Pipeline complete.")
    print(f"Metadata: {metadata_path}")
    print(f"Validation OK: {metadata['validation']['ok']}")


if __name__ == "__main__":
    main()
