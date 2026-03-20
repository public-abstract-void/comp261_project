from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple


BASE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt", "symbol"]
OUTPUT_COLUMNS = BASE_COLUMNS + [
    "type",
    "target_up_1d",
    "target_up_5d",
    "target_up_10d",
]


@dataclass
class BuildStats:
    input_rows: int = 0
    output_rows: int = 0
    symbols_seen: int = 0
    dropped_tail_rows: int = 0
    parse_errors: int = 0
    sort_violations: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a training-ready CSV by adding 'type' and 1d/5d/10d up/down targets. "
            "Assumes rows are sorted by symbol then Date."
        )
    )
    p.add_argument(
        "--input-csv",
        default="data/processed/full_stocks_cleaned.csv",
        help="Cleaned input CSV (must include Date/Open/High/Low/Close/Volume/OpenInt/symbol)",
    )
    p.add_argument(
        "--output-csv",
        default="data/processed/training_input.csv",
        help="Output training CSV",
    )
    p.add_argument(
        "--type",
        default="stock",
        help="Value for the 'type' column (default: stock)",
    )
    p.add_argument(
        "--keep-unlabeled",
        action="store_true",
        help="Keep last rows per symbol with empty targets (default: drop them)",
    )
    return p.parse_args()


def _float_or_none(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def build_training_table(input_csv: Path, output_csv: Path, type_value: str, keep_unlabeled: bool) -> BuildStats:
    stats = BuildStats()

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", newline="", encoding="utf-8") as fin, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")

        missing = [c for c in BASE_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")

        writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        current_symbol: Optional[str] = None
        last_date_for_symbol: Optional[str] = None
        buf: Deque[Dict[str, str]] = deque()

        def flush_symbol_buffer(final: bool) -> None:
            # Drop tail rows that can't be labeled unless keep_unlabeled is set.
            nonlocal buf
            if keep_unlabeled:
                while buf:
                    row = buf.popleft()
                    row_out = dict(row)
                    row_out["type"] = type_value
                    row_out["target_up_1d"] = ""
                    row_out["target_up_5d"] = ""
                    row_out["target_up_10d"] = ""
                    writer.writerow(row_out)
                    stats.output_rows += 1
                return

            stats.dropped_tail_rows += len(buf)
            buf.clear()

        def emit_labeled_row(base_row: Dict[str, str], f1: Dict[str, str], f5: Dict[str, str], f10: Dict[str, str]) -> None:
            c0 = _float_or_none(base_row["Close"])
            c1 = _float_or_none(f1["Close"])
            c5 = _float_or_none(f5["Close"])
            c10 = _float_or_none(f10["Close"])
            if c0 is None or c1 is None or c5 is None or c10 is None:
                stats.parse_errors += 1
                return

            out = dict(base_row)
            out["type"] = type_value
            out["target_up_1d"] = "1" if c1 > c0 else "0"
            out["target_up_5d"] = "1" if c5 > c0 else "0"
            out["target_up_10d"] = "1" if c10 > c0 else "0"
            writer.writerow(out)
            stats.output_rows += 1

        for row in reader:
            stats.input_rows += 1
            sym = row["symbol"]
            date = row["Date"]

            # Symbol boundary
            if current_symbol is None:
                current_symbol = sym
                stats.symbols_seen += 1
                last_date_for_symbol = None

            if sym != current_symbol:
                flush_symbol_buffer(final=True)
                buf.clear()
                current_symbol = sym
                stats.symbols_seen += 1
                last_date_for_symbol = None

            # Basic sort check (string compare works for YYYY-MM-DD)
            if last_date_for_symbol is not None and date < last_date_for_symbol:
                stats.sort_violations += 1
            last_date_for_symbol = date

            # Keep only the base columns (prevents unexpected extra columns leaking)
            base_row = {k: row[k] for k in BASE_COLUMNS}
            buf.append(base_row)

            # Need 10 future rows to label the oldest row in the buffer
            if len(buf) > 10:
                base = buf.popleft()
                f1 = buf[0]
                f5 = buf[4]
                f10 = buf[9]
                emit_labeled_row(base, f1, f5, f10)

        # End of file flush
        if current_symbol is not None:
            flush_symbol_buffer(final=True)

    return stats


def main() -> None:
    args = parse_args()
    stats = build_training_table(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        type_value=args.type,
        keep_unlabeled=args.keep_unlabeled,
    )

    print("Training table built.")
    print(f"Input rows: {stats.input_rows}")
    print(f"Output rows: {stats.output_rows}")
    print(f"Symbols seen: {stats.symbols_seen}")
    print(f"Dropped tail rows: {stats.dropped_tail_rows}")
    print(f"Parse errors: {stats.parse_errors}")
    print(f"Sort violations: {stats.sort_violations}")


if __name__ == "__main__":
    main()
