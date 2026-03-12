from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt", "symbol"]


@dataclass
class ValidationResult:
    total_rows: int
    missing_columns: List[str]
    invalid_date_rows: int
    missing_value_rows: int
    negative_volume_rows: int
    high_lt_low_rows: int

    @property
    def ok(self) -> bool:
        return (
            not self.missing_columns
            and self.invalid_date_rows == 0
            and self.missing_value_rows == 0
            and self.negative_volume_rows == 0
            and self.high_lt_low_rows == 0
        )

    def to_dict(self) -> Dict[str, int | List[str] | bool]:
        return {
            "total_rows": self.total_rows,
            "missing_columns": self.missing_columns,
            "invalid_date_rows": self.invalid_date_rows,
            "missing_value_rows": self.missing_value_rows,
            "negative_volume_rows": self.negative_volume_rows,
            "high_lt_low_rows": self.high_lt_low_rows,
            "ok": self.ok,
        }


def validate_csv(path: str, chunksize: int = 1_000_000) -> ValidationResult:
    total_rows = 0
    invalid_date_rows = 0
    missing_value_rows = 0
    negative_volume_rows = 0
    high_lt_low_rows = 0

    # Read header once for column check
    header_df = pd.read_csv(path, nrows=1)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in header_df.columns]

    if missing_columns:
        return ValidationResult(
            total_rows=0,
            missing_columns=missing_columns,
            invalid_date_rows=0,
            missing_value_rows=0,
            negative_volume_rows=0,
            high_lt_low_rows=0,
        )

    for chunk in pd.read_csv(path, chunksize=chunksize):
        total_rows += len(chunk)

        # Date validity
        dates = pd.to_datetime(chunk["Date"], errors="coerce")
        invalid_date_rows += int(dates.isna().sum())

        # Missing values
        missing_value_rows += int(chunk.isna().any(axis=1).sum())

        # Negative volume
        negative_volume_rows += int((chunk["Volume"] < 0).sum())

        # High < Low
        high_lt_low_rows += int((chunk["High"] < chunk["Low"]).sum())

    return ValidationResult(
        total_rows=total_rows,
        missing_columns=missing_columns,
        invalid_date_rows=invalid_date_rows,
        missing_value_rows=missing_value_rows,
        negative_volume_rows=negative_volume_rows,
        high_lt_low_rows=high_lt_low_rows,
    )
