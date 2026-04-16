from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from .config import PipelineConfig, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    total_records: int
    valid_records: int
    invalid_records: int
    schema_errors: int
    range_errors: int
    null_errors: int
    duplicate_errors: int


class DataValidator:
    STANDARD_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "symbol"]
    NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
    REQUIRED_COLUMNS = ["Date", "Close", "symbol"]

    def __init__(self, config: PipelineConfig):
        self.config = config

    def validate(self, df: pd.DataFrame, strict: bool = True) -> ValidationResult:
        errors = []
        warnings = []

        if df.empty:
            return ValidationResult(
                valid=True,
                records_checked=0,
                records_valid=0,
                errors=[],
                warnings=["Empty dataframe provided"],
            )

        schema_result = self._validate_schema(df, strict)
        errors.extend(schema_result.errors)
        warnings.extend(schema_result.warnings)

        range_result = self._validate_ranges(df)
        errors.extend(range_result.errors)
        warnings.extend(range_result.warnings)

        null_result = self._validate_nulls(df, strict)
        errors.extend(null_result.errors)
        warnings.extend(null_result.warnings)

        dup_result = self._validate_duplicates(df)
        errors.extend(dup_result.errors)
        warnings.extend(dup_result.warnings)

        final_valid = len(errors) == 0

        return ValidationResult(
            valid=final_valid,
            errors=errors,
            warnings=warnings,
            records_checked=len(df),
            records_valid=len(df) - len(errors),
        )

    def _validate_schema(self, df: pd.DataFrame, strict: bool) -> ValidationResult:
        errors = []
        warnings = []

        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        if strict:
            extra_cols = set(df.columns) - set(self.STANDARD_COLUMNS)
            if extra_cols:
                warnings.append(f"Extra columns (ignored): {extra_cols}")

        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' is not numeric")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            records_checked=len(df),
            records_valid=len(df) - len(errors),
        )

    def _validate_ranges(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        for col in self.NUMERIC_COLUMNS:
            if col not in df.columns:
                continue

            if (df[col] <= 0).any():
                non_positive = (df[col] <= 0).sum()
                errors.append(f"{col}: {non_positive} non-positive values")

            if col == "Volume":
                below_min = (df[col] < self.config.min_volume).sum()
                above_max = (df[col] > self.config.max_volume).sum()
                if below_min > 0:
                    warnings.append(
                        f"Volume: {below_min} values below min ({self.config.min_volume})"
                    )
                if above_max > 0:
                    warnings.append(
                        f"Volume: {above_max} values above max ({self.config.max_volume})"
                    )

        if "High" in df.columns and "Low" in df.columns:
            invalid_hl = (df["High"] < df["Low"]).sum()
            if invalid_hl > 0:
                errors.append(f"High < Low: {invalid_hl} records")

        if "Open" in df.columns and "Close" in df.columns:
            pass

        if "High" in df.columns and "Close" in df.columns:
            pass

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            records_checked=len(df),
            records_valid=len(df) - len(errors),
        )

    def _validate_nulls(self, df: pd.DataFrame, strict: bool) -> ValidationResult:
        errors = []
        warnings = []

        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                continue
            null_count = df[col].isna().sum()
            if null_count > 0:
                if strict:
                    errors.append(f"{col}: {null_count} null values")
                else:
                    warnings.append(f"{col}: {null_count} null values")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            records_checked=len(df),
            records_valid=len(df) - len(errors),
        )

    def _validate_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        dupes = df.duplicated(subset=["symbol", "Date"], keep=False).sum()
        if dupes > 0:
            warnings.append(f"Duplicate (symbol, Date) pairs: {dupes}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            records_checked=len(df),
            records_valid=len(df) - len(errors),
        )

    def validate_batch(
        self, df: pd.DataFrame, batch_size: int = 10000
    ) -> ValidationResult:
        if len(df) <= batch_size:
            return self.validate(df)

        results = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            results.append(self.validate(batch, strict=False))

        combined = results[0]
        for r in results[1:]:
            combined = combined.merge(r)

        return combined

    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                df = df.dropna(subset=[col])

        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df[df[col] > 0] if col != "Volume" else df

        df = df.drop_duplicates(subset=["symbol", "Date"], keep="last")

        return df
