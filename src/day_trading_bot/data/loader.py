from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_stock_file_path(data_dir: Path, ticker: str) -> Path:
    """Build stock file path, supporting case variants like aapl.us.txt."""
    ticker = ticker.upper().strip()
    candidates = [
        data_dir / f"{ticker}.us.txt",
        data_dir / f"{ticker.lower()}.us.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_stock_csv(file_path: Path) -> pd.DataFrame:
    """Load one stock CSV and parse Date column."""
    if not file_path.exists():
        raise FileNotFoundError(f"Stock file not found: {file_path}")

    df = pd.read_csv(file_path)

    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column but none was found.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """Return missing-value count per column."""
    return df.isna().sum()
