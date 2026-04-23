"""
bar_loader.py
-------------
Unified interface for daily and intraday OHLCV bars.
Intraday is scaffold only — schema and loader API are defined
so the ML teammate can plug in a real feed later without
changing any downstream code.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume", "symbol"}


class BarFrequency(str, Enum):
    DAILY      = "daily"
    INTRADAY_1M  = "1m"
    INTRADAY_5M  = "5m"
    INTRADAY_15M = "15m"
    INTRADAY_1H  = "1h"


@dataclass
class BarLoaderConfig:
    data_dir:        Path = Path("data/processed")
    daily_filename:  str  = "full_merged.parquet"
    intraday_dir:    str  = "intraday"


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Date to UTC-aware datetime and sort ascending."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if df fails basic schema checks."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df["Date"].isnull().any():
        raise ValueError("Null timestamps detected.")
    if df["Close"].isnull().any():
        raise ValueError("Null close prices detected.")
    if (df["High"] < df["Low"]).any():
        raise ValueError("high < low detected — data integrity error.")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Timestamps not sorted — lookahead risk.")


class BarLoader:
    """
    Load daily or intraday bars with a consistent interface.

    Usage:
        loader = BarLoader()
        df = loader.load(symbols=["AAPL"], frequency=BarFrequency.DAILY)
        df = loader.load(symbols=["AAPL"], frequency=BarFrequency.INTRADAY_5M)
    """

    def __init__(self, config: Optional[BarLoaderConfig] = None):
        self.cfg = config or BarLoaderConfig()

    def load(
        self,
        symbols:   Optional[list[str]] = None,
        frequency: BarFrequency = BarFrequency.DAILY,
        start:     Optional[str] = None,
        end:       Optional[str] = None,
    ) -> pd.DataFrame:
        if frequency == BarFrequency.DAILY:
            df = self._load_daily(symbols)
        else:
            df = self._load_intraday(symbols, frequency)

        df = normalize_timestamps(df)
        validate_schema(df)

        if start:
            df = df[df["Date"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df["Date"] <= pd.Timestamp(end, tz="UTC")]

        logger.info("Loaded %d rows | freq=%s | symbols=%s",
                    len(df), frequency.value, symbols or "ALL")
        return df

    def available_symbols(self) -> list[str]:
        try:
            df = self._load_daily(None)
            return sorted(df["symbol"].unique().tolist())
        except FileNotFoundError:
            return []

    def _load_daily(self, symbols: Optional[list[str]]) -> pd.DataFrame:
        path = self.cfg.data_dir / self.cfg.daily_filename
        if not path.exists():
            raise FileNotFoundError(
                f"Daily data not found: {path}\n"
                "Run: python scripts/daily_update.py rebuild"
            )
        df = pd.read_parquet(path)
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        return df

    def _load_intraday(
        self,
        symbols: Optional[list[str]],
        frequency: BarFrequency,
    ) -> pd.DataFrame:
        """
        Scaffold: returns empty DataFrame with correct schema.
        Replace with real feed (Polygon, Alpaca) when available.
        To add real data: drop parquet files into
        data/processed/intraday/{frequency}/{symbol}.parquet
        """
        logger.warning(
            "Intraday data (%s) not yet available — returning empty scaffold. "
            "Feed: add parquet files to data/processed/intraday/%s/",
            frequency.value, frequency.value,
        )
        return pd.DataFrame({
            "Date":    pd.Series(dtype="datetime64[ns, UTC]"),
            "Open":    pd.Series(dtype=float),
            "High":    pd.Series(dtype=float),
            "Low":     pd.Series(dtype=float),
            "Close":   pd.Series(dtype=float),
            "Volume":  pd.Series(dtype=float),
            "OpenInt": pd.Series(dtype=float),
            "symbol":  pd.Series(dtype=str),
            "type":    pd.Series(dtype=str),
        })