"""Tests for BarLoader interface."""
import pytest
import pandas as pd
from src.day_trading_bot.data.bar_loader import (
    BarLoader, BarFrequency, normalize_timestamps, validate_schema
)


def make_df(n=10):
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "Date":    dates,
        "Open":    [100.0] * n,
        "High":    [110.0] * n,
        "Low":     [90.0]  * n,
        "Close":   [105.0] * n,
        "Volume":  [1e6]   * n,
        "OpenInt": [0.0]   * n,
        "symbol":  "AAPL",
        "type":    "stock",
    })


def test_normalize_timestamps_sorts():
    df = make_df()
    df = df.iloc[::-1].reset_index(drop=True)
    out = normalize_timestamps(df)
    assert out["Date"].is_monotonic_increasing


def test_normalize_timestamps_utc():
    df = make_df()
    out = normalize_timestamps(df)
    assert str(out["Date"].dt.tz) == "UTC"


def test_validate_schema_passes_valid():
    df = make_df()
    validate_schema(df)  # should not raise


def test_validate_schema_catches_null_close():
    import numpy as np
    df = make_df()
    df.loc[0, "Close"] = np.nan
    with pytest.raises(ValueError, match="Null close"):
        validate_schema(df)


def test_validate_schema_catches_high_lt_low():
    df = make_df()
    df.loc[0, "High"] = 1.0
    df.loc[0, "Low"]  = 999.0
    with pytest.raises(ValueError, match="high < low"):
        validate_schema(df)


def test_intraday_scaffold_returns_empty():
    loader = BarLoader()
    df = loader.load(frequency=BarFrequency.INTRADAY_5M)
    assert len(df) == 0
    assert "Close" in df.columns


def test_intraday_scaffold_has_correct_columns():
    loader = BarLoader()
    df = loader.load(frequency=BarFrequency.INTRADAY_1M)
    required = {"Date","Open","High","Low","Close","Volume","symbol"}
    assert required.issubset(set(df.columns))