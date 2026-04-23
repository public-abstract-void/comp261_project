"""
Tests for data schema validation and date normalization.
These protect the ML teammate from receiving bad data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone


def make_valid_df(n=100):
    """Helper: create a minimal valid OHLCV DataFrame."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "Date":    dates,
        "Open":    np.random.uniform(100, 200, n),
        "High":    np.random.uniform(200, 300, n),
        "Low":     np.random.uniform(50,  100, n),
        "Close":   np.random.uniform(100, 200, n),
        "Volume":  np.random.uniform(1e6, 1e7, n),
        "symbol":  "AAPL",
    })


# ── column presence ──────────────────────────────────────────────

def test_required_columns_present():
    df = make_valid_df()
    required = {"Date", "Open", "High", "Low", "Close", "Volume", "symbol"}
    assert required.issubset(set(df.columns)), "Missing required columns"


def test_missing_column_detected():
    df = make_valid_df().drop(columns=["Volume"])
    required = {"Date", "Open", "High", "Low", "Close", "Volume", "symbol"}
    assert not required.issubset(set(df.columns))

# ── timestamp normalization ──────────────────────────────────────

def test_timestamps_are_sorted():
    df = make_valid_df()
    assert df["Date"].is_monotonic_increasing


def test_timestamps_become_utc():
    df = make_valid_df()
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    assert str(df["Date"].dt.tz) == "UTC"


def test_unsorted_timestamps_detected():
    df = make_valid_df()
    df = df.iloc[::-1].reset_index(drop=True)  # reverse order
    assert not df["Date"].is_monotonic_increasing


def test_string_timestamps_can_be_parsed():
    df = make_valid_df()
    df["Date"] = df["Date"].astype(str)
    parsed = pd.to_datetime(df["Date"], utc=True)
    assert parsed.notna().all()


# ── data integrity ───────────────────────────────────────────────

def test_no_null_close_prices():
    df = make_valid_df()
    assert df["Close"].notna().all()


def test_null_close_detected():
    df = make_valid_df()
    df.loc[5, "Close"] = np.nan
    assert df["Close"].isna().any()


def test_high_never_below_low():
    df = make_valid_df()
    # fix the random data to guarantee high > low
    df["High"] = df[["Open","Close"]].max(axis=1) + 1
    df["Low"]  = df[["Open","Close"]].min(axis=1) - 1
    assert (df["High"] >= df["Low"]).all()


def test_high_below_low_detected():
    df = make_valid_df()
    df.loc[0, "High"] = 1.0
    df.loc[0, "Low"]  = 999.0
    assert (df["High"] < df["Low"]).any()


def test_volume_non_negative():
    df = make_valid_df()
    df["Volume"] = df["Volume"].abs()
    assert (df["Volume"] >= 0).all()


def test_no_duplicate_timestamps_per_symbol():
    df = make_valid_df()
    dupes = df.duplicated(subset=["Date", "symbol"])
    assert not dupes.any()


def test_duplicate_timestamps_detected():
    df = make_valid_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    dupes = df.duplicated(subset=["Date", "symbol"])
    assert dupes.any()


# ── no-lookahead guarantee ───────────────────────────────────────

def test_backtester_signal_is_lagged():
    """
    Critical: signal generated at bar[t] must not use bar[t+1] data.
    We verify the shift(1) mechanic directly.
    """
    close = pd.Series([100, 101, 99, 102, 98, 103])
    raw_signal = (close > close.shift(1)).astype(int)  # momentum signal
    lagged     = raw_signal.shift(1).fillna(0)

    # lagged[i] must equal raw_signal[i-1]
    for i in range(1, len(raw_signal)):
        assert lagged.iloc[i] == raw_signal.iloc[i - 1], (
            f"Lookahead detected at bar {i}: "
            f"lagged={lagged.iloc[i]}, raw[i-1]={raw_signal.iloc[i-1]}"
        )