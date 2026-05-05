"""
==================================================
PRO FEATURE ENGINEERING PIPELINE (V4 - OPTIMIZED)
==================================================
"""

import pandas as pd
import numpy as np
import argparse

TARGET_HORIZONS = [1, 3, 5]
ROLL_WINDOWS = [5, 10, 20, 50]
EPS = 1e-9


# =========================
# HELPERS
# =========================

def safe_div(a, b):
    return a / (b + EPS)


def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + EPS)


# =========================
# RETURNS
# =========================

def add_returns(df):
    df["return_1d"] = df.groupby("symbol")["close"].pct_change()
    df["return_5d"] = df.groupby("symbol")["close"].pct_change(5)
    return df


# =========================
# MARKET REGIME (NEW)
# =========================

def add_market_regime(df):
    print("Market regime...")

    daily_vol = df.groupby("date")["return_1d"].std()
    daily_vol = daily_vol.rolling(20, min_periods=5).mean()

    df["market_vol_regime"] = df["date"].map(daily_vol)
    df["market_vol_regime"] = df["market_vol_regime"] / (
        df["market_vol_regime"].rolling(50).mean() + EPS
    )

    return df


# =========================
# FEATURES
# =========================

def add_momentum_features(df):
    for w in ROLL_WINDOWS:
        ma = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(w).mean()
        )
        df[f"ma_ratio_{w}"] = safe_div(df["close"], ma)
    return df


def add_volatility_features(df):
    for w in ROLL_WINDOWS:
        vol = df.groupby("symbol")["return_1d"].transform(
            lambda x: x.rolling(w).std()
        )
        df[f"vol_{w}"] = vol
    return df


def add_zscore_features(df):
    for w in ROLL_WINDOWS:
        df[f"zscore_{w}"] = df.groupby("symbol")["close"].transform(
            lambda x: zscore(x, w)
        )
    return df


def add_alpha_combos(df):
    df["momentum_vol_adj"] = safe_div(df["return_5d"], df["vol_20"])
    df["reversion_strength"] = -df["zscore_20"]
    df["combo_momentum"] = df["return_5d"] * df["ma_ratio_10"]
    return df


# =========================
# TARGETS
# =========================

def add_targets(df):
    for h in TARGET_HORIZONS:
        df[f"target_{h}"] = (
            df.groupby("symbol")["close"].shift(-h) / df["close"] - 1
        )
    return df


# =========================
# CLEANING
# =========================

def clean_df(df):
    df = df.loc[:, ~df.columns.duplicated()]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    return df


# =========================
# MAIN PIPELINE
# =========================

def run_feature_engineering(input_path, output_path, date_frac=1.0):
    print("Loading data...")
    df = pd.read_parquet(input_path)

    print("Sorting...")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # =========================
    # DATE FILTER (IMPORTANT FIX)
    # =========================
    if date_frac < 1.0:
        print(f"Filtering to latest {date_frac*100:.1f}%...")
        cutoff = df["date"].quantile(1 - date_frac)
        df = df[df["date"] >= cutoff]

    print("Adding returns...")
    df = add_returns(df)

    print("Momentum features...")
    df = add_momentum_features(df)

    print("Volatility features...")
    df = add_volatility_features(df)

    print("Z-score features...")
    df = add_zscore_features(df)

    print("Market regime...")
    df = add_market_regime(df)

    print("Alpha combos...")
    df = add_alpha_combos(df)

    print("Targets...")
    df = add_targets(df)

    print("Cleaning...")
    df = clean_df(df)

    print("Saving...")
    df.to_parquet(output_path, index=False)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--date_frac", type=float, default=1.0)

    args = parser.parse_args()

    run_feature_engineering(
        input_path=args.input,
        output_path=args.output,
        date_frac=args.date_frac,
    )
