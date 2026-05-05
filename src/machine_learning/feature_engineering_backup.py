"""
==================================================
PRO FEATURE ENGINEERING PIPELINE (V4)
==================================================
IMPROVEMENTS:
- Date fraction filtering (memory control)
- Removed lambda transforms (major speed/memory gain)
- Reduced intermediate allocations
- Fully compatible with existing pipeline
==================================================
"""

import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
TARGET_HORIZONS = [1, 3, 5]
ROLL_WINDOWS = [5, 10, 20, 50]
EPS = 1e-9

# =========================
# HELPERS
# =========================

def safe_div(a, b):
    return a / (b + EPS)


def zscore_fast(series, window):
    rolling = series.rolling(window)
    mean = rolling.mean()
    std = rolling.std()
    return (series - mean) / (std + EPS)


# =========================
# DATE FILTERING
# =========================

def filter_recent_dates(df, frac):
    if frac >= 1.0:
        return df

    unique_dates = np.sort(df["date"].unique())
    cutoff_idx = int(len(unique_dates) * (1 - frac))
    cutoff_date = unique_dates[cutoff_idx]

    return df[df["date"] >= cutoff_date]


# =========================
# FEATURES
# =========================

def add_returns(df):
    g = df.groupby("symbol")["close"]
    df["return_1d"] = g.pct_change()
    df["return_5d"] = g.pct_change(5)
    return df


def add_momentum_features(df):
    g = df.groupby("symbol")["close"]

    for w in ROLL_WINDOWS:
        ma = g.rolling(w).mean().reset_index(level=0, drop=True)
        df[f"ma_ratio_{w}"] = safe_div(df["close"], ma)

    return df


def add_volatility_features(df):
    g = df.groupby("symbol")["return_1d"]

    for w in ROLL_WINDOWS:
        vol = g.rolling(w).std().reset_index(level=0, drop=True)
        df[f"vol_{w}"] = vol

    return df


def add_zscore_features(df):
    g = df.groupby("symbol")["close"]

    for w in ROLL_WINDOWS:
        z = g.rolling(w)
        mean = z.mean().reset_index(level=0, drop=True)
        std = z.std().reset_index(level=0, drop=True)
        df[f"zscore_{w}"] = (df["close"] - mean) / (std + EPS)

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
    g = df.groupby("symbol")["close"]

    for h in TARGET_HORIZONS:
        future = g.shift(-h)
        df[f"target_{h}"] = future / df["close"] - 1

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
# MAIN
# =========================

def run_feature_engineering(input_path, output_path, date_frac):
    print("Loading data...")
    df = pd.read_parquet(input_path)

    print("Sorting...")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"Filtering to latest {date_frac*100:.1f}% of dates...")
    df = filter_recent_dates(df, date_frac)

    print("Adding returns...")
    df = add_returns(df)

    print("Momentum features...")
    df = add_momentum_features(df)

    print("Volatility features...")
    df = add_volatility_features(df)

    print("Z-score features...")
    df = add_zscore_features(df)

    print("Alpha combos...")
    df = add_alpha_combos(df)

    print("Targets...")
    df = add_targets(df)

    print("Cleaning...")
    df = clean_df(df)

    print("Saving...")
    df.to_parquet(output_path, index=False)

    print("Done.")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--date_frac",
        type=float,
        default=1.0,
        help="Fraction of most recent dates to keep (e.g. 0.5 = latest 50%%)"
    )

    args = parser.parse_args()

    run_feature_engineering(
        args.input,
        args.output,
        args.date_frac
    )
