# train_valid_v2.py

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm


# ============================================================
# BASIC CHECKS
# ============================================================

def basic_checks(df):
    print("\n=== BASIC CHECKS ===")
    print("Shape:", df.shape)
    print("NaNs:", df.isna().sum().sum())
    print("Infs:", np.isinf(df.select_dtypes(include=[np.number])).sum().sum())


# ============================================================
# IC METRICS
# ============================================================

def compute_ic(df):
    daily_ic = []

    for _, g in df.groupby("date"):
        if len(g) > 1:
            ic = np.corrcoef(g["prediction"], g["target"])[0, 1]
            daily_ic.append(ic)

    daily_ic = pd.Series(daily_ic)

    print("\n=== IC METRICS ===")
    print("Mean IC:", daily_ic.mean())
    print("Std IC:", daily_ic.std())

    return daily_ic


def compute_rank_ic(df):
    daily_ic = []

    for _, g in df.groupby("date"):
        if len(g) > 1:
            ic = np.corrcoef(
                g["prediction"].rank(),
                g["target"].rank()
            )[0, 1]
            daily_ic.append(ic)

    daily_ic = pd.Series(daily_ic)

    print("\n=== RANK IC ===")
    print("Mean Rank IC:", daily_ic.mean())

    return daily_ic


# ============================================================
# PORTFOLIO PERFORMANCE
# ============================================================

def portfolio_performance(df):
    long_returns = []
    short_returns = []
    ls_returns = []

    for _, g in tqdm(df.groupby("date"), desc="Portfolio"):
        g = g.copy()

        # ----------------------------------------------------
        # USE WEIGHTS (CRITICAL FIX)
        # ----------------------------------------------------
        g["weighted_return"] = g["weight"] * g["target"]

        # Long side (positive weights)
        longs = g[g["weight"] > 0]["weighted_return"]

        # Short side (negative weights → invert sign for readability)
        shorts = -g[g["weight"] < 0]["weighted_return"]

        long_returns.append(longs.sum() if len(longs) > 0 else 0)
        short_returns.append(shorts.sum() if len(shorts) > 0 else 0)

        ls = long_returns[-1] + short_returns[-1]

        # If no longs, interpret as short-only PnL
        if long_returns[-1] == 0:
            ls = short_returns[-1]

        ls_returns.append(ls)
        
    long_returns = pd.Series(long_returns)
    short_returns = pd.Series(short_returns)
    ls_returns = pd.Series(ls_returns)

    print("\n=== PORTFOLIO (WEIGHTED) ===")
    print("Avg Long Return:", long_returns.mean())
    print("Avg Short Return:", short_returns.mean())
    print("Avg Long-Short:", ls_returns.mean())

    sharpe = ls_returns.mean() / (ls_returns.std() + 1e-8)
    print("Daily Sharpe:", sharpe)

    # Drawdown
    equity = (1 + ls_returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    print("Max Drawdown:", drawdown.min())

    return ls_returns


# ============================================================
# DECILE ANALYSIS (VERY IMPORTANT)
# ============================================================

def decile_analysis(df):
    print("\n=== DECILE ANALYSIS ===")

    decile_returns = []

    for _, g in df.groupby("date"):
        if len(g) < 10:
            continue

        g = g.copy()
        g["decile"] = pd.qcut(g["prediction"], 10, labels=False)

        decile_means = g.groupby("decile")["target"].mean()
        decile_returns.append(decile_means)

    decile_returns = pd.DataFrame(decile_returns)

    avg = decile_returns.mean()

    print("\nAverage returns by decile:")
    print(avg)

    print("\nTop vs Bottom spread:",
          avg.iloc[-1] - avg.iloc[0])


# ============================================================
# PREDICTION DIAGNOSTICS
# ============================================================

def prediction_diagnostics(df):
    print("\n=== PREDICTION DIAGNOSTICS ===")

    print("Prediction mean:", df["prediction"].mean())
    print("Prediction std:", df["prediction"].std())

    daily_std = df.groupby("date")["prediction"].std()

    print("Avg daily prediction std:", daily_std.mean())

    if daily_std.mean() < 0.001:
        print("⚠️ Predictions too flat")


# ============================================================
# DIRECTIONAL ACCURACY
# ============================================================

def directional_accuracy(df):
    acc = np.mean(np.sign(df["prediction"]) == np.sign(df["target"]))
    print("\nDirectional Accuracy:", acc)


# ============================================================
# FINAL WARNINGS
# ============================================================

def warnings(df, ic, sharpe):
    print("\n=== WARNINGS ===")

    if ic.mean() < 0.01:
        print("⚠️ Weak signal")

    if ic.mean() > 0.1:
        print("⚠️ Suspiciously strong signal")

    if sharpe > 1:
        print("⚠️ Unrealistically high Sharpe (check leakage)")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Training Validator V2")

    parser.add_argument("--input", required=True)
    parser.add_argument("--sample_rows", type=int, default=1_000_000)

    args = parser.parse_args()

    print("Loading predictions (sample)...")
    df = pd.read_parquet(args.input)

    if len(df) > args.sample_rows:
        df = df.sample(args.sample_rows, random_state=42)

    print(f"Using {len(df)} rows")

    basic_checks(df)

    ic = compute_ic(df)
    rank_ic = compute_rank_ic(df)

    ls_returns = portfolio_performance(df)
    sharpe = ls_returns.mean() / (ls_returns.std() + 1e-8)

    decile_analysis(df)
    prediction_diagnostics(df)
    directional_accuracy(df)

    warnings(df, ic, sharpe)

    print("\nDone.")


if __name__ == "__main__":
    main()


# ============================================================
# HOW TO RUN
# ============================================================

"""
======================
VALIDATE MODEL OUTPUT
======================

python train_valid_v2.py \
  --input predictions.parquet


======================
TIPS
======================

- Run after every training
- Focus on:
  - IC stability
  - long vs short balance
  - decile spread
- Ignore directional accuracy (not important)
"""