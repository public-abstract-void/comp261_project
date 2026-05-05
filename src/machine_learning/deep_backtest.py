import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm


# ============================================================
# BACKTEST CORE
# ============================================================

def run_backtest(df, cost_per_trade=0.0005):
    """
    Realistic backtest:
    - Daily rebalancing
    - Transaction costs
    - Equity curve tracking
    - Robust to zero-weight days
    """

    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    equity = 1.0
    equity_curve = []
    daily_returns = []

    prev_weights = None
    skipped_days = 0

    for date, g in tqdm(df.groupby("date"), desc="Backtesting"):
        g = g.dropna(subset=["weight", "target"])

        if len(g) == 0:
            skipped_days += 1
            continue

        raw_weights = g["weight"].to_numpy()
        returns = g["target"].to_numpy()

        # ----------------------------------------------------
        # CHECK EXPOSURE
        # ----------------------------------------------------
        gross = np.sum(np.abs(raw_weights))

        if gross < 1e-12:
            skipped_days += 1
            continue

        # ----------------------------------------------------
        # RESCALE TO TARGET EXPOSURE
        # ----------------------------------------------------
        TARGET_GROSS = 1.0
        weights = raw_weights * (TARGET_GROSS / gross)

        # Use SAME weights everywhere (CRITICAL FIX)
        current = pd.Series(weights, index=g["symbol"])

        # ----------------------------------------------------
        # PNL
        # ----------------------------------------------------
        pnl = np.sum(weights * returns)

        # ----------------------------------------------------
        # TRANSACTION COSTS
        # ----------------------------------------------------
        if prev_weights is not None:
            aligned = pd.concat([current, prev_weights], axis=1).fillna(0)
            aligned.columns = ["current", "prev"]

            turnover = np.sum(np.abs(aligned["current"] - aligned["prev"]))
        else:
            turnover = np.sum(np.abs(current))

        cost = turnover * cost_per_trade
        net_return = pnl - cost

        # ----------------------------------------------------
        # UPDATE EQUITY
        # ----------------------------------------------------
        net_return = max(net_return, -0.99)
        equity *= (1 + net_return)

        equity_curve.append(equity)
        daily_returns.append(net_return)

        prev_weights = current.copy()

    print(f"\nSkipped days (no positions): {skipped_days}")

    return pd.Series(equity_curve), pd.Series(daily_returns)


# ============================================================
# VALIDATION
# ============================================================

def validate_backtest_inputs(df):
    print("\n=== BACKTEST INPUT CHECKS ===")

    print("Rows:", len(df))
    print("NaNs:", df.isna().sum().sum())

    if "weight" not in df.columns or "target" not in df.columns:
        raise ValueError("Missing required columns")

    daily_sums = df.groupby("date")["weight"].sum()
    print("Avg weight sum per day:", daily_sums.mean())

    if (df["weight"] > 0).sum() == 0:
        print("Detected: SHORT-ONLY PORTFOLIO")

    exposure = df.groupby("date")["weight"].apply(lambda x: np.sum(np.abs(x)))
    print("Avg gross exposure:", exposure.mean())


# ============================================================
# METRICS
# ============================================================

def compute_metrics(equity_curve, daily_returns):
    print("\n=== BACKTEST PERFORMANCE ===")

    if len(equity_curve) == 0:
        print("No trades executed. Check portfolio construction.")
        return 0.0, 0.0, 0.0

    total_return = equity_curve.iloc[-1] - 1

    sharpe = (
        daily_returns.mean() /
        (daily_returns.std() + 1e-8)
    )

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()

    print("Total Return:", total_return)
    print("Sharpe:", sharpe)
    print("Max Drawdown:", max_dd)

    return total_return, sharpe, max_dd


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Backtester V1")

    parser.add_argument("--input", required=True)
    parser.add_argument("--cost", type=float, default=0.0005)

    args = parser.parse_args()

    print("Loading predictions...")
    df = pd.read_parquet(args.input)

    print("Running backtest...")
    equity_curve, daily_returns = run_backtest(
        df,
        cost_per_trade=args.cost
    )

    compute_metrics(equity_curve, daily_returns)

    equity_curve.to_csv("equity_curve.csv", index=False)

    print("\nEquity curve saved → equity_curve.csv")
    print("Done.")

    print("\n\nValidation:")
    validate_backtest_inputs(df)


if __name__ == "__main__":
    main()


# ============================================================
# HOW TO RUN
# ============================================================

"""
python backtest_v1.py \
  --input predictions.parquet \
  --cost 0.0005
"""