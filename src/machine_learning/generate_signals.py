import pandas as pd
import numpy as np
import argparse


def generate_signals(df, long_pct=0.02, short_pct=0.02):
    df = df.copy()

    # normalize predictions per day
    df["pred_z"] = df.groupby("date")["prediction"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    df["rank_pct"] = df.groupby("date")["pred_z"].rank(pct=True)

    df["signal"] = "HOLD"

    df.loc[df["rank_pct"] >= (1 - long_pct), "signal"] = "BUY"
    df.loc[df["rank_pct"] <= short_pct, "signal"] = "SHORT"

    # confidence = abs(z-score), scaled
    df["confidence"] = df.groupby("date")["pred_z"].transform(
        lambda x: np.abs(x) / (np.abs(x).max() + 1e-8)
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="predictions.parquet")
    parser.add_argument("--output", default="signals.parquet")
    parser.add_argument("--long_pct", type=float, default=0.02)
    parser.add_argument("--short_pct", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=5)

    args = parser.parse_args()

    print("Loading predictions...")
    df = pd.read_parquet(args.input)

    print("Generating signals...")
    df = generate_signals(df, args.long_pct, args.short_pct)

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()

    latest["holding_days"] = args.horizon

    latest = latest.sort_values("confidence", ascending=False)

    print("\n=== TRADING SIGNALS (LATEST DATE) ===")
    print(
        latest[latest["signal"] != "HOLD"][
            ["symbol", "signal", "confidence", "holding_days"]
        ].head(20)
    )

    print("\nSaving signals...")
    df.to_parquet(args.output, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
