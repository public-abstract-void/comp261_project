"""
==================================================
HOW TO RUN
==================================================

FAST TEST:
python deep_train_model_v6.py --input features.parquet --sample_frac 0.1 --fast

FULL RUN:
python deep_train_model_v6.py --input features.parquet

OUTPUT CONTROL:
python deep_train_model_v6.py --input features.parquet --pred_out preds.parquet --model_out model.txt

NOTES:
- Optimized for ~32GB RAM
- Full-batch training (best accuracy)
- Memory spikes minimized
"""

import pandas as pd
import numpy as np
import argparse
import lightgbm as lgb
import gc

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

def optimize_dtypes(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df


# ============================================================
# SAFE CROSS-SECTIONAL NORMALIZATION
# (column-by-column to avoid memory explosion)
# ============================================================

def cross_sectional_zscore_safe(df, feature_cols):
    print("Applying cross-sectional normalization (safe)...")

    for col in feature_cols:
        mean = df.groupby("date")[col].transform("mean")
        std = df.groupby("date")[col].transform("std")

        df[col] = (df[col] - mean) / (std + 1e-8)

        del mean, std
        gc.collect()

    return df


# ============================================================
# METRICS
# ============================================================

def compute_ic(df):
    ic_list = []
    for _, g in df.groupby("date"):
        if len(g) > 1:
            ic = np.corrcoef(g["prediction"], g["target"])[0, 1]
            ic_list.append(ic)
    return np.nanmean(ic_list)


def compute_rank_ic(df):
    ic_list = []
    for _, g in df.groupby("date"):
        if len(g) > 1:
            ic = np.corrcoef(
                g["prediction"].rank(),
                g["target"].rank()
            )[0, 1]
            ic_list.append(ic)
    return np.nanmean(ic_list)


# ============================================================
# PORTFOLIO
# ============================================================

def create_portfolio(df, long_pct=0.03, short_pct=0.05):
    df = df.copy()

    df["prediction"] = df.groupby("date")["prediction"].transform(
        lambda x: x - x.mean()
    )

    df["rank"] = df.groupby("date")["prediction"].rank(method="first")
    df["rank_pct"] = df.groupby("date")["rank"].transform(
        lambda x: x / len(x)
    )

    df["weight"] = 0.0

    for date, g in df.groupby("date"):
        long_idx = g[g["rank_pct"] >= (1 - long_pct)].index
        short_idx = g[g["rank_pct"] <= short_pct].index

        if len(long_idx) > 0:
            df.loc[long_idx, "weight"] = 1.0 / len(long_idx)

        if len(short_idx) > 0:
            df.loc[short_idx, "weight"] = -1.0 / len(short_idx)

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="32GB Optimized Training")

    parser.add_argument("--input", required=True)
    parser.add_argument("--target", default="target_5")
    parser.add_argument("--sample_frac", type=float, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--pred_out", default="predictions.parquet")
    parser.add_argument("--model_out", default="model.txt")

    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.input)

    print("Optimizing dtypes...")
    df = optimize_dtypes(df)

    print("Initial rows:", len(df))

    # ========================================================
    # FILTER BAD SYMBOLS EARLY (BIG MEMORY WIN)
    # ========================================================
    print("Filtering symbols...")
    counts = df["symbol"].value_counts()
    good_symbols = counts[counts >= 60].index
    df = df[df["symbol"].isin(good_symbols)]

    print("Remaining rows:", len(df))

    # SAMPLE
    if args.sample_frac:
        symbols = df["symbol"].unique()
        n = int(len(symbols) * args.sample_frac)
        sampled = np.random.choice(symbols, n, replace=False)
        df = df[df["symbol"].isin(sampled)]
        print(f"Sampled {n} symbols")

    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    target = args.target
    target_cols = ["target_1", "target_3", "target_5"]

    feature_cols = [
        c for c in df.columns
        if c not in ["date", "symbol"] + target_cols
    ]

    print("Features:", len(feature_cols))

    # Keep only needed columns (CRITICAL)
    df = df[["date", "symbol"] + feature_cols + [target]]

    df["target_raw"] = df[target]

    # ========================================================
    # NORMALIZATION
    # ========================================================
    df = cross_sectional_zscore_safe(df, feature_cols)

    df[target] = (df[target] - df[target].mean()) / (df[target].std() + 1e-8)

    gc.collect()

    # ========================================================
    # SPLIT
    # ========================================================
    split_date = df["date"].quantile(0.8)

    train = df[df["date"] <= split_date]
    test = df[df["date"] > split_date]

    X_train = train[feature_cols]
    y_train = train[target]

    X_test = test[feature_cols]
    y_test = test[target]

    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))

    # ========================================================
    # MODEL (MEMORY-SAFE SETTINGS)
    # ========================================================
    params = dict(
        objective="regression",
        learning_rate=0.03 if args.fast else 0.02,
        num_leaves=64,
        max_bin=255,
        force_col_wise=True,   # 🔥 memory optimization
        n_estimators=1200 if not args.fast else 600,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print("Training model...")
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100)]
    )

    print("Best iteration:", model.best_iteration_)

    # ========================================================
    # PREDICTIONS
    # ========================================================
    print("Generating predictions...")

    preds = model.predict(X_test)

    pred_df = test[["date", "symbol"]].copy()
    pred_df["prediction"] = preds
    pred_df["target"] = test["target_raw"].values

    pred_df = create_portfolio(pred_df)

    pred_df.to_parquet(args.pred_out, index=False)

    # ========================================================
    # METRICS
    # ========================================================
    print("\n=== PERFORMANCE ===")
    print("Mean IC:", compute_ic(pred_df))
    print("Mean Rank IC:", compute_rank_ic(pred_df))

    # SAVE MODEL
    model.booster_.save_model(args.model_out)

    print("\nDone.")


if __name__ == "__main__":
    main()