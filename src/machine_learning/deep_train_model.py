"""
==================================================
PRO MODEL TRAINING PIPELINE (V7)
==================================================

GOAL:
- Train model on historical data
- Emphasize recent data (time-decay)
- Output top stocks + weights for latest date

==================================================
HOW TO RUN
==================================================

FAST TEST:
python deep_train_model_v7.py --input features.parquet --sample_frac 0.1 --fast

FULL RUN:
python deep_train_model_v7.py --input features.parquet

OUTPUT:
- predictions.parquet (for backtesting)
- model.txt (trained model)
- printed TOP STOCKS (latest date)

==================================================
REQUIREMENTS
==================================================
- ~32GB RAM
- Parquet input file with:
    date, symbol, features..., target_X

==================================================
"""

import pandas as pd
import numpy as np
import argparse
import lightgbm as lgb
import gc
from tqdm import tqdm

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
# CROSS-SECTIONAL NORMALIZATION (SAFE + PROGRESS)
# ============================================================

def cross_sectional_zscore_safe(df, feature_cols):
    print("Applying cross-sectional normalization...")

    for col in tqdm(feature_cols, desc="Normalizing features"):
        mean = df.groupby("date")[col].transform("mean")
        std = df.groupby("date")[col].transform("std")

        df[col] = (df[col] - mean) / (std + 1e-8)

        del mean, std
        gc.collect()

    return df


# ============================================================
# TIME DECAY WEIGHTING (CRITICAL UPGRADE)
# ============================================================

def add_time_decay_weights(df, strength=1.5):
    print("Applying time-decay weighting...")

    df = df.sort_values("date")

    # Rank dates from 0 → 1
    df["time_rank"] = df["date"].rank(pct=True)

    # Exponential weighting
    df["sample_weight"] = np.exp(strength * df["time_rank"])

    return df


# ============================================================
# PORTFOLIO (PRO VERSION - LONG ONLY)
# ============================================================

def create_portfolio(df, long_pct=0.02):
    df = df.copy()

    # Standardize predictions per day
    df["prediction"] = df.groupby("date")["prediction"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    df["rank"] = df.groupby("date")["prediction"].rank(method="first")
    df["rank_pct"] = df.groupby("date")["rank"].transform(
        lambda x: x / len(x)
    )

    df["weight"] = 0.0

    for date, g in df.groupby("date"):
        idx = g.index

        long_idx = g[g["rank_pct"] >= (1 - long_pct)].index
        scores = g.loc[long_idx, "prediction"].clip(lower=0)

        if len(scores) > 0:
            denom = scores.sum()
            if denom > 1e-12:
                df.loc[long_idx, "weight"] = scores / denom

        # Normalize to full capital
        weights = df.loc[idx, "weight"]
        total = weights.sum()

        if total > 0:
            df.loc[idx, "weight"] /= total

    return df


# ============================================================
# METRICS
# ============================================================

def compute_ic(df):
    ic_list = []
    for _, g in df.groupby("date"):
        if len(g) > 1:
            if g["prediction"].std() > 1e-12 and g["target"].std() > 1e-12:
                ic_list.append(
                    np.corrcoef(g["prediction"], g["target"])[0, 1]
                )
    return np.nanmean(ic_list)


def compute_rank_ic(df):
    ic_list = []
    for _, g in df.groupby("date"):
        if len(g) > 1:
            pred = g["prediction"].rank()
            tgt = g["target"].rank()

            if pred.std() > 1e-12 and tgt.std() > 1e-12:
                ic_list.append(np.corrcoef(pred, tgt)[0, 1])

    return np.nanmean(ic_list)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pro Training Pipeline V7")

    parser.add_argument("--input", required=True)
    parser.add_argument("--target", default="target_5")
    parser.add_argument("--sample_frac", type=float, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--pred_out", default="predictions.parquet")
    parser.add_argument("--model_out", default="model.txt")

    args = parser.parse_args()

    # ========================================================
    # LOAD
    # ========================================================
    print("Loading data...")
    df = pd.read_parquet(args.input)

    print("Optimizing dtypes...")
    df = optimize_dtypes(df)

    print("Initial rows:", len(df))

    # ========================================================
    # FILTER SYMBOLS
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

    # ========================================================
    # TARGET SMOOTHING (CRITICAL FIX)
    # ========================================================
    print("Applying target smoothing...")

    df["target_smooth"] = (
        df.groupby("symbol")[target]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # Clip extreme values
    df["target_smooth"] = df["target_smooth"].clip(-3, 3)

    # Preserve original BEFORE dropping anything
    df["target_raw"] = df[target]

    # Use smoothed target
    target = "target_smooth"

    # Exclude BOTH original + smoothed targets from features
    feature_cols = [
        c for c in df.columns
        if c not in ["date", "symbol"] + target_cols + ["target_smooth", "target_raw"]
    ]

    # Final dataset
    df = df[["date", "symbol"] + feature_cols + [target, "target_raw"]]

    # ========================================================
    # NORMALIZATION
    # ========================================================
    df = cross_sectional_zscore_safe(df, feature_cols)

    df[target] = (df[target] - df[target].mean()) / (df[target].std() + 1e-8)

    # ========================================================
    # TIME DECAY
    # ========================================================
    df = add_time_decay_weights(df)

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

    w_train = train["sample_weight"]

    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))

    # ========================================================
    # MODEL (FIXED UNDERFITTING)
    # ========================================================
    params = dict(
        objective="regression",
        learning_rate=0.01,          # slower learning
        num_leaves=256,              # more complexity
        max_depth=-1,
        min_child_samples=20,        # allow finer splits
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=4000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        force_col_wise=True,
    )

    print("Training model...")
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(50)]
    )

    print("Best iteration:", model.best_iteration_)

    # ========================================================
    # PREDICTIONS
    # ========================================================
    print("Generating predictions...")
    preds = model.predict(X_test)

    # Auto-detect correct direction using validation
    temp_df = test.copy()
    temp_df["prediction"] = preds

    ic = compute_ic(
        temp_df[["date", "prediction", "target_raw"]]
        .rename(columns={"target_raw": "target"})
    )

    if ic < 0:
        print("Flipping signal (negative IC detected)")
        preds = -preds

    pred_df = test[["date", "symbol"]].copy()
    pred_df["prediction"] = preds
    pred_df["target"] = test["target_raw"].values

    pred_df["prediction"] = pred_df["prediction"].clip(-5, 5)
    pred_df = create_portfolio(pred_df)
    pred_df = pred_df.dropna()

    pred_df.to_parquet(args.pred_out, index=False)

    # ========================================================
    # METRICS
    # ========================================================
    print("\n=== PERFORMANCE ===")
    print("Mean IC:", compute_ic(pred_df))
    print("Mean Rank IC:", compute_rank_ic(pred_df))

    # ========================================================
    # OUTPUT TOP STOCKS (YOUR GOAL)
    # ========================================================
    print("\n=== TOP STOCKS (LATEST DATE) ===")

    latest_date = pred_df["date"].max()
    latest = pred_df[pred_df["date"] == latest_date]

    top = latest.sort_values("prediction", ascending=False)

    print(top[["symbol", "prediction", "weight"]].head(20))

    # SAVE MODEL
    model.booster_.save_model(args.model_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
