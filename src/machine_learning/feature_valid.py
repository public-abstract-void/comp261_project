import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


# ============================================================
# LOAD
# ============================================================

def load_data(path, sample_rows):
    print("Loading data (sample)...")
    df = pd.read_parquet(path)

    if len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)

    print(f"Using {len(df)} rows")
    return df


# ============================================================
# BASIC INFO
# ============================================================

def dataset_summary(df):
    print("\n=== DATASET SUMMARY ===")
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print(df.head())


# ============================================================
# NAN / INF CHECK
# ============================================================

def nan_check(df):
    print("\n=== NAN / INF CHECK ===")

    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

    print("Total NaNs:", nan_count)
    print("Total Infs:", inf_count)

    if nan_count > 0:
        print("\nTop NaN columns:")
        print(df.isna().sum().sort_values(ascending=False).head(10))


# ============================================================
# TARGET ANALYSIS
# ============================================================

def target_analysis(df, targets):
    print("\n=== TARGET ANALYSIS ===")

    for t in targets:
        print(f"\n--- {t} ---")
        print(df[t].describe())

        extreme = np.mean(np.abs(df[t]) > 1)
        print("Extreme %:", extreme)


# ============================================================
# FEATURE QUALITY
# ============================================================

def feature_quality(df, feature_cols):
    print("\n=== FEATURE QUALITY ===")

    variances = df[feature_cols].var()

    low_var = variances[variances < 1e-6]
    const = variances[variances == 0]

    print("Low variance features:", len(low_var))
    print("Constant features:", len(const))

    if len(const) > 0:
        print("Examples:", const.index.tolist()[:5])


# ============================================================
# LEAKAGE CHECK (FIXED)
# ============================================================

def leakage_check(df, feature_cols, targets):
    print("\n=== LEAKAGE CHECK (FIXED) ===")

    for t in targets:
        print(f"\n--- {t} ---")

        corrs = []

        for f in feature_cols:
            try:
                c = np.corrcoef(df[f], df[t])[0, 1]
                if not np.isnan(c):
                    corrs.append((f, abs(c)))
            except:
                continue

        corrs = sorted(corrs, key=lambda x: -x[1])

        if len(corrs) == 0:
            continue

        top_feature, top_corr = corrs[0]

        print("Max correlation:", top_corr)

        if top_corr > 0.95:
            print("❌ REAL leakage detected:", top_feature)
        elif top_corr > 0.2:
            print("⚠️ Strong feature:", top_feature)

        print("Top correlated features:")
        for f, c in corrs[:5]:
            print(f, c)


# ============================================================
# CROSS-SECTIONAL CHECK
# ============================================================

def cross_sectional(df, feature_cols):
    print("\n=== CROSS-SECTIONAL ANALYSIS (FAST) ===")

    sample_dates = df["date"].drop_duplicates().sample(100, random_state=42)

    dispersions = []

    for d in tqdm(sample_dates):
        g = df[df["date"] == d]

        if len(g) > 1:
            dispersions.append(g[feature_cols].std().mean())

    print("Avg feature dispersion:", np.mean(dispersions))


# ============================================================
# SYMBOL HEALTH
# ============================================================

def symbol_health(df):
    print("\n=== SYMBOL HEALTH ===")

    counts = df["symbol"].value_counts()

    print("Total symbols:", len(counts))
    print("Min rows per symbol:", counts.min())
    print("Max rows per symbol:", counts.max())

    weak = (counts < 60).sum()

    if weak > 0:
        print("⚠️ Symbols with too few rows:", weak)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Feature Validator PRO")

    parser.add_argument("--input", required=True)
    parser.add_argument("--sample_rows", type=int, default=500_000)

    args = parser.parse_args()

    df = load_data(args.input, args.sample_rows)

    dataset_summary(df)
    nan_check(df)

    targets = ["target_1", "target_3", "target_5"]

    feature_cols = [
        c for c in df.columns
        if c not in ["date", "symbol"] + targets
    ]

    target_analysis(df, targets)
    feature_quality(df, feature_cols)
    leakage_check(df, feature_cols, targets)
    cross_sectional(df, feature_cols)
    symbol_health(df)

    print("\n=== FINAL VERDICT ===")

    if df.isna().sum().sum() > 0:
        print("⚠️ NaNs present (expected early in time-series)")
    else:
        print("✅ Clean dataset")

    print("Done.")


if __name__ == "__main__":
    main()