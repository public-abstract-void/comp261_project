# validate_features_v2.py

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm


# ============================================================
# SUMMARY
# ============================================================

def summarize(df):
    print("\n=== DATASET SUMMARY ===")
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print(df.head())


# ============================================================
# NAN / INF CHECK (DETAILED)
# ============================================================

def check_nans_infs(df):
    print("\n=== NAN / INF CHECK ===")

    nan_counts = df.isna().sum()
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()

    total_nans = nan_counts.sum()
    total_infs = inf_counts.sum()

    print("Total NaNs:", total_nans)
    print("Total Infs:", total_infs)

    if total_nans > 0:
        print("\nTop NaN columns:")
        print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(10))

    if total_infs > 0:
        print("\nTop Inf columns:")
        print(inf_counts[inf_counts > 0].sort_values(ascending=False).head(10))


# ============================================================
# TARGET VALIDATION
# ============================================================

def check_targets(df):
    print("\n=== TARGET ANALYSIS ===")

    targets = [c for c in df.columns if "target" in c]

    for t in targets:
        print(f"\n--- {t} ---")
        desc = df[t].describe()
        print(desc)

        # Check clipping behavior
        print("Min / Max:", df[t].min(), df[t].max())

        # Check proportion of extreme values
        extreme_pct = np.mean((df[t].abs() > 0.09))
        print("Extreme %:", extreme_pct)


# ============================================================
# FEATURE QUALITY
# ============================================================

def check_feature_quality(df):
    print("\n=== FEATURE QUALITY ===")

    numeric = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric.columns if "target" not in c]

    low_var = []
    constant = []

    for col in feature_cols:
        std = numeric[col].std()

        if std < 1e-5:
            low_var.append(col)

        if numeric[col].nunique() <= 1:
            constant.append(col)

    print("Low variance features:", len(low_var))
    print("Constant features:", len(constant))

    if low_var:
        print("Examples:", low_var[:10])

    if constant:
        print("Examples:", constant[:10])


# ============================================================
# LEAKAGE DETECTION (UPGRADED)
# ============================================================

def check_leakage_fast(df):
    print("\n=== LEAKAGE CHECK (FAST) ===")

    targets = [c for c in df.columns if "target" in c]
    numeric = df.select_dtypes(include=[np.number])

    for t in targets:
        print(f"\n--- {t} ---")

        corrs = {}

        for col in numeric.columns:
            if col == t:
                continue

            corr = np.corrcoef(numeric[col], numeric[t])[0, 1]
            if not np.isnan(corr):
                corrs[col] = abs(corr)

        top = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:5]

        max_corr = top[0][1] if top else 0
        print("Max correlation:", max_corr)

        if max_corr > 0.9:
            print("❌ Likely leakage detected")
        elif max_corr > 0.3:
            print("⚠️ Strong feature — inspect")

        print("Top correlated features:")
        for k, v in top:
            print(k, v)


# ============================================================
# CROSS-SECTIONAL CHECK (VERY IMPORTANT)
# ============================================================

def check_cross_sectional_fast(df):
    print("\n=== CROSS-SECTIONAL ANALYSIS (FAST) ===")

    if "date" not in df.columns:
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if "target" not in c]

    dates = df["date"].drop_duplicates()

    if len(dates) > 100:
        dates = dates.sample(100, random_state=42)

    dispersions = []

    for d in tqdm(dates):
        sub = df[df["date"] == d]

        if len(sub) < 5:
            continue

        dispersions.append(sub[feature_cols].std().mean())

    print("Avg feature dispersion:", np.mean(dispersions))
    

# ============================================================
# SYMBOL HEALTH
# ============================================================

def check_symbols(df):
    print("\n=== SYMBOL HEALTH ===")

    counts = df["symbol"].value_counts()

    print("Total symbols:", len(counts))
    print("Min rows per symbol:", counts.min())
    print("Max rows per symbol:", counts.max())

    bad = counts[counts < 50]

    if len(bad) > 0:
        print("⚠️ Symbols with too few rows:", len(bad))


# ============================================================
# FINAL VERDICT
# ============================================================

def final_verdict(df):
    issues = 0

    if df.isna().sum().sum() > 0:
        issues += 1

    if np.isinf(df.select_dtypes(include=[np.number])).sum().sum() > 0:
        issues += 1

    print("\n=== FINAL VERDICT ===")

    if issues == 0:
        print("✅ DATA IS CLEAN AND READY")
    else:
        print(f"⚠️ {issues} ISSUE TYPES DETECTED")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Feature Validator V2")

    parser.add_argument("--input", required=True)
    parser.add_argument("--sample_rows", type=int, default=500_000)

    args = parser.parse_args()

    print("Loading data (sample)...")

    df = pd.read_parquet(args.input)

    if len(df) > args.sample_rows:
        df = df.sample(args.sample_rows, random_state=42)

    print(f"Using {len(df)} rows")

    summarize(df)
    check_nans_infs(df)
    check_targets(df)
    check_feature_quality(df)
    check_leakage_fast(df)   # NEW SAFE VERSION
    check_cross_sectional_fast(df)  # NEW SAFE VERSION
    check_symbols(df)
    final_verdict(df)


if __name__ == "__main__":
    main()


# ============================================================
# HOW TO RUN
# ============================================================

"""
======================
QUICK TEST
======================

python validate_features_v2.py \
  --input features.parquet \
  --n_rows 100000


======================
FULL VALIDATION
======================

python validate_features_v2.py \
  --input features.parquet


======================
TIPS
======================

- Use n_rows first to test quickly
- Full run recommended before training
- Watch leakage warnings carefully
"""