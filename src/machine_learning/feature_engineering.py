#
# Feature engineering element for our daytrading bot.
# Requires a parquet as input via CLI argument
# 
# @Author(s): Christopher Yalch
#


import pandas as pd
import argparse
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

def compute_features_symbol(df):
    """
    Compute features for a single symbol dataframe.
    """
    df = df.sort_values("date")
    
    # Daily return
    df["return_1d"] = df["close"].pct_change()
    
    # 5-day momentum
    df["return_5d"] = df["close"].pct_change(5)
    
    # Rolling volatility 10 days
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    
    # Volume change
    df["volume_change"] = df["volume"].pct_change()
    
    # Moving averages
    df["moving_avg_10"] = df["close"].rolling(10).mean()
    df["moving_avg_50"] = df["close"].rolling(50).mean()
    
    # Price gap
    df["price_gap"] = (df["open"] / df["close"].shift(1)) - 1
    
    # High-low range
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    
    # Target: next-day direction
    df["target"] = (df["close"].shift(-1) / df["close"] - 1 > 0).astype(int)
    
    df = df.dropna()
    return df

def feature_engineering(input_file, output_file, tickers=None):
    """
    Process Parquet dataset symbol by symbol with progress bar.
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)

    if tickers is not None:
        df = df[df["symbol"].isin(tickers)]
        print(f"Filtering for tickers: {tickers}, {len(df)} rows remaining")

    symbols = df["symbol"].unique()
    print(f"Processing {len(symbols)} symbols...")

    writer = None
    total_rows = 0

    # Process per symbol with progress bar
    for symbol in tqdm(symbols, desc="Symbols processed"):
        df_symbol = df[df["symbol"] == symbol].copy()
        df_features = compute_features_symbol(df_symbol)

        if df_features.empty:
            continue

        table = pa.Table.from_pandas(df_features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema, compression="snappy")
        writer.write_table(table)

        total_rows += len(df_features)

    if writer:
        writer.close()

    print("\nFeature engineering complete!")
    print(f"Total rows: {total_rows}")
    print(f"Total columns: {len(df_features.columns)}")

    # Preview first 5 rows with all columns
    df_preview = pd.read_parquet(output_file).head(5)
    with pd.option_context('display.max_columns', None):
        print("\nPreview of engineered features (first 5 rows):")
        print(df_preview)

def main():
    parser = argparse.ArgumentParser(description="Feature engineering for stock/ETF dataset with progress bar")
    parser.add_argument("-i", "--input", required=True, help="Input Parquet file")
    parser.add_argument("-o", "--output", default="parquet_features.parquet", help="Output Parquet file")
    parser.add_argument("-t", "--tickers", nargs="+", default=None, help="Tickers to process (default: all)")
    args = parser.parse_args()

    feature_engineering(args.input, args.output, args.tickers)

if __name__ == "__main__":
    main()