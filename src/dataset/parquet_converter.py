#
# Go from CSV to Parquet for faster usage and just better formatting
#
#
# @Author(s): Christopher Yalch
#


import pandas as pd
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def optimize_chunk(df):

    # lowercase column names
    df.columns = [c.lower() for c in df.columns]

    # convert date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # convert type (stock=0, etf=1)
    if "type" in df.columns:
        df["type"] = (
            df["type"]
            .str.lower()
            .map({"stock": 0, "etf": 1})
            .astype("int8")
        )

    # optimize OHLC floats
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # optimize volume
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype("int32")

    # symbol category
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("category")

    return df


def convert_dataset(input_file, output_file, chunksize):

    writer = None
    total_rows = 0

    with tqdm(desc="Processing chunks") as pbar:

        for chunk in pd.read_csv(input_file, chunksize=chunksize):

            chunk = optimize_chunk(chunk)

            # sort values
            if {"symbol", "date"}.issubset(chunk.columns):
                chunk = chunk.sort_values(["symbol", "date"])

            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    output_file,
                    table.schema,
                    compression="snappy"
                )

            writer.write_table(table)

            total_rows += len(chunk)
            pbar.update(1)

    if writer:
        writer.close()

    print(f"\nConversion complete.")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Output written to: {output_file}")


def main():

    parser = argparse.ArgumentParser(
        description="Convert stock CSV dataset to optimized parquet file"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV file"
    )

    parser.add_argument(
        "-o", "--output",
        default="parquet_data.parquet",
        help="Output parquet file (default: parquet_data.parquet)"
    )

    parser.add_argument(
        "-c", "--chunksize",
        type=int,
        default=500000,
        help="Rows per chunk (default: 500000)"
    )

    args = parser.parse_args()

    convert_dataset(args.input, args.output, args.chunksize)


if __name__ == "__main__":
    main()