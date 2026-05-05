import pandas as pd
import argparse
from datetime import datetime


def safe_col(df, col, default=None):
    return df[col] if col in df.columns else default


def generate_report(df, output_path):

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()

    latest = latest.sort_values("confidence", ascending=False)

    buys = latest[latest["signal"] == "BUY"]
    shorts = latest[latest["signal"] == "SHORT"]

    avg_conf = latest["confidence"].mean()

    lines = []
    lines.append("=" * 60)
    lines.append("TRADING SIGNAL REPORT")
    lines.append(f"Generated: {datetime.utcnow()} UTC")
    lines.append(f"Latest Date: {latest_date}")
    lines.append("=" * 60)

    lines.append("\nSUMMARY")
    lines.append(f"Total Signals: {len(latest)}")
    lines.append(f"BUY: {len(buys)} | SHORT: {len(shorts)}")
    lines.append(f"Average Confidence: {avg_conf:.4f}")

    def format_row(row):
        hold = row.get("holding_days", "N/A")
        return f"{row['symbol']:10} | conf={row['confidence']:.3f} | hold={hold}"

    lines.append("\nTOP BUY SIGNALS")
    for _, row in buys.head(20).iterrows():
        lines.append(format_row(row))

    lines.append("\nTOP SHORT SIGNALS")
    for _, row in shorts.head(20).iterrows():
        lines.append(format_row(row))

    lines.append("\nAVAILABLE COLUMNS IN DATA:")
    lines.append(", ".join(latest.columns))

    lines.append("\nNOTES")
    lines.append("- Confidence = model strength of signal")
    lines.append("- Holding period may not exist in current pipeline")
    lines.append("- If missing, defaults to N/A")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="signal_report.txt")

    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    generate_report(df, args.output)
