#!/usr/bin/env python3
"""
CLI for daily data pipeline updates.

Usage:
    python scripts/daily_update.py status
    python scripts/daily_update.py update
    python scripts/daily_update.py rebuild
    python scripts/daily_update.py versions
    python scripts/daily_update.py get AAPL --start 2026-01-01
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, date

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from day_trading_bot.pipeline import TradingDataPipeline, PipelineConfig


def cmd_status(pipeline: TradingDataPipeline) -> int:
    status = pipeline.get_status()

    print("\n" + "=" * 50)
    print("PIPELINE STATUS")
    print("=" * 50)
    print(f"Data exists:     {status['data_exists']}")
    print(f"Total records:   {status['total_records']:,}")
    print(f"Symbol count:    {status['symbol_count']}")

    dr = status["date_range"]
    print(f"Date range:      {dr.get('min', 'N/A')} to {dr.get('max', 'N/A')}")

    print(f"Version count:   {status['version_count']}")
    print(f"Latest version:  {status['latest_version']}")
    print(f"Last fetch:      {status['last_fetch_date']}")
    print(f"Indexes loaded:  {status['indexes_loaded']}")
    print("=" * 50)

    return 0


def cmd_update(pipeline: TradingDataPipeline, args) -> int:
    symbols = None
    if args.symbols:
        symbols = set(args.symbols.split(","))

    result = pipeline.run_daily_update(symbols=symbols, force_full=args.force)

    print("\n" + "=" * 50)
    print("UPDATE RESULT")
    print("=" * 50)
    print(f"Success:         {result.success}")
    print(f"Records added:   {result.records_added}")
    print(f"Records modified: {result.records_modified}")
    print(f"Version tag:     {result.version_tag}")
    print(f"Time (seconds):  {result.execution_time_seconds:.2f}")

    if result.errors:
        print(f"Errors:          {result.errors}")

    print("=" * 50)

    return 0 if result.success else 1


def cmd_rebuild(pipeline: TradingDataPipeline, args) -> int:
    symbols = None
    if args.symbols:
        symbols = set(args.symbols.split(","))

    input_path = None
    if args.input:
        input_path = Path(args.input).expanduser().resolve()

    result = pipeline.run_full_rebuild(symbols=symbols, input_path=input_path)

    print("\n" + "=" * 50)
    print("REBUILD RESULT")
    print("=" * 50)
    print(f"Success:         {result.success}")
    print(f"Total records:   {result.records_added}")
    print(f"Version tag:     {result.version_tag}")
    print(f"Time (seconds):  {result.execution_time_seconds:.2f}")

    if result.errors:
        print(f"Errors:          {result.errors}")

    print("=" * 50)

    return 0 if result.success else 1


def cmd_versions(pipeline: TradingDataPipeline) -> int:
    versions = pipeline.list_versions()

    print("\n" + "=" * 80)
    print("VERSIONS")
    print("=" * 80)
    print(
        f"{'Version':<20} {'Created':<25} {'Records':>10} {'Symbols':>8} {'Checksum':<10}"
    )
    print("-" * 80)

    for v in versions:
        created = v.get("created_at", "N/A")[:19]
        records = v.get("record_count", 0)
        symbols = v.get("symbol_count", 0)
        checksum = v.get("checksum", "")[:8]
        version = v.get("version", "N/A")

        print(f"{version:<20} {created:<25} {records:>10,} {symbols:>8} {checksum:<10}")

    print("=" * 80)

    return 0


def cmd_get(pipeline: TradingDataPipeline, args) -> int:
    symbols = args.symbol.split(",") if "," in args.symbol else [args.symbol]

    start = None
    end = None

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()

    df = pipeline.get_data(symbols=symbols, start_date=start, end_date=end)

    print(f"\nRetrieved {len(df)} records for {symbols}")

    if args.preview and not df.empty:
        preview_cols = ["Date", "symbol", "Open", "High", "Low", "Close", "Volume"]
        available_cols = [c for c in preview_cols if c in df.columns]
        print(df[available_cols].head(20).to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to: {args.output}")

    return 0


def cmd_audit(pipeline: TradingDataPipeline, args) -> int:
    df = pipeline.get_audit_log(args.limit)

    if df.empty:
        print("No audit log entries")
        return 0

    print("\n" + "=" * 80)
    print("AUDIT LOG")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Day Trading Bot Data Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/daily_update.py status
    python scripts/daily_update.py update
    python scripts/daily_update.py update --symbols AAPL,MSFT
    python scripts/daily_update.py rebuild
    python scripts/daily_update.py versions
    python scripts/daily_update.py get AAPL --start 2026-01-01
    python scripts/daily_update.py audit --limit 20
        """,
    )

    parser.add_argument(
        "--config", default=None, help="Path to custom config (not implemented yet)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    status_parser = subparsers.add_parser("status", help="Show pipeline status")

    update_parser = subparsers.add_parser("update", help="Run daily incremental update")
    update_parser.add_argument(
        "--symbols",
        help="Comma-separated symbols to update (default: all scraper stocks)",
    )
    update_parser.add_argument(
        "--force", action="store_true", help="Force full re-check even if no new data"
    )

    rebuild_parser = subparsers.add_parser(
        "rebuild", help="Full rebuild from scraper data"
    )
    rebuild_parser.add_argument(
        "--symbols",
        help="Comma-separated symbols to rebuild (default: all scraper stocks)",
    )
    rebuild_parser.add_argument(
        "--input",
        default=None,
        help=(
            "Optional input file to rebuild from (CSV or Parquet). "
            "If omitted, rebuild uses existing full_merged.parquet if present."
        ),
    )

    versions_parser = subparsers.add_parser("versions", help="List data versions")

    get_parser = subparsers.add_parser("get", help="Get data for symbol(s)")
    get_parser.add_argument("symbol", help="Symbol or comma-separated symbols")
    get_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    get_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    get_parser.add_argument("--preview", action="store_true", help="Show preview")
    get_parser.add_argument("--output", help="Save to CSV file")

    audit_parser = subparsers.add_parser("audit", help="Show audit log")
    audit_parser.add_argument("--limit", type=int, default=50, help="Number of entries")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        config = PipelineConfig()
        pipeline = TradingDataPipeline(config)

        if args.command == "status":
            return cmd_status(pipeline)
        elif args.command == "update":
            return cmd_update(pipeline, args)
        elif args.command == "rebuild":
            return cmd_rebuild(pipeline, args)
        elif args.command == "versions":
            return cmd_versions(pipeline)
        elif args.command == "get":
            return cmd_get(pipeline, args)
        elif args.command == "audit":
            return cmd_audit(pipeline, args)
        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
