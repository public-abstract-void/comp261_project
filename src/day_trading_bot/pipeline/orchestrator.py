from __future__ import annotations
import logging
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig, UpdateResult
from .fetcher import DataFetcher
from .detector import ChangeDetector
from .validator import DataValidator
from .versioning import DataVersioner
from .indexer import PerformanceIndexer

logger = logging.getLogger(__name__)


class TradingDataPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self.fetcher = DataFetcher(self.config)
        self.detector = ChangeDetector(self.config)
        self.validator = DataValidator(self.config)
        self.versioner = DataVersioner(self.config)
        self.indexer = PerformanceIndexer(self.config)

        self._setup_logging()

        logger.info("TradingDataPipeline initialized")

    def _setup_logging(self) -> None:
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logging.basicConfig(level=log_level, handlers=[handler])

    def run_daily_update(
        self, symbols: Optional[set[str]] = None, force_full: bool = False
    ) -> UpdateResult:
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("STARTING DAILY UPDATE")
        logger.info("=" * 60)

        try:
            existing_data = self._load_existing_data()

            # Daily updates should be cheap: only fetch a small recent window.
            # If we already have data, start near the dataset's current max Date.
            # This avoids refetching 2017->today for thousands of symbols.
            start_date = None
            if not existing_data.empty and "Date" in existing_data.columns:
                try:
                    # Robust against mixed formats + timezone-aware strings.
                    max_dt = pd.to_datetime(
                        existing_data["Date"],
                        format="mixed",
                        utc=True,
                        errors="coerce",
                    ).max()
                    if pd.notna(max_dt):
                        start_date = (max_dt.date() - timedelta(days=10))
                except Exception:
                    start_date = None

            fetched_data = self.fetcher.fetch_incremental(
                symbols,
                start_date=start_date,
                end_date=date.today(),
                # Default to a small known-good set (config.scraper_stocks) unless the caller
                # explicitly supplies a symbol list. This keeps "daily update" practical on laptops.
                use_full_list=False,
            )

            if fetched_data.empty:
                logger.info("No new data fetched")
                return UpdateResult(
                    success=True,
                    records_added=0,
                    execution_time_seconds=time.time() - start_time,
                )

            logger.info(f"Fetched {len(fetched_data)} new records")

            validation_result = self.validator.validate(fetched_data, strict=False)

            if not validation_result.valid:
                logger.warning(f"Validation issues: {validation_result.errors}")

            valid_data = fetched_data
            if validation_result.errors:
                valid_data = self.validator.sanitize(fetched_data)
                logger.info(f"Sanitized data: {len(valid_data)} records")

            # Fast-path daily updates:
            # Instead of rewriting the full 24M-row dataset, write a small delta file.
            # The full merged dataset remains the "rebuild artifact".
            delta_df = valid_data.copy()
            if "Date" in delta_df.columns:
                delta_df["Date"] = pd.to_datetime(
                    delta_df["Date"], format="mixed", utc=True, errors="coerce"
                ).dt.date
                delta_df = delta_df.dropna(subset=["Date"])

            # Enforce 1 row per (symbol, Date) inside the delta itself.
            if "symbol" in delta_df.columns and "Date" in delta_df.columns:
                delta_df = delta_df.drop_duplicates(subset=["symbol", "Date"], keep="last")

            delta_file = self._write_delta(delta_df)
            added_count = len(delta_df)
            modified_count = 0

            # Cursor indicates the last time we attempted an update.
            self.fetcher.save_cursor(date.today())

            execution_time = time.time() - start_time

            logger.info("=" * 60)
            logger.info("DAILY UPDATE COMPLETE")
            logger.info(f"  Added: {added_count}, Modified: {modified_count}")
            logger.info(f"  Delta file: {delta_file}")
            logger.info(f"  Time: {execution_time:.2f}s")
            logger.info("=" * 60)

            return UpdateResult(
                success=True,
                records_added=added_count,
                records_modified=modified_count,
                version_tag=None,
                execution_time_seconds=execution_time,
                metadata={
                    "delta_file": str(delta_file),
                    "delta_records": int(added_count),
                    "delta_symbols": int(delta_df["symbol"].nunique())
                    if not delta_df.empty and "symbol" in delta_df.columns
                    else 0,
                },
            )

        except Exception as e:
            logger.error(f"Daily update failed: {e}", exc_info=True)
            return UpdateResult(
                success=False,
                errors=[str(e)],
                execution_time_seconds=time.time() - start_time,
            )

    def run_full_rebuild(
        self,
        symbols: Optional[set[str]] = None,
        input_path: Optional[Path] = None,
    ) -> UpdateResult:
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("STARTING FULL REBUILD")
        logger.info("=" * 60)

        try:
            merged_file = self.config.processed_dir / "merged_2017_2026.csv"
            training_file = self.config.processed_dir / "training_2017_2026.csv"

            # Safety rule:
            # - If you already have full_merged.parquet, rebuild should use it by default.
            #   That prevents accidentally overwriting a larger merged dataset with a smaller CSV.
            # - If you want to rebuild from a specific file, pass --input.
            use_file: Optional[Path] = None

            if input_path is not None:
                use_file = input_path
                logger.info(f"Using explicit input file: {use_file}")
            elif self.config.main_data_file.exists():
                use_file = self.config.main_data_file
                logger.info(f"Using existing main data file: {use_file}")
            elif merged_file.exists():
                use_file = merged_file
                logger.info(f"Using merged file: {use_file}")
            elif training_file.exists():
                use_file = training_file
                logger.info(f"Using training file: {use_file}")
            if use_file:
                if use_file.suffix.lower() in {".parquet", ".pq"}:
                    df = pd.read_parquet(use_file)
                else:
                    df = pd.read_csv(use_file)
                logger.info(f"Loaded {len(df)} records from {use_file.name}")

                if symbols:
                    df = df[df["symbol"].isin([s.upper() for s in symbols])]

                valid_data = df
            else:
                fetched_data = self.fetcher.fetch_full(symbols)

                if fetched_data.empty:
                    return UpdateResult(
                        success=False,
                        errors=["No data fetched"],
                        execution_time_seconds=time.time() - start_time,
                    )

                valid_data = self.validator.sanitize(fetched_data)
                logger.info(f"Using scraped data: {len(valid_data)} records")

            valid_data = valid_data.sort_values(["symbol", "Date"])

            self._save_combined_data(valid_data)

            version_tag = self.versioner.create_version(valid_data)

            self.indexer.build_indexes(valid_data)
            self.indexer.save_indexes()

            execution_time = time.time() - start_time

            logger.info("=" * 60)
            logger.info("FULL REBUILD COMPLETE")
            logger.info(f"  Total: {len(valid_data)} records")
            logger.info(f"  Version: {version_tag}")
            logger.info(f"  Time: {execution_time:.2f}s")
            logger.info("=" * 60)

            return UpdateResult(
                success=True,
                records_added=len(valid_data),
                version_tag=version_tag,
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            logger.error(f"Full rebuild failed: {e}", exc_info=True)
            return UpdateResult(
                success=False,
                errors=[str(e)],
                execution_time_seconds=time.time() - start_time,
            )

    _data_cache = None

    def get_data(
        self,
        symbols: Optional[list[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        if TradingDataPipeline._data_cache is None:
            TradingDataPipeline._data_cache = self._load_existing_data()
        df = TradingDataPipeline._data_cache

        deltas = self._load_deltas(symbols=symbols)
        if not deltas.empty:
            df = pd.concat([df, deltas], ignore_index=True)

        if df.empty:
            return df

        if symbols:
            df = df[df["symbol"].isin([s.upper() for s in symbols])]

        if start_date:
            df = df[pd.to_datetime(df["Date"]) >= pd.to_datetime(start_date)]

        if end_date:
            df = df[pd.to_datetime(df["Date"]) <= pd.to_datetime(end_date)]

        return df

    def get_latest_version(self) -> Optional[pd.DataFrame]:
        return self.versioner.load_latest()

    def list_versions(self) -> list[dict]:
        return self.versioner.list_versions()

    def get_audit_log(self, limit: int = 50) -> pd.DataFrame:
        return self.versioner.get_audit_log(limit)

    def get_status(self) -> dict:
        df = self._load_existing_data()
        deltas = self._list_delta_files()

        versions = self.list_versions()

        cursor_date = None
        if self.config.cursor_file.exists():
            cursor_date = self.config.cursor_file.read_text().strip()

        return {
            "data_exists": df.empty is not True,
            "total_records": len(df),
            "symbol_count": int(df["symbol"].nunique()) if not df.empty else 0,
            "date_range": {
                "min": str(df["Date"].min()) if not df.empty else None,
                "max": str(df["Date"].max()) if not df.empty else None,
            },
            "version_count": len(versions),
            "latest_version": versions[0].get("version") if versions else None,
            "last_fetch_date": cursor_date,
            "indexes_loaded": self.indexer._loaded,
            "delta_count": len(deltas),
            "latest_delta": str(deltas[0].name) if deltas else None,
        }

    def _delta_dir(self) -> Path:
        d = self.config.processed_dir / "deltas"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _list_delta_files(self) -> list[Path]:
        d = self._delta_dir()
        files = sorted(d.glob("delta_*.parquet"), reverse=True)
        return files

    def _write_delta(self, df: pd.DataFrame) -> Path:
        out_dir = self._delta_dir()
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"delta_{tag}.parquet"
        df.to_parquet(out, index=False)
        return out

    def _load_deltas(self, symbols: Optional[list[str]] = None) -> pd.DataFrame:
        files = self._list_delta_files()
        if not files:
            return pd.DataFrame()

        # Only the newest few deltas matter for "daily update" consumption.
        # Keep this small so reads remain fast.
        files = files[:10]
        frames = []
        for f in files:
            try:
                d = pd.read_parquet(f)
                if symbols and "symbol" in d.columns:
                    d = d[d["symbol"].isin([s.upper() for s in symbols])]
                if not d.empty:
                    frames.append(d)
            except Exception as e:
                logger.warning(f"Could not load delta {f}: {e}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _load_existing_data(self) -> pd.DataFrame:
        parquet_file = self.config.main_data_file

        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                logger.info(
                    f"Loaded existing data from {parquet_file}: {len(df)} records"
                )
                return df
            except Exception as e:
                logger.warning(f"Error loading {parquet_file}: {e}")

        csv_file = self.config.main_data_csv
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded existing data from {csv_file}: {len(df)} records")
                return df
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")

        possible_files = [
            self.config.legacy_data_file,
            self.config.src_data_file,
        ]

        for data_file in possible_files:
            if data_file.exists():
                try:
                    df = pd.read_csv(data_file)
                    logger.info(
                        f"Loaded existing data from {data_file}: {len(df)} records"
                    )
                    return df
                except Exception as e:
                    logger.warning(f"Error loading {data_file}: {e}")
                    continue

        logger.info("No existing data file found")
        return pd.DataFrame()

    def _save_combined_data(self, df: pd.DataFrame) -> None:
        output_parquet = self.config.main_data_file
        output_csv = self.config.main_data_csv
        output_parquet.parent.mkdir(parents=True, exist_ok=True)

        df = df.sort_values(["symbol", "Date"])

        df.to_parquet(output_parquet, index=False)
        logger.info(f"Saved combined data: {output_parquet} ({len(df)} records)")

        df.to_csv(output_csv, index=False)
        logger.info(f"Saved CSV backup: {output_csv} ({len(df)} records)")
