from __future__ import annotations
import logging
import time
from datetime import date, datetime, timezone
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

            fetched_data = self.fetcher.fetch_incremental(symbols)

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

            if existing_data.empty:
                logger.info("No existing data - creating full dataset")
                combined_data = valid_data
                added_count = len(valid_data)
                modified_count = 0
            else:
                combined_data, change_summary = self.detector.detect(
                    existing_data, valid_data
                )
                added_count = change_summary.added_count
                modified_count = change_summary.modified_count

            if added_count == 0 and modified_count == 0:
                logger.info("No changes detected")
                return UpdateResult(
                    success=True,
                    records_added=0,
                    records_modified=0,
                    execution_time_seconds=time.time() - start_time,
                )

            self._save_combined_data(combined_data)

            version_tag = self.versioner.create_version(combined_data)

            self.indexer.build_indexes(combined_data)
            self.indexer.save_indexes()

            self.fetcher.save_cursor(date.today())

            execution_time = time.time() - start_time

            logger.info("=" * 60)
            logger.info("DAILY UPDATE COMPLETE")
            logger.info(f"  Added: {added_count}, Modified: {modified_count}")
            logger.info(f"  Version: {version_tag}")
            logger.info(f"  Time: {execution_time:.2f}s")
            logger.info("=" * 60)

            return UpdateResult(
                success=True,
                records_added=added_count,
                records_modified=modified_count,
                version_tag=version_tag,
                execution_time_seconds=execution_time,
                metadata={
                    "total_records": len(combined_data),
                    "symbol_count": int(combined_data["symbol"].nunique())
                    if not combined_data.empty
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

    def run_full_rebuild(self, symbols: Optional[set[str]] = None) -> UpdateResult:
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("STARTING FULL REBUILD")
        logger.info("=" * 60)

        try:
            merged_file = self.config.processed_dir / "merged_2017_2026.csv"
            training_file = self.config.processed_dir / "training_2017_2026.csv"

            use_file = None

            if training_file.exists() and training_file.stat().st_size > 100_000_000:
                use_file = training_file
                logger.info(f"Using training file: {training_file}")
            elif merged_file.exists():
                use_file = merged_file
                logger.info(f"Using merged file: {merged_file}")

            if use_file:
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

    def get_data(
        self,
        symbols: Optional[list[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        df = self._load_existing_data()

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
        }

    def _load_existing_data(self) -> pd.DataFrame:
        possible_files = [
            self.config.legacy_data_file,
            self.config.src_data_file,
            self.config.main_data_file,
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
        output_file = self.config.main_data_file
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df = df.sort_values(["symbol", "Date"])

        df.to_csv(output_file, index=False)
        logger.info(f"Saved combined data: {output_file} ({len(df)} records)")
