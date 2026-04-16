from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class PerformanceIndexer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._symbol_index: Optional[pd.DataFrame] = None
        self._date_index: Optional[pd.DataFrame] = None
        self._loaded = False

    def build_indexes(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        logger.info(f"Building indexes for {len(df)} records...")

        symbol_idx = self._build_symbol_index(df)
        date_idx = self._build_date_index(df)
        self._symbol_index = symbol_idx
        self._date_index = date_idx
        self._loaded = True

        logger.info(
            f"Built indexes: {len(symbol_idx)} symbols, {len(date_idx)} date entries"
        )

        return {"symbol": symbol_idx, "date": date_idx}

    def _build_symbol_index(self, df: pd.DataFrame) -> pd.DataFrame:
        symbol_stats = (
            df.groupby("symbol")
            .agg(
                {
                    "Date": ["min", "max", "count"],
                    "Volume": ["sum", "mean"],
                    "Close": ["min", "max", "mean"],
                }
            )
            .reset_index()
        )

        symbol_stats.columns = [
            "symbol",
            "first_date",
            "last_date",
            "record_count",
            "total_volume",
            "avg_volume",
            "min_close",
            "max_close",
            "avg_close",
        ]

        return symbol_stats

    def _build_date_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"]).dt.strftime("%Y-%m-%d")

        date_stats = (
            df_copy.groupby("Date")
            .agg({"symbol": "count", "Volume": "sum", "Close": "mean"})
            .reset_index()
        )

        date_stats.columns = ["date", "symbol_count", "total_volume", "avg_close"]

        return date_stats

    def save_indexes(self) -> None:
        if not self._loaded:
            logger.warning("No indexes to save")
            return

        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        def save_parquet_or_csv(df, filepath):
            try:
                df.to_parquet(filepath, index=False)
            except Exception as e:
                logger.warning(f"Parquet not available, using CSV: {e}")
                filepath = filepath.with_suffix(".csv")
                df.to_csv(filepath, index=False)

        if self._symbol_index is not None:
            symbol_file = self.config.cache_dir / "index_symbols.parquet"
            save_parquet_or_csv(self._symbol_index, symbol_file)
            logger.info(f"Saved symbol index: {symbol_file}")

        if self._date_index is not None:
            date_file = self.config.cache_dir / "index_dates.parquet"
            save_parquet_or_csv(self._date_index, date_file)
            logger.info(f"Saved date index: {date_file}")

    def load_indexes(self) -> bool:
        symbol_file = self.config.cache_dir / "index_symbols.parquet"
        date_file = self.config.cache_dir / "index_dates.parquet"

        if not symbol_file.exists() or not date_file.exists():
            symbol_file = self.config.cache_dir / "index_symbols.csv"
            date_file = self.config.cache_dir / "index_dates.csv"
            if not symbol_file.exists() or not date_file.exists():
                logger.warning("Index files not found")
                return False

        try:
            if symbol_file.suffix == ".csv":
                self._symbol_index = pd.read_csv(symbol_file)
                self._date_index = pd.read_csv(date_file)
            else:
                self._symbol_index = pd.read_parquet(symbol_file)
                self._date_index = pd.read_parquet(date_file)
            self._loaded = True
            logger.info(
                f"Loaded indexes: {len(self._symbol_index)} symbols, {len(self._date_index)} dates"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            return False

        try:
            self._symbol_index = pd.read_parquet(symbol_file)
            self._date_index = pd.read_parquet(date_file)
            self._loaded = True
            logger.info(
                f"Loaded indexes: {len(self._symbol_index)} symbols, {len(self._date_index)} dates"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            return False

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        if not self._loaded:
            return None

        row = self._symbol_index[self._symbol_index["symbol"] == symbol.upper()]
        if row.empty:
            return None

        row = row.iloc[0]
        return {
            "symbol": row["symbol"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "record_count": row["record_count"],
            "total_volume": row["total_volume"],
            "avg_volume": row["avg_volume"],
            "min_close": row["min_close"],
            "max_close": row["max_close"],
            "avg_close": row["avg_close"],
        }

    def get_date_info(self, date: str) -> Optional[dict]:
        if not self._loaded:
            return None

        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        row = self._date_index[self._date_index["date"] == date_str]

        if row.empty:
            return None

        row = row.iloc[0]
        return {
            "date": row["date"],
            "symbol_count": row["symbol_count"],
            "total_volume": row["total_volume"],
            "avg_close": row["avg_close"],
        }

    def get_all_symbols(self) -> list[str]:
        if not self._loaded or self._symbol_index is None:
            return []
        return sorted(self._symbol_index["symbol"].tolist())

    def get_symbols_in_date_range(self, start_date: str, end_date: str) -> list[str]:
        if not self._loaded or self._date_index is None:
            return []

        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        mask = (self._date_index["date"] >= start) & (self._date_index["date"] <= end)
        dates = self._date_index[mask]["date"].tolist()

        return dates

    def query_fast(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        data_file = self.config.main_data_file

        if not data_file.exists():
            return pd.DataFrame()

        filters = [f"symbol == '{symbol.upper()}'"]

        if start_date:
            start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
            filters.append(f"Date >= '{start}'")

        if end_date:
            end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
            filters.append(f"Date <= '{end}'")

        query = " and ".join(filters)

        try:
            df = pd.read_csv(
                data_file,
                usecols=["Date", "Open", "High", "Low", "Close", "Volume", "symbol"],
            )
            df = df.query(query)
            return df
        except Exception as e:
            logger.error(f"Error in fast query: {e}")
            return pd.DataFrame()

    def rebuild_from_data(self, data_file: Optional[Path] = None) -> None:
        data_file = data_file or self.config.main_data_file

        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return

        logger.info(f"Rebuilding indexes from {data_file}...")

        df = pd.read_csv(data_file)

        self.build_indexes(df)
        self.save_indexes()

        logger.info("Indexes rebuilt and saved")
