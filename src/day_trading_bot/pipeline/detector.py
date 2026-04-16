from __future__ import annotations
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ChangeSummary:
    added_count: int
    modified_count: int  # Records with same (symbol, date) but different content
    deleted_count: int  # Records that existed before but not in new data
    unchanged_count: int
    total_old: int
    total_new: int

    @property
    def has_changes(self) -> bool:
        return self.added_count > 0 or self.modified_count > 0 or self.deleted_count > 0

    def to_dict(self) -> dict:
        return {
            "added": self.added_count,
            "modified": self.modified_count,
            "deleted": self.deleted_count,
            "unchanged": self.unchanged_count,
            "total_old": self.total_old,
            "total_new": self.total_new,
        }


class ChangeDetector:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._old_index: dict[tuple[str, str], str] = {}

    def build_index(self, df: pd.DataFrame) -> dict[tuple[str, str], str]:
        index = {}

        if df.empty:
            return index

        df = df.copy()

        df["_Date_str"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        for _, row in df.iterrows():
            key = (str(row.get("symbol", "")).upper(), str(row.get("_Date_str", "")))
            content = self._compute_content_hash(row)
            index[key] = content

        logger.info(f"Built index with {len(index)} entries")

        return index

    def _compute_content_hash(self, row: pd.Series) -> str:
        important_cols = ["Open", "High", "Low", "Close", "Volume"]
        values = [str(row.get(col, "")) for col in important_cols]
        return hashlib.sha256("|".join(values).encode()).hexdigest()[:16]

    def detect(
        self, old_data: pd.DataFrame, new_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, ChangeSummary]:
        if old_data.empty and new_data.empty:
            summary = ChangeSummary(
                added_count=0,
                modified_count=0,
                deleted_count=0,
                unchanged_count=0,
                total_old=0,
                total_new=0,
            )
            return pd.DataFrame(), summary

        if old_data.empty:
            summary = ChangeSummary(
                added_count=len(new_data),
                modified_count=0,
                deleted_count=0,
                unchanged_count=0,
                total_old=0,
                total_new=len(new_data),
            )
            return new_data.copy(), summary

        if new_data.empty:
            summary = ChangeSummary(
                added_count=0,
                modified_count=0,
                deleted_count=len(old_data),
                unchanged_count=0,
                total_old=len(old_data),
                total_new=0,
            )
            return pd.DataFrame(), summary

        self._old_index = self.build_index(old_data)

        new_data = new_data.copy()
        new_data["_Date_str"] = pd.to_datetime(new_data["Date"]).dt.strftime("%Y-%m-%d")

        added = []
        modified = []
        unchanged = []

        for idx, row in new_data.iterrows():
            key = (str(row.get("symbol", "")).upper(), str(row.get("_Date_str", "")))
            new_content = self._compute_content_hash(row)

            if key not in self._old_index:
                added.append(idx)
            elif self._old_index[key] != new_content:
                modified.append(idx)
            else:
                unchanged.append(idx)

        result_df = new_data.loc[added + modified].copy()
        result_df = result_df.drop(columns=["_Date_str"], errors="ignore")

        summary = ChangeSummary(
            added_count=len(added),
            modified_count=len(modified),
            deleted_count=0,
            unchanged_count=len(unchanged),
            total_old=len(old_data),
            total_new=len(new_data),
        )

        logger.info(
            f"Changes detected: {summary.added_count} added, {summary.modified_count} modified"
        )

        return result_df, summary

    def detect_full(
        self, old_data: pd.DataFrame, new_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, ChangeSummary]:
        if old_data.empty and new_data.empty:
            summary = ChangeSummary(
                added_count=0,
                modified_count=0,
                deleted_count=0,
                unchanged_count=0,
                total_old=0,
                total_new=0,
            )
            return pd.DataFrame(), summary

        if old_data.empty:
            summary = ChangeSummary(
                added_count=len(new_data),
                modified_count=0,
                deleted_count=0,
                unchanged_count=0,
                total_old=0,
                total_new=len(new_data),
            )
            return new_data.copy(), summary

        if new_data.empty:
            summary = ChangeSummary(
                added_count=0,
                modified_count=0,
                deleted_count=len(old_data),
                unchanged_count=0,
                total_old=len(old_data),
                total_new=0,
            )
            return pd.DataFrame(), summary

        self._old_index = self.build_index(old_data)

        old_data = old_data.copy()
        old_data["_Date_str"] = pd.to_datetime(old_data["Date"]).dt.strftime("%Y-%m-%d")

        new_data = new_data.copy()
        new_data["_Date_str"] = pd.to_datetime(new_data["Date"]).dt.strftime("%Y-%m-%d")

        new_symbols = set(new_data["symbol"].unique())
        old_symbols = set(old_data["symbol"].unique())

        deleted_symbols = old_symbols - new_symbols

        deleted_keys = set()
        for sym in deleted_symbols:
            for date_str in old_data[old_data["symbol"] == sym]["_Date_str"].unique():
                deleted_keys.add((sym, date_str))

        added = []
        modified = []
        unchanged = []

        for idx, row in new_data.iterrows():
            key = (str(row.get("symbol", "")).upper(), str(row.get("_Date_str", "")))
            new_content = self._compute_content_hash(row)

            if key not in self._old_index:
                added.append(idx)
            elif self._old_index[key] != new_content:
                modified.append(idx)
            else:
                unchanged.append(idx)

        added_count = len(added)
        modified_count = len(modified)
        deleted_count = len(deleted_keys)

        result_df = new_data.loc[added + modified].copy()
        result_df = result_df.drop(columns=["_Date_str"], errors="ignore")

        summary = ChangeSummary(
            added_count=added_count,
            modified_count=modified_count,
            deleted_count=deleted_count,
            unchanged_count=len(unchanged),
            total_old=len(old_data),
            total_new=len(new_data),
        )

        logger.info(
            f"Full changes detected: {summary.added_count} added, {summary.modified_count} modified, {summary.deleted_count} deleted"
        )

        return result_df, summary

    def load_existing_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        file_path = file_path or self.config.main_data_file

        if not file_path.exists():
            logger.warning(f"Existing data file not found: {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)
            logger.info(
                f"Loaded existing data: {len(df)} rows, {df['symbol'].nunique()} symbols"
            )
            return df
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            return pd.DataFrame()
