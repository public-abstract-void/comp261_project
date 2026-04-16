from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class DataVersioner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._current_version: Optional[str] = None

    def create_version(
        self, df: pd.DataFrame, version_tag: Optional[str] = None
    ) -> str:
        if version_tag is None:
            version_tag = self._generate_version_tag()

        version_dir = self.config.versions_dir / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)

        version_file = version_dir / "data.parquet"

        compression = self.config.compression

        try:
            df.to_parquet(version_file, compression=compression, index=False)
        except Exception as e:
            logger.warning(f"Parquet not available, using CSV: {e}")
            version_file = version_dir / "data.csv"
            df.to_csv(version_file, index=False)
            compression = "csv"

        checksum = self._compute_checksum(df)

        metadata = {
            "version": version_tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(df),
            "symbol_count": int(df["symbol"].nunique()) if not df.empty else 0,
            "date_range": {
                "min": str(df["Date"].min()) if not df.empty else None,
                "max": str(df["Date"].max()) if not df.empty else None,
            },
            "checksum": checksum,
            "file_size_bytes": version_file.stat().st_size,
            "compression": compression,
        }

        metadata_file = version_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        self._update_latest(version_tag)
        self._append_audit_log("CREATE", version_tag, len(df), checksum)

        logger.info(
            f"Created version {version_tag}: {len(df)} records, checksum={checksum[:8]}"
        )

        self._current_version = version_tag

        return version_tag

    def load_version(self, version_tag: str) -> pd.DataFrame:
        version_dir = self.config.versions_dir / version_tag
        version_file = version_dir / "data.parquet"

        if not version_file.exists():
            version_file = version_dir / "data.csv"
            if not version_file.exists():
                raise FileNotFoundError(f"Version not found: {version_tag}")
            df = pd.read_csv(version_file)
        else:
            df = pd.read_parquet(version_file)
        logger.info(f"Loaded version {version_tag}: {len(df)} records")

        return df

    def load_latest(self) -> Optional[pd.DataFrame]:
        latest_file = self.config.latest_version_file

        if not latest_file.exists():
            logger.warning("No latest version found")
            return None

        try:
            version_tag = latest_file.read_text().strip()
            return self.load_version(version_tag)
        except Exception as e:
            logger.error(f"Error loading latest version: {e}")
            return None

    def list_versions(self) -> list[dict]:
        versions = []

        if not self.config.versions_dir.exists():
            return versions

        for v_dir in sorted(self.config.versions_dir.iterdir()):
            if not v_dir.is_dir():
                continue

            metadata_file = v_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    versions.append(metadata)
                except Exception as e:
                    logger.warning(f"Error reading version {v_dir.name}: {e}")

        return sorted(versions, key=lambda x: x.get("created_at", ""), reverse=True)

    def cleanup_old_versions(self, keep_count: Optional[int] = None) -> int:
        keep_count = keep_count or self.config.keep_versions

        versions = self.list_versions()

        if len(versions) <= keep_count:
            logger.info(
                f"No cleanup needed: {len(versions)} versions, keeping {keep_count}"
            )
            return 0

        to_delete = versions[keep_count:]
        deleted_count = 0

        for v in to_delete:
            version_tag = v.get("version")
            if version_tag:
                version_dir = self.config.versions_dir / version_tag
                if version_dir.exists():
                    import shutil

                    shutil.rmtree(version_dir)
                    self._append_audit_log("DELETE", version_tag, 0, "")
                    deleted_count += 1
                    logger.info(f"Deleted old version: {version_tag}")

        logger.info(f"Cleaned up {deleted_count} old versions")

        return deleted_count

    def get_current_version(self) -> Optional[str]:
        if self._current_version:
            return self._current_version

        latest_file = self.config.latest_version_file
        if latest_file.exists():
            self._current_version = latest_file.read_text().strip()

        return self._current_version

    def verify_checksum(self, version_tag: str) -> bool:
        version_dir = self.config.versions_dir / version_tag
        metadata_file = version_dir / "metadata.json"
        version_file = version_dir / "data.parquet"

        if not metadata_file.exists() or not version_file.exists():
            return False

        try:
            metadata = json.loads(metadata_file.read_text())
            stored_checksum = metadata.get("checksum", "")

            version_file = version_dir / "data.parquet"
            if not version_file.exists():
                version_file = version_dir / "data.csv"

            if version_file.suffix == ".csv":
                df = pd.read_csv(version_file)
            else:
                df = pd.read_parquet(version_file)

            current_checksum = self._compute_checksum(df)

            return stored_checksum == current_checksum
        except Exception as e:
            logger.error(f"Error verifying checksum: {e}")
            return False

    def _generate_version_tag(self) -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y%m%d_%H%M%S")

    def _compute_checksum(self, df: pd.DataFrame) -> str:
        if df.empty:
            return hashlib.sha256(b"empty").hexdigest()

        sample = df.head(10000) if len(df) > 10000 else df

        content = json.dumps(
            {
                "len": len(df),
                "symbols": sorted(df["symbol"].unique().tolist())[:100],
                "date_range": [str(df["Date"].min()), str(df["Date"].max())],
            },
            sort_keys=True,
        )

        return hashlib.sha256(content.encode()).hexdigest()

    def _update_latest(self, version_tag: str) -> None:
        self.config.latest_version_file.write_text(version_tag)
        logger.debug(f"Updated latest version to {version_tag}")

    def _append_audit_log(
        self, action: str, version_tag: str, record_count: int, checksum: str
    ) -> None:
        audit_file = self.config.versions_dir / "audit_log.jsonl"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "version": version_tag,
            "record_count": record_count,
            "checksum": checksum,
        }

        with open(audit_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.debug(f"Audit log: {action} {version_tag}")

    def get_audit_log(self, limit: Optional[int] = None) -> pd.DataFrame:
        audit_file = self.config.versions_dir / "audit_log.jsonl"

        if not audit_file.exists():
            return pd.DataFrame()

        entries = []
        with open(audit_file, "r") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except:
                    pass

        df = pd.DataFrame(entries)

        if limit:
            df = df.tail(limit)

        return df
