from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )

    # Data directories
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    versions_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    # External sources
    kaggle_dir: Path = field(
        default_factory=lambda: Path("/Users/allanodora/Downloads/archive/Stocks")
    )
    scraper_dir: Path = field(
        default_factory=lambda: Path(
            "/Users/allanodora/Downloads/stock_scrapper/stock_data_api_hybrid"
        )
    )

    # Scraper stocks (currently working)
    scraper_stocks: set[str] = field(default_factory=lambda: {"AAPL", "MSFT", "NVDA"})

    # Trading symbols list (S&P 500 / top stocks)
    trading_symbols_file: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "data"
        / "trading_symbols.txt"
    )

    def load_trading_symbols(self) -> set[str]:
        if self.trading_symbols_file.exists():
            with open(self.trading_symbols_file) as f:
                return {line.strip().upper() for line in f if line.strip()}
        return self.scraper_stocks

    # Performance settings
    batch_size: int = 10000
    chunk_size: int = 5000
    max_workers: int = 4

    # Validation thresholds
    max_price_gap_pct: float = 50.0  # Max 50% gap from previous close
    min_volume: int = 100
    max_volume: int = 10_000_000_000

    # Versioning
    keep_versions: int = 30
    compression: str = "snappy"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.versions_dir = self.data_dir / "versions"
        self.cache_dir = self.data_dir / "cache"

        for d in [self.raw_dir, self.processed_dir, self.versions_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def main_data_file(self) -> Path:
        return self.processed_dir / "training_2017_2026.csv"

    @property
    def legacy_data_file(self) -> Path:
        return self.project_root / "training_2017_2026.csv"

    @property
    def src_data_file(self) -> Path:
        return (
            self.project_root / "src" / "data" / "processed" / "training_2017_2026.csv"
        )

    @property
    def latest_version_file(self) -> Path:
        return self.versions_dir / "latest.txt"

    @property
    def cursor_file(self) -> Path:
        return self.cache_dir / "last_update_cursor.txt"

    @property
    def index_file(self) -> Path:
        return self.cache_dir / "symbol_date_index.parquet"


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    records_checked: int = 0
    records_valid: int = 0

    def merge(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            records_checked=self.records_checked + other.records_checked,
            records_valid=self.records_valid + other.records_valid,
        )


@dataclass
class UpdateResult:
    success: bool
    records_added: int = 0
    records_modified: int = 0
    records_rejected: int = 0
    version_tag: Optional[str] = None
    execution_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
