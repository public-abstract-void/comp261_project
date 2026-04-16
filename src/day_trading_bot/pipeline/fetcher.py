from __future__ import annotations
import hashlib
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig, ValidationResult

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._last_fetch_date: Optional[date] = None
        self._load_cursor()

    def _load_cursor(self) -> None:
        cursor_file = self.config.cursor_file
        if cursor_file.exists():
            try:
                self._last_fetch_date = datetime.fromisoformat(
                    cursor_file.read_text().strip()
                ).date()
                logger.info(f"Loaded cursor: last fetch date = {self._last_fetch_date}")
            except Exception as e:
                logger.warning(f"Could not load cursor: {e}")
                self._last_fetch_date = None

    def save_cursor(self, fetch_date: date) -> None:
        self.config.cursor_file.write_text(fetch_date.isoformat())
        self._last_fetch_date = fetch_date
        logger.info(f"Saved cursor: fetch_date = {fetch_date}")

    def get_last_fetch_date(self) -> Optional[date]:
        return self._last_fetch_date

    def fetch_incremental(
        self,
        symbols: Optional[set[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        use_full_list: bool = True,
    ) -> pd.DataFrame:
        symbols = (
            symbols or self.config.load_trading_symbols()
            if use_full_list
            else self.config.scraper_stocks
        )
        start_date = start_date or self._last_fetch_date or date(2017, 1, 1)
        end_date = end_date or date.today()

        if start_date >= end_date:
            logger.info("No new data to fetch (start >= end)")
            return pd.DataFrame()

        logger.info(
            f"Fetching incremental data: {len(symbols)} symbols, {start_date} to {end_date}"
        )

        all_data = []

        for symbol in symbols:
            data = self._fetch_symbol(symbol, start_date, end_date)
            if not data.empty:
                all_data.append(data)

        if not all_data:
            logger.info("No data fetched from any symbol")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"Fetched {len(result)} total records")

        return result

    def _fetch_symbol(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        scraper_csv = self.config.scraper_dir / f"{symbol.lower()}.csv"

        if not scraper_csv.exists():
            logger.warning(f"Scraper file not found: {symbol}, fetching live...")
            return self._fetch_live(symbol, start_date, end_date)

        try:
            df = pd.read_csv(scraper_csv)

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])

            df["Date"] = df["Date"].dt.date

            mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
            filtered = df[mask].copy()

            if not filtered.empty:
                filtered["symbol"] = symbol.upper()

            last_scraper_date = filtered["Date"].max() if not filtered.empty else None
            if (
                last_scraper_date is not None
                and pd.notna(last_scraper_date)
                and last_scraper_date < end_date
            ):
                live_start = last_scraper_date + timedelta(days=1)
                live_data = self._fetch_live(symbol, live_start, end_date)
                if live_data is not None and not live_data.empty:
                    filtered = pd.concat([filtered, live_data], ignore_index=True)
            elif filtered.empty:
                live_data = self._fetch_live(symbol, start_date, end_date)
                if live_data is not None and not live_data.empty:
                    filtered = live_data

            return filtered

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_full(self, symbols: Optional[set[str]] = None) -> pd.DataFrame:
        symbols = symbols or self.config.scraper_stocks

        logger.info(f"Fetching full data: {len(symbols)} symbols")

        all_data = []

        for symbol in symbols:
            scraper_csv = self.config.scraper_dir / f"{symbol.lower()}.csv"

            if not scraper_csv.exists():
                logger.warning(f"Scraper file not found: {scraper_csv}")
                continue

            try:
                df = pd.read_csv(scraper_csv)
                df["symbol"] = symbol.upper()
                all_data.append(df)
                logger.info(f"Loaded {symbol}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        logger.info(
            f"Total loaded: {len(result)} records, {result['symbol'].nunique()} symbols"
        )

        return result

    def compute_record_hash(self, row: pd.Series) -> str:
        key = f"{row.get('symbol', '')}|{row.get('Date', '')}|{row.get('Close', '')}"
        return hashlib.md5(key.encode()).hexdigest()

    def _fetch_live(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        df = self._fetch_yfinance(symbol, start_date, end_date)

        if df.empty:
            logger.info(f"yfinance failed for {symbol}, trying Alpha Vantage...")
            df = self._fetch_alpha_vantage(symbol, start_date, end_date)

        return df

    def _fetch_yfinance(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        try:
            import yfinance

            ticker = yfinance.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            df = df.rename(
                columns={
                    "Date": "Date",
                    "Open": "Open",
                    "High": "High",
                    "Low": "Low",
                    "Close": "Close",
                    "Volume": "Volume",
                }
            )
            df["Date"] = df["Date"].dt.date
            df["symbol"] = symbol.upper()

            logger.info(f"Live fetch {symbol}: {len(df)} records")
            return df
        except ImportError:
            logger.warning(
                "yfinance not installed - install with: pip install yfinance"
            )
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_alpha_vantage(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        try:
            from alpha_vantage import foreignexchanger, stocktimeseries
            import os

            api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")

            ts = stocktimeseries.TimeSeries(key=api_key, output_format="pandas")
            data, meta = ts.get_daily_adjusted(symbol=symbol, outputsize="compact")

            if data is None or data.empty:
                return pd.DataFrame()

            data = data.reset_index()
            data.columns = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]
            data["Date"] = pd.to_datetime(data["Date"]).dt.date

            mask = (data["Date"] >= start_date) & (data["Date"] <= end_date)
            filtered = data[mask].copy()
            filtered["symbol"] = symbol.upper()

            logger.info(f"Alpha Vantage fetch {symbol}: {len(filtered)} records")
            return filtered
        except ImportError:
            logger.warning("alpha-vantage not installed")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_and_hash(self, symbols: Optional[set[str]] = None) -> pd.DataFrame:
        df = self.fetch_full(symbols)

        if df.empty:
            return df

        df["_row_hash"] = df.apply(self.compute_record_hash, axis=1)

        return df

    def fetch_intraday(
        self, symbol: str, interval: str = "1m", period: str = "5d"
    ) -> pd.DataFrame:
        """Fetch intraday 1-minute data for day trading.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval (1m, 5m, 15m, 1h, etc.)
            period: Data period (1d, 5d, 1mo, etc.)

        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            import yfinance

            ticker = yfinance.Ticker(symbol)
            df = ticker.history(interval=interval, period=period)

            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            df["Datetime"] = df["Datetime"].dt.tz_localize(None)
            df["symbol"] = symbol.upper()

            logger.info(f"Intraday {symbol}: {len(df)} {interval} bars")
            return df

        except Exception as e:
            logger.warning(f"Intraday fetch failed for {symbol}: {e}")
            return pd.DataFrame()
