from __future__ import annotations
import logging
from datetime import date, datetime
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetches market regime data: VIX, economic calendar, earnings."""

    def __init__(self):
        self._vix_cache: Optional[pd.DataFrame] = None

    def get_vix(self) -> Optional[float]:
        """Get current VIX (volatility index) for risk detection."""
        try:
            import yfinance

            vix = yfinance.Ticker("^VIX")
            df = vix.history(period="1d")

            if df.empty:
                return None

            current_vix = df["Close"].iloc[-1]
            logger.info(f"Current VIX: {current_vix:.2f}")
            return float(current_vix)

        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}")
            return None

    def get_vix_history(self, days: int = 30) -> pd.DataFrame:
        """Get VIX history for trend analysis."""
        try:
            import yfinance

            vix = yfinance.Ticker("^VIX")
            df = vix.history(period=f"{days}d")
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            df = df.rename(columns={"Close": "VIX"})

            return df[["Date", "VIX"]]

        except Exception as e:
            logger.warning(f"Failed to fetch VIX history: {e}")
            return pd.DataFrame()

    def get_earnings_calendar(
        self, symbols: list[str], days_ahead: int = 30
    ) -> pd.DataFrame:
        """Get upcoming earnings dates for symbols."""
        import yfinance

        earnings_data = []
        today = date.today()

        for symbol in symbols:
            try:
                ticker = yfinance.Ticker(symbol)
                info = ticker.info

                if info:
                    earnings_date = info.get("earningsDate")
                    if earnings_date:
                        days_until = None
                        if isinstance(earnings_date, (date, datetime)):
                            days_until = (earnings_date.date() - today).days

                        earnings_data.append(
                            {
                                "symbol": symbol.upper(),
                                "earnings_date": str(earnings_date)[:10],
                                "days_until": days_until,
                            }
                        )

            except Exception as e:
                logger.debug(f"Earnings fetch failed for {symbol}: {e}")
                continue

        if not earnings_data:
            return pd.DataFrame()

        df = pd.DataFrame(earnings_data)
        df = df.sort_values("earnings_date")

        logger.info(f"Found earnings dates for {len(df)} symbols")
        return df

    def get_fred_indicators(self) -> dict:
        """Fetch key economic indicators from FRED (Federal Reserve Economic Data)."""
        indicators = {
            "^DFF": "Federal Funds Rate",
            "^TNX": "10-Year Treasury Rate",
        }

        import yfinance

        results = {}

        for ticker, name in indicators.items():
            try:
                data = yfinance.Ticker(ticker)
                df = data.history(period="1d")

                if not df.empty:
                    results[name] = {
                        "value": float(df["Close"].iloc[-1]),
                        "date": str(df.index[-1])[:10],
                    }
            except Exception as e:
                logger.debug(f"FRED fetch failed for {ticker}: {e}")

        return results

    def is_high_risk_period(self, vix_threshold: float = 25.0) -> bool:
        """Check if current market regime is high risk."""
        vix = self.get_vix()

        if vix is None:
            return False

        is_high_risk = vix > vix_threshold

        if is_high_risk:
            logger.warning(f"HIGH RISK: VIX is {vix:.2f} (threshold: {vix_threshold})")

        return is_high_risk

    def get_market_status(self) -> dict:
        """Check if market is open, closed, or pre/after hours."""
        now = datetime.now()

        weekday = now.weekday()
        market_time = now.time()

        is_weekend = weekday >= 5
        market_open = market_time >= datetime.strptime("09:30", "%H:%M").time()
        market_close = market_time < datetime.strptime("16:00", "%H:%M").time()

        if is_weekend:
            status = "closed"
        elif not market_open:
            status = "pre_market"
        elif not market_close:
            status = "after_hours"
        else:
            status = "open"

        return {
            "status": status,
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "is_trading_hours": status == "open",
        }
