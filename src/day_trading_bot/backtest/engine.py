"""
Minimal backtester — correct over fancy.
- Time-ordered, no lookahead (signal at bar[t] executes at bar[t+1] open)
- Transaction costs + slippage
- Tracks: PnL, Sharpe, max drawdown, win rate
- Baseline strategies: SMA crossover, momentum
"""

from __future__ import annotations
import json, logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ANNUALISATION = 252


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001       # 0.1% per leg
    slippage_pct:   float = 0.0005      # 0.05% market impact
    position_size_pct: float = 1.0      # fraction of capital per trade
    risk_free_rate: float = 0.04
    report_dir: Path = Path("data/processed")


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    final_equity: float


# ── built-in strategies ──────────────────────────────────────────────────────

def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """1 = long, 0 = flat. Uses only past closes — no lookahead."""
    c = df["close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(int)


def momentum(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """1 = long if past-N return positive, else 0."""
    return (df["close"].pct_change(lookback) > 0).astype(int)


STRATEGIES: dict[str, Callable] = {
    "sma_crossover": sma_crossover,
    "momentum": momentum,
}


# ── engine ───────────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, df: pd.DataFrame, cfg: Optional[BacktestConfig] = None):
        self.df  = df.sort_values("timestamp").reset_index(drop=True)
        self.cfg = cfg or BacktestConfig()
        self._check_no_lookahead()

    def _check_no_lookahead(self):
        if not self.df["timestamp"].is_monotonic_increasing:
            raise ValueError("Data not sorted — lookahead risk detected.")

    def run(self, strategy: str | Callable = "sma_crossover",
            symbol: str = "UNKNOWN", **kwargs) -> BacktestResult:

        name = strategy if isinstance(strategy, str) else strategy.__name__
        fn   = STRATEGIES[strategy] if isinstance(strategy, str) else strategy

        # Signal at bar[t], execute at bar[t+1] open — the key no-lookahead shift
        signals = fn(self.df, **kwargs).shift(1).fillna(0)

        equity, trades = self._simulate(signals)
        result = self._metrics(equity, trades, name, symbol)
        self._save(result)
        return result

    def _simulate(self, signals: pd.Series):
        cap, shares, in_pos, entry = self.cfg.initial_capital, 0.0, False, 0.0
        equity, trades = [], []

        for i, row in self.df.iterrows():
            sig = int(signals.iloc[i])
            o, c = row["open"], row["close"]

            if sig == 1 and not in_pos:
                exec_p = o * (1 + self.cfg.slippage_pct)
                cost   = cap * self.cfg.position_size_pct
                shares = cost * (1 - self.cfg.commission_pct) / exec_p
                cap   -= cost; entry = exec_p; in_pos = True
                trades.append({"side": "BUY", "bar": i, "price": exec_p})

            elif sig == 0 and in_pos:
                exec_p  = o * (1 - self.cfg.slippage_pct)
                proceeds = shares * exec_p * (1 - self.cfg.commission_pct)
                trades.append({"side": "SELL", "bar": i, "price": exec_p,
                               "pnl": proceeds - shares * entry})
                cap += proceeds; shares = 0.0; in_pos = False

            equity.append(cap + shares * c)

        return pd.Series(equity), trades

    def _metrics(self, equity, trades, name, symbol) -> BacktestResult:
        init, final = self.cfg.initial_capital, equity.iloc[-1]
        yrs   = len(equity) / ANNUALISATION
        ret   = (final - init) / init * 100
        # CAGR formula: (final/initial)^(1/years) - 1
        pow_term = max(yrs, 1e-9)
        ann = ((final / init) ** (1.0 / pow_term) - 1.0) * 100.0
        dr    = equity.pct_change().dropna()
        ex    = dr - self.cfg.risk_free_rate / ANNUALISATION
        sharpe = ex.mean() / ex.std() * np.sqrt(ANNUALISATION) if ex.std() > 0 else 0.0
        dd    = ((equity - equity.cummax()) / equity.cummax()).min() * 100
        sells = [t for t in trades if t["side"] == "SELL"]
        wr    = sum(1 for t in sells if t["pnl"] > 0) / len(sells) * 100 if sells else 0.0

        return BacktestResult(
            strategy=name, symbol=symbol,
            start_date=str(self.df["timestamp"].iloc[0])[:10],
            end_date=str(self.df["timestamp"].iloc[-1])[:10],
            total_return_pct=round(ret, 4),
            annualised_return_pct=round(ann, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown_pct=round(dd, 4),
            win_rate_pct=round(wr, 2),
            num_trades=len(sells),
            final_equity=round(final, 2),
        )

    def _save(self, result: BacktestResult):
        p = Path(self.cfg.report_dir) / "backtest_report.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(result)), indent=2)
        logger.info("Report saved -> %s", p)