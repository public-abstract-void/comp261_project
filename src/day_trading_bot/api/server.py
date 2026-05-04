import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from datetime import date, datetime
import logging
import re

from ..pipeline import TradingDataPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Day Trading Bot API",
    description="API for stock data and predictions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = PipelineConfig()
pipeline = TradingDataPipeline(config)


# Lazy-loaded mapping: SYMBOL -> Company Name
_SYMBOL_NAMES = None

def _load_symbol_names() -> dict:
    global _SYMBOL_NAMES
    if _SYMBOL_NAMES is not None:
        return _SYMBOL_NAMES

    names_file = config.project_root / 'data' / 'reference' / 'symbol_names.csv'
    mapping = {}
    if names_file.exists():
        try:
            df = pd.read_csv(names_file)
            cols = {c.lower(): c for c in df.columns}
            sym_col = cols.get('symbol')
            name_col = cols.get('name')
            if sym_col and name_col:
                for _, r in df.iterrows():
                    sym = str(r.get(sym_col, '')).strip().upper()
                    name = str(r.get(name_col, '')).strip()
                    if sym and name:
                        mapping[sym] = name
        except Exception:
            mapping = {}

    _SYMBOL_NAMES = mapping
    return _SYMBOL_NAMES


def _get_stock_name(symbol: str) -> str:
    m = _load_symbol_names()
    return m.get(symbol.upper(), symbol.upper())


@app.get("/")
def root():
    return {"message": "Day Trading Bot API", "version": "1.0.0"}


@app.get("/status")
def get_status():
    status = pipeline.get_status()
    return {
        "status": "healthy",
        "data_exists": status["data_exists"],
        "total_records": status["total_records"],
        "symbol_count": status["symbol_count"],
        "date_range": status["date_range"],
        "latest_version": status["latest_version"],
    }


@app.get("/stocks")
def get_stocks(
    symbols: Optional[str] = Query(
        None, description="Comma-separated symbols, e.g. AAPL,MSFT"
    ),
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    limit: Optional[int] = Query(1000, description="Max records to return"),
):
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

    start_date = None
    end_date = None

    if start:
        try:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid start date format. Use YYYY-MM-DD"
            )

    if end:
        try:
            end_date = datetime.strptime(end, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid end date format. Use YYYY-MM-DD"
            )

    df = pipeline.get_data(
        symbols=symbol_list, start_date=start_date, end_date=end_date
    )

    if df.empty:
        return {"data": [], "metadata": {"count": 0, "symbols_requested": symbol_list}}

    if limit and len(df) > limit:
        df = df.head(limit)

    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "symbol": row.get("symbol", ""),
                "date": str(row.get("Date", "")),
                "open": float(row.get("Open", 0))
                if pd.notna(row.get("Open"))
                else None,
                "high": float(row.get("High", 0))
                if pd.notna(row.get("High"))
                else None,
                "low": float(row.get("Low", 0)) if pd.notna(row.get("Low")) else None,
                "close": float(row.get("Close", 0))
                if pd.notna(row.get("Close"))
                else None,
                "volume": int(row.get("Volume", 0))
                if pd.notna(row.get("Volume"))
                else None,
            }
        )

    return {
        "data": data,
        "metadata": {
            "count": len(data),
            "symbols_requested": symbol_list,
            "date_range": {"start": start, "end": end},
        },
    }


@app.get("/stocks/{symbol}")
def get_stock(symbol: str):
    symbol = symbol.upper()

    df = pipeline.get_data(symbols=[symbol])

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    df = df.tail(30)

    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "date": str(row.get("Date", "")),
                "close": float(row.get("Close", 0))
                if pd.notna(row.get("Close"))
                else None,
                "volume": int(row.get("Volume", 0))
                if pd.notna(row.get("Volume"))
                else None,
            }
        )

    return {"symbol": symbol, "recent_data": data, "metadata": {"count": len(data)}}


@app.get("/symbols")
def get_symbols():
    indexer = pipeline.indexer
    indexer.load_indexes()
    symbols = indexer.get_all_symbols()

    return {"symbols": symbols[:100], "total": len(symbols)}




@app.get("/predictions")
def get_predictions(
    limit: int = Query(50, description="Max predictions to return"),
    horizon: str = Query("1d", description="Prediction horizon: 1d, 5d, 10d"),
):
    """Placeholder batch predictions for frontend integration.

    Uses `data/trading_symbols.txt` as the preferred symbol list so the UI shows
    sensible tickers instead of every symbol in the dataset.
    """

    if horizon not in {"1d", "5d", "10d"}:
        raise HTTPException(status_code=400, detail="horizon must be one of: 1d, 5d, 10d")

    # Only show normal-looking tickers (e.g. AAPL, MSFT).
    ticker_re = re.compile(r'^[A-Z]{1,5}$')

    # Load symbol index for fast existence checks.
    pipeline.indexer.load_indexes()
    available = {s for s in pipeline.indexer.get_all_symbols() if ticker_re.match(s)}



    # Prefer S&P 500 template list if present, else use trading_symbols.txt, else scraper_stocks.
    sp500_file = config.project_root / "data" / "reference" / "sp500_symbols_template.csv"
    preferred = []
    if sp500_file.exists():
        try:
            import pandas as _pd

            _df = _pd.read_csv(sp500_file)
            # accept columns: ticker or symbol
            cols = {c.lower(): c for c in _df.columns}
            col = cols.get("ticker") or cols.get("symbol")
            if col:
                preferred = [str(x).strip().upper() for x in _df[col].tolist()]
        except Exception:
            preferred = []

    if not preferred:
        preferred = list(config.load_trading_symbols())

    preferred = [s for s in preferred if ticker_re.match(s)]

    # Keep only tickers that exist in our merged dataset.
    symbols = [s for s in preferred if s in available]

    # If the curated list is empty (or doesn't intersect), fallback to first N available symbols.
    if not symbols:
        symbols = sorted(list(available))

    symbols = symbols[: max(0, min(limit, len(symbols)))]

    preds = []
    for s in symbols:
        # Deterministic placeholder confidence based on symbol.
        conf = (sum(ord(c) for c in s) % 50) / 100.0 + 0.50
        preds.append(
            {
                "symbol": s,
                "stock_name": _get_stock_name(s),
                "confidence": round(conf, 3),
                "horizon": horizon,
                "model_version": "v0.1-placeholder",
            }
        )

    return {"predictions": preds, "count": len(preds)}


@app.post("/update")
def run_update(
    symbols: Optional[str] = Query(None, description="Comma-separated symbols, e.g. AAPL,MSFT"),
    force: bool = Query(False, description="Force full re-check"),
    dry_run: bool = Query(
        False,
        description="If true, only fetch+validate counts (does not merge/write the 25M-row dataset)",
    ),
):
    """Runs a pipeline daily update.

    Use dry_run=true for a fast "Update Info" button that won't rewrite the whole dataset.
    """

    symset = None
    if symbols:
        symset = {s.strip().upper() for s in symbols.split(",") if s.strip()}

    if dry_run:
        fetched = pipeline.fetcher.fetch_incremental(symset)
        v = pipeline.validator.validate(fetched, strict=False)
        return {
            "dry_run": True,
            "symbols": sorted(symset) if symset else None,
            "fetched_rows": int(len(fetched)),
            "valid": bool(v.valid),
            "errors": v.errors,
            "warnings": v.warnings,
        }

    result = pipeline.run_daily_update(symbols=symset, force_full=force)
    return {
        "dry_run": False,
        "success": result.success,
        "records_added": result.records_added,
        "records_modified": result.records_modified,
        "version_tag": result.version_tag,
        "errors": result.errors,
    }


@app.get("/predict")
def get_prediction(symbol: str = Query(..., description="Stock symbol")):
    symbol = symbol.upper()

    df = pipeline.get_data(symbols=[symbol])

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    last_row = df.iloc[-1]
    close = last_row.get("Close", 0)

    return {
        "symbol": symbol,
        "prediction": "HOLD",
        "confidence": 0.5,
        "model_version": "v0.1-placeholder",
        "reasoning": "ML model not yet integrated. Placeholder response.",
        "current_price": float(close) if pd.notna(close) else None,
        "note": "Chris needs to integrate his ML model here",
    }


@app.get("/versions")
def get_versions():
    versions = pipeline.list_versions()
    return {"versions": versions}


@app.get("/audit")
def get_audit(limit: int = Query(50, description="Number of entries")):
    df = pipeline.get_audit_log(limit)
    if df.empty:
        return {"audit": [], "message": "No audit entries"}
