import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from datetime import date, datetime
import logging

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


import pandas as pd


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

    return {"audit": df.to_dict(orient="records")}
