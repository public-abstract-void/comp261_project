"""
Structured logger for the data pipeline.
Logs: data update success/failure, validation events, API health.
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)

    # File handler — structured JSON lines
    fh = logging.FileHandler(LOG_DIR / "pipeline.log")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    return logger


def log_event(logger: logging.Logger, event: str, status: str, **kwargs):
    """Write a structured JSON log line."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status": status,
        **kwargs,
    }
    logger.info(json.dumps(record))