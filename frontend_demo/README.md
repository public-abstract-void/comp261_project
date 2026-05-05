# Frontend Demo (Local)

This folder is a lightweight frontend demo that consumes the backend API.

## Run

1. Start the backend API (from repo root):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src uvicorn day_trading_bot.api.server:app --reload --port 8001
```

2. Serve this folder (from repo root):

```sh
cd frontend_demo
python3 -m http.server 5173
```

3. Open:

`http://127.0.0.1:5173`

## What It Calls

- `GET /predictions?limit=10` for top picks (stock name + confidence)
- `GET /stocks/{symbol}` for latest close price
- (optional) `GET /status` for dataset stats
