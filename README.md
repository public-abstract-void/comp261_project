# Day Trading Bot (Academic Project)

Team: Allan, Christopher, Junah  

## Project Layout
- `data/raw/Stocks/`: Raw Kaggle stock CSV files
- `data/processed/`: Cleaned/derived datasets
- `notebooks/`: EDA notebooks (optional)
- `scripts/`: Runnable scripts for data tasks
- `src/day_trading_bot/`: Reusable project modules
- `tests/`: Unit tests (later)

---

# Data Pipeline (Allan)

## Quick Start
```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fetch new stock data
python scripts/daily_update.py

# Merge data for ML
python scripts/merge_for_ml.py
```

## Output
- `data/processed/full_cleaned_data.csv` - Merged dataset for ML model

## Merge Script Options
```bash
# Preview merge (no file written)
python scripts/merge_for_ml.py --dry-run

# Custom paths
python scripts/merge_for_ml.py --chris-data path/to/chris.csv --allan-data path/to/allan.csv
```

---

# Legacy / Chris's Pipeline

## Quick Start
1. Create a virtual environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Place Kaggle stock CSV files in `data/raw/Stocks/`
3. Run the prototype loader:
   - `python scripts/load_one_stock.py --ticker AAPL`

## Expected Kaggle File Pattern
- `data/raw/Stocks/AAPL.us.txt`

## S&P 500 Filter Tool (Week 1)
This is a data-prep script (not a UI feature). It filters raw stock files down to the symbols you provide.

### What it is for
- Keep only S&P 500 ticker files from a large raw dataset
- Produce a clean list for later cleaning/indicators/backtesting
- Handle empty files automatically

### Empty-file rule
- If empty files are few (<= threshold): ignore and continue
- If empty files are many (> threshold): skip them and generate a skip report

### Run example
```bash
python scripts/filter_sp500_files.py --stocks-dir "data/raw/Stocks" --symbols-file "data/reference/sp500_symbols_template.csv"
```

### Optional: copy filtered files into a folder
```bash
python scripts/filter_sp500_files.py --stocks-dir "data/raw/Stocks" --symbols-file "data/reference/sp500_symbols_template.csv" --copy-dir "data/processed/sp500_files"
```

## Professional Pipeline (Merge → Clean → Validate)
Run the full professional pipeline (local only):
- `python scripts/run_pipeline.py`

Optional parquet + metadata:
- `python scripts/run_pipeline.py --write-parquet`

Key outputs:
- `data/processed/full_stocks_merged.csv`
- `data/processed/full_stocks_cleaned.csv`
- `data/processed/full_stocks_cleaned.parquet` (optional)
- `data/processed/run_metadata.json`

Data contract validation:
- `src/day_trading_bot/data/contract.py`

## Training Table Contract
Locked output columns (exact order):
- `Date,Open,High,Low,Close,Volume,OpenInt,symbol,type,target_up_1d,target_up_5d,target_up_10d`

Targets are computed per `symbol` using future Close:
- `target_up_1d = Close(t+1) > Close(t)`
- `target_up_5d = Close(t+5) > Close(t)`
- `target_up_10d = Close(t+10) > Close(t)`

Training table builder:
- `python scripts/build_training_table.py`
