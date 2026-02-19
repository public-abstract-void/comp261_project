# Day Trading Bot (Academic Project)

Team: Allan, Christopher, Junah  
Phase: Week 1 - Stock Universe & Data Collection

## Current Scope
- Universe: S&P 500 (for later filtering)
- Data source: Kaggle Huge Stock Market Dataset
- Current mode: Prototype data loading and inspection (no live trading, no APIs)

## Project Layout
- `data/raw/Stocks/`: Raw Kaggle stock CSV files
- `data/processed/`: Cleaned/derived datasets (later)
- `notebooks/`: EDA notebooks (optional)
- `scripts/`: Runnable scripts for data tasks
- `src/day_trading_bot/`: Reusable project modules
- `tests/`: Unit tests (later)

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

### Run example (with your downloaded data)
- `python scripts/filter_sp500_files.py --stocks-dir "/Users/allanodora/Downloads/archive/Stocks" --symbols-file "data/reference/sp500_symbols_template.csv"`

### Optional: copy filtered files into a folder
- `python scripts/filter_sp500_files.py --stocks-dir "/Users/allanodora/Downloads/archive/Stocks" --symbols-file "data/reference/sp500_symbols_template.csv" --copy-dir "data/processed/sp500_files"`
