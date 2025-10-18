# Crypto Futures Backtester — Bachelor Thesis (UZH)

A modular Python framework for my **Bachelor Thesis at the University of Zurich**,
**“Evaluating Trading Strategies in Cryptocurrency Futures Markets: A Comparative Analysis of Bitcoin and Ethereum.”**

It provides a reproducible engine to **download data**, **simulate trading strategies**, and **analyze performance** on Binance perpetual futures (BTCUSDT, ETHUSDT).

---

## Repository structure

```
.
├── Backtester.py           # Core engine: handles portfolio, trades, PnL computation
├── Bot_Class.py            # Trading bot wrapper combining data, strategy, and execution
├── Config_File.py          # Configuration file (API keys, parameters, paths)
├── TradingStrats.py        # Strategy implementations (EMA cross, RSI, ATR breakout, etc.)
├── trade_models.py         # Trade, Position, and Order object definitions
├── trade_utils.py          # Helper functions for trade handling and performance metrics
├── data_utils.py           # Data loading, cleaning, and resampling (Binance/CSV/parquet)
├── bulk_data_downloader.py # Binance historical data downloader
├── Helper.py               # General-purpose utilities (math, date/time, etc.)
├── visualization.py        # Plotting equity curves, drawdowns, and metrics
├── run_grid.py             # Batch/parameter sweep runner for strategy combinations
├── debug_backtester.py     # Debugging/testing harness
├── test_Trading_Strats.py  # Unit tests for strategy logic
├── logger.py               # Logging setup for console and file outputs
├── __init__.py             # Marks package
└── .gitignore              # Ignore venvs, data, and temporary logs
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/aki2273/TradingProgram.git
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # (Windows: .venv\Scripts\activate)
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
   
---

## Configuration (API keys & parameters)

Edit **`Config_File.py`** before running anything:

```python
# Config_File.py
BINANCE_API_KEY = "your_api_key_here"
BINANCE_SECRET_KEY = "your_secret_key_here"

# General parameters
DATA_PATH = "data/"
RESULTS_PATH = "results/"
FEE_RATE = 0.0004
SLIPPAGE = 0.0001
LEVERAGE = 3
```

---

## How to use

### 1. Download historical data

Fetch OHLCV and funding data from Binance Futures:

```bash
python bulk_data_downloader.py --pair BTCUSDT --interval 1h --start 2019-01-01 --end 2025-06-30
```

Data will be saved in `data/` by default.

---

### 2. Run a single backtest

```bash
python Backtester.py --pair BTCUSDT --strategy ema_cross --start 2020-01-01 --end 2025-06-30
```

Example output:

```
Running EMA Cross strategy on BTCUSDT...
Final Equity: 1.284x
CAGR: 18.7%, Sharpe: 1.42, Max Drawdown: -22.5%
Results saved to: results/BTCUSDT_ema_cross_2025-06-30/
```

---

### 3. Strategy grid search

Explore multiple parameter combinations:

```bash
python run_grid.py --strategy rsi_band \
  --grid "rsi_len=[7,14,21]; rsi_low=[20,30]; rsi_high=[70,80]"
```

This will generate results for each combination and store metrics in a summary CSV.

---

### 4. Visualization

Generate plots for equity curves, drawdowns, and rolling Sharpe ratios:

```bash
python visualization.py --input results/BTCUSDT_ema_cross_2025-06-30/
```

---

## Included strategies (TradingStrats.py)

| Type                    | Strategy                 | Description                                  |
| ----------------------- | ------------------------ | -------------------------------------------- |
| **Momentum**            | Triple EMA, Golden Cross | Trend-following based on EMA crossovers      |
| **Mean Reversion**      | RSI Band, VWAP Deviation | Buys oversold, sells overbought conditions   |
| **Volatility Breakout** | ATR Channel              | Enters when volatility exceeds ATR threshold |

Each strategy implements:

```python
def generate_signals(df, params):
    # returns long/short/flat signals as +1/-1/0
```

---

## Testing & debugging

Run internal test scripts to verify logic:

```bash
python test_Trading_Strats.py
python debug_backtester.py
```

---

## Output files

Each backtest produces a timestamped results folder under `results/`:

```
results/
└── BTCUSDT_ema_cross_2025-06-30_12-00-00/
    ├── trades.csv
    ├── daily_equity.csv
    ├── metrics.json
    ├── config_snapshot.json
    └── equity_plot.png
```

---

## Example metrics

| Metric               | Description                           |
| -------------------- | ------------------------------------- |
| **CAGR**             | Compound annual growth rate           |
| **Volatility**       | Std. deviation of daily returns       |
| **Sharpe / Sortino** | Risk-adjusted performance ratios      |
| **Max Drawdown**     | Largest peak-to-trough equity decline |
| **Hit Rate**         | % of profitable trades                |
| **Profit Factor**    | Gross profit / gross loss             |
| **Exposure**         | % of time in market                   |

---

## Extending the framework

To add a new strategy:

1. Create a function in `TradingStrats.py`
2. Register it in the strategy dictionary inside `Bot_Class.py`
3. Pass your strategy name to `Backtester.py`

```python
# TradingStrats.py
def bollinger_band(df, params):
    ...
```

---

## Logging

All runtime logs (info, warnings, errors) are handled by `logger.py`.
They’re stored under `logs/{date}.log` for debugging or audit purposes.

---

## Disclaimer

This project is **for academic research only**.
It does **not** execute live trades and should not be used for financial decision-making.
All Binance trademarks and data belong to their respective owners.

---

## Citation

> **Akash Islam (2025)**
> *Evaluating Trading Strategies in Cryptocurrency Futures Markets: A Comparative Analysis of Bitcoin and Ethereum.*
> Bachelor Thesis, University of Zurich (UZH), Department of Banking & Finance.

---

## Author

**Akash Islam**
University of Zurich — Banking & Finance
📧 *akash.islam@uzh.ch*
💼 Research interests: Quantitative Finance, Algorithmic Trading, Machine Learning in Finance

---
