# run_full_grid.py  –  launches the *complete* parameter sweep
# -------------------------------------------------------------
from itertools import product
from joblib import Parallel, delayed
from Backtester import Backtester      

# ── 1) grab the SAME grids your Backtester demo uses ─────────
leverages = [5, 10]
time_frames = [
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h",
    "12h","1d","1w"
]
strategies = [
    "tripleEMA",
    "goldenCross",
    "ATRBreakoutstrategy",
    "pure_RSI_mean_reversion",
    "VWAP_mean_reversion"
]
tp_sl_choices = [
    "%",
    "x (ATR)",
    "x (Swing High/Low) level 1",
    "x (Swing High/Low) level 2",
    "x (Swing Close) level 1",
    "x (Swing Close) level 2"
]
symbols = ["BTCUSDT", "ETHUSDT"]

sl_mults = [1.0]
tp_mults = [2.0]

# ── 2) parameters that stay constant across the grid ─────────
base = dict(
    account_balance_start   = 50000,
    order_size              = 2,          
    start_date              = "01-01-2020",
    end_date                = "31-12-2024",
    max_open_trades         = 100,
    trade_all_symbols       = False,
    separate_accounts_per_coin = True,
    use_trailing_stop       = True,
    trailing_stop_callback  = 0.5,
    fee                     = 0.0004,
    slippage_range          = (0.0001, 0.0005),
    buffer_size             = 600,
    quick_test              = False,
    printing_on             = False,
    graph_folder            = "./results",
    auto_open_graph_images  = False,
    graph_before            = 5,
    graph_after             = 5,
    funding_mode            = "uniform",
    funding_bps             = 3.0,
    sl_first                = True,
    stop_on_bankruptcy      = True,
    make_trade_graphs       = False,
    signal_priority         = "entry",
)

# ── 3) single-run helper ─────────────────────────────────────
def run_one(sym, tf, lev, strat, tpsl, slm, tpm):
    kwargs = base | {
        "symbols"       : [sym],
        "time_interval" : tf,
        "leverage"      : lev,
        "strategy"      : strat,
        "tp_sl_choice"  : tpsl,
        "sl_mult"       : slm,
        "tp_mult"       : tpm,
    }
    Backtester(**kwargs).run_backtester()

# ── 4) farm every combo out to separate *processes* ──────────
if __name__ == "__main__":                 # mandatory on Windows
    grid = product(
        symbols, time_frames, leverages,
        strategies, tp_sl_choices, sl_mults, tp_mults
    )
    Parallel(n_jobs=-1, backend="loky")(
        delayed(run_one)(*g) for g in grid
    )
