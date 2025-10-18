import os # Used for operating system dependent functionalities like file path manipulation.
from datetime import datetime, timezone # For working with dates and times.
from concurrent.futures import ThreadPoolExecutor # For running operations in parallel using threads.
from joblib import load # For efficiently loading large NumPy arrays or Python objects.
import threading
import matplotlib # The main plotting library.
matplotlib.use('Agg') # Sets the backend of matplotlib to 'Agg', which is a non-interactive backend for writing to files.
import matplotlib.pyplot as plt # Provides a MATLAB-like plotting framework.
import numpy as np # Fundamental package for numerical computation in Python.
import pandas as pd # Powerful data analysis and manipulation library.

from binance.um_futures import UMFutures # Binance Futures API client.

from Bot_Class import Bot # Imports the Bot class, which encapsulates trading strategy logic.
try:
    from Config_File import API_KEY, API_SECRET
except (ModuleNotFoundError, ImportError):
    API_KEY    = os.getenv("BINANCE_KEY")
    API_SECRET = os.getenv("BINANCE_SECRET")
    if not (API_KEY and API_SECRET):
        raise RuntimeError(
            "Binance API credentials not found. "
            "Either create Config_File.py with API_KEY / API_SECRET variables "
            "or set BINANCE_KEY and BINANCE_SECRET environment variables."
            )
from data_utils import parse_time_interval
from trade_utils import check_take_profit, check_stop_loss, open_trade, close_trade_on_signal
from trade_models import Trade, TradeStats
from logger import get_logger
from visualization import (           # ← NEW
    get_candles_for_graphing,
    get_indicators_for_graphing,
    generate_trade_graphs as plot_trade_graphs,
)
log = get_logger()

# ──────────────────────────────────────────────────────────────────────────────
# Backtester CLASS
# ──────────────────────────────────────────────────────────────────────────────
class Backtester:
    """Run a candle‑by‑candle simulation over historical OHLCV data."""
    # This is useful for strategies that have specific conditions for closing trades outside of SL/TP.
    STRATEGIES_USING_MINUS_99_FOR_EXPLICIT_EXIT: list[str] = [ "VWAP_mean_reversion"]
    # ---------- static warm‑up table -----------------------------------------
    @staticmethod
    def _strategy_warmups():
        """
        Provides a static mapping of strategy names to their required warm-up periods in candles.
        This ensures that indicators have enough data to compute correctly before trading starts.
        """
        return {
            'tripleEMA': 50,                       
            'goldenCross': 100,
            'ATRBreakoutstrategy': 20,
            'pure_RSI_mean_reversion': 14,     
            'VWAP_mean_reversion': 20
        }

    # ---------- constructor ---------------------------------------------------
    def __init__(
        self,
        account_balance_start: float,
        leverage: float,
        order_size: float,
        start_date: str,
        end_date: str,
        time_interval: str,
        max_open_trades: int,
        trade_all_symbols: bool,
        separate_accounts_per_coin: bool,
        symbols: list,
        use_trailing_stop: bool,
        trailing_stop_callback: float,
        strategy: str,
        tp_sl_choice: str,
        sl_mult: float,
        tp_mult: float,
        fee: float,
        slippage_range: tuple[float, float] = (0.0001, 0.0005),
        buffer_size: int = 2500,
        quick_test: bool = False,
        printing_on: bool = False,
        graph_folder: str = "./results",
        auto_open_graph_images: bool = False,
        graph_before: int = 5,
        graph_after: int = 5,
        funding_mode: str = "uniform",
        funding_bps: float = 0.03,
        sl_first: bool = True,
        stop_on_bankruptcy: bool = True,
        make_trade_graphs: bool = False,
        signal_priority: str = "entry"

    ):
        """
        Initializes the Backtester with user-defined parameters and sets up necessary
        infrastructure for the backtesting process.
        """

        # -------- store user parameters
        # These lines store the configuration parameters passed by the user as instance attributes.
        self.account_balance_start = account_balance_start
        self.leverage = leverage
        self.order_size_percent = order_size / 100.0 # Converts order size from percentage to a decimal.
        self.start_date, self.end_date = start_date, end_date
        self.time_interval_str = time_interval  # e.g. "15m"
        self.max_open_trades = max_open_trades
        self.trade_all_symbols = trade_all_symbols
        self.separate_accounts_per_coin = separate_accounts_per_coin
        self.symbols = symbols
        self.use_trailing_stop = use_trailing_stop
        self.slippage_range = slippage_range
        self.funding_mode = funding_mode.lower() # Converts funding mode to lowercase for consistent checks.
        self.funding_max = funding_bps / 10000 # Converts funding basis points to a decimal rate.
        # Calculates the funding interval in terms of candles based on an 8-hour funding cycle.
        self.funding_interval = int((8*60)/parse_time_interval(self.time_interval_str))
        self.funding_interval = max(1, self.funding_interval) # Ensures funding interval is at least 1 candle.
        self.strategy = strategy
        self.tp_sl_choice, self.sl_mult, self.tp_mult = tp_sl_choice, sl_mult, tp_mult
        self.fee = fee
        self.buffer_size = buffer_size # User-defined additional warm-up buffer.
        self.quick_test = quick_test # If true, limits data length for faster testing.
        self.printing_on = printing_on # Controls whether to print trade actions during simulation.
        self.graph_folder = graph_folder
        self.auto_open_graph_images = auto_open_graph_images # Controls if graph images are opened automatically.
        self.graph_before, self.graph_after = graph_before, graph_after # Candles to show before/after trade in graphs.
        self.sl_first = sl_first
        self.trade_counter = 0 # Initializes a counter for unique trade IDs.
        self.stop_on_bankruptcy = stop_on_bankruptcy
        self.make_trade_graphs = make_trade_graphs
        self.signal_priority = signal_priority

        # -------- infra & containers
        # Initializes helper objects and data structures for the backtest.
        self.client = UMFutures(key=API_KEY, secret=API_SECRET) # Binance API client.
        self.time_interval = parse_time_interval(self.time_interval_str)  # Converts time interval string to minutes.
        self.bots: list[Bot] = [] # List to hold Bot instances.
        # Lists for tracking account balances, profit graphs, and daily returns across accounts/symbols.
        self.account_balances, self.profit_graphs, self.daily_returns = [], [], []
        self.daily_log_ret = [] # Stores daily logarithmic returns.
        self.daily_dates = [] # Stores dates corresponding to daily returns.
        self.daily_records = [] # Stores daily snapshots of indicator values (if implemented).
        self.buy_hold_equity = [] # Tracks equity for a buy-and-hold benchmark.
        self.buy_hold_log_ret = [] # Tracks log returns for the buy-and-hold benchmark.
        self.completed_trades: list[Trade] = []    
        self.trade_stats = TradeStats() 

        # -------- symbol handling
        # Determines the list of symbols to trade based on user settings.
        if self.trade_all_symbols:
            self.symbols = self._get_all_usdt_symbols() # Fetches all USDT perpetual futures symbols.
        self.coin_info = self._collect_coin_info() # Collects exchange information (precision, tick size) for symbols.

        # OHLCV lists
        # Initializes lists to store OHLCV data for each symbol.
        self.dates_interval_data, self.opens_interval_data, self.highs_interval_data = [], [], []
        self.lows_interval_data, self.closes_interval_data, self.volumes_interval_data = [], [], []

    # =========================================================================
    # EXCHANGE INFO HELPERS
    # =========================================================================
    def _get_all_usdt_symbols(self):
        """Fetches all trading symbols paired with USDT from Binance Futures."""
        tickers = self.client.ticker_price()
        # Filters for symbols containing 'USDT' and without '_' (to exclude quarterly contracts, etc.).
        return [t['symbol'] for t in tickers if 'USDT' in t['symbol'] and '_' not in t['symbol']]

    def _collect_coin_info(self):
        """
        Collects exchange information for all symbols, such as price and quantity precision,
        and tick size. This is important for order placement.
        """
        out = {}
        exchange_info = self.client.exchange_info()
        for s in exchange_info['symbols']:
            out[s['pair']] = {
                'pricePrecision': s['pricePrecision'],
                'quantityPrecision': s['quantityPrecision'],
                'tickSize': float(s['filters'][0]['tickSize']), # Assumes the first filter is PRICE_FILTER for tickSize.
            }
        return out

    # =========================================================================
    # PUBLIC PIPELINE
    # =========================================================================
    def run_backtester(self):
        """
        This is the main public method that orchestrates the entire backtesting process
        by calling various private helper methods in sequence.
        """
        self._print_settings() # Prints the current backtest configuration.
        self._load_data_for_backtest() # Loads historical price data.
        self._initialize_bots_and_balances() # Initializes Bot instances and account balances.
        self._simulate_trading_loop() # Runs the core candle-by-candle trading simulation.
        self._finalize_results() # Calculates performance metrics and saves results.

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------
    def _load_data_for_backtest(self):
        """
        Loads historical OHLCV data for each symbol specified in `self.symbols`.
        Applies quick-test slicing if enabled.
        """
        log.info("Loading price data …")
        for sym in self.symbols:
            # Loads data for each symbol using the _load_bulk_data helper.
            d = self._load_bulk_data(sym, self.time_interval_str, self.start_date, self.end_date)

            # Applies quick_test slicing: limits data to buffer_size if quick_test is True.
            limit = min(len(d['Close']), self.buffer_size) if self.quick_test else len(d['Close'])
            # Appends the (potentially sliced) data to the respective instance lists.
            self.dates_interval_data.append(d['Date'][:limit])
            self.opens_interval_data.append(d['Open'][:limit])
            self.highs_interval_data.append(d['High'][:limit])
            self.lows_interval_data.append(d['Low'][:limit])
            self.closes_interval_data.append(d['Close'][:limit])
            self.volumes_interval_data.append(d['Volume'][:limit])

        # Sets the main_date series (from the first symbol) as a reference for dates.
        self.main_date = self.dates_interval_data[0] if self.dates_interval_data else []

    def _load_bulk_data(self, symbol, interval, start_date, end_date):
        """
        Loads pre-downloaded historical data from a .joblib file for a given symbol and interval.
        Filters the data to the specified start and end dates.
        """
        # --- sanity check on date range ---------------------------------------
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date {self.start_date} must be earlier than end_date {self.end_date}"
            )
        from joblib import load # Local import, though already imported globally.
        fn = f"./bulk_data/{symbol}_{interval}.joblib" # Constructs the filename.
        data = load(fn) # Loads data from the joblib file.
        # Converts date strings to datetime objects for comparison.
        s_dt = datetime.strptime(start_date, "%d-%m-%Y")
        e_dt = datetime.strptime(end_date, "%d-%m-%Y")
        # Creates a mask to filter data within the specified date range.
        mask = [(s_dt <= dt <= e_dt) for dt in data['Date']]
        # Applies the mask to all data series (Open, High, Low, Close, Volume, Date).
        return {k: [v[i] for i in range(len(v)) if mask[i]] for k, v in data.items()}

    # -------------------------------------------------------------------------
    # INITIALISE BOTS + BALANCES
    # -------------------------------------------------------------------------
    def _initialize_bots_and_balances(self):
        """
        Create one Bot per symbol and set up the book-keeping containers.
        """
        self.daily_equity = []                 # reset (used for daily snapshots)

        # ── 1) build the bots ──────────────────────────────────────────────
        for idx, sym in enumerate(self.symbols):
            meta = self.coin_info.get(
                sym,
                {"pricePrecision": 2, "quantityPrecision": 2, "tickSize": 0.01},
            )

            self.bots.append(
                Bot(
                    sym,
                    self.opens_interval_data[idx],
                    self.closes_interval_data[idx],
                    self.highs_interval_data[idx],
                    self.lows_interval_data[idx],
                    self.volumes_interval_data[idx],
                    self.dates_interval_data[idx],
                    meta["quantityPrecision"],
                    meta["pricePrecision"],
                    idx,
                    meta["tickSize"],
                    self.strategy,
                    self.tp_sl_choice,
                    self.sl_mult,
                    self.tp_mult,
                    backtesting=1,
                    signal_priority=self.signal_priority,
                )
            )

            # ── 2a) per-coin cash accounts ────────────────────────────────
            if self.separate_accounts_per_coin:
                self.account_balances.append(self.account_balance_start)

                # equity / P&L tracking (one list per account)
                self.profit_graphs.append([self.account_balance_start])
                self.daily_returns.append([self.account_balance_start])
                self.daily_log_ret.append([])
                self.daily_dates.append([])

                # daily_equity & buy-and-hold start empty; they’ll be filled
                self.daily_equity.append([])
                self.buy_hold_equity.append([self.account_balance_start])
                self.buy_hold_log_ret.append([])

        # ── 2b) single shared account ─────────────────────────────────────
        if not self.separate_accounts_per_coin:
            self.account_balances   = [self.account_balance_start]
            self.profit_graphs      = [[self.account_balance_start]]
            self.daily_returns      = [[self.account_balance_start]]
            self.daily_log_ret      = [[]]
            self.daily_dates        = [[]]
            self.daily_equity       = [[]]                       # starts empty
            self.buy_hold_equity    = [[self.account_balance_start]]
            self.buy_hold_log_ret   = [[]]

        # ── 3) make sure daily_equity has the right shape ─────────────────
        #     (kept empty on purpose – snapshots are added later)
        self.daily_equity = [[] for _ in range(len(self.account_balances))]

        # ── 4) one balance lock per cash account (or one global lock) ─────
        if self.separate_accounts_per_coin:
            self._bal_locks = [threading.Lock() for _ in self.account_balances]
        else:
            self._bal_lock = threading.Lock()


    # -------------------------------------------------------------------------
    # MAIN SIMULATION LOOP
    # -------------------------------------------------------------------------
    def _simulate_trading_loop(self):
            """
            Executes the core candle-by-candle trading simulation.
            It handles warm-up periods, bot decision making, trade execution (opening/closing),
            SL/TP checks, funding fees, and daily equity snapshots.
            """
            active_trades = [] # List to keep track of currently open trades.
            # trade_counter is a class attribute, initialized in __init__.
            self.trade_counter = 0

            # adaptive warm‑up calculation (in candles).
            if not self.bots: # Ensures bots are initialized.
                log.error("Error: Bots not initialized before simulation loop.")
                return

            # Determines the maximum strategy-specific warm-up period required by any bot.
            strategy_specific_warmup = 0
            if self.bots:
                strategy_specific_warmup = max(self._strategy_warmups().get(b.strategy, 50) for b in self.bots)

            # Total warm-up is the max of strategy warm-up and user-defined buffer.
            warm = max(strategy_specific_warmup, self.buffer_size)

            # Ensures warm-up does not exceed available data length, leaving a few candles for trading.
            # max_possible_warmup ensures there are at least 11 candles after warm-up (10 for trading, 1 for next open).
            max_possible_warmup = len(self.closes_interval_data[0]) - 11 if self.closes_interval_data and self.closes_interval_data[0] else -11
            if max_possible_warmup < 0: max_possible_warmup = 0 # Prevents negative warm-up.

            warm = min(warm, max_possible_warmup) # Caps warm-up by available data.
            warm = max(0, warm) # Ensures warm-up is not negative after capping.

            # Checks if there's enough data for trading after the warm-up period.
            if not self.closes_interval_data or not self.closes_interval_data[0] or len(self.closes_interval_data[0]) <= warm + 10:
                log.info(f"⚠ Insufficient data for trading after warm-up. Warm-up candles: {warm}, Total candles: {len(self.closes_interval_data[0]) if self.closes_interval_data and self.closes_interval_data[0] else 0}.")
                log.info("Check buffer_size, strategy warm-ups, or data length.")
                return

            # Initializes `last_snapshot_day` to the date of the last warm-up candle.
            if not self.main_date: # Ensures main_date (reference date series) is available.
                log.error("Error: self.main_date is empty. Cannot proceed.")
                return

            if warm == 0 : # If no warm-up, use the date of the first candle.
                last_snapshot_day = self.main_date[0].date()
            elif warm > 0 and warm <= len(self.main_date):
                # main_date is 0-indexed; main_date[warm-1] is the date of the last warm-up candle.
                last_snapshot_day = self.main_date[warm - 1].date()
            else: # Fallback if warm-up index is out of bounds.
                log.error(f"Error: Warm-up index {warm-1} is problematic for main_date (len: {len(self.main_date)}).")
                if self.main_date:
                    last_snapshot_day = self.main_date[0].date()
                    log.info("Defaulting last_snapshot_day to the first available date.")
                else:
                    log.warning("Cannot determine last_snapshot_day as main_date is empty.")
                    return

            # ------------------------------------------------ loop
            # Iterates from the first candle *after* warm-up up to the second-to-last candle.
            # The loop stops at len-2 because trade opening uses open_px[i+1].
            for i in range(warm, len(self.closes_interval_data[0]) - 1):

                # --- Bot Update (set current candle index for decisions) ---
                # Sets the `current_index` for each bot to `i`, so bots know which candle's data to use.
                for b_idx, b_obj in enumerate(self.bots):
                    b_obj.current_index = i

                # --- Gather ALL Bot Decisions (Entry, Hold, or Strategy Exit) ---
                # Collects trading decisions (buy, sell, hold, or explicit exit) from all bots.
                all_bot_decisions = self._gather_new_trades() # Returns list of (bot_k_index, decision, sl, tp)

                # --- Process Strategy-Based Exits FIRST ---
                # Handles exits explicitly signaled by strategy logic before checking SL/TP or new entries.
                trades_closed_by_strategy = [] # Stores trades closed by strategy in this iteration.

                # Iterates over a copy of active_trades because the list might be modified.
                current_active_trades_iter = list(active_trades)

                for t_idx, t in enumerate(current_active_trades_iter):
                    if t.trade_status != 1: # Skips if trade is not currently active (already closed).
                        self.bots[t.index].active_trade_direction = None
                        continue

                    bot_for_trade = self.bots[t.index] # Gets the bot associated with the trade.

                    decision_for_this_bot = None # Initializes to indicate no explicit decision found yet.
                    for bd_k_idx, bd_decision, _, _ in all_bot_decisions:
                        if bd_k_idx == t.index: # Matches the bot's decision to the current trade's bot.
                            decision_for_this_bot = bd_decision
                            break

                    # If decision is -99 and strategy is in the predefined list, close the trade.
                    if decision_for_this_bot == -99 and bot_for_trade.strategy in Backtester.STRATEGIES_USING_MINUS_99_FOR_EXPLICIT_EXIT:

                        acc_idx = t.index if self.separate_accounts_per_coin else 0
                        # Exits at the current candle's close price.
                        exit_price_for_signal = self.closes_interval_data[t.index][i]

                        original_balance = self.account_balances[acc_idx]
                        # Closes the trade
                        closed_trade_obj, new_balance = close_trade_on_signal(
                            t, original_balance, exit_price_for_signal, self.fee, self.printing_on
                        )
                        self.account_balances[acc_idx] = new_balance

                        # Marks the trade as closed for later removal.
                        if closed_trade_obj.trade_status != 1:
                            trades_closed_by_strategy.append(closed_trade_obj)
                            bot_for_trade.active_trade_direction = None

                    elif decision_for_this_bot in (0, 1) and decision_for_this_bot != t.trade_direction:
                        acc_idx = t.index if self.separate_accounts_per_coin else 0
                        exit_price_for_signal = self.closes_interval_data[t.index][i]

                        original_balance = self.account_balances[acc_idx]
                        closed_trade_obj, new_balance = close_trade_on_signal(
                            t, original_balance, exit_price_for_signal, self.fee, self.printing_on
                        )
                        self.account_balances[acc_idx] = new_balance

                        if closed_trade_obj.trade_status != 1:
                            trades_closed_by_strategy.append(closed_trade_obj)
                            bot_for_trade.active_trade_direction = None

                # Removes trades that were closed by strategy signals from the main `active_trades` list.
                if trades_closed_by_strategy:
                    self.completed_trades.extend(trades_closed_by_strategy)
                    # NEW – update win / loss counters
                    for t in trades_closed_by_strategy:
                        self.trade_stats.total_number_of_trades += 1
                        if t.trade_info.trade_success:
                            self.trade_stats.wins   += 1
                        else:
                            self.trade_stats.losses += 1
                    active_trades = [
                        trade for trade in active_trades
                        # Ensures removal by checking a unique combination of order_id and symbol.
                        if all(closed_trade.order_id != trade.order_id or closed_trade.symbol != trade.symbol for closed_trade in trades_closed_by_strategy)
                    ]


                # --- Open New Trades ---
                # Identifies bots that already have an active trade.
                bots_with_active_trades = {t.index for t in active_trades}
                new_entry_signals_queue = []
                # Gathers new entry signals (decision 0 for short, 1 for long) from bots that don't have an active trade.
                for k_idx, decision, sl, tp in all_bot_decisions:
                    if decision in [0, 1] and k_idx not in bots_with_active_trades:
                        new_entry_signals_queue.append((k_idx, decision, sl, tp))

                # Opens new trades based on the gathered signals, respecting max_open_trades.
                # self.trade_counter is used internally by _open_new_trades.
                self._open_new_trades(new_entry_signals_queue, active_trades, i)


                # --- Funding Fee Calculation ---
                # Applies funding fees at the specified funding interval.
                if self.funding_mode != "none" and i != 0 and (i % self.funding_interval == 0):
                    if self.funding_mode == "constant":
                        funding_rate = self.funding_max # Uses a fixed max rate.
                    else:  # "uniform"
                        funding_rate = np.random.uniform(0, self.funding_max) # Uses a random rate up to max.

                    for t_fund in active_trades: # Iterates over trades still active.
                        notional = abs(t_fund.position_size * t_fund.entry_price) # Calculates notional value.
                        debit    = notional * funding_rate # Calculates funding fee.
                        acc_idx_funding = t_fund.index if self.separate_accounts_per_coin else 0
                        if 0 <= acc_idx_funding < len(self.account_balances):
                            self.account_balances[acc_idx_funding] -= debit # Deducts fee from account balance.
                        else:
                            log.warning(f"Warning: Invalid account index {acc_idx_funding} for funding on trade {t_fund.symbol}.")


                # --- Update Active Trades (SL/TP checks using H/L of candle 'i') ---
                # Checks stop-loss and take-profit conditions for all active trades using the H/L of the current candle 'i'.
                # This function modifies `active_trades` in place by removing trades that hit SL/TP.
                self._update_active_trades(active_trades, i)

                # ── bankruptcy guard ──────────────────────────────────────────────
                if self.stop_on_bankruptcy:
                    if not self.separate_accounts_per_coin:
                        if self.account_balances[0] < 0:
                            log.error(
                                "❌  Account balance fell below zero (%.2f) at candle %d. "
                                "Back-test aborted.",
                                self.account_balances[0], i
                            )
                            break
                    else:
                        negatives = [
                            f"{sym}:{bal:.2f}"
                            for sym, bal in zip(self.symbols, self.account_balances)
                            if bal < 0
                        ]
                        if negatives:
                            log.error(
                                "❌  Negative balance(s) detected (%s) at candle %d. "
                                "Back-test aborted.",
                                ", ".join(negatives), i
                            )
                            break
                        
                # --- Daily Equity Snapshot (after all operations for candle 'i') ---
                # Captures daily equity and performance metrics if the day has changed.
                current_day = self.main_date[i].date()
                if current_day != last_snapshot_day:
                    # Calculates equity (cash + UPNL) at the current candle.
                    equity_at_snapshot = self._equity_snapshot(active_trades, i)
                    # Records daily returns and other performance data.
                    self._capture_daily_returns(i, equity_at_snapshot)

                    # Appends the current account balance(s) to profit_graphs for later plotting.
                    if not self.separate_accounts_per_coin:
                        if self.account_balances and self.profit_graphs:
                            if len(self.profit_graphs) > 0 and self.profit_graphs[0] is not None:
                                self.profit_graphs[0].append(self.account_balances[0])
                            elif len(self.profit_graphs) > 0 :
                                self.profit_graphs[0] = [self.account_balances[0]]
                    else:
                        for acc_idx_pg, acc_bal_pg in enumerate(self.account_balances):
                            if acc_idx_pg < len(self.profit_graphs) and self.profit_graphs[acc_idx_pg] is not None:
                                self.profit_graphs[acc_idx_pg].append(acc_bal_pg)
                            elif acc_idx_pg < len(self.profit_graphs):
                                self.profit_graphs[acc_idx_pg] = [acc_bal_pg]
                    last_snapshot_day = current_day # Updates the last snapshot day.

            if active_trades:
                log.info("Auto-closing %d still-open trade(s) at end-of-back-test…",
                         len(active_trades))

                final_close_idx = len(self.closes_interval_data[0]) - 1  # very last candle

                for t in list(active_trades):          # iterate over a *copy*
                    acc_idx  = t.index if self.separate_accounts_per_coin else 0
                    exit_px  = self.closes_interval_data[t.index][final_close_idx]

                    # keep fee / PnL logic identical to normal signal exits
                    t, new_bal = close_trade_on_signal(
                        t,
                        self.account_balances[acc_idx],
                        exit_px,
                        self.fee,
                        self.printing_on,
                    )
                    t.exit_reason = "End-of-Backtest"
                    t.trade_info.exit_time = str(self.main_date[final_close_idx])

                    self.account_balances[acc_idx] = new_bal
                    self.completed_trades.append(t)
                    active_trades.remove(t)

                    # stats counters
                    self.trade_stats.total_number_of_trades += 1
                    if t.trade_info.trade_success:
                        self.trade_stats.wins   += 1
                    else:
                        self.trade_stats.losses += 1

                    # make the bot eligible for new trades on a subsequent run
                    self.bots[t.index].active_trade_direction = None            
            # Final capture for the last day's activity after the loop finishes.
            if self.closes_interval_data and self.closes_interval_data[0] and len(self.closes_interval_data[0]) > 1:
                # `last_candle_idx` is the index of the last candle for which decisions/updates were made (loop went up to len-2).
                last_candle_idx = len(self.closes_interval_data[0]) - 2
                if last_candle_idx >= warm : # Ensures the loop actually ran and processed some data.
                    # Performs a final equity calculation and return capture for the state after the last processed candle.
                    final_equity_state = self._equity_snapshot(active_trades, last_candle_idx)
                    self._capture_daily_returns(last_candle_idx, final_equity_state)

                    # Appends the final account balances to profit_graphs to ensure the last state is recorded for plotting.
                    if not self.separate_accounts_per_coin:
                        if self.account_balances and self.profit_graphs and len(self.profit_graphs) > 0 and self.profit_graphs[0] is not None:
                            self.profit_graphs[0].append(self.account_balances[0])
                    else:
                        for acc_idx_pg, acc_bal_pg in enumerate(self.account_balances):
                            if acc_idx_pg < len(self.profit_graphs) and self.profit_graphs[acc_idx_pg] is not None:
                                self.profit_graphs[acc_idx_pg].append(acc_bal_pg)

    # -------------------------------------------------------------------------
    # BOT / TRADE HELPERS
    # -------------------------------------------------------------------------
    def _gather_new_trades(self):
        """
        Collects trading decisions (entry, hold, or strategy-specific exit like -99)
        from all Bot instances for the current simulation candle.
        It uses a ThreadPoolExecutor to evaluate bot decisions in parallel for efficiency.
        """
        def evaluate(item):
            """Helper function to be run in parallel for each bot."""
            bot_index, bot_instance = item # `bot_index` is the bot's `k` (original index in self.bots).
            # `bot_instance.current_index` (the simulation candle `i`) must be set before calling Make_decision.
            # This is handled in the main loop (`_simulate_trading_loop`).
            decision, sl_val, tp_val = bot_instance.make_decision()
            return (bot_index, decision, sl_val, tp_val)

        # The bot's `current_index` (simulation candle `i`) is set in the main simulation loop.
        with ThreadPoolExecutor() as pool:
            # Creates a list of (bot_original_index, bot_object) tuples for parallel processing.
            bot_items = []
            for k, b_obj in enumerate(self.bots): # `k` is the bot's original index.
                bot_items.append((k, b_obj))

            results = pool.map(evaluate, bot_items)

        # `results` is a list of (bot_original_index, decision, sl_val, tp_val) tuples.
        # Decisions of -99 are kept as they might signify a strategy-specific exit.
        return list(results)


    def _open_new_trades(self, queue, active_trades, i):
        """
        Processes a queue of new trade signals (buy/sell decisions from bots).
        Opens new trades if the `max_open_trades` limit is not reached.
        Calculates order quantity, entry price considering slippage, and updates account balance.
        Creates a Trade object and adds it to `active_trades`.
        It uses `self.trade_counter` to assign unique order IDs.
        """
        while queue and len(active_trades) < self.max_open_trades:
            # `k` is the bot's original index, `direction` is 0 for short, 1 for long.
            k, direction, sl_val, tp_val = queue.pop(0)
            # Trades are opened at the open price of the *next* candle (i+1).
            open_px = self.opens_interval_data[k][i + 1]

            # Determines the account balance to use for this trade.
            bal = self.account_balances[k] if self.separate_accounts_per_coin else self.account_balances[0]
            # Calculates the notional value of the order based on balance, leverage, and order size percentage.
            notional = bal * self.leverage * self.order_size_percent
            # Calls open_trade to get quantity, actual entry price (with slippage), and new balance.
            qty, entry_px, bal_new, slip_val = open_trade(
                self.bots[k].symbol, notional, bal, open_px,
                self.fee, self.bots[k].OP, self.bots[k].CP, # OP/CP are quantity/price precision.
                direction, self.slippage_range
            )
            if qty == 0: # If quantity is zero (e.g., insufficient balance or notional too small), skip.
                continue

            # Updates the account balance.
            if self.separate_accounts_per_coin:
                self.account_balances[k] = bal_new
            else:
                self.account_balances[0] = bal_new

            # Calculates Take Profit and Stop Loss prices.
            tp_px = entry_px + tp_val if direction == 1 else entry_px - tp_val
            sl_px = entry_px - sl_val if direction == 1 else entry_px + sl_val
            # Rounds TP/SL prices to the symbol's price precision.
            cp = self.bots[k].CP # Price precision.
            tp_px = round(tp_px, cp) if cp else round(tp_px)
            sl_px = round(sl_px, cp) if cp else round(sl_px)

            # --- MODIFICATION HERE for order_id ---
            current_trade_id = self.trade_counter # Assigns a unique ID to the trade.
            # --------------------------------------
            # Creates a new Trade object.
            t = Trade(
                index=k, # Bot's original index.
                position_size=qty,
                TP_val=tp_px, SL_val=sl_px,
                trade_direction=direction,
                order_id=current_trade_id, # Uses the unique trade ID.
                symbol=self.bots[k].symbol
            )
            t.entry_price = entry_px # Sets entry price.
            entry_fee = entry_px * qty * self.fee
            t.trade_info.entry_fee = entry_fee
            ind_dict = getattr(self.bots[k], "indicators", None)
            if ind_dict:
                atr_values = ind_dict.get("ATR", {}).get("values", [])
                if i < len(atr_values):
                    t.trade_info.volatility_at_entry = atr_values[i]
            t.trade_info.entry_price = entry_px # Also stores in trade_info.
            t.trade_info.slippage    = slip_val
            # `self.bots[k].current_index` is `i` (the candle index on which decision was made).
            t.trade_info.trade_start_index = self.bots[k].current_index
            t.trade_info.start_time = str(self.dates_interval_data[k][i + 1]) # Time of the next candle's open.
            t.trade_status = 1 # Sets trade status to "in progress".
            active_trades.append(t) # Adds the new trade to the list of active trades.
            self.bots[k].active_trade_direction = direction
            self.trade_counter += 1 # Increments the global trade counter.
            # -------------------------------------------------------------

    def _update_active_trades(self, active_trades, i):
        """
        Walk through every active trade and evaluate Stop-Loss / Take-Profit hits
        on candle index `i`.

        Priority rule
        ------------
        • If self.sl_first == True  (default):  Stop-Loss is evaluated *before* Take-Profit.
        A candle that pierces both levels will be closed as a loss.

        • If self.sl_first == False: Take-Profit is evaluated first; the same candle
        closes as a win if both thresholds are breached.

        Trades closed inside this routine are appended to self.completed_trades and
        corresponding win / loss counters in self.trade_stats are updated.
        """
        def step(t: Trade):
            """Process SL/TP for a single trade under a balance lock."""
            if t.same_candle:
                t.same_candle = False
                return t
            idx = t.index
            hi, lo = self.highs_interval_data[idx][i], self.lows_interval_data[idx][i]
            atr_val = self.bots[idx].indicators["ATR"]["values"][i]
            ref_price = hi if t.trade_direction == 1 else lo
            dyn_cb = 0.005
            if atr_val and atr_val > 0 and ref_price:
                dyn_cb = 0.5 * atr_val / ref_price
            if hi > t.Highest_val:
                t.Highest_val = hi
            if lo < t.Lowest_val:
                t.Lowest_val = lo

            # -------- critical section on the relevant account balance ----------
            lock = self._bal_locks[idx] if self.separate_accounts_per_coin else self._bal_lock
            with lock:
                bal = (
                    self.account_balances[idx]
                    if self.separate_accounts_per_coin
                    else self.account_balances[0]
                )

                # ----- apply SL / TP in the configured order -------------------
                if self.sl_first:
                    if t.trade_status == 1:
                        t, bal = check_stop_loss(t, bal, hi, lo, self.fee)
                    if t.trade_status == 1:
                        t, bal = check_take_profit(
                            t, bal, hi, lo, self.fee,
                            self.use_trailing_stop, dyn_cb, self.bots[idx].CP
                        )
                else:  # TP priority
                    if t.trade_status == 1:
                        t, bal = check_take_profit(
                            t, bal, hi, lo, self.fee,
                            self.use_trailing_stop, dyn_cb, self.bots[idx].CP
                        )
                    if t.trade_status == 1:
                        t, bal = check_stop_loss(t, bal, hi, lo, self.fee)

                # ---------- write the updated balance back ---------------------
                if self.separate_accounts_per_coin:
                    self.account_balances[idx] = bal
                else:
                    self.account_balances[0] = bal
            # -------- end critical section -------------------------------------

            if t.trade_status != 1:
                t.trail_activated = False
                t.trade_info.exit_time = str(self.main_date[i])
                self.bots[idx].active_trade_direction = None
            return t

        # ---- parallel evaluation of all trades --------------------------------
        with ThreadPoolExecutor() as pool:
            updated = list(pool.map(step, active_trades))

        # ---- collect closed trades & update stats ------------------------------
        closed = [t for t in updated if t.trade_status != 1]
        if closed:
            self.completed_trades.extend(closed)
            for t in closed:
                self.trade_stats.total_number_of_trades += 1
                if t.trade_info.trade_success:
                    self.trade_stats.wins   += 1
                else:
                    self.trade_stats.losses += 1

        # ---- keep only still-open trades in the original list ------------------
        active_trades[:] = [t for t in updated if t.trade_status == 1]


    # -------------------------------------------------------------------------
    # BOOKKEEPING & RESULTS
    # -------------------------------------------------------------------------
    def _equity_snapshot(self, active_trades: list[Trade], i: int) -> list[float]:
        """
        Calculates the total equity for every account at candle index `i`.
        Equity is defined as cash balance + unrealized PnL (UPNL) of active trades.
        This works for both single-account and multi-account (separate_accounts_per_coin) modes.
        """
        # Starts with a copy of current cash balances.
        equity = self.account_balances.copy() if self.separate_accounts_per_coin else [self.account_balances[0]]
        num_accounts = len(equity)

        for t in active_trades:
            acc_idx = t.index if self.separate_accounts_per_coin else 0 # Determines account index for the trade.

            if 0 <= acc_idx < num_accounts: # Ensures account index is valid.
                side = 1 if t.trade_direction == 1 else -1 # 1 for long, -1 for short.
                # Ensures trade's symbol index and candle index are valid for price data.
                if 0 <= t.index < len(self.closes_interval_data) and 0 <= i < len(self.closes_interval_data[t.index]):
                    last_price = self.closes_interval_data[t.index][i] # Current close price for UPNL calculation.
                    upnl = side * (last_price - t.entry_price) * t.position_size # Calculates UPNL.
                    equity[acc_idx] += upnl # Adds UPNL to the respective account's equity.
                else:
                    log.warning(f"Warning: Trade index {t.index} or candle index {i} out of bounds for price data in _equity_snapshot.")
            else:
                log.warning(f"Warning: Account index {acc_idx} out of bounds for equity list in _equity_snapshot.")
        return equity


    def _capture_daily_returns(self, current_candle_idx: int, equity_now: list[float]):
        """
        Records the daily equity for each account, calculates daily logarithmic returns,
        and updates the buy-and-hold benchmark equity. This is typically called once per
        simulated day, using the equity calculated at that day's snapshot.
        """
        if self.separate_accounts_per_coin: # Logic for separate accounts per coin.
            for j, bal in enumerate(equity_now): # `j` is the account index (and symbol index).
                if j < len(self.closes_interval_data): # Ensures index `j` is valid for price data.
                    self.daily_equity[j].append(bal) # Appends current equity.
                    self.daily_returns[j].append(bal) # `daily_returns` stores the equity path.

                    # Buy-and-Hold benchmark calculation for this symbol.
                    if len(self.buy_hold_equity[j]) == 0: # Initializes B&H equity if first data point.
                        self.buy_hold_equity[j].append(self.account_balance_start)
                    else:
                        # Checks if candle index is valid for this symbol's price data.
                        if 0 <= current_candle_idx < len(self.closes_interval_data[j]) and len(self.closes_interval_data[j]) > 0:
                            price_start = self.closes_interval_data[j][0] # Price at the start of the backtest for this symbol.
                            price_now   = self.closes_interval_data[j][current_candle_idx] # Current price for this symbol.
                            if price_start != 0: # Avoids division by zero.
                                # Calculates B&H equity based on price change from start.
                                bh_equity_val = self.buy_hold_equity[j][0] * (price_now / price_start)
                                self.buy_hold_equity[j].append(bh_equity_val)
                            else:
                                self.buy_hold_equity[j].append(self.buy_hold_equity[j][-1]) # Maintains last B&H value.
                        else: # Fallback if price data is unavailable.
                             self.buy_hold_equity[j].append(self.buy_hold_equity[j][-1])

                    # Calculates daily log return for the strategy.
                    if len(self.daily_returns[j]) >= 2: # Needs at least two data points.
                        # Ensures non-negative values for log calculation.
                        if self.daily_returns[j][-1] > 0 and self.daily_returns[j][-2] > 0:
                            r = np.log(self.daily_returns[j][-1] / self.daily_returns[j][-2])
                            self.daily_log_ret[j].append(r)
                        else:
                            self.daily_log_ret[j].append(0) # Appends 0 if log cannot be calculated.
                        self.daily_dates[j].append(self.main_date[current_candle_idx]) # Records the date.

                    # Calculates daily log return for the Buy-and-Hold benchmark.
                    if len(self.buy_hold_equity[j]) >= 2:
                        if self.buy_hold_equity[j][-1] > 0 and self.buy_hold_equity[j][-2] > 0:
                             r_bh = np.log(self.buy_hold_equity[j][-1] / self.buy_hold_equity[j][-2])
                             self.buy_hold_log_ret[j].append(r_bh)
                        else:
                             self.buy_hold_log_ret[j].append(0)
                else:
                     log.warning(f"Warning: Account index {j} out of bounds for price data in _capture_daily_returns (separate accounts).")

        else: # Logic for a single, shared account.
            if equity_now: # Ensures equity_now (list) is not empty.
                bal = equity_now[0] # Equity of the single account.
                self.daily_equity[0].append(bal)
                self.daily_returns[0].append(bal)

                # Calculates daily log return for the strategy.
                if len(self.daily_returns[0]) >= 2:
                    if self.daily_returns[0][-1] > 0 and self.daily_returns[0][-2] > 0:
                        r = np.log(self.daily_returns[0][-1] / self.daily_returns[0][-2])
                        self.daily_log_ret[0].append(r)
                    else:
                        self.daily_log_ret[0].append(0)
                    self.daily_dates[0].append(self.main_date[current_candle_idx])

                # Buy-and-Hold benchmark (against the first symbol in `self.symbols`).
                if len(self.closes_interval_data) > 0 and self.closes_interval_data[0] and \
                   0 <= current_candle_idx < len(self.closes_interval_data[0]):
                    if len(self.buy_hold_equity[0]) == 0:
                        self.buy_hold_equity[0].append(self.account_balance_start)
                    else:
                        price_start = self.closes_interval_data[0][0] # Price of the first symbol at backtest start.
                        price_now   = self.closes_interval_data[0][current_candle_idx] # Current price of the first symbol.
                        if price_start != 0:
                            bh_equity_val = self.buy_hold_equity[0][0] * (price_now / price_start)
                            self.buy_hold_equity[0].append(bh_equity_val)
                        else:
                            self.buy_hold_equity[0].append(self.buy_hold_equity[0][-1])

                    # Calculates daily log return for the Buy-and-Hold benchmark.
                    if len(self.buy_hold_equity[0]) >= 2:
                        if self.buy_hold_equity[0][-1] > 0 and self.buy_hold_equity[0][-2] > 0:
                            r_bh = np.log(self.buy_hold_equity[0][-1] / self.buy_hold_equity[0][-2])
                            self.buy_hold_log_ret[0].append(r_bh)
                        else:
                            self.buy_hold_log_ret[0].append(0)
                elif len(self.buy_hold_equity[0]) > 0: # Fallback if price data is missing for B&H.
                    self.buy_hold_equity[0].append(self.buy_hold_equity[0][-1])
                    if len(self.buy_hold_equity[0]) >= 2: self.buy_hold_log_ret[0].append(0)


    def _finalize_results(self):
        """
        Calculates overall performance metrics (Total Return, Sharpe Ratio, Sortino Ratio, Max Drawdown)
        for each account/symbol. Saves these metrics, daily returns, and potentially other
        logged data (like daily indicator snapshots) to CSV files. Also saves profit graph images.
        """
        # Saves daily indicator snapshots if they were recorded.
        if self.daily_records: # `self.daily_records` is not populated in the current code version.
            pd.DataFrame(self.daily_records).to_csv(
                os.path.join(self.graph_folder, "daily_indicator_snapshot.csv"),
                index=False
            )

        records = [] # List to store summary metrics for each account/symbol.
        # Determines labels for symbols (individual symbols or "ALL" for shared account).
        for idx, sym in enumerate(self.symbols if self.separate_accounts_per_coin else ["ALL"]):
            # Checks if data exists for the current symbol/account.
            if idx >= len(self.daily_returns) or not self.daily_returns[idx]:
                log.warning(f"Warning: No daily returns data for {sym} (index {idx}). Skipping metrics calculation.")
                continue

            bal_path = np.array(self.daily_returns[idx]) # Equity curve.
            rets     = np.array(self.daily_log_ret[idx]) # Daily log returns.

            # Calculates metrics if sufficient data is available.
            if len(bal_path) < 2 or len(rets) == 0: # Needs at least 2 data points for returns.
                log.warning(f"Warning: Insufficient data for metrics for {sym} (index {idx}). bal_path len: {len(bal_path)}, rets len: {len(rets)}")
                total_ret, avg_ret, mdd, sharpe, downside, sortino, calmar = 0,0,0,0,0,0,0 # Default to zero or NaN.
            else:
                start_cap = self.account_balance_start
                total_ret = (bal_path[-1] - start_cap) / start_cap if start_cap != 0 else 0
                avg_ret   = rets.mean() # Average daily log return.
                # Max Drawdown calculation.
                bal_path_full = np.insert(bal_path, 0, start_cap) 
                peak_curve    = np.maximum.accumulate(bal_path_full)
                mdd           = ((peak_curve - bal_path_full) / peak_curve).max() \
                    if peak_curve.max() != 0 else 0
                # Sharpe Ratio (annualized, assuming 365 trading days).
                sharpe    = (avg_ret / rets.std()) * np.sqrt(365) if rets.std() != 0 else np.nan
                downside_rets = rets[rets < 0] # Considers only negative returns for Sortino.
                downside  = downside_rets.std() if len(downside_rets) > 0 and downside_rets.std() !=0 else np.nan # Downside deviation.
                # Sortino Ratio (annualized).
                sortino   = (avg_ret / downside) * np.sqrt(365) if downside is not np.nan and downside != 0 else np.nan
                # Calmar Ratio: Total Return / Max Drawdown
                delta_days = (self.end_date - self.start_date).days
                years = delta_days / 365.0
                if mdd == 0 or years == 0:
                    calmar = np.nan
                else:
                    cagr = cagr = (bal_path[-1] / start_cap) ** (1 / years) - 1
                    calmar = cagr / mdd
            # Appends metrics to the records list.
            records.append({
                "Symbol":       sym,
                "Strategy":     self.strategy,
                "Interval":     self.time_interval_str,
                "TP_SL":        self.tp_sl_choice,
                "TotalReturn":  total_ret,
                "AvgDailyRet":  avg_ret,
                "MaxDrawdown":  mdd,
                "Sharpe":       sharpe,
                "Sortino":      sortino,
                "Calmar":       calmar,
                "Start":        self.start_date,
                "End":          self.end_date,
                "Leverage":     self.leverage, 
                "SL_mult":      self.sl_mult,
                "TP_mult":      self.tp_mult
            })

        # Saves strategy metrics to a CSV file. Appends if file exists.
        csv_path = "./results/strategy_metrics.csv"
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        pd.DataFrame(records).to_csv(
             csv_path,
             mode="a",
             index=False,
             header=not os.path.exists(csv_path),
         )

        # Prepares and saves daily returns data to a CSV file.
        daily_rows = []
        label_list = self.symbols if self.separate_accounts_per_coin else ["ALL"]
        for idx, sym in enumerate(label_list):
            # Checks for data availability.
            if idx >= len(self.daily_log_ret) or idx >= len(self.daily_dates) or not self.daily_log_ret[idx]:
                log.warning(f"Warning: No daily log returns or dates for {sym} (index {idx}). Skipping daily rows generation.")
                continue

            # Calculates cumulative returns from log returns.
            cum_ret_vec = np.exp(np.cumsum(self.daily_log_ret[idx])) - 1
            # --- Buy-and-Hold reference for the same account/symbol -----------------
            bh_log_vec = self.buy_hold_log_ret[idx] if idx < len(self.buy_hold_log_ret) else []
            bh_cum_vec = np.exp(np.cumsum(bh_log_vec)) - 1 if bh_log_vec else []
            for k, (t, r, cum) in enumerate(zip(self.daily_dates[idx], self.daily_log_ret[idx], cum_ret_vec)):
                # safe defaults for days where B&H vector is shorter
                bh_r    = bh_log_vec[k] if k < len(bh_log_vec) else np.nan
                bh_cum  = bh_cum_vec[k] if k < len(bh_cum_vec) else np.nan
                # +1 offset because buy_hold_equity[0] is the initial balance
                bh_eqty = (self.buy_hold_equity[idx][k + 1] if idx < len(self.buy_hold_equity) and (k + 1) < len(self.buy_hold_equity[idx]) else np.nan)
                # Attempts to retrieve ATR values if available in bot indicators.
                atr_series = []
                # Uses first bot's ATR if not separate accounts, otherwise specific bot's ATR.
                bot_idx_for_atr = idx if self.separate_accounts_per_coin else 0
                if bot_idx_for_atr < len(self.bots):
                     atr_series = self.bots[bot_idx_for_atr].indicators.get("ATR", {}).get("values", [])

                atr_val = np.nan # Default ATR value.
                if atr_series:
                    try:
                        # Finds the candle index corresponding to the date `t` to get the correct ATR.
                        if self.main_date: # Ensure main_date reference exists.
                            candle_idx_for_atr = self.main_date.index(t)
                            if candle_idx_for_atr < len(atr_series):
                                atr_val = atr_series[candle_idx_for_atr]
                    except ValueError: # `t` not found in `self.main_date`.
                        pass
                    except IndexError: # `candle_idx_for_atr` out of bounds for `atr_series`.
                        pass

                # Appends daily data row.
                daily_rows.append({
                    "Date":     pd.to_datetime(t).date(),
                    "Symbol":   sym,
                    "Strategy": self.strategy,
                    "Interval": self.time_interval_str,
                    "TP_SL":    self.tp_sl_choice,
                    "Return":   r, # Daily log return.
                    "ATR":      atr_val,
                    "Leverage": self.leverage,
                    "SL_mult" : self.sl_mult,
                    "TP_mult" : self.tp_mult,
                    "CumReturn": cum, # Cumulative return.
                    "BH_Return": bh_r,          # buy-&-hold daily log-return
                    "BH_CumReturn": bh_cum,     # buy-&-hold cumulative return
                    "BH_Equity": bh_eqty,       # buy-&-hold equity curve
                    })

        # Saves daily returns to a CSV file. Appends if file exists.
        csv_path = "./results/daily_returns.csv"
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        pd.DataFrame(daily_rows).to_csv(
             csv_path,
             mode="a",
             index=False,
             header=not os.path.exists(csv_path),
         )

        # Creates directories for saving results if they don't exist.
        os.makedirs(self.graph_folder, exist_ok=True) # Base results folder.
        strat_dir = os.path.join(self.graph_folder, self.strategy) # Strategy-specific subfolder.
        os.makedirs(strat_dir, exist_ok=True)
        stamp = (
        f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        f"_sl{self.sl_mult:g}_tp{self.tp_mult:g}_lev{self.leverage:g}"
        )
        run_dir   = os.path.join(strat_dir, stamp)
        os.makedirs(run_dir, exist_ok=True)
        strat_dir = run_dir   

        # Saves daily candle data (if self.daily_records was populated).
        if self.daily_records:
            pd.DataFrame(self.daily_records).to_csv(
                os.path.join(strat_dir, f"{self.strategy}_daily_candle_data.csv"), index=False
            )

        # Saves per-account summaries and profit graphs.
        summaries = []
        if self.separate_accounts_per_coin:
            for idx, sym in enumerate(self.symbols):
                 # Ensures index is valid for account balances and profit graphs.
                 if idx < len(self.account_balances) and idx < len(self.profit_graphs):
                    summaries.append(self._save_account_stats(sym, idx, strat_dir))
                 else:
                    log.warning(f"Warning: Index {idx} for symbol {sym} is out of bounds for account_balances/profit_graphs. Skipping summary.")
        else: # Not separate accounts (single shared account).
            if self.account_balances and self.profit_graphs: # Ensures lists are not empty.
                 summaries.append(self._save_account_stats("AllCoins", 0, strat_dir))
            else:
                 log.warning("Warning: account_balances or profit_graphs is empty for 'AllCoins' summary. Skipping.")

        # Saves the collected summaries to a CSV file.
        if summaries:
            summary_csv = os.path.join(
                strat_dir, f"{self.strategy}_{stamp}_final_summary.csv"
                )
            pd.DataFrame(summaries).to_csv(summary_csv, index=False)
        # ── TRADE-GRAPH GENERATION ─────────────────────────────────────────────
        if self.make_trade_graphs and self.completed_trades:
            if self.completed_trades:
                log.info("Generating trade graphs for %d closed trades …", len(self.completed_trades))
                trades_for_graphs = []

                for t in self.completed_trades:
                    bot = self.bots[t.index]
                    # populate candle + indicator slices
                    get_candles_for_graphing(bot, t, self.graph_before, self.graph_after)
                    get_indicators_for_graphing(
                        bot.indicators,
                        t,
                        self.graph_before,
                        self.graph_after,
                        bot.current_index if bot.current_index >= 0 else len(bot.Close) - 1,
                    )
                    trades_for_graphs.append(t.trade_info)

                graphs_out_dir = os.path.join(strat_dir, "trade_graphs")
                plot_trade_graphs(trades_for_graphs, graphs_out_dir, self.auto_open_graph_images)
            else:
                log.info("No completed trades collected — skipping graph generation.")
        
        if self.completed_trades:
            trade_rows = []
            for t in self.completed_trades:
                info = t.trade_info
                trade_rows.append({
                    "Symbol"      : info.symbol,
                    "Direction"   : "Long" if info.trade_direction == 1 else "Short",
                    "EntryTime"   : info.start_time,
                    "EntryPrice"  : info.entry_price,
                    "ExitTime"    : info.exit_time, 
                    "ExitPrice"   : info.exit_price,
                    "ExitReason"  : info.exit_reason,
                    "PositionSize": t.position_size,
                    "TP"          : info.TP_price,
                    "SL"          : info.SL_price,
                    "TradeResult" : "Win" if info.trade_success else "Loss",
                })

            trades_csv = os.path.join(
                strat_dir, f"{self.strategy}_{stamp}_trades.csv"
            )
            pd.DataFrame(trade_rows).to_csv(trades_csv, index=False)
            log.info("Saved detailed trade ledger: %s", trades_csv)

        if self.trade_stats.total_number_of_trades:
            win_rate = (
                self.trade_stats.wins / self.trade_stats.total_number_of_trades * 100
            )
            log.info(
                "Trades %d   |   Wins %d   |   Losses %d   |   Win-rate %.2f%%",
                self.trade_stats.total_number_of_trades,
                self.trade_stats.wins,
                self.trade_stats.losses,
                win_rate,
            )
        log.info(f"results saved in {strat_dir}")

    def _save_account_stats(self, label, idx, out_dir):
        """
        Calculates final balance and percent gain for a specific account (identified by `idx`).
        Saves the profit graph (equity curve) for this account as a PNG image.
        Returns a dictionary with the symbol/label, final balance, and percent gain.
        """
        # Checks if data for the given index is available and valid.
        if idx >= len(self.account_balances) or idx >= len(self.profit_graphs) or not self.profit_graphs[idx]:
            log.warning(f"Warning: Cannot save stats for {label} (index {idx}). Data missing or profit graph empty.")
            return {"Symbol": label, "FinalBalance": np.nan, "PercentGain": np.nan} # Returns NaN values.

        bal_final = self.account_balances[idx] # Final cash balance of the account.
        # Calculates percentage gain from the starting balance.
        pct_gain = ((bal_final - self.account_balance_start) / self.account_balance_start * 100) if self.account_balance_start != 0 else 0

        # Plots and saves the profit graph (equity curve) for the account.
        plt.figure()
        plt.plot(self.profit_graphs[idx]) # profit_graphs stores the daily equity values.
        plt.title(f"{label} balance")
        plt.savefig(os.path.join(out_dir, f"{label}_profit_graph.png"))
        plt.close() # Closes the plot to free memory.

        log.info(f"{label:>10s}:  balance {bal_final:,.2f}  |  gain {pct_gain:.2f}%")
        return {"Symbol": label, "FinalBalance": bal_final, "PercentGain": round(pct_gain, 2)}

    # -------------------------------------------------------------------------
    # PRETTY SETTINGS PRINT
    # -------------------------------------------------------------------------
    def _print_settings(self):
        settings_txt = (
            "\n========== BACKTEST SETTINGS ==========\n"
            f" Symbols         : {self.symbols}\n"
            f" Strategy        : {self.strategy}\n"
            f" Date range      : {self.start_date} to {self.end_date}\n"
            f" Interval        : {self.time_interval_str} ({self.time_interval} min)\n"
            f" Start balance   : {self.account_balance_start:,}\n"
            f" Leverage        : {self.leverage}x  |  Order size {self.order_size_percent*100:.2f}%\n"
            f" Buffer (candles): {self.buffer_size}   |  Quick-test: {self.quick_test}\n"
            "========================================"
        )
        print(settings_txt)

# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────
def run_backtester(**kwargs):
    """A simple wrapper function to instantiate and run the Backtester class."""
    Backtester(**kwargs).run_backtester()

# --------------------------------------------------------------------
# helper – leave dates empty, caller must supply them
# --------------------------------------------------------------------


# -------------------------------------------------------------------------
# FULL PARAMETER GRID  (unchanged from your original script)
# -------------------------------------------------------------------------
def run_all_combos():
    """
    Iterate through the full grid of
      leverage × timeframe × strategy × TP/SL‑scheme.
    Each combination calls `run_backtester()`.
    This function serves as an example of how to automate running multiple
    backtests with different parameter combinations.
    """

    leverages = [5, 10, 20]
    time_frames = [
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h',
        '12h', '1d', '1w'
    ]
    single_symbols = ['ETHUSDT', 'BTCUSDT'] # Symbols to test single-asset strategies on.

    all_strategies = [ # List of strategies to iterate through.
        'tripleEMA',
        'goldenCross',
        'ATRBreakoutstrategy',
        'pure_RSI_mean_reversion',
        'VWAP_mean_reversion']

    tp_sl_choices = [ # Different Take Profit / Stop Loss calculation methods.
        '%',
        'x (ATR)',
        'x (Swing High/Low) level 1',
        'x (Swing High/Low) level 2',
        'x (Swing Close) level 1',
        'x (Swing Close) level 2'
    ]

    # Base parameters for all backtest runs in this grid search.
    base_kwargs = dict(
        account_balance_start   = 50000,
        order_size              = 10,
        start_date              = "01-01-2020",
        end_date                = "31-12-2024",
        max_open_trades         = 100,
        trade_all_symbols       = False, # Set to False as specific symbols are chosen below.
        separate_accounts_per_coin = True, # Assumes separate tracking for each symbol run.
        use_trailing_stop       = True,
        trailing_stop_callback  = 0.5, 
        slippage_range          = (0.0001, 0.0005),
        fee                     = 0.0004,
        buffer_size             = 600,
        quick_test              = False, # Full test for grid search.
        printing_on             = False, # Usually off for bulk runs.
        graph_folder            = "./results",
        auto_open_graph_images  = False,
        graph_before            = 5,
        graph_after             = 5,
        funding_mode            = "uniform",
        funding_bps             = 0.03
        )

    # ------------- cartesian product over the grid ------------------
    # Iterates through all combinations of leverages, timeframes, strategies, and TP/SL choices.
    for lev in leverages:
        for tf in time_frames:
            for strat in all_strategies:
                for tpsl in tp_sl_choices:

                    # Creates a copy of base_kwargs and updates with current loop parameters.
                    run_kwargs = base_kwargs.copy()
                    run_kwargs.update(
                        leverage       = lev,
                        time_interval  = tf,
                        strategy       = strat,
                        tp_sl_choice   = tpsl
                    )
                    for sym in single_symbols: # Runs the strategy for each specified single symbol.
                        run_kwargs.update(
                            symbols             = [sym],
                            sl_mult             = 1.0, # Example SL/TP multipliers for single strategies.
                            tp_mult             = 2.0
                            )
                        run_backtester(**run_kwargs)

    log.info("Finished running all parameter combinations!")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(947902)
    """
    This block is executed when the script is run directly.
    It calls `run_all_combos()` to start the grid search backtesting process.
    """
    run_all_combos()