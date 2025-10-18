try:
    from ta.momentum import rsi
    from ta.trend import ema_indicator, sma_indicator
    from ta.volatility import average_true_range
    from ta.volume import VolumeWeightedAveragePrice
except ImportError:               # offline / CI shim ───────────────
    def _passthrough(series, **_):
        """Tiny stub that simply returns the input unchanged."""
        return series

    rsi = ema_indicator = sma_indicator = average_true_range = _passthrough
    # volume
    class VolumeWeightedAveragePrice:
        def __init__(self, **kwargs):
            self._series = _passthrough(kwargs["close"])
        def volume_weighted_average_price(self):
            return self._series

import numpy as np
import pandas as pd
import re
# Local strategy module
import TradingStrats as TS


class Bot:
    """Lightweight trading‑bot wrapper around a single strategy.
    Notes
    -----
    Each Bot instance manages at most one live trade at a time. While that
    trade is open the attribute active_trade_direction is set to
    0 (short) or 1 (long). Every strategy helper must return
    -99 (the "NO_SIGNAL" sentinel value) while this flag is not
    None, otherwise the Back‑tester will ignore the signal.

    The back‑tester enforces single‑concurrency via the set
    bots_with_active_trades. A bot is removed from that set and allowed
    to place a new trade only after its open position is fully closed
    (any exit path sets active_trade_direction = None).
    """

    # ---------------------------------------------------------------------
    # Construction & static data
    # ---------------------------------------------------------------------
    def __init__(
        self,
        symbol: str,
        Open: list[float],
        Close: list[float],
        High: list[float],
        Low: list[float],
        Volume: list[float],
        Date: list[str],
        OP: int,
        CP: int,
        index: int,
        tick: float,
        strategy: str,
        TP_SL_choice: str,
        SL_mult: float,
        TP_mult: float,
        backtesting: bool = 1,
        vwap_window: int = 20,
        signal_priority: str = "entry"
    ) -> None:
        # --- raw OHLCV --------------------------------------------------
        self.symbol = symbol
        self.Date = Date

        # Align all arrays to *equal length* (pick shortest len)
        lengths = list(map(len, (Open, Close, High, Low, Volume, Date)))
        if len(set(lengths)) != 1:
            raise ValueError(f"OHLCV arrays length mismatch {lengths}")
        self.Open, self.Close, self.High, self.Low, self.Volume, self.Date = \
            Open, Close, High, Low, Volume, Date

        # --- misc symbol metadata --------------------------------------
        self.OP = OP  # qty precision
        self.CP = CP  # price precision
        self.index = index  # unique bot index inside backtester
        self.tick_size = tick

        # --- runtime state ---------------------------------------------
        self.current_index: int = -1        # updated by backtester each candle
        self.active_trade_direction: int | None = None  # None | 0 (short) | 1 (long)
        self.signal_priority = signal_priority

        # --- config -----------------------------------------------------
        self.strategy = strategy
        self.TP_SL_choice = TP_SL_choice
        self.SL_mult = SL_mult
        self.TP_mult = TP_mult
        self.vwap_window = vwap_window

        # --- pre‑alloc containers --------------------------------------
        self.indicators: dict[str, dict[str, list]] = {}
        self.take_profit_val: list[float] = []
        self.stop_loss_val: list[float] = []
        self.peaks: list[float]   = []
        self.troughs: list[float] = []

        # ----------------------------------------------------------------
        # one‑shot calculations (backtest mode)
        # ----------------------------------------------------------------
        if backtesting:
            self.update_indicators()

    # ---------------------------------------------------------------------
    # Indicator & SL/TP preparation
    # ---------------------------------------------------------------------
    def update_indicators(self) -> None:
        """Compute / refresh all indicator buffers for *all* candles."""
        Close, High, Low, Vol = map(pd.Series, (self.Close, self.High, self.Low, self.Volume))

        if self.strategy == 'tripleEMA':
            self.indicators = {
                'EMA_S': {'values': list(ema_indicator(Close, window=5)),  'plotting_axis': 1},
                'EMA_M': {'values': list(ema_indicator(Close, window=20)), 'plotting_axis': 1},
                'EMA_L': {'values': list(ema_indicator(Close, window=50)), 'plotting_axis': 1},
            }

        elif self.strategy == 'goldenCross':
            self.indicators = {
                'EMA_L': {'values': list(ema_indicator(Close, window=100)), 'plotting_axis': 1},
                'EMA_M': {'values': list(ema_indicator(Close, window=50)),  'plotting_axis': 1},
                'EMA_S': {'values': list(ema_indicator(Close, window=20)),  'plotting_axis': 1},
                'RSI':   {'values': list(rsi(Close)),                     'plotting_axis': 3},
            }

        elif self.strategy == 'ATRBreakoutstrategy':
            self.indicators = {
                'ATR':   {'values': list(average_true_range(High, Low, Close)), 'plotting_axis': 3},
                'SMA20': {'values': list(sma_indicator(Close, window=20)),       'plotting_axis': 1},
            }

        elif self.strategy == 'pure_RSI_mean_reversion':
            self.indicators = {'RSI': {'values': list(rsi(Close)), 'plotting_axis': 2}}

        elif self.strategy == 'VWAP_mean_reversion':
            vwap_vals = VolumeWeightedAveragePrice(high=High, low=Low, close=Close, volume=Vol,
                                                   window=self.vwap_window).volume_weighted_average_price()
            self.indicators = {'VWAP': {'values': list(vwap_vals), 'plotting_axis': 1}}

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.indicators.setdefault("ATR",{"values": list(average_true_range(High, Low, Close)), "plotting_axis": 3},)

        # whenever indicators refresh, recompute generic TP/SL scaffolding
        self.update_TP_SL()

    # .....................................................................
    def update_TP_SL(self) -> None:
        """Pre‑compute per‑candle TP & SL distances based on config."""
        n = len(self.Close)
        fallback = 1e-6
        self.take_profit_val = [fallback * self.TP_mult] * n
        self.stop_loss_val   = [fallback * self.SL_mult] * n
        Close, High, Low = map(pd.Series, (self.Close, self.High, self.Low))

        if self.TP_SL_choice == '%':
            for i in range(n):
                self.take_profit_val[i] = (self.TP_mult / 100.0) * Close[i]
                self.stop_loss_val[i]   = (self.SL_mult / 100.0) * Close[i]

        elif self.TP_SL_choice == 'x (ATR)':
            atr = average_true_range(High, Low, Close)
            for i in range(n):
                if not np.isnan(atr[i]):
                    self.take_profit_val[i] = self.TP_mult * atr[i]
                    self.stop_loss_val[i]   = self.SL_mult * atr[i]

        elif self.TP_SL_choice.startswith('x (Swing'):
            self.peaks   = [0.0] * n
            self.troughs = [0.0] * n
            match = re.search(r"\d+", self.TP_SL_choice)
            if not match:
                raise ValueError(
                    f"Could not parse swing-level from TP_SL_choice '{self.TP_SL_choice}'. "
                    "Use “… level N” where N is a positive integer."
                )
            level = int(match.group())  # extracts "1" .. "3"
            src_hi = High if 'High/Low' in self.TP_SL_choice else Close
            src_lo = Low  if 'High/Low' in self.TP_SL_choice else Close
            for i in range(level, n - level):
                # peak
                if all(src_hi[i] > src_hi[i - k] and src_hi[i] > src_hi[i + k] for k in range(1, level + 1)):
                    self.peaks[i] = src_hi[i]
                # trough
                if all(src_lo[i] < src_lo[i - k] and src_lo[i] < src_lo[i + k] for k in range(1, level + 1)):
                    self.troughs[i] = src_lo[i]
        else:
            raise ValueError(
                f"Unknown TP/SL scheme '{self.TP_SL_choice}'. "
                "Supported: '%', 'x (ATR)', "
                "'x (Swing High/Low) level n', 'x (Swing Close) level n'."
            )

    # ---------------------------------------------------------------------
    # Decision logic (per‑candle)
    # ---------------------------------------------------------------------
    def make_decision(self) -> tuple[int, float, float]:
        """Return (direction, sl_value, tp_value) for current candle index."""
        Trade_Direction: int = -99  # default = hold / no entry
        stop_loss_val:   float = -99
        take_profit_val: float = -99

        # -----------------------------------------------------------------
        # 1) Explicit‑exit first (only applies when already in market)
        # -----------------------------------------------------------------
        if self.strategy == 'VWAP_mean_reversion' and self.active_trade_direction is not None:
            if TS.VWAP_mean_reversion_exit(self.Close, self.indicators['VWAP']['values'],
                                           self.current_index, self.active_trade_direction) == -99:
                # request close → backtester interprets -99 as explicit exit
                return -99, -99, -99

        # -----------------------------------------------------------------
        # 2) Entry / regular signal
        # -----------------------------------------------------------------
        if self.strategy == 'tripleEMA':
            Trade_Direction = TS.tripleEMA(
                -99,
                self.indicators['EMA_S']['values'],
                self.indicators['EMA_M']['values'],
                self.indicators['EMA_L']['values'],
                self.current_index,
                signal_priority=self.signal_priority,
            )

        elif self.strategy == 'goldenCross':
            Trade_Direction = TS.goldenCross(
                -99,
                self.Close,
                self.indicators['EMA_L']['values'],
                self.indicators['EMA_M']['values'],
                self.indicators['EMA_S']['values'],
                self.indicators['RSI']['values'],
                self.current_index,
                signal_priority=self.signal_priority,
            )

        elif self.strategy == 'ATRBreakoutstrategy':
            Trade_Direction = TS.ATRBreakoutstrategy(
                -99,
                self.Close,
                self.indicators['ATR']['values'],
                self.indicators['SMA20']['values'],
                self.current_index,
                signal_priority=self.signal_priority,
            )

        elif self.strategy == 'pure_RSI_mean_reversion':
            Trade_Direction = TS.pure_RSI_mean_reversion(
                -99,
                self.indicators['RSI']['values'],
                self.current_index,
            )

        elif self.strategy == 'VWAP_mean_reversion':
            Trade_Direction = TS.VWAP_mean_reversion(
                -99,
                self.Close,
                self.indicators['VWAP']['values'],
                self.current_index,
                signal_priority=self.signal_priority,
            )

        # -----------------------------------------------------------------
        # 3) On *new* entry signal – compute SL/TP and remember direction
        # -----------------------------------------------------------------
        if Trade_Direction in (0, 1):
            self.active_trade_direction = Trade_Direction
            stop_loss_val, take_profit_val = TS.SetSLTP(
                self.stop_loss_val,
                self.take_profit_val,
                self.peaks,
                self.troughs,
                self.Close,
                self.High,
                self.Low,
                Trade_Direction,
                self.SL_mult,
                self.TP_mult,
                self.TP_SL_choice,
                self.current_index,
            )

        return Trade_Direction, stop_loss_val, take_profit_val
    

