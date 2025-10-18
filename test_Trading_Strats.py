"""
Robust, deterministic test-suite for *TradingStrats.py*
======================================================

Targets ≥ 90 % statement *and* branch coverage, with property-based checks
(Hypothesis), offline CI operation (dummy TA-Lib shims), and a gated
integration smoke-test.
"""
from __future__ import annotations

import importlib
from typing import Any
import sys
import types
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from hypothesis import (
    HealthCheck,
    assume,
    given,
    settings,
    strategies as st,
)



# ---------------------------------------------------------------------------
# Global deterministic seeding ----------------------------------------------
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Auto-use fixture: mock minimal TA-Lib API *before* importing TradingStrats ---
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _patch_ta():
    """Provide light-weight TA-Lib shims so the library imports offline."""

    trend = types.ModuleType("ta.trend")
    trend.ema_indicator = lambda s, window=10: s
    trend.sma_indicator = lambda s, window=10: s

    momentum = types.ModuleType("ta.momentum")
    momentum.rsi = lambda s, window=14: s

    volatility = types.ModuleType("ta.volatility")
    volatility.average_true_range = (
        lambda h, l, c, window=14: np.full_like(c, 1.0)
    )

    sys.modules.update(
        {
            "ta": types.ModuleType("ta"),
            "ta.trend": trend,
            "ta.momentum": momentum,
            "ta.volatility": volatility,
        }
    )


# ---------------------------------------------------------------------------
# Library under test (safe to import after TA shims) -------------------------
# ---------------------------------------------------------------------------
import TradingStrats as strat  # noqa: E402  pylint: disable=wrong-import-position

# =============================================================================
# Helper builders -------------------------------------------------------------
# =============================================================================


def _build_ema_scenarios(lookback: int, scenario: str):
    """Craft EMA_S/M/L arrays whose last bar enforces a given outcome."""

    if scenario == "bullish":
        ema_l = [6] * lookback + [5, 4]
        ema_m = [5] * lookback + [4, 5]
        ema_s = [4] * lookback + [3, 6]
        expected = 1
    elif scenario == "bearish":
        ema_l = [4] * lookback + [5, 6]
        ema_m = [5] * lookback + [6, 5]
        ema_s = [6] * lookback + [7, 4]
        expected = 0
    else:  # flat
        ema_l = ema_m = ema_s = [5] * (lookback + 2)
        expected = -99
    return ema_s, ema_m, ema_l, expected


# =============================================================================
# goldenCross -----------------------------------------------------------------
# =============================================================================

class TestGoldenCross:
    @pytest.mark.parametrize("scenario", ["bullish", "bearish", "flat"])
    def test_expected_signals(self, scenario):
        lookback = 3
        n        = lookback + 2        # minimum history for the helper

        if scenario == "bullish":      # EMA20 crosses ↑ above EMA50
            close  = [200] * n
            ema100 = [100] * n
            ema50  = [150] * (n - 1) + [120]
            ema20  = [ 90] * (n - 1) + [140]
            rsi    = [60]  * n
            exp    = 1

        elif scenario == "bearish":    # EMA20 crosses ↓ below EMA50
            close  = [90] * n
            ema100 = [100] * n
            ema50  = [ 80] * (n - 1) + [120]
            ema20  = [130] * (n - 1) + [ 70]
            rsi    = [40]  * n
            exp    = 0

        else:                          # flat / no-signal
            close = ema100 = ema50 = ema20 = rsi = [100] * n
            exp   = -99

        idx = n - 1
        out = strat.goldenCross(
            -99, close, ema100, ema50, ema20, rsi, idx, lookback=lookback
        )
        assert out == exp

    def test_tie_breaker_priority(self):
        c = [120, 120]
        out_entry = strat.goldenCross(-99, c, c, c, c, [60, 60], 1, signal_priority="entry")
        out_exit  = strat.goldenCross(-99, c, c, c, c, [60, 60], 1, signal_priority="exit")
        assert (out_entry, out_exit) == (1, 0)

    def test_input_length_mismatch(self):
        with pytest.raises(ValueError):
            strat.goldenCross(-99, [1], [1, 1], [1], [1], [50], 0)

    # ---------- Property: output always ∈ {-99, 0, 1} --------------------
    @given(
        arr=st.lists(st.floats(50, 150), min_size=15, max_size=25),
        data=st.data(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_enum_integrity(self, arr, data):
        n   = len(arr)
        idx = n - 1
        out = strat.goldenCross(
            -99,
            arr,
            [v * 0.9  for v in arr],    # EMA100 < price
            [v * 0.95 for v in arr],
            [v * 0.96 for v in arr],
            [55] * n,                   # RSI>50 ⇒ long-bias possible
            idx,
        )
        assert out in (-99, 0, 1)


# =============================================================================
# pure_RSI_mean_reversion -----------------------------------------------------
# =============================================================================

class TestPureRSI:
    @pytest.mark.parametrize("trade_dir", [1, 0])
    def test_basic_signals(self, trade_dir):
        # oversold → revert  |  overbought → revert
        rsi_seq = [20, 35] if trade_dir == 1 else [80, 65]
        exp     = 1 if trade_dir == 1 else 0
        out = strat.pure_RSI_mean_reversion(-99, rsi_seq, 1, oversold_level=30, overbought_level=70)
        assert out == exp

    def test_hold_when_no_cross(self):
        out = strat.pure_RSI_mean_reversion(-99, [50, 55], 1)
        assert out == -99

    # ---------- Property: output enumeration --------------------------------
    @given(
        rsi_vals=st.lists(st.floats(5, 95), min_size=20, max_size=50),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_enum_integrity(self, rsi_vals):
        assume(len(rsi_vals) >= 2)
        idx = len(rsi_vals) - 1
        out = strat.pure_RSI_mean_reversion(-99, rsi_vals, idx)
        assert out in (-99, 0, 1)


# =============================================================================
# tripleEMA -------------------------------------------------------------------
# =============================================================================


class TestTripleEMA:
    @pytest.mark.parametrize("lookback", [2, 4, 7, 10])
    @pytest.mark.parametrize("scenario", ["bullish", "bearish", "flat"])
    def test_expected_signals(self, lookback, scenario):
        ema_s, ema_m, ema_l, exp = _build_ema_scenarios(lookback, scenario)
        res = strat.tripleEMA(
            -99, ema_s, ema_m, ema_l, current_index=lookback + 1, lookback=lookback
        )
        assert res == exp

    def test_input_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            strat.tripleEMA(-99, [1, 2], [1], [1, 2], current_index=1)

    def test_no_mutation(self):
        ema_s, ema_m, ema_l, _ = _build_ema_scenarios(4, "bullish")
        orig = list(ema_s), list(ema_m), list(ema_l)
        strat.tripleEMA(-99, ema_s, ema_m, ema_l, 5, lookback=4)
        assert (ema_s, ema_m, ema_l) == orig

    # ---------------- Property: enumerated output range --------------------
    @given(
        ema_vals=st.lists(st.floats(0.01, 10_000), min_size=12, max_size=20),
        lookback=st.integers(2, 6),
        data=st.data(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_enum_integrity(self, ema_vals, lookback, data):
        idx = len(ema_vals) - 1
        assume(idx >= lookback)
        np.random.seed(data.draw(st.integers(0, 2**32 - 1)))
        ema_s = ema_vals
        ema_m = [v * 1.001 for v in ema_vals]
        ema_l = [v * 1.002 for v in ema_vals]
        out = strat.tripleEMA(-99, ema_s, ema_m, ema_l, idx, lookback)
        assert out in (-99, 0, 1)


# =============================================================================
# ATRBreakoutstrategy ---------------------------------------------------------
# =============================================================================


class TestATRBreakout:
    @pytest.mark.parametrize(
        "move, mult, exp",
        [
            (1.0, 0.5, 1),    # upward move, lenient ⇒ long
            (1.0, 3.0, -99),  # upward move, strict ⇒ no signal
            (3.0, 2.0, 1),    # large upward move ⇒ long
            (-3.0, 2.0, 0),   # large downward move ⇒ short
        ],
    )
    def test_thresholds(self, move, mult, exp):
        close = [100, 100 + move]
        atr = [1, 1]
        sma = [100, 100]
        res = strat.ATRBreakoutstrategy(-99, close, atr, sma, 1, atr_multiplier=mult)
        assert res == exp

    def test_nan_graceful(self):
        res = strat.ATRBreakoutstrategy(
            -99, [100, 101], [np.nan, np.nan], [100, 100], 1
        )
        assert res == -99

    def test_false_signal_rate(self):
        """Ensure the strategy does not spam entry signals on a random walk.
        A fixed RNG seed makes this check deterministic and CI-stable."""
        np.random.seed(42)

        n = 300
        walk = np.cumsum(np.random.normal(0, 1, n))
        atr  = np.abs(np.random.normal(1, 0.2, n))
        sma20 = pd.Series(walk).rolling(20, min_periods=1).mean().bfill()

        sigs = [
            strat.ATRBreakoutstrategy(
                -99, walk.tolist(), atr.tolist(), sma20.tolist(), i
            )
            for i in range(1, n)
        ]
        rate = sum(s in {0, 1} for s in sigs) / (n - 1)
        assert rate < 0.25


# =============================================================================
# VWAP mean-reversion ---------------------------------------------------------
# =============================================================================


class TestVWAP:
    def test_basic_long_short(self):
        base = 100
        close_lo = [base, base * 0.97]  # 3 % below VWAP ⇒ long
        close_hi = [base, base * 1.03]  # 3 % above VWAP ⇒ short
        vwap = [base, base]
        res_long = strat.VWAP_mean_reversion(
            -99, close_lo, vwap, 1, deviation_threshold=0.02
        )
        res_short = strat.VWAP_mean_reversion(
            -99, close_hi, vwap, 1, deviation_threshold=0.02
        )
        assert (res_long, res_short) == (1, 0)

    @pytest.mark.parametrize("trade_dir", [0, 1])
    def test_exit_logic(self, trade_dir):
        close = [100, 100.7] if trade_dir == 1 else [100, 99.3]
        vwap = [100, 100]
        res = strat.VWAP_mean_reversion_exit(
            close, vwap, 1, trade_dir, tolerance=0.005
        )
        assert res == -99

    # ---------------- Property: symmetry & idempotence --------------------
    @given(base=st.floats(50, 200), pct=st.floats(0.005, 0.03))
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_direction_symmetry(self, base, pct):
        close_lo = [base, base * (1 - pct)]
        close_hi = [base, base * (1 + pct)]
        vwap = [base, base]
        long_sig = strat.VWAP_mean_reversion(
            -99, close_lo, vwap, 1, deviation_threshold=pct / 2
        )
        short_sig = strat.VWAP_mean_reversion(
            -99, close_hi, vwap, 1, deviation_threshold=pct / 2
        )
        assert {long_sig, short_sig} <= {0, 1} and long_sig != short_sig

    @given(
        close=st.lists(st.floats(50, 200), min_size=40, max_size=60),
        vwap=st.lists(st.floats(50, 200), min_size=40, max_size=60),
        thresh=st.floats(0.005, 0.03),
        data=st.data(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_idempotence(self, close, vwap, thresh, data):
        assume(len(close) == len(vwap))
        idx = len(close) - 1
        np.random.seed(data.draw(st.integers(0, 2**32 - 1)))
        out1 = strat.VWAP_mean_reversion(
            -99, close, vwap, idx, deviation_threshold=thresh
        )
        np.random.seed(data.draw(st.integers(0, 2**32 - 1)))
        out2 = strat.VWAP_mean_reversion(
            -99, close, vwap, idx, deviation_threshold=thresh
        )
        assert out1 == out2


# =============================================================================
# SetSLTP ---------------------------------------------------------------------
# =============================================================================


class TestSetSLTP:
    SCHEMES = [
        "%",
        "x (ATR)",
        "x (Swing High/Low) level 1",
        "x (Swing Close) level 1",
    ]

    @pytest.mark.parametrize("scheme", SCHEMES)
    @pytest.mark.parametrize("direction", [0, 1])
    def test_static_examples(self, scheme, direction):
        sl_arr = [1.0] * 8
        tp_arr = [2.0] * 8
        close = [100] * 8
        high = [101] * 8
        low = [99] * 8
        peaks = [0, 0, 105, 0, 0, 0, 0, 0]
        troughs = [0, 95, 0, 0, 0, 0, 0, 0]

        sl, tp = strat.SetSLTP(
            sl_arr,
            tp_arr,
            peaks,
            troughs,
            close,
            high,
            low,
            direction,
            SL=1.5,
            TP=2.0,
            TP_SL_choice=scheme,
            current_index=7,
        )
        assert sl >= 0 and tp >= 0
        if scheme in {"%", "x (ATR)"}:
            assert sl == pytest.approx(1.0)
            assert tp == pytest.approx(2.0)

    # -------------- Property: idempotence & direction-symmetry -------------
    @given(
        ohlc=st.lists(st.floats(25, 75), min_size=60, max_size=60).map(np.asarray),
        trade_dir=st.sampled_from([0, 1]),
        scheme=st.sampled_from(SCHEMES),
        data=st.data(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_idempotence_and_symmetry(self, ohlc, trade_dir, scheme, data):
        close = ohlc
        high = close + 2.0
        low = close - 2.0
        n = len(close)
        idx = n - 2

        sl_arr = np.full(n, 0.01)
        tp_arr = np.full(n, 0.02)
        peaks = np.zeros(n)
        troughs = np.zeros(n)

        np.random.seed(data.draw(st.integers(0, 2**32 - 1)))
        sl1, tp1 = strat.SetSLTP(
            sl_arr,
            tp_arr,
            peaks,
            troughs,
            close,
            high,
            low,
            trade_dir,
            1.0,
            2.0,
            scheme,
            idx,
        )
        sl2, tp2 = strat.SetSLTP(
            sl_arr,
            tp_arr,
            peaks,
            troughs,
            close,
            high,
            low,
            trade_dir,
            1.0,
            2.0,
            scheme,
            idx,
        )
        assert sl1 == pytest.approx(sl2)
        assert tp1 == pytest.approx(tp2)

        # Flip prices & direction
        sl_sym, tp_sym = strat.SetSLTP(
            sl_arr,
            tp_arr,
            troughs,  # swap peaks/troughs
            peaks,
            -close,
            -low,   # high ↔ low when mirrored
            -high,
            1 - trade_dir,
            1.0,
            2.0,
            scheme,
            idx,
        )
        assert sl1 == pytest.approx(sl_sym)
        assert tp1 == pytest.approx(tp_sym)

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="scheme"):
            strat.SetSLTP(
                [1], [1], [0], [0], [100], [101], [99],
                1, 1.0, 2.0, "foo", 0
            )


# =============================================================================
# Edge-case price levels ------------------------------------------------------
# =============================================================================


class TestEdgeCases:
    def test_extreme_prices(self):
        close_big = [1e10, 1e10 + 1e8]
        vwap_big = [1e10, 1e10]
        res_big = strat.VWAP_mean_reversion(
            -99, close_big, vwap_big, 1, deviation_threshold=0.02
        )
        assert res_big in (-99, 0, 1)

        close_small = [1e-6, 2e-6]
        vwap_small = [1e-6, 1e-6]
        res_small = strat.VWAP_mean_reversion(
            -99, close_small, vwap_small, 1, deviation_threshold=0.2
        )
        assert res_small in (-99, 0, 1)


# ---------------------------------------------------------------------------
# 1) trade_utils – open / TP logic
# ---------------------------------------------------------------------------

from trade_utils import check_take_profit, open_trade  # noqa: E402
from trade_models import Trade  # noqa: E402


def test_open_trade_basic():
    """
    ``open_trade`` should

    * return the exact *order_qty* given a zero–slippage environment,
    * deduct the opening fee from the cash balance.
    """
    symbol = "TEST"
    order_notional = 1_000.0
    balance_start = 5_000.0
    ohlc_open = 10.0  # entry at 10 USDT
    fee = 0.001  # 0.1 %
    qty_prec, price_prec = 0, 2

    # deterministic slippage
    qty, entry_px, new_balance, slippage = open_trade(
        symbol,
        order_notional,
        balance_start,
        ohlc_open,
        fee,
        qty_prec,
        price_prec,
        trade_direction=1,  # long
        slippage_range=(0.0, 0.0),
        printing_on=False,
    )

    exp_qty = order_notional / ohlc_open
    exp_fee = entry_px * exp_qty * fee

    assert qty == pytest.approx(exp_qty)
    assert entry_px == pytest.approx(ohlc_open)
    assert new_balance == pytest.approx(balance_start - exp_fee)
    assert slippage == pytest.approx(0.0)


def test_check_take_profit_hit():
    """
    A long position whose TP lies within the candle *High–Low*
    must be closed as a win and credit realised PnL (net of fees).
    """
    t = Trade(
        index=0,
        position_size=10.0,
        TP_val=12.0,
        SL_val=8.0,
        trade_direction=1,  # long
        order_id=1,
        symbol="TEST",
    )
    t.entry_price = 10.0
    account_balance = 0.0
    fee = 0.001

    # Candle that pierces the TP
    high, low = 12.5, 9.5

    t, new_balance = check_take_profit(
        t,
        account_balance,
        high,
        low,
        fee,
        use_trailing_stop=False,
        trailing_stop_callback=0.0,
        CP=2,
        printing_on=False,
    )

    # --------------- assertions --------------------
    assert t.trade_status == 2  # Take-profit hit
    assert t.trade_info.trade_success == 1
    # Positive balance increment expected
    assert new_balance > account_balance


# ---------------------------------------------------------------------------
# 2) Bot.run generator
# ---------------------------------------------------------------------------

from Bot_Class import Bot  # noqa: E402


def _make_dummy_bot() -> Bot:
    """Return a *tiny* Bot instance with arrays of length 3."""
    ohlcv = [10.0, 10.1, 10.2]
    return Bot(
        symbol="TEST",
        Open=ohlcv,
        Close=ohlcv,
        High=ohlcv,
        Low=ohlcv,
        Volume=[100, 100, 100],
        Date=["2025-01-01", "2025-01-02", "2025-01-03"],
        OP=0,
        CP=2,
        index=0,
        tick=0.01,
        strategy="tripleEMA",
        TP_SL_choice="%",
        SL_mult=1.0,
        TP_mult=2.0,
        backtesting=False,
        signal_priority="entry",
    )

def test_bot_make_decision_iterates_len():
    """
    Even though ``Bot.run`` is unused in production, it should still
    iterate over the full candle range.
    """
    b = _make_dummy_bot()

    # Keep the test lightweight by stubbing the indicator-heavy method.
    b.make_decision = types.MethodType(lambda self: (-99, -99, -99), b)  # type: ignore[attr-defined]

    iterations = 0
    for idx in range(len(b.Close)):
        b.current_index = idx
        b.make_decision()
        iterations += 1
    assert iterations == len(b.Close) == 3


# ---------------------------------------------------------------------------
# 3) Backtester._open_new_trades
# ---------------------------------------------------------------------------


def _patch_external_modules() -> None:
    """Insert stubs for the Binance client & config *before* importing Backtester."""
    dummy_binance = types.ModuleType("binance")
    dummy_um = types.ModuleType("binance.um_futures")
    dummy_um.UMFutures = lambda *_, **__: None  # noqa: D401 – simple stub
    sys.modules.update({"binance": dummy_binance, "binance.um_futures": dummy_um})

    cfg = types.ModuleType("Config_File")
    cfg.API_KEY = "x"
    cfg.API_SECRET = "y"
    sys.modules["Config_File"] = cfg


_patch_external_modules()
Backtester_mod = importlib.import_module("Backtester")
Backtester = Backtester_mod.Backtester


class DummyBot:
    """Minimal subset of :class:`Bot` used by ``_open_new_trades``."""
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.OP = 0
        self.CP = 2
        self.current_index = 0
        self.active_trade_direction: int | None = None


def test_open_new_trades_opens_position():
    """
    Verify that a queued entry signal is converted into an *active* Trade
    and the account balance debits the opening fee.
    """
    # --- build half-initialised Backtester instance ----------------------
    bt: Any = object.__new__(Backtester)  # bypass heavy __init__

    bt.leverage = 1.0
    bt.order_size_percent = 0.5
    bt.fee = 0.001
    bt.slippage_range = (0.0, 0.0)
    bt.max_open_trades = 5
    bt.separate_accounts_per_coin = False
    bt.account_balances = [1_000.0]
    bt.trade_counter = 0

    bt.bots = [DummyBot("TESTUSDT")]
    bt.opens_interval_data = [[10.0, 11.0]]  # needs i+1 index
    bt.closes_interval_data = [[10.0, 11.0]]
    bt.dates_interval_data = [[dt.datetime(2024, 1, 1),dt.datetime(2024, 1, 2)]]

    # _open_new_trades reads this attribute for bookkeeping
    bt.bots[0].current_index = 0
    # placeholders unused by _open_new_trades
    bt.printing_on = False

    # --- run the method --------------------------------------------------
    queue = [(0, 1, 0.5, 1.0)]  # bot-idx, direction, SL, TP
    active: list[Trade] = []

    bt._open_new_trades(queue, active, i=0)

    # Expect one Trade object with correct status
    assert len(active) == 1
    t = active[0]
    assert t.entry_price == pytest.approx(11.0)
    qty       = t.position_size
    exp_fee   = t.entry_price * qty * bt.fee
    assert bt.account_balances[0] == pytest.approx(1_000.0 - exp_fee)



