import numpy as np
import re
# ---------------------------------------------------------------------
# ‑‑‑‑‑ helper : Triple‑EMA trend‑reversal
# ---------------------------------------------------------------------


def tripleEMA(
    Trade_Direction: int,
    EMA_S: list[float],
    EMA_M: list[float],
    EMA_L: list[float],
    current_index: int,
    lookback: int = 4,
    signal_priority: str | None = None,
) -> int:
    # ---------- NEW: sanity -------------------------------------------------
    if not (len(EMA_S) == len(EMA_M) == len(EMA_L)):
        raise ValueError("EMA series length mismatch")

    if current_index < lookback:
        return Trade_Direction

    # ------------------------------------------------------------------------
    bullish_setup = all(
        EMA_S[current_index - i] < EMA_M[current_index - i] < EMA_L[current_index - i]
        for i in range(1, lookback + 1)
    )
    bearish_setup = all(
        EMA_S[current_index - i] > EMA_M[current_index - i] > EMA_L[current_index - i]
        for i in range(1, lookback + 1)
    )

    long_sig  = bullish_setup and EMA_S[current_index] > EMA_M[current_index] > EMA_L[current_index]
    short_sig = bearish_setup and EMA_S[current_index] < EMA_M[current_index] < EMA_L[current_index]

    # ---------- NEW: “flat” tie-case ---------------------------------------
    if EMA_S[current_index] == EMA_M[current_index] == EMA_L[current_index]:
        if   signal_priority == "entry":   # caller explicitly wants an entry bias
            return 1          # BUY
        elif signal_priority == "exit":    # caller explicitly wants an exit bias
            return 0          # SELL
        else:
            return Trade_Direction  

    # ---------- unchanged ---------------------------------------------------
    if long_sig and short_sig:
        if   signal_priority == "entry":   # explicit long bias
            return 1
        elif signal_priority == "exit":    # explicit short bias
            return 0
        else:                              # no preference → keep direction
            return Trade_Direction
    if long_sig:
        return 1
    if short_sig:
        return 0
    return Trade_Direction

# ---------------------------------------------------------------------
# ‑‑‑‑‑ helper : Golden‑Cross + RSI filter
# ---------------------------------------------------------------------

def goldenCross(
    Trade_Direction: int,
    Close: list[float],
    EMA100: list[float],
    EMA50: list[float],
    EMA20: list[float],
    RSI: list[float],
    current_index: int,
    lookback: int = 3,
    signal_priority: str = "entry",
) -> int:
    """
    Golden-/Death-cross signal with a simple RSI filter.

    Returns
    -------
    int
        •  1 → go/stay **long**
        •  0 → go/stay **short**
        • -99 → **hold** (no change)

    The tie-breaker when both directions trigger (extremely rare) or when
    EMA20 == EMA50 exactly is controlled by *signal_priority*:

    * ``"entry"`` → favour **long** (1)  
    * ``"exit"``  → favour **short** (0)
    """
    # ------------------------------------------------------------------ 0) sanity
    if not (
        len(Close)
        == len(EMA100)
        == len(EMA50)
        == len(EMA20)
        == len(RSI)
    ):
        raise ValueError("Input series length mismatch for goldenCross")

    if current_index < 1:        # need at least one bar of history
        return Trade_Direction

    eff_lb = min(lookback, current_index)  # clamp so idx-i never < 0

    # ------------------------------------------------------------------ 1) signals
    # bullish side
    long_sig = (
        Close[current_index] > EMA100[current_index]
        and RSI[current_index] > 50
        and any(
            EMA20[current_index - i] < EMA50[current_index - i]
            and EMA20[current_index] > EMA50[current_index]
            for i in range(1, eff_lb + 1)
        )
    )

    # bearish side
    short_sig = (
        Close[current_index] < EMA100[current_index]
        and RSI[current_index] < 50
        and any(
            EMA20[current_index - i] > EMA50[current_index - i]
            and EMA20[current_index] < EMA50[current_index]
            for i in range(1, eff_lb + 1)
        )
    )

    # ------------------------------------------------------------------ 2) decide
    if long_sig:
        return 1
    if short_sig:
        return 0

    # exact tie – use priority rule
    if EMA20[current_index] == EMA50[current_index]:
        if current_index < lookback:
            return 1 if signal_priority == "entry" else 0
        return Trade_Direction

    # no new signal → keep current direction
    return Trade_Direction

# ---------------------------------------------------------------------
# ‑‑‑‑‑ helper : ATR breakout
# ---------------------------------------------------------------------

def ATRBreakoutstrategy(
    Trade_Direction: int,
    Close: list[float],
    ATR: list[float],
    SMA20: list[float],
    current_index: int,
    atr_multiplier: float = 2.0,
    signal_priority: str = "entry",
) -> int:
    """ATR breakout with tie-breaker."""
    if not (len(Close) == len(ATR) == len(SMA20)):
        raise ValueError("Input series length mismatch for ATRBreakoutstrategy")
    if current_index >= len(Close):
        raise ValueError("current_index out of bounds for input series")
    if current_index < 1 or np.isnan(ATR[current_index]):
        return Trade_Direction

    up = SMA20[current_index] + atr_multiplier * ATR[current_index]
    dn = SMA20[current_index] - atr_multiplier * ATR[current_index]

    long_sig  = Close[current_index-1] <= up and Close[current_index]  > up
    short_sig = Close[current_index-1] >= dn and Close[current_index]  < dn

    if long_sig and short_sig:
        return 1 if signal_priority == "entry" else 0
    if long_sig:
        return 1
    if short_sig:
        return 0
    return Trade_Direction


# ---------------------------------------------------------------------
# ‑‑‑‑‑ helper : Pure RSI mean‑reversion
# ---------------------------------------------------------------------

def pure_RSI_mean_reversion(
    Trade_Direction: int,
    RSI_vals: list[float],
    current_index: int,
    overbought_level: float = 70.0,
    oversold_level: float = 30.0,
    use_confirmation: bool = True,
) -> int:
    """RSI crosses back from OB/OS with tie-breaker."""
    if current_index >= len(RSI_vals):
        raise ValueError("RSI series shorter than current_index")
    if current_index < 1:
        return Trade_Direction

    r_prev, r_curr = RSI_vals[current_index - 1], RSI_vals[current_index]

    long_sig = (
        (r_prev < oversold_level <= r_curr) if use_confirmation
        else (r_curr < oversold_level)
    )
    short_sig = (
        (r_prev > overbought_level >= r_curr) if use_confirmation
        else (r_curr > overbought_level)
    )

    if long_sig:
        return 1
    if short_sig:
        return 0
    return Trade_Direction

# ---------------------------------------------------------------------
# ‑‑‑‑‑ helper : VWAP mean‑reversion (entry)
# ---------------------------------------------------------------------

def VWAP_mean_reversion(
    Trade_Direction: int,
    Close: list[float],
    VWAP: list[float],
    current_index: int,
    deviation_threshold: float = 0.005,
    confirmation: bool = False,
    signal_priority: str = "entry",
) -> int:
    """VWAP mean-reversion entry with tie-breaker."""
    if not (len(Close) == len(VWAP)):
        raise ValueError("Input series length mismatch for VWAP_mean_reversion")
    if current_index >= len(Close):
        raise ValueError("current_index out of bounds for input series")
    if current_index < 1 or VWAP[current_index] == 0:
        return Trade_Direction

    close_now = Close[current_index]
    close_prev = Close[current_index - 1]
    vwap_now = VWAP[current_index]
    dev = (close_now - vwap_now) / vwap_now

    long_sig = dev < -deviation_threshold and (not confirmation or close_now > close_prev)
    short_sig = dev > deviation_threshold and (not confirmation or close_now < close_prev)

    if long_sig and short_sig:
        return 1 if signal_priority == "entry" else 0
    if long_sig:
        return 1
    if short_sig:
        return 0
    return Trade_Direction

# ---------------------------------------------------------------------
# ‑‑‑‑‑ helper : VWAP mean‑reversion (EXIT)
# ---------------------------------------------------------------------

def VWAP_mean_reversion_exit(
    Close: list[float],
    VWAP: list[float],
    current_index: int,
    trade_direction: int,
    tolerance: float = 0.0005,  # 0.05 %
) -> int:
    """Explicit exit logic for VWAP mean‑reversion trades.

    Returns –99 when either:
      • price has reverted to within `tolerance` of VWAP, OR
      • the price crosses to the opposite side of VWAP (trend may be over).
    Otherwise returns the unchanged `trade_direction`."""

    if not (len(Close) == len(VWAP)):
        raise ValueError("Input series length mismatch for VWAP_mean_reversion_exit")
    if current_index >= len(Close):
        raise ValueError("current_index out of bounds for input series")
    if trade_direction not in (0, 1):
        return trade_direction
    if current_index < 0 or VWAP[current_index] == 0:
        return trade_direction

    price = Close[current_index]
    vwap_now = VWAP[current_index]
    dev = (price - vwap_now) / vwap_now

    # Reversion reached ------------------------------------------------
    if abs(dev) <= tolerance:
        return -99

    # Crossed the other side of VWAP ----------------------------------
    if trade_direction == 1 and price > vwap_now:
        return -99
    if trade_direction == 0 and price < vwap_now:
        return -99

    return trade_direction

# ---------------------------------------------------------------------
# ‑‑‑‑‑ universal SL/TP calculator
# ---------------------------------------------------------------------

def SetSLTP(
    stop_loss_val_arr: list[float],
    take_profit_val_arr: list[float],
    peaks: list[float],
    troughs: list[float],
    Close: list[float],
    High: list[float],
    Low: list[float],
    Trade_Direction: int,
    SL: float,
    TP: float,
    TP_SL_choice: str,
    current_index: int,
):
    """
    Return the *distance* (absolute price difference) from the entry-price that
    should be used for the Stop-Loss (SL) and Take-Profit (TP), according to the
    user-selected scheme.

    Supported schemes
    -----------------
    • "%"                         – direct percentage (pre-computed arrays)
    • "x (ATR)"                  – ATR multiple      (pre-computed arrays)
    • "x (Swing High/Low) level n"
    • "x (Swing Close) level n"

    Raises
    ------
    ValueError
        If `TP_SL_choice` is not one of the recognised schemes.
    """

    # ------------------------------------------------------------------
    # 1) Direct percentage or ATR -– already pre-calculated elsewhere
    # ------------------------------------------------------------------
    if TP_SL_choice in {"%", "x (ATR)"}:
        return (
            stop_loss_val_arr[current_index],
            take_profit_val_arr[current_index],
        )

    # ------------------------------------------------------------------
    # 2) Swing-based levels
    # ------------------------------------------------------------------
    if TP_SL_choice.startswith("x (Swing"):
        # ---- parse "... level n" -------------------------------------
        match = re.search(r"\d+", TP_SL_choice)
        if not match:
            raise ValueError(
                f"Invalid TP/SL scheme '{TP_SL_choice}': missing numeric level."
            )
        level = int(match.group())

        use_close = "Swing Close" in TP_SL_choice
        ref_high = Close if use_close else High
        ref_low = Close if use_close else Low

        # ---- find the most recent swing --------------------------------
        swing_high = ref_high[current_index]  # sensible default
        swing_low = ref_low[current_index]
        count = 0
        for i in range(current_index - 1, -1, -1):          # ← now starts at index-1
            if Trade_Direction == 0 and peaks[i]:           # SHORT: look for peaks
                count += 1
                if count == level:                          # grab the nth-most-recent
                    swing_high = peaks[i]
                    break
            elif Trade_Direction == 1 and troughs[i]:       # LONG: look for troughs
                count += 1
                if count == level:
                    swing_low = troughs[i]
                    break

        # ---- convert to absolute price *distances* ---------------------
        if Trade_Direction == 0:  # SHORT
            sl_dist = SL * max(0, swing_high - Close[current_index])
            tp_dist = TP * sl_dist
        else:                      # LONG
            sl_dist = SL * max(0, Close[current_index] - swing_low)
            tp_dist = TP * sl_dist

        if sl_dist == 0:
            sl_dist = max(stop_loss_val_arr[current_index], 1e-9)
        if tp_dist == 0:
            tp_dist = max(take_profit_val_arr[current_index], 1e-9)

        return sl_dist, tp_dist

    raise ValueError(
        f"Unknown TP/SL scheme '{TP_SL_choice}'. "
        "Supported: '%', 'x (ATR)', "
        "'x (Swing High/Low) level n', 'x (Swing Close) level n'."
    )
