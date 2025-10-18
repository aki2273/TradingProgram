from trade_models import Trade # Imports the Trade dataclass, representing an active trade.
from typing import Tuple # Used for type hinting, specifically for function return types.
import numpy as np # Fundamental package for numerical computation, used here for random slippage.
from datetime import datetime, timezone
# === Helper functions ===

def calculate_pnl(t: Trade, exit_price: float, fee: float, is_long: bool) -> float:
    """
    Calculates the Profit and Loss (PnL) for a trade.
    It considers the entry price, exit price, position size, trade direction, and trading fee.

    Args:
        t (Trade): The Trade object containing details like entry price and position size.
        exit_price (float): The price at which the trade is exited.
        fee (float): The trading fee rate (e.g., 0.0004 for 0.04%).
        is_long (bool): True if the trade is long, False if short.

    Returns:
        float: The calculated PnL for the trade.
    """
    # Determines the direction factor for PnL calculation (1 for long, -1 for short).
    direction_factor = 1 if is_long else -1
    # Calculates gross PnL based on price difference and position size.
    # Then subtracts the fee, which is typically applied to the notional value at exit.
    return direction_factor * (exit_price - t.entry_price) * t.position_size - exit_price * fee * t.position_size


def update_trailing_stop(t: Trade, price: float, callback: float, CP: int) -> float:
    """
    Updates the Trailing Stop (TS) value based on the current price and callback rate.
    The TS value is essentially a new Take Profit (TP) price that trails the market.

    Args:
        t (Trade): The Trade object, used to determine trade direction.
        price (float): The current market price (e.g., High for long, Low for short) used to adjust the TS.
        callback (float): The trailing stop callback rate (e.g., 0.01 for 1%).
        CP (int): Price precision (number of decimal places) for rounding the new TS value.

    Returns:
        float: The new, updated trailing stop price.
    """
    if t.trade_direction == 0:  # Short trade
        # For shorts, TS is above the current price; if price drops, TS also drops (moves closer).
        new_tp = price * (1 + callback)
    else:  # Long trade
        # For longs, TS is below the current price; if price rises, TS also rises (moves closer).
        new_tp = price * (1 - callback)
    # Rounds the new TS price to the symbol's specified price precision.
    return round(new_tp, CP) if CP else round(new_tp)


# === Core functions ===

def check_take_profit(
    t: Trade, account_balance: float, High: float, Low: float,
    fee: float, use_trailing_stop: bool, trailing_stop_callback: float, CP: int, printing_on: bool = False
) -> Tuple[Trade, float]:
    """
    Checks if a trade's Take Profit (TP) or Trailing Stop (TS) has been hit.
    Updates the trade status and account balance if a TP/TS is triggered.

    Args:
        t (Trade): The active Trade object.
        account_balance (float): The current account balance.
        High (float): The high price of the current candle.
        Low (float): The low price of the current candle.
        fee (float): The trading fee rate.
        use_trailing_stop (bool): Flag indicating if trailing stop is enabled.
        trailing_stop_callback (float): The callback rate for the trailing stop.
        CP (int): Price precision for rounding.
        printing_on (bool): Flag to enable/disable printing of trade actions (1 for on, 0 for off).

    Returns:
        Tuple[Trade, float]: The updated Trade object and the new account balance.
    """
    is_long = t.trade_direction == 1 # True if the trade is long.
    # Checks if the TP price was hit by the current candle's High/Low.
    price_hit = (t.TP_val <= High) if is_long else (t.TP_val >= Low)

    if not use_trailing_stop: # Standard Take Profit logic.
        if price_hit: # If TP is hit.
            if printing_on:
                print(f"Take Profit hit on {t.symbol}")
            # Calculates PnL and updates account balance.
            account_balance += calculate_pnl(t, t.TP_val, fee, is_long)
            t.trade_info.exit_fee = t.TP_val * t.position_size * fee
            # — flag this as a winning trade 
            t.trade_status = 2 
            t.trade_info.trade_success = 1 # Take Profit Hit
            t.trade_info.exit_price    = t.TP_val
            t.trade_info.exit_reason   = "Take-Profit"
    else: # Trailing Stop logic.
        # Case 1: Trailing Stop is already active and gets hit.
        if t.trail_activated and ((not is_long and High >= t.TP_val) or (is_long and Low <= t.TP_val)):
            if printing_on:
                print(f"Trailing Stop hit on {t.symbol}")
            account_balance += calculate_pnl(t, t.TP_val, fee, is_long)
            t.trade_info.exit_fee = t.TP_val * t.position_size * fee
            t.trade_status = 5 # Trailing-Stop-Hit ≡ win
            t.trade_info.trade_success = 1
            t.trade_info.exit_price  = t.TP_val
            t.trade_info.exit_reason = "Trailing-Stop"

        # Case 2: Trailing Stop is not active, but the initial TP (acting as trigger) is hit.
        # This activates the trailing stop.
        elif not t.trail_activated and price_hit:
            t.trail_activated = True
            # ---- place the first trail -----------------------------------
            t.TP_val = update_trailing_stop(
                t,
                High if is_long else Low,
                trailing_stop_callback,
                CP,
            )
            if printing_on:
                print(f"Trailing Stop activated on {t.symbol} to {t.TP_val}")

            # ---- immediate re-check on the same candle --------------
            ts_hit_now = (is_long and Low <= t.TP_val) or (not is_long and High >= t.TP_val)
            if ts_hit_now:
                if printing_on:
                    print(f"Trailing Stop *immediately* hit on {t.symbol}")
                account_balance += calculate_pnl(t, t.TP_val, fee, is_long)
                t.trade_info.exit_fee = t.TP_val * t.position_size * fee
                t.trade_status               = 5            # Trailing-Stop-Hit
                t.trade_info.trade_success   = 1
                t.trade_info.exit_price      = t.TP_val
                t.trade_info.exit_reason     = "Trailing-Stop"

        # Case 3: Trailing Stop is active and the market moves further in favor of the trade.
        # The TS value (new TP) is adjusted.
        elif t.trail_activated:
            # Reference price for adjusting TS (Low for short, High for long to tighten the stop).
            reference_price = Low if not is_long else High
            new_tp = update_trailing_stop(t, reference_price, trailing_stop_callback, CP)
            # Adjusts TP_val only if the new TS is more favorable (tighter).
            if (not is_long and new_tp < t.TP_val) or (is_long and new_tp > t.TP_val):
                t.TP_val = new_tp
                if printing_on:
                    print(f"Trailing Stop updated on {t.symbol} to {t.TP_val}")

    return t, account_balance


def check_stop_loss(
    t: Trade, account_balance: float, High: float, Low: float,
    fee: float, printing_on: bool = False
) -> Tuple[Trade, float]:
    """
    Checks if a trade's Stop Loss (SL) has been hit.
    Updates the trade status and account balance if an SL is triggered.

    Args:
        t (Trade): The active Trade object.
        account_balance (float): The current account balance.
        High (float): The high price of the current candle.
        Low (float): The low price of the current candle.
        fee (float): The trading fee rate.
        printing_on (bool): Flag to enable/disable printing of trade actions.

    Returns:
        Tuple[Trade, float]: The updated Trade object and the new account balance.
    """
    is_long = t.trade_direction == 1 # True if the trade is long.
    # Checks if the SL price was hit by the current candle's High/Low.
    price_hit = (t.SL_val >= Low) if is_long else (t.SL_val <= High)

    if price_hit: # If SL is hit.
        if printing_on:
            print(f"Stop Loss hit on {t.symbol}")
        # Calculates PnL (which will be a loss) and updates account balance.
        account_balance += calculate_pnl(t, t.SL_val, fee, is_long)
        t.trade_info.exit_fee = t.SL_val * t.position_size * fee
        t.trade_status = 3 # Stop-Loss Hit
        t.trade_info.trade_success = 0 # loss
        t.trade_info.exit_price    = t.SL_val
        t.trade_info.exit_reason   = "Stop-Loss"

    return t, account_balance


def open_trade(
    symbol: str, order_notional: float, account_balance: float, Open: float,
    fee: float, OP: int, CP: int, trade_direction: int, slippage_range: Tuple[float, float], printing_on: bool = False
) -> Tuple[float, float, float, float]:
    """
    Simulates opening a new trade.
    Calculates entry price considering random slippage, determines order quantity,
    and updates the account balance by deducting the initial fee.

    Args:
        symbol (str): The trading symbol.
        order_notional (float): The desired notional value of the order.
        account_balance (float): The current account balance.
        Open (float): The open price of the candle at which the trade is to be opened.
        fee (float): The trading fee rate.
        OP (int): Quantity precision for rounding the order quantity.
        CP (int): Price precision for rounding the entry price.
        trade_direction (int): The direction of the trade (1 for long, 0 for short).
        slippage_range (tuple[float, float]): A tuple (min_slippage, max_slippage) for random slippage simulation.
        printing_on (bool): Flag to enable/disable printing of trade actions.

    Returns:
        Tuple[float, float, float, float]:
            - order_qty (float): The calculated order quantity.
            - entry_price (float): The entry price after slippage.
            - account_balance (float): The updated account balance after deducting fees.
            - slippage (float): Passthrough of the slippage.
    """
    # Simulates random slippage within the specified range.
    slippage = np.random.uniform(*slippage_range)
    # Adjusts the open price for slippage.
    # For shorts, slippage makes the entry price worse (lower).
    # For longs, slippage makes the entry price worse (higher).
    adjusted_open = Open * (1 - slippage) if trade_direction == 0 else Open * (1 + slippage)
    # Rounds the entry price to the symbol's price precision.
    entry_price = round(adjusted_open, CP) if CP else round(adjusted_open)

    # Calculates the order quantity based on notional value and entry price.
    # Ensures entry_price is not zero to avoid division by zero error.
    order_qty = 0
    if entry_price != 0:
        order_qty = round(order_notional / entry_price, OP) if OP else round(order_notional / entry_price)


    if order_qty > 0: # If a valid quantity is calculated.
        # Deducts the opening fee from the account balance.
        # Fee is based on the notional value at entry.
        account_balance -= entry_price * order_qty * fee
        if printing_on:
            print(f"Trade Opened Successfully on {symbol}")
    else: # If order_qty is 0 (e.g., due to very small notional or high price).
        if printing_on:
            print(f"Trade not opened for {symbol} due to zero quantity. Notional: {order_notional}, Entry Price: {entry_price}")


    return order_qty, entry_price, account_balance, slippage

def close_trade_on_signal(
    t: Trade,
    account_balance: float,
    close_price: float, # The price at which the signal triggers the close.
    fee: float,
    printing_on: bool = False
) -> Tuple[Trade, float]:
    """
    Closes an active trade based on an explicit signal from the trading strategy
    (e.g., a strategy-defined exit condition, not SL/TP).

    Args:
        t (Trade): The active Trade object to be closed.
        account_balance (float): The current account balance.
        close_price (float): The price at which the trade is closed due to the signal.
        fee (float): The trading fee rate.
        printing_on (bool): Flag to enable/disable printing of trade actions.

    Returns:
        Tuple[Trade, float]: The updated (now closed) Trade object and the new account balance.
    """
    is_long = t.trade_direction == 1 # True if the trade is long.

    if printing_on:
        print(f"Strategy signal to close trade on {t.symbol} at price {close_price}")

    # Uses the existing `calculate_pnl` function to determine the PnL from this closure
    # and deducts the closing fee.
    pnl_change = calculate_pnl(t, close_price, fee, is_long)
    account_balance += pnl_change
    t.trade_info.exit_fee = close_price * t.position_size * fee
    t.trade_info.exit_time = str(datetime.now(timezone.utc))
    # Determine win/loss by realised PnL
    is_win = 1 if pnl_change > 0 else 0
    t.trade_status = 4 # Closed on Condition
    t.trade_info.trade_success = is_win
    t.trade_info.exit_price    = close_price
    t.trade_info.exit_reason   = "Signal"

    return t, account_balance