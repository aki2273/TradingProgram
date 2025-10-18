import os
import csv
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Tuple, Dict, Any

from tabulate import tabulate
from trade_models import Trade

# ───────────────────────────── logging setup ──────────────────────────────
CSV_HEADER: list[str] = [
    "Time", "AccountBalance", "Symbol", "Entry", "Size", "Close",
    "TP", "SL", "Direction",
    "Highest Candle",          
    "Lowest Candle",           
    "Trade Status",             
    "NetPnL", "PnLPercent",
    "Fees", "VolatilityAtEntry",
    "ExitPrice", "ExitReason",
]
LOG_PATH = "./logs"
MAIN_LOG_FILE = f"{LOG_PATH}/backtester.log"
ERROR_LOG_FILE = f"{LOG_PATH}/errors.log"
os.makedirs(LOG_PATH, exist_ok=True)

LOGGER_NAME = "backtester"
_logger = logging.getLogger(LOGGER_NAME)
_logger.setLevel(logging.DEBUG)  # Capture everything; handlers will filter.

# Console output – INFO and above
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")
)
_logger.addHandler(_console_handler)

# Rotating file – DEBUG and above
_file_handler = RotatingFileHandler(MAIN_LOG_FILE, maxBytes=2_000_000, backupCount=3)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")
)
_logger.addHandler(_file_handler)

# Separate error file – ERROR and above
_error_handler = RotatingFileHandler(ERROR_LOG_FILE, maxBytes=1_000_000, backupCount=2)
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")
)
_logger.addHandler(_error_handler)
_logger.propagate = False 

def get_logger() -> logging.Logger:
    """Public accessor so other modules can do::

        from logger import get_logger
        log = get_logger()
    """
    return _logger


# ─────────────────────────── compatibility shim ───────────────────────────

def log_error(message: str): 
    """Legacy helper retained for backward‑compatibility."""
    get_logger().error(message)


# ───────────────────────────── helper functions ───────────────────────────

def extract_trade_info(active_trades: List[Trade], trade_price: List[float]) -> Dict[str, List[Any]]:
    """Return a dict of human‑readable columns for *active_trades*.

    The resulting mapping can be fed straight into *tabulate* or turned into
    a DataFrame.
    """
    info: Dict[str, List[Any]] = {
        "Symbol": [],
        "Direction": [],
        "Entry": [],
        "Close": trade_price,  # current/closing price for PnL calc
        "Size": [],
        "TP": [],
        "SL": [],
        "PNL": [],
        "Trade Status": [],
        "Highest Candle": [],
        "Lowest Candle": [],
    }

    status_mapping = {
        0: "New position Opened",
        1: "In Progress",
        2: "Take Profit Hit",
        3: "Stop Loss Hit",
        4: "Closed on Condition",
        5: "Trailing Stop Hit",
    }

    for i, trade in enumerate(active_trades):
        (
            symbol,
            entry_price,
            position_size,
            TP_val,
            SL_val,
            direction,
            status,
            highest,
            lowest,
        ) = trade.print_vals()

        # PnL – sign depends on direction (0 = short, 1 = long)
        pnl = (
            (entry_price - trade_price[i]) * position_size
            if direction == 0
            else (trade_price[i] - entry_price) * position_size
        )

        info["Symbol"].append(symbol)
        info["Direction"].append("Short" if direction == 0 else "Long" if direction == 1 else "Closed")
        info["Entry"].append(entry_price)
        info["Size"].append(position_size)
        info["TP"].append(TP_val)
        info["SL"].append(SL_val)
        info["PNL"].append(pnl)
        info["Trade Status"].append(status_mapping.get(status, "Unknown"))
        info["Highest Candle"].append(highest if highest != float("-inf") else "N/A")
        info["Lowest Candle"].append(lowest if lowest != float("inf") else "N/A")
        info.setdefault('ExitPrice',  []).append(trade.exit_price)
        info.setdefault('ExitReason', []).append(trade.exit_reason)
        total_fees = (trade.trade_info.entry_fee or 0.0) + (trade.trade_info.exit_fee or 0.0)
        info.setdefault("Fees", []).append(total_fees)
        info.setdefault("VolatilityAtEntry", []).append(
            trade.trade_info.volatility_at_entry if trade.trade_info.volatility_at_entry is not None else "")

    return info


# ─────────────────────────── public log helpers ───────────────────────────

def log_info(
    active_trades: List[Trade],
    trade_price: List[float],
    dates: List[str],
    account_balance: float,
    csv_name: str,
    indicators: List[Tuple[str, List[Any]]],
):
    """Comprehensive snapshot: *INFO* to console, *DEBUG* to log, CSV row‑wise.

    The CSV file receives one row **per active trade** each time this function
    is called.  Additional *indicators* are appended as extra columns.
    """
    log = get_logger()

    info = extract_trade_info(active_trades, trade_price)
    info["Date"] = dates

    for name, values in indicators:
        info[name] = values

    # ── console & log output ────────────────────────────────────────────
    log.info("Account Balance: %.2f", account_balance)
    table_str = tabulate(info, headers="keys", tablefmt="fancy_grid")
    log.debug("\n%s\n%s", table_str, "-" * 60)

    # ── CSV logging ─────────────────────────────────────────────────────
    write_trades_to_csv(
        info,
        account_balance,
        dates[0] if dates else "",
        csv_name,
        active_trades,
        )


def print_trades(
    active_trades: List[Trade],
    trade_price: List[float],
    date: int,
    account_balance: List[float],
    change_occurred: bool,
    print_to_csv: bool,
    csv_name: str,
    csv_path: str,
    time_delta: int,
) -> Tuple[float, int]:
    """Real‑time logger called by the back‑tester loop.

    Returns
    -------
    Tuple[float, int]
        total_pnl : float
            Aggregate PnL of *all* open trades (single‑account mode only).
        bankruptcy_flag : int
            1 if `total_pnl + account_balance[0] < 0`, else 0 (single‑account).
    """

    if not change_occurred:
        return 0.0, 0

    log = get_logger()
    info = extract_trade_info(active_trades, trade_price)

    # convert *date* index plus *time_delta* to a displayable value
    time_str = date + time_delta  # still an int; caller decides meaning

    # ── single‑account mode ────────────────────────────────────────────
    if len(account_balance) == 1:
        log.info("Time: %s  |  Account Balance: %.2f", time_str, account_balance[0])
        log.debug("\n%s\n%s", tabulate(info, headers="keys", tablefmt="fancy_grid"), "-" * 60)

        if print_to_csv:
            write_trades_to_csv(
                info,
                account_balance[0],
                str(time_str),
                os.path.join(csv_path, csv_name),
                active_trades,
            )

        total_pnl = sum(info["PNL"])
        return total_pnl, int(total_pnl + account_balance[0] < 0)

    # ── multi‑account mode ─────────────────────────────────────────────
    account_balance_info = [account_balance[t.index] for t in active_trades]
    info["Account Balance"] = account_balance_info

    log.info("Time: %s", time_str)
    log.debug("\n%s\n%s", tabulate(info, headers="keys", tablefmt="fancy_grid"), "-" * 60)

    if print_to_csv:
        write_trades_to_csv(
            info,
            account_balance_info,
            str(time_str),
            os.path.join(csv_path, csv_name),
            active_trades,
        )

    return 0.0, 0


# ─────────────────────────── CSV helper ──────────────────────

def write_trades_to_csv(
    info: Dict[str, List[Any]],
    account_balance,
    time_str: str,
    file_path: str,
    active_trades: List[Trade],
):
    """Append *one row per active trade* to *file_path*.

    NOTE  The header is written on *every* call here.  In a production
    system you might want to check `os.path.isfile(...) and os.path.getsize(...)`
    first, but the file is typically rotated per run so the overhead is
    negligible.
    """
    header = CSV_HEADER
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        # write header once per file
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writerow(header)

        for i, trade in enumerate(active_trades):
            net_pnl = info["PNL"][i] if info.get("PNL") else 0.0

            pnl_percent = 0.0
            if trade.entry_price != 0 and trade.position_size != 0:
                pnl_percent = (
                    net_pnl / (abs(trade.entry_price) * abs(trade.position_size)) * 100
                )

            fees = info["Fees"][i] if info.get("Fees") else 0.0
            volatility_at_entry = (
                info["VolatilityAtEntry"][i] if info.get("VolatilityAtEntry") else 0.0
            )

            current_acc_balance = (
                account_balance if isinstance(account_balance, (float, int)) else account_balance[i]
            )

            writer.writerow(
                [
                    time_str,
                    current_acc_balance,
                    info["Symbol"][i],
                    info["Entry"][i],
                    info["Size"][i],
                    info["Close"][i],
                    info["TP"][i],
                    info["SL"][i],
                    info["Direction"][i],
                    info["Highest Candle"][i],
                    info["Lowest Candle"][i],
                    info["Trade Status"][i],
                    net_pnl,
                    pnl_percent,
                    fees,
                    volatility_at_entry,
                    info["ExitPrice"][i],
                    info["ExitReason"][i],  
                ]
            )
