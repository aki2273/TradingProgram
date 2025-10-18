from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from joblib import dump
from logger import get_logger

log = get_logger()

# ────────────────────────────── optional Binance client ──────────────────────
try:
    from binance.um_futures import UMFutures  # type: ignore
    _BINANCE_AVAILABLE = True
except ImportError:  # offline / CI environment
    _BINANCE_AVAILABLE = False

    class UMFutures:  # type: ignore
        """Offline stub mimicking the minimal interface we rely on."""
        def __init__(self, *_, **__):
            log.warning(
                "Package 'binance-um-futures' not installed – using offline stub. "
                "All network requests will raise RuntimeError."
            )

        def klines(self, *_, **__):  # noqa: D401  (simple stub)
            raise RuntimeError(
                "binance.um_futures unavailable: install 'binance-um-futures' "
                "or 'python-binance' to enable live downloads."
            )

        def __getattr__(self, name):
            raise AttributeError(
                f"Offline stub for binance.um_futures – '{name}' not available."
            )

# ────────────────────────────── credentials ────────────────────────────────
try:
    from Config_File import API_KEY, API_SECRET  # noqa: N804 – external naming
except (ModuleNotFoundError, ImportError):
    API_KEY = os.getenv("BINANCE_KEY")
    API_SECRET = os.getenv("BINANCE_SECRET")

if not (API_KEY and API_SECRET):
    log.warning(
        "Binance API credentials not found in Config_File.py or environment; "
        "public endpoints will still work but rate‑limits may be stricter."
    )

client = UMFutures(key=API_KEY, secret=API_SECRET)

# ────────────────────────────── helpers ─────────────────────────────

def to_milliseconds(date_str: str) -> int:
    """Convert *dd-mm-YYYY* to a UTC milliseconds epoch timestamp."""
    dt = datetime.strptime(date_str, "%d-%m-%Y")
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

# ─────────────────────────── downloader api ─────────────────────────

def download_history(
    symbol: str,
    interval: str,
    start_str: str,
    filename: str,
    *,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    raise_on_empty: bool = False,
) -> None:
    """Download historical klines and persist as a joblib file.

    Parameters
    ----------
    symbol, interval
        Market and candle timeframe understood by Binance (e.g. "BTCUSDT", "1h").
    start_str
        First candle date *dd-mm-YYYY*.
    filename
        Destination path written with :func:`joblib.dump`.
    max_retries, retry_delay
        How many *consecutive* request errors to tolerate and how long to wait
        between retries. Counter resets after any successful fetch.
    raise_on_empty
        If *True* and the query returns **zero** candles, raise
        :class:`RuntimeError` instead of just logging a warning.
    """

    if not _BINANCE_AVAILABLE:
        raise RuntimeError(
            "Cannot download data because 'binance-um-futures' is not installed."
        )

    log.info("Downloading %s %s – starting %s", symbol, interval, start_str)

    start_ts = to_milliseconds(start_str)
    end_ts   = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    limit    = 1_500  # Binance maximum per request

    data: dict[str, list] = {
        "Open": [],
        "High": [],
        "Low": [],
        "Close": [],
        "Volume": [],
        "Date": [],
    }

    current_ts = start_ts
    consecutive_errs = 0

    while current_ts < end_ts:
        try:
            klines = client.klines(
                symbol=symbol,
                interval=interval,
                startTime=current_ts,
                endTime=end_ts,
                limit=limit,
            )

            if not klines:
                if not data["Date"]:
                    log.warning(
                        "⚠ No candles returned for %s %s in requested range – %s to now.",
                        symbol, interval, start_str,
                    )
                break  # either first request (empty range) or data exhausted

            # ───── parse candles ───────────────────────────────────────
            for k in klines:
                data["Date"].append(datetime.fromtimestamp(k[0] / 1000))
                data["Open"].append(float(k[1]))
                data["High"].append(float(k[2]))
                data["Low"].append(float(k[3]))
                data["Close"].append(float(k[4]))
                data["Volume"].append(float(k[5]))

            dt0 = datetime.fromtimestamp(klines[0][0] / 1000, tz=timezone.utc).replace(tzinfo=None)
            dt1 = datetime.fromtimestamp(klines[-1][6] / 1000, tz=timezone.utc).replace(tzinfo=None)
            log.debug("Fetched %d candles from %s to %s", len(klines), dt0, dt1)

            # advance one millisecond past last close
            current_ts = klines[-1][6] + 1
            consecutive_errs = 0  # reset – success
            time.sleep(0.2)        # throttle

        except Exception as exc:  # noqa: BLE001 – broad to catch any client error
            consecutive_errs += 1
            log.warning(
                "%s %s: %s (retry %d/%d) – sleeping %.1fs",
                symbol, interval, exc, consecutive_errs, max_retries, retry_delay,
            )
            if consecutive_errs >= max_retries:
                log.error("Exceeded %d consecutive retries – aborting %s %s",
                          max_retries, symbol, interval)
                raise
            time.sleep(retry_delay)

    # ─────────────────────── persist / empty‑range guard ────────────────────
    if not data["Date"]:
        msg = f"No candles downloaded for {symbol} {interval}; nothing saved."
        if raise_on_empty:
            raise RuntimeError(msg)
        log.warning("⚠ %s", msg)
        return  # Skip file write

    log.info("Saving %d candles to %s", len(data["Date"]), filename)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    dump(data, filename)
    log.info("Finished %s %s", symbol, interval)

# ────────────────────────────── CLI helper ─────────────────────────────

def main() -> None:  # pragma: no cover – convenience wrapper
    """Download BTC/ETH data across common intervals into *./bulk_data*."""
    os.makedirs("./bulk_data", exist_ok=True)

    symbols = ["BTCUSDT", "ETHUSDT"]
    intervals = [
        "1w", "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h", "1d",
    ]

    for sym in symbols:
        for itv in intervals:
            fn = f"./bulk_data/{sym}_{itv}.joblib"
            download_history(sym, itv, start_str="01-01-2017", filename=fn)

if __name__ == "__main__":  # pragma: no cover
    main()
