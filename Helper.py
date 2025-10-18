from binance.um_futures import UMFutures
from datetime import datetime, timezone
from Config_File import API_KEY, API_SECRET
from joblib import load, dump
import sys, os
import multiprocessing

import Bot_Class

from trade_models import TradeStats, TradeInfo, Trade

from data_utils import (
    parse_time_interval,
    to_milliseconds,
    initialize_price_data,
    save_price_data,
    fetch_klines,
    aggregate_candle_data,
    get_klines,
    align_datasets,
    get_heikin_ashi,
    multiprocess_get_candles
)




from visualization import (
    generate_trade_graphs,
    get_indicators_for_graphing,
    get_candles_for_graphing
)
from logger import (
    log_error,
    log_info,
    print_trades
)
from trade_utils import (
    get_CAGR,
    check_take_profit,
    check_stop_loss,
    open_trade,
    close_position
)

client = UMFutures(key=API_KEY, secret=API_SECRET)
price_data_path = '.'
