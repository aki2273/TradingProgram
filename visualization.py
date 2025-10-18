import os # Provides functions for interacting with the operating system, like creating directories and paths.
from typing import List, Dict # Imports type hints for better code readability and static analysis.
import pandas as pd # Imports pandas, used here for creating DataFrames for plotting.
import plotly.graph_objs as go # Imports graph objects from Plotly for creating chart components.
from plotly.subplots import make_subplots # Imports function to create figures with multiple subplots.
import plotly # Imports the main Plotly library, used here for offline plotting.

# It seems `trade_models` (specifically TradeInfo, Trade) and `Bot_Class` would be imported
# if this file were part of a larger package and these functions were called with those objects.
# Since they are type hints or used as arguments, their import might be assumed by the context
# where this script is used. For standalone execution or clarity, explicit imports would be needed if not present.
from trade_models import TradeInfo, Trade # Assumed import for type hinting.
import Bot_Class # Assumed import for type hinting (Bot_Class.Bot).
from logger import get_logger
log = get_logger()
# === Constants ===
# Defines directory names for storing graphs of winning and losing trades.
WINNING_TRADES_DIR = "winning_trades"
LOSING_TRADES_DIR = "losing_trades"

# === Helper Functions ===
def ensure_folder(path: str):
    """
    Ensures that a directory exists at the given path.
    If the directory (and any necessary parent directories) does not exist, it creates it.

    Args:
        path (str): The directory path to check and create if necessary.
    """
    os.makedirs(path, exist_ok=True) # `exist_ok=True` prevents an error if the directory already exists.

def format_filename(name: str) -> str:
    """
    Formats a string to be suitable for use as a filename.
    It replaces spaces and colons with underscores.

    Args:
        name (str): The input string to format.

    Returns:
        str: The formatted string, safe for use as a filename.
    """
    return str(name).replace(' ', '_').replace(':', '_') # Converts to string first for safety.

# === Main Functions ===

def get_candles_for_graphing(
    Bot: Bot_Class.Bot,
    trade: Trade,
    graph_before: int,
    graph_after: int,
) -> Trade:
    """
    Slice the regular OHLCV arrays around *trade.trade_start_index* while
    clamping indices so we never run outside the available history.
    """
    start_idx = max(0, trade.trade_info.trade_start_index - graph_before)
    end_idx   = min(len(Bot.Close), Bot.current_index + graph_after)

    candles = {
        "Date":   Bot.Date[start_idx:end_idx],
        "Open":   Bot.Open[start_idx:end_idx],
        "Close":  Bot.Close[start_idx:end_idx],
        "High":   Bot.High[start_idx:end_idx],
        "Low":    Bot.Low[start_idx:end_idx],
        "Volume": Bot.Volume[start_idx:end_idx],
    }

    trade.trade_info.candles = candles
    return trade

def get_indicators_for_graphing(indicators: Dict, trade: Trade, graph_before: int, graph_after: int, current_index: int) -> Trade:
    """
    Extracts and slices relevant indicator data for a specific trade to be used in graphing.
    It aligns the indicator data with the candle data timeframe prepared by `get_candles_for_graphing`.

    Args:
        indicators (Dict): The Bot's `indicators` dictionary, where keys are indicator names
                           and values are dicts containing 'values' (list of indicator data)
                           and 'plotting_axis'.
        trade (Trade): The Trade object for which to prepare indicator data. Uses `trade.trade_info.trade_start_index`.
        graph_before (int): Number of data points to include before the trade's start index.
        graph_after (int): Number of data points to include after `current_index`.
        current_index (int): The index representing the end point for slicing (e.g., trade exit or current simulation point).

    Returns:
        Trade: The input `trade` object, with its `trade.trade_info.indicators` attribute populated
               with the dictionary of sliced indicator data.
    """
    # Iterates through each indicator stored in the Bot's `indicators` dictionary.
    for key, indicator_data in indicators.items(): # Changed `indicator` to `indicator_data` to avoid conflict with module name
        try:
            # Slices the indicator values to match the candle data range for the trade.
            values = indicator_data["values"][trade.trade_info.trade_start_index - graph_before : current_index + graph_after]
        except IndexError: # Handles potential IndexError during slicing.
            # Fallback: Slices data only up to `current_index`.
            values = indicator_data["values"][trade.trade_info.trade_start_index - graph_before : current_index]

        # Stores the sliced indicator values and their intended plotting axis in the trade's info.
        trade.trade_info.indicators[key] = {
            "values": values,
            "plotting_axis": indicator_data["plotting_axis"]
        }
    return trade # Returns the modified trade object.

def generate_trade_graphs(trades: List[TradeInfo], trade_graph_folder: str, auto_open_graph_images: bool):
    """
    Generates interactive HTML graphs for a list of completed trades using Plotly.
    Each graph displays candlesticks, volume, trade entry/TP/SL markers, and relevant indicators.
    Graphs are saved into subfolders based on whether the trade was winning or losing and by symbol.

    Args:
        trades (List[TradeInfo]): A list of `TradeInfo` objects. Each `TradeInfo` object should
                                  already have its `candles` and `indicators` attributes populated
                                  by `get_candles_for_graphing` and `get_indicators_for_graphing`.
        trade_graph_folder (str): The base directory where trade graphs will be saved.
        auto_open_graph_images (bool): If True, automatically opens the generated HTML graph files in a browser.
    """
    log.info("Generating Trade Graphs...")

    # Ensures base folders for winning and losing trades exist.
    ensure_folder(f"{trade_graph_folder}/{WINNING_TRADES_DIR}")
    ensure_folder(f"{trade_graph_folder}/{LOSING_TRADES_DIR}")

    # Iterates through each `TradeInfo` object in the provided list.
    for trade in trades:
        # Ensures per-symbol subfolders exist within winning/losing trade directories.
        ensure_folder(f"{trade_graph_folder}/{WINNING_TRADES_DIR}/{trade.symbol}")
        ensure_folder(f"{trade_graph_folder}/{LOSING_TRADES_DIR}/{trade.symbol}")

        # Creates a pandas DataFrame from the candle data stored in `trade.candles`.
        # This DataFrame is used by Plotly for creating the candlestick chart.
        df = pd.DataFrame({
            "Date": trade.candles["Date"],
            "Open": trade.candles["Open"],
            "High": trade.candles["High"],
            "Low": trade.candles["Low"],
            "Close": trade.candles["Close"],
            "Volume": trade.candles["Volume"],
        })

        # Determines the maximum number of subplot rows needed based on the 'plotting_axis' of indicators.
        # Default is 2 axes (1 for candles, 1 for volume if no other indicators).
        keys = list(trade.indicators.keys()) # List of indicator names for this trade.
        # Finds the highest plotting_axis value specified among all indicators.
        max_axes = max((detail["plotting_axis"] for detail in trade.indicators.values()), default=1,)
                        # `indicator_detail` replaces `indicator` to avoid conflict.
        max_axes = max(max_axes, 2)
        if max_axes < 2:
            max_axes = 2
        # Defines the layout of subplots: row heights and types.
        # The first row (for candlesticks) gets 50% of the height.
        # Subsequent rows (for indicators) share the remaining height.
        row_heights = [0.5] + [0.1] * (max_axes - 1) # Example: if max_axes=3, [0.5, 0.1, 0.1] - remaining 0.3 is distributed?
                                                     # This height definition might need adjustment if sum != 1 or for many axes.
                                                     # Plotly usually normalizes these.
        # Specifies the type for each subplot (Candlestick for the first, scatter for others).
        specs = [[{"type": "Candlestick"}]] + [[{"type": "scatter"}] for _ in range(max_axes - 1)]

        # Creates a Plotly figure with the defined subplots.
        fig = make_subplots(rows=max_axes, cols=1, row_heights=row_heights, specs=specs, shared_xaxes=True)

        # Adds the candlestick trace to the first subplot (row 1).
        fig.add_candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles",
            row=1, col=1 # Explicitly assigning to row 1, col 1.
        )

        # Adds a marker for the trade entry point on the candlestick chart.
        entry_marker = go.Scatter(
            x=[trade.start_time], # X-coordinate is the trade start time/date.
            y=[trade.entry_price], # Y-coordinate is the entry price.
            mode="markers",
            name="Entry",
            marker=dict(
                size=12,
                color="green" if trade.trade_direction == 1 else "red", # Green for long, red for short.
                symbol="triangle-up" if trade.trade_direction == 1 else "triangle-down" # Triangle points up for long, down for short.
            ),
            hovertext=f"Entry Price: {trade.entry_price}", # Text to display on hover.
            hoverinfo="text"
        )
        fig.add_trace(entry_marker, row=1, col=1) # Adds to the first subplot.

        # Adds volume as a bar chart to the second subplot (row 2), assuming max_axes >= 2.
        # If max_axes is 1 (no indicators requesting other axes), this might cause an error or plot on row 1.
        # The logic for `max_axes` (default=2) and `specs` implies row 2 should exist if `max_axes >= 2`.
        fig.add_trace(
            go.Bar(x=df["Date"], y=df["Volume"], name="Volume"),
            row=2, col=1
        )
        # If max_axes can be 1, a strategy to handle volume plotting would be needed (e.g., overlay or skip).

        # Adds traces for each indicator to their specified plotting axis.
        for key in keys: # Iterates through indicator names.
            indicator_detail = trade.indicators[key] # Gets data for the current indicator.
            axis_num = indicator_detail["plotting_axis"] # Gets the designated subplot row for this indicator.
            # Ensures the axis_num is valid given `max_axes`.
            if 1 <= axis_num <= max_axes:
                fig.add_trace(
                    go.Scatter(x=df["Date"], y=indicator_detail["values"], name=key),
                    row=axis_num, col=1
                )
            else:
                log.warning(f"Warning: Indicator '{key}' requests plotting_axis {axis_num} which is out of range (1-{max_axes}). Skipping.")


        # Adds horizontal lines for Take Profit (TP) and Stop Loss (SL) levels on the first subplot.
        fig.add_hline(y=trade.TP_price, line_color="lightseagreen", annotation_text="TP", row=1, col=1)
        fig.add_hline(y=trade.SL_price, line_color="red", annotation_text="SL", row=1, col=1)

        # Sets the layout for the figure, including title and template.
        # Determines if the trade was a win or loss for the title and save path.
        result_folder = WINNING_TRADES_DIR if trade.trade_success == 1 else LOSING_TRADES_DIR
        result_label = "Winning Trade" if trade.trade_success == 1 else "Losing Trade"

        fig.update_layout(
            title_text=f"{trade.start_time}: {trade.symbol} {result_label}", # Sets the graph title.
            template="plotly_dark", # Uses a dark theme for the plot.
            xaxis_rangeslider_visible=False # Hides the x-axis range slider for a cleaner look.
        )

        # Saves the generated plot as an HTML file.
        # The filename is based on the trade's start time.
        file_path = f"{trade_graph_folder}/{result_folder}/{trade.symbol}/{format_filename(trade.start_time)}.html"
        plotly.offline.plot(fig, filename=file_path, auto_open=auto_open_graph_images)

    log.info("Finished Generating Trade Graphs.")