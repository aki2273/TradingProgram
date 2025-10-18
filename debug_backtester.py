import os
import sys
import argparse
from Backtester import Backtester

def parse_args():
    parser = argparse.ArgumentParser(description='Debug backtester for trading strategies')
    parser.add_argument('-s', '--strategy', type=str, default='candle_wick',
                        help='Strategy to test (default: candle_wick)')
    parser.add_argument('-sym', '--symbol', type=str, default='BTCUSDT',
                        help='Symbol to test (default: BTCUSDT)')
    parser.add_argument('-sd', '--start_date', type=str, default='01-01-2020',
                        help='Start date (default: 01-01-2020)')
    parser.add_argument('-ed', '--end_date', type=str, default='30-03-2020',
                        help='End date (default: 30-03-2020)')
    parser.add_argument('-tf', '--timeframe', type=str, default='15m',
                        help='Timeframe (default: 15m)')
    parser.add_argument('-b', '--balance', type=float, default=50000,
                        help='Initial balance (default: 50000)')
    parser.add_argument('-l', '--leverage', type=float, default=5,
                        help='Leverage (default: 5)')
    parser.add_argument('-o', '--order_size', type=float, default=3,
                        help='Order size as percentage of balance (default: 3)')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug output (default: False)')
    return parser.parse_args()

def debug_single_strategy():
    """
    Run a single strategy with detailed debugging output to diagnose trade execution issues
    """
    # Parse command line arguments
    args = parse_args()
    
    print(f"Running debug backtester for {args.strategy} on {args.symbol}...")
    
    # Force printing to be enabled for all debug messages
    os.environ['DEBUG_MODE'] = 'True'
    
    # Create a backtester instance with the specified strategy
    bt = Backtester(
        strategy=args.strategy,
        symbols=[args.symbol],
        start_date=args.start_date,
        end_date=args.end_date,
        time_interval=args.timeframe,
        initial_balance=args.balance,
        leverage=args.leverage,
        order_size_percent=args.order_size,
        max_open_trades=10,
        tp_sl_choice="x",
        sl_mult=1.0,
        tp_mult=0.5,
        fee=0.0002,
        printing_on=True,  # Enable detailed printing
        buffer_size=200,
        quick_test=False,
        trade_session_on=True,
        graph_folder="./debug_results",
        plot_trade_graphs=False,
        slippage=0.01,
        use_trailing_stop=False,
        trailing_stop_callback=0.0,
        separate_accounts_per_coin=False,
        debug_mode=args.debug  # Pass the debug flag
    )
    
    # Monkey patch the _simulate_trading_loop method to add more debug prints
    original_simulate = bt._simulate_trading_loop
    
    def debug_simulate():
        print("\n=== STARTING SIMULATION WITH DETAILED DEBUGGING ===\n")
        # Track key metrics during simulation
        trades_attempted = 0
        trades_executed = 0
        
        # Initialize active trades list
        active_trades = []
        trade_count = 0
        trading_on = True
        
        # Loop through each candle
        for i in range(bt.buffer_size, len(bt.close_1min[0]) - 1):
            # Update each bot with current candle data
            for k, bot in enumerate(bt.bots):
                # Update bot state with current candle
                bot.current_index = i
                bot.current_price = bt.close_1min[k][i]
                bot.current_date = bt.date_1min[k][i]
                
                # Make trading decision
                Trade_Direction, stop_loss_val, take_profit_val = bot.Make_decision()
                
                # Debug print for trade signals
                if Trade_Direction != -99:
                    print(f"[DEBUG] Trade signal at index {i}: {bot.symbol}, Direction={Trade_Direction}, SL={stop_loss_val}, TP={take_profit_val}")
                    trades_attempted += 1
                
                # Check if we have a valid trade signal
                if Trade_Direction != -99 and stop_loss_val != -99 and take_profit_val != -99:
                    # Create a new trade
                    new_trades = [(k, Trade_Direction, stop_loss_val, take_profit_val)]
                    
                    # Process the new trade
                    old_count = len(active_trades)
                    trade_count = bt._open_new_trades(new_trades, active_trades, i, trade_count)
                    trades_executed += (len(active_trades) - old_count)
            
            # Update active trades
            bt._update_active_trades(active_trades, i)
            
            # Print progress every 1000 candles
            if i % 1000 == 0:
                print(f"Processing candle {i}/{len(bt.close_1min[0])-1}, Active trades: {len(active_trades)}")
        
        print("\n=== SIMULATION COMPLETE ===")
        print(f"Total trades attempted: {trades_attempted}")
        print(f"Total trades executed: {trades_executed}")
        print(f"Final trade count: {trade_count}")
        print(f"Final active trades: {len(active_trades)}")
        
        # Return the active trades and trade count for the original method
        return active_trades, trade_count
    
    # Replace the original method with our debug version
    bt._simulate_trading_loop = debug_simulate
    
    # Run the backtest
    bt.run_backtester()
    
    # Print summary
    print("\nBacktest Summary:")
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Initial balance: {args.balance}")
    print(f"Final balance: {bt.account_balances[0]}")
    print(f"Profit/Loss: {bt.account_balances[0] - args.balance}")
    print(f"Return: {((bt.account_balances[0] / args.balance) - 1) * 100:.2f}%")
    print(f"Number of trades: {len(bt.trades_for_graphing)}")
    
    return bt

if __name__ == "__main__":
    debug_single_strategy()
