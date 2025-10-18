from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TradeStats:
    total_number_of_trades: int = 0
    wins: int = 0
    losses: int = 0


@dataclass
class TradeInfo:
    symbol: str
    TP_price: float
    SL_price: float
    trade_direction: int
    slippage: float = None
    entry_price: float = None
    trade_success: bool = False
    exit_price:   float | None = None
    exit_reason:  str   | None = None
    trade_start_index: int = -99
    indicators: Dict[str, Any] = field(default_factory=dict)
    candles: Dict[str, Any] = field(default_factory=dict)
    start_time: str = ''
    exit_time:  str = ''
    entry_fee: float = 0.0
    exit_fee:  float = 0.0
    volatility_at_entry: float | None = None



@dataclass
class Trade:
    index: int
    position_size: float
    TP_val: float
    SL_val: float
    trade_direction: int
    order_id: int
    symbol: str
    entry_price: float = -99
    TP_id: str = ''
    SL_id: str = ''
    trade_status: int = 0  # 0 = not started · 1 = in-progress · 2 = Take-Profit · 3 = Stop-Loss · 4 = Closed-on-Signal · 5 = Trailing-Stop
    trade_start: str = ''
    Highest_val: float = float('-inf')
    Lowest_val: float = float('inf')
    trail_activated: bool = False
    same_candle: bool = True
    trade_info: TradeInfo = field(init=False)

    def __post_init__(self):
        self.trade_info = TradeInfo(
            symbol          = self.symbol,
            TP_price        = self.TP_val,
            SL_price        = self.SL_val,
            trade_direction = self.trade_direction,
        )

    def print_vals(self) -> tuple:
        """
        Return the exact tuple shape expected by ``logger.extract_trade_info``:
        (symbol, entry_price, position_size, TP_val, SL_val,
        trade_direction, trade_status, Highest_val, Lowest_val)
        """
        return(
            self.symbol,
            self.entry_price,
            self.position_size,
            self.TP_val,
            self.SL_val,
            self.trade_direction,
            self.trade_status,
            self.Highest_val,
            self.Lowest_val,
        )
    
    @property
    def exit_price(self):
        return self.trade_info.exit_price
    
    @exit_price.setter
    def exit_price(self, val):
        self.trade_info.exit_price = val

    @property
    def exit_reason(self):
        return self.trade_info.exit_reason
    
    @exit_reason.setter
    def exit_reason(self, val):
        self.trade_info.exit_reason = val