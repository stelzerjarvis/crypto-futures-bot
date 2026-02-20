from __future__ import annotations
from dataclasses import dataclass
import math
import pandas as pd

from bot.indicators import add_indicators


@dataclass
class BacktestResult:
    trades: int
    win_rate: float
    pnl: float
    max_drawdown: float
    sharpe: float


class BacktestEngine:
    def __init__(self, exchange, strategy, symbol: str, timeframe: str = "1h"):
        self.exchange = exchange
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe

    @staticmethod
    def _bars_for_days(days: int, timeframe: str) -> int:
        if not timeframe:
            return days * 24
        unit = timeframe[-1]
        try:
            value = int(timeframe[:-1])
        except ValueError:
            return days * 24
        if value <= 0:
            return days * 24
        if unit == "m":
            return math.ceil(days * 24 * 60 / value)
        if unit == "h":
            return math.ceil(days * 24 / value)
        if unit == "d":
            return math.ceil(days / value)
        return days * 24

    def _fetch_history(self, limit: int = 1000) -> pd.DataFrame:
        if hasattr(self.exchange, "fetch_ohlcv_extended"):
            ohlcv = self.exchange.fetch_ohlcv_extended(self.symbol, timeframe=self.timeframe, limit=limit)
        else:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return add_indicators(df)

    def run(self, days: int = 90) -> BacktestResult:
        bars_needed = self._bars_for_days(days, self.timeframe)
        warmup_bars = max(200, getattr(self.strategy, "min_bars", 30))
        limit = bars_needed + warmup_bars
        df = self._fetch_history(limit=limit)
        in_position = False
        entry_price = 0.0
        entry_index = 0
        position_size = 0.0
        trade_pnl = 0.0
        stop_loss = 0.0
        trailing_stop = 0.0
        take_profit = 0.0
        partial_trigger = 0.0
        partial_done = False
        highest_price = 0.0
        equity = 1.0
        equity_curve = []
        trades = 0
        wins = 0

        start_index = max(30, getattr(self.strategy, "min_bars", 30))
        for i in range(start_index, len(df)):
            slice_df = df.iloc[: i + 1]
            price = float(slice_df.iloc[-1]["close"])
            atr = float(slice_df.iloc[-1]["atr"]) if "atr" in slice_df.columns else 0.0

            if not in_position and self.strategy.should_enter(slice_df):
                if atr <= 0 or math.isnan(atr):
                    equity_curve.append(equity)
                    continue
                in_position = True
                entry_price = price
                entry_index = i
                position_size = 1.0
                trade_pnl = 0.0
                stop_distance = atr * self.strategy.stop_atr_mult
                stop_loss = entry_price - stop_distance
                trailing_stop = stop_loss
                take_profit = entry_price + stop_distance * self.strategy.risk_reward
                partial_trigger = entry_price + stop_distance * self.strategy.partial_profit_rr
                partial_done = False
                highest_price = entry_price
                trades += 1

            elif in_position:
                highest_price = max(highest_price, price)
                if atr > 0 and not math.isnan(atr):
                    trail_candidate = highest_price - atr * self.strategy.trail_atr_mult
                    trailing_stop = max(trailing_stop, trail_candidate)

                stop_level = max(stop_loss, trailing_stop)
                bars_held = i - entry_index

                if price <= stop_level:
                    pnl = (price - entry_price) / entry_price * position_size
                    trade_pnl += pnl
                    equity *= (1 + pnl)
                    wins += 1 if trade_pnl > 0 else 0
                    in_position = False
                elif price >= take_profit:
                    pnl = (price - entry_price) / entry_price * position_size
                    trade_pnl += pnl
                    equity *= (1 + pnl)
                    wins += 1 if trade_pnl > 0 else 0
                    in_position = False
                elif not partial_done and price >= partial_trigger:
                    partial_pct = self.strategy.partial_exit_pct
                    pnl = (price - entry_price) / entry_price * partial_pct
                    trade_pnl += pnl
                    equity *= (1 + pnl)
                    position_size -= partial_pct
                    partial_done = True
                    if position_size <= 0:
                        wins += 1 if trade_pnl > 0 else 0
                        in_position = False
                elif self.strategy.should_exit(slice_df) or bars_held >= self.strategy.max_hold_bars:
                    pnl = (price - entry_price) / entry_price * position_size
                    trade_pnl += pnl
                    equity *= (1 + pnl)
                    wins += 1 if trade_pnl > 0 else 0
                    in_position = False

            if in_position and entry_price > 0:
                marked = equity * (1 + (price - entry_price) / entry_price * position_size)
                equity_curve.append(marked)
            else:
                equity_curve.append(equity)

        pnl_total = equity - 1.0
        win_rate = (wins / trades) if trades else 0.0
        max_drawdown = self._max_drawdown(equity_curve)
        sharpe = self._sharpe_ratio(equity_curve)

        return BacktestResult(
            trades=trades,
            win_rate=win_rate,
            pnl=pnl_total,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
        )

    @staticmethod
    def _max_drawdown(equity_curve: list[float]) -> float:
        peak = -math.inf
        max_dd = 0.0
        for val in equity_curve:
            peak = max(peak, val)
            drawdown = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        return max_dd

    @staticmethod
    def _sharpe_ratio(equity_curve: list[float]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))
            if equity_curve[i - 1] > 0
        ]
        if not returns:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std = math.sqrt(variance)
        return mean_ret / std if std > 0 else 0.0
