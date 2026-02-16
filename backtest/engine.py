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

    def _fetch_history(self, limit: int = 1000) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return add_indicators(df)

    def run(self, days: int = 90) -> BacktestResult:
        limit = min(days * 24, 1000)
        df = self._fetch_history(limit=limit)
        in_position = False
        entry_price = 0.0
        equity = 1.0
        equity_curve = []
        trades = 0
        wins = 0

        for i in range(30, len(df)):
            slice_df = df.iloc[: i + 1]
            price = float(slice_df.iloc[-1]["close"])

            if not in_position and self.strategy.should_enter(slice_df):
                in_position = True
                entry_price = price
                trades += 1

            elif in_position and self.strategy.should_exit(slice_df):
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl)
                wins += 1 if pnl > 0 else 0
                in_position = False

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
