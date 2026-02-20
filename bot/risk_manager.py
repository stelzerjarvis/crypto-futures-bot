from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class RiskLimits:
    risk_per_trade: float
    max_positions: int
    max_daily_loss: float


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.daily_pnl = 0.0
        self.daily_reset = datetime.now(timezone.utc).date()

    def _reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self.daily_reset:
            self.daily_reset = today
            self.daily_pnl = 0.0

    def update_daily_pnl(self, pnl: float):
        self._reset_daily()
        self.daily_pnl += pnl

    def can_trade(self, open_positions: int) -> bool:
        self._reset_daily()
        if open_positions >= self.limits.max_positions:
            return False
        if self.daily_pnl <= -abs(self.limits.max_daily_loss):
            return False
        return True

    def position_size(self, equity: float, entry_price: float, stop_loss_price: float) -> float:
        if entry_price <= 0 or stop_loss_price <= 0:
            raise ValueError("Prices must be positive")
        risk_amount = equity * self.limits.risk_per_trade
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance == 0:
            raise ValueError("Stop loss distance cannot be zero")
        return risk_amount / stop_distance

    def compute_stop_take(self, entry_price: float, atr: float, risk_reward: float = 2.0, atr_mult: float = 1.5):
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if atr <= 0:
            raise ValueError("ATR must be positive")
        stop_distance = atr * atr_mult
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + stop_distance * risk_reward
        return stop_loss, take_profit
