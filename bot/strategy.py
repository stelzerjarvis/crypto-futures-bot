from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    name = "base"
    stop_atr_mult = 1.6
    trail_atr_mult = 2.2
    risk_reward = 2.0
    partial_profit_rr = 1.0
    partial_exit_pct = 0.5
    max_hold_bars = 48
    min_bars = 210

    @abstractmethod
    def should_enter(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError

    @abstractmethod
    def should_exit(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError
