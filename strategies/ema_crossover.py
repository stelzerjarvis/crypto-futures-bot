from __future__ import annotations
import pandas as pd

from bot.strategy import Strategy


class EmaCrossoverStrategy(Strategy):
    name = "ema_crossover"
    stop_atr_mult = 1.6
    trail_atr_mult = 2.0
    risk_reward = 2.1
    partial_profit_rr = 1.0
    partial_exit_pct = 0.5
    max_hold_bars = 36

    def should_enter(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        required = [
            "ema_9",
            "ema_21",
            "ema_50",
            "close",
        ]
        if any(col not in df.columns for col in required):
            return False
        if any(pd.isna(curr[col]) for col in required) or any(pd.isna(prev[col]) for col in ["ema_9", "ema_21"]):
            return False
        cross_up = float(prev["ema_9"]) <= float(prev["ema_21"]) and float(curr["ema_9"]) > float(curr["ema_21"])
        close = float(curr["close"])
        ema_50 = float(curr["ema_50"])
        trend_up = close > ema_50
        return cross_up and trend_up

    def should_exit(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        required = ["ema_9", "ema_21", "ema_50", "close"]
        if any(col not in df.columns for col in required):
            return False
        if any(pd.isna(curr[col]) for col in required) or any(pd.isna(prev[col]) for col in ["ema_9", "ema_21"]):
            return False
        cross_down = float(prev["ema_9"]) >= float(prev["ema_21"]) and float(curr["ema_9"]) < float(curr["ema_21"])
        close = float(curr["close"])
        ema_50 = float(curr["ema_50"])
        return cross_down or close < ema_50
