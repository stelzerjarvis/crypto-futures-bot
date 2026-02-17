from __future__ import annotations
import pandas as pd

from bot.strategy import Strategy


class BollingerBreakoutStrategy(Strategy):
    name = "bollinger_breakout"
    stop_atr_mult = 1.7
    trail_atr_mult = 2.3
    risk_reward = 2.0
    partial_profit_rr = 1.0
    partial_exit_pct = 0.5
    max_hold_bars = 24

    def should_enter(self, df: pd.DataFrame) -> bool:
        if len(df) < 1:
            return False
        curr = df.iloc[-1]
        required = [
            "bb_low",
            "rsi",
            "close",
        ]
        if any(col not in df.columns for col in required):
            return False
        if any(pd.isna(curr[col]) for col in required):
            return False
        close = float(curr["close"])
        rsi_14 = float(curr["rsi"])
        return rsi_14 < 40 and close < float(curr["bb_low"])

    def should_exit(self, df: pd.DataFrame) -> bool:
        if len(df) < 1:
            return False
        curr = df.iloc[-1]
        required = ["bb_mid", "ema_21", "close"]
        if any(col not in df.columns for col in required):
            return False
        if any(pd.isna(curr[col]) for col in required):
            return False
        close = float(curr["close"])
        bb_mid = float(curr["bb_mid"])
        ema_21 = float(curr["ema_21"])
        return close < bb_mid or close < ema_21
