from __future__ import annotations
import pandas as pd
from bot.strategy import Strategy


class RsiOversoldStrategy(Strategy):
    name = "rsi_oversold"
    stop_atr_mult = 1.5
    trail_atr_mult = 2.0
    risk_reward = 2.2
    partial_profit_rr = 1.0
    partial_exit_pct = 0.5
    max_hold_bars = 36

    def should_enter(self, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        latest = df.iloc[-1]
        required = [
            "rsi",
        ]
        if any(col not in df.columns for col in required):
            return False
        if any(pd.isna(latest[col]) for col in required):
            return False
        rsi_14 = float(latest["rsi"])
        return rsi_14 < 35

    def should_exit(self, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        latest = df.iloc[-1]
        if any(col not in df.columns for col in ["rsi_10", "bb_mid", "close"]):
            return False
        if any(pd.isna(latest[col]) for col in ["rsi_10", "bb_mid", "close"]):
            return False
        rsi_10 = float(latest["rsi_10"])
        close = float(latest["close"])
        bb_mid = float(latest["bb_mid"])
        return rsi_10 > 55 or close > bb_mid
