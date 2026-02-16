from __future__ import annotations
import pandas as pd

from bot.strategy import Strategy


class BollingerBreakoutStrategy(Strategy):
    name = "bollinger_breakout"

    def should_enter(self, df: pd.DataFrame) -> bool:
        if len(df) < 2 or "bb_high" not in df.columns or "close" not in df.columns:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        return float(prev["close"]) <= float(prev["bb_high"]) and float(curr["close"]) > float(curr["bb_high"])

    def should_exit(self, df: pd.DataFrame) -> bool:
        if len(df) < 2 or "bb_mid" not in df.columns or "close" not in df.columns:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        return float(prev["close"]) >= float(prev["bb_mid"]) and float(curr["close"]) < float(curr["bb_mid"])
