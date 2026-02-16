from __future__ import annotations
import pandas as pd

from bot.strategy import Strategy


class EmaCrossoverStrategy(Strategy):
    name = "ema_crossover"

    def should_enter(self, df: pd.DataFrame) -> bool:
        if len(df) < 2 or "ema_12" not in df.columns or "ema_26" not in df.columns:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        return float(prev["ema_12"]) <= float(prev["ema_26"]) and float(curr["ema_12"]) > float(curr["ema_26"])

    def should_exit(self, df: pd.DataFrame) -> bool:
        if len(df) < 2 or "ema_12" not in df.columns or "ema_26" not in df.columns:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        return float(prev["ema_12"]) >= float(prev["ema_26"]) and float(curr["ema_12"]) < float(curr["ema_26"])
