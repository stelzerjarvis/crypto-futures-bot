from __future__ import annotations
import pandas as pd

from bot.strategy import Strategy


class MacdCrossoverStrategy(Strategy):
    name = "macd_crossover"

    def should_enter(self, df: pd.DataFrame) -> bool:
        if len(df) < 2 or "macd" not in df.columns or "macd_signal" not in df.columns:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        return float(prev["macd"]) <= float(prev["macd_signal"]) and float(curr["macd"]) > float(curr["macd_signal"])

    def should_exit(self, df: pd.DataFrame) -> bool:
        if len(df) < 2 or "macd" not in df.columns or "macd_signal" not in df.columns:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        return float(prev["macd"]) >= float(prev["macd_signal"]) and float(curr["macd"]) < float(curr["macd_signal"])
