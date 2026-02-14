import pandas as pd
from bot.strategy import Strategy


class RsiOversoldStrategy(Strategy):
    name = "rsi_oversold"

    def should_enter(self, df: pd.DataFrame) -> bool:
        if df.empty or "rsi" not in df.columns:
            return False
        latest_rsi = float(df.iloc[-1]["rsi"])
        return latest_rsi < 30

    def should_exit(self, df: pd.DataFrame) -> bool:
        if df.empty or "rsi" not in df.columns:
            return False
        latest_rsi = float(df.iloc[-1]["rsi"])
        return latest_rsi > 70
