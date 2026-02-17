from __future__ import annotations
import pandas as pd

from bot.strategy import Strategy


class MacdCrossoverStrategy(Strategy):
    name = "macd_crossover"
    stop_atr_mult = 1.6
    trail_atr_mult = 2.1
    risk_reward = 2.2
    partial_profit_rr = 1.0
    partial_exit_pct = 0.5
    max_hold_bars = 48

    def should_enter(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        required = [
            "macd",
            "macd_signal",
            "ema_50",
            "adx_14",
            "close",
        ]
        if any(col not in df.columns for col in required):
            return False
        if any(pd.isna(curr[col]) for col in required) or any(pd.isna(prev[col]) for col in ["macd", "macd_signal"]):
            return False
        macd_cross = float(prev["macd"]) <= float(prev["macd_signal"]) and float(curr["macd"]) > float(curr["macd_signal"])
        close = float(curr["close"])
        ema_50 = float(curr["ema_50"])
        adx_14 = float(curr["adx_14"])
        trend_up = close > ema_50
        return macd_cross and trend_up and adx_14 > 15

    def should_exit(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        required = ["macd", "macd_signal", "ema_21", "close"]
        if any(col not in df.columns for col in required):
            return False
        if any(pd.isna(curr[col]) for col in required) or any(pd.isna(prev[col]) for col in ["macd", "macd_signal"]):
            return False
        macd_cross_down = float(prev["macd"]) >= float(prev["macd_signal"]) and float(curr["macd"]) < float(curr["macd_signal"])
        close = float(curr["close"])
        ema_21 = float(curr["ema_21"])
        return macd_cross_down or close < ema_21
