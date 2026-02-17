from __future__ import annotations
import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["rsi_10"] = ta.momentum.RSIIndicator(close=df["close"], window=10).rsi()
    df["ema_9"] = ta.trend.EMAIndicator(close=df["close"], window=9).ema_indicator()
    df["ema_12"] = ta.trend.EMAIndicator(close=df["close"], window=12).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(close=df["close"], window=26).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(close=df["close"], window=200).ema_indicator()

    macd = ta.trend.MACD(close=df["close"], window_fast=8, window_slow=21, window_sign=5)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2.2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_low"] = bb.bollinger_lband()

    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx_14"] = adx.adx()

    atr = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr.average_true_range()

    obv = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    df["obv"] = obv
    df["obv_ema_20"] = obv.ewm(span=20, adjust=False).mean()
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]
    df["volume_ratio"] = df["volume_ratio"].replace([float("inf"), -float("inf")], pd.NA)
    return df
