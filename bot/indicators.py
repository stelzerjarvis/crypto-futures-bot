from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import ta

MA_PERIODS_DEFAULT: tuple[int, ...] = (9, 21, 45, 100)


@dataclass
class DivergenceSignal:
    kind: Literal["bullish", "bearish"]
    pivot_a_idx: int
    pivot_b_idx: int
    price_points: tuple[float, float]
    rsi_points: tuple[float, float]
    strength: float


@dataclass
class CandlePatternResult:
    hammer: bool
    wick_retraction: bool
    bias: Literal["bullish", "bearish", "neutral"]


def add_indicators(df: pd.DataFrame, ma_periods: Sequence[int] | None = None) -> pd.DataFrame:
    """Append legacy indicators plus Divergence-4MA specific signals."""

    df = df.copy()
    ma_periods = tuple(ma_periods or MA_PERIODS_DEFAULT)

    # Core indicators kept for backward compatibility
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
    df["volume_ratio"] = (df["volume"] / df["volume_ma_20"]).replace([float("inf"), -float("inf")], pd.NA)

    # Divergence-4MA additions -------------------------------------------------
    for period in ma_periods:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()

    return df


def evaluate_ma_order(row: pd.Series, periods: Sequence[int] | None = None) -> str:
    periods = tuple(periods or MA_PERIODS_DEFAULT)
    values: list[float] = []
    for period in periods:
        key = f"sma_{period}"
        value = row.get(key)
        if pd.isna(value):
            return "unknown"
        values.append(float(value))
    decreasing = all(values[idx] >= values[idx + 1] for idx in range(len(values) - 1))
    increasing = all(values[idx] <= values[idx + 1] for idx in range(len(values) - 1))
    if decreasing:
        return "bullish"
    if increasing:
        return "bearish"
    return "crossing"


def detect_rsi_divergences(
    df: pd.DataFrame,
    lookback: int = 160,
    pivot_window: int = 3,
) -> list[DivergenceSignal]:
    if df.empty or "rsi" not in df.columns:
        return []
    rsi = df["rsi"].astype(float)
    closes = df["close"].astype(float)
    lows = df["low"].astype(float)
    highs = df["high"].astype(float)
    signals: list[DivergenceSignal] = []

    pivot_lows = _pivot_points(lows.values, pivot_window)
    pivot_highs = _pivot_points(highs.values, pivot_window, kind="high")
    cutoff = max(0, len(df) - lookback)
    pivot_lows = [idx for idx in pivot_lows if idx >= cutoff]
    pivot_highs = [idx for idx in pivot_highs if idx >= cutoff]

    if len(pivot_lows) >= 2:
        a, b = pivot_lows[-2], pivot_lows[-1]
        price_a, price_b = lows.iloc[a], lows.iloc[b]
        rsi_a, rsi_b = rsi.iloc[a], rsi.iloc[b]
        if price_b < price_a and rsi_b > rsi_a:
            strength = _divergence_strength(price_a, price_b, rsi_a, rsi_b)
            signals.append(
                DivergenceSignal(
                    kind="bullish",
                    pivot_a_idx=int(a),
                    pivot_b_idx=int(b),
                    price_points=(float(price_a), float(price_b)),
                    rsi_points=(float(rsi_a), float(rsi_b)),
                    strength=strength,
                )
            )

    if len(pivot_highs) >= 2:
        a, b = pivot_highs[-2], pivot_highs[-1]
        price_a, price_b = highs.iloc[a], highs.iloc[b]
        rsi_a, rsi_b = rsi.iloc[a], rsi.iloc[b]
        if price_b > price_a and rsi_b < rsi_a:
            strength = _divergence_strength(price_a, price_b, rsi_a, rsi_b)
            signals.append(
                DivergenceSignal(
                    kind="bearish",
                    pivot_a_idx=int(a),
                    pivot_b_idx=int(b),
                    price_points=(float(price_a), float(price_b)),
                    rsi_points=(float(rsi_a), float(rsi_b)),
                    strength=strength,
                )
            )

    return signals


def detect_candle_patterns(df: pd.DataFrame) -> CandlePatternResult:
    if df.empty:
        return CandlePatternResult(False, False, "neutral")
    latest = df.iloc[-1]
    open_price = float(latest["open"])
    close_price = float(latest["close"])
    high_price = float(latest["high"])
    low_price = float(latest["low"])
    body = abs(close_price - open_price)
    upper_wick = max(0.0, high_price - max(open_price, close_price))
    lower_wick = max(0.0, min(open_price, close_price) - low_price)
    average_range = max(1e-8, high_price - low_price)
    hammer = bool(lower_wick >= 2 * body and upper_wick <= body and (close_price > open_price))
    wick_ratio = max(upper_wick, lower_wick) / max(body, 1e-8)
    wick_retraction = bool(wick_ratio >= 2 and (body / average_range) <= 0.4)
    bias: Literal["bullish", "bearish", "neutral"] = "neutral"
    if wick_retraction:
        bias = "bullish" if lower_wick > upper_wick else "bearish"
    elif hammer:
        bias = "bullish"
    return CandlePatternResult(hammer=hammer, wick_retraction=wick_retraction, bias=bias)


def wick_detachment(df: pd.DataFrame, periods: int = 3, ma_column: str = "sma_9", threshold: float = 0.01) -> bool:
    if df.empty or ma_column not in df.columns:
        return False
    tail = df.tail(periods)
    ma = tail[ma_column]
    if ma.isna().any():
        return False
    deviation = (tail["close"] - ma).abs() / ma
    return bool((deviation >= threshold).all())


def _divergence_strength(price_a: float, price_b: float, rsi_a: float, rsi_b: float) -> float:
    price_delta = abs(price_b - price_a) / max(1e-8, abs(price_a))
    rsi_delta = abs(rsi_b - rsi_a) / max(1.0, abs(rsi_a))
    return round(float(rsi_delta - price_delta), 4)


def _pivot_points(values: Iterable[float], window: int, kind: str = "low") -> list[int]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < (window * 2 + 1):
        return []
    pivots: list[int] = []
    for idx in range(window, len(arr) - window):
        segment = arr[idx - window : idx + window + 1]
        center = arr[idx]
        if kind == "high":
            if center == segment.max() and center > segment[:-1].max() and center >= segment[1:].max():
                pivots.append(idx)
        else:
            if center == segment.min() and center < segment[:-1].min() and center <= segment[1:].min():
                pivots.append(idx)
    return pivots
