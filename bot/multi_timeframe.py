from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import pandas as pd

from bot.indicators import (
    MA_PERIODS_DEFAULT,
    CandlePatternResult,
    DivergenceSignal,
    add_indicators,
    detect_candle_patterns,
    detect_rsi_divergences,
    evaluate_ma_order,
)


@dataclass
class TimeframeSnapshot:
    timeframe: str
    df: pd.DataFrame
    latest: pd.Series
    ma_order: str
    divergences: list[DivergenceSignal]
    candle_pattern: CandlePatternResult

    @property
    def price(self) -> float:
        return float(self.latest.get("close", 0.0))

    @property
    def rsi(self) -> float | None:
        value = self.latest.get("rsi")
        if value is None or pd.isna(value):
            return None
        return float(value)


@dataclass
class MultiTimeframeResult:
    symbol: str
    snapshots: dict[str, TimeframeSnapshot]
    divergence_alignment: dict[str, list[str]]

    def count_divergences(self, kind: Literal["bullish", "bearish"]) -> int:
        return len(self.divergence_alignment.get(kind, []))


class MultiTimeframeAnalyzer:
    def __init__(
        self,
        exchange: Any,
        timeframes: Sequence[str] | None = None,
        ma_periods: Sequence[int] | None = None,
        limit: int = 320,
    ):
        self.exchange = exchange
        self.timeframes = list(timeframes or ["15m", "1h", "4h", "1d"])
        self.ma_periods = tuple(ma_periods or MA_PERIODS_DEFAULT)
        self.limit = limit

    def analyze(self, symbol: str) -> MultiTimeframeResult:
        snapshots: dict[str, TimeframeSnapshot] = {}
        for timeframe in self.timeframes:
            df = self._fetch(symbol, timeframe)
            if df.empty:
                continue
            latest = df.iloc[-1]
            divergences = detect_rsi_divergences(df)
            ma_order = evaluate_ma_order(latest, periods=self.ma_periods)
            candle_pattern = detect_candle_patterns(df)
            snapshots[timeframe] = TimeframeSnapshot(
                timeframe=timeframe,
                df=df,
                latest=latest,
                ma_order=ma_order,
                divergences=divergences,
                candle_pattern=candle_pattern,
            )
        alignment = self._align_divergences(snapshots)
        return MultiTimeframeResult(symbol=symbol, snapshots=snapshots, divergence_alignment=alignment)

    # ------------------------------------------------------------------
    def _fetch(self, symbol: str, timeframe: str) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=self.limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return add_indicators(df, ma_periods=self.ma_periods)

    def _align_divergences(self, snapshots: dict[str, TimeframeSnapshot]) -> dict[str, list[str]]:
        alignment = {"bullish": [], "bearish": []}
        for timeframe, snapshot in snapshots.items():
            if any(sig.kind == "bullish" for sig in snapshot.divergences):
                alignment["bullish"].append(timeframe)
            if any(sig.kind == "bearish" for sig in snapshot.divergences):
                alignment["bearish"].append(timeframe)
        return alignment
