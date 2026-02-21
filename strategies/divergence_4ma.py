from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import pandas as pd

from bot.indicators import CandlePatternResult, wick_detachment
from bot.multi_timeframe import MultiTimeframeResult, TimeframeSnapshot
from bot.strategy import Strategy

SignalDirection = Literal["LONG", "SHORT"]


@dataclass
class EntrySignal:
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    confirmations: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    signal_timeframes: list[str] = field(default_factory=list)
    btc_state: str = "unknown"
    timestamp: pd.Timestamp | None = None
    breakeven_moved: bool = False


class Divergence4MAStrategy(Strategy):
    """Implements the Divergence + 4-MA discretionary rules as a Strategy."""

    name = "divergence_4ma"
    min_bars = 320
    entry_timeframe = "15m"
    # Divergence-first: strong (3+ TF) needs no confirmation,
    # medium (2 TF) needs 1 confirmation, weak (1 TF) is skipped.
    strong_timeframes = 3       # divergence on 3+ TF → send straight to Mike
    medium_timeframes = 2       # divergence on 2 TF → need 1 confirmation
    min_timeframes = 2          # below this → skip
    # RSI is advisory context, not a hard gate
    oversold = 32.0
    overbought = 68.0
    drop_threshold = 0.05
    btc_wick_threshold = 0.03
    stop_lookback = 12
    min_confirmations = 1
    ma_tolerance = 0.004
    detachment_threshold = 0.015

    def __init__(
        self,
        reference_symbol: str = "BTC/USDT",
        trend_timeframes: Sequence[str] | None = None,
    ):
        super().__init__()
        self.reference_symbol = reference_symbol
        self.trend_timeframes = list(trend_timeframes or ("1h", "4h", "1d"))
        self.asset_context: MultiTimeframeResult | None = None
        self.reference_context: MultiTimeframeResult | None = None
        self._latest_long_signal: EntrySignal | None = None
        self._latest_short_signal: EntrySignal | None = None
        self._active_trade: EntrySignal | None = None
        self._btc_state = "unknown"
        self._btc_crashing = False
        self._btc_emergency = False

    # ------------------------------------------------------------------
    def update_context(
        self,
        asset_context: MultiTimeframeResult,
        reference_context: MultiTimeframeResult | None = None,
    ) -> None:
        self.asset_context = asset_context
        self.reference_context = reference_context
        if reference_context is not None:
            self._btc_state = self._infer_btc_state(reference_context)
            self._btc_crashing = self._detect_btc_crash(reference_context)
            self._btc_emergency = self._detect_btc_emergency(reference_context)
        else:
            self._btc_state = "unknown"
            self._btc_crashing = False
            self._btc_emergency = False

    def should_enter(self, df: pd.DataFrame) -> bool:
        if self.asset_context is None:
            self._latest_long_signal = None
            self._latest_short_signal = None
            return False

        self._latest_long_signal = self._evaluate_long_signal()
        self._latest_short_signal = self._evaluate_short_signal()

        if self._latest_long_signal:
            self._active_trade = EntrySignal(**vars(self._latest_long_signal))
            return True
        return False

    def should_exit(self, df: pd.DataFrame) -> bool:
        if self._active_trade is None or df.empty:
            return False
        exit_now = self._should_exit_trade(df, self._active_trade)
        if exit_now:
            self._active_trade = None
        return exit_now

    # Public helpers ---------------------------------------------------
    @property
    def last_long_signal(self) -> EntrySignal | None:
        return self._latest_long_signal

    @property
    def last_short_signal(self) -> EntrySignal | None:
        return self._latest_short_signal

    def wants_short(self) -> bool:
        return self._latest_short_signal is not None

    # Signal evaluation ------------------------------------------------
    def _evaluate_long_signal(self) -> EntrySignal | None:
        entry_snapshot = self._get_snapshot(self.asset_context, self.entry_timeframe)
        if entry_snapshot is None:
            return None

        # Primary driver: divergence across timeframes
        aligned = self._divergence_timeframes("bullish")
        if len(aligned) < self.min_timeframes:
            return None

        # Determine signal strength
        strength = "strong" if len(aligned) >= self.strong_timeframes else "medium"

        # Medium signals need at least 1 confirmation
        confirmations = self._long_confirmations(entry_snapshot)
        if strength == "medium" and len(confirmations) < self.min_confirmations:
            return None

        # BTC emergency is still a hard block
        if self._btc_emergency or self._btc_crashing:
            return None

        # RSI and filters are advisory context (sent to Mike, not hard gates)
        rsi = entry_snapshot.rsi
        _, filter_notes = self._long_filters_advisory()

        entry_price = entry_snapshot.price
        stop_loss = self._calc_stop_loss(entry_snapshot, "LONG")
        take_profit = self._calc_take_profit(entry_snapshot)

        rsi_note = f"RSI={rsi:.1f}" if rsi is not None else "RSI=N/A"
        reason = " / ".join(
            [
                f"{strength} bullish divergence",
                f"aligned on {', '.join(aligned)}",
                rsi_note,
            ]
        )
        return EntrySignal(
            direction="LONG",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            confirmations=confirmations,
            filters=filter_notes,
            signal_timeframes=aligned,
            btc_state=self._btc_state,
            timestamp=entry_snapshot.latest.get("timestamp"),
        )

    def _evaluate_short_signal(self) -> EntrySignal | None:
        entry_snapshot = self._get_snapshot(self.asset_context, self.entry_timeframe)
        if entry_snapshot is None:
            return None

        # Primary driver: divergence across timeframes
        aligned = self._divergence_timeframes("bearish")
        if len(aligned) < self.min_timeframes:
            return None

        # Determine signal strength
        strength = "strong" if len(aligned) >= self.strong_timeframes else "medium"

        # Medium signals need at least 1 confirmation
        confirmations = self._short_confirmations(entry_snapshot)
        if strength == "medium" and len(confirmations) < self.min_confirmations:
            return None

        # BTC emergency is still a hard block
        if self._btc_emergency:
            return None

        # RSI and filters are advisory context (sent to Mike, not hard gates)
        rsi = entry_snapshot.rsi
        _, filter_notes = self._short_filters_advisory()

        entry_price = entry_snapshot.price
        stop_loss = self._calc_stop_loss(entry_snapshot, "SHORT")
        take_profit = self._calc_take_profit(entry_snapshot)

        rsi_note = f"RSI={rsi:.1f}" if rsi is not None else "RSI=N/A"
        reason = " / ".join(
            [
                f"{strength} bearish divergence",
                f"aligned on {', '.join(aligned)}",
                rsi_note,
            ]
        )
        return EntrySignal(
            direction="SHORT",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            confirmations=confirmations,
            filters=filter_notes,
            signal_timeframes=aligned,
            btc_state=self._btc_state,
            timestamp=entry_snapshot.latest.get("timestamp"),
        )

    # Confirmations ----------------------------------------------------
    def _long_confirmations(self, snapshot: TimeframeSnapshot) -> list[str]:
        confirmations: list[str] = []
        pattern: CandlePatternResult = snapshot.candle_pattern
        if pattern.hammer:
            confirmations.append("hammer")
        if pattern.wick_retraction and pattern.bias == "bullish":
            confirmations.append("wick_retraction")
        if self._ma_bounce(snapshot, (21, 100)):
            confirmations.append("ma_bounce")
        return confirmations

    def _short_confirmations(self, snapshot: TimeframeSnapshot) -> list[str]:
        confirmations: list[str] = []
        if wick_detachment(snapshot.df, periods=3, ma_column="sma_9", threshold=self.detachment_threshold):
            confirmations.append("ma9_detachment")
        trend_snapshot = self._get_snapshot(self.asset_context, "4h")
        if trend_snapshot and trend_snapshot.ma_order == "bearish":
            confirmations.append("ma_bearish")
        if self._loss_of_momentum(snapshot.df):
            confirmations.append("momentum_loss")
        return confirmations

    # Filters (advisory — passed to Mike as context, not hard gates) ----
    def _long_filters_advisory(self) -> tuple[bool, list[str]]:
        notes: list[str] = []
        notes.append(f"btc_state={self._btc_state}")
        if self._btc_crashing:
            notes.append("⚠️ btc_crashing")
        ma_ok = self._ma_filter("bullish")
        notes.append("4h_ma_bullish" if ma_ok else "⚠️ 4h_ma_not_bullish")
        drop_ok = self._daily_drop_ok()
        notes.append("daily_drop_ok" if drop_ok else "daily_drop_missing")
        all_ok = ma_ok and drop_ok and not self._btc_crashing
        return all_ok, notes

    def _short_filters_advisory(self) -> tuple[bool, list[str]]:
        notes: list[str] = []
        notes.append(f"btc_state={self._btc_state}")
        if self._btc_state not in {"bearish", "sideways"}:
            notes.append("⚠️ btc_not_bearish")
        ma_ok = self._ma_filter("bearish")
        notes.append("4h_ma_bearish" if ma_ok else "⚠️ 4h_ma_not_bearish")
        all_ok = ma_ok and self._btc_state in {"bearish", "sideways"}
        return all_ok, notes

    # Legacy aliases for backward compatibility
    _long_filters = _long_filters_advisory
    _short_filters = _short_filters_advisory

    def _ma_filter(self, desired: Literal["bullish", "bearish"]) -> bool:
        snapshot = self._get_snapshot(self.asset_context, "4h")
        if snapshot is None:
            return False
        order = snapshot.ma_order
        price = snapshot.price
        trend_ma = snapshot.latest.get("sma_21")
        if trend_ma is None or pd.isna(trend_ma):
            return False
        if desired == "bullish":
            return order in {"bullish", "crossing"} and price >= float(trend_ma)
        return order == "bearish" and price <= float(trend_ma)

    def _daily_drop_ok(self) -> bool:
        snapshot = self._get_snapshot(self.asset_context, "1d")
        if snapshot is None:
            return False
        df = snapshot.df
        if len(df) < 2:
            return False
        prev_close = float(df.iloc[-2]["close"])
        latest_close = float(df.iloc[-1]["close"])
        if prev_close <= 0:
            return False
        drop = (latest_close - prev_close) / prev_close
        return drop <= -self.drop_threshold

    # Exit logic -------------------------------------------------------
    def _should_exit_trade(self, df: pd.DataFrame, trade: EntrySignal) -> bool:
        latest = df.iloc[-1]
        price = float(latest["close"])
        ma100 = latest.get("sma_100")
        if self._btc_emergency:
            return True
        if trade.direction == "LONG":
            self._maybe_move_breakeven(trade, price)
            if price <= trade.stop_loss:
                return True
            if ma100 is not None and not pd.isna(ma100) and price >= float(ma100):
                return True
        else:
            self._maybe_move_breakeven(trade, price)
            if price >= trade.stop_loss:
                return True
            if ma100 is not None and not pd.isna(ma100) and price <= float(ma100):
                return True
        return False

    def _maybe_move_breakeven(self, trade: EntrySignal, price: float) -> None:
        if trade.breakeven_moved:
            return
        if trade.direction == "LONG":
            risk = trade.entry_price - trade.stop_loss
            if risk <= 0:
                return
            if price - trade.entry_price >= risk:
                trade.stop_loss = trade.entry_price
                trade.breakeven_moved = True
        else:
            risk = trade.stop_loss - trade.entry_price
            if risk <= 0:
                return
            if trade.entry_price - price >= risk:
                trade.stop_loss = trade.entry_price
                trade.breakeven_moved = True

    # Utility helpers --------------------------------------------------
    def _get_snapshot(
        self, result: MultiTimeframeResult | None, timeframe: str
    ) -> TimeframeSnapshot | None:
        if result is None:
            return None
        return result.snapshots.get(timeframe)

    def _has_divergence(self, snapshot: TimeframeSnapshot, kind: Literal["bullish", "bearish"]) -> bool:
        return any(sig.kind == kind for sig in snapshot.divergences)

    def _divergence_timeframes(self, kind: Literal["bullish", "bearish"]) -> list[str]:
        if self.asset_context is None:
            return []
        return list(self.asset_context.divergence_alignment.get(kind, []))

    def _calc_stop_loss(self, snapshot: TimeframeSnapshot, direction: SignalDirection) -> float:
        window = snapshot.df.tail(self.stop_lookback)
        if direction == "LONG":
            return float(window["low"].min())
        return float(window["high"].max())

    def _calc_take_profit(self, snapshot: TimeframeSnapshot) -> float:
        ma_value = snapshot.latest.get("sma_100")
        if ma_value is None or pd.isna(ma_value):
            return snapshot.price
        return float(ma_value)

    def _ma_bounce(self, snapshot: TimeframeSnapshot, periods: Sequence[int]) -> bool:
        latest = snapshot.latest
        low = float(latest["low"])
        close = float(latest["close"])
        for period in periods:
            key = f"sma_{period}"
            ma_value = latest.get(key)
            if ma_value is None or pd.isna(ma_value):
                continue
            ma_value = float(ma_value)
            touched = (low <= ma_value <= close) or abs(close - ma_value) / max(ma_value, 1e-8) <= self.ma_tolerance
            if touched:
                return True
        return False

    def _loss_of_momentum(self, df: pd.DataFrame) -> bool:
        if len(df) < 6:
            return False
        bodies = (df["close"] - df["open"]).abs()
        recent = bodies.iloc[-3:]
        prev = bodies.iloc[-6:-3]
        if prev.mean() == 0:
            return False
        return recent.mean() <= prev.mean() * 0.7

    def _infer_btc_state(self, context: MultiTimeframeResult) -> str:
        hour = self._get_snapshot(context, "1h")
        four_hour = self._get_snapshot(context, "4h")
        if hour and four_hour:
            if hour.ma_order in {"bullish", "crossing"} and four_hour.ma_order in {"bullish", "crossing"}:
                return "bullish"
            if hour.ma_order == "bearish" and four_hour.ma_order == "bearish":
                return "bearish"
        return "sideways"

    def _detect_btc_crash(self, context: MultiTimeframeResult) -> bool:
        snapshot = self._get_snapshot(context, self.entry_timeframe)
        if snapshot is None:
            return False
        df = snapshot.df
        if len(df) < 2:
            return False
        prev_close = float(df.iloc[-2]["close"])
        latest_close = float(df.iloc[-1]["close"])
        if prev_close <= 0:
            return False
        change = (latest_close - prev_close) / prev_close
        wick = self._wick_percentage(snapshot)
        return change <= -0.02 or wick >= 0.025

    def _detect_btc_emergency(self, context: MultiTimeframeResult) -> bool:
        snapshot = self._get_snapshot(context, self.entry_timeframe)
        if snapshot is None:
            return False
        return self._wick_percentage(snapshot) >= self.btc_wick_threshold

    def _wick_percentage(self, snapshot: TimeframeSnapshot) -> float:
        latest = snapshot.latest
        open_price = float(latest["open"])
        close_price = float(latest["close"])
        high_price = float(latest["high"])
        low_price = float(latest["low"])
        upper = max(0.0, high_price - max(open_price, close_price))
        lower = max(0.0, min(open_price, close_price) - low_price)
        wick = max(upper, lower)
        reference = max(close_price, 1e-8)
        return wick / reference


__all__ = ["Divergence4MAStrategy", "EntrySignal"]
