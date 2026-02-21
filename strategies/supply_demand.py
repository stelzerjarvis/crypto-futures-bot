from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import pandas as pd

from bot.multi_timeframe import MultiTimeframeResult, TimeframeSnapshot
from bot.strategy import Strategy
from strategies.divergence_4ma import EntrySignal

SignalDirection = Literal["LONG", "SHORT"]
TrendDirection = Literal["uptrend", "downtrend", "range"]
ZoneKind = Literal["demand", "supply"]


def _to_float(value: object | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class SwingPoint:
    idx: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["high", "low"]


@dataclass
class StructureState:
    trend: TrendDirection
    last_valid_low: SwingPoint | None = None
    last_valid_high: SwingPoint | None = None


@dataclass
class Zone:
    kind: ZoneKind
    lower: float
    upper: float
    base_idx: int
    impulse_idx: int
    created_at: pd.Timestamp

    def bounds(self) -> tuple[float, float]:
        return (self.lower, self.upper)


class SupplyDemandStrategy(Strategy):
    """Charlie — Price Action Supply & Demand strategy (1h primary timeframe)."""

    name = "charlie_strategy"
    min_bars = 200
    entry_timeframe = "1h"
    higher_timeframe = "4h"

    fractal_size = 3
    consolidation_min_bars = 3
    consolidation_range_mult = 1.0
    impulse_body_mult = 1.5
    stop_buffer_pct = 0.001
    min_risk_reward = 2.5

    btc_wick_threshold = 0.03
    btc_volatility_threshold = 0.03

    def __init__(self, reference_symbol: str = "BTC/USDT") -> None:
        super().__init__()
        self.reference_symbol = reference_symbol
        self.asset_context: MultiTimeframeResult | None = None
        self.reference_context: MultiTimeframeResult | None = None

        self._latest_long_signal: EntrySignal | None = None
        self._latest_short_signal: EntrySignal | None = None
        self._active_trade: EntrySignal | None = None
        self._last_structure: StructureState | None = None
        self._consumed_zones: set[str] = set()

        self._btc_state = "unknown"
        self._btc_emergency = False
        self._btc_high_vol = False

    # ------------------------------------------------------------------
    def update_context(
        self,
        asset_context: MultiTimeframeResult,
        reference_context: MultiTimeframeResult | None = None,
    ) -> None:
        self.asset_context = asset_context
        self.reference_context = reference_context
        self._update_btc_filters(reference_context)

    def should_enter(self, df: pd.DataFrame) -> bool:
        self._latest_long_signal = None
        self._latest_short_signal = None

        if self.asset_context is None:
            return False
        if self._btc_emergency or self._btc_high_vol:
            return False

        snapshot = self._get_snapshot(self.asset_context, self.entry_timeframe)
        if snapshot is None:
            return False

        entry_df = df if not df.empty else snapshot.df
        if len(entry_df) < self.min_bars:
            return False

        swings = self._detect_swings(entry_df, self.fractal_size)
        structure = self._determine_structure(entry_df, swings)
        self._last_structure = structure
        if structure.trend not in {"uptrend", "downtrend"}:
            return False

        direction: SignalDirection = "LONG" if structure.trend == "uptrend" else "SHORT"
        if not self._trend_alignment_ok(direction):
            return False

        signal = self._build_signal(entry_df, swings, structure, direction)
        if signal is None:
            return False

        if signal.direction == "LONG":
            self._latest_long_signal = signal
        else:
            self._latest_short_signal = signal

        self._active_trade = replace(signal)
        return True

    def should_exit(self, df: pd.DataFrame) -> bool:
        if self._active_trade is None:
            return False

        snapshot_df = df if not df.empty else self._get_snapshot_df()
        if snapshot_df is None or snapshot_df.empty:
            return False

        trade = self._active_trade
        latest = snapshot_df.iloc[-1]
        price_high = float(latest["high"])
        price_low = float(latest["low"])
        close_price = float(latest["close"])

        # Emergency BTC action overrides everything
        if self._btc_emergency:
            self._active_trade = None
            return True

        self._maybe_move_breakeven(trade, close_price)

        if trade.direction == "LONG":
            if price_low <= trade.stop_loss:
                self._active_trade = None
                return True
            if price_high >= trade.take_profit:
                self._active_trade = None
                return True
        else:
            if price_high >= trade.stop_loss:
                self._active_trade = None
                return True
            if price_low <= trade.take_profit:
                self._active_trade = None
                return True

        swings = self._detect_swings(snapshot_df, self.fractal_size)
        structure = self._determine_structure(snapshot_df, swings)
        if structure.trend != "range":
            self._last_structure = structure

        if self._trend_flip_detected(snapshot_df, trade.direction, structure):
            self._active_trade = None
            return True

        return False

    # ------------------------------------------------------------------
    @property
    def last_long_signal(self) -> EntrySignal | None:
        return self._latest_long_signal

    @property
    def last_short_signal(self) -> EntrySignal | None:
        return self._latest_short_signal

    def wants_short(self) -> bool:
        return self._latest_short_signal is not None

    # ------------------------------------------------------------------
    def _build_signal(
        self,
        df: pd.DataFrame,
        swings: Sequence[SwingPoint],
        structure: StructureState,
        direction: SignalDirection,
    ) -> EntrySignal | None:
        kind: ZoneKind = "demand" if direction == "LONG" else "supply"
        zones = self._identify_zones(df, kind)
        if not zones:
            return None

        latest_idx = len(df) - 1
        latest_ts = df.iloc[-1]["timestamp"]
        for zone in zones:
            zone_key = self._zone_key(zone)
            if zone_key in self._consumed_zones:
                continue

            retest_idx = self._zone_retest_index(df, zone)
            if retest_idx is None:
                continue
            if retest_idx < 0:
                self._consumed_zones.add(zone_key)
                continue

            self._consumed_zones.add(zone_key)
            if retest_idx != latest_idx:
                continue

            plan = self._build_trade_plan(df, swings, zone, direction)
            if plan is None:
                continue

            entry, stop_loss, take_profit, rr = plan
            if rr < self.min_risk_reward:
                continue

            confirmations = self._confirmations(structure, zone, rr, direction, zone_key)
            filters = self._filters(rr)
            reason = self._signal_reason(direction, rr)

            return EntrySignal(
                direction=direction,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                confirmations=confirmations,
                filters=filters,
                signal_timeframes=[self.entry_timeframe],
                btc_state=self._btc_state,
                timestamp=latest_ts,
            )

        return None

    def _build_trade_plan(
        self,
        df: pd.DataFrame,
        swings: Sequence[SwingPoint],
        zone: Zone,
        direction: SignalDirection,
    ) -> tuple[float, float, float, float] | None:
        entry = zone.upper if direction == "LONG" else zone.lower
        buffer = entry * self.stop_buffer_pct
        if direction == "LONG":
            stop_loss = zone.lower - buffer
            target = self._recent_swing_price(swings, "high", len(df) - 1)
            if target is None or target <= entry:
                return None
            reward = target - entry
            risk = entry - stop_loss
        else:
            stop_loss = zone.upper + buffer
            target = self._recent_swing_price(swings, "low", len(df) - 1)
            if target is None or target >= entry:
                return None
            reward = entry - target
            risk = stop_loss - entry

        if risk <= 0 or reward <= 0:
            return None
        rr = reward / risk
        return entry, stop_loss, target, rr

    # ------------------------------------------------------------------
    def _detect_swings(self, df: pd.DataFrame, n: int) -> list[SwingPoint]:
        swings: list[SwingPoint] = []
        if len(df) < 2 * n + 1:
            return swings
        for idx in range(n, len(df) - n):
            row = df.iloc[idx]
            prev_window = df.iloc[idx - n : idx]
            next_window = df.iloc[idx + 1 : idx + 1 + n]
            if prev_window.empty or next_window.empty:
                continue
            high = float(row["high"])
            low = float(row["low"])
            if high > float(prev_window["high"].max()) and high > float(next_window["high"].max()):
                swings.append(
                    SwingPoint(
                        idx=idx,
                        timestamp=row["timestamp"],
                        price=high,
                        kind="high",
                    )
                )
            if low < float(prev_window["low"].min()) and low < float(next_window["low"].min()):
                swings.append(
                    SwingPoint(
                        idx=idx,
                        timestamp=row["timestamp"],
                        price=low,
                        kind="low",
                    )
                )
        swings.sort(key=lambda s: s.idx)
        return swings

    def _determine_structure(self, df: pd.DataFrame, swings: Sequence[SwingPoint]) -> StructureState:
        last_valid_low: SwingPoint | None = None
        last_valid_high: SwingPoint | None = None
        last_swing_high: SwingPoint | None = None
        last_swing_low: SwingPoint | None = None
        pending_low: tuple[SwingPoint, SwingPoint | None] | None = None
        pending_high: tuple[SwingPoint, SwingPoint | None] | None = None

        for swing in swings:
            if swing.kind == "low":
                if pending_high is not None:
                    high_point, reference_low = pending_high
                    if reference_low is not None and swing.price < reference_low.price:
                        last_valid_high = high_point
                        pending_high = None
                pending_low = (swing, last_swing_high)
                last_swing_low = swing
            else:
                if pending_low is not None:
                    low_point, reference_high = pending_low
                    if reference_high is not None and swing.price > reference_high.price:
                        last_valid_low = low_point
                        pending_low = None
                pending_high = (swing, last_swing_low)
                last_swing_high = swing

        latest_close = float(df.iloc[-1]["close"])
        trend: TrendDirection = "range"

        if last_valid_low and (not last_valid_high or last_valid_low.idx >= last_valid_high.idx):
            trend = "uptrend"
            if latest_close < last_valid_low.price:
                trend = "downtrend"
        elif last_valid_high:
            trend = "downtrend"
            if latest_close > last_valid_high.price:
                trend = "uptrend"

        return StructureState(trend=trend, last_valid_low=last_valid_low, last_valid_high=last_valid_high)

    # ------------------------------------------------------------------
    def _identify_zones(self, df: pd.DataFrame, kind: ZoneKind) -> list[Zone]:
        zones: list[Zone] = []
        min_bars = self.consolidation_min_bars
        if len(df) <= min_bars + 1:
            return zones

        for idx in range(min_bars, len(df) - 1):
            impulse_idx = idx + 1
            impulse_row = df.iloc[impulse_idx]
            if not self._is_impulsive(impulse_row, kind):
                continue
            window = df.iloc[idx - min_bars + 1 : idx + 1]
            if len(window) < min_bars:
                continue
            if not self._is_consolidation(window):
                continue
            base_candle = window.iloc[-1]
            lower = float(base_candle["low"])
            upper = float(base_candle["high"])
            zones.append(
                Zone(
                    kind=kind,
                    lower=min(lower, upper),
                    upper=max(lower, upper),
                    base_idx=idx,
                    impulse_idx=impulse_idx,
                    created_at=impulse_row["timestamp"],
                )
            )
        return zones

    def _is_consolidation(self, window: pd.DataFrame) -> bool:
        if window.empty:
            return False
        atr = _to_float(window.iloc[-1].get("atr"))
        if atr <= 0:
            return False
        window_range = float(window["high"].max()) - float(window["low"].min())
        return window_range <= atr * self.consolidation_range_mult

    def _is_impulsive(self, row: pd.Series, kind: ZoneKind) -> bool:
        atr = _to_float(row.get("atr"))
        if atr <= 0:
            return False
        open_price = float(row["open"])
        close_price = float(row["close"])
        body = abs(close_price - open_price)
        if body < atr * self.impulse_body_mult:
            return False
        return close_price > open_price if kind == "demand" else close_price < open_price

    def _zone_retest_index(self, df: pd.DataFrame, zone: Zone) -> int | None:
        for idx in range(zone.impulse_idx + 1, len(df)):
            row = df.iloc[idx]
            high = float(row["high"])
            low = float(row["low"])
            close_price = float(row["close"])
            touched = (low <= zone.upper) and (high >= zone.lower)
            if touched:
                if zone.kind == "demand" and close_price < zone.lower:
                    return -1
                if zone.kind == "supply" and close_price > zone.upper:
                    return -1
                return idx
            if zone.kind == "demand" and close_price < zone.lower:
                return -1
            if zone.kind == "supply" and close_price > zone.upper:
                return -1
        return None

    def _recent_swing_price(
        self,
        swings: Sequence[SwingPoint],
        kind: Literal["high", "low"],
        before_idx: int,
    ) -> float | None:
        for swing in reversed(swings):
            if swing.kind != kind:
                continue
            if swing.idx < before_idx:
                return swing.price
        return None

    # ------------------------------------------------------------------
    def _confirmations(
        self,
        structure: StructureState,
        zone: Zone,
        rr: float,
        direction: SignalDirection,
        zone_key: str,
    ) -> list[str]:
        confirmations = [f"trend={structure.trend}", f"zone_kind={zone.kind}", f"rr={rr:.2f}", zone_key]
        if direction == "LONG" and structure.last_valid_low:
            confirmations.append(f"valid_low@{structure.last_valid_low.price:.4f}")
        if direction == "SHORT" and structure.last_valid_high:
            confirmations.append(f"valid_high@{structure.last_valid_high.price:.4f}")
        return confirmations

    def _filters(self, rr: float) -> list[str]:
        filters = [
            f"btc_state={self._btc_state}",
            f"btc_emergency={self._btc_emergency}",
            f"btc_volatility={self._btc_high_vol}",
            f"rr_ok={rr >= self.min_risk_reward}",
        ]
        return filters

    def _signal_reason(self, direction: SignalDirection, rr: float) -> str:
        zone_label = "demand" if direction == "LONG" else "supply"
        return f"{direction} retest of {zone_label} zone / R:R={rr:.2f}"

    def _trend_flip_detected(
        self,
        df: pd.DataFrame,
        direction: SignalDirection,
        structure: StructureState,
    ) -> bool:
        reference = structure if structure.trend != "range" else self._last_structure
        if reference is None:
            return False
        latest_close = float(df.iloc[-1]["close"])
        if direction == "LONG" and reference.last_valid_low is not None:
            return latest_close < reference.last_valid_low.price
        if direction == "SHORT" and reference.last_valid_high is not None:
            return latest_close > reference.last_valid_high.price
        return False

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _trend_alignment_ok(self, direction: SignalDirection) -> bool:
        if self.asset_context is None:
            return True
        snapshot = self._get_snapshot(self.asset_context, self.higher_timeframe)
        if snapshot is None or len(snapshot.df) < self.fractal_size * 2 + 1:
            return True
        swings = self._detect_swings(snapshot.df, self.fractal_size)
        structure = self._determine_structure(snapshot.df, swings)
        if structure.trend == "range":
            return True
        if direction == "LONG":
            return structure.trend == "uptrend"
        return structure.trend == "downtrend"

    def _update_btc_filters(self, reference_context: MultiTimeframeResult | None) -> None:
        if reference_context is None:
            self._btc_state = "unknown"
            self._btc_emergency = False
            self._btc_high_vol = False
            return
        self._btc_state = self._infer_btc_state(reference_context)
        self._btc_emergency = self._detect_btc_emergency(reference_context)
        self._btc_high_vol = self._detect_btc_volatility(reference_context)

    def _infer_btc_state(self, context: MultiTimeframeResult) -> str:
        hour = self._get_snapshot(context, "1h")
        four_hour = self._get_snapshot(context, "4h")
        if hour and four_hour:
            if hour.ma_order in {"bullish", "crossing"} and four_hour.ma_order in {"bullish", "crossing"}:
                return "bullish"
            if hour.ma_order == "bearish" and four_hour.ma_order == "bearish":
                return "bearish"
        return "sideways"

    def _detect_btc_emergency(self, context: MultiTimeframeResult) -> bool:
        snapshot = self._get_snapshot(context, self.entry_timeframe)
        if snapshot is None:
            return False
        latest = snapshot.latest
        open_price = _to_float(latest.get("open"))
        close_price = _to_float(latest.get("close"))
        high_price = _to_float(latest.get("high"))
        low_price = _to_float(latest.get("low"))
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        upper_wick = max(0.0, high_price - body_top)
        lower_wick = max(0.0, body_bottom - low_price)
        wick = max(upper_wick, lower_wick)
        reference = max(close_price, 1e-8)
        return wick / reference >= self.btc_wick_threshold

    def _detect_btc_volatility(self, context: MultiTimeframeResult) -> bool:
        snapshot = self._get_snapshot(context, "4h")
        if snapshot is None:
            return False
        df = snapshot.df
        if len(df) < 2:
            return False
        prev_close = float(df.iloc[-2]["close"])
        latest_close = float(df.iloc[-1]["close"])
        if prev_close <= 0:
            return False
        change = abs(latest_close - prev_close) / prev_close
        return change >= self.btc_volatility_threshold

    def _get_snapshot(
        self, result: MultiTimeframeResult | None, timeframe: str
    ) -> TimeframeSnapshot | None:
        if result is None:
            return None
        return result.snapshots.get(timeframe)

    def _get_snapshot_df(self) -> pd.DataFrame | None:
        if self.asset_context is None:
            return None
        snapshot = self._get_snapshot(self.asset_context, self.entry_timeframe)
        return None if snapshot is None else snapshot.df

    def _zone_key(self, zone: Zone) -> str:
        return f"{zone.kind}:{zone.base_idx}:{zone.impulse_idx}:{zone.created_at.value}"


__all__ = ["SupplyDemandStrategy"]
