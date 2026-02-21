from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import pandas as pd

from bot.multi_timeframe import MultiTimeframeResult, TimeframeSnapshot
from bot.strategy import Strategy

SignalDirection = Literal["LONG", "SHORT"]
ZoneKind = Literal["demand", "supply"]
TrendDirection = Literal["uptrend", "downtrend", "range"]


def _to_float(value: float | int | str | None) -> float:
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
    close: float
    kind: Literal["high", "low"]


@dataclass
class MarketStructure:
    trend: TrendDirection
    last_valid_low: SwingPoint | None = None
    last_valid_high: SwingPoint | None = None


@dataclass
class Zone:
    kind: ZoneKind
    lower: float
    upper: float
    start_idx: int
    impulse_idx: int
    created_at: pd.Timestamp


@dataclass
class SupplyDemandSignal:
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
    rr_ratio: float = 0.0
    zone_kind: ZoneKind = "demand"
    zone_bounds: tuple[float, float] = (0.0, 0.0)


class SupplyDemandStrategy(Strategy):
    """Charlie: Price Action Supply & Demand strategy (1h primary timeframe)."""

    name = "charlie_supply_demand"
    entry_timeframe = "1h"
    min_bars = 220
    fractal_size = 3
    min_risk_reward = 2.5
    consolidation_min_bars = 3
    consolidation_range_mult = 1.0
    impulse_body_mult = 1.5
    stop_buffer_pct = 0.001

    def __init__(self, reference_symbol: str = "BTC/USDT"):
        super().__init__()
        self.reference_symbol = reference_symbol
        self.asset_context: MultiTimeframeResult | None = None
        self.reference_context: MultiTimeframeResult | None = None
        self._latest_long_signal: SupplyDemandSignal | None = None
        self._latest_short_signal: SupplyDemandSignal | None = None
        self._btc_state = "unknown"
        self._consumed_zones: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def update_context(
        self,
        asset_context: MultiTimeframeResult | None,
        reference_context: MultiTimeframeResult | None = None,
    ) -> None:
        self.asset_context = asset_context
        self.reference_context = reference_context
        if reference_context is not None:
            self._btc_state = self._infer_btc_state(reference_context)
        else:
            self._btc_state = "unknown"

    # ------------------------------------------------------------------
    def should_enter(self, df: pd.DataFrame) -> bool:
        self._latest_long_signal = None
        self._latest_short_signal = None
        signal = self._evaluate_structure_and_zones()
        if signal is None:
            return False
        if signal.direction == "LONG":
            self._latest_long_signal = signal
        else:
            self._latest_short_signal = signal
        return True

    def should_exit(self, df: pd.DataFrame) -> bool:
        # Charlie's exits are handled by the daemon (SL/TP + trend monitoring)
        return False

    # Public helpers ---------------------------------------------------
    @property
    def last_long_signal(self) -> SupplyDemandSignal | None:
        return self._latest_long_signal

    @property
    def last_short_signal(self) -> SupplyDemandSignal | None:
        return self._latest_short_signal

    # Core evaluation --------------------------------------------------
    def _evaluate_structure_and_zones(self) -> SupplyDemandSignal | None:
        snapshot = self._get_snapshot(self.asset_context, self.entry_timeframe)
        if snapshot is None:
            return None
        df = snapshot.df
        if len(df) < self.min_bars:
            return None
        swings = self._detect_swings(df)
        if not swings:
            return None
        structure = self._determine_structure(df, swings)
        if structure.trend == "uptrend":
            return self._evaluate_zone_signal(df, swings, structure, direction="LONG")
        if structure.trend == "downtrend":
            return self._evaluate_zone_signal(df, swings, structure, direction="SHORT")
        return None

    def _evaluate_zone_signal(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        structure: MarketStructure,
        direction: SignalDirection,
    ) -> SupplyDemandSignal | None:
        kind: ZoneKind = "demand" if direction == "LONG" else "supply"
        zones = self._identify_zones(df, kind)
        if not zones:
            return None
        latest_idx = len(df) - 1
        for zone in reversed(zones):
            key = self._zone_key(zone)
            retest_idx = self._zone_retest_index(df, zone)
            if retest_idx is None:
                continue
            # Zone invalid (price closed through) — mark consumed and skip
            if retest_idx < 0:
                self._consumed_zones[key] = retest_idx
                continue
            if key in self._consumed_zones:
                continue
            # Retest happened in the past — mark consumed, no new trade
            if retest_idx < latest_idx:
                self._consumed_zones[key] = retest_idx
                continue
            # Retest is the latest candle (fresh) → evaluate trade idea
            plan = self._build_trade_plan(df, swings, zone, direction)
            self._consumed_zones[key] = retest_idx
            if plan is None:
                continue
            entry, stop_loss, take_profit, rr = plan
            reason = self._signal_reason(direction, zone, rr)
            confirmations = self._confirmations(direction, structure, zone)
            filters = self._filters(structure, rr)
            timestamp = df.iloc[retest_idx]["timestamp"]
            return SupplyDemandSignal(
                direction=direction,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                confirmations=confirmations,
                filters=filters,
                signal_timeframes=[self.entry_timeframe],
                btc_state=self._btc_state,
                timestamp=timestamp,
                rr_ratio=rr,
                zone_kind=zone.kind,
                zone_bounds=(zone.lower, zone.upper),
            )
        return None

    # Zones ------------------------------------------------------------
    def _identify_zones(self, df: pd.DataFrame, kind: ZoneKind) -> List[Zone]:
        zones: List[Zone] = []
        min_bars = self.consolidation_min_bars
        for idx in range(min_bars + 1, len(df)):
            impulse_row = df.iloc[idx]
            if not self._is_impulsive(impulse_row, kind):
                continue
            window = df.iloc[idx - min_bars - 1 : idx - 1]
            if len(window) < min_bars:
                continue
            if not self._is_consolidation(window):
                continue
            zone_candle = df.iloc[idx - 1]
            lower = float(zone_candle["low"])
            upper = float(zone_candle["high"])
            created_at = impulse_row["timestamp"]
            zones.append(
                Zone(
                    kind=kind,
                    lower=min(lower, upper),
                    upper=max(lower, upper),
                    start_idx=idx - 1,
                    impulse_idx=idx,
                    created_at=created_at,
                )
            )
        return zones

    def _is_consolidation(self, window: pd.DataFrame) -> bool:
        if window.empty:
            return False
        atr = _to_float(window.iloc[-1].get("atr"))
        if atr <= 0:
            return False
        range_high = float(window["high"].max())
        range_low = float(window["low"].min())
        return (range_high - range_low) <= atr * self.consolidation_range_mult

    def _is_impulsive(self, row: pd.Series, kind: ZoneKind) -> bool:
        atr = _to_float(row.get("atr"))
        if atr <= 0:
            return False
        open_price = float(row["open"])
        close_price = float(row["close"])
        body = abs(close_price - open_price)
        if body < atr * self.impulse_body_mult:
            return False
        if kind == "demand":
            return close_price > open_price
        return close_price < open_price

    def _zone_retest_index(self, df: pd.DataFrame, zone: Zone) -> int | None:
        for idx in range(zone.impulse_idx + 1, len(df)):
            row = df.iloc[idx]
            high = float(row["high"])
            low = float(row["low"])
            close_price = float(row["close"])
            intersects = (low <= zone.upper) and (high >= zone.lower)
            if intersects:
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

    def _build_trade_plan(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        zone: Zone,
        direction: SignalDirection,
    ) -> Optional[tuple[float, float, float, float]]:
        latest_close = float(df.iloc[-1]["close"])
        if direction == "LONG":
            entry = zone.upper
            buffer = entry * self.stop_buffer_pct
            stop_loss = zone.lower - buffer
            target = self._recent_swing(swings, kind="high", before_idx=len(df) - 1)
            if target is None or target <= entry:
                return None
            reward = target - entry
            risk = entry - stop_loss
        else:
            entry = zone.lower
            buffer = entry * self.stop_buffer_pct
            stop_loss = zone.upper + buffer
            target = self._recent_swing(swings, kind="low", before_idx=len(df) - 1)
            if target is None or target >= entry:
                return None
            reward = entry - target
            risk = stop_loss - entry
        if risk <= 0 or reward <= 0:
            return None
        rr = reward / risk
        if rr < self.min_risk_reward:
            return None
        take_profit = target
        return entry, stop_loss, take_profit, rr

    def _recent_swing(self, swings: List[SwingPoint], kind: Literal["high", "low"], before_idx: int) -> float | None:
        for swing in reversed(swings):
            if swing.kind != kind:
                continue
            if swing.idx < before_idx:
                return swing.price
        return None

    # Structure --------------------------------------------------------
    def _detect_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        swings: List[SwingPoint] = []
        n = self.fractal_size
        last_index = len(df) - n
        for idx in range(n, last_index):
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
                        close=float(row["close"]),
                        kind="high",
                    )
                )
            if low < float(prev_window["low"].min()) and low < float(next_window["low"].min()):
                swings.append(
                    SwingPoint(
                        idx=idx,
                        timestamp=row["timestamp"],
                        price=low,
                        close=float(row["close"]),
                        kind="low",
                    )
                )
        swings.sort(key=lambda s: s.idx)
        return swings

    def _determine_structure(self, df: pd.DataFrame, swings: List[SwingPoint]) -> MarketStructure:
        last_valid_low: SwingPoint | None = None
        last_valid_high: SwingPoint | None = None
        prev_high: float | None = None
        prev_low: float | None = None
        pending_low: SwingPoint | None = None
        pending_high: SwingPoint | None = None

        for swing in swings:
            if swing.kind == "high":
                pending_high = swing
                if prev_high is None:
                    prev_high = swing.price
                elif swing.price > prev_high:
                    prev_high = swing.price
                    if pending_low is not None:
                        last_valid_low = pending_low
                        pending_low = None
                else:
                    prev_high = swing.price
            else:
                pending_low = swing
                if prev_low is None:
                    prev_low = swing.price
                elif swing.price < prev_low:
                    prev_low = swing.price
                    if pending_high is not None:
                        last_valid_high = pending_high
                        pending_high = None
                else:
                    prev_low = swing.price

        latest_close = float(df.iloc[-1]["close"])
        if last_valid_low and (not last_valid_high or last_valid_low.idx > last_valid_high.idx):
            if latest_close >= last_valid_low.price:
                return MarketStructure(trend="uptrend", last_valid_low=last_valid_low, last_valid_high=last_valid_high)
        if last_valid_high and (not last_valid_low or last_valid_high.idx > last_valid_low.idx):
            if latest_close <= last_valid_high.price:
                return MarketStructure(trend="downtrend", last_valid_low=last_valid_low, last_valid_high=last_valid_high)
        return MarketStructure(trend="range", last_valid_low=last_valid_low, last_valid_high=last_valid_high)

    # Messaging helpers -----------------------------------------------
    def _signal_reason(self, direction: SignalDirection, zone: Zone, rr: float) -> str:
        side = "demand" if direction == "LONG" else "supply"
        return (
            f"{direction} setup: retest of {side} zone from {zone.created_at.strftime('%Y-%m-%d %H:%M')} "
            f"(R:R={rr:.2f})"
        )

    def _confirmations(self, direction: SignalDirection, structure: MarketStructure, zone: Zone) -> list[str]:
        confirmations: list[str] = [f"trend={structure.trend}", f"zone_created={zone.created_at.strftime('%m-%d %H:%M')}"]
        if direction == "LONG" and structure.last_valid_low is not None:
            confirmations.append(f"valid_low_intact@{structure.last_valid_low.price:.4f}")
        if direction == "SHORT" and structure.last_valid_high is not None:
            confirmations.append(f"valid_high_intact@{structure.last_valid_high.price:.4f}")
        confirmations.append("fresh_zone")
        return confirmations

    def _filters(self, structure: MarketStructure, rr: float) -> list[str]:
        filters: list[str] = [f"btc_state={self._btc_state}", f"rr={rr:.2f}"]
        if structure.last_valid_low:
            filters.append(f"last_valid_low={structure.last_valid_low.price:.4f}")
        if structure.last_valid_high:
            filters.append(f"last_valid_high={structure.last_valid_high.price:.4f}")
        return filters

    # Utilities --------------------------------------------------------
    def _zone_key(self, zone: Zone) -> str:
        return f"{zone.kind}:{zone.created_at.isoformat()}:{zone.start_idx}:{zone.impulse_idx}"

    def _get_snapshot(
        self, result: MultiTimeframeResult | None, timeframe: str
    ) -> TimeframeSnapshot | None:
        if result is None:
            return None
        return result.snapshots.get(timeframe)

    def _infer_btc_state(self, context: MultiTimeframeResult) -> str:
        hour = self._get_snapshot(context, "1h")
        four_hour = self._get_snapshot(context, "4h")
        if hour and four_hour:
            if hour.ma_order in {"bullish", "crossing"} and four_hour.ma_order in {"bullish", "crossing"}:
                return "bullish"
            if hour.ma_order == "bearish" and four_hour.ma_order == "bearish":
                return "bearish"
        return "sideways"


__all__ = ["SupplyDemandStrategy", "SupplyDemandSignal"]
