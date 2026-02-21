from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from bot.multi_timeframe import MultiTimeframeResult, TimeframeSnapshot
from bot.strategy import Strategy

ZoneType = Literal["demand", "supply"]
SignalDirection = Literal["LONG", "SHORT"]
TrendState = Literal["uptrend", "downtrend", "range"]


@dataclass
class SwingPoint:
    kind: Literal["high", "low"]
    idx: int
    price: float
    timestamp: pd.Timestamp


@dataclass
class ValidSwing:
    swing: SwingPoint
    confirmed_at: int

    @property
    def price(self) -> float:
        return self.swing.price

    @property
    def idx(self) -> int:
        return self.swing.idx

    @property
    def timestamp(self) -> pd.Timestamp:
        return self.swing.timestamp


@dataclass
class MarketStructureStatus:
    trend: TrendState
    swing_highs: list[SwingPoint]
    swing_lows: list[SwingPoint]
    valid_highs: list[ValidSwing]
    valid_lows: list[ValidSwing]

    @property
    def last_swing_high(self) -> SwingPoint | None:
        return self.swing_highs[-1] if self.swing_highs else None

    @property
    def last_swing_low(self) -> SwingPoint | None:
        return self.swing_lows[-1] if self.swing_lows else None

    @property
    def last_valid_high(self) -> ValidSwing | None:
        return self.valid_highs[-1] if self.valid_highs else None

    @property
    def last_valid_low(self) -> ValidSwing | None:
        return self.valid_lows[-1] if self.valid_lows else None


@dataclass
class Zone:
    id: str
    direction: ZoneType
    zone_low: float
    zone_high: float
    candle_idx: int
    timestamp: pd.Timestamp
    consumed: bool = False
    invalidated: bool = False
    touches: int = 0


@dataclass
class EntrySignal:
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    trend: TrendState
    zone_id: str
    zone_bounds: tuple[float, float]
    risk_reward: float
    timestamp: pd.Timestamp
    btc_state: str
    filters: list[str] = field(default_factory=list)
    structure_notes: list[str] = field(default_factory=list)
    breakeven_moved: bool = False


class SupplyDemandStrategy(Strategy):
    """Charlie\'s Price Action Supply & Demand strategy."""

    name = "charlie_supply_demand"
    min_bars = 220
    primary_timeframe = "1h"
    swing_window = 3
    consolidation_bars = 3
    consolidation_range_atr = 1.0
    impulse_body_mult = 1.5
    rr_threshold = 2.5
    stop_buffer_pct = 0.001
    max_zone_history = 12
    btc_vol_threshold = 0.03
    btc_emergency_wick = 0.025
    breakeven_trigger_rr = 1.0

    def __init__(self, reference_symbol: str = "BTC/USDT"):
        super().__init__()
        self.reference_symbol = reference_symbol
        self.asset_context: MultiTimeframeResult | None = None
        self.reference_context: MultiTimeframeResult | None = None
        self._latest_long_signal: EntrySignal | None = None
        self._latest_short_signal: EntrySignal | None = None
        self._active_trade: EntrySignal | None = None
        self._zones: dict[str, Zone] = {}
        self._btc_state: str = "unknown"
        self._btc_change_4h: float | None = None
        self._btc_vol_ok: bool = True
        self._btc_emergency: bool = False

    # ------------------------------------------------------------------
    def update_context(
        self,
        asset_context: MultiTimeframeResult,
        reference_context: MultiTimeframeResult | None = None,
    ) -> None:
        self.asset_context = asset_context
        self.reference_context = reference_context
        self._update_reference_state(reference_context)

    # ------------------------------------------------------------------
    def should_enter(self, df: pd.DataFrame) -> bool:
        self._latest_long_signal = None
        self._latest_short_signal = None
        if self.asset_context is None:
            return False
        snapshot = self._get_snapshot(self.asset_context, self.primary_timeframe)
        if snapshot is None:
            return False
        structure = self._analyze_structure(snapshot.df)
        if structure.trend not in {"uptrend", "downtrend"}:
            return False
        if not self._btc_vol_ok or self._btc_emergency:
            return False

        desired_zone = "demand" if structure.trend == "uptrend" else "supply"
        self._refresh_zones(snapshot.df, desired_zone)
        zone = self._find_retest_zone(snapshot.df, desired_zone)
        if zone is None:
            return False
        signal = self._build_signal(snapshot, zone, structure)
        if signal is None:
            return False

        if signal.direction == "LONG":
            self._latest_long_signal = signal
        else:
            self._latest_short_signal = signal
        self._active_trade = EntrySignal(**vars(signal))
        zone.consumed = True
        return True

    def should_exit(self, df: pd.DataFrame) -> bool:
        if self._active_trade is None or df.empty:
            return False
        latest = df.iloc[-1]
        price = float(latest["close"])
        trade = self._active_trade
        exited = False

        if self._btc_emergency:
            exited = True
        elif trade.direction == "LONG":
            if price <= trade.stop_loss or price >= trade.take_profit:
                exited = True
        else:
            if price >= trade.stop_loss or price <= trade.take_profit:
                exited = True

        if not exited:
            structure = self._analyze_structure(df)
            if trade.direction == "LONG" and structure.last_valid_low and price < structure.last_valid_low.price:
                exited = True
            elif trade.direction == "SHORT" and structure.last_valid_high and price > structure.last_valid_high.price:
                exited = True

        if not exited:
            self._maybe_move_breakeven(trade, price)
            return False

        self._active_trade = None
        return True

    # ------------------------------------------------------------------
    @property
    def last_long_signal(self) -> EntrySignal | None:
        return self._latest_long_signal

    @property
    def last_short_signal(self) -> EntrySignal | None:
        return self._latest_short_signal

    # ------------------------------------------------------------------
    def _build_signal(
        self,
        snapshot: TimeframeSnapshot,
        zone: Zone,
        structure: MarketStructureStatus,
    ) -> EntrySignal | None:
        latest = snapshot.df.iloc[-1]
        signal_time = pd.Timestamp(latest["timestamp"])
        direction: SignalDirection = "LONG" if zone.direction == "demand" else "SHORT"
        entry_price = zone.zone_high if direction == "LONG" else zone.zone_low
        buffer = snapshot.price * self.stop_buffer_pct
        if direction == "LONG":
            stop_loss = zone.zone_low - buffer
            target_source = structure.last_swing_high
        else:
            stop_loss = zone.zone_high + buffer
            target_source = structure.last_swing_low
        if target_source is None:
            return None
        take_profit = float(target_source.price)
        if direction == "LONG" and take_profit <= entry_price:
            return None
        if direction == "SHORT" and take_profit >= entry_price:
            return None

        rr = self._calc_rr(direction, entry_price, stop_loss, take_profit)
        if rr is None or rr < self.rr_threshold:
            return None

        filters = [f"rr={rr:.2f}"]
        if self._btc_change_4h is not None:
            filters.append(f"btc_4h_change={self._btc_change_4h:.2%}")
        filters.append("btc_volatility_ok")
        filters.append(f"zone={zone.id}")

        structure_notes: list[str] = []
        if structure.last_valid_low:
            structure_notes.append(
                f"valid_low={structure.last_valid_low.price:.4f}"
            )
        if structure.last_valid_high:
            structure_notes.append(
                f"valid_high={structure.last_valid_high.price:.4f}"
            )

        reason = (
            f"{structure.trend} {zone.direction} zone retest @ {entry_price:.4f}"
        )

        return EntrySignal(
            direction=direction,
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            reason=reason,
            trend=structure.trend,
            zone_id=zone.id,
            zone_bounds=(zone.zone_low, zone.zone_high),
            risk_reward=float(rr),
            timestamp=signal_time,
            btc_state=self._btc_state,
            filters=filters,
            structure_notes=structure_notes,
        )

    def _refresh_zones(self, df: pd.DataFrame, direction: ZoneType) -> None:
        new_zones = self._scan_zones(df, direction)
        for zone in new_zones:
            if zone.id not in self._zones:
                self._zones[zone.id] = zone
        self._invalidate_zones(df)
        self._prune_zones()

    def _invalidate_zones(self, df: pd.DataFrame) -> None:
        latest_close = float(df.iloc[-1]["close"])
        for zone in self._zones.values():
            if zone.invalidated:
                continue
            if zone.direction == "demand" and latest_close < zone.zone_low:
                zone.invalidated = True
            elif zone.direction == "supply" and latest_close > zone.zone_high:
                zone.invalidated = True

    def _prune_zones(self) -> None:
        if len(self._zones) <= self.max_zone_history:
            return
        ordered = sorted(self._zones.values(), key=lambda z: z.candle_idx)
        for zone in ordered[:-self.max_zone_history]:
            self._zones.pop(zone.id, None)

    def _find_retest_zone(self, df: pd.DataFrame, direction: ZoneType) -> Zone | None:
        latest = df.iloc[-1]
        prev_close = float(df.iloc[-2]["close"]) if len(df) >= 2 else None
        ordered = sorted(
            (z for z in self._zones.values() if z.direction == direction and not z.consumed and not z.invalidated),
            key=lambda z: z.candle_idx,
            reverse=True,
        )
        for zone in ordered:
            if self._zone_retested(zone, latest, prev_close):
                zone.touches += 1
                return zone
        return None

    def _zone_retested(self, zone: Zone, latest: pd.Series, prev_close: float | None) -> bool:
        high = float(latest["high"])
        low = float(latest["low"])
        if zone.direction == "demand":
            if prev_close is not None and prev_close < zone.zone_low:
                return False
            return low <= zone.zone_high and high >= zone.zone_low
        if prev_close is not None and prev_close > zone.zone_high:
            return False
        return high >= zone.zone_low and low <= zone.zone_high

    def _scan_zones(self, df: pd.DataFrame, direction: ZoneType) -> list[Zone]:
        zones: list[Zone] = []
        if len(df) <= self.consolidation_bars + 1:
            return zones
        atr = df.get("atr")
        if atr is None or atr.isna().all():
            return zones
        for impulse_idx in range(self.consolidation_bars + 1, len(df)):
            row = df.iloc[impulse_idx]
            atr_value = row.get("atr")
            if pd.isna(atr_value) or atr_value <= 0:
                continue
            body = abs(float(row["close"]) - float(row["open"]))
            if body < self.impulse_body_mult * float(atr_value):
                continue
            is_bullish = float(row["close"]) > float(row["open"])
            if direction == "demand" and not is_bullish:
                continue
            if direction == "supply" and is_bullish:
                continue
            cons_start = impulse_idx - self.consolidation_bars
            window = df.iloc[cons_start:impulse_idx]
            cons_range = float(window["high"].max() - window["low"].min())
            if cons_range >= self.consolidation_range_atr * float(atr_value):
                continue
            zone_idx = impulse_idx - 1
            zone_row = df.iloc[zone_idx]
            zone_low = float(zone_row["low"])
            zone_high = float(zone_row["high"])
            if zone_high <= zone_low:
                continue
            zone_id = f"{zone_row['timestamp'].isoformat()}-{direction}"
            zones.append(
                Zone(
                    id=zone_id,
                    direction=direction,
                    zone_low=zone_low,
                    zone_high=zone_high,
                    candle_idx=zone_idx,
                    timestamp=zone_row["timestamp"],
                )
            )
        return zones

    # ------------------------------------------------------------------
    def _analyze_structure(self, df: pd.DataFrame) -> MarketStructureStatus:
        swing_highs, swing_lows = self._fractal_swings(df)
        valid_highs, valid_lows = self._validate_swings(df, swing_highs, swing_lows)
        trend = self._determine_trend(valid_highs, valid_lows)
        return MarketStructureStatus(
            trend=trend,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            valid_highs=valid_highs,
            valid_lows=valid_lows,
        )

    def _fractal_swings(self, df: pd.DataFrame) -> tuple[list[SwingPoint], list[SwingPoint]]:
        highs: list[SwingPoint] = []
        lows: list[SwingPoint] = []
        window = self.swing_window
        if len(df) < window * 2 + 1:
            return highs, lows
        high_values = df["high"].to_numpy()
        low_values = df["low"].to_numpy()
        timestamps = df["timestamp"].to_list()
        for idx in range(window, len(df) - window):
            center_high = high_values[idx]
            if center_high == max(high_values[idx - window : idx + window + 1]):
                highs.append(
                    SwingPoint(
                        kind="high",
                        idx=idx,
                        price=float(center_high),
                        timestamp=pd.Timestamp(timestamps[idx]),
                    )
                )
            center_low = low_values[idx]
            if center_low == min(low_values[idx - window : idx + window + 1]):
                lows.append(
                    SwingPoint(
                        kind="low",
                        idx=idx,
                        price=float(center_low),
                        timestamp=pd.Timestamp(timestamps[idx]),
                    )
                )
        return highs, lows

    def _validate_swings(
        self,
        df: pd.DataFrame,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
    ) -> tuple[list[ValidSwing], list[ValidSwing]]:
        close_vals = df["close"].to_numpy()
        valid_highs: list[ValidSwing] = []
        valid_lows: list[ValidSwing] = []
        for low_point in swing_lows:
            prev_high_candidates = [h for h in swing_highs if h.idx < low_point.idx]
            if not prev_high_candidates:
                continue
            prev_high = prev_high_candidates[-1]
            breakout = np.where(close_vals[low_point.idx + 1 :] > prev_high.price)[0]
            if breakout.size == 0:
                continue
            confirmed_idx = low_point.idx + 1 + int(breakout[0])
            valid_lows.append(ValidSwing(swing=low_point, confirmed_at=confirmed_idx))
        for high_point in swing_highs:
            prev_low_candidates = [l for l in swing_lows if l.idx < high_point.idx]
            if not prev_low_candidates:
                continue
            prev_low = prev_low_candidates[-1]
            breakdown = np.where(close_vals[high_point.idx + 1 :] < prev_low.price)[0]
            if breakdown.size == 0:
                continue
            confirmed_idx = high_point.idx + 1 + int(breakdown[0])
            valid_highs.append(ValidSwing(swing=high_point, confirmed_at=confirmed_idx))
        return valid_highs, valid_lows

    def _determine_trend(self, valid_highs: list[ValidSwing], valid_lows: list[ValidSwing]) -> TrendState:
        if len(valid_highs) >= 2 and len(valid_lows) >= 2:
            last_highs = valid_highs[-2:]
            last_lows = valid_lows[-2:]
            if last_highs[-1].price > last_highs[-2].price and last_lows[-1].price > last_lows[-2].price:
                return "uptrend"
            if last_highs[-1].price < last_highs[-2].price and last_lows[-1].price < last_lows[-2].price:
                return "downtrend"
        return "range"

    # ------------------------------------------------------------------
    def _calc_rr(
        self,
        direction: SignalDirection,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> float | None:
        if direction == "LONG":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        if risk <= 0 or reward <= 0:
            return None
        return reward / risk

    def _maybe_move_breakeven(self, trade: EntrySignal, price: float) -> None:
        if trade.breakeven_moved:
            return
        if trade.direction == "LONG":
            risk = trade.entry_price - trade.stop_loss
            if risk <= 0:
                return
            if price - trade.entry_price >= risk * self.breakeven_trigger_rr:
                trade.stop_loss = trade.entry_price
                trade.breakeven_moved = True
        else:
            risk = trade.stop_loss - trade.entry_price
            if risk <= 0:
                return
            if trade.entry_price - price >= risk * self.breakeven_trigger_rr:
                trade.stop_loss = trade.entry_price
                trade.breakeven_moved = True

    # ------------------------------------------------------------------
    def _get_snapshot(
        self,
        result: MultiTimeframeResult | None,
        timeframe: str,
    ) -> TimeframeSnapshot | None:
        if result is None:
            return None
        return result.snapshots.get(timeframe)

    def _update_reference_state(self, reference_context: MultiTimeframeResult | None) -> None:
        if reference_context is None:
            self._btc_state = "unknown"
            self._btc_change_4h = None
            self._btc_vol_ok = True
            self._btc_emergency = False
            return
        hour_snapshot = self._get_snapshot(reference_context, "1h")
        four_snapshot = self._get_snapshot(reference_context, "4h")
        if hour_snapshot and four_snapshot:
            if hour_snapshot.ma_order in {"bullish", "crossing"} and four_snapshot.ma_order in {"bullish", "crossing"}:
                self._btc_state = "bullish"
            elif hour_snapshot.ma_order == "bearish" and four_snapshot.ma_order == "bearish":
                self._btc_state = "bearish"
            else:
                self._btc_state = "sideways"
        else:
            self._btc_state = "unknown"
        self._btc_change_4h = self._compute_btc_change(hour_snapshot)
        self._btc_vol_ok = True
        if self._btc_change_4h is not None and abs(self._btc_change_4h) > self.btc_vol_threshold:
            self._btc_vol_ok = False
        fifteen_snapshot = self._get_snapshot(reference_context, "15m")
        self._btc_emergency = False
        if fifteen_snapshot is not None:
            wick = self._wick_percentage(fifteen_snapshot)
            if wick >= self.btc_emergency_wick:
                self._btc_emergency = True

    def _compute_btc_change(self, snapshot: TimeframeSnapshot | None) -> float | None:
        if snapshot is None:
            return None
        df = snapshot.df
        if len(df) < 5:
            return None
        recent = float(df.iloc[-1]["close"])
        prior = float(df.iloc[-5]["close"])
        if prior == 0:
            return None
        return (recent - prior) / prior

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


__all__ = ["SupplyDemandStrategy", "EntrySignal"]
