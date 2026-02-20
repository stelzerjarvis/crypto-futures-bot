from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

import pandas as pd

from bot.advisor import Advisor
from bot.exchange import BinanceFuturesTestnet
from bot.multi_timeframe import MultiTimeframeAnalyzer, MultiTimeframeResult, TimeframeSnapshot
from bot.notifier import Notifier
from bot.risk_manager import RiskLimits, RiskManager
from config.settings import Settings
from db.models import TradeDatabase
from strategies.divergence_4ma import Divergence4MAStrategy, EntrySignal
from utils.logger import get_logger


def normalize_symbol(symbol: str) -> str:
    if symbol is None:
        raise ValueError('symbol is required')
    symbol = symbol.strip().upper()
    if symbol.endswith(':USDT'):
        return symbol
    if '/' in symbol:
        return symbol
    if symbol.endswith('USDT'):
        base = symbol[:-4]
        return f"{base}/USDT"
    return symbol


def asset_key(symbol: str) -> str:
    if symbol is None:
        return ''
    cleaned = symbol.upper().replace(':USDT', '')
    if '/' in cleaned:
        base, quote = cleaned.split('/', 1)
        return f"{base}{quote}"
    return cleaned


class TradingDaemon:
    LOOP_SLEEP_SECONDS = 5
    MINUTE_INTERVAL = timedelta(seconds=60)
    ANALYSIS_INTERVAL = timedelta(minutes=15)
    DAILY_INTERVAL = timedelta(hours=24)

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger('daemon')
        self.exchange = BinanceFuturesTestnet(settings.api_key, settings.api_secret)
        self.notifier = Notifier()
        self.advisor = Advisor(
            exchange=self.exchange,
            timeframe='15m',
            model=settings.mike_model,
            enabled=settings.mike_enabled,
        )
        self.db = TradeDatabase()
        self.reference_symbol = normalize_symbol(settings.reference_symbol)
        seen: Dict[str, None] = {}
        self.assets = []
        for asset in settings.assets:
            normalized = normalize_symbol(asset)
            if normalized not in seen:
                seen[normalized] = None
                self.assets.append(normalized)
        self.asset_symbol_map = {asset: asset for asset in self.assets}
        self.asset_aliases = {asset_key(symbol): asset for asset, symbol in self.asset_symbol_map.items()}
        self.multi_analyzer = MultiTimeframeAnalyzer(
            exchange=self.exchange,
            timeframes=['15m', '1h', '4h', '1d'],
            ma_periods=settings.ma_periods,
            limit=360,
        )
        self.strategies = {
            asset: Divergence4MAStrategy(reference_symbol=self.reference_symbol)
            for asset in self.assets
        }
        self.risk_manager = RiskManager(
            RiskLimits(
                risk_per_trade=settings.risk_per_trade,
                max_positions=settings.max_positions,
                max_daily_loss=settings.max_daily_loss,
            )
        )
        self.summary_interval = timedelta(hours=max(1, settings.summary_interval_hours))
        now = datetime.now(timezone.utc)
        self._last_minute_check = now - self.MINUTE_INTERVAL
        self._next_analysis = self._next_boundary(now, self.ANALYSIS_INTERVAL)
        self._next_summary = now + self.summary_interval
        self._next_daily_recap = now + self.DAILY_INTERVAL
        self._processed_signals: Dict[Tuple[str, str], str | None] = {}
        self.logger.info('Configured assets: %s', ', '.join(self.assets))
        self._bootstrap_leverage()

    # ------------------------------------------------------------------
    def _bootstrap_leverage(self) -> None:
        for asset, symbol in self.asset_symbol_map.items():
            try:
                self.exchange.set_leverage(symbol, self.settings.default_leverage)
                self.logger.info('Leverage set for %s (%sx)', asset, self.settings.default_leverage)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning('Failed to set leverage for %s: %s', asset, exc)

    def _next_boundary(self, now: datetime, delta: timedelta) -> datetime:
        seconds = int(delta.total_seconds())
        if seconds <= 0:
            return now + delta
        epoch = int(now.timestamp())
        next_epoch = ((epoch // seconds) + 1) * seconds
        return datetime.fromtimestamp(next_epoch, tz=timezone.utc)

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.logger.info('Starting trading daemon …')
        while True:
            now = datetime.now(timezone.utc)
            try:
                if now - self._last_minute_check >= self.MINUTE_INTERVAL:
                    self._last_minute_check = now
                    self._minute_cycle()
                if now >= self._next_analysis:
                    self._analysis_cycle(now)
                    self._next_analysis = self._next_boundary(now, self.ANALYSIS_INTERVAL)
                if now >= self._next_summary:
                    self._send_summary(now)
                    self._next_summary = now + self.summary_interval
                if now >= self._next_daily_recap:
                    self._send_daily_recap(now)
                    self._next_daily_recap = now + self.DAILY_INTERVAL
            except Exception as exc:  # noqa: BLE001
                self.logger.error('Daemon loop error: %s', exc, exc_info=True)
            time.sleep(self.LOOP_SLEEP_SECONDS)

    # Minute tasks -----------------------------------------------------
    def _minute_cycle(self) -> None:
        self._check_btc_emergency()
        self._monitor_open_positions()

    def _check_btc_emergency(self) -> None:
        try:
            candles = self.exchange.fetch_ohlcv(self.reference_symbol, timeframe='1m', limit=3)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning('Failed to fetch BTC candles: %s', exc)
            return
        if not candles:
            return
        _, open_price, high_price, low_price, close_price, _ = candles[-1]
        upper = max(0.0, high_price - max(open_price, close_price))
        lower = max(0.0, min(open_price, close_price) - low_price)
        wick = max(upper, lower)
        reference = max(close_price, 1e-8)
        wick_pct = wick / reference
        if wick_pct >= self.settings.btc_volatility_threshold:
            reason = (
                f"BTC wick {wick_pct:.2%} >= "
                f"{self.settings.btc_volatility_threshold:.2%}"
            )
            self.logger.warning('Emergency exit triggered: %s', reason)
            self.notifier.emergency_exit(reason)
            self._exit_all_alt_positions(reason)

    # Position monitoring ----------------------------------------------
    def _monitor_open_positions(self) -> None:
        open_trades = self.db.fetch_open_trades()
        if not open_trades:
            return
        for trade in open_trades:
            asset = trade['asset']
            symbol = self.asset_symbol_map.get(asset, asset)
            price = self._fetch_price(symbol)
            if price is None:
                continue
            direction = str(trade['direction']).upper()
            entry = float(trade['entry_price'])
            stop_loss = float(trade['stop_loss'])
            take_profit = float(trade['take_profit']) if trade['take_profit'] is not None else None
            position_size = float(trade['position_size'])
            breakeven_moved = abs(stop_loss - entry) <= 1e-8
            if direction == 'LONG':
                risk = entry - stop_loss
                if risk > 0 and not breakeven_moved and price - entry >= risk:
                    self._move_stop_to_breakeven(trade, entry)
                    stop_loss = entry
                if price <= stop_loss:
                    self._close_trade(trade, price, 'stop_loss')
                    continue
                if take_profit is not None and price >= take_profit:
                    self._close_trade(trade, price, 'take_profit')
            else:
                risk = stop_loss - entry
                if risk > 0 and not breakeven_moved and entry - price >= risk:
                    self._move_stop_to_breakeven(trade, entry)
                    stop_loss = entry
                if price >= stop_loss:
                    self._close_trade(trade, price, 'stop_loss')
                    continue
                if take_profit is not None and price <= take_profit:
                    self._close_trade(trade, price, 'take_profit')

    def _move_stop_to_breakeven(self, trade, breakeven: float) -> None:
        trade_id = trade['id']
        asset = trade['asset']
        self.db.update_trade(trade_id, stop_loss=breakeven)
        self.db.log_trade_update(trade_id, 'breakeven_move', {'stop_loss': breakeven})
        self.notifier.stop_loss_moved(asset, breakeven)
        self.logger.info('Moved SL to breakeven for %s (trade %s)', asset, trade_id)

    def _close_trade(
        self,
        trade,
        exit_price: float,
        reason: str,
        status: str = 'CLOSED',
        place_order: bool = True,
    ) -> None:
        trade_id = trade['id']
        asset = trade['asset']
        symbol = self.asset_symbol_map.get(asset, asset)
        direction = str(trade['direction']).upper()
        position_size = float(trade['position_size'])
        side = 'sell' if direction == 'LONG' else 'buy'
        if place_order:
            try:
                self.exchange.close_position(symbol, side, position_size)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning('Failed to close %s on exchange: %s', asset, exc)
        entry_price = float(trade['entry_price'])
        if direction == 'LONG':
            pnl = (exit_price - entry_price) * position_size
        else:
            pnl = (entry_price - exit_price) * position_size
        pnl_pct = ((exit_price - entry_price) / entry_price) if direction == 'LONG' else ((entry_price - exit_price) / entry_price)
        self.db.update_trade(
            trade_id,
            status=status,
            exit_price=exit_price,
            exit_time=datetime.utcnow().isoformat(timespec='seconds'),
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        self.db.log_trade_update(trade_id, reason, {'exit_price': exit_price})
        self.risk_manager.update_daily_pnl(pnl)
        self.notifier.trade_closed(asset, direction, exit_price, pnl, pnl_pct)
        self.logger.info('Trade %s closed (%s) at %.4f | pnl=%.2f', trade_id, reason, exit_price, pnl)

    def _fetch_price(self, symbol: str) -> float | None:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            last = ticker.get('last') or ticker.get('close')
            return float(last) if last is not None else None
        except Exception as exc:  # noqa: BLE001
            self.logger.warning('Failed to fetch price for %s: %s', symbol, exc)
            return None

    # Emergency handling ----------------------------------------------
    def _exit_all_alt_positions(self, reason: str) -> None:
        try:
            positions = self.exchange.fetch_positions()
        except Exception as exc:  # noqa: BLE001
            self.logger.error('Unable to fetch positions during emergency: %s', exc)
            positions = []
        trades_by_asset = defaultdict(list)
        for trade in self.db.fetch_open_trades():
            trades_by_asset[trade['asset']].append(trade)
        for pos in positions or []:
            contracts = float(pos.get('contracts') or pos.get('positionAmt') or 0.0)
            if abs(contracts) <= 0:
                continue
            symbol = pos.get('symbol') or pos.get('info', {}).get('symbol')
            if not symbol:
                continue
            key = asset_key(symbol)
            asset_name = self.asset_aliases.get(key)
            if not asset_name:
                continue
            side = 'sell' if contracts > 0 else 'buy'
            amount = abs(contracts)
            exchange_symbol = self.asset_symbol_map.get(asset_name, symbol)
            try:
                self.exchange.close_position(exchange_symbol, side, amount)
            except Exception as exc:  # noqa: BLE001
                self.logger.error('Failed to close %s during emergency: %s', asset_name, exc)
                continue
            price = float(
                pos.get('markPrice')
                or pos.get('lastPrice')
                or self._fetch_price(exchange_symbol)
                or 0.0
            )
            for trade in trades_by_asset.get(asset_name, []):
                self._close_trade(trade, price or trade['entry_price'], 'emergency_exit', status='EMERGENCY', place_order=False)

    # Analysis cycle ---------------------------------------------------
    def _analysis_cycle(self, now: datetime) -> None:
        self.logger.info('Running analysis cycle @ %s', now.strftime('%H:%M'))
        btc_context = self._analyze_symbol(self.reference_symbol)
        for asset in self.assets:
            symbol = self.asset_symbol_map.get(asset)
            if not symbol:
                continue
            try:
                asset_context = self._analyze_symbol(symbol)
                if not asset_context.snapshots:
                    continue
                self._evaluate_signals(asset, asset_context, btc_context)
            except Exception as exc:  # noqa: BLE001
                self.logger.error('Analysis failed for %s: %s', asset, exc)

    def _analyze_symbol(self, symbol: str) -> MultiTimeframeResult:
        return self.multi_analyzer.analyze(symbol)

    def _evaluate_signals(
        self,
        asset: str,
        asset_context: MultiTimeframeResult,
        reference_context: MultiTimeframeResult,
    ) -> None:
        strategy = self.strategies[asset]
        strategy.update_context(asset_context, reference_context)
        entry_snapshot = asset_context.snapshots.get(strategy.entry_timeframe)
        if entry_snapshot is None:
            return
        entry_df = entry_snapshot.df
        if len(entry_df) < strategy.min_bars:
            return
        strategy.should_enter(entry_df)
        signals = [sig for sig in (strategy.last_long_signal, strategy.last_short_signal) if sig]
        for signal in signals:
            key = (asset, signal.direction)
            timestamp = None
            if isinstance(signal.timestamp, pd.Timestamp):
                timestamp = signal.timestamp.isoformat()
            if timestamp and self._processed_signals.get(key) == timestamp:
                continue
            self._processed_signals[key] = timestamp
            self._process_signal(asset, asset_context, reference_context, signal)

    def _process_signal(
        self,
        asset: str,
        asset_context: MultiTimeframeResult,
        reference_context: MultiTimeframeResult,
        signal: EntrySignal,
    ) -> None:
        self.notifier.signal_detected(asset, signal.direction, signal.reason)
        context_payload = self._build_signal_context(asset, asset_context, reference_context, signal)
        start = time.monotonic()
        decision = self.advisor.review_signal(context_payload)
        elapsed = time.monotonic() - start
        self.notifier.mike_decision(asset, decision.decision, decision.reasoning or '')
        if decision.decision == 'REJECT':
            self.logger.info('Mike rejected %s %s (%.2fs)', signal.direction, asset, elapsed)
            return
        if decision.decision not in {'APPROVE', 'MODIFY'}:
            self.logger.info('Mike returned %s — skipping', decision.decision)
            return
        self._execute_trade(asset, signal, decision, context_payload, elapsed)

    def _build_signal_context(
        self,
        asset: str,
        asset_context: MultiTimeframeResult,
        reference_context: MultiTimeframeResult,
        signal: EntrySignal,
    ) -> dict[str, Any]:
        def snapshot_payload(snapshot: TimeframeSnapshot) -> dict[str, Any]:
            return {
                'timeframe': snapshot.timeframe,
                'price': snapshot.price,
                'rsi': snapshot.rsi,
                'ma_order': snapshot.ma_order,
                'hammer': snapshot.candle_pattern.hammer,
                'wick_retraction': snapshot.candle_pattern.wick_retraction,
                'pattern_bias': snapshot.candle_pattern.bias,
            }

        return {
            'asset': asset,
            'direction': signal.direction,
            'strategy': Divergence4MAStrategy.name,
            'timeframes': list(asset_context.snapshots.keys()),
            'divergences': asset_context.divergence_alignment,
            'confirmations': signal.confirmations,
            'filters': signal.filters,
            'btc_state': signal.btc_state,
            'proposed': {
                'entry': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'reason': signal.reason,
            },
            'asset_snapshots': {tf: snapshot_payload(snapshot) for tf, snapshot in asset_context.snapshots.items()},
            'btc_snapshots': {tf: snapshot_payload(snapshot) for tf, snapshot in reference_context.snapshots.items()},
            'open_positions': self._open_positions(asset),
        }

    def _open_positions(self, asset: str) -> list[dict[str, Any]]:
        symbol = self.asset_symbol_map.get(asset, asset)
        try:
            positions = self.exchange.fetch_positions(symbols=[symbol])
        except Exception:  # noqa: BLE001
            return []
        cleaned: list[dict[str, Any]] = []
        for pos in positions:
            contracts = float(pos.get('contracts') or pos.get('positionAmt') or 0.0)
            if abs(contracts) <= 0:
                continue
            cleaned.append(
                {
                    'symbol': pos.get('symbol'),
                    'contracts': contracts,
                    'entryPrice': pos.get('entryPrice') or pos.get('entry_price'),
                    'unrealizedPnl': pos.get('unrealizedPnl') or pos.get('unrealizedProfit'),
                }
            )
        return cleaned

    def _execute_trade(
        self,
        asset: str,
        signal: EntrySignal,
        decision,
        context_payload: dict[str, Any],
        response_time: float,
    ) -> None:
        open_positions = len(self.db.fetch_open_trades())
        if not self.risk_manager.can_trade(open_positions):
            self.logger.warning('Risk manager denied trade for %s (positions=%s)', asset, open_positions)
            return
        symbol = self.asset_symbol_map.get(asset, asset)
        entry_price = decision.adjustments.get('entry') if decision.adjustments else None
        entry_price = entry_price or signal.entry_price
        stop_loss = decision.stop_loss or signal.stop_loss
        take_profit = decision.take_profit or signal.take_profit
        equity = self._account_equity()
        if equity is None:
            self.logger.warning('Cannot determine account equity; skipping trade for %s', asset)
            return
        if stop_loss is None or entry_price is None:
            self.logger.warning('Missing stops for %s', asset)
            return
        try:
            size = self.risk_manager.position_size(equity, entry_price, stop_loss)
        except Exception as exc:  # noqa: BLE001
            self.logger.error('Failed to size position for %s: %s', asset, exc)
            return
        size *= max(0.0, decision.position_size_pct) / 100.0
        if size <= 0:
            self.logger.info('Zero position size for %s — skipping entry', asset)
            return
        direction = signal.direction
        try:
            if direction == 'LONG':
                self.exchange.market_buy(symbol, size)
            else:
                self.exchange.market_sell(symbol, size)
        except Exception as exc:  # noqa: BLE001
            self.logger.error('Order placement failed for %s: %s', asset, exc)
            return
        self.notifier.trade_opened(asset, direction, entry_price, stop_loss, take_profit, self.settings.default_leverage)
        asset_snapshots = context_payload['asset_snapshots']
        trade_id = self.db.record_trade(
            asset=asset,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self.settings.default_leverage,
            position_size=size,
            status='OPEN',
            signal_timeframes=signal.signal_timeframes,
            signal_type='divergence_bullish' if direction == 'LONG' else 'divergence_bearish',
            confirmation_type=signal.confirmations,
            btc_state=signal.btc_state,
            mike_decision=decision.decision,
            mike_reasoning=decision.reasoning,
            mike_response_time=response_time,
            rsi_15m=self._snapshot_value(asset_snapshots.get('15m'), 'rsi'),
            rsi_1h=self._snapshot_value(asset_snapshots.get('1h'), 'rsi'),
            rsi_4h=self._snapshot_value(asset_snapshots.get('4h'), 'rsi'),
            ma_order_4h=asset_snapshots.get('4h', {}).get('ma_order') if '4h' in asset_snapshots else None,
        )
        self.logger.info('Opened trade %s for %s size=%.4f', trade_id, asset, size)

    def _snapshot_value(self, snapshot: dict[str, Any] | None, key: str) -> float | None:
        if not snapshot:
            return None
        value = snapshot.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _account_equity(self) -> float | None:
        try:
            balance = self.exchange.fetch_balance()
            total = balance.get('total', {})
            return float(total.get('USDT'))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning('Failed to fetch equity: %s', exc)
            return None

    # Notifications ----------------------------------------------------
    def _send_summary(self, now: datetime) -> None:
        open_trades = self.db.fetch_open_trades()
        btc_price = self._fetch_price(self.reference_symbol)
        realized = self._realized_pnl()
        lines = [
            f"Open trades: {len(open_trades)}",
            f"BTC: {btc_price:.2f}" if btc_price else 'BTC: n/a',
            f"Realized P&L: {realized:.2f} USDT",
        ]
        for trade in open_trades[:5]:
            lines.append(
                f"- {trade['asset']} {trade['direction']} @ {float(trade['entry_price']):.4f} (SL {float(trade['stop_loss']):.4f})"
            )
        self.notifier.summary('4h Summary', lines)

    def _realized_pnl(self, since: datetime | None = None) -> float:
        query = "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status != 'OPEN'"
        params: Tuple[Any, ...] = ()
        if since is not None:
            query += " AND exit_time >= ?"
            params = (since.isoformat(timespec='seconds'),)
        with self.db._lock:  # type: ignore[attr-defined]
            cursor = self.db._conn.execute(query, params)  # type: ignore[attr-defined]
            value = cursor.fetchone()[0]
        return float(value or 0.0)

    def _send_daily_recap(self, now: datetime) -> None:
        since = now - self.DAILY_INTERVAL
        trades = self._closed_trades_since(since)
        total = len(trades)
        wins = sum(1 for trade in trades if (trade['pnl'] or 0) > 0)
        pnl_sum = sum(float(trade['pnl'] or 0.0) for trade in trades)
        win_rate = (wins / total) if total else 0.0
        lines = [
            f"Trades closed: {total}",
            f"Win rate: {win_rate:.1%}",
            f"P&L last 24h: {pnl_sum:.2f} USDT",
        ]
        self.notifier.daily_recap(lines)

    def _closed_trades_since(self, since: datetime) -> list:
        query = "SELECT * FROM trades WHERE status != 'OPEN' AND exit_time >= ?"
        params = (since.isoformat(timespec='seconds'),)
        with self.db._lock:  # type: ignore[attr-defined]
            cursor = self.db._conn.execute(query, params)  # type: ignore[attr-defined]
            return cursor.fetchall()


__all__ = ['TradingDaemon', 'normalize_symbol', 'asset_key']
