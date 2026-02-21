from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence, TYPE_CHECKING

from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from utils.logger import get_logger

if TYPE_CHECKING:
    from config.settings import Settings
    from db.models import TradeDatabase


class TelegramCommandBot:
    """Telegram bot providing read-only trading commands."""

    def __init__(self, exchange: Any, db: 'TradeDatabase', settings: 'Settings'):
        self.exchange = exchange
        self.db = db
        self.settings = settings
        self.logger = get_logger('telegram.bot')
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        self._thread: threading.Thread | None = None
        self._application: Application | None = None
        if not self.enabled:
            self.logger.warning('Telegram command bot disabled — missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID')

    # -----------------------------------------------------------------
    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_polling, name='TelegramCommandBot', daemon=True)
        self._thread.start()

    def run_forever(self) -> None:
        if not self.enabled or not self._application:
            self.logger.warning('Telegram command bot is disabled; nothing to run')
            return
        self._run_polling()

    def _run_polling(self) -> None:
        self.logger.info('Starting Telegram command bot …')
        try:
            # Python 3.9: threads don't have an event loop by default
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            app = ApplicationBuilder().token(self.token).build()
            self._application = app
            self._register_handlers()
            app.run_polling(stop_signals=None, allowed_updates=Update.ALL_TYPES)
        except Exception as exc:  # noqa: BLE001
            self.logger.error('Telegram bot stopped: %s', exc, exc_info=True)

    def _register_handlers(self) -> None:
        assert self._application is not None
        commands = {
            'balance': self.handle_balance,
            'positions': self.handle_positions,
            'trades': self.handle_trades,
            'stats': self.handle_stats,
            'portfolio': self.handle_portfolio,
            'help': self.handle_help,
            'start': self.handle_help,
        }
        for name, callback in commands.items():
            self._application.add_handler(CommandHandler(name, callback))

    # Command handlers ------------------------------------------------
    async def handle_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        try:
            text = await self._balance_block()
        except Exception as exc:  # noqa: BLE001
            self.logger.error('Balance command failed: %s', exc, exc_info=True)
            text = '⚠️ Unable to fetch balance right now.'
        await self._send_reply(context, update, text)

    async def handle_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        try:
            text = await self._positions_block()
        except Exception as exc:  # noqa: BLE001
            self.logger.error('Positions command failed: %s', exc, exc_info=True)
            text = '⚠️ Unable to fetch positions right now.'
        await self._send_reply(context, update, text)

    async def handle_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        limit = 10
        if context.args:
            try:
                limit = max(1, min(50, int(context.args[0])))
            except ValueError:
                pass
        try:
            text = await self._trades_block(limit)
        except Exception as exc:  # noqa: BLE001
            self.logger.error('Trades command failed: %s', exc, exc_info=True)
            text = '⚠️ Unable to fetch trade history.'
        await self._send_reply(context, update, text)

    async def handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        try:
            text = await self._stats_block()
        except Exception as exc:  # noqa: BLE001
            self.logger.error('Stats command failed: %s', exc, exc_info=True)
            text = '⚠️ Unable to compute stats right now.'
        await self._send_reply(context, update, text)

    async def handle_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        try:
            sections = [await self._balance_block(), await self._positions_block(), await self._today_block()]
            text = '\n\n'.join(section for section in sections if section)
        except Exception as exc:  # noqa: BLE001
            self.logger.error('Portfolio command failed: %s', exc, exc_info=True)
            text = '⚠️ Unable to fetch portfolio overview.'
        await self._send_reply(context, update, text)

    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        commands = [
            '/balance',
            '/positions',
            '/trades [N]',
            '/stats',
            '/portfolio',
            '/help',
        ]
        text = '🤖 Available Commands\n' + '\n'.join(f"• {cmd}" for cmd in commands)
        await self._send_reply(context, update, text)

    # Message blocks --------------------------------------------------
    async def _balance_block(self) -> str:
        balance = await asyncio.to_thread(self.exchange.fetch_balance)
        total = self._extract_balance(balance, 'total')
        available = self._extract_balance(balance, 'free')
        unrealized = self._extract_unrealized(balance)
        return (
            "💰 Balance\n"
            f"Total: {self._format_usdt(total)}\n"
            f"Available: {self._format_usdt(available)}\n"
            f"Unrealized P&L: {self._format_signed_usdt(unrealized)}"
        )

    async def _positions_block(self) -> str:
        trades = await asyncio.to_thread(self.db.fetch_open_trades)
        positions = await asyncio.to_thread(self.exchange.fetch_positions)
        positions = positions or []
        symbols = self._symbols_for_positions(trades, positions)
        price_map = await asyncio.to_thread(self._fetch_prices, symbols)
        entries: list[str] = []
        handled_symbols: set[str] = set()
        for trade in trades:
            symbol = self._normalize_symbol(trade['asset'])
            handled_symbols.add(symbol)
            entry_price = float(trade['entry_price'])
            leverage = int(trade['leverage'])
            stop_loss = float(trade['stop_loss']) if trade['stop_loss'] is not None else None
            take_profit = float(trade['take_profit']) if trade['take_profit'] is not None else None
            size = float(trade['position_size'])
            direction = str(trade['direction']).upper()
            current = price_map.get(symbol)
            pnl_value, pnl_pct = self._pnl(direction, entry_price, current, size)
            entries.append(
                self._format_position(
                    label=self._display_symbol(symbol),
                    direction=direction,
                    entry=entry_price,
                    current=current,
                    pnl_value=pnl_value,
                    pnl_pct=pnl_pct,
                    leverage=leverage,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
            )
        for pos in positions:
            symbol = self._normalize_symbol(pos.get('symbol') or pos.get('info', {}).get('symbol'))
            if not symbol or symbol in handled_symbols:
                continue
            contracts = float(pos.get('contracts') or pos.get('positionAmt') or 0.0)
            if abs(contracts) <= 0:
                continue
            direction = 'LONG' if contracts > 0 else 'SHORT'
            entry_price = float(pos.get('entryPrice') or pos.get('entry_price') or 0.0)
            leverage = int(float(pos.get('leverage') or self.settings.default_leverage))
            current = price_map.get(symbol)
            pnl_value = float(pos.get('unrealizedPnl') or pos.get('unrealizedProfit') or 0.0)
            pnl_pct = self._percent(entry_price, current, direction)
            entries.append(
                self._format_position(
                    label=self._display_symbol(symbol),
                    direction=direction,
                    entry=entry_price or None,
                    current=current,
                    pnl_value=pnl_value,
                    pnl_pct=pnl_pct,
                    leverage=leverage,
                    stop_loss=None,
                    take_profit=None,
                )
            )
        header = f"📊 Open Positions ({len(entries)})"
        if not entries:
            return header + "\n\nNo open positions."
        return header + '\n\n' + '\n\n'.join(entries)

    async def _trades_block(self, limit: int) -> str:
        trades = await asyncio.to_thread(self.db.fetch_closed_trades, limit)
        if not trades:
            return f"📜 Last {limit} Trades\n\nNo closed trades recorded."
        lines = [f"📜 Last {limit} Trades", ""]
        for idx, trade in enumerate(trades, start=1):
            pnl = float(trade['pnl'] or 0.0)
            entry = float(trade['entry_price'])
            exit_price = float(trade['exit_price'] or trade['entry_price'])
            pct = float(trade['pnl_pct'] or 0.0) * 100
            direction = str(trade['direction']).upper()
            asset = self._short_asset_name(str(trade['asset']))
            icon = '✅' if pnl >= 0 else '❌'
            date_str = self._format_date(trade['exit_time'] or trade['entry_time'])
            lines.append(
                f"{idx}. {icon} {direction} {asset} @ {entry:.2f} → {exit_price:.2f} | "
                f"{self._format_signed_usd(pnl)} ({pct:+.2f}%) | {date_str}"
            )
        return '\n'.join(lines)

    async def _stats_block(self) -> str:
        stats = await asyncio.to_thread(self.db.fetch_trade_stats)
        today = await self._period_summary(self._start_of_day())
        week = await self._period_summary(datetime.now(timezone.utc) - timedelta(days=7))
        month = await self._period_summary(datetime.now(timezone.utc) - timedelta(days=30))
        lines = [
            '📈 Trading Stats',
            '',
            f"Total trades: {stats['total']}",
            f"Win rate: {stats['win_rate'] * 100:.1f}% ({stats['wins']}W / {stats['losses']}L)",
            f"Total P&L: {self._format_signed_usd(stats['total_pnl'])}",
            f"Best trade: {self._format_trade_summary(stats.get('best'))}",
            f"Worst trade: {self._format_trade_summary(stats.get('worst'))}",
            f"Avg P&L: {self._format_signed_usd(stats['avg'])}",
            '',
            self._format_period_line('Today', today),
            self._format_period_line('This week', week),
            self._format_period_line('This month', month),
        ]
        return '\n'.join(lines)

    async def _today_block(self) -> str:
        pnl, count = await self._period_summary(self._start_of_day())
        return f"🗓 Today: {self._format_signed_usd(pnl)} ({count} trades)"

    # Utility ---------------------------------------------------------
    async def _period_summary(self, since: datetime) -> tuple[float, int]:
        since_str = self._iso_timestamp(since)
        trades = await asyncio.to_thread(self.db.fetch_trades_since, since_str)
        pnl = sum(float(trade['pnl'] or 0.0) for trade in trades)
        return pnl, len(trades)

    def _start_of_day(self) -> datetime:
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _iso_timestamp(self, dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        utc_dt = dt.astimezone(timezone.utc)
        return utc_dt.replace(tzinfo=None).isoformat(timespec='seconds')

    def _symbols_for_positions(self, trades: Sequence[Any], positions: Sequence[Any]) -> set[str]:
        symbols: set[str] = set()
        for trade in trades:
            symbols.add(self._normalize_symbol(trade['asset']))
        for pos in positions:
            symbol = self._normalize_symbol(pos.get('symbol') or pos.get('info', {}).get('symbol'))
            if symbol:
                symbols.add(symbol)
        return {sym for sym in symbols if sym}

    def _fetch_prices(self, symbols: Iterable[str]) -> dict[str, float | None]:
        prices: dict[str, float | None] = {}
        for symbol in symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
            except Exception:  # noqa: BLE001
                self.logger.warning('Failed to fetch ticker for %s', symbol)
                prices[symbol] = None
                continue
            value = ticker.get('last') or ticker.get('close')
            prices[symbol] = float(value) if value is not None else None
        return prices

    def _format_position(
        self,
        *,
        label: str,
        direction: str,
        entry: float | None,
        current: float | None,
        pnl_value: float,
        pnl_pct: float,
        leverage: int,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> str:
        entry_str = f"${entry:.2f}" if entry is not None else 'n/a'
        current_str = f"${current:.2f}" if current is not None else 'n/a'
        sl_str = f"${stop_loss:.2f}" if stop_loss is not None else 'n/a'
        tp_str = f"${take_profit:.2f}" if take_profit is not None else 'n/a'
        return (
            f"{direction} {label}\n"
            f"Entry: {entry_str} | Current: {current_str}\n"
            f"P&L: {self._format_signed_usd(pnl_value)} ({pnl_pct:+.2f}%) | {leverage}x\n"
            f"SL: {sl_str} | TP: {tp_str}"
        )

    def _pnl(self, direction: str, entry: float | None, current: float | None, size: float) -> tuple[float, float]:
        if entry in (None, 0.0) or current is None:
            return 0.0, 0.0
        if direction == 'LONG':
            pnl_value = (current - entry) * size
            pct = ((current - entry) / entry) * 100 if entry else 0.0
        else:
            pnl_value = (entry - current) * size
            pct = ((entry - current) / entry) * 100 if entry else 0.0
        return pnl_value, pct

    def _percent(self, entry: float | None, current: float | None, direction: str) -> float:
        if entry in (None, 0.0) or current is None:
            return 0.0
        if direction == 'LONG':
            return ((current - entry) / entry) * 100
        return ((entry - current) / entry) * 100

    def _extract_balance(self, balance: dict[str, Any], key: str) -> float:
        bucket = balance.get(key) or {}
        value = bucket.get('USDT')
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _extract_unrealized(self, balance: dict[str, Any]) -> float:
        info = balance.get('info') or {}
        value = info.get('totalUnrealizedProfit') or info.get('unrealizedProfit')
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _format_usdt(self, value: float) -> str:
        return f"{value:,.2f} USDT"

    def _format_signed_usdt(self, value: float) -> str:
        sign = '+' if value >= 0 else '-'
        return f"{sign}{abs(value):,.2f} USDT"

    def _format_signed_usd(self, value: float) -> str:
        sign = '+' if value >= 0 else '-'
        return f"{sign}${abs(value):,.2f}"

    def _display_symbol(self, symbol: str) -> str:
        if not symbol:
            return symbol
        return symbol if '/' in symbol else symbol.replace('USDT', '/USDT')

    def _short_asset_name(self, asset: str) -> str:
        if '/' in asset:
            return asset.split('/', 1)[0]
        if asset.endswith('USDT'):
            return asset[:-4]
        return asset

    def _format_trade_summary(self, trade: dict[str, Any] | None) -> str:
        if not trade:
            return 'n/a'
        direction = trade.get('direction') or 'n/a'
        asset = trade.get('asset') or 'n/a'
        display = self._display_symbol(asset) or asset
        return f"{self._format_signed_usd(trade['pnl'])} ({direction} {display})"

    def _format_period_line(self, label: str, data: tuple[float, int]) -> str:
        pnl, count = data
        return f"{label}: {self._format_signed_usd(pnl)} ({count} trades)"

    def _format_date(self, value: str) -> str:
        try:
            dt = datetime.fromisoformat(value)
        except Exception:  # noqa: BLE001
            return value
        return dt.strftime('%b %d')

    def _normalize_symbol(self, symbol: str | None) -> str:
        if not symbol:
            return ''
        symbol = symbol.upper().replace(' ', '')
        if '/' in symbol:
            return symbol
        if symbol.endswith(':USDT'):
            base = symbol[:-5]
            return f"{base}/USDT"
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}/USDT"
        return symbol

    def _is_authorized(self, update: Update) -> bool:
        if not self.enabled:
            return False
        chat = update.effective_chat
        if not chat:
            return False
        if str(chat.id) != str(self.chat_id):
            self.logger.warning('Ignoring unauthorized chat id: %s', chat.id)
            return False
        return True

    async def _send_reply(self, context: ContextTypes.DEFAULT_TYPE, update: Update, text: str) -> None:
        if update.effective_message:
            await update.effective_message.reply_text(text)
            return
        if update.effective_chat:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


__all__ = ['TelegramCommandBot']
