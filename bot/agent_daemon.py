"""
Agent-Aware Trading Daemon
============================
Wraps TradingDaemon with agent-specific configuration.
Each agent runs as a separate process with its own DB, vault,
capital allocation, and strategy.

Usage:
    python main.py agent mike
    python main.py agent charlie
"""

from __future__ import annotations

import importlib
import os
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
from bot.vault import ProfitVault
from config.agent_config import AgentConfig, load_agent_config
from db.models import TradeDatabase
from utils.logger import get_logger


def normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if "/" not in s and s.endswith("USDT"):
        s = s[: -4] + "/USDT"
    return s


def asset_key(symbol: str) -> str:
    return symbol.replace("/", "").upper()


class AgentDaemon:
    """Agent-isolated trading daemon. Each agent gets its own everything."""

    LOOP_SLEEP_SECONDS = 5
    MINUTE_INTERVAL = timedelta(seconds=60)
    ANALYSIS_INTERVAL = timedelta(minutes=15)
    SUMMARY_INTERVAL = timedelta(hours=4)
    DAILY_INTERVAL = timedelta(hours=24)

    def __init__(self, agent_config: AgentConfig):
        self.config = agent_config
        self.logger = get_logger(f"daemon.{agent_config.agent_id}")
        self.logger.info(
            "%s %s daemon initializing (strategy=%s, capital=%.0f)",
            agent_config.emoji, agent_config.name,
            agent_config.strategy, agent_config.capital,
        )

        # Exchange (shared Binance account)
        self.exchange = BinanceFuturesTestnet(agent_config.api_key, agent_config.api_secret)

        # Agent-specific notifier (prefixed with emoji + name)
        self.notifier = AgentNotifier(agent_config)

        # Agent-specific advisor
        self.advisor = Advisor(
            exchange=self.exchange,
            timeframe="15m",
            model=agent_config.model,
            enabled=agent_config.mike_enabled,
            log_path=agent_config.decisions_log,
        )
        # Inject agent-specific strategy prompt
        self._inject_strategy_prompt(agent_config)

        # Agent-specific DB and vault
        self.db = TradeDatabase(db_path=agent_config.db_path)
        self.vault = ProfitVault(db_path=agent_config.db_path)

        # Assets
        self.reference_symbol = normalize_symbol(agent_config.reference_symbol)
        seen: Dict[str, None] = {}
        self.assets = []
        for a in agent_config.assets:
            normalized = normalize_symbol(a)
            if normalized not in seen:
                seen[normalized] = None
                self.assets.append(normalized)
        self.asset_symbol_map = {a: a for a in self.assets}

        # Multi-timeframe analyzer
        self.multi_analyzer = MultiTimeframeAnalyzer(
            exchange=self.exchange,
            timeframes=["15m", "1h", "4h", "1d"],
            ma_periods=(9, 21, 45, 100),
            limit=360,
        )

        # Strategy — load dynamically
        self.strategies = self._load_strategies(agent_config)

        # Risk manager
        self.risk_manager = RiskManager(
            RiskLimits(
                risk_per_trade=agent_config.risk_per_trade,
                max_positions=agent_config.max_positions,
                max_daily_loss=agent_config.max_daily_loss,
            )
        )

        # Timing
        now = datetime.now(timezone.utc)
        self._last_minute_check = now - self.MINUTE_INTERVAL
        self._next_analysis = self._next_boundary(now, self.ANALYSIS_INTERVAL)
        self._next_summary = now + self.SUMMARY_INTERVAL
        self._next_daily_recap = now + self.DAILY_INTERVAL
        self._processed_signals: Dict[Tuple[str, str], str | None] = {}

        self.logger.info(
            "%s Configured assets: %s",
            agent_config.emoji, ", ".join(self.assets),
        )

    def _inject_strategy_prompt(self, config: AgentConfig) -> None:
        """Override advisor prompts with agent-specific strategy prompt."""
        if not config.strategy_prompt:
            return
        try:
            mod = importlib.import_module(config.strategy_prompt)
            context = getattr(mod, "STRATEGY_CONTEXT", None)
            if context:
                self.advisor._system_prompt = lambda: context
                self.advisor._signal_system_prompt = lambda: context
                self.logger.info("Loaded strategy prompt from %s", config.strategy_prompt)
        except Exception as exc:
            self.logger.warning("Failed to load strategy prompt %s: %s", config.strategy_prompt, exc)

    def _load_strategies(self, config: AgentConfig) -> dict:
        """Load the correct strategy class for this agent."""
        if config.strategy == "divergence_4ma":
            from strategies.divergence_4ma import Divergence4MAStrategy
            return {
                asset: Divergence4MAStrategy(reference_symbol=self.reference_symbol)
                for asset in self.assets
            }
        elif config.strategy == "charlie_strategy":
            from strategies.supply_demand import SupplyDemandStrategy
            return {
                asset: SupplyDemandStrategy(reference_symbol=self.reference_symbol)
                for asset in self.assets
            }
        else:
            self.logger.error("Unknown strategy: %s", config.strategy)
            return {}

    def _next_boundary(self, now: datetime, delta: timedelta) -> datetime:
        seconds = int(delta.total_seconds())
        if seconds <= 0:
            return now + delta
        epoch = int(now.timestamp())
        next_epoch = ((epoch // seconds) + 1) * seconds
        return datetime.fromtimestamp(next_epoch, tz=timezone.utc)

    def _bootstrap_leverage(self) -> None:
        for asset, symbol in self.asset_symbol_map.items():
            try:
                self.exchange.set_leverage(symbol, self.config.leverage)
                self.logger.info("Leverage set for %s (%sx)", asset, self.config.leverage)
            except Exception as exc:
                self.logger.warning("Failed to set leverage for %s: %s", asset, exc)

    def _account_equity(self) -> float | None:
        """Return this agent's allocated capital (not full account balance)."""
        try:
            balance = self.exchange.fetch_balance()
            total = float(balance.get("total", {}).get("USDT", 0))
            # Each agent only sees their allocated capital
            return min(total, self.config.capital)
        except Exception as exc:
            self.logger.warning("Failed to fetch equity: %s", exc)
            return None

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.logger.info(
            "%s Starting %s trading daemon …",
            self.config.emoji, self.config.name,
        )
        self._bootstrap_leverage()

        # Initialize vault
        equity = self._account_equity()
        if equity is not None:
            self.vault.initialize(equity)

        self.logger.info(
            "Entering main loop. Next analysis at %s",
            self._next_analysis.strftime("%H:%M:%S"),
        )

        while True:
            now = datetime.now(timezone.utc)
            try:
                if now - self._last_minute_check >= self.MINUTE_INTERVAL:
                    self._last_minute_check = now
                    self._minute_cycle()

                if now >= self._next_analysis:
                    self.logger.info(
                        "%s Running analysis cycle @ %s",
                        self.config.emoji, now.strftime("%H:%M"),
                    )
                    self._analysis_cycle(now)
                    self._next_analysis = self._next_boundary(now, self.ANALYSIS_INTERVAL)
                    self.logger.info(
                        "Next analysis at %s", self._next_analysis.strftime("%H:%M:%S"),
                    )

                if now >= self._next_summary:
                    self._send_summary(now)
                    self._next_summary = now + self.SUMMARY_INTERVAL

                if now >= self._next_daily_recap:
                    self._send_daily_recap(now)
                    self._next_daily_recap = now + self.DAILY_INTERVAL

            except Exception as exc:
                self.logger.error("Daemon loop error: %s", exc, exc_info=True)

            time.sleep(self.LOOP_SLEEP_SECONDS)

    # Minute tasks -----------------------------------------------------
    def _minute_cycle(self) -> None:
        self._check_btc_emergency()
        self._monitor_open_positions()

    def _check_btc_emergency(self) -> None:
        try:
            candles = self.exchange.fetch_ohlcv(self.reference_symbol, timeframe="1m", limit=5)
        except Exception as exc:
            self.logger.warning("Failed to fetch BTC candles: %s", exc)
            return
        if not candles:
            return
        first_open = candles[0][1]
        overall_high = max(c[2] for c in candles)
        overall_low = min(c[3] for c in candles)
        reference = max(first_open, 1e-8)
        range_pct = (overall_high - overall_low) / reference
        if range_pct >= 0.03:
            reason = f"BTC 5m range {range_pct:.2%} >= 3%"
            self.logger.warning("%s BTC emergency: %s", self.config.emoji, reason)
            self._exit_all_alt_positions(reason)

    def _monitor_open_positions(self) -> None:
        open_trades = self.db.fetch_open_trades()
        for trade in open_trades:
            asset = trade["asset"]
            symbol = self.asset_symbol_map.get(asset, asset)
            price = self._fetch_price(symbol)
            if price is None:
                continue
            direction = str(trade["direction"]).upper()
            entry_price = float(trade["entry_price"])
            stop_loss = float(trade["stop_loss"])
            take_profit = trade["take_profit"]
            tp = float(take_profit) if take_profit is not None else None

            # Check stop loss
            if direction == "LONG" and price <= stop_loss:
                self._close_trade(trade, price, "stop_loss_hit", status="STOPPED")
            elif direction == "SHORT" and price >= stop_loss:
                self._close_trade(trade, price, "stop_loss_hit", status="STOPPED")
            # Check take profit
            elif tp is not None:
                if direction == "LONG" and price >= tp:
                    self._close_trade(trade, price, "take_profit_hit")
                elif direction == "SHORT" and price <= tp:
                    self._close_trade(trade, price, "take_profit_hit")
            # Move SL to breakeven
            else:
                self._maybe_move_breakeven(trade, price, entry_price, direction)

    def _maybe_move_breakeven(self, trade, price, entry_price, direction) -> None:
        trade_id = trade["id"]
        stop_loss = float(trade["stop_loss"])
        if stop_loss >= entry_price and direction == "LONG":
            return  # Already at breakeven
        if stop_loss <= entry_price and direction == "SHORT":
            return
        if direction == "LONG":
            risk = entry_price - stop_loss
            if risk > 0 and price - entry_price >= risk:
                self._move_stop_to_breakeven(trade, entry_price)
        else:
            risk = stop_loss - entry_price
            if risk > 0 and entry_price - price >= risk:
                self._move_stop_to_breakeven(trade, entry_price)

    def _move_stop_to_breakeven(self, trade, breakeven: float) -> None:
        trade_id = trade["id"]
        asset = trade["asset"]
        symbol = self.asset_symbol_map.get(asset, asset)
        self.db.update_trade(trade_id, stop_loss=breakeven)
        self.db.log_trade_update(trade_id, "sl_moved", {"new_sl": breakeven})
        direction = str(trade["direction"]).upper()
        sl_side = "sell" if direction == "LONG" else "buy"
        try:
            self.exchange.place_stop_loss(symbol, sl_side, float(trade["position_size"]), breakeven)
        except Exception as exc:
            self.logger.warning("Failed to update exchange SL for %s: %s", asset, exc)
        self.notifier.stop_loss_moved(asset, breakeven)
        self.logger.info("Moved SL to breakeven for %s (trade %s)", asset, trade_id)

    def _close_trade(self, trade, exit_price, reason, status="CLOSED", place_order=True) -> None:
        trade_id = trade["id"]
        asset = trade["asset"]
        symbol = self.asset_symbol_map.get(asset, asset)
        direction = str(trade["direction"]).upper()
        position_size = float(trade["position_size"])
        side = "sell" if direction == "LONG" else "buy"
        if place_order:
            try:
                self.exchange.close_position(symbol, side, position_size)
            except Exception as exc:
                self.logger.warning("Failed to close %s on exchange: %s", asset, exc)
        entry_price = float(trade["entry_price"])
        if direction == "LONG":
            pnl = (exit_price - entry_price) * position_size
        else:
            pnl = (entry_price - exit_price) * position_size
        pnl_pct = ((exit_price - entry_price) / entry_price) if direction == "LONG" else ((entry_price - exit_price) / entry_price)
        self.db.update_trade(
            trade_id, status=status, exit_price=exit_price,
            exit_time=datetime.utcnow().isoformat(timespec="seconds"),
            pnl=pnl, pnl_pct=pnl_pct,
        )
        self.db.log_trade_update(trade_id, reason, {"exit_price": exit_price})
        self.risk_manager.update_daily_pnl(pnl)
        skim = self.vault.process_trade_close(trade_id, pnl)
        if skim > 0:
            vault_state = self.vault.get_state()
            self.notifier.vault_skim(asset, skim, pnl, vault_state["vault_balance"])
        self.notifier.trade_closed(asset, direction, exit_price, pnl, pnl_pct)
        self.logger.info("Trade %s closed (%s) at %.4f | pnl=%.2f", trade_id, reason, exit_price, pnl)

    def _fetch_price(self, symbol: str) -> float | None:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            last = ticker.get("last") or ticker.get("close")
            return float(last) if last is not None else None
        except Exception as exc:
            self.logger.warning("Failed to fetch price for %s: %s", symbol, exc)
            return None

    def _exit_all_alt_positions(self, reason: str) -> None:
        open_trades = self.db.fetch_open_trades()
        if not open_trades:
            return
        self.notifier.emergency_exit(reason)
        for trade in open_trades:
            asset = trade["asset"]
            symbol = self.asset_symbol_map.get(asset, asset)
            price = self._fetch_price(symbol)
            if price is not None:
                self._close_trade(trade, price, "emergency_exit", status="EMERGENCY")

    # Analysis ---------------------------------------------------------
    def _analysis_cycle(self, now: datetime) -> None:
        if not self.strategies:
            return
        # Fetch BTC reference context
        ref_context = None
        try:
            ref_context = self.multi_analyzer.analyze(self.reference_symbol)
        except Exception as exc:
            self.logger.warning("Failed to analyze BTC: %s", exc)

        for asset in self.assets:
            try:
                self._analyze_and_signal(asset, ref_context)
            except Exception as exc:
                self.logger.error("Analysis error for %s: %s", asset, exc)

    def _analyze_and_signal(self, asset: str, ref_context) -> None:
        symbol = self.asset_symbol_map.get(asset, asset)
        strategy = self.strategies.get(asset)
        if strategy is None:
            return

        asset_context = self.multi_analyzer.analyze(symbol)
        strategy.update_context(asset_context, ref_context)

        # Create a simple DataFrame for should_enter check
        entry_snapshot = asset_context.snapshots.get("15m")
        if entry_snapshot is None:
            return
        df = entry_snapshot.df

        if strategy.should_enter(df):
            signal = strategy.last_long_signal or strategy.last_short_signal
            if signal is None:
                return
            self._process_signal(asset, signal, asset_context, ref_context)

    def _process_signal(self, asset, signal, asset_context, ref_context) -> None:
        self.notifier.signal_detected(asset, signal.direction, signal.reason)

        # Build context for advisor
        context = {
            "asset": asset,
            "direction": signal.direction,
            "strategy": self.config.strategy,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "reason": signal.reason,
            "confirmations": signal.confirmations,
            "filters": signal.filters,
            "signal_timeframes": signal.signal_timeframes,
            "btc_state": signal.btc_state,
            "open_positions": len(self.db.fetch_open_trades()),
        }

        # Add snapshot data
        asset_snapshots = {}
        for tf, snap in asset_context.snapshots.items():
            asset_snapshots[tf] = {
                "price": snap.price,
                "rsi": snap.rsi,
                "ma_order": snap.ma_order,
                "sma_9": snap.latest.get("sma_9"),
                "sma_21": snap.latest.get("sma_21"),
                "sma_45": snap.latest.get("sma_45"),
                "sma_100": snap.latest.get("sma_100"),
            }
        context["asset_snapshots"] = asset_snapshots
        context["timeframes"] = list(asset_snapshots.keys())

        # Ask advisor
        start = time.time()
        decision = self.advisor.review_signal(context)
        response_time = time.time() - start

        self.notifier.mike_decision(asset, decision.decision, decision.reasoning)

        if decision.decision in ("APPROVE", "GO", "MODIFY"):
            self._execute_trade(asset, signal, decision, context, response_time)

    def _execute_trade(self, asset, signal, decision, context, response_time) -> None:
        open_positions = len(self.db.fetch_open_trades())
        if not self.risk_manager.can_trade(open_positions):
            self.logger.warning("Risk manager denied trade for %s", asset)
            return

        symbol = self.asset_symbol_map.get(asset, asset)
        entry_price = (decision.adjustments or {}).get("entry") or signal.entry_price
        stop_loss = decision.stop_loss or signal.stop_loss
        take_profit = decision.take_profit or signal.take_profit

        raw_equity = self._account_equity()
        if raw_equity is None:
            self.logger.warning("Cannot determine equity; skipping %s", asset)
            return
        equity = self.vault.trading_equity(raw_equity)
        if equity <= 0:
            self.logger.warning("No trading equity (all in vault); skipping %s", asset)
            return

        if stop_loss is None or entry_price is None:
            self.logger.warning("Missing stops for %s", asset)
            return

        try:
            size = self.risk_manager.position_size(equity, entry_price, stop_loss)
        except Exception as exc:
            self.logger.error("Failed to size position for %s: %s", asset, exc)
            return

        size *= max(0.0, decision.position_size_pct) / 100.0
        if size <= 0:
            self.logger.info("Zero position size for %s — skipping", asset)
            return

        direction = signal.direction
        try:
            if direction == "LONG":
                self.exchange.market_buy(symbol, size)
            else:
                self.exchange.market_sell(symbol, size)
        except Exception as exc:
            self.logger.error("Order placement failed for %s: %s", asset, exc)
            return

        sl_side = "sell" if direction == "LONG" else "buy"
        try:
            self.exchange.place_stop_loss(symbol, sl_side, size, stop_loss)
        except Exception as exc:
            self.logger.warning("Failed to place exchange SL for %s: %s", asset, exc)

        self.notifier.trade_opened(
            asset, direction, entry_price, stop_loss, take_profit, self.config.leverage,
        )

        asset_snapshots = context.get("asset_snapshots", {})
        trade_id = self.db.record_trade(
            asset=asset, direction=direction, entry_price=entry_price,
            stop_loss=stop_loss, take_profit=take_profit,
            leverage=self.config.leverage, position_size=size, status="OPEN",
            signal_timeframes=signal.signal_timeframes,
            signal_type="divergence_bullish" if direction == "LONG" else "divergence_bearish",
            confirmation_type=signal.confirmations, btc_state=signal.btc_state,
            mike_decision=decision.decision, mike_reasoning=decision.reasoning,
            mike_response_time=response_time,
            rsi_15m=self._snapshot_val(asset_snapshots.get("15m"), "rsi"),
            rsi_1h=self._snapshot_val(asset_snapshots.get("1h"), "rsi"),
            rsi_4h=self._snapshot_val(asset_snapshots.get("4h"), "rsi"),
            ma_order_4h=asset_snapshots.get("4h", {}).get("ma_order"),
        )
        self.logger.info("Opened trade %s for %s size=%.4f", trade_id, asset, size)

    def _snapshot_val(self, snap: dict | None, key: str) -> float | None:
        if not snap:
            return None
        v = snap.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # Summary / Recap --------------------------------------------------
    def _send_summary(self, now: datetime) -> None:
        open_trades = self.db.fetch_open_trades()
        closed = self.db.fetch_closed_trades(limit=5)
        lines = [
            f"Open positions: {len(open_trades)}",
            f"Recent closed: {len(closed)}",
        ]
        vault = self.vault.get_state()
        lines.append(f"Vault: ${vault['vault_balance']:.2f}")
        self.notifier.summary(f"{self.config.name} 4h Summary", lines)

    def _send_daily_recap(self, now: datetime) -> None:
        stats = self.db.fetch_trade_stats()
        lines = [
            f"Total trades: {stats.get('total_trades', 0)}",
            f"Win rate: {stats.get('win_rate', 0):.1f}%",
            f"Total P&L: ${stats.get('total_pnl', 0):.2f}",
        ]
        vault = self.vault.get_state()
        lines.append(f"Vault: ${vault['vault_balance']:.2f}")
        self.notifier.summary(f"{self.config.name} Daily Recap", lines)


class AgentNotifier(Notifier):
    """Notifier that prefixes messages with agent emoji + name."""

    def __init__(self, config: AgentConfig):
        super().__init__(bot_token=config.telegram_bot_token, chat_id=config.telegram_chat_id)
        self.prefix = f"{config.emoji} {config.name}"

    def _send(self, text: str) -> None:
        super()._send(f"{self.prefix} | {text}")


def run_agent(agent_id: str) -> None:
    """Entry point to run a single agent daemon."""
    config = load_agent_config(agent_id)
    # Set log file env for the logger
    os.environ["BOT_LOG_FILE"] = config.log_file
    daemon = AgentDaemon(config)
    daemon.run()
