from __future__ import annotations

import os
import textwrap
from datetime import datetime
from typing import Any

import requests

from utils.logger import get_logger


class Notifier:
    """Telegram helper for real-time and periodic alerts."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        parse_mode: str = "Markdown",
    ):
        token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        chat = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.bot_token = token
        self.chat_id = chat
        self.parse_mode = parse_mode
        self.enabled = bool(token and chat)
        self.logger = get_logger("notifier")
        if not self.enabled:
            self.logger.warning("Telegram notifier disabled — missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    def _send(self, text: str) -> None:
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": textwrap.dedent(text).strip(), "parse_mode": self.parse_mode}
        try:
            response = requests.post(url, json=payload, timeout=10)
            if not response.ok:
                self.logger.warning("Telegram send failed: %s", response.text)
        except Exception as exc:
            self.logger.warning("Telegram exception: %s", exc)

    # Event helpers -----------------------------------------------------
    def signal_detected(self, asset: str, direction: str, reason: str) -> None:
        self._send(f"🔍 Signal: *{direction} {asset}* — {reason}\nWaiting for Mike's decision …")

    def mike_decision(self, asset: str, decision: str, reasoning: str) -> None:
        icon = {
            "APPROVE": "✅",
            "REJECT": "❌",
            "MODIFY": "✏️",
        }.get(decision.upper(), "ℹ️")
        self._send(f"{icon} Mike {decision.upper()} {asset}\n{reasoning}")

    def trade_opened(self, asset: str, direction: str, entry: float, sl: float, tp: float, leverage: int) -> None:
        self._send(
            f"📈 OPENED {direction} {asset} @ {entry:.4f}\nSL: {sl:.4f} | TP: {tp:.4f} | {leverage}x"
        )

    def trade_closed(self, asset: str, direction: str, exit_price: float, pnl: float, pnl_pct: float) -> None:
        icon = "💰" if pnl >= 0 else "🔻"
        self._send(f"{icon} CLOSED {direction} {asset} @ {exit_price:.4f}\nP&L: {pnl:.2f} ({pnl_pct:.2%})")

    def stop_loss_moved(self, asset: str, new_sl: float) -> None:
        self._send(f"🔒 {asset} SL moved to {new_sl:.4f}")

    def emergency_exit(self, reason: str) -> None:
        self._send(f"🚨 Emergency exit triggered\n{reason}")

    def summary(self, title: str, body_lines: list[str]) -> None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        body = "\n".join(body_lines)
        self._send(f"📊 *{title}* ({timestamp})\n{body}")

    def daily_recap(self, body_lines: list[str]) -> None:
        self.summary("Daily Recap", body_lines)
