"""
Profit Vault — Tiered Profit Skimming
======================================
After each winning trade, a percentage of the profit is moved to a virtual
"vault" that the bot won't use for position sizing. As cumulative gains grow,
the skim rate increases to protect more profits.

Tiers:
  - Cumulative P&L 0-10%  → skim 20% of each win
  - Cumulative P&L 10-25% → skim 30% of each win
  - Cumulative P&L 25%+   → skim 40% of each win
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger("vault")

# Tiered skim rates: (cumulative_pnl_pct_threshold, skim_rate)
TIERS = [
    (0.25, 0.40),   # 25%+ cumulative gain → skim 40%
    (0.10, 0.30),   # 10-25% cumulative gain → skim 30%
    (0.00, 0.20),   # 0-10% cumulative gain → skim 20%
]


class ProfitVault:
    """Manages the virtual profit vault in SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path("data/trades.db")
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS vault_transactions (
                id INTEGER PRIMARY KEY,
                trade_id INTEGER REFERENCES trades(id),
                amount REAL NOT NULL,
                skim_rate REAL NOT NULL,
                trade_pnl REAL NOT NULL,
                cumulative_pnl_pct REAL NOT NULL,
                tier_label TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS vault_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initial_capital REAL NOT NULL,
                vault_balance REAL NOT NULL DEFAULT 0.0,
                total_skimmed REAL NOT NULL DEFAULT 0.0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def initialize(self, initial_capital: float) -> None:
        """Set initial capital (call once at bot startup if not set)."""
        row = self.conn.execute("SELECT id FROM vault_state WHERE id = 1").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO vault_state (id, initial_capital, vault_balance, total_skimmed) VALUES (1, ?, 0.0, 0.0)",
                (initial_capital,),
            )
            self.conn.commit()
            logger.info("Vault initialized with capital: %.2f USDT", initial_capital)

    def get_state(self) -> dict[str, Any]:
        """Return current vault state."""
        row = self.conn.execute("SELECT * FROM vault_state WHERE id = 1").fetchone()
        if row is None:
            return {"initial_capital": 0, "vault_balance": 0, "total_skimmed": 0, "trading_capital": 0}
        state = dict(row)
        state["trading_capital"] = state["initial_capital"] + self._realized_pnl() - state["vault_balance"]
        return state

    def process_trade_close(self, trade_id: int, pnl: float) -> float:
        """
        Called when a trade closes profitably.
        Returns the amount skimmed to the vault (0 if trade was a loss).
        """
        if pnl <= 0:
            logger.debug("Trade %s closed with loss (%.2f) — no skim", trade_id, pnl)
            return 0.0

        state = self.conn.execute("SELECT * FROM vault_state WHERE id = 1").fetchone()
        if state is None:
            logger.warning("Vault not initialized — skipping skim")
            return 0.0

        initial_capital = state["initial_capital"]
        if initial_capital <= 0:
            return 0.0

        cumulative_pnl_pct = self._realized_pnl() / initial_capital
        skim_rate = self._tier_rate(cumulative_pnl_pct)
        tier_label = self._tier_label(cumulative_pnl_pct)
        skim_amount = pnl * skim_rate

        # Update vault balance
        self.conn.execute(
            "UPDATE vault_state SET vault_balance = vault_balance + ?, total_skimmed = total_skimmed + ?, updated_at = ? WHERE id = 1",
            (skim_amount, skim_amount, datetime.now(timezone.utc).isoformat()),
        )

        # Record transaction
        self.conn.execute(
            "INSERT INTO vault_transactions (trade_id, amount, skim_rate, trade_pnl, cumulative_pnl_pct, tier_label) VALUES (?, ?, ?, ?, ?, ?)",
            (trade_id, skim_amount, skim_rate, pnl, cumulative_pnl_pct, tier_label),
        )
        self.conn.commit()

        logger.info(
            "Vault skim: $%.2f (%.0f%% of $%.2f profit) | Tier: %s | Vault total: $%.2f",
            skim_amount, skim_rate * 100, pnl, tier_label,
            state["vault_balance"] + skim_amount,
        )
        return skim_amount

    def trading_equity(self, total_equity: float) -> float:
        """Return equity available for trading (total minus vault)."""
        state = self.get_state()
        vault = state.get("vault_balance", 0.0)
        available = total_equity - vault
        return max(0.0, available)

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent vault transactions."""
        rows = self.conn.execute(
            "SELECT * FROM vault_transactions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _realized_pnl(self) -> float:
        """Sum of all closed trade PnL."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE status IN ('CLOSED', 'STOPPED')"
        ).fetchone()
        return float(row["total"]) if row else 0.0

    def _tier_rate(self, cumulative_pnl_pct: float) -> float:
        for threshold, rate in TIERS:
            if cumulative_pnl_pct >= threshold:
                return rate
        return TIERS[-1][1]  # fallback to lowest tier

    def _tier_label(self, cumulative_pnl_pct: float) -> str:
        pct = cumulative_pnl_pct * 100
        if pct >= 25:
            return f"Tier 3 (≥25% gain, skim 40%) — at {pct:.1f}%"
        elif pct >= 10:
            return f"Tier 2 (10-25% gain, skim 30%) — at {pct:.1f}%"
        else:
            return f"Tier 1 (<10% gain, skim 20%) — at {pct:.1f}%"

    def close(self) -> None:
        self.conn.close()
