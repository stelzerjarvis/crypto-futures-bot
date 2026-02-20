from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trades.db"


class TradeDatabase:
    """Lightweight SQLite wrapper for persisting trades and market snapshots."""

    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_PATH.read_text())
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record_trade(self, **fields: Any) -> int:
        data = self._normalize_trade_fields(fields)
        return self._insert("trades", data)

    def update_trade(self, trade_id: int, **fields: Any) -> None:
        if not fields:
            return
        columns = ", ".join(f"{key} = ?" for key in fields.keys())
        values = [self._normalize_value(value) for value in fields.values()]
        values.append(trade_id)
        with self._lock:
            self._conn.execute(f"UPDATE trades SET {columns} WHERE id = ?", values)
            self._conn.commit()

    def log_trade_update(self, trade_id: int, event_type: str, details: dict[str, Any] | None = None) -> int:
        payload = {
            "trade_id": trade_id,
            "event_type": event_type,
            "details": json.dumps(details or {}, ensure_ascii=True),
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        return self._insert("trade_updates", payload)

    def log_market_snapshot(self, **snapshot: Any) -> int:
        payload = {**snapshot}
        if "created_at" not in payload:
            payload["created_at"] = datetime.utcnow().isoformat(timespec="seconds")
        for key in ("ma_9", "ma_21", "ma_45", "ma_100", "btc_price", "btc_rsi", "btc_volatility", "asset_price", "asset_rsi"):
            if key in payload:
                payload[key] = self._normalize_value(payload[key])
        return self._insert("market_snapshots", payload)

    def fetch_open_trades(self) -> list[sqlite3.Row]:
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            rows = cursor.fetchall()
        return rows

    def fetch_closed_trades(self, limit: int = 10) -> list[sqlite3.Row]:
        if limit <= 0:
            return []
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT * FROM trades
                WHERE status != 'OPEN' AND exit_time IS NOT NULL
                ORDER BY datetime(exit_time) DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            return cursor.fetchall()

    def fetch_trade_stats(self) -> dict[str, Any]:
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT asset, direction, pnl, pnl_pct
                FROM trades
                WHERE status != 'OPEN' AND pnl IS NOT NULL
                """
            )
            rows = cursor.fetchall()
        total = len(rows)
        total_pnl = sum(float(row['pnl'] or 0.0) for row in rows)
        wins = sum(1 for row in rows if (row['pnl'] or 0.0) > 0)
        losses = total - wins
        win_rate = (wins / total) if total else 0.0
        avg = (total_pnl / total) if total else 0.0

        def _row_summary(candidate: sqlite3.Row | None) -> dict[str, Any] | None:
            if candidate is None:
                return None
            return {
                'asset': candidate['asset'],
                'direction': candidate['direction'],
                'pnl': float(candidate['pnl'] or 0.0),
                'pnl_pct': float(candidate['pnl_pct'] or 0.0) if candidate['pnl_pct'] is not None else None,
            }

        best_row = max(rows, key=lambda row: float(row['pnl'] if row['pnl'] is not None else float('-inf')), default=None)
        worst_row = min(rows, key=lambda row: float(row['pnl'] if row['pnl'] is not None else float('inf')), default=None)
        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg': avg,
            'best': _row_summary(best_row),
            'worst': _row_summary(worst_row),
        }

    def fetch_trades_since(self, since: str) -> list[sqlite3.Row]:
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT * FROM trades
                WHERE status != 'OPEN' AND exit_time IS NOT NULL AND exit_time >= ?
                ORDER BY datetime(exit_time) DESC
                """,
                (since,),
            )
            return cursor.fetchall()

    def realized_pnl(self, since: str | None = None) -> float:
        query = "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status != 'OPEN'"
        params: tuple[Any, ...] = ()
        if since:
            query += " AND exit_time >= ?"
            params = (since,)
        with self._lock:
            cursor = self._conn.execute(query, params)
            value = cursor.fetchone()[0]
        return float(value or 0.0)

    def closed_trades_since(self, since: str) -> list[sqlite3.Row]:
        query = "SELECT * FROM trades WHERE status != 'OPEN' AND exit_time >= ?"
        with self._lock:
            cursor = self._conn.execute(query, (since,))
            return cursor.fetchall()

    def _insert(self, table: str, data: dict[str, Any]) -> int:
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = [self._normalize_value(value) for value in data.values()]
        with self._lock:
            cursor = self._conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)
            self._conn.commit()
            return int(cursor.lastrowid)

    def _normalize_trade_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        data = dict(fields)
        now = datetime.utcnow().isoformat(timespec="seconds")
        data.setdefault("created_at", now)
        data.setdefault("entry_time", now)
        for key in ("signal_timeframes", "confirmation_type"):
            if key in data and isinstance(data[key], (dict, list)):
                data[key] = json.dumps(data[key], ensure_ascii=True)
        return data

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds")
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=True)
        return value
