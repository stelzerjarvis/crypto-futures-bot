from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "trades.db"


def init_database(db_path: Path | str | None = None) -> None:
    """Initialize the SQLite database using schema.sql."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_PATH.read_text())


if __name__ == "__main__":
    init_database()
