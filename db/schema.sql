CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY,
    asset TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_loss REAL NOT NULL,
    take_profit REAL,
    leverage INTEGER NOT NULL,
    position_size REAL NOT NULL,
    entry_time DATETIME NOT NULL,
    exit_time DATETIME,
    status TEXT NOT NULL,
    pnl REAL,
    pnl_pct REAL,
    signal_timeframes TEXT,
    signal_type TEXT,
    confirmation_type TEXT,
    btc_state TEXT,
    mike_decision TEXT,
    mike_reasoning TEXT,
    mike_response_time REAL,
    rsi_15m REAL,
    rsi_1h REAL,
    rsi_4h REAL,
    ma_order_4h TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trade_updates (
    id INTEGER PRIMARY KEY,
    trade_id INTEGER REFERENCES trades(id),
    event_type TEXT NOT NULL,
    details TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    btc_price REAL,
    btc_rsi REAL,
    btc_volatility REAL,
    asset_price REAL,
    asset_rsi REAL,
    ma_9 REAL,
    ma_21 REAL,
    ma_45 REAL,
    ma_100 REAL,
    ma_order TEXT,
    divergence_detected BOOLEAN,
    divergence_type TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
