# Trading Strategy — Crypto Futures on Binance (Divergence 4MA)

> Strategy derived from a conversation with an experienced trader.
> Bot analyzes, Mike (GPT-5.2) makes final trade decisions.

## Architecture: Option 2

```
Bot (continuous analysis) → Signal detected → Context sent to Mike → Mike approves/rejects → Bot executes
```

## 1. Assets

| Crypto | Futures Ticker | Priority |
|--------|---------------|----------|
| ROSE | ROSEUSDT | High |
| THETA | THETAUSDT | High |
| ATOM | ATOMUSDT | High |
| AXS | AXSUSDT | High |
| SOL | SOLUSDT | High |
| AAVE | AAVEUSDT | High |
| BNB | BNBUSDT | High |

**Reference**: BTCUSDT (not traded, used as market filter)

> **Note**: Verify exact tickers on Binance Futures.

## 2. Indicators

### Moving Averages (4 SMA)

| MA | Period | Color | Function |
|----|--------|-------|----------|
| Fast MA | **9** | Green | Short-term signal |
| Trend MA | **21** | Yellow | **Most important.** Defines trend |
| Intermediate MA | **45** | Red | Intermediate trend confirmation |
| Slow MA | **100** | Brown | Strong bounce zone / TP target |

### RSI
- Period 14, standard overbought/oversold bands
- Used for **divergence detection**

## 3. Multi-Timeframe Analysis (Top-Down)

```
DAILY → 4 HOURS → 1 HOUR → 15 MINUTES
(context)  (trend)  (confirm)  (execution)
```

## 4. Entry Rules — LONG

### Mandatory (ALL required)
1. RSI oversold (lower band)
2. Bullish divergence: price lower low + RSI higher low
3. Divergence on at least 2 timeframes

### Confirmation (at least 1)
4. Hammer candle (long lower wick, small body top)
5. Wick with retraction (price tested level, rejected)
6. Bounce off MA 21 or MA 100

### Filters
- BTC not in sharp decline or extreme volatility
- MAs on 4h in bullish order or crossing bullish
- If crypto already dropped 5-7% daily → safer for long

## 5. Entry Rules — SHORT

### Mandatory (ALL required)
1. RSI overbought (upper band)
2. Bearish divergence: price higher high + RSI lower high
3. Divergence on at least 2 timeframes

### Confirmation (at least 1)
4. Detachment from MA 9 (2-3 candles away)
5. MAs in bearish order or crossing down
6. Loss of momentum (smaller candles, rejection wicks)

### Filters
- BTC bearish or showing weakness
- MAs on 4h in bearish order or crossing down

## 6. Exit Rules

### Take Profit
- Long: price reaches MA 100 or relevant upper MA
- Short: price reaches MA 100 or relevant lower MA
- If price touches MA 21 and bounces against position → consider closing

### Stop Loss
- Mandatory on every entry
- Below last relevant low (long) / above last relevant high (short)
- Move to breakeven when profitable

### Emergency Exit
- BTC strong wick >2-3% → exit altcoin positions immediately
- Analysis no longer holds → exit, don't wait for stop loss

## 7. Risk Management

- Leverage: **5X** (conservative start)
- Max position size: **5% of capital**
- Max loss per trade: **2%**
- Averaging down: only if analysis still holds

## 8. BTC Market Filter

| BTC State | Action |
|-----------|--------|
| Extreme volatility | DO NOT trade |
| Clear bearish 1h/4h | SHORT altcoins |
| Clear bullish 1h/4h | LONG altcoins |
| Sideways | Trade cautiously, smaller positions |
| Sudden wick >2-3% | EXIT all altcoin positions |

**Key**: 1% BTC move ≈ 3-4% altcoin move

## 9. Configurable Parameters

```python
ASSETS = ["ROSEUSDT", "THETAUSDT", "ATOMUSDT", "AXSUSDT", "SOLUSDT", "AAVEUSDT", "BNBUSDT"]
REFERENCE = "BTCUSDT"
LEVERAGE = 5
MA_PERIODS = [9, 21, 45, 100]
RSI_PERIOD = 14
ENTRY_TIMEFRAME = "15m"
TREND_TIMEFRAMES = ["1h", "4h"]
CONTEXT_TIMEFRAME = "1d"
MAX_POSITION_SIZE_PCT = 0.05
STOP_LOSS_PCT = 0.02
BTC_VOLATILITY_THRESHOLD = 0.03
```

## 10. Analysis Frequency

- **Every 1 minute**: Check BTC for emergency conditions (sudden wicks)
- **Every 15 minutes** (on candle close): Full analysis on all assets at 15m timeframe
- **Every 1 hour**: Recalculate 1h indicators and divergences
- **Every 4 hours**: Recalculate 4h trend and macro divergences
- **Daily**: Update daily context

## 11. Mike's Role (Final Decision Maker)

When the bot detects a signal, it sends Mike:
- Asset, direction (LONG/SHORT), timeframe
- All indicator values (MAs, RSI, divergence details)
- BTC market state
- Candle pattern detected
- Proposed entry, stop loss, take profit
- Current open positions

Mike responds with: APPROVE / REJECT / MODIFY (with adjusted params)

## 12. Notifications (Telegram)

### Real-time alerts:
- **Signal detected** → "🔍 Signal: LONG SOLUSDT — waiting for Mike's decision"
- **Mike's decision** → "✅ Mike APPROVED LONG SOLUSDT" or "❌ Mike REJECTED — reason: ..."
- **Trade executed** → "📈 OPENED LONG SOLUSDT @ $145.20 | SL: $142.10 | TP: $152.00 | 5X"
- **Stop loss moved** → "🔒 SOLUSDT SL moved to breakeven @ $145.20"
- **Trade closed** → "💰 CLOSED LONG SOLUSDT @ $151.80 | P&L: +$32.50 (+4.5%)"
- **Emergency exit** → "🚨 BTC wick detected! Exiting all positions"

### Periodic summary:
- **Every 4 hours**: Brief status — open positions, P&L, market conditions
- **Daily**: Full summary — trades opened/closed, total P&L, win rate

## 13. Trade History (Database)

Store all trades in SQLite:

```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    asset TEXT NOT NULL,
    direction TEXT NOT NULL,  -- LONG/SHORT
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_loss REAL NOT NULL,
    take_profit REAL,
    leverage INTEGER NOT NULL,
    position_size REAL NOT NULL,
    entry_time DATETIME NOT NULL,
    exit_time DATETIME,
    status TEXT NOT NULL,  -- OPEN/CLOSED/STOPPED/EMERGENCY
    pnl REAL,
    pnl_pct REAL,
    -- Signal context
    signal_timeframes TEXT,  -- JSON: which timeframes had divergence
    signal_type TEXT,  -- divergence_bullish, divergence_bearish
    confirmation_type TEXT,  -- hammer, wick_retraction, ma_bounce
    btc_state TEXT,  -- bullish, bearish, sideways
    -- Mike's input
    mike_decision TEXT,  -- APPROVE/REJECT/MODIFY
    mike_reasoning TEXT,
    mike_response_time REAL,  -- seconds
    -- Indicators at entry
    rsi_15m REAL,
    rsi_1h REAL,
    rsi_4h REAL,
    ma_order_4h TEXT,  -- bullish/bearish/crossing
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE trade_updates (
    id INTEGER PRIMARY KEY,
    trade_id INTEGER REFERENCES trades(id),
    event_type TEXT NOT NULL,  -- sl_moved, tp_adjusted, averaged_down, emergency_exit
    details TEXT,  -- JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE market_snapshots (
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
```
