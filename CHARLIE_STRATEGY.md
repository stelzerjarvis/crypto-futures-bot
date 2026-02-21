# Crypto Futures Trading Strategy — Price Action Supply & Demand

> **Source**: YouTube video transcription (`yt-strategy.txt`). A pure price action strategy with no indicators — 3 steps: Market Structure, Supply & Demand Zones, Risk:Reward Filter.

---

## 1. Strategy Overview

This strategy uses three steps applied in sequence. A trade is only taken when all three steps check out.

1. **Market Structure (Trend Identification)** — Determine whether the asset is in an uptrend or downtrend by tracking valid swing highs and swing lows. Only trade in the direction of the trend: longs in uptrends, shorts in downtrends.

2. **Supply & Demand Zones** — In an uptrend, identify demand zones (areas of consolidation before a sharp upward move). In a downtrend, identify supply zones (consolidation before a sharp downward move). Wait for price to retest the zone, then enter. Stop loss goes just beyond the zone; take profit targets the most recent swing high/low.

3. **Risk:Reward Filter** — Only take the trade if the risk:reward ratio is at least 2.5:1. If it's below 2.5, skip the trade even if steps 1 and 2 are satisfied.

---

## 2. Formal Rules

### Step 1: Market Structure — Trend Identification

**Definitions:**

| Concept | Rule | Transcription Reference |
|---------|------|------------------------|
| Uptrend | Price makes higher highs AND higher lows | Line 12 |
| Downtrend | Price makes lower lows AND lower highs | Line 13 |
| Valid low | A swing low is valid ONLY IF the rally from it broke the previous swing high | Lines 28, 32, 35 |
| Valid high | (Inverse of valid low) A swing high is valid ONLY IF the decline from it broke the previous swing low | Implied by symmetry with lines 53-55 |
| Trend continuation | Uptrend holds as long as price stays above the last valid low — "it can go up, down, sideways, literally anything as long as it doesn't break this low, we are in an uptrend" | Lines 39-40 |
| Valid low transfer | When price makes a new higher high, the valid low transfers to the most recent swing low before that new high | Line 42 |
| Trend flip to downtrend | Occurs ONLY when price breaks (closes below) the last valid low | Lines 38, 80, 98 |
| Trend flip to uptrend | (Inverse) Occurs ONLY when price breaks (closes above) the last valid high | Implied by symmetry |

**Trading direction rule:**

```
IF trend = UPTREND  → ONLY look for LONG trades (lines 37, 77-78)
IF trend = DOWNTREND → ONLY look for SHORT trades (line 99)
"Shorting in an uptrend is just silly." (line 79)
```

### Step 2: Supply & Demand Zones

**Zone identification:**

```
IF UPTREND:
  1. Find an area of consolidation (price moves sideways) (line 82)
  2. Followed by a sharp/impulsive move UPWARDS (lines 82, 84)
  3. Mark the DEMAND ZONE = low to high of the LAST CANDLE before the impulsive move (lines 85-87)

IF DOWNTREND:
  1. Find an area of consolidation (price moves sideways)
  2. Followed by a sharp/impulsive move DOWNWARDS (lines 53-54)
  3. Mark the SUPPLY ZONE = low to high of the LAST CANDLE before the impulsive move
```

**Entry, stop loss, and take profit:**

```
LONG (from demand zone in uptrend):
  ENTRY:       Wait for price to re-enter the demand zone (line 91)
  STOP LOSS:   "Right below the demand zone" (line 92)
  TAKE PROFIT: "At the recent highs" (line 92)

SHORT (from supply zone in downtrend):
  ENTRY:       Wait for price to re-enter the supply zone (line 101)
  STOP LOSS:   "Right above the area of supply" (line 102)
  TAKE PROFIT: "At recent lows" (line 103)
```

### Step 3: Risk:Reward Filter

```
CALCULATE R:R = (distance from entry to TP) / (distance from entry to SL)

IF R:R >= 2.5 → TAKE the trade (line 111)
IF R:R < 2.5  → DO NOT take the trade, even if steps 1 and 2 pass (line 112)

"This one rule by itself increases the profit rate of the strategy by a ton." (line 113)
```

### Complete Decision Flow (Pseudocode)

```
FOR each asset:

  1. DETERMINE trend direction (uptrend / downtrend) using valid swing highs/lows

  2. IF uptrend:
       SCAN for demand zones (consolidation → impulsive move UP)
       IGNORE all supply zones
     IF downtrend:
       SCAN for supply zones (consolidation → impulsive move DOWN)
       IGNORE all demand zones

  3. IF a valid zone exists AND price is approaching/entering the zone:
       CALCULATE entry, SL, TP
       CALCULATE R:R = (entry to TP) / (entry to SL)

       IF R:R >= 2.5:
         PLACE trade (limit order at zone edge)
       ELSE:
         SKIP — do not trade

  4. IF in a trade:
       IF price hits TP → EXIT with profit
       IF price hits SL → EXIT with loss
       IF valid low/high breaks (trend flips) → EXIT immediately
```

---

## 3. Parameters Table

The transcription does not specify these values precisely. Below are proposed defaults for bot implementation. Each can be overridden.

| # | Parameter | Proposed Default | Reasoning |
|---|-----------|-----------------|-----------|
| 1 | **Swing point detection** | Fractal method: N=3 candles. A swing low = a candle whose low is lower than the 3 candles before AND 3 candles after it. A swing high = a candle whose high is higher than the 3 candles before AND after it. | Standard SMC swing detection. Simple and widely used. N=3 provides a good balance — N=2 is too noisy, N=5 is too slow. |
| 2 | **"Impulsive move" threshold** | Candle body > 1.5x ATR(14) | A candle whose body is 1.5x the 14-period Average True Range clearly stands out as impulsive. ATR adapts to the asset's volatility. |
| 3 | **"Consolidation" definition** | 3+ consecutive candles where the total range of all candles < 1x ATR(14) | Small range relative to volatility = sideways movement. Minimum 3 candles confirms it's not just a one-candle pause. |
| 4 | **What "breaks" means** | Candle must CLOSE below/above the level, not just wick through it | Wicks create frequent false breaks. Requiring a close is more conservative and filters out stop-hunt wicks. |
| 5 | **Zone freshness** | One-time use only. A zone is consumed after its first retest (win or lose). A zone is also invalidated if price closes through it entirely without respecting it. | The video shows one entry per zone in every example. Most supply/demand traders treat zones as single-use. |
| 6 | **Entry precision** | Limit order at the edge of the zone: top of demand zone (for longs), bottom of supply zone (for shorts) | Most aggressive entry within the zone. Maximizes R:R ratio. Entering at the middle of the zone worsens R:R. |
| 7 | **Stop loss offset** | SL = zone boundary - 0.1% of current price. For longs: zone_low - (price * 0.001). For shorts: zone_high + (price * 0.001). | Small buffer beyond the zone edge to avoid stop hunts on exact levels while keeping SL tight. |
| 8 | **Take profit target** | The most recent swing high (for longs) or swing low (for shorts) in the current trend leg | Matches the transcription's "recent highs" / "recent lows". The nearest swing is the safest, most conservative target. |
| 9 | **Timeframe** | 1h as primary timeframe | The video doesn't specify any timeframe. 1h is a common default for S/D strategies — good balance between signal quality and trade frequency for crypto futures. |
| 10 | **Trend change during open trade** | Close the position immediately if the valid low (longs) or valid high (shorts) is broken while in a trade | If the trend premise is invalidated, the trade thesis no longer holds. The transcription says you only trade in the trend direction. |
| 11 | **Multiple zones** | Trade only the closest valid zone to current price. Maximum 1 active trade per asset at a time. | Simplest approach. Avoids overexposure to a single asset. If the closest zone fails, the next zone becomes eligible. |

---

## 4. Trade Lifecycle

### Step-by-step flowchart

```
┌─────────────────────────────────────────────────────────┐
│                    SCAN LOOP (every candle close)        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │  1. DETECT TREND       │
                │  Identify valid swing  │
                │  highs & lows.         │
                │  Determine: UPTREND    │
                │  or DOWNTREND?         │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  2. IDENTIFY ZONES     │
                │  Uptrend → demand zones│
                │  Downtrend → supply    │
                │  zones.                │
                │  Find consolidation +  │
                │  impulsive move. Mark  │
                │  last candle before    │
                │  impulse.              │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  3. WAIT FOR RETEST    │
                │  Is price entering or  │
                │  at the zone?          │
                │                        │
                │  NO → continue scanning│
                │  YES ▼                 │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  4. CALCULATE R:R      │
                │  Entry = zone edge     │
                │  SL = beyond zone      │
                │  TP = recent swing     │
                │  R:R = TP dist / SL    │
                │  dist                  │
                │                        │
                │  R:R < 2.5 → SKIP     │
                │  R:R >= 2.5 ▼          │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  5. ENTER TRADE        │
                │  Place limit order at  │
                │  zone edge. Set SL     │
                │  and TP orders.        │
                └───────────┬────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  6. MANAGE TRADE       │
                │                        │
                │  Monitor each candle:  │
                │  • TP hit → EXIT WIN   │
                │  • SL hit → EXIT LOSS  │
                │  • Valid low/high      │
                │    broken → EXIT       │
                │    IMMEDIATELY          │
                │  • Zone consumed →     │
                │    remove from list    │
                └────────────────────────┘
```

### Trade lifecycle summary

| Phase | Action | Detail |
|-------|--------|--------|
| **Pre-trade** | Trend detection | Track valid highs/lows, determine uptrend/downtrend |
| **Pre-trade** | Zone identification | Find consolidation + impulsive move, mark zone |
| **Pre-trade** | Wait for retest | Price must re-enter the zone |
| **Pre-trade** | R:R check | Must be >= 2.5:1 or trade is skipped |
| **Entry** | Place orders | Limit entry at zone edge, SL beyond zone, TP at recent swing |
| **In-trade** | Monitor | Check for TP hit, SL hit, or trend invalidation |
| **Exit** | TP hit | Close position, mark zone as consumed |
| **Exit** | SL hit | Close position, mark zone as consumed |
| **Exit** | Trend flip | Close immediately — valid low/high broken |
| **Post-trade** | Reset | Zone is consumed; scan for next zone in the trend |

---

## 5. Optional Enhancements

> **These are NOT from the YouTube transcription.** They are borrowed from an older indicator-based strategy (`trading_strategy.md`) and general best practices. Each is clearly marked as optional. They complement the core strategy without contradicting it.

### 5.1 BTC Volatility Filter (OPTIONAL — not from transcription)

**Rule:** Do not open new trades if BTCUSDT has moved more than 3% in the last 4 hours.

**Rationale:** In crypto, altcoins are highly correlated with BTC. A sudden BTC move of >3% often causes cascading liquidations across altcoins, making supply/demand zones unreliable.

```
IF abs(BTC_price_change_4h) > 3%:
  DO NOT open new trades
  (Existing trades are managed normally — SL/TP still apply)
```

**Parameter:** `BTC_VOLATILITY_THRESHOLD = 0.03` (adjustable)

### 5.2 Breakeven Stop Loss Management (OPTIONAL — not from transcription)

**Rule:** When the trade is in profit by at least 1x the risk (i.e., price has moved from entry toward TP by the same distance as entry-to-SL), move the stop loss to the entry price (breakeven).

**Rationale:** The transcription only defines the initial stop loss. Moving to breakeven after price confirms direction reduces risk to zero while allowing the trade to run to TP.

```
IF unrealized_profit >= 1 * risk_amount:
  MOVE SL to entry_price (breakeven)
```

**Parameter:** `BREAKEVEN_TRIGGER = 1.0` (in multiples of risk; adjustable)

### 5.3 Multi-Timeframe Confirmation (OPTIONAL — not from transcription)

**Rule:** Use a higher timeframe (4h) for trend determination and a lower timeframe (1h or 15m) for zone identification and entry.

**Rationale:** Trend on a higher timeframe is more reliable. Zones on a lower timeframe provide tighter entries with better R:R. This approach is common in SMC (Smart Money Concepts) trading.

```
TREND TIMEFRAME:  4h (determine uptrend/downtrend via valid highs/lows)
ENTRY TIMEFRAME:  1h or 15m (identify S/D zones and enter trades)

Only take trades where both timeframes agree on direction.
```

### 5.4 Position Sizing (OPTIONAL — not from transcription)

**Rule:** Risk a maximum of 2-5% of total capital per trade.

**Rationale:** The transcription discusses R:R ratio but never mentions position sizing or money management. For a bot, this is essential to prevent ruin.

```
position_size = (capital * risk_pct) / (entry_price - stop_loss_price)

Example: $10,000 capital, 2% risk = $200 max loss per trade
If entry = $100, SL = $98 → position_size = $200 / $2 = 100 units
```

**Parameters:**
- `RISK_PER_TRADE_PCT = 0.02` (2% of capital; adjustable)
- `MAX_OPEN_TRADES = 3` (across all assets; adjustable)

### 5.5 Emergency Exit (OPTIONAL — not from transcription)

**Rule:** Close ALL open positions immediately if BTC wicks more than 2-3% in a short timeframe (e.g., within 15 minutes).

**Rationale:** Flash crashes in BTC cause extreme altcoin moves that blow through stop losses via slippage. An emergency exit overrides normal SL/TP management.

```
IF BTC_wick_15m > 2.5%:
  CLOSE all open positions at market price
```

**Parameter:** `BTC_EMERGENCY_THRESHOLD = 0.025` (adjustable)

---

## 6. Visual References

### `valid-low.png` — Valid Low Concept

The screenshot shows a line chart with an uptrend (higher highs, higher lows). Two horizontal lines are drawn:

- **Green line (lower)** = the **valid low**. This is the swing low whose rally broke the previous swing high. As long as price stays above this line, the uptrend is intact.
- **Red line (upper)** = an **invalid low**. This is a swing low whose rally did NOT break the previous high. Many traders mistakenly think this level being broken means a downtrend — but it doesn't, because it was never a valid low.

**Confirms:** Lines 28-35 — "In order for a low to be validated, it needs to break the previous high." The chart shows that when price dips below the red (invalid) line, the market is still in an uptrend because the green (valid) line has not been broken.

### `demand-zone.png` — Demand Zone Marking

The screenshot shows a real candlestick chart. After a period of consolidation (several sideways candles), price makes a sharp impulsive move upward (large green candles). The demand zone is marked:

- **Green rectangle** = the demand zone, covering the low to the high of the **last candle before the impulsive move**.
- **Red line** below the zone = the **stop loss level**, placed "right below the demand zone."

**Confirms:** Lines 85-87 — "Mark from the low to the high of the previous candle before the big move. This is our area of demand." Also confirms line 92 — "Set your stop loss right below the demand zone."

### `downtrend.png` — Downtrend Confirmation via Valid Low Break

The screenshot shows a real candlestick chart that was previously in an uptrend (higher highs and higher lows), then price breaks below a key level:

- **Red horizontal line** = the **valid low**. Price closes below this level, confirming a trend change to downtrend.
- The text "DOWNTREND" labels the chart after the break.

**Confirms:** Lines 97-99 — "This low is what broke the previous high. So this is where price needs to break in order to be in a downtrend, which is exactly what ends up happening. So now we are in a downtrend and we only look for areas of supply or short trades."

---

## 7. Reference Comparison — New vs Old Strategy

| Aspect | New Strategy (this document) | Old Strategy (`trading_strategy.md`) |
|--------|------------------------------|--------------------------------------|
| **Approach** | Pure price action — no indicators | Indicator-heavy: 4 MAs + RSI divergences |
| **Trend detection** | Valid swing highs/lows (market structure) | Moving average order (9 > 21 > 45 > 100) |
| **Entry signal** | Price re-enters supply/demand zone | RSI divergence + candle patterns (hammer, wick retraction) + MA bounce |
| **Entry confirmation** | Zone retest + R:R >= 2.5 | Multi-timeframe RSI divergence on 2+ TFs |
| **Stop loss** | Just beyond the S/D zone | Below last relevant swing low/high |
| **Take profit** | Recent swing high/low | MA 100 or other relevant MA |
| **R:R filter** | Mandatory >= 2.5:1 | Not specified |
| **Timeframes** | Not specified (default: 1h) | Daily → 4h → 1h → 15min cascade |
| **BTC filter** | Not mentioned (optional add-on) | Core rule — don't trade in BTC extreme volatility |
| **Position management** | SL/TP only, exit on trend flip | Move SL to breakeven, emergency BTC exit, DCA in position |
| **Indicators required** | None (ATR only for parameter defaults) | SMA 9, 21, 45, 100 + RSI 14 |
| **Complexity** | Low — 3 simple steps | High — multi-indicator, multi-timeframe |

---

## Appendix: Transcription Line Reference Index

Key lines from `yt-strategy.txt` used in this document:

| Line(s) | Content | Used In |
|----------|---------|---------|
| 12 | Higher highs + higher lows = uptrend | Step 1 |
| 13 | Lower lows + lower highs = downtrend | Step 1 |
| 28-32 | Valid low requires breaking previous high | Step 1 |
| 35 | "If price does break this high, we now know this is a valid low" | Step 1 |
| 37-40 | Only look for bullish trades in uptrend; trend holds above valid low | Step 1 |
| 42 | Valid low transfers when new higher high is made | Step 1 |
| 47-48 | Demand in uptrends, supply in downtrends | Step 2 |
| 79 | "Shorting in an uptrend is just silly" | Step 1 |
| 80 | Valid low identification on real chart | Step 1 |
| 82 | "Area of consolidation or point where price moves sideways" | Step 2 |
| 84-87 | Demand zone marking: last candle before impulsive move | Step 2 |
| 91 | "Wait for price to re-enter into the zone" | Step 2 |
| 92 | "Stop loss right below the demand zone, take profit at recent highs" | Step 2 |
| 98-99 | Valid low broken → downtrend → only look for short trades | Step 1 |
| 101-103 | Supply zone entry, SL above supply, TP at recent lows | Step 2 |
| 111 | "Risk to reward above 2.5 to 1" | Step 3 |
| 112 | "If risk to reward is under 2.5, we do not take this trade" | Step 3 |
| 113 | "This one rule increases the profit rate by a ton" | Step 3 |
| 117-119 | Final example: SL below demand, TP at recent high, R:R check | Steps 2-3 |
