# Crypto Futures Bot (Binance Testnet)

A minimal, **testnet-only** crypto futures trading bot with a simple strategy system, risk management, backtesting, and Binance Futures Testnet integration via `ccxt`.

## Features
- Binance Futures **TESTNET** connectivity (ccxt)
- Strategy interface + example RSI oversold strategy
- Risk manager (position sizing, stop-loss, take-profit, limits)
- Trading loop with indicator calculations
- Simple backtesting engine and performance metrics
- Colored logging with timestamps

## Project Structure
```
crypto-futures-bot/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   ├── __init__.py
│   └── settings.py
├── bot/
│   ├── __init__.py
│   ├── exchange.py
│   ├── strategy.py
│   ├── risk_manager.py
│   ├── trader.py
│   └── indicators.py
├── strategies/
│   ├── __init__.py
│   └── rsi_oversold.py
├── backtest/
│   ├── __init__.py
│   └── engine.py
├── utils/
│   ├── __init__.py
│   └── logger.py
└── main.py
```

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your Binance Futures **Testnet** API key/secret.
3. Run a live paper trade:
   ```bash
   python main.py trade --symbol BTCUSDT --strategy rsi_oversold
   ```
4. Run a backtest:
   ```bash
   python main.py backtest --symbol BTCUSDT --strategy rsi_oversold --days 90
   ```

## Notes
- Trading defaults to **testnet** only.
- Always verify that your API keys are testnet keys.
- This is an educational template; use at your own risk.
