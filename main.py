import argparse

from config.settings import load_settings
from bot.exchange import BinanceFuturesTestnet
from bot.risk_manager import RiskLimits, RiskManager
from bot.trader import Trader
from strategies.rsi_oversold import RsiOversoldStrategy
from backtest.engine import BacktestEngine
from utils.logger import get_logger


def get_strategy(name: str):
    if name == "rsi_oversold":
        return RsiOversoldStrategy()
    raise ValueError(f"Unknown strategy: {name}")


def normalize_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT"
    return symbol


def run_trade(args):
    settings = load_settings()
    exchange = BinanceFuturesTestnet(settings.api_key, settings.api_secret)
    symbol = normalize_symbol(args.symbol)
    exchange.set_leverage(symbol, settings.default_leverage)
    strategy = get_strategy(args.strategy)
    limits = RiskLimits(
        risk_per_trade=settings.risk_per_trade,
        max_positions=settings.max_positions,
        max_daily_loss=settings.max_daily_loss,
    )
    risk_manager = RiskManager(limits)
    trader = Trader(exchange, strategy, risk_manager, symbol, timeframe=args.timeframe)
    trader.run()


def run_backtest(args):
    settings = load_settings()
    exchange = BinanceFuturesTestnet(settings.api_key, settings.api_secret)
    strategy = get_strategy(args.strategy)
    symbol = normalize_symbol(args.symbol)
    engine = BacktestEngine(exchange, strategy, symbol, timeframe=args.timeframe)
    result = engine.run(days=args.days)
    logger = get_logger("backtest")
    logger.info(
        f"Backtest result | trades={result.trades} win_rate={result.win_rate:.2f} "
        f"pnl={result.pnl:.2f} max_dd={result.max_drawdown:.2f} sharpe={result.sharpe:.2f}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Crypto Futures Bot (Binance Testnet)")
    subparsers = parser.add_subparsers(dest="command")

    trade = subparsers.add_parser("trade", help="Run live paper trading")
    trade.add_argument("--symbol", required=True, help="Symbol like BTC/USDT")
    trade.add_argument("--strategy", required=True, help="Strategy name")
    trade.add_argument("--timeframe", default="1m", help="Candle timeframe")

    backtest = subparsers.add_parser("backtest", help="Run backtest")
    backtest.add_argument("--symbol", required=True, help="Symbol like BTC/USDT")
    backtest.add_argument("--strategy", required=True, help="Strategy name")
    backtest.add_argument("--timeframe", default="1h", help="Candle timeframe")
    backtest.add_argument("--days", type=int, default=90, help="Days of historical data")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "trade":
        run_trade(args)
    elif args.command == "backtest":
        run_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
