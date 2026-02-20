from __future__ import annotations
import time
from datetime import datetime
import pandas as pd

from bot.advisor import Advisor
from bot.indicators import add_indicators
from utils.logger import get_logger


class Trader:
    def __init__(
        self,
        exchange,
        strategy,
        risk_manager,
        symbol: str,
        timeframe: str = "1m",
        poll_seconds: int = 30,
        advisor: Advisor | None = None,
    ):
        self.exchange = exchange
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.poll_seconds = poll_seconds
        self.advisor = advisor
        self.logger = get_logger("trader")

    def _fetch_dataframe(self, limit: int = 200) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return add_indicators(df)

    def _get_equity(self) -> float:
        balance = self.exchange.fetch_balance()
        total = balance.get("total", {})
        usdt = total.get("USDT", 0.0)
        return float(usdt)

    def _open_positions_count(self) -> int:
        positions = self.exchange.fetch_positions(symbols=[self.symbol])
        open_positions = [p for p in positions if abs(float(p.get("contracts", 0))) > 0]
        return len(open_positions)

    def _current_position(self):
        positions = self.exchange.fetch_positions(symbols=[self.symbol])
        for pos in positions:
            contracts = float(pos.get("contracts", 0))
            if abs(contracts) > 0:
                return contracts
        return 0.0

    def run(self):
        self.logger.info(f"Starting trader for {self.symbol} using {self.strategy.name}")
        while True:
            try:
                df = self._fetch_dataframe()
                latest_price = float(df.iloc[-1]["close"])

                open_positions = self._open_positions_count()
                if not self.risk_manager.can_trade(open_positions):
                    self.logger.warning("Risk limits reached. Skipping this cycle.")
                    time.sleep(self.poll_seconds)
                    continue

                if self.strategy.should_enter(df):
                    equity = self._get_equity()
                    atr = float(df.iloc[-1]["atr"]) if "atr" in df.columns else 0.0
                    if atr <= 0:
                        self.logger.warning("ATR not ready. Skipping entry.")
                        time.sleep(self.poll_seconds)
                        continue
                    stop_loss, take_profit = self.risk_manager.compute_stop_take(latest_price, atr)
                    size = self.risk_manager.position_size(equity, latest_price, stop_loss)
                    decision = None
                    if self.advisor:
                        decision = self.advisor.evaluate(
                            symbol=self.symbol,
                            strategy_name=self.strategy.name,
                            direction="LONG",
                            entry_price=latest_price,
                            df=df,
                            account_equity=equity,
                            open_positions=open_positions,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                        )

                    if decision:
                        self.logger.info(
                            f"Mike decision | {decision.decision} conf={decision.confidence} "
                            f"size_pct={decision.position_size_pct:.2f} reasoning={decision.reasoning}"
                        )
                        if decision.decision == "NO_GO":
                            self.logger.warning("Mike vetoed trade. Skipping entry.")
                            time.sleep(self.poll_seconds)
                            continue

                        size *= max(0.0, decision.position_size_pct) / 100.0
                        stop_loss = decision.stop_loss or stop_loss
                        take_profit = decision.take_profit or take_profit

                    self.logger.info(
                        f"ENTER signal at {latest_price:.2f} | size={size:.4f} | SL={stop_loss:.2f} TP={take_profit:.2f}"
                    )
                    if size > 0:
                        self.exchange.market_buy(self.symbol, size)
                    else:
                        self.logger.warning("Position size is zero. Skipping entry.")

                if self.strategy.should_exit(df):
                    contracts = self._current_position()
                    if contracts != 0:
                        side = "sell" if contracts > 0 else "buy"
                        self.logger.info(f"EXIT signal at {latest_price:.2f} | closing {abs(contracts):.4f}")
                        self.exchange.close_position(self.symbol, side, abs(contracts))

            except Exception as exc:
                self.logger.error(f"Trader error: {exc}")

            time.sleep(self.poll_seconds)
