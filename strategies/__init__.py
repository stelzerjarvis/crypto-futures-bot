from __future__ import annotations

from strategies.rsi_oversold import RsiOversoldStrategy
from strategies.macd_crossover import MacdCrossoverStrategy
from strategies.bollinger_breakout import BollingerBreakoutStrategy
from strategies.ema_crossover import EmaCrossoverStrategy
from strategies.divergence_4ma import Divergence4MAStrategy

__all__ = [
    "RsiOversoldStrategy",
    "MacdCrossoverStrategy",
    "BollingerBreakoutStrategy",
    "EmaCrossoverStrategy",
    "Divergence4MAStrategy",
]
