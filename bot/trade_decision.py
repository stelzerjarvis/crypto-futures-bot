from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TradeDecision:
    decision: str
    confidence: int
    reasoning: str
    position_size_pct: float
    stop_loss: float | None
    take_profit: float | None
    raw_response: str
    adjustments: dict[str, Any] = field(default_factory=dict)
