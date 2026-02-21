"""
Agent Configuration Loader
============================
Each trading agent (Mike, Charlie, etc.) has its own JSON config file
in config/agents/. This keeps agents fully isolated — separate DBs,
separate capital, separate strategies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()

AGENTS_DIR = Path(__file__).parent / "agents"


@dataclass(frozen=True)
class AgentConfig:
    # Identity
    name: str
    emoji: str
    agent_id: str  # e.g. "mike", "charlie"

    # Model
    model: str

    # Strategy
    strategy: str  # strategy module name
    strategy_prompt: str  # module path for the prompt

    # Capital & Risk
    capital: float
    leverage: int
    risk_per_trade: float
    max_positions: int
    max_daily_loss: float

    # Assets
    assets: Tuple[str, ...]
    reference_symbol: str

    # Paths
    db_path: str
    log_file: str
    decisions_log: str

    # Exchange credentials (shared — same Binance account)
    api_key: str = ""
    api_secret: str = ""

    # Telegram (shared)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # OpenAI
    openai_api_key: str = ""

    # Mike enabled (always true for agent-based)
    mike_enabled: bool = True


def load_agent_config(agent_id: str) -> AgentConfig:
    """Load agent config from config/agents/{agent_id}.json + env vars."""
    config_path = AGENTS_DIR / f"{agent_id}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config not found: {config_path}")

    with open(config_path) as f:
        data = json.load(f)

    return AgentConfig(
        name=data["name"],
        emoji=data.get("emoji", "⚪"),
        agent_id=agent_id,
        model=data.get("model", "gpt-5.2"),
        strategy=data["strategy"],
        strategy_prompt=data.get("strategy_prompt", ""),
        capital=float(data.get("capital", 2500)),
        leverage=int(data.get("leverage", 5)),
        risk_per_trade=float(data.get("risk_per_trade", 0.02)),
        max_positions=int(data.get("max_positions", 3)),
        max_daily_loss=float(data.get("max_daily_loss", 0.05)),
        assets=tuple(data.get("assets", [])),
        reference_symbol=data.get("reference_symbol", "BTCUSDT"),
        db_path=data.get("db_path", f"data/{agent_id}_trades.db"),
        log_file=data.get("log_file", f"logs/{agent_id}.log"),
        decisions_log=data.get("decisions_log", f"logs/{agent_id}_decisions.jsonl"),
        api_key=os.getenv("BINANCE_TESTNET_API_KEY", ""),
        api_secret=os.getenv("BINANCE_TESTNET_SECRET", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        mike_enabled=True,
    )


def list_agents() -> list[str]:
    """Return list of available agent IDs."""
    if not AGENTS_DIR.exists():
        return []
    return [p.stem for p in AGENTS_DIR.glob("*.json")]
