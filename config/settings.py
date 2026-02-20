from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


DEFAULT_ASSETS: Tuple[str, ...] = (
    "ROSEUSDT",
    "THETAUSDT",
    "ATOMUSDT",
    "AXSUSDT",
    "SOLUSDT",
    "AAVEUSDT",
    "BNBUSDT",
)
DEFAULT_MA_PERIODS: Tuple[int, ...] = (9, 21, 45, 100)


def _parse_assets_env(value: str | None) -> Tuple[str, ...]:
    if not value:
        return DEFAULT_ASSETS
    items = [chunk.strip().upper() for chunk in value.split(",") if chunk.strip()]
    return tuple(items or DEFAULT_ASSETS)


def _parse_ma_periods_env(value: str | None) -> Tuple[int, ...]:
    if not value:
        return DEFAULT_MA_PERIODS
    periods: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            periods.append(int(chunk))
        except ValueError:
            continue
    return tuple(periods or DEFAULT_MA_PERIODS)


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_secret: str
    default_leverage: int
    risk_per_trade: float
    max_positions: int
    max_daily_loss: float
    testnet: bool
    mike_model: str
    mike_enabled: bool
    assets: Tuple[str, ...]
    reference_symbol: str
    btc_volatility_threshold: float
    ma_periods: Tuple[int, ...]
    analysis_interval_minutes: int
    summary_interval_hours: int



def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    return Settings(
        api_key=_get_env("BINANCE_TESTNET_API_KEY"),
        api_secret=_get_env("BINANCE_TESTNET_SECRET"),
        default_leverage=int(os.getenv("DEFAULT_LEVERAGE", "3")),
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.02")),
        max_positions=int(os.getenv("MAX_POSITIONS", "3")),
        max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.05")),
        testnet=True,
        mike_model=os.getenv("MIKE_MODEL", "gpt-4o"),
        mike_enabled=os.getenv("MIKE_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"},
        assets=_parse_assets_env(os.getenv("ASSETS")),
        reference_symbol=os.getenv("REFERENCE_SYMBOL", "BTCUSDT").upper(),
        btc_volatility_threshold=float(os.getenv("BTC_VOLATILITY_THRESHOLD", "0.03")),
        ma_periods=_parse_ma_periods_env(os.getenv("MA_PERIODS")),
        analysis_interval_minutes=int(os.getenv("ANALYSIS_INTERVAL_MINUTES", "15")),
        summary_interval_hours=int(os.getenv("SUMMARY_INTERVAL_HOURS", "4")),
    )
