import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_secret: str
    default_leverage: int
    risk_per_trade: float
    max_positions: int
    max_daily_loss: float
    testnet: bool



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
    )
