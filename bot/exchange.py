from __future__ import annotations
import ccxt


TESTNET_URLS = {
    "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
    "fapiPublicV2": "https://testnet.binancefuture.com/fapi/v2",
    "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
    "fapiPrivateV3": "https://testnet.binancefuture.com/fapi/v3",
    "public": "https://testnet.binancefuture.com/fapi/v1",
}


class BinanceFuturesTestnet:
    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "fetchCurrencies": False,  # skip sapi endpoints not available on testnet
            },
        })
        # Override URLs to point to testnet directly (sandbox mode is broken in latest ccxt)
        self.exchange.urls["api"] = {**self.exchange.urls.get("api", {}), **TESTNET_URLS}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 200):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ohlcv_extended(self, symbol: str, timeframe: str = "1m", limit: int = 200):
        if limit <= 1000:
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        timeframe_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)
        all_ohlcv: list[list[float]] = []
        since = None
        remaining = limit

        while remaining > 0:
            batch_limit = min(1000, remaining)
            batch = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=batch_limit,
            )
            if not batch:
                break
            if all_ohlcv:
                last_ts = all_ohlcv[-1][0]
                while batch and batch[0][0] <= last_ts:
                    batch = batch[1:]
                if not batch:
                    break
            all_ohlcv.extend(batch)
            remaining = limit - len(all_ohlcv)
            if len(batch) < batch_limit:
                break
            since = all_ohlcv[-1][0] + timeframe_ms

        return all_ohlcv[:limit]

    def fetch_order_book(self, symbol: str, limit: int = 50):
        return self.exchange.fetch_order_book(symbol, limit=limit)

    def fetch_ticker(self, symbol: str):
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self):
        return self.exchange.fetch_balance()

    def fetch_positions(self, symbols: list[str] | None = None):
        return self.exchange.fetch_positions(symbols=symbols)

    def set_leverage(self, symbol: str, leverage: int):
        if leverage < 1 or leverage > 125:
            raise ValueError("Leverage must be between 1 and 125")
        return self.exchange.set_leverage(leverage, symbol)

    def create_order(self, symbol: str, side: str, order_type: str, amount: float, price: float | None = None):
        if amount <= 0:
            raise ValueError("Order amount must be positive")
        if order_type not in {"market", "limit"}:
            raise ValueError("Order type must be market or limit")
        if side not in {"buy", "sell"}:
            raise ValueError("Side must be buy or sell")
        if order_type == "limit" and price is None:
            raise ValueError("Limit orders require a price")
        return self.exchange.create_order(symbol, order_type, side, amount, price)

    def market_buy(self, symbol: str, amount: float):
        return self.create_order(symbol, "buy", "market", amount)

    def market_sell(self, symbol: str, amount: float):
        return self.create_order(symbol, "sell", "market", amount)

    def limit_buy(self, symbol: str, amount: float, price: float):
        return self.create_order(symbol, "buy", "limit", amount, price)

    def limit_sell(self, symbol: str, amount: float, price: float):
        return self.create_order(symbol, "sell", "limit", amount, price)

    def close_position(self, symbol: str, side: str, amount: float):
        if amount <= 0:
            raise ValueError("Close amount must be positive")
        if side not in {"buy", "sell"}:
            raise ValueError("Side must be buy or sell")
        params = {"reduceOnly": True}
        return self.exchange.create_order(symbol, "market", side, amount, None, params)
