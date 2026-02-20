from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from openai import OpenAI

from bot.trade_decision import TradeDecision
from utils.logger import get_logger


class Advisor:
    def __init__(
        self,
        exchange,
        timeframe: str,
        model: str,
        enabled: bool = True,
        log_path: str = "logs/decisions.jsonl",
    ):
        self.exchange = exchange
        self.timeframe = timeframe
        self.model = model
        self.enabled = enabled
        self.logger = get_logger("advisor")
        self.log_path = log_path
        self._client: OpenAI | None = None
        self._api_key = os.getenv("OPENAI_API_KEY", "")

        if self.enabled and not self._api_key:
            self.logger.warning("OPENAI_API_KEY missing. Mike advisor will return NO_GO.")

        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fetch_funding_rate(self, symbol: str) -> float | None:
        try:
            if hasattr(self.exchange, "fetch_funding_rate"):
                data = self.exchange.fetch_funding_rate(symbol)
                return self._extract_funding_rate(data)
            if hasattr(self.exchange, "exchange") and hasattr(self.exchange.exchange, "fetch_funding_rate"):
                data = self.exchange.exchange.fetch_funding_rate(symbol)
                return self._extract_funding_rate(data)
        except Exception:
            return None
        return None

    def _extract_funding_rate(self, data: Any) -> float | None:
        if isinstance(data, dict):
            return self._safe_float(data.get("fundingRate") or data.get("funding_rate"))
        return None

    def _get_equity(self) -> float | None:
        try:
            balance = self.exchange.fetch_balance()
            total = balance.get("total", {})
            usdt = total.get("USDT", 0.0)
            return float(usdt)
        except Exception:
            return None

    def _open_positions(self, symbol: str) -> list[dict[str, Any]]:
        try:
            return self.exchange.fetch_positions(symbols=[symbol])
        except Exception:
            return []

    def _open_positions_count(self, symbol: str) -> int:
        positions = self._open_positions(symbol)
        open_positions = [p for p in positions if abs(float(p.get("contracts", 0))) > 0]
        return len(open_positions)

    def _recent_pnl(self, positions: list[dict[str, Any]]) -> float | None:
        if not positions:
            return None
        total = 0.0
        found = False
        for pos in positions:
            for key in ("unrealizedPnl", "unrealizedProfit", "realizedPnl"):
                value = self._safe_float(pos.get(key))
                if value is not None:
                    total += value
                    found = True
        return total if found else None

    def _latest_indicators(self, df: pd.DataFrame) -> dict[str, float | None]:
        if df.empty:
            return {}
        latest = df.iloc[-1]
        return {
            "rsi": self._safe_float(latest.get("rsi")),
            "macd": self._safe_float(latest.get("macd")),
            "macd_signal": self._safe_float(latest.get("macd_signal")),
            "bb_high": self._safe_float(latest.get("bb_high")),
            "bb_mid": self._safe_float(latest.get("bb_mid")),
            "bb_low": self._safe_float(latest.get("bb_low")),
            "ema_9": self._safe_float(latest.get("ema_9")),
            "ema_12": self._safe_float(latest.get("ema_12")),
            "ema_21": self._safe_float(latest.get("ema_21")),
            "ema_26": self._safe_float(latest.get("ema_26")),
            "ema_50": self._safe_float(latest.get("ema_50")),
            "ema_200": self._safe_float(latest.get("ema_200")),
        }

    def _latest_candles(self, df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
        if df.empty:
            return []
        candles = []
        tail = df.tail(limit)
        for _, row in tail.iterrows():
            ts = row.get("timestamp")
            if hasattr(ts, "isoformat"):
                ts_value = ts.isoformat()
            else:
                ts_value = str(ts)
            candles.append(
                {
                    "timestamp": ts_value,
                    "open": self._safe_float(row.get("open")),
                    "high": self._safe_float(row.get("high")),
                    "low": self._safe_float(row.get("low")),
                    "close": self._safe_float(row.get("close")),
                    "volume": self._safe_float(row.get("volume")),
                }
            )
        return candles

    def _system_prompt(self) -> str:
        from bot.mike_strategy_prompt import STRATEGY_CONTEXT
        return STRATEGY_CONTEXT

    def _signal_system_prompt(self) -> str:
        from bot.mike_strategy_prompt import STRATEGY_CONTEXT
        return STRATEGY_CONTEXT

    def _user_prompt(self, context: dict[str, Any]) -> str:
        return (
            "Trade signal context (JSON):\n"
            f"{json.dumps(context, ensure_ascii=True)}"
        )

    def _parse_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _coerce_decision(
        self,
        data: dict[str, Any],
        stop_loss: float | None,
        take_profit: float | None,
    ) -> TradeDecision:
        decision = str(data.get("decision", "NO_GO")).strip().upper()
        if decision in {"NO-GO", "NOGO"}:
            decision = "NO_GO"
        if decision not in {"GO", "NO_GO", "REDUCE_SIZE"}:
            decision = "NO_GO"

        confidence = data.get("confidence", 5)
        try:
            confidence = int(confidence)
        except (TypeError, ValueError):
            confidence = 5
        confidence = max(1, min(10, confidence))

        reasoning = str(data.get("reasoning", ""))

        position_size_pct = data.get("position_size_pct", 100.0 if decision != "NO_GO" else 0.0)
        try:
            position_size_pct = float(position_size_pct)
        except (TypeError, ValueError):
            position_size_pct = 100.0 if decision != "NO_GO" else 0.0
        position_size_pct = max(0.0, min(100.0, position_size_pct))

        stop_loss_value = self._safe_float(data.get("stop_loss"))
        take_profit_value = self._safe_float(data.get("take_profit"))

        return TradeDecision(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            position_size_pct=position_size_pct,
            stop_loss=stop_loss_value if stop_loss_value is not None else (float(stop_loss) if stop_loss is not None else None),
            take_profit=take_profit_value if take_profit_value is not None else (float(take_profit) if take_profit is not None else None),
            raw_response=json.dumps(data, ensure_ascii=True),
        )

    def _coerce_signal_decision(
        self,
        data: dict[str, Any],
        fallback_entry: float | None,
        fallback_stop: float | None,
        fallback_take: float | None,
    ) -> TradeDecision:
        decision = str(data.get("decision", "REJECT")).strip().upper()
        if decision not in {"APPROVE", "REJECT", "MODIFY"}:
            decision = "REJECT"

        confidence = data.get("confidence", 5)
        try:
            confidence = int(confidence)
        except (TypeError, ValueError):
            confidence = 5
        confidence = max(1, min(10, confidence))

        reasoning = str(data.get("reasoning", "")).strip()
        position_size_pct = data.get("position_size_pct", 100.0 if decision == "APPROVE" else 0.0)
        try:
            position_size_pct = float(position_size_pct)
        except (TypeError, ValueError):
            position_size_pct = 100.0 if decision == "APPROVE" else 0.0
        position_size_pct = max(0.0, min(100.0, position_size_pct))

        entry = self._safe_float(data.get("entry")) or fallback_entry
        stop_loss = self._safe_float(data.get("stop_loss")) or fallback_stop
        take_profit = self._safe_float(data.get("take_profit")) or fallback_take
        adjustments: dict[str, Any] = {}
        if entry is not None:
            adjustments["entry"] = entry
        if stop_loss is not None:
            adjustments["stop_loss"] = stop_loss
        if take_profit is not None:
            adjustments["take_profit"] = take_profit
        notes = data.get("notes")
        if notes:
            adjustments["notes"] = notes

        return TradeDecision(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            position_size_pct=position_size_pct,
            stop_loss=stop_loss,
            take_profit=take_profit,
            raw_response=json.dumps(data, ensure_ascii=True),
            adjustments=adjustments,
        )

    def _log_decision(self, payload: dict[str, Any]) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as exc:
            self.logger.warning(f"Failed to log decision: {exc}")

    # ------------------------------------------------------------------
    def evaluate(
        self,
        symbol: str,
        strategy_name: str,
        direction: str,
        entry_price: float,
        df: pd.DataFrame,
        account_equity: float | None = None,
        open_positions: int | None = None,
        recent_pnl: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> TradeDecision:
        if account_equity is None:
            account_equity = self._get_equity()
        if open_positions is None:
            open_positions = self._open_positions_count(symbol)

        positions = self._open_positions(symbol)
        if recent_pnl is None:
            recent_pnl = self._recent_pnl(positions)

        context = {
            "symbol": symbol,
            "timeframe": self.timeframe,
            "strategy": strategy_name,
            "direction": direction,
            "entry_price": entry_price,
            "current_price": self._safe_float(df.iloc[-1]["close"]) if not df.empty else entry_price,
            "latest_20_candles": self._latest_candles(df, limit=20),
            "indicators": self._latest_indicators(df),
            "funding_rate": self._fetch_funding_rate(symbol),
            "account_equity": account_equity,
            "open_positions_count": open_positions,
            "recent_pnl": recent_pnl,
            "suggested_stop_loss": stop_loss,
            "suggested_take_profit": take_profit,
        }

        if not self.enabled:
            decision = TradeDecision(
                decision="GO",
                confidence=10,
                reasoning="Mike advisor disabled; defaulting to strategy signal.",
                position_size_pct=100.0,
                stop_loss=float(stop_loss) if stop_loss is not None else None,
                take_profit=float(take_profit) if take_profit is not None else None,
                raw_response="",
            )
            self._log_decision(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "direction": direction,
                    "entry_price": entry_price,
                    "decision": decision.decision,
                    "confidence": decision.confidence,
                    "position_size_pct": decision.position_size_pct,
                    "stop_loss": decision.stop_loss,
                    "take_profit": decision.take_profit,
                    "reasoning": decision.reasoning,
                    "raw_response": decision.raw_response,
                }
            )
            return decision

        if not self._api_key:
            self.logger.warning("Mike advisor unavailable: missing OPENAI_API_KEY.")
            decision = TradeDecision(
                decision="NO_GO",
                confidence=1,
                reasoning="AI advisor unreachable; missing OPENAI_API_KEY.",
                position_size_pct=0.0,
                stop_loss=float(stop_loss) if stop_loss is not None else None,
                take_profit=float(take_profit) if take_profit is not None else None,
                raw_response="",
            )
            self._log_decision(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "direction": direction,
                    "entry_price": entry_price,
                    "decision": decision.decision,
                    "confidence": decision.confidence,
                    "position_size_pct": decision.position_size_pct,
                    "stop_loss": decision.stop_loss,
                    "take_profit": decision.take_profit,
                    "reasoning": decision.reasoning,
                    "raw_response": decision.raw_response,
                }
            )
            return decision

        try:
            client = self._get_client()
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(context)},
            ]
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
            except TypeError:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                )

            raw = response.choices[0].message.content.strip()
            data = self._parse_json(raw)
            decision = self._coerce_decision(data, stop_loss, take_profit)
        except Exception as exc:
            self.logger.warning(f"Mike advisor unavailable: {exc}")
            decision = TradeDecision(
                decision="NO_GO",
                confidence=1,
                reasoning="AI advisor unreachable; skipping trade.",
                position_size_pct=0.0,
                stop_loss=float(stop_loss) if stop_loss is not None else None,
                take_profit=float(take_profit) if take_profit is not None else None,
                raw_response=str(exc),
            )

        self._log_decision(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "strategy": strategy_name,
                "direction": direction,
                "entry_price": entry_price,
                "decision": decision.decision,
                "confidence": decision.confidence,
                "position_size_pct": decision.position_size_pct,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "reasoning": decision.reasoning,
                "raw_response": decision.raw_response,
            }
        )
        return decision

    # ------------------------------------------------------------------
    def review_signal(self, context: dict[str, Any]) -> TradeDecision:
        proposed = context.get("proposed", {})
        fallback_entry = self._safe_float(proposed.get("entry"))
        fallback_stop = self._safe_float(proposed.get("stop_loss"))
        fallback_take = self._safe_float(proposed.get("take_profit"))

        if not self.enabled:
            adjustments = {}
            if fallback_entry is not None:
                adjustments["entry"] = fallback_entry
            return TradeDecision(
                decision="APPROVE",
                confidence=10,
                reasoning="Mike disabled; auto-approving signal.",
                position_size_pct=100.0,
                stop_loss=fallback_stop,
                take_profit=fallback_take,
                raw_response="",
                adjustments=adjustments,
            )

        if not self._api_key:
            self.logger.warning("Mike advisor unavailable for signal review: missing OPENAI_API_KEY.")
            return TradeDecision(
                decision="REJECT",
                confidence=1,
                reasoning="Advisor unreachable",
                position_size_pct=0.0,
                stop_loss=fallback_stop,
                take_profit=fallback_take,
                raw_response="",
            )

        payload = {
            "symbol": context.get("asset"),
            "strategy": context.get("strategy", "divergence_4ma"),
            "direction": context.get("direction"),
            "timeframes": context.get("timeframes"),
            "divergences": context.get("divergences"),
            "confirmations": context.get("confirmations"),
            "btc_state": context.get("btc_state"),
            "patterns": context.get("patterns"),
            "proposed": proposed,
            "open_positions": context.get("open_positions"),
        }

        messages = [
            {"role": "system", "content": self._signal_system_prompt()},
            {"role": "user", "content": self._user_prompt(payload)},
        ]
        start = time.monotonic()
        try:
            client = self._get_client()
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
            except TypeError:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                )
            raw = response.choices[0].message.content.strip()
            data = self._parse_json(raw)
            decision = self._coerce_signal_decision(data, fallback_entry, fallback_stop, fallback_take)
        except Exception as exc:
            self.logger.warning(f"Mike advisor failed to review signal: {exc}")
            decision = TradeDecision(
                decision="REJECT",
                confidence=1,
                reasoning="Advisor error",
                position_size_pct=0.0,
                stop_loss=fallback_stop,
                take_profit=fallback_take,
                raw_response=str(exc),
            )

        elapsed = time.monotonic() - start
        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": context.get("asset"),
            "strategy": context.get("strategy", "divergence_4ma"),
            "direction": context.get("direction"),
            "decision": decision.decision,
            "confidence": decision.confidence,
            "position_size_pct": decision.position_size_pct,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "reasoning": decision.reasoning,
            "raw_response": decision.raw_response,
            "context_type": "signal",
            "response_time": round(elapsed, 3),
        }
        self._log_decision(log_payload)
        return decision
