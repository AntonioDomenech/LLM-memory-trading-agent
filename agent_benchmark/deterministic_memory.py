from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from .schemas import BenchmarkConfig
from .warehouse.store import Warehouse


CONTEXT_SYMBOLS = ["SPY", "QQQ", "^VIX"]


def _date(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return str(value)[:10]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt_pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:+.2%}"


def _finite(value: Any) -> bool:
    try:
        return value is not None and not pd.isna(value) and math.isfinite(float(value))
    except Exception:
        return False


@dataclass
class MemorySummary:
    cases: int
    symbols: int
    train_start: str
    train_end: str


class DeterministicMarketMemory:
    """Point-in-time historical case memory built without LLM calls."""

    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self._case_cache: Dict[tuple, pd.DataFrame] = {}
        self._context_cache: Dict[tuple, pd.DataFrame] = {}

    def prepare(self, config: BenchmarkConfig, symbols: List[str]) -> MemorySummary:
        cases = self._cases(config, symbols)
        return MemorySummary(
            cases=int(len(cases)),
            symbols=int(cases["symbol"].nunique()) if not cases.empty else 0,
            train_start=config.train_start,
            train_end=config.train_end,
        )

    def retrieve(
        self,
        config: BenchmarkConfig,
        symbols: List[str],
        decision_date: str,
        current_market: Dict[str, Any],
        *,
        limit_per_symbol: int | None = None,
        max_items: int | None = None,
    ) -> List[Dict[str, Any]]:
        cases = self._cases(config, symbols)
        if cases.empty:
            return []
        limit_per_symbol = max(1, int(limit_per_symbol or config.deterministic_memory_per_symbol or 1))
        max_items = max(1, int(max_items or config.deterministic_memory_max_items or 50))
        knowledge_cutoff = min(_date(decision_date), _date(config.train_end))
        candidates = cases[cases["knowledge_timestamp"] <= knowledge_cutoff]
        if candidates.empty:
            return []

        current_context = self._current_context_features(decision_date)
        selected: List[tuple[float, Dict[str, Any]]] = []
        for symbol in symbols:
            snapshot = current_market.get(symbol) or {}
            group = candidates[candidates["symbol"] == symbol]
            if group.empty or not snapshot:
                continue
            scored = group.copy()
            scored["_distance"] = scored.apply(lambda row: self._distance(row, snapshot, current_context), axis=1)
            top = scored.sort_values(["_distance", "date"], ascending=[True, False]).head(limit_per_symbol)
            for _, row in top.iterrows():
                score = 1.0 / (1.0 + float(row["_distance"]))
                selected.append((score, self._row_to_memory(row, score)))

        selected.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in selected[:max_items]]

    def _cases(self, config: BenchmarkConfig, symbols: List[str]) -> pd.DataFrame:
        key = (tuple(sorted(symbols)), config.train_start, config.train_end)
        if key not in self._case_cache:
            self._case_cache[key] = self._load_cases(config, symbols)
        return self._case_cache[key]

    def _load_cases(self, config: BenchmarkConfig, symbols: List[str]) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        placeholders = ", ".join(["?"] * len(symbols))
        lookback_start = (datetime.fromisoformat(config.train_start) - timedelta(days=140)).date().isoformat()
        query = f"""
            WITH priced AS (
                SELECT
                    date,
                    symbol,
                    close,
                    volume,
                    return_1d,
                    close / NULLIF(LAG(close, 5) OVER (PARTITION BY symbol ORDER BY date), 0) - 1 AS return_5d,
                    close / NULLIF(LAG(close, 20) OVER (PARTITION BY symbol ORDER BY date), 0) - 1 AS return_20d,
                    close / NULLIF(LAG(close, 60) OVER (PARTITION BY symbol ORDER BY date), 0) - 1 AS return_60d,
                    STDDEV_SAMP(return_1d) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * SQRT(252) AS volatility_20d,
                    LEAD(date, 1) OVER (PARTITION BY symbol ORDER BY date) AS outcome_date_1d,
                    LEAD(close, 1) OVER (PARTITION BY symbol ORDER BY date) / NULLIF(close, 0) - 1 AS outcome_1d,
                    LEAD(date, 5) OVER (PARTITION BY symbol ORDER BY date) AS outcome_date_5d,
                    LEAD(close, 5) OVER (PARTITION BY symbol ORDER BY date) / NULLIF(close, 0) - 1 AS outcome_5d,
                    LEAD(date, 20) OVER (PARTITION BY symbol ORDER BY date) AS outcome_date_20d,
                    LEAD(close, 20) OVER (PARTITION BY symbol ORDER BY date) / NULLIF(close, 0) - 1 AS outcome_20d,
                    LEAD(date, 60) OVER (PARTITION BY symbol ORDER BY date) AS outcome_date_60d,
                    LEAD(close, 60) OVER (PARTITION BY symbol ORDER BY date) / NULLIF(close, 0) - 1 AS outcome_60d
                FROM asset_daily
                WHERE symbol IN ({placeholders})
                  AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                  AND ohlcv_available = true
                  AND close IS NOT NULL
            )
            SELECT *
            FROM priced
            WHERE date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND return_20d IS NOT NULL
              AND (outcome_1d IS NOT NULL OR outcome_5d IS NOT NULL OR outcome_20d IS NOT NULL OR outcome_60d IS NOT NULL)
            ORDER BY symbol, date
        """
        df = self.warehouse.conn.execute(
            query,
            [*symbols, lookback_start, config.train_end, config.train_start, config.train_end],
        ).fetchdf()
        if df.empty:
            return df
        df["date"] = df["date"].map(_date)
        for horizon in ("1d", "5d", "20d", "60d"):
            column = f"outcome_date_{horizon}"
            df[column] = df[column].map(lambda value: _date(value) if value is not None and not pd.isna(value) else "")
            df.loc[df[column] > config.train_end, [column, f"outcome_{horizon}"]] = ["", None]
        df = df[df[[f"outcome_{h}" for h in ("1d", "5d", "20d", "60d")]].notna().any(axis=1)].copy()
        if df.empty:
            return df
        df["knowledge_timestamp"] = df.apply(self._knowledge_timestamp, axis=1)
        df = df[df["knowledge_timestamp"] <= config.train_end].copy()
        news = self._news_counts(symbols, config.train_start, config.train_end)
        if not news.empty:
            df = df.merge(news, how="left", on=["symbol", "date"])
        if "news_count" not in df.columns:
            df["news_count"] = 0
        else:
            df["news_count"] = df["news_count"].fillna(0).astype(int)
        context = self._context_frame(config.train_start, config.train_end)
        if not context.empty:
            df = df.merge(context, how="left", on="date")
        for column in ("spy_return_20d", "qqq_return_20d", "vix_close"):
            if column not in df.columns:
                df[column] = None
        return df.reset_index(drop=True)

    def _knowledge_timestamp(self, row: pd.Series) -> str:
        dates = [row.get(f"outcome_date_{h}") for h in ("1d", "5d", "20d", "60d") if row.get(f"outcome_date_{h}")]
        return max(dates) if dates else _date(row.get("date"))

    def _news_counts(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        placeholders = ", ".join(["?"] * len(symbols))
        try:
            df = self.warehouse.conn.execute(
                f"""
                SELECT symbol, CAST(bucket_start AS VARCHAR) AS date, COUNT(*) AS news_count
                FROM news_articles
                WHERE symbol IN ({placeholders})
                  AND bucket_start BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                GROUP BY symbol, bucket_start
                """,
                [*symbols, start, end],
            ).fetchdf()
        except Exception:
            return pd.DataFrame(columns=["symbol", "date", "news_count"])
        if not df.empty:
            df["date"] = df["date"].map(_date)
        return df

    def _context_frame(self, start: str, end: str) -> pd.DataFrame:
        key = (start, end)
        if key in self._context_cache:
            return self._context_cache[key]
        placeholders = ", ".join(["?"] * len(CONTEXT_SYMBOLS))
        lookback_start = (datetime.fromisoformat(start) - timedelta(days=60)).date().isoformat()
        try:
            df = self.warehouse.conn.execute(
                f"""
                WITH ctx AS (
                    SELECT
                        date,
                        symbol,
                        close,
                        close / NULLIF(LAG(close, 20) OVER (PARTITION BY symbol ORDER BY date), 0) - 1 AS return_20d
                    FROM context_daily
                    WHERE symbol IN ({placeholders})
                      AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                      AND ohlcv_available = true
                      AND close IS NOT NULL
                )
                SELECT *
                FROM ctx
                WHERE date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                """,
                [*CONTEXT_SYMBOLS, lookback_start, end, start, end],
            ).fetchdf()
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            out = pd.DataFrame(columns=["date", "spy_return_20d", "qqq_return_20d", "vix_close"])
        else:
            df["date"] = df["date"].map(_date)
            returns = df.pivot_table(index="date", columns="symbol", values="return_20d", aggfunc="last")
            closes = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
            out = pd.DataFrame(index=returns.index)
            out["spy_return_20d"] = returns.get("SPY")
            out["qqq_return_20d"] = returns.get("QQQ")
            out["vix_close"] = closes.get("^VIX")
            out = out.reset_index()
        self._context_cache[key] = out
        return out

    def _current_context_features(self, decision_date: str) -> Dict[str, float | None]:
        start = (datetime.fromisoformat(_date(decision_date)) - timedelta(days=60)).date().isoformat()
        frame = self._context_frame(start, _date(decision_date))
        if frame.empty:
            return {"spy_return_20d": None, "qqq_return_20d": None, "vix_close": None}
        frame = frame[frame["date"] <= _date(decision_date)].sort_values("date")
        if frame.empty:
            return {"spy_return_20d": None, "qqq_return_20d": None, "vix_close": None}
        row = frame.iloc[-1]
        return {
            "spy_return_20d": _safe_float(row.get("spy_return_20d")),
            "qqq_return_20d": _safe_float(row.get("qqq_return_20d")),
            "vix_close": _safe_float(row.get("vix_close")),
        }

    def _distance(self, row: pd.Series, snapshot: Dict[str, Any], context: Dict[str, Any]) -> float:
        weighted_features = [
            ("return_5d", snapshot.get("return_5d"), row.get("return_5d"), 0.06, 1.2),
            ("return_20d", snapshot.get("return_20d"), row.get("return_20d"), 0.12, 1.7),
            ("return_60d", snapshot.get("return_60d"), row.get("return_60d"), 0.20, 0.8),
            ("volatility_20d", snapshot.get("volatility_20d"), row.get("volatility_20d"), 0.25, 1.1),
            ("spy_return_20d", context.get("spy_return_20d"), row.get("spy_return_20d"), 0.10, 0.9),
            ("qqq_return_20d", context.get("qqq_return_20d"), row.get("qqq_return_20d"), 0.12, 0.7),
            ("vix_close", context.get("vix_close"), row.get("vix_close"), 12.0, 0.8),
        ]
        distance = 0.0
        used = 0.0
        for _, current, historical, scale, weight in weighted_features:
            if not (_finite(current) and _finite(historical)):
                continue
            distance += min(4.0, abs(float(current) - float(historical)) / scale) * weight
            used += weight
        if used <= 0:
            return 999.0
        return distance / used

    def _row_to_memory(self, row: pd.Series, score: float) -> Dict[str, Any]:
        outcomes = {
            horizon: _safe_float(row.get(f"outcome_{horizon}"))
            for horizon in ("1d", "5d", "20d", "60d")
            if _finite(row.get(f"outcome_{horizon}"))
        }
        outcome_dates = {
            horizon: row.get(f"outcome_date_{horizon}")
            for horizon in outcomes
            if row.get(f"outcome_date_{horizon}")
        }
        state = {
            "return_5d": _safe_float(row.get("return_5d")),
            "return_20d": _safe_float(row.get("return_20d")),
            "return_60d": _safe_float(row.get("return_60d")),
            "volatility_20d": _safe_float(row.get("volatility_20d")),
            "spy_return_20d": _safe_float(row.get("spy_return_20d")),
            "qqq_return_20d": _safe_float(row.get("qqq_return_20d")),
            "vix_close": _safe_float(row.get("vix_close")),
            "news_count": int(row.get("news_count") or 0),
        }
        content = (
            f"{row['symbol']} deterministic case on {_date(row['date'])}: "
            f"r5={_fmt_pct(state['return_5d'])}, r20={_fmt_pct(state['return_20d'])}, "
            f"r60={_fmt_pct(state['return_60d'])}, vol20={_fmt_pct(state['volatility_20d'])}, "
            f"SPY20={_fmt_pct(state['spy_return_20d'])}, news={state['news_count']}. "
            f"Later returns: "
            + ", ".join(f"{h}={_fmt_pct(v)}" for h, v in outcomes.items())
        )
        return {
            "id": f"det:{row['symbol']}:{_date(row['date'])}",
            "model": "deterministic",
            "mode": "deterministic_market_cases",
            "portfolio_scope": "historical_market",
            "symbol": row["symbol"],
            "decision_timestamp": _date(row["date"]),
            "knowledge_timestamp": row["knowledge_timestamp"],
            "source_run_id": "warehouse",
            "memory_type": "deterministic_market_case",
            "content": content,
            "outcome_horizon": ",".join(outcomes.keys()),
            "outcome_available_at": row["knowledge_timestamp"],
            "metadata": {
                "retrieval_score": round(score, 6),
                "historical_state": state,
                "future_outcomes": outcomes,
                "outcome_dates": outcome_dates,
            },
            "created_at": "",
            "retrieval_score": round(score, 6),
        }
