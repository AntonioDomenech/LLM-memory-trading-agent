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


def _bounded(value: float | None, scale: float) -> float:
    if value is None or not scale:
        return 0.0
    return max(-1.0, min(1.0, float(value) / float(scale)))


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
        neighbor_count = max(3, int(getattr(config, "memory_k_neighbors", 50) or 50))
        examples_per_symbol = max(0, int(getattr(config, "memory_examples_per_symbol", 2) or 2))
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
            top = scored.sort_values(["_distance", "date"], ascending=[True, False]).head(neighbor_count)
            if top.empty:
                continue
            aggregate = self._aggregate_to_memory(symbol, top, examples_per_symbol)
            selected.append((float(aggregate.get("retrieval_score") or 0.0), aggregate))
            if limit_per_symbol > 1:
                for _, row in top.head(limit_per_symbol - 1).iterrows():
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

    def _aggregate_to_memory(self, symbol: str, rows: pd.DataFrame, examples_per_symbol: int) -> Dict[str, Any]:
        best = rows.iloc[0]
        best_score = 1.0 / (1.0 + float(best.get("_distance") or 0.0))
        stats: Dict[str, Dict[str, float | int | None]] = {}
        fragments = []
        for horizon in ("1d", "5d", "20d", "60d"):
            values = [
                float(value)
                for value in rows.get(f"outcome_{horizon}", pd.Series(dtype=float)).tolist()
                if _finite(value)
            ]
            if not values:
                continue
            series = pd.Series(values)
            hit_rate = float((series > 0).mean())
            downside_rate = float((series < 0).mean())
            mean_return = float(series.mean())
            median_return = float(series.median())
            stats[horizon] = {
                "cases": int(len(series)),
                "mean_return": mean_return,
                "median_return": median_return,
                "hit_rate": hit_rate,
                "downside_rate": downside_rate,
                "p10": float(series.quantile(0.10)),
                "p90": float(series.quantile(0.90)),
            }
            fragments.append(
                f"{horizon}: n={len(series)}, mean={_fmt_pct(mean_return)}, median={_fmt_pct(median_return)}, hit={hit_rate:.0%}, downside={downside_rate:.0%}"
            )
        example_rows = []
        for _, row in rows.head(examples_per_symbol).iterrows():
            score = 1.0 / (1.0 + float(row.get("_distance") or 0.0))
            example_rows.append(
                {
                    "id": f"det:{row['symbol']}:{_date(row['date'])}",
                    "date": _date(row["date"]),
                    "score": round(score, 6),
                    "outcomes": {
                        horizon: _safe_float(row.get(f"outcome_{horizon}"))
                        for horizon in ("1d", "5d", "20d", "60d")
                        if _finite(row.get(f"outcome_{horizon}"))
                    },
                }
            )
        confidence = self._aggregate_confidence(stats.get("20d", {}), best_score)
        suggested_exposure = self._suggested_exposure(stats, confidence)
        content = (
            f"{symbol} aggregate memory from {len(rows)} similar point-in-time cases. "
            f"Retrieval confidence={confidence}; best_score={best_score:.3f}. "
            + " | ".join(fragments)
        )
        if suggested_exposure:
            band = suggested_exposure["band"]
            content += (
                f" Suggested exposure evidence: {suggested_exposure['horizon']} "
                f"score={suggested_exposure['score']:+.2f}, band=[{band[0]:+.2f}, {band[1]:+.2f}], "
                f"base_rate={_fmt_pct(suggested_exposure['base_rate_return'])}, "
                f"hit={suggested_exposure['hit_rate']:.0%}, downside={suggested_exposure['downside_rate']:.0%}."
            )
        return {
            "id": f"detagg:{symbol}:{_date(best['date'])}",
            "model": "deterministic",
            "mode": "deterministic_market_cases",
            "portfolio_scope": "historical_market",
            "symbol": symbol,
            "decision_timestamp": _date(best["date"]),
            "knowledge_timestamp": best["knowledge_timestamp"],
            "source_run_id": "warehouse",
            "memory_type": "deterministic_market_aggregate",
            "content": content,
            "outcome_horizon": ",".join(stats.keys()),
            "outcome_available_at": best["knowledge_timestamp"],
            "metadata": {
                "retrieval_score": round(best_score, 6),
                "aggregate_stats": stats,
                "examples": example_rows,
                "confidence": confidence,
                "suggested_exposure": suggested_exposure,
                "neighbor_count": int(len(rows)),
            },
            "created_at": "",
            "retrieval_score": round(best_score, 6),
        }

    def _suggested_exposure(self, stats: Dict[str, Dict[str, Any]], confidence: str) -> Dict[str, Any]:
        horizon = next((name for name in ("20d", "60d", "5d", "1d") if name in stats), "")
        if not horizon:
            return {}
        horizon_stats = stats.get(horizon) or {}
        mean_return = _safe_float(horizon_stats.get("mean_return"))
        hit_rate = _safe_float(horizon_stats.get("hit_rate"))
        downside_rate = _safe_float(horizon_stats.get("downside_rate"))
        cases = int(horizon_stats.get("cases") or 0)
        score = 0.65 * _bounded(mean_return, 0.08) + 0.35 * _bounded((hit_rate - 0.5) if hit_rate is not None else None, 0.25)
        if confidence == "weak":
            score *= 0.75
        if cases < 10:
            score *= 0.65
        if horizon == "1d":
            score *= 0.45
        return {
            "horizon": horizon,
            "score": round(float(score), 6),
            "band": self._score_to_exposure_band(score),
            "base_rate_return": mean_return,
            "hit_rate": hit_rate,
            "downside_rate": downside_rate,
            "confidence": confidence,
            "cases": cases,
        }

    def _score_to_exposure_band(self, score: float) -> List[float]:
        if score >= 0.45:
            return [0.5, 0.85]
        if score >= 0.18:
            return [0.2, 0.5]
        if score <= -0.45:
            return [-0.85, -0.5]
        if score <= -0.18:
            return [-0.5, -0.2]
        return [0.0, 0.2]

    def _aggregate_confidence(self, stats: Dict[str, Any], best_score: float) -> str:
        cases = int(stats.get("cases") or 0)
        if cases >= 30 and best_score >= 0.85:
            return "moderate"
        if cases >= 10 and best_score >= 0.75:
            return "weak_to_moderate"
        return "weak"
