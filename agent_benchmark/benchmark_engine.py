from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Protocol

import pandas as pd

from .deterministic_memory import DeterministicMarketMemory
from .llm_client import call_json_model
from .memory import HybridMemory
from .news import fetch_news_bundle
from .portfolio import execute_target_weights, initial_book, mark_to_market
from .prompting import build_stage1_prompt, build_stage2_prompt
from .schemas import BenchmarkConfig, PortfolioBook, SecretConfig, model_to_dict
from .storage import BenchmarkStore
from .warehouse.store import Warehouse
from .warehouse.universe import STOCKS, STOCK_SYMBOLS


class RunControl(Protocol):
    def checkpoint(self, run_id: str, phase: str, progress: Dict[str, Any]) -> None: ...
    def should_cancel(self, run_id: str) -> bool: ...
    def wait_if_paused(self, run_id: str) -> None: ...


def _iso(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _chunks(items: List[str], size: int) -> Iterable[List[str]]:
    size = max(1, int(size or 10))
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 8)


class BenchmarkEngine:
    horizons = {"1d": 1, "5d": 5, "20d": 20, "60d": 60}
    prompt_context_symbols = {"SPY", "QQQ", "IWM", "^VIX", "^TNX"}

    def __init__(self, warehouse: Warehouse | None = None):
        self.warehouse = warehouse or Warehouse()
        self.deterministic_memory = DeterministicMarketMemory(self.warehouse)

    def close(self) -> None:
        self.warehouse.close()

    def estimate(self, config: BenchmarkConfig) -> Dict[str, Any]:
        symbols = self._symbols(config)
        train_pairs = self._trading_pairs(config.train_start, config.train_end, config.max_train_days)
        test_pairs = self._trading_pairs(config.test_start, config.test_end, config.max_test_days)
        chunks_per_day = math.ceil(len(symbols) / max(1, config.stage1_chunk_size))
        calls_per_day = chunks_per_day + 1
        deterministic = config.memory_mode == "deterministic_market_cases"
        lesson_upper_bound = 0 if deterministic else len(self.horizons)
        decision_days = len(test_pairs) if deterministic else len(train_pairs) + len(test_pairs)
        return {
            "symbols": len(symbols),
            "train_days": len(train_pairs),
            "test_days": len(test_pairs),
            "deterministic_memory_days": len(train_pairs) if deterministic else 0,
            "stage1_chunks_per_day": chunks_per_day,
            "decision_calls_per_day": calls_per_day,
            "estimated_decision_calls": decision_days * calls_per_day,
            "estimated_lesson_calls_upper_bound": decision_days * lesson_upper_bound,
            "uncapped": config.max_train_days == 0 or config.max_test_days == 0,
            "memory_mode": config.memory_mode,
        }

    def preview(self, config: BenchmarkConfig, secrets: SecretConfig, *, phase: str = "test", decision_date: str | None = None) -> Dict[str, Any]:
        symbols = self._symbols(config)
        pairs = self._trading_pairs(
            config.train_start if phase == "train" else config.test_start,
            config.train_end if phase == "train" else config.test_end,
            1,
        )
        if not pairs:
            raise ValueError("No trading dates are available for the selected preview window.")
        pair = pairs[0]
        if decision_date:
            pair = {"decision_date": decision_date, "fill_date": pairs[0]["fill_date"]}
        memory = HybridMemory(BenchmarkStore(), config, secrets)
        book = initial_book(config.initial_cash)
        bundle = self._build_bundle(config, memory, "preview", phase, pair["decision_date"], pair["fill_date"], symbols, book)
        first_chunk = symbols[: max(1, config.stage1_chunk_size)]
        stage1_system, stage1_user = build_stage1_prompt(self._stage1_bundle(bundle, first_chunk), first_chunk)
        stage2_system, stage2_user = build_stage2_prompt(self._stage2_bundle(bundle), [{"symbol": symbol, "stance": "preview"} for symbol in first_chunk])
        return {
            "estimate": self.estimate(config),
            "decision_pair": pair,
            "bundle": bundle,
            "stage1_prompt": {"system": stage1_system, "user": json.loads(stage1_user)},
            "stage2_prompt": {"system": stage2_system, "user": json.loads(stage2_user)},
        }

    def run(
        self,
        *,
        run_id: str,
        config: BenchmarkConfig,
        secrets: SecretConfig,
        store: BenchmarkStore,
        control: RunControl | None = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        symbols = self._symbols(config)
        memory = HybridMemory(store, config, secrets)
        book = initial_book(config.initial_cash)
        equity_curve: List[Dict[str, Any]] = []
        all_executions: List[Dict[str, Any]] = []
        all_decisions: List[Dict[str, Any]] = []
        model_calls = 0
        deterministic = config.memory_mode == "deterministic_market_cases"

        store.update_benchmark_run(run_id, status="running", phase="training", started=True, progress={"percent": 0, "message": "Preparing deterministic historical memory" if deterministic else "Starting training replay"})
        memory_summary = None
        if deterministic:
            memory_summary = self.deterministic_memory.prepare(config, symbols)
            store.append_benchmark_event(
                run_id,
                "training",
                "deterministic_memory_ready",
                memory_summary.__dict__,
            )
            store.update_benchmark_run(
                run_id,
                status="running",
                phase="training",
                progress={
                    "percent": 5,
                    "message": f"Built deterministic memory from {memory_summary.cases:,} historical cases",
                    "deterministic_memory_cases": memory_summary.cases,
                },
            )

        phases = [("test", config.test_start, config.test_end, config.max_test_days)] if deterministic else [
            ("training", config.train_start, config.train_end, config.max_train_days),
            ("test", config.test_start, config.test_end, config.max_test_days),
        ]
        total_days = sum(len(self._trading_pairs(start, end, limit)) for _, start, end, limit in phases)
        completed_days = 0

        for phase, start, end, limit in phases:
            pairs = self._trading_pairs(start, end, limit)
            for pair in pairs:
                if control:
                    control.wait_if_paused(run_id)
                    if control.should_cancel(run_id):
                        store.update_benchmark_run(run_id, status="cancelled", phase=phase, progress={"percent": self._percent(completed_days, total_days), "message": "Cancelled"}, finished=True)
                        return {"status": "cancelled"}

                decision_date = pair["decision_date"]
                fill_date = pair["fill_date"]
                close_prices = self._price_map(symbols, decision_date, field="close")
                if close_prices:
                    book = mark_to_market(book, close_prices)

                bundle = self._build_bundle(config, memory, run_id, phase, decision_date, fill_date, symbols, book)
                stage1_outputs = []
                for chunk in _chunks(symbols, config.stage1_chunk_size):
                    chunk_bundle = self._stage1_bundle(bundle, chunk)
                    system, user = build_stage1_prompt(chunk_bundle, chunk)
                    fallback = self._stage1_fallback(chunk)
                    output = call_json_model(config, secrets, system, user, dry_run=dry_run, fallback=fallback, cache_namespace="stage1")
                    model_calls += 0 if output.get("_api_status") in {"dry_run", "missing_key"} else 1
                    stage1_outputs.append(output)
                    store.save_benchmark_decision(
                        run_id=run_id,
                        phase=phase,
                        decision_date=decision_date,
                        fill_date=fill_date,
                        stage="stage1",
                        symbol=",".join(chunk),
                        input_payload=chunk_bundle,
                        output_payload=output,
                        execution_payload={},
                    )

                manager_bundle = self._stage2_bundle(bundle)
                system, user = build_stage2_prompt(manager_bundle, stage1_outputs)
                stage2_output = call_json_model(config, secrets, system, user, dry_run=dry_run, fallback=self._stage2_fallback(symbols), cache_namespace="stage2")
                model_calls += 0 if stage2_output.get("_api_status") in {"dry_run", "missing_key"} else 1
                stage2_output, repair_calls = self._repair_stage2_allocation_if_needed(
                    config,
                    secrets,
                    manager_bundle,
                    stage2_output,
                    symbols,
                    dry_run=dry_run,
                    cache_namespace="stage2-repair",
                )
                model_calls += repair_calls
                fill_prices = self._price_map(symbols, fill_date, field="open")
                if not fill_prices:
                    fill_prices = self._price_map(symbols, fill_date, field="close")
                target_weights = self._coerce_target_weights(stage2_output.get("target_weights") or {}, symbols)
                book, execution = execute_target_weights(book, target_weights, fill_prices, config)
                all_executions.append(execution)
                decision_record = {
                    "phase": phase,
                    "decision_date": decision_date,
                    "fill_date": fill_date,
                    "stage1_outputs": stage1_outputs,
                    "stage2_output": stage2_output,
                    "execution": execution,
                }
                all_decisions.append(decision_record)
                store.save_benchmark_decision(
                    run_id=run_id,
                    phase=phase,
                    decision_date=decision_date,
                    fill_date=fill_date,
                    stage="stage2",
                    symbol="PORTFOLIO",
                    input_payload=manager_bundle,
                    output_payload=stage2_output,
                    execution_payload=execution,
                )
                equity_curve.append(
                    {
                        "date": fill_date,
                        "phase": phase,
                        "equity": book.equity,
                        "cash": book.cash,
                        "gross_exposure": book.gross_exposure,
                        "net_exposure": book.net_exposure,
                    }
                )
                completed_days += 1
                progress = {
                    "percent": self._percent(completed_days, total_days),
                    "phase": phase,
                    "current_decision_date": decision_date,
                    "current_fill_date": fill_date,
                    "completed_days": completed_days,
                    "total_days": total_days,
                    "model_calls": model_calls,
                    "message": f"{phase.title()} {completed_days}/{total_days}",
                }
                if control:
                    control.checkpoint(run_id, phase, progress)
                else:
                    store.update_benchmark_run(run_id, status="running", phase=phase, progress=progress)

        summary = self._summary(config, symbols, equity_curve, all_executions, model_calls, dry_run)
        if memory_summary:
            summary["deterministic_memory"] = memory_summary.__dict__
        store.update_benchmark_run(run_id, status="completed", phase="completed", summary=summary, progress={"percent": 100, "message": "Completed", "model_calls": model_calls}, finished=True)
        return {"summary": summary, "decisions": all_decisions}

    def run_live_snapshot(
        self,
        *,
        run_id: str,
        config: BenchmarkConfig,
        secrets: SecretConfig,
        store: BenchmarkStore,
        portfolio_book: PortfolioBook | None = None,
        dry_run: bool = False,
        timestamp: datetime | None = None,
    ) -> Dict[str, Any]:
        symbols = self._symbols(config)
        now = timestamp or datetime.now()
        decision_timestamp = now.astimezone().isoformat(timespec="seconds")
        decision_date = now.date().isoformat()
        memory = HybridMemory(store, config, secrets)

        store.update_benchmark_run(
            run_id,
            status="running",
            phase="live",
            started=True,
            progress={"percent": 10, "message": "Collecting live market snapshot"},
        )
        market, fill_prices, source_status = self._live_market_snapshots(symbols)
        if not fill_prices:
            raise ValueError("No live prices were available for the selected symbols.")

        book = mark_to_market(portfolio_book or initial_book(config.initial_cash), fill_prices)
        bundle = self._build_live_bundle(
            config,
            secrets,
            memory,
            run_id,
            decision_timestamp,
            decision_date,
            symbols,
            book,
            market,
            source_status,
            dry_run=dry_run,
        )

        stage1_outputs = []
        model_calls = 0
        for chunk in _chunks(symbols, config.stage1_chunk_size):
            chunk_bundle = self._stage1_bundle(bundle, chunk)
            system, user = build_stage1_prompt(chunk_bundle, chunk)
            fallback = self._stage1_fallback(chunk)
            output = call_json_model(config, secrets, system, user, dry_run=dry_run, fallback=fallback, cache_namespace="live-stage1")
            model_calls += 0 if output.get("_api_status") in {"dry_run", "missing_key"} else 1
            stage1_outputs.append(output)
            store.save_benchmark_decision(
                run_id=run_id,
                phase="live",
                decision_date=decision_timestamp,
                fill_date=decision_timestamp,
                stage="stage1",
                symbol=",".join(chunk),
                input_payload=chunk_bundle,
                output_payload=output,
                execution_payload={},
            )

        store.update_benchmark_run(
            run_id,
            status="running",
            phase="live",
            progress={"percent": 65, "message": "Asking the portfolio manager pass"},
        )
        manager_bundle = self._stage2_bundle(bundle)
        system, user = build_stage2_prompt(manager_bundle, stage1_outputs)
        stage2_output = call_json_model(config, secrets, system, user, dry_run=dry_run, fallback=self._stage2_fallback(symbols), cache_namespace="live-stage2")
        model_calls += 0 if stage2_output.get("_api_status") in {"dry_run", "missing_key"} else 1
        stage2_output, repair_calls = self._repair_stage2_allocation_if_needed(
            config,
            secrets,
            manager_bundle,
            stage2_output,
            symbols,
            dry_run=dry_run,
            cache_namespace="live-stage2-repair",
        )
        model_calls += repair_calls
        target_weights = self._coerce_target_weights(stage2_output.get("target_weights") or {}, symbols)
        next_book, execution = execute_target_weights(book, target_weights, fill_prices, config)
        store.save_benchmark_decision(
            run_id=run_id,
            phase="live",
            decision_date=decision_timestamp,
            fill_date=decision_timestamp,
            stage="stage2",
            symbol="PORTFOLIO",
            input_payload=manager_bundle,
            output_payload=stage2_output,
            execution_payload=execution,
        )
        store.append_benchmark_event(
            run_id,
            "live",
            "live_snapshot_completed",
            {
                "decision_timestamp": decision_timestamp,
                "available_prices": len(fill_prices),
                "source_status": source_status,
            },
        )
        summary = self._summary(
            config,
            symbols,
            [
                {
                    "date": decision_timestamp,
                    "phase": "live",
                    "equity": next_book.equity,
                    "cash": next_book.cash,
                    "gross_exposure": next_book.gross_exposure,
                    "net_exposure": next_book.net_exposure,
                }
            ],
            [execution],
            model_calls,
            dry_run,
        )
        summary["live"] = {
            "decision_timestamp": decision_timestamp,
            "source_status": source_status,
            "prices_available": len(fill_prices),
            "portfolio": model_to_dict(next_book),
        }
        store.update_benchmark_run(
            run_id,
            status="completed",
            phase="live",
            summary=summary,
            progress={"percent": 100, "message": "Live snapshot completed", "model_calls": model_calls},
            finished=True,
        )
        return {"summary": summary, "portfolio": next_book}

    def _symbols(self, config: BenchmarkConfig) -> List[str]:
        if config.mode == "single_stock":
            return [config.symbol.upper()]
        if config.selected_symbols:
            return [symbol.upper() for symbol in config.selected_symbols]
        return list(STOCK_SYMBOLS)

    def _trading_pairs(self, start: str, end: str, limit: int) -> List[Dict[str, str]]:
        query = """
            SELECT date
            FROM context_daily
            WHERE symbol = 'SPY'
              AND market_open = true
              AND ohlcv_available = true
              AND date <= CAST(? AS DATE)
            ORDER BY date
        """
        rows = [self._date(value[0]) for value in self.warehouse.conn.execute(query, [end]).fetchall()]
        if not rows:
            rows = [self._date(value[0]) for value in self.warehouse.conn.execute(
                """
                SELECT date FROM asset_daily
                WHERE symbol = 'AAPL' AND market_open = true AND ohlcv_available = true AND date <= CAST(? AS DATE)
                ORDER BY date
                """,
                [end],
            ).fetchall()]
        pairs = []
        for index, fill_date in enumerate(rows):
            if fill_date < start or index == 0:
                continue
            pairs.append({"decision_date": rows[index - 1], "fill_date": fill_date})
        if limit and limit > 0:
            pairs = pairs[:limit]
        return pairs

    def _date(self, value: Any) -> str:
        return _iso(value)[:10]

    def _price_map(self, symbols: List[str], target_date: str, *, field: str = "close") -> Dict[str, float]:
        if not symbols:
            return {}
        placeholders = ", ".join(["?"] * len(symbols))
        rows = self.warehouse.conn.execute(
            f"""
            SELECT symbol, {field}
            FROM asset_daily
            WHERE symbol IN ({placeholders})
              AND date = CAST(? AS DATE)
              AND ohlcv_available = true
            """,
            [*symbols, target_date],
        ).fetchall()
        return {row[0]: float(row[1]) for row in rows if row[1] is not None}

    def _market_snapshots(self, symbols: List[str], decision_date: str) -> Dict[str, Any]:
        placeholders = ", ".join(["?"] * len(symbols))
        df = self.warehouse.conn.execute(
            f"""
            SELECT date, symbol, open, high, low, close, adj_close, volume, return_1d, market_open, listed, source
            FROM asset_daily
            WHERE symbol IN ({placeholders})
              AND date <= CAST(? AS DATE)
              AND ohlcv_available = true
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) <= 65
            ORDER BY symbol, date
            """,
            [*symbols, decision_date],
        ).fetchdf()
        snapshots: Dict[str, Any] = {}
        if df.empty:
            return snapshots
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values("date")
            latest = group.iloc[-1]
            closes = group["close"].astype(float)
            def trailing(days: int) -> float | None:
                if len(closes) <= days:
                    return None
                base = float(closes.iloc[-days - 1])
                return float(closes.iloc[-1]) / base - 1.0 if base else None
            snapshots[symbol] = {
                "symbol": symbol,
                "as_of_date": self._date(latest["date"]),
                "open": _safe_float(latest["open"]),
                "high": _safe_float(latest["high"]),
                "low": _safe_float(latest["low"]),
                "close": _safe_float(latest["close"]),
                "adj_close": _safe_float(latest["adj_close"]),
                "volume": _safe_float(latest["volume"]),
                "return_1d": _safe_float(latest["return_1d"]),
                "return_5d": _pct(trailing(5)),
                "return_20d": _pct(trailing(20)),
                "return_60d": _pct(trailing(60)),
                "volatility_20d": _safe_float(closes.pct_change().tail(20).std() * (252 ** 0.5)) if len(closes) > 20 else None,
                "source": latest.get("source", ""),
                "history_points": int(len(group)),
            }
        return snapshots

    def _context(self, decision_date: str) -> List[Dict[str, Any]]:
        rows = self.warehouse.conn.execute(
            """
            SELECT date, symbol, close, return_1d, source
            FROM context_daily
            WHERE date <= CAST(? AS DATE) AND ohlcv_available = true
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) = 1
            ORDER BY symbol
            """,
            [decision_date],
        ).fetchdf()
        return rows.to_dict(orient="records") if not rows.empty else []

    def _news(self, symbols: List[str], decision_date: str, limit_per_symbol: int) -> Dict[str, List[Dict[str, Any]]]:
        if not symbols or limit_per_symbol <= 0:
            return {}
        placeholders = ", ".join(["?"] * len(symbols))
        df = self.warehouse.conn.execute(
            f"""
            SELECT symbol, published_at, title, url, domain, source_country, source, raw_json
            FROM news_articles
            WHERE symbol IN ({placeholders})
              AND bucket_start BETWEEN CAST(? AS DATE) - INTERVAL 5 DAY AND CAST(? AS DATE)
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY bucket_start DESC, published_at DESC, article_id DESC) <= ?
            ORDER BY symbol, published_at DESC
            """,
            [*symbols, decision_date, decision_date, limit_per_symbol],
        ).fetchdf()
        out: Dict[str, List[Dict[str, Any]]] = {}
        if df.empty:
            return out
        for symbol, group in df.groupby("symbol"):
            items = []
            for _, row in group.iterrows():
                raw = {}
                try:
                    raw = json.loads(row.get("raw_json") or "{}")
                except Exception:
                    raw = {}
                items.append(
                    {
                        "published_at": str(row.get("published_at") or ""),
                        "title": row.get("title") or "",
                        "url": row.get("url") or "",
                        "domain": row.get("domain") or "",
                        "source_country": row.get("source_country") or "",
                        "source": row.get("source") or "",
                        "tone": raw.get("AvgTone"),
                        "mentions": raw.get("NumMentions"),
                        "event_code": raw.get("EventCode"),
                    }
                )
            out[symbol] = items
        return out

    def _fundamentals(self, symbols: List[str], decision_date: str) -> Dict[str, Any]:
        placeholders = ", ".join(["?"] * len(symbols))
        df = self.warehouse.conn.execute(
            f"""
            SELECT symbol, concept, value, unit, period_end, filed_date, form
            FROM sec_facts
            WHERE symbol IN ({placeholders}) AND filed_date <= CAST(? AS DATE)
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol, concept ORDER BY filed_date DESC, period_end DESC) = 1
            ORDER BY symbol, concept
            """,
            [*symbols, decision_date],
        ).fetchdf()
        out: Dict[str, Any] = {}
        if df.empty:
            return out
        for symbol, group in df.groupby("symbol"):
            out[symbol] = {
                str(row["concept"]): {
                    "value": _safe_float(row["value"]),
                    "unit": row["unit"],
                    "period_end": self._date(row["period_end"]),
                    "filed_date": self._date(row["filed_date"]),
                    "form": row["form"],
                }
                for _, row in group.iterrows()
            }
        return out

    def _macro(self, decision_date: str) -> List[Dict[str, Any]]:
        df = self.warehouse.conn.execute(
            """
            SELECT series_id, label, observation_date, value, source_status
            FROM macro_daily
            WHERE date <= CAST(? AS DATE)
            QUALIFY ROW_NUMBER() OVER (PARTITION BY series_id ORDER BY date DESC) = 1
            ORDER BY series_id
            """,
            [decision_date],
        ).fetchdf()
        return df.to_dict(orient="records") if not df.empty else []

    def _data_quality(self, symbols: List[str], decision_date: str) -> Dict[str, Any]:
        placeholders = ", ".join(["?"] * len(symbols))
        rows = self.warehouse.conn.execute(
            f"""
            SELECT symbol, ohlcv_available, listed, source_error
            FROM asset_daily
            WHERE symbol IN ({placeholders}) AND date = CAST(? AS DATE)
            """,
            [*symbols, decision_date],
        ).fetchdf()
        missing = []
        unavailable = []
        for _, row in rows.iterrows():
            if not row["ohlcv_available"]:
                missing.append(row["symbol"])
            if not row["listed"]:
                unavailable.append(row["symbol"])
        return {
            "checked_symbols": len(symbols),
            "missing_ohlcv": missing,
            "not_listed": unavailable,
            "status": "ok" if not missing else "partial",
        }

    def _build_bundle(
        self,
        config: BenchmarkConfig,
        memory: HybridMemory,
        run_id: str,
        phase: str,
        decision_date: str,
        fill_date: str,
        symbols: List[str],
        book: PortfolioBook,
    ) -> Dict[str, Any]:
        market = self._market_snapshots(symbols, decision_date)
        news = self._news(symbols, decision_date, config.max_news_per_symbol)
        query = json.dumps({"phase": phase, "date": decision_date, "portfolio": book.model_dump(), "market": market}, default=str)[:6000]
        if config.memory_mode == "deterministic_market_cases":
            memories = self.deterministic_memory.retrieve(
                config,
                symbols,
                decision_date,
                market,
                limit_per_symbol=config.deterministic_memory_per_symbol,
                max_items=config.deterministic_memory_max_items,
            )
        else:
            memories = memory.retrieve(decision_timestamp=decision_date, query=query, limit=14)
        return {
            "schema_version": "benchmark-input-v2",
            "mode": config.mode,
            "run_id": run_id,
            "phase": phase,
            "decision_date": decision_date,
            "fill_date": fill_date,
            "information_cutoff": f"Only information available on or before {decision_date} may be used. Trades fill at {fill_date} open.",
            "candidate_universe": self._symbol_metadata(symbols),
            "portfolio_state": book.model_dump() if hasattr(book, "model_dump") else book.dict(),
            "market_snapshots": market,
            "index_context": self._context(decision_date),
            "fundamentals": self._fundamentals(symbols, decision_date),
            "macro_context": self._macro(decision_date),
            "news_and_events": news,
            "data_quality": self._data_quality(symbols, decision_date),
            "memory": memories,
            "benchmark_rules": {
                "model_owns_decision": True,
                "simulator_role": "mechanics_only",
                "allow_short": config.allow_short,
                "max_gross_exposure": config.max_gross_exposure,
                "fill_timing": config.fill_timing,
            },
        }

    def _stage1_bundle(self, bundle: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        symbol_set = set(symbols)

        def by_symbol(section: Dict[str, Any] | None) -> Dict[str, Any]:
            return {symbol: value for symbol, value in (section or {}).items() if symbol in symbol_set}

        data_quality = dict(bundle.get("data_quality") or {})
        for key in ("missing_ohlcv", "not_listed", "missing_live_prices"):
            if isinstance(data_quality.get(key), list):
                data_quality[key] = [symbol for symbol in data_quality[key] if symbol in symbol_set]
        data_quality["checked_symbols"] = len(symbols)
        return {
            "schema_version": bundle.get("schema_version"),
            "mode": bundle.get("mode"),
            "run_id": bundle.get("run_id"),
            "phase": bundle.get("phase"),
            "decision_date": bundle.get("decision_date"),
            "fill_date": bundle.get("fill_date"),
            "information_cutoff": bundle.get("information_cutoff"),
            "symbols": symbols,
            "candidate_universe": [item for item in (bundle.get("candidate_universe") or []) if item.get("symbol") in symbol_set],
            "portfolio_state": bundle.get("portfolio_state"),
            "market_snapshots": by_symbol(bundle.get("market_snapshots")),
            "index_context": self._compact_context(bundle.get("index_context") or []),
            "fundamentals": by_symbol(bundle.get("fundamentals")),
            "macro_context": bundle.get("macro_context") or [],
            "news_and_events": self._compact_news(by_symbol(bundle.get("news_and_events"))),
            "data_quality": data_quality,
            "memory": self._filter_memory(bundle.get("memory") or [], symbol_set, limit=max(8, len(symbols) * 2)),
            "benchmark_rules": bundle.get("benchmark_rules") or {},
        }

    def _stage2_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        market = bundle.get("market_snapshots") or {}
        news = bundle.get("news_and_events") or {}
        symbol_table = []
        for item in bundle.get("candidate_universe") or []:
            symbol = item.get("symbol")
            snap = market.get(symbol) or {}
            symbol_table.append(
                {
                    "symbol": symbol,
                    "sector": item.get("sector", ""),
                    "r5": snap.get("return_5d"),
                    "r20": snap.get("return_20d"),
                    "r60": snap.get("return_60d"),
                    "vol20": snap.get("volatility_20d"),
                    "news_items": len(news.get(symbol) or []),
                    "data_as_of": snap.get("as_of_date"),
                }
            )
        return {
            "schema_version": bundle.get("schema_version"),
            "mode": bundle.get("mode"),
            "run_id": bundle.get("run_id"),
            "phase": bundle.get("phase"),
            "decision_date": bundle.get("decision_date"),
            "fill_date": bundle.get("fill_date"),
            "information_cutoff": bundle.get("information_cutoff"),
            "candidate_universe": bundle.get("candidate_universe") or [],
            "portfolio_state": bundle.get("portfolio_state"),
            "symbol_summary_table": symbol_table,
            "index_context": self._compact_context(bundle.get("index_context") or []),
            "macro_context": bundle.get("macro_context") or [],
            "data_quality": bundle.get("data_quality") or {},
            "memory": self._filter_memory(bundle.get("memory") or [], set(), limit=10),
            "benchmark_rules": bundle.get("benchmark_rules") or {},
            "omitted_raw_sections": ["market_snapshots", "fundamentals", "news_and_events"],
        }

    def _compact_context(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for item in context:
            symbol = item.get("symbol")
            if symbol not in self.prompt_context_symbols:
                continue
            compact.append(
                {
                    "symbol": symbol,
                    "date": item.get("date"),
                    "close": item.get("close"),
                    "return_1d": item.get("return_1d"),
                }
            )
        return compact

    def _compact_news(self, news: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        compact: Dict[str, List[Dict[str, Any]]] = {}
        for symbol, items in (news or {}).items():
            compact[symbol] = [
                {
                    "published_at": item.get("published_at"),
                    "title": item.get("title"),
                    "domain": item.get("domain"),
                    "tone": item.get("tone"),
                    "mentions": item.get("mentions"),
                    "event_code": item.get("event_code"),
                }
                for item in (items or [])
            ]
        return compact

    def _filter_memory(self, memories: List[Dict[str, Any]], symbols: set[str], *, limit: int) -> List[Dict[str, Any]]:
        filtered = []
        for item in memories:
            symbol = item.get("symbol") or ""
            if symbols and symbol and symbol not in symbols:
                continue
            filtered.append(
                {
                    "id": item.get("id"),
                    "memory_type": item.get("memory_type"),
                    "symbol": symbol,
                    "decision_timestamp": item.get("decision_timestamp"),
                    "knowledge_timestamp": item.get("knowledge_timestamp"),
                    "content": item.get("content"),
                    "retrieval_score": item.get("retrieval_score"),
                }
            )
            if len(filtered) >= limit:
                break
        return filtered

    def _build_live_bundle(
        self,
        config: BenchmarkConfig,
        secrets: SecretConfig,
        memory: HybridMemory,
        run_id: str,
        decision_timestamp: str,
        decision_date: str,
        symbols: List[str],
        book: PortfolioBook,
        market: Dict[str, Any],
        source_status: List[Dict[str, Any]],
        *,
        dry_run: bool,
    ) -> Dict[str, Any]:
        query = json.dumps({"phase": "live", "timestamp": decision_timestamp, "portfolio": model_to_dict(book), "market": market}, default=str)[:6000]
        if config.memory_mode == "deterministic_market_cases":
            memories = self.deterministic_memory.retrieve(
                config,
                symbols,
                decision_date,
                market,
                limit_per_symbol=config.deterministic_memory_per_symbol,
                max_items=config.deterministic_memory_max_items,
            )
        else:
            memories = memory.retrieve(decision_timestamp=decision_timestamp, query=query, limit=14)
        return {
            "schema_version": "benchmark-input-v2-live",
            "mode": config.mode,
            "run_id": run_id,
            "phase": "live",
            "decision_date": decision_timestamp,
            "fill_date": decision_timestamp,
            "information_cutoff": f"Only information available on or before {decision_timestamp} may be used. The paper trade fills at the latest available live price.",
            "candidate_universe": self._symbol_metadata(symbols),
            "portfolio_state": model_to_dict(book),
            "market_snapshots": market,
            "index_context": self._live_context(config, source_status),
            "fundamentals": self._fundamentals(symbols, decision_date),
            "macro_context": self._macro(decision_date),
            "news_and_events": self._live_news(config, secrets, symbols, decision_date, dry_run=dry_run),
            "data_quality": {
                "checked_symbols": len(symbols),
                "missing_live_prices": [symbol for symbol in symbols if symbol not in market],
                "source_status": source_status,
                "status": "ok" if len(market) == len(symbols) else "partial",
            },
            "memory": memories,
            "benchmark_rules": {
                "model_owns_decision": True,
                "simulator_role": "mechanics_only",
                "allow_short": config.allow_short,
                "max_gross_exposure": config.max_gross_exposure,
                "fill_timing": "live_latest_price",
            },
        }

    def _live_market_snapshots(self, symbols: List[str]) -> tuple[Dict[str, Any], Dict[str, float], List[Dict[str, Any]]]:
        snapshots: Dict[str, Any] = {}
        prices: Dict[str, float] = {}
        status: List[Dict[str, Any]] = []
        if not symbols:
            return snapshots, prices, status
        try:
            import yfinance as yf

            daily = yf.download(" ".join(symbols), period="6mo", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
            intraday = yf.download(" ".join(symbols), period="1d", interval="5m", group_by="ticker", auto_adjust=False, progress=False, threads=True)
            for symbol in symbols:
                day_frame = self._yf_symbol_frame(daily, symbol, symbols)
                intra_frame = self._yf_symbol_frame(intraday, symbol, symbols)
                snapshot = self._snapshot_from_live_frames(symbol, day_frame, intra_frame)
                if snapshot:
                    snapshots[symbol] = snapshot
                    prices[symbol] = float(snapshot["close"])
                    status.append({"symbol": symbol, "source": "yfinance", "status": "ok", "as_of": snapshot.get("as_of_timestamp")})
                else:
                    status.append({"symbol": symbol, "source": "yfinance", "status": "missing"})
        except Exception as exc:
            status.append({"source": "yfinance", "status": "error", "message": str(exc)})
        return snapshots, prices, status

    def _yf_symbol_frame(self, raw: pd.DataFrame, symbol: str, symbols: List[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        frame = pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            first_level = [str(value) for value in raw.columns.get_level_values(0)]
            second_level = [str(value) for value in raw.columns.get_level_values(1)]
            candidates = [symbol, symbol.replace(".", "-"), symbol.replace("-", ".")]
            for candidate in candidates:
                if candidate in first_level:
                    frame = raw[candidate].copy()
                    break
                if candidate in second_level:
                    frame = raw.xs(candidate, level=1, axis=1).copy()
                    break
        elif len(symbols) == 1:
            frame = raw.copy()
        if frame.empty:
            return frame
        frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]
        if "adj_close" not in frame and "adj_close" in frame.columns:
            frame["adj_close"] = frame["adj_close"]
        return frame.dropna(how="all")

    def _snapshot_from_live_frames(self, symbol: str, daily: pd.DataFrame, intraday: pd.DataFrame) -> Dict[str, Any] | None:
        if daily is None or daily.empty or "close" not in daily.columns:
            return None
        daily = daily.dropna(subset=["close"]).sort_index()
        if daily.empty:
            return None
        intraday = intraday.dropna(subset=["close"]).sort_index() if intraday is not None and not intraday.empty and "close" in intraday.columns else pd.DataFrame()
        latest_bar = intraday.iloc[-1] if not intraday.empty else daily.iloc[-1]
        latest_index = intraday.index[-1] if not intraday.empty else daily.index[-1]
        closes = daily["close"].astype(float)
        current = _safe_float(latest_bar.get("close")) or float(closes.iloc[-1])
        previous_close = float(closes.iloc[-2]) if len(closes) > 1 else current

        def trailing(days: int) -> float | None:
            if len(closes) <= days:
                return None
            base = float(closes.iloc[-days - 1])
            return current / base - 1.0 if base else None

        timestamp = latest_index.isoformat() if hasattr(latest_index, "isoformat") else str(latest_index)
        return {
            "symbol": symbol,
            "as_of_date": str(latest_index)[:10],
            "as_of_timestamp": timestamp,
            "source": "yfinance_live",
            "open": _safe_float(latest_bar.get("open")),
            "high": _safe_float(latest_bar.get("high")),
            "low": _safe_float(latest_bar.get("low")),
            "close": current,
            "adj_close": _safe_float(daily.iloc[-1].get("adj_close")) if "adj_close" in daily.columns else None,
            "volume": _safe_float(latest_bar.get("volume")),
            "previous_close": previous_close,
            "return_1d": current / previous_close - 1.0 if previous_close else None,
            "return_5d": _pct(trailing(5)),
            "return_20d": _pct(trailing(20)),
            "return_60d": _pct(trailing(60)),
            "volatility_20d": _safe_float(closes.pct_change().tail(20).std() * (252 ** 0.5)) if len(closes) > 20 else None,
            "history_points": int(len(daily)),
        }

    def _live_context(self, config: BenchmarkConfig, source_status: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not config.data_sources.include_index_context:
            return []
        context_symbols = [symbol.upper() for symbol in config.data_sources.index_symbols]
        snapshots, _, status = self._live_market_snapshots(context_symbols)
        source_status.extend([{"context_symbol": item.get("symbol"), **item} for item in status])
        return list(snapshots.values())

    def _live_news(self, config: BenchmarkConfig, secrets: SecretConfig, symbols: List[str], decision_date: str, *, dry_run: bool) -> Dict[str, List[Dict[str, Any]]]:
        if dry_run or config.max_news_per_symbol <= 0:
            return {}
        meta = {item["symbol"]: item for item in self._symbol_metadata(symbols)}
        news: Dict[str, List[Dict[str, Any]]] = {}
        for symbol in symbols:
            scoped = config.model_copy(deep=True) if hasattr(config, "model_copy") else BenchmarkConfig(**config.dict())
            scoped.symbol = symbol
            scoped.company_name = meta.get(symbol, {}).get("name") or symbol
            scoped.data_sources.max_news_per_day = config.max_news_per_symbol
            try:
                news[symbol] = fetch_news_bundle(scoped, secrets, decision_date).get("items", [])
            except Exception as exc:
                news[symbol] = [{"source": "news", "title": "Live news fetch failed", "summary": str(exc), "url": "", "published_at": decision_date}]
        return news

    def _symbol_metadata(self, symbols: List[str]) -> List[Dict[str, Any]]:
        meta = {item.symbol: item for item in STOCKS}
        return [{"symbol": symbol, "name": meta.get(symbol).name if symbol in meta else symbol, "sector": meta.get(symbol).sector if symbol in meta else ""} for symbol in symbols]

    def _stage1_fallback(self, symbols: List[str]) -> Dict[str, Any]:
        return {
            "analyses": [
                {
                    "symbol": symbol,
                    "stance": "neutral",
                    "confidence": 0.0,
                    "expected_return_bps": 0,
                    "horizon_days": 1,
                    "key_evidence": ["Fallback/no model call."],
                    "memory_refs": [],
                    "uncertainty": ["No model analysis was produced."],
                    "proposed_target_weight": 0.0,
                }
                for symbol in symbols
            ],
            "market_regime_notes": "Fallback/no model call.",
            "data_quality_notes": [],
        }

    def _stage2_fallback(self, symbols: List[str]) -> Dict[str, Any]:
        return {
            "target_weights": {symbol: 0.0 for symbol in symbols},
            "cash_target_weight": 1.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "confidence": 0.0,
            "portfolio_thesis": "Fallback/no model call.",
            "major_risks": [],
            "uncertainty": ["No model allocation was produced."],
            "expected_return_bps": 0,
            "horizon_days": 1,
        }

    def _repair_stage2_allocation_if_needed(
        self,
        config: BenchmarkConfig,
        secrets: SecretConfig,
        manager_bundle: Dict[str, Any],
        stage2_output: Dict[str, Any],
        symbols: List[str],
        *,
        dry_run: bool,
        cache_namespace: str,
    ) -> tuple[Dict[str, Any], int]:
        errors = self._allocation_errors(config, stage2_output, symbols)
        if not errors or stage2_output.get("_api_status") in {"dry_run", "missing_key"}:
            return stage2_output, 0

        repair_system = """You are the portfolio manager stage of an AI market benchmark.

Your previous Stage 2 allocation violated hard benchmark constraints. You still
own the decision. Correct your own target weights; the simulator will not scale
or improve them. Return only valid compact JSON with the same Stage 2 schema.
"""
        repair_user = json.dumps(
            {
                "task": "Repair your previous portfolio allocation and return valid JSON only.",
                "hard_rules": {
                    "allow_short": config.allow_short,
                    "max_gross_exposure": config.max_gross_exposure,
                    "max_nonzero_positions": 12,
                    "gross_exposure_formula": "sum(abs(target_weights.values()))",
                    "net_exposure_formula": "sum(target_weights.values())",
                    "zero_weight_policy": "omit zero weights",
                    "safe_fallback": "all cash is valid if you cannot make a compliant allocation",
                },
                "validation_errors": errors,
                "previous_stage2_output": self._strip_api_metadata(stage2_output),
                "portfolio_state": manager_bundle.get("portfolio_state"),
                "symbol_summary_table": manager_bundle.get("symbol_summary_table"),
                "benchmark_rules": manager_bundle.get("benchmark_rules"),
            },
            sort_keys=True,
            default=str,
        )
        try:
            repaired = call_json_model(
                config,
                secrets,
                repair_system,
                repair_user,
                dry_run=dry_run,
                fallback=stage2_output,
                cache_namespace=cache_namespace,
            )
        except Exception as exc:
            stage2_output["_allocation_repair_error"] = str(exc)
            stage2_output["_allocation_repair_errors"] = errors
            return stage2_output, 0

        repaired_errors = self._allocation_errors(config, repaired, symbols)
        repaired["_allocation_repair"] = {
            "attempted": True,
            "original_errors": errors,
            "remaining_errors": repaired_errors,
        }
        calls = 0 if repaired.get("_api_status") in {"dry_run", "missing_key"} else 1
        return repaired, calls

    def _allocation_errors(self, config: BenchmarkConfig, output: Dict[str, Any], symbols: List[str]) -> List[Dict[str, Any]]:
        weights = self._coerce_target_weights(output.get("target_weights") or {}, symbols)
        gross = sum(abs(value) for value in weights.values())
        errors: List[Dict[str, Any]] = []
        if gross > config.max_gross_exposure + 1e-9:
            errors.append({"type": "gross_exposure_exceeded", "actual": round(gross, 8), "max": config.max_gross_exposure})
        if not config.allow_short:
            shorts = {symbol: value for symbol, value in weights.items() if value < 0}
            if shorts:
                errors.append({"type": "shorts_not_allowed", "symbols": sorted(shorts)})
        return errors

    def _strip_api_metadata(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in (output or {}).items() if not str(key).startswith("_")}

    def _coerce_target_weights(self, raw: Dict[str, Any], symbols: List[str]) -> Dict[str, float]:
        allowed = set(symbols)
        weights = {}
        for symbol, value in (raw or {}).items():
            symbol = str(symbol).upper().strip()
            if symbol not in allowed:
                continue
            try:
                weights[symbol] = float(value)
            except Exception:
                continue
        return weights

    def _nth_trading_date_after(self, fill_date: str, days: int) -> str | None:
        rows = self.warehouse.conn.execute(
            """
            SELECT date
            FROM context_daily
            WHERE symbol = 'SPY'
              AND market_open = true
              AND ohlcv_available = true
              AND date > CAST(? AS DATE)
            ORDER BY date
            LIMIT ?
            """,
            [fill_date, days],
        ).fetchall()
        if len(rows) < days:
            return None
        return self._date(rows[-1][0])

    def _summary(
        self,
        config: BenchmarkConfig,
        symbols: List[str],
        equity_curve: List[Dict[str, Any]],
        executions: List[Dict[str, Any]],
        model_calls: int,
        dry_run: bool,
    ) -> Dict[str, Any]:
        initial = float(config.initial_cash)
        final = float(equity_curve[-1]["equity"]) if equity_curve else initial
        peak = initial
        max_drawdown = 0.0
        returns = []
        previous = initial
        for point in equity_curve:
            equity = float(point["equity"])
            if previous:
                returns.append(equity / previous - 1.0)
            previous = equity
            peak = max(peak, equity)
            if peak:
                max_drawdown = min(max_drawdown, equity / peak - 1.0)
        volatility = float(pd.Series(returns).std() * (252 ** 0.5)) if len(returns) > 1 else 0.0
        average_return = float(pd.Series(returns).mean() * 252) if returns else 0.0
        sharpe_like = average_return / volatility if volatility else 0.0
        event_count = sum(len(execution.get("events") or []) for execution in executions)
        model_failures = sum(1 for execution in executions if execution.get("model_failure"))
        fees = sum(float(execution.get("fees") or 0.0) for execution in executions)
        slippage = sum(float(execution.get("slippage_cost") or 0.0) for execution in executions)
        summary = {
            "mode": config.mode,
            "preset": config.run_preset,
            "symbol_count": len(symbols),
            "model": config.model or "unselected-model",
            "dry_run": dry_run,
            "days": len(equity_curve),
            "model_calls": model_calls,
            "metrics": {
                "final_equity": final,
                "total_return": final / initial - 1.0 if initial else 0.0,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "sharpe_like": sharpe_like,
                "event_count": event_count,
                "model_failures": model_failures,
                "fees": fees,
                "slippage_cost": slippage,
                "alpha_spy": None,
                "alpha_qqq": None,
                "alpha_equal_weight_balanced_50": None,
            },
            "equity_curve": equity_curve,
        }
        return self.enrich_summary_with_buy_hold(config, symbols, summary)

    def enrich_summary_with_buy_hold(
        self,
        config: BenchmarkConfig,
        symbols: List[str],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        equity_curve = summary.get("equity_curve") or []
        metrics = dict(summary.get("metrics") or {})
        if not equity_curve:
            summary["metrics"] = metrics
            return summary

        initial = float(config.initial_cash or 0.0)
        ai_return = _safe_float(metrics.get("total_return"))
        comparison = self._buy_hold_comparison(config, symbols, equity_curve, initial, ai_return)
        if comparison.get("benchmarks"):
            summary["buy_hold_comparison"] = comparison
            for benchmark in comparison["benchmarks"]:
                key = benchmark.get("id")
                if key == "spy":
                    metrics["alpha_spy"] = benchmark.get("excess_return")
                elif key == "qqq":
                    metrics["alpha_qqq"] = benchmark.get("excess_return")
                elif key == "balanced_50_equal_weight":
                    metrics["alpha_equal_weight_balanced_50"] = benchmark.get("excess_return")
                elif key == "selected_equal_weight":
                    metrics["alpha_equal_weight_selected"] = benchmark.get("excess_return")
                elif key == "single_stock":
                    metrics["alpha_single_stock_buy_hold"] = benchmark.get("excess_return")
        summary["metrics"] = metrics
        return summary

    def _buy_hold_comparison(
        self,
        config: BenchmarkConfig,
        symbols: List[str],
        equity_curve: List[Dict[str, Any]],
        initial: float,
        ai_return: float | None,
    ) -> Dict[str, Any]:
        start_date = self._date(equity_curve[0].get("date"))
        end_date = self._date(equity_curve[-1].get("date"))
        benchmarks: List[Dict[str, Any]] = []

        if config.mode == "single_stock" and symbols:
            single = self._instrument_buy_hold(
                table="asset_daily",
                symbol=symbols[0],
                label=f"{symbols[0]} buy & hold",
                benchmark_id="single_stock",
                start_date=start_date,
                end_date=end_date,
                initial=initial,
                ai_return=ai_return,
            )
            if single:
                benchmarks.append(single)

        for symbol, label in (("SPY", "SPY buy & hold"), ("QQQ", "QQQ buy & hold")):
            benchmark = self._instrument_buy_hold(
                table="context_daily",
                symbol=symbol,
                label=label,
                benchmark_id=symbol.lower(),
                start_date=start_date,
                end_date=end_date,
                initial=initial,
                ai_return=ai_return,
            )
            if benchmark:
                benchmarks.append(benchmark)

        if symbols:
            equal_weight_id = "balanced_50_equal_weight" if set(symbols) == set(STOCK_SYMBOLS) else "selected_equal_weight"
            equal_weight_label = (
                "Balanced 50 equal-weight buy & hold"
                if equal_weight_id == "balanced_50_equal_weight"
                else "Selected universe equal-weight buy & hold"
            )
            equal_weight = self._equal_weight_buy_hold(
                symbols=symbols,
                label=equal_weight_label,
                benchmark_id=equal_weight_id,
                start_date=start_date,
                end_date=end_date,
                initial=initial,
                ai_return=ai_return,
            )
            if equal_weight:
                benchmarks.append(equal_weight)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial,
            "ai_strategy": {
                "label": "AI strategy",
                "final_equity": _safe_float((equity_curve[-1] or {}).get("equity")),
                "total_return": ai_return,
            },
            "benchmarks": benchmarks,
        }

    def _instrument_buy_hold(
        self,
        *,
        table: str,
        symbol: str,
        label: str,
        benchmark_id: str,
        start_date: str,
        end_date: str,
        initial: float,
        ai_return: float | None,
    ) -> Dict[str, Any] | None:
        if table not in {"asset_daily", "context_daily"}:
            return None
        df = self.warehouse.conn.execute(
            f"""
            SELECT date, adj_close
            FROM {table}
            WHERE symbol = ?
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND ohlcv_available = true
              AND adj_close IS NOT NULL
              AND adj_close > 0
            ORDER BY date
            """,
            [symbol, start_date, end_date],
        ).fetchdf()
        if df.empty:
            return None
        first_price = float(df.iloc[0]["adj_close"])
        if not first_price:
            return None
        values = [initial * float(price) / first_price for price in df["adj_close"]]
        final = float(values[-1]) if values else initial
        total_return = final / initial - 1.0 if initial else 0.0
        return {
            "id": benchmark_id,
            "label": label,
            "symbol": symbol,
            "method": "single_instrument_buy_hold",
            "start_date": self._date(df.iloc[0]["date"]),
            "end_date": self._date(df.iloc[-1]["date"]),
            "observations": int(len(df)),
            "symbols": 1,
            "final_equity": final,
            "total_return": total_return,
            "excess_return": ai_return - total_return if ai_return is not None else None,
            "max_drawdown": self._series_max_drawdown(values, initial),
        }

    def _equal_weight_buy_hold(
        self,
        *,
        symbols: List[str],
        label: str,
        benchmark_id: str,
        start_date: str,
        end_date: str,
        initial: float,
        ai_return: float | None,
    ) -> Dict[str, Any] | None:
        if not symbols:
            return None
        placeholders = ", ".join(["?"] * len(symbols))
        start_end = self.warehouse.conn.execute(
            f"""
            SELECT symbol,
                   MAX(CASE WHEN date = CAST(? AS DATE) THEN adj_close END) AS start_price,
                   MAX(CASE WHEN date = CAST(? AS DATE) THEN adj_close END) AS end_price
            FROM asset_daily
            WHERE symbol IN ({placeholders})
              AND date IN (CAST(? AS DATE), CAST(? AS DATE))
              AND ohlcv_available = true
              AND listed = true
              AND adj_close IS NOT NULL
              AND adj_close > 0
            GROUP BY symbol
            """,
            [start_date, end_date, *symbols, start_date, end_date],
        ).fetchdf()
        if start_end.empty:
            return None
        eligible = start_end.dropna(subset=["start_price", "end_price"])
        eligible_symbols = [str(symbol) for symbol in eligible["symbol"].tolist()]
        if not eligible_symbols:
            return None

        placeholders = ", ".join(["?"] * len(eligible_symbols))
        df = self.warehouse.conn.execute(
            f"""
            SELECT date, symbol, adj_close
            FROM asset_daily
            WHERE symbol IN ({placeholders})
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND ohlcv_available = true
              AND listed = true
              AND adj_close IS NOT NULL
              AND adj_close > 0
            ORDER BY date, symbol
            """,
            [*eligible_symbols, start_date, end_date],
        ).fetchdf()
        if df.empty:
            return None

        first_prices = {
            str(row["symbol"]): float(row["start_price"])
            for _, row in eligible.iterrows()
            if float(row["start_price"] or 0.0) > 0
        }
        if not first_prices:
            return None
        allocation = initial / len(first_prices) if first_prices else 0.0
        shares = {symbol: allocation / price for symbol, price in first_prices.items()}

        pivot = df.pivot(index="date", columns="symbol", values="adj_close").sort_index().ffill()
        values = []
        for _, row in pivot.iterrows():
            value = sum(float(row.get(symbol) or 0.0) * shares[symbol] for symbol in shares)
            values.append(value)
        if not values:
            return None

        final = float(values[-1])
        total_return = final / initial - 1.0 if initial else 0.0
        missing_symbols = sorted(set(symbols) - set(first_prices))
        return {
            "id": benchmark_id,
            "label": label,
            "method": "equal_weight_buy_hold",
            "start_date": self._date(pivot.index[0]),
            "end_date": self._date(pivot.index[-1]),
            "observations": int(len(pivot)),
            "symbols": len(first_prices),
            "missing_symbols": missing_symbols,
            "final_equity": final,
            "total_return": total_return,
            "excess_return": ai_return - total_return if ai_return is not None else None,
            "max_drawdown": self._series_max_drawdown(values, initial),
        }

    def _series_max_drawdown(self, values: List[float], initial: float) -> float:
        peak = float(initial or 0.0)
        max_drawdown = 0.0
        for value in values:
            equity = float(value)
            peak = max(peak, equity)
            if peak:
                max_drawdown = min(max_drawdown, equity / peak - 1.0)
        return max_drawdown

    def _percent(self, done: int, total: int) -> int:
        if not total:
            return 0
        return int(round(100 * done / total))
