from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Protocol

import pandas as pd

from .api_usage import estimate_run_api_usage
from .deterministic_memory import DeterministicMarketMemory
from .llm_client import call_json_model
from .local_provider import validate_no_paid_api_mode
from .memory import HybridMemory
from .news import fetch_news_bundle
from .portfolio import estimate_target_turnover, execute_target_weights, initial_book, mark_to_market, reject_target_weights
from .prompting import build_exposure_critic_prompt, build_reflection_lesson_prompt, build_stage1_prompt, build_stage2_prompt
from .quality import canonicalize_fundamentals, is_synthetic_news_title
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


def _bounded(value: float | None, scale: float) -> float:
    if value is None or not scale:
        return 0.0
    return max(-1.0, min(1.0, float(value) / float(scale)))


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
        critic_calls_per_day = 1 if config.mode == "single_stock" and config.exposure_critic_enabled else 0
        calls_per_day = chunks_per_day + 1 + critic_calls_per_day
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
            "exposure_critic_calls_per_day": critic_calls_per_day,
            "estimated_decision_calls": decision_days * calls_per_day,
            "estimated_lesson_calls_upper_bound": decision_days * lesson_upper_bound,
            "uncapped": config.max_train_days == 0 or config.max_test_days == 0,
            "memory_mode": config.memory_mode,
        }

    def preview(self, config: BenchmarkConfig, secrets: SecretConfig, *, phase: str = "test", decision_date: str | None = None) -> Dict[str, Any]:
        validate_no_paid_api_mode(config, secrets)
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
        stage2_system, stage2_user = build_stage2_prompt(self._stage2_bundle(bundle, config), [{"symbol": symbol, "stance": "preview"} for symbol in first_chunk])
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
        resume: bool = False,
    ) -> Dict[str, Any]:
        validate_no_paid_api_mode(config, secrets)
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
        invalid_stage2_days = 0
        completed_keys = set()
        if resume:
            resume_state = self._load_resume_state(store, run_id, config)
            book = resume_state["book"]
            equity_curve = resume_state["equity_curve"]
            all_executions = resume_state["executions"]
            all_decisions = resume_state["decisions"]
            model_calls = resume_state["model_calls"]
            completed_days = len(all_decisions)
            invalid_stage2_days = resume_state["invalid_stage2_days"]
            completed_keys = resume_state["completed_keys"]
            store.update_benchmark_run(
                run_id,
                status="running",
                phase=(all_decisions[-1]["phase"] if all_decisions else "training"),
                progress={
                    "percent": self._percent(completed_days, total_days),
                    "message": f"Resuming from checkpoint {completed_days}/{total_days}",
                    "completed_days": completed_days,
                    "total_days": total_days,
                    "model_calls": model_calls,
                    "resume": True,
                },
            )

        for phase, start, end, limit in phases:
            pairs = self._trading_pairs(start, end, limit)
            for pair in pairs:
                pair_key = (phase, pair["decision_date"], pair["fill_date"])
                if pair_key in completed_keys:
                    continue
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

                model_calls += self._record_due_learning_lessons(memory, config, secrets, run_id, all_decisions, decision_date, dry_run=dry_run)
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

                manager_bundle = self._stage2_bundle(bundle, config)
                prompt_stage1_outputs = [self._strip_api_metadata(output) for output in stage1_outputs]
                critic_calls = self._maybe_run_exposure_critic(
                    config,
                    secrets,
                    manager_bundle,
                    prompt_stage1_outputs,
                    run_id=run_id,
                    phase=phase,
                    decision_date=decision_date,
                    fill_date=fill_date,
                    symbols=symbols,
                    store=store,
                    dry_run=dry_run,
                    cache_namespace="exposure-critic",
                )
                model_calls += critic_calls
                system, user = build_stage2_prompt(manager_bundle, prompt_stage1_outputs)
                stage2_output = call_json_model(config, secrets, system, user, dry_run=dry_run, fallback=self._stage2_fallback(symbols, manager_bundle), cache_namespace="stage2")
                model_calls += 0 if stage2_output.get("_api_status") in {"dry_run", "missing_key"} else 1
                fill_prices = self._price_map(symbols, fill_date, field="open")
                if not fill_prices:
                    fill_prices = self._price_map(symbols, fill_date, field="close")
                stage2_output = self._normalize_stage2_output(config, stage2_output, symbols, manager_bundle, book=book, prices=fill_prices)
                stage2_output, repair_calls = self._repair_stage2_allocation_if_needed(
                    config,
                    secrets,
                    manager_bundle,
                    stage2_output,
                    symbols,
                    book=book,
                    prices=fill_prices,
                    dry_run=dry_run,
                    cache_namespace="stage2-repair",
                )
                model_calls += repair_calls
                target_weights = self._coerce_target_weights(stage2_output.get("target_weights") or {}, symbols)
                validation_errors = self._allocation_errors(config, stage2_output, symbols, book=book, prices=fill_prices)
                if validation_errors and stage2_output.get("_api_status") not in {"dry_run", "missing_key"}:
                    stage2_output["_allocation_validation_errors"] = validation_errors
                    book, execution = reject_target_weights(book, target_weights, fill_prices, validation_errors)
                else:
                    book, execution = execute_target_weights(book, target_weights, fill_prices, config)
                execution["repair_attempted"] = bool((stage2_output.get("_allocation_repair") or {}).get("attempted"))
                execution["repair_count"] = int((stage2_output.get("_allocation_repair") or {}).get("attempts") or repair_calls or 0)
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

                if execution.get("model_failure") and not dry_run and config.run_preset in {"single_stock_official", "budget_official", "full_official", "local_gemma_aapl_full"}:
                    invalid_stage2_days += 1
                    invalid_rate = invalid_stage2_days / max(1, completed_days)
                    should_abort = invalid_stage2_days >= config.invalid_run_abort_count or (completed_days >= 20 and invalid_rate > config.invalid_run_abort_rate)
                    if should_abort:
                        reason = {
                            "invalid_stage2_days": invalid_stage2_days,
                            "completed_days": completed_days,
                            "invalid_rate": invalid_rate,
                            "max_invalid_count": config.invalid_run_abort_count,
                            "max_invalid_rate": config.invalid_run_abort_rate,
                        }
                        store.append_benchmark_event(run_id, phase, "run_aborted_invalid_allocations", reason)
                        summary = self._summary(config, symbols, equity_curve, all_executions, model_calls, dry_run)
                        summary = self._attach_api_usage(summary, store, run_id, config)
                        summary["official_status"] = "diagnostic"
                        summary["aborted_reason"] = "invalid_allocations"
                        summary["abort_details"] = reason
                        if memory_summary:
                            summary["deterministic_memory"] = memory_summary.__dict__
                        store.update_benchmark_run(
                            run_id,
                            status="failed",
                            phase="failed",
                            summary=summary,
                            error="Aborted after repeated invalid Stage 2 allocations.",
                            progress={"percent": self._percent(completed_days, total_days), "message": "Aborted after repeated invalid Stage 2 allocations.", "model_calls": model_calls},
                            finished=True,
                        )
                        return {"summary": summary, "decisions": all_decisions}

        summary = self._summary(config, symbols, equity_curve, all_executions, model_calls, dry_run)
        summary = self._attach_api_usage(summary, store, run_id, config)
        if memory_summary:
            summary["deterministic_memory"] = memory_summary.__dict__
        store.update_benchmark_run(run_id, status="completed", phase="completed", summary=summary, progress={"percent": 100, "message": "Completed", "model_calls": model_calls}, finished=True)
        return {"summary": summary, "decisions": all_decisions}

    def _load_resume_state(self, store: BenchmarkStore, run_id: str, config: BenchmarkConfig) -> Dict[str, Any]:
        run = store.get_benchmark_run(run_id)
        if not run:
            raise ValueError(f"Cannot resume missing benchmark run {run_id!r}.")
        decisions_by_day: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for item in run.get("decisions") or []:
            key = (str(item.get("phase")), str(item.get("decision_date")), str(item.get("fill_date")))
            bucket = decisions_by_day.setdefault(key, {"stage1": [], "stage2": None})
            if item.get("stage") == "stage1":
                bucket["stage1"].append(item.get("output") or {})
            elif item.get("stage") == "stage2":
                bucket["stage2"] = item

        recorded_memory = self._recorded_resume_memory(store, run_id)
        completed_keys = set()
        decisions: List[Dict[str, Any]] = []
        executions: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = []
        book = initial_book(config.initial_cash)
        invalid_stage2_days = 0
        for key in sorted(decisions_by_day, key=lambda value: (value[1], value[0], value[2])):
            bucket = decisions_by_day[key]
            stage2 = bucket.get("stage2")
            if not stage2:
                continue
            execution = stage2.get("execution") or {}
            stage2_output = stage2.get("output") or {}
            record = {
                "phase": key[0],
                "decision_date": key[1],
                "fill_date": key[2],
                "stage1_outputs": bucket.get("stage1") or [],
                "stage2_output": stage2_output,
                "execution": execution,
            }
            if ("diagnostic_lesson", key[1]) in recorded_memory:
                record["diagnostic_lesson_recorded"] = True
            if ("llm_reflection_lesson", key[1]) in recorded_memory:
                record["llm_reflection_lesson_recorded"] = True
            decisions.append(record)
            executions.append(execution)
            completed_keys.add(key)
            if execution.get("model_failure"):
                invalid_stage2_days += 1
            after = execution.get("portfolio_after") or {}
            if after:
                book = PortfolioBook(**after)
                equity_curve.append(
                    {
                        "date": key[2],
                        "phase": key[0],
                        "equity": book.equity,
                        "cash": book.cash,
                        "gross_exposure": book.gross_exposure,
                        "net_exposure": book.net_exposure,
                    }
                )
        model_calls = self._resume_model_call_count(store, run_id, run)
        return {
            "book": book,
            "equity_curve": equity_curve,
            "executions": executions,
            "decisions": decisions,
            "model_calls": model_calls,
            "invalid_stage2_days": invalid_stage2_days,
            "completed_keys": completed_keys,
        }

    def _recorded_resume_memory(self, store: BenchmarkStore, run_id: str) -> set[tuple[str, str]]:
        if not hasattr(store, "_connect"):
            return set()
        with store._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_type, decision_timestamp
                FROM benchmark_memory
                WHERE source_run_id = ?
                  AND memory_type IN ('diagnostic_lesson', 'llm_reflection_lesson')
                """,
                (run_id,),
            ).fetchall()
        return {(str(row["memory_type"]), str(row["decision_timestamp"])) for row in rows}

    def _resume_model_call_count(self, store: BenchmarkStore, run_id: str, run: Dict[str, Any]) -> int:
        calls = 0
        for item in run.get("decisions") or []:
            output = item.get("output") or {}
            if output.get("_api_status") not in {"dry_run", "missing_key"}:
                calls += 1
        if not hasattr(store, "_connect"):
            return calls
        with store._connect() as conn:
            rows = conn.execute(
                """
                SELECT metadata_json
                FROM benchmark_memory
                WHERE source_run_id = ?
                  AND memory_type = 'llm_reflection_lesson'
                """,
                (run_id,),
            ).fetchall()
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            if metadata.get("api_status") not in {"dry_run", "missing_key"}:
                calls += 1
        return calls

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
        validate_no_paid_api_mode(config, secrets)
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
        manager_bundle = self._stage2_bundle(bundle, config)
        prompt_stage1_outputs = [self._strip_api_metadata(output) for output in stage1_outputs]
        critic_calls = self._maybe_run_exposure_critic(
            config,
            secrets,
            manager_bundle,
            prompt_stage1_outputs,
            run_id=run_id,
            phase="live",
            decision_date=decision_timestamp,
            fill_date=decision_timestamp,
            symbols=symbols,
            store=store,
            dry_run=dry_run,
            cache_namespace="live-exposure-critic",
        )
        model_calls += critic_calls
        system, user = build_stage2_prompt(manager_bundle, prompt_stage1_outputs)
        stage2_output = call_json_model(config, secrets, system, user, dry_run=dry_run, fallback=self._stage2_fallback(symbols, manager_bundle), cache_namespace="live-stage2")
        model_calls += 0 if stage2_output.get("_api_status") in {"dry_run", "missing_key"} else 1
        stage2_output = self._normalize_stage2_output(config, stage2_output, symbols, manager_bundle, book=book, prices=fill_prices)
        stage2_output, repair_calls = self._repair_stage2_allocation_if_needed(
            config,
            secrets,
            manager_bundle,
            stage2_output,
            symbols,
            book=book,
            prices=fill_prices,
            dry_run=dry_run,
            cache_namespace="live-stage2-repair",
        )
        model_calls += repair_calls
        target_weights = self._coerce_target_weights(stage2_output.get("target_weights") or {}, symbols)
        validation_errors = self._allocation_errors(config, stage2_output, symbols, book=book, prices=fill_prices)
        if validation_errors and stage2_output.get("_api_status") not in {"dry_run", "missing_key"}:
            stage2_output["_allocation_validation_errors"] = validation_errors
            next_book, execution = reject_target_weights(book, target_weights, fill_prices, validation_errors)
        else:
            next_book, execution = execute_target_weights(book, target_weights, fill_prices, config)
        execution["repair_attempted"] = bool((stage2_output.get("_allocation_repair") or {}).get("attempted"))
        execution["repair_count"] = int((stage2_output.get("_allocation_repair") or {}).get("attempts") or repair_calls or 0)
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
        summary = self._attach_api_usage(summary, store, run_id, config)
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
        df = self.warehouse.conn.execute(
            """
            SELECT date, symbol, close, return_1d, source
            FROM context_daily
            WHERE date <= CAST(? AS DATE) AND ohlcv_available = true
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) <= 65
            ORDER BY symbol, date
            """,
            [decision_date],
        ).fetchdf()
        if df.empty:
            return []
        context = []
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values("date")
            latest = group.iloc[-1]
            closes = group["close"].astype(float)

            def trailing(days: int) -> float | None:
                if len(closes) <= days:
                    return None
                base = float(closes.iloc[-days - 1])
                return float(closes.iloc[-1]) / base - 1.0 if base else None

            context.append(
                {
                    "date": latest.get("date"),
                    "symbol": symbol,
                    "close": _safe_float(latest.get("close")),
                    "return_1d": _safe_float(latest.get("return_1d")),
                    "return_5d": _pct(trailing(5)),
                    "return_20d": _pct(trailing(20)),
                    "return_60d": _pct(trailing(60)),
                    "volatility_20d": _safe_float(closes.pct_change().tail(20).std() * (252 ** 0.5)) if len(closes) > 20 else None,
                    "source": latest.get("source", ""),
                }
            )
        return context

    def _news(self, symbols: List[str], decision_date: str, limit_per_symbol: int, config: BenchmarkConfig) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        if not symbols or limit_per_symbol <= 0:
            return {}, {"status": "disabled", "total_rows": 0}
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
        quality = {
            "status": "ok",
            "total_rows": int(len(df)),
            "real_headline_rows": 0,
            "synthetic_or_nan_rows": 0,
            "policy": config.news_policy,
            "by_symbol": {},
        }
        if df.empty:
            quality["status"] = "missing"
            return out, quality
        for symbol, group in df.groupby("symbol"):
            items = []
            event_codes = Counter()
            domains = Counter()
            tones = []
            mentions = 0
            synthetic_rows = 0
            real_rows = 0
            for _, row in group.iterrows():
                raw = {}
                try:
                    raw = json.loads(row.get("raw_json") or "{}")
                except Exception:
                    raw = {}
                title = row.get("title") or ""
                tone = _safe_float(raw.get("AvgTone"))
                if tone is not None:
                    tones.append(tone)
                event_code = str(raw.get("EventCode") or "")
                if event_code:
                    event_codes[event_code] += 1
                domain = row.get("domain") or ""
                if domain:
                    domains[domain] += 1
                try:
                    mentions += int(float(raw.get("NumMentions") or 0))
                except Exception:
                    pass
                if is_synthetic_news_title(title):
                    synthetic_rows += 1
                    continue
                real_rows += 1
                items.append(
                    {
                        "type": "article",
                        "published_at": str(row.get("published_at") or ""),
                        "title": title,
                        "url": row.get("url") or "",
                        "domain": domain,
                        "source_country": row.get("source_country") or "",
                        "source": row.get("source") or "",
                        "tone": raw.get("AvgTone"),
                        "mentions": raw.get("NumMentions"),
                        "event_code": raw.get("EventCode"),
                    }
                )
            if synthetic_rows:
                items.append(
                    {
                        "type": "event_summary",
                        "window_days": 5,
                        "event_rows": synthetic_rows,
                        "avg_tone": sum(tones) / len(tones) if tones else None,
                        "total_mentions": mentions,
                        "top_event_codes": [{"code": code, "count": count} for code, count in event_codes.most_common(5)],
                        "top_domains": [{"domain": domain, "count": count} for domain, count in domains.most_common(5)],
                        "note": "Synthetic GDELT event rows were aggregated and not passed as article headlines.",
                    }
                )
            quality["real_headline_rows"] += real_rows
            quality["synthetic_or_nan_rows"] += synthetic_rows
            quality["by_symbol"][symbol] = {"real_headline_rows": real_rows, "synthetic_or_nan_rows": synthetic_rows, "items_sent": len(items)}
            out[symbol] = items
        total = max(1, quality["total_rows"])
        quality["real_headline_rate"] = quality["real_headline_rows"] / total
        if quality["real_headline_rows"] == 0 and quality["synthetic_or_nan_rows"] > 0:
            quality["status"] = "event_aggregate_only"
        return out, quality

    def _fundamentals(self, symbols: List[str], decision_date: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
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
        quality: Dict[str, Any] = {}
        if df.empty:
            return out, quality
        for symbol, group in df.groupby("symbol"):
            canonical, symbol_quality = canonicalize_fundamentals(group.to_dict(orient="records"), decision_date)
            out[symbol] = canonical
            quality[symbol] = symbol_quality
        return out, quality

    def _macro(self, config: BenchmarkConfig, decision_date: str) -> List[Dict[str, Any]]:
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
        if df.empty:
            return []
        if config.macro_policy == "omit_if_missing":
            df = df[df["source_status"] == "ok"].copy()
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
        news, news_quality = self._news(symbols, decision_date, config.max_news_per_symbol, config)
        fundamentals, fundamental_quality = self._fundamentals(symbols, decision_date)
        macro = self._macro(config, decision_date)
        data_quality = self._data_quality(symbols, decision_date)
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
            memories.extend(self._diagnostic_lesson_memories(memory, config, decision_date, query, limit=6))
        else:
            lesson_limit = max(1, min(14, int(config.memory_examples_per_symbol or 2) * max(1, len(symbols))))
            memories = memory.retrieve(decision_timestamp=decision_date, query=query, limit=lesson_limit)
        payload = {
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
            "fundamentals": fundamentals,
            "fundamental_quality": fundamental_quality,
            "macro_context": macro,
            "news_and_events": news,
            "news_quality": news_quality,
            "data_quality": data_quality,
            "input_quality": {
                "macro_available": bool(macro),
                "macro_policy": config.macro_policy,
                "news_quality": news_quality,
                "fundamental_quality": fundamental_quality,
                "warnings": self._input_quality_warnings(macro, news_quality, fundamental_quality),
            },
            "memory": memories,
            "benchmark_rules": {
                "model_owns_decision": True,
                "simulator_role": "mechanics_only",
                "allow_short": config.allow_short,
                "max_gross_exposure": config.max_gross_exposure,
                "max_nonzero_positions": config.max_nonzero_positions,
                "max_daily_turnover": config.max_daily_turnover,
                "turnover_edge_multiplier": config.turnover_edge_multiplier,
                "turnover_prompt_buffer": config.turnover_prompt_buffer,
                "slippage_bps": config.slippage_bps,
                "fill_timing": config.fill_timing,
                "opportunity_cost_policy": config.opportunity_cost_policy,
                "exposure_critic_enabled": config.exposure_critic_enabled,
                "outcome_learning_mode": config.outcome_learning_mode,
            },
        }
        payload["decision_support"] = self._decision_support(payload)
        return payload

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
            "fundamental_quality": by_symbol(bundle.get("fundamental_quality")),
            "macro_context": bundle.get("macro_context") or [],
            "news_and_events": self._compact_news(by_symbol(bundle.get("news_and_events"))),
            "news_quality": bundle.get("news_quality") or {},
            "data_quality": data_quality,
            "input_quality": bundle.get("input_quality") or {},
            "decision_support": self._filter_decision_support(bundle.get("decision_support") or {}, symbol_set),
            "memory": self._filter_memory(bundle.get("memory") or [], symbol_set, limit=max(2, len(symbols) * 2)),
            "benchmark_rules": bundle.get("benchmark_rules") or {},
        }

    def _stage2_bundle(self, bundle: Dict[str, Any], config: BenchmarkConfig | None = None) -> Dict[str, Any]:
        market = bundle.get("market_snapshots") or {}
        news = bundle.get("news_and_events") or {}
        portfolio_state = bundle.get("portfolio_state") or {}
        current_position_weights = self._current_position_weights(portfolio_state, market)
        decision_support = bundle.get("decision_support") or {}
        support_by_symbol = {
            item.get("symbol"): item
            for item in decision_support.get("ranked_symbols", [])
            if item.get("symbol")
        }
        symbol_table = []
        for item in bundle.get("candidate_universe") or []:
            symbol = item.get("symbol")
            snap = market.get(symbol) or {}
            support = support_by_symbol.get(symbol) or {}
            memory_signal = support.get("memory_signal") or {}
            symbol_table.append(
                {
                    "symbol": symbol,
                    "sector": item.get("sector", ""),
                    "r5": snap.get("return_5d"),
                    "r20": snap.get("return_20d"),
                    "r60": snap.get("return_60d"),
                    "vol20": snap.get("volatility_20d"),
                    "current_weight": current_position_weights.get(symbol, 0.0),
                    "signal_rank": support.get("rank"),
                    "signal_score": support.get("score"),
                    "stance_hint": support.get("stance_hint"),
                    "memory_horizon": memory_signal.get("horizon"),
                    "memory_base_rate_return": memory_signal.get("base_rate_return"),
                    "memory_mean_return": memory_signal.get("mean_return"),
                    "memory_hit_rate": memory_signal.get("hit_rate"),
                    "memory_downside_rate": memory_signal.get("downside_rate"),
                    "memory_confidence": memory_signal.get("confidence"),
                    "memory_suggested_exposure_band": memory_signal.get("suggested_exposure_band"),
                    "memory_cases": memory_signal.get("cases"),
                    "news_items": len(news.get(symbol) or []),
                    "data_as_of": snap.get("as_of_date"),
                }
            )
        manager = {
            "schema_version": bundle.get("schema_version"),
            "mode": bundle.get("mode"),
            "run_id": bundle.get("run_id"),
            "phase": bundle.get("phase"),
            "decision_date": bundle.get("decision_date"),
            "fill_date": bundle.get("fill_date"),
            "information_cutoff": bundle.get("information_cutoff"),
            "candidate_universe": bundle.get("candidate_universe") or [],
            "portfolio_state": portfolio_state,
            "current_position_weights": current_position_weights,
            "no_trade_target_weights": current_position_weights,
            "symbol_summary_table": symbol_table,
            "index_context": self._compact_context(bundle.get("index_context") or []),
            "macro_context": bundle.get("macro_context") or [],
            "input_quality": bundle.get("input_quality") or {},
            "data_quality": bundle.get("data_quality") or {},
            "decision_support": self._compact_decision_support(decision_support),
            "memory": self._filter_memory(bundle.get("memory") or [], set(), limit=4),
            "benchmark_rules": bundle.get("benchmark_rules") or {},
            "omitted_raw_sections": ["market_snapshots", "fundamentals", "news_and_events"],
        }
        if bundle.get("mode") == "single_stock":
            symbol = str((bundle.get("candidate_universe") or [{}])[0].get("symbol") or (config.symbol if config else "AAPL")).upper()
            scoped_config = config or BenchmarkConfig(mode="single_stock", symbol=symbol)
            valid_range = self._valid_target_exposure_range(scoped_config, current_position_weights, [symbol])
            context_by_symbol = {item.get("symbol"): item for item in manager.get("index_context") or []}
            manager.update(
                {
                    "target_exposure_symbol": symbol,
                    "valid_target_exposure_range": valid_range,
                    "single_stock_contract": {
                        "model_returns": "target_exposure",
                        "simulator_computes": ["target_weights", "cash_weight", "gross_exposure", "net_exposure", "estimated_turnover", "estimated_slippage_cost_bps"],
                        "required_opportunity_cost_fields": [
                            "cash_drag_justification",
                            "why_not_buy_hold",
                            "stage1_alignment",
                            "stage1_veto_reason",
                        ],
                    },
                    "single_stock_opportunity_cost": {
                        "policy": scoped_config.opportunity_cost_policy,
                        "stock_symbol": symbol,
                        "current_exposure": valid_range.get("current_exposure"),
                        "benchmark_context": {
                            "SPY": context_by_symbol.get("SPY"),
                            "QQQ": context_by_symbol.get("QQQ"),
                        },
                        "instruction": "Low exposure is valid, but explain why cash beats buy-and-hold participation when stock or benchmark context is favorable.",
                    },
                }
            )
        return manager

    def _current_position_weights(self, portfolio_state: Dict[str, Any], market: Dict[str, Any]) -> Dict[str, float]:
        positions = portfolio_state.get("positions") or {}
        equity = _safe_float(portfolio_state.get("equity")) or 0.0
        if equity <= 0:
            return {}
        weights: Dict[str, float] = {}
        for symbol, shares in positions.items():
            snap = market.get(symbol) or {}
            price = _safe_float(snap.get("close") or snap.get("open") or snap.get("adj_close"))
            share_count = _safe_float(shares)
            if price is None or share_count is None:
                continue
            weight = share_count * price / equity
            if abs(weight) > 1e-9:
                weights[str(symbol).upper()] = round(float(weight), 8)
        return weights

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
                    "return_5d": item.get("return_5d"),
                    "return_20d": item.get("return_20d"),
                    "return_60d": item.get("return_60d"),
                    "volatility_20d": item.get("volatility_20d"),
                }
            )
        return compact

    def _compact_news(self, news: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        compact: Dict[str, List[Dict[str, Any]]] = {}
        for symbol, items in (news or {}).items():
            compact_items = []
            for item in items or []:
                if item.get("type") == "event_summary":
                    compact_items.append(
                        {
                            "type": "event_summary",
                            "window_days": item.get("window_days"),
                            "event_rows": item.get("event_rows"),
                            "avg_tone": item.get("avg_tone"),
                            "total_mentions": item.get("total_mentions"),
                            "top_event_codes": item.get("top_event_codes"),
                            "note": item.get("note"),
                        }
                    )
                    continue
                compact_items.append(
                    {
                        "type": "article",
                        "published_at": item.get("published_at"),
                        "title": item.get("title"),
                        "domain": item.get("domain"),
                        "tone": item.get("tone"),
                        "mentions": item.get("mentions"),
                        "event_code": item.get("event_code"),
                    }
                )
            compact[symbol] = compact_items
        return compact

    def _input_quality_warnings(self, macro: List[Dict[str, Any]], news_quality: Dict[str, Any], fundamental_quality: Dict[str, Any]) -> List[str]:
        warnings = []
        if not macro:
            warnings.append("macro_unavailable_or_omitted")
        if news_quality.get("status") in {"missing", "event_aggregate_only"}:
            warnings.append(f"news_{news_quality.get('status')}")
        weak_fundamentals = [symbol for symbol, quality in (fundamental_quality or {}).items() if quality.get("status") != "ok"]
        if weak_fundamentals:
            warnings.append(f"weak_fundamentals:{len(weak_fundamentals)}")
        return warnings

    def _diagnostic_lesson_memories(
        self,
        memory: HybridMemory,
        config: BenchmarkConfig,
        decision_timestamp: str,
        query: str,
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if config.outcome_learning_mode != "diagnostic_lessons":
            return []
        lessons = memory.retrieve(decision_timestamp=decision_timestamp, query=query, limit=limit)
        return [item for item in lessons if item.get("memory_type") == "diagnostic_lesson"][:limit]

    def _filter_memory(self, memories: List[Dict[str, Any]], symbols: set[str], *, limit: int) -> List[Dict[str, Any]]:
        filtered = []
        for item in memories:
            symbol = item.get("symbol") or ""
            if symbols and symbol and symbol not in symbols:
                continue
            content = str(item.get("content") or "")
            if len(content) > 260:
                content = content[:257].rstrip() + "..."
            filtered.append(
                {
                    "id": item.get("id"),
                    "memory_type": item.get("memory_type"),
                    "symbol": symbol,
                    "decision_timestamp": item.get("decision_timestamp"),
                    "knowledge_timestamp": item.get("knowledge_timestamp"),
                    "content": content,
                    "retrieval_score": item.get("retrieval_score"),
                }
            )
            if len(filtered) >= limit:
                break
        return filtered

    def _decision_support(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Build compact point-in-time features that help the model compare symbols."""
        market = bundle.get("market_snapshots") or {}
        memories = bundle.get("memory") or []
        news = bundle.get("news_and_events") or {}
        memory_by_symbol = self._memory_signal_by_symbol(memories)
        news_by_symbol = self._news_signal_by_symbol(news)
        rows = []
        for item in bundle.get("candidate_universe") or []:
            symbol = item.get("symbol")
            if not symbol:
                continue
            snap = market.get(symbol) or {}
            memory_signal = memory_by_symbol.get(symbol, {})
            news_signal = news_by_symbol.get(symbol, {})
            r5 = _safe_float(snap.get("return_5d"))
            r20 = _safe_float(snap.get("return_20d"))
            r60 = _safe_float(snap.get("return_60d"))
            vol20 = _safe_float(snap.get("volatility_20d"))
            trend_score = (
                0.25 * _bounded(r5, 0.06)
                + 0.45 * _bounded(r20, 0.12)
                + 0.30 * _bounded(r60, 0.25)
            )
            memory_score = _safe_float(memory_signal.get("score")) or 0.0
            news_score = _bounded(_safe_float(news_signal.get("avg_tone")), 5.0)
            risk_penalty = max(0.0, _bounded((vol20 or 0.0) - 0.35, 0.35)) if vol20 is not None else 0.0
            score = 0.42 * trend_score + 0.43 * memory_score + 0.10 * news_score - 0.10 * risk_penalty
            rows.append(
                {
                    "symbol": symbol,
                    "sector": item.get("sector", ""),
                    "score": round(float(score), 6),
                    "trend_score": round(float(trend_score), 6),
                    "memory_score": round(float(memory_score), 6),
                    "news_score": round(float(news_score), 6),
                    "risk_penalty": round(float(risk_penalty), 6),
                    "stance_hint": self._stance_hint(score),
                    "memory_signal": memory_signal,
                    "news_signal": news_signal,
                    "market_features": {
                        "return_5d": r5,
                        "return_20d": r20,
                        "return_60d": r60,
                        "volatility_20d": vol20,
                    },
                }
            )

        rows.sort(key=lambda item: (item.get("score") or 0.0), reverse=True)
        for index, item in enumerate(rows, start=1):
            item["rank"] = index
        weak = list(reversed(rows[-10:])) if rows else []
        return {
            "method": "point_in_time_signal_blend",
            "description": "Deterministic support features from current trend, retrieved historical memory, news tone, and volatility. These are evidence features, not simulator orders.",
            "ranked_symbols": rows,
            "top_long_candidates": [
                {"symbol": item["symbol"], "rank": item["rank"], "score": item["score"], "stance_hint": item["stance_hint"]}
                for item in rows[:10]
            ],
            "weak_or_hedge_candidates": [
                {"symbol": item["symbol"], "rank": item["rank"], "score": item["score"], "stance_hint": item["stance_hint"]}
                for item in weak[:10]
            ],
        }

    def _memory_signal_by_symbol(self, memories: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        preferred_horizons = ("20d", "60d", "5d", "1d")
        for item in memories:
            symbol = item.get("symbol")
            if not symbol or item.get("memory_type") != "deterministic_market_aggregate":
                continue
            metadata = item.get("metadata") or {}
            stats = metadata.get("aggregate_stats") or {}
            horizon = next((name for name in preferred_horizons if name in stats), "")
            if not horizon:
                continue
            horizon_stats = stats.get(horizon) or {}
            mean_return = _safe_float(horizon_stats.get("mean_return"))
            hit_rate = _safe_float(horizon_stats.get("hit_rate"))
            cases = int(horizon_stats.get("cases") or 0)
            confidence = metadata.get("confidence") or "weak"
            score = 0.65 * _bounded(mean_return, 0.08) + 0.35 * _bounded((hit_rate - 0.5) if hit_rate is not None else None, 0.25)
            if cases < 10:
                score *= 0.65
            if horizon == "1d":
                score *= 0.45
            suggested = metadata.get("suggested_exposure") or {}
            suggested_band = suggested.get("band")
            if not suggested_band:
                suggested_band = self._score_to_exposure_band(score)
            out[str(symbol)] = {
                "horizon": horizon,
                "base_rate_return": _safe_float(suggested.get("base_rate_return")) if suggested else mean_return,
                "mean_return": mean_return,
                "median_return": _safe_float(horizon_stats.get("median_return")),
                "hit_rate": hit_rate,
                "downside_rate": _safe_float(horizon_stats.get("downside_rate")),
                "cases": cases,
                "confidence": confidence,
                "retrieval_score": _safe_float(metadata.get("retrieval_score") or item.get("retrieval_score")),
                "score": round(float(score), 6),
                "suggested_exposure_score": _safe_float(suggested.get("score")) if suggested else round(float(score), 6),
                "suggested_exposure_band": suggested_band,
            }
        return out

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

    def _news_signal_by_symbol(self, news: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for symbol, items in (news or {}).items():
            tones = []
            event_rows = 0
            article_rows = 0
            mentions = 0
            for item in items or []:
                tone = _safe_float(item.get("tone") if item.get("type") != "event_summary" else item.get("avg_tone"))
                if tone is not None:
                    tones.append(tone)
                if item.get("type") == "event_summary":
                    event_rows += int(item.get("event_rows") or 0)
                    mentions += int(item.get("total_mentions") or 0)
                else:
                    article_rows += 1
                    try:
                        mentions += int(float(item.get("mentions") or 0))
                    except Exception:
                        pass
            out[str(symbol)] = {
                "avg_tone": sum(tones) / len(tones) if tones else None,
                "tone_observations": len(tones),
                "event_rows": event_rows,
                "article_rows": article_rows,
                "mentions": mentions,
            }
        return out

    def _stance_hint(self, score: float) -> str:
        if score >= 0.35:
            return "favorable"
        if score <= -0.35:
            return "unfavorable"
        if score >= 0.12:
            return "slightly_favorable"
        if score <= -0.12:
            return "slightly_unfavorable"
        return "neutral"

    def _filter_decision_support(self, support: Dict[str, Any], symbols: set[str]) -> Dict[str, Any]:
        ranked = [item for item in support.get("ranked_symbols", []) if item.get("symbol") in symbols]
        return {
            "method": support.get("method"),
            "description": support.get("description"),
            "ranked_symbols": ranked,
            "top_long_candidates": [item for item in support.get("top_long_candidates", []) if item.get("symbol") in symbols],
            "weak_or_hedge_candidates": [item for item in support.get("weak_or_hedge_candidates", []) if item.get("symbol") in symbols],
        }

    def _compact_decision_support(self, support: Dict[str, Any], *, limit: int = 20) -> Dict[str, Any]:
        ranked = []
        for item in support.get("ranked_symbols", [])[:limit]:
            memory_signal = item.get("memory_signal") or {}
            ranked.append(
                {
                    "symbol": item.get("symbol"),
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "trend_score": item.get("trend_score"),
                    "memory_score": item.get("memory_score"),
                    "news_score": item.get("news_score"),
                    "risk_penalty": item.get("risk_penalty"),
                    "stance_hint": item.get("stance_hint"),
                    "memory_horizon": memory_signal.get("horizon"),
                    "memory_base_rate_return": memory_signal.get("base_rate_return"),
                    "memory_mean_return": memory_signal.get("mean_return"),
                    "memory_hit_rate": memory_signal.get("hit_rate"),
                    "memory_downside_rate": memory_signal.get("downside_rate"),
                    "memory_confidence": memory_signal.get("confidence"),
                    "memory_suggested_exposure_band": memory_signal.get("suggested_exposure_band"),
                    "memory_cases": memory_signal.get("cases"),
                }
            )
        return {
            "method": support.get("method"),
            "description": support.get("description"),
            "ranked_symbols": ranked,
            "top_long_candidates": support.get("top_long_candidates", [])[:10],
            "weak_or_hedge_candidates": support.get("weak_or_hedge_candidates", [])[:10],
        }

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
            memories.extend(self._diagnostic_lesson_memories(memory, config, decision_timestamp, query, limit=6))
        else:
            memories = memory.retrieve(decision_timestamp=decision_timestamp, query=query, limit=14)
        live_news, live_news_quality = self._live_news(config, secrets, symbols, decision_date, dry_run=dry_run)
        fundamentals, fundamental_quality = self._fundamentals(symbols, decision_date)
        macro = self._macro(config, decision_date)
        data_quality = {
            "checked_symbols": len(symbols),
            "missing_live_prices": [symbol for symbol in symbols if symbol not in market],
            "source_status": source_status,
            "status": "ok" if len(market) == len(symbols) else "partial",
        }
        payload = {
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
            "fundamentals": fundamentals,
            "fundamental_quality": fundamental_quality,
            "macro_context": macro,
            "news_and_events": live_news,
            "news_quality": live_news_quality,
            "data_quality": data_quality,
            "input_quality": {
                "macro_available": bool(macro),
                "macro_policy": config.macro_policy,
                "news_quality": live_news_quality,
                "fundamental_quality": fundamental_quality,
                "warnings": self._input_quality_warnings(macro, live_news_quality, fundamental_quality),
            },
            "memory": memories,
            "benchmark_rules": {
                "model_owns_decision": True,
                "simulator_role": "mechanics_only",
                "allow_short": config.allow_short,
                "max_gross_exposure": config.max_gross_exposure,
                "max_nonzero_positions": config.max_nonzero_positions,
                "max_daily_turnover": config.max_daily_turnover,
                "turnover_edge_multiplier": config.turnover_edge_multiplier,
                "turnover_prompt_buffer": config.turnover_prompt_buffer,
                "slippage_bps": config.slippage_bps,
                "fill_timing": "live_latest_price",
                "opportunity_cost_policy": config.opportunity_cost_policy,
                "exposure_critic_enabled": config.exposure_critic_enabled,
                "outcome_learning_mode": config.outcome_learning_mode,
            },
        }
        payload["decision_support"] = self._decision_support(payload)
        return payload

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

    def _live_news(self, config: BenchmarkConfig, secrets: SecretConfig, symbols: List[str], decision_date: str, *, dry_run: bool) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        quality: Dict[str, Any] = {
            "status": "disabled" if dry_run or config.max_news_per_symbol <= 0 else "ok",
            "total_rows": 0,
            "real_headline_rows": 0,
            "synthetic_or_nan_rows": 0,
            "policy": config.news_policy,
            "by_symbol": {},
        }
        if dry_run or config.max_news_per_symbol <= 0:
            return {}, quality
        meta = {item["symbol"]: item for item in self._symbol_metadata(symbols)}
        news: Dict[str, List[Dict[str, Any]]] = {}
        for symbol in symbols:
            scoped = config.model_copy(deep=True) if hasattr(config, "model_copy") else BenchmarkConfig(**config.dict())
            scoped.symbol = symbol
            scoped.company_name = meta.get(symbol, {}).get("name") or symbol
            scoped.data_sources.max_news_per_day = config.max_news_per_symbol
            try:
                raw_items = fetch_news_bundle(scoped, secrets, decision_date).get("items", [])
            except Exception as exc:
                quality["by_symbol"][symbol] = {"status": "error", "message": str(exc), "real_headline_rows": 0, "synthetic_or_nan_rows": 0, "items_sent": 0}
                news[symbol] = []
                continue

            items = []
            synthetic_rows = 0
            real_rows = 0
            for item in raw_items or []:
                quality["total_rows"] += 1
                title = item.get("title") or item.get("headline") or ""
                if is_synthetic_news_title(title):
                    synthetic_rows += 1
                    continue
                real_rows += 1
                items.append(
                    {
                        "type": "article",
                        "published_at": item.get("published_at") or item.get("datetime") or decision_date,
                        "title": title,
                        "url": item.get("url") or "",
                        "domain": item.get("domain") or item.get("source") or "",
                        "source_country": item.get("source_country") or "",
                        "source": item.get("source") or "live_news",
                        "summary": item.get("summary") or item.get("description") or "",
                    }
                )
            quality["real_headline_rows"] += real_rows
            quality["synthetic_or_nan_rows"] += synthetic_rows
            quality["by_symbol"][symbol] = {"real_headline_rows": real_rows, "synthetic_or_nan_rows": synthetic_rows, "items_sent": len(items)}
            news[symbol] = items
        if quality["total_rows"] == 0:
            quality["status"] = "missing"
        elif quality["real_headline_rows"] == 0 and quality["synthetic_or_nan_rows"] > 0:
            quality["status"] = "event_aggregate_only"
        quality["real_headline_rate"] = quality["real_headline_rows"] / max(1, quality["total_rows"])
        return news, quality

    def _symbol_metadata(self, symbols: List[str]) -> List[Dict[str, Any]]:
        meta = {item.symbol: item for item in STOCKS}
        return [{"symbol": symbol, "name": meta.get(symbol).name if symbol in meta else symbol, "sector": meta.get(symbol).sector if symbol in meta else ""} for symbol in symbols]

    def _valid_target_exposure_range(
        self,
        config: BenchmarkConfig,
        current_position_weights: Dict[str, float],
        symbols: List[str],
    ) -> Dict[str, Any]:
        symbol = symbols[0] if symbols else config.symbol.upper()
        current = float(current_position_weights.get(symbol, 0.0) or 0.0)
        turnover_limit = max(0.0, float(config.max_daily_turnover or 0.0))
        buffer = max(0.0, float(config.turnover_prompt_buffer or 0.0))
        allowed_change = max(0.0, turnover_limit - buffer) if turnover_limit else float(config.max_gross_exposure)
        lower_bound = -float(config.max_gross_exposure) if config.allow_short else 0.0
        upper_bound = float(config.max_gross_exposure)
        minimum = max(lower_bound, current - allowed_change)
        maximum = min(upper_bound, current + allowed_change)
        if minimum > maximum:
            minimum = maximum = max(lower_bound, min(upper_bound, current))
        return {
            "symbol": symbol,
            "current_exposure": round(current, 8),
            "min": round(minimum, 8),
            "max": round(maximum, 8),
            "max_daily_turnover": turnover_limit,
            "turnover_prompt_buffer": buffer,
            "rule": "Choose target_exposure inside [min, max] to avoid prompt-time turnover overflow.",
        }

    def _single_stock_exposure(self, output: Dict[str, Any], symbols: List[str], manager_bundle: Dict[str, Any] | None = None) -> float:
        symbol = symbols[0] if symbols else ""
        raw = _safe_float(output.get("target_exposure"))
        if raw is not None:
            return float(raw)
        weights = output.get("target_weights") or {}
        if symbol and symbol in weights:
            weight = _safe_float(weights.get(symbol))
            if weight is not None:
                return float(weight)
        current = ((manager_bundle or {}).get("current_position_weights") or {}).get(symbol)
        return float(_safe_float(current) or 0.0)

    def _normalize_stage2_output(
        self,
        config: BenchmarkConfig,
        output: Dict[str, Any],
        symbols: List[str],
        manager_bundle: Dict[str, Any] | None = None,
        *,
        book: PortfolioBook | None = None,
        prices: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        if config.mode != "single_stock":
            return output
        normalized = dict(output or {})
        symbol = symbols[0] if symbols else config.symbol.upper()
        exposure = self._single_stock_exposure(normalized, [symbol], manager_bundle)
        target_weights = {symbol: exposure} if abs(exposure) > 1e-9 else {}
        gross = abs(exposure)
        normalized["target_exposure"] = round(float(exposure), 8)
        normalized["target_weights"] = {key: round(float(value), 8) for key, value in target_weights.items()}
        normalized["gross_exposure"] = round(gross, 8)
        normalized["net_exposure"] = round(float(exposure), 8)
        normalized["cash_weight"] = round(1.0 - gross, 8)
        normalized["cash_target_weight"] = normalized["cash_weight"]
        if "expected_holding_days" not in normalized and normalized.get("horizon_days") is not None:
            normalized["expected_holding_days"] = normalized.get("horizon_days")
        if book is not None and prices:
            turnover = estimate_target_turnover(book, target_weights, prices)
        else:
            current = (((manager_bundle or {}).get("current_position_weights") or {}).get(symbol)) or 0.0
            turnover = abs(float(exposure) - float(_safe_float(current) or 0.0))
        normalized["estimated_turnover"] = round(float(turnover), 8)
        normalized["estimated_slippage_cost_bps"] = round(float(turnover) * float(config.slippage_bps or 0.0), 8)
        return normalized

    def _exposure_critic_fallback(self, manager_bundle: Dict[str, Any], stage1_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_range = manager_bundle.get("valid_target_exposure_range") or {}
        symbol = valid_range.get("symbol") or "symbol"
        analysis = next(
            (
                item
                for output in stage1_outputs
                for item in (output.get("analyses") or [])
                if item.get("symbol") == symbol
            ),
            {},
        )
        stance = analysis.get("stance") or "unknown"
        return {
            "bull_exposure_case": f"Stage 1 stance is {stance}; participate only if evidence remains favorable.",
            "defensive_case": "Respect weak data, drawdown risk, and the prompt-time turnover range.",
            "cash_drag_risk": "Cash can underperform if the stock and benchmarks continue higher.",
            "recommended_exposure_band": [valid_range.get("min", 0.0), valid_range.get("max", 0.0)],
            "key_disagreement": "Fallback critic: no model critique was produced.",
        }

    def _maybe_run_exposure_critic(
        self,
        config: BenchmarkConfig,
        secrets: SecretConfig,
        manager_bundle: Dict[str, Any],
        stage1_outputs: List[Dict[str, Any]],
        *,
        run_id: str,
        phase: str,
        decision_date: str,
        fill_date: str,
        symbols: List[str],
        store: BenchmarkStore,
        dry_run: bool,
        cache_namespace: str,
    ) -> int:
        if config.mode != "single_stock" or not config.exposure_critic_enabled:
            return 0
        system, user = build_exposure_critic_prompt(manager_bundle, stage1_outputs)
        fallback = self._exposure_critic_fallback(manager_bundle, stage1_outputs)
        input_payload = dict(manager_bundle)
        output = call_json_model(config, secrets, system, user, dry_run=dry_run, fallback=fallback, cache_namespace=cache_namespace)
        manager_bundle["exposure_critic"] = self._strip_api_metadata(output)
        store.save_benchmark_decision(
            run_id=run_id,
            phase=phase,
            decision_date=decision_date,
            fill_date=fill_date,
            stage="exposure_critic",
            symbol=symbols[0] if symbols else config.symbol.upper(),
            input_payload=input_payload,
            output_payload=output,
            execution_payload={},
        )
        return 0 if output.get("_api_status") in {"dry_run", "missing_key"} else 1

    def _record_due_diagnostic_lessons(
        self,
        memory: HybridMemory,
        config: BenchmarkConfig,
        run_id: str,
        decisions: List[Dict[str, Any]],
        decision_date: str,
    ) -> int:
        if config.outcome_learning_mode != "diagnostic_lessons" or config.mode != "single_stock":
            return 0
        symbol = self._symbols(config)[0]
        for record in decisions:
            if record.get("diagnostic_lesson_recorded"):
                continue
            stage2 = record.get("stage2_output") or {}
            horizon = int(_safe_float(stage2.get("expected_holding_days", stage2.get("horizon_days"))) or 1)
            horizon = max(1, horizon)
            outcome_date = self._nth_trading_date_after(str(record.get("fill_date")), horizon)
            if not outcome_date or outcome_date > decision_date:
                continue
            start_prices = self._price_map([symbol], str(record.get("fill_date")), field="open") or self._price_map([symbol], str(record.get("fill_date")), field="close")
            end_prices = self._price_map([symbol], outcome_date, field="close")
            start = _safe_float(start_prices.get(symbol) if start_prices else None)
            end = _safe_float(end_prices.get(symbol) if end_prices else None)
            if start is None or end is None or start <= 0:
                continue
            realized_return = end / start - 1.0
            exposure = self._single_stock_exposure(stage2, [symbol])
            if realized_return > 0 and exposure < 0.25:
                lesson = "cash_drag"
            elif realized_return > 0 and exposure > 0:
                lesson = "participation_helped"
            elif realized_return < 0 and exposure > 0:
                lesson = "exposure_hurt"
            elif realized_return < 0 and exposure <= 0:
                lesson = "defense_helped"
            else:
                lesson = "flat_outcome"
            content = (
                f"{symbol} diagnostic lesson from {record.get('decision_date')} known on {outcome_date}: "
                f"target_exposure={exposure:.2f}, horizon={horizon}d, realized_return={realized_return:+.2%}, lesson={lesson}."
            )
            memory.add(
                portfolio_scope=config.mode,
                symbol=symbol,
                decision_timestamp=str(record.get("decision_date")),
                knowledge_timestamp=outcome_date,
                source_run_id=run_id,
                memory_type="diagnostic_lesson",
                content=content,
                outcome_horizon=f"{horizon}d",
                outcome_available_at=outcome_date,
                metadata={
                    "target_exposure": exposure,
                    "realized_return": realized_return,
                    "lesson": lesson,
                    "fill_date": record.get("fill_date"),
                    "outcome_date": outcome_date,
                },
            )
            record["diagnostic_lesson_recorded"] = True
        return 0

    def _record_due_learning_lessons(
        self,
        memory: HybridMemory,
        config: BenchmarkConfig,
        secrets: SecretConfig,
        run_id: str,
        decisions: List[Dict[str, Any]],
        decision_date: str,
        *,
        dry_run: bool,
    ) -> int:
        calls = self._record_due_diagnostic_lessons(memory, config, run_id, decisions, decision_date)
        calls += self._record_due_llm_reflection_lessons(memory, config, secrets, run_id, decisions, decision_date, dry_run=dry_run)
        return calls

    def _record_due_llm_reflection_lessons(
        self,
        memory: HybridMemory,
        config: BenchmarkConfig,
        secrets: SecretConfig,
        run_id: str,
        decisions: List[Dict[str, Any]],
        decision_date: str,
        *,
        dry_run: bool,
    ) -> int:
        if config.outcome_learning_mode != "llm_reflection_lessons" or config.mode != "single_stock":
            return 0
        symbol = self._symbols(config)[0]
        calls = 0
        for record in decisions:
            if record.get("llm_reflection_lesson_recorded") or record.get("phase") != "training":
                continue
            stage2 = record.get("stage2_output") or {}
            horizon = int(_safe_float(stage2.get("expected_holding_days", stage2.get("horizon_days"))) or 1)
            horizon = max(1, min(60, horizon))
            outcome_date = self._nth_trading_date_after(str(record.get("fill_date")), horizon)
            if not outcome_date or outcome_date > decision_date:
                continue
            start_prices = self._price_map([symbol], str(record.get("fill_date")), field="open") or self._price_map([symbol], str(record.get("fill_date")), field="close")
            end_prices = self._price_map([symbol], outcome_date, field="close")
            start = _safe_float(start_prices.get(symbol) if start_prices else None)
            end = _safe_float(end_prices.get(symbol) if end_prices else None)
            if start is None or end is None or start <= 0:
                continue
            realized_return = end / start - 1.0
            exposure = self._single_stock_exposure(stage2, [symbol])
            fallback = self._reflection_lesson_fallback(symbol, record, horizon, outcome_date, realized_return, exposure)
            system, user = build_reflection_lesson_prompt(
                {
                    "symbol": symbol,
                    "decision_date": record.get("decision_date"),
                    "fill_date": record.get("fill_date"),
                    "knowledge_timestamp": outcome_date,
                    "outcome_available_at": outcome_date,
                    "horizon_days": horizon,
                    "target_exposure": exposure,
                    "realized_return": realized_return,
                    "stage1_outputs": record.get("stage1_outputs") or [],
                    "stage2_output": self._strip_api_metadata(stage2),
                    "execution": record.get("execution") or {},
                }
            )
            output = call_json_model(
                config,
                secrets,
                system,
                user,
                dry_run=dry_run,
                fallback=fallback,
                cache_namespace="reflection-lesson",
            )
            if output.get("_api_status") not in {"dry_run", "missing_key"}:
                calls += 1
            lesson = str(output.get("summary_lesson") or output.get("lesson") or fallback["summary_lesson"]).strip()
            content = (
                f"{symbol} LLM reflection from {record.get('decision_date')} known on {outcome_date}: "
                f"{lesson[:900]}"
            )
            memory.add(
                portfolio_scope=config.mode,
                symbol=symbol,
                decision_timestamp=str(record.get("decision_date")),
                knowledge_timestamp=outcome_date,
                source_run_id=run_id,
                memory_type="llm_reflection_lesson",
                content=content,
                outcome_horizon=f"{horizon}d",
                outcome_available_at=outcome_date,
                metadata={
                    "target_exposure": exposure,
                    "realized_return": realized_return,
                    "outcome_date": outcome_date,
                    "lesson_tags": output.get("lesson_tags") or [],
                    "use_in_future_if": output.get("use_in_future_if", ""),
                    "avoid_if": output.get("avoid_if", ""),
                    "confidence": _safe_float(output.get("confidence")),
                    "api_status": output.get("_api_status"),
                },
            )
            record["llm_reflection_lesson_recorded"] = True
        return calls

    def _reflection_lesson_fallback(
        self,
        symbol: str,
        record: Dict[str, Any],
        horizon: int,
        outcome_date: str,
        realized_return: float,
        exposure: float,
    ) -> Dict[str, Any]:
        if realized_return > 0 and exposure < 0.25:
            tag = "cash_drag"
            lesson = "Low exposure missed a positive realized move; require stronger evidence before staying mostly in cash."
        elif realized_return > 0 and exposure > 0:
            tag = "participation_helped"
            lesson = "Positive exposure participated in a positive realized move; similar trend evidence can justify long exposure."
        elif realized_return < 0 and exposure > 0:
            tag = "exposure_hurt"
            lesson = "Long exposure lost money over the realized horizon; similar weak evidence should reduce sizing."
        elif realized_return < 0 and exposure <= 0:
            tag = "defense_helped"
            lesson = "Defensive exposure avoided a negative realized move; weak setups can justify cash."
        else:
            tag = "flat_outcome"
            lesson = "The realized move was flat; avoid over-reading this setup."
        return {
            "summary_lesson": lesson,
            "lesson_tags": [tag],
            "use_in_future_if": f"Similar {symbol} setup appears after {outcome_date}.",
            "avoid_if": "Input evidence differs materially.",
            "confidence": 0.5,
            "source_decision_date": record.get("decision_date"),
            "outcome_horizon": f"{horizon}d",
            "realized_return": realized_return,
        }

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

    def _stage2_fallback(self, symbols: List[str], manager_bundle: Dict[str, Any] | None = None) -> Dict[str, Any]:
        allowed = set(symbols)
        current_weights: Dict[str, float] = {}
        for symbol, raw_weight in ((manager_bundle or {}).get("current_position_weights") or {}).items():
            symbol = str(symbol).upper()
            if symbol not in allowed:
                continue
            weight = _safe_float(raw_weight)
            if weight is not None and abs(weight) > 1e-9:
                current_weights[symbol] = float(weight)
        gross = sum(abs(value) for value in current_weights.values())
        net = sum(current_weights.values())
        if (manager_bundle or {}).get("mode") == "single_stock":
            symbol = symbols[0] if symbols else str((manager_bundle or {}).get("target_exposure_symbol") or "AAPL").upper()
            exposure = float(current_weights.get(symbol, 0.0))
            gross = abs(exposure)
            target_weights = {symbol: exposure} if abs(exposure) > 1e-9 else {}
            return {
                "target_exposure": round(exposure, 8),
                "target_weights": {key: round(value, 8) for key, value in target_weights.items()},
                "cash_weight": round(1.0 - gross, 8),
                "cash_target_weight": round(1.0 - gross, 8),
                "gross_exposure": round(gross, 8),
                "net_exposure": round(exposure, 8),
                "confidence": 0.0,
                "portfolio_thesis": "Fallback/no model call; keep current single-stock exposure.",
                "major_risks": [],
                "uncertainty": ["No model allocation was produced."],
                "expected_return_bps": 0,
                "horizon_days": 1,
                "expected_holding_days": 1,
                "estimated_turnover": 0.0,
                "estimated_slippage_cost_bps": 0.0,
                "rebalance_reason": "fallback_no_model_call_keep_current",
                "input_evidence_refs": [],
                "data_quality_warnings_used": [],
                "cash_drag_justification": "Fallback/no model call; no new cash decision.",
                "why_not_buy_hold": "Fallback/no model call; kept existing exposure instead of changing toward buy-and-hold.",
                "stage1_alignment": "partial",
                "stage1_veto_reason": "Fallback/no model call.",
            }
        return {
            "target_weights": current_weights,
            "cash_weight": round(1.0 - gross, 8),
            "cash_target_weight": round(1.0 - gross, 8),
            "gross_exposure": round(gross, 8),
            "net_exposure": round(net, 8),
            "confidence": 0.0,
            "portfolio_thesis": "Fallback/no model call; keep current portfolio weights.",
            "major_risks": [],
            "uncertainty": ["No model allocation was produced."],
            "expected_return_bps": 0,
            "horizon_days": 1,
            "expected_holding_days": 1,
            "estimated_turnover": 0.0,
            "estimated_slippage_cost_bps": 0.0,
            "rebalance_reason": "fallback_no_model_call_keep_current",
            "input_evidence_refs": [],
            "data_quality_warnings_used": [],
        }

    def _repair_stage2_allocation_if_needed(
        self,
        config: BenchmarkConfig,
        secrets: SecretConfig,
        manager_bundle: Dict[str, Any],
        stage2_output: Dict[str, Any],
        symbols: List[str],
        *,
        book: PortfolioBook | None = None,
        prices: Dict[str, float] | None = None,
        dry_run: bool,
        cache_namespace: str,
    ) -> tuple[Dict[str, Any], int]:
        current = stage2_output
        errors = self._allocation_errors(config, current, symbols, book=book, prices=prices)
        if not errors or current.get("_api_status") in {"dry_run", "missing_key"}:
            return current, 0

        if config.mode == "single_stock":
            repair_system = """You are the portfolio manager stage of a single-stock AI market benchmark.

Your previous Stage 2 target_exposure violated hard benchmark constraints.
Correct your own target_exposure and required explanation fields. The simulator
will convert target_exposure into weights and compute portfolio arithmetic.
Return only valid compact JSON with the same single-stock Stage 2 schema.
"""
        else:
            repair_system = """You are the portfolio manager stage of an AI market benchmark.

Your previous Stage 2 allocation violated hard benchmark constraints. You still
own the decision. Correct your own target weights; the simulator will not scale
or improve them. Return only valid compact JSON with the same Stage 2 schema.
"""
        original_errors = errors
        original_usage = current.get("_api_usage")
        original_cache_hit = bool(current.get("_api_cache_hit"))
        attempt_usages = []
        calls = 0
        for attempt in range(1, 3):
            repair_user = json.dumps(
                {
                    "task": "Repair your previous single-stock target_exposure and return valid JSON only." if config.mode == "single_stock" else "Repair your previous portfolio allocation and return valid JSON only.",
                    "attempt": attempt,
                    "max_attempts": 2,
                    "hard_rules": {
                        "allow_short": config.allow_short,
                        "max_gross_exposure": config.max_gross_exposure,
                        "max_nonzero_positions": config.max_nonzero_positions,
                        "max_daily_turnover": config.max_daily_turnover,
                        "turnover_prompt_buffer": config.turnover_prompt_buffer,
                        "turnover_edge_multiplier": config.turnover_edge_multiplier,
                        "slippage_bps": config.slippage_bps,
                        "valid_target_exposure_range": manager_bundle.get("valid_target_exposure_range"),
                        "single_stock_contract": manager_bundle.get("single_stock_contract"),
                        "gross_exposure_formula": "sum(abs(target_weights.values()))",
                        "net_exposure_formula": "sum(target_weights.values())",
                        "cash_weight_formula": "1 - gross_exposure",
                        "zero_weight_policy": "omitted symbols, including currently held positions, are target weight 0",
                        "current_position_weights": manager_bundle.get("current_position_weights") or {},
                        "safe_fallback": "to make no trade, copy current_position_weights into target_weights; all cash is a sell-to-zero order and may violate turnover/cost rules",
                    },
                    "validation_errors": errors,
                    "previous_stage2_output": self._strip_api_metadata(current),
                    "portfolio_state": manager_bundle.get("portfolio_state"),
                    "current_position_weights": manager_bundle.get("current_position_weights") or {},
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
                    fallback=current,
                    cache_namespace=f"{cache_namespace}-{attempt}",
                )
            except Exception as exc:
                current["_allocation_repair_error"] = str(exc)
                current["_allocation_repair_errors"] = errors
                return current, calls

            repaired = self._normalize_stage2_output(config, repaired, symbols, manager_bundle, book=book, prices=prices)
            if repaired.get("_api_status") not in {"dry_run", "missing_key"}:
                calls += 1
                attempt_usages.append(
                    {
                        "attempt": attempt,
                        "usage": repaired.get("_api_usage") or {},
                        "cache_hit": bool(repaired.get("_api_cache_hit")),
                    }
                )
            errors = self._allocation_errors(config, repaired, symbols, book=book, prices=prices)
            repaired["_allocation_repair"] = {
                "attempted": True,
                "attempts": attempt,
                "original_errors": original_errors,
                "remaining_errors": errors,
                "original_api_usage": original_usage or {},
                "original_cache_hit": original_cache_hit,
                "attempt_api_usages": attempt_usages,
            }
            current = repaired
            if not errors:
                return current, calls
        return current, calls

    def _attach_api_usage(self, summary: Dict[str, Any], store: BenchmarkStore, run_id: str, config: BenchmarkConfig) -> Dict[str, Any]:
        run = store.get_benchmark_run(run_id)
        if not run:
            return summary
        summary["api_usage_estimate"] = estimate_run_api_usage(run, config, summary_override=summary)
        return summary

    def _allocation_errors(
        self,
        config: BenchmarkConfig,
        output: Dict[str, Any],
        symbols: List[str],
        *,
        book: PortfolioBook | None = None,
        prices: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        if config.mode == "single_stock":
            output = self._normalize_stage2_output(config, output, symbols, book=book, prices=prices)
        weights = self._coerce_target_weights(output.get("target_weights") or {}, symbols)
        gross = sum(abs(value) for value in weights.values())
        net = sum(weights.values())
        expected_cash = 1.0 - gross
        errors: List[Dict[str, Any]] = []
        if config.mode == "single_stock":
            if _safe_float(output.get("target_exposure")) is None:
                errors.append({"type": "missing_or_invalid_target_exposure"})
            for field in ("cash_drag_justification", "why_not_buy_hold"):
                if not isinstance(output.get(field), str) or not output.get(field, "").strip():
                    errors.append({"type": f"missing_{field}"})
            alignment = output.get("stage1_alignment")
            if alignment not in {"follow", "partial", "veto"}:
                errors.append({"type": "missing_or_invalid_stage1_alignment", "allowed": ["follow", "partial", "veto"]})
            veto_reason = output.get("stage1_veto_reason")
            if not isinstance(veto_reason, str):
                errors.append({"type": "missing_stage1_veto_reason"})
            elif alignment == "veto" and not veto_reason.strip():
                errors.append({"type": "empty_stage1_veto_reason_for_veto"})
        nonzero = [symbol for symbol, value in weights.items() if abs(value) > 1e-9]
        if gross > config.max_gross_exposure + 1e-9:
            errors.append({"type": "gross_exposure_exceeded", "actual": round(gross, 8), "max": config.max_gross_exposure})
        if len(nonzero) > config.max_nonzero_positions:
            errors.append({"type": "too_many_nonzero_positions", "actual": len(nonzero), "max": config.max_nonzero_positions})
        if not config.allow_short:
            shorts = {symbol: value for symbol, value in weights.items() if value < 0}
            if shorts:
                errors.append({"type": "shorts_not_allowed", "symbols": sorted(shorts)})
        declared_gross = _safe_float(output.get("gross_exposure"))
        declared_net = _safe_float(output.get("net_exposure"))
        declared_cash = _safe_float(output.get("cash_weight", output.get("cash_target_weight")))
        if config.mode != "single_stock":
            if declared_gross is None:
                errors.append({"type": "missing_gross_exposure"})
            elif abs(declared_gross - gross) > 1e-4:
                errors.append({"type": "gross_exposure_mismatch", "declared": declared_gross, "actual": round(gross, 8)})
            if declared_net is None:
                errors.append({"type": "missing_net_exposure"})
            elif abs(declared_net - net) > 1e-4:
                errors.append({"type": "net_exposure_mismatch", "declared": declared_net, "actual": round(net, 8)})
            if declared_cash is None:
                errors.append({"type": "missing_cash_weight"})
            elif abs(declared_cash - expected_cash) > 1e-4:
                errors.append({"type": "cash_weight_mismatch", "declared": declared_cash, "actual": round(expected_cash, 8), "rule": "cash_weight must equal 1 - gross_exposure"})

        holding_days = _safe_float(output.get("expected_holding_days", output.get("horizon_days")))
        if holding_days is None or holding_days <= 0:
            errors.append({"type": "missing_or_invalid_expected_holding_days"})
        if "input_evidence_refs" not in output or not isinstance(output.get("input_evidence_refs"), list):
            errors.append({"type": "missing_input_evidence_refs"})
        if "data_quality_warnings_used" not in output or not isinstance(output.get("data_quality_warnings_used"), list):
            errors.append({"type": "missing_data_quality_warnings_used"})

        if book is not None and prices:
            actual_turnover = estimate_target_turnover(book, weights, prices)
            declared_turnover = _safe_float(output.get("estimated_turnover"))
            if declared_turnover is None:
                errors.append({"type": "missing_estimated_turnover", "actual": actual_turnover})
            elif config.mode != "single_stock" and abs(declared_turnover - actual_turnover) > 0.05:
                errors.append({"type": "estimated_turnover_mismatch", "declared": declared_turnover, "actual": actual_turnover})
            turnover_limit = _safe_float(config.max_daily_turnover)
            if turnover_limit is not None and turnover_limit > 0 and actual_turnover > turnover_limit + 1e-9:
                errors.append(
                    {
                        "type": "max_daily_turnover_exceeded",
                        "actual_turnover": actual_turnover,
                        "max_daily_turnover": turnover_limit,
                        "rule": "estimated_turnover and actual target turnover must stay within the daily trading budget",
                    }
                )
                estimated_cost_bps = actual_turnover * float(config.slippage_bps or 0.0)
                expected_edge_bps = abs(float(_safe_float(output.get("expected_return_bps")) or 0.0))
                required_edge_bps = estimated_cost_bps * float(config.turnover_edge_multiplier or 1.0)
                if expected_edge_bps + 1e-9 < required_edge_bps:
                    errors.append(
                        {
                            "type": "turnover_cost_hurdle_failed",
                            "actual_turnover": actual_turnover,
                            "max_daily_turnover": turnover_limit,
                            "estimated_cost_bps": round(estimated_cost_bps, 8),
                            "expected_edge_bps": round(expected_edge_bps, 8),
                            "required_edge_bps": round(required_edge_bps, 8),
                        }
                    )
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
        repair_count = sum(int(execution.get("repair_count") or 0) for execution in executions)
        turnover_values = []
        long_pnl = 0.0
        short_pnl = 0.0
        for execution in executions:
            before = execution.get("portfolio_before") or {}
            equity_before = max(float(before.get("equity") or initial or 1.0), 1e-9)
            traded_value = 0.0
            for trade in execution.get("trades") or []:
                delta = float(trade.get("signed_delta") or 0.0)
                reference = float(trade.get("reference_price") or 0.0)
                traded_value += abs(delta) * reference
                if delta >= 0:
                    long_pnl -= float(trade.get("shares") or 0.0) * float(trade.get("fill_price") or reference)
                else:
                    short_pnl += float(trade.get("shares") or 0.0) * float(trade.get("fill_price") or reference)
            if traded_value:
                turnover_values.append(traded_value / equity_before)
        total_turnover = float(sum(turnover_values))
        avg_daily_turnover = float(pd.Series(turnover_values).mean()) if turnover_values else 0.0
        gross_of_cost_final = final + fees + slippage
        summary = {
            "mode": config.mode,
            "preset": config.run_preset,
            "symbol_count": len(symbols),
            "model": config.model or "unselected-model",
            "model_provider": config.model_provider,
            "no_paid_api_mode": config.no_paid_api_mode,
            "local_model_base_url": config.local_model_base_url,
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
                "invalid_allocation_count": model_failures,
                "repair_count": repair_count,
                "fees": fees,
                "slippage_cost": slippage,
                "gross_of_cost_final_equity": gross_of_cost_final,
                "gross_of_cost_return": gross_of_cost_final / initial - 1.0 if initial else 0.0,
                "net_of_cost_return": final / initial - 1.0 if initial else 0.0,
                "slippage_drag": slippage / initial if initial else 0.0,
                "fee_drag": fees / initial if initial else 0.0,
                "total_turnover": total_turnover,
                "avg_daily_turnover": avg_daily_turnover,
                "long_trade_cashflow": long_pnl,
                "short_trade_cashflow": short_pnl,
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
