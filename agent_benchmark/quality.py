from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .schemas import BenchmarkConfig, SecretConfig
from .local_provider import validate_no_paid_api_mode
from .storage import BenchmarkStore
from .warehouse.store import Warehouse
from .warehouse.universe import STOCK_SYMBOLS


OFFICIAL_PRESETS = {
    "single_stock_official",
    "budget_official",
    "full_official",
    "local_gemma_aapl_full",
    "local_gemma_aapl_online",
}
CORE_CONTEXT_SYMBOLS = {"SPY", "QQQ"}
STALE_FACT_DAYS = 730


def symbols_for_config(config: BenchmarkConfig) -> List[str]:
    if config.mode == "single_stock":
        return [config.symbol.upper()]
    return [symbol.upper() for symbol in (config.selected_symbols or STOCK_SYMBOLS)]


def is_official_full_run(config: BenchmarkConfig) -> bool:
    return config.run_preset in OFFICIAL_PRESETS and int(config.max_test_days or 0) == 0


def is_synthetic_news_title(title: Any) -> bool:
    text = str(title or "").strip().lower()
    return not text or text.startswith("gdelt event") or "nan" in text


def _payload_contains_synthetic_news_title(value: Any) -> bool:
    if isinstance(value, dict):
        if "title" in value and is_synthetic_news_title(value.get("title")):
            return True
        return any(_payload_contains_synthetic_news_title(child) for child in value.values())
    if isinstance(value, list):
        return any(_payload_contains_synthetic_news_title(child) for child in value)
    return False


def _missing_stage2_contract_fields(output: Dict[str, Any]) -> List[str]:
    required = [
        "target_weights",
        "cash_weight",
        "gross_exposure",
        "net_exposure",
        "expected_holding_days",
        "estimated_turnover",
        "estimated_slippage_cost_bps",
        "rebalance_reason",
        "input_evidence_refs",
        "data_quality_warnings_used",
    ]
    return [field for field in required if field not in (output or {})]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except Exception:
        return None


def _date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)[:10]


def _days_between(start: str, end: str) -> int | None:
    try:
        return (datetime.fromisoformat(end[:10]) - datetime.fromisoformat(start[:10])).days
    except Exception:
        return None


def _add_check(report: Dict[str, Any], check: Dict[str, Any]) -> None:
    report["checks"].append(check)
    if check.get("status") == "fail":
        report["blocking_issues"].append(check)
    elif check.get("status") == "warn":
        report["warnings"].append(check)


def _trading_dates(warehouse: Warehouse, start: str, end: str) -> List[str]:
    df = warehouse.conn.execute(
        """
        SELECT CAST(date AS VARCHAR) AS date
        FROM context_daily
        WHERE symbol = 'SPY'
          AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          AND market_open = true
          AND ohlcv_available = true
        ORDER BY date
        """,
        [start, end],
    ).fetchdf()
    return [str(value)[:10] for value in df["date"].tolist()] if not df.empty else []


def _calendar_coverage_end(warehouse: Warehouse, symbol: str, end: str) -> str:
    row = warehouse.conn.execute(
        """
        SELECT MAX(date)
        FROM context_daily
        WHERE symbol = ? AND date <= CAST(? AS DATE)
        """,
        [symbol, end],
    ).fetchone()
    return str(row[0])[:10] if row and row[0] is not None else ""


def build_preflight_report(
    config: BenchmarkConfig,
    secrets: SecretConfig,
    warehouse: Warehouse,
    *,
    store: BenchmarkStore | None = None,
) -> Dict[str, Any]:
    symbols = symbols_for_config(config)
    report: Dict[str, Any] = {
        "status": "pass",
        "official_run": config.run_preset in OFFICIAL_PRESETS,
        "full_run": is_official_full_run(config),
        "symbols": len(symbols),
        "train_start": config.train_start,
        "train_end": config.train_end,
        "test_start": config.test_start,
        "test_end": config.test_end,
        "benchmark_contract": {
            "version": getattr(config, "benchmark_contract_version", "legacy"),
            "evaluation_mode": getattr(config, "evaluation_mode", "legacy"),
            "selection_cutoff": getattr(config, "selection_cutoff", ""),
            "fixed_evaluation_cutoff": getattr(config, "fixed_evaluation_cutoff", ""),
            "globally_pristine": bool(getattr(config, "globally_pristine", False)),
            "historical_holdout_reveal_count_lower_bound": int(
                getattr(config, "historical_holdout_reveal_count_lower_bound", 0) or 0
            ),
            "historical_prompt_blinding": bool(
                getattr(config, "historical_prompt_blinding", False)
            ),
            "historical_prompt_blinding_contract": getattr(
                config, "historical_prompt_blinding_contract", ""
            ),
            "model_training_data_cutoff": getattr(config, "model_training_data_cutoff", ""),
            "historical_decision_authority": getattr(
                config, "historical_decision_authority", "llm"
            ),
            "historical_price_basis": getattr(config, "historical_price_basis", "legacy"),
            "action_space": config.single_stock_action_space,
            "online_test_learning": bool(getattr(config, "online_test_learning", False)),
            "memory_namespace": getattr(config, "memory_namespace", "legacy"),
            "memory_base_snapshot_id": getattr(config, "memory_base_snapshot_id", ""),
            "memory_online_stream_id": getattr(config, "memory_online_stream_id", "") or "bound_to_run_id",
        },
        "checks": [],
        "blocking_issues": [],
        "warnings": [],
    }
    try:
        validate_no_paid_api_mode(config, secrets)
        if config.evaluation_mode != "legacy":
            _add_check(
                report,
                {
                    "id": "evaluation_split",
                    "status": "pass",
                    "message": (
                        f"{config.train_start} through {config.selection_cutoff} is learning data; "
                        f"{config.test_start} through {config.test_end} is {config.evaluation_mode}."
                    ),
                    "test_outcomes_used_for_learning": bool(config.online_test_learning),
                },
            )
            _add_check(
                report,
                {
                    "id": "parametric_lookahead_guard",
                    "status": "pass" if config.historical_prompt_blinding else "fail",
                    "message": (
                        "Historical LLM prompts are identity/date blinded and raw identifying amounts are removed."
                        if config.historical_prompt_blinding
                        else "Historical LLM prompts can expose facts memorized during model pretraining."
                    ),
                    "model_training_data_cutoff": config.model_training_data_cutoff,
                    "blinding_contract": config.historical_prompt_blinding_contract,
                    "decision_authority": config.historical_decision_authority,
                },
            )
        if config.no_paid_api_mode:
            _add_check(
                report,
                {
                    "id": "no_paid_api_mode",
                    "status": "pass",
                    "message": "Model endpoint, embeddings, and news sources satisfy local-only no-paid safeguards.",
                    "model_provider": config.model_provider,
                    "base_url": config.local_model_base_url or secrets.openai_base_url,
                },
            )
    except Exception as exc:
        _add_check(report, {"id": "no_paid_api_mode", "status": "fail", "message": str(exc)})
    trading_dates = _trading_dates(warehouse, config.test_start, config.test_end)
    if config.evaluation_mode != "legacy":
        covered_through = _calendar_coverage_end(warehouse, "SPY", config.test_end)
        _add_check(
            report,
            {
                "id": "fixed_evaluation_end_coverage",
                "status": "pass" if covered_through == config.test_end else "fail",
                "message": (
                    f"Evaluation calendar is covered through {covered_through or 'no date'}; "
                    f"the frozen contract requires {config.test_end}."
                ),
                "covered_through": covered_through,
                "required_end": config.test_end,
            },
        )
    if int(config.max_test_days or 0) > 0:
        trading_dates = trading_dates[: int(config.max_test_days)]
    test_days = len(trading_dates)
    quantitative_historical_authority = bool(
        config.evaluation_mode in {"frozen_holdout", "causal_online_replay"}
        and config.historical_decision_authority == "precutoff_quantitative_policy"
    )
    stage1_chunks = (
        0
        if quantitative_historical_authority
        else math.ceil(len(symbols) / max(1, int(config.stage1_chunk_size or 1)))
    )
    exposure_critic_calls = (
        1
        if not quantitative_historical_authority
        and config.mode == "single_stock"
        and config.exposure_critic_enabled
        and config.single_stock_action_space not in {"trinary_all_in", "long_cash_hold"}
        else 0
    )
    decision_calls_per_day = (
        0
        if quantitative_historical_authority
        else stage1_chunks + 1 + exposure_critic_calls
    )
    cadence = getattr(config, "decision_cadence", "daily")
    if cadence == "weekly_event":
        scheduled_decision_days = len(
            {
                datetime.fromisoformat(value[:10]).isocalendar()[:2]
                for value in trading_dates
            }
        )
    else:
        scheduled_decision_days = test_days
    report["estimate"] = {
        "test_trading_days": test_days,
        "decision_calls_per_day": decision_calls_per_day,
        "historical_decision_authority": config.historical_decision_authority,
        "decision_cadence": cadence,
        "scheduled_decision_days": scheduled_decision_days,
        "estimated_model_calls": scheduled_decision_days * decision_calls_per_day,
        "estimated_model_calls_upper_bound": test_days * decision_calls_per_day,
        "event_trigger_calls_in_estimate": (
            "not_applicable_no_historical_llm_authority"
            if quantitative_historical_authority
            else "included_only_in_upper_bound"
        ),
        "stage1_chunks_per_day": stage1_chunks,
        "exposure_critic_calls_per_day": exposure_critic_calls,
        "llm_reflection_cadence": config.llm_reflection_cadence,
        "online_lessons_require_model_calls": False,
    }

    _price_coverage_check(report, warehouse, symbols, trading_dates)
    _context_coverage_check(report, warehouse, config, trading_dates)
    _macro_check(report, warehouse, config, secrets)
    _news_check(report, warehouse, config, symbols)
    _fundamental_check(report, warehouse, config, symbols)
    _memory_check(report, warehouse, config, symbols)
    if store is not None:
        _micro_pilot_check(report, store, config, symbols)

    if report["blocking_issues"]:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "warn"
    return report


def _price_coverage_check(report: Dict[str, Any], warehouse: Warehouse, symbols: List[str], trading_dates: List[str]) -> None:
    if not trading_dates:
        _add_check(report, {"id": "price_coverage", "status": "fail", "message": "No SPY trading calendar rows found for the test window."})
        return
    placeholders = ", ".join(["?"] * len(symbols))
    df = warehouse.conn.execute(
        f"""
        SELECT symbol,
               COUNT(*) AS rows,
               SUM(CASE WHEN ohlcv_available = true AND listed = true THEN 1 ELSE 0 END) AS available_rows
        FROM asset_daily
        WHERE symbol IN ({placeholders})
          AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          AND market_open = true
        GROUP BY symbol
        """,
        [*symbols, trading_dates[0], trading_dates[-1]],
    ).fetchdf()
    coverage = {row["symbol"]: int(row["available_rows"] or 0) for _, row in df.iterrows()} if not df.empty else {}
    missing = {symbol: len(trading_dates) - coverage.get(symbol, 0) for symbol in symbols if coverage.get(symbol, 0) < len(trading_dates)}
    _add_check(
        report,
        {
            "id": "price_coverage",
            "status": "fail" if missing else "pass",
            "message": "Stock OHLCV coverage is complete." if not missing else f"{len(missing)} symbols have missing test-window price rows.",
            "required_dates": len(trading_dates),
            "missing_by_symbol": missing,
        },
    )


def _context_coverage_check(report: Dict[str, Any], warehouse: Warehouse, config: BenchmarkConfig, trading_dates: List[str]) -> None:
    symbols = sorted(set(config.data_sources.index_symbols or []) | CORE_CONTEXT_SYMBOLS)
    if not trading_dates or not symbols:
        return
    placeholders = ", ".join(["?"] * len(symbols))
    df = warehouse.conn.execute(
        f"""
        SELECT symbol, COUNT(*) AS available_rows
        FROM context_daily
        WHERE symbol IN ({placeholders})
          AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          AND market_open = true
          AND ohlcv_available = true
        GROUP BY symbol
        """,
        [*symbols, trading_dates[0], trading_dates[-1]],
    ).fetchdf()
    coverage = {row["symbol"]: int(row["available_rows"] or 0) for _, row in df.iterrows()} if not df.empty else {}
    missing = {symbol: len(trading_dates) - coverage.get(symbol, 0) for symbol in symbols if coverage.get(symbol, 0) < len(trading_dates)}
    _add_check(
        report,
        {
            "id": "context_coverage",
            "status": "fail" if missing else "pass",
            "message": "Index/context coverage is complete." if not missing else f"{len(missing)} context symbols have missing rows.",
            "missing_by_symbol": missing,
        },
    )


def _macro_check(report: Dict[str, Any], warehouse: Warehouse, config: BenchmarkConfig, secrets: SecretConfig) -> None:
    if not config.data_sources.include_fred_macro or config.macro_policy == "omit_if_missing" and not secrets.fred_api_key:
        _add_check(report, {"id": "macro_quality", "status": "pass", "message": "Macro is unavailable and will be omitted from prompts.", "policy": config.macro_policy})
        return
    df = warehouse.conn.execute(
        """
        SELECT source_status, COUNT(*) AS rows
        FROM macro_daily
        WHERE date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        GROUP BY source_status
        """,
        [config.test_start, config.test_end],
    ).fetchdf()
    counts = {str(row["source_status"]): int(row["rows"]) for _, row in df.iterrows()} if not df.empty else {}
    ok_rows = counts.get("ok", 0)
    _add_check(
        report,
        {
            "id": "macro_quality",
            "status": "pass" if ok_rows > 0 else "fail",
            "message": "FRED macro rows are available." if ok_rows > 0 else "Macro was requested but no usable FRED observations are available.",
            "source_status_counts": counts,
        },
    )


def _news_check(report: Dict[str, Any], warehouse: Warehouse, config: BenchmarkConfig, symbols: List[str]) -> None:
    if int(config.max_news_per_symbol or 0) <= 0 or int(config.data_sources.max_news_per_day or 0) <= 0:
        _add_check(
            report,
            {
                "id": "news_quality",
                "status": "pass",
                "message": "News is deliberately disabled; no event labels or synthetic titles enter the prompt.",
                "total_rows": 0,
                "real_headline_rows": 0,
                "synthetic_or_nan_rows": 0,
                "real_headline_rate": 0.0,
                "policy": config.news_policy,
            },
        )
        return
    placeholders = ", ".join(["?"] * len(symbols))
    df = warehouse.conn.execute(
        f"""
        SELECT title
        FROM news_articles
        WHERE symbol IN ({placeholders})
          AND bucket_start BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        """,
        [*symbols, config.test_start, config.test_end],
    ).fetchdf()
    total = int(len(df))
    synthetic = int(sum(1 for title in df["title"].tolist() if is_synthetic_news_title(title))) if total else 0
    real = total - synthetic
    real_rate = real / total if total else 0.0
    status = "pass"
    message = "News rows have usable real titles."
    if total == 0:
        status = "warn"
        message = "No news rows found; prompts will disclose missing news."
    elif synthetic and config.news_policy != "real_titles_or_aggregate_events":
        status = "fail"
        message = "Synthetic GDELT event labels would be sent as raw headlines under the current news policy."
    elif real_rate < 0.5:
        status = "warn"
        message = "Most news rows are event-only GDELT records; they will be aggregated as event features, not fake headlines."
    _add_check(
        report,
        {
            "id": "news_quality",
            "status": status,
            "message": message,
            "total_rows": total,
            "real_headline_rows": real,
            "synthetic_or_nan_rows": synthetic,
            "real_headline_rate": real_rate,
            "policy": config.news_policy,
        },
    )


def _fundamental_check(report: Dict[str, Any], warehouse: Warehouse, config: BenchmarkConfig, symbols: List[str]) -> None:
    if not config.data_sources.include_sec_fundamentals:
        _add_check(report, {"id": "fundamental_quality", "status": "pass", "message": "SEC fundamentals are disabled."})
        return
    placeholders = ", ".join(["?"] * len(symbols))
    df = warehouse.conn.execute(
        f"""
        SELECT symbol, concept, value, unit, period_end, filed_date, form
        FROM sec_facts
        WHERE symbol IN ({placeholders}) AND filed_date <= CAST(? AS DATE)
        """,
        [*symbols, config.test_start],
    ).fetchdf()
    missing = []
    stale = []
    for symbol in symbols:
        group = df[df["symbol"] == symbol] if not df.empty else pd.DataFrame()
        if group.empty:
            missing.append(symbol)
            continue
        _, quality = canonicalize_fundamentals(group.to_dict(orient="records"), config.test_start)
        if quality["status"] != "ok":
            stale.append(symbol)
    status = "fail" if missing or len(stale) > max(3, len(symbols) * 0.15) else "pass"
    _add_check(
        report,
        {
            "id": "fundamental_quality",
            "status": status,
            "message": "Canonical fundamentals are usable." if status == "pass" else "Too many symbols lack fresh canonical fundamentals.",
            "missing_symbols": missing,
            "stale_or_weak_symbols": stale,
        },
    )


def _memory_check(report: Dict[str, Any], warehouse: Warehouse, config: BenchmarkConfig, symbols: List[str]) -> None:
    placeholders = ", ".join(["?"] * len(symbols))
    adjusted = getattr(config, "historical_price_basis", "legacy") == "adjusted"
    usable_price = (
        "open IS NOT NULL AND close IS NOT NULL AND adj_close IS NOT NULL "
        "AND open * adj_close / NULLIF(close, 0) > 0"
        if adjusted
        else "close IS NOT NULL"
    )
    df = warehouse.conn.execute(
        f"""
        SELECT symbol, COUNT(*) AS rows
        FROM asset_daily
        WHERE symbol IN ({placeholders})
          AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          AND ohlcv_available = true
          AND {usable_price}
        GROUP BY symbol
        """,
        [*symbols, config.train_start, config.train_end],
    ).fetchdf()
    counts = {row["symbol"]: int(row["rows"] or 0) for _, row in df.iterrows()} if not df.empty else {}
    weak = {symbol: counts.get(symbol, 0) for symbol in symbols if counts.get(symbol, 0) < 250}
    _add_check(
        report,
        {
            "id": "memory_coverage",
            "status": "fail" if weak else "pass",
            "message": "Historical memory coverage is sufficient." if not weak else f"{len(weak)} symbols have weak training coverage.",
            "weak_symbols": weak,
            "min_rows_per_symbol": min(counts.values()) if counts else 0,
            "memory_k_neighbors": config.memory_k_neighbors,
            "historical_price_basis": getattr(config, "historical_price_basis", "legacy"),
        },
    )


def _micro_pilot_check(report: Dict[str, Any], store: BenchmarkStore, config: BenchmarkConfig, symbols: List[str]) -> None:
    if not is_official_full_run(config) or not config.require_paid_micro_pilot:
        return
    candidates = []
    for run in store.list_benchmark_runs(limit=100):
        cfg = run.get("config") or {}
        summary = run.get("summary") or {}
        metrics = summary.get("metrics") or {}
        if run.get("status") != "completed":
            continue
        if summary.get("dry_run"):
            continue
        if cfg.get("model") != config.model or cfg.get("mode") != config.mode:
            continue
        if int(cfg.get("max_test_days") or 0) <= 0 or int(cfg.get("max_test_days") or 0) > 5:
            continue
        if metrics.get("model_failures", 0) == 0:
            candidates.append({"id": run["id"], "created_at": run["created_at"], "metrics": metrics})
    _add_check(
        report,
        {
            "id": "paid_micro_pilot",
            "status": "pass" if candidates else "fail",
            "message": "A clean paid micro-pilot exists." if candidates else "Run a clean paid 5-day micro-pilot before spending tokens on a full official run.",
            "matching_runs": candidates[:5],
        },
    )


def canonicalize_fundamentals(rows: Iterable[Dict[str, Any]], decision_date: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    concepts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    excluded_stale = 0
    for row in rows:
        filed = _date(row.get("filed_date"))
        age = _days_between(filed, decision_date) if filed else None
        if age is None or age > STALE_FACT_DAYS:
            excluded_stale += 1
            continue
        concepts[str(row.get("concept") or "")].append(row)

    def latest(names: List[str]) -> Dict[str, Any] | None:
        candidates = []
        for name in names:
            candidates.extend(concepts.get(name, []))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (_date(item.get("filed_date")), _date(item.get("period_end"))), reverse=True)[0]

    def value(names: List[str]) -> float | None:
        row = latest(names)
        return _safe_float(row.get("value")) if row else None

    revenue = value(["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"])
    net_income = value(["NetIncomeLoss", "ProfitLoss"])
    operating_income = value(["OperatingIncomeLoss"])
    operating_cash = value(["NetCashProvidedByUsedInOperatingActivities"])
    capex = value(["PaymentsToAcquirePropertyPlantAndEquipment"])
    assets = value(["Assets"])
    liabilities = value(["Liabilities"])
    equity = value(["StockholdersEquity"])
    debt = value(["LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"])
    cash = value(["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalentsAtFairValue"])
    eps = value(["EarningsPerShareDiluted"])
    latest_rows = [row for values in concepts.values() for row in values]
    latest_filing = max((_date(row.get("filed_date")) for row in latest_rows), default="")
    latest_age = _days_between(latest_filing, decision_date) if latest_filing else None
    fcf = operating_cash - abs(capex) if operating_cash is not None and capex is not None else None

    canonical = {
        "revenue_ttm": revenue,
        "net_income_ttm": net_income,
        "operating_income": operating_income,
        "operating_margin": operating_income / revenue if operating_income is not None and revenue else None,
        "net_margin": net_income / revenue if net_income is not None and revenue else None,
        "free_cash_flow": fcf,
        "debt_to_equity": debt / equity if debt is not None and equity else None,
        "cash_to_assets": cash / assets if cash is not None and assets else None,
        "assets": assets,
        "liabilities": liabilities,
        "eps_diluted": eps,
        "latest_filing_date": latest_filing,
        "latest_filing_age_days": latest_age,
    }
    canonical = {key: value for key, value in canonical.items() if value is not None and not (isinstance(value, float) and not math.isfinite(value))}
    quality = {
        "status": "ok" if revenue is not None and latest_age is not None and latest_age <= STALE_FACT_DAYS else "weak",
        "facts_used": len(latest_rows),
        "stale_facts_excluded": excluded_stale,
        "latest_filing_date": latest_filing,
        "latest_filing_age_days": latest_age,
    }
    return canonical, quality


def build_run_diagnostics(run: Dict[str, Any], config: BenchmarkConfig, warehouse: Warehouse | None = None) -> Dict[str, Any]:
    all_decisions = run.get("decisions") or []
    decisions = [item for item in all_decisions if item.get("stage") == "stage2"]
    initial = float(config.initial_cash or 0.0)
    if not decisions:
        return {"status": "empty", "message": "No Stage 2 decisions are available."}

    symbols = symbols_for_config(config)
    primary_symbol = symbols[0] if symbols else config.symbol.upper()
    stage1_by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in all_decisions:
        if item.get("stage") != "stage1":
            continue
        for analysis in (item.get("output") or {}).get("analyses") or []:
            stage1_by_date[str(item.get("decision_date"))].append(analysis)

    rows = []
    legacy_stage2_schema_days = 0
    previous_equity = None
    symbol_cashflows: Dict[str, float] = defaultdict(float)
    symbol_traded: Dict[str, float] = defaultdict(float)
    trade_counts: Counter[str] = Counter()
    stage1_alignment_counts: Counter[str] = Counter()
    bullish_but_underexposed_days = 0
    for item in decisions:
        execution = item.get("execution") or {}
        output = item.get("output") or {}
        missing_contract_fields = _missing_stage2_contract_fields(output)
        if config.mode == "single_stock":
            for field in ("target_exposure", "cash_drag_justification", "why_not_buy_hold", "stage1_alignment", "stage1_veto_reason"):
                if field not in output and field not in missing_contract_fields:
                    missing_contract_fields.append(field)
            alignment = output.get("stage1_alignment")
            if alignment:
                stage1_alignment_counts[str(alignment)] += 1
        if missing_contract_fields:
            legacy_stage2_schema_days += 1
        before = execution.get("portfolio_before") or {}
        after = execution.get("portfolio_after") or {}
        equity = float(after.get("equity") or 0.0)
        weights = execution.get("target_weights") or (item.get("output") or {}).get("target_weights") or {}
        target_exposure = _safe_float(output.get("target_exposure"))
        if target_exposure is None and config.mode == "single_stock":
            target_exposure = _safe_float(weights.get(primary_symbol))
        if target_exposure is None:
            target_exposure = _safe_float(after.get("net_exposure")) or 0.0
        stage1_rows = stage1_by_date.get(str(item.get("decision_date")), [])
        primary_stage1 = next((row for row in stage1_rows if row.get("symbol") == primary_symbol), {})
        if config.mode == "single_stock" and primary_stage1.get("stance") == "bullish" and abs(float(target_exposure or 0.0)) < 0.25:
            bullish_but_underexposed_days += 1
        turnover_value = 0.0
        for trade in execution.get("trades") or []:
            symbol = str(trade.get("symbol"))
            delta = float(trade.get("signed_delta") or 0.0)
            price = float(trade.get("reference_price") or trade.get("fill_price") or 0.0)
            fill = float(trade.get("fill_price") or price)
            turnover_value += abs(delta) * price
            if delta > 0:
                symbol_cashflows[symbol] -= delta * fill
            else:
                symbol_cashflows[symbol] += abs(delta) * fill
            symbol_traded[symbol] += abs(delta) * price
            trade_counts[symbol] += 1
        equity_before = max(float(before.get("equity") or initial or 1.0), 1e-9)
        rows.append(
            {
                "decision_date": item.get("decision_date"),
                "fill_date": item.get("fill_date"),
                "equity": equity,
                "daily_return": equity / previous_equity - 1.0 if previous_equity else None,
                "gross_exposure": float(after.get("gross_exposure") or 0.0),
                "net_exposure": float(after.get("net_exposure") or 0.0),
                "short_exposure": float(after.get("short_exposure") or 0.0),
                "turnover": turnover_value / equity_before,
                "slippage_cost": float(execution.get("slippage_cost") or 0.0),
                "model_failure": bool(execution.get("model_failure")),
                "target_exposure": float(target_exposure or 0.0),
                "stage1_stance": primary_stage1.get("stance"),
                "stage1_alignment": output.get("stage1_alignment"),
                "target_gross": sum(abs(float(value)) for value in weights.values()),
                "nonzero_positions": sum(abs(float(value)) > 1e-9 for value in weights.values()),
                "missing_contract_fields": missing_contract_fields,
                "thesis": output.get("portfolio_thesis", ""),
                "decision_kind": output.get("decision_kind") or execution.get("decision_kind") or "model_decision",
                "requested_action": output.get("requested_action") or execution.get("requested_action") or output.get("action"),
                "executed_action": output.get("executed_action") or execution.get("executed_action") or output.get("action"),
                "deferred_action": (output.get("hysteresis") or execution.get("hysteresis") or {}).get("status") == "deferred",
                "mechanical_deleverage": bool(
                    execution.get("mechanical_deleverage")
                    or any(event.get("type") == "mechanical_gross_deleverage" for event in execution.get("events") or [])
                ),
            }
        )
        previous_equity = equity

    df = pd.DataFrame(rows)
    slippage = float(df["slippage_cost"].sum())
    final_equity = float(df.iloc[-1]["equity"])
    gross_of_cost_final = final_equity + slippage
    worst_days = df.sort_values("daily_return", na_position="last").head(10).to_dict(orient="records")
    max_participation_exposure = max(float(config.max_gross_exposure or 1.0), 1e-9)
    avg_target_exposure = float(df["target_exposure"].mean()) if "target_exposure" in df else 0.0
    if config.mode == "single_stock":
        participation_ratio = float(df["target_exposure"].clip(lower=0.0).mean() / max_participation_exposure)
    else:
        participation_ratio = float(df["gross_exposure"].mean() / max_participation_exposure)
    cash_drag_proxy = None
    missed_upside_days = None
    if warehouse is not None and config.mode == "single_stock" and len(df) > 1:
        missed_upside_days = 0
        cash_drag_proxy = 0.0
        ordered = df.sort_values("fill_date").reset_index(drop=True)
        for index in range(0, len(ordered) - 1):
            start_date = str(ordered.iloc[index]["fill_date"])
            end_date = str(ordered.iloc[index + 1]["fill_date"])
            try:
                prices = warehouse.conn.execute(
                    """
                    SELECT CAST(date AS VARCHAR) AS date, COALESCE(adj_close, close) AS price
                    FROM asset_daily
                    WHERE symbol = ?
                      AND date IN (CAST(? AS DATE), CAST(? AS DATE))
                      AND ohlcv_available = true
                    """,
                    [primary_symbol, start_date, end_date],
                ).fetchdf()
            except Exception:
                continue
            if prices.empty or len(prices) < 2:
                continue
            by_date = {str(row["date"])[:10]: _safe_float(row["price"]) for _, row in prices.iterrows()}
            start_price = by_date.get(start_date)
            end_price = by_date.get(end_date)
            if start_price is None or end_price is None or start_price <= 0:
                continue
            stock_return = end_price / start_price - 1.0
            exposure = float(ordered.iloc[index]["target_exposure"] or 0.0)
            if stock_return > 0:
                if exposure < 0.25:
                    missed_upside_days += 1
                cash_drag_proxy += stock_return * max(0.0, 1.0 - exposure)
    monthly = []
    if not df.empty:
        df["month"] = df["fill_date"].astype(str).str[:7]
        for month, group in df.groupby("month"):
            monthly.append(
                {
                    "month": month,
                    "days": int(len(group)),
                    "return": float(group.iloc[-1]["equity"] / group.iloc[0]["equity"] - 1.0) if len(group) > 1 and group.iloc[0]["equity"] else 0.0,
                    "avg_gross": float(group["gross_exposure"].mean()),
                    "avg_net": float(group["net_exposure"].mean()),
                    "turnover": float(group["turnover"].sum()),
                    "slippage": float(group["slippage_cost"].sum()),
                    "model_failures": int(group["model_failure"].sum()),
                }
            )

    symbol_pnl = []
    if warehouse is not None:
        final_date = str(decisions[-1].get("fill_date"))
        price_expression = (
            "open * adj_close / NULLIF(close, 0)"
            if getattr(config, "historical_price_basis", "legacy") == "adjusted"
            else "open"
        )
        final_positions = (decisions[-1].get("execution") or {}).get("portfolio_after", {}).get("positions") or {}
        for symbol, shares in final_positions.items():
            try:
                price = warehouse.conn.execute(
                    f"SELECT {price_expression} FROM asset_daily WHERE symbol = ? AND date = CAST(? AS DATE)",
                    [symbol, final_date],
                ).fetchone()
                final_value = float(shares) * float(price[0]) if price and price[0] is not None else 0.0
            except Exception:
                final_value = 0.0
            symbol_cashflows[str(symbol)] += final_value
    for symbol, pnl in symbol_cashflows.items():
        symbol_pnl.append(
            {
                "symbol": symbol,
                "pnl": float(pnl),
                "trades": int(trade_counts.get(symbol, 0)),
                "traded_value": float(symbol_traded.get(symbol, 0.0)),
            }
        )
    symbol_pnl.sort(key=lambda item: item["pnl"])
    synthetic_news_prompt_records = sum(1 for item in all_decisions if _payload_contains_synthetic_news_title(item.get("input") or {}))
    run_id = str(run.get("id") or "")
    cross_run_online_memory = 0
    if not getattr(config, "memory_online_stream_id", ""):
        for item in decisions:
            for memory_item in (item.get("input") or {}).get("memory") or []:
                if (
                    memory_item.get("memory_layer") == "online"
                    and memory_item.get("source_run_id")
                    and str(memory_item.get("source_run_id")) != run_id
                ):
                    cross_run_online_memory += 1
    maximum_observed_gross = float(df["gross_exposure"].max()) if not df.empty else 0.0
    post_execution_gross_breaches = int((df["gross_exposure"] > float(config.max_gross_exposure) + 1e-8).sum())
    diagnostic_reasons = []
    invalid_allocations = int(df["model_failure"].sum())
    if invalid_allocations:
        diagnostic_reasons.append("invalid_allocations")
    if legacy_stage2_schema_days:
        diagnostic_reasons.append("legacy_stage2_schema")
    if synthetic_news_prompt_records:
        diagnostic_reasons.append("synthetic_news_titles_in_prompt")
    if cross_run_online_memory:
        diagnostic_reasons.append("cross_run_online_memory")
    if post_execution_gross_breaches:
        diagnostic_reasons.append("post_execution_gross_breach")

    diagnostics = {
        "status": "ok",
        "official_status": "diagnostic" if diagnostic_reasons else "official_candidate",
        "diagnostic_reasons": diagnostic_reasons,
        "metrics": {
            "final_equity": final_equity,
            "net_return": final_equity / initial - 1.0 if initial else 0.0,
            "gross_of_cost_final_equity": gross_of_cost_final,
            "gross_of_cost_return": gross_of_cost_final / initial - 1.0 if initial else 0.0,
            "slippage_drag": slippage / initial if initial else 0.0,
            "slippage_cost": slippage,
            "total_turnover": float(df["turnover"].sum()),
            "avg_daily_turnover": float(df["turnover"].mean()),
            "avg_gross_exposure": float(df["gross_exposure"].mean()),
            "avg_net_exposure": float(df["net_exposure"].mean()),
            "avg_short_exposure": float(df["short_exposure"].mean()),
            "avg_target_exposure": avg_target_exposure,
            "participation_ratio": participation_ratio,
            "cash_drag_proxy": cash_drag_proxy,
            "missed_upside_days": missed_upside_days,
            "bullish_but_underexposed_days": bullish_but_underexposed_days,
            "stage1_follow_rate": float(stage1_alignment_counts.get("follow", 0) / max(1, sum(stage1_alignment_counts.values()))),
            "stage1_partial_rate": float(stage1_alignment_counts.get("partial", 0) / max(1, sum(stage1_alignment_counts.values()))),
            "stage1_veto_rate": float(stage1_alignment_counts.get("veto", 0) / max(1, sum(stage1_alignment_counts.values()))),
            "invalid_allocations": invalid_allocations,
            "legacy_stage2_schema_days": legacy_stage2_schema_days,
            "synthetic_news_prompt_records": synthetic_news_prompt_records,
            "cadence_hold_count": int((df["decision_kind"] == "cadence_hold").sum()),
            "model_decision_count": int((df["decision_kind"] == "model_decision").sum()),
            "deferred_action_change_count": int(df["deferred_action"].sum()),
            "mechanical_deleverage_count": int(df["mechanical_deleverage"].sum()),
            "requested_action_counts": dict(Counter(str(value) for value in df["requested_action"].dropna())),
            "executed_action_counts": dict(Counter(str(value) for value in df["executed_action"].dropna())),
            "cross_run_online_memory_count": cross_run_online_memory,
            "maximum_observed_gross": maximum_observed_gross,
            "post_execution_gross_breach_count": post_execution_gross_breaches,
            "historical_price_basis": getattr(config, "historical_price_basis", "legacy"),
            "benchmark_contract_version": getattr(config, "benchmark_contract_version", "legacy"),
            "avg_nonzero_positions": float(df["nonzero_positions"].mean()),
        },
        "worst_days": worst_days,
        "monthly": monthly,
        "symbol_pnl_worst": symbol_pnl[:10],
        "symbol_pnl_best": list(reversed(symbol_pnl[-10:])),
    }
    return diagnostics
