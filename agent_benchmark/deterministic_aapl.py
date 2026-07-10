from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yfinance as yf


REQUIRED_MARKET_COLUMNS = (
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "spy_adj_close",
)


@dataclass(frozen=True)
class CostAssumptions:
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    commission_per_share: float = 0.0
    annual_margin_rate: float = 0.08

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class StrategySpec:
    name: str = "trend_regime_tilt_v1"
    symbol: str = "AAPL"
    aapl_sma_days: int = 150
    spy_sma_days: int = 200
    risk_on_exposure: float = 1.10
    risk_off_exposure: float = 0.975
    mixed_exposure: float = 1.00
    rebalance: str = "daily"
    max_exposure: float = 1.10

    def validate(self) -> None:
        if self.symbol != "AAPL":
            raise ValueError("This goal and runner support AAPL only; reject APPL and all other symbols.")
        if self.name not in {
            "trend_regime_tilt_v1",
            "exhaustion_cash_v1",
            "downside_rebound_v1",
            "buy_hold",
        }:
            raise ValueError(f"Unsupported deterministic strategy: {self.name}")
        if self.rebalance not in {"daily", "on_target_change"}:
            raise ValueError("rebalance must be daily or on_target_change")
        if self.aapl_sma_days < 2 or self.spy_sma_days < 2:
            raise ValueError("Moving-average lookbacks must be at least two sessions")
        for field in ("risk_on_exposure", "risk_off_exposure", "mixed_exposure"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0 or value > self.max_exposure:
                raise ValueError(f"{field} must be between zero and max_exposure")


@dataclass(frozen=True)
class EvaluationPeriod:
    name: str
    start: str
    end: str


DEFAULT_PERIODS = (
    EvaluationPeriod("2024", "2024-01-01", "2024-12-31"),
    EvaluationPeriod("2025", "2025-01-01", "2025-12-31"),
    EvaluationPeriod("2026_ytd", "2026-01-01", "2026-07-09"),
)


def _iso_day(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def canonical_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    if "date" in clean.columns:
        clean["date"] = pd.to_datetime(clean["date"], errors="raise")
        clean = clean.set_index("date")
    clean.index = pd.DatetimeIndex(pd.to_datetime(clean.index, errors="raise")).tz_localize(None)
    clean = clean.sort_index()
    if clean.index.has_duplicates:
        raise ValueError("Market data contains duplicate trading dates")
    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in clean.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")
    for column in REQUIRED_MARKET_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=list(REQUIRED_MARKET_COLUMNS))
    if clean.empty:
        raise ValueError("No complete AAPL/SPY observations remain after validation")
    if (clean[list(REQUIRED_MARKET_COLUMNS)] <= 0).any().any():
        raise ValueError("Market prices must be strictly positive")
    clean["aapl_adj_open"] = clean["aapl_open"] * clean["aapl_adj_close"] / clean["aapl_close"]
    if (~np.isfinite(clean["aapl_adj_open"]) | (clean["aapl_adj_open"] <= 0)).any():
        raise ValueError("Adjusted AAPL open could not be constructed")
    return clean[
        ["aapl_open", "aapl_close", "aapl_adj_close", "aapl_adj_open", "spy_adj_close"]
    ].astype(float)


def _flatten_download(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    if isinstance(clean.columns, pd.MultiIndex):
        clean.columns = clean.columns.get_level_values(0)
    clean.columns = [str(column).lower().replace(" ", "_") for column in clean.columns]
    return clean


def download_market_frame(start: str, end_inclusive: str) -> pd.DataFrame:
    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).date().isoformat()
    downloaded: Dict[str, pd.DataFrame] = {}
    for symbol in ("AAPL", "SPY"):
        raw = yf.download(
            symbol,
            start=start,
            end=end_exclusive,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise RuntimeError(f"Yahoo Finance returned no data for {symbol}")
        downloaded[symbol] = _flatten_download(raw)
    aapl = downloaded["AAPL"][["open", "close", "adj_close"]].rename(
        columns={
            "open": "aapl_open",
            "close": "aapl_close",
            "adj_close": "aapl_adj_close",
        }
    )
    spy = downloaded["SPY"][["adj_close"]].rename(columns={"adj_close": "spy_adj_close"})
    return canonical_market_frame(aapl.join(spy, how="inner"))


def load_or_download_market_frame(
    cache_path: Path,
    *,
    start: str,
    end_inclusive: str,
    refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end_inclusive)

    def requested_slice(source: pd.DataFrame) -> pd.DataFrame:
        sliced = source.loc[(source.index >= requested_start) & (source.index <= requested_end)]
        if sliced.empty:
            raise ValueError("Cached/downloaded market data does not overlap the requested range")
        return sliced

    if cache_path.exists() and not refresh:
        cached = canonical_market_frame(pd.read_csv(cache_path))
        # A requested calendar start can be a weekend/holiday (the default is
        # 1999-01-01 while the first AAPL session is 1999-01-04).  Accept the
        # first observed joint session within one week instead of repeatedly
        # redownloading an otherwise complete immutable cache.
        start_coverage_deadline = pd.Timestamp(start) + pd.Timedelta(days=7)
        if (
            cached.index.min() <= start_coverage_deadline
            and cached.index.max() >= pd.Timestamp(end_inclusive)
        ):
            return requested_slice(cached), "cache"
    frame = download_market_frame(start, end_inclusive)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(
        frame.reset_index(names="date"),
        cache_path,
        index=False,
        float_format="%.17g",
    )
    # Evaluate the exact serialized snapshot that later cached runs will use.
    serialized = canonical_market_frame(pd.read_csv(cache_path))
    return requested_slice(serialized), "yfinance"


def market_data_sha256(frame: pd.DataFrame) -> str:
    canonical = canonical_market_frame(frame)
    payload = canonical.reset_index(names="date").to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12g",
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strategy_sha256(spec: StrategySpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_target_exposure(frame: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    spec.validate()
    data = canonical_market_frame(frame)
    if spec.name == "buy_hold":
        return pd.Series(1.0, index=data.index, name="target_exposure")
    if spec.name == "trend_regime_tilt_v1":
        aapl_sma = data["aapl_adj_close"].rolling(
            spec.aapl_sma_days,
            min_periods=spec.aapl_sma_days,
        ).mean()
        spy_sma = data["spy_adj_close"].rolling(
            spec.spy_sma_days,
            min_periods=spec.spy_sma_days,
        ).mean()
        aapl_above = data["aapl_adj_close"] > aapl_sma
        spy_above = data["spy_adj_close"] > spy_sma
        values = np.select(
            [aapl_above & spy_above, (~aapl_above) & (~spy_above)],
            [spec.risk_on_exposure, spec.risk_off_exposure],
            default=spec.mixed_exposure,
        )
        # A complete warm-up is mandatory.  The configured evaluation periods
        # begin years after it, but failing closed here prevents a partial SMA
        # from silently becoming a different strategy.
        values = pd.Series(values, index=data.index, dtype=float)
        values.loc[aapl_sma.isna() | spy_sma.isna()] = np.nan
        return values.rename("target_exposure")
    intraday = data["aapl_close"] / data["aapl_open"] - 1.0
    if spec.name == "exhaustion_cash_v1":
        q_aapl = intraday.rolling(1260, min_periods=1260).quantile(0.975).shift(1)
        # This frozen failed candidate deliberately omits the old QQQ leg
        # because the shared compact dataset contains only SPY context.
        trigger = intraday > np.maximum(0.025, q_aapl)
        values = pd.Series(np.where(trigger, 0.0, 1.0), index=data.index, dtype=float)
        values.loc[q_aapl.isna()] = np.nan
        return values.rename("target_exposure")
    if spec.name == "downside_rebound_v1":
        lower_tail = intraday.rolling(756, min_periods=756).quantile(0.05).shift(1)
        trigger = (intraday < lower_tail) & (intraday < -0.025)
        values = pd.Series(np.where(trigger, 1.10, 1.0), index=data.index, dtype=float)
        values.loc[lower_tail.isna()] = np.nan
        return values.rename("target_exposure")
    raise AssertionError(f"Unhandled strategy: {spec.name}")


def _trade_delta_for_target(
    *,
    equity: float,
    current_market_value: float,
    price: float,
    target: float,
    costs: CostAssumptions,
) -> float:
    desired_without_costs = target * equity - current_market_value
    if abs(desired_without_costs) <= 1e-12:
        return 0.0
    per_trade = costs.commission_per_trade
    per_share_friction = price * costs.slippage_bps / 10_000.0 + costs.commission_per_share
    if desired_without_costs > 0:
        denominator = price + target * per_share_friction
        numerator = target * (equity - per_trade) - current_market_value
        return max(0.0, numerator / max(denominator, 1e-12))
    denominator = price - target * per_share_friction
    numerator = current_market_value - target * (equity - per_trade)
    return -max(0.0, numerator / max(denominator, 1e-12))


def simulate_period(
    frame: pd.DataFrame,
    target_at_close: pd.Series,
    period: EvaluationPeriod,
    costs: CostAssumptions,
    *,
    initial_cash: float = 1000.0,
    max_exposure: float = 1.10,
    rebalance: str = "daily",
) -> pd.DataFrame:
    costs.validate()
    if rebalance not in {"daily", "on_target_change"}:
        raise ValueError("rebalance must be daily or on_target_change")
    data = canonical_market_frame(frame)
    target = pd.to_numeric(target_at_close.reindex(data.index), errors="coerce")
    decision_dates = data.index.to_series().shift(1)
    executed_target = target.shift(1)
    mask = (data.index >= pd.Timestamp(period.start)) & (data.index <= pd.Timestamp(period.end))
    fill_data = data.loc[mask]
    if fill_data.empty:
        raise ValueError(f"No AAPL fills are available for period {period.name}")
    fill_targets = executed_target.loc[fill_data.index]
    fill_decisions = decision_dates.loc[fill_data.index]
    if fill_targets.isna().any() or fill_decisions.isna().any():
        bad = fill_targets.index[fill_targets.isna()].tolist()
        raise ValueError(f"Strategy lacks causal warm-up before fills: {bad[:3]}")
    if (fill_targets < 0).any() or (fill_targets > max_exposure + 1e-12).any():
        raise ValueError("Target exposure is outside the declared long-only leverage bounds")

    cash = float(initial_cash)
    shares = 0.0
    previous_fill_date: pd.Timestamp | None = None
    previous_requested_target: float | None = None
    rows: list[Dict[str, Any]] = []
    for fill_date, market_row in fill_data.iterrows():
        fill_date = pd.Timestamp(fill_date)
        price = float(market_row["aapl_adj_open"])
        margin_interest = 0.0
        if previous_fill_date is not None and cash < 0:
            elapsed_days = max(1, int((fill_date - previous_fill_date).days))
            margin_interest = (-cash) * costs.annual_margin_rate * elapsed_days / 365.0
            cash -= margin_interest
        equity_before = cash + shares * price
        if not math.isfinite(equity_before) or equity_before <= 0:
            raise RuntimeError(f"Portfolio equity became non-positive on {_iso_day(fill_date)}")
        holding_exposure = shares * price / equity_before
        requested_target = float(fill_targets.loc[fill_date])
        should_rebalance = (
            rebalance == "daily"
            or previous_requested_target is None
            or abs(requested_target - previous_requested_target) > 1e-12
        )
        delta = 0.0
        fees = 0.0
        slippage = 0.0
        fill_price = price
        if should_rebalance:
            delta = _trade_delta_for_target(
                equity=equity_before,
                current_market_value=shares * price,
                price=price,
                target=requested_target,
                costs=costs,
            )
            if abs(delta) > 1e-12:
                is_buy = delta > 0
                fill_price = price * (
                    1.0 + costs.slippage_bps / 10_000.0
                    if is_buy
                    else 1.0 - costs.slippage_bps / 10_000.0
                )
                fees = costs.commission_per_trade + abs(delta) * costs.commission_per_share
                slippage = abs(delta) * abs(fill_price - price)
                cash -= delta * fill_price + fees
                shares += delta
                # Eliminate harmless floating-point dust at the exact
                # all-cash/all-invested boundaries.  This matters for the
                # unleveraged runner: a mathematical 100% target must not be
                # reported as borrowed cash or exposure infinitesimally above
                # one merely because of binary arithmetic.
                if abs(cash) <= 1e-10:
                    cash = 0.0
                if abs(shares) <= 1e-12:
                    shares = 0.0
        equity_after = cash + shares * price
        actual_exposure = shares * price / equity_after
        if actual_exposure > max_exposure + 1e-8 or actual_exposure < -1e-10:
            raise RuntimeError(
                f"Executed exposure {actual_exposure} breached bounds on {_iso_day(fill_date)}"
            )
        turnover = abs(delta) * price / equity_before
        rows.append(
            {
                "decision_date": _iso_day(fill_decisions.loc[fill_date]),
                "fill_date": _iso_day(fill_date),
                "adjusted_open": price,
                "equity_before_fill": equity_before,
                "equity": equity_after,
                "cash": cash,
                "shares": shares,
                "holding_exposure_for_return": holding_exposure,
                "target_exposure": requested_target,
                "new_exposure_after_fill": actual_exposure,
                "signed_share_delta": delta,
                "reference_price": price,
                "fill_price": fill_price,
                "turnover": turnover,
                "fees": fees,
                "slippage": slippage,
                "margin_interest": margin_interest,
                "trade_executed": bool(abs(delta) > 1e-12),
            }
        )
        previous_fill_date = fill_date
        previous_requested_target = requested_target
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise AssertionError("Simulation produced no ledger rows")
    ledger["daily_return"] = ledger["equity"].pct_change()
    ledger.loc[0, "daily_return"] = ledger.loc[0, "equity"] / float(initial_cash) - 1.0
    ledger["monetary_pnl"] = ledger["equity"].diff()
    ledger.loc[0, "monetary_pnl"] = ledger.loc[0, "equity"] - float(initial_cash)
    running_peak = np.maximum.accumulate(np.r_[float(initial_cash), ledger["equity"].to_numpy()])[1:]
    ledger["drawdown"] = ledger["equity"].to_numpy() / running_peak - 1.0
    return ledger


def _drawdown_details(ledger: pd.DataFrame, initial_cash: float) -> Dict[str, Any]:
    values = np.r_[float(initial_cash), ledger["equity"].to_numpy(dtype=float)]
    dates: list[str] = ["initial"] + ledger["fill_date"].astype(str).tolist()
    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1.0
    trough_index = int(np.argmin(drawdowns))
    peak_index = int(np.argmax(values[: trough_index + 1]))
    recovery_index: int | None = None
    peak_value = float(values[peak_index])
    for index in range(trough_index + 1, len(values)):
        if values[index] >= peak_value:
            recovery_index = index
            break
    longest_underwater = 0
    current_underwater = 0
    for drawdown in drawdowns:
        if drawdown < -1e-12:
            current_underwater += 1
            longest_underwater = max(longest_underwater, current_underwater)
        else:
            current_underwater = 0
    return {
        "max_drawdown": float(drawdowns[trough_index]),
        "peak_date": dates[peak_index],
        "trough_date": dates[trough_index],
        "recovery_date": dates[recovery_index] if recovery_index is not None else None,
        "longest_underwater_observations": int(longest_underwater),
        "underwater_observations": int(np.count_nonzero(drawdowns < -1e-12)),
    }


def performance_metrics(
    ledger: pd.DataFrame,
    *,
    initial_cash: float = 1000.0,
) -> Dict[str, Any]:
    if ledger.empty:
        raise ValueError("Cannot evaluate an empty ledger")
    returns = pd.to_numeric(ledger["daily_return"], errors="raise").astype(float)
    equity = pd.to_numeric(ledger["equity"], errors="raise").astype(float)
    observations = int(len(ledger))
    years = observations / 252.0
    total_return = float(equity.iloc[-1] / initial_cash - 1.0)
    annualized_return = (
        float((equity.iloc[-1] / initial_cash) ** (1.0 / years) - 1.0)
        if years > 0 and equity.iloc[-1] > 0
        else 0.0
    )
    volatility = float(returns.std(ddof=1) * math.sqrt(252.0)) if observations > 1 else 0.0
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0))
        if observations > 1 and returns.std(ddof=1) > 0
        else 0.0
    )
    downside = returns.clip(upper=0.0)
    downside_deviation = float(math.sqrt(float((downside**2).mean())) * math.sqrt(252.0))
    sortino = (
        float(returns.mean() * 252.0 / downside_deviation)
        if downside_deviation > 0
        else 0.0
    )
    drawdown = _drawdown_details(ledger, initial_cash)
    calmar = (
        float(annualized_return / abs(drawdown["max_drawdown"]))
        if drawdown["max_drawdown"] < 0
        else 0.0
    )
    best_index = int(returns.idxmax())
    worst_index = int(returns.idxmin())
    active_turnover = ledger.loc[ledger["turnover"] > 1e-12, "turnover"]
    post_fill_exposures = pd.to_numeric(
        ledger["new_exposure_after_fill"], errors="raise"
    ).astype(float)
    holding_exposures = pd.to_numeric(
        ledger["holding_exposure_for_return"], errors="raise"
    ).astype(float)
    requested_exposures = pd.to_numeric(ledger["target_exposure"], errors="raise").astype(float)
    realized_max_exposure = float(max(post_fill_exposures.max(), holding_exposures.max()))
    fill_dates = pd.to_datetime(ledger["fill_date"])
    monthly_returns: Dict[str, float] = {}
    for month, month_returns in returns.groupby(fill_dates.dt.to_period("M")):
        monthly_returns[str(month)] = float((1.0 + month_returns).prod() - 1.0)
    return {
        "observations": observations,
        "initial_equity": float(initial_cash),
        "final_equity": float(equity.iloc[-1]),
        "minimum_equity": float(equity.min()),
        "minimum_return_from_initial": float(equity.min() / initial_cash - 1.0),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": sharpe,
        "downside_deviation": downside_deviation,
        "sortino_zero_cash_rate": sortino,
        "calmar": calmar,
        **drawdown,
        "best_daily_return": float(returns.loc[best_index]),
        "best_daily_date": str(ledger.loc[best_index, "fill_date"]),
        "best_daily_monetary_pnl": float(ledger.loc[best_index, "monetary_pnl"]),
        "worst_daily_return": float(returns.loc[worst_index]),
        "worst_daily_date": str(ledger.loc[worst_index, "fill_date"]),
        "worst_daily_monetary_pnl": float(ledger.loc[worst_index, "monetary_pnl"]),
        "positive_day_rate": float((returns > 0).mean()),
        "executed_order_count": int(ledger["trade_executed"].sum()),
        "active_trading_days": int((ledger["turnover"] > 1e-12).sum()),
        "total_turnover": float(ledger["turnover"].sum()),
        "average_daily_turnover_all_days": float(ledger["turnover"].mean()),
        "average_turnover_active_days": float(active_turnover.mean()) if not active_turnover.empty else 0.0,
        "mean_exposure": float(post_fill_exposures.mean()),
        "median_exposure": float(post_fill_exposures.median()),
        "maximum_exposure": realized_max_exposure,
        "minimum_exposure": float(min(post_fill_exposures.min(), holding_exposures.min())),
        "maximum_requested_target_exposure": float(requested_exposures.max()),
        "maximum_post_fill_exposure": float(post_fill_exposures.max()),
        "maximum_holding_exposure_for_return": float(holding_exposures.max()),
        "mean_holding_exposure_for_return": float(holding_exposures.mean()),
        "cash_day_rate": float((post_fill_exposures <= 1e-6).mean()),
        "partial_exposure_day_rate": float(
            ((post_fill_exposures > 1e-6) & (post_fill_exposures < 0.999)).mean()
        ),
        "fully_invested_day_rate": float(
            ((post_fill_exposures >= 0.999) & (post_fill_exposures <= 1.001)).mean()
        ),
        "leveraged_day_rate": float((post_fill_exposures > 1.001).mean()),
        "fees": float(ledger["fees"].sum()),
        "slippage": float(ledger["slippage"].sum()),
        "margin_interest": float(ledger["margin_interest"].sum()),
        "nominal_total_costs": float(
            ledger[["fees", "slippage", "margin_interest"]].sum(axis=1).sum()
        ),
        "monthly_returns": monthly_returns,
    }


def compare_ledgers(
    strategy_ledger: pd.DataFrame,
    benchmark_ledger: pd.DataFrame,
    *,
    initial_cash: float = 1000.0,
    success_epsilon: float = 1e-8,
) -> Dict[str, Any]:
    strategy_dates = strategy_ledger["fill_date"].astype(str).tolist()
    benchmark_dates = benchmark_ledger["fill_date"].astype(str).tolist()
    if strategy_dates != benchmark_dates:
        raise ValueError("Strategy and AAPL buy-and-hold ledgers are not date-aligned")
    strategy = performance_metrics(strategy_ledger, initial_cash=initial_cash)
    benchmark = performance_metrics(benchmark_ledger, initial_cash=initial_cash)
    strategy_returns = strategy_ledger["daily_return"].astype(float).to_numpy()
    benchmark_returns = benchmark_ledger["daily_return"].astype(float).to_numpy()
    active = strategy_returns - benchmark_returns
    tracking_error = float(np.std(active, ddof=1) * math.sqrt(252.0)) if len(active) > 1 else 0.0
    information_ratio = (
        float(np.mean(active) / np.std(active, ddof=1) * math.sqrt(252.0))
        if len(active) > 1 and np.std(active, ddof=1) > 0
        else 0.0
    )
    correlation = (
        float(np.corrcoef(strategy_returns, benchmark_returns)[0, 1])
        if len(active) > 1
        and np.std(strategy_returns, ddof=1) > 0
        and np.std(benchmark_returns, ddof=1) > 0
        else 0.0
    )
    excess = float(strategy["total_return"] - benchmark["total_return"])
    relative_wealth = float(
        (1.0 + strategy["total_return"]) / (1.0 + benchmark["total_return"]) - 1.0
    )
    return {
        "strategy": strategy,
        "aapl_buy_hold": benchmark,
        "excess_return_vs_aapl_buy_hold": excess,
        "relative_wealth_vs_aapl_buy_hold": relative_wealth,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "daily_return_correlation": correlation,
        "max_drawdown_difference": float(strategy["max_drawdown"] - benchmark["max_drawdown"]),
        "worst_day_difference": float(strategy["worst_daily_return"] - benchmark["worst_daily_return"]),
        "requested_success": bool(excess > success_epsilon),
        "material_one_basis_point_success": bool(excess > 0.0001),
        "success_epsilon": success_epsilon,
    }


def _git_state(repo_root: Path) -> Dict[str, Any]:
    def command(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": command("rev-parse", "HEAD"),
            "branch": command("branch", "--show-current"),
            "dirty": bool(command("status", "--porcelain")),
        }
    except Exception as exc:
        return {"commit": None, "branch": None, "dirty": None, "error": str(exc)}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _atomic_write_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def terminal_close_sensitivity(
    ledger: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    initial_cash: float = 1000.0,
) -> Dict[str, Any]:
    if ledger.empty:
        raise ValueError("Cannot terminal-mark an empty ledger")
    data = canonical_market_frame(frame)
    last_fill = pd.Timestamp(str(ledger.iloc[-1]["fill_date"]))
    if last_fill not in data.index:
        raise ValueError("Terminal fill date is absent from the market frame")
    terminal_price = float(data.loc[last_fill, "aapl_adj_close"])
    terminal_equity = float(ledger.iloc[-1]["cash"]) + float(ledger.iloc[-1]["shares"]) * terminal_price
    open_equity = float(ledger.iloc[-1]["equity"])
    values = np.r_[float(initial_cash), ledger["equity"].to_numpy(dtype=float), terminal_equity]
    drawdowns = values / np.maximum.accumulate(values) - 1.0
    return {
        "terminal_date": _iso_day(last_fill),
        "terminal_price_basis": "adjusted_close",
        "terminal_adjusted_close": terminal_price,
        "terminal_equity": terminal_equity,
        "terminal_total_return": float(terminal_equity / initial_cash - 1.0),
        "last_session_open_to_close_return": float(terminal_equity / open_equity - 1.0),
        "terminal_curve_max_drawdown": float(drawdowns.min()),
    }


def evaluate_suite(
    frame: pd.DataFrame,
    spec: StrategySpec,
    periods: Sequence[EvaluationPeriod],
    costs: CostAssumptions,
    *,
    initial_cash: float = 1000.0,
    success_epsilon: float = 1e-8,
) -> tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    suite_started = time.perf_counter()
    target = build_target_exposure(frame, spec)
    benchmark_target = build_target_exposure(frame, replace(spec, name="buy_hold"))
    static_leverage_target = pd.Series(1.10, index=canonical_market_frame(frame).index)
    period_reports: Dict[str, Any] = {}
    ledgers: Dict[str, pd.DataFrame] = {}
    for period in periods:
        strategy_ledger = simulate_period(
            frame,
            target,
            period,
            costs,
            initial_cash=initial_cash,
            max_exposure=spec.max_exposure,
            rebalance=spec.rebalance,
        )
        benchmark_ledger = simulate_period(
            frame,
            benchmark_target,
            period,
            costs,
            initial_cash=initial_cash,
            max_exposure=1.0,
            rebalance="daily",
        )
        static_leverage_ledger = simulate_period(
            frame,
            static_leverage_target,
            period,
            costs,
            initial_cash=initial_cash,
            max_exposure=1.10,
            rebalance="daily",
        )
        comparison = compare_ledgers(
            strategy_ledger,
            benchmark_ledger,
            initial_cash=initial_cash,
            success_epsilon=success_epsilon,
        )
        strategy_terminal = terminal_close_sensitivity(
            strategy_ledger,
            frame,
            initial_cash=initial_cash,
        )
        benchmark_terminal = terminal_close_sensitivity(
            benchmark_ledger,
            frame,
            initial_cash=initial_cash,
        )
        terminal_excess = float(
            strategy_terminal["terminal_total_return"]
            - benchmark_terminal["terminal_total_return"]
        )
        comparison["terminal_close_sensitivity"] = {
            "strategy": strategy_terminal,
            "aapl_buy_hold": benchmark_terminal,
            "excess_return_vs_aapl_buy_hold": terminal_excess,
            "requested_success": bool(terminal_excess > success_epsilon),
            "material_one_basis_point_success": bool(terminal_excess > 0.0001),
        }
        static_metrics = performance_metrics(static_leverage_ledger, initial_cash=initial_cash)
        static_terminal = terminal_close_sensitivity(
            static_leverage_ledger,
            frame,
            initial_cash=initial_cash,
        )
        comparison["static_1_10_aapl_benchmark"] = {
            **static_metrics,
            "terminal_close_sensitivity": static_terminal,
            "strategy_excess_return_vs_static_1_10": float(
                comparison["strategy"]["total_return"] - static_metrics["total_return"]
            ),
        }
        # Explicitly remove every friction for the counterfactual.
        zero_cost_ledger = simulate_period(
            frame,
            target,
            period,
            CostAssumptions(slippage_bps=0.0, annual_margin_rate=0.0),
            initial_cash=initial_cash,
            max_exposure=spec.max_exposure,
            rebalance=spec.rebalance,
        )
        zero_cost_metrics = performance_metrics(zero_cost_ledger, initial_cash=initial_cash)
        comparison["zero_cost_counterfactual_return"] = zero_cost_metrics["total_return"]
        comparison["compounded_cost_drag"] = float(
            zero_cost_metrics["total_return"] - comparison["strategy"]["total_return"]
        )
        merged = strategy_ledger.add_prefix("strategy_").copy()
        benchmark_prefixed = benchmark_ledger.add_prefix("buy_hold_")
        static_prefixed = static_leverage_ledger.add_prefix("static_1_10_")
        merged = pd.concat([merged, benchmark_prefixed, static_prefixed], axis=1)
        merged["active_daily_return"] = (
            merged["strategy_daily_return"] - merged["buy_hold_daily_return"]
        )
        period_reports[period.name] = {
            "period": asdict(period),
            **comparison,
        }
        ledgers[period.name] = merged
    chained_relative_wealth = float(
        np.prod(
            [
                1.0 + report["relative_wealth_vs_aapl_buy_hold"]
                for report in period_reports.values()
            ]
        )
        - 1.0
    )
    elapsed = time.perf_counter() - suite_started
    report = {
        "strategy": asdict(spec),
        "strategy_hash": strategy_sha256(spec),
        "costs": asdict(costs),
        "initial_cash": initial_cash,
        "periods": period_reports,
        "all_periods_requested_success": bool(
            all(item["requested_success"] for item in period_reports.values())
        ),
        "all_periods_material_success": bool(
            all(item["material_one_basis_point_success"] for item in period_reports.values())
        ),
        "all_periods_terminal_close_material_success": bool(
            all(
                item["terminal_close_sensitivity"]["material_one_basis_point_success"]
                for item in period_reports.values()
            )
        ),
        "all_periods_open_and_close_material_success": bool(
            all(
                item["material_one_basis_point_success"]
                and item["terminal_close_sensitivity"]["material_one_basis_point_success"]
                for item in period_reports.values()
            )
        ),
        "geometrically_chained_relative_wealth": chained_relative_wealth,
        "evaluation_seconds": elapsed,
        "completed_within_3600_seconds": bool(elapsed <= 3600.0),
        "model_calls": 0,
        "estimated_external_cost_usd": 0.0,
        "api_cost_display": "$0.00",
    }
    return report, ledgers


def historical_selection_audit(
    frame: pd.DataFrame,
    spec: StrategySpec,
    costs: CostAssumptions,
    *,
    first_year: int = 2005,
    last_year: int = 2023,
    initial_cash: float = 1000.0,
) -> Dict[str, Any]:
    if last_year >= 2024:
        raise ValueError("Selection audit must end before the first final evaluation year, 2024")
    periods = [
        EvaluationPeriod(str(year), f"{year}-01-01", f"{year}-12-31")
        for year in range(first_year, last_year + 1)
    ]
    report, _ = evaluate_suite(
        frame,
        spec,
        periods,
        costs,
        initial_cash=initial_cash,
    )
    annual_excess = {
        name: float(item["excess_return_vs_aapl_buy_hold"])
        for name, item in report["periods"].items()
    }
    values = np.asarray(list(annual_excess.values()), dtype=float)
    early = [value for year, value in annual_excess.items() if int(year) <= 2014]
    late = [value for year, value in annual_excess.items() if int(year) >= 2015]
    return {
        "selection_window": {"first_year": first_year, "last_year": last_year},
        "selection_data_cutoff": f"{last_year}-12-31",
        "annual_excess_returns": annual_excess,
        "years_evaluated": int(len(values)),
        "years_beating_buy_hold": int(np.count_nonzero(values > 1e-8)),
        "beat_rate": float(np.mean(values > 1e-8)),
        "early_subperiod_wins": int(np.count_nonzero(np.asarray(early) > 1e-8)),
        "early_subperiod_years": int(len(early)),
        "late_subperiod_wins": int(np.count_nonzero(np.asarray(late) > 1e-8)),
        "late_subperiod_years": int(len(late)),
        "mean_annual_excess": float(np.mean(values)),
        "median_annual_excess": float(np.median(values)),
        "worst_annual_excess": float(np.min(values)),
        "best_annual_excess": float(np.max(values)),
        "selection_rule_data_cutoff": f"{last_year}-12-31",
        "holdout_provenance": (
            "Retrospective development evidence only: the implementation was created after all final "
            "periods occurred, so no pre-2024 commit can prove an untouched holdout."
        ),
        "evidence_classification": "retrospective_historical_fit",
    }


def _periods_through_available_data(
    frame: pd.DataFrame,
    requested: Sequence[EvaluationPeriod],
) -> tuple[EvaluationPeriod, ...]:
    last_day = canonical_market_frame(frame).index.max().date()
    periods: list[EvaluationPeriod] = []
    for period in requested:
        start = pd.Timestamp(period.start).date()
        requested_end = pd.Timestamp(period.end).date()
        if last_day < start:
            raise ValueError(f"No market data is available for required period {period.name}")
        actual_end = min(last_day, requested_end)
        periods.append(EvaluationPeriod(period.name, period.start, actual_end.isoformat()))
    return tuple(periods)


def run_experiment(
    *,
    repo_root: Path,
    output_dir: Path,
    cache_path: Path,
    spec: StrategySpec,
    base_costs: CostAssumptions,
    stress_costs: Mapping[str, CostAssumptions],
    periods: Sequence[EvaluationPeriod] = DEFAULT_PERIODS,
    data_start: str = "1999-01-01",
    data_end: str = "2026-07-09",
    refresh_data: bool = False,
    initial_cash: float = 1000.0,
) -> Dict[str, Any]:
    wall_started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    git_state = _git_state(repo_root)
    spec.validate()
    base_costs.validate()
    for scenario in stress_costs.values():
        scenario.validate()
    frame, data_origin = load_or_download_market_frame(
        cache_path,
        start=data_start,
        end_inclusive=data_end,
        refresh=refresh_data,
    )
    periods = _periods_through_available_data(frame, periods)
    data_hash = market_data_sha256(frame)
    implementation_path = Path(__file__).resolve()
    implementation_hash = file_sha256(implementation_path)
    selection = historical_selection_audit(
        frame,
        spec,
        base_costs,
        initial_cash=initial_cash,
    )
    base_report, base_ledgers = evaluate_suite(
        frame,
        spec,
        periods,
        base_costs,
        initial_cash=initial_cash,
    )
    scenarios: Dict[str, Any] = {"base": base_report}
    scenario_ledgers: Dict[str, Dict[str, pd.DataFrame]] = {"base": base_ledgers}
    for name, scenario_costs in stress_costs.items():
        scenario_report, ledgers = evaluate_suite(
            frame,
            spec,
            periods,
            scenario_costs,
            initial_cash=initial_cash,
        )
        scenarios[name] = scenario_report
        scenario_ledgers[name] = ledgers

    run_id = f"aapl-{spec.name}-{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = run_dir / "market_snapshot.csv"
    _atomic_write_csv(
        frame.reset_index(names="date"),
        snapshot_path,
        index=False,
        float_format="%.17g",
    )
    for scenario_name, period_ledgers in scenario_ledgers.items():
        for period_name, ledger in period_ledgers.items():
            _atomic_write_csv(
                ledger,
                run_dir / f"daily_{scenario_name}_{period_name}.csv",
                index=False,
                float_format="%.12g",
            )
    elapsed = time.perf_counter() - wall_started
    integrity_errors: list[str] = []
    if git_state.get("dirty") is not False:
        integrity_errors.append("dirty_or_unverifiable_git_worktree_at_run_start")
    if not git_state.get("commit"):
        integrity_errors.append("missing_git_commit")
    if market_data_sha256(pd.read_csv(snapshot_path)) != data_hash:
        integrity_errors.append("saved_market_snapshot_hash_mismatch")
    base_open_close_success = bool(base_report["all_periods_open_and_close_material_success"])
    standard_stress_name = "stress_10bps_12pct_margin"
    standard_stress_success = bool(
        scenarios.get(standard_stress_name, {}).get(
            "all_periods_open_and_close_material_success",
            False,
        )
    )
    candidate_hash = hashlib.sha256(
        f"{strategy_sha256(spec)}:{implementation_hash}:{data_hash}".encode("utf-8")
    ).hexdigest()
    report = {
        "run_id": run_id,
        "created_at_utc": started_at.isoformat(),
        "artifact_dir": str(run_dir.resolve()),
        "strategy": asdict(spec),
        "strategy_hash": strategy_sha256(spec),
        "implementation_sha256": implementation_hash,
        "candidate_hash": candidate_hash,
        "data": {
            "source": "Yahoo Finance via yfinance",
            "origin_this_run": data_origin,
            "cache_path": str(cache_path.resolve()),
            "saved_snapshot_path": str(snapshot_path.resolve()),
            "sha256": data_hash,
            "first_observation": _iso_day(frame.index.min()),
            "last_observation": _iso_day(frame.index.max()),
            "observations": int(len(frame)),
            "adjusted_open_formula": "raw_open * adjusted_close / raw_close",
        },
        "execution_contract": {
            "decision_information_cutoff": "completed close on decision_date",
            "fill": "next AAPL trading session adjusted open",
            "valuation": "adjusted open",
            "dividends_and_splits": "implicit in adjusted prices; not separately credited",
            "fractional_shares": True,
            "shorting": False,
            "margin": "negative cash allowed only for declared target exposure above 1.0",
            "cash_interest_rate": 0.0,
        },
        "selection_audit": selection,
        "scenarios": scenarios,
        "promotion": {
            "requested_base_success": bool(base_report["all_periods_requested_success"]),
            "material_base_success": bool(base_report["all_periods_material_success"]),
            "open_and_terminal_close_material_base_success": base_open_close_success,
            "stress_success": {
                name: bool(report_item["all_periods_open_and_close_material_success"])
                for name, report_item in scenarios.items()
                if name != "base"
            },
            "required_standard_stress": standard_stress_name,
            "capital_promotion_success": bool(
                base_open_close_success
                and standard_stress_success
                and not integrity_errors
                and elapsed <= 3600.0
            ),
            "same_strategy_hash_all_periods": True,
            "integrity_errors": integrity_errors,
            "evidence_classification": "retrospective_historical_fit_not_prospective_proof",
        },
        "reproducibility": {
            "git": git_state,
            "random_seed": None,
            "model": None,
            "model_calls": 0,
            "token_counts": 0,
            "estimated_external_cost_usd": 0.0,
            "api_cost_display": "$0.00",
            "command_runtime_seconds": elapsed,
            "completed_within_3600_seconds": bool(elapsed <= 3600.0),
        },
    }
    report_path = run_dir / "report.json"
    _atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    latest_path = output_dir / "latest.json"
    _atomic_write_text(
        latest_path,
        json.dumps(
            {"run_id": run_id, "report": str(report_path.resolve())},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    # Include a complete report and latest-pointer serialization in the
    # promised one-hour wall clock, then rewrite only the final timing fields.
    final_elapsed = time.perf_counter() - wall_started
    report["reproducibility"]["command_runtime_seconds"] = final_elapsed
    report["reproducibility"]["completed_within_3600_seconds"] = bool(final_elapsed <= 3600.0)
    _atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    # Runtime is now final; a run barely crossing the cap during serialization
    # must not retain an earlier promotion decision.
    if final_elapsed > 3600.0:
        report["promotion"]["capital_promotion_success"] = False
        if "runtime_exceeded_3600_seconds" not in report["promotion"]["integrity_errors"]:
            report["promotion"]["integrity_errors"].append("runtime_exceeded_3600_seconds")
        _atomic_write_text(
            report_path,
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        )
    return report


def _strategy_from_args(args: argparse.Namespace) -> StrategySpec:
    return StrategySpec(
        name=args.strategy,
        symbol=args.symbol,
        aapl_sma_days=args.aapl_sma_days,
        spy_sma_days=args.spy_sma_days,
        risk_on_exposure=args.risk_on_exposure,
        risk_off_exposure=args.risk_off_exposure,
        mixed_exposure=args.mixed_exposure,
        rebalance=args.rebalance,
        max_exposure=args.max_exposure,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a causal, zero-API-cost deterministic AAPL strategy suite.",
    )
    parser.add_argument("--strategy", default="trend_regime_tilt_v1")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--aapl-sma-days", type=int, default=150)
    parser.add_argument("--spy-sma-days", type=int, default=200)
    parser.add_argument("--risk-on-exposure", type=float, default=1.10)
    parser.add_argument("--risk-off-exposure", type=float, default=0.975)
    parser.add_argument("--mixed-exposure", type=float, default=1.0)
    parser.add_argument("--max-exposure", type=float, default=1.10)
    parser.add_argument("--rebalance", choices=["daily", "on_target_change"], default="daily")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--annual-margin-rate", type=float, default=0.08)
    parser.add_argument("--initial-cash", type=float, default=1000.0)
    parser.add_argument("--data-start", default="1999-01-01")
    parser.add_argument("--data-end", default="2026-07-09")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--cache-path", default="data/deterministic_aapl/aapl_spy.csv")
    parser.add_argument("--output-dir", default="data/deterministic_aapl/runs")
    parser.add_argument("--no-stress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    spec = _strategy_from_args(args)
    base_costs = CostAssumptions(
        slippage_bps=args.slippage_bps,
        annual_margin_rate=args.annual_margin_rate,
    )
    stresses: Dict[str, CostAssumptions] = {}
    if not args.no_stress:
        stresses = {
            "stress_10bps_12pct_margin": CostAssumptions(
                slippage_bps=10.0,
                annual_margin_rate=0.12,
            ),
            "severe_20bps_12pct_margin": CostAssumptions(
                slippage_bps=20.0,
                annual_margin_rate=0.12,
            ),
        }
    report = run_experiment(
        repo_root=repo_root,
        output_dir=(repo_root / args.output_dir).resolve(),
        cache_path=(repo_root / args.cache_path).resolve(),
        spec=spec,
        base_costs=base_costs,
        stress_costs=stresses,
        data_start=args.data_start,
        data_end=args.data_end,
        refresh_data=args.refresh_data,
        initial_cash=args.initial_cash,
    )
    compact = {
        "run_id": report["run_id"],
        "artifact_dir": report["artifact_dir"],
        "requested_base_success": report["promotion"]["requested_base_success"],
        "material_base_success": report["promotion"]["material_base_success"],
        "open_and_terminal_close_material_base_success": report["promotion"][
            "open_and_terminal_close_material_base_success"
        ],
        "capital_promotion_success": report["promotion"]["capital_promotion_success"],
        "integrity_errors": report["promotion"]["integrity_errors"],
        "stress_success": report["promotion"]["stress_success"],
        "runtime_seconds": report["reproducibility"]["command_runtime_seconds"],
        "external_cost": report["reproducibility"]["api_cost_display"],
        "periods": {
            name: {
                "strategy_return": item["strategy"]["total_return"],
                "buy_hold_return": item["aapl_buy_hold"]["total_return"],
                "excess_return": item["excess_return_vs_aapl_buy_hold"],
                "strategy_max_drawdown": item["strategy"]["max_drawdown"],
                "buy_hold_max_drawdown": item["aapl_buy_hold"]["max_drawdown"],
                "terminal_close_excess_return": item["terminal_close_sensitivity"][
                    "excess_return_vs_aapl_buy_hold"
                ],
                "excess_return_vs_static_1_10": item["static_1_10_aapl_benchmark"][
                    "strategy_excess_return_vs_static_1_10"
                ],
            }
            for name, item in report["scenarios"]["base"]["periods"].items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if report["promotion"]["capital_promotion_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
