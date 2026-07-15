from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.deterministic_aapl import EvaluationPeriod
from agent_benchmark.binary_regime_union_selector import (
    restore_pending_regime_lessons,
    restore_regime_states,
    serialize_pending_regime_lessons,
    serialize_regime_states,
)
import agent_benchmark.binary_regime_longrun_audit as audit


def _market_frame(
    periods: int = 900,
    *,
    start: str = "2004-01-02",
    daily_aapl_return: float = 0.0002,
) -> pd.DataFrame:
    """A wholly synthetic market frame whose rows all predate 2024."""

    index = pd.bdate_range(start, periods=periods)
    assert index.max() < pd.Timestamp("2024-01-01")
    positions = np.arange(periods, dtype=float)
    aapl = 100.0 * np.exp(daily_aapl_return * positions)
    spy = 200.0 * np.exp(0.00015 * positions)
    qqq = 150.0 * np.exp(0.00018 * positions)
    return pd.DataFrame(
        {
            "aapl_open": aapl,
            "aapl_close": aapl,
            "aapl_adj_close": aapl,
            "spy_adj_close": spy,
            "qqq_adj_close": qqq,
        },
        index=index,
    )


def _period(name: str, start: str, end: str) -> EvaluationPeriod:
    return EvaluationPeriod(name, start, end)


def _manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    value["manifest_sha256"] = audit._sha256(audit._canonical_json_bytes(value))
    return value


def _no_leverage_proof() -> dict[str, Any]:
    return {
        "passed": True,
        "maximum_requested_target": 1.0,
        "maximum_post_fill_exposure": 1.0,
        "maximum_holding_exposure": 1.0,
        "minimum_cash": 0.0,
        "minimum_shares": 0.0,
        "total_margin_interest": 0.0,
        "shorting": False,
        "borrowing": False,
    }


def _longrun_period_names() -> list[str]:
    return [str(year) for year in range(2005, 2026)] + ["2026_ytd"]


def _benchmark_period_return(name: str) -> float:
    if name == "2008":
        return -0.30
    return -0.10 if name == "2022" else 0.10


def _policy_equity_path(period_edges: Mapping[str, float]) -> np.ndarray:
    equity = 1000.0
    values: list[float] = []
    for name in _longrun_period_names():
        equity *= (1.0 + _benchmark_period_return(name)) * math.exp(
            float(period_edges[name])
        )
        values.append(equity)
    return np.asarray(values, dtype=float)


def _path_max_drawdown(equity: np.ndarray) -> float:
    values = np.r_[1000.0, np.asarray(equity, dtype=float)]
    return float(np.min(values / np.maximum.accumulate(values) - 1.0))


def _always_long_result() -> dict[str, Any]:
    names = _longrun_period_names()
    edges = {name: 0.0 for name in names}
    equity = _policy_equity_path(edges)
    total_return = float(equity[-1] / 1000.0 - 1.0)
    max_drawdown = _path_max_drawdown(equity)
    return {
        "total_active_log_edge": 0.0,
        "continuous_account_active_log_edge": 0.0,
        "no_leverage_proof": _no_leverage_proof(),
        "comparison": {
            "relative_wealth_vs_aapl_buy_hold": 0.0,
            "strategy": {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
            },
            "aapl_buy_hold": {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
            },
        },
        "periods": {
            name: {
                "active_log_edge": 0.0,
                "strategy_return": _benchmark_period_return(name),
                "aapl_buy_hold_return": _benchmark_period_return(name),
                "ledger_boundary_active_log_edge": 0.0,
            }
            for name in names
        },
    }


def _passing_metrics() -> dict[str, Any]:
    names = _longrun_period_names()
    # The 22 entry-attributed periods add to the 150 complete episode edges.
    edges = {name: 0.03 for name in names}
    edges[names[-1]] = 0.12
    metrics: dict[str, Any] = {}
    for cost_name, _ in audit.COST_SCENARIOS:
        periods = {}
        for name, edge in edges.items():
            benchmark_return = _benchmark_period_return(name)
            strategy_return = (1.0 + benchmark_return) * math.exp(edge) - 1.0
            periods[name] = {
                "active_log_edge": edge,
                "strategy_return": strategy_return,
                "aapl_buy_hold_return": benchmark_return,
                "ledger_boundary_active_log_edge": edge,
            }
        # The strict diagnostic consumes these same continuous-account results.
        for name in ("2024", "2025", "2026_ytd"):
            periods[name]["active_log_edge"] = max(
                float(periods[name]["active_log_edge"]), 0.01
            )
        union_periods = copy.deepcopy(periods)
        union_periods["2026_ytd"]["active_log_edge"] -= 0.10
        union_periods["2026_ytd"]["strategy_return"] = (
            1.0 + union_periods["2026_ytd"]["aapl_buy_hold_return"]
        ) * math.exp(union_periods["2026_ytd"]["active_log_edge"]) - 1.0
        union_periods["2026_ytd"]["ledger_boundary_active_log_edge"] = (
            union_periods["2026_ytd"]["active_log_edge"]
        )
        incremental_periods = {
            name: float(
                periods[name]["active_log_edge"]
                - union_periods[name]["active_log_edge"]
            )
            for name in names
        }
        selector_equity = _policy_equity_path(
            {name: float(periods[name]["ledger_boundary_active_log_edge"]) for name in names}
        )
        union_equity = _policy_equity_path(
            {
                name: float(union_periods[name]["ledger_boundary_active_log_edge"])
                for name in names
            }
        )
        benchmark_equity = _policy_equity_path({name: 0.0 for name in names})
        selector_return = float(selector_equity[-1] / 1000.0 - 1.0)
        union_return = float(union_equity[-1] / 1000.0 - 1.0)
        benchmark_return = float(benchmark_equity[-1] / 1000.0 - 1.0)
        benchmark_drawdown = _path_max_drawdown(benchmark_equity)
        metrics[cost_name] = {
            "selector": {
                "continuous_account_active_log_edge": 0.75,
                "continuous_account_attributed_episode_active_log_edge": 0.75,
                "continuous_account_episode_ledger_identity_error": 0.0,
                "total_active_log_edge": 0.75,
                "attributed_episode_active_log_edge": 0.75,
                "cash_episode_count": 150,
                "no_leverage_proof": _no_leverage_proof(),
                "periods": periods,
                "comparison": {
                    "strategy": {
                        "max_drawdown": _path_max_drawdown(selector_equity),
                        "total_return": selector_return,
                    },
                    "aapl_buy_hold": {
                        "max_drawdown": benchmark_drawdown,
                        "total_return": benchmark_return,
                    },
                },
            },
            "union": {
                "continuous_account_active_log_edge": 0.65,
                "continuous_account_attributed_episode_active_log_edge": 0.65,
                "continuous_account_episode_ledger_identity_error": 0.0,
                "total_active_log_edge": 0.65,
                "attributed_episode_active_log_edge": 0.65,
                "cash_episode_count": len(names),
                "periods": union_periods,
                "comparison": {
                    "strategy": {
                        "total_return": union_return,
                        "max_drawdown": _path_max_drawdown(union_equity),
                    },
                    "aapl_buy_hold": {
                        "total_return": benchmark_return,
                        "max_drawdown": benchmark_drawdown,
                    },
                },
                "no_leverage_proof": _no_leverage_proof(),
            },
            "always_long": _always_long_result(),
            "selector_vs_union": {
                "total_active_log_edge": 0.10,
                "continuous_account_total_active_log_edge": 0.10,
                "periods": incremental_periods,
                "veto_benefit": {
                    "veto_count": 1,
                    "total_veto_benefit": 0.10,
                    "periods": incremental_periods,
                },
                "veto_benefit_identity_error": 0.0,
                "continuous_account_veto_benefit": 0.10,
                "continuous_account_veto_benefit_identity_error": 0.0,
            },
        }
    return metrics


def _passing_episodes() -> dict[str, dict[str, pd.DataFrame]]:
    # 100 positive and 50 negative episodes: 66.7% win rate, positive mean and
    # median, low concentration, and +0.70 after the five largest are removed.
    values = np.asarray([0.01] * 100 + [-0.005] * 50, dtype=float)
    dates = pd.bdate_range("2010-01-04", periods=len(values))
    assert dates.max() < pd.Timestamp("2024-01-01")
    return {
        cost_name: {
            "selector": pd.DataFrame(
                {
                    "entry_date": dates,
                    "net_active_log_edge": values,
                }
            )
        }
        for cost_name, _ in audit.COST_SCENARIOS
    }


def _passing_integrity() -> dict[str, Any]:
    return {
        "passed": True,
        "continuous_account_start": "2005-01-01",
        "account_cooldown_reset_count_after_inception": 0,
        **{name: True for name in audit.REQUIRED_INTEGRITY_TRUE_FIELDS},
    }


def _synthetic_online_artifacts(
    metrics: Mapping[str, Any],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, pd.DataFrame]],
    dict[str, pd.DataFrame],
]:
    ledgers: dict[str, pd.DataFrame] = {}
    episodes: dict[str, dict[str, pd.DataFrame]] = {}
    benefits: dict[str, pd.DataFrame] = {}
    for cost_name, cost_bps in audit.COST_SCENARIOS:
        friction = math.log(
            (1.0 - cost_bps / 10_000.0) / (1.0 + cost_bps / 10_000.0)
        )
        selector_rows: list[dict[str, Any]] = []
        for period_name in _longrun_period_names():
            year = 2026 if period_name == "2026_ytd" else int(period_name)
            expected_count = int(
                metrics[cost_name]["selector"]["cash_episode_count"]
            )
            count = 24 if period_name == "2026_ytd" else 6
            if expected_count == 149 and period_name == "2005":
                count -= 1
            assert expected_count in {149, 150}
            net = float(
                metrics[cost_name]["selector"]["periods"][period_name][
                    "active_log_edge"
                ]
            ) / count
            values = [net] * count
            base = pd.Timestamp(year=year, month=1, day=2)
            for index, net in enumerate(values):
                decision = base + pd.Timedelta(days=3 * index)
                entry = decision + pd.Timedelta(days=1)
                exit_date = decision + pd.Timedelta(days=2)
                selector_rows.append(
                    {
                        "decision_date": decision.date().isoformat(),
                        "entry_date": entry.date().isoformat(),
                        "exit_date": exit_date.date().isoformat(),
                        "raw_active_log_edge": net - friction,
                        "net_active_log_edge": net,
                        "win": net > 0.0,
                    }
                )

        union_rows: list[dict[str, Any]] = []
        for period_name, period in metrics[cost_name]["union"]["periods"].items():
            year = 2026 if period_name == "2026_ytd" else int(period_name)
            entry = pd.Timestamp(
                year=year,
                month=4 if year == 2026 else 7,
                day=16,
            )
            net = float(period["active_log_edge"])
            union_rows.append(
                {
                    "decision_date": (entry - pd.Timedelta(days=1)).date().isoformat(),
                    "entry_date": entry.date().isoformat(),
                    "exit_date": (entry + pd.Timedelta(days=1)).date().isoformat(),
                    "raw_active_log_edge": net - friction,
                    "net_active_log_edge": net,
                    "win": net > 0.0,
                }
            )
        selector_frame = pd.DataFrame(
            selector_rows, columns=audit._selector_runner._EPISODE_COLUMNS
        )
        union_frame = pd.DataFrame(
            union_rows, columns=audit._selector_runner._EPISODE_COLUMNS
        )
        episodes[cost_name] = {
            "selector": selector_frame,
            "union": union_frame,
        }

        net_union = -0.10
        entry = pd.Timestamp("2026-05-06")
        benefits[cost_name] = pd.DataFrame(
            [
                {
                    "decision_date": (entry - pd.Timedelta(days=1)).date().isoformat(),
                    "entry_date": entry.date().isoformat(),
                    "exit_date": (entry + pd.Timedelta(days=1)).date().isoformat(),
                    "raw_union_cash_edge": net_union - friction,
                    "net_union_cash_edge": net_union,
                    "veto_benefit": -net_union,
                    "beneficial_veto": True,
                    "risk_on": True,
                    "spy_return_20": 0.01,
                    "qqq_return_20": 0.01,
                    "selector_regime_n_eff": 10.0,
                    "selector_regime_mean": 0.01,
                    "selector_regime_ready": True,
                    "selector_cash_prediction": False,
                    "selector_skip_prediction": True,
                }
            ],
            columns=audit._selector_runner._VETO_BENEFIT_COLUMNS,
        )

        price_by_date: dict[pd.Timestamp, float] = {}
        fill_dates: set[pd.Timestamp] = set()
        for frame in (selector_frame, union_frame):
            for row in frame.to_dict(orient="records"):
                entry_date = pd.Timestamp(row["entry_date"])
                exit_date = pd.Timestamp(row["exit_date"])
                raw_edge = float(row["raw_active_log_edge"])
                fill_dates.update((entry_date, exit_date))
                price_by_date[entry_date] = 100.0
                price_by_date[exit_date] = 100.0 / math.exp(raw_edge)
        # The real 5/10 bps ledgers share one physical session inventory even
        # when a synthetic failing gate uses one fewer attributed episode.
        for period_name in _longrun_period_names():
            year = 2026 if period_name == "2026_ytd" else int(period_name)
            canonical_count = 24 if period_name == "2026_ytd" else 6
            base = pd.Timestamp(year=year, month=1, day=2)
            for index in range(canonical_count):
                decision = base + pd.Timedelta(days=3 * index)
                fill_dates.update(
                    (
                        decision + pd.Timedelta(days=1),
                        decision + pd.Timedelta(days=2),
                    )
                )
        fill_dates.update(
            pd.Timestamp(f"{year}-12-31") for year in range(2005, 2026)
        )
        fill_dates.add(pd.Timestamp("2026-07-09"))
        fill_dates.add(pd.Timestamp(benefits[cost_name].loc[0, "entry_date"]))
        fill_dates.add(pd.Timestamp(benefits[cost_name].loc[0, "exit_date"]))
        ordered_dates = sorted(fill_dates)

        policy_edges = {
            "selector": {
                name: float(
                    metrics[cost_name]["selector"]["periods"][name][
                        "ledger_boundary_active_log_edge"
                    ]
                )
                for name in _longrun_period_names()
            },
            "union": {
                name: float(
                    metrics[cost_name]["union"]["periods"][name][
                        "ledger_boundary_active_log_edge"
                    ]
                )
                for name in _longrun_period_names()
            },
            "always_long": {name: 0.0 for name in _longrun_period_names()},
            "aapl_buy_hold": {name: 0.0 for name in _longrun_period_names()},
        }
        all_rows: list[dict[str, Any]] = []
        for policy, edges in policy_edges.items():
            period_final_equity = dict(
                zip(
                    _longrun_period_names(),
                    _policy_equity_path(edges),
                    strict=True,
                )
            )
            previous_equity = 1000.0
            previous_shares = 0.0
            previous_post_fill = 0.0
            running_peak = 1000.0
            for row_index, fill_date in enumerate(ordered_dates):
                period_name = (
                    "2026_ytd" if fill_date.year == 2026 else str(fill_date.year)
                )
                is_period_end = fill_date == (
                    pd.Timestamp("2026-07-09")
                    if period_name == "2026_ytd"
                    else pd.Timestamp(f"{period_name}-12-31")
                )
                equity = (
                    float(period_final_equity[period_name])
                    if is_period_end
                    else previous_equity
                )
                adjusted_open = float(price_by_date.get(fill_date, 100.0))
                target = 0.0 if row_index == 0 else 1.0
                cash = equity if target == 0.0 else 0.0
                shares = 0.0 if target == 0.0 else equity / adjusted_open
                delta = shares - previous_shares
                daily_return = equity / previous_equity - 1.0
                running_peak = max(running_peak, equity)
                all_rows.append(
                    {
                        "policy": policy,
                        "ledger_role": (
                            "benchmark"
                            if policy == "aapl_buy_hold"
                            else "strategy"
                        ),
                        "decision_date": (
                            fill_date - pd.Timedelta(days=1)
                        ).date().isoformat(),
                        "fill_date": fill_date.date().isoformat(),
                        "adjusted_open": adjusted_open,
                        "equity_before_fill": equity,
                        "equity": equity,
                        "cash": cash,
                        "shares": shares,
                        "holding_exposure_for_return": previous_post_fill,
                        "target_exposure": target,
                        "new_exposure_after_fill": target,
                        "signed_share_delta": delta,
                        "reference_price": adjusted_open,
                        "fill_price": adjusted_open,
                        "turnover": abs(delta) * adjusted_open / equity,
                        "fees": 0.0,
                        "slippage": 0.0,
                        "margin_interest": 0.0,
                        "trade_executed": bool(abs(delta) > 1e-12),
                        "daily_return": daily_return,
                        "monetary_pnl": equity - previous_equity,
                        "drawdown": equity / running_peak - 1.0,
                    }
                )
                previous_equity = equity
                previous_shares = shares
                previous_post_fill = target
        ledgers[cost_name] = pd.DataFrame(all_rows)
    return ledgers, episodes, benefits


def _gate_name(cost_name: str, suffix: str) -> str:
    return f"{cost_name}_{suffix}"


def _rejected_parent_fixture() -> tuple[
    Path, Path, dict[str, Any], dict[str, bytes], dict[str, Any]
]:
    root = Path("C:/synthetic-repo")
    manifest_path = root / audit.PARENT_MANIFEST_PATH
    dependency_identity = {
        "tracked_dependency_sha256": {
            name: "sha256:" + "1" * 64
            for name in audit.PARENT_DEPENDENCY_PATHS
        },
        "runtime_versions": {"python": "synthetic"},
    }
    embedded = _manifest(
        {
            "contract_version": audit.PARENT_CONTRACT_VERSION,
            "stage": "development",
            "stage_pass": True,
            "run_id": "synthetic-development",
            "git_identity": dependency_identity,
        }
    )
    failures = sorted(audit.EXPECTED_VALIDATION_FAILURES)
    gate = {
        "passed": False,
        "gates": {name: False for name in failures},
        "failures": failures,
    }
    report = {
        "contract_version": audit.PARENT_CONTRACT_VERSION,
        "stage": "validation",
        "run_id": audit.PARENT_RUN_ID,
        "post_2023_outcomes_accessed": False,
        "gate_report": gate,
    }
    payloads = {
        name: audit._pretty_json_bytes({"synthetic": name})
        for name in audit._selector_runner._required_parent_payloads(
            "validation"
        )
    }
    payloads["validation_gate_report.json"] = audit._pretty_json_bytes(gate)
    payloads["report.json"] = audit._pretty_json_bytes(report)
    payloads["authorized_development_manifest.json"] = audit._pretty_json_bytes(
        embedded
    )
    manifest = _manifest(
        {
            "contract_version": audit.PARENT_CONTRACT_VERSION,
            "stage": "validation",
            "stage_pass": False,
            "run_id": audit.PARENT_RUN_ID,
            "payload_sha256": {
                name: audit._sha256(value)
                for name, value in sorted(payloads.items())
            },
            "parent_manifest_sha256": embedded["manifest_sha256"],
            "source_provenance": {
                "bounded_last_date": "2023-12-29",
                "physical_snapshot_has_later_rows": False,
                "rows_after_bound_returned": False,
            },
            "git_identity": dependency_identity,
        }
    )
    return root, manifest_path, manifest, payloads, embedded


def _install_parent_fixture(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    embedded: Mapping[str, Any],
) -> None:
    manifest_bytes = audit._pretty_json_bytes(manifest)
    embedded_bytes = audit._pretty_json_bytes(embedded)

    def fake_committed(
        _root: Path, path: Path, *, description: str
    ) -> bytes:
        del _root, description
        resolved = path.resolve()
        if resolved == manifest_path.resolve():
            return manifest_bytes
        if resolved.name == "stage_manifest.json" and resolved != manifest_path.resolve():
            return embedded_bytes
        if resolved.name in payloads:
            return payloads[resolved.name]
        raise AssertionError(f"unexpected synthetic parent path: {resolved}")

    monkeypatch.setattr(audit, "_committed_local_bytes", fake_committed)
    monkeypatch.setattr(audit, "PARENT_MANIFEST_SHA256", manifest["manifest_sha256"])
    monkeypatch.setattr(
        audit._selector_runner,
        "_validated_prior_manifest",
        lambda **_kwargs: copy.deepcopy(dict(embedded)),
    )


def _adaptive_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int = 10,
    years: int = 2,
    orientation: str = "online_long_frozen_cash",
    crossing: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    index = pd.bdate_range("2019-01-02", "2023-12-29")
    assert index.max() < pd.Timestamp("2024-01-01")
    positions = np.arange(len(index), dtype=float)
    aapl = 100.0 * np.exp(0.0005 * positions)
    frame = pd.DataFrame(
        {
            "aapl_open": aapl,
            "aapl_close": aapl,
            "aapl_adj_close": aapl,
            "spy_adj_close": 200.0 * np.exp(0.0002 * positions),
            "qqq_adj_close": 150.0 * np.exp(0.0003 * positions),
        },
        index=index,
    )
    monkeypatch.setattr(audit, "VALIDATION_END", pd.Timestamp("2020-12-31"))

    def positions_in(year: int, amount: int) -> list[int]:
        candidates = np.flatnonzero(index.year == year)
        return [int(value) for value in candidates[10 : 10 + amount * 5 : 5]]

    if years == 1:
        divergence_positions = positions_in(2021, count)
    else:
        first = count // 2
        divergence_positions = positions_in(2021, first) + positions_in(
            2022, count - first
        )
    online_target = pd.Series(1.0, index=index, dtype=float)
    frozen_target = pd.Series(1.0, index=index, dtype=float)
    union_target = pd.Series(1.0, index=index, dtype=float)
    for position in divergence_positions:
        union_target.iloc[position] = 0.0
        if orientation == "online_long_frozen_cash":
            frozen_target.iloc[position] = 0.0
        elif orientation == "online_cash_frozen_long":
            online_target.iloc[position] = 0.0
        elif orientation == "balanced":
            if len(np.flatnonzero(online_target.eq(0.0))) < count // 2:
                online_target.iloc[position] = 0.0
            else:
                frozen_target.iloc[position] = 0.0
        else:
            raise AssertionError(orientation)

    def forecast(*, online: bool) -> pd.DataFrame:
        value = pd.DataFrame(index=index)
        value["shadow_lesson_added_now"] = False
        value["shadow_matured_signal_risk_on"] = False
        value["shadow_signal_close"] = pd.NaT
        value["shadow_label_10bps"] = np.nan
        for regime in audit.REGIME_NAMES:
            value[f"{regime}_cash_selected"] = False
            value[f"{regime}_mean"] = 0.0
            value[f"{regime}_ready"] = False
            value[f"{regime}_n_raw"] = 20
            value[f"{regime}_n_eff"] = 20.0
        if online and crossing:
            position = int(index.searchsorted(pd.Timestamp("2021-01-08")))
            value.loc[index[position]:, "risk_on_cash_selected"] = True
            value.loc[index[position]:, "risk_on_mean"] = (
                audit.POSITIVE_MEAN_THRESHOLD + 0.01
            )
            value.loc[index[position]:, "risk_on_ready"] = True
            value.iloc[
                position,
                value.columns.get_loc("shadow_lesson_added_now"),
            ] = True
            value.iloc[
                position,
                value.columns.get_loc("shadow_matured_signal_risk_on"),
            ] = True
            value.iloc[
                position, value.columns.get_loc("shadow_signal_close")
            ] = index[position - 2]
            value.iloc[
                position, value.columns.get_loc("shadow_label_10bps")
            ] = 0.01
        value.attrs["synthetic_selector_target"] = (
            online_target.copy() if online else frozen_target.copy()
        )
        value.attrs["synthetic_union_target"] = union_target.copy()
        return value

    online_forecast = forecast(online=True)
    frozen_forecast = forecast(online=False)

    def fake_stage_targets(
        data: pd.DataFrame,
        synthetic_forecast: pd.DataFrame,
        *,
        administrative_start: pd.Timestamp,
    ):
        del administrative_start
        target = synthetic_forecast.attrs["synthetic_selector_target"].reindex(
            data.index
        )
        union = synthetic_forecast.attrs["synthetic_union_target"].reindex(
            data.index
        )
        return {
            "selector": target,
            "union": union,
            "always_long": pd.Series(1.0, index=data.index),
        }, _passing_integrity()

    monkeypatch.setattr(audit._selector_runner, "_stage_targets", fake_stage_targets)

    online_metrics: dict[str, Any] = {}
    frozen_metrics: dict[str, Any] = {}
    adjusted_open = frame["aapl_open"].to_numpy(dtype=float)
    period_names = [str(year) for year in range(2005, 2026)] + ["2026_ytd"]
    for cost_name, cost_bps in audit.COST_SCENARIOS:
        friction = math.log(
            (1.0 - cost_bps / 10_000.0) / (1.0 + cost_bps / 10_000.0)
        )
        observed: list[tuple[int, float]] = []
        for position in divergence_positions:
            cash_edge = math.log(
                adjusted_open[position + 1] / adjusted_open[position + 2]
            ) + friction
            signed = (
                cash_edge if online_target.iloc[position] == 0.0 else -cash_edge
            )
            observed.append((index[position + 1].year, signed))
        period_edges = {
            name: float(
                sum(value for year, value in observed if name == str(year))
            )
            for name in period_names
        }
        online_metrics[cost_name] = {
            "selector": {
                "continuous_account_active_log_edge": float(
                    sum(value for _, value in observed)
                ),
                "periods": {
                    name: {"active_log_edge": edge}
                    for name, edge in period_edges.items()
                },
            }
        }
        frozen_metrics[cost_name] = {
            "selector": {
                "continuous_account_active_log_edge": 0.0,
                "periods": {
                    name: {"active_log_edge": 0.0} for name in period_names
                },
            }
        }
    return frame, online_forecast, frozen_forecast, online_metrics, frozen_metrics


def test_frozen_contract_identity_and_bounds_are_exact():
    assert audit.CONTRACT_VERSION == "aapl-binary-regime-longrun-audit-v1"
    assert audit.PARENT_MANIFEST_SHA256 == (
        "sha256:0354355ac460042f96663d1a45cf5e9a8cf4fe873ddf5cc87afefd9c4b5d81dc"
    )
    assert set(audit.EXPECTED_VALIDATION_FAILURES) == {
            "base_5bps_minimum_two_positive_incremental_years",
            "stress_10bps_minimum_two_positive_incremental_years",
            "stress_10bps_veto_benefit_not_concentrated",
        }
    assert audit.ACCOUNT_START == pd.Timestamp("2005-01-01")
    assert audit.VALIDATION_END == pd.Timestamp("2023-12-31")
    assert audit.AUDIT_END == pd.Timestamp("2026-07-09")
    assert audit.COST_SCENARIOS == (("base_5bps", 5.0), ("stress_10bps", 10.0))
    assert audit.AUDIT_RUN_ID == "binary-regime-longrun-audit-v1"
    assert audit.AUDIT_OUTPUT_PATH == Path("e/binary_regime_longrun_audit_v1")
    assert audit.FINAL_ROWS == 6_875
    assert audit.FINAL_FIRST_SESSION == pd.Timestamp("1999-03-10")
    assert audit.FINAL_DATE_SEQUENCE_SHA256 == (
        "b88df14b4ec60534ace68645ee19c8a0b7d03d0c2c1829a3ad48f8a0a24c9299"
    )
    assert audit.FINAL_BOUNDED_RESULT_SHA256 == (
        "sha256:c01447f975d4a90e49c315f23177f357966363b1ec4790632fa54c0dee250b21"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "nonfinite_continuous_edge",
        "nonfinite_period_edge",
        "missing_period",
        "mutated_period_return",
    ],
)
def test_always_long_control_rejects_nonfinite_or_period_corruption(
    mutation: str,
):
    metrics = _passing_metrics()
    assert audit._always_long_matches(metrics) is True
    control = metrics["stress_10bps"]["always_long"]
    if mutation == "nonfinite_continuous_edge":
        control["continuous_account_active_log_edge"] = math.nan
    elif mutation == "nonfinite_period_edge":
        control["periods"]["2010"]["active_log_edge"] = math.inf
    elif mutation == "missing_period":
        del control["periods"]["2010"]
    else:
        control["periods"]["2010"]["strategy_return"] += 0.01
    assert audit._always_long_matches(metrics) is False


def test_strict_recent_history_requires_all_six_cells_and_exact_ytd_label():
    report = audit.apply_strict_recent_history_gates(
        _passing_metrics(), _passing_integrity()
    )
    assert report["passed"] is True
    assert report["strict_recent_history_pass"] is True
    assert report["failures"] == []
    assert report["evidence_classification"] == (
        "repeated_historical_target_non_confirmatory"
    )
    assert report["ytd_label"] == "2026 YTD through 2026-07-09"


@pytest.mark.parametrize("cost_name", ["base_5bps", "stress_10bps"])
@pytest.mark.parametrize("period_name", ["2024", "2025", "2026_ytd"])
def test_strict_recent_history_point_001_is_a_failure(
    cost_name: str, period_name: str
):
    metrics = _passing_metrics()
    metrics[cost_name]["selector"]["periods"][period_name][
        "active_log_edge"
    ] = 0.001
    report = audit.apply_strict_recent_history_gates(metrics, _passing_integrity())
    key = f"{cost_name}_{period_name}_active_log_edge_above_001"
    assert report["passed"] is False
    assert report["gates"][key] is False
    assert key in report["failures"]


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf])
def test_strict_recent_history_missing_or_nonfinite_fails_closed(bad: Any):
    metrics = _passing_metrics()
    metrics["stress_10bps"]["selector"]["periods"]["2026_ytd"][
        "active_log_edge"
    ] = bad
    report = audit.apply_strict_recent_history_gates(metrics, _passing_integrity())
    assert report["strict_recent_history_pass"] is False


def test_post_hoc_long_run_passing_case_discloses_every_gate_input():
    report = audit.apply_post_hoc_long_run_gates(
        _passing_metrics(), _passing_episodes(), _passing_integrity()
    )
    assert report["passed"] is True
    assert report["post_hoc_long_run_robustness_pass"] is True
    assert report["post_hoc_non_confirmatory"] is True
    assert set(report["gate_inputs"]) == {
        "base_5bps",
        "stress_10bps",
    }
    for inputs in report["gate_inputs"].values():
        assert inputs["reporting_period_count"] == 22
        assert inputs["positive_period_count"] == 22
        assert inputs["complete_cash_episode_count"] == 150
        assert inputs["beneficial_episode_rate"] == pytest.approx(2.0 / 3.0)
        assert inputs["active_log_edge_after_five_largest_episodes"] > 0.0
        assert inputs["strategy_max_drawdown"] >= inputs["aapl_max_drawdown"]


def test_post_hoc_long_run_requires_all_22_periods():
    metrics = _passing_metrics()
    del metrics["base_5bps"]["selector"]["periods"]["2005"]
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    key = _gate_name("base_5bps", "minimum_12_strictly_positive_periods")
    assert report["gates"][key] is False


@pytest.mark.parametrize("cost_name", ["base_5bps", "stress_10bps"])
def test_post_hoc_total_edge_must_be_strictly_above_point_001(cost_name: str):
    metrics = _passing_metrics()
    metrics[cost_name]["selector"]["continuous_account_active_log_edge"] = 0.001
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    assert report["gates"][
        _gate_name(cost_name, "total_active_log_edge_above_001")
    ] is False


def test_post_hoc_requires_twelve_strictly_positive_periods():
    metrics = _passing_metrics()
    periods = metrics["stress_10bps"]["selector"]["periods"]
    for name in list(periods)[11:]:
        periods[name]["active_log_edge"] = 0.0
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    key = _gate_name("stress_10bps", "minimum_12_strictly_positive_periods")
    assert report["gates"][key] is False
    assert report["gate_inputs"]["stress_10bps"]["positive_period_count"] == 11


def test_post_hoc_exactly_twelve_strictly_positive_periods_passes_that_gate():
    metrics = _passing_metrics()
    periods = metrics["stress_10bps"]["selector"]["periods"]
    for name in list(periods)[12:]:
        periods[name]["active_log_edge"] = 0.0
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    key = _gate_name("stress_10bps", "minimum_12_strictly_positive_periods")
    assert report["gates"][key] is True
    assert report["gate_inputs"]["stress_10bps"]["positive_period_count"] == 12


def test_post_hoc_must_survive_removing_best_period():
    metrics = _passing_metrics()
    selector = metrics["base_5bps"]["selector"]
    for period in selector["periods"].values():
        period["active_log_edge"] = 0.0
    selector["periods"]["2005"]["active_log_edge"] = 0.75
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    assert report["gates"][
        _gate_name(
            "base_5bps", "positive_after_subtracting_largest_period"
        )
    ] is False


def test_post_hoc_negative_aapl_period_aggregate_is_strict_and_nonempty():
    metrics = _passing_metrics()
    periods = metrics["base_5bps"]["selector"]["periods"]
    for period in periods.values():
        period["aapl_buy_hold_return"] = 0.0
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    key = _gate_name(
        "base_5bps", "positive_negative_aapl_period_aggregate"
    )
    assert report["gates"][key] is False


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf])
def test_post_hoc_missing_or_nonfinite_aapl_period_return_fails_closed(bad: Any):
    metrics = _passing_metrics()
    metrics["stress_10bps"]["selector"]["periods"]["2010"][
        "aapl_buy_hold_return"
    ] = bad
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    assert report["post_hoc_long_run_robustness_pass"] is False


def test_post_hoc_requires_150_complete_count_matching_ledger():
    metrics = _passing_metrics()
    episodes = _passing_episodes()
    episodes["stress_10bps"]["selector"] = episodes["stress_10bps"][
        "selector"
    ].iloc[:149]
    metrics["stress_10bps"]["selector"]["cash_episode_count"] = 149
    report = audit.apply_post_hoc_long_run_gates(
        metrics, episodes, _passing_integrity()
    )
    assert report["gates"][
        _gate_name(
            "stress_10bps", "minimum_150_complete_cash_episodes"
        )
    ] is False


def test_post_hoc_beneficial_rate_counts_only_strictly_positive_edges():
    episodes = _passing_episodes()
    values = np.asarray([0.01] * 74 + [0.0] * 26 + [-0.001] * 50)
    episodes["base_5bps"]["selector"]["net_active_log_edge"] = values
    report = audit.apply_post_hoc_long_run_gates(
        _passing_metrics(), episodes, _passing_integrity()
    )
    assert report["gates"][
        _gate_name(
            "base_5bps", "beneficial_episode_rate_at_least_50pct"
        )
    ] is False


@pytest.mark.parametrize(
    ("values", "gate_suffix"),
    [
        (np.asarray([0.01] * 75 + [-0.01] * 75), "positive_mean_episode_edge"),
        (np.asarray([0.01] * 75 + [-0.01] * 75), "positive_median_episode_edge"),
    ],
)
def test_post_hoc_episode_location_statistics_are_strictly_positive(
    values: np.ndarray, gate_suffix: str
):
    episodes = _passing_episodes()
    episodes["stress_10bps"]["selector"]["net_active_log_edge"] = values
    report = audit.apply_post_hoc_long_run_gates(
        _passing_metrics(), episodes, _passing_integrity()
    )
    assert report["gates"][_gate_name("stress_10bps", gate_suffix)] is False


def test_post_hoc_must_survive_five_largest_complete_episodes():
    metrics = _passing_metrics()
    metrics["base_5bps"]["selector"]["continuous_account_active_log_edge"] = 0.05
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    assert report["gates"][
        _gate_name(
            "base_5bps",
            "positive_after_subtracting_five_largest_episodes",
        )
    ] is False


def test_post_hoc_positive_episode_concentration_cap():
    episodes = _passing_episodes()
    values = np.asarray([0.26] + [0.74 / 99.0] * 99 + [-0.001] * 50)
    episodes["stress_10bps"]["selector"]["net_active_log_edge"] = values
    report = audit.apply_post_hoc_long_run_gates(
        _passing_metrics(), episodes, _passing_integrity()
    )
    assert report["gates"][
        _gate_name(
            "stress_10bps", "no_episode_above_25pct_positive_edge"
        )
    ] is False


@pytest.mark.parametrize(
    ("strategy_drawdown", "expected"),
    [(-0.30, True), (-0.20, True), (-0.3000001, False)],
)
def test_post_hoc_drawdown_direction_uses_negative_values(
    strategy_drawdown: float, expected: bool
):
    metrics = _passing_metrics()
    metrics["stress_10bps"]["selector"]["comparison"]["strategy"][
        "max_drawdown"
    ] = strategy_drawdown
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    assert report["gates"][
        _gate_name("stress_10bps", "max_drawdown_no_worse_than_aapl")
    ] is expected


@pytest.mark.parametrize("cost_name", ["base_5bps", "stress_10bps"])
def test_post_hoc_selector_must_beat_fixed_union_at_each_cost(cost_name: str):
    metrics = _passing_metrics()
    metrics[cost_name]["selector_vs_union"][
        "continuous_account_total_active_log_edge"
    ] = 0.0
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    assert report["gates"][
        _gate_name(
            cost_name, "selector_minus_fixed_union_strictly_positive"
        )
    ] is False
    assert report["post_hoc_long_run_robustness_pass"] is False


@pytest.mark.parametrize(
    "case",
    ["after_best_period", "negative_aapl_aggregate", "selector_vs_union"],
)
def test_post_hoc_zero_comparisons_use_one_e_minus_12_tolerance(case: str):
    metrics = _passing_metrics()
    selector = metrics["base_5bps"]["selector"]
    if case == "after_best_period":
        for period in selector["periods"].values():
            period["active_log_edge"] = 0.0
        selector["periods"]["2005"]["active_log_edge"] = (
            selector["continuous_account_active_log_edge"] - 0.5e-12
        )
        suffix = "positive_after_subtracting_largest_period"
    elif case == "negative_aapl_aggregate":
        selector["periods"]["2008"]["active_log_edge"] = 0.25e-12
        selector["periods"]["2022"]["active_log_edge"] = 0.25e-12
        suffix = "positive_negative_aapl_period_aggregate"
    else:
        metrics["base_5bps"]["selector_vs_union"][
            "continuous_account_total_active_log_edge"
        ] = 0.5e-12
        suffix = "selector_minus_fixed_union_strictly_positive"
    report = audit.apply_post_hoc_long_run_gates(
        metrics, _passing_episodes(), _passing_integrity()
    )
    assert report["gates"][_gate_name("base_5bps", suffix)] is False


def test_post_hoc_empty_and_nonfinite_inputs_fail_closed():
    empty = audit.apply_post_hoc_long_run_gates({}, {}, {})
    assert empty["passed"] is False
    metrics = _passing_metrics()
    episodes = _passing_episodes()
    episodes["base_5bps"]["selector"].loc[0, "net_active_log_edge"] = math.nan
    report = audit.apply_post_hoc_long_run_gates(
        metrics, episodes, _passing_integrity()
    )
    assert report["post_hoc_long_run_robustness_pass"] is False


def test_exact_rejected_parent_is_authorized_without_laundering_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    root, path, manifest, payloads, embedded = _rejected_parent_fixture()
    _install_parent_fixture(
        monkeypatch, root, path, manifest, payloads, embedded
    )
    result = audit.validate_rejected_validation_parent(
        repo_root=root, path=path
    )
    assert result["stage_pass"] is False
    assert result["manifest_sha256"] == manifest["manifest_sha256"]


def test_rejected_parent_requires_exact_self_hash(
    monkeypatch: pytest.MonkeyPatch,
):
    root, path, manifest, payloads, embedded = _rejected_parent_fixture()
    expected = manifest["manifest_sha256"]
    manifest["manifest_sha256"] = "sha256:" + "0" * 64
    _install_parent_fixture(
        monkeypatch, root, path, manifest, payloads, embedded
    )
    monkeypatch.setattr(audit, "PARENT_MANIFEST_SHA256", expected)
    with pytest.raises(audit.BinaryRegimeLongrunAuditError, match="self-hash"):
        audit.validate_rejected_validation_parent(repo_root=root, path=path)


@pytest.mark.parametrize("mutation", ["missing", "extra", "passed", "report"])
def test_rejected_parent_requires_exact_three_failures_and_report_equality(
    monkeypatch: pytest.MonkeyPatch, mutation: str
):
    root, path, manifest, payloads, embedded = _rejected_parent_fixture()
    gate = json.loads(payloads["validation_gate_report.json"])
    report = json.loads(payloads["report.json"])
    if mutation == "missing":
        removed = gate["failures"].pop()
        gate["gates"].pop(removed)
    elif mutation == "extra":
        gate["failures"].append("synthetic_extra_failure")
        gate["gates"]["synthetic_extra_failure"] = False
    elif mutation == "passed":
        gate["passed"] = True
    else:
        report["run_id"] = "different-run"
    if mutation != "report":
        report["gate_report"] = gate
    payloads["validation_gate_report.json"] = audit._pretty_json_bytes(gate)
    payloads["report.json"] = audit._pretty_json_bytes(report)
    manifest["payload_sha256"] = {
        name: audit._sha256(value) for name, value in sorted(payloads.items())
    }
    manifest.pop("manifest_sha256")
    manifest = _manifest(manifest)
    _install_parent_fixture(
        monkeypatch, root, path, manifest, payloads, embedded
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError,
        match="exact three gate failures disagree",
    ):
        audit.validate_rejected_validation_parent(repo_root=root, path=path)


def test_rejected_parent_payload_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    root, path, manifest, payloads, embedded = _rejected_parent_fixture()
    payloads["validation_online_metrics.json"] += b"tamper"
    _install_parent_fixture(
        monkeypatch, root, path, manifest, payloads, embedded
    )
    with pytest.raises(audit.BinaryRegimeLongrunAuditError, match="checksum changed"):
        audit.validate_rejected_validation_parent(repo_root=root, path=path)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_rejected_parent_payload_inventory_must_be_exact(
    monkeypatch: pytest.MonkeyPatch, mutation: str
):
    root, path, manifest, payloads, embedded = _rejected_parent_fixture()
    if mutation == "missing":
        payloads.pop("validation_online_metrics.json")
    else:
        payloads["synthetic_extra.json"] = b"{}\n"
    manifest["payload_sha256"] = {
        name: audit._sha256(value) for name, value in sorted(payloads.items())
    }
    manifest.pop("manifest_sha256")
    manifest = _manifest(manifest)
    _install_parent_fixture(
        monkeypatch, root, path, manifest, payloads, embedded
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="inventory is not exact"
    ):
        audit.validate_rejected_validation_parent(repo_root=root, path=path)


def test_rejected_parent_requires_passing_embedded_development(
    monkeypatch: pytest.MonkeyPatch,
):
    root, path, manifest, payloads, embedded = _rejected_parent_fixture()
    embedded["stage_pass"] = False
    embedded.pop("manifest_sha256")
    embedded = _manifest(embedded)
    payloads["authorized_development_manifest.json"] = audit._pretty_json_bytes(
        embedded
    )
    manifest["parent_manifest_sha256"] = embedded["manifest_sha256"]
    manifest["payload_sha256"] = {
        name: audit._sha256(value) for name, value in sorted(payloads.items())
    }
    manifest.pop("manifest_sha256")
    manifest = _manifest(manifest)
    _install_parent_fixture(
        monkeypatch, root, path, manifest, payloads, embedded
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="passing parent"
    ):
        audit.validate_rejected_validation_parent(repo_root=root, path=path)


def test_rejected_parent_cannot_contain_post_2023_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    root, path, manifest, payloads, embedded = _rejected_parent_fixture()
    manifest["source_provenance"]["physical_snapshot_has_later_rows"] = True
    manifest.pop("manifest_sha256")
    manifest = _manifest(manifest)
    _install_parent_fixture(
        monkeypatch, root, path, manifest, payloads, embedded
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="post-2023 rows"
    ):
        audit.validate_rejected_validation_parent(repo_root=root, path=path)


def test_dependency_continuity_binds_parent_runtime_and_new_audit_files():
    required_current = {
        *(path.as_posix() for path in audit.IMPLEMENTATION_PATHS),
        audit.CONTRACT_PATH.as_posix(),
    }
    # The current identity must include the inherited parent dependencies as
    # well as the new audit runner and contract.
    current_hashes = {
        name: "sha256:" + "1" * 64
        for name in required_current | set(audit.PARENT_DEPENDENCY_PATHS)
    }
    parent_hashes = {
        name: current_hashes[name] for name in audit.PARENT_DEPENDENCY_PATHS
    }
    current = {
        "tracked_dependency_sha256": current_hashes,
        "runtime_versions": {"python": "synthetic"},
    }
    parent = {
        "git_identity": {
            "tracked_dependency_sha256": parent_hashes,
            "runtime_versions": {"python": "synthetic"},
        }
    }
    audit._require_dependency_continuity(current, parent)
    assert audit.CONTRACT_PATH.as_posix() in current_hashes
    assert Path(audit.__file__).name in {
        Path(name).name for name in current_hashes
    }

    changed = copy.deepcopy(current)
    changed["runtime_versions"]["python"] = "changed"
    with pytest.raises(audit.BinaryRegimeLongrunAuditError, match="dependency|runtime"):
        audit._require_dependency_continuity(changed, parent)

    changed_hash = copy.deepcopy(current)
    changed_hash["tracked_dependency_sha256"][next(iter(parent_hashes))] = (
        "sha256:" + "9" * 64
    )
    with pytest.raises(audit.BinaryRegimeLongrunAuditError, match="dependency|runtime"):
        audit._require_dependency_continuity(changed_hash, parent)


def test_clean_git_identity_hashes_committed_blobs_and_requires_pushed_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = Path("runner.py")
    contract = Path("contract.md")
    price = Path("prices.csv")
    price_object_id = "b" * 40
    (tmp_path / runner).write_bytes(b"runner\r\n")
    (tmp_path / contract).write_bytes(b"contract\r\n")
    committed = {runner.as_posix(): b"runner\n", contract.as_posix(): b"contract\n"}

    def fake_git_text(_root: Path, *args: str) -> str:
        return {
            ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()),
            (
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                ".",
                f":(top,exclude,literal){price.as_posix()}",
            ): "",
            ("symbolic-ref", "--quiet", "--short", "HEAD"): "codex/synthetic",
            ("rev-parse", "HEAD"): "a" * 40,
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "origin/codex/synthetic",
            ("rev-parse", "@{upstream}"): "a" * 40,
            ("rev-parse", f"HEAD:{price.as_posix()}"): price_object_id,
            (
                "ls-files",
                "--stage",
                "--",
                f":(top,literal){price.as_posix()}",
            ): f"100644 {price_object_id} 0\t{price.as_posix()}",
        }[args]

    def fake_git_bytes(_root: Path, *args: str) -> bytes:
        assert price.as_posix() not in " ".join(args)
        if args[0] == "ls-files":
            return (args[-1] + "\n").encode()
        return committed[args[1].removeprefix("HEAD:")]

    monkeypatch.setattr(audit, "IMPLEMENTATION_PATHS", (runner,))
    monkeypatch.setattr(audit, "CONTRACT_PATH", contract)
    monkeypatch.setattr(audit, "_git_text", fake_git_text)
    monkeypatch.setattr(audit, "_git_bytes", fake_git_bytes)
    identity = audit._clean_git_identity(tmp_path, tmp_path / price)
    assert identity["head_equals_upstream"] is True
    assert identity["tracked_dependency_sha256"][runner.as_posix()] == audit._sha256(
        b"runner\n"
    )
    assert identity["tracked_dependency_sha256"][runner.as_posix()] != audit._sha256(
        b"runner\r\n"
    )
    assert identity["pre_lock_final_input_head_index_equal"] is True
    assert identity["final_input_verified_clean_after_attempt_lock"] is False

    original = audit._git_text

    def unpushed(root: Path, *args: str) -> str:
        if args == ("rev-parse", "@{upstream}"):
            return "b" * 40
        return original(root, *args)

    monkeypatch.setattr(audit, "_git_text", unpushed)
    with pytest.raises(audit.BinaryRegimeLongrunAuditError, match="pushed upstream"):
        audit._clean_git_identity(tmp_path, tmp_path / price)


def test_post_lock_input_identity_binds_head_index_local_and_clean_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    price = tmp_path / "prices.csv"
    payload = b"synthetic prices\n"
    price.write_bytes(payload)
    object_id = "c" * 40

    def fake_git_text(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD:prices.csv"):
            return object_id
        if args == (
            "ls-files",
            "--stage",
            "--",
            ":(top,literal)prices.csv",
        ):
            return f"100644 {object_id} 0\tprices.csv"
        if args == (
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ":(top,literal)prices.csv",
        ):
            return ""
        raise AssertionError(args)

    def fake_git_bytes(_root: Path, *args: str) -> bytes:
        if args in (("show", "HEAD:prices.csv"), ("show", ":prices.csv")):
            return payload
        raise AssertionError(args)

    monkeypatch.setattr(audit, "_git_text", fake_git_text)
    monkeypatch.setattr(audit, "_git_bytes", fake_git_bytes)
    identity = audit._tracked_input_identity(tmp_path, price)
    assert identity["head_index_local_equal"] is True
    assert identity["path_status_clean"] is True
    assert identity["verified_after_attempt_lock"] is True
    assert identity["sha256"] == audit._sha256(payload)

    def changed_index(_root: Path, *args: str) -> bytes:
        if args == ("show", ":prices.csv"):
            return b"changed index\n"
        return fake_git_bytes(_root, *args)

    monkeypatch.setattr(audit, "_git_bytes", changed_index)
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="HEAD, index, local"
    ):
        audit._tracked_input_identity(tmp_path, price)


@pytest.mark.parametrize(
    ("orientation", "expected_status", "demonstrated"),
    [
        ("online_long_frozen_cash", "exercised_positive", True),
        ("online_cash_frozen_long", "exercised_negative", False),
        ("balanced", "exercised_flat", False),
    ],
)
def test_adaptive_status_sign_and_exact_xor_orientation(
    monkeypatch: pytest.MonkeyPatch,
    orientation: str,
    expected_status: str,
    demonstrated: bool,
):
    frame, online, frozen, online_metrics, frozen_metrics = _adaptive_fixture(
        monkeypatch, orientation=orientation
    )
    diagnostic, differences, crossings, runs = audit._adaptive_value_analysis(
        frame, online, frozen, online_metrics, frozen_metrics
    )
    assert diagnostic["adaptive_status"] == expected_status
    assert diagnostic["online_learning_historical_value_demonstrated"] is demonstrated
    assert diagnostic["reliable_prospective_evidence"] is False
    assert diagnostic["state_threshold_crossing_count"] == 1
    assert len(crossings) == 1
    assert crossings.iloc[0]["transition"] == "LONG_TO_CASH"
    assert diagnostic["complete_xor_differing_episode_count"] == 10
    assert diagnostic["distinct_entry_year_count"] == 2
    assert diagnostic["sufficient_exposure"] is True
    assert diagnostic["divergence_run_count"] == len(runs) == 10
    allowed = {
        "online_cash_frozen_long",
        "online_long_frozen_cash",
    }
    for cost_name, _ in audit.COST_SCENARIOS:
        assert set(differences[cost_name]["orientation"]) <= allowed
        reconciliation = diagnostic["reconciliations"][cost_name]
        assert abs(reconciliation["full_account_identity_error"]) <= 1e-10
        assert all(
            abs(value) <= 1e-10
            for value in reconciliation["period_identity_errors"].values()
        )


def test_adaptive_xor_charges_both_legs_at_each_cost(
    monkeypatch: pytest.MonkeyPatch,
):
    frame, online, frozen, online_metrics, frozen_metrics = _adaptive_fixture(
        monkeypatch, count=1, years=1, orientation="online_cash_frozen_long"
    )
    _, differences, _, _ = audit._adaptive_value_analysis(
        frame, online, frozen, online_metrics, frozen_metrics
    )
    base = differences["base_5bps"].iloc[0]
    stress = differences["stress_10bps"].iloc[0]
    expected_delta = math.log((0.999 / 1.001) / (0.9995 / 1.0005))
    assert stress["net_cash_edge"] - base["net_cash_edge"] == pytest.approx(
        expected_delta
    )
    assert stress["online_minus_frozen_active_log_edge"] == pytest.approx(
        stress["net_cash_edge"]
    )


@pytest.mark.parametrize(
    ("count", "years"), [(9, 2), (10, 1)]
)
def test_adaptive_exposure_requires_ten_episodes_and_two_entry_years(
    monkeypatch: pytest.MonkeyPatch, count: int, years: int
):
    args = _adaptive_fixture(
        monkeypatch,
        count=count,
        years=years,
        orientation="online_long_frozen_cash",
    )
    diagnostic = audit._adaptive_value_diagnostic(*args)
    assert diagnostic["adaptive_status"] == "exercised_insufficient_evidence"
    assert diagnostic["sufficient_exposure"] is False
    assert "not been demonstrated" in diagnostic["conclusion"]


@pytest.mark.parametrize(
    ("count", "crossing"), [(10, False), (0, True)]
)
def test_adaptive_is_unexercised_without_crossing_or_action_difference(
    monkeypatch: pytest.MonkeyPatch, count: int, crossing: bool
):
    args = _adaptive_fixture(
        monkeypatch,
        count=count,
        years=2,
        orientation="online_long_frozen_cash",
        crossing=crossing,
    )
    diagnostic = audit._adaptive_value_diagnostic(*args)
    assert diagnostic["adaptive_status"] == "unexercised"
    assert diagnostic["online_learning_historical_value_demonstrated"] is False
    assert "not been demonstrated" in diagnostic["conclusion"]


def test_adaptive_rejects_noncausal_latch_transition(
    monkeypatch: pytest.MonkeyPatch,
):
    frame, online, frozen, online_metrics, frozen_metrics = _adaptive_fixture(
        monkeypatch
    )
    online["shadow_lesson_added_now"] = False
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError,
        match="not a causal threshold crossing",
    ):
        audit._adaptive_value_diagnostic(
            frame, online, frozen, online_metrics, frozen_metrics
        )


def test_adaptive_full_and_period_difference_identity_is_gate_bearing(
    monkeypatch: pytest.MonkeyPatch,
):
    frame, online, frozen, online_metrics, frozen_metrics = _adaptive_fixture(
        monkeypatch
    )
    online_metrics["stress_10bps"]["selector"][
        "continuous_account_active_log_edge"
    ] += 2e-10
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="does not reconcile"
    ):
        audit._adaptive_value_diagnostic(
            frame, online, frozen, online_metrics, frozen_metrics
        )


@pytest.mark.parametrize(
    "orientation", ["online_long_frozen_cash", "online_cash_frozen_long"]
)
def test_adaptive_exact_plus_or_minus_tolerance_is_flat(
    monkeypatch: pytest.MonkeyPatch, orientation: str
):
    args = _adaptive_fixture(monkeypatch, orientation=orientation)
    initial = audit._adaptive_value_diagnostic(*args)
    observed = abs(initial["stress_10bps_incremental_active_log_edge"])
    assert observed > 0.0
    monkeypatch.setattr(audit, "ZERO_TOLERANCE", observed)
    report = audit._adaptive_value_diagnostic(*args)
    assert report["adaptive_status"] == "exercised_flat"


def test_adaptive_nonfinite_incremental_edge_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    frame, online, frozen, online_metrics, frozen_metrics = _adaptive_fixture(
        monkeypatch
    )
    online_metrics["stress_10bps"]["selector"][
        "continuous_account_active_log_edge"
    ] = math.nan
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="nonfinite|reconcile"
    ):
        audit._adaptive_value_diagnostic(
            frame, online, frozen, online_metrics, frozen_metrics
        )


def test_adaptive_entry_open_year_controls_period_attribution(
    monkeypatch: pytest.MonkeyPatch,
):
    frame, online, frozen, online_metrics, frozen_metrics = _adaptive_fixture(
        monkeypatch, count=1, years=1
    )
    target_position = int(frame.index.searchsorted(pd.Timestamp("2021-12-31")))
    # The decision on Dec 31 enters at the next session's open, so it belongs
    # to 2022 even though the decision itself is in 2021.
    online_target = pd.Series(1.0, index=frame.index)
    frozen_target = pd.Series(1.0, index=frame.index)
    union_target = pd.Series(1.0, index=frame.index)
    frozen_target.iloc[target_position] = 0.0
    union_target.iloc[target_position] = 0.0
    online.attrs["synthetic_selector_target"] = online_target
    online.attrs["synthetic_union_target"] = union_target
    frozen.attrs["synthetic_selector_target"] = frozen_target
    frozen.attrs["synthetic_union_target"] = union_target
    opens = frame["aapl_open"].to_numpy(dtype=float)
    for cost_name, cost_bps in audit.COST_SCENARIOS:
        edge = -(
            math.log(opens[target_position + 1] / opens[target_position + 2])
            + math.log(
                (1.0 - cost_bps / 10_000.0)
                / (1.0 + cost_bps / 10_000.0)
            )
        )
        online_metrics[cost_name]["selector"][
            "continuous_account_active_log_edge"
        ] = edge
        for period in online_metrics[cost_name]["selector"]["periods"].values():
            period["active_log_edge"] = 0.0
        online_metrics[cost_name]["selector"]["periods"]["2022"][
            "active_log_edge"
        ] = edge
    _, differences, _, _ = audit._adaptive_value_analysis(
        frame, online, frozen, online_metrics, frozen_metrics
    )
    assert differences["stress_10bps"].iloc[0]["decision_date"] == "2021-12-31"
    assert differences["stress_10bps"].iloc[0]["entry_period"] == "2022"


def test_continuous_account_starts_once_and_ledgers_stay_unleveraged():
    frame = _market_frame(900)
    target = pd.Series(1.0, index=frame.index)
    signal = int(frame.index.searchsorted(pd.Timestamp("2006-06-01")))
    target.iloc[signal] = 0.0
    period = _period(
        "continuous_synthetic",
        "2005-01-01",
        frame.index[-1].date().isoformat(),
    )
    result, strategy, benchmark, episodes = (
        audit._selector_runner._continuous_policy_result(
            frame, target, periods=(period,), cost_bps=10.0
        )
    )
    assert result["account_period"]["start"] == "2005-01-01"
    assert result["account_inception_reset_count"] == 0
    assert result["cash_episode_count"] == len(episodes) == 1
    assert abs(result["continuous_account_episode_ledger_identity_error"]) <= 1e-10
    assert result["no_leverage_proof"]["passed"] is True
    assert strategy["cash"].min() >= -1e-10
    assert benchmark["cash"].min() >= -1e-10


def test_cross_year_cash_episode_is_attributed_by_entry_open():
    frame = _market_frame(800)
    target = pd.Series(1.0, index=frame.index)
    decision = int(frame.index.searchsorted(pd.Timestamp("2005-12-30")))
    assert frame.index[decision] == pd.Timestamp("2005-12-30")
    assert frame.index[decision + 1].year == 2006
    target.iloc[decision] = 0.0
    periods = (
        _period("2005", "2005-01-01", "2005-12-31"),
        _period("2006_plus", "2006-01-01", frame.index[-1].date().isoformat()),
    )
    result, _, _, episodes = audit._selector_runner._continuous_policy_result(
        frame, target, periods=periods, cost_bps=5.0
    )
    assert len(episodes) == 1
    assert pd.Timestamp(episodes.iloc[0]["entry_date"]).year == 2006
    assert result["periods"]["2005"]["active_log_edge"] == pytest.approx(0.0)
    assert result["periods"]["2006_plus"]["active_log_edge"] == pytest.approx(
        result["total_active_log_edge"]
    )


def test_online_and_frozen_comparators_must_be_identical():
    online = _passing_metrics()
    frozen = copy.deepcopy(online)
    gates = audit._require_comparator_identity(
        online, frozen, _passing_integrity(), _passing_integrity()
    )
    assert all(gates.values())
    frozen["stress_10bps"]["union"][
        "continuous_account_active_log_edge"
    ] += 0.001
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="comparator identity"
    ):
        audit._require_comparator_identity(
            online, frozen, _passing_integrity(), _passing_integrity()
        )


def test_final_checkpoint_preserves_all_continuation_state():
    frame = _market_frame(200)
    forecast = audit.build_binary_regime_union_selector_forecast(
        frame, learning_mode=audit.CAUSAL_ONLINE_MODE
    )
    checkpoint = audit._selector_runner._checkpoint_from_forecast(
        frame,
        forecast,
        cutoff=frame.index[-1],
        learning_mode=audit.CAUSAL_ONLINE_MODE,
    )
    assert checkpoint["account_start"] == "2005-01-01"
    assert checkpoint["account_reset_count_after_inception"] == 0
    assert set(checkpoint["serialized_regime_states"]["states"]) == set(
        audit.REGIME_NAMES
    )
    assert "lessons" in checkpoint["serialized_pending_shadow_lessons"]
    assert isinstance(checkpoint["pending_shadow_opportunities"], list)
    assert isinstance(checkpoint["account_trailing_cooldown_context"], list)
    assert len(checkpoint["account_trailing_cooldown_context"]) <= 2

    restored_states = restore_regime_states(
        checkpoint["serialized_regime_states"]
    )
    assert serialize_regime_states(restored_states) == (
        checkpoint["serialized_regime_states"]
    )
    restored_pending = restore_pending_regime_lessons(
        checkpoint["serialized_pending_shadow_lessons"]
    )
    assert serialize_pending_regime_lessons(restored_pending) == checkpoint[
        "serialized_pending_shadow_lessons"
    ]


def test_final_input_identity_can_be_verified_with_synthetic_pre2024_bounds(
    monkeypatch: pytest.MonkeyPatch,
):
    frame = _market_frame(100, start="2010-01-04")
    first = frame.index.min()
    last = frame.index.max()
    monkeypatch.setattr(audit, "FINAL_ROWS", len(frame))
    monkeypatch.setattr(audit, "FINAL_FIRST_SESSION", first)
    monkeypatch.setattr(audit, "AUDIT_END", last)
    monkeypatch.setattr(audit, "FINAL_DATE_SEQUENCE_SHA256", "synthetic-dates")
    monkeypatch.setattr(
        audit, "FINAL_BOUNDED_RESULT_SHA256", "sha256:" + "2" * 64
    )
    provenance = {
        "bounded_rows": len(frame),
        "bounded_first_date": first.date().isoformat(),
        "bounded_last_date": last.date().isoformat(),
        "bounded_result_sha256": "sha256:" + "2" * 64,
        "expected_bounded_result_sha256": "sha256:" + "2" * 64,
        "session_coverage": {"date_sequence_sha256": "synthetic-dates"},
        "physical_snapshot_has_later_rows": False,
        "rows_after_bound_returned": False,
    }
    audit._require_final_input_identity(frame, provenance)

    invalid = frame.copy()
    invalid.iloc[0, invalid.columns.get_loc("aapl_open")] = math.nan
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="nonfinite|nonpositive"
    ):
        audit._require_final_input_identity(invalid, provenance)


def _known_pre2024_gate_report() -> dict[str, Any]:
    failures = [
        "base_5bps_minimum_150_complete_cash_episodes",
        "stress_10bps_minimum_150_complete_cash_episodes",
    ]
    return {
        "passed": False,
        "failures": failures,
        "gates": {name: False for name in failures},
        "gate_inputs": {
            "base_5bps": {
                "positive_period_count": 16,
                "complete_cash_episode_count": 137,
                "beneficial_episode_rate": 0.642,
                "total_active_log_edge": 1.2767,
            },
            "stress_10bps": {
                "positive_period_count": 14,
                "complete_cash_episode_count": 137,
                "beneficial_episode_rate": 0.620,
                "total_active_log_edge": 1.1397,
            },
        },
    }


def test_pre2024_disclosure_reproduces_known_status_and_failed_episode_gate():
    disclosure = audit._pre_2024_disclosure(_known_pre2024_gate_report())
    assert disclosure["known_before_new_2024_2026_audit"] is True
    assert disclosure[
        "post_hoc_thresholds_selected_after_this_history_was_known"
    ] is True
    assert set(disclosure["gate_report"]["failures"]) == {
        "base_5bps_minimum_150_complete_cash_episodes",
        "stress_10bps_minimum_150_complete_cash_episodes",
    }
    assert disclosure["summary"]["base_5bps"]["positive_period_count"] == 16
    assert disclosure["summary"]["stress_10bps"]["positive_period_count"] == 14
    assert disclosure["summary"]["base_5bps"][
        "beneficial_episode_rate_percent_rounded_1dp"
    ] == 64.2
    assert disclosure["summary"]["stress_10bps"][
        "beneficial_episode_rate_percent_rounded_1dp"
    ] == 62.0


@pytest.mark.parametrize(
    ("cost_name", "field", "replacement"),
    [
        ("base_5bps", "positive_period_count", 15),
        ("stress_10bps", "complete_cash_episode_count", 138),
        ("base_5bps", "beneficial_episode_rate", 0.641),
        ("stress_10bps", "total_active_log_edge", 1.1396),
    ],
)
def test_pre2024_disclosure_fails_if_known_history_changes(
    cost_name: str, field: str, replacement: Any
):
    report = _known_pre2024_gate_report()
    report["gate_inputs"][cost_name][field] = replacement
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="known gate inputs changed"
    ):
        audit._pre_2024_disclosure(report)


@pytest.mark.parametrize("mutation", ["passed", "gate_value", "extra_failure"])
def test_pre2024_disclosure_requires_exact_failed_gate_status(mutation: str):
    report = _known_pre2024_gate_report()
    if mutation == "passed":
        report["passed"] = True
    elif mutation == "gate_value":
        report["gates"][report["failures"][0]] = True
    else:
        report["failures"].append("synthetic_extra")
        report["gates"]["synthetic_extra"] = False
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="episode-count gate|gate status"
    ):
        audit._pre_2024_disclosure(report)


def test_invalid_rejected_parent_precedes_all_final_input_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    events: list[str] = []
    monkeypatch.setattr(
        audit,
        "_require_frozen_run_destination",
        lambda **_kwargs: audit.AUDIT_RUN_ID,
    )
    monkeypatch.setattr(
        audit,
        "_clean_git_identity",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("Git identity must not run before parent rejection")
        ),
    )

    def reject_parent(**_kwargs):
        events.append("parent")
        raise audit.BinaryRegimeLongrunAuditError("synthetic rejected parent")

    monkeypatch.setattr(audit, "_validated_rejected_validation_parent", reject_parent)
    monkeypatch.setattr(
        audit,
        "load_bounded_prices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("price loader must not run")
        ),
    )
    monkeypatch.setattr(
        audit,
        "_tracked_input_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("raw input must not be read")
        ),
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="synthetic rejected parent"
    ):
        audit.run_longrun_audit(
            repo_root=tmp_path,
            price_artifact=tmp_path / "must-not-open.csv",
            validation_manifest=tmp_path / "bad-parent.json",
            output_dir=tmp_path / "out",
        )
    assert events == ["parent"]


def test_one_run_destination_is_frozen_and_retry_proof(tmp_path: Path):
    expected_output = tmp_path / audit.AUDIT_OUTPUT_PATH
    assert audit._require_frozen_run_destination(
        repo_root=tmp_path,
        output_dir=expected_output,
        run_id=None,
    ) == audit.AUDIT_RUN_ID
    assert audit._require_frozen_run_destination(
        repo_root=tmp_path,
        output_dir=expected_output,
        run_id=audit.AUDIT_RUN_ID,
    ) == audit.AUDIT_RUN_ID

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="alternate|retry"
    ):
        audit._require_frozen_run_destination(
            repo_root=tmp_path,
            output_dir=tmp_path / "different-output",
            run_id=audit.AUDIT_RUN_ID,
        )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="alternate|retry"
    ):
        audit._require_frozen_run_destination(
            repo_root=tmp_path,
            output_dir=expected_output,
            run_id="binary-regime-longrun-audit-retry",
        )

    run_dir = expected_output / audit.AUDIT_RUN_ID
    run_dir.mkdir(parents=True)
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="already exists"
    ):
        audit._require_frozen_run_destination(
            repo_root=tmp_path,
            output_dir=expected_output,
            run_id=audit.AUDIT_RUN_ID,
        )


def test_stale_sealing_attempt_blocks_the_only_audit_run(tmp_path: Path):
    expected_output = tmp_path / audit.AUDIT_OUTPUT_PATH
    expected_output.mkdir(parents=True)
    stale = expected_output / f".{audit.AUDIT_RUN_ID}.synthetic.sealing"
    stale.mkdir()
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="stale sealing"
    ):
        audit._require_frozen_run_destination(
            repo_root=tmp_path,
            output_dir=expected_output,
            run_id=audit.AUDIT_RUN_ID,
        )


def test_short_local_stale_sealing_attempt_blocks_the_only_audit_run(
    tmp_path: Path,
):
    expected_output = tmp_path / audit.AUDIT_OUTPUT_PATH
    expected_output.mkdir(parents=True)
    (expected_output / ".lr-deadbeef0000.sealing").mkdir()
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="stale sealing"
    ):
        audit._require_frozen_run_destination(
            repo_root=tmp_path,
            output_dir=expected_output,
            run_id=audit.AUDIT_RUN_ID,
        )


def test_deep_output_path_fails_before_attempt_lock(tmp_path: Path):
    deep_output = tmp_path / ("deep" * 20) / audit.AUDIT_OUTPUT_PATH
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="too deep|path"
    ):
        audit._require_seal_path_feasibility(
            output_dir=deep_output,
            path_limit_chars=180,
        )
    assert not (deep_output / audit.ATTEMPT_LOCK_FILENAME).exists()


def test_persistent_attempt_lock_is_exclusive_durable_and_hash_bound(
    tmp_path: Path,
):
    output = tmp_path / audit.AUDIT_OUTPUT_PATH
    parent = {
        "manifest_sha256": audit.PARENT_MANIFEST_SHA256,
        "bounded_result_sha256": "sha256:" + "2" * 64,
    }
    git_identity = {"commit": "a" * 40}
    prefix = {"bounded_result_sha256": parent["bounded_result_sha256"]}
    pre_status = {"known_before_new_2024_2026_audit": True}

    identity, payload = audit._create_persistent_attempt_lock(
        repo_root=tmp_path,
        output_dir=output,
        parent=parent,
        git_identity=git_identity,
        prefix_provenance=prefix,
        pre_2024_gate_status=pre_status,
    )
    lock_path = output / audit.ATTEMPT_LOCK_FILENAME
    assert lock_path.read_bytes() == payload
    assert identity["sha256"] == audit._sha256(payload)
    assert identity["path"] == (
        audit.AUDIT_OUTPUT_PATH / audit.ATTEMPT_LOCK_FILENAME
    ).as_posix()
    assert json.loads(payload)["pre_2024_gate_status_sha256"] == audit._sha256(
        audit._canonical_json_bytes(pre_status)
    )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="already been consumed"
    ):
        audit._create_persistent_attempt_lock(
            repo_root=tmp_path,
            output_dir=output,
            parent=parent,
            git_identity=git_identity,
            prefix_provenance=prefix,
            pre_2024_gate_status=pre_status,
        )
    assert lock_path.read_bytes() == payload
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="already attempted"
    ):
        audit._require_frozen_run_destination(
            repo_root=tmp_path,
            output_dir=output,
            run_id=audit.AUDIT_RUN_ID,
        )


def test_existing_run_blocks_before_any_price_input_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    events: list[str] = []
    output = tmp_path / audit.AUDIT_OUTPUT_PATH
    (output / audit.AUDIT_RUN_ID).mkdir(parents=True)
    monkeypatch.setattr(audit, "_clean_git_identity", lambda _root, _price: {})
    monkeypatch.setattr(
        audit,
        "_validated_rejected_validation_parent",
        lambda **_kwargs: events.append("parent") or {},
    )
    monkeypatch.setattr(
        audit,
        "_require_dependency_continuity",
        lambda *_args: events.append("dependency"),
    )
    monkeypatch.setattr(
        audit,
        "load_bounded_prices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no price input may be opened")
        ),
    )
    monkeypatch.setattr(
        audit,
        "_tracked_input_identity",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("raw price blob may not be opened")
        ),
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="already exists"
    ):
        audit.run_longrun_audit(
            repo_root=tmp_path,
            price_artifact=tmp_path / "must-not-open.csv",
            validation_manifest=tmp_path / "synthetic-parent.json",
            output_dir=output,
            run_id=audit.AUDIT_RUN_ID,
        )
    assert events == ["parent"]


def _install_loader_order_stubs(
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame,
    events: list[tuple[str, Any]],
) -> None:
    monkeypatch.setattr(
        audit,
        "_require_frozen_run_destination",
        lambda **_kwargs: audit.AUDIT_RUN_ID,
    )
    monkeypatch.setattr(audit, "_clean_git_identity", lambda _root, _price: {})
    monkeypatch.setattr(
        audit,
        "_validated_rejected_validation_parent",
        lambda **_kwargs: {"manifest_sha256": "sha256:" + "1" * 64},
    )
    monkeypatch.setattr(audit, "_require_dependency_continuity", lambda *_args: None)

    def loader(
        _path: Path, *, end: pd.Timestamp, required_last_session: pd.Timestamp
    ):
        del required_last_session
        events.append(("load", pd.Timestamp(end)))
        return frame, {
            "source_path": "synthetic.csv",
            "bounded_result_sha256": "sha256:" + "2" * 64,
        }

    monkeypatch.setattr(audit, "load_bounded_prices", loader)
    monkeypatch.setattr(
        audit._selector_runner,
        "_require_source_continuity",
        lambda *_args, **_kwargs: events.append(("source", None)),
    )
    monkeypatch.setattr(
        audit,
        "build_binary_regime_union_selector_forecast",
        lambda value, **_kwargs: pd.DataFrame(index=value.index),
    )
    monkeypatch.setattr(
        audit._selector_runner,
        "_require_checkpoint_continuity",
        lambda *_args, **_kwargs: events.append(("checkpoint", None)),
    )
    monkeypatch.setattr(
        audit._selector_runner,
        "_evaluate_policy_set",
        lambda *_args, **_kwargs: ({}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        audit,
        "apply_post_hoc_long_run_gates",
        lambda *_args, **_kwargs: _known_pre2024_gate_report(),
    )
    monkeypatch.setattr(
        audit,
        "_pre_2024_disclosure",
        lambda _report: events.append(("pre_gate", None)) or {},
    )
    monkeypatch.setattr(
        audit,
        "_tracked_input_identity",
        lambda *_args: events.append(("raw_identity", None)) or {},
    )
    monkeypatch.setattr(
        audit,
        "_create_persistent_attempt_lock",
        lambda **_kwargs: (
            events.append(("attempt_lock", None))
            or ({"sha256": "sha256:" + "4" * 64}, b"synthetic-lock")
        ),
    )


def test_prefix_and_checkpoint_finish_before_raw_or_later_value_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    frame = _market_frame(80, start="2018-01-02")
    events: list[tuple[str, Any]] = []
    _install_loader_order_stubs(monkeypatch, frame, events)

    class StopAfterLaterLoad(RuntimeError):
        pass

    monkeypatch.setattr(
        audit._selector_runner,
        "_require_exact_physical_stage_bound",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StopAfterLaterLoad("later load reached")
        ),
    )
    with pytest.raises(StopAfterLaterLoad, match="later load reached"):
        audit.run_longrun_audit(
            repo_root=tmp_path,
            price_artifact=tmp_path / "synthetic.csv",
            validation_manifest=tmp_path / "parent.json",
            output_dir=tmp_path / "out",
        )
    assert events == [
        ("load", audit.VALIDATION_END),
        ("source", None),
        ("checkpoint", None),
        ("pre_gate", None),
        ("attempt_lock", None),
        ("raw_identity", None),
        ("load", audit.AUDIT_END),
    ]


def test_checkpoint_failure_prevents_raw_identity_and_full_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    frame = _market_frame(80, start="2018-01-02")
    events: list[tuple[str, Any]] = []
    _install_loader_order_stubs(monkeypatch, frame, events)

    def reject_checkpoint(*_args, **_kwargs):
        events.append(("checkpoint_failure", None))
        raise audit.BinaryRegimeLongrunAuditError("synthetic checkpoint failure")

    monkeypatch.setattr(
        audit._selector_runner, "_require_checkpoint_continuity", reject_checkpoint
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="checkpoint failure"
    ):
        audit.run_longrun_audit(
            repo_root=tmp_path,
            price_artifact=tmp_path / "synthetic.csv",
            validation_manifest=tmp_path / "parent.json",
            output_dir=tmp_path / "out",
        )
    assert events == [
        ("load", audit.VALIDATION_END),
        ("source", None),
        ("checkpoint_failure", None),
    ]


def test_post_lock_failure_leaves_lock_and_public_retry_is_rejected_before_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    frame = _market_frame(80, start="2018-01-02")
    output = tmp_path / audit.AUDIT_OUTPUT_PATH
    price = tmp_path / "synthetic.csv"
    parent = {
        "manifest_sha256": audit.PARENT_MANIFEST_SHA256,
        "bounded_result_sha256": "sha256:" + "2" * 64,
    }
    object_id = "c" * 40
    git_identity = {
        "commit": "a" * 40,
        "final_input_path": "synthetic.csv",
        "final_input_head_object_id": object_id,
        "final_input_index_object_id": object_id,
    }
    loader_calls: list[pd.Timestamp] = []

    monkeypatch.setattr(
        audit, "_validated_rejected_validation_parent", lambda **_kwargs: parent
    )
    monkeypatch.setattr(
        audit, "_clean_git_identity", lambda _root, _price: dict(git_identity)
    )
    monkeypatch.setattr(audit, "_require_dependency_continuity", lambda *_args: None)

    def loader(
        _path: Path, *, end: pd.Timestamp, required_last_session: pd.Timestamp
    ):
        del required_last_session
        loader_calls.append(pd.Timestamp(end))
        if len(loader_calls) != 1:
            raise AssertionError("full loader must not run after forced raw failure")
        return frame, {
            "source_path": "synthetic.csv",
            "bounded_result_sha256": parent["bounded_result_sha256"],
        }

    monkeypatch.setattr(audit, "load_bounded_prices", loader)
    monkeypatch.setattr(
        audit._selector_runner, "_require_source_continuity", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        audit,
        "build_binary_regime_union_selector_forecast",
        lambda value, **_kwargs: pd.DataFrame(index=value.index),
    )
    monkeypatch.setattr(
        audit._selector_runner, "_require_checkpoint_continuity", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        audit._selector_runner,
        "_evaluate_policy_set",
        lambda *_a, **_k: ({}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        audit,
        "apply_post_hoc_long_run_gates",
        lambda *_a, **_k: _known_pre2024_gate_report(),
    )

    class ForcedPostLockFailure(RuntimeError):
        pass

    def fail_raw_identity(_root: Path, _path: Path):
        assert (output / audit.ATTEMPT_LOCK_FILENAME).is_file()
        raise ForcedPostLockFailure("synthetic post-lock raw failure")

    monkeypatch.setattr(audit, "_tracked_input_identity", fail_raw_identity)
    with pytest.raises(ForcedPostLockFailure, match="post-lock raw failure"):
        audit.run_longrun_audit(
            repo_root=tmp_path,
            price_artifact=price,
            validation_manifest=tmp_path / "parent.json",
            output_dir=output,
        )
    lock_path = output / audit.ATTEMPT_LOCK_FILENAME
    persisted = lock_path.read_bytes()
    assert loader_calls == [audit.VALIDATION_END]

    monkeypatch.setattr(
        audit,
        "_clean_git_identity",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("retry must reject before Git")
        ),
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="already attempted"
    ):
        audit.run_longrun_audit(
            repo_root=tmp_path,
            price_artifact=price,
            validation_manifest=tmp_path / "parent.json",
            output_dir=output,
        )
    assert lock_path.read_bytes() == persisted
    assert loader_calls == [audit.VALIDATION_END]


_ADAPTIVE_EPISODE_COLUMNS = (
    "decision_date",
    "entry_date",
    "exit_date",
    "entry_period",
    "orientation",
    "online_target",
    "frozen_target",
    "raw_cash_edge",
    "net_cash_edge",
    "online_minus_frozen_active_log_edge",
)


def _synthetic_adaptive_frames(
    adaptive_status: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    if adaptive_status == "unexercised":
        count, total_edge, crossing_count = 0, 0.0, 0
    elif adaptive_status == "exercised_insufficient_evidence":
        count, total_edge, crossing_count = 1, 0.01, 1
    else:
        count, crossing_count = 10, 1
        total_edge = {
            "exercised_positive": 0.01,
            "exercised_negative": -0.01,
            "exercised_flat": 0.0,
        }[adaptive_status]
    stress_friction = math.log((1.0 - 10.0 / 10_000.0) / (1.0 + 10.0 / 10_000.0))
    stress_signed = total_edge / count if count else 0.0
    raw_edge = stress_signed - stress_friction if count else 0.0
    identity_rows: list[dict[str, Any]] = []
    for position in range(count):
        year = 2024 if count == 1 or position < count // 2 else 2025
        day = position + 4 if year == 2024 else position - count // 2 + 4
        entry = pd.Timestamp(year=year, month=1, day=day)
        identity_rows.append(
            {
                "decision_date": (entry - pd.Timedelta(days=1)).date().isoformat(),
                "entry_date": entry.date().isoformat(),
                "exit_date": (entry + pd.Timedelta(days=1)).date().isoformat(),
                "entry_period": str(year),
                "orientation": "online_cash_frozen_long",
                "online_target": 0.0,
                "frozen_target": 1.0,
                "raw_cash_edge": raw_edge,
            }
        )
    differences: dict[str, pd.DataFrame] = {}
    for cost_name, cost_bps in audit.COST_SCENARIOS:
        friction = math.log(
            (1.0 - cost_bps / 10_000.0) / (1.0 + cost_bps / 10_000.0)
        )
        rows = []
        for identity in identity_rows:
            net = float(identity["raw_cash_edge"]) + friction
            rows.append(
                {
                    **identity,
                    "net_cash_edge": net,
                    "online_minus_frozen_active_log_edge": net,
                }
            )
        differences[cost_name] = pd.DataFrame(
            rows, columns=_ADAPTIVE_EPISODE_COLUMNS
        )
    crossing_columns = (
        "crossing_date",
        "regime",
        "transition",
        "pre_n_raw",
        "pre_n_eff",
        "pre_mean",
        "pre_cash_selected",
        "post_n_raw",
        "post_n_eff",
        "post_mean",
        "post_cash_selected",
        "matured_signal_close",
        "matured_label_10bps",
    )
    divergence_columns = (
        "run_id",
        "first_decision_date",
        "last_decision_date",
        "sessions",
    )
    if crossing_count:
        first_decision = identity_rows[0]["decision_date"]
        last_decision = identity_rows[-1]["decision_date"]
        crossing_date = (
            pd.Timestamp(first_decision) - pd.Timedelta(days=1)
        ).date().isoformat()
        crossings = pd.DataFrame(
            [
                {
                    "crossing_date": crossing_date,
                    "regime": "risk_on",
                    "transition": "LONG_TO_CASH",
                    "pre_n_raw": 9,
                    "pre_n_eff": 9.0,
                    "pre_mean": 0.0,
                    "pre_cash_selected": False,
                    "post_n_raw": 10,
                    "post_n_eff": 10.0,
                    "post_mean": audit.POSITIVE_MEAN_THRESHOLD + 0.1,
                    "post_cash_selected": True,
                    "matured_signal_close": (
                        pd.Timestamp(crossing_date) - pd.Timedelta(days=1)
                    ).date().isoformat(),
                    "matured_label_10bps": 0.01,
                }
            ],
            columns=crossing_columns,
        )
        divergence_runs = pd.DataFrame(
            [
                {
                    "run_id": 1,
                    "first_decision_date": first_decision,
                    "last_decision_date": last_decision,
                    "sessions": count,
                }
            ],
            columns=divergence_columns,
        )
    else:
        crossings = pd.DataFrame(columns=crossing_columns)
        divergence_runs = pd.DataFrame(columns=divergence_columns)
    return differences, crossings, divergence_runs


def _sealed_report(
    *,
    post_hoc: bool = True,
    strict: bool = True,
    adaptive_status: str = "exercised_positive",
) -> dict[str, Any]:
    fixed = post_hoc and strict
    demonstrated = adaptive_status == "exercised_positive"
    online = fixed and demonstrated
    if adaptive_status == "unexercised":
        crossings, action_count, complete, years, edge = 0, 0, 0, [], 0.0
    elif adaptive_status == "exercised_insufficient_evidence":
        crossings, action_count, complete, years, edge = 1, 1, 1, [2024], 0.01
    else:
        crossings, action_count, complete, years = 1, 10, 10, [2024, 2025]
        edge = {
            "exercised_positive": 0.01,
            "exercised_negative": -0.01,
            "exercised_flat": 0.0,
        }[adaptive_status]
    adaptive = {
        "adaptive_status": adaptive_status,
        "state_threshold_crossing_count": crossings,
        "state_threshold_crossings": [{}] * crossings,
        "action_difference_count": action_count,
        "divergence_run_count": crossings,
        "divergence_runs": [{}] * crossings,
        "complete_xor_differing_episode_count": complete,
        "distinct_entry_years": years,
        "distinct_entry_year_count": len(years),
        "sufficient_exposure": complete >= 10 and len(years) >= 2,
        "stress_10bps_incremental_active_log_edge": edge,
        "reconciliations": {name: {} for name, _ in audit.COST_SCENARIOS},
        "xor_orientation_values": [
            "online_cash_frozen_long",
            "online_long_frozen_cash",
        ],
        "online_learning_historical_value_demonstrated": demonstrated,
        "reliable_prospective_evidence": False,
        "conclusion": audit.ADAPTIVE_CONCLUSIONS[adaptive_status],
    }
    online_metrics = _passing_metrics()
    if not strict:
        selector_periods = online_metrics["stress_10bps"]["selector"]["periods"]
        union_periods = online_metrics["stress_10bps"]["union"]["periods"]
        shifted = selector_periods["2026_ytd"]["active_log_edge"] - 0.001
        selector_periods["2026_ytd"]["active_log_edge"] = 0.001
        selector_periods["2025"]["active_log_edge"] += shifted
        union_periods["2026_ytd"]["active_log_edge"] = -0.099
        union_periods["2025"]["active_log_edge"] += shifted
        for policy in ("selector", "union"):
            result = online_metrics["stress_10bps"][policy]
            boundary_edges: dict[str, float] = {}
            for name in _longrun_period_names():
                period = result["periods"][name]
                edge_value = float(period["active_log_edge"])
                benchmark_return = float(period["aapl_buy_hold_return"])
                period["strategy_return"] = (
                    (1.0 + benchmark_return) * math.exp(edge_value) - 1.0
                )
                period["ledger_boundary_active_log_edge"] = edge_value
                boundary_edges[name] = edge_value
            policy_equity = _policy_equity_path(boundary_edges)
            benchmark_equity = _policy_equity_path(
                {name: 0.0 for name in _longrun_period_names()}
            )
            result["comparison"]["strategy"].update(
                {
                    "total_return": float(policy_equity[-1] / 1000.0 - 1.0),
                    "max_drawdown": _path_max_drawdown(policy_equity),
                }
            )
            result["comparison"]["aapl_buy_hold"].update(
                {
                    "total_return": float(
                        benchmark_equity[-1] / 1000.0 - 1.0
                    ),
                    "max_drawdown": _path_max_drawdown(benchmark_equity),
                }
            )
    if not post_hoc:
        online_metrics["stress_10bps"]["selector"]["cash_episode_count"] = 149
    frozen_metrics = copy.deepcopy(online_metrics)
    adaptive_frames, crossing_frame, divergence_frame = (
        _synthetic_adaptive_frames(adaptive_status)
    )
    adaptive["state_threshold_crossings"] = crossing_frame.to_dict(
        orient="records"
    )
    adaptive["state_threshold_crossing_count"] = len(crossing_frame)
    adaptive["divergence_runs"] = divergence_frame.to_dict(orient="records")
    adaptive["divergence_run_count"] = len(divergence_frame)
    period_names = tuple(str(year) for year in range(2005, 2026)) + (
        "2026_ytd",
    )
    reconciliations: dict[str, Any] = {}
    for cost_name, _ in audit.COST_SCENARIOS:
        frame = adaptive_frames[cost_name]
        observed_full = float(
            frame["online_minus_frozen_active_log_edge"].sum()
        )
        period_edges = {
            name: float(
                frame.loc[
                    frame["entry_period"] == name,
                    "online_minus_frozen_active_log_edge",
                ].sum()
            )
            for name in period_names
        }
        frozen_metrics[cost_name]["selector"][
            "continuous_account_active_log_edge"
        ] = (
            online_metrics[cost_name]["selector"][
                "continuous_account_active_log_edge"
            ]
            - observed_full
        )
        period_errors: dict[str, float] = {}
        for name in period_names:
            frozen_metrics[cost_name]["selector"]["periods"][name][
                "active_log_edge"
            ] = (
                online_metrics[cost_name]["selector"]["periods"][name][
                    "active_log_edge"
                ]
                - period_edges[name]
            )
            expected_period = (
                online_metrics[cost_name]["selector"]["periods"][name][
                    "active_log_edge"
                ]
                - frozen_metrics[cost_name]["selector"]["periods"][name][
                    "active_log_edge"
                ]
            )
            period_errors[name] = expected_period - period_edges[name]
        expected_full = (
            online_metrics[cost_name]["selector"][
                "continuous_account_active_log_edge"
            ]
            - frozen_metrics[cost_name]["selector"][
                "continuous_account_active_log_edge"
            ]
        )
        reconciliations[cost_name] = {
            "complete_xor_episode_count": len(frame),
            "full_account_online_minus_frozen_active_log_edge": expected_full,
            "xor_episode_incremental_edge": observed_full,
            "full_account_identity_error": expected_full - observed_full,
            "period_incremental_edges": period_edges,
            "period_identity_errors": period_errors,
            "identity_tolerance": audit.IDENTITY_TOLERANCE,
        }
    adaptive["reconciliations"] = reconciliations
    adaptive["stress_10bps_incremental_active_log_edge"] = reconciliations[
        "stress_10bps"
    ]["xor_episode_incremental_edge"]
    _, online_episodes, _ = _synthetic_online_artifacts(online_metrics)
    online_integrity = _passing_integrity()
    frozen_integrity = _passing_integrity()
    strict_gate = audit.apply_strict_recent_history_gates(
        online_metrics, online_integrity
    )
    long_gate = audit.apply_post_hoc_long_run_gates(
        online_metrics, online_episodes, online_integrity
    )
    assert strict_gate["strict_recent_history_pass"] is strict
    assert long_gate["post_hoc_long_run_robustness_pass"] is post_hoc
    return {
        "contract_version": audit.CONTRACT_VERSION,
        "stage": "post_hoc_long_run_audit",
        "run_id": audit.AUDIT_RUN_ID,
        "evidence_classification": audit.EVIDENCE_CLASSIFICATION,
        "parent_validation_rejection_remains_final": True,
        "parent_validation_failures": list(audit.EXPECTED_VALIDATION_FAILURES),
        "physical_data_end": "2026-07-09",
        "ytd_label": "2026 YTD through 2026-07-09",
        "continuous_account_start": "2005-01-01",
        "continuous_account_reset_count_after_inception": 0,
        "sentiment_inputs": audit.SENTIMENT_DISCLOSURE,
        "pre_2024_gate_status": audit._pre_2024_disclosure(
            _known_pre2024_gate_report()
        ),
        "online_metrics_2005_through_2026_07_09": online_metrics,
        "frozen_2023_metrics_2005_through_2026_07_09": frozen_metrics,
        "policy_comparison": audit._policy_comparison(
            online_metrics, frozen_metrics
        ),
        "online_integrity": online_integrity,
        "frozen_2023_integrity": frozen_integrity,
        "comparator_integrity": audit._require_comparator_identity(
            online_metrics,
            frozen_metrics,
            online_integrity,
            frozen_integrity,
        ),
        "strict_recent_history_gate_report": strict_gate,
        "post_hoc_long_run_gate_report": long_gate,
        "post_hoc_long_run_robustness_pass": post_hoc,
        "strict_recent_history_pass": strict,
        "fixed_policy_candidate_for_prospective_paper": fixed,
        "adaptive_value": adaptive,
        "online_learning_historical_value_demonstrated": demonstrated,
        "online_learning_candidate_for_paper": online,
        "online_arm_role_for_prospective_paper": (
            "learning_candidate" if online else "shadow_challenger"
        ),
        "historical_results_authorize_real_capital": False,
        "runtime": {
            "seconds_before_seal": 1.0,
            "limit_seconds": 3600.0,
            "within_limit": True,
            "network_access": False,
            "news_calls": 0,
            "llm_calls": 0,
            "api_calls": 0,
            "external_cost_usd": 0.0,
        },
    }


class _AcceptDeadline:
    def check(self, _location: str) -> None:
        return None

    def elapsed(self) -> float:
        return 1.0


def _seal_synthetic(
    output_dir: Path,
    report: Mapping[str, Any],
    *,
    run_id: str = audit.AUDIT_RUN_ID,
    deadline: Any | None = None,
    mutate_payloads: Any | None = None,
) -> dict[str, Any]:
    parent_manifest = {
        "manifest_sha256": audit.PARENT_MANIFEST_SHA256,
        "bounded_result_sha256": "sha256:" + "2" * 64,
    }
    price_object_id = "c" * 40
    git_identity = {
        "commit": "a" * 40,
        "pre_lock_worktree_clean_excluding_final_input": True,
        "pre_lock_final_input_head_index_equal": True,
        "final_input_path": "synthetic.csv",
        "final_input_head_object_id": price_object_id,
        "final_input_index_object_id": price_object_id,
        "final_input_verified_clean_after_attempt_lock": True,
    }
    pre_2024 = report["pre_2024_gate_status"]
    lock_content = {
        "schema_version": audit.ATTEMPT_LOCK_SCHEMA_VERSION,
        "contract_version": audit.CONTRACT_VERSION,
        "run_id": audit.AUDIT_RUN_ID,
        "one_run_no_retry": True,
        "persistent_after_success_or_failure": True,
        "created_after_through_2023_checkpoint_replay": True,
        "created_before_raw_blob_or_2024_value_read": True,
        "parent_manifest_sha256": audit.PARENT_MANIFEST_SHA256,
        "git_commit": git_identity["commit"],
        "through_2023_bounded_result_sha256": "sha256:" + "2" * 64,
        "pre_2024_gate_status_sha256": audit._sha256(
            audit._canonical_json_bytes(pre_2024)
        ),
    }
    lock_bytes = audit._pretty_json_bytes(lock_content)
    attempt_lock = {
        "path": (audit.AUDIT_OUTPUT_PATH / audit.ATTEMPT_LOCK_FILENAME).as_posix(),
        "sha256": audit._sha256(lock_bytes),
        "content": lock_content,
        "left_in_place_for_commit": True,
    }
    sealed_report = dict(report)
    sealed_report["attempt_lock"] = attempt_lock
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / audit.ATTEMPT_LOCK_FILENAME).write_bytes(lock_bytes)
    source_provenance = {
        "source_path": "synthetic.csv",
        "bounded_result_sha256": audit.FINAL_BOUNDED_RESULT_SHA256,
        "expected_bounded_result_sha256": audit.FINAL_BOUNDED_RESULT_SHA256,
        "bounded_rows": audit.FINAL_ROWS,
        "bounded_first_date": "1999-03-10",
        "bounded_last_date": "2026-07-09",
        "physical_snapshot_has_later_rows": False,
        "rows_after_bound_returned": False,
        "tracked_input": {
            "path": "synthetic.csv",
            "sha256": "sha256:" + "9" * 64,
            "head_blob_sha256": "sha256:" + "9" * 64,
            "index_blob_sha256": "sha256:" + "9" * 64,
            "local_file_sha256": "sha256:" + "9" * 64,
            "head_object_id": price_object_id,
            "index_object_id": price_object_id,
            "head_index_local_equal": True,
            "path_status_clean": True,
            "verified_after_attempt_lock": True,
        },
        "authorization_preflight": {
            "persistent_attempt_lock_sha256": attempt_lock["sha256"],
            "final_input_verified_clean_after_attempt_lock": True,
        },
    }
    prefix_provenance = {
        "bounded_result_sha256": parent_manifest["bounded_result_sha256"]
    }
    checkpoint = {
        "learning_mode": audit.CAUSAL_ONLINE_MODE,
        "checkpoint_cutoff": "2026-07-09",
        "last_observed_session": "2026-07-09",
        "serialized_regime_states": {},
        "serialized_pending_shadow_lessons": {},
        "account_trailing_cooldown_context": [],
    }
    payloads = {
        name: b"synthetic\n" for name in audit._required_audit_payload_names()
    }
    payloads.update(
        {
            audit.ATTEMPT_LOCK_FILENAME: lock_bytes,
            "authorized_rejected_validation_manifest.json": audit._pretty_json_bytes(
                parent_manifest
            ),
            "input_provenance.json": audit._pretty_json_bytes(source_provenance),
            "prefix_preflight_provenance.json": audit._pretty_json_bytes(
                prefix_provenance
            ),
            "pre_2024_gate_status.json": audit._pretty_json_bytes(pre_2024),
            "pre_2024_metrics.json": audit._pretty_json_bytes({"synthetic": True}),
            "audit_online_metrics.json": audit._pretty_json_bytes(
                sealed_report["online_metrics_2005_through_2026_07_09"]
            ),
            "audit_frozen_2023_metrics.json": audit._pretty_json_bytes(
                sealed_report["frozen_2023_metrics_2005_through_2026_07_09"]
            ),
            "audit_policy_comparison.json": audit._pretty_json_bytes(
                sealed_report["policy_comparison"]
            ),
            "strict_recent_history_gate_report.json": audit._pretty_json_bytes(
                sealed_report["strict_recent_history_gate_report"]
            ),
            "post_hoc_long_run_gate_report.json": audit._pretty_json_bytes(
                sealed_report["post_hoc_long_run_gate_report"]
            ),
            "adaptive_value.json": audit._pretty_json_bytes(
                sealed_report["adaptive_value"]
            ),
            "online_checkpoint_through_2026_07_09.json": audit._pretty_json_bytes(
                checkpoint
            ),
        }
    )
    online_ledgers, online_episodes, online_benefits = _synthetic_online_artifacts(
        sealed_report["online_metrics_2005_through_2026_07_09"]
    )
    for cost_name, _ in audit.COST_SCENARIOS:
        payloads[f"longrun_online_{cost_name}_ledgers.csv"] = (
            audit._frame_csv_bytes(online_ledgers[cost_name])
        )
        payloads[f"longrun_online_{cost_name}_union_episodes.csv"] = (
            audit._frame_csv_bytes(online_episodes[cost_name]["union"])
        )
        payloads[f"longrun_online_{cost_name}_selector_episodes.csv"] = (
            audit._frame_csv_bytes(online_episodes[cost_name]["selector"])
        )
        payloads[f"longrun_online_{cost_name}_veto_benefits.csv"] = (
            audit._frame_csv_bytes(online_benefits[cost_name])
        )
    adaptive_frames, crossings, divergence_runs = _synthetic_adaptive_frames(
        report["adaptive_value"]["adaptive_status"]
    )
    for cost_name, _ in audit.COST_SCENARIOS:
        payloads[f"adaptive_{cost_name}_xor_differing_episodes.csv"] = (
            audit._frame_csv_bytes(adaptive_frames[cost_name])
        )
    payloads["adaptive_threshold_crossings.csv"] = audit._frame_csv_bytes(
        crossings
    )
    payloads["adaptive_divergence_runs.csv"] = audit._frame_csv_bytes(
        divergence_runs
    )
    if mutate_payloads is not None:
        mutate_payloads(payloads)
    return audit._stage_bundle(
        output_dir=output_dir,
        run_id=run_id,
        report=sealed_report,
        payloads=payloads,
        source_provenance=source_provenance,
        git_identity=git_identity,
        parent_manifest=parent_manifest,
        deadline=deadline or _AcceptDeadline(),
    )


def test_sealed_bundle_is_atomic_deterministic_safe_and_zero_cost(tmp_path: Path):
    first = _seal_synthetic(tmp_path / "one", _sealed_report())
    second = _seal_synthetic(tmp_path / "two", _sealed_report())
    first_dir = Path(first["artifact_dir"])
    second_dir = Path(second["artifact_dir"])
    first_manifest = json.loads((first_dir / "stage_manifest.json").read_text())
    second_manifest = json.loads((second_dir / "stage_manifest.json").read_text())
    assert first_manifest == second_manifest
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (first_dir / "checksums.json").read_bytes() == (
        second_dir / "checksums.json"
    ).read_bytes()
    assert first_manifest["stage_pass"] is True
    assert first_manifest["parent_validation_rejection_remains_final"] is True
    execution = first_manifest["execution"]
    assert execution["actions"] == ["LONG_100_PERCENT", "CASH_100_PERCENT"]
    assert execution["account_resets_after_inception"] == 0
    assert execution["maximum_target_exposure"] == pytest.approx(1.0)
    for forbidden in (
        "shorting",
        "leverage",
        "borrowing",
        "negative_cash",
        "cash_interest",
        "margin_interest",
        "network_access",
    ):
        assert execution[forbidden] is False
    assert execution["news_calls"] == 0
    assert execution["llm_calls"] == 0
    assert execution["api_calls"] == 0
    assert execution["estimated_external_cost_usd"] == pytest.approx(0.0)
    assert execution["cost_scenarios_bps_per_changing_leg"] == [5.0, 10.0]
    assert execution["ytd_label"] == "through 2026-07-09"


def test_seal_csv_reparse_preserves_nontrivial_ieee_floats_exactly():
    values = np.asarray(
        [
            np.nextafter(0.1, 1.0),
            np.nextafter(-0.005, -1.0),
            np.nextafter(1.0 / 3.0, 0.0),
        ],
        dtype=np.float64,
    )
    payloads = {
        "synthetic.csv": audit._frame_csv_bytes(
            pd.DataFrame({"net_active_log_edge": values})
        )
    }
    reparsed = audit._csv_payload_frame(payloads, "synthetic.csv")[
        "net_active_log_edge"
    ].to_numpy(dtype=np.float64)
    assert np.array_equal(values.view(np.uint64), reparsed.view(np.uint64))


def test_sealer_rejects_inconsistent_candidate_flags(tmp_path: Path):
    report = _sealed_report(post_hoc=False, strict=True)
    report["fixed_policy_candidate_for_prospective_paper"] = True
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="candidate|inconsistent"
    ):
        _seal_synthetic(tmp_path, report)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("evidence_classification", "prospective_confirmation"),
        ("parent_validation_rejection_remains_final", False),
        ("historical_results_authorize_real_capital", True),
    ],
)
def test_sealer_rejects_changed_evidence_or_capital_classification(
    tmp_path: Path, field: str, replacement: Any
):
    report = _sealed_report()
    report[field] = replacement
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="inconsistent"
    ):
        _seal_synthetic(tmp_path, report)


def test_sealer_rejects_report_payload_cross_binding_tamper(tmp_path: Path):
    def tamper(payloads: dict[str, bytes]) -> None:
        payloads["audit_policy_comparison.json"] = audit._pretty_json_bytes(
            {"tampered": True}
        )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="does not match|contradicts"
    ):
        _seal_synthetic(
            tmp_path,
            _sealed_report(),
            mutate_payloads=tamper,
        )


def test_sealer_rejects_passed_integrity_with_false_invariant(tmp_path: Path):
    report = _sealed_report()
    report["online_integrity"]["all_policy_ledgers_unleveraged"] = False
    assert report["online_integrity"]["passed"] is True
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="comparator identity|integrity"
    ):
        _seal_synthetic(tmp_path, report)


def test_sealer_rejects_shorting_hidden_inside_passing_proof(tmp_path: Path):
    report = _sealed_report()
    report["online_metrics_2005_through_2026_07_09"]["base_5bps"][
        "selector"
    ]["no_leverage_proof"]["shorting"] = True
    report["policy_comparison"] = audit._policy_comparison(
        report["online_metrics_2005_through_2026_07_09"],
        report["frozen_2023_metrics_2005_through_2026_07_09"],
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="comparator identity|leverage"
    ):
        _seal_synthetic(tmp_path, report)


def test_sealer_recomputes_online_frozen_comparator_identity(tmp_path: Path):
    report = _sealed_report()
    report["frozen_2023_metrics_2005_through_2026_07_09"]["base_5bps"][
        "union"
    ]["continuous_account_active_log_edge"] = 123.0
    report["policy_comparison"] = audit._policy_comparison(
        report["online_metrics_2005_through_2026_07_09"],
        report["frozen_2023_metrics_2005_through_2026_07_09"],
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="comparator identity"
    ):
        _seal_synthetic(tmp_path, report)


def test_sealer_recomputes_strict_gate_from_bound_online_metrics(tmp_path: Path):
    report = _sealed_report()
    for cost_name, _ in audit.COST_SCENARIOS:
        report["online_metrics_2005_through_2026_07_09"][cost_name]["selector"][
            "periods"
        ]["2024"]["active_log_edge"] = -9.0
    report["policy_comparison"] = audit._policy_comparison(
        report["online_metrics_2005_through_2026_07_09"],
        report["frozen_2023_metrics_2005_through_2026_07_09"],
    )
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="do not recompute"
    ):
        _seal_synthetic(tmp_path, report)


def test_sealer_recomputes_long_gate_from_bound_episode_csv(tmp_path: Path):
    def tamper(payloads: dict[str, bytes]) -> None:
        episodes = _passing_episodes()["stress_10bps"]["selector"].copy()
        episodes["net_active_log_edge"] = -0.01
        payloads["longrun_online_stress_10bps_selector_episodes.csv"] = (
            audit._frame_csv_bytes(episodes)
        )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="do not recompute"
    ):
        _seal_synthetic(
            tmp_path,
            _sealed_report(),
            mutate_payloads=tamper,
        )


def test_sealer_rejects_adaptive_nested_reconciliation_relabel(tmp_path: Path):
    report = _sealed_report(adaptive_status="exercised_positive")
    report["adaptive_value"]["reconciliations"]["stress_10bps"][
        "xor_episode_incremental_edge"
    ] = -99.0
    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="Adaptive.*reconciliation"
    ):
        _seal_synthetic(tmp_path, report)


def test_sealer_rejects_adaptive_xor_csv_signed_edge_tamper(tmp_path: Path):
    def tamper(payloads: dict[str, bytes]) -> None:
        frames, _, _ = _synthetic_adaptive_frames("exercised_positive")
        frame = frames["stress_10bps"].copy()
        frame.loc[0, "online_minus_frozen_active_log_edge"] = -1.0
        payloads["adaptive_stress_10bps_xor_differing_episodes.csv"] = (
            audit._frame_csv_bytes(frame)
        )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError,
        match="Adaptive.*reconciliation|edge|invalid evidence",
    ):
        _seal_synthetic(
            tmp_path,
            _sealed_report(adaptive_status="exercised_positive"),
            mutate_payloads=tamper,
        )


def test_sealer_rejects_adaptive_xor_orientation_target_contradiction(
    tmp_path: Path,
):
    def tamper(payloads: dict[str, bytes]) -> None:
        frames, _, _ = _synthetic_adaptive_frames("exercised_positive")
        frame = frames["base_5bps"].copy()
        frame.loc[0, "orientation"] = "online_long_frozen_cash"
        payloads["adaptive_base_5bps_xor_differing_episodes.csv"] = (
            audit._frame_csv_bytes(frame)
        )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="invalid evidence"
    ):
        _seal_synthetic(
            tmp_path,
            _sealed_report(adaptive_status="exercised_positive"),
            mutate_payloads=tamper,
        )


def test_sealer_rejects_crossing_csv_that_contradicts_diagnostic(tmp_path: Path):
    def tamper(payloads: dict[str, bytes]) -> None:
        _, crossings, _ = _synthetic_adaptive_frames("exercised_positive")
        crossings.loc[0, "post_mean"] = -99.0
        payloads["adaptive_threshold_crossings.csv"] = audit._frame_csv_bytes(
            crossings
        )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="crossing.*contradicts"
    ):
        _seal_synthetic(
            tmp_path,
            _sealed_report(adaptive_status="exercised_positive"),
            mutate_payloads=tamper,
        )


def test_sealer_rejects_divergence_sessions_that_do_not_match_actions(
    tmp_path: Path,
):
    def tamper(payloads: dict[str, bytes]) -> None:
        _, _, divergence = _synthetic_adaptive_frames("exercised_positive")
        divergence.loc[0, "sessions"] = 9
        payloads["adaptive_divergence_runs.csv"] = audit._frame_csv_bytes(
            divergence
        )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="divergence.*contradicts"
    ):
        _seal_synthetic(
            tmp_path,
            _sealed_report(adaptive_status="exercised_positive"),
            mutate_payloads=tamper,
        )


def test_sealer_rejects_gate_bearing_benchmark_period_ledger_mutation(
    tmp_path: Path,
):
    report = _sealed_report()

    def tamper(payloads: dict[str, bytes]) -> None:
        ledgers, _, _ = _synthetic_online_artifacts(
            report["online_metrics_2005_through_2026_07_09"]
        )
        frame = ledgers["base_5bps"].copy()
        for policy in ("always_long", "aapl_buy_hold"):
            policy_indices = frame.index[frame["policy"] == policy].to_numpy()
            period_end_index = int(
                frame.index[
                    (frame["policy"] == policy)
                    & (frame["fill_date"] == "2008-12-31")
                ][0]
            )
            local_position = int(np.flatnonzero(policy_indices == period_end_index)[0])
            previous_index = int(policy_indices[local_position - 1])
            frame.loc[period_end_index, "equity"] = (
                float(frame.loc[previous_index, "equity"]) * 1.05
            )
            policy_rows = frame.loc[policy_indices]
            equities = policy_rows["equity"].to_numpy(dtype=float)
            previous_equities = np.r_[1000.0, equities[:-1]]
            opens = policy_rows["adjusted_open"].to_numpy(dtype=float)
            targets = policy_rows["target_exposure"].to_numpy(dtype=float)
            shares = np.where(targets == 0.0, 0.0, equities / opens)
            cash = np.where(targets == 0.0, equities, 0.0)
            deltas = np.diff(np.r_[0.0, shares])
            running_peak = np.maximum.accumulate(np.r_[1000.0, equities])[1:]
            frame.loc[policy_indices, "equity_before_fill"] = equities
            frame.loc[policy_indices, "cash"] = cash
            frame.loc[policy_indices, "shares"] = shares
            frame.loc[policy_indices, "signed_share_delta"] = deltas
            frame.loc[policy_indices, "turnover"] = (
                np.abs(deltas) * opens / equities
            )
            frame.loc[policy_indices, "trade_executed"] = np.abs(deltas) > 1e-12
            frame.loc[policy_indices, "daily_return"] = (
                equities / previous_equities - 1.0
            )
            frame.loc[policy_indices, "monetary_pnl"] = (
                equities - previous_equities
            )
            frame.loc[policy_indices, "drawdown"] = equities / running_peak - 1.0
        payloads["longrun_online_base_5bps_ledgers.csv"] = (
            audit._frame_csv_bytes(frame)
        )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="ledger"
    ):
        _seal_synthetic(tmp_path, report, mutate_payloads=tamper)


def test_sealer_rejects_union_episode_edge_mutation(tmp_path: Path):
    report = _sealed_report()

    def tamper(payloads: dict[str, bytes]) -> None:
        _, episodes, _ = _synthetic_online_artifacts(
            report["online_metrics_2005_through_2026_07_09"]
        )
        frame = episodes["stress_10bps"]["union"].copy()
        frame.loc[0, "net_active_log_edge"] += 0.25
        payloads["longrun_online_stress_10bps_union_episodes.csv"] = (
            audit._frame_csv_bytes(frame)
        )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="union episode"
    ):
        _seal_synthetic(tmp_path, report, mutate_payloads=tamper)


def test_sealer_rejects_veto_benefit_mutation(tmp_path: Path):
    report = _sealed_report()

    def tamper(payloads: dict[str, bytes]) -> None:
        _, _, benefits = _synthetic_online_artifacts(
            report["online_metrics_2005_through_2026_07_09"]
        )
        frame = benefits["base_5bps"].copy()
        frame.loc[0, "veto_benefit"] += 0.25
        payloads["longrun_online_base_5bps_veto_benefits.csv"] = (
            audit._frame_csv_bytes(frame)
        )

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="veto"
    ):
        _seal_synthetic(tmp_path, report, mutate_payloads=tamper)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_sealer_requires_exact_payload_inventory(tmp_path: Path, mutation: str):
    def mutate(payloads: dict[str, bytes]) -> None:
        if mutation == "missing":
            payloads.pop("adaptive_threshold_crossings.csv")
        else:
            payloads["synthetic-extra.json"] = b"{}\n"

    with pytest.raises(
        audit.BinaryRegimeLongrunAuditError, match="inventory"
    ):
        _seal_synthetic(
            tmp_path,
            _sealed_report(),
            mutate_payloads=mutate,
        )


def test_fixed_and_online_candidate_truth_table_is_seal_bound(tmp_path: Path):
    cases = [
        (False, False, "exercised_positive", False, False),
        (False, True, "exercised_positive", False, False),
        (True, False, "exercised_positive", False, False),
        (True, True, "exercised_negative", True, False),
        (True, True, "exercised_positive", True, True),
    ]
    for offset, (post, strict, adaptive, fixed, online) in enumerate(cases):
        result = _seal_synthetic(
            tmp_path / str(offset),
            _sealed_report(
                post_hoc=post, strict=strict, adaptive_status=adaptive
            ),
        )
        manifest = json.loads(Path(result["stage_manifest"]).read_text())
        assert manifest["fixed_policy_candidate_for_prospective_paper"] is fixed
        assert manifest["online_learning_candidate_for_paper"] is online
        assert manifest["online_learning_historical_value_demonstrated"] is (
            adaptive == "exercised_positive"
        )


def test_deadline_failure_before_promotion_leaves_no_partial_bundle(tmp_path: Path):
    class RejectDeadline(_AcceptDeadline):
        def check(self, location: str) -> None:
            assert location == "before long-run artifact promotion"
            raise audit.BinaryRegimeLongrunAuditError("synthetic deadline")

    with pytest.raises(audit.BinaryRegimeLongrunAuditError, match="synthetic deadline"):
        _seal_synthetic(tmp_path, _sealed_report(), deadline=RejectDeadline())
    assert not (tmp_path / audit.AUDIT_RUN_ID).exists()
    assert not list(tmp_path.glob(audit.SEAL_TEMP_GLOB))


def test_atomic_promotion_has_one_strict_pre_rename_check_and_no_post_check(
    tmp_path: Path,
):
    class SequencedClock:
        def __init__(self) -> None:
            self.values = iter((0.0, 3599.999, 3600.0))
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return next(self.values)

    clock = SequencedClock()
    deadline = audit._Deadline(clock)
    run_dir = tmp_path / "promoted"
    checksums = audit._seal_audit_bundle(
        run_dir,
        {"payload.txt": b"sealed\n"},
        before_promote=lambda: deadline.check("before synthetic promotion"),
    )
    assert clock.calls == 2
    assert run_dir.is_dir()
    assert (run_dir / "payload.txt").read_bytes() == b"sealed\n"
    assert checksums == {"payload.txt": audit._sha256(b"sealed\n")}
    assert not list(tmp_path.glob(audit.SEAL_TEMP_GLOB))


def test_sealed_bundle_cannot_be_overwritten(tmp_path: Path):
    _seal_synthetic(tmp_path, _sealed_report())
    with pytest.raises(Exception, match="exists|overwrite|already"):
        _seal_synthetic(tmp_path, _sealed_report())


@pytest.mark.parametrize(
    ("elapsed", "within"), [(3599.999999, True), (3600.0, False)]
)
def test_runtime_report_uses_strict_under_one_hour(elapsed: float, within: bool):
    class RuntimeDeadline:
        def elapsed(self) -> float:
            return elapsed

    report = audit._runtime_report(RuntimeDeadline())
    assert report["within_limit"] is within


def test_public_runner_synthetic_end_to_end_locks_verifies_and_seals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    real_long_gates = audit.apply_post_hoc_long_run_gates
    real_strict_gates = audit.apply_strict_recent_history_gates
    template = _sealed_report()
    online_metrics = copy.deepcopy(
        template["online_metrics_2005_through_2026_07_09"]
    )
    frozen_metrics = copy.deepcopy(
        template["frozen_2023_metrics_2005_through_2026_07_09"]
    )
    frame = _market_frame(80, start="2018-01-02")
    output = tmp_path / audit.AUDIT_OUTPUT_PATH
    price = tmp_path / "synthetic.csv"
    object_id = "c" * 40
    parent = {
        "manifest_sha256": audit.PARENT_MANIFEST_SHA256,
        "bounded_result_sha256": "sha256:" + "2" * 64,
    }
    git_identity = {
        "commit": "a" * 40,
        "pre_lock_worktree_clean_excluding_final_input": True,
        "pre_lock_final_input_head_index_equal": True,
        "final_input_path": "synthetic.csv",
        "final_input_head_object_id": object_id,
        "final_input_index_object_id": object_id,
        "final_input_verified_clean_after_attempt_lock": False,
    }
    input_identity = {
        "path": "synthetic.csv",
        "sha256": "sha256:" + "9" * 64,
        "head_blob_sha256": "sha256:" + "9" * 64,
        "index_blob_sha256": "sha256:" + "9" * 64,
        "local_file_sha256": "sha256:" + "9" * 64,
        "head_object_id": object_id,
        "index_object_id": object_id,
        "head_index_local_equal": True,
        "path_status_clean": True,
        "verified_after_attempt_lock": True,
    }
    events: list[str] = []

    monkeypatch.setattr(
        audit,
        "_validated_rejected_validation_parent",
        lambda **_kwargs: events.append("parent") or parent,
    )
    monkeypatch.setattr(
        audit,
        "_clean_git_identity",
        lambda _root, _price: events.append("git") or copy.deepcopy(git_identity),
    )
    monkeypatch.setattr(audit, "_require_dependency_continuity", lambda *_a: None)

    load_count = 0

    def loader(
        _path: Path, *, end: pd.Timestamp, required_last_session: pd.Timestamp
    ):
        nonlocal load_count
        del required_last_session
        load_count += 1
        if load_count == 1:
            assert pd.Timestamp(end) == audit.VALIDATION_END
            events.append("prefix")
            return frame, {
                "source_path": "synthetic.csv",
                "bounded_result_sha256": parent["bounded_result_sha256"],
            }
        assert load_count == 2
        assert pd.Timestamp(end) == audit.AUDIT_END
        events.append("full")
        return frame, {
            "source_path": "synthetic.csv",
            "bounded_result_sha256": audit.FINAL_BOUNDED_RESULT_SHA256,
            "expected_bounded_result_sha256": audit.FINAL_BOUNDED_RESULT_SHA256,
            "bounded_rows": audit.FINAL_ROWS,
            "bounded_first_date": "1999-03-10",
            "bounded_last_date": "2026-07-09",
            "physical_snapshot_has_later_rows": False,
            "rows_after_bound_returned": False,
        }

    monkeypatch.setattr(audit, "load_bounded_prices", loader)
    monkeypatch.setattr(
        audit._selector_runner, "_require_source_continuity", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        audit._selector_runner,
        "_require_exact_physical_stage_bound",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(audit, "_require_final_input_identity", lambda *_a: None)
    monkeypatch.setattr(
        audit,
        "build_binary_regime_union_selector_forecast",
        lambda value, **_kwargs: pd.DataFrame(index=value.index),
    )
    monkeypatch.setattr(
        audit._selector_runner, "_require_checkpoint_continuity", lambda *_a, **_k: None
    )

    evaluation_count = 0

    def evaluate(*_args: Any, **_kwargs: Any):
        nonlocal evaluation_count
        evaluation_count += 1
        if evaluation_count == 1:
            return ({"synthetic": True}, {}, {}, {}, _passing_integrity())
        if evaluation_count == 2:
            return (online_metrics, {}, {}, {}, _passing_integrity())
        assert evaluation_count == 3
        return (frozen_metrics, {}, {}, {}, _passing_integrity())

    monkeypatch.setattr(audit._selector_runner, "_evaluate_policy_set", evaluate)
    long_gate_count = 0

    def long_gates(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal long_gate_count
        long_gate_count += 1
        if long_gate_count == 1:
            return _known_pre2024_gate_report()
        if long_gate_count == 2:
            return copy.deepcopy(template["post_hoc_long_run_gate_report"])
        assert long_gate_count == 3
        return real_long_gates(*_args, **_kwargs)

    monkeypatch.setattr(audit, "apply_post_hoc_long_run_gates", long_gates)
    monkeypatch.setattr(
        audit,
        "apply_strict_recent_history_gates",
        real_strict_gates,
    )
    monkeypatch.setattr(
        audit,
        "_require_comparator_identity",
        lambda *_a, **_k: {"synthetic": True},
    )
    monkeypatch.setattr(
        audit,
        "_adaptive_value_analysis",
        lambda *_a, **_k: (
            copy.deepcopy(template["adaptive_value"]),
            *_synthetic_adaptive_frames("exercised_positive"),
        ),
    )

    def tracked_after_lock(_root: Path, _path: Path) -> dict[str, Any]:
        assert (output / audit.ATTEMPT_LOCK_FILENAME).is_file()
        events.append("post_lock_input")
        return copy.deepcopy(input_identity)

    monkeypatch.setattr(audit, "_tracked_input_identity", tracked_after_lock)
    monkeypatch.setattr(
        audit._selector_runner,
        "_checkpoint_from_forecast",
        lambda *_a, **_k: {
            "learning_mode": audit.CAUSAL_ONLINE_MODE,
            "checkpoint_cutoff": "2026-07-09",
            "last_observed_session": "2026-07-09",
            "serialized_regime_states": {},
            "serialized_pending_shadow_lessons": {},
            "account_trailing_cooldown_context": [],
        },
    )
    monkeypatch.setattr(
        audit._selector_runner,
        "_flatten_forecast",
        lambda value: value.reset_index(names="date"),
    )

    def ledger_payloads(_value: Any, *, prefix: str) -> dict[str, bytes]:
        if prefix == "longrun_online":
            ledgers, _, _ = _synthetic_online_artifacts(online_metrics)
            return {
                f"{prefix}_{cost_name}_ledgers.csv": audit._frame_csv_bytes(
                    ledgers[cost_name]
                )
                for cost_name, _ in audit.COST_SCENARIOS
            }
        return {
            f"{prefix}_{cost_name}_ledgers.csv": b"synthetic\n"
            for cost_name, _ in audit.COST_SCENARIOS
        }

    def episode_payloads(_value: Any, *, prefix: str) -> dict[str, bytes]:
        result = {
            f"{prefix}_{cost_name}_{policy}_episodes.csv": b"synthetic\n"
            for cost_name, _ in audit.COST_SCENARIOS
            for policy in ("selector", "union")
        }
        if prefix == "longrun_online":
            _, passing, _ = _synthetic_online_artifacts(online_metrics)
            for cost_name, _ in audit.COST_SCENARIOS:
                result[f"{prefix}_{cost_name}_selector_episodes.csv"] = (
                    audit._frame_csv_bytes(passing[cost_name]["selector"])
                )
                result[f"{prefix}_{cost_name}_union_episodes.csv"] = (
                    audit._frame_csv_bytes(passing[cost_name]["union"])
                )
        return result

    def benefit_payloads(_value: Any, *, prefix: str) -> dict[str, bytes]:
        if prefix == "longrun_online":
            _, _, benefits = _synthetic_online_artifacts(online_metrics)
            return {
                f"{prefix}_{cost_name}_veto_benefits.csv": (
                    audit._frame_csv_bytes(benefits[cost_name])
                )
                for cost_name, _ in audit.COST_SCENARIOS
            }
        return {
            f"{prefix}_{cost_name}_veto_benefits.csv": b"synthetic\n"
            for cost_name, _ in audit.COST_SCENARIOS
        }

    monkeypatch.setattr(audit._selector_runner, "_ledger_payloads", ledger_payloads)
    monkeypatch.setattr(audit._selector_runner, "_episode_payloads", episode_payloads)
    monkeypatch.setattr(audit._selector_runner, "_benefit_payloads", benefit_payloads)

    result = audit.run_longrun_audit(
        repo_root=tmp_path,
        price_artifact=price,
        validation_manifest=tmp_path / "parent.json",
        output_dir=output,
    )
    run_dir = output / audit.AUDIT_RUN_ID
    assert result["stage_pass"] is True
    assert run_dir.is_dir()
    assert (output / audit.ATTEMPT_LOCK_FILENAME).is_file()
    assert (run_dir / audit.ATTEMPT_LOCK_FILENAME).read_bytes() == (
        output / audit.ATTEMPT_LOCK_FILENAME
    ).read_bytes()
    report = json.loads((run_dir / "report.json").read_text())
    manifest = json.loads((run_dir / "stage_manifest.json").read_text())
    assert report["fixed_policy_candidate_for_prospective_paper"] is True
    assert manifest["git_identity"][
        "final_input_verified_clean_after_attempt_lock"
    ] is True
    assert manifest["source_provenance"]["authorization_preflight"][
        "final_input_verified_clean_after_attempt_lock"
    ] is True
    assert events == ["parent", "git", "prefix", "post_lock_input", "full"]
