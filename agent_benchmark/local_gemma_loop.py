from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import requests

from .benchmark_engine import BenchmarkEngine
from .config_store import DATA_DIR
from .llm_client import call_json_model
from .memory import build_frozen_system_manifest, verify_frozen_system_manifest
from .historical_blinding import HISTORICAL_BLINDING_CONTRACT
from .local_provider import (
    LOCAL_OLLAMA_MODEL,
    local_gemma_aapl_config,
    local_gemma_aapl_causal_replay_config,
    local_gemma_aapl_online_config,
    local_gemma_secret_config,
    validate_no_paid_api_mode,
)
from .monitoring import ResourceMonitor, summarize_call_metrics
from .quality import build_preflight_report, build_run_diagnostics
from .schemas import BenchmarkConfig, SecretConfig, model_to_dict
from .storage import BenchmarkStore
from .warehouse.store import Warehouse

ALLOWLISTED_PATCH_CATEGORIES = {
    "prompts",
    "exposure_policy",
    "critic_framing",
    "memory_lesson_formatting",
    "diagnostics",
    "benchmark_config",
}


@dataclass
class LocalPatch:
    category: str
    reason: str
    config_updates: Dict[str, Any]


class LocalBenchmarkControl:
    def __init__(self, store: BenchmarkStore, monitor: ResourceMonitor | None = None):
        self.store = store
        self.monitor = monitor

    def checkpoint(self, run_id: str, phase: str, progress: Dict[str, Any]) -> None:
        if self.monitor and self.monitor.abort_reason:
            progress = {**progress, "monitor_abort_reason": self.monitor.abort_reason}
        self.store.update_benchmark_run(run_id, status="running", phase=phase, progress=progress)

    def should_cancel(self, run_id: str) -> bool:
        return bool(self.monitor and self.monitor.abort_reason)

    def wait_if_paused(self, run_id: str) -> None:
        return None


def ensure_ollama_model(model: str = LOCAL_OLLAMA_MODEL, *, pull: bool = True) -> Dict[str, Any]:
    ollama = ollama_executable()
    tags = _ollama_tags(ollama)
    installed = {item.get("name") for item in tags.get("models", []) if item.get("name")}
    installed.update({item.get("model") for item in tags.get("models", []) if item.get("model")})
    if model in installed:
        record = next(
            (
                item
                for item in tags.get("models", [])
                if model in {item.get("name"), item.get("model")}
            ),
            {},
        )
        return {
            "status": "present",
            "model": model,
            "digest": str(record.get("digest") or ""),
            "installed_models": sorted(installed),
        }
    if not pull:
        raise RuntimeError(f"Ollama model {model!r} is not installed.")
    _run_checked([ollama, "pull", model], timeout=60 * 60)
    tags = _ollama_tags(ollama)
    installed = {item.get("name") for item in tags.get("models", []) if item.get("name")}
    installed.update({item.get("model") for item in tags.get("models", []) if item.get("model")})
    if model not in installed:
        raise RuntimeError(f"Pulled {model!r}, but Ollama does not list it as installed.")
    record = next(
        (
            item
            for item in tags.get("models", [])
            if model in {item.get("name"), item.get("model")}
        ),
        {},
    )
    return {
        "status": "pulled",
        "model": model,
        "digest": str(record.get("digest") or ""),
        "installed_models": sorted(installed),
    }


def ollama_executable() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("Ollama is not installed or not on PATH. Install Ollama, then rerun the local Gemma loop.")


def run_local_json_smoke(config: BenchmarkConfig, secrets: SecretConfig) -> Dict[str, Any]:
    validate_no_paid_api_mode(config, secrets)
    result = call_json_model(
        config,
        secrets,
        "Return only JSON.",
        json.dumps({"task": "Return {'ok': true, 'model': model}.", "model": config.model}),
        dry_run=False,
        fallback={"ok": False},
        cache_namespace="local-smoke",
    )
    if result.get("ok") is not True:
        raise RuntimeError(f"Local model JSON smoke test failed: {result}")
    return result


def _requires_local_model_smoke(config: BenchmarkConfig) -> bool:
    """Return whether this run gives the local LLM any decision authority."""

    return not (
        config.evaluation_mode in {"frozen_holdout", "causal_online_replay"}
        and config.historical_decision_authority == "precutoff_quantitative_policy"
    )


def evaluate_success(run: Dict[str, Any], config: BenchmarkConfig) -> Dict[str, Any]:
    summary = run.get("summary") or {}
    metrics = summary.get("metrics") or {}
    usage = summary.get("api_usage_estimate") or {}
    evaluation_metrics = summary.get("test_metrics") or metrics
    comparison = summary.get("test_buy_hold_comparison") or summary.get("buy_hold_comparison") or {}
    single = next((item for item in comparison.get("benchmarks", []) if item.get("id") == "single_stock"), {})
    ai_return = _float(evaluation_metrics.get("total_return"))
    buy_hold_return = _float(single.get("total_return"))
    invalid = int(metrics.get("invalid_allocation_count") or metrics.get("model_failures") or 0)
    local_only = bool(summary.get("no_paid_api_mode") and usage.get("local_only") and usage.get("estimated_cost_usd") == 0.0)
    overall_beat_buy_hold = ai_return is not None and buy_hold_return is not None and ai_return > buy_hold_return
    period_results = ((summary.get("frozen_holdout") or {}).get("periods") or {})
    expected_periods = set()
    if config.evaluation_mode == "frozen_holdout":
        start_year = date.fromisoformat(config.test_start).year
        end_date = date.fromisoformat(config.test_end)
        for year in range(start_year, end_date.year + 1):
            expected_periods.add(
                f"{year}_ytd"
                if year == end_date.year and (end_date.month, end_date.day) != (12, 31)
                else str(year)
            )
    required_periods_present = bool(expected_periods) and set(period_results) == expected_periods
    all_required_periods_beat = (
        required_periods_present
        and all(bool(item.get("beat_buy_hold")) for item in period_results.values())
        if config.evaluation_mode == "frozen_holdout"
        else True
    )
    beat_buy_hold = bool(overall_beat_buy_hold and all_required_periods_beat)
    learning_proof = summary.get("frozen_learning_state_proof") or {}
    learning_before = learning_proof.get("before") or {}
    learning_after = learning_proof.get("after") or {}
    data_proof = summary.get("evaluation_data_snapshot") or {}
    data_before = data_proof.get("before") or {}
    data_after = data_proof.get("after") or {}
    manifest = summary.get("frozen_system_manifest") or {}
    manifest_config = manifest.get("config") or {}
    historical_model_calls = [
        item
        for item in (run.get("decisions") or [])
        if item.get("phase") == "test"
        and (item.get("output") or {}).get("_api_status") == "ok"
    ]
    frozen_certification_checks = {
        "learning_state_unchanged": bool(
            learning_proof.get("unchanged")
            and learning_proof.get("test_outcomes_used_for_learning") is False
            and learning_before.get("sha256")
            and learning_before.get("sha256") == learning_after.get("sha256")
        ),
        "evaluation_data_unchanged": bool(
            data_proof.get("unchanged_during_run")
            and data_before.get("sha256")
            and data_before.get("sha256") == data_after.get("sha256")
            and int(data_after.get("session_count") or 0) > 0
        ),
        "manifest_hash_valid": verify_frozen_system_manifest(manifest),
        "manifest_matches_run_config": bool(manifest_config == model_to_dict(config)),
        "manifest_binds_evaluation_data": bool(
            manifest.get("evaluation_data_snapshot") == data_after
        ),
        "manifest_binds_training_data": bool(manifest.get("base_content_hash")),
        "model_digest_bound": bool(
            config.local_model_digest
            and manifest.get("local_model_digest") == config.local_model_digest
        ),
        "implementation_commit_bound": bool(
            config.implementation_commit
            and manifest.get("git_commit") == config.implementation_commit
        ),
        "prompt_and_implementation_bound": bool(
            manifest.get("prompt_contract_sha256")
            and manifest.get("implementation_sha256")
        ),
        "parametric_lookahead_guard_bound": bool(
            config.historical_prompt_blinding
            and bool(config.model_training_data_cutoff)
            and config.historical_prompt_blinding_contract == HISTORICAL_BLINDING_CONTRACT
            and manifest_config.get("historical_prompt_blinding") is True
            and manifest_config.get("historical_prompt_blinding_contract")
            == HISTORICAL_BLINDING_CONTRACT
            and manifest_config.get("model_training_data_cutoff")
            == config.model_training_data_cutoff
        ),
        "cutoff_safe_decision_authority": bool(
            config.historical_decision_authority == "precutoff_quantitative_policy"
            and (summary.get("benchmark_contract") or {}).get(
                "historical_decision_authority"
            )
            == "precutoff_quantitative_policy"
        ),
        "no_historical_llm_market_inference": bool(
            not historical_model_calls
            and int(summary.get("model_calls") or 0) == 0
        ),
        "no_allocation_repairs": int(metrics.get("repair_count") or 0) == 0,
        "report_role_is_frozen": bool(
            (summary.get("benchmark_contract") or {}).get("evaluation_mode")
            == "frozen_holdout"
            and (summary.get("frozen_holdout") or {}).get("test_evidence") is True
        ),
    }
    frozen_certified = bool(
        config.evaluation_mode == "frozen_holdout"
        and all(frozen_certification_checks.values())
    )
    evidence_eligible = frozen_certified
    pristine_test_evidence = bool(
        frozen_certified
        and config.globally_pristine
        and int(config.historical_holdout_reveal_count_lower_bound or 0) == 0
    )
    evidence_scope = (
        "prospective_pristine"
        if pristine_test_evidence
        else (
            "candidate_specific_frozen_not_globally_pristine"
            if frozen_certified
            else "not_certified"
        )
    )
    return {
        "success": bool(beat_buy_hold and invalid == 0 and local_only and evidence_eligible),
        "beat_buy_hold": beat_buy_hold,
        "overall_beat_buy_hold": overall_beat_buy_hold,
        "all_required_periods_beat_buy_hold": all_required_periods_beat,
        "required_periods_present": required_periods_present,
        "expected_periods": sorted(expected_periods),
        "period_results": period_results,
        "ai_return": ai_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": ai_return - buy_hold_return if ai_return is not None and buy_hold_return is not None else None,
        "invalid_decisions": invalid,
        "local_only_cost_proof": local_only,
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "evaluation_mode": config.evaluation_mode,
        "eligible_as_frozen_test_evidence": evidence_eligible,
        "eligible_as_pristine_test_evidence": pristine_test_evidence,
        "evidence_scope": evidence_scope,
        "globally_pristine": bool(config.globally_pristine),
        "historical_holdout_reveal_count_lower_bound": int(
            config.historical_holdout_reveal_count_lower_bound or 0
        ),
        "frozen_certified": frozen_certified,
        "frozen_certification_checks": frozen_certification_checks,
        "api_cost_display": usage.get("estimated_cost_display"),
        "model": summary.get("model") or config.model,
        "model_provider": summary.get("model_provider") or config.model_provider,
        "local_model_base_url": summary.get("local_model_base_url") or config.local_model_base_url,
        "evaluation_window": summary.get("success_evaluation_window") or {
            "name": "full_run_legacy",
            "phase": None,
            "start_date": comparison.get("start_date"),
            "end_date": comparison.get("end_date"),
            "reason": "Legacy summary did not include a separate test-window comparison.",
        },
    }


def propose_allowlisted_patch(diagnostics: Dict[str, Any], evaluation: Dict[str, Any], config: BenchmarkConfig) -> LocalPatch | None:
    metrics = diagnostics.get("metrics") or {}
    if int(evaluation.get("invalid_decisions") or 0) > 0:
        return LocalPatch(
            category="exposure_policy",
            reason="Invalid decisions occurred; keep shorting available and enforce the trinary all-in action contract without a turnover cap.",
            config_updates={"allow_short": True, "single_stock_action_space": "trinary_all_in", "max_daily_turnover": 0.0, "turnover_prompt_buffer": 0.0, "turnover_edge_multiplier": 0.0},
        )
    if metrics.get("cash_drag_proxy") and float(metrics["cash_drag_proxy"]) > 0:
        return LocalPatch(
            category="benchmark_config",
            reason="Diagnostics show cash drag; allow full buy-and-hold participation when evidence supports it.",
            config_updates={"allow_short": True, "single_stock_action_space": "trinary_all_in", "max_daily_turnover": 0.0, "turnover_prompt_buffer": 0.0, "max_gross_exposure": 1.0},
        )
    if metrics.get("bullish_but_underexposed_days"):
        return LocalPatch(
            category="critic_framing",
            reason="Stage 1 was bullish while Stage 2 stayed underexposed; keep trinary all-in choices fast and widen no-turnover target range.",
            config_updates={"exposure_critic_enabled": False, "max_daily_turnover": 0.0, "turnover_prompt_buffer": 0.0},
        )
    if not evaluation.get("beat_buy_hold"):
        return LocalPatch(
            category="benchmark_config",
            reason="AI did not beat AAPL buy-and-hold; ensure the config permits full long/short all-in actions without turnover throttling.",
            config_updates={"allow_short": True, "single_stock_action_space": "trinary_all_in", "max_daily_turnover": 0.0, "turnover_prompt_buffer": 0.0, "turnover_edge_multiplier": 0.0},
        )
    return None


def apply_allowlisted_patch(config: BenchmarkConfig, patch: LocalPatch) -> BenchmarkConfig:
    if patch.category not in ALLOWLISTED_PATCH_CATEGORIES:
        raise ValueError(f"Patch category {patch.category!r} is not allowlisted.")
    payload = model_to_dict(config)
    payload.update(patch.config_updates)
    return BenchmarkConfig(**payload)


def run_iteration(
    *,
    iteration: int,
    config: BenchmarkConfig,
    secrets: SecretConfig,
    store: BenchmarkStore,
    repo_root: Path,
    commit_before_run: bool = True,
    run_id: str | None = None,
    resume: bool = False,
) -> Dict[str, Any]:
    validate_no_paid_api_mode(config, secrets)
    if run_id:
        existing = store.get_benchmark_run(run_id)
        if not existing:
            raise RuntimeError(f"Cannot resume missing benchmark run {run_id!r}.")
        if not _is_resumable_run(existing):
            raise RuntimeError(f"Cannot resume run {run_id!r} from status {existing.get('status')!r}.")
        commit_hash = current_git_hash(repo_root)
        if config.evaluation_mode != "legacy":
            if not config.implementation_commit:
                raise RuntimeError("Frozen/causal resume lacks its original implementation commit")
            if commit_hash != config.implementation_commit:
                raise RuntimeError(
                    "Resume implementation commit differs from the run's frozen contract"
                )
    else:
        commit_hash = (
            commit_current_state(repo_root, f"local gemma benchmark iteration {iteration}")
            if commit_before_run
            else current_git_hash(repo_root)
        )
        if config.evaluation_mode != "legacy":
            config_payload = model_to_dict(config)
            config_payload["implementation_commit"] = commit_hash
            config = BenchmarkConfig(**config_payload)
        run_id = f"local-gemma-aapl-{iteration}-{uuid.uuid4().hex[:8]}"
        store.create_benchmark_run(run_id, model_to_dict(config))
    if config.evaluation_mode != "legacy":
        if not config.local_model_digest:
            raise RuntimeError("Frozen/causal runs require the exact local Ollama model digest")
        dirty = _run_checked(["git", "status", "--porcelain"], cwd=repo_root).stdout.strip()
        if dirty:
            raise RuntimeError("Frozen/causal runs require a clean committed worktree")
    run_dir = DATA_DIR / "local_gemma_runs" / run_id
    preflight_report = _preflight(config, secrets, store)
    store.save_benchmark_report(run_id, "preflight", preflight_report)
    if preflight_report.get("status") == "fail":
        issues = preflight_report.get("blocking_issues") or []
        message = "Local Gemma preflight failed: " + "; ".join(str(item.get("message") or item.get("id")) for item in issues)
        store.update_benchmark_run(run_id, status="failed", phase="preflight", error=message, progress={"message": message}, finished=True)
        raise RuntimeError(message)

    monitor_path = run_dir / "monitoring.jsonl"
    monitor = ResourceMonitor(monitor_path, config) if config.monitoring_enabled else None
    control = LocalBenchmarkControl(store, monitor)
    engine = BenchmarkEngine()
    try:
        if monitor:
            monitor.start()
        engine.run(run_id=run_id, config=config, secrets=secrets, store=store, control=control, dry_run=False, resume=resume)
    except Exception as exc:
        store.append_benchmark_event(run_id, "error", "run_failed", {"error": str(exc)})
        store.update_benchmark_run(
            run_id,
            status="failed",
            phase="failed",
            error=str(exc),
            progress={"message": str(exc), "monitor_abort_reason": monitor.abort_reason if monitor else ""},
            finished=True,
        )
        raise
    finally:
        if monitor:
            monitor.stop()
        engine.close()
    if config.evaluation_mode != "legacy":
        if current_git_hash(repo_root) != commit_hash:
            message = "Implementation commit changed during the frozen/causal run"
            store.update_benchmark_run(
                run_id,
                status="failed",
                phase="failed",
                error=message,
                finished=True,
            )
            raise RuntimeError(message)
        dirty_after = _run_checked(["git", "status", "--porcelain"], cwd=repo_root).stdout.strip()
        if dirty_after:
            message = "Worktree changed during the frozen/causal run"
            store.update_benchmark_run(
                run_id,
                status="failed",
                phase="failed",
                error=message,
                finished=True,
            )
            raise RuntimeError(message)
    run = store.get_benchmark_run(run_id) or {"id": run_id, "summary": {}, "decisions": []}
    summary = run.get("summary") or {}
    summary["commit_hash"] = commit_hash
    if config.evaluation_mode != "legacy":
        prompt_path = Path(__file__).with_name("prompting.py")
        implementation_paths = (
            Path(__file__).with_name("benchmark_engine.py"),
            Path(__file__).with_name("deterministic_memory.py"),
            Path(__file__).with_name("historical_blinding.py"),
            Path(__file__).with_name("llm_client.py"),
            Path(__file__).with_name("online_policy.py"),
        )
        prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        implementation_hasher = hashlib.sha256()
        for path in implementation_paths:
            implementation_hasher.update(path.name.encode("utf-8"))
            implementation_hasher.update(path.read_bytes())
        base_content_hash = str(
            ((summary.get("deterministic_memory") or {}).get("content_hash")) or ""
        )
        summary["frozen_system_manifest"] = build_frozen_system_manifest(
            config,
            base_content_hash=base_content_hash,
            evaluation_data_snapshot=(summary.get("evaluation_data_snapshot") or {}).get("after") or {},
            git_commit=commit_hash,
            prompt_contract_sha256=prompt_hash,
            implementation_sha256=implementation_hasher.hexdigest(),
        )
    summary["monitoring"] = {
        "enabled": bool(monitor),
        "path": str(monitor_path) if monitor else "",
        "samples": monitor.samples if monitor else 0,
        "abort_reason": monitor.abort_reason if monitor else "",
        "call_metrics": summarize_call_metrics(run.get("decisions") or []),
    }
    store.update_benchmark_run(run_id, summary=summary)
    run = store.get_benchmark_run(run_id) or run
    evaluation = evaluate_success(run, config)
    diagnostics = _diagnostics(run, config)
    report = {
        "iteration": iteration,
        "run_id": run_id,
        "commit_hash": commit_hash,
        "resumed": resume,
        "config": model_to_dict(config),
        "evaluation": evaluation,
        "diagnostics": diagnostics,
        "preflight": preflight_report,
        "monitoring_log": str(monitor_path) if monitor else "",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iteration_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    store.save_benchmark_report(run_id, "local_gemma_iteration", report)
    if monitor and monitor.abort_reason:
        raise RuntimeError(monitor.abort_reason)
    return report


def _is_resumable_run(run: Dict[str, Any]) -> bool:
    status = run.get("status")
    if status in {"paused", "running"}:
        return True
    if status == "cancelled":
        monitoring = (run.get("summary") or {}).get("monitoring") or {}
        return bool(monitoring.get("abort_reason"))
    return False


def _config_for_resume(run: Dict[str, Any]) -> BenchmarkConfig:
    payload = dict(run.get("config") or {})
    config = BenchmarkConfig(**payload)
    if config.run_preset in {"local_gemma_aapl_full", "local_gemma_aapl_online"} and int(config.warehouse_recycle_interval_days or 0) <= 0:
        defaults = local_gemma_aapl_online_config() if config.run_preset == "local_gemma_aapl_online" else local_gemma_aapl_config()
        config_payload = model_to_dict(config)
        config_payload["warehouse_recycle_interval_days"] = defaults.warehouse_recycle_interval_days
        config = BenchmarkConfig(**config_payload)
    return config


def _config_for_preset(preset: str) -> BenchmarkConfig:
    if preset == "legacy":
        return local_gemma_aapl_config()
    if preset == "aapl-online":
        return local_gemma_aapl_online_config()
    if preset == "aapl-causal-replay":
        return local_gemma_aapl_causal_replay_config()
    raise ValueError(f"Unknown local Gemma preset {preset!r}.")


def _with_day_limits(
    config: BenchmarkConfig,
    *,
    max_train_days: int | None,
    max_test_days: int | None,
) -> BenchmarkConfig:
    updates: Dict[str, Any] = {}
    for field, value in (("max_train_days", max_train_days), ("max_test_days", max_test_days)):
        if value is None:
            continue
        if int(value) < 0:
            raise ValueError(f"{field} cannot be negative.")
        updates[field] = int(value)
    if not updates:
        return config
    payload = model_to_dict(config)
    payload.update(updates)
    return BenchmarkConfig(**payload)


def run_loop(
    max_iterations: int = 3,
    *,
    pull_model: bool = True,
    commit_before_run: bool = True,
    resume_run_id: str = "",
    preset: str = "aapl-online",
    max_train_days: int | None = None,
    max_test_days: int | None = None,
    preflight_only: bool = False,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    config = _config_for_preset(preset)
    secrets = local_gemma_secret_config()
    store = BenchmarkStore()
    if resume_run_id:
        if preflight_only:
            raise ValueError("--preflight-only cannot be combined with --resume-run-id.")
        if max_train_days is not None or max_test_days is not None:
            raise ValueError("Day limits cannot be changed while resuming a saved run.")
        existing = store.get_benchmark_run(resume_run_id)
        if not existing:
            raise RuntimeError(f"Cannot resume missing benchmark run {resume_run_id!r}.")
        config = _config_for_resume(existing)
    else:
        config = _with_day_limits(config, max_train_days=max_train_days, max_test_days=max_test_days)
    validate_no_paid_api_mode(config, secrets)
    if preflight_only:
        preflight_report = _preflight(config, secrets, store)
        return {
            "status": f"preflight_{preflight_report.get('status', 'unknown')}",
            "preset": preset,
            "config": model_to_dict(config),
            "model_status": "not_checked",
            "smoke": "not_run",
            "preflight": preflight_report,
        }

    online_preset = config.run_preset == "local_gemma_aapl_online"
    if online_preset:
        max_iterations = 1
    expected_resume_digest = str(config.local_model_digest or "") if resume_run_id else ""
    model_status = ensure_ollama_model(config.model, pull=pull_model)
    observed_digest = str(model_status.get("digest") or "")
    if (
        resume_run_id
        and config.evaluation_mode != "legacy"
        and (not expected_resume_digest or observed_digest != expected_resume_digest)
    ):
        raise RuntimeError("Resume Ollama model digest differs from the run's frozen contract")
    config_payload = model_to_dict(config)
    config_payload["local_model_digest"] = observed_digest
    config = BenchmarkConfig(**config_payload)
    smoke = (
        run_local_json_smoke(config, secrets)
        if _requires_local_model_smoke(config)
        else {
            "status": "skipped",
            "reason": "no_historical_llm_decision_authority",
        }
    )
    reports: List[Dict[str, Any]] = []
    first_iteration = 1
    if resume_run_id:
        report = run_iteration(
            iteration=1,
            config=config,
            secrets=secrets,
            store=store,
            repo_root=repo_root,
            commit_before_run=commit_before_run,
            run_id=resume_run_id,
            resume=True,
        )
        reports.append(report)
        if report["evaluation"]["success"]:
            return {"status": "success", "model_status": model_status, "smoke": smoke, "reports": reports}
        if online_preset:
            return {"status": "target_not_met", "model_status": model_status, "smoke": smoke, "reports": reports}
        patch = propose_allowlisted_patch(report.get("diagnostics") or {}, report.get("evaluation") or {}, config)
        if not patch:
            return {"status": "blocked", "reason": "No allowlisted autonomous patch was available.", "model_status": model_status, "smoke": smoke, "reports": reports}
        run_backend_tests(repo_root)
        config = apply_allowlisted_patch(config, patch)
        reports[-1]["next_patch"] = {"category": patch.category, "reason": patch.reason, "config_updates": patch.config_updates}
        first_iteration = 2
    for iteration in range(first_iteration, max_iterations + 1):
        report = run_iteration(
            iteration=iteration,
            config=config,
            secrets=secrets,
            store=store,
            repo_root=repo_root,
            commit_before_run=commit_before_run,
        )
        reports.append(report)
        if report["evaluation"]["success"]:
            return {"status": "success", "model_status": model_status, "smoke": smoke, "reports": reports}
        if online_preset:
            return {"status": "target_not_met", "model_status": model_status, "smoke": smoke, "reports": reports}
        patch = propose_allowlisted_patch(report.get("diagnostics") or {}, report.get("evaluation") or {}, config)
        if not patch:
            return {"status": "blocked", "reason": "No allowlisted autonomous patch was available.", "model_status": model_status, "smoke": smoke, "reports": reports}
        run_backend_tests(repo_root)
        config = apply_allowlisted_patch(config, patch)
        reports[-1]["next_patch"] = {"category": patch.category, "reason": patch.reason, "config_updates": patch.config_updates}
    return {"status": "max_iterations_reached", "model_status": model_status, "smoke": smoke, "reports": reports}


def commit_current_state(repo_root: Path, message: str) -> str:
    status = _run_checked(["git", "status", "--porcelain"], cwd=repo_root).stdout.strip()
    if status:
        _run_checked(["git", "add", "-A"], cwd=repo_root)
        _run_checked(["git", "commit", "-m", message], cwd=repo_root)
    return current_git_hash(repo_root)


def current_git_hash(repo_root: Path) -> str:
    return _run_checked(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()


def run_backend_tests(repo_root: Path) -> None:
    _run_checked([sys.executable, "-m", "pytest", "tests"], cwd=repo_root, timeout=20 * 60)


def _preflight(config: BenchmarkConfig, secrets: SecretConfig, store: BenchmarkStore) -> Dict[str, Any]:
    warehouse = Warehouse()
    try:
        return build_preflight_report(config, secrets, warehouse, store=store)
    finally:
        warehouse.close()


def _diagnostics(run: Dict[str, Any], config: BenchmarkConfig) -> Dict[str, Any]:
    warehouse = Warehouse()
    try:
        return build_run_diagnostics(run, config, warehouse)
    finally:
        warehouse.close()


def _ollama_tags(ollama: str) -> Dict[str, Any]:
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
        if resp.status_code < 400:
            return resp.json()
    except Exception:
        pass
    _run_checked([ollama, "list"], timeout=20)
    resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _run_checked(command: List[str], *, cwd: Path | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    kwargs: Dict[str, Any] = {
        "cwd": str(cwd) if cwd else None,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "check": False,
    }
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.run(command, **kwargs)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed with exit code {proc.returncode}: {proc.stderr or proc.stdout}")
    return proc


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local-only Ollama Gemma AAPL benchmark loop.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Legacy loop limit; the aapl-online preset always runs exactly one iteration.")
    parser.add_argument(
        "--preset",
        choices=("aapl-online", "aapl-causal-replay", "legacy"),
        default="aapl-online",
        help="Use the chronological online-learning system (default) or the preserved legacy loop.",
    )
    parser.add_argument("--max-train-days", type=int, default=None, help="Override the preset training-day cap; 0 means the full window.")
    parser.add_argument("--max-test-days", type=int, default=None, help="Override the preset test-day cap; 0 means the full window.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate configuration and local data without pulling/calling Gemma or starting a benchmark.",
    )
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--no-commit-before-run", action="store_true")
    parser.add_argument("--resume-run-id", default="", help="Resume a paused or monitor-cancelled local Gemma benchmark run from its saved checkpoint.")
    args = parser.parse_args()
    result = run_loop(
        max_iterations=args.max_iterations,
        pull_model=not args.no_pull,
        commit_before_run=not args.no_commit_before_run,
        resume_run_id=args.resume_run_id,
        preset=args.preset,
        max_train_days=args.max_train_days,
        max_test_days=args.max_test_days,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
