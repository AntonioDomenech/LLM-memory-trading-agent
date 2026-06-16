from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import requests

from .benchmark_engine import BenchmarkEngine
from .config_store import DATA_DIR
from .llm_client import call_json_model
from .local_provider import (
    LOCAL_OLLAMA_MODEL,
    local_gemma_aapl_config,
    local_gemma_secret_config,
    validate_no_paid_api_mode,
)
from .monitoring import ResourceMonitor, summarize_call_metrics
from .quality import build_run_diagnostics
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
        return {"status": "present", "model": model, "installed_models": sorted(installed)}
    if not pull:
        raise RuntimeError(f"Ollama model {model!r} is not installed.")
    _run_checked([ollama, "pull", model], timeout=60 * 60)
    tags = _ollama_tags(ollama)
    installed = {item.get("name") for item in tags.get("models", []) if item.get("name")}
    installed.update({item.get("model") for item in tags.get("models", []) if item.get("model")})
    if model not in installed:
        raise RuntimeError(f"Pulled {model!r}, but Ollama does not list it as installed.")
    return {"status": "pulled", "model": model, "installed_models": sorted(installed)}


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


def evaluate_success(run: Dict[str, Any], config: BenchmarkConfig) -> Dict[str, Any]:
    summary = run.get("summary") or {}
    metrics = summary.get("metrics") or {}
    usage = summary.get("api_usage_estimate") or {}
    comparison = summary.get("buy_hold_comparison") or {}
    single = next((item for item in comparison.get("benchmarks", []) if item.get("id") == "single_stock"), {})
    ai_return = _float(metrics.get("total_return"))
    buy_hold_return = _float(single.get("total_return"))
    invalid = int(metrics.get("invalid_allocation_count") or metrics.get("model_failures") or 0)
    local_only = bool(summary.get("no_paid_api_mode") and usage.get("local_only") and usage.get("estimated_cost_usd") == 0.0)
    beat_buy_hold = ai_return is not None and buy_hold_return is not None and ai_return > buy_hold_return
    return {
        "success": bool(beat_buy_hold and invalid == 0 and local_only),
        "beat_buy_hold": beat_buy_hold,
        "ai_return": ai_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": ai_return - buy_hold_return if ai_return is not None and buy_hold_return is not None else None,
        "invalid_decisions": invalid,
        "local_only_cost_proof": local_only,
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "api_cost_display": usage.get("estimated_cost_display"),
        "model": summary.get("model") or config.model,
        "model_provider": summary.get("model_provider") or config.model_provider,
        "local_model_base_url": summary.get("local_model_base_url") or config.local_model_base_url,
    }


def propose_allowlisted_patch(diagnostics: Dict[str, Any], evaluation: Dict[str, Any], config: BenchmarkConfig) -> LocalPatch | None:
    metrics = diagnostics.get("metrics") or {}
    if int(evaluation.get("invalid_decisions") or 0) > 0:
        return LocalPatch(
            category="exposure_policy",
            reason="Invalid decisions occurred; tighten to long-only single-stock execution with direct turnover math.",
            config_updates={"allow_short": False, "max_daily_turnover": 1.0, "turnover_prompt_buffer": 0.0, "turnover_edge_multiplier": 0.0},
        )
    if metrics.get("cash_drag_proxy") and float(metrics["cash_drag_proxy"]) > 0:
        return LocalPatch(
            category="benchmark_config",
            reason="Diagnostics show cash drag; allow full buy-and-hold participation when evidence supports it.",
            config_updates={"allow_short": False, "max_daily_turnover": 1.0, "turnover_prompt_buffer": 0.0, "max_gross_exposure": 1.0},
        )
    if metrics.get("bullish_but_underexposed_days"):
        return LocalPatch(
            category="critic_framing",
            reason="Stage 1 was bullish while Stage 2 stayed underexposed; keep exposure critic enabled and widen target range.",
            config_updates={"exposure_critic_enabled": True, "max_daily_turnover": 1.0, "turnover_prompt_buffer": 0.0},
        )
    if not evaluation.get("beat_buy_hold"):
        return LocalPatch(
            category="benchmark_config",
            reason="AI did not beat AAPL buy-and-hold; ensure the config permits full long exposure without turnover throttling.",
            config_updates={"allow_short": False, "max_daily_turnover": 1.0, "turnover_prompt_buffer": 0.0, "turnover_edge_multiplier": 0.0},
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
) -> Dict[str, Any]:
    validate_no_paid_api_mode(config, secrets)
    commit_hash = commit_current_state(repo_root, f"local gemma benchmark iteration {iteration}") if commit_before_run else current_git_hash(repo_root)
    run_id = f"local-gemma-aapl-{iteration}-{uuid.uuid4().hex[:8]}"
    store.create_benchmark_run(run_id, model_to_dict(config))
    run_dir = DATA_DIR / "local_gemma_runs" / run_id
    monitor_path = run_dir / "monitoring.jsonl"
    monitor = ResourceMonitor(monitor_path, config) if config.monitoring_enabled else None
    control = LocalBenchmarkControl(store, monitor)
    engine = BenchmarkEngine()
    try:
        if monitor:
            monitor.start()
        engine.run(run_id=run_id, config=config, secrets=secrets, store=store, control=control, dry_run=False)
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
    run = store.get_benchmark_run(run_id) or {"id": run_id, "summary": {}, "decisions": []}
    summary = run.get("summary") or {}
    summary["commit_hash"] = commit_hash
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
        "config": model_to_dict(config),
        "evaluation": evaluation,
        "diagnostics": diagnostics,
        "monitoring_log": str(monitor_path) if monitor else "",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iteration_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    store.save_benchmark_report(run_id, "local_gemma_iteration", report)
    if monitor and monitor.abort_reason:
        raise RuntimeError(monitor.abort_reason)
    return report


def run_loop(max_iterations: int = 3, *, pull_model: bool = True, commit_before_run: bool = True) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    config = local_gemma_aapl_config()
    secrets = local_gemma_secret_config()
    validate_no_paid_api_mode(config, secrets)
    model_status = ensure_ollama_model(config.model, pull=pull_model)
    smoke = run_local_json_smoke(config, secrets)
    store = BenchmarkStore()
    reports: List[Dict[str, Any]] = []
    for iteration in range(1, max_iterations + 1):
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
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--no-commit-before-run", action="store_true")
    args = parser.parse_args()
    result = run_loop(
        max_iterations=args.max_iterations,
        pull_model=not args.no_pull,
        commit_before_run=not args.no_commit_before_run,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
