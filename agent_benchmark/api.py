from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config_store import load_local_config, public_config, save_local_config
from .benchmark_engine import BenchmarkEngine
from .information import build_information_bundle, information_manifest
from .jobs import BenchmarkJobManager, LiveScheduler
from .llm_client import list_openai_models
from .quality import OFFICIAL_PRESETS, build_preflight_report, build_run_diagnostics
from .runner import run_benchmark
from .schemas import BenchmarkConfig, BenchmarkRunRequest, LiveSnapshotRequest, PreviewRequest, PreviewRequestV2, RunRequest, SaveConfigRequest
from .serialization import json_safe
from .storage import BenchmarkStore
from .warehouse.store import Warehouse

app = FastAPI(title="LLM Memory Trading Benchmark", version="1.0.0")
job_manager = BenchmarkJobManager()
live_scheduler = LiveScheduler(job_manager)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    return json_safe(public_config())


@app.put("/api/config")
def put_config(payload: SaveConfigRequest):
    saved = save_local_config(payload)
    return json_safe(public_config(saved))


@app.get("/api/source-plan")
def source_plan():
    config = load_local_config()
    return json_safe(information_manifest(config.benchmark))


@app.get("/api/models")
def models():
    config = load_local_config()
    try:
        return json_safe(list_openai_models(config.secrets))
    except Exception as exc:
        return {"models": [], "error": str(exc)}


@app.post("/api/preview")
def preview(payload: PreviewRequest):
    local = load_local_config()
    config = payload.config or local.benchmark
    as_of_date = payload.as_of_date or config.start_date
    try:
        return json_safe(build_information_bundle(config, local.secrets, as_of_date))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/benchmark/preview")
def benchmark_preview(payload: PreviewRequestV2):
    local = load_local_config()
    config = payload.config or local.benchmark
    engine = BenchmarkEngine()
    try:
        return json_safe(engine.preview(config, local.secrets, phase=payload.phase, decision_date=payload.decision_date))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        engine.close()


@app.post("/api/benchmark/preflight")
def benchmark_preflight(payload: BenchmarkRunRequest):
    local = load_local_config()
    config = payload.config or local.benchmark
    wh = Warehouse()
    try:
        report = build_preflight_report(config, local.secrets, wh, store=job_manager.store)
        return json_safe(report)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        wh.close()


@app.post("/api/benchmark/runs")
def start_benchmark_run(payload: BenchmarkRunRequest):
    local = load_local_config()
    config = payload.config or local.benchmark
    preflight_report = None
    if not payload.dry_run and config.strict_preflight and config.run_preset in OFFICIAL_PRESETS:
        wh = Warehouse()
        try:
            preflight_report = build_preflight_report(config, local.secrets, wh, store=job_manager.store)
        finally:
            wh.close()
        if preflight_report.get("status") == "fail":
            issues = preflight_report.get("blocking_issues") or []
            labels = ", ".join(str(item.get("id") or "check") for item in issues[:4])
            raise HTTPException(status_code=400, detail=f"Preflight failed before paid official run: {labels}. Open Run control and run Preflight for details.")
    try:
        run = job_manager.start(config, local.secrets, dry_run=payload.dry_run)
        if preflight_report and run.get("id"):
            job_manager.store.save_benchmark_report(run["id"], "preflight", preflight_report)
            run["preflight"] = preflight_report
        return json_safe(run)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/benchmark/runs")
def list_benchmark_runs():
    return json_safe({"runs": BenchmarkStore().list_benchmark_runs()})


def _enrich_run_summary(run: dict) -> dict:
    summary = run.get("summary") or {}
    if not summary.get("equity_curve"):
        return run
    try:
        benchmark_config = BenchmarkConfig(**(run.get("config") or {}))
    except Exception:
        return run

    engine = BenchmarkEngine()
    try:
        symbols = engine._symbols(benchmark_config)
        run["summary"] = engine.enrich_summary_with_buy_hold(benchmark_config, symbols, summary)
        return run
    finally:
        engine.close()


@app.get("/api/benchmark/runs/{run_id}")
def get_benchmark_run(run_id: str):
    run = BenchmarkStore().get_benchmark_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return json_safe(_enrich_run_summary(run))


@app.get("/api/benchmark/runs/{run_id}/diagnostics")
def benchmark_run_diagnostics(run_id: str):
    store = BenchmarkStore()
    run = store.get_benchmark_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    try:
        config = BenchmarkConfig(**(run.get("config") or {}))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Run config cannot be parsed: {exc}") from exc
    wh = Warehouse()
    try:
        report = build_run_diagnostics(run, config, wh)
        stored_preflight = store.latest_benchmark_report(run_id, "preflight")
        if stored_preflight:
            report["preflight"] = stored_preflight
        store.save_benchmark_report(run_id, "diagnostics", report)
        return json_safe(report)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        wh.close()


@app.post("/api/benchmark/runs/{run_id}/mark-diagnostic")
def mark_benchmark_run_diagnostic(run_id: str, reason: str = "Marked diagnostic by user"):
    run = BenchmarkStore().mark_benchmark_run_diagnostic(run_id, reason)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return json_safe(run)


@app.post("/api/benchmark/runs/{run_id}/pause")
def pause_benchmark_run(run_id: str):
    try:
        return json_safe(job_manager.pause(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Active run not found") from exc


@app.post("/api/benchmark/runs/{run_id}/resume")
def resume_benchmark_run(run_id: str):
    try:
        return json_safe(job_manager.resume(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Active run not found") from exc


@app.post("/api/benchmark/runs/{run_id}/cancel")
def cancel_benchmark_run(run_id: str):
    try:
        return json_safe(job_manager.cancel(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Active run not found") from exc


@app.post("/api/live/start")
def start_live(payload: LiveSnapshotRequest):
    local = load_local_config()
    config = payload.config or local.benchmark
    try:
        return json_safe(live_scheduler.start(config, local.secrets, dry_run=payload.dry_run))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/live/stop")
def stop_live():
    return json_safe(live_scheduler.stop())


@app.post("/api/live/snapshot")
def live_snapshot(payload: LiveSnapshotRequest):
    local = load_local_config()
    config = payload.config or local.benchmark
    try:
        return json_safe(live_scheduler.snapshot(config, local.secrets, dry_run=payload.dry_run, force=payload.force))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/live/status")
def live_status():
    return json_safe(live_scheduler.status())


@app.get("/api/memory")
def list_memory(model: str = "", limit: int = 100):
    return json_safe({"items": BenchmarkStore().list_memory(model=model, limit=limit)})


@app.post("/api/runs")
def create_run(payload: RunRequest):
    local = load_local_config()
    config = payload.config or local.benchmark
    try:
        return json_safe(run_benchmark(config, local.secrets, dry_run=payload.dry_run))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs")
def list_runs():
    return json_safe({"runs": BenchmarkStore().list_runs()})


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = BenchmarkStore().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return json_safe(run)


@app.get("/api/warehouse/status")
def warehouse_status():
    wh = Warehouse()
    try:
        return json_safe(wh.status())
    finally:
        wh.close()
