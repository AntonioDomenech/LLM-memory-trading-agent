from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config_store import load_local_config, public_config, save_local_config
from .information import build_information_bundle, information_manifest
from .llm_client import list_openai_models
from .runner import run_benchmark
from .schemas import PreviewRequest, RunRequest, SaveConfigRequest
from .storage import BenchmarkStore
from .warehouse.store import Warehouse

app = FastAPI(title="LLM Memory Trading Benchmark", version="0.2.0")

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
    return public_config()


@app.put("/api/config")
def put_config(payload: SaveConfigRequest):
    saved = save_local_config(payload)
    return public_config(saved)


@app.get("/api/source-plan")
def source_plan():
    config = load_local_config()
    return information_manifest(config.benchmark)


@app.get("/api/models")
def models():
    config = load_local_config()
    try:
        return list_openai_models(config.secrets)
    except Exception as exc:
        return {"models": [], "error": str(exc)}


@app.post("/api/preview")
def preview(payload: PreviewRequest):
    local = load_local_config()
    config = payload.config or local.benchmark
    as_of_date = payload.as_of_date or config.start_date
    try:
        return build_information_bundle(config, local.secrets, as_of_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs")
def create_run(payload: RunRequest):
    local = load_local_config()
    config = payload.config or local.benchmark
    try:
        return run_benchmark(config, local.secrets, dry_run=payload.dry_run)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs")
def list_runs():
    return {"runs": BenchmarkStore().list_runs()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = BenchmarkStore().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/warehouse/status")
def warehouse_status():
    wh = Warehouse()
    try:
        return wh.status()
    finally:
        wh.close()
