from __future__ import annotations

import ctypes
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .schemas import BenchmarkConfig


@dataclass
class AbortThresholds:
    gpu_temp_c: float = 86.0
    vram_fraction: float = 0.98
    ram_fraction: float = 0.95


def parse_nvidia_smi_csv(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        util, mem_used, mem_total, temp, power = parts[:5]
        rows.append(
            {
                "gpu_utilization_pct": _float(util),
                "vram_used_mb": _float(mem_used),
                "vram_total_mb": _float(mem_total),
                "temperature_c": _float(temp),
                "power_w": _float(power),
            }
        )
    return rows


def sample_nvidia_smi() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        kwargs: Dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": 10,
            "check": False,
        }
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.run(command, **kwargs)
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    if proc.returncode != 0:
        return {"available": False, "error": (proc.stderr or proc.stdout or "").strip(), "gpus": []}
    return {"available": True, "gpus": parse_nvidia_smi_csv(proc.stdout)}


def sample_system_ram() -> Dict[str, Any]:
    if not hasattr(ctypes, "windll"):
        return {"available": False}

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    memory = MEMORYSTATUSEX()
    memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if not ok:
        return {"available": False}
    total = float(memory.ullTotalPhys or 0)
    available = float(memory.ullAvailPhys or 0)
    used = max(0.0, total - available)
    return {
        "available": True,
        "ram_used_mb": round(used / 1024 / 1024, 3),
        "ram_total_mb": round(total / 1024 / 1024, 3),
        "ram_used_fraction": used / total if total else None,
        "memory_load_pct": float(memory.dwMemoryLoad),
    }


def evaluate_abort(sample: Dict[str, Any], thresholds: AbortThresholds) -> str:
    for gpu in ((sample.get("gpu") or {}).get("gpus") or []):
        temp = _float(gpu.get("temperature_c"))
        if temp is not None and temp >= thresholds.gpu_temp_c:
            return f"GPU temperature {temp:.1f}C exceeded abort threshold {thresholds.gpu_temp_c:.1f}C."
        used = _float(gpu.get("vram_used_mb"))
        total = _float(gpu.get("vram_total_mb"))
        if used is not None and total and used / total >= thresholds.vram_fraction:
            return f"VRAM usage {used:.0f}/{total:.0f} MB exceeded abort threshold {thresholds.vram_fraction:.0%}."
    ram = sample.get("system_ram") or {}
    ram_fraction = _float(ram.get("ram_used_fraction"))
    if ram_fraction is not None and ram_fraction >= thresholds.ram_fraction:
        return f"System RAM usage {ram_fraction:.0%} exceeded abort threshold {thresholds.ram_fraction:.0%}."
    return ""


def sample_resources() -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": sample_nvidia_smi(),
        "system_ram": sample_system_ram(),
    }


class ResourceMonitor:
    def __init__(self, path: Path | str, config: BenchmarkConfig):
        self.path = Path(path)
        self.interval_seconds = max(1.0, float(config.monitoring_interval_seconds or 5.0))
        self.thresholds = AbortThresholds(
            gpu_temp_c=float(config.monitoring_gpu_temp_abort_c or 86.0),
            vram_fraction=float(config.monitoring_vram_abort_fraction or 0.98),
            ram_fraction=float(config.monitoring_ram_abort_fraction or 0.95),
        )
        self.abort_reason = ""
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def sample_once(self) -> Dict[str, Any]:
        sample = sample_resources()
        reason = evaluate_abort(sample, self.thresholds)
        if reason and not self.abort_reason:
            self.abort_reason = reason
        self.samples += 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**sample, "abort_reason": reason}, default=str) + "\n")
        return sample

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_seconds)


def summarize_call_metrics(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = []
    chars_per_second = []
    tokens_per_second = []
    malformed_or_repaired = 0
    for decision in decisions:
        output = decision.get("output") or {}
        latency = _float(output.get("_api_latency_seconds"))
        if latency is not None:
            latencies.append(latency)
        cps = _float(output.get("_api_chars_per_second"))
        if cps is not None:
            chars_per_second.append(cps)
        tps = _float(output.get("_api_output_tokens_per_second"))
        if tps is not None:
            tokens_per_second.append(tps)
        if output.get("_api_retry_count") or (output.get("_allocation_repair") or {}).get("attempted"):
            malformed_or_repaired += 1
    calls = len([item for item in decisions if (item.get("output") or {}).get("_api_status") == "ok"])
    return {
        "calls_with_latency": len(latencies),
        "avg_call_latency_seconds": _mean(latencies),
        "max_call_latency_seconds": max(latencies) if latencies else None,
        "avg_chars_per_second": _mean(chars_per_second),
        "avg_output_tokens_per_second": _mean(tokens_per_second),
        "json_repair_or_retry_rate": malformed_or_repaired / calls if calls else 0.0,
    }


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None
