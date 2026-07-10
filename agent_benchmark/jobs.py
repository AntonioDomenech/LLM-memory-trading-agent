from __future__ import annotations

import threading
import time
import uuid
from datetime import date, datetime, time as clock_time, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from .benchmark_engine import BenchmarkEngine
from .local_provider import with_verified_local_runtime_identity
from .schemas import BenchmarkConfig, PortfolioBook, SecretConfig, model_to_dict
from .storage import BenchmarkStore


class JobControl:
    def __init__(self, store: BenchmarkStore):
        self.store = store
        self._pause: Dict[str, threading.Event] = {}
        self._cancel: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def register(self, run_id: str) -> None:
        with self._lock:
            pause = threading.Event()
            pause.set()
            self._pause[run_id] = pause
            self._cancel[run_id] = threading.Event()

    def checkpoint(self, run_id: str, phase: str, progress: Dict[str, Any]) -> None:
        self.store.update_benchmark_run(run_id, status="running", phase=phase, progress=progress)

    def should_cancel(self, run_id: str) -> bool:
        event = self._cancel.get(run_id)
        return bool(event and event.is_set())

    def wait_if_paused(self, run_id: str) -> None:
        event = self._pause.get(run_id)
        if event:
            event.wait()

    def pause(self, run_id: str) -> bool:
        event = self._pause.get(run_id)
        if not event:
            return False
        event.clear()
        self.store.update_benchmark_run(run_id, status="paused", progress={"message": "Paused by user"})
        return True

    def resume(self, run_id: str) -> bool:
        event = self._pause.get(run_id)
        if not event:
            return False
        event.set()
        self.store.update_benchmark_run(run_id, status="running", progress={"message": "Resumed"})
        return True

    def cancel(self, run_id: str) -> bool:
        event = self._cancel.get(run_id)
        if not event:
            return False
        event.set()
        pause = self._pause.get(run_id)
        if pause:
            pause.set()
        self.store.update_benchmark_run(run_id, status="cancelling", progress={"message": "Cancellation requested"})
        return True

    def release(self, run_id: str) -> None:
        with self._lock:
            self._pause.pop(run_id, None)
            self._cancel.pop(run_id, None)


class BenchmarkJobManager:
    def __init__(self):
        self.store = BenchmarkStore()
        self.control = JobControl(self.store)
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def start(self, config: BenchmarkConfig, secrets: SecretConfig, *, dry_run: bool = False) -> Dict[str, Any]:
        with self._lock:
            active = [
                run for run in self.store.list_benchmark_runs(limit=20)
                if run["status"] in {"queued", "running", "paused", "cancelling"}
            ]
            if active:
                raise RuntimeError(f"Run {active[0]['id']} is already active. Finish, cancel, or resume it before starting another.")
            run_id = str(uuid.uuid4())
            config_payload = config.model_dump() if hasattr(config, "model_dump") else config.dict()
            self.store.create_benchmark_run(run_id, config_payload)
            self.control.register(run_id)
            thread = threading.Thread(target=self._worker, args=(run_id, config, secrets, dry_run), daemon=True)
            self._threads[run_id] = thread
            thread.start()
        return self.store.get_benchmark_run(run_id) or {"id": run_id, "status": "queued"}

    def _worker(self, run_id: str, config: BenchmarkConfig, secrets: SecretConfig, dry_run: bool) -> None:
        engine = BenchmarkEngine()
        try:
            engine.run(run_id=run_id, config=config, secrets=secrets, store=self.store, control=self.control, dry_run=dry_run)
        except Exception as exc:
            self.store.append_benchmark_event(run_id, "error", "run_failed", {"error": str(exc)})
            self.store.update_benchmark_run(run_id, status="failed", phase="failed", error=str(exc), progress={"message": str(exc)}, finished=True)
        finally:
            engine.close()
            self.control.release(run_id)

    def pause(self, run_id: str) -> Dict[str, Any]:
        if not self.control.pause(run_id):
            raise KeyError(run_id)
        return self.store.get_benchmark_run(run_id) or {"id": run_id}

    def resume(self, run_id: str) -> Dict[str, Any]:
        if not self.control.resume(run_id):
            raise KeyError(run_id)
        return self.store.get_benchmark_run(run_id) or {"id": run_id}

    def cancel(self, run_id: str) -> Dict[str, Any]:
        if not self.control.cancel(run_id):
            raise KeyError(run_id)
        return self.store.get_benchmark_run(run_id) or {"id": run_id}


class LiveScheduler:
    def __init__(self, manager: BenchmarkJobManager):
        self.manager = manager
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._config: Optional[BenchmarkConfig] = None
        self._secrets: Optional[SecretConfig] = None
        self._dry_run = False
        self._book: Optional[PortfolioBook] = None
        self.last_snapshot: Dict[str, Any] | None = None

    def start(self, config: BenchmarkConfig, secrets: SecretConfig, *, dry_run: bool = False) -> Dict[str, Any]:
        if self._thread and self._thread.is_alive():
            return self.status()
        self._config = config
        self._secrets = secrets
        self._dry_run = dry_run
        if config.memory_online_stream_id:
            self._book = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        return self.status()

    def snapshot(self, config: BenchmarkConfig, secrets: SecretConfig, *, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
        session = self._market_session()
        if not session["market_open"] and not force:
            self.last_snapshot = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "skipped",
                "reason": session["reason"],
                "market": session,
            }
            return {
                "status": "skipped",
                "phase": "live",
                "message": session["message"],
                "market": session,
                "last_snapshot": self.last_snapshot,
            }

        active = [
            run for run in self.manager.store.list_benchmark_runs(limit=20)
            if run["status"] in {"queued", "running", "paused", "cancelling"}
        ]
        if active:
            raise RuntimeError(f"Run {active[0]['id']} is already active. Finish or cancel it before taking a live snapshot.")

        now = datetime.now(ZoneInfo("America/New_York"))
        run_id = str(uuid.uuid4())
        live_config = config.model_copy(deep=True) if hasattr(config, "model_copy") else BenchmarkConfig(**config.dict())
        live_config.run_preset = config.run_preset
        if live_config.evaluation_mode == "live_learning" and not dry_run:
            live_config = with_verified_local_runtime_identity(live_config)
        payload = live_config.model_dump() if hasattr(live_config, "model_dump") else live_config.dict()
        payload["live_started_at"] = now.isoformat(timespec="seconds")
        self.manager.store.create_benchmark_run(run_id, payload)

        engine = BenchmarkEngine()
        try:
            result = engine.run_live_snapshot(
                run_id=run_id,
                config=live_config,
                secrets=secrets,
                store=self.manager.store,
                portfolio_book=None if live_config.memory_online_stream_id else self._book,
                dry_run=dry_run,
                timestamp=now,
            )
            if not dry_run:
                self._book = result.get("portfolio") or self._book
            run = self.manager.store.get_benchmark_run(run_id) or {"id": run_id, "status": "completed"}
            self.last_snapshot = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "status": run.get("status"),
                "market": session,
                "portfolio": model_to_dict(self._book) if self._book else None,
            }
            return run
        except Exception as exc:
            self.manager.store.append_benchmark_event(run_id, "live", "live_snapshot_failed", {"error": str(exc), "market": session})
            self.manager.store.update_benchmark_run(run_id, status="failed", phase="live", error=str(exc), progress={"message": str(exc)}, finished=True)
            self.last_snapshot = {"created_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "status": "failed", "error": str(exc), "market": session}
            raise
        finally:
            engine.close()

    def status(self) -> Dict[str, Any]:
        running = bool(self._thread and self._thread.is_alive() and not self._stop.is_set())
        session = self._market_session()
        return {
            "running": running,
            "frequency": self._config.live_frequency if self._config else "hourly",
            "last_snapshot": self.last_snapshot,
            "market": session,
            "portfolio": model_to_dict(self._book) if self._book else None,
            "message": (
                (
                    "Daily-open local paper benchmark is active."
                    if self._config and self._config.live_frequency == "daily_open"
                    else "Hourly local paper benchmark is active and will run during US market hours."
                )
                if running
                else "Live scheduler is stopped."
            ),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._config and self._config.live_frequency == "daily_open":
                delay = self._seconds_until_daily_open_snapshot()
                if delay > 1.0:
                    self._stop.wait(min(delay, 60 * 60))
                    continue
            if self._config and self._secrets:
                try:
                    self.snapshot(self._config, self._secrets, dry_run=self._dry_run, force=False)
                except Exception as exc:
                    self.last_snapshot = {"created_at": datetime.now(timezone.utc).isoformat(), "error": str(exc)}
            self._stop.wait(60 if self._config and self._config.live_frequency == "daily_open" else 60 * 60)

    def _seconds_until_daily_open_snapshot(self, now: datetime | None = None) -> float:
        ny_tz = ZoneInfo("America/New_York")
        current = (now or datetime.now(ny_tz)).astimezone(ny_tz)
        successful_today = bool(
            self.last_snapshot
            and str(self.last_snapshot.get("created_at") or "")[:10] == current.date().isoformat()
            and self.last_snapshot.get("status") == "completed"
        )
        target_date = current.date()
        target = datetime.combine(target_date, clock_time(9, 35), ny_tz)
        if successful_today or current.time() > clock_time(10, 0):
            target_date += timedelta(days=1)
        while target_date.weekday() >= 5 or target_date in _nyse_holidays(target_date.year):
            target_date += timedelta(days=1)
        target = datetime.combine(target_date, clock_time(9, 35), ny_tz)
        if target_date == current.date() and current >= target:
            return 0.0
        return max(0.0, (target - current).total_seconds())

    def _market_session(self, now: datetime | None = None) -> Dict[str, Any]:
        ny_tz = ZoneInfo("America/New_York")
        current = (now or datetime.now(ny_tz)).astimezone(ny_tz)
        today = current.date()
        open_at = datetime.combine(today, clock_time(9, 30), ny_tz)
        close_at = datetime.combine(today, clock_time(16, 0), ny_tz)
        holiday = today in _nyse_holidays(today.year)
        weekday = current.weekday() < 5
        market_open = weekday and not holiday and open_at <= current < close_at
        if market_open:
            reason = "open"
            message = "US market is open. A live snapshot can run now."
        elif not weekday:
            reason = "weekend"
            message = "US market is closed for the weekend. The scheduler will wait for the next session."
        elif holiday:
            reason = "holiday"
            message = "US market is closed for a market holiday. The scheduler will wait for the next session."
        elif current < open_at:
            reason = "before_open"
            message = "US market has not opened yet. The scheduler is waiting."
        else:
            reason = "after_close"
            message = "US market is closed for the day. The scheduler will wait for the next session."
        return {
            "timezone": "America/New_York",
            "now": current.isoformat(timespec="seconds"),
            "regular_open": open_at.isoformat(timespec="seconds"),
            "regular_close": close_at.isoformat(timespec="seconds"),
            "market_open": market_open,
            "reason": reason,
            "message": message,
        }


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    value = date(year, month, day)
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    value = date(year, month, 1)
    while value.weekday() != weekday:
        value += timedelta(days=1)
    return value + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    value = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while value.weekday() != weekday:
        value -= timedelta(days=1)
    return value


def _easter_date(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nyse_holidays(year: int) -> set[date]:
    return {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_date(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 6, 19),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(year, 12, 25),
    }
