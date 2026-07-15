"""Staged, bounded runner for the chronological exhaustion-expert experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .chronological_exhaustion_expert import (
    CAUSAL_ONLINE_MODE,
    FROZEN_CUTOFF_MODE,
    build_chronological_exhaustion_forecast,
    build_fixed_expert_signals,
    build_unfiltered_expert_targets,
    canonicalize_one_session_signals,
)
from .deterministic_aapl import (
    CostAssumptions,
    EvaluationPeriod,
    compare_ledgers,
)
from .unleveraged_aapl import (
    assert_unleveraged_ledger,
    canonical_context_frame,
    simulate_unleveraged_period,
)


CONTRACT_VERSION = "aapl-chronological-exhaustion-expert-v1"
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
VALIDATION_END = pd.Timestamp("2023-12-31")
FINAL_END = pd.Timestamp("2026-07-09")
DEVELOPMENT_START = pd.Timestamp("2005-01-01")
VALIDATION_START = pd.Timestamp("2019-01-01")
FINAL_START = pd.Timestamp("2024-01-01")
INITIAL_CASH = 1000.0
RUN_TIME_LIMIT_SECONDS = 3600.0
MIN_MATERIAL_ACTIVE_LOG_EDGE = 0.001
COST_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("base_5bps", 5.0),
    ("stress_10bps", 10.0),
)
CONTRACT_PATH = Path("docs/aapl_chronological_exhaustion_expert_v1.md")
IMPLEMENTATION_PATHS = (
    Path("agent_benchmark/chronological_exhaustion_expert.py"),
    Path("agent_benchmark/chronological_exhaustion_experiment.py"),
    Path("agent_benchmark/deterministic_aapl.py"),
    Path("agent_benchmark/unleveraged_aapl.py"),
)
PHYSICAL_SNAPSHOT_COLUMNS = (
    "date",
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "spy_adj_close",
    "qqq_adj_close",
)
STAGE_SESSION_COVERAGE: dict[str, dict[str, Any]] = {
    "2018-12-31": {
        "first_session": "1999-03-10",
        "last_session": "2018-12-31",
        "observations": 4986,
        "date_sequence_sha256": (
            "3b8844f939269760624816e166ebf15705c27549aa89dd578ddb66bd50d87df9"
        ),
    },
    "2023-12-29": {
        "first_session": "1999-03-10",
        "last_session": "2023-12-29",
        "observations": 6244,
        "date_sequence_sha256": (
            "77098a2d35b6cee78ccb100514e599ee4ef6dac55738e7084b02d0e0dd0b63c1"
        ),
    },
    "2026-07-09": {
        "first_session": "1999-03-10",
        "last_session": "2026-07-09",
        "observations": 6875,
        "date_sequence_sha256": (
            "b88df14b4ec60534ace68645ee19c8a0b7d03d0c2c1829a3ad48f8a0a24c9299"
        ),
    },
}
STAGE_BOUNDED_RESULT_SHA256 = {
    "2018-12-31": (
        "sha256:31b56551b8d1b837f2e69178ab7f206b6bf3c19d11d9d47be0e33cf501db3f45"
    ),
    "2023-12-29": (
        "sha256:3b5e02acaa69a56fa47a0fd34275472d226b62c0f61741c13d3680b239b82535"
    ),
    "2026-07-09": (
        "sha256:c01447f975d4a90e49c315f23177f357966363b1ec4790632fa54c0dee250b21"
    ),
}
REQUIRED_PARENT_PAYLOADS = {
    "development": frozenset(
        {
            ".gitattributes",
            "report.json",
            "development_prices_through_2018.csv",
            "development_metrics.json",
            "development_gate_report.json",
            "development_checkpoint_through_2018.json",
            "input_provenance.json",
        }
    ),
    "validation": frozenset(
        {
            ".gitattributes",
            "report.json",
            "validation_prices_through_2023.csv",
            "validation_frozen_metrics.json",
            "validation_gate_report.json",
            "validation_checkpoint_through_2023.json",
            "input_provenance.json",
        }
    ),
}
PARENT_GATE_REPORT_FILENAME = {
    "development": "development_gate_report.json",
    "validation": "validation_gate_report.json",
}
PRICE_QUERY = """
SELECT
  CAST(date AS DATE) AS date,
  CAST(aapl_open AS DOUBLE) AS aapl_open,
  CAST(aapl_close AS DOUBLE) AS aapl_close,
  CAST(aapl_adj_close AS DOUBLE) AS aapl_adj_close,
  CAST(spy_adj_close AS DOUBLE) AS spy_adj_close,
  CAST(qqq_adj_close AS DOUBLE) AS qqq_adj_close
FROM read_csv_auto(?, header = true)
ORDER BY CAST(date AS DATE)
""".strip()


class ChronologicalExhaustionExperimentError(RuntimeError):
    """Raised when the frozen experiment contract is violated."""


class _Deadline:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started = float(clock())

    def elapsed(self) -> float:
        return float(self._clock()) - self._started

    def check(self, location: str) -> None:
        if self.elapsed() > RUN_TIME_LIMIT_SECONDS:
            raise ChronologicalExhaustionExperimentError(
                f"Experiment exceeded {RUN_TIME_LIMIT_SECONDS:.0f}s at {location}"
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _safe_run_id(value: str | None, *, prefix: str) -> str:
    result = value or (
        f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", result):
        raise ChronologicalExhaustionExperimentError("run_id is not a safe name")
    return result


def _manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(payload), "manifest_sha256": _sha256(_canonical_json_bytes(payload))}


def _seal_bundle(
    run_dir: Path,
    payloads: Mapping[str, bytes],
    *,
    before_promote: Callable[[], None] | None = None,
) -> dict[str, str]:
    if run_dir.exists():
        raise ChronologicalExhaustionExperimentError(
            f"Artifact directory already exists: {run_dir}"
        )
    temporary = run_dir.with_name(f".{run_dir.name}.{uuid.uuid4().hex}.sealing")
    if temporary.exists():
        raise ChronologicalExhaustionExperimentError(
            f"Temporary artifact directory exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    promoted = False
    try:
        checksums = {name: _sha256(data) for name, data in sorted(payloads.items())}
        for name, data in payloads.items():
            destination = temporary / name
            if destination.parent != temporary:
                raise ChronologicalExhaustionExperimentError(
                    "Artifact filenames must be flat"
                )
            _atomic_write(destination, data)
        _atomic_write(temporary / "checksums.json", _pretty_json_bytes(checksums))
        for name, expected in checksums.items():
            if _sha256((temporary / name).read_bytes()) != expected:
                raise ChronologicalExhaustionExperimentError(
                    f"Artifact checksum mismatch: {name}"
                )
        if before_promote is not None:
            before_promote()
        temporary.replace(run_dir)
        promoted = True
        return checksums
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True
    ).stdout


def _git_text(root: Path, *args: str) -> str:
    return _git_bytes(root, *args).decode("utf-8", errors="strict").strip()


def _clean_git_identity(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    try:
        actual = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
        status = _git_text(root, "status", "--porcelain", "--untracked-files=all")
        branch = _git_text(root, "symbolic-ref", "--quiet", "--short", "HEAD")
        commit = _git_text(root, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise ChronologicalExhaustionExperimentError(
            "Stage requires a valid Git repository"
        ) from exc
    if actual != root:
        raise ChronologicalExhaustionExperimentError(
            "repo_root must be the actual Git root"
        )
    if status:
        raise ChronologicalExhaustionExperimentError(
            "Stage requires a completely clean worktree"
        )
    if not branch or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ChronologicalExhaustionExperimentError(
            "Stage requires an attached valid Git commit"
        )
    tracked_hashes: dict[str, str] = {}
    for relative in (*IMPLEMENTATION_PATHS, CONTRACT_PATH):
        name = relative.as_posix()
        try:
            _git_bytes(root, "ls-files", "--error-unmatch", "--", name)
            committed = _git_bytes(root, "show", f"HEAD:{name}")
        except subprocess.CalledProcessError as exc:
            raise ChronologicalExhaustionExperimentError(
                f"Frozen dependency is not tracked: {name}"
            ) from exc
        if not (root / relative).is_file():
            raise ChronologicalExhaustionExperimentError(
                f"Frozen dependency is missing locally: {name}"
            )
        # The clean-worktree check above proves the filtered working file
        # corresponds to HEAD. Hash the canonical Git blob so Windows CRLF
        # checkout policy cannot create a false cross-stage mismatch.
        tracked_hashes[name] = _sha256(committed)
    return {
        "branch": branch,
        "commit": commit,
        "dirty": False,
        "tracked_dependency_sha256": tracked_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "duckdb": __import__("duckdb").__version__,
        },
    }


def _tracked_input_identity(repo_root: Path, path: Path) -> dict[str, str]:
    root = repo_root.resolve()
    source = path.resolve()
    try:
        relative = source.relative_to(root).as_posix()
        _git_bytes(root, "ls-files", "--error-unmatch", "--", relative)
        committed = _git_bytes(root, "show", f"HEAD:{relative}")
        local = source.read_bytes()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise ChronologicalExhaustionExperimentError(
            "Physical stage snapshot must be an exact tracked file at HEAD"
        ) from exc
    if committed != local:
        raise ChronologicalExhaustionExperimentError(
            "Physical stage snapshot differs from its committed blob"
        )
    return {"path": relative, "sha256": _sha256(local)}


def _validated_payload_inventory(
    value: Any, *, expected_stage: str
) -> dict[str, str]:
    required = REQUIRED_PARENT_PAYLOADS.get(expected_stage, frozenset())
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or any(
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(expected, str)
            for filename, expected in value.items()
        )
    ):
        raise ChronologicalExhaustionExperimentError(
            "Prior manifest has an incomplete or unsafe payload inventory"
        )
    return dict(value)


def _require_pass_evidence_consistency(
    manifest: Mapping[str, Any],
    gate: Any,
    report: Any,
    *,
    expected_stage: str,
) -> None:
    if (
        manifest.get("stage_pass") is not True
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
        or not isinstance(report, dict)
        or report.get("contract_version") != CONTRACT_VERSION
        or report.get("stage") != expected_stage
        or report.get("run_id") != manifest.get("run_id")
        or report.get("gate_report") != gate
    ):
        raise ChronologicalExhaustionExperimentError(
            "Prior manifest, report, and gate result do not agree on a pass"
        )


def _validated_prior_manifest(
    *, repo_root: Path, path: Path, expected_stage: str
) -> dict[str, Any]:
    root = repo_root.resolve()
    manifest_path = path.resolve()
    try:
        relative = manifest_path.relative_to(root).as_posix()
        _git_bytes(root, "ls-files", "--error-unmatch", "--", relative)
        committed = _git_bytes(root, "show", f"HEAD:{relative}")
        local = manifest_path.read_bytes()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise ChronologicalExhaustionExperimentError(
            "Prior manifest must be an exact tracked file at HEAD"
        ) from exc
    if committed != local:
        raise ChronologicalExhaustionExperimentError(
            "Prior manifest differs from its committed blob"
        )
    try:
        value = json.loads(local.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChronologicalExhaustionExperimentError(
            "Prior manifest is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ChronologicalExhaustionExperimentError(
            "Prior manifest must be a JSON object"
        )
    payload = dict(value)
    recorded = payload.pop("manifest_sha256", None)
    if recorded != _sha256(_canonical_json_bytes(payload)):
        raise ChronologicalExhaustionExperimentError(
            "Prior manifest self-hash is invalid"
        )
    if (
        value.get("contract_version") != CONTRACT_VERSION
        or value.get("stage") != expected_stage
        or value.get("stage_pass") is not True
    ):
        raise ChronologicalExhaustionExperimentError(
            "Prior manifest did not pass the required stage"
        )
    hashes = _validated_payload_inventory(
        value.get("payload_sha256"), expected_stage=expected_stage
    )
    verified_payloads: dict[str, bytes] = {}
    for filename, expected in hashes.items():
        artifact = manifest_path.parent / filename
        try:
            artifact_relative = artifact.resolve().relative_to(root).as_posix()
            _git_bytes(root, "ls-files", "--error-unmatch", "--", artifact_relative)
            artifact_committed = _git_bytes(root, "show", f"HEAD:{artifact_relative}")
            artifact_local = artifact.read_bytes()
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            raise ChronologicalExhaustionExperimentError(
                f"Prior artifact is not tracked: {filename}"
            ) from exc
        if artifact_committed != artifact_local or _sha256(artifact_local) != expected:
            raise ChronologicalExhaustionExperimentError(
                f"Prior artifact checksum changed: {filename}"
            )
        verified_payloads[filename] = artifact_local
    try:
        gate = json.loads(
            verified_payloads[PARENT_GATE_REPORT_FILENAME[expected_stage]].decode(
                "utf-8"
            )
        )
        report = json.loads(verified_payloads["report.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChronologicalExhaustionExperimentError(
            "Prior stage pass evidence is unreadable"
        ) from exc
    _require_pass_evidence_consistency(
        value, gate, report, expected_stage=expected_stage
    )
    return value


def _require_dependency_continuity(
    current_git_identity: Mapping[str, Any], parent: Mapping[str, Any]
) -> None:
    current = current_git_identity.get("tracked_dependency_sha256")
    previous = parent.get("git_identity", {}).get("tracked_dependency_sha256")
    current_runtime = current_git_identity.get("runtime_versions")
    previous_runtime = parent.get("git_identity", {}).get("runtime_versions")
    if (
        not isinstance(current, dict)
        or current != previous
        or not isinstance(current_runtime, dict)
        or current_runtime != previous_runtime
    ):
        raise ChronologicalExhaustionExperimentError(
            "Frozen implementation dependencies changed after the parent stage"
        )


def _validate_physical_snapshot_rows(
    source: Path, *, required_last_session: pd.Timestamp
) -> dict[str, Any]:
    dates: list[pd.Timestamp] = []
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PHYSICAL_SNAPSHOT_COLUMNS:
                raise ChronologicalExhaustionExperimentError(
                    "Physical stage snapshot has unexpected columns"
                )
            previous: pd.Timestamp | None = None
            for row in reader:
                if None in row or any(row[column] in (None, "") for column in row):
                    raise ChronologicalExhaustionExperimentError(
                        "Physical stage snapshot contains an incomplete raw row"
                    )
                current = pd.Timestamp(row["date"])
                if (
                    pd.isna(current)
                    or current.tzinfo is not None
                    or current.time() != datetime.min.time()
                    or current.date().isoformat() != row["date"]
                    or (previous is not None and current <= previous)
                ):
                    raise ChronologicalExhaustionExperimentError(
                        "Physical stage snapshot dates are not strict ISO sessions"
                    )
                for column in PHYSICAL_SNAPSHOT_COLUMNS[1:]:
                    value = float(row[column])
                    if not math.isfinite(value) or value <= 0.0:
                        raise ChronologicalExhaustionExperimentError(
                            "Physical stage snapshot contains an invalid raw price"
                        )
                dates.append(current)
                previous = current
    except ChronologicalExhaustionExperimentError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ChronologicalExhaustionExperimentError(
            "Physical stage snapshot raw validation failed"
        ) from exc
    if not dates or dates[-1] != required_last_session:
        raise ChronologicalExhaustionExperimentError(
            "Physical stage snapshot does not end on the required session"
        )
    session_payload = "".join(
        f"{value.date().isoformat()}\n" for value in dates
    ).encode("ascii")
    observed = {
        "first_session": dates[0].date().isoformat(),
        "last_session": dates[-1].date().isoformat(),
        "observations": len(dates),
        "date_sequence_sha256": hashlib.sha256(session_payload).hexdigest(),
    }
    expected = STAGE_SESSION_COVERAGE.get(required_last_session.date().isoformat())
    if observed != expected:
        raise ChronologicalExhaustionExperimentError(
            "Physical stage snapshot does not match the frozen raw session sequence"
        )
    return observed


def load_bounded_prices(
    path: Path, *, end: pd.Timestamp, required_last_session: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a physically stage-bounded snapshot, never a future-filled source."""

    source = path.resolve()
    if not source.is_file():
        raise ChronologicalExhaustionExperimentError(
            f"Price artifact does not exist: {source}"
        )
    raw_coverage = _validate_physical_snapshot_rows(
        source, required_last_session=required_last_session
    )
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            raw = connection.execute(
                PRICE_QUERY,
                [str(source)],
            ).fetchdf()
        finally:
            connection.close()
    except Exception as exc:
        raise ChronologicalExhaustionExperimentError(
            "Bounded price query failed"
        ) from exc
    try:
        frame = canonical_context_frame(raw)
    except (TypeError, ValueError) as exc:
        raise ChronologicalExhaustionExperimentError(
            "Bounded price rows are not canonical"
        ) from exc
    if frame.empty or frame.index.max() != required_last_session:
        raise ChronologicalExhaustionExperimentError(
            "Physical stage snapshot does not end on the required session"
        )
    if frame.index.min() > pd.Timestamp("1999-06-01") or frame.index.max() > end:
        raise ChronologicalExhaustionExperimentError(
            "Physical stage snapshot contains an unauthorized date range"
        )
    session_payload = "".join(
        f"{value.date().isoformat()}\n" for value in frame.index
    ).encode("ascii")
    observed_coverage = {
        "first_session": frame.index.min().date().isoformat(),
        "last_session": frame.index.max().date().isoformat(),
        "observations": int(len(frame)),
        "date_sequence_sha256": hashlib.sha256(session_payload).hexdigest(),
    }
    if observed_coverage != raw_coverage:
        raise ChronologicalExhaustionExperimentError(
            "Canonical stage rows differ from the validated raw snapshot"
        )
    payload = _frame_csv_bytes(frame.reset_index(names="date"))
    bounded_result_sha256 = _sha256(payload)
    expected_result_sha256 = STAGE_BOUNDED_RESULT_SHA256.get(
        required_last_session.date().isoformat()
    )
    if bounded_result_sha256 != expected_result_sha256:
        raise ChronologicalExhaustionExperimentError(
            "Physical stage snapshot prices do not match the preregistered source"
        )
    return frame, {
        "source_type": "physically_bounded_local_csv_duckdb_query",
        "source_path": str(source),
        "query": PRICE_QUERY,
        "query_parameters": [str(source)],
        "bounded_first_date": frame.index.min().date().isoformat(),
        "bounded_last_date": frame.index.max().date().isoformat(),
        "bounded_rows": int(len(frame)),
        "bounded_result_sha256": bounded_result_sha256,
        "expected_bounded_result_sha256": expected_result_sha256,
        "session_coverage": raw_coverage,
        "physical_snapshot_has_later_rows": False,
        "rows_after_bound_returned": False,
        "network_access": False,
    }


def _period_return(
    ledger: pd.DataFrame, period: EvaluationPeriod, *, initial_cash: float
) -> float:
    dates = pd.to_datetime(ledger["fill_date"])
    equity = ledger["equity"].astype(float).reset_index(drop=True)
    indices = np.flatnonzero(
        (dates >= pd.Timestamp(period.start)) & (dates <= pd.Timestamp(period.end))
    )
    if not len(indices):
        raise ChronologicalExhaustionExperimentError(
            f"Ledger lacks period {period.name}"
        )
    first = int(indices[0])
    last = int(indices[-1])
    start_equity = initial_cash if first == 0 else float(equity.iloc[first - 1])
    return float(equity.iloc[last] / start_equity - 1.0)


def _episode_rows(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
) -> list[dict[str, Any]]:
    data = canonical_context_frame(frame)
    values = pd.to_numeric(target.reindex(data.index), errors="raise").astype(float)
    opens = data["aapl_adj_open"].to_numpy(dtype=float)
    dates = data.index
    cost = float(cost_bps) / 10_000.0
    friction = math.log((1.0 - cost) / (1.0 + cost))
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(values.to_numpy() == 0.0):
        if index + 2 >= len(data):
            continue
        if index > 0 and values.iloc[index - 1] == 0.0:
            continue
        entry = dates[index + 1]
        exit_date = dates[index + 2]
        if entry < start or exit_date > end:
            continue
        raw_edge = math.log(opens[index + 1] / opens[index + 2])
        net_edge = raw_edge + friction
        rows.append(
            {
                "decision_date": dates[index].date().isoformat(),
                "entry_date": entry.date().isoformat(),
                "exit_date": exit_date.date().isoformat(),
                "raw_active_log_edge": raw_edge,
                "net_active_log_edge": net_edge,
                "win": bool(net_edge > 0.0),
            }
        )
    return rows


def _periods_for_years(first: int, last: int) -> tuple[EvaluationPeriod, ...]:
    return tuple(
        EvaluationPeriod(str(year), f"{year}-01-01", f"{year}-12-31")
        for year in range(first, last + 1)
    )


def _evaluate_policy(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    periods: Sequence[EvaluationPeriod],
    cost_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not periods:
        raise ChronologicalExhaustionExperimentError("No evaluation periods")
    full = EvaluationPeriod("continuous", periods[0].start, periods[-1].end)
    costs = CostAssumptions(slippage_bps=cost_bps, annual_margin_rate=0.0)
    data_index = canonical_context_frame(frame).index
    evaluation_target = pd.to_numeric(
        target.reindex(data_index), errors="raise"
    ).astype(float)
    # Both accounts begin with the same all-in AAPL purchase. Decisions before
    # the evaluation account exists cannot make it start in cash for free.
    evaluation_target.loc[
        evaluation_target.index < pd.Timestamp(full.start)
    ] = 1.0
    benchmark_target = pd.Series(1.0, index=data_index)
    strategy = simulate_unleveraged_period(
        frame, evaluation_target, full, costs, initial_cash=INITIAL_CASH
    )
    benchmark = simulate_unleveraged_period(
        frame, benchmark_target, full, costs, initial_cash=INITIAL_CASH
    )
    comparison = compare_ledgers(strategy, benchmark, initial_cash=INITIAL_CASH)
    episodes = _episode_rows(
        frame,
        evaluation_target,
        start=pd.Timestamp(periods[0].start),
        end=pd.Timestamp(periods[-1].end),
        cost_bps=cost_bps,
    )
    episode_edges = np.asarray(
        [row["net_active_log_edge"] for row in episodes], dtype=float
    )
    ledger_total_active_edge = float(
        math.log1p(comparison["strategy"]["total_return"])
        - math.log1p(comparison["aapl_buy_hold"]["total_return"])
    )
    attributed_total_active_edge = float(np.sum(episode_edges))
    identity_error = ledger_total_active_edge - attributed_total_active_edge
    if abs(identity_error) > 1e-10:
        raise ChronologicalExhaustionExperimentError(
            "Ledger active edge does not reconcile to executed cash episodes"
        )
    returns: dict[str, Any] = {}
    for period in periods:
        strategy_return = _period_return(strategy, period, initial_cash=INITIAL_CASH)
        benchmark_return = _period_return(benchmark, period, initial_cash=INITIAL_CASH)
        ledger_boundary_edge = float(
            math.log1p(strategy_return) - math.log1p(benchmark_return)
        )
        attributed_edge = float(
            np.sum(
                [
                    row["net_active_log_edge"]
                    for row in episodes
                    if pd.Timestamp(period.start)
                    <= pd.Timestamp(row["entry_date"])
                    <= pd.Timestamp(period.end)
                ]
            )
        )
        returns[period.name] = {
            "strategy_return": strategy_return,
            "aapl_buy_hold_return": benchmark_return,
            "active_log_edge": attributed_edge,
            "ledger_boundary_active_log_edge": ledger_boundary_edge,
            "active_edge_attribution": "cash_episode_entry_open_date",
        }
    episode_frame = pd.DataFrame(episodes)
    result = {
        "cost_bps_per_changing_leg": float(cost_bps),
        "period": asdict(full),
        "comparison": comparison,
        "total_active_log_edge": ledger_total_active_edge,
        "attributed_episode_active_log_edge": attributed_total_active_edge,
        "episode_ledger_identity_error": identity_error,
        "periods": returns,
        "cash_episode_count": int(len(episodes)),
        "cash_episode_win_rate": float(np.mean(episode_edges > 0.0))
        if len(episode_edges)
        else None,
        "mean_cash_episode_edge": float(np.mean(episode_edges))
        if len(episode_edges)
        else None,
        "median_cash_episode_edge": float(np.median(episode_edges))
        if len(episode_edges)
        else None,
        "maximum_positive_episode_share": (
            float(np.max(episode_edges[episode_edges > 0.0]) / np.sum(episode_edges[episode_edges > 0.0]))
            if np.any(episode_edges > 0.0)
            else None
        ),
        "no_leverage_proof": assert_unleveraged_ledger(strategy),
    }
    merged = pd.concat(
        [strategy.add_prefix("strategy_"), benchmark.add_prefix("buy_hold_")],
        axis=1,
    )
    merged["relative_wealth"] = (
        merged["strategy_equity"] / merged["buy_hold_equity"] - 1.0
    )
    return result, strategy, benchmark, episode_frame


def _policy_targets(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    administrative_start: pd.Timestamp | None = None,
) -> dict[str, pd.Series]:
    unfiltered = build_unfiltered_expert_targets(frame)
    targets = {
        "learner": forecast["target_exposure"].astype(float),
        "contextual": unfiltered[
            "unfiltered_contextual_target_exposure"
        ].astype(float),
        "weak_trend": unfiltered[
            "unfiltered_weak_trend_target_exposure"
        ].astype(float),
        "union": unfiltered["unfiltered_union_target_exposure"].astype(float),
        "always_long": unfiltered["always_long_target_exposure"].astype(float),
    }
    if administrative_start is None:
        return targets

    signals = build_fixed_expert_signals(frame)
    eligible = signals["stage_outcome_available"].astype(bool)

    def stage_target(candidate: pd.Series) -> pd.Series:
        allowed = candidate.astype(bool) & eligible
        allowed.loc[allowed.index < administrative_start] = False
        accepted = canonicalize_one_session_signals(allowed)
        return pd.Series(
            np.where(accepted, 0.0, 1.0),
            index=allowed.index,
            dtype=float,
        )

    # Account cooldowns restart only after earlier, non-executable candidates
    # are removed. Expert virtual lesson streams remain continuous.
    targets["learner"] = stage_target(
        forecast["combined_trusted_candidate_signal"]
    )
    targets["contextual"] = stage_target(
        signals["contextual_virtual_signal"]
    )
    targets["weak_trend"] = stage_target(
        signals["weak_trend_virtual_signal"]
    )
    targets["union"] = stage_target(
        signals["unfiltered_union_candidate_signal"]
    )
    return targets


def _evaluate_policy_set(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    periods: Sequence[EvaluationPeriod],
    administrative_start: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    targets = _policy_targets(
        frame, forecast, administrative_start=administrative_start
    )
    metrics: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    episodes: dict[str, pd.DataFrame] = {}
    for cost_name, cost_bps in COST_SCENARIOS:
        metrics[cost_name] = {}
        ledger_rows: list[pd.DataFrame] = []
        for policy_name, target in targets.items():
            stage_target = target.copy()
            if administrative_start is not None:
                # Defense in depth; candidates were already removed before
                # account-level cooldowns were recomputed.
                stage_target.loc[stage_target.index < administrative_start] = 1.0
            result, strategy, benchmark, episode_frame = _evaluate_policy(
                frame,
                stage_target,
                periods=periods,
                cost_bps=cost_bps,
            )
            metrics[cost_name][policy_name] = result
            strategy_copy = strategy.copy()
            strategy_copy.insert(0, "policy", policy_name)
            strategy_copy.insert(1, "ledger_role", "strategy")
            ledger_rows.append(strategy_copy)
            if policy_name == "always_long":
                benchmark_copy = benchmark.copy()
                benchmark_copy.insert(0, "policy", "aapl_buy_hold")
                benchmark_copy.insert(1, "ledger_role", "benchmark")
                ledger_rows.append(benchmark_copy)
            if policy_name == "learner":
                episodes[cost_name] = episode_frame
        ledgers[cost_name] = pd.concat(ledger_rows, ignore_index=True)
    return metrics, ledgers, episodes


def _annual_values(result: Mapping[str, Any], first: int, last: int) -> list[float]:
    return [
        float(result["periods"][str(year)]["active_log_edge"])
        for year in range(first, last + 1)
    ]


def _positive_concentration(values: Sequence[float]) -> float | None:
    positive = np.asarray([value for value in values if value > 0.0], dtype=float)
    if not len(positive):
        return None
    return float(np.max(positive) / np.sum(positive))


def _always_long_gate(metrics: Mapping[str, Any]) -> bool:
    for cost_name, _ in COST_SCENARIOS:
        result = metrics[cost_name]["always_long"]
        if (
            abs(float(result["total_active_log_edge"])) > 1e-12
            or abs(
                float(
                    result["comparison"]["relative_wealth_vs_aapl_buy_hold"]
                )
            )
            > 1e-12
        ):
            return False
    return True


def apply_development_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    gates: dict[str, bool] = {"always_long_matches_buy_hold": _always_long_gate(metrics)}
    for cost_name, _ in COST_SCENARIOS:
        result = metrics[cost_name]["learner"]
        annual = _annual_values(result, 2005, 2018)
        folds = [
            float(result["periods"][f"{start}_{start + 1}"]["active_log_edge"])
            for start in range(2005, 2019, 2)
        ]
        negative_years = [
            float(result["periods"][str(year)]["active_log_edge"])
            for year in (2008, 2015, 2018)
        ]
        concentration = _positive_concentration(annual)
        prefix = f"{cost_name}_"
        gates.update(
            {
                prefix + "positive_total_active_log_edge": float(
                    result["total_active_log_edge"]
                )
                > 0.0,
                prefix + "positive_relative_wealth": float(
                    result["comparison"]["relative_wealth_vs_aapl_buy_hold"]
                )
                > 0.0,
                prefix + "minimum_eight_cash_episodes": int(
                    result["cash_episode_count"]
                )
                >= 8,
                prefix + "minimum_eight_positive_years": int(
                    np.count_nonzero(np.asarray(annual) > 0.0)
                )
                >= 8,
                prefix + "minimum_four_positive_folds": int(
                    np.count_nonzero(np.asarray(folds) > 0.0)
                )
                >= 4,
                prefix + "positive_after_removing_best_year": float(
                    np.sum(annual) - np.max(annual)
                )
                > 0.0,
                prefix + "annual_positive_edge_not_concentrated": (
                    concentration is not None and concentration <= 0.50
                ),
                prefix + "positive_negative_year_aggregate": float(
                    np.sum(negative_years)
                )
                > 0.0,
                prefix + "two_of_three_negative_years_positive": int(
                    np.count_nonzero(np.asarray(negative_years) > 0.0)
                )
                >= 2,
            }
        )
    stress = metrics["stress_10bps"]["learner"]
    stress_union = metrics["stress_10bps"]["union"]
    gates.update(
        {
            "stress_10bps_episode_win_rate_at_least_55pct": (
                stress["cash_episode_win_rate"] is not None
                and float(stress["cash_episode_win_rate"]) >= 0.55
            ),
            "stress_10bps_positive_mean_episode_edge": (
                stress["mean_cash_episode_edge"] is not None
                and float(stress["mean_cash_episode_edge"]) > 0.0
            ),
            "stress_10bps_positive_median_episode_edge": (
                stress["median_cash_episode_edge"] is not None
                and float(stress["median_cash_episode_edge"]) > 0.0
            ),
            "learner_strictly_beats_union_at_10bps": float(
                stress["total_active_log_edge"]
            )
            > float(stress_union["total_active_log_edge"]),
        }
    )
    failures = sorted(name for name, passed in gates.items() if not passed)
    return {"passed": not failures, "gates": gates, "failures": failures}


def apply_validation_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    gates: dict[str, bool] = {"always_long_matches_buy_hold": _always_long_gate(metrics)}
    for cost_name, _ in COST_SCENARIOS:
        result = metrics[cost_name]["learner"]
        annual = _annual_values(result, 2019, 2023)
        prefix = f"{cost_name}_"
        gates.update(
            {
                prefix + "positive_total_active_log_edge": float(
                    result["total_active_log_edge"]
                )
                > 0.0,
                prefix + "minimum_three_positive_years": int(
                    np.count_nonzero(np.asarray(annual) > 0.0)
                )
                >= 3,
                prefix + "nonnegative_2022_active_log_edge": float(
                    result["periods"]["2022"]["active_log_edge"]
                )
                >= 0.0,
            }
        )
    stress = metrics["stress_10bps"]["learner"]
    gates.update(
        {
            "minimum_three_cash_episodes": int(stress["cash_episode_count"]) >= 3,
            "positive_mean_10bps_episode_edge": (
                stress["mean_cash_episode_edge"] is not None
                and float(stress["mean_cash_episode_edge"]) > 0.0
            ),
            "positive_median_10bps_episode_edge": (
                stress["median_cash_episode_edge"] is not None
                and float(stress["median_cash_episode_edge"]) > 0.0
            ),
            "episode_positive_edge_not_concentrated": (
                stress["maximum_positive_episode_share"] is not None
                and float(stress["maximum_positive_episode_share"]) <= 0.50
            ),
        }
    )
    failures = sorted(name for name, passed in gates.items() if not passed)
    return {"passed": not failures, "gates": gates, "failures": failures}


def apply_final_gates(
    frozen_metrics: Mapping[str, Any], lifetime_online_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    gates: dict[str, bool] = {
        "frozen_always_long_matches_buy_hold": _always_long_gate(frozen_metrics),
        "online_always_long_matches_buy_hold": _always_long_gate(
            lifetime_online_metrics
        ),
    }
    for cost_name, _ in COST_SCENARIOS:
        frozen = frozen_metrics[cost_name]["learner"]
        for period_name in ("2024", "2025", "2026_ytd"):
            gates[
                f"strict_{cost_name}_{period_name}_material_active_log_edge"
            ] = (
                float(frozen["periods"][period_name]["active_log_edge"])
                > MIN_MATERIAL_ACTIVE_LOG_EDGE
            )
        online = lifetime_online_metrics[cost_name]["learner"]
        annual = [
            float(value["active_log_edge"])
            for name, value in online["periods"].items()
            if name.isdigit()
        ]
        gates[f"relaxed_{cost_name}_positive_continuous_relative_wealth"] = float(
            online["comparison"]["relative_wealth_vs_aapl_buy_hold"]
        ) > 0.0
        gates[f"relaxed_{cost_name}_more_positive_than_negative_years"] = int(
            np.count_nonzero(np.asarray(annual) > 0.0)
        ) > int(np.count_nonzero(np.asarray(annual) < 0.0))
    strict_names = [name for name in gates if name.startswith("strict_")]
    relaxed_names = [name for name in gates if name.startswith("relaxed_")]
    integrity_names = [
        "frozen_always_long_matches_buy_hold",
        "online_always_long_matches_buy_hold",
    ]
    integrity_pass = all(gates[name] for name in integrity_names)
    strict_pass = integrity_pass and all(gates[name] for name in strict_names)
    relaxed_pass = integrity_pass and all(gates[name] for name in relaxed_names)
    failures = sorted(name for name, passed in gates.items() if not passed)
    return {
        "passed": strict_pass,
        "integrity_pass": integrity_pass,
        "strict_pass": strict_pass,
        "relaxed_long_run_pass": relaxed_pass,
        "gates": gates,
        "failures": failures,
    }


def _flatten_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reset_index(names="decision_date")


def _stage_bundle(
    *,
    output_dir: Path,
    run_id: str,
    stage: str,
    stage_pass: bool,
    report: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    source_provenance: Mapping[str, Any],
    git_identity: Mapping[str, Any],
    parent_manifest: Mapping[str, Any] | None,
    deadline: _Deadline,
) -> dict[str, Any]:
    data_payloads = {".gitattributes": b"* -text\n", **dict(payloads)}
    data_payloads["report.json"] = _pretty_json_bytes(report)
    hashes = {name: _sha256(data) for name, data in sorted(data_payloads.items())}
    manifest_payload = {
        "contract_version": CONTRACT_VERSION,
        "stage": stage,
        "stage_pass": bool(stage_pass),
        "run_id": run_id,
        "source_provenance": dict(source_provenance),
        "source_path": source_provenance["source_path"],
        "bounded_result_sha256": source_provenance["bounded_result_sha256"],
        "git_identity": dict(git_identity),
        "parent_manifest_sha256": (
            parent_manifest.get("manifest_sha256") if parent_manifest else None
        ),
        "payload_sha256": hashes,
        "execution": {
            "asset": "AAPL",
            "actions": ["LONG_100_PERCENT", "CASH_100_PERCENT"],
            "maximum_target_exposure": 1.0,
            "shorting": False,
            "leverage": False,
            "borrowing": False,
            "network_access": False,
            "llm_calls": 0,
            "api_calls": 0,
            "estimated_external_cost_usd": 0.0,
            "minimum_material_active_log_edge": MIN_MATERIAL_ACTIVE_LOG_EDGE,
        },
    }
    manifest = _manifest(manifest_payload)
    data_payloads["stage_manifest.json"] = _pretty_json_bytes(manifest)
    run_dir = output_dir.resolve() / run_id
    checksums = _seal_bundle(
        run_dir,
        data_payloads,
        before_promote=lambda: deadline.check("before artifact promotion"),
    )
    return {
        "stage": stage,
        "stage_pass": bool(stage_pass),
        "run_id": run_id,
        "artifact_dir": str(run_dir),
        "stage_manifest": str(run_dir / "stage_manifest.json"),
        "manifest_sha256": manifest["manifest_sha256"],
        "checksums": checksums,
    }


def _development_periods() -> tuple[EvaluationPeriod, ...]:
    annual = _periods_for_years(2005, 2018)
    folds = tuple(
        EvaluationPeriod(
            f"{start}_{start + 1}",
            f"{start}-01-01",
            f"{start + 1}-12-31",
        )
        for start in range(2005, 2019, 2)
    )
    return (*annual, *folds)


def _validation_periods() -> tuple[EvaluationPeriod, ...]:
    return _periods_for_years(2019, 2023)


def _final_periods() -> tuple[EvaluationPeriod, ...]:
    return (
        EvaluationPeriod("2024", "2024-01-01", "2024-12-31"),
        EvaluationPeriod("2025", "2025-01-01", "2025-12-31"),
        EvaluationPeriod("2026_ytd", "2026-01-01", "2026-07-09"),
    )


def _lifetime_periods() -> tuple[EvaluationPeriod, ...]:
    return (
        *_periods_for_years(2005, 2025),
        EvaluationPeriod("2026_ytd", "2026-01-01", "2026-07-09"),
    )


def _prefix_sha256(frame: pd.DataFrame, *, end: pd.Timestamp) -> str:
    prefix = canonical_context_frame(frame).loc[:end]
    return _sha256(_frame_csv_bytes(prefix.reset_index(names="date")))


def _require_source_continuity(
    frame: pd.DataFrame,
    provenance: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    parent_end: pd.Timestamp,
) -> None:
    if (
        provenance.get("source_type")
        != parent.get("source_provenance", {}).get("source_type")
    ):
        raise ChronologicalExhaustionExperimentError(
            "Physical price-snapshot lineage changed between authorized stages"
        )
    if _prefix_sha256(frame, end=parent_end) != parent.get(
        "bounded_result_sha256"
    ):
        raise ChronologicalExhaustionExperimentError(
            "Authorized historical price prefix changed"
        )


def _checkpoint_from_forecast(
    forecast: pd.DataFrame, *, cutoff: pd.Timestamp, learning_mode: str
) -> dict[str, Any]:
    eligible = forecast.loc[:cutoff]
    if eligible.empty:
        raise ChronologicalExhaustionExperimentError("Checkpoint has no rows")
    row = eligible.iloc[-1]
    experts: dict[str, Any] = {}
    for expert in ("contextual", "weak_trend"):
        experts[expert] = {
            "matured_count": int(row[f"{expert}_matured_count"]),
            "matured_wins": int(row[f"{expert}_matured_wins"]),
            "sum_net_edge": float(row[f"{expert}_sum_net_edge"]),
            "sum_squared_net_edge": float(
                row[f"{expert}_sum_squared_net_edge"]
            ),
        }
    trailing_columns = [
        "contextual_virtual_signal",
        "weak_trend_virtual_signal",
        "combined_cash_signal",
        "target_exposure",
    ]
    trailing = (
        eligible.loc[:, trailing_columns]
        .tail(2)
        .reset_index(names="decision_date")
        .to_dict(orient="records")
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "learning_mode": learning_mode,
        "checkpoint_cutoff": cutoff.date().isoformat(),
        "last_observed_session": eligible.index[-1].date().isoformat(),
        "experts": experts,
        "trailing_cooldown_and_pending_context": trailing,
    }


def _require_checkpoint_continuity(
    forecast: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    parent_manifest_path: Path,
    checkpoint_filename: str,
) -> None:
    checkpoint_path = parent_manifest_path.resolve().parent / checkpoint_filename
    try:
        parent = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChronologicalExhaustionExperimentError(
            "Committed parent checkpoint is unreadable"
        ) from exc
    current = _checkpoint_from_forecast(
        forecast,
        cutoff=cutoff,
        learning_mode=str(parent.get("learning_mode")),
    )
    comparable_fields = (
        "contract_version",
        "learning_mode",
        "checkpoint_cutoff",
        "last_observed_session",
        "experts",
    )

    def pending_context(value: Mapping[str, Any]) -> list[tuple[str, bool, bool]]:
        rows = value.get("trailing_cooldown_and_pending_context")
        if not isinstance(rows, list):
            return []
        return [
            (
                pd.Timestamp(row["decision_date"]).date().isoformat(),
                bool(row["contextual_virtual_signal"]),
                bool(row["weak_trend_virtual_signal"]),
            )
            for row in rows
        ]

    if (
        parent.get("learning_mode") != CAUSAL_ONLINE_MODE
        or any(parent.get(name) != current.get(name) for name in comparable_fields)
        or pending_context(parent) != pending_context(current)
    ):
        raise ChronologicalExhaustionExperimentError(
            "Regenerated learner state does not match the committed parent checkpoint"
        )


def _ledger_payloads(
    ledgers: Mapping[str, pd.DataFrame], *, prefix: str
) -> dict[str, bytes]:
    return {
        f"{prefix}_{cost_name}_ledgers.csv": _frame_csv_bytes(frame)
        for cost_name, frame in ledgers.items()
    }


def _episode_payloads(
    episodes: Mapping[str, pd.DataFrame], *, prefix: str
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for cost_name, frame in episodes.items():
        value = frame.copy()
        if value.empty:
            value = pd.DataFrame(
                columns=[
                    "decision_date",
                    "entry_date",
                    "exit_date",
                    "raw_active_log_edge",
                    "net_active_log_edge",
                    "win",
                ]
            )
        payloads[f"{prefix}_{cost_name}_episodes.csv"] = _frame_csv_bytes(value)
    return payloads


def run_development(
    *,
    repo_root: Path,
    price_artifact: Path,
    output_dir: Path,
    run_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    deadline = _Deadline(clock)
    git_identity = _clean_git_identity(repo_root)
    input_identity = _tracked_input_identity(repo_root, price_artifact)
    frame, provenance = load_bounded_prices(
        price_artifact,
        end=DEVELOPMENT_END,
        required_last_session=pd.Timestamp("2018-12-31"),
    )
    provenance["tracked_input"] = input_identity
    deadline.check("bounded development load")
    forecast = build_chronological_exhaustion_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    metrics, ledgers, episodes = _evaluate_policy_set(
        frame,
        forecast,
        periods=_development_periods(),
        administrative_start=DEVELOPMENT_START,
    )
    gates = apply_development_gates(metrics)
    deadline.check("development replay and gates")
    resolved_run_id = _safe_run_id(run_id, prefix="exhaustion-expert-development")
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "development",
        "run_id": resolved_run_id,
        "evidence_classification": "causal_prequential_2005_2018_not_globally_pristine",
        "physical_data_end": DEVELOPMENT_END.date().isoformat(),
        "later_outcomes_accessed": False,
        "metrics": metrics,
        "gate_report": gates,
        "runtime": {
            "seconds_before_seal": deadline.elapsed(),
            "limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "within_limit": deadline.elapsed() <= RUN_TIME_LIMIT_SECONDS,
            "llm_calls": 0,
            "api_calls": 0,
            "external_cost_usd": 0.0,
        },
    }
    checkpoint = _checkpoint_from_forecast(
        forecast, cutoff=DEVELOPMENT_END, learning_mode=CAUSAL_ONLINE_MODE
    )
    payloads = {
        "development_prices_through_2018.csv": _frame_csv_bytes(
            frame.reset_index(names="date")
        ),
        "development_forecast_through_2018.csv": _frame_csv_bytes(
            _flatten_forecast(forecast)
        ),
        "development_metrics.json": _pretty_json_bytes(metrics),
        "development_gate_report.json": _pretty_json_bytes(gates),
        "development_checkpoint_through_2018.json": _pretty_json_bytes(checkpoint),
        "input_provenance.json": _pretty_json_bytes(provenance),
        **_ledger_payloads(ledgers, prefix="development"),
        **_episode_payloads(episodes, prefix="development"),
    }
    deadline.check("before development seal")
    result = _stage_bundle(
        output_dir=output_dir,
        run_id=resolved_run_id,
        stage="development",
        stage_pass=bool(gates["passed"]),
        report=report,
        payloads=payloads,
        source_provenance=provenance,
        git_identity=git_identity,
        parent_manifest=None,
        deadline=deadline,
    )
    deadline.check("development seal")
    return {**result, "gate_report": gates, "runtime_seconds": deadline.elapsed()}


def run_validation(
    *,
    repo_root: Path,
    price_artifact: Path,
    development_manifest: Path,
    output_dir: Path,
    run_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    deadline = _Deadline(clock)
    git_identity = _clean_git_identity(repo_root)
    parent = _validated_prior_manifest(
        repo_root=repo_root,
        path=development_manifest,
        expected_stage="development",
    )
    _require_dependency_continuity(git_identity, parent)
    input_identity = _tracked_input_identity(repo_root, price_artifact)
    # Authority is validated before this first operation capable of loading a
    # 2019-2023 market row.
    frame, provenance = load_bounded_prices(
        price_artifact,
        end=VALIDATION_END,
        required_last_session=pd.Timestamp("2023-12-29"),
    )
    provenance["tracked_input"] = input_identity
    _require_source_continuity(
        frame, provenance, parent, parent_end=DEVELOPMENT_END
    )
    deadline.check("authorized validation load")
    frozen = build_chronological_exhaustion_forecast(
        frame,
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=DEVELOPMENT_END,
    )
    _require_checkpoint_continuity(
        frozen,
        cutoff=DEVELOPMENT_END,
        parent_manifest_path=development_manifest,
        checkpoint_filename="development_checkpoint_through_2018.json",
    )
    online = build_chronological_exhaustion_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    frozen_metrics, frozen_ledgers, frozen_episodes = _evaluate_policy_set(
        frame,
        frozen,
        periods=_validation_periods(),
        administrative_start=VALIDATION_START,
    )
    online_metrics, online_ledgers, online_episodes = _evaluate_policy_set(
        frame,
        online,
        periods=_validation_periods(),
        administrative_start=VALIDATION_START,
    )
    gates = apply_validation_gates(frozen_metrics)
    deadline.check("validation replay and gates")
    resolved_run_id = _safe_run_id(run_id, prefix="exhaustion-expert-validation")
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "validation",
        "run_id": resolved_run_id,
        "evidence_classification": (
            "candidate_frozen_2019_2023_confirmation_not_globally_pristine"
        ),
        "physical_data_end": VALIDATION_END.date().isoformat(),
        "post_2023_outcomes_accessed": False,
        "primary_frozen_metrics": frozen_metrics,
        "secondary_causal_online_metrics": online_metrics,
        "gate_report": gates,
        "online_cannot_rescue_frozen_failure": True,
        "runtime": {
            "seconds_before_seal": deadline.elapsed(),
            "limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "within_limit": deadline.elapsed() <= RUN_TIME_LIMIT_SECONDS,
            "llm_calls": 0,
            "api_calls": 0,
            "external_cost_usd": 0.0,
        },
    }
    checkpoint = _checkpoint_from_forecast(
        online, cutoff=VALIDATION_END, learning_mode=CAUSAL_ONLINE_MODE
    )
    payloads = {
        "authorized_development_manifest.json": _pretty_json_bytes(parent),
        "validation_prices_through_2023.csv": _frame_csv_bytes(
            frame.reset_index(names="date")
        ),
        "validation_frozen_forecast.csv": _frame_csv_bytes(
            _flatten_forecast(frozen.loc[VALIDATION_START:VALIDATION_END])
        ),
        "validation_online_forecast.csv": _frame_csv_bytes(
            _flatten_forecast(online.loc[VALIDATION_START:VALIDATION_END])
        ),
        "validation_frozen_metrics.json": _pretty_json_bytes(frozen_metrics),
        "validation_online_metrics.json": _pretty_json_bytes(online_metrics),
        "validation_gate_report.json": _pretty_json_bytes(gates),
        "validation_checkpoint_through_2023.json": _pretty_json_bytes(checkpoint),
        "input_provenance.json": _pretty_json_bytes(provenance),
        **_ledger_payloads(frozen_ledgers, prefix="validation_frozen"),
        **_ledger_payloads(online_ledgers, prefix="validation_online"),
        **_episode_payloads(frozen_episodes, prefix="validation_frozen"),
        **_episode_payloads(online_episodes, prefix="validation_online"),
    }
    deadline.check("before validation seal")
    result = _stage_bundle(
        output_dir=output_dir,
        run_id=resolved_run_id,
        stage="validation",
        stage_pass=bool(gates["passed"]),
        report=report,
        payloads=payloads,
        source_provenance=provenance,
        git_identity=git_identity,
        parent_manifest=parent,
        deadline=deadline,
    )
    deadline.check("validation seal")
    return {**result, "gate_report": gates, "runtime_seconds": deadline.elapsed()}


def run_final(
    *,
    repo_root: Path,
    price_artifact: Path,
    validation_manifest: Path,
    output_dir: Path,
    run_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    deadline = _Deadline(clock)
    git_identity = _clean_git_identity(repo_root)
    parent = _validated_prior_manifest(
        repo_root=repo_root,
        path=validation_manifest,
        expected_stage="validation",
    )
    _require_dependency_continuity(git_identity, parent)
    input_identity = _tracked_input_identity(repo_root, price_artifact)
    # Authority is validated before this first operation capable of loading a
    # 2024+ market row.
    frame, provenance = load_bounded_prices(
        price_artifact,
        end=FINAL_END,
        required_last_session=FINAL_END,
    )
    provenance["tracked_input"] = input_identity
    _require_source_continuity(frame, provenance, parent, parent_end=VALIDATION_END)
    deadline.check("authorized final load")
    frozen = build_chronological_exhaustion_forecast(
        frame,
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=VALIDATION_END,
    )
    _require_checkpoint_continuity(
        frozen,
        cutoff=VALIDATION_END,
        parent_manifest_path=validation_manifest,
        checkpoint_filename="validation_checkpoint_through_2023.json",
    )
    online = build_chronological_exhaustion_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    frozen_metrics, frozen_ledgers, frozen_episodes = _evaluate_policy_set(
        frame,
        frozen,
        periods=_final_periods(),
        administrative_start=FINAL_START,
    )
    online_final_metrics, online_final_ledgers, online_final_episodes = (
        _evaluate_policy_set(
            frame,
            online,
            periods=_final_periods(),
            administrative_start=FINAL_START,
        )
    )
    lifetime_metrics, lifetime_ledgers, lifetime_episodes = _evaluate_policy_set(
        frame,
        online,
        periods=_lifetime_periods(),
        administrative_start=DEVELOPMENT_START,
    )
    gates = apply_final_gates(frozen_metrics, lifetime_metrics)
    deadline.check("final replay and gates")
    resolved_run_id = _safe_run_id(run_id, prefix="exhaustion-expert-final")
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "final",
        "run_id": resolved_run_id,
        "evidence_classification": "repeated_2024_2026_historical_audit",
        "physical_data_end": FINAL_END.date().isoformat(),
        "primary_frozen_metrics": frozen_metrics,
        "secondary_causal_online_final_metrics": online_final_metrics,
        "lifetime_causal_online_2005_2026_ytd_metrics": lifetime_metrics,
        "gate_report": gates,
        "online_cannot_rescue_frozen_strict_failure": True,
        "runtime": {
            "seconds_before_seal": deadline.elapsed(),
            "limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "within_limit": deadline.elapsed() <= RUN_TIME_LIMIT_SECONDS,
            "llm_calls": 0,
            "api_calls": 0,
            "external_cost_usd": 0.0,
        },
    }
    checkpoint = _checkpoint_from_forecast(
        online, cutoff=FINAL_END, learning_mode=CAUSAL_ONLINE_MODE
    )
    payloads = {
        "authorized_validation_manifest.json": _pretty_json_bytes(parent),
        "final_prices_through_2026_ytd.csv": _frame_csv_bytes(
            frame.reset_index(names="date")
        ),
        "final_frozen_forecast.csv": _frame_csv_bytes(
            _flatten_forecast(frozen.loc[FINAL_START:FINAL_END])
        ),
        "final_online_forecast.csv": _frame_csv_bytes(
            _flatten_forecast(online.loc[FINAL_START:FINAL_END])
        ),
        "final_frozen_metrics.json": _pretty_json_bytes(frozen_metrics),
        "final_online_metrics.json": _pretty_json_bytes(online_final_metrics),
        "lifetime_online_metrics.json": _pretty_json_bytes(lifetime_metrics),
        "final_gate_report.json": _pretty_json_bytes(gates),
        "online_checkpoint_through_2026_ytd.json": _pretty_json_bytes(checkpoint),
        "input_provenance.json": _pretty_json_bytes(provenance),
        **_ledger_payloads(frozen_ledgers, prefix="final_frozen"),
        **_ledger_payloads(online_final_ledgers, prefix="final_online"),
        **_ledger_payloads(lifetime_ledgers, prefix="lifetime_online"),
        **_episode_payloads(frozen_episodes, prefix="final_frozen"),
        **_episode_payloads(online_final_episodes, prefix="final_online"),
        **_episode_payloads(lifetime_episodes, prefix="lifetime_online"),
    }
    deadline.check("before final seal")
    result = _stage_bundle(
        output_dir=output_dir,
        run_id=resolved_run_id,
        stage="final",
        stage_pass=bool(gates["strict_pass"]),
        report=report,
        payloads=payloads,
        source_provenance=provenance,
        git_identity=git_identity,
        parent_manifest=parent,
        deadline=deadline,
    )
    deadline.check("final seal")
    return {**result, "gate_report": gates, "runtime_seconds": deadline.elapsed()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("develop", "validate", "final"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--price-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    common = {
        "repo_root": args.repo_root,
        "price_artifact": args.price_artifact,
        "output_dir": args.output_dir,
        "run_id": args.run_id,
    }
    if args.command == "develop":
        if args.development_manifest or args.validation_manifest:
            raise SystemExit("develop does not accept a prior manifest")
        result = run_development(**common)
    elif args.command == "validate":
        if args.development_manifest is None or args.validation_manifest:
            raise SystemExit("validate requires only --development-manifest")
        result = run_validation(
            **common, development_manifest=args.development_manifest
        )
    else:
        if args.validation_manifest is None or args.development_manifest:
            raise SystemExit("final requires only --validation-manifest")
        result = run_final(**common, validation_manifest=args.validation_manifest)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ChronologicalExhaustionExperimentError",
    "apply_development_gates",
    "apply_final_gates",
    "apply_validation_gates",
    "load_bounded_prices",
    "main",
    "run_development",
    "run_final",
    "run_validation",
]
