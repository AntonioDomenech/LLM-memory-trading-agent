from __future__ import annotations

import copy

import pandas as pd
import pytest

from agent_benchmark.sec_filing_gemma_extractor_schema import DIMENSION_NAMES, FLAG_NAMES
from agent_benchmark import sec_gemma_content_risk_inputs as content_inputs
from agent_benchmark.sec_gemma_content_risk_experiment import (
    SecGemmaContentRiskError,
    _validate_model_identity_guard,
    build_content_target,
    corroborated_adverse_signal,
)


def _output(*, flag: bool, dimension: bool, quality: str = "usable") -> dict:
    dimensions = {
        name: {
            "current_impact": "neutral",
            "change_vs_prior": "stable",
            "evidence_sentence_ids": [],
        }
        for name in DIMENSION_NAMES
    }
    flags = {
        name: {"present": False, "evidence_sentence_ids": []} for name in FLAG_NAMES
    }
    if dimension:
        dimensions["demand"]["change_vs_prior"] = "deteriorating"
    if flag:
        flags["new_material_risk"]["present"] = True
    return {
        "schema_version": "sec-filing-extractor-v1",
        "document_quality": quality,
        "dimensions": dimensions,
        "flags": flags,
    }


def _frame() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=80)
    values = pd.Series(range(100, 180), index=dates, dtype=float)
    return pd.DataFrame(
        {
            "aapl_open": values,
            "aapl_close": values,
            "aapl_adj_close": values,
            "spy_adj_close": values,
            "qqq_adj_close": values,
        },
        index=dates,
    )


def _manifest_row(frame: pd.DataFrame, position: int, sequence: int) -> dict:
    day = frame.index[position]
    return {
        "sequence": sequence,
        "accession_number": f"accession-{sequence}",
        "form": "10-Q",
        "filing_date": (day - pd.Timedelta(days=1)).date().isoformat(),
        "availability_session": day.date().isoformat(),
    }


def test_signal_requires_both_independent_warning_types() -> None:
    assert corroborated_adverse_signal(_output(flag=True, dimension=True))
    assert not corroborated_adverse_signal(_output(flag=True, dimension=False))
    assert not corroborated_adverse_signal(_output(flag=False, dimension=True))
    assert not corroborated_adverse_signal(_output(flag=True, dimension=True, quality="unusable"))
    assert not corroborated_adverse_signal(None)


def test_management_transition_alone_is_not_adverse() -> None:
    output = _output(flag=False, dimension=True)
    output["flags"]["management_transition"]["present"] = True
    assert not corroborated_adverse_signal(output)


def test_valid_signal_uses_t_plus_1_t_plus_21_without_extension() -> None:
    frame = _frame()
    manifest = [
        _manifest_row(frame, 5, 1),
        _manifest_row(frame, 10, 2),
        _manifest_row(frame, 26, 3),
    ]
    results = [
        {
            "sequence": row["sequence"],
            "availability_session": row["availability_session"],
            "status": "valid",
            "extractor_output": _output(flag=True, dimension=True),
        }
        for row in manifest
    ]
    target, schedule = build_content_target(frame, manifest, results)
    assert schedule["scheduled"].tolist() == [True, False, True]
    assert (target.iloc[5:25] == 0.0).all()
    assert target.iloc[25] == 1.0
    assert (target.iloc[26:46] == 0.0).all()
    assert schedule.iloc[0]["entry_open"] == frame.index[6].date().isoformat()
    assert schedule.iloc[0]["exit_open"] == frame.index[26].date().isoformat()


def test_invalid_output_stays_long() -> None:
    frame = _frame()
    manifest = [_manifest_row(frame, 5, 1)]
    results = [
        {
            "sequence": 1,
            "availability_session": manifest[0]["availability_session"],
            "status": "invalid",
            "extractor_output": None,
        }
    ]
    target, schedule = build_content_target(frame, manifest, results)
    assert (target == 1.0).all()
    assert not bool(schedule.iloc[0]["corroborated_adverse_signal"])


def test_model_identity_guard_requires_exact_unchanged_gemma() -> None:
    identity = {
        "schema_version": content_inputs.MODEL_IDENTITY_SCHEMA_VERSION,
        "ollama_version": "0.11.4",
        "model_name": content_inputs.MODEL_NAME,
        "model_manifest_sha256": content_inputs.MODEL_MANIFEST_SHA256,
        "semantic_runtime_fingerprint_sha256": (
            content_inputs.SEMANTIC_RUNTIME_FINGERPRINT_SHA256
        ),
        "version_response_sha256": "a" * 64,
        "tags_response_sha256": "b" * 64,
    }
    guard = {"before": copy.deepcopy(identity), "after": copy.deepcopy(identity)}
    assert _validate_model_identity_guard(guard)["passed"]

    observational_change = copy.deepcopy(guard)
    observational_change["after"]["tags_response_sha256"] = "d" * 64
    assert _validate_model_identity_guard(observational_change)["passed"]

    changed = copy.deepcopy(guard)
    changed["after"]["model_manifest_sha256"] = "c" * 64
    with pytest.raises(SecGemmaContentRiskError, match="Gemma identity"):
        _validate_model_identity_guard(changed)
