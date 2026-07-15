from __future__ import annotations

import copy
import hashlib
import json

import pytest

from agent_benchmark import contextual_expert_aggregation_2024_audit_input as audit_input


def _receipt_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"


def test_receipt_schema_is_strict_in_addition_to_its_raw_hash(monkeypatch) -> None:
    payload = _receipt_bytes(audit_input.RECEIPT_EXPECTED)
    monkeypatch.setattr(
        audit_input,
        "RECEIPT_RAW_SHA256",
        f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
    parsed = audit_input.parse_sanitized_receipt_bytes(payload)
    assert parsed == audit_input.RECEIPT_EXPECTED


def test_receipt_rejects_bool_integer_type_substitution(monkeypatch) -> None:
    changed = copy.deepcopy(audit_input.RECEIPT_EXPECTED)
    changed["preparation_attestations"]["llm_calls"] = False
    payload = _receipt_bytes(changed)
    monkeypatch.setattr(
        audit_input,
        "RECEIPT_RAW_SHA256",
        f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
    with pytest.raises(audit_input.AuditInputError, match="type changed"):
        audit_input.parse_sanitized_receipt_bytes(payload)


def test_input_module_does_not_name_the_quarantined_record_path() -> None:
    source = __import__("inspect").getsource(audit_input)
    assert "preparation_quarantine" not in source
    assert "through_2025" not in source
    assert "through_2026" not in source
