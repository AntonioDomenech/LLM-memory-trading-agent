from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from typing import Any, Mapping

import pytest

from agent_benchmark.sec_audit_transport import ResponseAudit
from agent_benchmark.sec_filing_content import normalize_filing_text
from agent_benchmark.sec_filing_gemma_contract import (
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_SECONDS,
    SecFilingGemmaContractError,
    build_corpus_universe_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_corpus import (
    DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION,
    DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION,
    MAIN_SUBMISSIONS_URL,
    SecCorpusBudget,
    SecFilingGemmaCorpusError,
    acquire_authorized_stage_documents,
    acquire_official_sec_catalog,
    validate_detached_catalog_replay,
    validate_detached_stage_content_replay,
)
from agent_benchmark.sec_point_in_time import (
    SecAuditLimitError,
    content_sha256,
    validate_sec_user_agent,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


USER_AGENT = "Private Researcher contact@real-domain-for-tests.dev"


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class FakeTransport:
    def __init__(
        self,
        payloads: Mapping[str, bytes],
        *,
        user_agent: str = USER_AGENT,
        audit_overrides: Mapping[str, Any] | None = None,
        security_overrides: Mapping[str, Any] | None = None,
        transport_max_requests: int = MAX_SEC_REQUESTS,
        transport_max_bytes: int = MAX_SEC_BYTES,
        transport_max_seconds: float = float(MAX_SEC_SECONDS),
        on_fetch: Any = None,
    ) -> None:
        self.payloads = dict(payloads)
        self.user_agent_audit = validate_sec_user_agent(user_agent)
        self.audit_overrides = dict(audit_overrides or {})
        self.security_overrides = dict(security_overrides or {})
        self.transport_max_requests = transport_max_requests
        self.transport_max_bytes = transport_max_bytes
        self.transport_max_seconds = transport_max_seconds
        self.on_fetch = on_fetch
        self.calls: list[str] = []

    def acquisition_security_state(self) -> dict[str, Any]:
        state = {
            "trust_env": False,
            "proxies": False,
            "follow_redirects": False,
            "max_retries": 0,
            "max_redirects": 0,
            "allow_cache_reads": False,
            "allow_cache_writes": False,
            "streaming_body": True,
            "content_length_preflight": True,
            "incremental_byte_budget": True,
            "transport_max_requests": self.transport_max_requests,
            "transport_max_bytes": self.transport_max_bytes,
            "transport_max_seconds": self.transport_max_seconds,
        }
        state.update(self.security_overrides)
        return state

    def fetch(self, url: str) -> tuple[bytes, ResponseAudit]:
        self.calls.append(url)
        if self.on_fetch is not None:
            self.on_fetch(url)
        if url not in self.payloads:
            raise LookupError("missing fake response containing secret headers")
        payload = self.payloads[url]
        content_type = (
            "application/json; charset=utf-8"
            if url.startswith("https://data.sec.gov/")
            else "text/html; charset=iso-8859-1"
        )
        audit = ResponseAudit(
            url=url,
            status_code=200,
            content_type=content_type,
            size_bytes=len(payload),
            content_sha256=content_sha256(payload),
            cache_hit=False,
            user_agent_sha256=self.user_agent_audit.sha256,
            network_requests=1,
            retries=0,
            redirects=0,
        )
        return payload, replace(audit, **self.audit_overrides)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _row(
    serial: int,
    *,
    year: int,
    form: str = "10-Q",
    filing_date: str | None = None,
    acceptance: str | None = None,
    accession_prefix: str = "0000320193",
    primary_document: str | None = None,
    filing_date_change: str = "",
) -> dict[str, Any]:
    filed = filing_date or f"{year:04d}-04-04"
    accepted = acceptance or f"{year:04d}0404120000"
    return {
        "accessionNumber": f"{accession_prefix}-{year % 100:02d}-{serial:06d}",
        "acceptanceDateTime": accepted,
        "form": form,
        "primaryDocument": primary_document or f"apple-{year}-{serial}.htm",
        "items": "",
        "filingDate": filed,
        "reportDate": f"{year:04d}-03-31",
        "isXBRL": 1,
        "dateOfFilingDateChange": filing_date_change,
    }


def _columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    keys = (
        "accessionNumber",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "items",
        "filingDate",
        "reportDate",
        "isXBRL",
        "dateOfFilingDateChange",
    )
    return {key: [row[key] for row in rows] for key in keys}


def _historical_url(name: str) -> str:
    return f"https://data.sec.gov/submissions/{name}"


def _main(
    recent: list[dict[str, Any]],
    references: list[dict[str, Any]] | None = None,
    *,
    cik: Any = 320193,
) -> bytes:
    return _json_bytes(
        {
            "cik": cik,
            "filings": {
                "recent": _columns(recent),
                "files": references or [],
            },
        }
    )


_DEFAULT_HISTORY_NAME = "CIK0000320193-submissions-999.json"


def _add_default_history_if_needed(
    payloads: Mapping[str, bytes],
) -> dict[str, bytes]:
    result = dict(payloads)
    if MAIN_SUBMISSIONS_URL not in result:
        return result
    main = json.loads(result[MAIN_SUBMISSIONS_URL].decode("utf-8"))
    files = main.get("filings", {}).get("files")
    if files != []:
        return result
    row = _row(999_999, year=1995, form="8-K")
    main["filings"]["files"] = [_reference(_DEFAULT_HISTORY_NAME, [row])]
    result[MAIN_SUBMISSIONS_URL] = _json_bytes(main)
    result[_historical_url(_DEFAULT_HISTORY_NAME)] = _json_bytes(_columns([row]))
    return result


def _reference(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dates = [row["filingDate"] for row in rows]
    return {
        "name": name,
        "filingCount": len(rows),
        "filingFrom": min(dates),
        "filingTo": max(dates),
    }


def _catalog(
    payloads: Mapping[str, bytes],
    *,
    transport: FakeTransport | None = None,
    budget: SecCorpusBudget | None = None,
    clock: Clock | None = None,
):
    local_clock = clock or Clock()
    prepared = _add_default_history_if_needed(payloads)
    selected_budget = budget or SecCorpusBudget(clock=local_clock)
    client = transport or FakeTransport(
        prepared,
        transport_max_requests=selected_budget.max_requests,
        transport_max_bytes=selected_budget.max_bytes,
        transport_max_seconds=selected_budget.max_seconds,
    )
    if transport is not None:
        transport.payloads = dict(prepared)
    result = acquire_official_sec_catalog(
        transport=client,
        user_agent=USER_AGENT,
        budget=selected_budget,
        session_dates=EXPECTED_SESSIONS,
    )
    return result, client


def _primary_url(accession: str, filename: str) -> str:
    return (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        f"{accession.replace('-', '')}/{filename}"
    )


def _universe() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    serial = 1
    for year in range(2000, 2027):
        if year <= 2014:
            count = 4
        elif year <= 2018:
            count = 3
        elif year <= 2022:
            count = 4
        elif year <= 2025:
            count = 3
        else:
            count = 1
        dates = ("02-01", "05-01", "08-01", "11-01")
        forms = ("10-K", "10-Q", "10-Q", "10-Q")
        for position in range(count):
            filing_date = f"{year:04d}-{dates[position]}"
            acceptance = filing_date.replace("-", "") + "120000"
            records.append(
                {
                    "accession_number": (
                        f"0000320193-{year % 100:02d}-{serial:06d}"
                    ),
                    "subject_cik": "0000320193",
                    "form": forms[position],
                    "acceptance_datetime": acceptance,
                    "filing_date": filing_date,
                    "filing_date_change": None,
                    "primary_document": f"filing-{year}-{position}.htm",
                    "source_record_sha256": hashlib.sha256(
                        f"source-{serial}".encode("ascii")
                    ).hexdigest(),
                }
            )
            serial += 1
    return build_corpus_universe_manifest(
        catalog_artifact_sha256="a" * 64,
        calendar_artifact_sha256="b" * 64,
        catalog_total_record_count=len(records),
        catalog_eligible_record_count=len(records),
        session_dates=EXPECTED_SESSIONS,
        records=records,
    )


def _catalog_detached_replay_inputs() -> dict[str, Any]:
    old_a = [_row(1, year=2000, form="10-K")]
    old_b = [_row(2, year=2001, form="10-Q")]
    name_a = "CIK0000320193-submissions-001.json"
    name_b = "CIK0000320193-submissions-002.json"
    recent = [_row(3, year=2025, form="10-Q")]
    payloads = {
        MAIN_SUBMISSIONS_URL: _main(
            recent,
            [_reference(name_b, old_b), _reference(name_a, old_a)],
        ),
        _historical_url(name_a): _json_bytes(_columns(old_a)),
        _historical_url(name_b): _json_bytes(_columns(old_b)),
    }
    result, _transport = _catalog(payloads)
    calendar_hash = "c" * 64
    universe = build_corpus_universe_manifest(
        catalog_artifact_sha256=result.catalog_artifact_sha256,
        calendar_artifact_sha256=calendar_hash,
        catalog_total_record_count=result.catalog_total_record_count,
        catalog_eligible_record_count=result.catalog_eligible_record_count,
        session_dates=EXPECTED_SESSIONS,
        records=[dict(record) for record in result.universe_records],
    )
    receipts = json.loads(result.request_receipts_json)
    return {
        "source_payloads": [
            {"name": source.name, "payload": source.payload}
            for source in result.sources
        ],
        "request_receipts": receipts,
        "catalog_artifact": json.loads(result.artifact_json),
        "corpus_universe_manifest": universe,
        "expected_source_payload_sha256s": {
            source.name: source.payload_sha256 for source in result.sources
        },
        "expected_request_receipt_sha256s": {
            source.name: receipt["request_receipt_sha256"]
            for source, receipt in zip(result.sources, receipts)
        },
        "expected_request_receipts_sha256": result.artifact[
            "request_receipts_sha256"
        ],
        "expected_catalog_artifact_sha256": result.catalog_artifact_sha256,
        "expected_corpus_universe_sha256": universe["universe_sha256"],
        "expected_calendar_artifact_sha256": calendar_hash,
        "session_dates": list(EXPECTED_SESSIONS),
    }


def _stage_detached_replay_inputs() -> dict[str, Any]:
    universe = _universe()
    records = [
        record
        for record in universe["records"]
        if record["artifact_stage"] == "development"
    ]
    payloads = {
        _primary_url(record["accession_number"], record["primary_document"]): (
            f"<html><body><p>Exact filing {record['accession_number']} text.</p></body></html>"
        ).encode("ascii")
        for record in records
    }
    result = acquire_authorized_stage_documents(
        transport=FakeTransport(payloads),
        user_agent=USER_AGENT,
        budget=SecCorpusBudget(clock=Clock()),
        authorized_stage="development",
        universe_manifest=universe,
        expected_universe_sha256=universe["universe_sha256"],
        session_dates=EXPECTED_SESSIONS,
    )
    receipts = json.loads(result.request_receipts_json)
    return {
        "authorized_stage": "development",
        "document_payloads": [
            {
                "accession_number": document.accession_number,
                "payload": document.raw_primary_document,
            }
            for document in result.documents
        ],
        "request_receipts": receipts,
        "content_manifest": json.loads(result.content_manifest_json),
        "stage_artifact": json.loads(result.artifact_json),
        "corpus_universe_manifest": universe,
        "expected_document_sha256s": {
            document.accession_number: document.primary_document_sha256
            for document in result.documents
        },
        "expected_normalized_text_sha256s": {
            document.accession_number: document.normalized_text_sha256
            for document in result.documents
        },
        "expected_request_receipt_sha256s": {
            document.accession_number: receipt["request_receipt_sha256"]
            for document, receipt in zip(result.documents, receipts)
        },
        "expected_request_receipts_sha256": result.artifact[
            "request_receipts_sha256"
        ],
        "expected_content_manifest_sha256": result.content_manifest[
            "content_manifest_sha256"
        ],
        "expected_stage_artifact_sha256": result.stage_artifact_sha256,
        "expected_corpus_universe_sha256": universe["universe_sha256"],
        "session_dates": list(EXPECTED_SESSIONS),
    }


def test_detached_catalog_replay_rebuilds_catalogue_and_universe_from_bytes() -> None:
    inputs = _catalog_detached_replay_inputs()
    inputs["request_receipts"] = tuple(inputs["request_receipts"])

    receipt = validate_detached_catalog_replay(**inputs)
    assert (
        receipt["schema_version"]
        == DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION
    )

    assert receipt["authorizing"] is False
    assert receipt["fresh_network_provenance_verified"] is False
    assert receipt["network_receipt_claims_replayed_not_observed"] is True
    assert receipt["catalog_artifact_sha256"] == inputs[
        "expected_catalog_artifact_sha256"
    ]
    assert receipt["corpus_universe_sha256"] == inputs[
        "expected_corpus_universe_sha256"
    ]
    assert receipt["source_payload_sha256s"] == inputs[
        "expected_source_payload_sha256s"
    ]
    with pytest.raises(TypeError):
        receipt["source_payload_sha256s"]["CIK0000320193.json"] = "0" * 64


def test_detached_catalog_replay_rejects_byte_substitution_omission_and_order() -> None:
    inputs = _catalog_detached_replay_inputs()
    attacks: list[dict[str, Any]] = []

    one_byte = deepcopy(inputs)
    historical = one_byte["source_payloads"][1]["payload"]
    one_byte["source_payloads"][1]["payload"] = historical.replace(
        b"10-K", b"10-Q", 1
    )
    attacks.append(one_byte)

    substitution = deepcopy(inputs)
    substitution["source_payloads"][1]["payload"], substitution[
        "source_payloads"
    ][2]["payload"] = (
        substitution["source_payloads"][2]["payload"],
        substitution["source_payloads"][1]["payload"],
    )
    attacks.append(substitution)

    omission = deepcopy(inputs)
    omission["source_payloads"].pop()
    attacks.append(omission)

    source_order = deepcopy(inputs)
    source_order["source_payloads"][1:] = reversed(
        source_order["source_payloads"][1:]
    )
    attacks.append(source_order)

    receipt_order = deepcopy(inputs)
    receipt_order["request_receipts"][0], receipt_order["request_receipts"][1] = (
        receipt_order["request_receipts"][1],
        receipt_order["request_receipts"][0],
    )
    attacks.append(receipt_order)

    for attacked in attacks:
        with pytest.raises(SecFilingGemmaCorpusError):
            validate_detached_catalog_replay(**attacked)


def test_detached_catalog_replay_rejects_adversarial_containers() -> None:
    class AdversarialList(list):
        pass

    class AdversarialDict(dict):
        pass

    list_attack = _catalog_detached_replay_inputs()
    list_attack["source_payloads"] = AdversarialList(list_attack["source_payloads"])
    with pytest.raises(SecFilingGemmaCorpusError, match="exact ordered"):
        validate_detached_catalog_replay(**list_attack)

    dict_attack = _catalog_detached_replay_inputs()
    dict_attack["catalog_artifact"] = AdversarialDict(
        dict_attack["catalog_artifact"]
    )
    with pytest.raises(SecFilingGemmaCorpusError, match="plain JSON"):
        validate_detached_catalog_replay(**dict_attack)

    cycle_attack = _catalog_detached_replay_inputs()
    cycle_attack["catalog_artifact"]["cycle"] = cycle_attack["catalog_artifact"]
    with pytest.raises(SecFilingGemmaCorpusError, match="cyclic"):
        validate_detached_catalog_replay(**cycle_attack)

    depth_attack = _catalog_detached_replay_inputs()
    nested: list[Any] = []
    for _index in range(70):
        nested = [nested]
    depth_attack["catalog_artifact"]["deep"] = nested
    with pytest.raises(SecFilingGemmaCorpusError, match="nesting limit"):
        validate_detached_catalog_replay(**depth_attack)


def test_detached_replay_rejects_internally_impossible_transport_caps() -> None:
    catalog_inputs = _catalog_detached_replay_inputs()
    catalog_artifact = catalog_inputs["catalog_artifact"]
    catalog_artifact["transport_security"]["transport_max_requests"] = 1
    catalog_body = {
        key: value
        for key, value in catalog_artifact.items()
        if key != "catalog_artifact_sha256"
    }
    catalog_hash = canonical_sha256(catalog_body)
    catalog_artifact["catalog_artifact_sha256"] = catalog_hash
    catalog_inputs["expected_catalog_artifact_sha256"] = catalog_hash
    with pytest.raises(SecFilingGemmaCorpusError, match="transport ceilings"):
        validate_detached_catalog_replay(**catalog_inputs)

    stage_inputs = _stage_detached_replay_inputs()
    stage_artifact = stage_inputs["stage_artifact"]
    stage_artifact["transport_security"]["transport_max_bytes"] = 1
    stage_body = {
        key: value
        for key, value in stage_artifact.items()
        if key != "stage_artifact_sha256"
    }
    stage_hash = canonical_sha256(stage_body)
    stage_artifact["stage_artifact_sha256"] = stage_hash
    stage_inputs["expected_stage_artifact_sha256"] = stage_hash
    with pytest.raises(SecFilingGemmaCorpusError, match="transport ceilings"):
        validate_detached_stage_content_replay(**stage_inputs)


def test_detached_stage_replay_rebuilds_hashes_and_manifests_from_bytes() -> None:
    inputs = _stage_detached_replay_inputs()

    receipt = validate_detached_stage_content_replay(**inputs)
    assert (
        receipt["schema_version"]
        == DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION
    )

    assert receipt["authorizing"] is False
    assert receipt["fresh_network_provenance_verified"] is False
    assert receipt["network_receipt_claims_replayed_not_observed"] is True
    assert receipt["content_manifest_sha256"] == inputs[
        "expected_content_manifest_sha256"
    ]
    assert receipt["stage_artifact_sha256"] == inputs[
        "expected_stage_artifact_sha256"
    ]
    assert receipt["primary_document_sha256s"] == inputs[
        "expected_document_sha256s"
    ]
    assert receipt["normalized_text_sha256s"] == inputs[
        "expected_normalized_text_sha256s"
    ]


def test_detached_stage_replay_rejects_byte_substitution_omission_and_order() -> None:
    inputs = _stage_detached_replay_inputs()
    attacks: list[dict[str, Any]] = []

    one_byte = deepcopy(inputs)
    payload = one_byte["document_payloads"][0]["payload"]
    one_byte["document_payloads"][0]["payload"] = payload.replace(
        b"Exact", b"Exabt", 1
    )
    attacks.append(one_byte)

    substitution = deepcopy(inputs)
    substitution["document_payloads"][0]["payload"], substitution[
        "document_payloads"
    ][1]["payload"] = (
        substitution["document_payloads"][1]["payload"],
        substitution["document_payloads"][0]["payload"],
    )
    attacks.append(substitution)

    omission = deepcopy(inputs)
    omission["document_payloads"].pop()
    attacks.append(omission)

    document_order = deepcopy(inputs)
    document_order["document_payloads"][0], document_order["document_payloads"][1] = (
        document_order["document_payloads"][1],
        document_order["document_payloads"][0],
    )
    attacks.append(document_order)

    receipt_order = deepcopy(inputs)
    receipt_order["request_receipts"][0], receipt_order["request_receipts"][1] = (
        receipt_order["request_receipts"][1],
        receipt_order["request_receipts"][0],
    )
    attacks.append(receipt_order)

    forged_count = deepcopy(inputs)
    forged_count["content_manifest"]["document_count"] -= 1
    attacks.append(forged_count)

    for attacked in attacks:
        with pytest.raises(SecFilingGemmaCorpusError):
            validate_detached_stage_content_replay(**attacked)


def test_detached_stage_replay_rejects_adversarial_containers() -> None:
    class AdversarialList(list):
        pass

    class AdversarialDict(dict):
        pass

    list_attack = _stage_detached_replay_inputs()
    list_attack["document_payloads"] = AdversarialList(
        list_attack["document_payloads"]
    )
    with pytest.raises(SecFilingGemmaCorpusError, match="exact ordered"):
        validate_detached_stage_content_replay(**list_attack)

    dict_attack = _stage_detached_replay_inputs()
    dict_attack["stage_artifact"] = AdversarialDict(dict_attack["stage_artifact"])
    with pytest.raises(SecFilingGemmaCorpusError, match="plain JSON"):
        validate_detached_stage_content_replay(**dict_attack)


def test_catalog_fetches_main_and_every_reference_and_builds_universe_rows() -> None:
    old_a = [_row(1, year=2000, form="10-K")]
    old_b = [_row(2, year=2001, form="10-Q")]
    name_a = "CIK0000320193-submissions-001.json"
    name_b = "CIK0000320193-submissions-002.json"
    recent = [
        _row(3, year=2025, form="10-Q"),
        _row(4, year=2025, form="10-Q/A"),
        _row(5, year=2025, form="8-K"),
    ]
    # Deliberately reverse references; acquisition order is canonical by name.
    main = _main(recent, [_reference(name_b, old_b), _reference(name_a, old_a)])
    payloads = {
        MAIN_SUBMISSIONS_URL: main,
        _historical_url(name_a): _json_bytes(_columns(old_a)),
        _historical_url(name_b): _json_bytes(_columns(old_b)),
    }
    result, transport = _catalog(payloads)

    assert transport.calls == [
        MAIN_SUBMISSIONS_URL,
        _historical_url(name_a),
        _historical_url(name_b),
    ]
    assert [source.payload for source in result.sources] == [
        payloads[url] for url in transport.calls
    ]
    assert [source.payload_sha256 for source in result.sources] == [
        __import__("hashlib").sha256(payloads[url]).hexdigest()
        for url in transport.calls
    ]
    assert result.catalog_total_record_count == 5
    assert result.catalog_eligible_record_count == 3
    assert {row["form"] for row in result.universe_records} == {"10-K", "10-Q"}
    assert tuple(result.artifact["eligible_records"]) == tuple(
        dict(row) for row in result.universe_records
    )
    assert result.artifact["exclusion_counts"] == {
        "amendment": 1,
        "other_form": 1,
        "outside_contract_availability_window": 0,
    }
    assert result.artifact["evidence_boundary"][
        "legacy_24_slot_audit_is_exhaustive_catalog_proof"
    ] is False
    assert result.artifact["evidence_boundary"][
        "sgml_or_master_index_reconciliation_claimed_for_2025_2026"
    ] is False
    assert result.artifact["evidence_boundary"][
        "full_predictive_corpus_master_index_sgml_or_index_reconciled"
    ] is False
    serialized = result.artifact_json.decode("utf-8")
    assert USER_AGENT not in serialized
    assert "contact@real-domain-for-tests.dev" not in serialized
    assert json.loads(result.artifact_json)["catalog_artifact_sha256"] == (
        result.catalog_artifact_sha256
    )
    assert result.artifact["outer_budget_role"] == (
        "post_transport_reconciliation_not_streaming_protection"
    )
    with pytest.raises(TypeError):
        result.artifact["sources"][0]["name"] = "mutated"
    with pytest.raises(TypeError):
        result.artifact["eligible_records"][0]["form"] = "8-K"
    with pytest.raises(TypeError):
        result.request_receipts[0]["size_bytes"] = 0

    universe = build_corpus_universe_manifest(
        catalog_artifact_sha256=result.catalog_artifact_sha256,
        calendar_artifact_sha256="c" * 64,
        catalog_total_record_count=result.catalog_total_record_count,
        catalog_eligible_record_count=result.catalog_eligible_record_count,
        session_dates=EXPECTED_SESSIONS,
        records=[dict(record) for record in result.universe_records],
    )
    assert universe["catalog_eligible_record_count"] == 3
    assert universe["stage_counts"] == {
        "development": 2,
        "intermediate": 0,
        "final": 1,
    }


def test_exact_duplicates_are_deduplicated_but_conflicts_fail_closed() -> None:
    duplicate = _row(1, year=2001, form="10-Q")
    name = "CIK0000320193-submissions-001.json"
    main = _main([duplicate, deepcopy(duplicate)], [_reference(name, [duplicate])])
    historical = _json_bytes(_columns([deepcopy(duplicate)]))
    result, _ = _catalog(
        {MAIN_SUBMISSIONS_URL: main, _historical_url(name): historical}
    )
    assert result.catalog_total_record_count == 1
    assert result.catalog_eligible_record_count == 1
    assert result.artifact["within_source_exact_duplicate_count"] == 1
    assert result.artifact["cross_source_exact_duplicate_count"] == 1

    conflict = deepcopy(duplicate)
    conflict["primaryDocument"] = "conflicting.htm"
    bad_payloads = {
        MAIN_SUBMISSIONS_URL: _main([duplicate], [_reference(name, [conflict])]),
        _historical_url(name): _json_bytes(_columns([conflict])),
    }
    with pytest.raises(SecFilingGemmaCorpusError, match="Conflicting metadata"):
        _catalog(bad_payloads)


@pytest.mark.parametrize(("main_extra", "historical_extra"), [(True, 1), ("x", "y")])
def test_cross_source_raw_rows_preserve_extra_columns_and_exact_types(
    main_extra: Any,
    historical_extra: Any,
) -> None:
    row = _row(1, year=2001, form="10-Q")
    name = "CIK0000320193-submissions-001.json"
    main_columns = _columns([row])
    historical_columns = _columns([deepcopy(row)])
    main_columns["newExtraColumn"] = [main_extra]
    historical_columns["newExtraColumn"] = [historical_extra]
    main = _json_bytes(
        {
            "cik": 320193,
            "filings": {
                "recent": main_columns,
                "files": [_reference(name, [row])],
            },
        }
    )
    with pytest.raises(SecFilingGemmaCorpusError, match="Conflicting metadata"):
        _catalog(
            {
                MAIN_SUBMISSIONS_URL: main,
                _historical_url(name): _json_bytes(historical_columns),
            }
        )

    historical_columns["newExtraColumn"] = [main_extra]
    result, _ = _catalog(
        {
            MAIN_SUBMISSIONS_URL: main,
            _historical_url(name): _json_bytes(historical_columns),
        }
    )
    assert result.artifact["cross_source_exact_duplicate_count"] == 1
    assert all(
        len(source["raw_row_set_sha256"]) == 64
        for source in result.artifact["sources"]
    )


def test_catalog_uses_acceptance_filing_and_change_dates_at_contract_boundaries() -> None:
    before_start = _row(
        1,
        year=1999,
        form="10-K",
        filing_date="1999-12-31",
        acceptance="19991231120000",
    )
    eligible_2026 = _row(
        2,
        year=2026,
        form="10-Q",
        filing_date="2026-07-06",
        acceptance="20260706120000",
    )
    changed_beyond_cutoff = _row(
        3,
        year=2026,
        form="10-Q",
        filing_date="2026-07-08",
        acceptance="20260708120000",
        filing_date_change="2026-07-09",
    )
    accepted_beyond_cutoff = _row(
        4,
        year=2026,
        form="10-Q",
        filing_date="2026-07-08",
        acceptance="20260710120000",
    )
    result, _ = _catalog(
        {
            MAIN_SUBMISSIONS_URL: _main(
                [
                    before_start,
                    eligible_2026,
                    changed_beyond_cutoff,
                    accepted_beyond_cutoff,
                ]
            )
        }
    )
    assert [
        row["accession_number"] for row in result.universe_records
    ] == [
        before_start["accessionNumber"],
        eligible_2026["accessionNumber"],
    ]
    assert result.artifact["exclusion_counts"][
        "outside_contract_availability_window"
    ] == 2


def test_catalog_normalizes_timezone_qualified_submissions_acceptance_to_et() -> None:
    row = _row(
        1,
        year=2025,
        filing_date="2025-04-04",
        acceptance="2025-04-04T16:00:00.000Z",
    )
    result, _ = _catalog({MAIN_SUBMISSIONS_URL: _main([row])})
    assert result.universe_records[0]["acceptance_datetime"] == "20250404120000"


def test_historical_reference_count_range_and_subject_claim_are_checked() -> None:
    row = _row(1, year=2001)
    name = "CIK0000320193-submissions-001.json"
    reference = _reference(name, [row])
    reference["filingCount"] = 2
    with pytest.raises(SecFilingGemmaCorpusError, match="row count"):
        _catalog(
            {
                MAIN_SUBMISSIONS_URL: _main([], [reference]),
                _historical_url(name): _json_bytes(_columns([row])),
            }
        )

    payload = {"cik": 1, **_columns([row])}
    with pytest.raises(SecFilingGemmaCorpusError, match="non-Apple CIK"):
        _catalog(
            {
                MAIN_SUBMISSIONS_URL: _main([], [_reference(name, [row])]),
                _historical_url(name): _json_bytes(payload),
            }
        )

    exact_reference = _reference(name, [row])
    exact_reference["filingFrom"] = "2001-04-03"
    with pytest.raises(SecFilingGemmaCorpusError, match="minimum filing date"):
        _catalog(
            {
                MAIN_SUBMISSIONS_URL: _main([], [exact_reference]),
                _historical_url(name): _json_bytes(_columns([row])),
            }
        )

    exact_reference = _reference(name, [row])
    exact_reference["filingTo"] = "2001-04-05"
    with pytest.raises(SecFilingGemmaCorpusError, match="maximum filing date"):
        _catalog(
            {
                MAIN_SUBMISSIONS_URL: _main([], [exact_reference]),
                _historical_url(name): _json_bytes(_columns([row])),
            }
        )


def test_historical_scalar_cik_is_authenticated_then_removed_before_columns() -> None:
    row = _row(1, year=2001, form="10-K")
    name = "CIK0000320193-submissions-001.json"
    payload = {"cik": "320193", **_columns([row])}
    result, _ = _catalog(
        {
            MAIN_SUBMISSIONS_URL: _main([], [_reference(name, [row])]),
            _historical_url(name): _json_bytes(payload),
        }
    )
    assert result.catalog_eligible_record_count == 1


@pytest.mark.parametrize(
    "missing_key",
    ["name", "filingCount", "filingFrom", "filingTo"],
)
def test_historical_reference_requires_exact_official_metadata(
    missing_key: str,
) -> None:
    row = _row(1, year=2001)
    reference = _reference("CIK0000320193-submissions-001.json", [row])
    reference.pop(missing_key)
    transport = FakeTransport(
        {MAIN_SUBMISSIONS_URL: _main([_row(2, year=2025)], [reference])}
    )
    with pytest.raises(SecFilingGemmaCorpusError, match="exact frozen metadata"):
        _catalog(transport.payloads, transport=transport)
    assert transport.calls == [MAIN_SUBMISSIONS_URL]


@pytest.mark.parametrize("files_value", [None, []])
def test_main_apple_submissions_requires_nonempty_files_array(
    files_value: Any,
) -> None:
    main: dict[str, Any] = {
        "cik": 320193,
        "filings": {"recent": _columns([_row(1, year=2025)])},
    }
    if files_value is not None:
        main["filings"]["files"] = files_value
    payload = _json_bytes(main)
    transport = FakeTransport({MAIN_SUBMISSIONS_URL: payload})
    with pytest.raises(SecFilingGemmaCorpusError, match="filings.files"):
        acquire_official_sec_catalog(
            transport=transport,
            user_agent=USER_AGENT,
            budget=SecCorpusBudget(clock=Clock()),
            session_dates=EXPECTED_SESSIONS,
        )
    assert transport.calls == [MAIN_SUBMISSIONS_URL]


@pytest.mark.parametrize(
    ("recent", "cik", "match"),
    [
        ([_row(1, year=2001)], 123456, "not Apple"),
        (
            [_row(1, year=2001, accession_prefix="0000000001")],
            320193,
            "non-Apple accession",
        ),
        (
            [_row(1, year=2001, form="8-K", accession_prefix="0000000001")],
            320193,
            "non-Apple accession",
        ),
        ([_row(1, year=2001, form="10-q")], 320193, "form spelling"),
        (
            [_row(1, year=2001, filing_date="2001-02-30")],
            320193,
            "strict validation",
        ),
        (
            [_row(1, year=2001, primary_document="../escape.htm")],
            320193,
            "unsafe primary document",
        ),
    ],
)
def test_catalog_rejects_subject_form_date_and_primary_identity_attacks(
    recent: list[dict[str, Any]], cik: Any, match: str
) -> None:
    with pytest.raises(SecFilingGemmaCorpusError, match=match):
        _catalog({MAIN_SUBMISSIONS_URL: _main(recent, cik=cik)})


@pytest.mark.parametrize(
    "reference",
    [
        {
            "name": "../CIK0000320193-submissions-001.json",
            "filingCount": 1,
            "filingFrom": "2001-04-04",
            "filingTo": "2001-04-04",
        },
        {
            "name": "CIK0000320193-submissions-001.json",
            "filingCount": 1,
            "filingFrom": "2001-04-04",
            "filingTo": "2001-04-04",
            "url": "https://evil.invalid/stolen.json",
        },
        {
            "name": "CIK0000320193-submissions-001.json",
            "filingCount": 1,
            "filingFrom": "2001-04-04",
            "filingTo": "2001-04-04",
            "href": "http://data.sec.gov/submissions/CIK0000320193-submissions-001.json",
        },
    ],
)
def test_catalog_rejects_historical_reference_url_attacks(
    reference: dict[str, Any]
) -> None:
    transport = FakeTransport(
        {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001)], [reference])}
    )
    with pytest.raises((SecFilingGemmaCorpusError, ValueError)):
        _catalog(transport.payloads, transport=transport)
    assert transport.calls == [MAIN_SUBMISSIONS_URL]


def test_catalog_rejects_duplicate_historical_reference_before_second_fetch() -> None:
    rows = [_row(1, year=2001)]
    name = "CIK0000320193-submissions-001.json"
    reference = _reference(name, rows)
    transport = FakeTransport(
        {MAIN_SUBMISSIONS_URL: _main([], [reference, deepcopy(reference)])}
    )
    with pytest.raises(SecFilingGemmaCorpusError, match="duplicate filename"):
        _catalog(transport.payloads, transport=transport)
    assert transport.calls == [MAIN_SUBMISSIONS_URL]


@pytest.mark.parametrize(
    "audit_overrides",
    [
        {"cache_hit": True, "network_requests": 0},
        {"retries": 1, "network_requests": 2},
        {"redirects": 1, "network_requests": 2},
        {"url": "https://www.sec.gov/substituted"},
    ],
)
def test_catalog_rejects_cache_retry_redirect_and_substitution(
    audit_overrides: dict[str, Any]
) -> None:
    payloads = {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001)])}
    transport = FakeTransport(payloads, audit_overrides=audit_overrides)
    with pytest.raises(SecFilingGemmaCorpusError, match="one fresh exact-URL"):
        _catalog(payloads, transport=transport)


@pytest.mark.parametrize(
    "audit_overrides",
    [
        {"status_code": True},
        {"size_bytes": True},
        {"network_requests": True},
        {"retries": False},
        {"redirects": False},
    ],
)
def test_response_audit_rejects_boolean_numeric_fields(
    audit_overrides: dict[str, Any]
) -> None:
    payloads = {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001)])}
    transport = FakeTransport(payloads, audit_overrides=audit_overrides)
    with pytest.raises(SecFilingGemmaCorpusError, match="one fresh exact-URL"):
        _catalog(payloads, transport=transport)


def test_response_audit_mapping_exception_cannot_expose_private_sentinel() -> None:
    sentinel = "PRIVATE-HEADER-SENTINEL contact@private.invalid"
    payloads = _add_default_history_if_needed(
        {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001)])}
    )

    class ExplodingAudit(Mapping[str, Any]):
        def __getitem__(self, _key: str) -> Any:
            raise RuntimeError(sentinel)

        def __iter__(self):
            raise RuntimeError(sentinel)

        def __len__(self) -> int:
            raise RuntimeError(sentinel)

    class UnsafeAuditTransport(FakeTransport):
        def fetch(self, url: str):
            self.calls.append(url)
            return self.payloads[url], ExplodingAudit()

    transport = UnsafeAuditTransport(payloads)
    with pytest.raises(SecFilingGemmaCorpusError) as caught:
        _catalog(payloads, transport=transport)
    assert sentinel not in str(caught.value)
    assert "contact@private.invalid" not in str(caught.value)


def test_same_class_audit_exception_cannot_bypass_redaction() -> None:
    sentinel = "PRIVATE-SAME-CLASS contact@private.invalid"
    payloads = _add_default_history_if_needed(
        {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001)])}
    )

    class ExplodingAudit(Mapping[str, Any]):
        def __getitem__(self, _key: str) -> Any:
            raise SecFilingGemmaCorpusError(sentinel)

        def __iter__(self):
            raise SecFilingGemmaCorpusError(sentinel)

        def __len__(self) -> int:
            raise SecFilingGemmaCorpusError(sentinel)

    class UnsafeAuditTransport(FakeTransport):
        def fetch(self, url: str):
            self.calls.append(url)
            return self.payloads[url], ExplodingAudit()

    with pytest.raises(SecFilingGemmaCorpusError) as caught:
        _catalog(payloads, transport=UnsafeAuditTransport(payloads))
    assert sentinel not in str(caught.value)
    assert "contact@private.invalid" not in str(caught.value)


@pytest.mark.parametrize(
    ("security_overrides", "user_agent"),
    [
        ({"trust_env": True}, USER_AGENT),
        ({"proxies": True}, USER_AGENT),
        ({"follow_redirects": True}, USER_AGENT),
        ({"max_retries": 1}, USER_AGENT),
        ({"allow_cache_reads": True}, USER_AGENT),
        ({"streaming_body": False}, USER_AGENT),
        ({"content_length_preflight": False}, USER_AGENT),
        ({"incremental_byte_budget": False}, USER_AGENT),
        ({}, "Different Person other@real-domain-for-tests.dev"),
    ],
)
def test_catalog_rejects_unsafe_transport_policy_and_user_agent_mismatch(
    security_overrides: dict[str, Any], user_agent: str
) -> None:
    payloads = {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001)])}
    transport = FakeTransport(
        payloads,
        user_agent=user_agent,
        security_overrides=security_overrides,
    )
    with pytest.raises(SecFilingGemmaCorpusError):
        _catalog(payloads, transport=transport)
    assert not transport.calls

@pytest.mark.parametrize(
    ("cap_name", "cap_value"),
    [
        ("transport_max_requests", MAX_SEC_REQUESTS + 1),
        ("transport_max_bytes", MAX_SEC_BYTES + 1),
        ("transport_max_seconds", float(MAX_SEC_SECONDS) + 1.0),
        ("transport_max_requests", True),
        ("transport_max_bytes", True),
        ("transport_max_seconds", True),
    ],
)
def test_transport_capability_attestation_must_be_typed_and_within_budget(
    cap_name: str,
    cap_value: Any,
) -> None:
    payloads = {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001)])}
    transport = FakeTransport(payloads, security_overrides={cap_name: cap_value})
    with pytest.raises(SecFilingGemmaCorpusError, match="security evidence"):
        _catalog(payloads, transport=transport)
    assert not transport.calls


def test_transport_state_exception_cannot_expose_private_contact() -> None:
    payloads = {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001)])}

    class UnsafeStateTransport(FakeTransport):
        def acquisition_security_state(self) -> dict[str, Any]:
            raise RuntimeError(USER_AGENT)

    with pytest.raises(SecFilingGemmaCorpusError) as caught:
        _catalog(payloads, transport=UnsafeStateTransport(payloads))
    assert USER_AGENT not in str(caught.value)
    assert "contact@real-domain-for-tests.dev" not in str(caught.value)


def test_same_class_transport_state_exception_is_also_redacted() -> None:
    payloads = {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001)])}

    class UnsafeStateTransport(FakeTransport):
        def acquisition_security_state(self) -> dict[str, Any]:
            raise SecFilingGemmaCorpusError(USER_AGENT)

    with pytest.raises(SecFilingGemmaCorpusError) as caught:
        _catalog(payloads, transport=UnsafeStateTransport(payloads))
    assert USER_AGENT not in str(caught.value)
    assert "contact@real-domain-for-tests.dev" not in str(caught.value)


def test_catalog_fails_on_request_byte_time_and_missing_reference() -> None:
    historical = [_row(1, year=2001)]
    name = "CIK0000320193-submissions-001.json"
    main = _main([], [_reference(name, historical)])
    payloads = {MAIN_SUBMISSIONS_URL: main}
    clock = Clock()
    with pytest.raises(SecAuditLimitError, match="request ceiling"):
        _catalog(
            payloads,
            budget=SecCorpusBudget(clock=clock, max_requests=1),
            clock=clock,
        )

    clock = Clock()
    with pytest.raises(SecAuditLimitError, match="byte ceiling"):
        _catalog(
            payloads,
            budget=SecCorpusBudget(clock=clock, max_bytes=len(main) - 1),
            clock=clock,
        )

    clock = Clock()

    def advance(_url: str) -> None:
        clock.value = 2.0

    transport = FakeTransport(
        payloads,
        on_fetch=advance,
        transport_max_seconds=1,
    )
    with pytest.raises(SecAuditLimitError, match="wall-clock"):
        _catalog(
            payloads,
            transport=transport,
            budget=SecCorpusBudget(clock=clock, max_seconds=1),
            clock=clock,
        )

    with pytest.raises(SecFilingGemmaCorpusError, match="fetch failed"):
        _catalog(payloads)


def test_stage_fetches_exact_authorized_bytes_only_and_is_deterministic() -> None:
    universe = _universe()
    expected_hash = universe["universe_sha256"]
    development_records = [
        row for row in universe["records"] if row["artifact_stage"] == "development"
    ]
    development = development_records[0]
    later_records = [
        row for row in universe["records"] if row["artifact_stage"] != "development"
    ]
    raw = (
        b"<html><head><style>hidden</style></head><body>\r\n"
        b"<p>Demand improved while costs remained controlled.</p>\r\n"
        b"<script>secret()</script></body></html>"
    )
    dev_urls = [
        _primary_url(row["accession_number"], row["primary_document"])
        for row in development_records
    ]
    later_urls = {
        _primary_url(row["accession_number"], row["primary_document"]): b"later"
        for row in later_records
    }
    payloads = {**{url: raw for url in dev_urls}, **later_urls}

    def run():
        clock = Clock()
        transport = FakeTransport(payloads)
        result = acquire_authorized_stage_documents(
            transport=transport,
            user_agent=USER_AGENT,
            budget=SecCorpusBudget(clock=clock),
            authorized_stage="development",
            universe_manifest=universe,
            expected_universe_sha256=expected_hash,
            session_dates=EXPECTED_SESSIONS,
        )
        return result, transport

    first, first_transport = run()
    second, second_transport = run()
    assert first_transport.calls == second_transport.calls == dev_urls
    assert len(first.documents) == 72
    assert first.documents[0].raw_primary_document == raw
    expected_normalized = normalize_filing_text(raw.decode("latin-1")).text.encode("utf-8")
    assert first.documents[0].normalized_text == expected_normalized
    assert first.raw_documents_by_accession[development["accession_number"]] == raw
    assert (
        first.normalized_documents_by_accession[development["accession_number"]]
        == expected_normalized
    )
    manifest_row = first.content_manifest["documents"][0]
    assert manifest_row["primary_document_bytes"] == len(raw)
    assert manifest_row["normalized_text_bytes"] == len(expected_normalized)
    assert manifest_row["primary_document_sha256"] == __import__("hashlib").sha256(raw).hexdigest()
    assert manifest_row["normalized_text_sha256"] == __import__("hashlib").sha256(
        expected_normalized
    ).hexdigest()
    assert first.content_manifest == second.content_manifest
    assert first.artifact == second.artifact
    assert first.artifact_json == second.artifact_json
    assert first.content_manifest_json == second.content_manifest_json
    assert first.request_receipts_json == second.request_receipts_json
    assert first.stage_artifact_sha256 == second.stage_artifact_sha256
    assert first.artifact["arbitrary_urls_or_accessions_accepted"] is False
    assert first.artifact["sampling_dropping_cache_or_substitution_allowed"] is False
    assert first.artifact["contains_outcomes_market_data_or_model_output"] is False
    assert first.artifact["evidence_boundary"][
        "legacy_24_slot_audit_is_exhaustive_catalog_proof"
    ] is False
    assert first.artifact["evidence_boundary"][
        "full_predictive_corpus_master_index_sgml_or_index_reconciled"
    ] is False
    assert first.artifact["outer_budget_role"] == (
        "post_transport_reconciliation_not_streaming_protection"
    )
    with pytest.raises(TypeError):
        first.artifact["documents"][0]["url"] = "https://evil.invalid"
    with pytest.raises(TypeError):
        first.content_manifest["documents"][0]["primary_document_bytes"] = 0
    with pytest.raises(TypeError):
        first.request_receipts[0]["status_code"] = 500


def test_stage_rejects_stage_and_universe_attacks_before_fetch() -> None:
    universe = _universe()
    expected_hash = universe["universe_sha256"]
    transport = FakeTransport({})
    clock = Clock()
    with pytest.raises(SecFilingGemmaCorpusError, match="stage is invalid"):
        acquire_authorized_stage_documents(
            transport=transport,
            user_agent=USER_AGENT,
            budget=SecCorpusBudget(clock=clock),
            authorized_stage="development,final",
            universe_manifest=universe,
            expected_universe_sha256=expected_hash,
            session_dates=EXPECTED_SESSIONS,
        )
    assert not transport.calls

    mutated = deepcopy(universe)
    final = next(
        row for row in mutated["records"] if row["artifact_stage"] == "final"
    )
    final["artifact_stage"] = "development"
    with pytest.raises(Exception, match="canonical|pin|stage|universe"):
        acquire_authorized_stage_documents(
            transport=transport,
            user_agent=USER_AGENT,
            budget=SecCorpusBudget(clock=Clock()),
            authorized_stage="development",
            universe_manifest=mutated,
            expected_universe_sha256=expected_hash,
            session_dates=EXPECTED_SESSIONS,
        )
    assert not transport.calls


def test_stage_uses_one_detached_universe_snapshot_without_toctou() -> None:
    universe = _universe()
    expected_hash = universe["universe_sha256"]
    mutated = deepcopy(universe)
    moved = next(
        row for row in mutated["records"] if row["artifact_stage"] == "final"
    )
    moved["artifact_stage"] = "development"
    mutated["stage_counts"]["development"] += 1
    mutated["stage_counts"]["final"] -= 1

    class SwitchingUniverse(dict):
        # Canonical JSON snapshotting sees the underlying validated dictionary;
        # any later caller read would instead expose the forged view.
        def __getitem__(self, key):
            return mutated[key]

        def get(self, key, default=None):
            return mutated.get(key, default)

    switching = SwitchingUniverse(universe)
    raw = b"<html><body><p>Complete filing text.</p></body></html>"
    development = [
        row for row in universe["records"] if row["artifact_stage"] == "development"
    ]
    development_urls = [
        _primary_url(row["accession_number"], row["primary_document"])
        for row in development
    ]
    moved_url = _primary_url(moved["accession_number"], moved["primary_document"])
    payloads = {url: raw for url in development_urls}
    payloads[moved_url] = b"<html><body><p>Protected final filing.</p></body></html>"
    transport = FakeTransport(payloads)

    result = acquire_authorized_stage_documents(
        transport=transport,
        user_agent=USER_AGENT,
        budget=SecCorpusBudget(clock=Clock()),
        authorized_stage="development",
        universe_manifest=switching,
        expected_universe_sha256=expected_hash,
        session_dates=EXPECTED_SESSIONS,
    )

    assert len(result.documents) == len(development) == 72
    assert transport.calls == development_urls
    assert moved_url not in transport.calls

def test_public_stage_api_rejects_canonical_but_incomplete_universe() -> None:
    record = {
        "accession_number": "0000320193-18-000001",
        "subject_cik": "0000320193",
        "form": "10-K",
        "acceptance_datetime": "20180201120000",
        "filing_date": "2018-02-01",
        "filing_date_change": None,
        "primary_document": "development.htm",
        "source_record_sha256": "1" * 64,
    }
    incomplete = build_corpus_universe_manifest(
        catalog_artifact_sha256="a" * 64,
        calendar_artifact_sha256="b" * 64,
        catalog_total_record_count=1,
        catalog_eligible_record_count=1,
        session_dates=EXPECTED_SESSIONS,
        records=[record],
    )
    transport = FakeTransport({})
    with pytest.raises(SecFilingGemmaContractError, match="fewer than 72"):
        acquire_authorized_stage_documents(
            transport=transport,
            user_agent=USER_AGENT,
            budget=SecCorpusBudget(clock=Clock()),
            authorized_stage="development",
            universe_manifest=incomplete,
            expected_universe_sha256=incomplete["universe_sha256"],
            session_dates=EXPECTED_SESSIONS,
        )
    assert not transport.calls

def test_stage_fails_closed_on_missing_cached_retried_empty_or_over_budget() -> None:
    universe = _universe()
    expected_hash = universe["universe_sha256"]
    development = next(
        row for row in universe["records"] if row["artifact_stage"] == "development"
    )
    url = _primary_url(
        development["accession_number"], development["primary_document"]
    )

    def acquire(transport: FakeTransport, budget: SecCorpusBudget | None = None):
        return acquire_authorized_stage_documents(
            transport=transport,
            user_agent=USER_AGENT,
            budget=budget or SecCorpusBudget(clock=Clock()),
            authorized_stage="development",
            universe_manifest=universe,
            expected_universe_sha256=expected_hash,
            session_dates=EXPECTED_SESSIONS,
        )

    with pytest.raises(SecFilingGemmaCorpusError, match="fetch failed"):
        acquire(FakeTransport({}))
    with pytest.raises(SecFilingGemmaCorpusError, match="one fresh exact-URL"):
        acquire(
            FakeTransport(
                {url: b"<p>text</p>"},
                audit_overrides={"cache_hit": True, "network_requests": 0},
            )
        )
    with pytest.raises(SecFilingGemmaCorpusError, match="one fresh exact-URL"):
        acquire(
            FakeTransport(
                {url: b"<p>text</p>"},
                audit_overrides={"retries": 1, "network_requests": 2},
            )
        )
    with pytest.raises(SecFilingGemmaCorpusError, match="empty"):
        acquire(FakeTransport({url: b""}))
    with pytest.raises(SecFilingGemmaCorpusError, match="no normalized visible text"):
        acquire(FakeTransport({url: b"<script>nothing visible</script>"}))
    clock = Clock()
    with pytest.raises(SecAuditLimitError, match="byte ceiling"):
        acquire(
            FakeTransport(
                {url: b"<p>text larger than cap</p>"},
                transport_max_bytes=5,
            ),
            SecCorpusBudget(clock=clock, max_bytes=5),
        )


def test_catalog_repeat_is_byte_and_hash_deterministic() -> None:
    payloads = {MAIN_SUBMISSIONS_URL: _main([_row(1, year=2001, form="10-K")])}
    first, _ = _catalog(payloads)
    second, _ = _catalog(payloads)
    assert first.artifact == second.artifact
    assert first.request_receipts == second.request_receipts
    assert first.universe_records == second.universe_records
    assert first.sources[0].payload == second.sources[0].payload
    assert first.catalog_artifact_sha256 == second.catalog_artifact_sha256
    assert first.artifact_json == second.artifact_json
    assert first.universe_records_json == second.universe_records_json
    assert first.request_receipts_json == second.request_receipts_json
