from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable
from typing import Any

import pytest

from agent_benchmark.sec_filing_content import normalize_filing_text
from agent_benchmark import sec_gemma_lean_science_v39_bridge as bridge
from agent_benchmark.sec_gemma_lean_science_v39_bridge import (
    AuthenticatedV38Authority,
    BridgeViolation,
    ScienceBridgeProjection,
    authenticate_v38_private_root,
    build_streaming_science_projection,
)
from agent_benchmark.sec_gemma_lean_v38_source import (
    CompleteSourceEvidence,
    LEGACY_SCIENCE_PROJECTION_SCHEMA_VERSION,
    LegacyScienceDocument,
    StageSourceBundle,
)
from agent_benchmark import sec_gemma_lean_v38_source as legacy_source
from agent_benchmark import sec_filing_gemma_corpus as legacy_corpus
from agent_benchmark import (
    sec_gemma_online_risk_overlay_acquisition as legacy_acquisition,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _FakeSourceApi:
    def __init__(
        self,
        *,
        output: Any,
        receipt: dict[str, Any],
        evidence_by_accession: dict[str, dict[str, Any]],
    ) -> None:
        self.output = output
        self.receipt = receipt
        self.evidence_by_accession = evidence_by_accession
        self.parse_calls: list[str] = []

    def rehydrate_compact_stage(
        self,
        stage: str,
        checkpoint: dict[str, Any],
        role_manifests: tuple[dict[str, Any], ...],
        load_blob: Any,
        prior: Any,
    ) -> Any:
        assert stage == "development"
        assert prior is None
        # The real v3.8 replay visits every role through this same callback.  A
        # synthetic replay only proves the dependency boundary; projection
        # tests below exercise every complete blob exactly once.
        assert callable(load_blob)
        assert checkpoint["role_manifest_count"] == len(role_manifests)
        return SimpleNamespace(stage_output=self.output, receipt=self.receipt)

    def parse_complete(
        self,
        stage: str,
        target: dict[str, Any],
        payload: bytes,
        reconciliation_evidence: dict[str, Any],
    ) -> Any:
        assert stage == "development"
        assert reconciliation_evidence["complete_targets"]
        accession = target["submissions"]["accession_number"]
        self.parse_calls.append(accession)
        return SimpleNamespace(evidence=self.evidence_by_accession[accession])


@dataclass
class _Fixture:
    authority: AuthenticatedV38Authority
    payloads: dict[str, bytes]
    rows: list[dict[str, Any]]
    source_api: _FakeSourceApi
    load_calls: list[str]


def _make_fixture(
    *,
    count: int = 75,
    mutate_before_authentication: Callable[[Any], None] | None = None,
) -> _Fixture:
    payloads: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    evidence_by_accession: dict[str, dict[str, Any]] = {}
    targets: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    base_text = " ".join(
        [
            "Apple business revenue risk operations management financial "
            "condition material changes products services customers suppliers "
            "technology competition strategy."
        ]
        * 20
    )
    for index in range(count):
        accession = f"0000320193-{index % 100:02d}-{index + 1:06d}"
        form = "10-K" if index % 2 == 0 else "10-Q"
        filename = f"filing{index:03d}.htm"
        raw = f"{base_text} Filing sequence {index}.".encode("latin-1")
        payload = b"HEAD" + raw + b"TAIL"
        normalized_result = normalize_filing_text(raw.decode("latin-1"))
        assert normalized_result.usable
        normalized = normalized_result.text.encode("utf-8")
        complete_hash = _sha(payload)
        selected_hash = _sha(raw)
        normalized_hash = _sha(normalized)
        # Keep the first two on one session with reverse acceptance ordering so
        # the source order and downstream event order are observably distinct.
        day = 3 if index < 2 else 3 + index
        availability = f"2000-01-{min(day, 28):02d}"
        acceptance = "20000103120000" if index == 0 else (
            "20000103110000" if index == 1 else f"200001{min(day - 1, 28):02d}120000"
        )
        typed_values = {
            "accessionNumber": accession,
            "acceptanceDateTime": acceptance,
            "form": form,
            "primaryDocument": filename,
            "items": "",
            "filingDate": "2000-01-03",
            "reportDate": "1999-12-31",
            "isXBRL": 0,
        }
        semantic = {
            "accession_number": accession,
            "subject_cik": "0000320193",
            "typed_values": typed_values,
            "date_of_filing_date_change_present": False,
        }
        selection = {
            "selected_document_ordinal": 1,
            "sequence": 1,
            "type": form,
            "raw_type": form,
            "sec_filename": filename,
            "document_identity": filename,
            "sgml_document_identity": filename,
            "submissions_filename": filename,
            "sgml_filename": filename,
            "submissions_filename_missing": False,
            "sgml_filename_missing": False,
        }
        complete = {
            "url": (
                "https://www.sec.gov/Archives/edgar/data/320193/"
                f"{accession}.txt"
            ),
            "sha256": f"sha256:{complete_hash}",
            "length": len(payload),
        }
        extracted = {
            "start_byte": 4,
            "end_byte": 4 + len(raw),
            "length": len(raw),
            "sha256": f"sha256:{selected_hash}",
            "encoding": "latin-1 exact one-to-one response slice",
        }
        normalized_meta = {
            "length": len(normalized),
            "character_count": normalized_result.character_count,
            "sha256": f"sha256:{normalized_hash}",
            "normalizer": "inherited normalize_filing_text byte-identical rules",
        }
        prefix = {
            "accession_number": accession,
            "submissions_semantic_identity": semantic,
            "submissions_semantic_identity_sha256": _canonical_sha(semantic),
            "master_identity": {},
            "header_identity": {},
            "acceptance": {
                "submissions_source": acceptance,
                "header_source": acceptance,
                "normalized_et": acceptance,
                "submissions_missing": False,
                "header_missing": False,
                "exact": True,
            },
            "filing_date_change": {
                "submissions_source": None,
                "submissions_column_present": False,
                "header_source": None,
                "normalized": None,
                "submissions_missing": True,
                "header_missing": True,
            },
            "availability_session": availability,
            "availability_not_before_date": "2000-01-02",
            "immutable_stage_assignment": "D",
            "complete_response": complete,
            "primary_selection": selection,
            "extracted_text": extracted,
            "normalized_text": normalized_meta,
        }
        row = {
            "accession_number": accession,
            "filing_date": "2000-01-03",
            "form": form,
            "acquisition_stage": "development",
            "stage_assignment": "D",
            "availability_session": availability,
            "exact_acceptance": True,
            "submissions_layout_provenance": {
                "source_name": "CIK0000320193.json",
                "source_content_sha256": f"sha256:{'1' * 64}",
                "source_position": index,
                "source_row_order_sha256": "2" * 64,
            },
            "complete_response": complete,
            "primary_selection": selection,
            "extracted_text": extracted,
            "normalized_text": normalized_meta,
            "frozen_prefix": prefix,
            "frozen_prefix_sha256": _canonical_sha(prefix),
        }
        evidence_body = {
            "schema_version": "synthetic-complete-v1",
            "stage": "development",
            "role_id": f"complete/{accession}",
            "accession_number": accession,
            "seal_row": row,
        }
        evidence = {
            **evidence_body,
            "source_evidence_sha256": _canonical_sha(evidence_body),
        }
        manifest_body = {
            "schema_version": "synthetic-role-manifest-v1",
            "sequence": index,
            "role_id": f"complete/{accession}",
            "url": complete["url"],
            "blob_name": f"{index:06d}-{complete_hash}.blob",
            "body_sha256": f"sha256:{complete_hash}",
            "body_bytes": len(payload),
            "transport_receipt_sha256": f"{index + 1000:064x}",
            "parse_receipt_sha256": f"{index + 2000:064x}",
            "source_evidence_sha256": evidence["source_evidence_sha256"],
        }
        manifest = {
            **manifest_body,
            "role_manifest_sha256": _canonical_sha(manifest_body),
        }
        payloads[manifest["blob_name"]] = payload
        rows.append(row)
        evidence_by_accession[accession] = evidence
        targets.append({"submissions": {"accession_number": accession}})
        manifests.append(manifest)
    seal_hash = "3" * 64
    output = SimpleNamespace(
        seal={
            "stage": "development",
            "stage_source_seal_sha256": seal_hash,
            "target_rows": rows,
            # Reverse the membership list: the bridge must derive the frozen
            # source order rather than trust caller ordering.
            "d_accessions": [row["accession_number"] for row in reversed(rows)],
        },
        stage_source_seal_sha256=seal_hash,
        complete_evidence=tuple(evidence_by_accession.values()),
        reconciliation_evidence={"complete_targets": targets},
    )
    manifests_hash = _canonical_sha(manifests)
    checkpoint_body = {
        "stage": "development",
        "role_manifest_count": len(manifests),
        "role_manifests_sha256": manifests_hash,
        "expected_stage_source_seal_sha256": seal_hash,
    }
    logical_checkpoint = _canonical_sha(checkpoint_body)
    checkpoint = {**checkpoint_body, "checkpoint_sha256": logical_checkpoint}
    replay_body = {
        "stage": "development",
        "checkpoint_sha256": logical_checkpoint,
        "role_manifests_sha256": manifests_hash,
        "stage_source_seal_sha256": seal_hash,
        "exact_source_seal_match": True,
        "peak_live_role_payload_count": 1,
        "fresh_network_provenance_claimed": False,
    }
    replay_hash = _canonical_sha(replay_body)
    receipt = {**replay_body, "compact_replay_sha256": replay_hash}
    source_api = _FakeSourceApi(
        output=output,
        receipt=receipt,
        evidence_by_accession=evidence_by_accession,
    )
    if mutate_before_authentication is not None:
        mutate_before_authentication(output)
    load_calls: list[str] = []
    active = False

    def loader(name: str) -> bytes:
        nonlocal active
        # A nested loader call would prove that more than one response was
        # being acquired at once.
        assert not active
        active = True
        try:
            load_calls.append(name)
            return payloads[name]
        finally:
            active = False

    expectations = bridge._AuthorityExpectations(
        checkpoint_file_sha256="4" * 64,
        logical_checkpoint_sha256=logical_checkpoint,
        stage_source_seal_sha256=seal_hash,
        compact_replay_sha256=replay_hash,
        role_manifests_sha256=manifests_hash,
        role_plan_sha256="5" * 64,
        document_count=count,
    )
    authority = bridge._authenticate_compact_replay(
        checkpoint_file_sha256="4" * 64,
        checkpoint=checkpoint,
        role_manifests=manifests,
        blob_loader=loader,
        source_api=source_api,
        expectations=expectations,
    )
    # Compact replay itself did not need synthetic bodies; only the exact
    # projection pass should load them.
    assert load_calls == []
    return _Fixture(authority, payloads, rows, source_api, load_calls)


def test_streaming_projection_recreates_all_75_legacy_documents_and_records() -> None:
    fixture = _make_fixture()
    result = build_streaming_science_projection(fixture.authority)

    assert type(result) is ScienceBridgeProjection
    assert result.stage == "development"
    assert len(result.legacy_documents) == 75
    assert all(type(item) is LegacyScienceDocument for item in result.legacy_documents)
    assert len(result.records) == len(result.primary_documents) == 75
    assert len(fixture.load_calls) == 75
    assert len(set(fixture.load_calls)) == 75
    assert fixture.source_api.parse_calls == [
        item["accession_number"] for item in result.source_order
    ]
    assert [item.accession_number for item in result.legacy_documents] == [
        item["accession_number"] for item in result.source_order
    ]
    assert set(result.records[0]) == {
        "accession_number",
        "subject_cik",
        "form",
        "acceptance_datetime",
        "filing_date",
        "filing_date_change",
        "primary_document",
        "source_record_sha256",
    }
    assert set(result.primary_documents[0]) == {
        "accession_number",
        "form",
        "availability_session",
        "acceptance_datetime",
        "official_url",
        "body",
    }
    assert result.manifest["document_count"] == 75
    assert result.manifest["peak_live_complete_submission_blob_count"] == 1
    assert result.manifest["sec_request_count"] == 0
    assert result.manifest["set_parity"] is True
    assert result.manifest["typed_identity_parity"] is True


def test_typed_identity_and_primary_documents_match_frozen_legacy_code() -> None:
    fixture = _make_fixture()
    result = build_streaming_science_projection(fixture.authority)
    first_source = result.source_order[0]
    row = next(
        item
        for item in fixture.rows
        if item["accession_number"] == first_source["accession_number"]
    )
    typed = row["frozen_prefix"]["submissions_semantic_identity"]["typed_values"]
    typed_hash = _sha(
        _canonical_bytes(legacy_corpus._typed_json_identity(typed))
    )
    assert first_source["typed_row_sha256"] == typed_hash
    source_body = {
        "source_url": "https://data.sec.gov/submissions/CIK0000320193.json",
        "source_content_sha256": "1" * 64,
        "raw_row_identity_sha256": typed_hash,
        "accession_number": row["accession_number"],
        "subject_cik": "0000320193",
        "form": row["form"],
        "acceptance_datetime_source": typed["acceptanceDateTime"],
        "acceptance_datetime_et": typed["acceptanceDateTime"],
        "filing_date": row["filing_date"],
        "filing_date_change": None,
        "primary_document": typed["primaryDocument"],
    }
    record = next(
        item
        for item in result.records
        if item["accession_number"] == row["accession_number"]
    )
    assert record["source_record_sha256"] == _canonical_sha(source_body)

    normalized = legacy_acquisition._normalize_documents(
        [dict(item) for item in result.primary_documents],
        stage="development",
    )
    assert len(normalized) == 75


def test_source_order_prior_links_and_event_order_are_exact_and_distinct() -> None:
    result = build_streaming_science_projection(_make_fixture().authority)
    source_accessions = [item["accession_number"] for item in result.source_order]
    assert source_accessions == sorted(source_accessions)
    assert result.event_order[0]["accession_number"] == source_accessions[1]
    assert result.event_order[1]["accession_number"] == source_accessions[0]

    prior_by_form: dict[str, str] = {}
    for link in result.prior_links:
        assert link["prior_accession_number"] == prior_by_form.get(link["form"])
        prior_by_form[link["form"]] = link["accession_number"]
    assert result.prior_links[0]["prior_accession_number"] is None
    assert result.prior_links[1]["prior_accession_number"] is None
    assert result.prior_links[2]["prior_accession_number"] == source_accessions[0]
    assert result.prior_links[3]["prior_accession_number"] == source_accessions[1]


def test_streaming_manifest_is_byte_identical_to_v38_legacy_manifest_shape() -> None:
    result = build_streaming_science_projection(_make_fixture().authority)
    rows = [
        {
            "accession_number": item.accession_number,
            "official_complete_submission_url": item.official_complete_submission_url,
            "raw_primary_document_sha256": item.primary_document_sha256,
            "normalized_text_sha256": item.normalized_text_sha256,
            "complete_response_sha256": item.complete_response_sha256,
            "selected_document_identity": item.selected_document_identity,
            "raw_primary_document_semantics": "selected_embedded_TEXT_bytes",
        }
        for item in result.legacy_documents
    ]
    body = {
        "schema_version": LEGACY_SCIENCE_PROJECTION_SCHEMA_VERSION,
        "stage": "development",
        "stage_source_seal_sha256": "3" * 64,
        "document_count": 75,
        "documents": rows,
        "ordering": "availability_session_then_accession",
        "compatibility_boundary": (
            "raw_primary_document is extracted TEXT, never the complete response"
        ),
    }
    expected = {**body, "projection_sha256": _canonical_sha(body)}
    assert dict(result.legacy_projection_manifest) == expected
    assert result.manifest["legacy_projection_sha256"] == expected["projection_sha256"]


def test_synthetic_streaming_and_existing_non_streaming_projection_are_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture()
    streaming = build_streaming_science_projection(fixture.authority)
    sources: list[CompleteSourceEvidence] = []
    manifests_by_accession = {
        item["role_id"].removeprefix("complete/"): item
        for item in fixture.authority.role_manifests
    }
    for row in fixture.rows:
        accession = row["accession_number"]
        payload = fixture.payloads[manifests_by_accession[accession]["blob_name"]]
        start = row["extracted_text"]["start_byte"]
        end = row["extracted_text"]["end_byte"]
        raw = payload[start:end]
        normalized = normalize_filing_text(raw.decode("latin-1")).text.encode("utf-8")
        sources.append(
            CompleteSourceEvidence(
                accession_number=accession,
                acquisition_stage="development",
                complete_response=payload,
                extracted_primary=raw,
                normalized_text=normalized,
                seal_row=row,
            )
        )
    bundle = StageSourceBundle(
        config=SimpleNamespace(stage="development"),  # type: ignore[arg-type]
        submissions_snapshot=None,  # type: ignore[arg-type]
        master_reconciliation=None,  # type: ignore[arg-type]
        complete_sources=tuple(sources),
        session_dates=(),
        current_complete_accessions=tuple(
            row["accession_number"] for row in fixture.rows
        ),
        prior_seal=None,
        prior_bundle=None,
        seal=fixture.authority.stage_output.seal,
        seal_json=b"synthetic",
    )
    # The synthetic fixture has no 100-quarter master corpus.  The parity test
    # isolates the already-authenticated projection function after its replay
    # boundary; production never patches this check.
    monkeypatch.setattr(legacy_source, "detached_replay_stage_source", lambda _: {})
    non_streaming = legacy_source.build_legacy_science_projection(bundle)

    assert non_streaming.manifest_json == _canonical_bytes(
        dict(streaming.legacy_projection_manifest)
    )
    assert non_streaming.documents == streaming.legacy_documents


def test_public_manifest_contains_aggregates_but_no_private_row_values() -> None:
    fixture = _make_fixture()
    result = build_streaming_science_projection(fixture.authority)
    public = result.manifest_json
    assert public == _canonical_bytes(dict(result.manifest))
    for row in fixture.rows:
        assert row["accession_number"].encode("ascii") not in public
        assert row["primary_selection"]["document_identity"].encode("ascii") not in public
        assert row["complete_response"]["url"].encode("ascii") not in public
    assert b"Apple business revenue" not in public


def test_blob_hash_tamper_is_rejected_before_document_escape() -> None:
    fixture = _make_fixture()
    first = fixture.authority.role_manifests[0]["blob_name"]

    def tampered(name: str) -> bytes:
        payload = fixture.payloads[name]
        return payload + b"x" if name == first else payload

    with pytest.raises(BridgeViolation, match="blob_identity_invalid"):
        build_streaming_science_projection(fixture.authority, blob_loader=tampered)


def test_duplicate_or_missing_development_membership_is_rejected() -> None:
    def mutate(output: Any) -> None:
        output.seal["d_accessions"][1] = output.seal["d_accessions"][0]

    fixture = _make_fixture(mutate_before_authentication=mutate)
    with pytest.raises(BridgeViolation, match="development_membership_invalid"):
        build_streaming_science_projection(fixture.authority)


def test_stored_complete_evidence_mismatch_is_rejected() -> None:
    def mutate(output: Any) -> None:
        output.complete_evidence[0]["source_evidence_sha256"] = "9" * 64

    fixture = _make_fixture(mutate_before_authentication=mutate)
    with pytest.raises(BridgeViolation, match="complete_submission_parity_invalid"):
        build_streaming_science_projection(fixture.authority)


def test_normalized_text_hash_mismatch_is_rejected() -> None:
    def mutate(output: Any) -> None:
        output.seal["target_rows"][0]["normalized_text"]["sha256"] = (
            f"sha256:{'f' * 64}"
        )

    fixture = _make_fixture(mutate_before_authentication=mutate)
    # Keep authenticated parse evidence internally equal so the independent
    # normalization proof, not a shallow object comparison, is the rejection.
    with pytest.raises(
        BridgeViolation,
        match="normalized_text_identity_invalid|complete_submission_parity_invalid",
    ):
        build_streaming_science_projection(fixture.authority)


def test_authority_cannot_be_replaced_by_a_caller_mapping_or_blank_instance() -> None:
    with pytest.raises(BridgeViolation, match="authority_object_invalid"):
        build_streaming_science_projection({})  # type: ignore[arg-type]
    blank = AuthenticatedV38Authority()
    with pytest.raises((BridgeViolation, AttributeError)):
        build_streaming_science_projection(blank)


def test_private_root_requires_readable_contact_before_any_replay(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    private = repo / "data" / "aapl_sec_gemma_lean_evidence_v3_8"
    private.mkdir(parents=True)
    with pytest.raises(BridgeViolation, match="private_contact_required"):
        authenticate_v38_private_root(repo, private)


def test_bridge_has_no_acquisition_constructor_or_external_effect_surface() -> None:
    source = inspect.getsource(bridge)
    forbidden = (
        "DiskBackedSecAcquisition",
        "run_production_acquisition",
        "strict_transport_factory",
        "urllib.request",
        "requests.",
        "http.client",
        "api/chat",
        "query1.finance.yahoo.com",
    )
    assert all(value not in source for value in forbidden)
    assert "del payload" in source
