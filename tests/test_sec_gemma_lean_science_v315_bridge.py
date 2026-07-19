from __future__ import annotations

import copy
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
from agent_benchmark.sec_filing_gemma_contract import build_extractor_model_payload
from agent_benchmark import sec_gemma_lean_science_v315_bridge as bridge
from agent_benchmark import sec_gemma_lean_science_v315_contract as contract
from agent_benchmark.sec_gemma_lean_science_v315_bridge import (
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
        neither_filename = index in {0, 2}
        sgml_only = index == 1
        submissions_raw = "" if neither_filename or sgml_only else filename
        submissions_filename = submissions_raw or None
        sgml_filename = None if neither_filename else filename
        sec_filename = submissions_filename or sgml_filename
        selected_identity = sec_filename or "legacy-sequence-1-no-filename"
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
            "primaryDocument": submissions_raw,
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
            "sec_filename": sec_filename,
            "document_identity": selected_identity,
            "sgml_document_identity": (
                sgml_filename or "legacy-sequence-1-no-filename"
            ),
            "submissions_filename": submissions_filename,
            "sgml_filename": sgml_filename,
            "submissions_filename_missing": submissions_filename is None,
            "sgml_filename_missing": sgml_filename is None,
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


@pytest.fixture(scope="module")
def nullable_projection() -> tuple[_Fixture, ScienceBridgeProjection]:
    fixture = _make_fixture()
    return fixture, build_streaming_science_projection(fixture.authority)


def _blinded_request(
    proof: dict[str, Any] | Any,
    *,
    marker: str,
    request_bytes: bytes | None = None,
) -> dict[str, Any]:
    current = proof["current_record"]
    marker_number = int(marker)
    letters = ""
    remainder = marker_number
    while True:
        remainder, digit = divmod(remainder, 26)
        letters = chr(ord("a") + digit) + letters
        if remainder == 0:
            break
    sentences = [
        {
            "id": "C0001",
            "text": f"Safe public business sentence variant {letters}.",
        }
    ]
    if proof["prior_same_form_record"] is not None:
        sentences.append(
            {
                "id": "P0001",
                "text": f"Safe prior business sentence variant {letters}.",
            }
        )
    payload = (
        contract.canonical_json_bytes(
            build_extractor_model_payload(sentences)
        )
        if request_bytes is None
        else request_bytes
    )
    return {
        "schema_version": contract.BLINDED_MODEL_REQUEST_SCHEMA_VERSION,
        "accession_number": current["accession_number"],
        "form": current["form"],
        "availability_session": current["availability_session"],
        "preprocessed_event_sha256": f"{int(marker) + 1:064x}",
        "supplied_sentence_ids": [sentence["id"] for sentence in sentences],
        "request_sha256": _sha(payload),
        "request_bytes": payload,
    }


def _assert_contract_code(code: str, function: Any, *args: Any, **kwargs: Any) -> None:
    with pytest.raises(contract.ContractViolation) as captured:
        function(*args, **kwargs)
    assert captured.value.code == code


def test_nullable_provenance_modes_preserve_both_sgml_only_and_neither(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    fixture, result = nullable_projection
    records = {
        item["accession_number"]: item for item in result.compatibility_records
    }

    neither = records[fixture.rows[0]["accession_number"]]
    assert neither["primary_document_filename"] is None
    assert neither["official_primary_document_url"] is None
    assert (
        neither["selected_document_identity"]
        == contract.LEGACY_MISSING_DOCUMENT_IDENTITY
    )

    sgml_only = records[fixture.rows[1]["accession_number"]]
    assert fixture.rows[1]["primary_selection"]["submissions_filename"] is None
    assert fixture.rows[1]["primary_selection"]["sgml_filename"] == "filing001.htm"
    assert sgml_only["primary_document_filename"] == "filing001.htm"
    assert sgml_only["selected_document_identity"] == "filing001.htm"
    assert sgml_only["official_primary_document_url"] == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019301000002/filing001.htm"
    )

    both = records[fixture.rows[3]["accession_number"]]
    assert fixture.rows[3]["primary_selection"]["submissions_filename"] == (
        fixture.rows[3]["primary_selection"]["sgml_filename"]
    )
    assert both["primary_document_filename"] == "filing003.htm"
    assert both["selected_document_identity"] == "filing003.htm"


def test_all_nullable_projection_authorities_validate_exactly(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    assert contract.validate_compatibility_manifest(
        result.compatibility_manifest
    ) == dict(result.compatibility_manifest)
    assert contract.validate_nullable_universe_manifest(result.universe) == dict(
        result.universe
    )
    assert contract.validate_nullable_content_manifest(
        result.content_manifest
    ) == dict(result.content_manifest)
    proof = result.universe_event_proofs[-1]
    assert contract.validate_nullable_universe_event_proof(
        proof,
        universe=result.universe,
        content_manifest=result.content_manifest,
        compatibility_records=result.compatibility_records,
    ) == dict(proof)


def test_proof_rejects_a_self_consistent_wrong_top_level_authority_hash(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proof_body = copy.deepcopy(dict(result.universe_event_proofs[-1]))
    proof_body.pop("universe_event_proof_sha256")
    proof_body["content_manifest_sha256"] = "f" * 64
    tampered = contract.add_self_sha256(
        proof_body, field="universe_event_proof_sha256"
    )
    _assert_contract_code(
        "v315_contract_universe_event_proof_identity",
        contract.validate_nullable_universe_event_proof,
        tampered,
        universe=result.universe,
        content_manifest=result.content_manifest,
        compatibility_records=result.compatibility_records,
    )


def test_proof_rejects_duplicate_compatibility_membership(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    records = [copy.deepcopy(dict(item)) for item in result.compatibility_records]
    records[1] = copy.deepcopy(records[0])
    _assert_contract_code(
        "v315_contract_proof_compatibility_duplicate",
        contract.validate_nullable_universe_event_proof,
        result.universe_event_proofs[-1],
        universe=result.universe,
        content_manifest=result.content_manifest,
        compatibility_records=records,
    )


def test_blinded_request_rejects_current_private_locator_bytes(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proof = next(
        item
        for item in result.universe_event_proofs
        if item["current_record"]["primary_document_filename"] is not None
    )
    valid = _blinded_request(proof, marker="0")
    assert contract.validate_blinded_model_request(valid, proof=proof) == valid
    leaked_url = proof["current_record"]["official_primary_document_url"].encode(
        "utf-8"
    )
    leaked = _blinded_request(proof, marker="0", request_bytes=leaked_url)
    _assert_contract_code(
        "v315_contract_blinded_request_private_identity",
        contract.validate_blinded_model_request,
        leaked,
        proof=proof,
    )


def test_blinded_request_rejects_prior_private_locator_bytes(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proof = next(
        item
        for item in result.universe_event_proofs
        if item["prior_same_form_record"] is not None
    )
    prior_url = proof["prior_same_form_record"][
        "official_complete_submission_url"
    ].encode("utf-8")
    leaked = _blinded_request(proof, marker="1", request_bytes=prior_url)
    _assert_contract_code(
        "v315_contract_blinded_request_private_identity",
        contract.validate_blinded_model_request,
        leaked,
        proof=proof,
    )


def test_blinded_request_rejects_every_current_source_and_proof_identity_token(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proof = next(
        item
        for item in result.universe_event_proofs
        if item["current_record"]["primary_document_filename"] is not None
    )
    current = proof["current_record"]
    current_content = proof["current_content_record"]
    record_fields = (
        "accession_number",
        "subject_cik",
        "primary_document_filename",
        "selected_document_identity",
        "official_complete_submission_url",
        "official_primary_document_url",
        "source_record_sha256",
        "selected_text_sha256",
        "normalized_text_sha256",
        "complete_response_sha256",
        "compatibility_record_sha256",
    )
    proof_fields = (
        "compatibility_manifest_sha256",
        "universe_sha256",
        "calendar_sessions_sha256",
        "current_record_sha256",
        "current_content_record_sha256",
        "current_filing_sha256",
        "content_manifest_sha256",
        "universe_event_proof_sha256",
    )
    tokens = {
        value
        for record in (current, current_content)
        for field in record_fields
        if type(value := record.get(field)) is str and value
    }
    tokens.update(
        value
        for field in proof_fields
        if type(value := proof.get(field)) is str and value
    )
    for token in sorted(tokens):
        payload = contract.canonical_json_bytes(
            {"sentences": [{"id": "C0001", "text": f"safe {token} sentence"}]}
        )
        leaked = _blinded_request(proof, marker="2", request_bytes=payload)
        with pytest.raises(contract.ContractViolation) as captured:
            contract.validate_blinded_model_request(leaked, proof=proof)
        assert captured.value.code == (
            "v315_contract_blinded_request_private_identity"
        ), token


def test_blinded_request_rejects_every_prior_source_and_proof_identity_token(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proof = next(
        item
        for item in result.universe_event_proofs
        if item["prior_same_form_record"] is not None
        and item["prior_same_form_record"]["primary_document_filename"] is not None
    )
    prior = proof["prior_same_form_record"]
    prior_content = proof["prior_same_form_content_record"]
    record_fields = (
        "accession_number",
        "subject_cik",
        "primary_document_filename",
        "selected_document_identity",
        "official_complete_submission_url",
        "official_primary_document_url",
        "source_record_sha256",
        "selected_text_sha256",
        "normalized_text_sha256",
        "complete_response_sha256",
        "compatibility_record_sha256",
    )
    proof_fields = (
        "prior_same_form_record_sha256",
        "prior_same_form_content_record_sha256",
        "prior_same_form_filing_sha256",
    )
    tokens = {
        value
        for record in (prior, prior_content)
        for field in record_fields
        if type(value := record.get(field)) is str and value
    }
    tokens.update(
        value
        for field in proof_fields
        if type(value := proof.get(field)) is str and value
    )
    for token in sorted(tokens):
        payload = contract.canonical_json_bytes(
            {"sentences": [{"id": "P0001", "text": f"safe {token} sentence"}]}
        )
        leaked = _blinded_request(proof, marker="3", request_bytes=payload)
        with pytest.raises(contract.ContractViolation) as captured:
            contract.validate_blinded_model_request(leaked, proof=proof)
        assert captured.value.code == (
            "v315_contract_blinded_request_private_identity"
        ), token


def test_blinded_request_rejects_structured_current_and_prior_missingness(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proof = next(
        item
        for item in result.universe_event_proofs
        if item["prior_same_form_record"] is not None
    )
    safe_payload = build_extractor_model_payload(
        [
            {"id": "C0001", "text": "Safe public business sentence."},
            {"id": "P0001", "text": "Safe prior business sentence."},
        ]
    )
    safe = _blinded_request(
        proof, marker="4", request_bytes=contract.canonical_json_bytes(safe_payload)
    )
    safe["supplied_sentence_ids"] = ["C0001", "P0001"]
    assert contract.validate_blinded_model_request(safe, proof=proof) == safe

    for field in (
        "primary_document_filename_missing",
        "prior_official_primary_document_url_is_null",
        "source_locator_present",
    ):
        leaked_payload = build_extractor_model_payload(
            [
                {
                    "id": "C0001",
                    "text": "Safe public business sentence.",
                    field: True,
                },
                {"id": "P0001", "text": "Safe prior business sentence."},
            ]
        )
        leaked = _blinded_request(
            proof,
            marker="4",
            request_bytes=contract.canonical_json_bytes(leaked_payload),
        )
        leaked["supplied_sentence_ids"] = ["C0001", "P0001"]
        _assert_contract_code(
            "v315_contract_blinded_request_private_identity",
            contract.validate_blinded_model_request,
            leaked,
            proof=proof,
        )


@pytest.mark.parametrize(
    "leaked_text",
    (
        "AAPL demand improved.",
        "Apple demand improved.",
        "A.A.P.L. demand improved.",
        "Ap-ple demand improved.",
        "A\u200bAPL demand improved.",
        "\uff21\uff21\uff30\uff2c demand improved.",
        "\\u0041\\u0041\\u0050\\u004c demand improved.",
        "%41%41%50%4C demand improved.",
        "A&#65;PL demand improved.",
        "QUFQTA== demand improved.",
        "Conditions changed on 2018-01-01.",
        "Conditions changed on 05/17/18.",
        "Conditions changed on 05.17.18.",
        "Conditions changed on \uff10\uff15/\uff11\uff17/\uff11\uff18.",
        "Conditions changed in two thousand eighteen.",
        "The price return benchmark improved.",
        "Shares rose sharply.",
        "Shares surged sharply.",
        "The market rose sharply.",
        "The security gained value.",
        "The stock was up.",
        "Trading recommendation: BUY.",
        "Action: BUY.",
        "Position: LONG.",
        "Rating: BUY.",
        "Decision: HOLD.",
        "Buy recommendation.",
        "The model chose CASH.",
        "Score: positive.",
        "Score: 0.8.",
        "Score: high.",
        "Label: risk on.",
        "The return was positive.",
        "Buy & hold comparison.",
        "The filename missing flag was set.",
        "The official_primary_document_url is null.",
        "The URL is present.",
        "No filename was supplied.",
        "The filename's value is missing.",
        "The filename had no value.",
        "The URL field was populated.",
        "The file name is missing.",
        "See the URL for details.",
        "The filename follows.",
        "The accession number identifies the filing.",
        "The source locator changed.",
        "The official primary document is attached.",
        "The file-name follows.",
        "See the U.R.L. for details.",
    ),
)
def test_blinded_request_rejects_plain_text_identity_outcome_and_missingness(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
    leaked_text: str,
) -> None:
    _, result = nullable_projection
    proof = next(
        item
        for item in result.universe_event_proofs
        if item["current_record"]["primary_document_filename"] is not None
    )
    payload = build_extractor_model_payload([{"id": "C0001", "text": leaked_text}])
    leaked = _blinded_request(
        proof,
        marker="5",
        request_bytes=contract.canonical_json_bytes(payload),
    )
    leaked["supplied_sentence_ids"] = ["C0001"]
    _assert_contract_code(
        "v315_contract_blinded_request_private_identity",
        contract.validate_blinded_model_request,
        leaked,
        proof=proof,
    )


def test_blinded_request_plain_text_guard_preserves_unrelated_business_language(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proof = result.universe_event_proofs[0]
    safe_texts = (
        "Management may be missing a revenue estimate as market demand changes.",
        "The risk on our supply chain increased.",
        "Customers may return products under the standard warranty policy.",
        "Our trading partners may change supply terms.",
        "We benchmark operating costs against peers.",
        "Our market share increased as customer demand improved.",
        "Our market share rose as customer demand improved.",
        "Product prices increased because component costs rose.",
        "Product prices declined as competition increased.",
        "Our competitive position was positive as customer demand improved.",
        "Our legal position was neutral while litigation risk declined.",
        "The classification is positive for regulatory compliance.",
    )
    for safe_text in safe_texts:
        payload = build_extractor_model_payload(
            [{"id": "C0001", "text": safe_text}]
        )
        safe = _blinded_request(
            proof,
            marker="6",
            request_bytes=contract.canonical_json_bytes(payload),
        )
        safe["supplied_sentence_ids"] = ["C0001"]
        assert contract.validate_blinded_model_request(safe, proof=proof) == safe, safe_text


def test_blinded_request_requires_prior_sentence_parity_with_proof(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    without_prior = next(
        proof
        for proof in result.universe_event_proofs
        if proof["prior_same_form_record"] is None
    )
    with_prior = next(
        proof
        for proof in result.universe_event_proofs
        if proof["prior_same_form_record"] is not None
    )
    cases = (
        (
            without_prior,
            [
                {"id": "C0001", "text": "Safe public business sentence."},
                {"id": "P0001", "text": "Safe prior business sentence."},
            ],
        ),
        (
            with_prior,
            [{"id": "C0001", "text": "Safe public business sentence."}],
        ),
    )
    for marker, (proof, sentences) in enumerate(cases, start=20):
        leaked = _blinded_request(
            proof,
            marker=str(marker),
            request_bytes=contract.canonical_json_bytes(
                build_extractor_model_payload(sentences)
            ),
        )
        leaked["supplied_sentence_ids"] = [sentence["id"] for sentence in sentences]
        _assert_contract_code(
            "v315_contract_blinded_request_private_identity",
            contract.validate_blinded_model_request,
            leaked,
            proof=proof,
        )


@pytest.mark.parametrize(
    "sentence",
    (
        {"id": "C0001", "text": "Safe public business sentence.", "ticker": True},
        {"id": "C0001", "text": [65, 65, 80, 76]},
        {
            "id": "C0001",
            "text": "Safe public business sentence.",
            "ticker_codepoints": [65, 65, 80, 76],
        },
        {"id": "C0002", "text": "Safe public business sentence."},
        {"id": "P0001", "text": "Safe public business sentence."},
    ),
)
def test_blinded_request_rejects_noncanonical_sentence_structure(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
    sentence: dict[str, Any],
) -> None:
    _, result = nullable_projection
    proof = result.universe_event_proofs[0]
    payload = build_extractor_model_payload([sentence])
    leaked = _blinded_request(
        proof,
        marker="9",
        request_bytes=contract.canonical_json_bytes(payload),
    )
    _assert_contract_code(
        "v315_contract_blinded_request_private_identity",
        contract.validate_blinded_model_request,
        leaked,
        proof=proof,
    )


def test_blinded_request_rejects_case_changed_private_identity_and_malformed_json(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proof = next(
        item
        for item in result.universe_event_proofs
        if item["current_record"]["official_primary_document_url"] is not None
    )
    private_url = proof["current_record"]["official_primary_document_url"]
    for marker, request_bytes in (
        ("7", private_url.upper().encode("utf-8")),
        ("8", b"not canonical JSON"),
    ):
        leaked = _blinded_request(proof, marker=marker, request_bytes=request_bytes)
        _assert_contract_code(
            "v315_contract_blinded_request_private_identity",
            contract.validate_blinded_model_request,
            leaked,
            proof=proof,
        )


def test_model_slice_excludes_bytes_but_request_hash_binds_them(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proofs = list(result.universe_event_proofs)
    requests = [
        _blinded_request(proof, marker=str(index))
        for index, proof in enumerate(proofs)
    ]
    model_slice = contract.build_nullable_model_slice(requests, proofs)
    assert contract.validate_nullable_model_slice(model_slice) == model_slice

    changed_requests = copy.deepcopy(requests)
    changed_requests[0]["request_bytes"] = contract.canonical_json_bytes(
        build_extractor_model_payload(
            [
                {
                    "id": "C0001",
                    "text": "Safe public business sentence changed.",
                }
            ]
        )
    )
    changed_requests[0]["request_sha256"] = _sha(
        changed_requests[0]["request_bytes"]
    )
    changed_slice = contract.build_nullable_model_slice(changed_requests, proofs)
    assert changed_slice["model_slice_sha256"] != model_slice["model_slice_sha256"]

    stale = copy.deepcopy(model_slice)
    stale["model_requests"][0]["request_bytes"] += b"tamper"
    _assert_contract_code(
        "v315_contract_blinded_request_identity",
        contract.validate_nullable_model_slice,
        stale,
    )


def test_model_slice_rejects_wrong_request_proof_order_and_count(
    nullable_projection: tuple[_Fixture, ScienceBridgeProjection],
) -> None:
    _, result = nullable_projection
    proofs = list(result.universe_event_proofs)
    requests = [
        _blinded_request(proof, marker=str(index))
        for index, proof in enumerate(proofs)
    ]
    requests[0], requests[1] = requests[1], requests[0]
    _assert_contract_code(
        "v315_contract_blinded_request_identity",
        contract.build_nullable_model_slice,
        requests,
        proofs,
    )
    _assert_contract_code(
        "v315_contract_model_slice_count",
        contract.build_nullable_model_slice,
        requests[:-1],
        proofs[:-1],
    )


def test_bridge_rejects_any_count_other_than_73_present_and_two_null() -> None:
    def make_one_missing(output: Any) -> None:
        selection = output.seal["target_rows"][2]["primary_selection"]
        selection["sgml_filename"] = "filing002.htm"
        selection["sec_filename"] = "filing002.htm"
        selection["document_identity"] = "filing002.htm"
        selection["sgml_document_identity"] = "filing002.htm"
        selection["sgml_filename_missing"] = False

    fixture = _make_fixture(mutate_before_authentication=make_one_missing)
    with pytest.raises(BridgeViolation) as captured:
        build_streaming_science_projection(fixture.authority)
    assert captured.value.code == "nullable_filename_count_invalid"


@pytest.mark.parametrize(
    "mutate_hash",
    (
        lambda value: value.removeprefix("sha256:"),
        lambda value: "SHA256:" + value.removeprefix("sha256:"),
        lambda value: "sha256:" + value,
    ),
    ids=("bare", "uppercase-prefix", "doubled-prefix"),
)
def test_projection_requires_one_exact_lowercase_source_content_sha256_tag(
    mutate_hash: Callable[[str], str],
) -> None:
    def mutate(output: Any) -> None:
        layout = output.seal["target_rows"][0]["submissions_layout_provenance"]
        layout["source_content_sha256"] = mutate_hash(
            layout["source_content_sha256"]
        )

    fixture = _make_fixture(mutate_before_authentication=mutate)
    with pytest.raises(BridgeViolation) as captured:
        build_streaming_science_projection(fixture.authority)
    assert captured.value.code == "source_record_invalid"


def test_streaming_projection_recreates_all_75_legacy_documents_and_records() -> None:
    fixture = _make_fixture()
    result = build_streaming_science_projection(fixture.authority)

    assert type(result) is ScienceBridgeProjection
    assert result.stage == "development"
    assert len(result.legacy_documents) == 75
    assert all(type(item) is LegacyScienceDocument for item in result.legacy_documents)
    assert (
        len(result.compatibility_records)
        == len(result.records)
        == len(result.primary_documents)
        == len(result.universe_event_proofs)
        == 75
    )
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
        "availability_session",
        "artifact_stage",
        "primary_document_filename",
        "selected_document_identity",
        "official_complete_submission_url",
        "official_primary_document_url",
        "source_record_sha256",
        "selected_text_sha256",
        "normalized_text_sha256",
        "complete_response_sha256",
        "compatibility_record_sha256",
    }
    assert set(result.primary_documents[0]) == {
        "accession_number",
        "form",
        "availability_session",
        "acceptance_datetime",
        "primary_document_filename",
        "selected_document_identity",
        "official_complete_submission_url",
        "official_primary_document_url",
        "body",
    }
    assert result.manifest["document_count"] == 75
    assert result.manifest["filename_present_count"] == 73
    assert result.manifest["filename_missing_count"] == 2
    assert result.manifest["peak_live_complete_submission_blob_count"] == 1
    assert result.manifest["sec_request_count"] == 0
    assert result.manifest["set_parity"] is True
    assert result.manifest["typed_identity_parity"] is True
    assert result.manifest["nullable_filename_parity"] is True
    assert result.manifest["no_fabricated_primary_url"] is True
    null_rows = [
        item for item in result.records if item["primary_document_filename"] is None
    ]
    assert len(null_rows) == 2
    assert all(item["official_primary_document_url"] is None for item in null_rows)
    assert all(
        item["selected_document_identity"] == "legacy-sequence-1-no-filename"
        for item in null_rows
    )


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
    assert set(first_source) == {
        "sequence",
        "accession_number",
        "form",
        "filing_date",
        "acceptance_datetime",
        "availability_session",
        "primary_document_filename",
        "selected_document_identity",
        "source_record_sha256",
        "compatibility_record_sha256",
        "complete_response_sha256",
        "selected_text_sha256",
        "normalized_text_sha256",
        "source_order_row_sha256",
    }
    source_body = {
        "schema_version": (
            "aapl-sec-gemma-lean-science-v3-15-private-source-identity-v1"
        ),
        "source_url": "https://data.sec.gov/submissions/CIK0000320193.json",
        "source_content_sha256": "1" * 64,
        "raw_row_identity_sha256": typed_hash,
        "accession_number": row["accession_number"],
        "subject_cik": "0000320193",
        "form": row["form"],
        "acceptance_datetime_source": typed["acceptanceDateTime"],
        "acceptance_datetime": typed["acceptanceDateTime"],
        "filing_date": row["filing_date"],
        "filing_date_change": None,
        "submissions_primary_document_raw": typed["primaryDocument"],
        "submissions_filename": row["primary_selection"]["submissions_filename"],
        "sgml_filename": row["primary_selection"]["sgml_filename"],
        "sec_filename": row["primary_selection"]["sec_filename"],
        "selected_document_identity": row["primary_selection"][
            "document_identity"
        ],
        "complete_response_sha256": row["complete_response"]["sha256"],
    }
    record = next(
        item
        for item in result.compatibility_records
        if item["accession_number"] == row["accession_number"]
    )
    assert record["source_record_sha256"] == _canonical_sha(source_body)
    assert record["primary_document_filename"] is None
    assert record["official_primary_document_url"] is None
    assert result.universe["universe_sha256"] == result.manifest["universe_sha256"]
    assert (
        result.content_manifest["content_manifest_sha256"]
        == result.manifest["content_manifest_sha256"]
    )


def test_source_order_prior_links_and_event_order_are_exact_and_distinct() -> None:
    result = build_streaming_science_projection(_make_fixture().authority)
    source_accessions = [item["accession_number"] for item in result.source_order]
    assert source_accessions == sorted(source_accessions)
    assert result.event_order[0]["accession_number"] == source_accessions[1]
    assert result.event_order[1]["accession_number"] == source_accessions[0]

    prior_by_form: dict[str, str] = {}
    for link in result.prior_links:
        assert link["prior_accession_number"] == prior_by_form.get(link["form"])
        assert link["policy"] == (
            "immediate_predecessor_same_form_by_availability_session_and_accession"
        )
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


def test_hash_path_accepts_windows_fstat_ctime_alias_but_rejects_visible_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stable.bin"
    payload = b"stable public authority bytes"
    path.write_bytes(payload)
    actual = path.stat()

    def details(*, ctime_ns: int) -> SimpleNamespace:
        return SimpleNamespace(
            st_dev=actual.st_dev,
            st_ino=actual.st_ino,
            st_mode=actual.st_mode,
            st_size=actual.st_size,
            st_mtime_ns=actual.st_mtime_ns,
            st_ctime_ns=ctime_ns,
            st_nlink=actual.st_nlink,
            st_file_attributes=getattr(actual, "st_file_attributes", 0),
        )

    visible = details(ctime_ns=101)
    opened = details(ctime_ns=actual.st_mtime_ns)
    monkeypatch.setattr(bridge, "_secure_file", lambda _path: visible)
    monkeypatch.setattr(bridge.os, "fstat", lambda _descriptor: opened)

    digest, size, returned = bridge._hash_path(path)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)
    assert returned is visible

    changed = details(ctime_ns=102)
    snapshots = iter((visible, changed))
    monkeypatch.setattr(bridge, "_secure_file", lambda _path: next(snapshots))
    with pytest.raises(BridgeViolation, match="private_authority_changed_during_read"):
        bridge._hash_path(path)


def test_canonical_mapping_has_no_second_read_swap_restore_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "authority.json"
    original = {"authority": "authenticated", "sequence": 1}
    replacement = {"authority": "swapped", "sequence": 2}
    path.write_bytes(bridge._canonical_bytes(original))
    second_reads: list[Path] = []

    def swap_read_restore(target: Path) -> bytes:
        second_reads.append(target)
        target.write_bytes(bridge._canonical_bytes(replacement))
        try:
            return bridge._canonical_bytes(replacement)
        finally:
            target.write_bytes(bridge._canonical_bytes(original))

    monkeypatch.setattr(Path, "read_bytes", swap_read_restore)
    assert bridge._read_canonical_mapping(path) == original
    assert second_reads == []


def test_authenticated_blob_has_no_hash_then_reopen_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "000000-authenticated.blob"
    payload = b"one descriptor supplies both the digest and returned bytes"
    path.write_bytes(payload)
    second_reads: list[Path] = []

    def forbidden_second_read(target: Path) -> bytes:
        second_reads.append(target)
        raise AssertionError("authenticated blob was reopened")

    monkeypatch.setattr(Path, "read_bytes", forbidden_second_read)
    assert bridge._read_authenticated_blob(
        path,
        expected_sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        expected_size=len(payload),
    ) == payload
    assert second_reads == []


def test_public_source_authentication_never_reopens_hashed_json_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    second_reads: list[tuple[Path, tuple[Any, ...], dict[str, Any]]] = []

    def forbidden_read_text(
        target: Path, *args: Any, **kwargs: Any
    ) -> str:
        second_reads.append((target, args, kwargs))
        raise AssertionError("authenticated public JSON was reopened")

    monkeypatch.setattr(Path, "read_text", forbidden_read_text)
    terminal, public = bridge._authenticate_public_source(repo_root)
    assert terminal["acquisition_sha256"] == contract.V38_TERMINAL_INTERNAL_SHA256
    assert type(public) is dict
    assert second_reads == []


def test_inventory_digest_uses_exact_v38_historical_key_spellings(
    tmp_path: Path,
) -> None:
    rows = (
        bridge._FileSnapshot(
            "blob", 0, tmp_path / "blob", "1" * 64, 123, 1, 2, 1, 3, 4
        ),
        bridge._FileSnapshot(
            "journal", 7, tmp_path / "journal", "a" * 64, 456, 1, 3, 1, 5, 6
        ),
    )
    assert bridge._inventory_digest(rows) == (
        "f523d02e054f8c8c7d19849053c70d6f5a8ed8c97d1ec08a3ea010ca734d3e12"
    )


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
