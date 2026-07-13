from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import agent_benchmark.sec_filing_gemma_market_acquirer as market_acquirer
import agent_benchmark.sec_filing_gemma_reveal_store as reveal_store_module
from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_market_evidence import MARKET_SYMBOLS
from agent_benchmark.sec_filing_gemma_reveal_store import (
    CURRENT_TIP_PENDING_SCHEMA_VERSION,
    DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME,
    DEVELOPMENT_MARKET_COMPLETE_MARKER_SCHEMA_VERSION,
    DEVELOPMENT_MARKET_COMPONENT_ID,
    MARKET_SOURCE_COMPONENT_DIRECTORY_NAME,
    MAX_MARKET_BATCH_FILES,
    STAGE_OUTPUTS_DIRECTORY_NAME,
    SecFilingGemmaRevealStore,
    SecFilingGemmaRevealStoreError,
)
from tests import test_sec_filing_gemma_market_acquirer as market_scaffold
from tests import test_sec_filing_gemma_reveal_store as store_scaffold


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def synthetic_acquisition_bundle() -> dict:
    """Build contract-valid owned-policy evidence without performing network I/O.

    The acquisition test seam correctly labels its injected transport as
    untrusted.  This store test changes only that receipt policy and recomputes
    every affected identity so the finalizer can exercise its production-policy
    gate without pretending the fixture itself is real acquired evidence.
    """

    bundle = copy.deepcopy(market_scaffold.acquisition_bundle.__wrapped__())
    acquisition_receipt = bundle["acquisition_receipt"]
    acquisition_receipt["transport_policy"] = (
        market_acquirer._receipt_transport_policy(trusted_transport=True)
    )
    receipt_body = {
        key: value
        for key, value in acquisition_receipt.items()
        if key != "acquisition_receipt_sha256"
    }
    receipt_hash = canonical_sha256(receipt_body)
    acquisition_receipt["acquisition_receipt_sha256"] = receipt_hash
    bundle["acquisition_receipt_sha256"] = receipt_hash
    bundle_body = {
        key: bundle[key]
        for key in (
            "schema_version",
            "artifact_stage",
            "acquisition_receipt_sha256",
            "acquisition_plan_sha256",
            "source_manifest_sha256",
            "market_stage_manifest_sha256",
            "source_reconciliation_sha256",
            "raw_response_sha256s",
            "artifact_sha256s",
            "window_sha256s",
        )
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle_body)
    _validate_acquisition_bundle(bundle)
    return bundle


@pytest.fixture(scope="module")
def owned_response_bodies() -> dict[str, bytes]:
    return market_scaffold._development_response_bodies()


def _owned_bundle_and_capability_for_claim(
    claim: dict,
    *,
    response_bodies: dict[str, bytes],
) -> tuple[dict, object]:
    owned_result = market_scaffold._owned_acquisition_without_network(
        response_bodies
    )
    bundle, capability = (
        market_acquirer._unwrap_owned_development_market_acquisition(
            owned_result
        )
    )
    assert bundle["acquisition_plan_sha256"] == claim[
        "market_acquisition_plan_sha256"
    ]
    market_acquirer._bind_owned_market_transport_capability_to_claim(
        capability,
        development_root_scope_sha256=claim[
            "development_root_scope_sha256"
        ],
        claim_sha256=claim["claim_sha256"],
    )
    return bundle, capability


def _new_store(tmp_path: Path, label: str) -> SecFilingGemmaRevealStore:
    base = tmp_path / label
    base.mkdir()
    return store_scaffold._store(base)


def _prepare_terminal_development_sec_root(
    store: SecFilingGemmaRevealStore,
    *,
    salt: str,
) -> dict:
    registered, candidate, universe, plan = (
        store_scaffold._registered_development_root(
            store,
            salt=salt,
        )
    )
    sec_claim_result = store.claim_owned_development_sec_root_execution(
        development_content_root_plan=plan,
        sec_user_agent_sha256=store_scaffold.SEC_TEST_USER_AGENT_SHA256,
    )
    sec_claim = sec_claim_result["claim"]
    store_scaffold._write_fixed_development_sec_root(
        store,
        sec_claim,
        plan,
    )
    sec_reader = store._record_owned_development_sec_root_reader_output(
        development_root_scope_sha256=plan["development_root_scope_sha256"],
    )
    return {
        "registered": registered,
        "candidate": candidate,
        "universe": universe,
        "plan": plan,
        "sec_claim": sec_claim,
        "sec_reader": sec_reader,
    }


def _prepare_market_claim(
    store: SecFilingGemmaRevealStore,
    *,
    salt: str,
) -> dict:
    prepared = _prepare_terminal_development_sec_root(store, salt=salt)
    scope_hash = prepared["plan"]["development_root_scope_sha256"]
    claim_result = store.claim_owned_development_market_execution(
        development_root_scope_sha256=scope_hash,
    )
    return {
        **prepared,
        "scope_hash": scope_hash,
        "claim_result": claim_result,
        "claim": claim_result["claim"],
    }


def _validate_acquisition_bundle(bundle: dict) -> dict:
    return market_acquirer.validate_development_market_acquisition_bundle(
        bundle,
        expected_acquisition_plan_sha256=bundle["acquisition_plan_sha256"],
        expected_acquisition_receipt_sha256=bundle[
            "acquisition_receipt_sha256"
        ],
        expected_bundle_sha256=bundle["bundle_sha256"],
    )


def _write_fixed_market_component(
    store: SecFilingGemmaRevealStore,
    claim: dict,
    bundle: dict,
    *,
    marker_overrides: dict | None = None,
) -> tuple[Path, Path, list[dict], dict]:
    """Persist the exact 22 payload files and canonical twenty-third marker."""

    validation = _validate_acquisition_bundle(bundle)
    component_directory = (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / claim["claim_sha256"]
        / MARKET_SOURCE_COMPONENT_DIRECTORY_NAME
    )
    component_directory.mkdir(parents=True)
    payloads = {
        **{
            f"raw-response-{symbol}.json": bundle[
                "raw_response_bytes_by_symbol"
            ][symbol]
            for symbol in MARKET_SYMBOLS
        },
        **{
            f"artifact-{symbol}.json": bundle["artifact_bytes_by_symbol"][
                symbol
            ]
            for symbol in MARKET_SYMBOLS
        },
        **{
            f"window-{symbol}.json": bundle["window_bytes_by_symbol"][symbol]
            for symbol in MARKET_SYMBOLS
        },
        "source-manifest.json": reveal_store_module._encoded_state(
            bundle["source_manifest"]
        ),
        "stage-manifest.json": reveal_store_module._encoded_state(
            bundle["stage_manifest"]
        ),
        "reconciliation-receipt.json": reveal_store_module._encoded_state(
            bundle["reconciliation_receipt"]
        ),
        "acquisition-receipt.json": reveal_store_module._encoded_state(
            bundle["acquisition_receipt"]
        ),
    }
    byte_index: list[dict] = []
    for ordinal, (logical_id, relative_path) in enumerate(
        reveal_store_module._expected_market_byte_layout(),
        start=1,
    ):
        payload = payloads[relative_path]
        (component_directory / relative_path).write_bytes(payload)
        byte_index.append(
            {
                "ordinal": ordinal,
                "logical_id": logical_id,
                "relative_path": relative_path,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    marker_body = {
        "schema_version": DEVELOPMENT_MARKET_COMPLETE_MARKER_SCHEMA_VERSION,
        "development_root_scope_sha256": claim[
            "development_root_scope_sha256"
        ],
        "claim_sha256": claim["claim_sha256"],
        "market_acquisition_plan_sha256": claim[
            "market_acquisition_plan_sha256"
        ],
        "acquisition_receipt_sha256": bundle[
            "acquisition_receipt_sha256"
        ],
        "acquisition_bundle_sha256": bundle["bundle_sha256"],
        "acquisition_validation_sha256": validation["validation_sha256"],
        "source_manifest_sha256": bundle["source_manifest_sha256"],
        "market_stage_manifest_sha256": bundle[
            "market_stage_manifest_sha256"
        ],
        "source_reconciliation_sha256": bundle[
            "source_reconciliation_sha256"
        ],
        "market_component_id": DEVELOPMENT_MARKET_COMPONENT_ID,
        "byte_index": byte_index,
        "byte_index_sha256": canonical_sha256(byte_index),
        "byte_count_total": sum(item["byte_count"] for item in byte_index),
    }
    if marker_overrides is not None:
        marker_body.update(marker_overrides)
    marker = {
        **marker_body,
        "marker_sha256": canonical_sha256(marker_body),
    }
    marker_path = component_directory / DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME
    marker_path.write_bytes(reveal_store_module._encoded_state(marker))
    return component_directory, marker_path, byte_index, validation


def test_genesis_tip_contains_empty_market_lifecycle_maps(tmp_path: Path) -> None:
    store = _new_store(tmp_path, "genesis")
    state = store.initialize()
    tip = store.load_current_tip_anchor()

    assert state == store.load()
    assert tip["development_market_execution_claims"] == {}
    assert tip["development_market_reader_receipts"] == {}
    assert tip["development_market_execution_aborts"] == {}


def test_market_claim_requires_terminal_development_sec_reader_and_is_idempotent(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path, "claim")
    _registered, candidate, _universe, plan = (
        store_scaffold._registered_development_root(
            store,
            salt="market-claim",
        )
    )
    scope_hash = plan["development_root_scope_sha256"]
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="Could not claim the exact development market batch",
    ):
        store.claim_owned_development_market_execution(
            development_root_scope_sha256=scope_hash,
        )

    sec_claim = store.claim_owned_development_sec_root_execution(
        development_content_root_plan=plan,
        sec_user_agent_sha256=store_scaffold.SEC_TEST_USER_AGENT_SHA256,
    )["claim"]
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="Could not claim the exact development market batch",
    ):
        store.claim_owned_development_market_execution(
            development_root_scope_sha256=scope_hash,
        )

    store_scaffold._write_fixed_development_sec_root(store, sec_claim, plan)
    sec_reader = store._record_owned_development_sec_root_reader_output(
        development_root_scope_sha256=scope_hash,
    )
    first = store.claim_owned_development_market_execution(
        development_root_scope_sha256=scope_hash,
    )
    stable_tip = store.current_tip_anchor_path.read_bytes()
    repeated = store.claim_owned_development_market_execution(
        development_root_scope_sha256=scope_hash,
    )
    claim = first["claim"]

    assert first["created"] is True
    assert repeated == {
        "claim": claim,
        "created": False,
        "reader_receipt": None,
        "abort": None,
    }
    assert store.current_tip_anchor_path.read_bytes() == stable_tip
    assert claim["development_sec_execution_claim_sha256"] == sec_claim[
        "claim_sha256"
    ]
    assert claim["development_sec_reader_receipt_sha256"] == sec_reader[
        "receipt_sha256"
    ]
    assert claim["market_acquisition_plan"] == (
        market_acquirer.build_development_market_acquisition_plan()
    )
    assert claim["execution_source_hashes"] == (
        reveal_store_module._market_execution_source_hashes(
            store_scaffold.REPO_ROOT
        )
    )
    assert claim["candidate_source_hashes_sha256"] == canonical_sha256(
        candidate["bindings"]["source_hashes"]
    )


def test_market_reader_replays_exact_twenty_three_file_component_and_is_idempotent(
    tmp_path: Path,
    owned_response_bodies: dict[str, bytes],
) -> None:
    store = _new_store(tmp_path, "reader")
    prepared = _prepare_market_claim(store, salt="market-reader")
    acquisition_bundle, capability = _owned_bundle_and_capability_for_claim(
        prepared["claim"],
        response_bodies=owned_response_bodies,
    )
    component, marker_path, byte_index, validation = _write_fixed_market_component(
        store,
        prepared["claim"],
        acquisition_bundle,
    )
    receipt = store._record_owned_development_market_reader_output(
        development_root_scope_sha256=prepared["scope_hash"],
        owned_transport_capability=capability,
    )
    stable_tip = store.current_tip_anchor_path.read_bytes()
    repeated = store._record_owned_development_market_reader_output(
        development_root_scope_sha256=prepared["scope_hash"],
    )
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="replay without capabilities",
    ):
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
            owned_transport_capability=capability,
        )

    assert marker_path.is_file()
    assert len(list(component.iterdir())) == MAX_MARKET_BATCH_FILES == 23
    assert len(byte_index) == 22
    assert receipt == repeated
    assert store.current_tip_anchor_path.read_bytes() == stable_tip
    assert receipt["acquisition_receipt_sha256"] == (
        acquisition_bundle["acquisition_receipt_sha256"]
    )
    assert receipt["acquisition_bundle_sha256"] == (
        acquisition_bundle["bundle_sha256"]
    )
    assert receipt["acquisition_validation_sha256"] == validation[
        "validation_sha256"
    ]
    assert receipt["raw_response_sha256s"] == acquisition_bundle[
        "raw_response_sha256s"
    ]
    assert receipt["artifact_sha256s"] == acquisition_bundle[
        "artifact_sha256s"
    ]
    assert receipt["window_sha256s"] == acquisition_bundle[
        "window_sha256s"
    ]
    assert receipt["fresh_network_provenance_claimed"] is False
    assert receipt["provider_response_normalization_replayed_by_store"] is True
    assert receipt["owned_transport_attested_by_store"] is True
    assert "capability" not in repr(receipt).lower()


def test_prewritten_trusted_marker_without_owned_capability_cannot_mint_receipt(
    tmp_path: Path,
    synthetic_acquisition_bundle: dict,
) -> None:
    store = _new_store(tmp_path, "synthetic-marker")
    prepared = _prepare_market_claim(store, salt="market-synthetic-marker")
    _write_fixed_market_component(
        store,
        prepared["claim"],
        synthetic_acquisition_bundle,
    )

    for capability in (None, object()):
        with pytest.raises(
            SecFilingGemmaRevealStoreError,
            match="same-execution owned Yahoo transport capability",
        ):
            store._record_owned_development_market_reader_output(
                development_root_scope_sha256=prepared["scope_hash"],
                owned_transport_capability=capability,
            )
        assert store.load_current_tip_anchor()[
            "development_market_reader_receipts"
        ] == {}


def test_market_reader_enforces_aggregate_raw_response_cap_before_commit(
    tmp_path: Path,
    synthetic_acquisition_bundle: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path, "aggregate-cap")
    prepared = _prepare_market_claim(store, salt="market-aggregate-cap")
    _write_fixed_market_component(
        store,
        prepared["claim"],
        synthetic_acquisition_bundle,
    )
    raw_total = sum(
        len(payload)
        for payload in synthetic_acquisition_bundle[
            "raw_response_bytes_by_symbol"
        ].values()
    )
    monkeypatch.setattr(
        reveal_store_module,
        "YAHOO_MAX_TOTAL_RESPONSE_BYTES",
        raw_total - 1,
    )

    with patch.object(
        reveal_store_module,
        "_read_owned_sec_indexed_payloads",
        side_effect=AssertionError("payload bytes were read before prevalidation"),
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="aggregate byte ceiling",
    ):
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
        )
    assert store.load_current_tip_anchor()[
        "development_market_reader_receipts"
    ] == {}


def test_market_reader_rejects_declared_oversized_file_before_payload_read(
    tmp_path: Path,
    synthetic_acquisition_bundle: dict,
) -> None:
    store = _new_store(tmp_path, "declared-file-cap")
    prepared = _prepare_market_claim(store, salt="market-declared-file-cap")
    _component, marker_path, byte_index, _validation = (
        _write_fixed_market_component(
            store,
            prepared["claim"],
            synthetic_acquisition_bundle,
        )
    )
    oversized_index = copy.deepcopy(byte_index)
    oversized_index[0]["byte_count"] = market_acquirer.YAHOO_MAX_RESPONSE_BYTES + 1
    marker = json.loads(marker_path.read_bytes())
    marker_body = {
        **{
            key: value
            for key, value in marker.items()
            if key != "marker_sha256"
        },
        "byte_index": oversized_index,
        "byte_index_sha256": canonical_sha256(oversized_index),
        "byte_count_total": sum(
            item["byte_count"] for item in oversized_index
        ),
    }
    marker_path.write_bytes(
        reveal_store_module._encoded_state(
            {
                **marker_body,
                "marker_sha256": canonical_sha256(marker_body),
            }
        )
    )

    with patch.object(
        reveal_store_module,
        "_read_owned_sec_indexed_payloads",
        side_effect=AssertionError("payload bytes were read before prevalidation"),
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="exact layout",
    ):
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
        )


def test_market_reader_rejects_underdeclared_actual_file_before_payload_read(
    tmp_path: Path,
    synthetic_acquisition_bundle: dict,
) -> None:
    store = _new_store(tmp_path, "underdeclared-file-size")
    prepared = _prepare_market_claim(store, salt="market-underdeclared-file-size")
    component, _marker_path, _byte_index, _validation = (
        _write_fixed_market_component(
            store,
            prepared["claim"],
            synthetic_acquisition_bundle,
        )
    )
    raw_path = component / "raw-response-AAPL.json"
    raw_path.write_bytes(raw_path.read_bytes() + b" ")

    with patch.object(
        reveal_store_module,
        "_read_owned_sec_indexed_payloads",
        side_effect=AssertionError("payload bytes were read before prevalidation"),
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="file size differs",
    ):
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
        )


def test_market_reader_counts_completion_marker_inside_component_cap(
    tmp_path: Path,
    synthetic_acquisition_bundle: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path, "marker-total-cap")
    prepared = _prepare_market_claim(store, salt="market-marker-total-cap")
    _component, marker_path, byte_index, _validation = (
        _write_fixed_market_component(
            store,
            prepared["claim"],
            synthetic_acquisition_bundle,
        )
    )
    payload_total = sum(item["byte_count"] for item in byte_index)
    monkeypatch.setattr(
        reveal_store_module,
        "MAX_MARKET_BATCH_TOTAL_BYTES",
        payload_total + len(marker_path.read_bytes()) - 1,
    )

    with patch.object(
        reveal_store_module,
        "_read_owned_sec_indexed_payloads",
        side_effect=AssertionError("payload bytes were read before prevalidation"),
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="byte index is inconsistent",
    ):
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
        )


def test_market_reader_and_abort_are_mutually_exclusive_and_abort_is_idempotent(
    tmp_path: Path,
    owned_response_bodies: dict[str, bytes],
) -> None:
    aborted_store = _new_store(tmp_path, "abort")
    aborted = _prepare_market_claim(aborted_store, salt="market-abort")
    abort = aborted_store.abort_owned_development_market_execution(
        development_root_scope_sha256=aborted["scope_hash"],
        reason="external_effect_failed_or_completion_unknown",
    )
    stable_abort_tip = aborted_store.current_tip_anchor_path.read_bytes()
    repeated_abort = aborted_store.abort_owned_development_market_execution(
        development_root_scope_sha256=aborted["scope_hash"],
        reason="external_effect_failed_or_completion_unknown",
    )
    assert repeated_abort == abort
    assert aborted_store.current_tip_anchor_path.read_bytes() == stable_abort_tip
    assert abort["external_effect_retry_permitted"] is False
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="Aborted development market execution",
    ):
        aborted_store._record_owned_development_market_reader_output(
            development_root_scope_sha256=aborted["scope_hash"],
        )

    completed_store = _new_store(tmp_path, "completed")
    completed = _prepare_market_claim(completed_store, salt="market-completed")
    acquisition_bundle, capability = _owned_bundle_and_capability_for_claim(
        completed["claim"],
        response_bodies=owned_response_bodies,
    )
    _write_fixed_market_component(
        completed_store,
        completed["claim"],
        acquisition_bundle,
    )
    completed_store._record_owned_development_market_reader_output(
        development_root_scope_sha256=completed["scope_hash"],
        owned_transport_capability=capability,
    )
    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="Completed development market execution cannot be aborted",
    ):
        completed_store.abort_owned_development_market_execution(
            development_root_scope_sha256=completed["scope_hash"],
            reason="external_effect_failed_or_completion_unknown",
        )


def test_market_reader_rejects_source_substitution_after_durable_claim(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path, "source-substitution")
    prepared = _prepare_market_claim(store, salt="market-source-substitution")
    changed_sources = copy.deepcopy(prepared["claim"]["execution_source_hashes"])
    changed_sources["market_acquirer"] = _digest("substituted-market-acquirer")

    with patch.object(
        reveal_store_module,
        "_market_execution_source_hashes",
        return_value=changed_sources,
    ), pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="sources changed after the durable claim",
    ):
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
        )


@pytest.mark.parametrize("attack", ("altered", "extra", "case_collision"))
def test_market_reader_rejects_altered_extra_or_case_colliding_files(
    tmp_path: Path,
    synthetic_acquisition_bundle: dict,
    attack: str,
) -> None:
    store = _new_store(tmp_path, f"layout-{attack}")
    prepared = _prepare_market_claim(store, salt=f"market-layout-{attack}")
    component, _marker, _index, _validation = _write_fixed_market_component(
        store,
        prepared["claim"],
        synthetic_acquisition_bundle,
    )
    if attack == "altered":
        target = component / "raw-response-AAPL.json"
        target.write_bytes(target.read_bytes() + b" ")
    elif attack == "extra":
        (component / "unexpected.json").write_bytes(b"{}")
    else:
        target = component / "RAW-RESPONSE-AAPL.JSON"
        target.write_bytes((component / "raw-response-AAPL.json").read_bytes())
        names = [item.name for item in component.iterdir()]
        if len(names) == len({name.casefold() for name in names}):
            pytest.skip("case-colliding filenames are unavailable on this filesystem")

    with pytest.raises(SecFilingGemmaRevealStoreError):
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
        )


def test_market_reader_rejects_marker_with_wrong_acquisition_validation_binding(
    tmp_path: Path,
    synthetic_acquisition_bundle: dict,
) -> None:
    store = _new_store(tmp_path, "validation-binding")
    prepared = _prepare_market_claim(store, salt="market-validation-binding")
    _write_fixed_market_component(
        store,
        prepared["claim"],
        synthetic_acquisition_bundle,
        marker_overrides={
            "acquisition_validation_sha256": _digest(
                "wrong-acquisition-validation"
            )
        },
    )

    with pytest.raises(
        SecFilingGemmaRevealStoreError,
        match="marker differs from semantic replay",
    ):
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
        )


def test_market_claim_pending_tip_wal_recovers_once(tmp_path: Path) -> None:
    store = _new_store(tmp_path, "claim-wal")
    prepared = _prepare_terminal_development_sec_root(
        store,
        salt="market-claim-wal",
    )
    scope_hash = prepared["plan"]["development_root_scope_sha256"]
    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()
    real_atomic_replace = reveal_store_module._atomic_replace
    crashed = False

    def crash_after_pending_tip(path: Path, payload: bytes) -> None:
        nonlocal crashed
        real_atomic_replace(path, payload)
        parsed = json.loads(payload)
        if (
            not crashed
            and path == store.current_tip_anchor_path
            and parsed.get("schema_version") == CURRENT_TIP_PENDING_SCHEMA_VERSION
        ):
            crashed = True
            raise RuntimeError("simulated development market claim pending-tip crash")

    def operation() -> dict:
        return store.claim_owned_development_market_execution(
            development_root_scope_sha256=scope_hash,
        )

    with patch.object(
        reveal_store_module,
        "_atomic_replace",
        crash_after_pending_tip,
    ), pytest.raises(RuntimeError, match="market claim pending-tip crash"):
        operation()

    pending = json.loads(store.current_tip_anchor_path.read_bytes())
    assert pending["schema_version"] == CURRENT_TIP_PENDING_SCHEMA_VERSION
    expected_claim = pending["next_tip_anchor"][
        "development_market_execution_claims"
    ][scope_hash]

    recovered = operation()
    assert recovered == {
        "claim": expected_claim,
        "created": False,
        "reader_receipt": None,
        "abort": None,
    }
    assert store.state_path.read_bytes() == state_bytes
    recovered_tip = store.load_current_tip_anchor()
    assert recovered_tip["revision"] == tip_before["revision"] + 1
    assert recovered_tip["development_market_execution_claims"] == {
        scope_hash: expected_claim
    }
    stable_tip = store.current_tip_anchor_path.read_bytes()
    assert operation() == recovered
    assert store.current_tip_anchor_path.read_bytes() == stable_tip


def test_market_reader_pending_tip_wal_recovers_after_capability_consumption(
    tmp_path: Path,
    owned_response_bodies: dict[str, bytes],
) -> None:
    store = _new_store(tmp_path, "reader-wal")
    prepared = _prepare_market_claim(store, salt="market-reader-wal")
    bundle, capability = _owned_bundle_and_capability_for_claim(
        prepared["claim"],
        response_bodies=owned_response_bodies,
    )
    _write_fixed_market_component(store, prepared["claim"], bundle)
    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()
    real_atomic_replace = reveal_store_module._atomic_replace
    crashed = False

    def crash_after_pending_tip(path: Path, payload: bytes) -> None:
        nonlocal crashed
        real_atomic_replace(path, payload)
        parsed = json.loads(payload)
        if (
            not crashed
            and path == store.current_tip_anchor_path
            and parsed.get("schema_version") == CURRENT_TIP_PENDING_SCHEMA_VERSION
        ):
            crashed = True
            raise RuntimeError("simulated development market reader pending-tip crash")

    with patch.object(
        reveal_store_module,
        "_atomic_replace",
        crash_after_pending_tip,
    ), pytest.raises(RuntimeError, match="market reader pending-tip crash"):
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
            owned_transport_capability=capability,
        )

    pending = json.loads(store.current_tip_anchor_path.read_bytes())
    assert pending["schema_version"] == CURRENT_TIP_PENDING_SCHEMA_VERSION
    expected_receipt = pending["next_tip_anchor"][
        "development_market_reader_receipts"
    ][prepared["scope_hash"]]

    recovered = store._record_owned_development_market_reader_output(
        development_root_scope_sha256=prepared["scope_hash"],
    )
    assert recovered == expected_receipt
    assert store.state_path.read_bytes() == state_bytes
    recovered_tip = store.load_current_tip_anchor()
    assert recovered_tip["revision"] == tip_before["revision"] + 1
    assert recovered_tip["development_market_reader_receipts"] == {
        prepared["scope_hash"]: expected_receipt
    }
    stable_tip = store.current_tip_anchor_path.read_bytes()
    assert (
        store._record_owned_development_market_reader_output(
            development_root_scope_sha256=prepared["scope_hash"],
        )
        == recovered
    )
    assert store.current_tip_anchor_path.read_bytes() == stable_tip


def test_market_abort_pending_tip_wal_recovers_once(tmp_path: Path) -> None:
    store = _new_store(tmp_path, "abort-wal")
    prepared = _prepare_market_claim(store, salt="market-abort-wal")
    state_bytes = store.state_path.read_bytes()
    tip_before = store.load_current_tip_anchor()
    real_atomic_replace = reveal_store_module._atomic_replace
    crashed = False
    reason = "external_effect_failed_or_completion_unknown"

    def crash_after_pending_tip(path: Path, payload: bytes) -> None:
        nonlocal crashed
        real_atomic_replace(path, payload)
        parsed = json.loads(payload)
        if (
            not crashed
            and path == store.current_tip_anchor_path
            and parsed.get("schema_version") == CURRENT_TIP_PENDING_SCHEMA_VERSION
        ):
            crashed = True
            raise RuntimeError("simulated development market abort pending-tip crash")

    def operation() -> dict:
        return store.abort_owned_development_market_execution(
            development_root_scope_sha256=prepared["scope_hash"],
            reason=reason,
        )

    with patch.object(
        reveal_store_module,
        "_atomic_replace",
        crash_after_pending_tip,
    ), pytest.raises(RuntimeError, match="market abort pending-tip crash"):
        operation()

    pending = json.loads(store.current_tip_anchor_path.read_bytes())
    assert pending["schema_version"] == CURRENT_TIP_PENDING_SCHEMA_VERSION
    expected_abort = pending["next_tip_anchor"][
        "development_market_execution_aborts"
    ][prepared["scope_hash"]]

    recovered = operation()
    assert recovered == expected_abort
    assert store.state_path.read_bytes() == state_bytes
    recovered_tip = store.load_current_tip_anchor()
    assert recovered_tip["revision"] == tip_before["revision"] + 1
    assert recovered_tip["development_market_execution_aborts"] == {
        prepared["scope_hash"]: expected_abort
    }
    stable_tip = store.current_tip_anchor_path.read_bytes()
    assert operation() == recovered
    assert store.current_tip_anchor_path.read_bytes() == stable_tip
