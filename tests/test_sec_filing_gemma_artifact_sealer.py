from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any

import pytest

import agent_benchmark.sec_filing_gemma_artifact_sealer as sealer_module
from agent_benchmark.sec_filing_gemma_artifact_sealer import (
    STATE_FILENAME,
    SecFilingGemmaArtifactSealerError,
    SecFilingGemmaPredictionArtifactSealer,
    build_prediction_artifact_bytes,
    validate_prediction_artifact_seal_receipt,
)
from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    PREDICTION_PREFIX_SCHEMA_VERSION,
    PREDICTION_ROW_SCHEMA_VERSION,
    _PREDICTION_ROW_KEYS,
)


CANDIDATE_SHA256 = "a" * 64
OTHER_CANDIDATE_SHA256 = "b" * 64
STAGE = "development"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _parse(payload: bytes) -> dict[str, Any]:
    parsed = json.loads(payload)
    assert isinstance(parsed, dict)
    return parsed


def _rehash_state(state: dict[str, Any]) -> bytes:
    body = {key: state[key] for key in state if key != "state_sha256"}
    state["state_sha256"] = _digest(body)
    return _canonical_bytes(state)


def _prediction_prefix(
    prior: dict[str, Any] | None = None,
    *,
    candidate_sha256: str = CANDIDATE_SHA256,
    stage: str = STAGE,
    nonce: str = "base",
    forbidden_field: tuple[str, Any] | None = None,
    parent_prefix_override: str | None | object = ...,
) -> dict[str, Any]:
    if prior is None:
        rows: list[dict[str, Any]] = []
        sequence = 1
        genesis = _digest({"test-genesis": candidate_sha256, "stage": stage})
        parent_prefix: str | None = None
        parent_tip = genesis
    else:
        rows = copy.deepcopy(prior["rows"])
        sequence = len(rows) + 1
        genesis = prior["genesis_sha256"]
        parent_prefix = prior["prediction_prefix_sha256"]
        parent_tip = prior["tip_sha256"]
    if parent_prefix_override is not ...:
        parent_prefix = parent_prefix_override  # type: ignore[assignment]
    row_body: dict[str, Any] = {
        key: None
        for key in _PREDICTION_ROW_KEYS
        if key != "prediction_row_sha256"
    }
    row_body.update({
        "schema_version": PREDICTION_ROW_SCHEMA_VERSION,
        "sequence_number": sequence,
        "prior_prediction_prefix_sha256": parent_prefix,
        "parent_prediction_sha256": parent_tip,
        "candidate_sha256": candidate_sha256,
        "stage": stage,
        "decision_session": f"200{sequence}-01-02",
        "fold_context": {},
        "fold_context_sha256": _digest(
            {"sequence": sequence, "nonce": nonce}
        ),
    })
    if forbidden_field is not None:
        row_body[forbidden_field[0]] = forbidden_field[1]
    row = {**row_body, "prediction_row_sha256": _digest(row_body)}
    rows.append(row)
    prefix_body = {
        "schema_version": PREDICTION_PREFIX_SCHEMA_VERSION,
        "contract_sha256": "c" * 64,
        "candidate_sha256": candidate_sha256,
        "corpus_universe_sha256": "d" * 64,
        "calendar_sessions_sha256": "e" * 64,
        "initial_event_sequence_sha256": "f" * 64,
        "event_sequence_sha256": _digest(
            [item["prediction_row_sha256"] for item in rows]
        ),
        "row_count": len(rows),
        "genesis_sha256": genesis,
        "parent_prefix_sha256": parent_prefix,
        "parent_tip_sha256": parent_tip,
        "appended_row_sha256": row["prediction_row_sha256"],
        "tip_sha256": row["prediction_row_sha256"],
        "rows_sha256": _digest(rows),
        "rows": rows,
    }
    return {
        **prefix_body,
        "prediction_prefix_sha256": _digest(prefix_body),
    }


def _new_store(tmp_path: Path) -> SecFilingGemmaPredictionArtifactSealer:
    return SecFilingGemmaPredictionArtifactSealer(
        store_directory=tmp_path / "artifact-seals"
    )


def test_exact_bytes_receipt_is_purely_verifiable_and_reloadable(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    initial_pin = store.initialize(
        candidate_sha256=CANDIDATE_SHA256, stage=STAGE
    )
    artifact = build_prediction_artifact_bytes(_prediction_prefix())

    result = store.compare_and_swap_seal(
        artifact_bytes=artifact,
        external_prior_pin_bytes=initial_pin,
    )
    independent = validate_prediction_artifact_seal_receipt(
        artifact_bytes=artifact,
        receipt_bytes=result.receipt_bytes,
        external_prior_pin_bytes=initial_pin,
    )

    assert independent == result.validation
    assert result.artifact_bytes == artifact
    assert independent.artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert independent.prediction_sequence_number == 1
    assert independent.candidate_sha256 == CANDIDATE_SHA256
    assert independent.stage == STAGE
    assert independent.next_external_pin_bytes == result.next_external_pin_bytes
    assert store.load(external_pin_bytes=result.next_external_pin_bytes) == (
        result.next_external_pin_bytes
    )

    receipt = _parse(result.receipt_bytes)
    assert receipt["artifact_size_bytes"] == len(artifact)
    assert receipt["artifact_sha256"] == hashlib.sha256(artifact).hexdigest()
    assert not any("outcome" in key or key == "labels" for key in receipt)


def test_noncanonical_or_arbitrary_checksum_bytes_are_never_trusted(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    pin = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    artifact = build_prediction_artifact_bytes(_prediction_prefix())

    with pytest.raises(SecFilingGemmaArtifactSealerError, match="canonical"):
        store.compare_and_swap_seal(
            artifact_bytes=artifact + b"\n", external_prior_pin_bytes=pin
        )
    arbitrary_map = _canonical_bytes(
        {"prediction_prefix_sha256": "1" * 64, "sha256": "2" * 64}
    )
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="keys"):
        store.compare_and_swap_seal(
            artifact_bytes=arbitrary_map, external_prior_pin_bytes=pin
        )
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="canonical"):
        store.load(external_pin_bytes=pin + b" ")
    tagged = copy.deepcopy(_prediction_prefix())
    tagged["prediction_prefix_sha256"] = "sha256:" + "1" * 64
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="lowercase SHA"):
        build_prediction_artifact_bytes(tagged)


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "future_return",
        "futureReturn",
        "realizedReturn",
        "cashBeatsLong10bps",
        "cash_beats_long_10bps",
        "outcome_ledger_row_sha256",
        "label_binding_sha256",
        "return",
        "result",
        "score",
        "labels",
    ],
)
def test_even_rehashed_label_or_outcome_fields_are_rejected(
    forbidden_key: str,
) -> None:
    prefix = _prediction_prefix(
        forbidden_field=(
            forbidden_key,
            "1" * 64 if forbidden_key.endswith("sha256") else 1.0,
        )
    )
    with pytest.raises(
        SecFilingGemmaArtifactSealerError,
        match="forbidden label/outcome|Invalid prediction row",
    ):
        build_prediction_artifact_bytes(prefix)


def test_duplicate_skip_reorder_and_stale_compare_and_swap_are_rejected(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    prefix1 = _prediction_prefix()
    artifact1 = build_prediction_artifact_bytes(prefix1)
    seal1 = store.compare_and_swap_seal(
        artifact_bytes=artifact1, external_prior_pin_bytes=initial
    )

    with pytest.raises(SecFilingGemmaArtifactSealerError, match="compare-and-swap"):
        store.compare_and_swap_seal(
            artifact_bytes=artifact1, external_prior_pin_bytes=initial
        )
    with pytest.raises(
        SecFilingGemmaArtifactSealerError, match="duplicated, reordered, skipped"
    ):
        store.compare_and_swap_seal(
            artifact_bytes=artifact1,
            external_prior_pin_bytes=seal1.next_external_pin_bytes,
        )

    prefix2 = _prediction_prefix(prefix1, nonce="two")
    prefix3 = _prediction_prefix(prefix2, nonce="three")
    with pytest.raises(
        SecFilingGemmaArtifactSealerError, match="duplicated, reordered, skipped"
    ):
        store.compare_and_swap_seal(
            artifact_bytes=build_prediction_artifact_bytes(prefix3),
            external_prior_pin_bytes=seal1.next_external_pin_bytes,
        )

    wrong_parent = _prediction_prefix(
        prefix1, nonce="wrong-parent", parent_prefix_override="9" * 64
    )
    with pytest.raises(
        SecFilingGemmaArtifactSealerError, match="exact child"
    ):
        store.compare_and_swap_seal(
            artifact_bytes=build_prediction_artifact_bytes(wrong_parent),
            external_prior_pin_bytes=seal1.next_external_pin_bytes,
        )

    forged_history = copy.deepcopy(prefix2)
    forged_first_body = {
        key: value
        for key, value in forged_history["rows"][0].items()
        if key != "prediction_row_sha256"
    }
    forged_first_body["fold_context_sha256"] = "7" * 64
    forged_history["rows"][0] = {
        **forged_first_body,
        "prediction_row_sha256": _digest(forged_first_body),
    }
    forged_history["rows_sha256"] = _digest(forged_history["rows"])
    forged_history["event_sequence_sha256"] = _digest(
        [row["prediction_row_sha256"] for row in forged_history["rows"]]
    )
    forged_prefix_body = {
        key: value
        for key, value in forged_history.items()
        if key != "prediction_prefix_sha256"
    }
    forged_history["prediction_prefix_sha256"] = _digest(forged_prefix_body)
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="exact child"):
        store.compare_and_swap_seal(
            artifact_bytes=build_prediction_artifact_bytes(forged_history),
            external_prior_pin_bytes=seal1.next_external_pin_bytes,
        )


def test_external_pin_detects_store_rollback(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    prefix1 = _prediction_prefix()
    seal1 = store.compare_and_swap_seal(
        artifact_bytes=build_prediction_artifact_bytes(prefix1),
        external_prior_pin_bytes=initial,
    )
    state_after_one = store.state_path.read_bytes()
    prefix2 = _prediction_prefix(prefix1, nonce="two")
    seal2 = store.compare_and_swap_seal(
        artifact_bytes=build_prediction_artifact_bytes(prefix2),
        external_prior_pin_bytes=seal1.next_external_pin_bytes,
    )

    with pytest.raises(SecFilingGemmaArtifactSealerError, match="rolled back"):
        store.load(external_pin_bytes=seal1.next_external_pin_bytes)

    store.state_path.write_bytes(state_after_one)
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="rolled back"):
        store.load(external_pin_bytes=seal2.next_external_pin_bytes)


def test_rehashed_artifact_corruption_and_receipt_reorder_fail_reload(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    prefix1 = _prediction_prefix()
    seal1 = store.compare_and_swap_seal(
        artifact_bytes=build_prediction_artifact_bytes(prefix1),
        external_prior_pin_bytes=initial,
    )
    prefix2 = _prediction_prefix(prefix1, nonce="two")
    seal2 = store.compare_and_swap_seal(
        artifact_bytes=build_prediction_artifact_bytes(prefix2),
        external_prior_pin_bytes=seal1.next_external_pin_bytes,
    )
    pristine = store.state_path.read_bytes()

    state = _parse(pristine)
    first = base64.b64decode(state["entries"][0]["artifact_bytes_base64"])
    state["entries"][0]["artifact_bytes_base64"] = base64.b64encode(
        first + b" "
    ).decode("ascii")
    store.state_path.write_bytes(_rehash_state(state))
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="canonical"):
        store.load(external_pin_bytes=seal2.next_external_pin_bytes)

    store.state_path.write_bytes(pristine)
    state = _parse(pristine)
    state["entries"].reverse()
    store.state_path.write_bytes(_rehash_state(state))
    with pytest.raises(SecFilingGemmaArtifactSealerError):
        store.load(external_pin_bytes=seal2.next_external_pin_bytes)


def test_receipt_tamper_fails_even_when_state_hash_is_recomputed(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    seal = store.compare_and_swap_seal(
        artifact_bytes=build_prediction_artifact_bytes(_prediction_prefix()),
        external_prior_pin_bytes=initial,
    )
    state = _parse(store.state_path.read_bytes())
    state["entries"][0]["receipt"]["artifact_sha256"] = "0" * 64
    store.state_path.write_bytes(_rehash_state(state))

    with pytest.raises(SecFilingGemmaArtifactSealerError, match="receipt"):
        store.load(external_pin_bytes=seal.next_external_pin_bytes)


def test_atomic_replace_failure_preserves_prior_state_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    artifact = build_prediction_artifact_bytes(_prediction_prefix())
    original_replace = sealer_module.os.replace

    def fail_state_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        if Path(target) == store.state_path:
            raise OSError("simulated crash before atomic commit")
        original_replace(source, target)

    monkeypatch.setattr(sealer_module.os, "replace", fail_state_replace)
    with pytest.raises(OSError, match="simulated crash"):
        store.compare_and_swap_seal(
            artifact_bytes=artifact, external_prior_pin_bytes=initial
        )
    monkeypatch.setattr(sealer_module.os, "replace", original_replace)

    assert store.load(external_pin_bytes=initial) == initial
    assert not list(store.store_directory.glob(f".{STATE_FILENAME}.*.tmp"))


def test_interrupted_regular_temp_is_cleaned(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    regular_temp = store.store_directory / f".{STATE_FILENAME}.{'1' * 32}.tmp"
    regular_temp.write_bytes(b"partial")
    assert store.load(external_pin_bytes=initial) == initial
    assert not regular_temp.exists()


def test_interrupted_link_temp_fails_closed(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    link_temp = store.store_directory / f".{STATE_FILENAME}.{'2' * 32}.tmp"
    target = tmp_path / "outside-temp"
    target.write_bytes(b"outside")
    try:
        link_temp.symlink_to(target)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="single-link"):
        store.load(external_pin_bytes=initial)


def test_concurrent_same_pin_allows_exactly_one_cas(tmp_path: Path) -> None:
    directory = tmp_path / "artifact-seals"
    first = SecFilingGemmaPredictionArtifactSealer(store_directory=directory)
    second = SecFilingGemmaPredictionArtifactSealer(store_directory=directory)
    initial = first.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    artifact = build_prediction_artifact_bytes(_prediction_prefix())
    barrier = threading.Barrier(2)

    def attempt(store: SecFilingGemmaPredictionArtifactSealer) -> str:
        barrier.wait(timeout=5)
        try:
            store.compare_and_swap_seal(
                artifact_bytes=artifact, external_prior_pin_bytes=initial
            )
        except SecFilingGemmaArtifactSealerError:
            return "rejected"
        return "sealed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, (first, second)))
    assert sorted(outcomes) == ["rejected", "sealed"]

    state = _parse(first.state_path.read_bytes())
    assert len(state["entries"]) == 1


def test_path_configuration_is_immutable(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    with pytest.raises(AttributeError, match="immutable"):
        store._store_directory = tmp_path / "redirected"  # type: ignore[misc]


def test_symlink_store_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "real-store"
    target.mkdir()
    redirect = tmp_path / "store-link"
    try:
        redirect.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create directory symlinks")
    redirected_store = SecFilingGemmaPredictionArtifactSealer(
        store_directory=redirect
    )
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="reparse point|link"):
        redirected_store.initialize(
            candidate_sha256=CANDIDATE_SHA256, stage=STAGE
        )


def test_state_file_hardlink_is_rejected(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    hardlink = tmp_path / "hardlinked-state.json"
    os.link(store.state_path, hardlink)
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="single-link"):
        store.load(external_pin_bytes=initial)


def test_state_file_symlink_is_rejected(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    state_bytes = store.state_path.read_bytes()
    target = tmp_path / "redirected-state.json"
    target.write_bytes(state_bytes)
    store.state_path.unlink()
    try:
        store.state_path.symlink_to(target)
    except OSError:
        pytest.skip("This Windows account cannot create file symlinks")
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="single-link"):
        store.load(external_pin_bytes=initial)


def test_advanced_store_cannot_mint_a_fresh_trusted_pin(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    seal = store.compare_and_swap_seal(
        artifact_bytes=build_prediction_artifact_bytes(_prediction_prefix()),
        external_prior_pin_bytes=initial,
    )
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="use load"):
        store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    assert store.load(external_pin_bytes=seal.next_external_pin_bytes) == (
        seal.next_external_pin_bytes
    )


def test_candidate_and_stage_are_bound_at_genesis(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="external pin"):
        store.initialize(candidate_sha256=OTHER_CANDIDATE_SHA256, stage=STAGE)


def test_genesis_rollback_cannot_be_retrusted_through_initialize(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    initial = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    genesis_state = store.state_path.read_bytes()
    seal = store.compare_and_swap_seal(
        artifact_bytes=build_prediction_artifact_bytes(_prediction_prefix()),
        external_prior_pin_bytes=initial,
    )
    store.state_path.write_bytes(genesis_state)

    with pytest.raises(SecFilingGemmaArtifactSealerError, match="external pin"):
        store.initialize(candidate_sha256=CANDIDATE_SHA256, stage=STAGE)
    with pytest.raises(SecFilingGemmaArtifactSealerError, match="rolled back"):
        store.load(external_pin_bytes=seal.next_external_pin_bytes)


def test_valid_cumulative_stage_transitions_advance_external_pin(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    pin = store.initialize(candidate_sha256=CANDIDATE_SHA256, stage="development")
    prefix: dict[str, Any] | None = None
    for expected_sequence, stage in enumerate(
        ("development", "intermediate", "final"), start=1
    ):
        prefix = _prediction_prefix(prefix, stage=stage, nonce=stage)
        result = store.compare_and_swap_seal(
            artifact_bytes=build_prediction_artifact_bytes(prefix),
            external_prior_pin_bytes=pin,
        )
        assert result.validation.stage == stage
        assert result.validation.prediction_sequence_number == expected_sequence
        assert _parse(result.next_external_pin_bytes)["stage"] == stage
        pin = result.next_external_pin_bytes
    assert store.load(external_pin_bytes=pin) == pin


@pytest.mark.parametrize(
    "stages",
    [
        ("intermediate",),
        ("development", "final"),
        ("development", "intermediate", "development"),
        ("development", "final", "intermediate"),
    ],
)
def test_stage_start_skip_regression_or_mixed_reorder_is_rejected(
    stages: tuple[str, ...],
) -> None:
    prefix: dict[str, Any] | None = None
    for index, stage in enumerate(stages, start=1):
        prefix = _prediction_prefix(prefix, stage=stage, nonce=f"stage-{index}")
    assert prefix is not None
    with pytest.raises(
        SecFilingGemmaArtifactSealerError,
        match="begin in development|regress|skip|reordered",
    ):
        build_prediction_artifact_bytes(prefix)


def test_nested_camel_case_outcome_field_is_rejected_after_rehash() -> None:
    prefix = _prediction_prefix()
    row = prefix["rows"][-1]
    row["fold_context"] = {"futureReturn": 0.25}
    row_body = {
        key: value for key, value in row.items() if key != "prediction_row_sha256"
    }
    row["prediction_row_sha256"] = _digest(row_body)
    prefix["rows_sha256"] = _digest(prefix["rows"])
    prefix["appended_row_sha256"] = row["prediction_row_sha256"]
    prefix["tip_sha256"] = row["prediction_row_sha256"]
    prefix_body = {
        key: value
        for key, value in prefix.items()
        if key != "prediction_prefix_sha256"
    }
    prefix["prediction_prefix_sha256"] = _digest(prefix_body)
    with pytest.raises(
        SecFilingGemmaArtifactSealerError, match="forbidden label/outcome"
    ):
        build_prediction_artifact_bytes(prefix)
