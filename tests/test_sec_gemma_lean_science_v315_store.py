from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType

import pytest

from agent_benchmark import sec_gemma_lean_science_v315_store as store_module
from agent_benchmark.sec_gemma_lean_science_v315_journal import (
    HashChainJournal,
    V315JournalError,
    canonical_json_bytes,
    canonical_sha256,
    replay_journal,
    sha256_bytes,
)
from agent_benchmark.sec_gemma_lean_science_v315_store import (
    ATTEMPT_ID,
    CHECKPOINT_SCHEMA_VERSION,
    PAUSE_THRESHOLD_NS,
    PAUSE_CHECKPOINT_STATE_SCHEMA_VERSION,
    PILOT_COUNT,
    RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION,
    RESPONSE_PAYLOAD_SCHEMA_VERSION,
    AttemptStore,
    V315StoreConflict,
    V315StoreError,
    V315StorePoisoned,
)


def _authority() -> dict[str, object]:
    return {
        "plan": {"sha256": "1" * 64},
        "attempt": {"attempt_id": ATTEMPT_ID},
        "implementation": {"commit": "2" * 40, "tree": "3" * 40},
        "preflight": {"sha256": "4" * 64},
        "source": {"sha256": "5" * 64},
        "science": {"sha256": "6" * 64},
        "effect_budget": {"sha256": "7" * 64},
        "request_order": {"sha256": "8" * 64},
        "pilot_order": {"sha256": "9" * 64},
    }


def _path(tmp_path: Path, name: str = "attempt") -> Path:
    return (tmp_path / name).resolve()


def _hash(label: str) -> str:
    return sha256_bytes(label.encode("ascii"))


def _append_binding(
    store: AttemptStore,
    checkpoint_name: str,
    *,
    invocation_kind: str = "development",
):
    return store.append_runtime_binding_receipt(
        checkpoint_name=checkpoint_name,
        invocation_kind=invocation_kind,
        head="a" * 40,
        tree="b" * 40,
        clean_state_sha256=_hash("clean-state"),
        repository_runtime_manifest_sha256=_hash("repository-runtime"),
        execution_dependency_manifest_sha256=_hash("execution-dependencies"),
        loaded_code_manifest_sha256=_hash("loaded-code"),
        runtime_binding_authority_sha256=_hash("runtime-authority"),
        route_argv_sha256=_hash("route-argv"),
        process_environment_sha256=_hash("environment"),
        interpreter_identity_sha256=_hash("interpreter"),
    )


def _append_recovery_binding(store: AttemptStore):
    return store.append_recovery_runtime_binding_receipt(
        head="a" * 40,
        tree="b" * 40,
        clean_state_sha256=_hash("recovery-clean-state"),
        repository_runtime_manifest_sha256=_hash("repository-runtime"),
        execution_dependency_manifest_sha256=_hash("execution-dependencies"),
        loaded_code_manifest_sha256=_hash("loaded-code"),
        runtime_binding_authority_sha256=_hash("runtime-authority"),
        route_argv_sha256=_hash("recovery-route-argv"),
        process_environment_sha256=_hash("environment"),
        interpreter_identity_sha256=_hash("interpreter"),
    )


def _build_candidate(
    store: AttemptStore,
    *,
    status: str,
    terminal_code: str,
    evidence: object,
    public_result: bytes | None = None,
    comparison_before: bytes = b"# comparison\n",
    comparison_after: bytes = b"# comparison\nupdated\n",
):
    return store.build_terminal_candidate(
        proposed_status=status,
        proposed_terminal_code=terminal_code,
        evidence=evidence,  # type: ignore[arg-type]
        public_result=(
            public_result
            if public_result is not None
            else canonical_json_bytes(
                {"status": status, "terminal_code": terminal_code}
            )
        ),
        comparison_before=comparison_before,
        comparison_after=comparison_after,
    )


def _begin_candidate_recovery(
    store: AttemptStore,
    *,
    target_kind: str,
    candidate,
    public_artifact_sha256: str | None = None,
    public_artifact_bytes: int | None = None,
    comparison_before_sha256: str | None = None,
    comparison_after_sha256: str | None = None,
    attempt_terminal_event_sha256: str | None = None,
    observed_state_sha256: str | None = None,
):
    return store.begin_recovery(
        target_kind=target_kind,
        candidate_sha256=candidate.candidate_sha256,
        attempt_terminal_event_sha256=attempt_terminal_event_sha256,
        public_artifact_sha256=(
            candidate.public_result_sha256
            if public_artifact_sha256 is None
            else public_artifact_sha256
        ),
        public_artifact_bytes=(
            candidate.public_result_bytes
            if public_artifact_bytes is None
            else public_artifact_bytes
        ),
        comparison_before_sha256=(
            candidate.comparison_before_sha256
            if comparison_before_sha256 is None
            else comparison_before_sha256
        ),
        comparison_after_sha256=(
            candidate.comparison_after_sha256
            if comparison_after_sha256 is None
            else comparison_after_sha256
        ),
        observed_state_sha256=(
            _hash("observed-state")
            if observed_state_sha256 is None
            else observed_state_sha256
        ),
    )


def _seal_zero_effect_rejection(
    store: AttemptStore,
    *,
    evidence: object | None = None,
):
    store.record_runtime_binding_failure(checkpoint_name="attempt_open")
    candidate = _build_candidate(
        store,
        status="rejected",
        terminal_code="runtime_binding_mismatch",
        evidence=(
            {"result": "rejected"} if evidence is None else evidence
        ),
    )
    _begin_candidate_recovery(
        store,
        target_kind="zero_effect_binding_rejection",
        candidate=candidate,
    )
    binding = _append_recovery_binding(store)
    candidate = store.persist_terminal_candidate(candidate)
    terminal = store.append_attempt_terminal(
        publication_binding_receipt_sha256=(
            binding.runtime_binding_receipt_sha256
        )
    )
    store.finish_recovery(
        outcome="published", final_state_sha256=_hash("published-state")
    )
    return terminal


def _seal(
    store: AttemptStore,
    *,
    request_id: str,
    effect_kind: str,
    model_phase: str = "none",
    segment_id: str | None = None,
    duration_ns: int | None = None,
) -> None:
    intent = store.begin_request(
        request_id=request_id,
        effect_kind=effect_kind,
        request_sha256=_hash("request:" + request_id),
        model_phase=model_phase,
        segment_id=segment_id,
    )
    response = store.commit_response(
        request_intent_event_sha256=intent.event_sha256,
        body=("body:" + request_id).encode("ascii"),
        metadata={"status": 200, "request_id_sha256": _hash(request_id)},
        duration_ns=duration_ns,
    )
    store.commit_checkpoint(
        subject_event_sha256=response.event_sha256,
        state=store.derive_pending_checkpoint_state(),
    )


def _six_yahoo(store: AttemptStore) -> None:
    for index in range(6):
        _seal(store, request_id=f"yahoo-{index}", effect_kind="yahoo")


def _pre_probe(store: AttemptStore, segment: str) -> None:
    _seal(
        store,
        request_id=f"{segment}-pre-version",
        effect_kind="identity",
        model_phase="pre_probe_start",
        segment_id=segment,
    )
    _seal(
        store,
        request_id=f"{segment}-pre-show",
        effect_kind="identity",
        model_phase="pre_probe",
        segment_id=segment,
    )


def _post_probe(store: AttemptStore, segment: str) -> None:
    _seal(
        store,
        request_id=f"{segment}-post-version",
        effect_kind="identity",
        model_phase="post_probe",
        segment_id=segment,
    )
    _seal(
        store,
        request_id=f"{segment}-post-show",
        effect_kind="identity",
        model_phase="post_probe_close",
        segment_id=segment,
    )


def test_create_is_one_shot_locked_and_binds_full_authority(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path)
    authority = _authority()
    with AttemptStore.create(path, authority=authority) as store:
        snapshot = store.snapshot
        assert snapshot.status == "active"
        assert snapshot.event_count == 1
        assert snapshot.yahoo_intent_count == 0
        assert snapshot.identity_intent_count == 0
        assert snapshot.gemma_intent_count == 0
        assert snapshot.continuation_authorized is False
        assert snapshot.continuation_sha256 is None
        assert snapshot.continuation_permission_sha256 is None
        assert snapshot.continuation_commit is None
        with pytest.raises(V315StoreConflict) as caught:
            AttemptStore.open(path, authority=authority)
        assert caught.value.code == "attempt_locked"

    with pytest.raises(V315StoreConflict) as caught:
        AttemptStore.create(path, authority=authority)
    assert caught.value.code == "attempt_already_exists"

    wrong = _authority()
    wrong["science"] = {"sha256": "0" * 64}
    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(path, authority=wrong)
    assert caught.value.code == "attempt_lock_invalid"


def test_fresh_development_attempt_open_chains_and_ambiguous_intent_cannot_reopen(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path, "fresh-process-opens")
    with AttemptStore.create(path, authority=_authority()) as store:
        first = _append_binding(store, "attempt_open")
        _append_binding(store, "yahoo_01_pre")
        second = _append_binding(store, "attempt_open")
        assert second.previous_runtime_binding_receipt_sha256 != (
            first.previous_runtime_binding_receipt_sha256
        )
        assert second.previous_runtime_binding_receipt_sha256 == (
            store.runtime_binding_receipts()[-2].runtime_binding_receipt_sha256
        )
        _append_binding(store, "yahoo_01_pre")
        store.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
        with pytest.raises(V315StoreError) as caught:
            _append_binding(store, "attempt_open")
        assert caught.value.code == "runtime_binding_receipt_invalid"

def test_interrupted_initialization_never_exposes_partial_attempt_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _path(tmp_path)

    def interrupt_before_intent(*_args, **_kwargs):
        raise V315JournalError("synthetic_initialization_interruption")

    monkeypatch.setattr(HashChainJournal, "append", interrupt_before_intent)
    with pytest.raises(V315JournalError) as caught:
        AttemptStore.create(path, authority=_authority())
    assert caught.value.code == "synthetic_initialization_interruption"
    assert not path.exists()


@pytest.mark.parametrize("foreign_kind", ["empty_directory", "symlink"])
def test_initialization_publication_race_never_replaces_foreign_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_kind: str,
) -> None:
    path = _path(tmp_path, f"attempt-{foreign_kind}")
    foreign_target = _path(tmp_path, f"foreign-{foreign_kind}")
    foreign_target.mkdir()
    (foreign_target / "marker.txt").write_bytes(b"foreign")
    original = store_module._publish_attempt_root_noreplace
    inserted: dict[str, object] = {}

    def insert_foreign_then_publish(source: Path, destination: Path) -> None:
        assert destination == path
        if foreign_kind == "symlink":
            try:
                destination.symlink_to(foreign_target, target_is_directory=True)
                inserted["symlink"] = True
            except OSError:
                # Directory symlinks may require elevated Windows privileges.
                # An empty directory exercises the same no-replace primitive.
                destination.mkdir()
                inserted["symlink"] = False
        else:
            destination.mkdir()
            inserted["symlink"] = False
        details = destination.lstat()
        inserted["device"] = details.st_dev
        inserted["inode"] = details.st_ino
        original(source, destination)

    monkeypatch.setattr(
        store_module,
        "_publish_attempt_root_noreplace",
        insert_foreign_then_publish,
    )
    with pytest.raises(V315StoreConflict) as caught:
        AttemptStore.create(path, authority=_authority())
    assert caught.value.code == "attempt_already_exists"

    visible = path.lstat()
    assert visible.st_dev == inserted["device"]
    assert visible.st_ino == inserted["inode"]
    if inserted["symlink"] is True:
        assert path.is_symlink()
        assert path.resolve(strict=True) == foreign_target
    else:
        assert path.is_dir()
        assert tuple(path.iterdir()) == ()
    assert (foreign_target / "marker.txt").read_bytes() == b"foreign"


def test_exact_intent_response_payload_checkpoint_marker_transition(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path)
    authority = _authority()
    with AttemptStore.create(path, authority=authority) as store:
        intent = store.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
        assert store.snapshot.active_request_id == "yahoo-0"
        response = store.commit_response(
            request_intent_event_sha256=intent.event_sha256,
            body=b"response",
            metadata={"status": 200},
        )
        assert "response" not in repr(response)
        assert store.snapshot.pending_checkpoint_event_sha256 == response.event_sha256
        with pytest.raises(V315StoreError) as caught:
            store.begin_request(
                request_id="yahoo-1",
                effect_kind="yahoo",
                request_sha256=_hash("request:yahoo-1"),
            )
        assert caught.value.code == "request_not_authorized"
        expected_checkpoint = dict(store.derive_pending_checkpoint_state())
        assert expected_checkpoint == {
            "schema_version": RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION,
            "phase": "none",
            "request_sha256": _hash("request:yahoo-0"),
            "body_sha256": sha256_bytes(b"response"),
        }
        for invalid in (
            {**expected_checkpoint, "phase": "generation"},
            {**expected_checkpoint, "request_sha256": "0" * 64},
            {**expected_checkpoint, "body_sha256": "0" * 64},
            {**expected_checkpoint, "extra": True},
        ):
            with pytest.raises(V315StoreError) as caught:
                store.commit_checkpoint(
                    subject_event_sha256=response.event_sha256,
                    state=invalid,
                )
            assert caught.value.code == "checkpoint_state_invalid"
        store.commit_checkpoint(
            subject_event_sha256=response.event_sha256,
            state=expected_checkpoint,
        )
        assert store.snapshot.yahoo_response_count == 1
        assert store.snapshot.pending_checkpoint_event_sha256 is None
        with pytest.raises(V315StoreError) as caught:
            store.committed_responses(effect_kind="yahoo")
        assert caught.value.code == "response_batch_unsealed"

        replay = replay_journal(
            path / "journal", authority_sha256=store.authority_sha256
        )
        assert [event.event_type for event in replay.events] == [
            "attempt_intent",
            "request_intent",
            "response_committed",
            "checkpoint_committed",
        ]
        response_files = list((path / "payloads").glob("*.json"))
        checkpoint_files = list((path / "checkpoints").glob("*.json"))
        assert len(response_files) == len(checkpoint_files) == 1
        assert response_files[0].stem == sha256_bytes(response_files[0].read_bytes())
        assert checkpoint_files[0].stem == sha256_bytes(
            checkpoint_files[0].read_bytes()
        )


def test_committed_yahoo_responses_release_only_as_complete_immutable_batch(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path)
    with AttemptStore.create(path, authority=_authority()) as store:
        _six_yahoo(store)
        with pytest.raises(V315StoreError) as caught:
            store.committed_responses(effect_kind="yahoo")
        assert caught.value.code == "response_values_not_opened"
        receipts = store.committed_response_receipts(effect_kind="yahoo")
        assert [item.intent.request_id for item in receipts] == [
            f"yahoo-{index}" for index in range(6)
        ]
        assert store.snapshot.yahoo_body_bytes == sum(
            len(f"body:yahoo-{index}".encode("ascii")) for index in range(6)
        )
        assert all(item.checkpoint_event_sha256 for item in receipts)
        assert all(item.checkpoint_sha256 for item in receipts)
        assert all(not hasattr(item, "body") for item in receipts)
        with pytest.raises(V315StoreError) as caught:
            store.committed_responses(
                effect_kind="yahoo", segment_id="segment-1"
            )
        assert caught.value.code == "response_filter_invalid"


def test_crash_before_intent_is_unconsumed_but_open_intent_is_indeterminate(
    tmp_path: Path,
) -> None:
    authority = _authority()
    safe_path = _path(tmp_path, "before-intent")
    AttemptStore.create(safe_path, authority=authority).close()
    with AttemptStore.open(safe_path, authority=authority) as store:
        _seal(store, request_id="yahoo-0", effect_kind="yahoo")
        assert store.snapshot.yahoo_response_count == 1

    path = _path(tmp_path, "after-intent")
    with AttemptStore.create(path, authority=authority) as store:
        _append_binding(store, "attempt_open")
        intent = store.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
    with AttemptStore.open(path, authority=authority) as recovered:
        assert recovered.snapshot.status == "indeterminate"
        assert recovered.snapshot.terminal_code == "request_outcome_unknown"
        assert recovered.snapshot.yahoo_intent_count == 1
        assert recovered.snapshot.yahoo_response_count == 0
        with pytest.raises(V315StoreError) as caught:
            recovered.commit_response(
                request_intent_event_sha256=intent.event_sha256,
                body=b"late",
                metadata={"status": 200},
            )
        assert caught.value.code == "response_commit_invalid"
        with pytest.raises(V315StoreError) as caught:
            recovered.begin_request(
                request_id="yahoo-0-again",
                effect_kind="yahoo",
                request_sha256=_hash("again"),
            )
        assert caught.value.code == "request_not_authorized"
        for status in ("rejected", "completed"):
            with pytest.raises(V315StoreError) as caught:
                _build_candidate(
                    recovered,
                    status=status,
                    terminal_code="wrong_open_intent_outcome",
                    evidence={"result": status},
                )
            assert caught.value.code == "terminal_candidate_invalid"
        candidate = _build_candidate(
            recovered,
            status="indeterminate",
            terminal_code="request_outcome_unknown",
            evidence={"result": "indeterminate"},
        )
        _begin_candidate_recovery(
            recovered,
            target_kind="terminal_indeterminate_created",
            candidate=candidate,
        )
        binding = _append_recovery_binding(recovered)
        recovered.persist_terminal_candidate(candidate)
        recovered.append_attempt_terminal(
            publication_binding_receipt_sha256=(
                binding.runtime_binding_receipt_sha256
            )
        )
        recovered.finish_recovery(
            outcome="published",
            final_state_sha256=_hash("open-intent-published"),
        )
        assert recovered.snapshot.status == "indeterminate"


def test_replay_rejects_nonindeterminate_terminal_after_open_intent(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path)
    authority = _authority()
    with AttemptStore.create(path, authority=authority) as attempt:
        attempt.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
        authority_sha256 = attempt.authority_sha256
    journal = HashChainJournal(
        path / "journal", authority_sha256=authority_sha256
    )
    prefix = journal.replay().head_event_sha256
    journal.append(
        "attempt_terminal",
        {
            "status": "rejected",
            "terminal_code": "wrong_open_intent_outcome",
            "evidence_sha256": "a" * 64,
            "evidence_bytes": 1,
            "journal_head_before_terminal_sha256": prefix,
        },
    )

    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(path, authority=authority)
    assert caught.value.code == "terminal_event_invalid"


def test_yahoo_response_without_marker_is_rebuilt_then_next_request_resumes(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path)
    authority = _authority()
    with AttemptStore.create(path, authority=authority) as store:
        intent = store.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
        response = store.commit_response(
            request_intent_event_sha256=intent.event_sha256,
            body=b"sealed-yahoo",
            metadata={"status": 200},
        )

    with AttemptStore.open(path, authority=authority) as recovered:
        assert recovered.snapshot.status == "checkpoint_recovery"
        seen: list[str] = []

        with pytest.raises(V315StoreError) as caught:
            recovered.commit_checkpoint(
                subject_event_sha256=response.event_sha256,
                state={"caller_supplied": "not-an-independent-rebuild"},
            )
        assert caught.value.code == "checkpoint_subject_invalid"

        def rebuild(context):
            assert context.response is not None
            assert not hasattr(context.response, "body")
            seen.append(context.response.body_sha256)
            return recovered.derive_pending_checkpoint_state()

        recovered.recover_pending_checkpoint(rebuild)
        assert seen == [sha256_bytes(b"sealed-yahoo")]
        assert recovered.snapshot.status == "active"
        assert recovered.snapshot.yahoo_response_count == 1
        assert recovered.snapshot.yahoo_body_bytes == len(b"sealed-yahoo")
        _seal(recovered, request_id="yahoo-1", effect_kind="yahoo")
        assert recovered.snapshot.yahoo_response_count == 2
        assert response.event_sha256 in {
            event.payload.get("subject_event_sha256")
            for event in replay_journal(
                path / "journal",
                authority_sha256=recovered.authority_sha256,
            ).events
            if event.event_type == "checkpoint_committed"
        }


def test_existing_marker_last_checkpoint_candidate_requires_exact_rebuild(
    tmp_path: Path,
) -> None:
    authority = _authority()
    for name, matching in (("match", True), ("mismatch", False)):
        path = _path(tmp_path, name)
        with AttemptStore.create(path, authority=authority) as store:
            intent = store.begin_request(
                request_id="yahoo-0",
                effect_kind="yahoo",
                request_sha256=_hash("request:yahoo-0"),
            )
            response = store.commit_response(
                request_intent_event_sha256=intent.event_sha256,
                body=b"sealed-yahoo",
                metadata={"status": 200},
            )
        state = {
            "schema_version": RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION,
            "phase": "none",
            "request_sha256": _hash("request:yahoo-0"),
            "body_sha256": sha256_bytes(b"sealed-yahoo"),
        }
        candidate_state = state if matching else {**state, "phase": "generation"}
        encoded = canonical_json_bytes(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "subject_event_sha256": response.event_sha256,
                "state": candidate_state,
            }
        )
        digest = sha256_bytes(encoded)
        (path / "checkpoints" / f"{digest}.json").write_bytes(encoded)

        if matching:
            with AttemptStore.open(path, authority=authority) as recovered:
                assert recovered.snapshot.status == "checkpoint_recovery"
                recovered.recover_pending_checkpoint(lambda _context: state)
                assert recovered.snapshot.status == "active"
        else:
            with pytest.raises(V315StorePoisoned) as caught:
                AttemptStore.open(path, authority=authority)
            assert caught.value.code == "orphan_checkpoint"


def test_hash_consistent_but_nonderived_checkpoint_poisons_replay(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path)
    authority = _authority()
    with AttemptStore.create(path, authority=authority) as attempt:
        _seal(attempt, request_id="yahoo-0", effect_kind="yahoo")
        authority_sha256 = attempt.authority_sha256

    replay = replay_journal(
        path / "journal", authority_sha256=authority_sha256
    )
    marker = replay.events[-1]
    assert marker.event_type == "checkpoint_committed"
    marker.path.unlink()
    original = (
        path
        / "checkpoints"
        / f"{marker.payload['checkpoint_sha256']}.json"
    )
    original.unlink()
    wrong_state = {
        "schema_version": RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION,
        "phase": "generation",
        "request_sha256": _hash("request:yahoo-0"),
        "body_sha256": sha256_bytes(b"body:yahoo-0"),
    }
    encoded = canonical_json_bytes(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "subject_event_sha256": marker.payload[
                "subject_event_sha256"
            ],
            "state": wrong_state,
        }
    )
    digest = sha256_bytes(encoded)
    (path / "checkpoints" / f"{digest}.json").write_bytes(encoded)
    HashChainJournal(
        path / "journal", authority_sha256=authority_sha256
    ).append(
        "checkpoint_committed",
        {
            **dict(marker.payload),
            "checkpoint_sha256": digest,
            "checkpoint_bytes": len(encoded),
        },
    )

    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(path, authority=authority)
    assert caught.value.code == "checkpoint_invalid"


def test_orphan_payload_missing_checkpoint_and_extra_root_entry_poison(
    tmp_path: Path,
) -> None:
    authority = _authority()
    orphan_path = _path(tmp_path, "orphan")
    with AttemptStore.create(orphan_path, authority=authority) as store:
        intent = store.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
    body = b"uncommitted"
    payload = canonical_json_bytes(
        {
            "schema_version": RESPONSE_PAYLOAD_SCHEMA_VERSION,
            "request_intent_event_sha256": intent.event_sha256,
            "request_id": "yahoo-0",
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_bytes": len(body),
            "body_sha256": sha256_bytes(body),
            "metadata": {"status": 200},
        }
    )
    (orphan_path / "payloads" / f"{sha256_bytes(payload)}.json").write_bytes(
        payload
    )
    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(orphan_path, authority=authority)
    assert caught.value.code == "orphan_response_payload"

    missing_path = _path(tmp_path, "missing-checkpoint")
    with AttemptStore.create(missing_path, authority=authority) as store:
        _seal(store, request_id="yahoo-0", effect_kind="yahoo")
    next((missing_path / "checkpoints").glob("*.json")).unlink()
    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(missing_path, authority=authority)
    assert caught.value.code == "checkpoint_missing"

    extra_path = _path(tmp_path, "extra")
    AttemptStore.create(extra_path, authority=authority).close()
    (extra_path / "foreign.txt").write_text("x", encoding="ascii")
    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(extra_path, authority=authority)
    assert caught.value.code == "attempt_root_poisoned"


def test_model_segment_interruption_is_never_resumable(tmp_path: Path) -> None:
    path = _path(tmp_path)
    authority = _authority()
    with AttemptStore.create(path, authority=authority) as store:
        _six_yahoo(store)
        _seal(
            store,
            request_id="segment-1-pre-version",
            effect_kind="identity",
            model_phase="pre_probe_start",
            segment_id="segment-1",
        )

    with AttemptStore.open(path, authority=authority) as recovered:
        snapshot = recovered.snapshot
        assert snapshot.status == "indeterminate"
        assert snapshot.terminal_code == "irreversible_state_ambiguous"
        assert snapshot.open_model_segment_id == "segment-1"
        with pytest.raises(V315StoreError) as caught:
            recovered.begin_request(
                request_id="segment-1-pre-show",
                effect_kind="identity",
                request_sha256=_hash("pre-show"),
                model_phase="pre_probe",
                segment_id="segment-1",
            )
        assert caught.value.code == "request_not_authorized"


def test_closed_model_segment_post_probe_checkpoint_can_be_rebuilt(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path)
    authority = _authority()
    with AttemptStore.create(path, authority=authority) as store:
        _append_binding(store, "attempt_open")
        _six_yahoo(store)
        _pre_probe(store, "segment-1")
        pre_identity = store.committed_responses(
            effect_kind="identity", segment_id="segment-1"
        )
        assert [item.intent.model_phase for item in pre_identity] == [
            "pre_probe_start",
            "pre_probe",
        ]
        for index in range(75):
            _seal(
                store,
                request_id=f"gemma-{index:02d}",
                effect_kind="gemma",
                model_phase="generation",
                segment_id="segment-1",
                duration_ns=1,
            )
        with pytest.raises(V315StoreError) as caught:
            store.committed_responses(
                effect_kind="gemma", segment_id="segment-1"
            )
        assert caught.value.code == "response_batch_unsealed"
        _seal(
            store,
            request_id="segment-1-post-version",
            effect_kind="identity",
            model_phase="post_probe",
            segment_id="segment-1",
        )
        intent = store.begin_request(
            request_id="segment-1-post-show",
            effect_kind="identity",
            request_sha256=_hash("segment-1-post-show"),
            model_phase="post_probe_close",
            segment_id="segment-1",
        )
        store.commit_response(
            request_intent_event_sha256=intent.event_sha256,
            body=b"post-show",
            metadata={"status": 200},
        )

    with AttemptStore.open(path, authority=authority) as recovered:
        assert recovered.snapshot.status == "checkpoint_recovery"
        assert recovered.snapshot.open_model_segment_id is None
        invalid = dict(recovered.derive_pending_checkpoint_state())
        invalid["phase"] = "none"
        with pytest.raises(V315StoreError) as caught:
            recovered.recover_pending_checkpoint(lambda _context: invalid)
        assert caught.value.code == "checkpoint_rebuild_failed"
        recovered.recover_pending_checkpoint(
            lambda _context: recovered.derive_pending_checkpoint_state()
        )
        assert recovered.snapshot.status == "indeterminate"
        assert recovered.snapshot.terminal_code == "model_post_binding_missing"
        _append_binding(recovered, "model_initial_post")
        snapshot = recovered.snapshot
        assert snapshot.status == "active"
        assert snapshot.gemma_response_count == 75
        assert snapshot.identity_response_count == 4
        assert snapshot.segment_summaries[0].generation_count == 75
        with pytest.raises(V315StoreError) as caught:
            recovered.committed_responses(
                effect_kind="gemma", segment_id="segment-1"
            )
        assert caught.value.code == "response_values_not_opened"
        gemma_receipts = recovered.committed_response_receipts(
            effect_kind="gemma", segment_id="segment-1"
        )
        assert len(gemma_receipts) == 75
        assert [item.intent.request_id for item in gemma_receipts] == [
            f"gemma-{index:02d}" for index in range(75)
        ]
        assert len(
            recovered.committed_responses(
                effect_kind="identity", segment_id="segment-1"
            )
        ) == 4
        with pytest.raises(V315StoreError) as caught:
            recovered.committed_responses()
        assert caught.value.code == "response_values_not_opened"
        assert recovered.snapshot.market_values_opened is False
        assert recovered.snapshot.model_responses_opened is False
        recovered.mark_market_values_opened()
        assert recovered.snapshot.market_values_opened is True
        assert recovered.snapshot.model_responses_opened is False
        yahoo = recovered.committed_responses(effect_kind="yahoo")
        assert len(yahoo) == 6

    with AttemptStore.open(path, authority=authority) as reopened:
        assert reopened.snapshot.market_values_opened is True
        assert reopened.snapshot.model_responses_opened is False
        reopened.mark_model_responses_opened()
        assert reopened.snapshot.model_responses_opened is True
        assert len(
            reopened.committed_responses(
                effect_kind="gemma", segment_id="segment-1"
            )
        ) == 75
        assert len(reopened.committed_responses()) == 85
        with pytest.raises(V315StoreError) as caught:
            reopened.mark_model_responses_opened()
        assert caught.value.code == "private_values_open_not_authorized"

    with AttemptStore.open(path, authority=authority) as terminal:
        assert terminal.snapshot.status == "active"
        assert terminal.snapshot.market_values_opened is True
        assert terminal.snapshot.model_responses_opened is True


def test_strict_pause_boundary_and_clean_pause_resume(tmp_path: Path) -> None:
    authority = _authority()

    exact_path = _path(tmp_path, "exact-threshold")
    with AttemptStore.create(exact_path, authority=authority) as exact:
        _six_yahoo(exact)
        _pre_probe(exact, "segment-1")
        exact_duration = PAUSE_THRESHOLD_NS // 75
        assert exact_duration * 75 == PAUSE_THRESHOLD_NS
        for index in range(PILOT_COUNT):
            _seal(
                exact,
                request_id=f"pilot-{index}",
                effect_kind="gemma",
                model_phase="generation",
                segment_id="segment-1",
                duration_ns=exact_duration,
            )
        sixth = exact.begin_request(
            request_id="remaining-0",
            effect_kind="gemma",
            request_sha256=_hash("remaining-0"),
            model_phase="generation",
            segment_id="segment-1",
        )
        assert sixth.request_id == "remaining-0"

    pause_path = _path(tmp_path, "strictly-over")
    with AttemptStore.create(pause_path, authority=authority) as paused:
        initial_open = _append_binding(paused, "attempt_open")
        assert initial_open.previous_runtime_binding_receipt_sha256 is None
        _six_yahoo(paused)
        _pre_probe(paused, "segment-1")
        for index in range(PILOT_COUNT):
            _seal(
                paused,
                request_id=f"pilot-{index}",
                effect_kind="gemma",
                model_phase="generation",
                segment_id="segment-1",
                duration_ns=exact_duration + 1,
            )
        with pytest.raises(V315StoreError) as caught:
            paused.begin_request(
                request_id="forbidden-sixth",
                effect_kind="gemma",
                request_sha256=_hash("forbidden-sixth"),
                model_phase="generation",
                segment_id="segment-1",
            )
        assert caught.value.code == "pilot_pause_required"
        _post_probe(paused, "segment-1")
        _append_binding(paused, "model_initial_post")
        pause_candidate = paused.write_pause_candidate(
            pause_artifact=canonical_json_bytes(
                {"status": "paused", "pilot_count": PILOT_COUNT}
            )
        )
        pause_binding = _append_binding(paused, "pause_publish_pre")
        with pytest.raises(V315StoreError) as caught:
            paused.commit_planned_pause(
                publication_binding_receipt_sha256=(
                    pause_binding.runtime_binding_receipt_sha256
                ),
                pause_candidate_sha256=(
                    pause_candidate.pause_candidate_sha256
                ),
                checkpoint_state={"caller_chosen_pause": True}
            )
        assert caught.value.code == "checkpoint_state_invalid"
        assert paused.snapshot.status == "checkpoint_recovery"
        with pytest.raises(V315StoreError) as caught:
            paused.authorize_continuation(
                continuation_sha256="a" * 64,
                permission_sha256="b" * 64,
                continuation_commit="c" * 40,
            )
        assert caught.value.code == "continuation_not_authorized"
        context = paused.pending_checkpoint_context()
        marker = paused.commit_checkpoint(
            subject_event_sha256=context.subject_event_sha256,
            state=paused.derive_pending_checkpoint_state(),
        )
        pause = next(
            event
            for event in replay_journal(
                pause_path / "journal",
                authority_sha256=paused.authority_sha256,
            ).events
            if event.event_type == "paused_for_justification"
        )
        assert pause.event_sha256
        assert marker.event_type == "checkpoint_committed"
        assert paused.snapshot.status == "paused"
        assert paused.snapshot.identity_response_count == 4
        assert paused.snapshot.yahoo_intent_count == 6
        assert paused.snapshot.identity_intent_count == 4
        assert paused.snapshot.gemma_intent_count == 5
        assert paused.snapshot.continuation_authorized is False
        assert paused.snapshot.continuation_sha256 is None
        assert paused.snapshot.continuation_permission_sha256 is None
        assert paused.snapshot.continuation_commit is None
        checkpoint_path = (
            pause_path
            / "checkpoints"
            / f"{marker.payload['checkpoint_sha256']}.json"
        )
        pause_state = json.loads(checkpoint_path.read_text(encoding="utf-8"))[
            "state"
        ]
        assert pause_state["schema_version"] == (
            PAUSE_CHECKPOINT_STATE_SCHEMA_VERSION
        )
        assert pause_state["pilot_count"] == PILOT_COUNT
        assert len(pause_state["pilot_response_event_sha256s"]) == PILOT_COUNT
        assert len(
            paused.committed_responses(
                effect_kind="identity", segment_id="segment-1"
            )
        ) == 4
        with pytest.raises(V315StoreError) as caught:
            paused.committed_responses(
                effect_kind="gemma", segment_id="segment-1"
            )
        assert caught.value.code == "response_batch_unsealed"
        pilot_receipts = paused.committed_response_receipts(
            effect_kind="gemma", segment_id="segment-1"
        )
        assert len(pilot_receipts) == PILOT_COUNT
        assert all(item.duration_ns == exact_duration + 1 for item in pilot_receipts)
        assert all(not hasattr(item, "body") for item in pilot_receipts)

    missing_commit_path = _path(tmp_path, "missing-continuation-commit")
    shutil.copytree(pause_path, missing_commit_path)
    missing_replay = replay_journal(
        missing_commit_path / "journal",
        authority_sha256=canonical_sha256(authority),
    )
    missing_pause = next(
        event
        for event in missing_replay.events
        if event.event_type == "paused_for_justification"
    )
    HashChainJournal(
        missing_commit_path / "journal",
        authority_sha256=canonical_sha256(authority),
    ).append(
        "continuation_authorized",
        {
            "pause_event_sha256": missing_pause.event_sha256,
            "continuation_sha256": "a" * 64,
            "permission_sha256": "b" * 64,
        },
    )
    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(missing_commit_path, authority=authority)
    assert caught.value.code == "continuation_event_invalid"

    with AttemptStore.open(pause_path, authority=authority) as resumed:
        assert resumed.snapshot.status == "paused"
        with pytest.raises(V315StoreError) as caught:
            _append_binding(
                resumed, "attempt_open", invocation_kind="continuation"
            )
        assert caught.value.code == "runtime_binding_receipt_invalid"
        with pytest.raises(V315StoreError) as caught:
            resumed.begin_request(
                request_id="segment-2-pre-version",
                effect_kind="identity",
                request_sha256=_hash("segment-2-pre-version"),
                model_phase="pre_probe_start",
                segment_id="segment-2",
            )
        assert caught.value.code == "request_not_authorized"
        continuation_sha256 = "a" * 64
        permission_sha256 = "b" * 64
        continuation_commit = "c" * 40
        for invalid_commit in ("c" * 39, "C" * 40, "g" * 40):
            with pytest.raises(V315StoreError) as caught:
                resumed.authorize_continuation(
                    continuation_sha256=continuation_sha256,
                    permission_sha256=permission_sha256,
                    continuation_commit=invalid_commit,
                )
            assert caught.value.code == "continuation_not_authorized"
        resumed.authorize_continuation(
            continuation_sha256=continuation_sha256,
            permission_sha256=permission_sha256,
            continuation_commit=continuation_commit,
        )
        assert resumed.snapshot.continuation_authorized is True
        assert resumed.snapshot.continuation_sha256 == continuation_sha256
        assert (
            resumed.snapshot.continuation_permission_sha256
            == permission_sha256
        )
        assert resumed.snapshot.continuation_commit == continuation_commit
        with pytest.raises(V315StoreError) as caught:
            _append_binding(
                resumed,
                "model_continuation_pre",
                invocation_kind="continuation",
            )
        assert caught.value.code == "runtime_binding_receipt_invalid"
        continuation_open = _append_binding(
            resumed, "attempt_open", invocation_kind="continuation"
        )
        assert continuation_open.previous_runtime_binding_receipt_sha256 == (
            pause_binding.runtime_binding_receipt_sha256
        )
        with pytest.raises(V315StoreError) as caught:
            _append_binding(
                resumed, "attempt_open", invocation_kind="continuation"
            )
        assert caught.value.code == "runtime_binding_receipt_invalid"
        _append_binding(
            resumed,
            "model_continuation_pre",
            invocation_kind="continuation",
        )
        restart_permission_sha256 = _hash("fresh-restart-permission")
        resumed.authorize_continuation(
            continuation_sha256=continuation_sha256,
            permission_sha256=restart_permission_sha256,
            continuation_commit=continuation_commit,
        )
        restart_open = _append_binding(
            resumed, "attempt_open", invocation_kind="continuation"
        )
        assert restart_open.previous_runtime_binding_receipt_sha256 != (
            continuation_open.previous_runtime_binding_receipt_sha256
        )
        _append_binding(
            resumed,
            "model_continuation_pre",
            invocation_kind="continuation",
        )
        _seal(
            resumed,
            request_id="segment-2-pre-version",
            effect_kind="identity",
            model_phase="pre_probe_start",
            segment_id="segment-2",
        )
        _seal(
            resumed,
            request_id="segment-2-pre-show",
            effect_kind="identity",
            model_phase="pre_probe",
            segment_id="segment-2",
        )
        assert resumed.snapshot.status == "active"
        assert resumed.snapshot.open_model_segment_id == "segment-2"

    with AttemptStore.open(pause_path, authority=authority) as replayed:
        assert replayed.snapshot.continuation_authorized is True
        assert replayed.snapshot.continuation_sha256 == continuation_sha256
        assert (
            replayed.snapshot.continuation_permission_sha256
            == restart_permission_sha256
        )
        assert replayed.snapshot.continuation_commit == continuation_commit


@pytest.mark.parametrize(
    "event_type",
    [
        "attempt_intent",
        "request_intent",
        "response_committed",
        "checkpoint_committed",
        "paused_for_justification",
        "continuation_authorized",
        "market_values_opened",
        "model_responses_opened",
        "attempt_terminal",
        "foreign_but_chain_valid",
    ],
)
def test_terminal_event_is_universally_last(
    tmp_path: Path, event_type: str
) -> None:
    path = _path(tmp_path, event_type)
    authority = _authority()
    with AttemptStore.create(path, authority=authority) as store:
        _seal_zero_effect_rejection(store)
        authority_sha256 = store.authority_sha256

    # This is a fully canonical, correctly sequenced, correctly hash-chained
    # journal event.  Its payload is deliberately irrelevant: after terminal,
    # replay must reject the event before interpreting its type or contents.
    journal = HashChainJournal(
        path / "journal", authority_sha256=authority_sha256
    )
    journal.append(event_type, {})

    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(path, authority=authority)
    assert caught.value.code == "event_after_terminal"


def test_native_float_cannot_enter_response_or_checkpoint_state(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path)
    with AttemptStore.create(path, authority=_authority()) as store:
        intent = store.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
        with pytest.raises(V315JournalError) as caught:
            store.commit_response(
                request_intent_event_sha256=intent.event_sha256,
                body=b"response",
                metadata={"elapsed": 1.5},
            )
        assert getattr(caught.value, "code", None) == "canonical_json_float_forbidden"


def test_terminal_evidence_is_content_addressed_marker_bound_and_replay_audited(
    tmp_path: Path,
) -> None:
    path = _path(tmp_path)
    authority = _authority()
    evidence = {
        "schema_version": "synthetic-private-terminal-v1",
        "diagnostics": {"reason": "private-value-should-not-print"},
        "effect_counts": {"yahoo": 0, "identity": 0, "gemma": 0},
    }
    with AttemptStore.create(path, authority=authority) as store:
        receipt = _seal_zero_effect_rejection(store, evidence=evidence)
        assert store.snapshot.status == "rejected"
        candidate_path = (
            path
            / "publication_candidates"
            / "terminal"
            / "candidate.json"
        )
        encoded_candidate = candidate_path.read_bytes()
        decoded_candidate = json.loads(encoded_candidate)
        assert decoded_candidate["candidate_sha256"] == receipt.evidence_sha256
        assert len(encoded_candidate) == receipt.evidence_bytes
        assert "private-value-should-not-print" not in repr(receipt)
        committed = store.committed_terminal_evidence()
        assert committed == receipt
        with pytest.raises(TypeError):
            committed.evidence["new"] = True  # type: ignore[index]

        replay = replay_journal(
            path / "journal", authority_sha256=store.authority_sha256
        )
        terminal = replay.events[-1]
        assert terminal.event_type == "attempt_terminal"
        assert terminal.payload["candidate_sha256"] == receipt.evidence_sha256
        assert terminal.payload["candidate_bytes"] == receipt.evidence_bytes
        assert terminal.payload["journal_head_before_terminal_sha256"] == (
            terminal.previous_event_sha256
        )

    with AttemptStore.open(path, authority=authority) as reopened:
        committed = reopened.committed_terminal_evidence()
        assert committed.evidence["schema_version"] == (
            "synthetic-private-terminal-v1"
        )
        assert reopened.snapshot.status == "rejected"


def test_recovery_api_is_reachable_end_to_end_with_ordered_demotion_and_verification_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Drive the real runner across demotion, publication, and verification."""

    from agent_benchmark import sec_gemma_lean_science_v315_contract as contract
    from agent_benchmark import sec_gemma_lean_science_v315_runner as runner
    from tests.test_sec_gemma_lean_science_v315_runner import (
        _development_test_dependencies,
    )

    dependencies, trace, authority, _plan = _development_test_dependencies(
        monkeypatch,
        generation_durations=[],
        fail_first_yahoo=True,
    )
    root = tmp_path.resolve()

    # The real development entrypoint records an irreversible request intent
    # whose callback fails.  No network is used: the injected Yahoo dependency
    # is the deterministic offline failure above.
    with pytest.raises(runner.V315RunnerError) as caught:
        runner.run_development(root, dependencies=dependencies)
    assert caught.value.code == "development_worker_failed"

    # Recovery ordinal 1 persists the deterministic candidate, then the
    # simulated process dies before that candidate is terminal-bound.
    original_append_terminal = AttemptStore.append_attempt_terminal
    append_crash = {"pending": True}

    def crash_after_candidate_persistence(self, **kwargs):
        if append_crash["pending"]:
            append_crash["pending"] = False
            raise KeyboardInterrupt("crash after candidate persistence")
        return original_append_terminal(self, **kwargs)

    monkeypatch.setattr(
        AttemptStore, "append_attempt_terminal", crash_after_candidate_persistence
    )
    with pytest.raises(KeyboardInterrupt, match="candidate persistence"):
        runner.recover_publication(root, dependencies=dependencies)
    monkeypatch.setattr(
        AttemptStore, "append_attempt_terminal", original_append_terminal
    )

    # Ordinal 2 sees the unreferenced candidate, demotes it, terminal-binds it,
    # and publishes exact safe bytes.  The crash is deliberately after the
    # publisher returns but before recovery_terminal is journaled.
    original_finish = AttemptStore.finish_recovery
    finish_crash = {"pending": True}

    def crash_after_exact_publication(self, **kwargs):
        if finish_crash["pending"]:
            finish_crash["pending"] = False
            raise KeyboardInterrupt("crash after exact publication")
        return original_finish(self, **kwargs)

    monkeypatch.setattr(
        AttemptStore, "finish_recovery", crash_after_exact_publication
    )
    with pytest.raises(KeyboardInterrupt, match="exact publication"):
        runner.recover_publication(root, dependencies=dependencies)
    assert len(trace["result_publications"]) == 1

    # Ordinal 3 restarts verification-only.  The exact public bytes are already
    # present, so the injected publisher must not be called a second time.
    recovered = runner.recover_publication(root, dependencies=dependencies)
    assert recovered["status"] == "indeterminate"
    assert recovered["terminal_code"] == "unbound_terminal_candidate"
    assert len(trace["result_publications"]) == 1
    runtime_count = len(trace["runtime_bindings"])
    assert runner.recover_publication(root, dependencies=dependencies) == recovered
    assert len(trace["result_publications"]) == 1
    assert len(trace["runtime_bindings"]) == runtime_count

    attempt_path = root / contract.PRIVATE_DEVELOPMENT_NAMESPACE
    with AttemptStore.open(attempt_path, authority=authority) as audited:
        assert audited.snapshot.status == "indeterminate"
        assert audited.snapshot.terminal_code == "unbound_terminal_candidate"
        assert [
            event.payload["target_kind"] for event in audited.recovery_intents()
        ] == [
            "terminal_indeterminate_created",
            "terminal_demoted",
            "terminal_demoted_bound",
        ]
        assert [
            event.payload["ordinal"] for event in audited.recovery_terminals()
        ] == [3]
        assert audited.latest_recovery_terminal is not None
        assert audited.latest_recovery_terminal.outcome == "published"


def test_nested_mappingproxytype_evidence_is_recursively_thawed_and_canonicalized(
    tmp_path: Path,
) -> None:
    nested = MappingProxyType(
        {
            "outer": MappingProxyType(
                {
                    "items": (
                        MappingProxyType(
                            {"value": "kept", "flags": (True, False)}
                        ),
                    )
                }
            )
        }
    )
    path = _path(tmp_path, "nested-immutable-evidence")
    with AttemptStore.create(path, authority=_authority()) as store:
        receipt = _seal_zero_effect_rejection(store, evidence=nested)
        assert receipt.evidence["outer"]["items"][0]["value"] == "kept"
        with pytest.raises(TypeError):
            receipt.evidence["outer"]["new"] = True  # type: ignore[index]

    candidate = json.loads(
        (
            path
            / "publication_candidates"
            / "terminal"
            / "candidate.json"
        ).read_text(encoding="utf-8")
    )
    assert candidate["evidence"] == {
        "outer": {"items": [{"flags": [True, False], "value": "kept"}]}
    }


def test_missing_or_malformed_terminal_candidate_poisons_replay(
    tmp_path: Path,
) -> None:
    authority = _authority()
    missing_path = _path(tmp_path, "missing-terminal")
    with AttemptStore.create(missing_path, authority=authority) as store:
        _seal_zero_effect_rejection(store)
    (
        missing_path
        / "publication_candidates"
        / "terminal"
        / "candidate.json"
    ).unlink()
    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(missing_path, authority=authority)
    assert caught.value.code == "terminal_candidate_directory_poisoned"

    malformed_path = _path(tmp_path, "malformed-terminal")
    with AttemptStore.create(malformed_path, authority=authority):
        pass
    bundle = malformed_path / "publication_candidates" / "terminal"
    bundle.mkdir()
    (bundle / "candidate.json").write_bytes(b"{}\n")
    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(malformed_path, authority=authority)
    assert caught.value.code == "terminal_candidate_directory_poisoned"


def test_yahoo_cumulative_body_bytes_replay_and_cap_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _path(tmp_path)
    authority = _authority()
    monkeypatch.setattr(store_module, "YAHOO_BATCH_BODY_CAP_BYTES", 5)
    with AttemptStore.create(path, authority=authority) as store:
        first = store.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
        store.commit_response_and_checkpoint(
            request_intent_event_sha256=first.event_sha256,
            body=b"abc",
            metadata={"status": 200},
            checkpoint_state={
                "schema_version": RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION,
                "phase": "none",
                "request_sha256": _hash("request:yahoo-0"),
                "body_sha256": sha256_bytes(b"abc"),
            },
        )
        assert store.snapshot.yahoo_body_bytes == 3
        second = store.begin_request(
            request_id="yahoo-1",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-1"),
        )
        with pytest.raises(V315StorePoisoned) as caught:
            store.commit_response(
                request_intent_event_sha256=second.event_sha256,
                body=b"def",
                metadata={"status": 200},
            )
    assert caught.value.code == "yahoo_batch_body_cap_exceeded"


def test_per_response_caps_are_enforced_on_commit_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority()
    commit_path = _path(tmp_path, "commit-cap")
    with AttemptStore.create(commit_path, authority=authority) as attempt:
        intent = attempt.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
        monkeypatch.setattr(store_module, "YAHOO_RESPONSE_CAP_BYTES", 3)
        with pytest.raises(V315StoreError) as caught:
            attempt.commit_response(
                request_intent_event_sha256=intent.event_sha256,
                body=b"four",
                metadata={"status": 200},
            )
        assert caught.value.code == "yahoo_response_bytes_invalid"

    monkeypatch.setattr(
        store_module, "YAHOO_RESPONSE_CAP_BYTES", 64 * 1024 * 1024
    )
    yahoo_replay_path = _path(tmp_path, "yahoo-replay-cap")
    with AttemptStore.create(yahoo_replay_path, authority=authority) as attempt:
        _seal(attempt, request_id="yahoo-0", effect_kind="yahoo")
    monkeypatch.setattr(store_module, "YAHOO_RESPONSE_CAP_BYTES", 3)
    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(yahoo_replay_path, authority=authority)
    assert caught.value.code == "yahoo_response_bytes_invalid"

    monkeypatch.setattr(
        store_module, "YAHOO_RESPONSE_CAP_BYTES", 64 * 1024 * 1024
    )
    model_replay_path = _path(tmp_path, "model-replay-cap")
    with AttemptStore.create(model_replay_path, authority=authority) as attempt:
        _six_yahoo(attempt)
        _pre_probe(attempt, "segment-1")
        intent = attempt.begin_request(
            request_id="gemma-0",
            effect_kind="gemma",
            request_sha256=_hash("request:gemma-0"),
            model_phase="generation",
            segment_id="segment-1",
        )
        attempt.commit_response_and_checkpoint(
            request_intent_event_sha256=intent.event_sha256,
            body=b"x" * 100,
            metadata={"status": 200},
            duration_ns=1,
            checkpoint_state={
                "schema_version": RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION,
                "phase": "generation",
                "request_sha256": _hash("request:gemma-0"),
                "body_sha256": sha256_bytes(b"x" * 100),
            },
        )
    monkeypatch.setattr(store_module, "MODEL_RESPONSE_CAP_BYTES", 50)
    with pytest.raises(V315StorePoisoned) as caught:
        AttemptStore.open(model_replay_path, authority=authority)
    assert caught.value.code == "model_response_bytes_invalid"


def test_content_replay_cap_is_checked_before_any_payload_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _path(tmp_path)
    with AttemptStore.create(path, authority=_authority()) as attempt:
        intent = attempt.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
        monkeypatch.setattr(store_module, "_MAX_CONTENT_BYTES", 32)
        with pytest.raises(V315StoreError) as caught:
            attempt.commit_response(
                request_intent_event_sha256=intent.event_sha256,
                body=b"bounded",
                metadata={"status": 200},
            )
        assert caught.value.code == "content_size_invalid"
        assert tuple((path / "payloads").iterdir()) == ()


def test_content_write_never_follows_foreign_symlink_after_presence_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _path(tmp_path)
    foreign = (tmp_path / "foreign.txt").resolve()
    foreign.write_bytes(b"do-not-touch")
    with AttemptStore.create(path, authority=_authority()) as attempt:
        intent = attempt.begin_request(
            request_id="yahoo-0",
            effect_kind="yahoo",
            request_sha256=_hash("request:yahoo-0"),
        )
        original = store_module._write_exclusive
        attacked = False

        def install_link_then_write(
            pending: Path, encoded: bytes, *, code: str
        ) -> None:
            nonlocal attacked
            if not attacked:
                try:
                    pending.symlink_to(foreign)
                except OSError:
                    os.link(foreign, pending)
                attacked = True
            original(pending, encoded, code=code)

        monkeypatch.setattr(
            store_module, "_write_exclusive", install_link_then_write
        )
        with pytest.raises(V315StorePoisoned) as caught:
            attempt.commit_response(
                request_intent_event_sha256=intent.event_sha256,
                body=b"bounded",
                metadata={"status": 200},
            )
        assert caught.value.code == "content_write_failed"
    assert foreign.read_bytes() == b"do-not-touch"
