from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import agent_benchmark.sec_gemma_lean_v35_journal as journal_module
from agent_benchmark.sec_point_in_time import validate_sec_user_agent
from agent_benchmark.sec_gemma_lean_v35_journal import (
    AcquisitionJournal,
    COMPLETE_SUBMISSION_BODY_LIMIT,
    CompleteSubmissionTarget,
    HISTORICAL_SUBMISSIONS_BODY_LIMIT,
    INVOCATION_DEADLINE_MS,
    LIFETIME_INTENT_CAP,
    LIFETIME_RECEIVED_BYTE_CAP,
    MAIN_SUBMISSIONS_BODY_LIMIT,
    MASTER_COMPRESSED_BODY_LIMIT,
    MASTER_DECOMPRESSED_BODY_LIMIT,
    MASTER_DECOMPRESSED_BYTE_CAP,
    PrivateSecContact,
    REQUEST_DEADLINE_MS,
    ROLE_DEADLINE_MS,
    SOURCE_DIAGNOSTIC_CODES,
    STAGE_ACTIVE_TIME_CAP_MS,
    SUCCESSFUL_RESPONSE_BYTE_CAP,
    SecGemmaLeanV35JournalError,
    build_stage_role_plan,
    validate_detached_journal,
    validate_stage_role_plan,
)


CONTACT_TEXT = (
    'Alder Investigación "Desk" sec+desk@alder-research-739184.com'
)
BODY_SHA256 = hashlib.sha256(b"x").hexdigest()
TRANSPORT_RECEIPT_SHA256 = hashlib.sha256(b"transport-receipt").hexdigest()
PARSE_RECEIPT_SHA256 = hashlib.sha256(b"parse-receipt").hexdigest()
CHECKPOINT_SHA256 = hashlib.sha256(b"checkpoint").hexdigest()


def _root(tmp_path: Path, name: str = "journal") -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def _contact() -> PrivateSecContact:
    return PrivateSecContact(CONTACT_TEXT)


def _development_plan(*, historical: tuple[str, ...] = ()):
    return build_stage_role_plan("development", historical, ())


def _journal(tmp_path: Path, *, root_name: str = "journal"):
    root = _root(tmp_path, root_name)
    contact = _contact()
    plan = _development_plan()
    return root, contact, plan, AcquisitionJournal(
        root, stage="development", contact=contact
    )


def _seal_next(journal: AcquisitionJournal, *, wait_ms: int = 1_000) -> None:
    role = journal.state.planned_roles[journal.state.role_seals]
    intent = journal.record_role_intent(role.role_id, dispatch_wait_ms=wait_ms)
    response = journal.record_response_complete(
        role.role_id,
        intent_event_sha256=intent,
        body_bytes=1,
        body_sha256=BODY_SHA256,
        transport_receipt_sha256=TRANSPORT_RECEIPT_SHA256,
        request_duration_ms=1,
    )
    journal.record_role_seal(
        role.role_id,
        intent_event_sha256=intent,
        response_event_sha256=response,
        blob_sha256=BODY_SHA256,
        parse_receipt_sha256=PARSE_RECEIPT_SHA256,
        body_bytes=1,
        decompressed_bytes=1 if role.phase == "quarterly_master" else None,
        role_duration_ms=2,
    )


def _build_maximum_formula_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage: str,
    historical_count: int,
    complete_count: int,
) -> tuple[Path, PrivateSecContact, AcquisitionJournal]:
    """Build real event files while avoiding quadratic repeated disk reads."""

    root = _root(tmp_path, f"{stage}-maximum")
    contact = _contact()
    cached_events: list[dict[str, object]] = []
    real_read_events = journal_module._read_events
    real_rename = journal_module.os.rename

    def cached_read_events(candidate: Path) -> list[dict[str, object]]:
        if Path(candidate) == root:
            return list(cached_events)
        return real_read_events(candidate)

    def capturing_rename(source: object, target: object) -> None:
        real_rename(source, target)
        target_path = Path(target)
        if target_path.parent == root:
            cached_events.append(
                json.loads(target_path.read_text(encoding="ascii"))
            )

    monkeypatch.setattr(journal_module.os, "fsync", lambda _descriptor: None)
    monkeypatch.setattr(journal_module, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(journal_module, "_read_events", cached_read_events)
    monkeypatch.setattr(journal_module.os, "rename", capturing_rename)

    journal = AcquisitionJournal(root, stage=stage, contact=contact)
    historical = tuple(
        f"CIK0000320193-submissions-{index:03d}.json"
        for index in range(historical_count)
    )
    complete = tuple(
        CompleteSubmissionTarget(
            "2000-01-01", f"0000320193-00-{index:06d}"
        )
        for index in range(complete_count)
    )

    journal.open_invocation()
    _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=2)
    journal.append_historical_phase(historical)
    journal.open_invocation()
    while journal.state.next_role_id is not None:
        _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=max(1, historical_count * 2))
    journal.append_master_phase(
        submissions_snapshot_receipt_sha256=TRANSPORT_RECEIPT_SHA256
    )
    journal.open_invocation()
    master_count = journal.state.quarterly_master_count
    assert master_count is not None
    while journal.state.next_role_id is not None:
        _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=master_count * 2)
    journal.append_complete_phase(
        complete,
        reconciliation_and_prior_chain_receipt_sha256=PARSE_RECEIPT_SHA256,
    )
    journal.open_invocation()
    while journal.state.next_role_id is not None:
        _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=max(1, complete_count * 2))
    journal.seal_terminal_pass(checkpoint_sha256=CHECKPOINT_SHA256)

    # Restore the real reader before the assertion below.  The proof must be a
    # detached reconstruction from the numbered files, not the build cache.
    monkeypatch.setattr(journal_module, "_read_events", real_read_events)
    monkeypatch.setattr(journal_module.os, "rename", real_rename)
    return root, contact, journal


def _assert_rehashed_basis_tamper_rejected(
    root: Path,
    contact: PrivateSecContact,
    *,
    stage: str,
    phase: str,
) -> None:
    phase_path: Path | None = None
    for candidate in sorted(root.glob("*.json")):
        event = json.loads(candidate.read_text(encoding="ascii"))
        if (
            event["event_type"] == "phase_plan"
            and event["payload"]["phase"] == phase
        ):
            phase_path = candidate
            break
    assert phase_path is not None
    original = phase_path.read_bytes()
    event = json.loads(original.decode("ascii"))
    replacement = "0" * 64
    if event["payload"]["basis_receipt_sha256"] == replacement:
        replacement = "f" * 64
    event["payload"]["basis_receipt_sha256"] = replacement
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    event["event_sha256"] = hashlib.sha256(
        (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
    ).hexdigest()
    phase_path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="",
    )
    try:
        with pytest.raises(SecGemmaLeanV35JournalError) as caught:
            validate_detached_journal(
                root, stage, contact.fingerprint_sha256
            )
        assert caught.value.code == "journal_invalid"
    finally:
        phase_path.write_bytes(original)


def test_frozen_stage_formula_order_and_exact_caps() -> None:
    historical = tuple(
        f"CIK0000320193-submissions-{index:03d}.json" for index in range(16)
    )
    complete = tuple(
        CompleteSubmissionTarget(
            "2000-01-01", f"0000320193-00-{index:06d}"
        )
        for index in range(128)
    )
    development = build_stage_role_plan("development", historical, complete)
    intermediate = build_stage_role_plan("intermediate", historical, complete[:24])
    final = build_stage_role_plan("final", historical, complete[:16])

    assert development.expected_successful_requests == 1 + 16 + 100 + 128 == 245
    assert intermediate.expected_successful_requests == 1 + 16 + 120 + 24 == 161
    assert final.expected_successful_requests == 1 + 16 + 131 + 16 == 164
    assert [role.ordinal for role in development.roles] == list(range(1, 246))
    assert development.roles[0].role_id == "submissions/main"
    assert development.roles[1].role_id.endswith("submissions-000.json")
    assert development.roles[16].role_id.endswith("submissions-015.json")
    assert development.roles[17].role_id == "master/1994/QTR1"
    assert development.roles[18].role_id == "master/1994/QTR2"
    assert development.roles[116].role_id == "master/2018/QTR4"
    assert development.roles[117].role_id == "complete/0000320193-00-000000"
    assert sum(role.body_limit_bytes for role in development.roles) == (
        SUCCESSFUL_RESPONSE_BYTE_CAP
    )
    assert sum(role.reservation_bytes for role in development.roles) == (
        SUCCESSFUL_RESPONSE_BYTE_CAP + 245
    )
    assert sum(
        role.decompressed_body_limit_bytes or 0 for role in final.roles
    ) == MASTER_DECOMPRESSED_BYTE_CAP
    assert LIFETIME_INTENT_CAP == 256
    assert LIFETIME_RECEIVED_BYTE_CAP == 20 * 1024**3


@pytest.mark.parametrize(
    ("stage", "quarter_count", "maximum_requests"),
    (
        ("development", 100, 245),
        ("intermediate", 120, 161),
        ("final", 131, 164),
    ),
)
def test_every_stage_starts_with_contiguous_1994_q1_q2_roles(
    stage: str, quarter_count: int, maximum_requests: int
) -> None:
    plan = build_stage_role_plan(stage, (), ())
    config = journal_module.STAGE_CONFIGS[stage]
    master_roles = tuple(
        role for role in plan.roles if role.phase == "quarterly_master"
    )

    assert config.as_dict()["master_start"] == "1994-Q1"
    assert config.quarterly_master_count == quarter_count
    assert config.maximum_successful_requests == maximum_requests
    assert len(master_roles) == quarter_count
    assert [role.role_id for role in master_roles[:3]] == [
        "master/1994/QTR1",
        "master/1994/QTR2",
        "master/1994/QTR3",
    ]


@pytest.mark.parametrize(
    ("stage", "historical_count", "complete_count", "expected_formula"),
    [
        ("development", 16, 128, 245),
        ("intermediate", 16, 24, 161),
        ("final", 16, 16, 164),
    ],
)
def test_maximum_formula_passes_real_detached_replay_and_binds_q_and_p(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    historical_count: int,
    complete_count: int,
    expected_formula: int,
) -> None:
    root, contact, journal = _build_maximum_formula_journal(
        tmp_path,
        monkeypatch,
        stage=stage,
        historical_count=historical_count,
        complete_count=complete_count,
    )
    detached = validate_detached_journal(
        root, stage, contact.fingerprint_sha256
    )
    assert detached == journal.state
    assert detached.formula_requests == expected_formula
    assert (
        detached.lifetime_intents
        == detached.http_200_responses
        == detached.role_seals
        == expected_formula
    )
    assert detached.accounting_complete
    assert not detached.source_authoritative
    assert detached.passed

    _assert_rehashed_basis_tamper_rejected(
        root, contact, stage=stage, phase="quarterly_master"
    )
    assert validate_detached_journal(
        root, stage, contact.fingerprint_sha256
    ).passed
    _assert_rehashed_basis_tamper_rejected(
        root, contact, stage=stage, phase="complete_submission"
    )
    assert validate_detached_journal(
        root, stage, contact.fingerprint_sha256
    ).passed


def test_phase_limits_and_canonical_urls_are_frozen() -> None:
    historical = ("CIK0000320193-submissions-001.json",)
    complete = (CompleteSubmissionTarget("2000-01-01", "0000912057-00-023442"),)
    plan = build_stage_role_plan("development", historical, complete)
    roles = {role.role_id: role for role in plan.roles}

    assert roles["submissions/main"].body_limit_bytes == MAIN_SUBMISSIONS_BODY_LIMIT
    assert (
        roles["submissions/historical/CIK0000320193-submissions-001.json"]
        .body_limit_bytes
        == HISTORICAL_SUBMISSIONS_BODY_LIMIT
    )
    assert roles["master/1994/QTR1"].body_limit_bytes == MASTER_COMPRESSED_BODY_LIMIT
    assert (
        roles["master/1994/QTR1"].decompressed_body_limit_bytes
        == MASTER_DECOMPRESSED_BODY_LIMIT
    )
    assert roles["master/1994/QTR2"].body_limit_bytes == MASTER_COMPRESSED_BODY_LIMIT
    complete_role = roles["complete/0000912057-00-023442"]
    assert complete_role.body_limit_bytes == COMPLETE_SUBMISSION_BODY_LIMIT
    assert complete_role.url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "0000912057-00-023442.txt"
    )
    assert all(role.reservation_bytes == role.body_limit_bytes + 1 for role in plan.roles)


def test_complete_target_boundary_includes_1994_q1_without_special_case() -> None:
    observed = CompleteSubmissionTarget(
        "1994-01-26", "0000320193-94-000002"
    )
    first_day = CompleteSubmissionTarget(
        "1994-01-01", "0000320193-94-000001"
    )
    plan = build_stage_role_plan(
        "development", (), (first_day, observed)
    )
    complete_roles = tuple(
        role for role in plan.roles if role.phase == "complete_submission"
    )

    assert [(role.filing_date, role.role_id) for role in complete_roles] == [
        ("1994-01-01", "complete/0000320193-94-000001"),
        ("1994-01-26", "complete/0000320193-94-000002"),
    ]

    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        build_stage_role_plan(
            "development",
            (),
            (
                CompleteSubmissionTarget(
                    "1993-12-31", "0000320193-93-000001"
                ),
            ),
        )
    assert caught.value.code == "complete_target_plan_invalid"


def test_plan_rejects_noncanonical_order_duplicates_and_cap() -> None:
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        build_stage_role_plan(
            "development",
            (
                "CIK0000320193-submissions-002.json",
                "CIK0000320193-submissions-001.json",
            ),
            (),
        )
    assert caught.value.code == "historical_plan_invalid"

    duplicate = CompleteSubmissionTarget("2000-01-01", "0000320193-00-000001")
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        build_stage_role_plan("development", (), (duplicate, duplicate))
    assert caught.value.code == "complete_target_plan_invalid"

    too_many = tuple(
        CompleteSubmissionTarget("2000-01-01", f"0000320193-00-{i:06d}")
        for i in range(129)
    )
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        build_stage_role_plan("development", (), too_many)
    assert caught.value.code == "complete_request_cap_exceeded"

    valid = _development_plan()
    forged_role = replace(valid.roles[0], url="https://example.invalid/forged")
    forged_plan = replace(valid, roles=(forged_role, *valid.roles[1:]))
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        validate_stage_role_plan(forged_plan)
    assert caught.value.code == "role_plan_invalid"


def test_clean_resume_only_between_sealed_roles_and_detached_state(tmp_path: Path) -> None:
    root, contact, plan, journal = _journal(tmp_path)
    journal.open_invocation()
    _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=10)

    detached = validate_detached_journal(
        root, "development", contact.fingerprint_sha256
    )
    assert detached.resume_allowed
    assert detached.sealed_role_ids == ("submissions/main",)
    assert detached.planned_roles == (plan.roles[0],)
    assert detached.next_required_expansion == "historical_submissions"
    assert detached.lifetime_intents == detached.http_200_responses == detached.role_seals == 1
    assert detached.successful_response_bytes == 1
    assert detached.lifetime_received_bytes == 1

    resumed = AcquisitionJournal(root, stage="development", contact=contact)
    resumed.append_historical_phase(())
    resumed.append_master_phase(
        submissions_snapshot_receipt_sha256=TRANSPORT_RECEIPT_SHA256
    )
    resumed.open_invocation()
    _seal_next(resumed)
    resumed.close_invocation(invocation_duration_ms=11)
    assert resumed.state.role_seals == 2
    assert resumed.state.cumulative_active_ms == 21
    assert resumed.state == validate_detached_journal(
        root, "development", contact.fingerprint_sha256
    )


def test_full_formula_requires_one_intent_http_200_and_seal_per_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The formula test writes about 300 tiny files.  File fsync behavior is tested
    # separately; skipping physical flushes here keeps this offline test quick.
    monkeypatch.setattr(journal_module.os, "fsync", lambda _descriptor: None)
    monkeypatch.setattr(journal_module, "_fsync_directory", lambda _path: None)
    root, contact, plan, journal = _journal(tmp_path)
    journal.open_invocation()
    _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=2)
    historical_event = journal.append_historical_phase(())
    assert historical_event
    master_event = journal.append_master_phase(
        submissions_snapshot_receipt_sha256=TRANSPORT_RECEIPT_SHA256
    )
    assert master_event
    journal.open_invocation()
    while journal.state.next_role_id is not None:
        _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=1_000)
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_complete_phase(
            (
                CompleteSubmissionTarget("2000-01-02", "0000320193-00-000002"),
                CompleteSubmissionTarget("2000-01-01", "0000320193-00-000001"),
            ),
            reconciliation_and_prior_chain_receipt_sha256=PARSE_RECEIPT_SHA256,
        )
    assert caught.value.code == "complete_target_plan_invalid"
    complete_event = journal.append_complete_phase(
        (),
        reconciliation_and_prior_chain_receipt_sha256=PARSE_RECEIPT_SHA256,
    )
    assert complete_event
    phase_events = [
        json.loads(path.read_text(encoding="ascii"))["payload"]
        for path in sorted(root.glob("*.json"))
        if json.loads(path.read_text(encoding="ascii"))["event_type"]
        == "phase_plan"
    ]
    assert [payload["phase"] for payload in phase_events] == [
        "historical_submissions",
        "quarterly_master",
        "complete_submission",
    ]
    assert phase_events[0]["basis_kind"] == "main_parse_receipt"
    assert phase_events[0]["basis_receipt_sha256"] == PARSE_RECEIPT_SHA256
    assert phase_events[0]["role_count"] == 0
    assert phase_events[1]["basis_receipt_sha256"] == TRANSPORT_RECEIPT_SHA256
    assert phase_events[1]["role_count"] == 100
    assert phase_events[2]["basis_receipt_sha256"] == PARSE_RECEIPT_SHA256
    assert phase_events[2]["role_count"] == 0
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_complete_phase(
            (),
            reconciliation_and_prior_chain_receipt_sha256=PARSE_RECEIPT_SHA256,
        )
    assert caught.value.code == "phase_expansion_order_invalid"
    assert journal.state.can_finalize_pass
    assert (
        journal.state.lifetime_intents
        == journal.state.http_200_responses
        == journal.state.role_seals
        == 1 + 0 + 100 + 0
        == 101
    )
    assert journal.state.phase_expansions == (
        "historical_submissions",
        "quarterly_master",
        "complete_submission",
    )
    assert journal.state.formula_requests == 101
    journal.seal_terminal_pass(checkpoint_sha256=CHECKPOINT_SHA256)
    state = validate_detached_journal(
        root, "development", contact.fingerprint_sha256
    )
    assert state.passed
    assert state.terminal_status == "passed"
    assert not state.resume_allowed


def test_phase_expansions_cannot_be_skipped_repeated_or_reordered(
    tmp_path: Path,
) -> None:
    _root_path, _contact_value, _plan, journal = _journal(tmp_path)
    assert journal.state.planned_roles == (_development_plan().roles[0],)
    assert journal.state.phase_expansions == ()

    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_historical_phase(())
    assert caught.value.code == "phase_boundary_not_reached"
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_master_phase(
            submissions_snapshot_receipt_sha256=TRANSPORT_RECEIPT_SHA256
        )
    assert caught.value.code == "phase_boundary_not_reached"

    journal.open_invocation()
    _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=2)
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_master_phase(
            submissions_snapshot_receipt_sha256=TRANSPORT_RECEIPT_SHA256
        )
    assert caught.value.code == "phase_expansion_order_invalid"
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_complete_phase(
            (),
            reconciliation_and_prior_chain_receipt_sha256=PARSE_RECEIPT_SHA256,
        )
    assert caught.value.code == "phase_expansion_order_invalid"

    historical_name = "CIK0000320193-submissions-001.json"
    journal.append_historical_phase((historical_name,))
    assert journal.state.phase_expansions == ("historical_submissions",)
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_historical_phase(())
    assert caught.value.code == "phase_boundary_not_reached"
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_master_phase(
            submissions_snapshot_receipt_sha256=TRANSPORT_RECEIPT_SHA256
        )
    assert caught.value.code == "phase_boundary_not_reached"
    journal.open_invocation()
    assert journal.state.next_role_id == f"submissions/historical/{historical_name}"
    _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=2)
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_historical_phase(())
    assert caught.value.code == "phase_expansion_order_invalid"
    journal.append_master_phase(
        submissions_snapshot_receipt_sha256=TRANSPORT_RECEIPT_SHA256
    )
    assert journal.state.phase_expansions == (
        "historical_submissions",
        "quarterly_master",
    )
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.append_master_phase(
            submissions_snapshot_receipt_sha256=TRANSPORT_RECEIPT_SHA256
        )
    assert caught.value.code == "phase_boundary_not_reached"


@pytest.mark.parametrize("mutation", ("swap_q1_q2", "omit_q1"))
def test_rehashed_master_plan_cannot_swap_or_omit_q1_q2(
    tmp_path: Path, mutation: str
) -> None:
    root, contact, _plan, journal = _journal(
        tmp_path, root_name=f"tampered-master-{mutation}"
    )
    journal.open_invocation()
    _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=2)
    journal.append_historical_phase(())
    journal.append_master_phase(
        submissions_snapshot_receipt_sha256=TRANSPORT_RECEIPT_SHA256
    )

    phase_path: Path | None = None
    event: dict[str, object] | None = None
    for candidate in sorted(root.glob("*.json")):
        candidate_event = json.loads(candidate.read_text(encoding="ascii"))
        if (
            candidate_event["event_type"] == "phase_plan"
            and candidate_event["payload"]["phase"] == "quarterly_master"
        ):
            phase_path = candidate
            event = candidate_event
            break
    assert phase_path is not None
    assert event is not None
    payload = event["payload"]
    assert isinstance(payload, dict)
    roles = payload["roles"]
    assert isinstance(roles, list)
    assert [role["role_id"] for role in roles[:2]] == [
        "master/1994/QTR1",
        "master/1994/QTR2",
    ]

    if mutation == "swap_q1_q2":
        roles[0], roles[1] = roles[1], roles[0]
    else:
        roles.pop(0)
        payload["role_count"] = len(roles)
    for ordinal, role in enumerate(roles, start=2):
        role["ordinal"] = ordinal

    unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
    event["event_sha256"] = hashlib.sha256(
        (json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
    ).hexdigest()
    phase_path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="",
    )

    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        validate_detached_journal(
            root, "development", contact.fingerprint_sha256
        )
    assert caught.value.code == "journal_invalid"


def test_rehashed_tampered_phase_expansion_fails_detached_replay(
    tmp_path: Path,
) -> None:
    root, contact, _plan, journal = _journal(tmp_path)
    journal.open_invocation()
    _seal_next(journal)
    journal.close_invocation(invocation_duration_ms=2)
    journal.append_historical_phase(())
    phase_path = root / "00000006.json"
    event = json.loads(phase_path.read_text(encoding="ascii"))
    event["payload"]["basis_receipt_sha256"] = "0" * 64
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    body_bytes = (
        json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    event["event_sha256"] = hashlib.sha256(body_bytes).hexdigest()
    phase_path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="",
    )
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        validate_detached_journal(
            root, "development", contact.fingerprint_sha256
        )
    assert caught.value.code == "journal_invalid"


def test_l_plus_one_rejects_and_retains_reservation(tmp_path: Path) -> None:
    root, contact, plan, journal = _journal(tmp_path)
    journal.open_invocation()
    role = plan.roles[0]
    intent = journal.record_role_intent(role.role_id, dispatch_wait_ms=1_000)
    before = journal.state.event_count

    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.record_response_complete(
            role.role_id,
            intent_event_sha256=intent,
            body_bytes=role.body_limit_bytes + 1,
            body_sha256=BODY_SHA256,
            transport_receipt_sha256=TRANSPORT_RECEIPT_SHA256,
            request_duration_ms=1,
        )
    assert caught.value.code == "body_limit_exceeded"
    assert journal.state.event_count == before

    journal.record_role_failure(
        role.role_id,
        intent_event_sha256=intent,
        error_code="body_limit_exceeded",
        observed_body_bytes=role.body_limit_bytes + 1,
        request_duration_ms=1,
        role_duration_ms=2,
    )
    assert journal.state.lifetime_received_bytes == role.body_limit_bytes + 1
    journal.close_invocation(invocation_duration_ms=3)
    failed = AcquisitionJournal(root, stage="development", contact=contact)
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        failed.open_invocation()
    assert caught.value.code == "body_limit_exceeded"
    failed.seal_terminal_rejection()
    assert failed.state.terminal_code == "body_limit_exceeded"


def test_observed_bytes_cannot_exceed_one_sentinel(tmp_path: Path) -> None:
    _root_path, _contact_value, plan, journal = _journal(tmp_path)
    journal.open_invocation()
    role = plan.roles[0]
    intent = journal.record_role_intent(role.role_id, dispatch_wait_ms=1_000)
    before = journal.state.event_count
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.record_role_failure(
            role.role_id,
            intent_event_sha256=intent,
            error_code="body_limit_exceeded",
            observed_body_bytes=role.body_limit_bytes + 2,
            request_duration_ms=1,
            role_duration_ms=2,
        )
    assert caught.value.code == "journal_invalid"
    assert journal.state.event_count == before


@pytest.mark.parametrize(
    ("with_intent", "expected_code"),
    [(False, "unclosed_invocation"), (True, "open_intent")],
)
def test_crash_open_is_permanently_nonpassable(
    tmp_path: Path, with_intent: bool, expected_code: str
) -> None:
    root, contact, plan, journal = _journal(
        tmp_path, root_name=f"crash-{expected_code}"
    )
    journal.open_invocation()
    if with_intent:
        journal.record_role_intent(plan.roles[0].role_id, dispatch_wait_ms=1_000)

    recovered = AcquisitionJournal(root, stage="development", contact=contact)
    assert recovered.state.derived_terminal_code == expected_code
    assert not recovered.state.resume_allowed
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        recovered.open_invocation()
    assert caught.value.code == expected_code
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        recovered.seal_terminal_rejection()
    assert caught.value.code == "unclean_duration_required"
    recovered.seal_terminal_rejection(unclean_active_ms=1)
    assert recovered.state.terminal_status == "rejected"
    assert recovered.state.terminal_code == expected_code
    assert recovered.state.terminal_unclean_active_ms == 1


def test_failure_has_fixed_code_and_role_cannot_be_redispatched(tmp_path: Path) -> None:
    root, contact, plan, journal = _journal(tmp_path)
    journal.open_invocation()
    role = plan.roles[0]
    intent = journal.record_role_intent(role.role_id, dispatch_wait_ms=1_000)

    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.record_role_failure(
            role.role_id,
            intent_event_sha256=intent,
            error_code="socket said secret details",
            observed_body_bytes=0,
            request_duration_ms=1,
            role_duration_ms=1,
        )
    assert caught.value.code == "unsafe_error_code"

    journal.record_role_failure(
        role.role_id,
        intent_event_sha256=intent,
        error_code="transport_error",
        observed_body_bytes=0,
        request_duration_ms=1,
        role_duration_ms=1,
    )
    journal.close_invocation(invocation_duration_ms=2)
    recovered = AcquisitionJournal(root, stage="development", contact=contact)
    assert recovered.state.next_role_id == role.role_id
    assert recovered.state.lifetime_intents == 1
    assert recovered.state.lifetime_received_bytes == role.reservation_bytes
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        recovered.open_invocation()
    assert caught.value.code == "transport_error"


def test_only_exact_source_diagnostic_is_terminal_safe_and_detached(
    tmp_path: Path,
) -> None:
    assert SOURCE_DIAGNOSTIC_CODES == frozenset(
        {"master_boundary_incomplete"}
    )
    root = _root(tmp_path, "exact-source-diagnostic")
    contact = _contact()
    acquisition = AcquisitionJournal(
        root, stage="development", contact=contact
    )
    acquisition.seal_terminal_rejection(
        code="master_boundary_incomplete"
    )

    detached = validate_detached_journal(
        root, "development", contact.fingerprint_sha256
    )
    assert detached.terminal_status == "rejected"
    assert detached.terminal_code == "master_boundary_incomplete"
    assert detached.active_invocation is None
    assert detached.open_role_id is None
    assert not detached.resume_allowed
    assert not detached.passed

    for index, unsafe in enumerate(
        (
            "another_lowercase_code",
            "master_boundary_incomplete_extra",
            "source said secret details",
            CONTACT_TEXT,
        )
    ):
        unsafe_root = _root(tmp_path, f"unsafe-source-diagnostic-{index}")
        unsafe_journal = AcquisitionJournal(
            unsafe_root, stage="development", contact=contact
        )
        with pytest.raises(SecGemmaLeanV35JournalError) as caught:
            unsafe_journal.seal_terminal_rejection(code=unsafe)
        assert caught.value.code == "unsafe_error_code"
        assert not list(unsafe_root.iterdir())


def test_private_contact_scans_raw_json_html_and_percent_forms() -> None:
    contact = _contact()
    needles = contact.serialized_echo_needles()
    assert CONTACT_TEXT.encode("utf-8") in needles
    assert any(b"\\u00f3" in needle for needle in needles)
    assert any(b"&quot;" in needle for needle in needles)
    assert any(b"%22" in needle and b"%40" in needle for needle in needles)
    assert CONTACT_TEXT not in repr(contact)
    assert contact.request_header_closure()() == CONTACT_TEXT
    assert validate_sec_user_agent(CONTACT_TEXT).sha256 == (
        f"sha256:{contact.fingerprint_sha256}"
    )

    for needle in needles:
        assert contact.contains_echo(b"prefix:" + needle + b":suffix")
        with pytest.raises(SecGemmaLeanV35JournalError) as caught:
            contact.assert_redacted({"nested": [b"safe", needle]})
        assert caught.value.code == "privacy_echo"
    contact.assert_redacted(
        {
            "error_code": "transport_error",
            "contact_fingerprint_sha256": contact.fingerprint_sha256,
        }
    )


@pytest.mark.parametrize(
    "invalid_contact",
    [
        "Alder Research Compliance",
        "Alder Example sec@alder-research-739184.com",
        "Alder Research sec@example.com",
        "Alder Placeholder sec@alder-research-739184.com",
    ],
)
def test_private_contact_rejects_missing_or_placeholder_identity(
    invalid_contact: str,
) -> None:
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        PrivateSecContact(invalid_contact)
    assert caught.value.code == "private_contact_invalid"


def test_tamper_breaks_canonical_hash_chain(tmp_path: Path) -> None:
    root, contact, plan, journal = _journal(tmp_path)
    journal.open_invocation()
    journal.close_invocation(invocation_duration_ms=1)
    path = root / "00000002.json"
    event = json.loads(path.read_text(encoding="ascii"))
    event["payload"]["invocation_duration_ms"] = 2
    path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
        newline="",
    )
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        validate_detached_journal(root, "development", contact.fingerprint_sha256)
    assert caught.value.code == "journal_invalid"


def test_invalid_duration_or_gap_is_rejected_before_persistence(tmp_path: Path) -> None:
    _root_path, _contact_value, plan, journal = _journal(tmp_path)
    journal.open_invocation()
    role = plan.roles[0]
    before = journal.state.event_count
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.record_role_intent(role.role_id, dispatch_wait_ms=999)
    assert caught.value.code == "dispatch_gap_too_short"
    assert journal.state.event_count == before

    intent = journal.record_role_intent(role.role_id, dispatch_wait_ms=1_000)
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.record_response_complete(
            role.role_id,
            intent_event_sha256=intent,
            body_bytes=1,
            body_sha256=BODY_SHA256,
            transport_receipt_sha256=TRANSPORT_RECEIPT_SHA256,
            request_duration_ms=REQUEST_DEADLINE_MS + 1,
        )
    assert caught.value.code == "journal_invalid"
    assert journal.state.event_count == before + 1


def test_transport_and_parse_receipt_commitments_are_required_before_seal(
    tmp_path: Path,
) -> None:
    _root_path, _contact_value, _plan, journal = _journal(tmp_path)
    journal.open_invocation()
    role = journal.state.planned_roles[0]
    intent = journal.record_role_intent(role.role_id, dispatch_wait_ms=1_000)
    before_response = journal.state.event_count
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.record_response_complete(
            role.role_id,
            intent_event_sha256=intent,
            body_bytes=1,
            body_sha256=BODY_SHA256,
            transport_receipt_sha256="not-a-hash",
            request_duration_ms=1,
        )
    assert caught.value.code == "journal_invalid"
    assert journal.state.event_count == before_response

    response = journal.record_response_complete(
        role.role_id,
        intent_event_sha256=intent,
        body_bytes=1,
        body_sha256=BODY_SHA256,
        transport_receipt_sha256=TRANSPORT_RECEIPT_SHA256,
        request_duration_ms=1,
    )
    before_seal = journal.state.event_count
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.record_role_seal(
            role.role_id,
            intent_event_sha256=intent,
            response_event_sha256=response,
            blob_sha256=BODY_SHA256,
            parse_receipt_sha256="not-a-hash",
            body_bytes=1,
            decompressed_bytes=None,
            role_duration_ms=2,
        )
    assert caught.value.code == "journal_invalid"
    assert journal.state.event_count == before_seal


def test_duration_caps_are_exact_constants() -> None:
    assert REQUEST_DEADLINE_MS == 30_000
    assert ROLE_DEADLINE_MS == 600_000
    assert INVOCATION_DEADLINE_MS == 14_400_000
    assert STAGE_ACTIVE_TIME_CAP_MS == 43_200_000


def test_every_event_fsyncs_file_then_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    contact = _contact()
    plan = _development_plan()
    calls: list[tuple[str, object]] = []

    def file_fsync(descriptor: int) -> None:
        calls.append(("file", descriptor))

    def directory_fsync(path: Path) -> None:
        calls.append(("directory", path))

    monkeypatch.setattr(journal_module.os, "fsync", file_fsync)
    monkeypatch.setattr(journal_module, "_fsync_directory", directory_fsync)
    journal = AcquisitionJournal(root, stage="development", contact=contact)
    journal.open_invocation()
    assert [kind for kind, _value in calls] == [
        "file",
        "directory",
        "directory",
    ]
    assert calls[1][1] == root
    assert calls[2][1] == root


def test_close_directory_fsync_failure_cannot_replay_as_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, contact, _plan, journal = _journal(tmp_path)
    journal.open_invocation()
    directory_calls = 0

    def fail_after_close_rename(_path: Path) -> None:
        nonlocal directory_calls
        directory_calls += 1
        if directory_calls == 2:
            raise SecGemmaLeanV35JournalError(
                "journal_directory_fsync_failed"
            )

    monkeypatch.setattr(
        journal_module, "_fsync_directory", fail_after_close_rename
    )
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.close_invocation(invocation_duration_ms=1)
    assert caught.value.code == "journal_append_durability_failed"
    assert (root / "00000002.json").is_file()
    assert list(root.glob(".append-failed-00000002-*.json"))

    with pytest.raises(SecGemmaLeanV35JournalError) as replay_caught:
        AcquisitionJournal(root, stage="development", contact=contact)
    assert replay_caught.value.code == "journal_append_incomplete"


def test_role_event_directory_fsync_failure_cannot_replay_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, contact, plan, journal = _journal(tmp_path)
    journal.open_invocation()
    directory_calls = 0

    def fail_after_role_rename(_path: Path) -> None:
        nonlocal directory_calls
        directory_calls += 1
        if directory_calls == 2:
            raise SecGemmaLeanV35JournalError(
                "journal_directory_fsync_failed"
            )

    monkeypatch.setattr(
        journal_module, "_fsync_directory", fail_after_role_rename
    )
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        journal.record_role_intent(
            plan.roles[0].role_id, dispatch_wait_ms=1_000
        )
    assert caught.value.code == "journal_append_durability_failed"
    assert (root / "00000002.json").is_file()
    assert list(root.glob(".append-failed-00000002-*.json"))

    with pytest.raises(SecGemmaLeanV35JournalError) as replay_caught:
        validate_detached_journal(
            root, "development", contact.fingerprint_sha256
        )
    assert replay_caught.value.code == "journal_append_incomplete"


@pytest.mark.parametrize(
    "marker_name",
    [
        ".pending-00000001-" + ("0" * 64) + ".json",
        ".append-failed-00000001-" + ("0" * 64) + ".json",
    ],
)
def test_leftover_append_marker_is_never_accepted(
    tmp_path: Path, marker_name: str
) -> None:
    root = _root(tmp_path)
    (root / marker_name).write_bytes(b"{}\n")
    contact = _contact()

    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        validate_detached_journal(
            root, "development", contact.fingerprint_sha256
        )
    assert caught.value.code == "journal_append_incomplete"


def test_root_must_be_absolute_existing_directory(tmp_path: Path) -> None:
    plan = _development_plan()
    contact = _contact()
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        AcquisitionJournal(Path("relative"), stage="development", contact=contact)
    assert caught.value.code == "journal_root_invalid"
    with pytest.raises(SecGemmaLeanV35JournalError) as caught:
        AcquisitionJournal(
            tmp_path / "missing", stage="development", contact=contact
        )
    assert caught.value.code == "journal_root_invalid"
