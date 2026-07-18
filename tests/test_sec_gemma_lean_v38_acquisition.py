from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import agent_benchmark.sec_gemma_lean_v38_acquisition as acquisition_module
import agent_benchmark.sec_gemma_lean_v38_journal as journal_module
from agent_benchmark.sec_gemma_lean_v38_acquisition import (
    DiskBackedSecAcquisition,
    GLOBAL_LEDGER_NAME,
    ROOT_NAMESPACE,
    SecGemmaLeanV38AcquisitionError,
    SharedSecDispatchLedger,
)
from agent_benchmark.sec_gemma_lean_v38_journal import PrivateSecContact
from agent_benchmark.sec_gemma_lean_v38_source import (
    SecGemmaLeanV38SourceError,
)
from agent_benchmark.sec_gemma_lean_v38_transport import (
    TRANSPORT_SCHEMA_VERSION,
    TransportResult,
    build_transport_receipt,
    derive_sec_role,
    execution_identity,
)


CONTACT = "Alder Research Compliance sec-filings@alder-research-739184.com"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash_value(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _evidence(body: dict[str, object]) -> dict[str, object]:
    return {**body, "source_evidence_sha256": _hash_value(body)}


@dataclass(frozen=True)
class _RoleOutput:
    evidence: dict[str, object]
    historical_filenames: tuple[str, ...] = ()
    decompressed_bytes: int | None = None


@dataclass(frozen=True)
class _PhaseOutput:
    evidence: dict[str, object]


@dataclass(frozen=True)
class _ReconciliationOutput:
    evidence: dict[str, object]
    complete_targets: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _StageOutput:
    stage_source_seal_sha256: str
    prior_stage_source_seal_sha256: str | None = None


@dataclass(frozen=True)
class _CompactReplayInput:
    stage: str
    checkpoint: dict[str, object]
    role_manifests: tuple[dict[str, object], ...]
    prior: _CompactReplayInput | None = None


@dataclass(frozen=True)
class _ReplayResult:
    stage_output: _StageOutput
    receipt: dict[str, object]


class _FakeSource:
    CompactReplayInput = _CompactReplayInput

    def __init__(
        self,
        *,
        historical_count: int = 0,
        complete_counts: dict[str, int] | None = None,
    ) -> None:
        self.historical_count = historical_count
        self.complete_counts = {} if complete_counts is None else complete_counts

    @staticmethod
    def _role(role_id: str, url: str, payload: bytes) -> _RoleOutput:
        return _RoleOutput(
            evidence=_evidence(
                {
                    "schema_version": "fake-role-v1",
                    "role_id": role_id,
                    "url": url,
                    "body_sha256": hashlib.sha256(payload).hexdigest(),
                    "body_bytes": len(payload),
                }
            )
        )

    def parse_main(self, stage: str, payload: bytes) -> _RoleOutput:
        del stage
        parsed = self._role(
            "submissions/main",
            "https://data.sec.gov/submissions/CIK0000320193.json",
            payload,
        )
        return _RoleOutput(
            evidence=parsed.evidence,
            historical_filenames=tuple(
                f"CIK0000320193-submissions-{index:03d}.json"
                for index in range(self.historical_count)
            ),
        )

    def parse_historical(
        self,
        stage: str,
        filename: str,
        payload: bytes,
        main_evidence: dict[str, object],
    ) -> _RoleOutput:
        del stage, main_evidence
        return self._role(
            f"submissions/historical/{filename}",
            f"https://data.sec.gov/submissions/{filename}",
            payload,
        )

    def finalize_submissions(
        self,
        stage: str,
        main_evidence: dict[str, object],
        historical_evidence: tuple[dict[str, object], ...],
        prior: object | None,
    ) -> _PhaseOutput:
        del historical_evidence, prior
        return _PhaseOutput(
            _evidence(
                {
                    "schema_version": "fake-submissions-v1",
                    "stage": stage,
                    "main_source_evidence_sha256": main_evidence[
                        "source_evidence_sha256"
                    ],
                }
            )
        )

    def parse_master(
        self, stage: str, year: int, quarter: int, payload: bytes
    ) -> _RoleOutput:
        del stage
        return self._role(
            f"master/{year}/QTR{quarter}",
            "https://www.sec.gov/Archives/edgar/full-index/"
            f"{year}/QTR{quarter}/master.gz",
            payload,
        ).__class__(
            evidence=self._role(
                f"master/{year}/QTR{quarter}",
                "https://www.sec.gov/Archives/edgar/full-index/"
                f"{year}/QTR{quarter}/master.gz",
                payload,
            ).evidence,
            decompressed_bytes=len(payload),
        )

    def reconcile_masters(
        self,
        stage: str,
        submissions_evidence: dict[str, object],
        master_evidence: tuple[dict[str, object], ...],
        prior: object | None,
    ) -> _ReconciliationOutput:
        del prior
        stage_year = {"development": "00", "intermediate": "19", "final": "24"}[
            stage
        ]
        filing_date = {
            "development": "2000-01-01",
            "intermediate": "2019-01-01",
            "final": "2024-01-01",
        }[stage]
        complete_targets = tuple(
            {
                "submissions": {
                    "accession_number": f"0000320193-{stage_year}-{index:06d}",
                    "filing_date": filing_date,
                },
                "master": {
                    "complete_submission_url": (
                        "https://www.sec.gov/Archives/edgar/data/320193/"
                        f"0000320193-{stage_year}-{index:06d}.txt"
                    )
                },
            }
            for index in range(self.complete_counts.get(stage, 0))
        )
        return _ReconciliationOutput(
            evidence=_evidence(
                {
                    "schema_version": "fake-reconciliation-v1",
                    "stage": stage,
                    "submissions_source_evidence_sha256": (
                        submissions_evidence["source_evidence_sha256"]
                    ),
                    "master_source_evidence_sha256": [
                        item["source_evidence_sha256"] for item in master_evidence
                    ],
                }
            ),
            complete_targets=complete_targets,
        )

    def parse_complete(
        self,
        stage: str,
        target: dict[str, object],
        payload: bytes,
        reconciliation_evidence: dict[str, object],
    ) -> _RoleOutput:
        del stage, reconciliation_evidence
        submissions = target["submissions"]
        master = target["master"]
        return self._role(
            f"complete/{submissions['accession_number']}",
            master["complete_submission_url"],
            payload,
        )

    def finalize_stage(
        self,
        stage: str,
        submissions_evidence: dict[str, object],
        reconciliation_evidence: dict[str, object],
        complete_evidence: tuple[dict[str, object], ...],
        prior: object | None,
    ) -> _StageOutput:
        prior_hash = (
            None if prior is None else prior.stage_source_seal_sha256
        )
        return _StageOutput(
            _hash_value(
                {
                    "stage": stage,
                    "submissions": submissions_evidence[
                        "source_evidence_sha256"
                    ],
                    "reconciliation": reconciliation_evidence[
                        "source_evidence_sha256"
                    ],
                    "complete": [
                        item["source_evidence_sha256"]
                        for item in complete_evidence
                    ],
                    "prior": prior_hash,
                }
            ),
            prior_hash,
        )

    def build_compact_role_manifest(self, **kwargs: object) -> dict[str, object]:
        output = kwargs["parse_output"]
        payload = kwargs["payload"]
        evidence = output.evidence
        assert evidence["role_id"] == kwargs["role_id"]
        assert evidence["url"] == kwargs["url"]
        assert evidence["body_sha256"] == hashlib.sha256(payload).hexdigest()
        body = {
            "schema_version": "fake-role-manifest-v1",
            "sequence": kwargs["sequence"],
            "role_id": kwargs["role_id"],
            "url": kwargs["url"],
            "blob_name": kwargs["blob_name"],
            "body_sha256": (
                "sha256:" + hashlib.sha256(payload).hexdigest()
            ),
            "body_bytes": len(payload),
            "transport_receipt_sha256": kwargs[
                "transport_receipt_sha256"
            ],
            "parse_receipt_sha256": kwargs["parse_receipt_sha256"],
            "source_evidence_sha256": evidence["source_evidence_sha256"],
        }
        return {**body, "role_manifest_sha256": _hash_value(body)}

    def build_compact_checkpoint(self, **kwargs: object) -> dict[str, object]:
        manifests = list(kwargs["role_manifests"])
        stage_output = kwargs["stage_output"]
        main_parse_receipt_sha256 = kwargs["main_parse_receipt_sha256"]
        assert main_parse_receipt_sha256 == manifests[0]["parse_receipt_sha256"]
        body = {
            "schema_version": "fake-checkpoint-v1",
            "stage": kwargs["stage"],
            "role_manifest_count": len(manifests),
            "role_manifests_sha256": _hash_value(manifests),
            "expected_stage_source_seal_sha256": (
                stage_output.stage_source_seal_sha256
            ),
            "prior_stage_source_seal_sha256": (
                stage_output.prior_stage_source_seal_sha256
            ),
            "main_parse_receipt_sha256": main_parse_receipt_sha256,
            "submissions_snapshot_receipt_sha256": kwargs[
                "submissions_snapshot_receipt_sha256"
            ],
            "reconciliation_and_prior_chain_receipt_sha256": kwargs[
                "reconciliation_and_prior_chain_receipt_sha256"
            ],
        }
        return {**body, "checkpoint_sha256": _hash_value(body)}

    def rehydrate_compact_stage(
        self,
        stage: str,
        checkpoint: dict[str, object],
        role_manifests: tuple[dict[str, object], ...],
        load_blob,
        prior: object | None = None,
    ) -> _ReplayResult:
        assert checkpoint["stage"] == stage
        assert checkpoint["role_manifest_count"] == len(role_manifests)
        assert checkpoint["role_manifests_sha256"] == _hash_value(
            list(role_manifests)
        )
        prior_hash = None
        if prior is not None:
            prior_result = self.rehydrate_compact_stage(
                prior.stage,
                prior.checkpoint,
                prior.role_manifests,
                load_blob,
                prior.prior,
            )
            prior_hash = prior_result.stage_output.stage_source_seal_sha256
        assert checkpoint["prior_stage_source_seal_sha256"] == prior_hash
        for manifest in role_manifests:
            payload = load_blob(manifest["blob_name"])
            assert len(payload) == manifest["body_bytes"]
            assert (
                hashlib.sha256(payload).hexdigest()
                == manifest["body_sha256"].removeprefix("sha256:")
            )
            del payload
        receipt = {
            "schema_version": "fake-replay-v1",
            "stage": stage,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "role_manifest_count": len(role_manifests),
            "role_manifests_sha256": checkpoint["role_manifests_sha256"],
            "stage_source_seal_sha256": checkpoint[
                "expected_stage_source_seal_sha256"
            ],
            "exact_source_seal_match": True,
            "peak_live_role_payload_count": 1,
            "main_parse_receipt_sha256": checkpoint[
                "main_parse_receipt_sha256"
            ],
            "submissions_snapshot_receipt_sha256": checkpoint[
                "submissions_snapshot_receipt_sha256"
            ],
            "reconciliation_and_prior_chain_receipt_sha256": checkpoint[
                "reconciliation_and_prior_chain_receipt_sha256"
            ],
        }
        return _ReplayResult(
            stage_output=_StageOutput(
                checkpoint["expected_stage_source_seal_sha256"],
                prior_hash,
            ),
            receipt=receipt,
        )

    def detached_replay(self, *args: object, **kwargs: object) -> dict[str, object]:
        return self.rehydrate_compact_stage(*args, **kwargs).receipt


class _SlowFakeSource(_FakeSource):
    def __init__(self, clock: _FakeClock, *, delay_seconds: float) -> None:
        super().__init__()
        self.clock = clock
        self.delay_seconds = delay_seconds

    def _delay(self, value: _RoleOutput) -> _RoleOutput:
        self.clock.advance(self.delay_seconds)
        return value

    def parse_main(self, stage: str, payload: bytes) -> _RoleOutput:
        return self._delay(super().parse_main(stage, payload))

    def parse_historical(
        self,
        stage: str,
        filename: str,
        payload: bytes,
        main_evidence: dict[str, object],
    ) -> _RoleOutput:
        return self._delay(
            super().parse_historical(
                stage, filename, payload, main_evidence
            )
        )

    def parse_master(
        self, stage: str, year: int, quarter: int, payload: bytes
    ) -> _RoleOutput:
        return self._delay(super().parse_master(stage, year, quarter, payload))

    def parse_complete(
        self,
        stage: str,
        target: dict[str, object],
        payload: bytes,
        reconciliation_evidence: dict[str, object],
    ) -> _RoleOutput:
        return self._delay(
            super().parse_complete(
                stage, target, payload, reconciliation_evidence
            )
        )


class _RejectingSource(_FakeSource):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def finalize_submissions(
        self,
        stage: str,
        main_evidence: dict[str, object],
        historical_evidence: tuple[dict[str, object], ...],
        prior: object | None,
    ) -> _PhaseOutput:
        del stage, main_evidence, historical_evidence, prior
        raise self.error


class _SourceErrorSubclass(SecGemmaLeanV38SourceError):
    pass


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0.0
        self.sleeps.append(seconds)
        self.value += seconds

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeTransport:
    def __init__(self, temporary: Path, fingerprint: str) -> None:
        self.temporary = temporary
        self.fingerprint = fingerprint
        self.calls: list[str] = []

    def safe_state(self) -> dict[str, object]:
        return {
            "temporary_directory": str(self.temporary),
            "contact_fingerprint_sha256": self.fingerprint,
        }

    def fetch(
        self,
        url: str,
        *,
        role_id: str,
        intent_event_sha256: str,
        body_limit: int,
    ) -> TransportResult:
        assert role_id not in self.calls
        self.calls.append(role_id)
        role = derive_sec_role(url)
        assert role.role_id == role_id
        assert role.body_limit_bytes == body_limit
        payload = (role_id + "\n").encode("ascii")
        path = self.temporary / f"{len(self.calls):06d}.response.tmp"
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        raw_headers = b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n"
        identity = execution_identity()
        metadata = {
            "request_url": url,
            "observed_url": url,
            "role_class": role.role_class,
            "role_id": role_id,
            "intent_event_sha256": intent_event_sha256,
            "body_limit_bytes": body_limit,
            "temporary_blob_path": str(path),
            "status_code": 200,
            "framing": "content-length",
            "declared_content_length": len(payload),
            "content_encoding": None,
            "body_bytes": len(payload),
            "body_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_headers_sha256": hashlib.sha256(raw_headers).hexdigest(),
            "raw_headers_bytes": len(raw_headers),
            "contact_fingerprint_sha256": self.fingerprint,
            "execution_identity": identity,
        }
        receipt = build_transport_receipt(metadata)
        return TransportResult(
            schema_version=TRANSPORT_SCHEMA_VERSION,
            request_url=url,
            observed_url=url,
            role_class=role.role_class,
            role_id=role_id,
            intent_event_sha256=intent_event_sha256,
            body_limit_bytes=body_limit,
            temporary_blob_path=path,
            status_code=200,
            framing="content-length",
            declared_content_length=len(payload),
            content_encoding=None,
            body_bytes=len(payload),
            body_sha256=hashlib.sha256(payload).hexdigest(),
            raw_headers_sha256=hashlib.sha256(raw_headers).hexdigest(),
            raw_headers_bytes=len(raw_headers),
            contact_fingerprint_sha256=self.fingerprint,
            execution_identity=identity,
            transport_receipt=receipt,
            transport_receipt_sha256=receipt["transport_receipt_sha256"],
            elapsed_milliseconds=1,
        )


class _SlowFakeTransport(_FakeTransport):
    def __init__(
        self,
        temporary: Path,
        fingerprint: str,
        clock: _FakeClock,
        *,
        advance_seconds: float = 500.0,
    ) -> None:
        super().__init__(temporary, fingerprint)
        self.clock = clock
        self.advance_seconds = advance_seconds

    def fetch(self, *args: object, **kwargs: object) -> TransportResult:
        self.clock.advance(self.advance_seconds)
        return super().fetch(*args, **kwargs)


class _CrashOpenTransport(_FakeTransport):
    def fetch(
        self,
        url: str,
        *,
        role_id: str,
        intent_event_sha256: str,
        body_limit: int,
    ) -> TransportResult:
        del url, intent_event_sha256, body_limit
        self.calls.append(role_id)
        path = self.temporary / "crash-open.response.tmp"
        with path.open("xb") as handle:
            handle.write(b"partial")
            handle.flush()
            os.fsync(handle.fileno())
        raise KeyboardInterrupt


def _roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("data/\n", encoding="utf-8")
    private = repo / "data" / ROOT_NAMESPACE
    public = repo / "e" / ROOT_NAMESPACE
    private.mkdir(parents=True)
    public.mkdir(parents=True)
    monkeypatch.setattr(acquisition_module, "_repository_root", lambda: repo)
    return repo, private, public


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("allowed", "master_boundary_incomplete"),
        ("code_shaped", "source_rejected"),
        ("arbitrary", "source_rejected"),
        ("secret_bearing", "source_rejected"),
        ("wrong_type", "source_rejected"),
        ("subclass", "source_rejected"),
    ),
)
def test_source_diagnostic_is_exact_safe_and_control_flow_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_code: str,
) -> None:
    assert journal_module.SOURCE_DIAGNOSTIC_CODES == frozenset(
        {"master_boundary_incomplete"}
    )
    messages = {
        "allowed": "master_boundary_incomplete",
        "code_shaped": "role_deadline_exceeded",
        "arbitrary": "unexpected parser sentence with details",
        "secret_bearing": (
            f"secret={CONTACT} url=https://example.invalid/private "
            "<SEC-DOCUMENT>hostile</SEC-DOCUMENT>"
        ),
        "wrong_type": "master_boundary_incomplete",
        "subclass": "master_boundary_incomplete",
    }
    message = messages[case]
    if case == "wrong_type":
        error: BaseException = RuntimeError(message)
    elif case == "subclass":
        error = _SourceErrorSubclass(message)
    else:
        error = SecGemmaLeanV38SourceError(message)

    _repo, private, public = _roots(tmp_path, monkeypatch)
    fingerprint = PrivateSecContact(CONTACT).fingerprint_sha256
    clock = _FakeClock()
    transports: list[_FakeTransport] = []

    def factory(temporary: Path) -> _FakeTransport:
        value = _FakeTransport(temporary, fingerprint)
        transports.append(value)
        return value

    ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=private / GLOBAL_LEDGER_NAME,
    )
    acquisition = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=_RejectingSource(error),
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition.run()
    assert caught.value.code == expected_code
    assert str(caught.value) == expected_code
    assert transports[0].calls == ["submissions/main"]
    state = acquisition.journal_state
    assert state.terminal_status == "rejected"
    assert state.terminal_code == expected_code
    assert state.role_seals == 1
    assert state.lifetime_intents == 1
    assert state.http_200_responses == 1
    assert list((private / "development" / "phase_receipts").iterdir()) == []

    detached = journal_module.validate_detached_journal(
        private / "development" / "journal",
        "development",
        fingerprint,
    )
    assert detached.terminal_status == "rejected"
    assert detached.terminal_code == expected_code
    disk_bytes = b"".join(
        path.read_bytes()
        for root in (private, public)
        for path in root.rglob("*")
        if path.is_file()
    )
    assert CONTACT.encode("utf-8") not in disk_bytes
    if expected_code == "source_rejected":
        assert message.encode("utf-8") not in disk_bytes

    restarted = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=_RejectingSource(
            AssertionError("terminal rejection must not call source")
        ),
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    )
    records = restarted._load_role_records()
    assert len(records) == 1
    assert records[0].manifest["body_sha256"].startswith("sha256:")
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        restarted.run()
    assert caught.value.code == expected_code
    assert transports[1].calls == []


@pytest.mark.parametrize(
    ("variant", "expected_code"),
    (
        ("canonical", "master_boundary_incomplete"),
        ("bare", "artifact_inventory_invalid"),
        ("uppercase", "artifact_inventory_invalid"),
        ("double_tagged", "artifact_inventory_invalid"),
        ("malformed", "artifact_inventory_invalid"),
        ("mismatch", "artifact_inventory_invalid"),
    ),
)
def test_terminal_inventory_requires_exact_tagged_manifest_body_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expected_code: str,
) -> None:
    _repo, private, public = _roots(tmp_path, monkeypatch)
    fingerprint = PrivateSecContact(CONTACT).fingerprint_sha256
    clock = _FakeClock()
    transports: list[_FakeTransport] = []

    def factory(temporary: Path) -> _FakeTransport:
        value = _FakeTransport(temporary, fingerprint)
        transports.append(value)
        return value

    ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=private / GLOBAL_LEDGER_NAME,
    )
    acquisition = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=_RejectingSource(
            SecGemmaLeanV38SourceError("master_boundary_incomplete")
        ),
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition.run()
    assert caught.value.code == "master_boundary_incomplete"
    assert transports[0].calls == ["submissions/main"]

    manifest_path = next(
        (private / "development" / "manifests").glob("*.json")
    )
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    canonical = manifest["body_sha256"]
    assert acquisition_module._TAGGED_SHA256_RE.fullmatch(canonical)
    digest = canonical.removeprefix("sha256:")
    replacements = {
        "bare": digest,
        "uppercase": "sha256:" + digest.upper(),
        "double_tagged": "sha256:" + canonical,
        "malformed": "sha256:" + digest[:-1],
        "mismatch": "sha256:" + ("0" * 64),
    }
    if variant != "canonical":
        manifest["body_sha256"] = replacements[variant]
        unsigned = dict(manifest)
        unsigned.pop("role_manifest_sha256")
        manifest["role_manifest_sha256"] = _hash_value(unsigned)
        changed_path = manifest_path.with_name(
            f"000000-{manifest['role_manifest_sha256']}.json"
        )
        manifest_path.write_bytes(_canonical(manifest))
        manifest_path.rename(changed_path)

    restarted = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=_RejectingSource(
            AssertionError("terminal rejection must not resume")
        ),
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        restarted.run()
    assert caught.value.code == expected_code
    assert transports[1].calls == []


def test_dispatch_measures_prior_gap_and_persists_cross_instance(
    tmp_path: Path,
) -> None:
    clock = _FakeClock()
    ledger_path = tmp_path / "dispatch.json"
    ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=ledger_path,
    )
    assert not ledger_path.exists()
    waits: list[int] = []
    first = ledger.dispatch(
        purpose="stage-a",
        first_in_invocation=True,
        record_intent=lambda wait: waits.append(wait) or ("0" * 64),
        dispatch_callback=lambda _intent: "first",
    )
    assert first.waited_milliseconds == 1_000
    clock.advance(1.0)
    second = ledger.dispatch(
        purpose="stage-a",
        first_in_invocation=False,
        record_intent=lambda wait: waits.append(wait) or ("1" * 64),
        dispatch_callback=lambda _intent: "second",
    )
    assert second.waited_milliseconds == 1_000
    assert waits == [1_000, 1_000]

    other = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=ledger_path,
    )
    third = other.dispatch(
        purpose="different-purpose",
        first_in_invocation=True,
        record_intent=lambda wait: ("2" * 64),
        dispatch_callback=lambda _intent: "third",
    )
    assert third.waited_milliseconds == 1_000
    assert ledger_path.is_file()


def test_file_lock_conflict_is_nonblocking_and_body_oserror_is_preserved(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "dispatch.json"
    clock = _FakeClock()
    ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=ledger_path,
        lock_timeout_seconds=0.0,
    )
    with acquisition_module._locked_file(
        ledger_path.with_suffix(".lock"), busy_code="outer_conflict"
    ):
        with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
            ledger.dispatch(
                purpose="conflict",
                first_in_invocation=True,
                record_intent=lambda _wait: "0" * 64,
                dispatch_callback=lambda _intent: None,
            )
        assert caught.value.code == "dispatch_lock_conflict"

    with pytest.raises(OSError, match="body failure"):
        with acquisition_module._locked_file(
            tmp_path / "body.lock", busy_code="conflict"
        ):
            raise OSError("body failure")


def test_concurrent_ledgers_serialize_callbacks_and_keep_half_second_starts(
    tmp_path: Path,
) -> None:
    clock = _FakeClock()
    ledger_path = tmp_path / "shared-dispatch.json"
    ledgers = [
        SharedSecDispatchLedger(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            ledger_path=ledger_path,
        )
        for _index in range(2)
    ]
    barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    starts: list[float] = []
    active = 0
    maximum_active = 0
    errors: list[Exception] = []

    def worker(index: int) -> None:
        nonlocal active, maximum_active
        try:
            barrier.wait(timeout=5.0)

            def callback(_intent: str) -> str:
                nonlocal active, maximum_active
                with state_lock:
                    starts.append(clock.monotonic())
                    active += 1
                    maximum_active = max(maximum_active, active)
                with state_lock:
                    active -= 1
                return "ok"

            ledgers[index].dispatch(
                purpose=f"concurrent-{index}",
                first_in_invocation=True,
                record_intent=lambda _wait: str(index) * 64,
                dispatch_callback=callback,
            )
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert maximum_active == 1
    assert len(starts) == 2
    starts.sort()
    assert starts[1] - starts[0] >= 0.5


def test_release_failure_is_not_silently_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        import msvcrt

        real_locking = msvcrt.locking

        def locking(descriptor: int, mode: int, count: int) -> None:
            if mode == msvcrt.LK_UNLCK:
                raise OSError("unlock")
            real_locking(descriptor, mode, count)

        monkeypatch.setattr(msvcrt, "locking", locking)
    else:
        import fcntl

        real_flock = fcntl.flock

        def flock(descriptor: int, operation: int) -> None:
            if operation == fcntl.LOCK_UN:
                raise OSError("unlock")
            real_flock(descriptor, operation)

        monkeypatch.setattr(fcntl, "flock", flock)
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        with acquisition_module._locked_file(
            tmp_path / "release.lock", busy_code="conflict"
        ):
            pass
    assert caught.value.code == "run_lock_release_failed"


def test_import_is_side_effect_free_and_exact_names_are_exported() -> None:
    assert ROOT_NAMESPACE == "aapl_sec_gemma_lean_evidence_v3_8"
    assert GLOBAL_LEDGER_NAME == "global_dispatch.json"
    repository = Path(acquisition_module.__file__).resolve().parents[1]
    private_root = repository / "data" / ROOT_NAMESPACE

    def snapshot() -> tuple[object, ...]:
        if not private_root.exists():
            return ("missing",)
        values: list[object] = []
        for path in sorted(private_root.iterdir(), key=lambda item: item.name):
            details = path.lstat()
            digest = None
            if path.is_file() and details.st_size <= 1024 * 1024:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            values.append((path.name, details.st_mode, details.st_size, digest))
        return tuple(values)

    before = snapshot()
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import agent_benchmark.sec_gemma_lean_v38_acquisition",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert snapshot() == before


def test_exact_root_and_authorized_dispatch_ledger_namespace_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, private, public = _roots(tmp_path, monkeypatch)
    fingerprint = PrivateSecContact(CONTACT).fingerprint_sha256
    source = _FakeSource()

    def factory(temporary: Path) -> _FakeTransport:
        return _FakeTransport(temporary, fingerprint)

    wrong_ledger = SharedSecDispatchLedger(
        ledger_path=private / "different-ledger.json"
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        DiskBackedSecAcquisition(
            stage="development",
            private_root=private,
            public_root=public,
            private_contact=CONTACT,
            transport_factory=factory,
            source_adapter=source,
            dispatch_ledger=wrong_ledger,
        )
    assert caught.value.code == "dispatch_ledger_namespace_invalid"
    assert not (private / "development").exists()

    wrong_private = repo / "data" / "wrong"
    wrong_private.mkdir()
    wrong_public = repo / "e" / "wrong"
    wrong_public.mkdir()
    matching_wrong_ledger = SharedSecDispatchLedger(
        ledger_path=wrong_private / GLOBAL_LEDGER_NAME
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        DiskBackedSecAcquisition(
            stage="development",
            private_root=wrong_private,
            public_root=wrong_public,
            private_contact=CONTACT,
            transport_factory=factory,
            source_adapter=source,
            dispatch_ledger=matching_wrong_ledger,
        )
    assert caught.value.code == "root_namespace_invalid"
    assert not (wrong_private / "development").exists()


def test_atomic_blob_rename_is_durable_without_fsync_test_patches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, private, public = _roots(tmp_path, monkeypatch)
    fingerprint = PrivateSecContact(CONTACT).fingerprint_sha256
    holder: list[_FakeTransport] = []

    def factory(temporary: Path) -> _FakeTransport:
        value = _FakeTransport(temporary, fingerprint)
        holder.append(value)
        return value

    acquisition = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=_FakeSource(),
        dispatch_ledger=SharedSecDispatchLedger(
            ledger_path=private / GLOBAL_LEDGER_NAME
        ),
    )
    url = "https://data.sec.gov/submissions/CIK0000320193.json"
    role = derive_sec_role(url)
    result = holder[0].fetch(
        url,
        role_id=role.role_id,
        intent_event_sha256="0" * 64,
        body_limit=role.body_limit_bytes,
    )
    blob_name, blob_path = acquisition._promote_blob(result, sequence=0)
    assert blob_path.name == blob_name
    assert blob_path.read_bytes() == (role.role_id + "\n").encode("ascii")
    assert list((private / "development" / "temporary").iterdir()) == []


def test_production_entrypoint_wires_only_development_and_later_stages_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, private, public = _roots(tmp_path, monkeypatch)
    gate = acquisition_module._ProductionDevelopmentGate(
        repository_root=repo,
        private_root=private,
        public_root=public,
        authorized_head="a" * 40,
        preflight_sha256="b" * 64,
        private_contact=CONTACT,
    )
    ledger = SharedSecDispatchLedger(
        ledger_path=private / GLOBAL_LEDGER_NAME
    )
    captured: dict[str, object] = {}
    sentinel = object()

    class StubAcquisition:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self) -> object:
            return sentinel

    result = acquisition_module._run_development_with_gate(
        gate,
        transport_factory=lambda _temporary: object(),
        source_adapter=_FakeSource(),
        dispatch_ledger=ledger,
        acquisition_factory=StubAcquisition,
    )
    assert result is sentinel
    assert captured["stage"] == "development"
    assert captured["private_root"] == private
    assert captured["public_root"] == public
    assert captured["prior"] is None
    assert captured["dispatch_ledger"] is ledger
    assert CONTACT not in repr(gate)

    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition_module.run_production_acquisition("intermediate", repo)
    assert (
        caught.value.code
        == "later_stage_scientific_authorization_unavailable"
    )
    assert acquisition_module.main(
        ["final", "--repo-root", str(repo)]
    ) == 1
    printed = capsys.readouterr().out
    assert "later_stage_scientific_authorization_unavailable" in printed
    assert CONTACT not in printed


def test_production_development_binds_closure_then_rechecks_and_wires_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, private, public = _roots(tmp_path, monkeypatch)
    gate = acquisition_module._ProductionDevelopmentGate(
        repository_root=repo,
        private_root=private,
        public_root=public,
        authorized_head="a" * 40,
        preflight_sha256="b" * 64,
        private_contact=CONTACT,
    )
    source_adapter = object()
    transport_factory = object()
    sentinel = object()
    events: list[str] = []
    captured: dict[str, object] = {}

    def verify_gate(root: Path):
        assert root == repo
        events.append("strict_gate")
        return gate

    def load_closure() -> dict[str, object]:
        events.append("imports")
        return {"agent_benchmark.sec_gemma_lean_v38_source": source_adapter}

    def bind_closure(
        root: Path,
        head: str,
        *,
        loaded_modules: object,
    ) -> dict[str, str]:
        assert root == repo
        assert head == gate.authorized_head
        assert loaded_modules == {
            "agent_benchmark.sec_gemma_lean_v38_source": source_adapter
        }
        events.append("closure_bound")
        return {}

    def git_text(_root: Path, *arguments: str) -> str:
        events.append("git_text:" + " ".join(arguments))
        if arguments == ("branch", "--show-current"):
            return acquisition_module.PRODUCTION_BRANCH
        if arguments == (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            return acquisition_module.PRODUCTION_UPSTREAM
        if arguments in {
            ("rev-parse", "HEAD"),
            ("rev-parse", "@{upstream}"),
        }:
            return gate.authorized_head
        raise AssertionError(arguments)

    def git(_root: Path, *arguments: str) -> bytes:
        assert arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        events.append("clean_status")
        return b""

    def make_transport(contact: str) -> object:
        assert contact == CONTACT
        events.append("transport_factory")
        return transport_factory

    def run_with_gate(value: object, **kwargs: object) -> object:
        assert value is gate
        events.append("runner")
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        acquisition_module, "_verify_production_development_gate", verify_gate
    )
    monkeypatch.setattr(
        acquisition_module, "_load_production_module_closure", load_closure
    )
    monkeypatch.setattr(
        acquisition_module, "_verify_production_module_closure", bind_closure
    )
    monkeypatch.setattr(acquisition_module, "_production_git_text", git_text)
    monkeypatch.setattr(acquisition_module, "_production_git", git)
    monkeypatch.setattr(
        acquisition_module, "strict_transport_factory", make_transport
    )
    monkeypatch.setattr(
        acquisition_module, "_run_development_with_gate", run_with_gate
    )

    assert acquisition_module.run_production_acquisition(
        "development", repo
    ) is sentinel
    assert events[:3] == ["strict_gate", "imports", "closure_bound"]
    assert events[-3:] == ["clean_status", "transport_factory", "runner"]
    assert captured["transport_factory"] is transport_factory
    assert captured["source_adapter"] is source_adapter
    ledger = captured["dispatch_ledger"]
    assert isinstance(ledger, SharedSecDispatchLedger)
    assert ledger.ledger_path == private / GLOBAL_LEDGER_NAME


def test_production_module_closure_binds_loaded_paths_and_exact_head_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    loaded: dict[str, object] = {}
    committed: dict[str, bytes] = {}
    for index, relative in enumerate(
        acquisition_module.PRODUCTION_MODULE_CLOSURE
    ):
        path = repo / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"# fixed production source {index}\n".encode("ascii")
        path.write_bytes(payload)
        module_name = acquisition_module._production_module_name(relative)
        loaded[module_name] = SimpleNamespace(__file__=str(path))
        committed[relative] = payload

    monkeypatch.setattr(acquisition_module, "_repository_root", lambda: repo)
    reads: list[str] = []

    def read_blob(_root: Path, _head: str, relative: str) -> bytes:
        reads.append(relative)
        return committed[relative]

    hashes = acquisition_module._verify_production_module_closure(
        repo,
        "a" * 40,
        loaded_modules=loaded,
        blob_reader=read_blob,
    )
    assert set(hashes) == set(acquisition_module.PRODUCTION_MODULE_CLOSURE)
    assert reads == list(acquisition_module.PRODUCTION_MODULE_CLOSURE)

    first = acquisition_module.PRODUCTION_MODULE_CLOSURE[0]
    first_module = acquisition_module._production_module_name(first)
    outside = tmp_path / "outside.py"
    outside.write_bytes(committed[first])
    loaded[first_module] = SimpleNamespace(__file__=str(outside))
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition_module._verify_production_module_closure(
            repo,
            "a" * 40,
            loaded_modules=loaded,
            blob_reader=read_blob,
        )
    assert caught.value.code == "production_execution_closure_invalid"

    first_path = repo / Path(first)
    loaded[first_module] = SimpleNamespace(__file__=str(first_path))
    first_path.write_bytes(committed[first] + b"x")
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition_module._verify_production_module_closure(
            repo,
            "a" * 40,
            loaded_modules=loaded,
            blob_reader=read_blob,
        )
    assert caught.value.code == "production_execution_closure_invalid"


def test_production_git_uses_absolute_executable_and_minimal_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_binary = tmp_path / "git.exe"
    git_binary.write_bytes(b"not executed")
    git_binary.chmod(0o700)
    monkeypatch.setattr(
        acquisition_module.shutil, "which", lambda _name: str(git_binary)
    )
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "hostile-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "hostile-worktree"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "hostile-config"))
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object):
        captured["command"] = command
        captured["environment"] = kwargs.get("env")
        return subprocess.CompletedProcess(
            command, 0, stdout=b"expected\n", stderr=b""
        )

    monkeypatch.setattr(acquisition_module.subprocess, "run", fake_run)
    assert acquisition_module._production_git(
        tmp_path, "rev-parse", "HEAD"
    ) == b"expected\n"
    command = captured["command"]
    assert isinstance(command, list)
    assert Path(command[0]) == git_binary.resolve(strict=True)
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["LC_ALL"] == "C"
    assert "GIT_DIR" not in environment
    assert "GIT_WORK_TREE" not in environment
    assert "PATH" not in environment

    linked_git = tmp_path / "git-link.exe"
    try:
        linked_git.symlink_to(git_binary)
    except OSError:
        return
    monkeypatch.setattr(
        acquisition_module.shutil, "which", lambda _name: str(linked_git)
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition_module._production_git(tmp_path, "rev-parse", "HEAD")
    assert caught.value.code == "production_git_check_unavailable"


def test_production_gate_rejects_minimal_forged_preflight_via_strict_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_benchmark import sec_gemma_lean_v38_preflight as preflight

    repo, _private, public = _roots(tmp_path, monkeypatch)
    (repo / ".git").mkdir()
    minimal = {
        "schema_version": acquisition_module.PRODUCTION_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "preflight_sha256": "0" * 64,
    }
    (public / "PREFLIGHT.json").write_text(
        json.dumps(minimal, sort_keys=True) + "\n", encoding="ascii"
    )
    seen: list[Path] = []

    def reject_forgery(root: Path) -> dict[str, object]:
        seen.append(root)
        assert json.loads((public / "PREFLIGHT.json").read_text("ascii")) == minimal
        raise preflight.SecGemmaLeanV38PreflightError(
            "committed_preflight_invalid"
        )

    monkeypatch.setattr(
        preflight,
        "validate_committed_preflight_for_acquisition",
        reject_forgery,
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition_module._verify_production_development_gate(repo)
    assert caught.value.code == "production_preflight_invalid"
    assert seen == [repo]


def test_production_gate_accepts_strict_result_and_rechecks_private_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_benchmark import sec_gemma_lean_v38_preflight as preflight

    repo, private, public = _roots(tmp_path, monkeypatch)
    (repo / ".git").mkdir()
    (repo / "data" / "local_config.json").write_text(
        json.dumps({"secrets": {"sec_user_agent": CONTACT}}, indent=2),
        encoding="utf-8",
    )
    head = "d" * 40
    contact_sha256 = (
        "sha256:" + PrivateSecContact(CONTACT).fingerprint_sha256
    )
    validation: dict[str, object] = {
        "schema_version": (
            "aapl-sec-gemma-lean-evidence-v3-8-acquisition-gate-v1"
        ),
        "validated": True,
        "read_only_validation": True,
        "evidence_commit": head,
        "evidence_tree": "e" * 40,
        "implementation_commit": "c" * 40,
        "implementation_tree": "b" * 40,
        "preflight_sha256": "a" * 64,
        "implementation_delta_manifest_sha256": "9" * 64,
        "contact_sha256": contact_sha256,
        "durable_probe_state": {
            "exact_request_sequence_verified": True,
        },
        "artifact_only_evidence_commit": True,
        "raw_artifact_equals_committed_blob": True,
        "current_branch_upstream_clean": True,
    }
    validator_calls: list[Path] = []

    def validate(root: Path) -> dict[str, object]:
        validator_calls.append(root)
        return dict(validation)

    monkeypatch.setattr(
        preflight,
        "validate_committed_preflight_for_acquisition",
        validate,
    )
    monkeypatch.setattr(
        acquisition_module,
        "_production_git_text",
        lambda root, *args: str(root)
        if args == ("rev-parse", "--show-toplevel")
        else (_ for _ in ()).throw(AssertionError(args)),
    )
    git_calls: list[tuple[str, ...]] = []

    def safe_git(_root: Path, *arguments: str) -> bytes:
        git_calls.append(arguments)
        if arguments == (
            "ls-files",
            "--",
            acquisition_module.PRIVATE_CONFIG_PATH,
        ):
            return b""
        if arguments == (
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            acquisition_module.PRIVATE_CONFIG_PATH,
        ):
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr(acquisition_module, "_production_git", safe_git)
    gate = acquisition_module._verify_production_development_gate(repo)
    assert gate.private_root == private
    assert gate.public_root == public
    assert gate.authorized_head == head
    assert gate.preflight_sha256 == "a" * 64
    assert validator_calls == [repo]
    assert len(git_calls) == 2
    assert CONTACT not in repr(gate)

    validation["contact_sha256"] = "sha256:" + "f" * 64
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition_module._verify_production_development_gate(repo)
    assert caught.value.code == "private_contact_changed"
    assert CONTACT not in repr(caught.value)


def test_ledger_updates_are_atomic_and_lock_state_paths_do_not_follow_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock()
    ledger_path = tmp_path / "atomic-dispatch.json"
    real_replace = acquisition_module.os.replace
    replacements: list[tuple[Path, Path]] = []

    def checked_replace(source: object, target: object) -> None:
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(acquisition_module.os, "replace", checked_replace)
    ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=ledger_path,
    )
    ledger.dispatch(
        purpose="atomic",
        first_in_invocation=True,
        record_intent=lambda _wait: "0" * 64,
        dispatch_callback=lambda _intent: "ok",
    )
    assert len(replacements) == 2
    assert all(target == ledger_path for _source, target in replacements)
    assert all(".pending-" in source.name for source, _target in replacements)
    assert not any(
        path.name.startswith(f".{ledger_path.name}.pending-")
        for path in tmp_path.iterdir()
    )

    hard_target = tmp_path / "hard-target"
    hard_target.write_bytes(b"\0")
    unsafe_lock = tmp_path / "hard-linked.lock"
    os.link(hard_target, unsafe_lock)
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        with acquisition_module._locked_file(
            unsafe_lock, busy_code="busy"
        ):
            pass
    assert caught.value.code == "run_lock_invalid"

    state_target = tmp_path / "state-target.json"
    state_target.write_bytes(ledger_path.read_bytes())
    unsafe_state = tmp_path / "unsafe-state.json"
    os.link(state_target, unsafe_state)
    unsafe_ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=unsafe_state,
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        unsafe_ledger.dispatch(
            purpose="unsafe",
            first_in_invocation=True,
            record_intent=lambda _wait: "1" * 64,
            dispatch_callback=lambda _intent: pytest.fail("must not dispatch"),
        )
    assert caught.value.code == "dispatch_ledger_invalid"

    symlink_target = tmp_path / "symlink-target"
    symlink_target.write_bytes(b"\0")
    symlink_lock = tmp_path / "symlink.lock"
    try:
        symlink_lock.symlink_to(symlink_target)
    except OSError:
        pass
    else:
        with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
            with acquisition_module._locked_file(
                symlink_lock, busy_code="busy"
            ):
                pass
        assert caught.value.code == "run_lock_invalid"

    pending_ledger_path = tmp_path / "pending-state.json"
    pending_marker = tmp_path / (
        f".{pending_ledger_path.name}.pending-hostile"
    )
    pending_marker.write_bytes(b"partial")
    pending_ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=pending_ledger_path,
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        pending_ledger.dispatch(
            purpose="pending",
            first_in_invocation=True,
            record_intent=lambda _wait: "2" * 64,
            dispatch_callback=lambda _intent: pytest.fail("must not dispatch"),
        )
    assert caught.value.code == "dispatch_indeterminate"


def test_actual_callback_overrun_is_journaled_as_hard_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, private, public = _roots(tmp_path, monkeypatch)
    fingerprint = PrivateSecContact(CONTACT).fingerprint_sha256
    clock = _FakeClock()
    transport: _SlowFakeTransport | None = None

    def factory(temporary: Path) -> _SlowFakeTransport:
        nonlocal transport
        transport = _SlowFakeTransport(
            temporary,
            fingerprint,
            clock,
            advance_seconds=31.0,
        )
        return transport

    acquisition = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=_FakeSource(),
        dispatch_ledger=SharedSecDispatchLedger(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            ledger_path=private / GLOBAL_LEDGER_NAME,
        ),
        monotonic=clock.monotonic,
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition.run()
    assert caught.value.code == "hard_timeout"
    assert str(caught.value) == "hard_timeout"
    assert transport is not None and len(transport.calls) == 1
    state = acquisition.journal_state
    assert state.terminal_status == "rejected"
    assert state.terminal_code == "hard_timeout"
    assert state.lifetime_intents == 1
    assert state.http_200_responses == 0
    assert state.role_seals == 0
    assert CONTACT not in repr(caught.value)


def test_pre_dispatch_deadline_gates_roll_invocations_and_stop_before_stage_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, private, public = _roots(tmp_path, monkeypatch)
    fingerprint = PrivateSecContact(CONTACT).fingerprint_sha256
    clock = _FakeClock()
    transports: list[_FakeTransport] = []
    journal_root = private / "development" / "journal"
    cached_events: list[dict[str, object]] = []
    real_read_events = journal_module._read_events
    real_rename = journal_module.os.rename

    def cached_read_events(root: Path):
        if Path(root) == journal_root:
            return list(cached_events)
        return real_read_events(root)

    def capturing_rename(source: object, target: object) -> None:
        real_rename(source, target)
        target_path = Path(target)
        if target_path.parent == journal_root and target_path.name.endswith(".json"):
            cached_events.append(
                json.loads(target_path.read_text(encoding="ascii"))
            )

    monkeypatch.setattr(journal_module.os, "fsync", lambda _descriptor: None)
    monkeypatch.setattr(journal_module, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(journal_module, "_read_events", cached_read_events)
    monkeypatch.setattr(journal_module.os, "rename", capturing_rename)

    def factory(temporary: Path) -> _FakeTransport:
        value = _FakeTransport(temporary, fingerprint)
        transports.append(value)
        return value

    acquisition = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=_SlowFakeSource(clock, delay_seconds=500.0),
        dispatch_ledger=SharedSecDispatchLedger(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            ledger_path=private / GLOBAL_LEDGER_NAME,
        ),
        monotonic=clock.monotonic,
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        acquisition.run()
    assert caught.value.code == "cumulative_source_deadline_exceeded"
    state = acquisition.journal_state
    assert state.terminal_status == "rejected"
    assert state.terminal_code == "cumulative_source_deadline_exceeded"
    assert state.role_seals == len(transports[0].calls)
    assert 1 < len(transports[0].calls) < 101
    assert state.cumulative_active_ms <= journal_module.STAGE_ACTIVE_TIME_CAP_MS
    invocation_closes = [
        event
        for event in cached_events
        if event["event_type"] == "invocation_clean_close"
    ]
    assert len(invocation_closes) >= 2
    assert all(
        event["payload"]["invocation_duration_ms"]
        <= journal_module.INVOCATION_DEADLINE_MS
        for event in invocation_closes
    )


def test_crash_open_intent_and_temporary_blob_never_authorize_resend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, private, public = _roots(tmp_path, monkeypatch)
    fingerprint = PrivateSecContact(CONTACT).fingerprint_sha256
    clock = _FakeClock()
    created: list[_CrashOpenTransport | _FakeTransport] = []

    def crash_factory(temporary: Path) -> _CrashOpenTransport:
        value = _CrashOpenTransport(temporary, fingerprint)
        created.append(value)
        return value

    ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=private / GLOBAL_LEDGER_NAME,
    )
    acquisition = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=crash_factory,
        source_adapter=_FakeSource(),
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    )
    with pytest.raises(KeyboardInterrupt):
        acquisition.run()
    assert len(created[0].calls) == 1
    temporary = private / "development" / "temporary"
    crash_files = list(temporary.iterdir())
    assert len(crash_files) == 1
    crash_files[0].unlink()

    def safe_factory(temporary_directory: Path) -> _FakeTransport:
        value = _FakeTransport(temporary_directory, fingerprint)
        created.append(value)
        return value

    restarted = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=safe_factory,
        source_adapter=_FakeSource(),
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    )
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        restarted.run()
    assert caught.value.code in {
        "open_intent",
        "unclosed_invocation",
        "unclean_open_intent",
    }
    assert len(created) == 2
    assert created[1].calls == []


def test_complete_fake_acquisition_restarts_without_resend_and_rejects_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, private, public = _roots(tmp_path, monkeypatch)
    fingerprint = PrivateSecContact(CONTACT).fingerprint_sha256
    clock = _FakeClock()
    ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=private / GLOBAL_LEDGER_NAME,
    )
    transports: list[_FakeTransport] = []

    def factory(temporary: Path) -> _FakeTransport:
        value = _FakeTransport(temporary, fingerprint)
        transports.append(value)
        return value

    journal_root = private / "development" / "journal"
    cached_events: list[dict[str, object]] = []
    promotions: list[tuple[Path, Path]] = []
    real_read_events = journal_module._read_events
    real_rename = journal_module.os.rename

    def cached_read_events(root: Path):
        if Path(root) == journal_root:
            return list(cached_events)
        return real_read_events(root)

    def capturing_rename(source: object, target: object) -> None:
        real_rename(source, target)
        source_path = Path(source)
        target_path = Path(target)
        if target_path.parent == private / "development" / "blobs":
            promotions.append((source_path, target_path))
        if target_path.parent == journal_root and target_path.name.endswith(".json"):
            cached_events.append(
                json.loads(target_path.read_text(encoding="ascii"))
            )

    monkeypatch.setattr(journal_module.os, "fsync", lambda _descriptor: None)
    monkeypatch.setattr(journal_module, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(journal_module, "_read_events", cached_read_events)
    monkeypatch.setattr(journal_module.os, "rename", capturing_rename)

    acquisition = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=_FakeSource(),
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    )
    result = acquisition.run()
    assert len(transports) == 1
    assert len(transports[0].calls) == 101
    assert len(set(transports[0].calls)) == 101
    assert len(promotions) == 101
    assert all(
        source.parent == private / "development" / "temporary"
        and target.parent == private / "development" / "blobs"
        for source, target in promotions
    )
    assert result.accounting_complete
    assert result.source_authoritative
    assert result.journal_state.formula_requests == 101
    assert result.journal_state.lifetime_intents == 101
    assert result.journal_state.http_200_responses == 101
    assert result.journal_state.role_seals == 101
    assert result.journal_state.accounting_complete
    assert not result.journal_state.source_authoritative
    assert result.checkpoint_path.parent == private / "development"
    assert result.public_authority_path.parent == public / "development"
    assert hashlib.sha256(result.checkpoint_path.read_bytes()).hexdigest() == (
        result.checkpoint_file_sha256
    )
    assert not (public / "development" / "journal").exists()
    assert len(list((public / "development").iterdir())) == 1
    contact_bytes = CONTACT.encode("utf-8")
    assert all(
        contact_bytes not in path.read_bytes()
        for root in (private, public)
        for path in root.rglob("*")
        if path.is_file()
    )

    monkeypatch.setattr(journal_module, "_read_events", real_read_events)
    monkeypatch.setattr(journal_module.os, "rename", real_rename)
    restarted = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=_FakeSource(),
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    )
    replayed = restarted.run()
    assert replayed.checkpoint_file_sha256 == result.checkpoint_file_sha256
    assert len(transports) == 2
    assert transports[1].calls == []

    with acquisition_module._locked_file(
        private / "development" / ".run.lock",
        busy_code="outer",
    ):
        with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
            restarted.run()
        assert caught.value.code == "active_run_conflict"

    transport_path = sorted(
        (private / "development" / "transport_receipts").glob("*.json")
    )[0]
    original_path = transport_path
    original_bytes = original_path.read_bytes()
    transport_receipt = json.loads(original_bytes.decode("ascii"))
    transport_receipt["raw_response_headers"]["sha256"] = "f" * 64
    unsigned_transport = dict(transport_receipt)
    unsigned_transport.pop("transport_receipt_sha256")
    transport_receipt["transport_receipt_sha256"] = _hash_value(
        unsigned_transport
    )
    tampered_path = original_path.with_name(
        "000000-" + transport_receipt["transport_receipt_sha256"] + ".json"
    )
    original_path.write_bytes(_canonical(transport_receipt))
    original_path.rename(tampered_path)
    with pytest.raises(SecGemmaLeanV38AcquisitionError):
        restarted.run()
    tampered_path.rename(original_path)
    original_path.write_bytes(original_bytes)

    parse_path = sorted(
        (private / "development" / "parse_receipts").glob("*.json")
    )[0]
    original_bytes = parse_path.read_bytes()
    parse_receipt = json.loads(original_bytes.decode("ascii"))
    parse_receipt["historical_filenames"] = [
        "CIK0000320193-submissions-000.json"
    ]
    tampered_sha = hashlib.sha256(_canonical(parse_receipt)).hexdigest()
    tampered_path = parse_path.with_name(f"000000-{tampered_sha}.json")
    parse_path.write_bytes(_canonical(parse_receipt))
    parse_path.rename(tampered_path)
    with pytest.raises(SecGemmaLeanV38AcquisitionError):
        restarted.run()
    tampered_path.rename(parse_path)
    parse_path.write_bytes(original_bytes)

    manifest_path = sorted(
        (private / "development" / "manifests").glob("*.json")
    )[0]
    original_bytes = manifest_path.read_bytes()
    manifest = json.loads(original_bytes.decode("ascii"))
    manifest["transport_receipt_sha256"] = "a" * 64
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("role_manifest_sha256")
    manifest["role_manifest_sha256"] = _hash_value(unsigned_manifest)
    tampered_path = manifest_path.with_name(
        f"000000-{manifest['role_manifest_sha256']}.json"
    )
    manifest_path.write_bytes(_canonical(manifest))
    manifest_path.rename(tampered_path)
    with pytest.raises(SecGemmaLeanV38AcquisitionError):
        restarted.run()
    tampered_path.rename(manifest_path)
    manifest_path.write_bytes(original_bytes)

    checkpoint_path = result.checkpoint_path
    original_bytes = checkpoint_path.read_bytes()
    checkpoint = json.loads(original_bytes.decode("ascii"))
    checkpoint["role_manifest_count"] += 1
    unsigned_checkpoint = dict(checkpoint)
    unsigned_checkpoint.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = _hash_value(unsigned_checkpoint)
    tampered_sha = hashlib.sha256(_canonical(checkpoint)).hexdigest()
    tampered_path = checkpoint_path.with_name(f"checkpoint-{tampered_sha}.json")
    checkpoint_path.write_bytes(_canonical(checkpoint))
    checkpoint_path.rename(tampered_path)
    with pytest.raises(SecGemmaLeanV38AcquisitionError):
        restarted.run()
    tampered_path.rename(checkpoint_path)
    checkpoint_path.write_bytes(original_bytes)

    public_path = result.public_authority_path
    original_bytes = public_path.read_bytes()
    public_receipt = json.loads(original_bytes.decode("ascii"))
    public_receipt["source_replay_receipt"]["stage"] = "rehashed-tamper"
    tampered_sha = hashlib.sha256(_canonical(public_receipt)).hexdigest()
    tampered_path = public_path.with_name(f"source-authority-{tampered_sha}.json")
    public_path.write_bytes(_canonical(public_receipt))
    public_path.rename(tampered_path)
    with pytest.raises(SecGemmaLeanV38AcquisitionError):
        restarted.run()
    tampered_path.rename(public_path)
    public_path.write_bytes(original_bytes)

    manifests = sorted(
        (private / "development" / "manifests").glob("*.json")
    )
    first_bytes, second_bytes = manifests[0].read_bytes(), manifests[1].read_bytes()
    manifests[0].write_bytes(second_bytes)
    manifests[1].write_bytes(first_bytes)
    with pytest.raises(SecGemmaLeanV38AcquisitionError):
        restarted.run()
    manifests[0].write_bytes(first_bytes)
    manifests[1].write_bytes(second_bytes)

    temporary_orphan = private / "development" / "temporary" / "crash.tmp"
    temporary_orphan.write_bytes(b"partial")
    with pytest.raises(SecGemmaLeanV38AcquisitionError):
        restarted.run()
    temporary_orphan.unlink()

    missing_blob = sorted(
        (private / "development" / "blobs").glob("*.blob")
    )[0]
    held_blob = missing_blob.with_suffix(".held")
    missing_blob.rename(held_blob)
    with pytest.raises(SecGemmaLeanV38AcquisitionError):
        restarted.run()
    held_blob.rename(missing_blob)

    orphan = private / "development" / "blobs" / ("9" * 64 + ".blob")
    orphan.write_bytes(b"orphan")
    with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
        restarted.run()
    assert caught.value.code == "artifact_inventory_invalid"


def test_full_size_three_stage_orchestrator_rehearsal_uses_disk_prior_rehydrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, private, public = _roots(tmp_path, monkeypatch)
    fingerprint = PrivateSecContact(CONTACT).fingerprint_sha256
    clock = _FakeClock()
    ledger = SharedSecDispatchLedger(
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        ledger_path=private / GLOBAL_LEDGER_NAME,
    )
    source = _FakeSource(
        historical_count=16,
        complete_counts={"development": 128, "intermediate": 24, "final": 16},
    )
    transports: list[_FakeTransport] = []

    def factory(temporary: Path) -> _FakeTransport:
        transport = _FakeTransport(temporary, fingerprint)
        transports.append(transport)
        return transport

    journal_roots = {
        private / stage / "journal" for stage in ("development", "intermediate", "final")
    }
    event_cache: dict[Path, list[dict[str, object]]] = {
        root: [] for root in journal_roots
    }
    real_read_events = journal_module._read_events
    real_rename = journal_module.os.rename

    def cached_read_events(root: Path):
        candidate = Path(root)
        if candidate in event_cache:
            return list(event_cache[candidate])
        return real_read_events(root)

    def capturing_rename(source_path: object, target_path: object) -> None:
        real_rename(source_path, target_path)
        target = Path(target_path)
        if target.parent in event_cache and target.name.endswith(".json"):
            event_cache[target.parent].append(
                json.loads(target.read_text(encoding="ascii"))
            )

    monkeypatch.setattr(journal_module.os, "fsync", lambda _descriptor: None)
    monkeypatch.setattr(journal_module, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(journal_module, "_read_events", cached_read_events)
    monkeypatch.setattr(journal_module.os, "rename", capturing_rename)

    development = DiskBackedSecAcquisition(
        stage="development",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=source,
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    ).run()

    def assert_prior_rejected(hostile_prior: object) -> None:
        hostile_transports: list[_FakeTransport] = []

        def hostile_factory(temporary: Path) -> _FakeTransport:
            value = _FakeTransport(temporary, fingerprint)
            hostile_transports.append(value)
            return value

        candidate = DiskBackedSecAcquisition(
            stage="intermediate",
            private_root=private,
            public_root=public,
            private_contact=CONTACT,
            transport_factory=hostile_factory,
            source_adapter=source,
            prior=hostile_prior,
            dispatch_ledger=ledger,
            monotonic=clock.monotonic,
        )
        with pytest.raises(SecGemmaLeanV38AcquisitionError) as caught:
            candidate.run()
        assert caught.value.code == "prior_stage_invalid"
        assert len(hostile_transports) == 1
        assert hostile_transports[0].calls == []

    bad_checkpoint = dict(development.checkpoint)
    bad_checkpoint["role_manifest_count"] += 1
    unsigned_checkpoint = dict(bad_checkpoint)
    unsigned_checkpoint.pop("checkpoint_sha256")
    bad_checkpoint["checkpoint_sha256"] = _hash_value(unsigned_checkpoint)
    assert_prior_rejected(
        replace(
            development,
            checkpoint=bad_checkpoint,
            compact_replay_input=replace(
                development.compact_replay_input,
                checkpoint=bad_checkpoint,
            ),
        )
    )

    bad_manifests = list(development.compact_replay_input.role_manifests)
    bad_manifest = dict(bad_manifests[0])
    bad_manifest["transport_receipt_sha256"] = "f" * 64
    unsigned_manifest = dict(bad_manifest)
    unsigned_manifest.pop("role_manifest_sha256")
    bad_manifest["role_manifest_sha256"] = _hash_value(unsigned_manifest)
    bad_manifests[0] = bad_manifest
    assert_prior_rejected(
        replace(
            development,
            compact_replay_input=replace(
                development.compact_replay_input,
                role_manifests=tuple(bad_manifests),
            ),
        )
    )

    bad_receipt = dict(development.replay_receipt)
    bad_receipt["checkpoint_sha256"] = "e" * 64
    assert_prior_rejected(replace(development, replay_receipt=bad_receipt))

    first_blob = next(iter(development.blob_paths.values()))
    original_blob = first_blob.read_bytes()
    first_blob.write_bytes(b"X" * len(original_blob))
    assert_prior_rejected(development)
    first_blob.write_bytes(original_blob)

    held_blob = private / "held-prior-blob"
    first_blob.rename(held_blob)
    try:
        first_blob.symlink_to(held_blob)
    except OSError:
        os.link(held_blob, first_blob)
    assert_prior_rejected(development)
    first_blob.unlink()
    held_blob.rename(first_blob)

    prior_temporary = private / "development" / "temporary" / "prior.tmp"
    prior_temporary.write_bytes(b"partial")
    assert_prior_rejected(development)
    prior_temporary.unlink()

    extra_transport = (
        private
        / "development"
        / "transport_receipts"
        / f"999999-{'9' * 64}.json"
    )
    extra_transport.write_bytes(b"{}")
    assert_prior_rejected(development)
    extra_transport.unlink()

    # The next stage must ignore this deliberately bogus in-memory object and
    # rehydrate the authenticated prior from its checkpoint/manifests/blobs.
    development_from_disk = replace(development, stage_output=object())
    intermediate = DiskBackedSecAcquisition(
        stage="intermediate",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=source,
        prior=development_from_disk,
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    ).run()
    intermediate_from_disk = replace(intermediate, stage_output=object())
    final = DiskBackedSecAcquisition(
        stage="final",
        private_root=private,
        public_root=public,
        private_contact=CONTACT,
        transport_factory=factory,
        source_adapter=source,
        prior=intermediate_from_disk,
        dispatch_ledger=ledger,
        monotonic=clock.monotonic,
    ).run()

    assert [
        development.journal_state.formula_requests,
        intermediate.journal_state.formula_requests,
        final.journal_state.formula_requests,
    ] == [245, 161, 164]
    assert [len(transport.calls) for transport in transports] == [245, 161, 164]
    assert all(
        result.accounting_complete and result.source_authoritative
        for result in (development, intermediate, final)
    )
    assert final.stage_output.prior_stage_source_seal_sha256 == (
        intermediate.stage_output.stage_source_seal_sha256
    )
    assert len(final.blob_paths) == 245 + 161 + 164
    assert all(
        len(list((public / stage).glob("source-authority-*.json"))) == 1
        for stage in ("development", "intermediate", "final")
    )

    monkeypatch.setattr(journal_module, "_read_events", real_read_events)
    monkeypatch.setattr(journal_module.os, "rename", real_rename)
