from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time
from typing import Any

import pytest

import agent_benchmark.sec_gemma_online_risk_overlay_acquisition as acquisition_module
import agent_benchmark.sec_gemma_online_risk_overlay_production as production_module
import agent_benchmark.sec_gemma_online_risk_overlay_runner as runner_module
import agent_benchmark.sec_gemma_online_risk_overlay_store as store_module
import agent_benchmark.sec_gemma_online_risk_overlay_vault as vault_module
from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
    ACQUISITION_VALIDATION_SCHEMA_VERSION,
    ACQUISITION_VALIDATION_VERIFIER_ID,
    CONFIRMATION as ACQUISITION_CONFIRMATION,
    DEVELOPMENT as ACQUISITION_DEVELOPMENT,
    FINAL as ACQUISITION_FINAL,
)
from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    ATTEMPT_ID_BY_KIND,
    ATTEMPT_KIND_BY_ID,
    CONFIRMATION_SCORING,
    CONSUMED,
    DEVELOPMENT_ACQUISITION,
    DEVELOPMENT_SCORING,
    IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
    PLANNED,
    REPORT_RECORD_TABLES,
    TERMINAL_FAIL,
    TERMINAL_INDETERMINATE,
    TERMINAL_PASS,
    build_attempt_plan,
    build_attempt_transition,
    validate_implementation_manifest,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    ACQUISITION_TERMINAL_EVIDENCE_FIELDS,
    BRANCH_NAME,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    EXTERNAL_TAG_REF_TEMPLATE,
    SCORED_TERMINAL_EVIDENCE_FIELDS,
    SOURCE_PIN_FILES,
    SOURCE_PINS,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
    ACQUISITION_PASS,
    PUBLICATION_GENESIS_SHA256,
    SCORED_FAILED_GATE,
    SCORED_PASS,
    _issue_verified_external_publication,
    build_external_publication,
    build_external_tag_message,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    CONTRACT_SOURCE_PATH,
    EXPECTED_ORIGIN_URL,
    PREREGISTRATION_COMMIT,
    REQUIRED_NEW_SOURCE_PATHS,
    SOURCE_VERIFICATION_SCHEMA_VERSION,
)
from agent_benchmark.sec_gemma_online_risk_overlay_store import (
    StoreRecordReceipt,
)
from agent_benchmark.sec_gemma_online_risk_overlay_runner import (
    CONFIRMATION_COMMAND,
    DETERMINISTIC_EVALUATION_SCHEMA_VERSION,
    DEVELOPMENT_ACQUISITION_COMMAND,
    DEVELOPMENT_COMMAND,
    FINAL_COMMAND,
    LOCAL_PREFLIGHT,
    PRODUCTION_EFFECT_BLOCKERS,
    RunnerDependencies,
    SEALED_STAGE_RESULT_SCHEMA_VERSION,
    SecGemmaOnlineRiskOverlayRunner,
    SecGemmaOnlineRiskOverlayRunnerError,
    SecGemmaOnlineRiskOverlayRunnerIndeterminate,
    build_phase_output,
    build_runner_plan,
    validate_runner_plan,
    validate_sealed_stage_result,
)
from tests.sec_gemma_online_risk_overlay_helpers import (
    build_synthetic_numerical_time_distributions,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_ZERO_COUNTERS = {
    "sec_request_count": 0,
    "market_request_count": 0,
    "model_call_count": 0,
    "retry_count": 0,
    "fallback_count": 0,
    "model_pull_count": 0,
    "paid_api_call_count": 0,
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _manifest() -> dict[str, Any]:
    contract_source = {
        "path": CONTRACT_SOURCE_PATH,
        "sha256": _digest("contract-source"),
    }
    reused = [
        {
            "role": role,
            "path": SOURCE_PIN_FILES[role],
            "sha256": SOURCE_PINS[role],
        }
        for role in sorted(SOURCE_PINS)
    ]
    new = [
        {
            "role": role,
            "path": REQUIRED_NEW_SOURCE_PATHS[role],
            "sha256": _digest(f"new:{role}"),
        }
        for role in sorted(REQUIRED_NEW_SOURCE_PATHS)
    ]
    commit = "1" * 40
    numerical_time_distributions = (
        build_synthetic_numerical_time_distributions(
            digest=_digest,
            canonical_sha256=canonical_sha256,
        )
    )
    dependency_material = {
        "dependency_sources": [],
        "external_distributions": [],
        "numerical_time_distributions": numerical_time_distributions,
    }
    dependency_closure = canonical_sha256(dependency_material)
    source_tree = {
        "contract_source": contract_source,
        "reused_sources": reused,
        "new_sources": new,
        **dependency_material,
        "dependency_closure_sha256": dependency_closure,
    }
    verification_material = {
        "schema_version": SOURCE_VERIFICATION_SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "branch": BRANCH_NAME,
        "origin_url": EXPECTED_ORIGIN_URL,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "head_commit": commit,
        "upstream_commit": commit,
        "contract_source": contract_source,
        "reused_sources": reused,
        "new_sources": new,
        **dependency_material,
        "dependency_closure_sha256": dependency_closure,
    }
    body = {
        "schema_version": IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "contract_source": contract_source,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "implementation_commit": commit,
        "branch": BRANCH_NAME,
        "upstream_ref": f"origin/{BRANCH_NAME}",
        "upstream_commit": commit,
        "origin_url": EXPECTED_ORIGIN_URL,
        "clean_tracked_tree": True,
        "head_matches_upstream": True,
        "preregistration_is_ancestor": True,
        "reused_sources": reused,
        "new_sources": new,
        **dependency_material,
        "dependency_closure_sha256": dependency_closure,
        "source_tree_sha256": canonical_sha256(source_tree),
        "source_verification_sha256": canonical_sha256(
            verification_material
        ),
        "effects_permitted": False,
    }
    return validate_implementation_manifest(
        {
            **body,
            "implementation_manifest_sha256": canonical_sha256(body),
        }
    )


class FakeCapability:
    def __init__(
        self,
        attempt_id: str,
        *,
        operation_kind: str | None = None,
        operation_sha256: str | None = None,
    ) -> None:
        self.attempt_id = attempt_id
        self.operation_kind = operation_kind
        self.operation_sha256 = operation_sha256


class FakeDurableAuthority:
    def __init__(
        self,
        material: dict[str, Any],
        store_receipt: StoreRecordReceipt,
        *,
        external_publication: Any | None = None,
    ) -> None:
        self._material = copy.deepcopy(material)
        self.store_receipt = store_receipt
        self.external_publication = external_publication

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._material)


class FakeStore:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.store_instance_id = _digest("fake-store-instance")
        self.store_session_nonce_sha256 = _digest(
            "fake-store-session-nonce"
        )
        self.plans: dict[str, dict[str, Any]] = {}
        self.histories: dict[str, list[dict[str, Any]]] = {}
        self.records: list[dict[str, Any]] = []
        self.governance: dict[
            str, list[FakeDurableAuthority]
        ] = {}
        self.events: list[str] = []
        self.anchor_bindings: dict[str, dict[str, Any]] = {}
        self.sequence = 0
        self.journal_tip_sha256 = _digest("fake-journal-genesis")

    def seed_pass(
        self,
        kind: str,
        predecessor: dict[str, Any] | None,
    ) -> dict[str, Any]:
        attempt_id = ATTEMPT_ID_BY_KIND[kind]
        plan = build_attempt_plan(
            implementation_manifest=self.manifest,
            attempt_id=attempt_id,
            prerequisite_terminal_transition=predecessor,
        )
        planned = build_attempt_transition(
            attempt_plan=plan,
            implementation_manifest=self.manifest,
            status=PLANNED,
        )
        consumed = build_attempt_transition(
            attempt_plan=plan,
            implementation_manifest=self.manifest,
            status=CONSUMED,
            prior_transition=planned,
        )
        terminal = build_attempt_transition(
            attempt_plan=plan,
            implementation_manifest=self.manifest,
            status=TERMINAL_PASS,
            prior_transition=consumed,
        )
        self.plans[attempt_id] = plan
        self.histories[attempt_id] = [planned, consumed, terminal]
        return terminal

    def register_attempt(self, attempt_plan: dict[str, Any]) -> StoreRecordReceipt:
        attempt_id = attempt_plan["attempt_id"]
        self.events.append(f"register:{attempt_id}")
        self.plans[attempt_id] = copy.deepcopy(attempt_plan)
        planned = build_attempt_transition(
            attempt_plan=attempt_plan,
            implementation_manifest=self.manifest,
            status=PLANNED,
        )
        self.histories[attempt_id] = [planned]
        return self._receipt("attempts", f"{attempt_id}:1", attempt_id, planned)

    def consume_attempt(self, attempt_id: str) -> FakeCapability:
        self.events.append(f"consume:{attempt_id}")
        plan = self.plans[attempt_id]
        consumed = build_attempt_transition(
            attempt_plan=plan,
            implementation_manifest=self.manifest,
            status=CONSUMED,
            prior_transition=self.histories[attempt_id][-1],
        )
        self.histories[attempt_id].append(consumed)
        return FakeCapability(attempt_id)

    def authorize_effect(self, capability: FakeCapability, effect: str) -> None:
        assert self.histories[capability.attempt_id][-1]["status"] == CONSUMED
        self.events.append(f"authorize:{effect}")

    def _receipt(
        self,
        table: str,
        identity: str,
        attempt_id: str,
        payload: dict[str, Any],
    ) -> StoreRecordReceipt:
        self.sequence += 1
        payload_hash = canonical_sha256(payload)
        receipt = StoreRecordReceipt(
            table=table,
            identity=identity,
            attempt_id=attempt_id,
            payload_sha256=payload_hash,
            journal_sequence=self.sequence,
            journal_entry_sha256=_digest(
                f"{self.sequence}:{table}:{identity}:{payload_hash}"
            ),
        )
        self.journal_tip_sha256 = receipt.journal_entry_sha256
        return receipt

    def _append(
        self,
        table: str,
        capability: FakeCapability,
        effect: str,
        identity: str,
        payload: dict[str, Any],
    ) -> StoreRecordReceipt:
        self.authorize_effect(capability, effect)
        receipt = self._receipt(
            table, identity, capability.attempt_id, payload
        )
        self.records.append(
            {
                "receipt": receipt,
                "payload": copy.deepcopy(payload),
            }
        )
        return receipt

    def append_evidence(self, **kwargs: Any) -> StoreRecordReceipt:
        return self._append(
            "evidence",
            kwargs["capability"],
            kwargs["effect"],
            kwargs["identity"],
            kwargs["payload"],
        )

    def append_feature(self, **kwargs: Any) -> StoreRecordReceipt:
        return self._append(
            "features",
            kwargs["capability"],
            kwargs["effect"],
            kwargs["identity"],
            kwargs["payload"],
        )

    def append_prediction(self, **kwargs: Any) -> StoreRecordReceipt:
        return self._append(
            "predictions",
            kwargs["capability"],
            kwargs["effect"],
            kwargs["identity"],
            kwargs["payload"],
        )

    def append_lesson(self, **kwargs: Any) -> StoreRecordReceipt:
        return self._append(
            "lessons",
            kwargs["capability"],
            kwargs["effect"],
            kwargs["identity"],
            kwargs["payload"],
        )

    def append_ledger(self, **kwargs: Any) -> StoreRecordReceipt:
        return self._append(
            "ledgers",
            kwargs["capability"],
            kwargs["effect"],
            kwargs["identity"],
            kwargs["payload"],
        )

    def append_artifact(self, **kwargs: Any) -> StoreRecordReceipt:
        return self._append(
            "artifacts",
            kwargs["capability"],
            kwargs["effect"],
            kwargs["identity"],
            kwargs["payload"],
        )

    def terminal_evidence_material(
        self, capability: FakeCapability
    ) -> dict[str, Any]:
        rows = [
            item["receipt"]
            for item in self.records
            if item["receipt"].attempt_id == capability.attempt_id
        ]
        counts = {
            table: sum(row.table == table for row in rows)
            for table in REPORT_RECORD_TABLES
        }
        commitment_rows = [
            {
                "table": row.table,
                "identity": row.identity,
                "payload_sha256": row.payload_sha256,
                "journal_sequence": row.journal_sequence,
                "journal_entry_sha256": row.journal_entry_sha256,
            }
            for row in rows
        ]
        return {
            "attempt_id": capability.attempt_id,
            "attempt_plan_sha256": self.plans[capability.attempt_id][
                "attempt_plan_sha256"
            ],
            "record_counts": counts,
            "record_commitment_sha256": canonical_sha256(
                commitment_rows
            ),
        }

    def _governance_authority(
        self,
        *,
        table: str,
        identity: str,
        attempt_id: str,
        material: dict[str, Any],
        external_publication: Any | None = None,
    ) -> FakeDurableAuthority:
        receipt = self._receipt(
            table, identity, attempt_id, material
        )
        self.events.append(f"commit:{table}:{attempt_id}")
        authority = FakeDurableAuthority(
            material,
            receipt,
            external_publication=external_publication,
        )
        self.governance.setdefault(table, []).append(authority)
        return authority

    def attempt_plan(self, attempt_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.plans[attempt_id])

    def governance_records(
        self,
        table: str,
        *,
        attempt_id: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = [
            authority.as_dict()
            for authority in self.governance.get(table, [])
        ]
        if attempt_id is not None:
            rows = [
                row
                for row in rows
                if row.get("attempt_id") == attempt_id
            ]
        return rows

    def _governance_lookup(
        self,
        table: str,
        attempt_id: str,
        *,
        hash_field: str | None = None,
        digest: str | None = None,
    ) -> FakeDurableAuthority:
        matches = [
            authority
            for authority in self.governance.get(table, [])
            if authority.as_dict().get("attempt_id") == attempt_id
            and (
                hash_field is None
                or authority.as_dict().get(hash_field) == digest
            )
        ]
        assert len(matches) == 1
        return matches[0]

    def terminal_reconstruction_authority(
        self, attempt_id: str
    ) -> FakeDurableAuthority:
        return self._governance_lookup(
            "terminal_reconstruction_materials", attempt_id
        )

    def publication_intent_authority(
        self, attempt_id: str
    ) -> FakeDurableAuthority:
        return self._governance_lookup(
            "publication_intents", attempt_id
        )

    def publication_receipt_authority(
        self,
        attempt_id: str,
        external_publication: Any,
    ) -> FakeDurableAuthority:
        authority = self._governance_lookup(
            "publication_receipts", attempt_id
        )
        assert (
            authority.as_dict()["external_publication_sha256"]
            == external_publication.publication_sha256
        )
        authority.external_publication = external_publication
        return authority

    def remote_observation_authority(
        self,
        attempt_id: str,
        remote_observation_sha256: str,
    ) -> FakeDurableAuthority:
        return self._governance_lookup(
            "publication_remote_observations",
            attempt_id,
            hash_field="publication_remote_observation_sha256",
            digest=remote_observation_sha256,
        )

    def publication_worker_ownership_authority(
        self,
        attempt_id: str,
        worker_ownership_sha256: str,
    ) -> FakeDurableAuthority:
        return self._governance_lookup(
            "publication_worker_ownership",
            attempt_id,
            hash_field="worker_ownership_sha256",
            digest=worker_ownership_sha256,
        )

    def pre_push_authorization_authority(
        self,
        attempt_id: str,
        pre_push_authorization_sha256: str,
    ) -> FakeDurableAuthority:
        return self._governance_lookup(
            "publication_pre_push_authorizations",
            attempt_id,
            hash_field="pre_push_authorization_sha256",
            digest=pre_push_authorization_sha256,
        )

    def recovery_completion_authority(
        self,
        attempt_id: str,
        recovery_invocation_completion_sha256: str,
    ) -> FakeDurableAuthority:
        return self._governance_lookup(
            "publication_recovery_invocation_completions",
            attempt_id,
            hash_field="recovery_invocation_completion_sha256",
            digest=recovery_invocation_completion_sha256,
        )

    def reconcile_publication_recovery_state(
        self, attempt_id: str
    ) -> None:
        self.events.append(f"reconcile-publication:{attempt_id}")

    def terminalization_claim_authority(
        self, attempt_id: str
    ) -> FakeDurableAuthority:
        return self._governance_lookup(
            "terminalization_claims", attempt_id
        )

    def terminal_artifact_payload_and_receipt(
        self, attempt_id: str
    ) -> tuple[dict[str, Any], StoreRecordReceipt]:
        matches = [
            item
            for item in self.records
            if item["receipt"].table == "artifacts"
            and item["receipt"].attempt_id == attempt_id
            and item["receipt"].identity
            == f"terminal_artifact:{attempt_id}"
        ]
        assert len(matches) == 1
        return (
            copy.deepcopy(matches[0]["payload"]),
            matches[0]["receipt"],
        )

    def pending_acquisition_phase_evidence(
        self, attempt_id: str
    ) -> dict[str, Any]:
        matches = [
            copy.deepcopy(item["payload"])
            for item in self.records
            if item["receipt"].table == "evidence"
            and item["receipt"].attempt_id == attempt_id
        ]
        return {
            item["phase"]: item
            for item in matches
            if "phase" in item
        }

    def commit_terminal_reconstruction_material(
        self,
        capability: FakeCapability,
        material: dict[str, Any],
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="terminal_reconstruction_materials",
            identity=f"terminal_reconstruction:{capability.attempt_id}",
            attempt_id=capability.attempt_id,
            material=material,
        )

    def commit_publication_intent(
        self,
        capability: FakeCapability,
        material: dict[str, Any],
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="publication_intents",
            identity=f"publication_intent:{capability.attempt_id}",
            attempt_id=capability.attempt_id,
            material=material,
        )

    def issue_publication_recovery_capability(
        self,
        attempt_id: str,
        *,
        operation_kind: str,
        operation_sha256: str | None = None,
    ) -> FakeCapability:
        self.events.append(f"publication-capability:{attempt_id}")
        return FakeCapability(
            attempt_id,
            operation_kind=operation_kind,
            operation_sha256=operation_sha256,
        )

    def commit_recovery_start(
        self,
        capability: FakeCapability,
        start: dict[str, Any],
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="publication_recovery_invocation_starts",
            identity=(
                "publication_recovery_start:"
                f"{capability.attempt_id}:{start['invocation_ordinal']}"
            ),
            attempt_id=capability.attempt_id,
            material=start,
        )

    def commit_transport_manifest(
        self,
        capability: FakeCapability,
        manifest: dict[str, Any],
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="publication_transports",
            identity=f"publication_transport:{capability.attempt_id}",
            attempt_id=capability.attempt_id,
            material=manifest,
        )

    def claim_publication_worker_ownership(
        self,
        capability: FakeCapability,
        ownership: dict[str, Any],
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="publication_worker_ownership",
            identity=f"publication_owner:{capability.attempt_id}",
            attempt_id=capability.attempt_id,
            material=ownership,
        )

    def commit_worker_quiescence(
        self,
        capability: FakeCapability,
        quiescence: dict[str, Any],
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="publication_worker_quiescence",
            identity=f"publication_quiescence:{capability.attempt_id}",
            attempt_id=capability.attempt_id,
            material=quiescence,
        )

    def commit_remote_observation(
        self,
        capability: FakeCapability,
        readback_evidence: dict[str, Any],
        observation: dict[str, Any],
    ) -> FakeDurableAuthority:
        del readback_evidence
        return self._governance_authority(
            table="publication_remote_observations",
            identity=(
                "publication_observation:"
                f"{capability.attempt_id}:{self.sequence + 1}"
            ),
            attempt_id=capability.attempt_id,
            material=observation,
        )

    def commit_pre_push_authorization(
        self,
        capability: FakeCapability,
        authorization: dict[str, Any],
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="publication_pre_push_authorizations",
            identity=f"publication_authorization:{capability.attempt_id}",
            attempt_id=capability.attempt_id,
            material=authorization,
        )

    def commit_publication_conflict(
        self,
        capability: FakeCapability,
        conflict: dict[str, Any],
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="publication_conflicts",
            identity=f"publication_conflict:{capability.attempt_id}",
            attempt_id=capability.attempt_id,
            material=conflict,
        )

    def commit_recovery_completion(
        self,
        capability: FakeCapability,
        completion: dict[str, Any],
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="publication_recovery_invocation_completions",
            identity=(
                "publication_recovery_completion:"
                f"{capability.attempt_id}:"
                f"{completion['invocation_ordinal']}"
            ),
            attempt_id=capability.attempt_id,
            material=completion,
        )

    def commit_publication_receipt(
        self,
        capability: FakeCapability,
        receipt: dict[str, Any],
        external_publication: Any,
    ) -> FakeDurableAuthority:
        return self._governance_authority(
            table="publication_receipts",
            identity=f"publication_receipt:{capability.attempt_id}",
            attempt_id=capability.attempt_id,
            material=receipt,
            external_publication=external_publication,
        )

    def release_publication_capability(
        self, capability: FakeCapability
    ) -> None:
        self.events.append(
            f"release-publication:{capability.attempt_id}"
        )

    def issue_terminalization_capability(
        self, attempt_id: str
    ) -> FakeCapability:
        self.events.append(f"terminalization-capability:{attempt_id}")
        return FakeCapability(attempt_id)

    def commit_terminalization_claim(
        self,
        capability: FakeCapability,
        terminal_evidence: Any,
        terminal_status: str,
    ) -> FakeDurableAuthority:
        material = {
            "attempt_id": capability.attempt_id,
            "terminal_status": terminal_status,
            "terminal_evidence_sha256": terminal_evidence.evidence[
                "terminal_evidence_sha256"
            ],
        }
        return self._governance_authority(
            table="terminalization_claims",
            identity=f"terminalization_claim:{capability.attempt_id}",
            attempt_id=capability.attempt_id,
            material=material,
        )

    def complete_terminalized_attempt(
        self,
        claim: FakeDurableAuthority,
        verified_terminal_evidence: Any,
    ) -> StoreRecordReceipt:
        claim_material = claim.as_dict()
        attempt_id = claim_material["attempt_id"]
        terminal_status = claim_material["terminal_status"]
        self.events.append(f"finish:{terminal_status}")
        publication = (
            verified_terminal_evidence.external_publication
        )
        self.anchor_bindings[attempt_id] = {
            "terminal_evidence": copy.deepcopy(
                verified_terminal_evidence.evidence
            ),
            "external_publication": copy.deepcopy(
                publication.publication
            ),
            "artifact_receipt": {
                field: getattr(
                    verified_terminal_evidence.artifact_receipt,
                    field,
                )
                for field in (
                    "table",
                    "identity",
                    "attempt_id",
                    "payload_sha256",
                    "journal_sequence",
                    "journal_entry_sha256",
                )
            },
        }
        terminal = build_attempt_transition(
            attempt_plan=self.plans[attempt_id],
            implementation_manifest=self.manifest,
            status=terminal_status,
            prior_transition=self.histories[attempt_id][-1],
        )
        self.histories[attempt_id].append(terminal)
        return self._receipt(
            "attempts",
            f"{attempt_id}:3",
            attempt_id,
            terminal,
        )

    def finish_attempt(
        self,
        capability: FakeCapability,
        *,
        terminal_status: str,
        verified_terminal_evidence: Any = None,
    ) -> StoreRecordReceipt:
        self.events.append(f"finish:{terminal_status}")
        if terminal_status == TERMINAL_PASS:
            assert verified_terminal_evidence is not None
        if verified_terminal_evidence is not None:
            self.anchor_bindings[capability.attempt_id] = {
                "terminal_evidence": copy.deepcopy(
                    verified_terminal_evidence.evidence
                ),
                "external_publication": copy.deepcopy(
                    verified_terminal_evidence.external_publication.publication
                ),
                "artifact_receipt": {
                    field: getattr(
                        verified_terminal_evidence.artifact_receipt,
                        field,
                    )
                    for field in (
                        "table",
                        "identity",
                        "attempt_id",
                        "payload_sha256",
                        "journal_sequence",
                        "journal_entry_sha256",
                    )
                },
            }
        plan = self.plans[capability.attempt_id]
        terminal = build_attempt_transition(
            attempt_plan=plan,
            implementation_manifest=self.manifest,
            status=terminal_status,
            prior_transition=self.histories[capability.attempt_id][-1],
        )
        self.histories[capability.attempt_id].append(terminal)
        return self._receipt(
            "attempts",
            f"{capability.attempt_id}:3",
            capability.attempt_id,
            terminal,
        )

    def attempt_history(self, attempt_id: str) -> list[dict[str, Any]]:
        return copy.deepcopy(self.histories.get(attempt_id, []))

    def terminal_anchor_binding(
        self, attempt_id: str
    ) -> dict[str, Any]:
        if attempt_id in self.anchor_bindings:
            return copy.deepcopy(self.anchor_bindings[attempt_id])
        kind = ATTEMPT_KIND_BY_ID[attempt_id]
        report_kind = (
            ACQUISITION_PASS
            if kind == DEVELOPMENT_ACQUISITION
            else SCORED_PASS
        )
        publication = _fake_verified_publication(
            self.manifest,
            attempt_id=attempt_id,
            terminal_status=TERMINAL_PASS,
            report_kind=report_kind,
            artifact_sha256=_digest(f"seed:{attempt_id}"),
            predecessor_publication_sha256=(
                PUBLICATION_GENESIS_SHA256
            ),
        )
        return {
            "terminal_evidence": {
                "attempt_id": attempt_id,
                "terminal_status": TERMINAL_PASS,
                "external_publication_sha256": (
                    publication.publication_sha256
                ),
            },
            "external_publication": publication.publication,
            "artifact_receipt": {
                "payload_sha256": _digest(f"seed-receipt:{attempt_id}")
            },
        }

    def predecessor_feature_rows(
        self,
        stage: str,
    ) -> list[dict[str, Any]]:
        predecessor_ids = {
            "development": (),
            "confirmation": (
                ATTEMPT_ID_BY_KIND[DEVELOPMENT_SCORING],
            ),
            "final": (
                ATTEMPT_ID_BY_KIND[DEVELOPMENT_SCORING],
                ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING],
            ),
        }[stage]
        rows = [
            copy.deepcopy(item["payload"])
            for item in self.records
            if item["receipt"].table == "features"
            and item["receipt"].attempt_id in predecessor_ids
        ]
        rows.sort(
            key=lambda row: (
                row["decision_session"],
                row["acceptance_datetime"] or "",
                row["accession_number"],
                row["feature_row_sha256"],
            )
        )
        return rows

    def snapshot(self) -> dict[str, Any]:
        return {
            "chain_valid": True,
            "journal_entry_count": self.sequence,
            "journal_tip_sha256": self.journal_tip_sha256,
            "attempt_states": {
                attempt_id: history[-1]["status"]
                for attempt_id, history in self.histories.items()
            },
        }


class MutableClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if not self.values:
            raise AssertionError("clock sequence exhausted")
        return self.values.pop(0)


def _phase_payload(command: str, phase: str) -> dict[str, Any]:
    if phase == "runtime_identity":
        return {
            "version_response_hex": "00",
            "show_response_hex": "00",
        }
    if phase == "gemma":
        semantic_hash = _digest(f"{command}:semantic")
        latency_body = {
            "stage": {
                DEVELOPMENT_COMMAND: "development",
                CONFIRMATION_COMMAND: "confirmation",
                FINAL_COMMAND: "final",
            }[command],
            "selection": "test_exact_order",
            "model_call_count": _phase_counters(
                command,
                phase,
            )["model_call_count"],
        }
        latency_receipt = {
            **latency_body,
            "latency_preflight_receipt_sha256": canonical_sha256(
                latency_body
            ),
        }
        return {
            "semantic_extraction_rows": [
                {
                    "semantic_extraction_row_sha256": semantic_hash,
                    "latency_preflight_receipt": latency_receipt,
                    "latency_preflight_receipt_sha256": (
                        latency_receipt[
                            "latency_preflight_receipt_sha256"
                        ]
                    ),
                }
            ],
            "semantic_extraction_row_sha256s": [semantic_hash],
            "semantic_batch_receipt_sha256": _digest(
                f"{command}:semantic-batch"
            ),
        }
    cutoff = {
        DEVELOPMENT_COMMAND: "2018-12-31",
        CONFIRMATION_COMMAND: "2023-12-29",
        FINAL_COMMAND: "2026-07-09",
    }[command]
    market = [
        {"session": "2000-01-03"},
        {"session": cutoff},
    ]
    signals = [
        {"session": "2000-01-03"},
        {"session": cutoff},
    ]
    feature_hash = _digest(f"{command}:feature")
    feature_row = {
        "accession_number": f"{command}-feature",
        "decision_session": cutoff,
        "acceptance_datetime": (
            cutoff.replace("-", "") + "160000"
        ),
        "feature_row_sha256": feature_hash,
    }
    return {
        "market_rows": market,
        "market_rows_sha256": canonical_sha256(market),
        "baseline_signals": signals,
        "baseline_signals_sha256": canonical_sha256(signals),
        "feature_rows": [feature_row],
        "feature_row_sha256s": [feature_hash],
        "semantic_batch_receipt_sha256": _digest(
            f"{command}:semantic-batch"
        ),
        "source_commitments": {
            "sec": _digest(f"{command}:sec"),
            "market": _digest(f"{command}:market"),
        },
    }


def _phase_counters(command: str, phase: str) -> dict[str, int]:
    counters = dict(_ZERO_COUNTERS)
    if phase == "gemma":
        counters["model_call_count"] = (
            5 if command == DEVELOPMENT_COMMAND else 2
        )
    return counters


class FakeExecutor:
    def __init__(
        self,
        store: FakeStore,
        clock: MutableClock | SequenceClock,
        *,
        durations: dict[str, float] | None = None,
        raise_phase: str | None = None,
        mutate_payload: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.store = store
        self.clock = clock
        self.durations = durations or {}
        self.raise_phase = raise_phase
        self.mutate_payload = mutate_payload or {}
        self.events: list[str] = []

    def execute(
        self,
        *,
        command: str,
        phase: str,
        permit: Any,
        deadline_monotonic: float,
        prior_phase_outputs: dict[str, Any],
        acquisition_execution: Any,
    ) -> dict[str, Any]:
        del deadline_monotonic, prior_phase_outputs, acquisition_execution
        self.events.append(phase)
        if command != LOCAL_PREFLIGHT:
            assert permit is not None
            assert self.store.histories[permit.attempt_id][-1][
                "status"
            ] == CONSUMED
            assert any(
                event == f"consume:{permit.attempt_id}"
                for event in self.store.events
            )
        else:
            assert permit is None
        if self.raise_phase == phase:
            raise RuntimeError("worker crashed")
        if isinstance(self.clock, MutableClock):
            self.clock.value += self.durations.get(phase, 1.0)
        payload = _phase_payload(command, phase)
        payload.update(self.mutate_payload.get(phase, {}))
        return build_phase_output(
            command=command,
            phase=phase,
            counters=_phase_counters(command, phase),
            payload=payload,
        )


def _fake_ledger(label: str, cost: str) -> dict[str, Any]:
    return {
        "policy_id": label,
        "cost_bps": 5 if cost == "cost_5bps" else 10,
        "ledger_sha256": _digest(f"{label}:{cost}:ledger"),
    }


def _online_replay(
    *,
    stage: str,
    arm: str,
) -> dict[str, Any]:
    costs: dict[str, Any] = {}
    for cost in ("cost_5bps", "cost_10bps"):
        costs[cost] = {
            "combined": _fake_ledger(f"{arm}:combined", cost),
            "baseline": _fake_ledger("baseline", cost),
            "aapl_buy_and_hold": _fake_ledger("aapl", cost),
        }
    return {
        "chronological_replay_sha256": _digest(
            f"{stage}:{arm}:online-replay"
        ),
        "learner_lessons": [],
        "primary": {
            "predictions": [],
            "ledgers": costs,
            "target_stream": {"target_rows": []},
        },
    }


def _frozen_replay(
    *,
    stage: str,
    control_id: str,
) -> dict[str, Any]:
    costs = {
        cost: {
            "combined": _fake_ledger(
                f"semantic:frozen:{control_id}", cost
            )
        }
        for cost in ("cost_5bps", "cost_10bps")
    }
    return {
        "chronological_replay_sha256": _digest(
            f"{stage}:{control_id}:frozen-replay"
        ),
        "frozen_control": {
            "predictions": [],
            "ledgers": costs,
            "target_stream": {"target_rows": []},
        },
    }


def _evaluation(
    stage: str,
    input_bundle: dict[str, Any],
    *,
    missing_arm: bool = False,
    missing_proof: bool = False,
    failed_gate: bool = False,
    fabricated_checks: bool = False,
    tamper_replay_hash: bool = False,
) -> dict[str, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_metrics import (
        FROZEN_CONTROL_IDS,
    )

    replays = {
        arm: {
            "online": _online_replay(stage=stage, arm=arm),
            "frozen_controls": (
                {
                    control_id: _frozen_replay(
                        stage=stage, control_id=control_id
                    )
                    for control_id in FROZEN_CONTROL_IDS[stage]
                }
                if arm == "semantic"
                else {}
            ),
        }
        for arm in (
            "semantic",
            "no_filing_meaning",
            "no_gemma_channel",
        )
    }
    if missing_arm:
        replays.pop("no_gemma_channel")
    checks = (
        {"foo": True}
        if fabricated_checks
        else {
            key: True
            for key in build_contract_manifest()["gates"][stage]
        }
    )
    if failed_gate and not fabricated_checks:
        first = next(iter(checks))
        checks[first] = False
    failures = [key for key, value in checks.items() if not value]
    gate_body = {
        "checks": checks,
        "passed": not failures,
        "failed_checks": failures,
    }
    gate = {
        **gate_body,
        "gate_report_sha256": canonical_sha256(gate_body),
    }
    metrics_input = {
        "stage_metrics_input_sha256": _digest(
            f"{stage}:metrics-input"
        )
    }
    stage_metrics = {
        "stage_metrics_sha256": _digest(f"{stage}:metrics")
    }
    proofs = {
        f"{account}:{cost}": {
            "proof_sha256": _digest(f"{account}:{cost}:proof")
        }
        for account in (
            "semantic",
            "baseline",
            "aapl_buy_and_hold",
            "no_filing_meaning",
            "no_gemma_channel",
        )
        for cost in ("cost_5bps", "cost_10bps")
    }
    for control_id in FROZEN_CONTROL_IDS[stage]:
        for cost in ("cost_5bps", "cost_10bps"):
            proof_id = f"frozen:{control_id}:{cost}"
            proofs[proof_id] = {
                "proof_sha256": _digest(f"{proof_id}:proof")
            }
    if missing_proof:
        proofs.pop(next(iter(proofs)))
    body = {
        "schema_version": DETERMINISTIC_EVALUATION_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": stage,
        "stage_input_bundle_sha256": input_bundle[
            "stage_input_bundle_sha256"
        ],
        "replays": replays,
        "replays_sha256": (
            _digest("tampered")
            if tamper_replay_hash
            else canonical_sha256(replays)
        ),
        "metrics_input": metrics_input,
        "metrics_input_sha256": metrics_input[
            "stage_metrics_input_sha256"
        ],
        "stage_metrics": stage_metrics,
        "stage_metrics_sha256": stage_metrics[
            "stage_metrics_sha256"
        ],
        "gate_report": gate,
        "gate_report_sha256": gate["gate_report_sha256"],
        "no_leverage_proofs": proofs,
        "no_leverage_proofs_sha256": canonical_sha256(proofs),
    }
    return {
        **body,
        "deterministic_evaluation_sha256": canonical_sha256(body),
    }


class FakeEvaluator:
    def __init__(self, **options: bool) -> None:
        self.options = options

    def evaluate(
        self, *, stage: str, input_bundle: dict[str, Any]
    ) -> dict[str, Any]:
        return _evaluation(stage, input_bundle, **self.options)

    def validate(
        self,
        evaluation: dict[str, Any],
        *,
        stage: str,
        input_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        del stage, input_bundle
        return copy.deepcopy(evaluation)


def _opaque_acquisition_report(
    payload: dict[str, Any],
    vault_path: Path,
) -> Any:
    vault = vault_module.open_test_acquisition_vault(vault_path)
    attempt_id = payload["attempt_id"]
    capability = store_module.EffectCapability(
        attempt_id=attempt_id,
        allowed_effects=("official_sec_network", "market_network"),
        receipt=StoreRecordReceipt(
            table="attempts",
            identity="attempt:test-recovery-report",
            attempt_id=attempt_id,
            payload_sha256=_digest("test-recovery-attempt"),
            journal_sequence=1,
            journal_entry_sha256=_digest(
                "test-recovery-attempt-journal"
            ),
        ),
        store_instance_id=_digest("test-recovery-store"),
        store_nonce=_digest("test-recovery-store-nonce"),
        transition_sha256=_digest("test-recovery-transition"),
        _sentinel=store_module._CAPABILITY_SENTINEL,
    )

    class VaultAuthorizer:
        @staticmethod
        def authorize_effect(
            supplied: Any, effect: str
        ) -> None:
            assert supplied is capability
            assert effect in capability.allowed_effects

    handle = vault_module._seal_quarantine(
        vault,
        store=VaultAuthorizer(),
        capability=capability,
        stage=payload["stage"],
        attempt_id=attempt_id,
        bundle_sha256=payload["bundle_sha256"],
        manifest_sha256=payload["manifest_sha256"],
        private_index_sha256=payload["private_index_sha256"],
        predecessor_handles=(),
        quarantine={},
    )
    return acquisition_module.VerifiedAcquisitionReport(
        payload,
        vault=vault,
        handle=handle,
        _sentinel=acquisition_module._VERIFIED_REPORT_SENTINEL,
    )


class FakeAcquisitionAdapter:
    def __init__(
        self,
        clock: MutableClock | SequenceClock,
        *,
        duration: float = 1.0,
        market_elapsed: float = 1.0,
        manifest_overrides: dict[str, Any] | None = None,
        report_checks: dict[str, str] | None = None,
        recovery_vault_path: Path | None = None,
    ) -> None:
        self.clock = clock
        self.duration = duration
        self.market_elapsed = market_elapsed
        self.manifest_overrides = manifest_overrides or {}
        self.report_checks = report_checks
        self.recovery_vault_path = recovery_vault_path
        self.calls: list[str] = []
        self.rehydrate_calls: list[str] = []
        self._recovery_report: Any | None = None

    def acquire(
        self,
        *,
        command: str,
        store: FakeStore,
        capability: FakeCapability,
        deadline_monotonic: float,
    ) -> dict[str, Any]:
        del deadline_monotonic
        assert store.histories[capability.attempt_id][-1][
            "status"
        ] == CONSUMED
        self.calls.append(command)
        if isinstance(self.clock, MutableClock):
            self.clock.value += self.duration
        stage = {
            DEVELOPMENT_ACQUISITION_COMMAND: ACQUISITION_DEVELOPMENT,
            CONFIRMATION_COMMAND: ACQUISITION_CONFIRMATION,
            FINAL_COMMAND: ACQUISITION_FINAL,
        }[command]
        checks = self.report_checks or {
            name: _digest(f"{command}:{name}")
            for name in (
                "exact_raw_bytes_replayed_sha256",
                "request_receipts_reconciled_sha256",
                "stage_and_attempt_scope_bound_sha256",
                "private_identity_digest_only_sha256",
                "market_prefix_continuity_replayed_sha256",
                "blinded_model_requests_replayed_sha256",
                "request_byte_retry_redirect_caps_reconciled_sha256",
            )
        }
        bundle_hash = _digest(f"{command}:bundle")
        manifest_hash = _digest(f"{command}:manifest")
        report_body = {
            "schema_version": ACQUISITION_VALIDATION_SCHEMA_VERSION,
            "verifier_id": ACQUISITION_VALIDATION_VERIFIER_ID,
            "verdict": "pass",
            "stage": stage,
            "attempt_id": capability.attempt_id,
            "attempt_kind": ATTEMPT_KIND_BY_ID[capability.attempt_id],
            "acquisition_plan_sha256": (
                acquisition_module.build_acquisition_plan(stage)[
                    "acquisition_plan_sha256"
                ]
            ),
            "bundle_sha256": bundle_hash,
            "manifest_sha256": manifest_hash,
            "private_index_sha256": _digest(
                f"{command}:private-index"
            ),
            "predecessor_chain_bundle_sha256s": [],
            "checks": checks,
            "check_set_sha256": canonical_sha256(checks),
        }
        report = {
            **report_body,
            "validation_sha256": canonical_sha256(report_body),
        }
        if self.recovery_vault_path is not None:
            self._recovery_report = _opaque_acquisition_report(
                report, self.recovery_vault_path
            )
        summary_body = {
            "schema_version": (
                "aapl-sec-gemma-online-risk-overlay-v2-2-"
                "public-summary-v1"
            ),
            "stage": stage,
            "attempt_id": capability.attempt_id,
            "attempt_kind": ATTEMPT_KIND_BY_ID[capability.attempt_id],
            "acquisition_plan_sha256": report[
                "acquisition_plan_sha256"
            ],
            "predecessor_bundle_sha256": None,
            "bundle_sha256": bundle_hash,
            "manifest_sha256": manifest_hash,
            "private_index_sha256": report["private_index_sha256"],
            "sec_catalog_source_count": 3,
            "sec_primary_document_count": 3,
            "market_response_count": 6,
            "model_request_count": (
                72
                if command == DEVELOPMENT_ACQUISITION_COMMAND
                else 2
            ),
            "model_slice_sha256": _digest(
                f"{command}:model-slice"
            ),
            "stage_slice_sha256": _digest(
                f"{command}:stage-slice"
            ),
            "market_elapsed_seconds_hex": self.market_elapsed.hex(),
            "total_raw_byte_count": 100,
            "complete_batch": True,
            "quarantine_only": True,
            "production_authority": False,
        }
        summary_body.update(self.manifest_overrides)
        summary = {
            **summary_body,
            "public_summary_sha256": canonical_sha256(summary_body),
        }
        accounting_body = {
            "schema_version": (
                "aapl-sec-gemma-online-risk-overlay-v2-2-"
                "request-accounting-v1"
            ),
            "stage": stage,
            "attempt_id": capability.attempt_id,
            "sec_request_count": 3,
            "market_request_count": 6,
            "sec_bytes": 50,
            "market_bytes": 50,
            "network_request_count": 9,
            "retry_count": 0,
            "redirect_count": 0,
            "market_elapsed_seconds_hex": self.market_elapsed.hex(),
        }
        accounting = {
            **accounting_body,
            "accounting_sha256": canonical_sha256(accounting_body),
        }
        return {
            "verified_report": report,
            "public_summary": summary,
            "request_accounting": accounting,
        }

    def rehydrate_pending_acquisition_report(
        self, *, attempt_id: str
    ) -> Any:
        self.rehydrate_calls.append(attempt_id)
        assert self.calls == [DEVELOPMENT_ACQUISITION_COMMAND]
        assert self._recovery_report is not None
        assert self._recovery_report["attempt_id"] == attempt_id
        return self._recovery_report


def _fake_verified_publication(
    manifest: dict[str, Any],
    *,
    attempt_id: str,
    terminal_status: str,
    report_kind: str,
    artifact_sha256: str,
    predecessor_publication_sha256: str,
) -> Any:
    message = build_external_tag_message(
        implementation_manifest=manifest,
        attempt_id=attempt_id,
        terminal_status=terminal_status,
        report_kind=report_kind,
        artifact_sha256=artifact_sha256,
        predecessor_publication_sha256=(
            predecessor_publication_sha256
        ),
    )
    tag_ref = EXTERNAL_TAG_REF_TEMPLATE.format(
        attempt_id=attempt_id,
        report_kind=report_kind,
        artifact_sha256=artifact_sha256,
    )
    publication = build_external_publication(
        implementation_manifest=manifest,
        tag_message=message,
        tag_ref=tag_ref,
        remote_name="origin",
        remote_url=manifest["origin_url"],
        remote_tag_object_sha1="2" * 40,
        remote_peeled_commit=manifest["implementation_commit"],
    )
    return _issue_verified_external_publication(
        publication,
        implementation_manifest=manifest,
    )


class FakePublicationReadback:
    def __init__(
        self,
        *,
        evidence: dict[str, Any],
        observation: dict[str, Any] | None,
        observed_ref_state: str,
    ) -> None:
        self.evidence = copy.deepcopy(evidence)
        self.observation = copy.deepcopy(observation)
        self.observed_ref_state = observed_ref_state


class FakePushReceipt:
    def __init__(self, push_command_sha256: str) -> None:
        self.push_command_sha256 = push_command_sha256


class FakePublisher:
    def __init__(
        self,
        *,
        remote_states: list[str | None] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.read_calls: list[str] = []
        self.push_calls: list[str] = []
        self.manifest: dict[str, Any] | None = None
        self.remote_states = (
            None
            if remote_states is None
            else list(remote_states)
        )

    def publish(
        self,
        *,
        attempt_id: str,
        terminal_status: str,
        report_kind: str,
        artifact_sha256: str,
        predecessor_publication_sha256: str,
        deadline_monotonic: float,
    ) -> Any:
        del deadline_monotonic
        assert self.manifest is not None
        self.calls.append(artifact_sha256)
        return _fake_verified_publication(
            self.manifest,
            attempt_id=attempt_id,
            terminal_status=terminal_status,
            report_kind=report_kind,
            artifact_sha256=artifact_sha256,
            predecessor_publication_sha256=(
                predecessor_publication_sha256
            ),
        )

    def read_remote(self, **kwargs: Any) -> FakePublicationReadback:
        prepared = kwargs["prepared_publication"]
        expected = prepared.expected_publication
        self.read_calls.append(kwargs["observation_phase"])
        state = (
            "exact_expected"
            if self.remote_states is None
            else self.remote_states.pop(0)
        )
        evidence_body = {
            "attempt_id": expected["attempt_id"],
            "observation_ordinal": kwargs["observation_ordinal"],
            "observation_phase": kwargs["observation_phase"],
        }
        evidence = {
            **evidence_body,
            "publication_readback_evidence_sha256": canonical_sha256(
                evidence_body
            ),
        }
        if state is None:
            return FakePublicationReadback(
                evidence=evidence,
                observation=None,
                observed_ref_state="unknown",
            )
        if state == "exact_expected":
            published = self.publish(
                attempt_id=expected["attempt_id"],
                terminal_status=expected["terminal_status"],
                report_kind=expected["report_kind"],
                artifact_sha256=expected["artifact_sha256"],
                predecessor_publication_sha256=expected[
                    "predecessor_publication_sha256"
                ],
                deadline_monotonic=kwargs["timeout_seconds"],
            )
            if not hasattr(published, "publication"):
                return FakePublicationReadback(
                    evidence=evidence,
                    observation=None,
                    observed_ref_state="unknown",
                )
        observation_body = {
            "attempt_id": expected["attempt_id"],
            "tag_ref": prepared.tag_ref,
            "expected_tag_object_sha1": (
                prepared.expected_tag_object_sha1
            ),
            "observation_operation_kind": kwargs[
                "operation_kind"
            ],
            "observation_operation_sha256": kwargs[
                "operation_sha256"
            ],
            "worker_ownership_sha256": kwargs[
                "worker_ownership"
            ].as_dict()["worker_ownership_sha256"],
            "observation_ordinal": kwargs[
                "observation_ordinal"
            ],
            "observation_phase": kwargs["observation_phase"],
            "observed_ref_state": state,
            "remote_tag_object_sha1": expected[
                "remote_tag_object_sha1"
            ],
            "remote_peeled_commit": expected[
                "remote_peeled_commit"
            ],
        }
        observation = {
            **observation_body,
            "publication_remote_observation_sha256": canonical_sha256(
                observation_body
            ),
        }
        return FakePublicationReadback(
            evidence=evidence,
            observation=observation,
            observed_ref_state=state,
        )

    def push_once(self, **kwargs: Any) -> FakePushReceipt:
        artifact_sha256 = kwargs[
            "prepared_publication"
        ].expected_publication["artifact_sha256"]
        self.push_calls.append(artifact_sha256)
        return FakePushReceipt(
            _digest(f"fake-push-command:{len(self.push_calls)}")
        )


class FakePublicationWorkerHandle:
    def __init__(self, *, attempt_id: str) -> None:
        self._transport_root = (
            WORKSPACE_ROOT / ".test-publication-transport"
        )
        self._source_object_directory = WORKSPACE_ROOT / ".git"
        self._executable_pins = {
            "fake_executable_sha256": _digest("fake-executable")
        }
        self._host_environment_values = {
            "SystemRoot": "C:\\Windows"
        }
        ownership_body = {
            "attempt_id": attempt_id,
            "owner_pid": 1,
        }
        self._ownership_material = {
            **ownership_body,
            "worker_ownership_sha256": canonical_sha256(
                ownership_body
            ),
        }
        self.aborted = False
        self.quiesced = False

    @property
    def transport_root(self) -> Path:
        return self._transport_root

    @property
    def source_object_directory(self) -> Path:
        return self._source_object_directory

    @property
    def executable_pins(self) -> dict[str, Any]:
        return copy.deepcopy(self._executable_pins)

    @property
    def host_environment_values(self) -> dict[str, str]:
        return copy.deepcopy(self._host_environment_values)

    @property
    def ownership_material(self) -> dict[str, Any]:
        return copy.deepcopy(self._ownership_material)

    def verify_executable_pins(
        self, pins: dict[str, Any]
    ) -> None:
        assert pins == self._executable_pins

    def abort_uncommitted(self) -> None:
        self.aborted = True

    def quiesce(
        self, *, worker_ownership_sha256: str
    ) -> dict[str, Any]:
        assert worker_ownership_sha256 == self._ownership_material[
            "worker_ownership_sha256"
        ]
        self.quiesced = True
        body = {
            "worker_ownership_sha256": worker_ownership_sha256,
            "worker_quiescent": True,
        }
        return {
            **body,
            "worker_quiescence_sha256": canonical_sha256(body),
        }


class FakePublicationRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.handles: list[FakePublicationWorkerHandle] = []

    def begin(self, **kwargs: Any) -> FakePublicationWorkerHandle:
        self.calls.append(kwargs["attempt_id"])
        handle = FakePublicationWorkerHandle(
            attempt_id=kwargs["attempt_id"]
        )
        self.handles.append(handle)
        return handle


class FakePreparedPublicationTransport:
    def __init__(
        self,
        *,
        prepared_publication: Any,
        pre_transport_store_journal_sequence: int,
        pre_transport_store_journal_tip_sha256: str,
    ) -> None:
        body = {
            "attempt_id": prepared_publication.expected_publication[
                "attempt_id"
            ],
            "tag_ref": prepared_publication.tag_ref,
            "pre_transport_store_journal_sequence": (
                pre_transport_store_journal_sequence
            ),
            "pre_transport_store_journal_tip_sha256": (
                pre_transport_store_journal_tip_sha256
            ),
        }
        self.manifest = {
            **body,
            "publication_transport_manifest_sha256": canonical_sha256(
                body
            ),
        }


@pytest.fixture(autouse=True)
def _fake_publication_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prepare_transport(**kwargs: Any) -> FakePreparedPublicationTransport:
        kwargs["identity_verifier"](kwargs["executable_pins"])
        return FakePreparedPublicationTransport(
            prepared_publication=kwargs["prepared_publication"],
            pre_transport_store_journal_sequence=kwargs[
                "pre_transport_store_journal_sequence"
            ],
            pre_transport_store_journal_tip_sha256=kwargs[
                "pre_transport_store_journal_tip_sha256"
            ],
        )

    def issue_publication(**kwargs: Any) -> Any:
        prepared = kwargs["prepared_publication"]
        return _issue_verified_external_publication(
            prepared.expected_publication,
            implementation_manifest=prepared.implementation_manifest,
        )

    monkeypatch.setattr(
        runner_module,
        "prepare_isolated_publication_transport",
        prepare_transport,
    )
    monkeypatch.setattr(
        runner_module,
        "issue_verified_external_publication_from_observation",
        issue_publication,
    )


class FakeFinalRegistryAuthority:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def authorize_final(self, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


class FakeTerminalEvidence:
    def __init__(
        self,
        *,
        evidence: dict[str, Any],
        artifact_payload: dict[str, Any],
        artifact_receipt: StoreRecordReceipt,
        external_publication: Any,
    ) -> None:
        self.evidence = copy.deepcopy(evidence)
        self.artifact_payload = copy.deepcopy(artifact_payload)
        self.artifact_receipt = artifact_receipt
        self.external_publication = external_publication


def _store_receipt_material(
    receipt: StoreRecordReceipt,
) -> dict[str, Any]:
    return {
        field: getattr(receipt, field)
        for field in (
            "table",
            "identity",
            "attempt_id",
            "payload_sha256",
            "journal_sequence",
            "journal_entry_sha256",
        )
    }


def _issue_fake_acquisition_terminal_evidence(
    **kwargs: Any,
) -> FakeTerminalEvidence:
    report = copy.deepcopy(dict(kwargs["acquisition_report"]))
    material = copy.deepcopy(kwargs["report_material"])
    receipt = kwargs["acquisition_artifact_receipt"]
    publication_intent = kwargs["publication_intent"]
    publication_receipt = kwargs["publication_receipt"]
    publication = publication_receipt.external_publication
    receipt_material = _store_receipt_material(receipt)
    intent_material = publication_intent.as_dict()
    durable_receipt_material = publication_receipt.as_dict()
    body = {
        "schema_version": (
            "aapl-sec-gemma-online-risk-overlay-v2-2-"
            "acquisition-terminal-evidence-v1"
        ),
        "verifier_id": (
            "aapl-sec-gemma-online-risk-overlay-v2-2-"
            "acquisition-terminal-evidence-verifier-v1"
        ),
        "verdict": "pass",
        "terminal_status": TERMINAL_PASS,
        "stage": "development",
        "attempt_id": report["attempt_id"],
        "attempt_kind": report["attempt_kind"],
        "attempt_plan_sha256": kwargs["attempt_plan"][
            "attempt_plan_sha256"
        ],
        "acquisition_validation_sha256": report["validation_sha256"],
        "bundle_sha256": report["bundle_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "private_index_sha256": report["private_index_sha256"],
        "check_set_sha256": report["check_set_sha256"],
        "record_commitment_sha256": material[
            "record_commitment_sha256"
        ],
        "acquisition_artifact_receipt_sha256": canonical_sha256(
            receipt_material
        ),
        "publication_intent_sha256": intent_material[
            "publication_intent_sha256"
        ],
        "publication_intent_store_receipt_sha256": canonical_sha256(
            _store_receipt_material(publication_intent.store_receipt)
        ),
        "publication_receipt_sha256": durable_receipt_material[
            "publication_receipt_sha256"
        ],
        "publication_receipt_store_receipt_sha256": canonical_sha256(
            _store_receipt_material(publication_receipt.store_receipt)
        ),
        "external_publication_sha256": (
            publication.publication_sha256
        ),
    }
    evidence = {
        **body,
        "terminal_evidence_sha256": canonical_sha256(body),
    }
    assert set(evidence) == set(ACQUISITION_TERMINAL_EVIDENCE_FIELDS)
    return FakeTerminalEvidence(
        evidence=evidence,
        artifact_payload=report,
        artifact_receipt=receipt,
        external_publication=publication,
    )


def _issue_fake_scored_terminal_evidence(
    **kwargs: Any,
) -> FakeTerminalEvidence:
    plan = copy.deepcopy(kwargs["attempt_plan"])
    joint = copy.deepcopy(kwargs["joint_stage_report"])
    material = copy.deepcopy(kwargs["report_material"])
    receipt = kwargs["joint_artifact_receipt"]
    gate_checks = copy.deepcopy(kwargs["gate_checks"])
    publication_intent = kwargs["publication_intent"]
    publication_receipt = kwargs["publication_receipt"]
    publication = publication_receipt.external_publication
    evaluation = joint["deterministic_evaluation"]
    failed = [
        name for name, passed in gate_checks.items() if passed is not True
    ]
    terminal_status = (
        TERMINAL_PASS if not failed else TERMINAL_FAIL
    )
    body = {
        "schema_version": (
            "aapl-sec-gemma-online-risk-overlay-v2-2-"
            "scored-terminal-evidence-v1"
        ),
        "verifier_id": (
            "aapl-sec-gemma-online-risk-overlay-v2-2-"
            "scored-terminal-evidence-verifier-v1"
        ),
        "verdict": "pass" if not failed else "failed_gate",
        "terminal_status": terminal_status,
        "stage": joint["metric_stage"],
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "attempt_plan_sha256": plan["attempt_plan_sha256"],
        "stage_input_bundle_sha256": evaluation[
            "stage_input_bundle_sha256"
        ],
        "deterministic_evaluation_sha256": evaluation[
            "deterministic_evaluation_sha256"
        ],
        "stage_metrics_input_sha256": evaluation[
            "metrics_input_sha256"
        ],
        "stage_metrics_sha256": evaluation[
            "stage_metrics_sha256"
        ],
        "gate_report_sha256": evaluation["gate_report_sha256"],
        "no_leverage_proofs_sha256": evaluation[
            "no_leverage_proofs_sha256"
        ],
        "joint_stage_report_sha256": joint[
            "joint_stage_report_sha256"
        ],
        "record_counts": material["record_counts"],
        "record_commitment_sha256": material[
            "record_commitment_sha256"
        ],
        "joint_artifact_receipt_sha256": canonical_sha256(
            _store_receipt_material(receipt)
        ),
        "gate_checks": gate_checks,
        "gate_check_set_sha256": canonical_sha256(gate_checks),
        "failed_gate_names": failed,
        "publication_intent_sha256": publication_intent.as_dict()[
            "publication_intent_sha256"
        ],
        "publication_intent_store_receipt_sha256": canonical_sha256(
            _store_receipt_material(publication_intent.store_receipt)
        ),
        "publication_receipt_sha256": publication_receipt.as_dict()[
            "publication_receipt_sha256"
        ],
        "publication_receipt_store_receipt_sha256": canonical_sha256(
            _store_receipt_material(publication_receipt.store_receipt)
        ),
        "external_publication_sha256": (
            publication.publication_sha256
        ),
    }
    evidence = {
        **body,
        "terminal_evidence_sha256": canonical_sha256(body),
    }
    assert set(evidence) == set(SCORED_TERMINAL_EVIDENCE_FIELDS)
    return FakeTerminalEvidence(
        evidence=evidence,
        artifact_payload=joint,
        artifact_receipt=receipt,
        external_publication=publication,
    )


def _dependencies(
    manifest: dict[str, Any],
    evaluator: FakeEvaluator | None = None,
    runtime_verifier: Any | None = None,
) -> RunnerDependencies:
    return RunnerDependencies(
        live_implementation_manifest=lambda _root: copy.deepcopy(
            manifest
        ),
        runtime_verifier=(
            runtime_verifier
            if runtime_verifier is not None
            else lambda payload: {
                "runtime_probe_payload_sha256": canonical_sha256(
                    payload
                )
            }
        ),
        evaluator=evaluator or FakeEvaluator(),
        issue_acquisition_terminal_evidence_fn=(
            _issue_fake_acquisition_terminal_evidence
        ),
        issue_scored_terminal_evidence_fn=(
            _issue_fake_scored_terminal_evidence
        ),
    )


def _runner(
    manifest: dict[str, Any],
    store: FakeStore,
    executor: FakeExecutor,
    publisher: FakePublisher,
    *,
    clock: Any,
    evaluator: FakeEvaluator | None = None,
    runtime_verifier: Any | None = None,
    acquisition_adapter: FakeAcquisitionAdapter | None = None,
    test_only_allow_effects: bool = True,
    final_registry_authority: FakeFinalRegistryAuthority | None = None,
    publication_runtime: FakePublicationRuntime | None = None,
    production_authorities: Any | None = None,
) -> SecGemmaOnlineRiskOverlayRunner:
    publisher.manifest = manifest
    return SecGemmaOnlineRiskOverlayRunner(
        repo_root=WORKSPACE_ROOT,
        implementation_manifest=manifest,
        store=store,
        phase_executor=executor,
        acquisition_adapter=(
            acquisition_adapter
            if acquisition_adapter is not None
            else FakeAcquisitionAdapter(clock)
        ),
        report_publisher=publisher,
        publication_runtime=(
            publication_runtime
            if publication_runtime is not None
            else FakePublicationRuntime()
        ),
        final_registry_authority=final_registry_authority,
        production_authorities=production_authorities,
        clock=clock,
        dependencies=_dependencies(
            manifest,
            evaluator,
            runtime_verifier,
        ),
        test_only_allow_effects=test_only_allow_effects,
    )


def _seed_confirmation_predecessor(store: FakeStore) -> None:
    acquisition = store.seed_pass(DEVELOPMENT_ACQUISITION, None)
    store.seed_pass(DEVELOPMENT_SCORING, acquisition)


def _seed_final_predecessor(store: FakeStore) -> None:
    _seed_confirmation_predecessor(store)
    development = store.histories[
        ATTEMPT_ID_BY_KIND[DEVELOPMENT_SCORING]
    ][-1]
    store.seed_pass(CONFIRMATION_SCORING, development)


def _final_registry_authorization(
    **overrides: Any,
) -> dict[str, Any]:
    access = build_contract_manifest()["stage_access"]["final"]
    body = {
        "predecessor_registry_pin_file_sha256": access[
            "predecessor_registry_pin_file_sha256"
        ],
        "predecessor_registry_sha256": access[
            "predecessor_registry_sha256"
        ],
        "predecessor_registry_tip_sha256": access[
            "predecessor_registry_tip_sha256"
        ],
        "historical_final_reveal_count": access[
            "historical_final_reveal_count_lower_bound"
        ],
        "successor_registered": True,
        "successor_externally_pinned": True,
        "successor_pin_sha256": _digest("final-successor-pin"),
    }
    body.update(overrides)
    return {
        **body,
        "authorization_sha256": canonical_sha256(body),
    }


def test_runner_plan_is_exact_and_self_hashed() -> None:
    manifest = _manifest()
    plan = build_runner_plan(
        command=DEVELOPMENT_ACQUISITION_COMMAND,
        implementation_manifest=manifest,
    )

    assert validate_runner_plan(
        plan, implementation_manifest=manifest
    ) == plan
    assert plan["phases"][0]["phase"] == "acquisition"
    assert plan["market_requests_exactly"] == 6
    assert plan["total_seconds_strictly_below"] == 3600

    changed = copy.deepcopy(plan)
    changed["market_requests_exactly"] = 5
    changed["runner_plan_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "runner_plan_sha256"
        }
    )
    with pytest.raises(SecGemmaOnlineRiskOverlayRunnerError):
        validate_runner_plan(changed, implementation_manifest=manifest)


def test_attempt_is_consumed_before_effect_and_releases_no_scores() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest, store, executor, publisher, clock=clock
    )

    result = runner.run(DEVELOPMENT_ACQUISITION_COMMAND)

    attempt_id = ATTEMPT_ID_BY_KIND[DEVELOPMENT_ACQUISITION]
    consume_position = store.events.index(f"consume:{attempt_id}")
    first_authorize = next(
        index
        for index, event in enumerate(store.events)
        if event.startswith("authorize:")
    )
    assert consume_position < first_authorize
    assert result["passed"] is True
    assert result["terminal_artifact"]["verdict"] == "pass"
    assert result["terminal_evidence"]["stage"] == "development"
    assert publisher.calls == [
        result["terminal_artifact"]["validation_sha256"]
    ]


def test_production_mode_refuses_generic_adapters_before_registration() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        test_only_allow_effects=False,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="production authorities",
    ):
        runner.run(DEVELOPMENT_ACQUISITION_COMMAND)

    assert store.events == []
    assert executor.events == []


@pytest.mark.parametrize(
    "worker_authority",
    [None, object()],
)
def test_production_recovery_cannot_run_without_opaque_supervised_worker(
    worker_authority: Any,
) -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        test_only_allow_effects=False,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="opaque supervised-worker authority",
    ):
        runner.recover_publication(
            attempt_id,
            supervised_worker_authority=worker_authority,
        )

    assert store.events == []
    assert publisher.read_calls == []
    assert publisher.push_calls == []


def test_test_recovery_cannot_claim_supervised_worker_authority() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="Test publication recovery cannot claim",
    ):
        runner.recover_publication(
            ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING],
            supervised_worker_authority=object(),
        )

    assert store.events == []


def test_supervised_worker_authority_binds_source_composition_and_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    clock.value = 10.0
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    production_authorities = object()
    calls: list[dict[str, Any]] = []

    class FakeExactWorkerAuthority:
        def authorize_runner_recovery(
            self, **kwargs: Any
        ) -> tuple[float, float]:
            calls.append(kwargs)
            return 5.0, 305.0

    worker_authority = FakeExactWorkerAuthority()
    monkeypatch.setattr(
        production_module,
        "is_verified_supervised_publication_recovery_worker_authority",
        lambda value: value is worker_authority,
        raising=False,
    )
    entered: list[tuple[str, float, float]] = []

    def enter_worker(
        self: SecGemmaOnlineRiskOverlayRunner,
        attempt_id: str,
        *,
        recovery_started_at: float,
        invocation_deadline: float,
    ) -> dict[str, Any]:
        entered.append(
                (
                    attempt_id,
                    recovery_started_at,
                    invocation_deadline,
                )
        )
        return {"sealed": True}

    monkeypatch.setattr(
        SecGemmaOnlineRiskOverlayRunner,
        "_recover_publication_in_supervised_worker",
        enter_worker,
    )
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        test_only_allow_effects=False,
        production_authorities=production_authorities,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    result = runner.recover_publication(
        attempt_id,
        supervised_worker_authority=worker_authority,
    )

    assert result == {"sealed": True}
    assert len(calls) == 1
    assert calls[0]["repo_root"] == WORKSPACE_ROOT
    assert calls[0]["implementation_manifest"] == manifest
    assert (
        calls[0]["production_authorities"]
        is production_authorities
    )
    assert calls[0]["attempt_id"] == attempt_id
    assert calls[0]["worker_entry_monotonic"] == 10.0
    assert calls[0]["worker_deadline_monotonic"] == 310.0
    assert entered == [(attempt_id, 5.0, 305.0)]


def test_supervised_worker_authority_cannot_extend_300_second_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()

    class ExtendingWorkerAuthority:
        def authorize_runner_recovery(
            self, **kwargs: Any
        ) -> tuple[float, float]:
            return (
                kwargs["worker_entry_monotonic"],
                kwargs["worker_deadline_monotonic"] + 1.0,
            )

    worker_authority = ExtendingWorkerAuthority()
    monkeypatch.setattr(
        production_module,
        "is_verified_supervised_publication_recovery_worker_authority",
        lambda value: value is worker_authority,
        raising=False,
    )
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        test_only_allow_effects=False,
        production_authorities=object(),
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="window is invalid",
    ):
        runner.recover_publication(
            ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING],
            supervised_worker_authority=worker_authority,
        )

    assert store.events == []


def test_recovery_elapsed_includes_pre_child_supervisor_time() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    publisher = FakePublisher(remote_states=[None, None])
    runner = _runner(
        manifest,
        store,
        FakeExecutor(store, clock),
        publisher,
        clock=clock,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    initial = runner.run(CONFIRMATION_COMMAND)
    assert initial["result_status"] == "publication_pending"
    clock.value = 10.0

    recovered = runner._recover_publication_in_supervised_worker(
        attempt_id,
        recovery_started_at=5.0,
        invocation_deadline=305.0,
    )

    assert recovered["result_status"] == "publication_pending"
    completions = store.governance_records(
        "publication_recovery_invocation_completions",
        attempt_id=attempt_id,
    )
    assert len(completions) == 1
    assert float.fromhex(completions[0]["elapsed_seconds"]) == 5.0


def test_production_readiness_precedes_registration_and_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    executor = FakeExecutor(store, time.monotonic)
    adapter = FakeAcquisitionAdapter(time.monotonic)
    publisher = FakePublisher()
    publisher.manifest = manifest
    publication_runtime = FakePublicationRuntime()

    class FakeExactProductionAuthorities:
        def __init__(self) -> None:
            self.phase_executor = executor
            self.acquisition_adapter = adapter
            self.report_publisher = publisher
            self.publication_runtime = publication_runtime
            self.final_registry_authority = None
            self.store = store
            self.ready_calls: list[str] = []

        def reverify(self) -> None:
            return None

        def assert_ready_for_command(self, command: str) -> None:
            self.ready_calls.append(command)
            if len(self.ready_calls) == 3:
                raise RuntimeError("retained acquisition vanished")
            return None

    authority = FakeExactProductionAuthorities()
    monkeypatch.setattr(
        production_module,
        "is_verified_production_authorities",
        lambda value: value is authority,
    )
    monkeypatch.setattr(
        runner_module,
        "_default_live_implementation_manifest",
        lambda _root: copy.deepcopy(manifest),
    )
    runner = SecGemmaOnlineRiskOverlayRunner(
        repo_root=WORKSPACE_ROOT,
        implementation_manifest=manifest,
        store=store,
        phase_executor=executor,
        acquisition_adapter=adapter,
        report_publisher=publisher,
        publication_runtime=publication_runtime,
        production_authorities=authority,
        clock=time.monotonic,
        test_only_allow_effects=False,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="restart-safe",
    ):
        runner.run(DEVELOPMENT_ACQUISITION_COMMAND)

    attempt_id = ATTEMPT_ID_BY_KIND[DEVELOPMENT_ACQUISITION]
    assert authority.ready_calls == [
        DEVELOPMENT_ACQUISITION_COMMAND,
        DEVELOPMENT_ACQUISITION_COMMAND,
        DEVELOPMENT_ACQUISITION_COMMAND,
    ]
    assert store.events == [f"register:{attempt_id}"]
    assert executor.events == []


def test_production_readiness_precedes_final_registry_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    executor = FakeExecutor(store, time.monotonic)
    adapter = FakeAcquisitionAdapter(time.monotonic)
    publisher = FakePublisher()
    publisher.manifest = manifest
    publication_runtime = FakePublicationRuntime()

    class RecordingRegistry:
        def __init__(self) -> None:
            self.calls = 0

        def authorize_final(self, **_kwargs: Any) -> None:
            self.calls += 1
            return None

    registry = RecordingRegistry()

    class NotReadyProductionAuthorities:
        def __init__(self) -> None:
            self.phase_executor = executor
            self.acquisition_adapter = adapter
            self.report_publisher = publisher
            self.publication_runtime = publication_runtime
            self.final_registry_authority = registry
            self.store = store

        def reverify(self) -> None:
            return None

        def assert_ready_for_command(self, command: str) -> None:
            assert command == FINAL_COMMAND
            raise RuntimeError("fresh bundle has no predecessors")

    authority = NotReadyProductionAuthorities()
    monkeypatch.setattr(
        production_module,
        "is_verified_production_authorities",
        lambda value: value is authority,
    )
    runner = SecGemmaOnlineRiskOverlayRunner(
        repo_root=WORKSPACE_ROOT,
        implementation_manifest=manifest,
        store=store,
        phase_executor=executor,
        acquisition_adapter=adapter,
        report_publisher=publisher,
        publication_runtime=publication_runtime,
        final_registry_authority=registry,
        production_authorities=authority,
        clock=time.monotonic,
        test_only_allow_effects=False,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="restart-safe",
    ):
        runner.run(FINAL_COMMAND)

    assert registry.calls == 0
    assert store.events == []


def test_local_preflight_reports_effectful_production_blockers() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        test_only_allow_effects=False,
    )

    result = runner.run_local_preflight()

    assert result["passed"] is False
    assert result["effectful_production_ready"] is False
    assert result["production_effect_blockers"] == list(
        PRODUCTION_EFFECT_BLOCKERS
    )
    assert result["production_effect_blockers"] == [
        "implementation_must_be_committed_clean_and_pushed_to_frozen_origin",
        (
            "exact_verified_production_authorities_must_be_supplied_before_"
            "registration"
        ),
        (
            "live_sec_yahoo_loopback_ollama_and_origin_tag_publication_are_"
            "not_exercised_by_local_preflight"
        ),
    ]
    assert store.events == []


def test_runtime_identity_is_verified_before_gemma() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    acquisition = store.seed_pass(DEVELOPMENT_ACQUISITION, None)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    verifier_calls: list[dict[str, Any]] = []

    def runtime_verifier(payload: dict[str, Any]) -> dict[str, Any]:
        assert executor.events == ["runtime_identity"]
        verifier_calls.append(copy.deepcopy(payload))
        return {
            "runtime_probe_payload_sha256": canonical_sha256(payload)
        }

    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        runtime_verifier=runtime_verifier,
    )
    result = runner.run(DEVELOPMENT_COMMAND)

    assert acquisition["status"] == TERMINAL_PASS
    assert result["passed"] is True
    assert len(verifier_calls) == 1
    assert executor.events == [
        "runtime_identity",
        "gemma",
        "deterministic",
    ]


def test_runtime_identity_failure_prevents_gemma_calls() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    store.seed_pass(DEVELOPMENT_ACQUISITION, None)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()

    def runtime_verifier(_payload: dict[str, Any]) -> dict[str, Any]:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "pinned runtime identity changed"
        )

    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        runtime_verifier=runtime_verifier,
    )
    result = runner.run(DEVELOPMENT_COMMAND)

    assert result["passed"] is False
    assert result["terminal_status"] == TERMINAL_INDETERMINATE
    assert result["diagnostic_code"] == "post_consumption_indeterminate"
    assert result["terminal_evidence"] is None
    assert executor.events == ["runtime_identity"]
    assert publisher.calls == []


def test_later_stage_predictions_inherit_all_matured_feature_history() -> None:
    class ContinuityEvaluator(FakeEvaluator):
        def evaluate(
            self,
            *,
            stage: str,
            input_bundle: dict[str, Any],
        ) -> dict[str, Any]:
            evaluation = _evaluation(stage, input_bundle)
            feature_hashes = input_bundle["feature_row_sha256s"]
            prior_hashes = input_bundle[
                "predecessor_feature_row_sha256s"
            ]
            lessons = []
            for digest in prior_hashes:
                lesson_body = {
                    "source_feature_row_sha256": digest,
                    "admitted_before_current_prediction": True,
                }
                lessons.append(
                    {
                        **lesson_body,
                        "lesson_row_sha256": canonical_sha256(
                            lesson_body
                        ),
                    }
                )
            prediction_body = {
                "decision_feature_row_sha256": feature_hashes[-1],
                "admitted_lesson_feature_row_sha256s": prior_hashes,
                "training_feature_row_sha256s": feature_hashes[:-1],
            }
            prediction = {
                **prediction_body,
                "prediction_row_sha256": canonical_sha256(
                    prediction_body
                ),
            }
            online = evaluation["replays"]["semantic"]["online"]
            online["learner_lessons"] = lessons
            online["primary"]["predictions"] = [prediction]
            online["chronological_replay_sha256"] = canonical_sha256(
                {
                    "stage": stage,
                    "feature_row_sha256s": feature_hashes,
                    "learner_lessons": lessons,
                    "predictions": [prediction],
                }
            )
            body = {
                key: value
                for key, value in evaluation.items()
                if key != "deterministic_evaluation_sha256"
            }
            body["replays_sha256"] = canonical_sha256(
                body["replays"]
            )
            return {
                **body,
                "deterministic_evaluation_sha256": canonical_sha256(
                    body
                ),
            }

    def outputs(command: str) -> dict[str, dict[str, Any]]:
        return {
            phase: build_phase_output(
                command=command,
                phase=phase,
                counters=_phase_counters(command, phase),
                payload=_phase_payload(command, phase),
            )
            for phase in ("gemma", "deterministic")
        }

    evaluator = ContinuityEvaluator()
    development = runner_module._stage_input_bundle(
        stage="development",
        outputs=outputs(DEVELOPMENT_COMMAND),
    )
    development_row = development["feature_rows"][0]
    confirmation = runner_module._stage_input_bundle(
        stage="confirmation",
        outputs=outputs(CONFIRMATION_COMMAND),
        predecessor_feature_rows=[development_row],
    )
    confirmation_row = next(
        row
        for row in confirmation["feature_rows"]
        if row["feature_row_sha256"]
        in confirmation["current_feature_row_sha256s"]
    )
    final = runner_module._stage_input_bundle(
        stage="final",
        outputs=outputs(FINAL_COMMAND),
        predecessor_feature_rows=[
            development_row,
            confirmation_row,
        ],
    )

    confirmation_evaluation = evaluator.evaluate(
        stage="confirmation",
        input_bundle=confirmation,
    )
    final_evaluation = evaluator.evaluate(
        stage="final",
        input_bundle=final,
    )
    confirmation_prediction = confirmation_evaluation["replays"][
        "semantic"
    ]["online"]["primary"]["predictions"][0]
    final_prediction = final_evaluation["replays"]["semantic"][
        "online"
    ]["primary"]["predictions"][0]

    assert confirmation["feature_row_sha256s"] == [
        development_row["feature_row_sha256"],
        confirmation_row["feature_row_sha256"],
    ]
    assert confirmation_prediction[
        "admitted_lesson_feature_row_sha256s"
    ] == [development_row["feature_row_sha256"]]
    assert final["feature_row_sha256s"] == [
        development_row["feature_row_sha256"],
        confirmation_row["feature_row_sha256"],
        final["current_feature_row_sha256s"][0],
    ]
    assert final_prediction[
        "admitted_lesson_feature_row_sha256s"
    ] == [
        development_row["feature_row_sha256"],
        confirmation_row["feature_row_sha256"],
    ]


def test_confirmation_success_is_jointly_sealed_and_verifiable() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest, store, executor, publisher, clock=clock
    )

    result = runner.run(CONFIRMATION_COMMAND)
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]
    plan = store.plans[attempt_id]

    assert result["passed"] is True
    assert result["terminal_status"] == TERMINAL_PASS
    assert result["terminal_evidence"]["gate_checks"] == result[
        "terminal_artifact"
    ]["deterministic_evaluation"]["gate_report"]["checks"]
    budget = result["terminal_artifact"]["phase_budget_report"]
    assert budget["market_elapsed_seconds_hex"] == (1.0).hex()
    acquisition_receipt = next(
        receipt
        for receipt in budget["phase_receipts"]
        if receipt["phase"] == "acquisition"
    )
    assert acquisition_receipt["market_elapsed_seconds_hex"] == (
        1.0
    ).hex()
    gemma_evidence = next(
        item["payload"]
        for item in store.records
        if item["receipt"].table == "evidence"
        and item["receipt"].identity
        == f"phase:{attempt_id}:gemma"
        and item["receipt"].attempt_id == attempt_id
    )
    assert gemma_evidence["semantic_payload"] == _phase_payload(
        CONFIRMATION_COMMAND,
        "gemma",
    )
    latency = gemma_evidence["latency_preflight_receipt"]
    assert gemma_evidence[
        "latency_preflight_receipt_sha256"
    ] == latency["latency_preflight_receipt_sha256"]
    assert validate_sealed_stage_result(
        result,
        implementation_manifest=manifest,
        attempt_plan=plan,
    ) == result
    verified = runner.verify(sealed_result=result, attempt_plan=plan)
    assert verified["passed"] is True


def test_budget_report_includes_parent_deterministic_evaluation_time() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()

    class TimedEvaluator(FakeEvaluator):
        def evaluate(
            self,
            *,
            stage: str,
            input_bundle: dict[str, Any],
        ) -> dict[str, Any]:
            clock.value += 20.0
            return super().evaluate(
                stage=stage,
                input_bundle=input_bundle,
            )

        def validate(
            self,
            evaluation: dict[str, Any],
            *,
            stage: str,
            input_bundle: dict[str, Any],
        ) -> dict[str, Any]:
            clock.value += 10.0
            return super().validate(
                evaluation,
                stage=stage,
                input_bundle=input_bundle,
            )

    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        evaluator=TimedEvaluator(),
    )

    result = runner.run(CONFIRMATION_COMMAND)

    budget = result["terminal_artifact"]["phase_budget_report"]
    groups = {
        key: float.fromhex(value)
        for key, value in budget["group_elapsed_seconds_hex"].items()
    }
    total_bound = float.fromhex(
        budget["total_elapsed_seconds_hex"]
    )
    contingency = build_contract_manifest()["runtime"][
        "contingency_seconds"
    ]
    assert groups["deterministic"] == 32.0
    assert total_bound == sum(groups.values()) + contingency
    assert total_bound > 4.0


def test_publication_crossing_its_90_second_partition_is_pending() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)

    class SlowPublisher(FakePublisher):
        def publish(self, **kwargs: Any) -> Any:
            result = super().publish(**kwargs)
            clock.value += 100.0
            return result

    publisher = SlowPublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )

    result = runner.run(CONFIRMATION_COMMAND)
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    assert publisher.calls
    assert result["result_status"] == "publication_pending"
    assert result["semantic_result_released"] is False
    assert result["next_stage_authority_blocked"] is True
    assert result["publication_recovery_required"] is True
    assert store.histories[attempt_id][-1]["status"] == CONSUMED
    assert attempt_id not in store.anchor_bindings


def test_publication_command_timeout_preserves_quiescence_reserve() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    runner = _runner(
        manifest,
        store,
        FakeExecutor(store, clock),
        FakePublisher(),
        clock=clock,
    )

    clock.value = 100.0
    assert runner._publication_timeout(200.0, "test publication") == 85.0
    assert runner._publication_timeout(500.0, "test publication") == 300.0

    clock.value = 185.0
    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerIndeterminate,
        match="lacks the frozen publication quiescence reserve",
    ):
        runner._publication_timeout(200.0, "test publication")


def test_normal_publication_timeout_quiesces_inside_its_partition() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()

    class DeadlinePublisher(FakePublisher):
        def __init__(self) -> None:
            super().__init__(remote_states=[None])
            self.command_timeouts: list[float] = []

        def read_remote(self, **kwargs: Any) -> FakePublicationReadback:
            timeout = kwargs["timeout_seconds"]
            self.command_timeouts.append(timeout)
            clock.value += timeout
            return super().read_remote(**kwargs)

    class CleanupHandle(FakePublicationWorkerHandle):
        def quiesce(
            self, *, worker_ownership_sha256: str
        ) -> dict[str, Any]:
            result = super().quiesce(
                worker_ownership_sha256=worker_ownership_sha256
            )
            clock.value += 14.0
            return result

    class CleanupRuntime(FakePublicationRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.publication_started_at: float | None = None

        def begin(self, **kwargs: Any) -> FakePublicationWorkerHandle:
            self.calls.append(kwargs["attempt_id"])
            self.publication_started_at = clock.value
            handle = CleanupHandle(attempt_id=kwargs["attempt_id"])
            self.handles.append(handle)
            return handle

    publisher = DeadlinePublisher()
    runtime = CleanupRuntime()
    runner = _runner(
        manifest,
        store,
        FakeExecutor(store, clock),
        publisher,
        clock=clock,
        publication_runtime=runtime,
    )

    result = runner.run(CONFIRMATION_COMMAND)

    assert result["result_status"] == "publication_pending"
    assert publisher.command_timeouts == [75.0]
    assert runtime.handles[0].quiesced is True
    assert runtime.publication_started_at is not None
    assert clock.value - runtime.publication_started_at == 89.0


def test_publication_does_not_spawn_inside_quiescence_reserve() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()

    class ReserveExhaustingRuntime(FakePublicationRuntime):
        def begin(self, **kwargs: Any) -> FakePublicationWorkerHandle:
            handle = super().begin(**kwargs)
            clock.value += 75.0
            return handle

    publisher = FakePublisher(remote_states=[None])
    runtime = ReserveExhaustingRuntime()
    runner = _runner(
        manifest,
        store,
        FakeExecutor(store, clock),
        publisher,
        clock=clock,
        publication_runtime=runtime,
    )

    result = runner.run(CONFIRMATION_COMMAND)

    assert result["result_status"] == "publication_pending"
    assert publisher.read_calls == []
    assert publisher.push_calls == []
    assert runtime.handles[0].quiesced is True


def test_intent_commit_cannot_borrow_from_publication_partition() -> None:
    manifest = _manifest()
    clock = MutableClock()

    class SlowIntentStore(FakeStore):
        def commit_publication_intent(
            self,
            capability: FakeCapability,
            material: dict[str, Any],
        ) -> FakeDurableAuthority:
            result = super().commit_publication_intent(
                capability, material
            )
            clock.value += 90.0
            return result

    store = SlowIntentStore(manifest)
    _seed_confirmation_predecessor(store)
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )

    result = runner.run(CONFIRMATION_COMMAND)

    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]
    assert result["result_status"] == "publication_pending"
    assert publisher.read_calls == []
    assert store.histories[attempt_id][-1]["status"] == CONSUMED


def test_pending_handoff_cannot_borrow_past_its_60_second_partition() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)

    class TooSlowPublisher(FakePublisher):
        def publish(self, **kwargs: Any) -> Any:
            result = super().publish(**kwargs)
            clock.value += 240.0
            return result

    publisher = TooSlowPublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerIndeterminate,
        match="exceptional publication-pending handoff",
    ):
        runner.run(CONFIRMATION_COMMAND)

    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]
    assert store.histories[attempt_id][-1]["status"] == CONSUMED
    assert attempt_id not in store.anchor_bindings


def test_parent_deadline_is_checked_after_canonical_result_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )
    original_success_result = runner_module._success_result
    constructed_hashes: list[str] = []

    def delayed_success_result(**kwargs: Any) -> dict[str, Any]:
        result = original_success_result(**kwargs)
        constructed_hashes.append(result["sealed_stage_result_sha256"])
        clock.value += 240.0
        return result

    monkeypatch.setattr(
        runner_module,
        "_success_result",
        delayed_success_result,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerIndeterminate,
        match="strict parent deadline",
    ):
        runner.run(CONFIRMATION_COMMAND)

    assert len(constructed_hashes) == 1
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]
    assert store.histories[attempt_id][-1]["status"] == TERMINAL_PASS


def test_exact_total_runtime_boundary_is_terminal_indeterminate() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = SequenceClock(
        [
            0.0,
            0.0,
            700.0,
            700.0,
            800.0,
            800.0,
            2960.0,
            2960.0,
            3340.0,
            3600.0,
        ]
    )
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest, store, executor, publisher, clock=clock
    )

    result = runner.run(CONFIRMATION_COMMAND)

    assert result["terminal_status"] == TERMINAL_INDETERMINATE
    assert result["terminal_artifact"] is None
    assert publisher.calls == []


def test_combined_acquisition_phase_overrun_is_terminal_indeterminate() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    acquisition = FakeAcquisitionAdapter(clock, duration=721.0)
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        acquisition_adapter=acquisition,
    )

    result = runner.run(DEVELOPMENT_ACQUISITION_COMMAND)

    assert result["terminal_status"] == TERMINAL_INDETERMINATE
    assert result["terminal_evidence"] is None
    assert publisher.calls == []


def test_market_subdeadline_overrun_stops_before_later_phases() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    acquisition = FakeAcquisitionAdapter(
        clock,
        market_elapsed=211.0,
    )
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        acquisition_adapter=acquisition,
    )

    result = runner.run(CONFIRMATION_COMMAND)

    assert result["terminal_status"] == TERMINAL_INDETERMINATE
    assert result["terminal_evidence"] is None
    assert executor.events == []
    assert publisher.calls == []


def test_confirmation_cannot_bypass_exact_predecessor() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest, store, executor, publisher, clock=clock
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="predecessor",
    ):
        runner.run(CONFIRMATION_COMMAND)

    assert executor.events == []
    assert not any(event.startswith("consume:") for event in store.events)


def test_worker_failure_never_releases_partial_semantic_output() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(
        store, clock, raise_phase="deterministic"
    )
    publisher = FakePublisher()
    runner = _runner(
        manifest, store, executor, publisher, clock=clock
    )

    result = runner.run(CONFIRMATION_COMMAND)

    assert result["terminal_status"] == TERMINAL_INDETERMINATE
    assert result["terminal_evidence"] is None
    assert result["terminal_artifact"] is None
    assert set(result) == {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "command",
        "attempt_id",
        "terminal_status",
        "passed",
        "diagnostic_code",
        "terminal_transition",
        "terminal_transition_sha256",
        "terminal_evidence",
        "external_publication",
        "terminal_artifact",
        "terminal_artifact_receipt",
        "sealed_stage_result_sha256",
    }


@pytest.mark.parametrize(
    "option",
    ["missing_arm", "missing_proof", "tamper_replay_hash"],
)
def test_missing_arm_proof_or_tampered_replay_is_terminal_indeterminate(
    option: str,
) -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    evaluator = FakeEvaluator(**{option: True})
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        evaluator=evaluator,
    )

    result = runner.run(CONFIRMATION_COMMAND)

    assert result["terminal_status"] == TERMINAL_INDETERMINATE
    assert result["diagnostic_code"] == "post_consumption_indeterminate"
    assert result["terminal_evidence"] is None
    assert publisher.calls == []


def test_failed_gate_is_terminal_and_blocks_report_publication() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        evaluator=FakeEvaluator(failed_gate=True),
    )

    result = runner.run(CONFIRMATION_COMMAND)

    assert result["terminal_status"] == TERMINAL_FAIL
    assert result["diagnostic_code"] == "gate_failure_sealed"
    assert result["terminal_artifact"] is not None
    assert result["terminal_evidence"] is not None
    assert publisher.calls == [
        result["terminal_artifact"]["joint_stage_report_sha256"]
    ]
    plan = store.plans[ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]]
    assert validate_sealed_stage_result(
        result,
        implementation_manifest=manifest,
        attempt_plan=plan,
    ) == result


def test_invalid_publisher_receipt_after_effect_is_pending() -> None:
    class InvalidReceiptPublisher(FakePublisher):
        def publish(self, **kwargs: Any) -> dict[str, Any]:
            super().publish(**kwargs)
            return {}

    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = InvalidReceiptPublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )

    result = runner.run(CONFIRMATION_COMMAND)
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    assert publisher.calls
    assert result["result_status"] == "publication_pending"
    assert result["semantic_result_released"] is False
    assert result["next_stage_authority_blocked"] is True
    assert result["publication_recovery_required"] is True
    assert store.histories[attempt_id][-1]["status"] == CONSUMED
    assert attempt_id not in store.anchor_bindings


def test_recovery_pre_push_exact_terminalizes_pending_attempt() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher(
        remote_states=[None, "exact_expected"]
    )
    publication_runtime = FakePublicationRuntime()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        publication_runtime=publication_runtime,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    pending = runner.run(CONFIRMATION_COMMAND)
    recovered = runner.recover_publication(attempt_id)

    assert pending["result_status"] == "publication_pending"
    assert recovered["passed"] is True
    assert recovered["terminal_status"] == TERMINAL_PASS
    assert store.histories[attempt_id][-1]["status"] == TERMINAL_PASS
    assert publisher.read_calls == ["pre_push", "pre_push"]
    assert publisher.push_calls == []
    completions = store.governance_records(
        "publication_recovery_invocation_completions",
        attempt_id=attempt_id,
    )
    assert [row["outcome"] for row in completions] == [
        "remote_exact_without_push"
    ]
    assert len(
        store.governance_records(
            "publication_receipts", attempt_id=attempt_id
        )
    ) == 1
    assert len(
        store.governance_records(
            "terminalization_claims", attempt_id=attempt_id
        )
    ) == 1


def test_recovery_cap_starts_before_local_reconciliation() -> None:
    manifest = _manifest()
    clock = MutableClock()

    class SlowReconciliationStore(FakeStore):
        def reconcile_publication_recovery_state(
            self, attempt_id: str
        ) -> None:
            super().reconcile_publication_recovery_state(attempt_id)
            clock.value += 301.0

    store = SlowReconciliationStore(manifest)
    _seed_confirmation_predecessor(store)
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher(remote_states=[None])
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    pending = runner.run(CONFIRMATION_COMMAND)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerIndeterminate,
        match="local publication recovery reconciliation",
    ):
        runner.recover_publication(attempt_id)

    assert pending["result_status"] == "publication_pending"
    assert publisher.read_calls == ["pre_push"]
    assert store.governance_records(
        "publication_recovery_invocation_starts",
        attempt_id=attempt_id,
    ) == []


def test_same_session_orphan_exact_observation_is_reconciled_locally() -> None:
    manifest = _manifest()

    class OrphanObservationStore(FakeStore):
        def __init__(self, value: dict[str, Any]) -> None:
            super().__init__(value)
            self.completion_failures = 2

        def commit_recovery_completion(
            self,
            capability: FakeCapability,
            completion: dict[str, Any],
        ) -> FakeDurableAuthority:
            if self.completion_failures:
                self.completion_failures -= 1
                raise RuntimeError("injected completion interruption")
            return super().commit_recovery_completion(
                capability, completion
            )

        def reconcile_publication_recovery_state(
            self, attempt_id: str
        ) -> None:
            super().reconcile_publication_recovery_state(attempt_id)
            starts = self.governance_records(
                "publication_recovery_invocation_starts",
                attempt_id=attempt_id,
            )
            completions = self.governance_records(
                "publication_recovery_invocation_completions",
                attempt_id=attempt_id,
            )
            observations = [
                row
                for row in self.governance_records(
                    "publication_remote_observations",
                    attempt_id=attempt_id,
                )
                if row["observation_operation_kind"]
                == "publication_recovery"
                and row["observed_ref_state"] == "exact_expected"
            ]
            if not starts or completions or not observations:
                return
            start = self._governance_lookup(
                "publication_recovery_invocation_starts",
                attempt_id,
            )
            observation = self.remote_observation_authority(
                attempt_id,
                observations[-1][
                    "publication_remote_observation_sha256"
                ],
            )
            material = (
                runner_module._build_publication_recovery_completion(
                    implementation_manifest=self.manifest,
                    store=self,
                    attempt_id=attempt_id,
                    publication_intent=(
                        self.publication_intent_authority(attempt_id)
                    ),
                    recovery_start=start,
                    outcome="remote_exact_without_push",
                    remote_observation=observation,
                    pre_push_authorization=None,
                    push_command_count_upper_bound=0,
                    elapsed_seconds=1.0,
                )
            )
            super().commit_recovery_completion(
                FakeCapability(
                    attempt_id,
                    operation_kind="publication_recovery",
                    operation_sha256=start.as_dict()[
                        "recovery_invocation_start_sha256"
                    ],
                ),
                material,
            )

    store = OrphanObservationStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher(
        remote_states=[None, "exact_expected"]
    )
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    pending = runner.run(CONFIRMATION_COMMAND)
    recovered = runner.recover_publication(attempt_id)

    assert pending["result_status"] == "publication_pending"
    assert recovered["terminal_status"] == TERMINAL_PASS
    assert publisher.read_calls == ["pre_push", "pre_push"]
    assert len(
        store.governance_records(
            "publication_recovery_invocation_completions",
            attempt_id=attempt_id,
        )
    ) == 1


def test_durable_normal_conflict_is_poisoned_without_a_new_remote_read() -> None:
    manifest = _manifest()

    class InterruptedConflictStore(FakeStore):
        def __init__(self, value: dict[str, Any]) -> None:
            super().__init__(value)
            self.fail_conflict_once = True

        def commit_publication_conflict(
            self,
            capability: FakeCapability,
            conflict: dict[str, Any],
        ) -> FakeDurableAuthority:
            if self.fail_conflict_once:
                self.fail_conflict_once = False
                raise RuntimeError("injected conflict interruption")
            return super().commit_publication_conflict(
                capability, conflict
            )

    store = InterruptedConflictStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher(remote_states=["conflicting"])
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    initial = runner.run(CONFIRMATION_COMMAND)
    recovered = runner.recover_publication(attempt_id)

    assert initial["result_status"] == "publication_pending"
    assert recovered["result_status"] == "publication_pending"
    assert publisher.read_calls == ["pre_push"]
    conflicts = store.governance_records(
        "publication_conflicts", attempt_id=attempt_id
    )
    assert len(conflicts) == 1
    assert conflicts[0]["conflict_reason"] == (
        "durable_remote_ref_conflict"
    )


def test_development_acquisition_recovery_rehydrates_without_rerun(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    acquisition = FakeAcquisitionAdapter(
        clock,
        recovery_vault_path=tmp_path / "acquisition-recovery-vault",
    )
    publisher = FakePublisher(
        remote_states=[None, "exact_expected"]
    )
    publication_runtime = FakePublicationRuntime()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        acquisition_adapter=acquisition,
        publication_runtime=publication_runtime,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[DEVELOPMENT_ACQUISITION]

    pending = runner.run(DEVELOPMENT_ACQUISITION_COMMAND)
    acquisition_calls_before = list(acquisition.calls)
    executor_events_before = list(executor.events)
    recovered = runner.recover_publication(attempt_id)

    assert pending["result_status"] == "publication_pending"
    assert recovered["passed"] is True
    assert recovered["terminal_status"] == TERMINAL_PASS
    assert recovered["terminal_artifact"]["verdict"] == "pass"
    assert acquisition.calls == acquisition_calls_before == [
        DEVELOPMENT_ACQUISITION_COMMAND
    ]
    assert acquisition.rehydrate_calls == [attempt_id]
    assert executor.events == executor_events_before == []
    assert store.histories[attempt_id][-1]["status"] == TERMINAL_PASS
    assert publisher.read_calls == ["pre_push", "pre_push"]
    assert publisher.push_calls == []
    assert len(publication_runtime.calls) == 2
    assert (
        store.governance_records(
            "publication_recovery_invocation_completions",
            attempt_id=attempt_id,
        )[0]["outcome"]
        == "remote_exact_without_push"
    )


def test_recovery_absent_push_then_exact_terminalizes() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher(
        remote_states=[
            None,
            "absent",
            "exact_expected",
        ]
    )
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    pending = runner.run(CONFIRMATION_COMMAND)
    recovered = runner.recover_publication(attempt_id)

    assert pending["result_status"] == "publication_pending"
    assert recovered["passed"] is True
    assert recovered["terminal_status"] == TERMINAL_PASS
    assert publisher.read_calls == [
        "pre_push",
        "pre_push",
        "post_push",
    ]
    assert publisher.push_calls == [
        recovered["terminal_artifact"][
            "joint_stage_report_sha256"
        ]
    ]
    authorizations = store.governance_records(
        "publication_pre_push_authorizations",
        attempt_id=attempt_id,
    )
    assert len(authorizations) == 1
    assert authorizations[0]["authorization_operation_kind"] == (
        "publication_recovery"
    )
    completions = store.governance_records(
        "publication_recovery_invocation_completions",
        attempt_id=attempt_id,
    )
    assert [row["outcome"] for row in completions] == [
        "published_exact_after_push"
    ]
    assert completions[0]["push_command_count_upper_bound"] == 1


def test_recovery_conflict_stays_pending_without_semantic_release() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher(
        remote_states=[None, "conflicting"]
    )
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    initial = runner.run(CONFIRMATION_COMMAND)
    first_recovery = runner.recover_publication(attempt_id)
    read_count = len(publisher.read_calls)
    second_recovery = runner.recover_publication(attempt_id)

    assert initial["result_status"] == "publication_pending"
    assert first_recovery["result_status"] == "publication_pending"
    assert second_recovery == first_recovery
    assert first_recovery["semantic_result_released"] is False
    assert first_recovery["next_stage_authority_blocked"] is True
    assert first_recovery["publication_recovery_required"] is True
    assert store.histories[attempt_id][-1]["status"] == CONSUMED
    assert len(publisher.read_calls) == read_count
    assert publisher.push_calls == []
    assert len(
        store.governance_records(
            "publication_conflicts", attempt_id=attempt_id
        )
    ) == 1
    assert (
        store.governance_records(
            "publication_recovery_invocation_completions",
            attempt_id=attempt_id,
        )[0]["outcome"]
        == "remote_conflict_poisoned_without_push"
    )
    assert (
        store.governance_records(
            "publication_receipts", attempt_id=attempt_id
        )
        == []
    )
    assert attempt_id not in store.anchor_bindings


def test_recover_publication_is_idempotent_when_already_terminal() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    publication_runtime = FakePublicationRuntime()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        publication_runtime=publication_runtime,
    )
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]

    original = runner.run(CONFIRMATION_COMMAND)
    history_before = store.attempt_history(attempt_id)
    governance_counts_before = {
        table: len(rows)
        for table, rows in store.governance.items()
    }
    runtime_calls_before = list(publication_runtime.calls)
    read_calls_before = list(publisher.read_calls)

    recovered = runner.recover_publication(attempt_id)

    assert recovered == original
    assert store.attempt_history(attempt_id) == history_before
    assert {
        table: len(rows)
        for table, rows in store.governance.items()
    } == governance_counts_before
    assert publication_runtime.calls == runtime_calls_before
    assert publisher.read_calls == read_calls_before


def test_sealed_validator_rejects_unsubstantiated_scored_terminal_fail() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()

    def runtime_verifier(_payload: dict[str, Any]) -> dict[str, Any]:
        raise SecGemmaOnlineRiskOverlayRunnerError(
            "pinned runtime identity changed"
        )

    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        runtime_verifier=runtime_verifier,
    )
    result = runner.run(CONFIRMATION_COMMAND)
    attempt_id = ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]
    plan = store.plans[attempt_id]
    forged = copy.deepcopy(result)
    forged_transition = build_attempt_transition(
        attempt_plan=plan,
        implementation_manifest=manifest,
        status=TERMINAL_FAIL,
        prior_transition=store.histories[attempt_id][-2],
    )
    forged["terminal_status"] = TERMINAL_FAIL
    forged["diagnostic_code"] = "gate_failure_sealed"
    forged["terminal_transition"] = forged_transition
    forged["terminal_transition_sha256"] = forged_transition[
        "transition_sha256"
    ]
    forged["sealed_stage_result_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in forged.items()
            if key != "sealed_stage_result_sha256"
        }
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="complete sealed scored gate failure",
    ):
        validate_sealed_stage_result(
            forged,
            implementation_manifest=manifest,
            attempt_plan=plan,
        )


def test_fabricated_all_true_check_cannot_create_terminal_pass() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        evaluator=FakeEvaluator(fabricated_checks=True),
    )

    result = runner.run(CONFIRMATION_COMMAND)

    assert result["terminal_status"] == TERMINAL_INDETERMINATE
    assert result["diagnostic_code"] == "post_consumption_indeterminate"
    assert result["terminal_evidence"] is None
    assert publisher.calls == []


def test_development_acquisition_rejects_semantic_or_score_payload() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    acquisition = FakeAcquisitionAdapter(
        clock,
        manifest_overrides={
            "semantic_extractions_returned": True
        },
    )
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        acquisition_adapter=acquisition,
    )

    result = runner.run(DEVELOPMENT_ACQUISITION_COMMAND)

    assert result["terminal_status"] == TERMINAL_INDETERMINATE
    assert result["diagnostic_code"] == "post_consumption_indeterminate"
    assert result["terminal_artifact"] is None
    assert publisher.calls == []


def test_acquisition_report_must_use_exact_validator_check_set() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    acquisition = FakeAcquisitionAdapter(
        clock,
        report_checks={"foo": _digest("foo")},
    )
    publisher = FakePublisher()
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        acquisition_adapter=acquisition,
    )

    result = runner.run(DEVELOPMENT_ACQUISITION_COMMAND)

    assert result["terminal_status"] == TERMINAL_INDETERMINATE
    assert result["diagnostic_code"] == "post_consumption_indeterminate"
    assert publisher.calls == []


def test_final_plan_cannot_use_an_arbitrary_hex_pin() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_final_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest, store, executor, publisher, clock=clock
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="registry successor",
    ):
        runner.plan(FINAL_COMMAND)


def test_final_registry_authority_must_match_exact_frozen_predecessor() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_final_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    authority = FakeFinalRegistryAuthority(
        _final_registry_authorization(
            predecessor_registry_tip_sha256=_digest("wrong-tip")
        )
    )
    runner = _runner(
        manifest,
        store,
        executor,
        publisher,
        clock=clock,
        final_registry_authority=authority,
    )
    events_before = list(store.events)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRunnerError,
        match="no opaque authorization",
    ):
        runner.run(FINAL_COMMAND)

    assert store.events == events_before
    assert executor.events == []


def test_sealed_report_tampering_is_rejected() -> None:
    manifest = _manifest()
    store = FakeStore(manifest)
    _seed_confirmation_predecessor(store)
    clock = MutableClock()
    executor = FakeExecutor(store, clock)
    publisher = FakePublisher()
    runner = _runner(
        manifest, store, executor, publisher, clock=clock
    )
    result = runner.run(CONFIRMATION_COMMAND)
    plan = store.plans[ATTEMPT_ID_BY_KIND[CONFIRMATION_SCORING]]

    changed = copy.deepcopy(result)
    changed["terminal_evidence"]["record_commitment_sha256"] = "f" * 64
    changed["sealed_stage_result_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "sealed_stage_result_sha256"
        }
    )

    with pytest.raises(Exception):
        validate_sealed_stage_result(
            changed,
            implementation_manifest=manifest,
            attempt_plan=plan,
        )
