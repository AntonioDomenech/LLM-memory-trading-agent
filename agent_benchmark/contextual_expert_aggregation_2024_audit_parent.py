"""Exact parent-only verifier for the preregistered 2024 continuation audit.

This module treats the sealed through-2023 post-rejection audit as an immutable
trust root.  It verifies that bundle's complete byte inventory, parses its
continuation checkpoint with the existing canonical parser, and returns only
the selected ``online_full`` model plus the cost-specific account/ledger
prefixes needed to start the two 2024 scenarios.

It deliberately contains no 2024 input path, policy replay, scoring, or gate
evaluation.  Ledger replay below validates accounting and hash-chain
continuity only; it consumes the already-sealed target decisions.
"""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from . import contextual_expert_aggregation_audit_artifacts as _audit_artifacts
from . import contextual_expert_aggregation_experiment as _experiment
from . import contextual_expert_aggregation_ledger as _ledger
from . import contextual_expert_aggregation_replay as _replay
from . import contextual_expert_aggregation_stage as _v2_stage


VERIFIER_ID = "contextual-expert-aggregation-2024-audit-parent-verifier-v2"
VERIFIER_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_2024_audit_parent.py"
)

PARENT_DIRECTORY = Path(
    "e/aapl_causal_contextual_expert_aggregation_audit_v2/"
    "contextual-expert-aggregation-post-rejection-2019-2023-audit-v2"
)
PARENT_PRESERVATION_COMMIT = "e4783a93391f8b6ba632bcbff8192c554dd374e7"
PARENT_RUN_COMMIT = "f2085fcf776d32a29918727ad8d9b64932cb17d9"
PARENT_CONTRACT_VERSION = (
    "aapl-causal-contextual-expert-aggregation-2019-2023-audit-v2"
)
PARENT_STAGE = "audit"
PARENT_RUN_ID = (
    "contextual-expert-aggregation-post-rejection-2019-2023-audit-v2"
)
PARENT_STATUS = "POST_REJECTION_AUDIT_POLICY_CANDIDATE"
PARENT_EVIDENCE_CLASSIFICATION = (
    "post_rejection_reused_historical_continuation_audit"
)

PARENT_MANIFEST_FILE_SHA256 = (
    "sha256:fd4c7427bdc9ec2a5463dcc2781310aeab5c498836db0617c0c41ebf81267bbd"
)
PARENT_MANIFEST_SELF_SHA256 = (
    "sha256:c71e201a8a3d1ef2b0ebe914d46abfdd0f5a9154d4f106ce723168a9bce009e8"
)
PARENT_CHECKSUMS_FILE_SHA256 = (
    "sha256:4d7fd305fb0ae6c8584b993528366af433c5b3401455745446fe2e36c36085b6"
)
PARENT_REPORT_FILE_SHA256 = (
    "sha256:5877434e4decb79f35a765793c9408638434d7ab1a590f29fe2f03da46522502"
)
PARENT_REPORT_SELF_SHA256 = (
    "sha256:01c58f963566517c5abe88ea293e34fa84cad22d7a8fd42d79bde36d6dba940e"
)
PARENT_GATE_REPORT_FILE_SHA256 = (
    "sha256:2caf58663fbe6b5fdc255708e6cbb4c3b5ed8383fa487c7ae602e868c741d7a2"
)
PARENT_METRICS_FILE_SHA256 = (
    "sha256:415b709a44a177706227162b5778ccfa80161ba5f4154bfd7d9579ad12359280"
)
PARENT_INTEGRITY_FILE_SHA256 = (
    "sha256:a4442f5d45bdcecf5f2ec32f356643593aed57ef91b554d61f97f05df45009f3"
)
PARENT_INTEGRITY_SELF_SHA256 = (
    "sha256:4960b6c06b7915e8cf0883b97e70d3a3291b385d3a7d8459e7e773ecb8cfee73"
)
PARENT_CHECKPOINT_FILE_SHA256 = (
    "sha256:560571500f580f9e0fd4319a93905b4ad1e498f1c1f27de9d647da455b6d5477"
)
PARENT_CHECKPOINT_SELF_SHA256 = (
    "sha256:4aaf549c9352ac322f36c46cb8dcdd03f672a1eba580813ba8a1947d323a85eb"
)

CHECKPOINT_FILENAME = "audit_continuation_checkpoint_through_2023.json"
REPORT_FILENAME = "report.json"
GATE_REPORT_FILENAME = "audit_gate_report.json"
METRICS_FILENAME = "audit_metrics.json"
INTEGRITY_FILENAME = "audit_integrity_evidence.json"

SELECTED_POLICY = "online_full"
REQUIRED_POLICIES = (
    SELECTED_POLICY,
    "aapl_buy_hold",
    "always_long",
    "exact_union_cash",
    "contextual_only",
)
SCENARIO_PARENT_POLICY: Mapping[str, str] = MappingProxyType(
    {
        "frozen_2023_lead": SELECTED_POLICY,
        "online_2024_shadow": SELECTED_POLICY,
    }
)

SELECTED_REPLAY_DIGEST = (
    "349ca26a1694970a1b24762fc3c9dfe289c837128780503f07dbda126848f1b8"
)
SELECTED_MODEL_STATE_SHA256 = (
    "781d9ae7f10335c7167b43b57f42f7c483052839b9f32da99e4a68fe286f502b"
)
SELECTED_PENDING_STATE_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
PARENT_LAST_SESSION = "2023-12-29"
PARENT_SOURCE_SESSION_COUNT = 6_244
PARENT_LEDGER_ROW_COUNT = 4_781

_PINNED_RAW_PAYLOAD_SHA256: Mapping[str, str] = MappingProxyType(
    {
        CHECKPOINT_FILENAME: PARENT_CHECKPOINT_FILE_SHA256,
        REPORT_FILENAME: PARENT_REPORT_FILE_SHA256,
        GATE_REPORT_FILENAME: PARENT_GATE_REPORT_FILE_SHA256,
        METRICS_FILENAME: PARENT_METRICS_FILE_SHA256,
        INTEGRITY_FILENAME: PARENT_INTEGRITY_FILE_SHA256,
    }
)

_ACCOUNT_STATE_SHA256: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "base_5bps": MappingProxyType(
            {
                "online_full": (
                    "sha256:ed9c7d4cb720ddf27c7db0429e66c16a9656dbc2561fca8a3ee41101eb056970"
                ),
                "aapl_buy_hold": (
                    "sha256:ae06a2f96d286e8a9a0c33db9aba22c58a9ef2fc15247498f71832d91c8e4cc6"
                ),
                "always_long": (
                    "sha256:618ee307f83735f810ab47f547772ad7993bc2b621f9fc407fda7fa96f7fd5b5"
                ),
                "exact_union_cash": (
                    "sha256:121e91e39e8ee9821455b2e31d9e079c604376914efb42e13dea1d95a394bc50"
                ),
                "contextual_only": (
                    "sha256:d50ccb3be50d80310169eac244760057b3b77a3762fc6382676fd6db6f60fa30"
                ),
            }
        ),
        "stress_10bps": MappingProxyType(
            {
                "online_full": (
                    "sha256:6ca2a8f11ab4ce875cfd10057e978de83d91f0d6401fbe1bf62f4d0bba962683"
                ),
                "aapl_buy_hold": (
                    "sha256:35ccf533717a511fac464f6778d0a75cef64285f9d10c3cad2c54d5c14b98ba9"
                ),
                "always_long": (
                    "sha256:0da8604bacdfceffab5a1d10255df15472512b51901e1ec536548ccc85cf82f1"
                ),
                "exact_union_cash": (
                    "sha256:92882101074b5e149c5023332ae250419e6516efb8c9893099ff069cbea50293"
                ),
                "contextual_only": (
                    "sha256:4030d8a3099b3319420e2b927a6d8281536eb1deb9cfe8d988f07259d5ecc3db"
                ),
            }
        ),
    }
)

_LEDGER_TIP_SHA256: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "base_5bps": MappingProxyType(
            {
                "online_full": (
                    "sha256:8c7b1c17ff2ea17086a09df78e342235ad1e2bd57fcbe05816350f317f01549a"
                ),
                "aapl_buy_hold": (
                    "sha256:fd6441546a54da6696ac9b553d9b8369878fb8e4abaf55ea67d024eff5ce5f3e"
                ),
                "always_long": (
                    "sha256:a26e056036be39fca6b50267c25dc161490edfb3f0d1c2d57a317ed999164a1f"
                ),
                "exact_union_cash": (
                    "sha256:5b8d4d25dccbf9275be1308ab60a340d5925576df718156114c4a07e4285a5f4"
                ),
                "contextual_only": (
                    "sha256:cf293ad8bdfe5038e936c83b5a55bd5ef8d21191f0ff65d67a1ab1edc4aa1e89"
                ),
            }
        ),
        "stress_10bps": MappingProxyType(
            {
                "online_full": (
                    "sha256:05f10e984ae3f3044c36078fb4ea6dd1bee2f78e7db393ce2a478913e3c1b167"
                ),
                "aapl_buy_hold": (
                    "sha256:66b0f2dffc131c640f6f43780adecb79027f786608330c3db07616878bbaef75"
                ),
                "always_long": (
                    "sha256:f301f7b239e7d7e76f2ec94ac30e28897252a00100ecd6a886c7429228260266"
                ),
                "exact_union_cash": (
                    "sha256:cadcf7afccefa03edb1aa8aa65a7c7f6d5a00b87d61586d1a63f6db8fae7c5f3"
                ),
                "contextual_only": (
                    "sha256:b4453107598e3c00452f0fee402665e3fbbb0712c23b46aff1e42f5dfd651da7"
                ),
            }
        ),
    }
)


class ParentBundleVerificationError(RuntimeError):
    """The exact sealed through-2023 parent failed closed verification."""


@dataclass(frozen=True)
class ParentBundleEvidence:
    """Byte-bound parent state ready for the 2024 runner to fork."""

    verifier_id: str
    verifier_dependency_path: str
    verified: bool
    parent_directory: str
    preservation_commit: str
    original_run_commit: str
    contract_version: str
    stage: str
    run_id: str
    status: str
    stage_pass: bool
    historical_policy_candidate: bool
    learning_candidate: bool
    passed_gate_count: int
    total_gate_count: int
    manifest_file_sha256: str
    manifest_self_sha256: str
    checksums_file_sha256: str
    report_file_sha256: str
    report_self_sha256: str
    gate_report_file_sha256: str
    metrics_file_sha256: str
    integrity_file_sha256: str
    integrity_self_sha256: str
    checkpoint_file_sha256: str
    checkpoint_self_sha256: str
    exact_payload_names: tuple[str, ...]
    payload_sha256: Mapping[str, str]
    bundle: _experiment.VerifiedBundle
    manifest_bytes: bytes
    checksums_bytes: bytes
    checkpoint_bytes: bytes
    parsed_checkpoint: Mapping[str, Any]
    checkpoint: _replay.ReplayCheckpoint
    model_payload: Mapping[str, Any]
    accounts: Mapping[str, Mapping[str, _ledger.AccountState]]
    account_checkpoint_sha256: Mapping[str, Mapping[str, str]]
    ledgers: Mapping[str, Mapping[str, pd.DataFrame]]

    @staticmethod
    def _policy(policy_or_scenario: str) -> str:
        policy = SCENARIO_PARENT_POLICY.get(policy_or_scenario, policy_or_scenario)
        if policy not in REQUIRED_POLICIES:
            raise KeyError(f"unauthorized parent policy: {policy_or_scenario}")
        return policy

    def account_for(
        self, cost: str, policy_or_scenario: str
    ) -> _ledger.AccountState:
        """Return one immutable parent account, resolving scenario aliases."""

        return self.accounts[cost][self._policy(policy_or_scenario)]

    def ledger_for(
        self, cost: str, policy_or_scenario: str
    ) -> pd.DataFrame:
        """Return a defensive copy of one verified continuous parent ledger."""

        return self.ledgers[cost][self._policy(policy_or_scenario)].copy(deep=True)

    def lock_payload(self) -> dict[str, Any]:
        """Return the stable JSON-compatible identity subset for an attempt lock."""

        account_identity = {
            cost: {
                policy: {
                    "account_state_sha256": self.account_checkpoint_sha256[cost][
                        policy
                    ],
                    "ledger_tip_sha256": self.accounts[cost][
                        policy
                    ].ledger_tip_sha256,
                    "ledger_row_count": self.accounts[cost][
                        policy
                    ].ledger_row_count,
                }
                for policy in REQUIRED_POLICIES
            }
            for cost in _audit_artifacts.COST_ORDER
        }
        selected_state = self.checkpoint.model_payload["state"]
        return {
            "verifier_id": self.verifier_id,
            "verifier_dependency_path": self.verifier_dependency_path,
            "verified": self.verified,
            "parent_directory": self.parent_directory,
            "preservation_commit": self.preservation_commit,
            "original_run_commit": self.original_run_commit,
            "contract_version": self.contract_version,
            "stage": self.stage,
            "run_id": self.run_id,
            "status": self.status,
            "stage_pass": self.stage_pass,
            "historical_policy_candidate": self.historical_policy_candidate,
            "learning_candidate": self.learning_candidate,
            "parent_rejection_remains_final": True,
            "post_2023_market_values_accessed": False,
            "passed_gate_count": self.passed_gate_count,
            "total_gate_count": self.total_gate_count,
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_self_sha256": self.manifest_self_sha256,
            "checksums_file_sha256": self.checksums_file_sha256,
            "report_file_sha256": self.report_file_sha256,
            "report_self_sha256": self.report_self_sha256,
            "gate_report_file_sha256": self.gate_report_file_sha256,
            "metrics_file_sha256": self.metrics_file_sha256,
            "integrity_file_sha256": self.integrity_file_sha256,
            "integrity_self_sha256": self.integrity_self_sha256,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "checkpoint_self_sha256": self.checkpoint_self_sha256,
            "exact_payload_names": list(self.exact_payload_names),
            "payload_sha256": dict(self.payload_sha256),
            "selected_checkpoint": {
                "policy": SELECTED_POLICY,
                "digest_sha256": self.checkpoint.digest_sha256,
                "checkpoint_date": self.checkpoint.checkpoint_date,
                "source_session_count": self.checkpoint.source_session_count,
                "runtime": dict(self.checkpoint.model_payload["runtime"]),
                "processed_session_count": selected_state[
                    "processed_session_count"
                ],
                "admitted_lesson_count": selected_state[
                    "admitted_lesson_count"
                ],
                "matured_lesson_count": selected_state[
                    "matured_lesson_count"
                ],
                "pending_lesson_count": len(self.checkpoint.pending_lessons),
                "model_state_sha256": self.checkpoint.model_state_sha256,
                "pending_state_sha256": self.checkpoint.pending_state_sha256,
            },
            "account_identity": account_identity,
        }


def _git_run(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ParentBundleVerificationError(
            "parent Git trust-root verification failed"
        ) from exc


def _git_text(repo_root: Path, *args: str) -> str:
    result = _git_run(repo_root, *args)
    if result.returncode != 0:
        raise ParentBundleVerificationError(
            "parent Git trust-root verification failed"
        )
    try:
        return result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ParentBundleVerificationError(
            "parent Git metadata is not UTF-8"
        ) from exc


def _require_repository_root(repo_root: Path) -> Path:
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ParentBundleVerificationError(
            "parent verifier requires an existing repository root"
        )
    actual = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if actual != root:
        raise ParentBundleVerificationError(
            "repo_root must be the actual Git repository root"
        )
    return root


def _require_parent_git_identity(repo_root: Path) -> None:
    for commit in (PARENT_RUN_COMMIT, PARENT_PRESERVATION_COMMIT):
        if _git_text(repo_root, "rev-parse", f"{commit}^{{commit}}") != commit:
            raise ParentBundleVerificationError(
                "a frozen parent commit identity changed"
            )
    for older, newer in (
        (PARENT_RUN_COMMIT, PARENT_PRESERVATION_COMMIT),
        (PARENT_PRESERVATION_COMMIT, "HEAD"),
    ):
        if (
            _git_run(repo_root, "merge-base", "--is-ancestor", older, newer).returncode
            != 0
        ):
            raise ParentBundleVerificationError(
                "frozen parent commit ancestry changed"
            )
    if (
        _git_run(
            repo_root,
            "diff",
            "--quiet",
            PARENT_PRESERVATION_COMMIT,
            "HEAD",
            "--",
            PARENT_DIRECTORY.as_posix(),
        ).returncode
        != 0
    ):
        raise ParentBundleVerificationError(
            "sealed parent tree changed after its preservation commit"
        )


def _load_exact_parent_bundle(
    repo_root: Path,
) -> tuple[_experiment.VerifiedBundle, bytes, bytes]:
    directory = repo_root / PARENT_DIRECTORY
    manifest_path = directory / "stage_manifest.json"
    checksums_path = directory / "checksums.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        checksums_bytes = checksums_path.read_bytes()
    except OSError as exc:
        raise ParentBundleVerificationError(
            "exact parent seal metadata is unreadable"
        ) from exc
    if (
        _experiment.sha256_bytes(manifest_bytes) != PARENT_MANIFEST_FILE_SHA256
        or _experiment.sha256_bytes(checksums_bytes)
        != PARENT_CHECKSUMS_FILE_SHA256
    ):
        raise ParentBundleVerificationError(
            "exact parent manifest or checksums raw bytes changed"
        )
    try:
        bundle = _experiment._verify_bundle_identity(
            directory,
            expected_contract_version=PARENT_CONTRACT_VERSION,
            expected_stage=PARENT_STAGE,
            expected_manifest_sha256=PARENT_MANIFEST_SELF_SHA256,
            expected_payload_names=_audit_artifacts.AUDIT_PAYLOAD_NAMES,
            require_stage_pass=True,
            repo_root=repo_root,
        )
    except _experiment.ContextualExpertAggregationExperimentError as exc:
        raise ParentBundleVerificationError(
            "exact parent byte bundle or inventory verification failed"
        ) from exc
    if any(
        bundle.payload_sha256.get(filename) != expected
        for filename, expected in _PINNED_RAW_PAYLOAD_SHA256.items()
    ):
        raise ParentBundleVerificationError(
            "a pinned parent payload raw hash changed"
        )
    return bundle, manifest_bytes, checksums_bytes


def _canonical_object(
    bundle: _experiment.VerifiedBundle, filename: str
) -> tuple[dict[str, Any], bytes]:
    try:
        payload = (bundle.directory / filename).read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ParentBundleVerificationError(
            f"parent payload is unreadable: {filename}"
        ) from exc
    if not isinstance(value, dict) or _experiment.pretty_json_bytes(value) != payload:
        raise ParentBundleVerificationError(
            f"parent payload is not canonical object JSON: {filename}"
        )
    return value, payload


def _require_omitted_field_self_hash(
    value: Mapping[str, Any],
    *,
    field: str,
    expected: str,
    label: str,
) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    computed = _experiment.sha256_bytes(
        _experiment.canonical_json_bytes(unsigned)
    )
    if recorded != expected or computed != expected:
        raise ParentBundleVerificationError(f"parent {label} self-hash changed")


def _require_parent_claims(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    gate: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> None:
    checks = gate.get("checks")
    integrity_checks = integrity.get("checks")
    if (
        not isinstance(checks, Mapping)
        or len(checks) != 64
        or any(type(value) is not bool or value is not True for value in checks.values())
        or gate.get("stage") != PARENT_STAGE
        or gate.get("passed") is not True
        or gate.get("passed_count") != 64
        or gate.get("total_count") != 64
        or gate.get("failed_checks") != []
        or gate.get("fatal_integrity_rejection") is not False
        or gate.get("fatal_reasons") != []
        or gate.get("learning_candidate_for_2024_audit") is not False
        or gate.get("normal_economic_gates_evaluated") is not True
        or gate.get("original_v2_rejection_remains_final") is not True
    ):
        raise ParentBundleVerificationError(
            "parent exact 64-of-64 gate result changed"
        )
    if (
        manifest.get("contract_version") != PARENT_CONTRACT_VERSION
        or manifest.get("stage") != PARENT_STAGE
        or manifest.get("run_id") != PARENT_RUN_ID
        or manifest.get("stage_pass") is not True
        or manifest.get("report_status") != PARENT_STATUS
        or manifest.get("evidence_classification")
        != PARENT_EVIDENCE_CLASSIFICATION
        or manifest.get("historical_policy_candidate_for_2024_audit") is not True
        or manifest.get("learning_candidate_for_2024_audit") is not False
        or manifest.get("parent_rejection_remains_final") is not True
        or manifest.get("post_2023_market_values_accessed") is not False
        or not isinstance(manifest.get("git_identity"), Mapping)
        or manifest["git_identity"].get("commit") != PARENT_RUN_COMMIT
        or manifest["git_identity"].get("upstream_commit") != PARENT_RUN_COMMIT
    ):
        raise ParentBundleVerificationError("parent manifest claims changed")
    if (
        report.get("contract_version") != PARENT_CONTRACT_VERSION
        or report.get("stage") != PARENT_STAGE
        or report.get("run_id") != PARENT_RUN_ID
        or report.get("status") != PARENT_STATUS
        or report.get("stage_pass") is not True
        or report.get("evidence_classification")
        != PARENT_EVIDENCE_CLASSIFICATION
        or report.get("historical_policy_candidate_for_2024_audit") is not True
        or report.get("learning_candidate_for_2024_audit") is not False
        or report.get("parent_rejection_remains_final") is not True
        or report.get("post_2023_market_values_accessed") is not False
        or report.get("later_data_access_authorized") is not False
        or report.get("account_reset_count_after_inception") != 0
        or report.get("continuous_account_start") != "2005-01-01"
        or report.get("calendar_cutoff") != "2023-12-31"
        or report.get("physical_data_end") != PARENT_LAST_SESSION
        or report.get("historical_results_authorize_real_capital") is not False
        or report.get("is_confirmation") is not False
        or report.get("is_holdout_test") is not False
        or report.get("is_prospective") is not False
        or report.get("gate_report") != dict(gate)
    ):
        raise ParentBundleVerificationError("parent report claims changed")
    if (
        not isinstance(integrity_checks, Mapping)
        or not integrity_checks
        or any(
            type(value) is not bool or value is not True
            for value in integrity_checks.values()
        )
        or integrity_checks.get("exact_rejected_parent_authorization") is not True
        or integrity_checks.get("post_2023_market_values_not_accessed") is not True
        or integrity_checks.get("single_continuous_account_exact") is not True
    ):
        raise ParentBundleVerificationError(
            "parent integrity and continuity claims changed"
        )


def _require_selected_checkpoint(
    checkpoint: _replay.ReplayCheckpoint,
) -> None:
    state = checkpoint.model_payload.get("state")
    runtime = checkpoint.model_payload.get("runtime")
    if (
        not isinstance(state, Mapping)
        or runtime
        != {
            "learning_mode": "causal_online",
            "frozen_cutoff": None,
            "ablation_mode": "full",
        }
        or checkpoint.digest_sha256 != SELECTED_REPLAY_DIGEST
        or checkpoint.checkpoint_date != PARENT_LAST_SESSION
        or checkpoint.source_session_count != PARENT_SOURCE_SESSION_COUNT
        or checkpoint.model_state_sha256 != SELECTED_MODEL_STATE_SHA256
        or checkpoint.pending_state_sha256 != SELECTED_PENDING_STATE_SHA256
        or state.get("last_session_date") != PARENT_LAST_SESSION
        or state.get("processed_session_count") != PARENT_SOURCE_SESSION_COUNT
        or state.get("admitted_lesson_count") != 202
        or state.get("matured_lesson_count") != 205
        or state.get("pending_lessons") != []
        or checkpoint.pending_lessons != ()
    ):
        raise ParentBundleVerificationError(
            "selected online_full checkpoint, runtime, or model state changed"
        )


def _require_account_identity(
    *,
    cost: str,
    policy: str,
    checkpoint_value: Mapping[str, Any],
    state: _ledger.AccountState,
) -> None:
    expected_hash = _ACCOUNT_STATE_SHA256[cost][policy]
    expected_tip = _LEDGER_TIP_SHA256[cost][policy]
    if (
        checkpoint_value.get("account_state_sha256") != expected_hash
        or state.policy_name != policy
        or state.cost_bps != _audit_artifacts.COST_BPS[cost]
        or state.last_session_date != PARENT_LAST_SESSION
        or state.ledger_row_count != PARENT_LEDGER_ROW_COUNT
        or state.ledger_tip_sha256 != expected_tip
        or state.inception_count != 1
        or state.held_target != 1
        or state.previous_requested_target != 1
        or state.pending_target_exposure != 1
        or state.cash != 0.0
        or state.shares <= 0.0
        or state.open_cash_entry_decision_date is not None
        or state.open_cash_entry_fill_date is not None
        or state.open_cash_entry_reference_price is not None
        or state.open_cash_fill_observations != 0
    ):
        raise ParentBundleVerificationError(
            f"pinned parent account identity changed: {cost}/{policy}"
        )


def _verify_ledger_frame(
    frame: pd.DataFrame,
    *,
    cost: str,
    policy: str,
    expected_end_state: _ledger.AccountState,
) -> None:
    try:
        if (
            len(frame) != PARENT_LEDGER_ROW_COUNT
            or tuple(frame.columns) != _ledger.LEDGER_COLUMNS
            or frame.iloc[0]["row_index"] != 1
            or frame.iloc[-1]["row_index"] != PARENT_LEDGER_ROW_COUNT
            or frame.iloc[-1]["fill_date"] != PARENT_LAST_SESSION
            or frame.iloc[-1]["row_sha256"]
            != _LEDGER_TIP_SHA256[cost][policy]
        ):
            raise ParentBundleVerificationError(
                f"parent ledger boundary changed: {cost}/{policy}"
            )
        initial = _ledger.AccountState.initial(
            policy_name=policy,
            cost_bps=_audit_artifacts.COST_BPS[cost],
        )
        _ledger.verify_ledger(
            frame,
            start_state=initial,
            expected_end_state=expected_end_state,
        )
    except ParentBundleVerificationError:
        raise
    except (_ledger.BinaryLedgerError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ParentBundleVerificationError(
            f"parent ledger accounting/hash-chain verification failed: {cost}/{policy}"
        ) from exc


def _parse_checkpoint_accounts_and_ledgers(
    bundle: _experiment.VerifiedBundle,
) -> tuple[
    bytes,
    Mapping[str, Any],
    _replay.ReplayCheckpoint,
    Mapping[str, Mapping[str, _ledger.AccountState]],
    Mapping[str, Mapping[str, str]],
    Mapping[str, Mapping[str, pd.DataFrame]],
]:
    try:
        checkpoint_bytes = (bundle.directory / CHECKPOINT_FILENAME).read_bytes()
        parsed = _audit_artifacts.parse_audit_checkpoint_bytes(checkpoint_bytes)
        if parsed.get("checkpoint_sha256") != PARENT_CHECKPOINT_SELF_SHA256:
            raise ParentBundleVerificationError(
                "parent checkpoint self-hash changed"
            )
        selected = _replay.ReplayCheckpoint.from_dict(
            parsed["replay_checkpoints"][SELECTED_POLICY]
        )
    except ParentBundleVerificationError:
        raise
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        _audit_artifacts.ContextualExpertAggregationAuditArtifactError,
    ) as exc:
        raise ParentBundleVerificationError(
            "parent continuation checkpoint is invalid"
        ) from exc
    _require_selected_checkpoint(selected)

    accounts: dict[str, Mapping[str, _ledger.AccountState]] = {}
    account_hashes: dict[str, Mapping[str, str]] = {}
    ledgers: dict[str, Mapping[str, pd.DataFrame]] = {}
    for cost in _audit_artifacts.COST_ORDER:
        cost_accounts: dict[str, _ledger.AccountState] = {}
        cost_hashes: dict[str, str] = {}
        cost_ledgers: dict[str, pd.DataFrame] = {}
        for policy in REQUIRED_POLICIES:
            try:
                raw_checkpoint = parsed["administrative_accounts"][cost][policy]
                state = _ledger.AccountState.from_checkpoint(raw_checkpoint)
            except (KeyError, TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
                raise ParentBundleVerificationError(
                    f"parent account checkpoint is invalid: {cost}/{policy}"
                ) from exc
            _require_account_identity(
                cost=cost,
                policy=policy,
                checkpoint_value=raw_checkpoint,
                state=state,
            )
            ledger_filename = f"audit_ledger__{cost}__{policy}.table.json"
            try:
                ledger_payload = (bundle.directory / ledger_filename).read_bytes()
                frame = _v2_stage.parse_ledger_table_bytes(ledger_payload)
            except (
                OSError,
                _audit_artifacts.ContextualExpertAggregationAuditArtifactError,
                _v2_stage.ContextualExpertAggregationStageError,
            ) as exc:
                raise ParentBundleVerificationError(
                    f"parent ledger artifact is invalid: {cost}/{policy}"
                ) from exc
            _verify_ledger_frame(
                frame,
                cost=cost,
                policy=policy,
                expected_end_state=state,
            )
            cost_accounts[policy] = state
            cost_hashes[policy] = raw_checkpoint["account_state_sha256"]
            cost_ledgers[policy] = frame
        accounts[cost] = MappingProxyType(cost_accounts)
        account_hashes[cost] = MappingProxyType(cost_hashes)
        ledgers[cost] = MappingProxyType(cost_ledgers)
    return (
        checkpoint_bytes,
        copy.deepcopy(parsed),
        selected,
        MappingProxyType(accounts),
        MappingProxyType(account_hashes),
        MappingProxyType(ledgers),
    )


def verify_parent_bundle(repo_root: Path) -> ParentBundleEvidence:
    """Verify and return the sole authorized through-2023 parent state."""

    root = _require_repository_root(repo_root)
    _require_parent_git_identity(root)
    bundle, manifest_bytes, checksums_bytes = _load_exact_parent_bundle(root)
    manifest = dict(bundle.manifest)
    report, _ = _canonical_object(bundle, REPORT_FILENAME)
    gate, _ = _canonical_object(bundle, GATE_REPORT_FILENAME)
    integrity, _ = _canonical_object(bundle, INTEGRITY_FILENAME)
    _require_omitted_field_self_hash(
        manifest,
        field="manifest_sha256",
        expected=PARENT_MANIFEST_SELF_SHA256,
        label="manifest",
    )
    _require_omitted_field_self_hash(
        report,
        field="report_sha256",
        expected=PARENT_REPORT_SELF_SHA256,
        label="report",
    )
    _require_omitted_field_self_hash(
        integrity,
        field="integrity_evidence_sha256",
        expected=PARENT_INTEGRITY_SELF_SHA256,
        label="integrity evidence",
    )
    _require_parent_claims(manifest, report, gate, integrity)
    (
        checkpoint_bytes,
        parsed_checkpoint,
        checkpoint,
        accounts,
        account_hashes,
        ledgers,
    ) = _parse_checkpoint_accounts_and_ledgers(bundle)
    if (
        manifest.get("checkpoint_sha256") != PARENT_CHECKPOINT_SELF_SHA256
        or report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SELF_SHA256
    ):
        raise ParentBundleVerificationError(
            "parent seal/report checkpoint binding changed"
        )

    return ParentBundleEvidence(
        verifier_id=VERIFIER_ID,
        verifier_dependency_path=VERIFIER_IMPLEMENTATION_PATH.as_posix(),
        verified=True,
        parent_directory=PARENT_DIRECTORY.as_posix(),
        preservation_commit=PARENT_PRESERVATION_COMMIT,
        original_run_commit=PARENT_RUN_COMMIT,
        contract_version=PARENT_CONTRACT_VERSION,
        stage=PARENT_STAGE,
        run_id=PARENT_RUN_ID,
        status=PARENT_STATUS,
        stage_pass=True,
        historical_policy_candidate=True,
        learning_candidate=False,
        passed_gate_count=64,
        total_gate_count=64,
        manifest_file_sha256=PARENT_MANIFEST_FILE_SHA256,
        manifest_self_sha256=PARENT_MANIFEST_SELF_SHA256,
        checksums_file_sha256=PARENT_CHECKSUMS_FILE_SHA256,
        report_file_sha256=PARENT_REPORT_FILE_SHA256,
        report_self_sha256=PARENT_REPORT_SELF_SHA256,
        gate_report_file_sha256=PARENT_GATE_REPORT_FILE_SHA256,
        metrics_file_sha256=PARENT_METRICS_FILE_SHA256,
        integrity_file_sha256=PARENT_INTEGRITY_FILE_SHA256,
        integrity_self_sha256=PARENT_INTEGRITY_SELF_SHA256,
        checkpoint_file_sha256=PARENT_CHECKPOINT_FILE_SHA256,
        checkpoint_self_sha256=PARENT_CHECKPOINT_SELF_SHA256,
        exact_payload_names=tuple(sorted(bundle.payload_sha256)),
        payload_sha256=MappingProxyType(dict(bundle.payload_sha256)),
        bundle=bundle,
        manifest_bytes=manifest_bytes,
        checksums_bytes=checksums_bytes,
        checkpoint_bytes=checkpoint_bytes,
        parsed_checkpoint=parsed_checkpoint,
        checkpoint=checkpoint,
        model_payload=copy.deepcopy(dict(checkpoint.model_payload)),
        accounts=accounts,
        account_checkpoint_sha256=account_hashes,
        ledgers=ledgers,
    )


__all__ = [
    "CHECKPOINT_FILENAME",
    "PARENT_CHECKPOINT_FILE_SHA256",
    "PARENT_CHECKPOINT_SELF_SHA256",
    "PARENT_CHECKSUMS_FILE_SHA256",
    "PARENT_CONTRACT_VERSION",
    "PARENT_DIRECTORY",
    "PARENT_MANIFEST_FILE_SHA256",
    "PARENT_MANIFEST_SELF_SHA256",
    "PARENT_PRESERVATION_COMMIT",
    "PARENT_RUN_COMMIT",
    "PARENT_RUN_ID",
    "PARENT_STAGE",
    "PARENT_STATUS",
    "ParentBundleEvidence",
    "ParentBundleVerificationError",
    "REQUIRED_POLICIES",
    "SCENARIO_PARENT_POLICY",
    "SELECTED_POLICY",
    "VERIFIER_ID",
    "VERIFIER_IMPLEMENTATION_PATH",
    "verify_parent_bundle",
]
