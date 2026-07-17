from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
from typing import Any

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
    validate_implementation_manifest,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    BRANCH_NAME,
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    EXTERNAL_PUBLICATION_FIELDS,
    EXTERNAL_TAG_MESSAGE_FIELDS,
    EXTERNAL_TAG_REF_TEMPLATE,
    FINAL_ATTEMPT_ID,
    FINAL_REGISTRY_TAG_REF_TEMPLATE,
    PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256,
    PUBLICATION_NORMAL_OPERATION_SHA256,
    PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS,
    PUBLICATION_REMOTE_OBSERVATION_FIELDS,
    PUBLICATION_REMOTE_REF_ABSENT_SENTINEL,
    PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
    PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL,
    SOURCE_PIN_FILES,
    SOURCE_PINS,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
    ALLOWED_REMOTE_URL,
    EXTERNAL_PUBLICATION_SCHEMA_VERSION,
    EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION,
    FINAL_REGISTRY_SUCCESSOR,
    FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE,
    NORMAL_PUBLICATION,
    PRE_PUSH,
    PUBLICATION_GENESIS_SHA256,
    REGISTERED_UNRUN,
    SCORED_PASS,
    TERMINAL_PASS,
    ExternalGitTagPublisher,
    PublicationProcessExecution,
    SecGemmaOnlineRiskOverlayPublisherError,
    VerifiedExternalPublication,
    build_transport_executable_identity_pins,
    git_runtime_dependency_closure_sha256,
    is_verified_external_publication,
    issue_verified_external_publication_from_observation,
    prepare_external_publication,
    prepare_isolated_publication_transport,
    validate_external_publication,
    verify_pinned_transport_executables,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    CONTRACT_SOURCE_PATH,
    EXPECTED_ORIGIN_URL,
    PREREGISTRATION_COMMIT,
    REQUIRED_NEW_SOURCE_PATHS,
    SOURCE_VERIFICATION_SCHEMA_VERSION,
)
from tests.sec_gemma_online_risk_overlay_helpers import (
    build_synthetic_numerical_time_distributions,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _manifest(commit: str = "1" * 40) -> dict[str, Any]:
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


def _pins() -> dict[str, Any]:
    body = {
        "git_executable_path": (
            "C:/Program Files/Git/mingw64/bin/git.exe"
        ),
        "git_executable_sha256": _digest("git"),
        "git_version_stdout_sha256": _digest("git-version"),
        "git_exec_path": (
            "C:/Program Files/Git/mingw64/libexec/git-core"
        ),
        "git_exec_path_directory_manifest_sha256": _digest("exec-path"),
        "git_remote_https_executable_path": (
            "C:/Program Files/Git/mingw64/libexec/"
            "git-core/git-remote-https.exe"
        ),
        "git_remote_https_executable_sha256": _digest("remote-https"),
        "credential_helper_config_value": (
            FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE
        ),
        "credential_helper_executable_path": (
            "C:/Program Files/Git/mingw64/bin/"
            "git-credential-manager.exe"
        ),
        "credential_helper_executable_sha256": _digest("gcm"),
        "credential_helper_version_stdout_sha256": _digest("gcm-version"),
        "command_interpreter_executable_path": (
            "C:/Program Files/Git/usr/bin/sh.exe"
        ),
        "command_interpreter_executable_sha256": _digest("shell"),
    }
    return {
        **body,
        "transport_executable_closure_manifest_sha256": (
            canonical_sha256(body)
        ),
    }


def test_git_runtime_pins_bind_real_executables_and_full_dependency_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher

    root = (tmp_path / "Git").resolve()
    for relative in (
        "cmd",
        "bin",
        "mingw64/bin",
        "mingw64/libexec/git-core",
        "usr/bin",
        "etc",
        "mingw64/etc",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    git = root / "mingw64" / "bin" / "git.exe"
    remote_https = (
        root
        / "mingw64"
        / "libexec"
        / "git-core"
        / "git-remote-https.exe"
    )
    helper = (
        root
        / "mingw64"
        / "bin"
        / "git-credential-manager.exe"
    )
    shell = root / "usr" / "bin" / "sh.exe"
    dependency = root / "mingw64" / "etc" / "gitconfig"
    wrapper = root / "cmd" / "git.exe"
    for path, payload in (
        (git, b"real-git"),
        (remote_https, b"real-remote-helper"),
        (helper, b"real-credential-helper"),
        (shell, b"real-compiled-shell"),
        (dependency, b"dependency-v1"),
        (wrapper, b"real-git"),
    ):
        path.write_bytes(payload)
    monkeypatch.setattr(
        publisher,
        "FROZEN_CREDENTIAL_HELPER_PATH",
        str(helper).replace("\\", "/"),
    )

    pins = build_transport_executable_identity_pins(
        git_executable_path=git,
        git_version_stdout=b"git version test\n",
        git_exec_path=remote_https.parent,
        git_remote_https_executable_path=remote_https,
        credential_helper_executable_path=helper,
        credential_helper_version_stdout=b"gcm test\n",
        command_interpreter_executable_path=shell,
    )
    durable_closure = pins[
        "transport_executable_closure_manifest_sha256"
    ]

    assert durable_closure == git_runtime_dependency_closure_sha256(root)
    verify_pinned_transport_executables(
        pins,
        expected_dependency_closure_sha256=durable_closure,
    )

    wrapper_pins = {**pins, "git_executable_path": str(wrapper)}
    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="not the real executable",
    ):
        verify_pinned_transport_executables(wrapper_pins)

    dependency.write_bytes(b"dependency-v2")
    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="dependency closure changed",
    ):
        verify_pinned_transport_executables(
            pins,
            expected_dependency_closure_sha256=durable_closure,
        )


def _host_environment() -> dict[str, str]:
    return {
        "SystemRoot": "C:/Windows",
        "WINDIR": "C:/Windows",
        "COMSPEC": "C:/Windows/System32/cmd.exe",
        "PATH": "C:/Program Files/Git/cmd",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "TEMP": "C:/Temp",
        "TMP": "C:/Temp",
        "USERPROFILE": "C:/Users/test",
        "LOCALAPPDATA": "C:/Users/test/AppData/Local",
        "APPDATA": "C:/Users/test/AppData/Roaming",
        "HOME": "C:/Users/test",
    }


def _prepared():
    return prepare_external_publication(
        implementation_manifest=_manifest(),
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        terminal_status=TERMINAL_PASS,
        report_kind=SCORED_PASS,
        artifact_sha256=_digest("artifact"),
        predecessor_publication_sha256=PUBLICATION_GENESIS_SHA256,
    )


def _transport(tmp_path: Path, prepared=None):
    if prepared is None:
        prepared = _prepared()
    source_objects = tmp_path / "source-objects"
    source_objects.mkdir(parents=True)
    return prepare_isolated_publication_transport(
        transport_root=(tmp_path / "isolated").resolve(),
        source_object_directory=source_objects.resolve(),
        prepared_publication=prepared,
        executable_pins=_pins(),
        host_environment_values=_host_environment(),
        identity_verifier=lambda pins: None,
        store_instance_id=_digest("store"),
        store_session_nonce_sha256=_digest("session"),
        publication_intent_sha256=_digest("intent"),
        operation_kind=NORMAL_PUBLICATION,
        operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
        pre_transport_store_journal_sequence=12,
        pre_transport_store_journal_tip_sha256=_digest("tip"),
    )


class _Executor:
    def __init__(self, results: list[PublicationProcessExecution]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> PublicationProcessExecution:
        self.calls.append(copy.deepcopy(kwargs))
        return self.results.pop(0)


class _FinalRegistryRunner:
    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        existing_tag: tuple[str, str, str] | None = None,
    ) -> None:
        self.manifest = manifest
        self.remote_tag = existing_tag
        self.calls: list[tuple[str, ...]] = []
        self.raw_commands: list[tuple[str, ...]] = []

    def __call__(self, command: list[str], **kwargs: Any):
        self.raw_commands.append(tuple(command))
        index = 1
        while command[index] == "-c":
            index += 2
        args = tuple(command[index:])
        self.calls.append(args)
        stdout = b""
        stderr = b""
        returncode = 0
        if args[:3] == ("rev-parse", "--verify", "HEAD"):
            stdout = (
                self.manifest["implementation_commit"].encode("ascii")
                + b"\n"
            )
        elif args[:2] == ("status", "--porcelain=v1"):
            stdout = b""
        elif args == ("config", "--get", "remote.origin.url"):
            stdout = ALLOWED_REMOTE_URL.encode("ascii") + b"\n"
        elif args == (
            "config",
            "--get-all",
            "remote.origin.pushurl",
        ):
            returncode = 1
        elif args[:2] == ("config", "--get-regexp"):
            returncode = 1
        elif args[:2] == ("ls-remote", ALLOWED_REMOTE_URL):
            ref = args[2]
            stdout = (
                f"{self.manifest['implementation_commit']}\t{ref}\n"
            ).encode("ascii")
        elif args[:2] == ("ls-remote", "--tags"):
            if self.remote_tag is not None:
                oid, peeled, ref = self.remote_tag
                stdout = (
                    f"{oid}\t{ref}\n"
                    f"{peeled}\t{ref}^{{}}\n"
                ).encode("ascii")
        elif args == ("mktag",):
            payload = kwargs["input"]
            material = (
                f"tag {len(payload)}\0".encode("ascii") + payload
            )
            oid = hashlib.sha1(
                material,
                usedforsecurity=False,
            ).hexdigest()
            stdout = oid.encode("ascii") + b"\n"
        elif args[:4] == (
            "push",
            "--porcelain",
            "--no-verify",
            ALLOWED_REMOTE_URL,
        ):
            oid, ref = args[4].split(":", 1)
            self.remote_tag = (
                oid,
                self.manifest["implementation_commit"],
                ref,
            )
            stdout = b"ok\n"
        else:
            raise AssertionError(f"Unexpected final-registry command: {args}")
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout,
            stderr=stderr,
        )


def _final_registry_repo(tmp_path: Path) -> Path:
    repo = (tmp_path / "repo").resolve()
    git_directory = repo / ".git"
    git_directory.mkdir(parents=True)
    (git_directory / "config").write_bytes(
        (
            "[core]\n"
            "\trepositoryformatversion = 0\n"
            "\tfilemode = false\n"
            "\tbare = false\n"
            "\tlogallrefupdates = true\n"
            '[remote "origin"]\n'
            f"\turl = {ALLOWED_REMOTE_URL}\n"
            "\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
        ).encode("utf-8")
    )
    return repo


def _test_only_bare_remote(tmp_path: Path) -> Path:
    remote = (tmp_path / "remote.git").resolve()
    for name in ("objects", "refs", "hooks"):
        (remote / name).mkdir(parents=True, exist_ok=True)
    (remote / "HEAD").write_text(
        "ref: refs/heads/main\n",
        encoding="utf-8",
    )
    (remote / "config").write_text(
        (
            "[core]\n"
            "\trepositoryformatversion = 0\n"
            "\tfilemode = false\n"
            "\tbare = true\n"
            "\tlongpaths = true\n"
        ),
        encoding="utf-8",
    )
    return remote


def _add_exact_test_only_registry_config(repo: Path, remote: Path) -> None:
    config_path = repo / ".git" / "config"
    config_path.write_bytes(
        config_path.read_bytes()
        + (
            "[core]\n"
            "\tautocrlf = false\n"
            "\tlongpaths = true\n"
            "[user]\n"
            "\tname = Registry Test\n"
            "\temail = registry@example.invalid\n"
            f'[url "{remote.as_uri()}"]\n'
            f"\tinsteadOf = {ALLOWED_REMOTE_URL}\n"
        ).encode("utf-8")
    )


def _read(
    *,
    tmp_path: Path,
    execution: PublicationProcessExecution,
):
    prepared = _prepared()
    transport = _transport(tmp_path, prepared)
    executor = _Executor([execution])
    result = ExternalGitTagPublisher(
        process_executor=executor
    ).read_remote(
        prepared_publication=prepared,
        transport=transport,
        durable_transport_manifest=_Opaque(transport.manifest),
        transport_manifest_validator=_opaque_validator,
        store_instance_id=_digest("store"),
        store_session_nonce_sha256=_digest("session"),
        publication_intent_sha256=_digest("intent"),
        operation_kind=NORMAL_PUBLICATION,
        operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
        worker_ownership=_Opaque(_ownership(transport)),
        worker_ownership_validator=_opaque_validator,
        observation_ordinal=1,
        observation_phase=PRE_PUSH,
        prior_push_command_sha256=(
            PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
        ),
        pre_observation_store_journal_sequence=20,
        pre_observation_store_journal_tip_sha256=_digest("observation-tip"),
        timeout_seconds=90,
    )
    return prepared, transport, executor, result


def _opaque_validator(value: Any) -> dict[str, Any]:
    return copy.deepcopy(value.payload)


class _Opaque:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = copy.deepcopy(payload)


def _ownership(transport) -> dict[str, Any]:
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-worker-ownership-v1"
        ),
        "owner_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-worker-owner-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": transport.manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": transport.manifest[
            "implementation_commit"
        ],
        "store_instance_id": _digest("store"),
        "store_session_nonce_sha256": _digest("session"),
        "attempt_id": CONFIRMATION_ATTEMPT_ID,
        "publication_intent_sha256": _digest("intent"),
        "operation_kind": NORMAL_PUBLICATION,
        "operation_sha256": PUBLICATION_NORMAL_OPERATION_SHA256,
        "owner_nonce_sha256": _digest("owner-nonce"),
        "owner_process_id": 1234,
        "owner_process_creation_filetime_hex": "0x1.0000000000000p+0",
        "job_object_name_sha256": _digest("job"),
        "owner_mutex_name_sha256": _digest("mutex"),
        "kill_on_parent_exit": True,
        "child_assignment_before_resume_required": True,
        "ownership_status": "claimed",
    }
    return {
        **body,
        "worker_ownership_sha256": canonical_sha256(body),
    }


def _authorization(
    *,
    prepared,
    transport,
    observation: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-pre-push-authorization-v1"
        ),
        "authorization_verifier_id": (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-pre-push-authorization-verifier-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": transport.manifest[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": transport.manifest[
            "implementation_commit"
        ],
        "store_instance_id": _digest("store"),
        "store_session_nonce_sha256": _digest("session"),
        "attempt_id": CONFIRMATION_ATTEMPT_ID,
        "publication_intent_sha256": _digest("intent"),
        "authorization_operation_kind": NORMAL_PUBLICATION,
        "authorization_operation_sha256": (
            PUBLICATION_NORMAL_OPERATION_SHA256
        ),
        "authorization_operation_ordinal": 0,
        "worker_ownership_sha256": observation[
            "worker_ownership_sha256"
        ],
        "pre_push_authorization_marker_key": _digest("marker"),
        "remote_observation_sha256": observation[
            "publication_remote_observation_sha256"
        ],
        "tag_ref": prepared.tag_ref,
        "expected_tag_object_sha1": (
            prepared.expected_tag_object_sha1
        ),
        "authorization_status": "push_authorized_once",
        "push_command_limit": 1,
    }
    result = {
        **body,
        "pre_push_authorization_sha256": canonical_sha256(body),
    }
    assert tuple(result) == PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS
    return result


def test_v2_2_tag_message_and_stable_ref_are_exact() -> None:
    first = _prepared()
    second = prepare_external_publication(
        implementation_manifest=_manifest(),
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        terminal_status=TERMINAL_PASS,
        report_kind=SCORED_PASS,
        artifact_sha256=_digest("different-artifact"),
        predecessor_publication_sha256=PUBLICATION_GENESIS_SHA256,
    )

    assert tuple(first.tag_message) == EXTERNAL_TAG_MESSAGE_FIELDS
    assert first.tag_message["schema_version"] == (
        EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION
    )
    assert "v2-2" in first.tag_message["schema_version"]
    assert first.tag_ref == EXTERNAL_TAG_REF_TEMPLATE.format(
        attempt_id=CONFIRMATION_ATTEMPT_ID
    )
    assert second.tag_ref == first.tag_ref
    assert second.expected_tag_object_sha1 != (
        first.expected_tag_object_sha1
    )


def test_final_registry_uses_separate_stable_namespace() -> None:
    prepared = prepare_external_publication(
        implementation_manifest=_manifest(),
        attempt_id=FINAL_ATTEMPT_ID,
        terminal_status=REGISTERED_UNRUN,
        report_kind=FINAL_REGISTRY_SUCCESSOR,
        artifact_sha256=_digest("registry"),
        predecessor_publication_sha256=_digest("prior-publication"),
    )

    assert prepared.tag_ref == FINAL_REGISTRY_TAG_REF_TEMPLATE.format(
        attempt_id=FINAL_ATTEMPT_ID
    )


def test_canonical_tag_object_and_expected_publication_are_precomputed() -> None:
    prepared = _prepared()
    payload = prepared.tag_object_bytes
    material = f"tag {len(payload)}\0".encode("ascii") + payload
    expected_oid = hashlib.sha1(
        material,
        usedforsecurity=False,
    ).hexdigest()

    assert prepared.expected_tag_object_sha1 == expected_oid
    assert b" 0 +0000\n\n" in payload
    assert prepared.intent_material["expected_tag_object_sha1"] == expected_oid
    assert prepared.intent_material["expected_publication_sha256"] == (
        prepared.expected_publication["publication_sha256"]
    )


def test_isolated_transport_has_exact_config_environment_and_loose_tag(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    transport = _transport(tmp_path, prepared)
    manifest = transport.manifest
    config = (transport.directory / "config").read_text(encoding="utf-8")
    hooks = Path(manifest["empty_hooks_directory_path"])
    object_path = (
        transport.directory
        / "objects"
        / prepared.expected_tag_object_sha1[:2]
        / prepared.expected_tag_object_sha1[2:]
    )

    assert hooks.is_dir()
    assert list(hooks.iterdir()) == []
    assert "remote" not in config
    assert "url." not in config
    assert FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE in config
    assert manifest["remote_url"] == ALLOWED_REMOTE_URL
    assert manifest["path_lookup_forbidden"] is True
    assert object_path.is_file()
    environment = transport.environment
    assert tuple(environment) == (
        "SystemRoot",
        "WINDIR",
        "COMSPEC",
        "PATH",
        "PATHEXT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "HOME",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_COUNT",
        "GIT_TERMINAL_PROMPT",
        "GIT_ALLOW_PROTOCOL",
        "GIT_PROTOCOL_FROM_USER",
        "GIT_EXEC_PATH",
        "GCM_INTERACTIVE",
        "LANG",
        "LC_ALL",
    )
    for forbidden in (
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_SSH_COMMAND",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ):
        assert forbidden not in environment


def test_empty_readback_is_absent_but_nonzero_is_unknown(
    tmp_path: Path,
) -> None:
    _, _, executor, absent = _read(
        tmp_path=tmp_path,
        execution=PublicationProcessExecution(
            process_exit_status="exited",
            process_exit_code=0,
            stdout=b"",
            stderr=b"",
        ),
    )

    assert absent.observed_ref_state == "absent"
    assert absent.evidence["ref_lookup_status"] == "absent"
    assert absent.evidence["raw_ref_object_value"] == (
        PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
    )
    assert executor.calls[0]["argv"][0] == _pins()["git_executable_path"]
    assert ALLOWED_REMOTE_URL in executor.calls[0]["argv"]
    assert "origin" not in executor.calls[0]["argv"]

    _, _, _, failed = _read(
        tmp_path=(tmp_path / "failed"),
        execution=PublicationProcessExecution(
            process_exit_status="exited",
            process_exit_code=128,
            stdout=b"",
            stderr=b"network unavailable",
        ),
    )
    assert failed.observation is None
    assert failed.evidence["transport_status"] == "unavailable"
    assert failed.evidence["ref_lookup_status"] == "unknown"
    assert failed.evidence["raw_ref_object_value"] == (
        PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
    )


def test_malformed_raw_bytes_never_become_an_observation(
    tmp_path: Path,
) -> None:
    _, _, _, result = _read(
        tmp_path=tmp_path,
        execution=PublicationProcessExecution(
            process_exit_status="exited",
            process_exit_code=0,
            stdout=b"not-a-git-row\n",
            stderr=b"",
        ),
    )

    assert result.observation is None
    assert result.evidence["transport_status"] == "protocol_error"
    assert result.evidence["ref_lookup_status"] == "unknown"
    assert result.evidence["raw_ref_object_value"] == (
        PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL
    )


def test_exact_readback_issues_recovery_friendly_opaque_receipt(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    stdout = (
        f"{prepared.expected_tag_object_sha1}\t{prepared.tag_ref}\n"
        f"{prepared.expected_peeled_commit}\t{prepared.tag_ref}^{{}}\n"
    ).encode("ascii")
    prepared, _, _, result = _read(
        tmp_path=tmp_path,
        execution=PublicationProcessExecution(
            process_exit_status="exited",
            process_exit_code=0,
            stdout=stdout,
            stderr=b"",
        ),
    )
    observation = result.observation
    assert observation is not None
    assert tuple(observation) == PUBLICATION_REMOTE_OBSERVATION_FIELDS
    assert observation["observed_ref_state"] == "exact_expected"

    receipt = issue_verified_external_publication_from_observation(
        prepared_publication=prepared,
        durable_remote_observation=_Opaque(observation),
        observation_validator=_opaque_validator,
    )

    assert is_verified_external_publication(receipt)
    assert tuple(receipt.publication) == EXTERNAL_PUBLICATION_FIELDS
    assert receipt.publication["schema_version"] == (
        EXTERNAL_PUBLICATION_SCHEMA_VERSION
    )
    assert validate_external_publication(
        receipt.publication,
        implementation_manifest=_manifest(),
    ) == receipt.publication
    with pytest.raises(SecGemmaOnlineRiskOverlayPublisherError):
        VerifiedExternalPublication(
            publication=receipt.publication,
            _sentinel=object(),
        )


def test_conflicting_present_ref_cannot_issue_receipt(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    stdout = (
        f"{'a' * 40}\t{prepared.tag_ref}\n"
        f"{prepared.expected_peeled_commit}\t{prepared.tag_ref}^{{}}\n"
    ).encode("ascii")
    prepared, _, _, result = _read(
        tmp_path=tmp_path,
        execution=PublicationProcessExecution(
            process_exit_status="exited",
            process_exit_code=0,
            stdout=stdout,
            stderr=b"",
        ),
    )
    assert result.observed_ref_state == "conflicting"

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="does not prove",
    ):
        issue_verified_external_publication_from_observation(
            prepared_publication=prepared,
            durable_remote_observation=_Opaque(result.observation),
            observation_validator=_opaque_validator,
        )


def test_push_is_one_literal_no_force_no_hook_command(
    tmp_path: Path,
) -> None:
    prepared, transport, _, absent = _read(
        tmp_path=tmp_path,
        execution=PublicationProcessExecution(
            process_exit_status="exited",
            process_exit_code=0,
            stdout=b"",
            stderr=b"",
        ),
    )
    observation = absent.observation
    assert observation is not None
    authorization = _authorization(
        prepared=prepared,
        transport=transport,
        observation=observation,
    )
    executor = _Executor(
        [
            PublicationProcessExecution(
                process_exit_status="exited",
                process_exit_code=0,
                stdout=b"To github\n",
                stderr=b"",
            )
        ]
    )
    publisher = ExternalGitTagPublisher(process_executor=executor)
    result = publisher.push_once(
        prepared_publication=prepared,
        transport=transport,
        pre_push_authorization=_Opaque(authorization),
        authorization_validator=_opaque_validator,
        absent_remote_observation=_Opaque(observation),
        observation_validator=_opaque_validator,
        timeout_seconds=90,
    )

    assert result.push_command_count_upper_bound == 1
    assert len(executor.calls) == 1
    argv = executor.calls[0]["argv"]
    assert argv == (
        _pins()["git_executable_path"],
        "push",
        "--porcelain",
        "--no-verify",
        ALLOWED_REMOTE_URL,
        f"{prepared.expected_tag_object_sha1}:{prepared.tag_ref}",
    )
    assert "--force" not in argv
    assert "-f" not in argv
    assert executor.calls[0]["cwd"] == transport.directory
    assert executor.calls[0]["env"] == transport.environment

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="already consumed",
    ):
        publisher.push_once(
            prepared_publication=prepared,
            transport=transport,
            pre_push_authorization=_Opaque(authorization),
            authorization_validator=_opaque_validator,
            absent_remote_observation=_Opaque(observation),
            observation_validator=_opaque_validator,
            timeout_seconds=90,
        )


def test_transport_tampering_fails_before_executor_is_called(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    transport = _transport(tmp_path, prepared)
    Path(transport.manifest["empty_hooks_directory_path"]).joinpath(
        "pre-push"
    ).write_bytes(b"malicious")
    executor = _Executor(
        [
            PublicationProcessExecution(
                process_exit_status="exited",
                process_exit_code=0,
                stdout=b"",
                stderr=b"",
            )
        ]
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="transport directory|filesystem or environment changed",
    ):
        ExternalGitTagPublisher(
            process_executor=executor
        ).read_remote(
            prepared_publication=prepared,
            transport=transport,
            durable_transport_manifest=_Opaque(transport.manifest),
            transport_manifest_validator=_opaque_validator,
            store_instance_id=_digest("store"),
            store_session_nonce_sha256=_digest("session"),
            publication_intent_sha256=_digest("intent"),
            operation_kind=NORMAL_PUBLICATION,
            operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
            worker_ownership=_Opaque(_ownership(transport)),
            worker_ownership_validator=_opaque_validator,
            observation_ordinal=1,
            observation_phase=PRE_PUSH,
            prior_push_command_sha256=(
                PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
            ),
            pre_observation_store_journal_sequence=20,
            pre_observation_store_journal_tip_sha256=_digest("tip-2"),
            timeout_seconds=90,
        )
    assert executor.calls == []


def test_remote_url_is_literal_and_cannot_be_replaced() -> None:
    assert ALLOWED_REMOTE_URL == EXPECTED_ORIGIN_URL
    manifest = _manifest()
    changed = copy.deepcopy(manifest)
    changed["origin_url"] = "https://example.invalid/repo.git"
    body = {
        key: value
        for key, value in changed.items()
        if key != "implementation_manifest_sha256"
    }
    changed["implementation_manifest_sha256"] = canonical_sha256(body)

    with pytest.raises(Exception):
        prepare_external_publication(
            implementation_manifest=changed,
            attempt_id=CONFIRMATION_ATTEMPT_ID,
            terminal_status=TERMINAL_PASS,
            report_kind=SCORED_PASS,
            artifact_sha256=_digest("artifact"),
            predecessor_publication_sha256=PUBLICATION_GENESIS_SHA256,
        )


def test_final_registry_only_method_publishes_exact_stable_ref(
    tmp_path: Path,
) -> None:
    repo = _final_registry_repo(tmp_path)
    manifest = _manifest()
    runner = _FinalRegistryRunner(manifest=manifest)
    publisher = ExternalGitTagPublisher(
        repo_root=repo,
        implementation_manifest=manifest,
        final_registry_subprocess_runner=runner,
    )

    receipt = publisher.publish_or_recover_final_registry(
        attempt_id=FINAL_ATTEMPT_ID,
        terminal_status=REGISTERED_UNRUN,
        report_kind=FINAL_REGISTRY_SUCCESSOR,
        artifact_sha256=_digest("successor-registry"),
        predecessor_publication_sha256=_digest("confirmation-publication"),
        deadline_monotonic=10**12,
    )

    assert is_verified_external_publication(receipt)
    assert receipt.publication["tag_ref"] == (
        FINAL_REGISTRY_TAG_REF_TEMPLATE.format(
            attempt_id=FINAL_ATTEMPT_ID
        )
    )
    push_calls = [call for call in runner.calls if call[:1] == ("push",)]
    assert len(push_calls) == 1
    push = push_calls[0]
    assert push[:4] == (
        "push",
        "--porcelain",
        "--no-verify",
        ALLOWED_REMOTE_URL,
    )
    assert "--force" not in push
    reset = (
        "-c",
        "credential.helper=",
        "-c",
        f"credential.helper={FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE}",
    )
    assert runner.raw_commands
    assert all(
        any(
            command[index : index + len(reset)] == reset
            for index in range(len(command) - len(reset) + 1)
        )
        for command in runner.raw_commands
    )
    assert not hasattr(publisher, "publish")


def test_final_registry_rehashes_full_git_closure_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher_module

    repo = _final_registry_repo(tmp_path)
    manifest = _manifest()
    runner = _FinalRegistryRunner(manifest=manifest)
    publisher = ExternalGitTagPublisher(
        repo_root=repo,
        implementation_manifest=manifest,
        final_registry_subprocess_runner=runner,
    )
    monkeypatch.setattr(
        publisher_module,
        "git_runtime_dependency_closure_sha256",
        lambda root: _digest(f"changed:{root}"),
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="dependency closure changed",
    ):
        publisher.publish_or_recover_final_registry(
            attempt_id=FINAL_ATTEMPT_ID,
            terminal_status=REGISTERED_UNRUN,
            report_kind=FINAL_REGISTRY_SUCCESSOR,
            artifact_sha256=_digest("successor-registry"),
            predecessor_publication_sha256=(
                _digest("confirmation-publication")
            ),
            deadline_monotonic=10**12,
        )

    assert runner.calls == []


_HOSTILE_FINAL_REGISTRY_CONFIGS = (
    pytest.param("[include]\n\tpath = C:/evil/config\n", id="include"),
    pytest.param(
        '[includeIf "gitdir:C:/repo/"]\n\tpath = C:/evil/config\n',
        id="include-if",
    ),
    pytest.param(
        '[credential]\n\thelper = !"C:/evil/helper.exe"\n',
        id="credential-helper",
    ),
    pytest.param(
        "[core]\n\taskPass = C:/evil/askpass.exe\n",
        id="core-askpass",
    ),
    pytest.param(
        "[core]\n\tfsmonitor = C:/evil/fsmonitor.exe\n",
        id="core-fsmonitor",
    ),
    pytest.param(
        "[core]\n\tautocrlf = false\n",
        id="production-core-autocrlf",
    ),
    pytest.param(
        "[core]\n\tlongpaths = true\n",
        id="production-core-longpaths",
    ),
    pytest.param(
        "[user]\n\tname = Registry Test\n",
        id="production-user-name",
    ),
    pytest.param(
        "[user]\n\temail = registry@example.invalid\n",
        id="production-user-email",
    ),
    pytest.param(
        "[http]\n\tproxy = http://evil.invalid\n",
        id="http-proxy",
    ),
    pytest.param(
        '[http "https://github.com/"]\n\textraHeader = X-Evil: 1\n',
        id="scoped-http-extra-header",
    ),
    pytest.param(
        '[url "https://evil.invalid/"]\n\tinsteadOf = https://github.com/\n',
        id="url-rewrite",
    ),
    pytest.param(
        '[remote "origin"]\n\tpushurl = https://evil.invalid/repo.git\n',
        id="push-url",
    ),
    pytest.param(
        "[core]\n\tsshCommand = C:/evil/ssh.exe\n",
        id="ssh-command",
    ),
    pytest.param(
        "[core]\n\tgitProxy = C:/evil/proxy.exe\n",
        id="git-proxy",
    ),
    pytest.param(
        "[core]\n\thooksPath = C:/evil/hooks\n",
        id="hooks-path",
    ),
    pytest.param(
        '[filter "escape"]\n\tprocess = C:/evil/filter.exe\n',
        id="filter-process",
    ),
    pytest.param(
        "[diff]\n\texternal = C:/evil/diff.exe\n",
        id="diff-external",
    ),
    pytest.param(
        '[difftool "escape"]\n\tcmd = C:/evil/difftool.exe\n',
        id="difftool-command",
    ),
    pytest.param(
        '[mergetool "escape"]\n\tcmd = C:/evil/mergetool.exe\n',
        id="mergetool-command",
    ),
    pytest.param(
        "[alias]\n\tstatus = !C:/evil/alias.exe\n",
        id="external-alias",
    ),
    pytest.param(
        "[pager]\n\tstatus = C:/evil/pager.exe\n",
        id="pager-command",
    ),
    pytest.param(
        "[gpg]\n\tprogram = C:/evil/gpg.exe\n",
        id="gpg-program",
    ),
    pytest.param(
        '[tar "escape"]\n\tcommand = C:/evil/tar.exe\n',
        id="tar-command",
    ),
    pytest.param(
        '[remote "origin"]\n\tuploadpack = C:/evil/upload-pack.exe\n',
        id="remote-upload-pack",
    ),
    pytest.param(
        "[core]\n\teditor = C:/evil/editor.exe\n",
        id="core-editor",
    ),
    pytest.param(
        "[sequence]\n\teditor = C:/evil/editor.exe\n",
        id="sequence-editor",
    ),
    pytest.param(
        "[interactive]\n\tdiffFilter = C:/evil/filter.exe\n",
        id="interactive-diff-filter",
    ),
)


@pytest.mark.parametrize("target", ("config", "config.worktree"))
@pytest.mark.parametrize("hostile_config", _HOSTILE_FINAL_REGISTRY_CONFIGS)
def test_final_registry_rejects_hostile_local_and_worktree_config_without_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    hostile_config: str,
) -> None:
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher_module

    repo = _final_registry_repo(tmp_path)
    config_path = repo / ".git" / target
    if target == "config":
        config_path.write_bytes(
            config_path.read_bytes() + hostile_config.encode("utf-8")
        )
    else:
        config_path.write_text(hostile_config, encoding="utf-8")
    manifest = _manifest()
    runner = _FinalRegistryRunner(manifest=manifest)
    monkeypatch.setattr(
        publisher_module,
        "git_runtime_dependency_closure_sha256",
        lambda root: _digest(f"closure:{root}"),
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="local/worktree Git configuration is unsafe",
    ):
        ExternalGitTagPublisher(
            repo_root=repo,
            implementation_manifest=manifest,
            final_registry_subprocess_runner=runner,
        )

    assert runner.calls == []
    assert runner.raw_commands == []


def test_production_final_registry_rejects_exact_test_file_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher_module

    repo = _final_registry_repo(tmp_path)
    remote = _test_only_bare_remote(tmp_path)
    config_path = repo / ".git" / "config"
    config_path.write_bytes(
        config_path.read_bytes()
        + (
            f'[url "{remote.as_uri()}"]\n'
            f"\tinsteadOf = {ALLOWED_REMOTE_URL}\n"
        ).encode("utf-8")
    )
    manifest = _manifest()
    runner = _FinalRegistryRunner(manifest=manifest)
    monkeypatch.setattr(
        publisher_module,
        "git_runtime_dependency_closure_sha256",
        lambda root: _digest(f"closure:{root}"),
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="local/worktree Git configuration is unsafe",
    ):
        ExternalGitTagPublisher(
            repo_root=repo,
            implementation_manifest=manifest,
            final_registry_subprocess_runner=runner,
        )

    assert runner.raw_commands == []


def test_test_only_final_registry_accepts_only_explicit_exact_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher_module

    repo = _final_registry_repo(tmp_path)
    remote = _test_only_bare_remote(tmp_path)
    _add_exact_test_only_registry_config(repo, remote)
    manifest = _manifest()
    runner = _FinalRegistryRunner(manifest=manifest)
    monkeypatch.setattr(
        publisher_module,
        "git_runtime_dependency_closure_sha256",
        lambda root: _digest(f"closure:{root}"),
    )

    publisher = ExternalGitTagPublisher(
        repo_root=repo,
        implementation_manifest=manifest,
        test_only_allow_url_rewrite=True,
        test_only_url_rewrite_target=remote,
        final_registry_subprocess_runner=runner,
    )
    publisher._verify_final_registry_mode()

    assert publisher._final_test_url_rewrite_target == remote
    assert runner.raw_commands == []


@pytest.mark.parametrize(
    "mutation",
    (
        "autocrlf-value",
        "longpaths-value",
        "user-name",
        "user-email",
        "rewrite-target",
        "rewrite-source",
        "duplicate-field",
    ),
)
def test_test_only_final_registry_rejects_every_fixture_deviation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher_module

    repo = _final_registry_repo(tmp_path)
    remote = _test_only_bare_remote(tmp_path)
    _add_exact_test_only_registry_config(repo, remote)
    config_path = repo / ".git" / "config"
    text = config_path.read_text(encoding="utf-8")
    replacements = {
        "autocrlf-value": ("\tautocrlf = false\n", "\tautocrlf = true\n"),
        "longpaths-value": ("\tlongpaths = true\n", "\tlongpaths = false\n"),
        "user-name": ("\tname = Registry Test\n", "\tname = Other\n"),
        "user-email": (
            "\temail = registry@example.invalid\n",
            "\temail = other@example.invalid\n",
        ),
        "rewrite-target": (
            remote.as_uri(),
            (tmp_path / "other.git").resolve().as_uri(),
        ),
        "rewrite-source": (
            f"\tinsteadOf = {ALLOWED_REMOTE_URL}\n",
            "\tinsteadOf = https://example.invalid/repo.git\n",
        ),
    }
    if mutation == "duplicate-field":
        text += "[core]\n\tautocrlf = false\n"
    else:
        old, new = replacements[mutation]
        text = text.replace(old, new, 1)
    config_path.write_text(text, encoding="utf-8")
    manifest = _manifest()
    runner = _FinalRegistryRunner(manifest=manifest)
    monkeypatch.setattr(
        publisher_module,
        "git_runtime_dependency_closure_sha256",
        lambda root: _digest(f"closure:{root}"),
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="local/worktree Git configuration is unsafe",
    ):
        ExternalGitTagPublisher(
            repo_root=repo,
            implementation_manifest=manifest,
            test_only_allow_url_rewrite=True,
            test_only_url_rewrite_target=remote,
            final_registry_subprocess_runner=runner,
        )

    assert runner.raw_commands == []


def test_test_only_final_registry_requires_matching_flag_and_explicit_target(
    tmp_path: Path,
) -> None:
    repo = _final_registry_repo(tmp_path)
    remote = _test_only_bare_remote(tmp_path)
    _add_exact_test_only_registry_config(repo, remote)
    manifest = _manifest()

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="requires one explicit target",
    ):
        ExternalGitTagPublisher(
            repo_root=repo,
            implementation_manifest=manifest,
            test_only_allow_url_rewrite=True,
        )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="configuration is incomplete",
    ):
        ExternalGitTagPublisher(
            repo_root=repo,
            implementation_manifest=manifest,
            test_only_url_rewrite_target=remote,
        )


def test_test_only_final_registry_reverifies_bare_target_identity_before_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher_module

    repo = _final_registry_repo(tmp_path)
    remote = _test_only_bare_remote(tmp_path)
    _add_exact_test_only_registry_config(repo, remote)
    manifest = _manifest()
    runner = _FinalRegistryRunner(manifest=manifest)
    monkeypatch.setattr(
        publisher_module,
        "git_runtime_dependency_closure_sha256",
        lambda root: _digest(f"closure:{root}"),
    )
    publisher = ExternalGitTagPublisher(
        repo_root=repo,
        implementation_manifest=manifest,
        test_only_allow_url_rewrite=True,
        test_only_url_rewrite_target=remote,
        final_registry_subprocess_runner=runner,
    )
    bare_config = remote / "config"
    bare_config.write_bytes(bare_config.read_bytes() + b"\n# changed\n")

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="rewrite target identity changed",
    ):
        publisher.publish_or_recover_final_registry(
            attempt_id=FINAL_ATTEMPT_ID,
            terminal_status=REGISTERED_UNRUN,
            report_kind=FINAL_REGISTRY_SUCCESSOR,
            artifact_sha256=_digest("successor-registry"),
            predecessor_publication_sha256=(
                _digest("confirmation-publication")
            ),
            deadline_monotonic=10**12,
        )

    assert runner.raw_commands == []


@pytest.mark.parametrize("target", ("config", "config.worktree"))
def test_final_registry_reverifies_pinned_config_bytes_before_any_git_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher_module

    repo = _final_registry_repo(tmp_path)
    if target == "config.worktree":
        (repo / ".git" / target).write_text(
            '[branch "safe"]\n\tremote = origin\n',
            encoding="utf-8",
        )
    manifest = _manifest()
    runner = _FinalRegistryRunner(manifest=manifest)
    monkeypatch.setattr(
        publisher_module,
        "git_runtime_dependency_closure_sha256",
        lambda root: _digest(f"closure:{root}"),
    )
    publisher = ExternalGitTagPublisher(
        repo_root=repo,
        implementation_manifest=manifest,
        final_registry_subprocess_runner=runner,
    )
    config_path = repo / ".git" / target
    config_path.write_bytes(config_path.read_bytes() + b"\n# changed\n")

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="local/worktree Git configuration changed",
    ):
        publisher.publish_or_recover_final_registry(
            attempt_id=FINAL_ATTEMPT_ID,
            terminal_status=REGISTERED_UNRUN,
            report_kind=FINAL_REGISTRY_SUCCESSOR,
            artifact_sha256=_digest("successor-registry"),
            predecessor_publication_sha256=(
                _digest("confirmation-publication")
            ),
            deadline_monotonic=10**12,
        )

    assert runner.calls == []
    assert runner.raw_commands == []


def test_final_registry_method_recovers_only_identical_existing_tag(
    tmp_path: Path,
) -> None:
    repo = _final_registry_repo(tmp_path)
    manifest = _manifest()
    prepared = prepare_external_publication(
        implementation_manifest=manifest,
        attempt_id=FINAL_ATTEMPT_ID,
        terminal_status=REGISTERED_UNRUN,
        report_kind=FINAL_REGISTRY_SUCCESSOR,
        artifact_sha256=_digest("successor-registry"),
        predecessor_publication_sha256=_digest("confirmation-publication"),
    )
    runner = _FinalRegistryRunner(
        manifest=manifest,
        existing_tag=(
            prepared.expected_tag_object_sha1,
            prepared.expected_peeled_commit,
            prepared.tag_ref,
        ),
    )
    publisher = ExternalGitTagPublisher(
        repo_root=repo,
        implementation_manifest=manifest,
        final_registry_subprocess_runner=runner,
    )

    recovered = publisher.publish_or_recover_final_registry(
        attempt_id=FINAL_ATTEMPT_ID,
        terminal_status=REGISTERED_UNRUN,
        report_kind=FINAL_REGISTRY_SUCCESSOR,
        artifact_sha256=_digest("successor-registry"),
        predecessor_publication_sha256=_digest("confirmation-publication"),
        deadline_monotonic=10**12,
    )

    assert recovered.publication == prepared.expected_publication
    assert not any(call[:1] == ("push",) for call in runner.calls)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="restricted to the exact final registry",
    ):
        publisher.publish_or_recover_final_registry(
            attempt_id=CONFIRMATION_ATTEMPT_ID,
            terminal_status=TERMINAL_PASS,
            report_kind=SCORED_PASS,
            artifact_sha256=_digest("ordinary"),
            predecessor_publication_sha256=PUBLICATION_GENESIS_SHA256,
            deadline_monotonic=10**12,
        )
