from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
from typing import Any

import pytest

from agent_benchmark import sec_audit_cli as cli
from agent_benchmark.sec_audit_artifact import ArtifactVerification
from agent_benchmark.sec_audit_artifact import seal_artifact
from agent_benchmark.sec_audit_runner import CONTRACT_VERSION, SOURCE_FILES


PRIVATE_USER_AGENT = "Antonio Test valid-contact@antoniodomenech.dev"
UNRELATED_API_SECRET = "DO-NOT-PRINT-UNRELATED-API-SECRET"


def _source_repo(root: Path) -> Path:
    repo = root / "source-repo"
    repo.mkdir()
    for command in (
        ("init",),
        ("config", "user.email", "tests@local.invalid"),
        ("config", "user.name", "SEC Audit Tests"),
    ):
        subprocess.run(
            ["git", "-C", str(repo), *command],
            check=True,
            capture_output=True,
        )
    workspace = Path(__file__).resolve().parents[1]
    for relative in SOURCE_FILES:
        source = workspace / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (repo / ".gitignore").write_text("data/\ne/\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "freeze source"],
        check=True,
        capture_output=True,
    )
    return repo


def _config(path: Path, *, user_agent: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "benchmark": {"model": "unrelated"},
                "secrets": {
                    "openai_api_key": UNRELATED_API_SECRET,
                    "sec_user_agent": user_agent,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _args(repo: Path, config: Path, *, execute: bool = False) -> list[str]:
    return [
        "--execute-live" if execute else "--preflight",
        "--source-repo",
        str(repo),
        "--config",
        str(config),
        "--artifact-dir",
        "e/sec_point_in_time_audit_v1/fixed-test-artifact",
    ]


def test_preflight_is_read_only_and_prints_only_safe_evidence(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "local-config.json", user_agent=PRIVATE_USER_AGENT)
    result = cli.main(_args(repo, config))
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 0
    assert payload["status"] == "ready"
    assert payload["network_requests_performed"] == 0
    assert payload["paid_api_calls"] == 0
    assert payload["llm_or_model_calls"] == 0
    assert payload["sec_user_agent_real_contact_validated"] is True
    assert str(payload["sec_user_agent_sha256"]).startswith("sha256:")
    assert str(payload["artifact_reference"]).startswith("sha256:")
    assert "fixed-test-artifact" not in output
    assert PRIVATE_USER_AGENT not in output
    assert UNRELATED_API_SECRET not in output
    assert not (repo / "data").exists()
    assert not (repo / "e").exists()


def test_blank_user_agent_fails_closed_without_printing_other_secrets(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "blank-config.json", user_agent="")
    result = cli.main(_args(repo, config))
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 2
    assert payload == {
        "llm_or_model_calls": 0,
        "network_requests_performed": 0,
        "paid_api_calls": 0,
        "reason_code": "sec_user_agent_missing",
        "status": "blocked",
    }
    assert UNRELATED_API_SECRET not in output


def test_artifact_cannot_escape_frozen_repository_root(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "config.json", user_agent=PRIVATE_USER_AGENT)
    args = _args(repo, config)
    args[-1] = str(tmp_path / "outside-artifact")
    assert cli.main(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason_code"] == "artifact_path_outside_frozen_root"


def test_execute_live_passes_private_contact_only_to_internal_live_runner(
    tmp_path: Path,
    capsys: Any,
    monkeypatch: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "execute-config.json", user_agent=PRIVATE_USER_AGENT)
    captured: dict[str, Any] = {}

    def fake_live(**kwargs: Any) -> ArtifactVerification:
        captured.update(kwargs)
        artifact = Path(kwargs["artifact_dir"])
        source_commit = subprocess.run(
            ["git", "-C", str(kwargs["source_repo"]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        report = {
            "contract_version": CONTRACT_VERSION,
            "source_commit": source_commit,
            "overall_pass": True,
            "transport_trust": {
                "trusted_production_transport": True,
                "fresh_official_retrieval_gate_passed": True,
            },
            "behavior_evidence": {
                "paid_api_calls": 0,
                "llm_or_model_calls": 0,
            },
        }
        verification = seal_artifact(
            artifact,
            {
                "audit_report.json": (
                    json.dumps(report, sort_keys=True) + "\n"
                ).encode("utf-8")
            },
            source_commit=source_commit,
            audit_contract_version=CONTRACT_VERSION,
        )
        captured["verification"] = verification
        captured["report"] = report
        return verification

    def fake_semantic(*args: Any, **kwargs: Any):
        del args, kwargs
        return captured["verification"], captured["report"]

    monkeypatch.setattr(cli, "run_live_sec_audit", fake_live)
    monkeypatch.setattr(
        cli, "verify_production_sec_audit_artifact", fake_semantic
    )
    result = cli.main(_args(repo, config, execute=True))
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 0
    assert captured["user_agent"] == PRIVATE_USER_AGENT
    assert captured["source_repo"] == repo
    assert payload["status"] == "passed"
    assert payload["overall_pass"] is True
    assert PRIVATE_USER_AGENT not in output
    assert UNRELATED_API_SECRET not in output


def test_invalid_arguments_use_fixed_json_error(capsys: Any) -> None:
    result = cli.main(["--preflight", "--execute-live"])
    payload = json.loads(capsys.readouterr().out)
    assert result == 2
    assert payload["status"] == "blocked"
    assert payload["reason_code"] == "invalid_arguments"
    assert payload["network_requests_performed"] == 0


def test_invalid_arguments_never_echo_raw_secret(
    capsys: Any,
) -> None:
    raw_secret = "raw-contact-and-api-key-DO-NOT-ECHO"

    result = cli.main(["--sec-user-agent", raw_secret, raw_secret])
    captured = capsys.readouterr()

    assert result == 2
    assert json.loads(captured.out)["reason_code"] == "invalid_arguments"
    assert raw_secret not in captured.out
    assert raw_secret not in captured.err


def test_prepare_state_redacts_private_contact_from_repr(
    tmp_path: Path,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "repr-config.json", user_agent=PRIVATE_USER_AGENT)
    private_artifact_name = "sk-proj-private-artifact-token-123456"

    prepared = cli._prepare(
        source_repo=repo,
        config=config,
        cache_dir="data/cache/sec_point_in_time_audit_v1",
        artifact_dir=f"e/sec_point_in_time_audit_v1/{private_artifact_name}",
    )
    diagnostics = repr(prepared) + repr(prepared.reveal_user_agent_for_transport)

    assert PRIVATE_USER_AGENT not in diagnostics
    assert UNRELATED_API_SECRET not in diagnostics
    assert private_artifact_name not in diagnostics
    assert "redacted" in diagnostics
    with pytest.raises(TypeError):
        asdict(prepared)
    with pytest.raises(TypeError):
        pickle.dumps(prepared)


def test_malformed_config_exception_retains_no_secret_content(
    tmp_path: Path,
) -> None:
    raw_secret = "PRIVATE-CONTACT-RETAINED-IN-MALFORMED-JSON"
    config = tmp_path / "malformed-config.json"
    config.write_text(
        '{"secrets":{"sec_user_agent":"' + raw_secret + '",',
        encoding="utf-8",
    )

    with pytest.raises(cli.SecAuditCliError) as captured:
        cli._load_sec_user_agent(config)

    error = captured.value
    assert error.code == "sec_config_unreadable"
    assert error.__cause__ is None
    assert error.__context__ is None
    diagnostics = [repr(error)]
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_globals.get("__name__") == cli.__name__:
            diagnostics.extend(
                repr(value) for value in traceback.tb_frame.f_locals.values()
            )
        traceback = traceback.tb_next
    assert raw_secret not in "\n".join(diagnostics)


def test_invalid_contact_exception_retains_no_secret_content(
    tmp_path: Path,
) -> None:
    raw_secret = "PRIVATE-INVALID-CONTACT-WITHOUT-AN-EMAIL"
    config = _config(tmp_path / "invalid-contact.json", user_agent=raw_secret)

    with pytest.raises(cli.SecAuditCliError) as captured:
        cli._load_sec_user_agent(config)

    error = captured.value
    assert error.code == "sec_user_agent_invalid"
    assert error.__cause__ is None
    assert error.__context__ is None
    diagnostics: list[str] = [repr(error)]
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_globals.get("__name__") == cli.__name__:
            diagnostics.extend(
                repr(value) for value in traceback.tb_frame.f_locals.values()
            )
        traceback = traceback.tb_next
    assert raw_secret not in "\n".join(diagnostics)


def test_caller_artifact_name_is_never_echoed(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "artifact-config.json", user_agent=PRIVATE_USER_AGENT)
    raw_secret = "api-key-shaped-secret@private.example"
    args = _args(repo, config)
    args[-1] = f"e/sec_point_in_time_audit_v1/{raw_secret}"

    assert cli.main(args) == 2
    captured = capsys.readouterr()

    assert json.loads(captured.out)["reason_code"] == "artifact_path_invalid"
    assert raw_secret not in captured.out
    assert raw_secret not in captured.err


def test_preflight_fails_when_live_dependency_is_missing(
    tmp_path: Path,
    capsys: Any,
    monkeypatch: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "runtime-config.json", user_agent=PRIVATE_USER_AGENT)
    original = cli.importlib_metadata.version

    def missing(name: str) -> str:
        if name == "requests":
            raise cli.importlib_metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(cli.importlib_metadata, "version", missing)
    assert cli.main(_args(repo, config)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason_code"] == "live_runtime_dependency_missing"
    assert payload["network_requests_performed"] == 0


def test_config_symlink_is_rejected_before_read(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = _source_repo(tmp_path)
    target = _config(tmp_path / "real-config.json", user_agent=PRIVATE_USER_AGENT)
    link = tmp_path / "linked-config.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("file symlinks are unavailable on this Windows host")
    assert cli.main(_args(repo, link)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason_code"] == "sec_config_missing_or_unsafe"


def test_cache_root_symlink_cannot_escape_repository(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "config.json", user_agent=PRIVATE_USER_AGENT)
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    cache_parent = repo / "data" / "cache"
    cache_parent.mkdir(parents=True)
    link = cache_parent / "sec_point_in_time_audit_v1"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this Windows host")
    assert cli.main(_args(repo, config)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason_code"] == "cache_path_outside_frozen_root"


@pytest.mark.skipif(
    not (hasattr(Path("."), "is_junction") or hasattr(os.path, "isjunction")),
    reason="Windows junction detection unavailable",
)
def test_cache_root_junction_cannot_escape_repository(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "junction-config.json", user_agent=PRIVATE_USER_AGENT)
    outside = tmp_path / "outside-junction-cache"
    outside.mkdir()
    cache_parent = repo / "data" / "cache"
    cache_parent.mkdir(parents=True)
    junction = cache_parent / "sec_point_in_time_audit_v1"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        check=False,
        capture_output=True,
        text=True,
    )
    path_junction = getattr(junction, "is_junction", None)
    os_junction = getattr(os.path, "isjunction", None)
    detected = bool(
        (path_junction is not None and path_junction())
        or (os_junction is not None and os_junction(junction))
    )
    if created.returncode != 0 or not detected:
        pytest.skip("directory junctions are unavailable on this Windows host")
    try:
        assert cli.main(_args(repo, config)) == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["reason_code"] == "cache_path_outside_frozen_root"
    finally:
        junction.rmdir()


def test_mocked_cache_reparse_point_is_always_rejected(
    tmp_path: Path,
    capsys: Any,
    monkeypatch: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "mocked-link-config.json", user_agent=PRIVATE_USER_AGENT)
    original = cli._is_link_or_junction

    def fake_reparse(path: Path) -> bool:
        if path.name == "sec_point_in_time_audit_v1" and "data" in path.parts:
            return True
        return original(path)

    monkeypatch.setattr(cli, "_is_link_or_junction", fake_reparse)
    assert cli.main(_args(repo, config)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason_code"] == "cache_path_outside_frozen_root"


def test_live_exception_cannot_print_private_contact(
    tmp_path: Path,
    capsys: Any,
    monkeypatch: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "failure-config.json", user_agent=PRIVATE_USER_AGENT)

    def fail_live(**kwargs: Any) -> ArtifactVerification:
        raise RuntimeError(f"injected failure {kwargs['user_agent']}")

    monkeypatch.setattr(cli, "run_live_sec_audit", fail_live)
    result = cli.main(_args(repo, config, execute=True))
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 2
    assert payload["reason_code"] == "live_execution_failed_safely"
    assert payload["network_requests_performed"] == (
        "unknown_but_bounded_by_audit_transport"
    )
    assert PRIVATE_USER_AGENT not in output
    assert UNRELATED_API_SECRET not in output


def test_completed_failed_audit_returns_exit_three(
    tmp_path: Path,
    capsys: Any,
    monkeypatch: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "failed-audit-config.json", user_agent=PRIVATE_USER_AGENT)

    def failed_live(**kwargs: Any) -> ArtifactVerification:
        commit = subprocess.run(
            ["git", "-C", str(kwargs["source_repo"]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        report = {
            "contract_version": CONTRACT_VERSION,
            "source_commit": commit,
            "overall_pass": False,
        }
        return seal_artifact(
            kwargs["artifact_dir"],
            {
                "audit_report.json": (
                    json.dumps(report, sort_keys=True) + "\n"
                ).encode("utf-8")
            },
            source_commit=commit,
            audit_contract_version=CONTRACT_VERSION,
        )

    monkeypatch.setattr(cli, "run_live_sec_audit", failed_live)
    result = cli.main(_args(repo, config, execute=True))
    payload = json.loads(capsys.readouterr().out)
    assert result == 3
    assert payload["status"] == "failed"
    assert payload["overall_pass"] is False


def test_one_file_self_declared_pass_is_rejected_semantically(
    tmp_path: Path,
    capsys: Any,
    monkeypatch: Any,
) -> None:
    repo = _source_repo(tmp_path)
    config = _config(tmp_path / "incomplete-config.json", user_agent=PRIVATE_USER_AGENT)

    def incomplete_live(**kwargs: Any) -> ArtifactVerification:
        commit = subprocess.run(
            ["git", "-C", str(kwargs["source_repo"]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        report = {
            "contract_version": CONTRACT_VERSION,
            "source_commit": commit,
            "overall_pass": True,
            "transport_trust": {
                "trusted_production_transport": True,
                "fresh_official_retrieval_gate_passed": True,
            },
            "behavior_evidence": {
                "paid_api_calls": 0,
                "llm_or_model_calls": 0,
            },
        }
        return seal_artifact(
            kwargs["artifact_dir"],
            {
                "audit_report.json": (
                    json.dumps(report, sort_keys=True) + "\n"
                ).encode("utf-8")
            },
            source_commit=commit,
            audit_contract_version=CONTRACT_VERSION,
        )

    monkeypatch.setattr(cli, "run_live_sec_audit", incomplete_live)
    assert cli.main(_args(repo, config, execute=True)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["reason_code"] == "live_execution_failed_safely"
