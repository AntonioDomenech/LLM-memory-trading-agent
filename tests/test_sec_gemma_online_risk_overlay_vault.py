from __future__ import annotations

from pathlib import Path
import sqlite3
import subprocess
from types import MethodType
from typing import Any

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_store import (
    EffectCapability,
    SecGemmaOnlineRiskOverlayStore,
    StoreRecordReceipt,
    _CAPABILITY_SENTINEL,
)
from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
    PRODUCTION_VAULT_RELATIVE_PATH,
    ProductionAcquisitionVault,
    SecGemmaOnlineRiskOverlayVaultError,
    VaultHandle,
    _VAULT_CONSTRUCTOR_SENTINEL,
    _load_current_production_handle_for_recovery,
    _read_quarantine_for_replay,
    _read_quarantine_for_stage_slice,
    _seal_quarantine,
    is_current_vault_handle,
    open_production_acquisition_vault,
    open_test_acquisition_vault,
)


ATTEMPT_ID = "aapl-sec-gemma-online-risk-overlay-v2-2-development-acquisition"


def _capability(
    effects: tuple[str, ...],
    *,
    attempt_id: str = ATTEMPT_ID,
) -> EffectCapability:
    return EffectCapability(
        attempt_id=attempt_id,
        allowed_effects=effects,
        receipt=StoreRecordReceipt(
            table="attempts",
            identity="attempt:1",
            attempt_id=attempt_id,
            payload_sha256="0" * 64,
            journal_sequence=1,
            journal_entry_sha256="1" * 64,
        ),
        store_instance_id="2" * 64,
        store_nonce="3" * 64,
        transition_sha256="4" * 64,
        _sentinel=_CAPABILITY_SENTINEL,
    )


class FakeStore:
    def __init__(self, capability: EffectCapability) -> None:
        self.capability = capability
        self.calls: list[str] = []

    def authorize_effect(
        self,
        capability: EffectCapability,
        effect: str,
    ) -> None:
        self.calls.append(effect)
        if capability is not self.capability:
            raise RuntimeError("foreign")
        if effect not in capability.allowed_effects:
            raise RuntimeError("forbidden")


def _sealed(
    path: Path,
) -> tuple[Any, VaultHandle, FakeStore, EffectCapability]:
    vault = open_test_acquisition_vault(path)
    capability = _capability(
        (
            "official_sec_network",
            "market_network",
            "canonical_market_value_read",
        )
    )
    store = FakeStore(capability)
    handle = _seal_quarantine(
        vault,
        store=store,
        capability=capability,
        stage="development",
        attempt_id=ATTEMPT_ID,
        bundle_sha256="a" * 64,
        manifest_sha256="b" * 64,
        private_index_sha256="c" * 64,
        predecessor_handles=(),
        quarantine={
            "private_quarantine": {
                "secret_bytes": b"future-provider-metadata",
                "stage_slice": {
                    "safe": True,
                },
            }
        },
    )
    return vault, handle, store, capability


def test_vault_is_durable_opaque_and_reopenable(tmp_path: Path) -> None:
    vault, handle, store, capability = _sealed(tmp_path)
    reopened = open_test_acquisition_vault(tmp_path)

    assert reopened.vault_id == vault.vault_id
    assert reopened.production_authority is False
    assert is_current_vault_handle(reopened, handle)
    assert handle.bundle_sha256 == "a" * 64
    assert handle.production_authority is False
    assert "future-provider-metadata" not in repr(handle)
    assert "database_path" not in vault.safe_state()
    public_names = {
        name for name in dir(vault) if not name.startswith("_")
    }
    assert not {
        "read",
        "read_raw",
        "raw_bundle",
        "quarantine",
    }.intersection(public_names)

    raw = _read_quarantine_for_replay(
        reopened,
        handle,
        store=store,
        capability=capability,
    )
    assert raw["private_quarantine"]["secret_bytes"] == (
        b"future-provider-metadata"
    )


def test_vault_effects_are_exactly_capability_gated(tmp_path: Path) -> None:
    vault, handle, _store, _capability_value = _sealed(tmp_path)
    wrong = _capability(("canonical_market_value_read",))
    wrong_store = FakeStore(wrong)
    with pytest.raises(SecGemmaOnlineRiskOverlayVaultError):
        _read_quarantine_for_replay(
            vault,
            handle,
            store=wrong_store,
            capability=wrong,
        )

    safe = _read_quarantine_for_stage_slice(
        vault,
        handle,
        store=wrong_store,
        capability=wrong,
    )
    assert safe["private_quarantine"]["stage_slice"] == {"safe": True}
    assert wrong_store.calls == ["canonical_market_value_read"]


def test_handle_is_immutable_and_stale_after_database_mutation(
    tmp_path: Path,
) -> None:
    vault, handle, _store, _capability_value = _sealed(tmp_path)
    with pytest.raises(AttributeError):
        handle._payload_sha256 = "0" * 64  # type: ignore[misc]

    database_path = object.__getattribute__(vault, "_database_path")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE quarantine_entries SET payload=? WHERE entry_id=?",
            (b"changed", object.__getattribute__(handle, "_entry_id")),
        )
        connection.commit()
    assert not is_current_vault_handle(vault, handle)


def test_production_factory_rejects_test_store(
    tmp_path: Path,
) -> None:
    capability = _capability(
        ("official_sec_network", "market_network")
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayVaultError,
        match="exact reviewed durable store",
    ):
        open_production_acquisition_vault(
            repo_root=tmp_path,
            store=FakeStore(capability),  # type: ignore[arg-type]
        )


def _uninitialized_exact_store(
    capability: EffectCapability,
) -> SecGemmaOnlineRiskOverlayStore:
    store = object.__new__(SecGemmaOnlineRiskOverlayStore)
    store._store_instance_id = "2" * 64

    def authorize_effect(
        self: SecGemmaOnlineRiskOverlayStore,
        observed: EffectCapability,
        effect: str,
    ) -> None:
        assert self is store
        if observed is not capability or effect not in observed.allowed_effects:
            raise RuntimeError("foreign")

    store.authorize_effect = MethodType(  # type: ignore[method-assign]
        authorize_effect,
        store,
    )
    return store


def test_production_vault_open_and_reopen_do_not_dirty_repository(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q"],
        cwd=repo,
        check=True,
    )
    project_ignore = (
        Path(__file__).resolve().parents[1] / ".gitignore"
    ).read_text(encoding="utf-8")
    (repo / ".gitignore").write_text(
        project_ignore,
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", ".gitignore"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Vault Test",
            "-c",
            "user.email=vault-test@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        cwd=repo,
        check=True,
    )
    capability = _capability(
        ("official_sec_network", "market_network")
    )
    store = _uninitialized_exact_store(capability)

    opened = open_production_acquisition_vault(
        repo_root=repo,
        store=store,
    )
    reopened = open_production_acquisition_vault(
        repo_root=repo,
        store=store,
    )
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert PRODUCTION_VAULT_RELATIVE_PATH.parts[0] == "data"
    assert object.__getattribute__(
        opened,
        "_database_path",
    ) == repo / PRODUCTION_VAULT_RELATIVE_PATH
    assert reopened.vault_id == opened.vault_id
    assert status == ""


def test_production_recovery_loads_fresh_metadata_handle_without_decode(
    tmp_path: Path,
) -> None:
    capability = _capability(
        ("official_sec_network", "market_network")
    )
    store = _uninitialized_exact_store(capability)
    path = tmp_path / "production-quarantine.sqlite3"
    vault = ProductionAcquisitionVault(
        database_path=path,
        production_authority=True,
        bound_store_instance_id="2" * 64,
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )
    original = _seal_quarantine(
        vault,
        store=store,
        capability=capability,
        stage="development",
        attempt_id=ATTEMPT_ID,
        bundle_sha256="a" * 64,
        manifest_sha256="b" * 64,
        private_index_sha256="c" * 64,
        predecessor_handles=(),
        quarantine={
            "private_quarantine": {
                "secret_bytes": b"must-not-be-decoded-by-recovery"
            }
        },
    )
    reopened = ProductionAcquisitionVault(
        database_path=path,
        production_authority=True,
        bound_store_instance_id="2" * 64,
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )

    recovered = _load_current_production_handle_for_recovery(
        reopened,
        store=store,
        stage="development",
        attempt_id=ATTEMPT_ID,
        predecessor_handles=(),
    )

    assert type(recovered) is VaultHandle
    assert recovered is not original
    assert recovered.bundle_sha256 == original.bundle_sha256
    assert recovered.seal_sha256 == original.seal_sha256
    assert "must-not-be-decoded" not in repr(recovered)


def test_production_recovery_rejects_changed_opaque_blob(
    tmp_path: Path,
) -> None:
    capability = _capability(
        ("official_sec_network", "market_network")
    )
    store = _uninitialized_exact_store(capability)
    path = tmp_path / "production-quarantine.sqlite3"
    vault = ProductionAcquisitionVault(
        database_path=path,
        production_authority=True,
        bound_store_instance_id="2" * 64,
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )
    handle = _seal_quarantine(
        vault,
        store=store,
        capability=capability,
        stage="development",
        attempt_id=ATTEMPT_ID,
        bundle_sha256="a" * 64,
        manifest_sha256="b" * 64,
        private_index_sha256="c" * 64,
        predecessor_handles=(),
        quarantine={"private_quarantine": {"secret": b"sealed"}},
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE quarantine_entries SET payload=? WHERE entry_id=?",
            (b"changed", object.__getattribute__(handle, "_entry_id")),
        )
        connection.commit()

    with pytest.raises(
        SecGemmaOnlineRiskOverlayVaultError,
        match="changed after sealing",
    ):
        _load_current_production_handle_for_recovery(
            vault,
            store=store,
            stage="development",
            attempt_id=ATTEMPT_ID,
            predecessor_handles=(),
        )
