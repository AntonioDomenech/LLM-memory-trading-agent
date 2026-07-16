"""Durable opaque quarantine vault for the SEC/Gemma v2.1 overlay.

The vault deliberately has no public raw-read method.  Acquisition code seals
one exact recursive value and receives an opaque :class:`VaultHandle`; detached
replay uses the module-private capability-gated reader.  Public callers may
check that a handle is still current, but cannot obtain SEC catalogue bytes,
Yahoo response bytes, future metadata, or the transport-only final row through
this module's supported API.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    canonical_json_bytes,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_store import (
    EffectCapability,
    STATE_RELATIVE_DIRECTORY,
    SecGemmaOnlineRiskOverlayStore,
)


VAULT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-vault-v1"
)
VAULT_HANDLE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-vault-handle-v1"
)
PRODUCTION_VAULT_RELATIVE_PATH: Final[Path] = (
    STATE_RELATIVE_DIRECTORY / "quarantine.sqlite3"
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_STAGES: Final[tuple[str, ...]] = ("development", "confirmation", "final")
_VAULT_CONSTRUCTOR_SENTINEL = object()
_HANDLE_SENTINEL = object()


class SecGemmaOnlineRiskOverlayVaultError(RuntimeError):
    """The durable quarantine is absent, stale, corrupt, or unauthorized."""


def _sha256_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault payload must be exact bytes"
        )
    return hashlib.sha256(value).hexdigest()


def _strict_sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayVaultError(
            f"{location} must be a lowercase SHA-256"
        )
    return value


def _strict_text(value: Any, location: str, *, maximum: int = 256) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or any(ord(char) < 32 or ord(char) > 126 for char in value)
    ):
        raise SecGemmaOnlineRiskOverlayVaultError(
            f"{location} must be bounded printable text"
        )
    return value


def _encode_opaque(value: Any, location: str, *, depth: int = 0) -> Any:
    """Encode exact Python scalar/container types without pickle execution."""

    if depth > 96:
        raise SecGemmaOnlineRiskOverlayVaultError(
            f"{location} exceeds the vault nesting limit"
        )
    if value is None:
        return {"t": "null"}
    if type(value) is bool:
        return {"t": "bool", "v": value}
    if type(value) is int:
        return {"t": "int", "v": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise SecGemmaOnlineRiskOverlayVaultError(
                f"{location} contains a non-finite float"
            )
        return {"t": "float", "v": value.hex()}
    if type(value) is str:
        return {"t": "str", "v": value}
    if type(value) is bytes:
        return {"t": "bytes", "v": value.hex()}
    if type(value) is list:
        return {
            "t": "list",
            "v": [
                _encode_opaque(
                    item,
                    f"{location}[{index}]",
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ],
        }
    if type(value) is tuple:
        return {
            "t": "tuple",
            "v": [
                _encode_opaque(
                    item,
                    f"{location}[{index}]",
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ],
        }
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise SecGemmaOnlineRiskOverlayVaultError(
                f"{location} keys must be exact strings"
            )
        return {
            "t": "dict",
            "v": [
                [
                    key,
                    _encode_opaque(
                        value[key],
                        f"{location}.{key}",
                        depth=depth + 1,
                    ),
                ]
                for key in sorted(value)
            ],
        }
    raise SecGemmaOnlineRiskOverlayVaultError(
        f"{location} contains an unsupported runtime object"
    )


def _decode_opaque(value: Any, location: str, *, depth: int = 0) -> Any:
    if depth > 96 or type(value) is not dict or set(value) not in (
        {"t"},
        {"t", "v"},
    ):
        raise SecGemmaOnlineRiskOverlayVaultError(
            f"{location} has a corrupt opaque encoding"
        )
    tag = value.get("t")
    raw = value.get("v")
    if tag == "null" and set(value) == {"t"}:
        return None
    if tag == "bool" and type(raw) is bool:
        return raw
    if tag == "int" and type(raw) is str and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", raw):
        return int(raw)
    if tag == "float" and type(raw) is str:
        try:
            result = float.fromhex(raw)
        except ValueError:
            result = math.nan
        if math.isfinite(result) and result.hex() == raw:
            return result
    if tag == "str" and type(raw) is str:
        return raw
    if tag == "bytes" and type(raw) is str:
        try:
            result = bytes.fromhex(raw)
        except ValueError:
            result = b""
        if result.hex() == raw:
            return result
    if tag in {"list", "tuple"} and type(raw) is list:
        result = [
            _decode_opaque(
                item,
                f"{location}[{index}]",
                depth=depth + 1,
            )
            for index, item in enumerate(raw)
        ]
        return result if tag == "list" else tuple(result)
    if tag == "dict" and type(raw) is list:
        result: dict[str, Any] = {}
        prior: str | None = None
        for index, pair in enumerate(raw):
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or pair[0] in result
                or (prior is not None and pair[0] <= prior)
            ):
                raise SecGemmaOnlineRiskOverlayVaultError(
                    f"{location} has a corrupt opaque dictionary"
                )
            key = pair[0]
            prior = key
            result[key] = _decode_opaque(
                pair[1],
                f"{location}.{key}",
                depth=depth + 1,
            )
        return result
    raise SecGemmaOnlineRiskOverlayVaultError(
        f"{location} has a corrupt opaque value"
    )


def _opaque_bytes(value: Any) -> bytes:
    return canonical_json_bytes(_encode_opaque(value, "vault quarantine"))


def _opaque_value(payload: bytes) -> Any:
    if type(payload) is not bytes or not payload:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault entry payload is absent"
        )
    try:
        encoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault entry payload is corrupt"
        ) from None
    return _decode_opaque(encoded, "vault quarantine")


class VaultHandle:
    """Opaque immutable identity for one sealed durable quarantine entry."""

    __slots__ = (
        "_attempt_id",
        "_bundle_sha256",
        "_entry_id",
        "_generation",
        "_locked",
        "_manifest_sha256",
        "_payload_sha256",
        "_private_index_sha256",
        "_production_authority",
        "_seal_sha256",
        "_sentinel",
        "_stage",
        "_vault_id",
    )

    def __init__(
        self,
        *,
        vault_id: str,
        entry_id: str,
        stage: str,
        attempt_id: str,
        generation: int,
        bundle_sha256: str,
        manifest_sha256: str,
        private_index_sha256: str,
        payload_sha256: str,
        production_authority: bool,
        seal_sha256: str,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _HANDLE_SENTINEL:
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Vault handles can only be issued by the durable vault"
            )
        object.__setattr__(self, "_locked", False)
        object.__setattr__(self, "_vault_id", _strict_sha256(vault_id, "vault id"))
        object.__setattr__(self, "_entry_id", _strict_sha256(entry_id, "entry id"))
        if stage not in _STAGES:
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Vault handle stage is invalid"
            )
        object.__setattr__(self, "_stage", stage)
        object.__setattr__(
            self,
            "_attempt_id",
            _strict_text(attempt_id, "vault attempt id"),
        )
        if type(generation) is not int or generation < 1:
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Vault handle generation is invalid"
            )
        object.__setattr__(self, "_generation", generation)
        object.__setattr__(
            self,
            "_bundle_sha256",
            _strict_sha256(bundle_sha256, "bundle hash"),
        )
        object.__setattr__(
            self,
            "_manifest_sha256",
            _strict_sha256(manifest_sha256, "manifest hash"),
        )
        object.__setattr__(
            self,
            "_private_index_sha256",
            _strict_sha256(private_index_sha256, "private-index hash"),
        )
        object.__setattr__(
            self,
            "_payload_sha256",
            _strict_sha256(payload_sha256, "vault payload hash"),
        )
        if type(production_authority) is not bool:
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Vault authority flag is invalid"
            )
        object.__setattr__(
            self, "_production_authority", production_authority
        )
        object.__setattr__(
            self,
            "_seal_sha256",
            _strict_sha256(seal_sha256, "vault seal hash"),
        )
        object.__setattr__(self, "_sentinel", _sentinel)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("VaultHandle is immutable")
        object.__setattr__(self, name, value)

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def bundle_sha256(self) -> str:
        return self._bundle_sha256

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def private_index_sha256(self) -> str:
        return self._private_index_sha256

    @property
    def seal_sha256(self) -> str:
        return self._seal_sha256

    @property
    def production_authority(self) -> bool:
        return self._production_authority

    def __repr__(self) -> str:
        return (
            "VaultHandle("
            f"stage={self._stage!r}, "
            f"bundle_sha256={self._bundle_sha256!r}, "
            f"production_authority={self._production_authority!r})"
        )


class _AcquisitionVault:
    __slots__ = (
        "_bound_store_instance_id",
        "_database_path",
        "_production_authority",
        "_vault_id",
    )

    def __init__(
        self,
        *,
        database_path: Path,
        production_authority: bool,
        bound_store_instance_id: str | None,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _VAULT_CONSTRUCTOR_SENTINEL:
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Acquisition vaults must be opened by a reviewed factory"
            )
        path = Path(database_path).expanduser().resolve()
        if path.exists() and not path.is_file():
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Vault database path is not a regular file"
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Vault directory could not be created"
            ) from None
        self._database_path = path
        self._production_authority = production_authority
        self._bound_store_instance_id = bound_store_instance_id
        self._vault_id = self._initialize()

    @property
    def production_authority(self) -> bool:
        return self._production_authority

    @property
    def vault_id(self) -> str:
        return self._vault_id

    def safe_state(self) -> dict[str, Any]:
        return {
            "schema_version": VAULT_SCHEMA_VERSION,
            "vault_id": self._vault_id,
            "contract_sha256": CONTRACT_SHA256,
            "production_authority": self._production_authority,
            "raw_read_api_exposed": False,
            "durable": True,
        }

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=30.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA journal_mode=DELETE")
            return connection
        except sqlite3.Error:
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Vault database could not be opened"
            ) from None

    def _initialize(self) -> str:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vault_meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS quarantine_entries(
                    entry_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    bundle_sha256 TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    private_index_sha256 TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    predecessor_entry_ids_json BLOB NOT NULL,
                    production_authority INTEGER NOT NULL,
                    metadata_sha256 TEXT NOT NULL,
                    seal_sha256 TEXT NOT NULL,
                    UNIQUE(stage, attempt_id)
                ) WITHOUT ROWID
                """
            )
            rows = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key,value FROM vault_meta"
                ).fetchall()
            }
            expected_keys = {
                "schema_version",
                "contract_sha256",
                "vault_id",
                "production_authority",
                "bound_store_instance_id",
            }
            if not rows:
                vault_id = secrets.token_hex(32)
                bound = self._bound_store_instance_id or ""
                entries = {
                    "schema_version": VAULT_SCHEMA_VERSION,
                    "contract_sha256": CONTRACT_SHA256,
                    "vault_id": vault_id,
                    "production_authority": (
                        "1" if self._production_authority else "0"
                    ),
                    "bound_store_instance_id": bound,
                }
                connection.executemany(
                    "INSERT INTO vault_meta(key,value) VALUES(?,?)",
                    list(entries.items()),
                )
            else:
                if set(rows) != expected_keys:
                    raise SecGemmaOnlineRiskOverlayVaultError(
                        "Vault metadata schema changed"
                    )
                vault_id = _strict_sha256(rows["vault_id"], "vault id")
                expected_bound = self._bound_store_instance_id or ""
                if (
                    rows["schema_version"] != VAULT_SCHEMA_VERSION
                    or rows["contract_sha256"] != CONTRACT_SHA256
                    or rows["production_authority"]
                    != ("1" if self._production_authority else "0")
                    or rows["bound_store_instance_id"] != expected_bound
                ):
                    raise SecGemmaOnlineRiskOverlayVaultError(
                        "Vault metadata does not match this contract authority"
                    )
            connection.execute("COMMIT")
            try:
                os.chmod(self._database_path, 0o600)
            except OSError:
                pass
            return vault_id
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()


class ProductionAcquisitionVault(_AcquisitionVault):
    """Exact production vault type returned only by its fixed-path factory."""

    __slots__ = ()


class TestAcquisitionVault(_AcquisitionVault):
    """Explicit non-production vault for deterministic local tests only."""

    __slots__ = ()
    __test__ = False


def open_production_acquisition_vault(
    *,
    repo_root: Path,
    store: SecGemmaOnlineRiskOverlayStore,
) -> ProductionAcquisitionVault:
    """Open the contract-fixed production vault bound to one exact store."""

    if type(store) is not SecGemmaOnlineRiskOverlayStore:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production vault requires the exact reviewed durable store"
        )
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir() or not (root / ".git").exists():
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production vault repository root is invalid"
        )
    store_id = getattr(store, "_store_instance_id", None)
    _strict_sha256(store_id, "production store id")
    return ProductionAcquisitionVault(
        database_path=root / PRODUCTION_VAULT_RELATIVE_PATH,
        production_authority=True,
        bound_store_instance_id=store_id,
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )


def open_test_acquisition_vault(path: Path) -> TestAcquisitionVault:
    """Open a visibly non-production vault under a caller-owned test path."""

    root = Path(path).expanduser().resolve()
    database_path = (
        root if root.suffix.lower() in {".db", ".sqlite", ".sqlite3"} else root / "quarantine.sqlite3"
    )
    return TestAcquisitionVault(
        database_path=database_path,
        production_authority=False,
        bound_store_instance_id=None,
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )


def _vault_type(value: Any) -> _AcquisitionVault:
    if type(value) not in {ProductionAcquisitionVault, TestAcquisitionVault}:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault must be an exact reviewed production or test instance"
        )
    return value


def _authorize_vault_effect(
    vault: _AcquisitionVault,
    *,
    store: Any,
    capability: EffectCapability,
    expected_attempt_id: str | None,
    required_effects: tuple[str, ...],
) -> None:
    if (
        type(capability) is not EffectCapability
        or not callable(getattr(store, "authorize_effect", None))
        or (
            expected_attempt_id is not None
            and capability.attempt_id != expected_attempt_id
        )
        or type(required_effects) is not tuple
        or not required_effects
        or any(
            type(effect) is not str
            or effect not in capability.allowed_effects
            for effect in required_effects
        )
    ):
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault capability is absent, wrong-stage, or unauthorized"
        )
    if vault.production_authority:
        if (
            type(vault) is not ProductionAcquisitionVault
            or type(store) is not SecGemmaOnlineRiskOverlayStore
            or getattr(store, "_store_instance_id", None)
            != vault._bound_store_instance_id
        ):
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Production vault authority is foreign or test-only"
            )
    try:
        for effect in required_effects:
            store.authorize_effect(capability, effect)
    except Exception:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault capability is stale, foreign, or not consumed"
        ) from None


def _entry_metadata(
    *,
    vault_id: str,
    entry_id: str,
    stage: str,
    attempt_id: str,
    generation: int,
    bundle_sha256: str,
    manifest_sha256: str,
    private_index_sha256: str,
    payload_sha256: str,
    predecessor_entry_ids: list[str],
    production_authority: bool,
) -> dict[str, Any]:
    return {
        "schema_version": VAULT_SCHEMA_VERSION,
        "vault_id": vault_id,
        "entry_id": entry_id,
        "stage": stage,
        "attempt_id": attempt_id,
        "generation": generation,
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": manifest_sha256,
        "private_index_sha256": private_index_sha256,
        "payload_sha256": payload_sha256,
        "predecessor_entry_ids": predecessor_entry_ids,
        "production_authority": production_authority,
    }


def _handle_from_metadata(metadata: Mapping[str, Any]) -> VaultHandle:
    body = dict(metadata)
    metadata_hash = canonical_sha256(body)
    seal_hash = canonical_sha256(
        {
            "metadata_sha256": metadata_hash,
            "payload_sha256": body["payload_sha256"],
        }
    )
    return VaultHandle(
        vault_id=body["vault_id"],
        entry_id=body["entry_id"],
        stage=body["stage"],
        attempt_id=body["attempt_id"],
        generation=body["generation"],
        bundle_sha256=body["bundle_sha256"],
        manifest_sha256=body["manifest_sha256"],
        private_index_sha256=body["private_index_sha256"],
        payload_sha256=body["payload_sha256"],
        production_authority=body["production_authority"],
        seal_sha256=seal_hash,
        _sentinel=_HANDLE_SENTINEL,
    )


def _seal_quarantine(
    vault: _AcquisitionVault,
    *,
    store: Any,
    capability: EffectCapability,
    stage: str,
    attempt_id: str,
    bundle_sha256: str,
    manifest_sha256: str,
    private_index_sha256: str,
    predecessor_handles: tuple[VaultHandle, ...],
    quarantine: dict[str, Any],
) -> VaultHandle:
    """Seal exact raw material; intentionally private to acquisition code."""

    fixed_vault = _vault_type(vault)
    _authorize_vault_effect(
        fixed_vault,
        store=store,
        capability=capability,
        expected_attempt_id=attempt_id,
        required_effects=("official_sec_network", "market_network"),
    )
    if stage not in _STAGES or type(quarantine) is not dict:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault quarantine stage or payload is invalid"
        )
    bundle_hash = _strict_sha256(bundle_sha256, "bundle hash")
    manifest_hash = _strict_sha256(manifest_sha256, "manifest hash")
    index_hash = _strict_sha256(
        private_index_sha256, "private-index hash"
    )
    if type(predecessor_handles) is not tuple:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault predecessor handles must be one exact tuple"
        )
    predecessor_entry_ids: list[str] = []
    for expected_stage, handle in zip(
        _STAGES[: _STAGES.index(stage)],
        predecessor_handles,
        strict=False,
    ):
        if (
            type(handle) is not VaultHandle
            or handle._sentinel is not _HANDLE_SENTINEL
            or handle._vault_id != fixed_vault.vault_id
            or handle.stage != expected_stage
        ):
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Vault predecessor chain is foreign or reordered"
            )
        _assert_vault_handle_current(fixed_vault, handle)
        predecessor_entry_ids.append(handle._entry_id)
    if len(predecessor_entry_ids) != _STAGES.index(stage):
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault predecessor chain is incomplete or excessive"
        )

    payload = _opaque_bytes(quarantine)
    payload_hash = _sha256_bytes(payload)
    entry_body = {
        "vault_id": fixed_vault.vault_id,
        "stage": stage,
        "attempt_id": attempt_id,
        "bundle_sha256": bundle_hash,
        "manifest_sha256": manifest_hash,
        "private_index_sha256": index_hash,
        "payload_sha256": payload_hash,
        "predecessor_entry_ids": predecessor_entry_ids,
        "production_authority": fixed_vault.production_authority,
    }
    entry_id = canonical_sha256(entry_body)
    generation = 1
    metadata = _entry_metadata(
        vault_id=fixed_vault.vault_id,
        entry_id=entry_id,
        stage=stage,
        attempt_id=attempt_id,
        generation=generation,
        bundle_sha256=bundle_hash,
        manifest_sha256=manifest_hash,
        private_index_sha256=index_hash,
        payload_sha256=payload_hash,
        predecessor_entry_ids=predecessor_entry_ids,
        production_authority=fixed_vault.production_authority,
    )
    metadata_hash = canonical_sha256(metadata)
    seal_hash = canonical_sha256(
        {
            "metadata_sha256": metadata_hash,
            "payload_sha256": payload_hash,
        }
    )
    predecessor_json = canonical_json_bytes(predecessor_entry_ids)
    connection = fixed_vault._connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT * FROM quarantine_entries
            WHERE stage=? AND attempt_id=?
            """,
            (stage, attempt_id),
        ).fetchone()
        row_values = (
            entry_id,
            stage,
            attempt_id,
            generation,
            bundle_hash,
            manifest_hash,
            index_hash,
            payload_hash,
            payload,
            predecessor_json,
            1 if fixed_vault.production_authority else 0,
            metadata_hash,
            seal_hash,
        )
        if existing is None:
            connection.execute(
                """
                INSERT INTO quarantine_entries(
                    entry_id,stage,attempt_id,generation,bundle_sha256,
                    manifest_sha256,private_index_sha256,payload_sha256,
                    payload,predecessor_entry_ids_json,production_authority,
                    metadata_sha256,seal_sha256
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row_values,
            )
        else:
            observed = tuple(existing[key] for key in existing.keys())
            if observed != row_values:
                raise SecGemmaOnlineRiskOverlayVaultError(
                    "Vault stage already binds different immutable bytes"
                )
        connection.execute("COMMIT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()
    handle = _handle_from_metadata(metadata)
    _assert_vault_handle_current(fixed_vault, handle)
    return handle


def _verified_entry(
    vault: _AcquisitionVault,
    handle: VaultHandle,
) -> tuple[sqlite3.Row, dict[str, Any]]:
    fixed_vault = _vault_type(vault)
    if (
        type(handle) is not VaultHandle
        or handle._sentinel is not _HANDLE_SENTINEL
        or handle._vault_id != fixed_vault.vault_id
        or handle.production_authority != fixed_vault.production_authority
    ):
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault handle is forged, foreign, or test-only"
        )
    connection = fixed_vault._connect()
    try:
        row = connection.execute(
            "SELECT * FROM quarantine_entries WHERE entry_id=?",
            (handle._entry_id,),
        ).fetchone()
    except sqlite3.Error:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault entry could not be read"
        ) from None
    finally:
        connection.close()
    if row is None:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault entry is absent or stale"
        )
    try:
        predecessor_bytes = bytes(row["predecessor_entry_ids_json"])
        predecessor_ids = json.loads(
            predecessor_bytes.decode("utf-8", errors="strict")
        )
        if (
            type(predecessor_ids) is not list
            or not all(
                type(item) is str and _SHA256_RE.fullmatch(item)
                for item in predecessor_ids
            )
            or canonical_json_bytes(predecessor_ids) != predecessor_bytes
        ):
            raise ValueError
        production = row["production_authority"]
        metadata = _entry_metadata(
            vault_id=fixed_vault.vault_id,
            entry_id=str(row["entry_id"]),
            stage=str(row["stage"]),
            attempt_id=str(row["attempt_id"]),
            generation=int(row["generation"]),
            bundle_sha256=str(row["bundle_sha256"]),
            manifest_sha256=str(row["manifest_sha256"]),
            private_index_sha256=str(row["private_index_sha256"]),
            payload_sha256=str(row["payload_sha256"]),
            predecessor_entry_ids=predecessor_ids,
            production_authority=(
                True if production == 1 else False if production == 0 else None
            ),
        )
        payload = bytes(row["payload"])
        metadata_hash = canonical_sha256(metadata)
        seal_hash = canonical_sha256(
            {
                "metadata_sha256": metadata_hash,
                "payload_sha256": _sha256_bytes(payload),
            }
        )
        exact = (
            metadata["stage"] in _STAGES
            and metadata["generation"] >= 1
            and metadata["production_authority"]
            is fixed_vault.production_authority
            and _strict_sha256(
                metadata["entry_id"], "vault entry id"
            )
            == handle._entry_id
            and _strict_sha256(
                metadata["bundle_sha256"], "vault bundle hash"
            )
            == handle.bundle_sha256
            and _strict_sha256(
                metadata["manifest_sha256"], "vault manifest hash"
            )
            == handle.manifest_sha256
            and _strict_sha256(
                metadata["private_index_sha256"],
                "vault private-index hash",
            )
            == handle.private_index_sha256
            and _strict_sha256(
                metadata["payload_sha256"], "vault payload hash"
            )
            == handle._payload_sha256
            and metadata["generation"] == handle._generation
            and metadata["attempt_id"] == handle.attempt_id
            and metadata["stage"] == handle.stage
            and metadata["payload_sha256"] == _sha256_bytes(payload)
            and row["metadata_sha256"] == metadata_hash
            and row["seal_sha256"] == seal_hash
            and handle.seal_sha256 == seal_hash
        )
    except Exception:
        exact = False
    if not exact:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault entry changed after sealing"
        )
    return row, metadata


def _production_recovery_sealed_stages(
    vault: ProductionAcquisitionVault,
    *,
    store: SecGemmaOnlineRiskOverlayStore,
) -> tuple[str, ...]:
    """Return only an exact contiguous production-vault stage prefix."""

    if (
        type(vault) is not ProductionAcquisitionVault
        or vault.production_authority is not True
        or type(store) is not SecGemmaOnlineRiskOverlayStore
        or getattr(store, "_store_instance_id", None)
        != vault._bound_store_instance_id
    ):
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production recovery requires the exact reopened store"
        )
    connection = vault._connect()
    try:
        rows = connection.execute(
            """
            SELECT stage, generation, production_authority
            FROM quarantine_entries
            ORDER BY CASE stage
                WHEN 'development' THEN 1
                WHEN 'confirmation' THEN 2
                WHEN 'final' THEN 3
                ELSE 4
            END
            """
        ).fetchall()
    except sqlite3.Error:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production recovery metadata could not be read"
        ) from None
    finally:
        connection.close()
    stages = tuple(str(row["stage"]) for row in rows)
    if (
        len(rows) > len(_STAGES)
        or stages != _STAGES[: len(stages)]
        or any(
            row["generation"] != 1
            or row["production_authority"] != 1
            for row in rows
        )
    ):
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production recovery vault stages are gapped or reordered"
        )
    return stages


def _load_current_production_handle_for_recovery(
    vault: ProductionAcquisitionVault,
    *,
    store: SecGemmaOnlineRiskOverlayStore,
    stage: str,
    attempt_id: str,
    predecessor_handles: tuple[VaultHandle, ...],
) -> VaultHandle | None:
    """Recover only current sealed metadata, never decoded quarantine bytes."""

    if (
        type(vault) is not ProductionAcquisitionVault
        or vault.production_authority is not True
        or type(store) is not SecGemmaOnlineRiskOverlayStore
        or getattr(store, "_store_instance_id", None)
        != vault._bound_store_instance_id
        or stage not in _STAGES
        or type(attempt_id) is not str
        or not attempt_id
        or type(predecessor_handles) is not tuple
        or len(predecessor_handles) != _STAGES.index(stage)
    ):
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production recovery requires one exact stage and predecessor chain"
        )
    expected_predecessor_ids: list[str] = []
    for expected_stage, handle in zip(
        _STAGES[: _STAGES.index(stage)],
        predecessor_handles,
        strict=True,
    ):
        if (
            type(handle) is not VaultHandle
            or handle._sentinel is not _HANDLE_SENTINEL
            or handle._vault_id != vault.vault_id
            or handle.production_authority is not True
            or handle.stage != expected_stage
        ):
            raise SecGemmaOnlineRiskOverlayVaultError(
                "Production recovery predecessor is foreign or reordered"
            )
        _assert_vault_handle_current(vault, handle)
        expected_predecessor_ids.append(handle._entry_id)
    connection = vault._connect()
    try:
        rows = connection.execute(
            "SELECT * FROM quarantine_entries WHERE stage=?",
            (stage,),
        ).fetchall()
    except sqlite3.Error:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production recovery metadata could not be read"
        ) from None
    finally:
        connection.close()
    if not rows:
        return None
    if len(rows) != 1 or str(rows[0]["attempt_id"]) != attempt_id:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production recovery stage binds a foreign attempt"
        )
    row = rows[0]
    try:
        predecessor_bytes = bytes(row["predecessor_entry_ids_json"])
        predecessor_ids = json.loads(
            predecessor_bytes.decode("utf-8", errors="strict")
        )
        if (
            type(predecessor_ids) is not list
            or predecessor_ids != expected_predecessor_ids
            or canonical_json_bytes(predecessor_ids)
            != predecessor_bytes
        ):
            raise ValueError
        production = row["production_authority"]
        metadata = _entry_metadata(
            vault_id=vault.vault_id,
            entry_id=str(row["entry_id"]),
            stage=str(row["stage"]),
            attempt_id=str(row["attempt_id"]),
            generation=int(row["generation"]),
            bundle_sha256=str(row["bundle_sha256"]),
            manifest_sha256=str(row["manifest_sha256"]),
            private_index_sha256=str(row["private_index_sha256"]),
            payload_sha256=str(row["payload_sha256"]),
            predecessor_entry_ids=predecessor_ids,
            production_authority=(
                True
                if production == 1
                else False
                if production == 0
                else None
            ),
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production recovery metadata is corrupt"
        ) from None
    handle = _handle_from_metadata(metadata)
    verified_row, verified_metadata = _verified_entry(vault, handle)
    if (
        verified_metadata != metadata
        or verified_metadata["predecessor_entry_ids"]
        != expected_predecessor_ids
        or hashlib.sha256(bytes(verified_row["payload"])).hexdigest()
        != verified_metadata["payload_sha256"]
    ):
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Production recovery seal verification changed"
        )
    return handle


def _assert_vault_handle_current(
    vault: _AcquisitionVault,
    handle: VaultHandle,
) -> None:
    """Check durable seal freshness without releasing raw entry bytes."""

    _verified_entry(vault, handle)


def is_current_vault_handle(
    vault: Any,
    handle: Any,
) -> bool:
    """Return true only for an intact handle naming current durable bytes."""

    try:
        _assert_vault_handle_current(_vault_type(vault), handle)
    except (SecGemmaOnlineRiskOverlayVaultError, sqlite3.Error, OSError):
        return False
    return True


def _read_quarantine_with_effects(
    vault: _AcquisitionVault,
    handle: VaultHandle,
    *,
    store: Any,
    capability: EffectCapability,
    required_effects: tuple[str, ...],
) -> dict[str, Any]:
    """Capability-gated raw read used only by acquisition detached replay."""

    fixed_vault = _vault_type(vault)
    _authorize_vault_effect(
        fixed_vault,
        store=store,
        capability=capability,
        expected_attempt_id=None,
        required_effects=required_effects,
    )
    row, _metadata = _verified_entry(fixed_vault, handle)
    value = _opaque_value(bytes(row["payload"]))
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayVaultError(
            "Vault quarantine root is corrupt"
        )
    return value


def _read_quarantine_for_replay(
    vault: _AcquisitionVault,
    handle: VaultHandle,
    *,
    store: Any,
    capability: EffectCapability,
) -> dict[str, Any]:
    return _read_quarantine_with_effects(
        vault,
        handle,
        store=store,
        capability=capability,
        required_effects=("official_sec_network", "market_network"),
    )


def _read_quarantine_for_stage_slice(
    vault: _AcquisitionVault,
    handle: VaultHandle,
    *,
    store: Any,
    capability: EffectCapability,
) -> dict[str, Any]:
    return _read_quarantine_with_effects(
        vault,
        handle,
        store=store,
        capability=capability,
        required_effects=("canonical_market_value_read",),
    )


def _read_quarantine_for_model_slice(
    vault: _AcquisitionVault,
    handle: VaultHandle,
    *,
    store: Any,
    capability: EffectCapability,
) -> dict[str, Any]:
    return _read_quarantine_with_effects(
        vault,
        handle,
        store=store,
        capability=capability,
        required_effects=("gemma_batch",),
    )


__all__ = [
    "PRODUCTION_VAULT_RELATIVE_PATH",
    "VAULT_HANDLE_SCHEMA_VERSION",
    "VAULT_SCHEMA_VERSION",
    "ProductionAcquisitionVault",
    "SecGemmaOnlineRiskOverlayVaultError",
    "TestAcquisitionVault",
    "VaultHandle",
    "is_current_vault_handle",
    "open_production_acquisition_vault",
    "open_test_acquisition_vault",
]
