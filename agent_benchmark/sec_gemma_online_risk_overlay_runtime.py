"""Exact local Gemma runtime verification for the online risk overlay.

The pure validator accepts detached bytes and streaming layer digests.  The
owned filesystem wrapper reads only the fixed local Ollama model location and
does not invoke, discover, pull, or mutate a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    MODEL_CONFIG_DIGEST,
    MODEL_LAYER_DIGESTS,
    MODEL_MANIFEST_SHA256,
    MODEL_NAME,
    OLLAMA_VERSION,
    RUNTIME_FINGERPRINT_SHA256,
    build_runtime_fingerprint_material,
    canonical_sha256,
)


RUNTIME_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-runtime-receipt-v1"
)
OLLAMA_VERSION_RESPONSE_SHA256: Final[str] = (
    "2bd89ec9b983123a225f3df0381c737a45302bb7417e345bf9ef92304e4388cf"
)
OLLAMA_SHOW_SEMANTIC_SHA256: Final[str] = (
    "5ccdf8b9a40bb762ea998dc9f5691ab2855d96833d60940b1d93686d08eb47e6"
)
OLLAMA_SHOW_SEMANTIC_EXCLUDED_KEYS: Final[tuple[str, ...]] = (
    "modified_at",
)
OLLAMA_SHOW_RESPONSE_SHA256_DIAGNOSTIC: Final[str] = (
    "5f56fb0fb2214ddcb9fa21c66aa31e37297f553e8758aeda5958f0f287d70893"
)
MODEL_INFO_SHA256: Final[str] = (
    "d21c1c125758901fcea224a7cb9df1057aeba7ebb5177b82d6ba1a096d65fc7b"
)
MANIFEST_RELATIVE_PATH: Final[Path] = Path(
    "manifests/registry.ollama.ai/library/gemma4/12b"
)
_DIGEST_RE = re.compile(r"sha256:([0-9a-f]{64})\Z")
_FROM_RE = re.compile(
    r"FROM\s+.*[/\\]sha256[-:]([0-9a-f]{64})\s*\Z"
)


class SecGemmaOnlineRiskOverlayRuntimeError(ValueError):
    """Raised when the installed local runtime differs from its exact pin."""


@dataclass(frozen=True)
class _RuntimePins:
    manifest_sha256: str
    config_digest: str
    layer_digests: tuple[str, ...]
    ollama_version: str
    version_response_sha256: str
    show_semantic_sha256: str
    show_semantic_excluded_keys: tuple[str, ...]
    model_info_sha256: str
    runtime_fingerprint_sha256: str


_PINNED = _RuntimePins(
    manifest_sha256=MODEL_MANIFEST_SHA256,
    config_digest=MODEL_CONFIG_DIGEST,
    layer_digests=MODEL_LAYER_DIGESTS,
    ollama_version=OLLAMA_VERSION,
    version_response_sha256=OLLAMA_VERSION_RESPONSE_SHA256,
    show_semantic_sha256=OLLAMA_SHOW_SEMANTIC_SHA256,
    show_semantic_excluded_keys=OLLAMA_SHOW_SEMANTIC_EXCLUDED_KEYS,
    model_info_sha256=MODEL_INFO_SHA256,
    runtime_fingerprint_sha256=RUNTIME_FINGERPRINT_SHA256,
)


def _strict_json_bytes(value: bytes, location: str) -> dict[str, Any]:
    if not isinstance(value, bytes) or not value:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            f"{location} must be nonempty bytes"
        )

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if key in result:
                raise SecGemmaOnlineRiskOverlayRuntimeError(
                    f"{location} contains a duplicate JSON key"
                )
            result[key] = child
        return result

    try:
        decoded = value.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SecGemmaOnlineRiskOverlayRuntimeError(
                    f"{location} contains nonfinite JSON token {token}"
                )
            ),
        )
    except SecGemmaOnlineRiskOverlayRuntimeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            f"{location} is not strict UTF-8 JSON"
        ) from exc
    if type(parsed) is not dict:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            f"{location} must contain one JSON object"
        )
    return parsed


def _digest(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _manifest_digest(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            f"{location} must be a manifest digest"
        )
    match = _DIGEST_RE.fullmatch(value)
    if match is None:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            f"{location} must be a sha256 manifest digest"
        )
    return match.group(1)


def _canonical_model_info_sha256(value: Any) -> str:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Ollama show model_info must be an exact object"
        )
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _runtime_fingerprint_material(
    *,
    pins: _RuntimePins,
) -> dict[str, Any]:
    return {
        "schema_version": "sec-gemma-v2-1-local-runtime-pin-v1",
        "model_name": MODEL_NAME,
        "ollama_version": pins.ollama_version,
        "model_manifest_sha256": pins.manifest_sha256,
        "model_config_digest": pins.config_digest,
        "model_layer_digests": list(pins.layer_digests),
        "version_response_sha256": pins.version_response_sha256,
        "show_semantic_sha256": pins.show_semantic_sha256,
        "show_semantic_excluded_keys": list(
            pins.show_semantic_excluded_keys
        ),
        "model_info_sha256": pins.model_info_sha256,
    }


def _validate_runtime_bundle_against_pins(
    *,
    manifest_bytes: bytes,
    config_bytes: bytes,
    layer_content_sha256s: Sequence[str],
    version_response_bytes: bytes,
    show_response_bytes: bytes,
    pins: _RuntimePins,
) -> dict[str, Any]:
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if not hmac.compare_digest(manifest_hash, pins.manifest_sha256):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model manifest hash changed"
        )
    manifest = _strict_json_bytes(manifest_bytes, "model manifest")
    if set(manifest) != {"schemaVersion", "mediaType", "config", "layers"}:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model manifest keys changed"
        )
    if (
        type(manifest["schemaVersion"]) is not int
        or manifest["schemaVersion"] != 2
        or manifest["mediaType"]
        != "application/vnd.docker.distribution.manifest.v2+json"
    ):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model manifest schema changed"
        )
    config = manifest["config"]
    layers = manifest["layers"]
    if type(config) is not dict or set(config) != {
        "mediaType",
        "digest",
        "size",
    }:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model config descriptor changed"
        )
    if (
        config["mediaType"]
        != "application/vnd.docker.container.image.v1+json"
        or _manifest_digest(config["digest"], "config digest")
        != pins.config_digest
        or type(config["size"]) is not int
        or config["size"] != len(config_bytes)
        or hashlib.sha256(config_bytes).hexdigest() != pins.config_digest
    ):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model config bytes changed"
        )
    config_value = _strict_json_bytes(config_bytes, "model config")
    rootfs = config_value.get("rootfs")
    if type(rootfs) is not dict or set(rootfs) != {"type", "diff_ids"}:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model config rootfs changed"
        )
    expected_diff_ids = [
        f"sha256:{digest}" for digest in pins.layer_digests
    ]
    if (
        rootfs["type"] != "layers"
        or rootfs["diff_ids"] != expected_diff_ids
    ):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model config layer order changed"
        )
    if type(layers) is not list or len(layers) != len(pins.layer_digests):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model layer inventory changed"
        )
    descriptor_digests: list[str] = []
    for ordinal, (descriptor, expected_digest) in enumerate(
        zip(layers, pins.layer_digests, strict=True),
        start=1,
    ):
        if type(descriptor) is not dict or set(descriptor) not in (
            {"mediaType", "digest", "size"},
            {"mediaType", "digest", "size", "from"},
        ):
            raise SecGemmaOnlineRiskOverlayRuntimeError(
                f"Installed model layer {ordinal} descriptor changed"
            )
        observed = _manifest_digest(
            descriptor["digest"], f"layer {ordinal} digest"
        )
        if observed != expected_digest:
            raise SecGemmaOnlineRiskOverlayRuntimeError(
                "Installed model layer order or digest changed"
            )
        if type(descriptor["size"]) is not int or descriptor["size"] <= 0:
            raise SecGemmaOnlineRiskOverlayRuntimeError(
                "Installed model layer size is invalid"
            )
        descriptor_digests.append(observed)
    if (
        isinstance(layer_content_sha256s, (str, bytes))
        or not isinstance(layer_content_sha256s, Sequence)
        or tuple(layer_content_sha256s) != pins.layer_digests
    ):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model layer content hashes changed"
        )
    for ordinal, digest in enumerate(layer_content_sha256s, start=1):
        _digest(digest, f"layer content digest {ordinal}")

    version_hash = hashlib.sha256(version_response_bytes).hexdigest()
    if not hmac.compare_digest(
        version_hash, pins.version_response_sha256
    ):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Ollama version response bytes changed"
        )
    version = _strict_json_bytes(
        version_response_bytes, "Ollama version response"
    )
    if set(version) != {"version"} or version["version"] != pins.ollama_version:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Ollama version changed"
        )

    show_raw_hash = hashlib.sha256(show_response_bytes).hexdigest()
    show = _strict_json_bytes(show_response_bytes, "Ollama show response")
    required_show_keys = {
        "capabilities",
        "details",
        "license",
        "model_info",
        "modelfile",
        "modified_at",
        "parameters",
        "projector_info",
        "requires",
        "template",
        "tensors",
    }
    if set(show) != required_show_keys:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Ollama show response keys changed"
        )
    if (
        pins.show_semantic_excluded_keys
        != OLLAMA_SHOW_SEMANTIC_EXCLUDED_KEYS
        or show["modified_at"] is None
        or type(show["modified_at"]) is not str
    ):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Ollama show modified_at boundary changed"
        )
    show_semantic = {
        key: value
        for key, value in show.items()
        if key not in pins.show_semantic_excluded_keys
    }
    show_semantic_hash = canonical_sha256(show_semantic)
    if not hmac.compare_digest(
        show_semantic_hash, pins.show_semantic_sha256
    ):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Ollama show semantic content changed"
        )
    if _canonical_model_info_sha256(show["model_info"]) != pins.model_info_sha256:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Ollama model_info changed"
        )
    modelfile = show["modelfile"]
    if not isinstance(modelfile, str):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Ollama Modelfile is invalid"
        )
    from_digests: list[str] = []
    for line in modelfile.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("FROM "):
            match = _FROM_RE.fullmatch(stripped)
            if match is None:
                raise SecGemmaOnlineRiskOverlayRuntimeError(
                    "Ollama Modelfile FROM line is not an exact blob path"
                )
            from_digests.append(match.group(1))
    if from_digests != list(pins.layer_digests[:2]):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Ollama Modelfile does not bind the exact model and projector"
        )

    material = _runtime_fingerprint_material(pins=pins)
    if canonical_sha256(material) != pins.runtime_fingerprint_sha256:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Derived runtime fingerprint changed"
        )
    body = {
        "schema_version": RUNTIME_RECEIPT_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "manifest_sha256": manifest_hash,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "ordered_layer_sha256s": descriptor_digests,
        "ordered_layer_content_sha256s": list(layer_content_sha256s),
        "version_response_sha256": version_hash,
        "show_response_raw_sha256": show_raw_hash,
        "show_response_semantic_sha256": show_semantic_hash,
        "show_semantic_excluded_keys": list(
            pins.show_semantic_excluded_keys
        ),
        "model_info_sha256": pins.model_info_sha256,
        "active_from_blob_sha256s": from_digests,
        "runtime_fingerprint_material": material,
        "runtime_fingerprint_sha256": pins.runtime_fingerprint_sha256,
        "manifest_config_layers_replayed": True,
        "all_layer_contents_hashed": True,
        "exact_two_active_from_blobs_verified": True,
    }
    return {**body, "runtime_receipt_sha256": canonical_sha256(body)}


def validate_pinned_runtime_bundle(
    *,
    manifest_bytes: bytes,
    config_bytes: bytes,
    layer_content_sha256s: Sequence[str],
    version_response_bytes: bytes,
    show_response_bytes: bytes,
) -> dict[str, Any]:
    """Validate detached runtime evidence against only the frozen production pins."""

    receipt = _validate_runtime_bundle_against_pins(
        manifest_bytes=manifest_bytes,
        config_bytes=config_bytes,
        layer_content_sha256s=layer_content_sha256s,
        version_response_bytes=version_response_bytes,
        show_response_bytes=show_response_bytes,
        pins=_PINNED,
    )
    if build_runtime_fingerprint_material() != receipt[
        "runtime_fingerprint_material"
    ]:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Runtime receipt differs from the preregistered contract material"
        )
    return receipt


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Pinned Ollama model file cannot be read"
        ) from exc
    return digest.hexdigest()


def verify_installed_pinned_runtime(
    *,
    version_response_bytes: bytes,
    show_response_bytes: bytes,
) -> dict[str, Any]:
    """Hash the fixed installed model tree and validate probe bytes."""

    root = Path.home() / ".ollama" / "models"
    manifest_path = root / MANIFEST_RELATIVE_PATH
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Pinned Ollama manifest cannot be read"
        ) from exc
    manifest = _strict_json_bytes(manifest_bytes, "installed model manifest")
    config_digest = _manifest_digest(
        manifest.get("config", {}).get("digest"),
        "installed config digest",
    )
    layers = manifest.get("layers")
    if type(layers) is not list:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model layer inventory is invalid"
        )
    layer_digests = [
        _manifest_digest(layer.get("digest"), f"installed layer {ordinal}")
        for ordinal, layer in enumerate(layers, start=1)
        if type(layer) is dict
    ]
    if len(layer_digests) != len(layers):
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Installed model layer descriptor is invalid"
        )
    config_path = root / "blobs" / f"sha256-{config_digest}"
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayRuntimeError(
            "Pinned Ollama config cannot be read"
        ) from exc
    content_hashes = [
        _file_sha256(root / "blobs" / f"sha256-{digest}")
        for digest in layer_digests
    ]
    return validate_pinned_runtime_bundle(
        manifest_bytes=manifest_bytes,
        config_bytes=config_bytes,
        layer_content_sha256s=content_hashes,
        version_response_bytes=version_response_bytes,
        show_response_bytes=show_response_bytes,
    )


__all__ = [
    "MANIFEST_RELATIVE_PATH",
    "MODEL_INFO_SHA256",
    "OLLAMA_SHOW_RESPONSE_SHA256_DIAGNOSTIC",
    "OLLAMA_SHOW_SEMANTIC_EXCLUDED_KEYS",
    "OLLAMA_SHOW_SEMANTIC_SHA256",
    "OLLAMA_VERSION_RESPONSE_SHA256",
    "RUNTIME_RECEIPT_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayRuntimeError",
    "validate_pinned_runtime_bundle",
    "verify_installed_pinned_runtime",
]
