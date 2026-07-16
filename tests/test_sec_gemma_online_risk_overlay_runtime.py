from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_runtime import (
    SecGemmaOnlineRiskOverlayRuntimeError,
    _RuntimePins,
    _validate_runtime_bundle_against_pins,
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fixture() -> dict[str, Any]:
    layers = tuple(hashlib.sha256(f"layer-{i}".encode()).hexdigest() for i in range(4))
    config = _json_bytes(
        {
            "rootfs": {
                "type": "layers",
                "diff_ids": [f"sha256:{digest}" for digest in layers],
            }
        }
    )
    descriptors = [
        {
            "mediaType": (
                "application/vnd.ollama.image.model"
                if index == 0
                else "application/vnd.ollama.image.projector"
                if index == 1
                else "application/vnd.ollama.image.license"
                if index == 2
                else "application/vnd.ollama.image.params"
            ),
            "digest": f"sha256:{digest}",
            "size": index + 1,
            **(
                {"from": "projector.gguf"} if index == 1 else {}
            ),
        }
        for index, digest in enumerate(layers)
    ]
    manifest = _json_bytes(
        {
            "schemaVersion": 2,
            "mediaType": (
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
            "config": {
                "mediaType": (
                    "application/vnd.docker.container.image.v1+json"
                ),
                "digest": f"sha256:{hashlib.sha256(config).hexdigest()}",
                "size": len(config),
            },
            "layers": descriptors,
        }
    )
    version = b'{"version":"0.32.0"}'
    model_info = {"general.architecture": "gemma4"}
    show = _json_bytes(
        {
            "capabilities": ["completion"],
            "details": {"family": "gemma4"},
            "license": "license",
            "model_info": model_info,
            "modelfile": (
                f"FROM C:\\models\\sha256-{layers[0]}\n"
                f"FROM C:\\models\\sha256-{layers[1]}\n"
            ),
            "modified_at": "2026-01-01T00:00:00Z",
            "parameters": "temperature 1",
            "projector_info": {},
            "requires": "0.30.5",
            "template": "{{ .Prompt }}",
            "tensors": [],
        }
    )
    model_info_hash = hashlib.sha256(_json_bytes(model_info)).hexdigest()
    pins_without_fingerprint = {
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "config_digest": hashlib.sha256(config).hexdigest(),
        "layer_digests": layers,
        "ollama_version": "0.32.0",
        "version_response_sha256": hashlib.sha256(version).hexdigest(),
        "show_semantic_sha256": canonical_sha256(
            {
                key: value
                for key, value in json.loads(show).items()
                if key != "modified_at"
            }
        ),
        "show_semantic_excluded_keys": ("modified_at",),
        "model_info_sha256": model_info_hash,
    }
    material = {
        "schema_version": "sec-gemma-v2-1-local-runtime-pin-v1",
        "model_name": "gemma4:12b",
        "ollama_version": "0.32.0",
        "model_manifest_sha256": pins_without_fingerprint[
            "manifest_sha256"
        ],
        "model_config_digest": pins_without_fingerprint["config_digest"],
        "model_layer_digests": list(layers),
        "version_response_sha256": pins_without_fingerprint[
            "version_response_sha256"
        ],
        "show_semantic_sha256": pins_without_fingerprint[
            "show_semantic_sha256"
        ],
        "show_semantic_excluded_keys": ["modified_at"],
        "model_info_sha256": model_info_hash,
    }
    pins = _RuntimePins(
        **pins_without_fingerprint,
        runtime_fingerprint_sha256=canonical_sha256(material),
    )
    return {
        "manifest": manifest,
        "config": config,
        "layers": layers,
        "version": version,
        "show": show,
        "pins": pins,
    }


def _validate(case: dict[str, Any]) -> dict[str, Any]:
    return _validate_runtime_bundle_against_pins(
        manifest_bytes=case["manifest"],
        config_bytes=case["config"],
        layer_content_sha256s=case["layers"],
        version_response_bytes=case["version"],
        show_response_bytes=case["show"],
        pins=case["pins"],
    )


def test_exact_manifest_config_four_layers_and_two_from_blobs_pass() -> None:
    case = _fixture()
    receipt = _validate(case)

    assert receipt["manifest_config_layers_replayed"] is True
    assert receipt["all_layer_contents_hashed"] is True
    assert receipt["exact_two_active_from_blobs_verified"] is True
    assert receipt["active_from_blob_sha256s"] == list(case["layers"][:2])
    assert receipt["show_response_raw_sha256"] == hashlib.sha256(
        case["show"]
    ).hexdigest()
    assert receipt["show_response_semantic_sha256"] == (
        case["pins"].show_semantic_sha256
    )
    assert "show_response_sha256" not in receipt[
        "runtime_fingerprint_material"
    ]


@pytest.mark.parametrize(
    "part",
    ["manifest", "config", "layers", "version"],
)
def test_any_runtime_identity_mutation_fails_closed(part: str) -> None:
    case = _fixture()
    changed = copy.deepcopy(case)
    if part in {"manifest", "config", "version"}:
        changed[part] = changed[part] + b" "
    else:
        changed["layers"] = (
            hashlib.sha256(b"wrong").hexdigest(),
            *changed["layers"][1:],
        )

    with pytest.raises(SecGemmaOnlineRiskOverlayRuntimeError):
        _validate(changed)


def test_raw_whitespace_key_order_and_modified_at_only_changes_pass() -> None:
    case = _fixture()
    show = json.loads(case["show"])
    show["modified_at"] = "2099-12-31T23:59:59Z"
    reordered = {
        key: show[key]
        for key in reversed(list(show))
    }
    case["show"] = json.dumps(
        reordered,
        sort_keys=False,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

    receipt = _validate(case)

    assert receipt["show_response_semantic_sha256"] == (
        case["pins"].show_semantic_sha256
    )
    assert receipt["show_response_raw_sha256"] != hashlib.sha256(
        _fixture()["show"]
    ).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("semantic", "semantic content"),
        ("missing_modified_at", "keys changed"),
        ("non_string_modified_at", "modified_at boundary"),
        ("extra_key", "keys changed"),
    ],
)
def test_show_mutations_other_than_modified_at_fail(
    mutation: str,
    message: str,
) -> None:
    case = _fixture()
    show = json.loads(case["show"])
    if mutation == "semantic":
        show["details"]["family"] = "not-gemma"
    elif mutation == "missing_modified_at":
        del show["modified_at"]
    elif mutation == "non_string_modified_at":
        show["modified_at"] = 1
    else:
        show["unexpected"] = True
    case["show"] = _json_bytes(show)

    with pytest.raises(SecGemmaOnlineRiskOverlayRuntimeError, match=message):
        _validate(case)


def test_one_from_or_reordered_from_lines_fail() -> None:
    case = _fixture()
    show = json.loads(case["show"])
    show["modelfile"] = (
        f"FROM C:\\models\\sha256-{case['layers'][1]}\n"
        f"FROM C:\\models\\sha256-{case['layers'][0]}\n"
    )
    changed_show = _json_bytes(show)
    semantic_hash = canonical_sha256(
        {
            key: value
            for key, value in show.items()
            if key != "modified_at"
        }
    )
    pins = _RuntimePins(
        **{
            **case["pins"].__dict__,
            "show_semantic_sha256": semantic_hash,
        }
    )
    material = {
        "schema_version": "sec-gemma-v2-1-local-runtime-pin-v1",
        "model_name": "gemma4:12b",
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
    pins = _RuntimePins(
        **{
            **pins.__dict__,
            "runtime_fingerprint_sha256": canonical_sha256(material),
        }
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRuntimeError,
        match="exact model and projector",
    ):
        _validate_runtime_bundle_against_pins(
            manifest_bytes=case["manifest"],
            config_bytes=case["config"],
            layer_content_sha256s=case["layers"],
            version_response_bytes=case["version"],
            show_response_bytes=changed_show,
            pins=pins,
        )
