from __future__ import annotations

import base64
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent_benchmark.sec_filing_content import normalize_filing_text
from agent_benchmark.sec_filing_gemma_contract import (
    DIMENSION_NAMES,
    EXTRACTOR_SCHEMA_VERSION,
    FLAG_NAMES,
    build_extractor_model_payload,
)
from agent_benchmark.sec_gemma_content_risk_inputs import (
    EVENT_SEQUENCES,
    MODEL_MANIFEST_SHA256,
    MODEL_NAME,
    OLLAMA_CHAT_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    OLLAMA_VERSION_ENDPOINT,
    PreparedRequest,
    SecGemmaContentRiskInputError,
    input_commitment_bytes,
    inspect_model_identity,
    load_model_results,
    prepare_requests,
    run_model_batch,
)
from agent_benchmark.sec_gemma_lean_science_v319_contract import canonical_json_bytes


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _build_private_v38(root: Path) -> None:
    (root / "parse_receipts").mkdir(parents=True)
    (root / "blobs").mkdir()
    start = date(2000, 1, 1)
    for sequence in range(122, 199):
        form = "10-K" if sequence % 2 == 0 else "10-Q"
        # Unique wording lets the test prove the immediately-prior same-form
        # selection while remaining fully synthetic and market-free.
        paragraph = (
            "Operations remained stable and customer demand was steady. "
            "Costs increased because several suppliers raised their service fees. "
            "The business continued normal product development and distribution. "
        )
        html = (
            "<html><body><p>"
            + paragraph * 5
            + f"Document token word{sequence} remained internal.</p></body></html>"
        ).encode("latin-1")
        blob_hash = _sha(html)
        blob_name = f"{sequence:06d}-development-{blob_hash}.blob"
        (root / "blobs" / blob_name).write_bytes(html)
        normalized = normalize_filing_text(html.decode("latin-1"))
        normalized_bytes = normalized.text.encode("utf-8")
        frozen_prefix = {"synthetic_sequence": sequence}
        seal = {
            "accession_number": f"0000000000-00-{sequence:06d}",
            "availability_session": (start + timedelta(days=sequence - 122)).isoformat(),
            "form": form,
            "complete_response": {
                "length": len(html),
                "sha256": f"sha256:{blob_hash}",
            },
            "extracted_text": {
                "encoding": "latin-1 exact one-to-one response slice",
                "start_byte": 0,
                "end_byte": len(html),
                "length": len(html),
                "sha256": f"sha256:{blob_hash}",
            },
            "normalized_text": {
                "length": len(normalized_bytes),
                "character_count": normalized.character_count,
                "sha256": normalized.sha256,
            },
            "frozen_prefix": frozen_prefix,
            "frozen_prefix_sha256": _sha(canonical_json_bytes(frozen_prefix)),
        }
        receipt = {
            "sequence": sequence,
            "stage": "development",
            "blob_name": blob_name,
            "body_bytes": len(html),
            "body_sha256": blob_hash,
            "source_evidence": {"seal_row": seal},
        }
        receipt_bytes = canonical_json_bytes(receipt)
        receipt_name = f"{sequence:06d}-{_sha(receipt_bytes)}.json"
        (root / "parse_receipts" / receipt_name).write_bytes(receipt_bytes)


def _prepared(ordinal: int, *, sentence_text: str | None = None) -> PreparedRequest:
    sentences = [
        {
            "id": "C0001",
            "text": sentence_text or f"Anonymous current business sentence {ordinal}.",
        },
        {"id": "P0001", "text": "Anonymous prior business sentence."},
    ]
    payload = build_extractor_model_payload(sentences)
    request_bytes = canonical_json_bytes(payload)
    digest = _sha(request_bytes)
    filler = "a" * 64
    return PreparedRequest(
        ordinal=ordinal,
        sequence=123 + ordinal,
        accession_number=f"0000000000-00-{ordinal:06d}",
        form="10-Q",
        availability_session=f"2000-01-{ordinal:02d}",
        prior_same_form_sequence=121 + ordinal,
        request_sha256=digest,
        model_payload_sha256=digest,
        request_byte_count=len(request_bytes),
        supplied_sentence_ids=("C0001", "P0001"),
        source_preprocessed_event_sha256=filler,
        sanitized_preprocessed_event_sha256=filler,
        removed_current_sentence_count=0,
        removed_prior_sentence_count=0,
        retained_sentence_count=2,
        current_complete_response_sha256=filler,
        current_selected_text_sha256=filler,
        current_normalized_text_sha256=filler,
        prior_normalized_text_sha256=filler,
        request_bytes=request_bytes,
    )


def _extractor_output(quality: str = "usable") -> dict[str, Any]:
    return {
        "schema_version": EXTRACTOR_SCHEMA_VERSION,
        "document_quality": quality,
        "dimensions": {
            name: {
                "current_impact": "not_stated",
                "change_vs_prior": "not_stated",
                "evidence_sentence_ids": [],
            }
            for name in DIMENSION_NAMES
        },
        "flags": {
            name: {"present": False, "evidence_sentence_ids": []}
            for name in FLAG_NAMES
        },
    }


class _Response:
    def __init__(self, url: str, value: Any, *, status_code: int = 200) -> None:
        self.url = url
        self.status_code = status_code
        self.history: list[Any] = []
        self._body = canonical_json_bytes(value)
        self.closed = False

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self._body

    def close(self) -> None:
        self.closed = True


class _Transport:
    def __init__(
        self,
        outputs: list[Any] | None = None,
        *,
        fail_posts: set[int] | None = None,
        crash_posts: set[int] | None = None,
        envelope_extras: dict[str, Any] | None = None,
        message_extras: dict[str, Any] | None = None,
        extra_models: list[dict[str, Any]] | None = None,
    ):
        self.outputs = list(outputs or [])
        self.fail_posts = set(fail_posts or set())
        self.crash_posts = set(crash_posts or set())
        self.envelope_extras = dict(envelope_extras or {})
        self.message_extras = dict(message_extras or {})
        self.extra_models = list(extra_models or [])
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.post_count = 0

    def request(self, method: str, url: str, **kwargs: Any):
        self.calls.append((method, url, kwargs))
        if method == "GET" and url == OLLAMA_VERSION_ENDPOINT:
            return _Response(url, {"version": "0.12.1"})
        if method == "GET" and url == OLLAMA_TAGS_ENDPOINT:
            return _Response(
                url,
                {
                    "models": [
                        {
                            "name": MODEL_NAME,
                            "model": MODEL_NAME,
                            "digest": MODEL_MANIFEST_SHA256,
                        },
                        *self.extra_models,
                    ]
                },
            )
        assert method == "POST" and url == OLLAMA_CHAT_ENDPOINT
        self.post_count += 1
        if self.post_count in self.crash_posts:
            raise KeyboardInterrupt("synthetic interrupted process")
        if self.post_count in self.fail_posts:
            raise ConnectionError("synthetic transport failure")
        output = self.outputs.pop(0)
        content = output if isinstance(output, str) else json.dumps(output)
        envelope = {
                "model": MODEL_NAME,
                "created_at": "2026-07-19T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": content,
                    **self.message_extras,
                },
                "done": True,
                "done_reason": "stop",
                "total_duration": 10,
                "load_duration": 1,
                "prompt_eval_count": 100,
                "prompt_eval_duration": 2,
                "eval_count": 20,
                "eval_duration": 7,
                **self.envelope_extras,
            }
        return _Response(url, envelope)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        result = self.value
        self.value += 1.0
        return result


def test_prepare_requests_authenticates_bytes_and_uses_immediate_prior(tmp_path: Path):
    private_root = tmp_path / "development"
    _build_private_v38(private_root)

    requests = prepare_requests(private_root, verify_frozen=False)

    assert len(requests) == 75
    assert [item.sequence for item in requests] == list(EVENT_SEQUENCES)
    assert requests[0].prior_same_form_sequence == 122
    assert requests[1].prior_same_form_sequence == 123
    assert requests[2].prior_same_form_sequence == 124
    commitment = json.loads(input_commitment_bytes(requests))
    assert set(commitment[0]) == {
        "sequence",
        "accession_sha256",
        "availability_session",
        "model_payload_sha256",
        "request_bytes",
        "request_sha256",
        "sentences",
        "removed_current",
        "removed_prior",
    }
    assert "0000000000-00-000124" not in input_commitment_bytes(requests).decode()


def test_prepare_requests_rejects_tampered_complete_response(tmp_path: Path):
    private_root = tmp_path / "development"
    _build_private_v38(private_root)
    receipt_path = next((private_root / "parse_receipts").glob("000124-*.json"))
    receipt = json.loads(receipt_path.read_bytes())
    blob_path = private_root / "blobs" / receipt["blob_name"]
    blob_path.write_bytes(blob_path.read_bytes() + b"tamper")

    with pytest.raises(SecGemmaContentRiskInputError, match="complete SEC response"):
        prepare_requests(private_root, verify_frozen=False)


def test_inspect_model_identity_uses_only_version_and_tags():
    transport = _Transport()

    identity = inspect_model_identity(transport=transport)

    assert identity["ollama_version"] == "0.12.1"
    assert identity["model_manifest_sha256"] == MODEL_MANIFEST_SHA256
    assert [(method, url) for method, url, _ in transport.calls] == [
        ("GET", OLLAMA_VERSION_ENDPOINT),
        ("GET", OLLAMA_TAGS_ENDPOINT),
    ]


def test_inspect_model_identity_rejects_changed_digest():
    transport = _Transport()
    original = transport.request

    def changed(method: str, url: str, **kwargs: Any):
        if url == OLLAMA_TAGS_ENDPOINT:
            return _Response(
                url,
                {"models": [{"name": MODEL_NAME, "digest": "b" * 64}]},
            )
        return original(method, url, **kwargs)

    transport.request = changed  # type: ignore[method-assign]
    with pytest.raises(SecGemmaContentRiskInputError, match="digest changed"):
        inspect_model_identity(transport=transport)


def test_batch_is_one_call_per_ordinal_resumable_and_public_safe(tmp_path: Path):
    requests = [
        _prepared(1, sentence_text="Private source sentence must not be checkpointed."),
        _prepared(2),
    ]
    transport = _Transport([_extractor_output(), "not-json"])
    progress: list[dict[str, Any]] = []

    first = run_model_batch(
        requests,
        tmp_path,
        ordinals=[1, 2],
        progress=progress.append,
        transport=transport,
        clock=_Clock(),
    )

    assert transport.post_count == 2
    assert [row["status"] for row in first] == ["valid", "invalid"]
    assert first[0]["supplied_sentence_ids"] == ["C0001", "P0001"]
    assert first[0]["extractor_output_sha256"] == _sha(
        canonical_json_bytes(first[0]["extractor_output"])
    )
    assert first[1]["extractor_output"] is None
    assert len(progress) == 2
    assert set(progress[0]) >= {
        "ordinal",
        "status",
        "elapsed_seconds",
        "completed_count",
        "rolling_median_seconds",
        "estimated_remaining_seconds",
    }
    checkpoint_text = (tmp_path / "model-call-0001.json").read_text()
    assert "Private source sentence" not in checkpoint_text
    assert str(tmp_path) not in checkpoint_text

    resume_transport = _Transport([])
    second = run_model_batch(
        requests,
        tmp_path,
        ordinals=[1, 2],
        progress=lambda event: pytest.fail(f"unexpected repeated call: {event}"),
        transport=resume_transport,
        clock=_Clock(),
    )
    assert resume_transport.post_count == 0
    assert second == first


def test_transport_error_is_permanent_and_later_call_continues(tmp_path: Path):
    requests = [_prepared(1), _prepared(2)]
    transport = _Transport([_extractor_output()], fail_posts={1})

    results = run_model_batch(
        requests,
        tmp_path,
        ordinals=[1, 2],
        progress=lambda event: None,
        transport=transport,
        clock=_Clock(),
    )

    assert transport.post_count == 2
    assert [row["status"] for row in results] == ["transport_error", "valid"]


def test_interrupted_in_progress_attempt_is_never_resent(tmp_path: Path):
    request = _prepared(1)
    with pytest.raises(KeyboardInterrupt, match="interrupted process"):
        run_model_batch(
            [request],
            tmp_path,
            ordinals=[1],
            progress=lambda event: None,
            transport=_Transport([_extractor_output()], crash_posts={1}),
            clock=_Clock(),
        )
    marker = json.loads((tmp_path / "model-call-0001.json").read_bytes())
    assert marker["status"] == "in_progress"

    resume_transport = _Transport([])
    [result] = run_model_batch(
        [request],
        tmp_path,
        ordinals=[1],
        progress=lambda event: pytest.fail(f"attempt was repeated: {event}"),
        transport=resume_transport,
        clock=_Clock(),
    )
    assert resume_transport.post_count == 0
    assert result["status"] == "transport_error"
    assert result["reason"] == "interrupted_attempt_no_retry"


def test_harmless_ollama_metadata_is_allowed_but_output_extras_are_invalid(
    tmp_path: Path,
):
    requests = [_prepared(1), _prepared(2)]
    harmless = _Transport(
        [_extractor_output(), _extractor_output()],
        envelope_extras={"context": [1, 2, 3], "runtime_metadata": {"gpu": True}},
        message_extras={"metadata": {"format": "json"}},
    )
    [first] = run_model_batch(
        requests,
        tmp_path / "safe",
        ordinals=[1],
        progress=lambda event: None,
        transport=harmless,
        clock=_Clock(),
    )
    assert first["status"] == "valid"

    dangerous = _Transport(
        [_extractor_output()], message_extras={"thinking": "hidden output"}
    )
    results = run_model_batch(
        requests,
        tmp_path / "unsafe",
        ordinals=[2],
        progress=lambda event: None,
        transport=dangerous,
        clock=_Clock(),
    )
    second = next(row for row in results if row["ordinal"] == 2)
    assert second["status"] == "invalid"


def test_schema_valid_unusable_output_is_permanent_invalid(tmp_path: Path):
    request = _prepared(1)
    transport = _Transport([_extractor_output("unusable")])

    [result] = run_model_batch(
        [request],
        tmp_path,
        ordinals=[1],
        progress=lambda event: None,
        transport=transport,
        clock=_Clock(),
    )

    assert result["status"] == "invalid"
    assert result["extractor_output"] is None
    assert result["extractor_output_sha256"] is None
    assert result["reason"] == "unusable_document_no_retry_no_repair"


def test_default_batch_runs_fixed_pilot_then_stops_below_five_valid(tmp_path: Path):
    requests = [_prepared(ordinal) for ordinal in range(1, 76)]
    outputs = [
        _extractor_output(),
        _extractor_output(),
        _extractor_output(),
        _extractor_output(),
        "invalid-json",
        "invalid-json",
    ]
    transport = _Transport(outputs)
    progress: list[dict[str, Any]] = []

    results = run_model_batch(
        requests,
        tmp_path,
        progress=progress.append,
        transport=transport,
        clock=_Clock(),
    )

    assert transport.post_count == 6
    assert [event["ordinal"] for event in progress] == [1, 15, 30, 45, 60, 75]
    assert len(results) == 6
    assert sum(row["status"] == "valid" for row in results) == 4


def test_default_batch_continues_after_healthy_fixed_pilot(tmp_path: Path):
    requests = [_prepared(ordinal) for ordinal in range(1, 76)]
    transport = _Transport([_extractor_output() for _ in range(75)])
    progress: list[dict[str, Any]] = []

    results = run_model_batch(
        requests,
        tmp_path,
        progress=progress.append,
        transport=transport,
        clock=_Clock(),
    )

    assert transport.post_count == 75
    assert [event["ordinal"] for event in progress[:6]] == [1, 15, 30, 45, 60, 75]
    assert {row["ordinal"] for row in results} == set(range(1, 76))
    assert all(row["status"] == "valid" for row in results)


def test_sealed_preoutput_pilot_can_continue_without_repeating_pilot(tmp_path: Path):
    requests = [_prepared(ordinal) for ordinal in range(1, 76)]
    pilot_transport = _Transport(
        [_extractor_output() for _ in range(6)],
        message_extras={"thinking": "legacy wrapper extra"},
    )
    first = run_model_batch(
        requests,
        tmp_path,
        progress=lambda event: None,
        transport=pilot_transport,
        clock=_Clock(),
    )
    assert len(first) == 6
    assert all(row["status"] == "invalid" for row in first)
    assert all(row["raw_output_sha256"] is None for row in first)

    continuation_transport = _Transport(
        [_extractor_output() for _ in range(69)]
    )
    results = run_model_batch(
        requests,
        tmp_path,
        progress=lambda event: None,
        transport=continuation_transport,
        clock=_Clock(),
        allow_sealed_preoutput_pilot_continuation=True,
    )
    assert continuation_transport.post_count == 69
    assert len(results) == 75
    assert sum(row["status"] == "valid" for row in results) == 69


def test_explicit_nonpilot_cannot_bypass_fixed_pilot(tmp_path: Path):
    requests = [_prepared(ordinal) for ordinal in range(1, 76)]
    blocked_transport = _Transport([])
    with pytest.raises(SecGemmaContentRiskInputError, match="healthy fixed pilot"):
        run_model_batch(
            requests,
            tmp_path,
            ordinals=[2],
            progress=lambda event: None,
            transport=blocked_transport,
            clock=_Clock(),
        )
    assert blocked_transport.post_count == 0

    pilot_transport = _Transport([_extractor_output() for _ in range(6)])
    run_model_batch(
        requests,
        tmp_path,
        ordinals=[1, 15, 30, 45, 60, 75],
        progress=lambda event: None,
        transport=pilot_transport,
        clock=_Clock(),
    )
    nonpilot_transport = _Transport([_extractor_output()])
    results = run_model_batch(
        requests,
        tmp_path,
        ordinals=[2],
        progress=lambda event: None,
        transport=nonpilot_transport,
        clock=_Clock(),
    )
    assert nonpilot_transport.post_count == 1
    assert next(row for row in results if row["ordinal"] == 2)["status"] == "valid"


def test_resume_rejects_changed_original_runtime_identity(tmp_path: Path):
    request = _prepared(1)
    run_model_batch(
        [request],
        tmp_path,
        ordinals=[1],
        progress=lambda event: None,
        transport=_Transport([_extractor_output()]),
        clock=_Clock(),
    )
    before_path = tmp_path / "runtime-before.json"
    before = json.loads(before_path.read_bytes())
    before["ollama_version"] = "0.12.0"
    before_path.write_bytes(canonical_json_bytes(before))

    transport = _Transport([])
    with pytest.raises(SecGemmaContentRiskInputError, match="differs from its first"):
        run_model_batch(
            [request],
            tmp_path,
            ordinals=[1],
            progress=lambda event: None,
            transport=transport,
            clock=_Clock(),
        )
    assert transport.post_count == 0


def test_resume_ignores_observational_tag_hash_and_unrelated_models(tmp_path: Path):
    request = _prepared(1)
    run_model_batch(
        [request],
        tmp_path,
        ordinals=[1],
        progress=lambda event: None,
        transport=_Transport([_extractor_output()]),
        clock=_Clock(),
    )
    changed_tags = _Transport(
        [],
        extra_models=[{"name": "unrelated:latest", "digest": "c" * 64}],
    )
    results = run_model_batch(
        [request],
        tmp_path,
        ordinals=[1],
        progress=lambda event: None,
        transport=changed_tags,
        clock=_Clock(),
    )
    assert results[0]["status"] == "valid"
    assert changed_tags.post_count == 0


def test_progress_counts_only_selected_ordinals(tmp_path: Path):
    requests = [_prepared(1), _prepared(2)]
    run_model_batch(
        requests,
        tmp_path,
        ordinals=[1],
        progress=lambda event: None,
        transport=_Transport([_extractor_output()]),
        clock=_Clock(),
    )
    progress: list[dict[str, Any]] = []
    run_model_batch(
        requests,
        tmp_path,
        ordinals=[2],
        progress=progress.append,
        transport=_Transport([_extractor_output()]),
        clock=_Clock(),
    )
    assert progress[0]["completed_count"] == 1
    assert progress[0]["estimated_remaining_seconds"] == 0.0


def test_load_revalidates_output_evidence_ids_and_hashes(tmp_path: Path):
    request = _prepared(1)
    run_model_batch(
        [request],
        tmp_path,
        ordinals=[1],
        progress=lambda event: None,
        transport=_Transport([_extractor_output()]),
        clock=_Clock(),
    )
    path = tmp_path / "model-call-0001.json"
    checkpoint = json.loads(path.read_bytes())
    dimension = next(iter(checkpoint["extractor_output"]["dimensions"].values()))
    dimension["current_impact"] = "unfavorable"
    dimension["evidence_sentence_ids"] = ["C9999"]
    exact = canonical_json_bytes(checkpoint["extractor_output"])
    checkpoint["exact_output_json_utf8_base64"] = base64.b64encode(exact).decode()
    checkpoint["raw_output_sha256"] = _sha(exact)
    checkpoint["extractor_output_sha256"] = _sha(exact)
    checkpoint["output_byte_count"] = len(exact)
    path.write_bytes(canonical_json_bytes(checkpoint))

    with pytest.raises(Exception, match="Unknown evidence id"):
        load_model_results(tmp_path, [request])


def test_model_post_uses_frozen_bytes_generous_timeout_and_no_retry(tmp_path: Path):
    request = _prepared(1)
    transport = _Transport([_extractor_output()])

    run_model_batch(
        [request],
        tmp_path,
        ordinals=[1],
        progress=lambda event: None,
        transport=transport,
        clock=_Clock(),
    )

    posts = [call for call in transport.calls if call[0] == "POST"]
    assert len(posts) == 1
    kwargs = posts[0][2]
    assert kwargs["data"] == request.request_bytes
    assert kwargs["timeout"] == (2.0, 600.0)
    assert kwargs["allow_redirects"] is False
    assert "json" not in kwargs
