from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

import agent_benchmark.sec_gemma_lean_v37_transport as transport


PRIVATE_CONTACT = (
    'Alder evidence "A&B" sec+contact@alder-research-739184.com'
)
INTENT_SHA256 = "1" * 64
_SELF_SIGNED_KEY_DER_BASE64 = (
    "MIIEogIBAAKCAQEAmw3D0peVNLJMu1xeJcK2EVg/D8Mak7dWBj43nciqVZXDxYQ2"
    "J97b/zM8Fl8NstZeXJJcUbPPyiYRotmIOuoNvmxQv1xk+J0659bPnHTrnYjMQKGe"
    "MrM8hCX6AcxhgaD/M9tj7Lf60hOKQ5igZexL56tQMw9WLtSCeYwZss2Zt+kdtHdP"
    "ocVlazfhLBYEGj+VXlg+1OlDJv/DqdT4HCTmE1M+5ICGOzHmzggs0LK63K4l+U6D"
    "4S39OvHdMm45Xr6+YZCAMe3AYA/1O44tWuNv43zSOmo+yGZKja7KKZ00DJ1sqq7k"
    "Smz/2es6f5X+T1PJdMsrVKM8g4bounKoabM8/wIDAQABAoIBAChKP9+phtpeGGU5"
    "H7e1LEU+hohsfKv0oFRX93C3E1fQ5kGdVMswdD8Qi6UOhws4++UXHQkX7b8L/Foy"
    "J12TswWPtL283vnhNUzH+0Oe+BiD0vtaY7at72QxUBEGkDG0aYwKjqEUv4a2EB/3"
    "eXpyl9i3ocvayNy7WLoHxye4mCB6+aeyP27gunWpfE4tjpwRHEBlqn1cFLGGDeC3"
    "4pOFhdtnz5b+NqeqABmea1z9u3bYzZhPgFz6AAvhx2qT7Gble4/7GfJSeXMUJDUz"
    "SrbZPRc9q8w770f/dUl8NXRU2HSxbjQHpmDkaywCcXJtf2Elaeq7CIBSqWOWBNvz"
    "Su/L/LkCgYEAzWWA0qINN8LpsBCXi6YVi9WQLfbp0KSSAADt3KJwa68sHasovOaH"
    "LJvV9AlSJy/qassJvAmF0eGft+PeZDuJn/A1GYTkOth8z/Y5pQng69pkc3bbQgWA"
    "qrxIeDgLs3vRrZP1u2DHH+66o2iSAa5JIlGc6NLlmyb3IgVfNZFy2/kCgYEAwUEc"
    "lEw55AlITpd2q/Yp/MONoLj6ic8ET6XmV/+lK0hTIXPSvpd4VWhExb/pfPlnq+u5"
    "jWU8vrtMo6bUN3/bIjGrGxHmiU1QZVuWD7ybe3eYzy4dVvGXsqPKWU0yJxru9GU7"
    "5LVh1eaFOaQNtVlkJE4q5YBc4F4YqkX7VI1rbrcCgYBKQVDyVU2sBBZR5Z20CeYu"
    "OJY4V9St7mEuCSf1JPC5rYmobDF9IWZrFSejYx9FNYhs9Vxek57CguwoIgRLfk7B"
    "+KhpwmZ0c8GjuWObq9eZzMmWCLk5xB2BHDKi67gnOjNSqnQjOtiiTb9BxlNKskSU"
    "WKb+cQg7MDoWHZUPG7dHmQKBgHcKFYGfxpUZxqlqkRXxxwFEr8vNxb33f89UAKQ5"
    "+9LCdTqI7sqp4NyzIpGw2jE6K8rxo9VeA+H024t+6v/YyPGyKJS/HQN8VUZp+PBu"
    "nFOcYstTu4zfujK2w6DodTkzVPfEF/WUaNRqb6wGys7nZlEauT+vJwapz4WrH9qA"
    "VinXAoGACF7UZ2YyRPRNHPcfENAQYdIGyJebiZ1QNA47YKKjkMtCKS/8FkkumRW8"
    "+tu93jNtK84kzaEc60wwDQg77cnomv20fMTN0WFFPytzMukajr1DMVtAnfiYyVvv"
    "qOgCVUUm1QFv4zsy05kEFYSY48y6VYDF4WBkUST9CQrhjQwr3GM="
)
_SELF_SIGNED_CERT_DER_BASE64 = (
    "MIIC1DCCAbygAwIBAgIBATANBgkqhkiG9w0BAQsFADAUMRIwEAYDVQQDDAlsb2Nh"
    "bGhvc3QwIBcNMjAwMTAxMDAwMDAwWhgPMjEyMDAxMDEwMDAwMDBaMBQxEjAQBgNV"
    "BAMMCWxvY2FsaG9zdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAJsN"
    "w9KXlTSyTLtcXiXCthFYPw/DGpO3VgY+N53IqlWVw8WENife2/8zPBZfDbLWXlyS"
    "XFGzz8omEaLZiDrqDb5sUL9cZPidOufWz5x0652IzEChnjKzPIQl+gHMYYGg/zPb"
    "Y+y3+tITikOYoGXsS+erUDMPVi7UgnmMGbLNmbfpHbR3T6HFZWs34SwWBBo/lV5Y"
    "PtTpQyb/w6nU+Bwk5hNTPuSAhjsx5s4ILNCyutyuJflOg+Et/Trx3TJuOV6+vmGQ"
    "gDHtwGAP9TuOLVrjb+N80jpqPshmSo2uyimdNAydbKqu5Eps/9nrOn+V/k9TyXTL"
    "K1SjPIOG6LpyqGmzPP8CAwEAAaMvMC0wGgYDVR0RBBMwEYIJbG9jYWxob3N0hwR/"
    "AAABMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBADWPb1GSOJp7"
    "4/r2DllPCFyQYe7frSvIEg7Q3cvj5fNVo8Vo+VxYOVoii4TkTmmidCUWy94Rj17Z"
    "uxrFEd/C1Gin1OX8aFdtg5vtriT0EVv7sYn447NNfCatsb+VHwSz1MWVbIvTSICB"
    "Bqoov+nWz93iunJwekDa6N1gm6miAOkoulu/SLu6dfxDVCcZFhry49d4p/JkyOrH"
    "Dzd/GCEa6OEoYHKxnh1dkJS5yoUBi6kFjYELNCYqKAJvFE64T9BG2lbiFixhxGml"
    "5B3ENH9oJh/c+Peq8CtSZQAeNTosOGXZrOcal76pgDk1x19NWuNsFZ33iBKyqIUj"
    "0CR1NzkdWvQ="
)


def _write_pem_from_der_base64(
    path: Path, *, label: str, der_base64: str
) -> None:
    der = base64.b64decode(der_base64, validate=True)
    encoded = base64.b64encode(der).decode("ascii")
    lines = [encoded[index : index + 64] for index in range(0, len(encoded), 64)]
    path.write_text(
        f"-----BEGIN {label}-----\n"
        + "\n".join(lines)
        + f"\n-----END {label}-----\n",
        encoding="ascii",
        newline="\n",
    )


def _raw(
    body: bytes = b"",
    *,
    headers: tuple[bytes, ...] = (),
    status: bytes = b"200 OK",
) -> bytes:
    return (
        b"HTTP/1.1 "
        + status
        + b"\r\n"
        + b"".join(header + b"\r\n" for header in headers)
        + b"\r\n"
        + body
    )


def _error_code(payload: bytes, *, limit: int = 32) -> str:
    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        transport.parse_raw_http_response(payload, body_limit=limit)
    assert captured.value.__cause__ is None
    return captured.value.code


def test_capability_is_exactly_fail_closed() -> None:
    capability = transport.transport_capability()
    assert capability.official_hosts == ("data.sec.gov", "www.sec.gov")
    assert capability.certificate_required is True
    assert capability.hostname_checking is True
    assert capability.trust_store == "runtime_default"
    assert capability.custom_ca_allowed is False
    assert capability.proxy_allowed is False
    assert capability.redirects_allowed is False
    assert capability.retries_allowed is False
    assert capability.cache_reads_allowed is False
    assert capability.cache_writes_allowed is False
    assert capability.accept_encoding == "identity"
    assert capability.hard_deadline_seconds == 30.0
    assert capability.isolated_child is True
    assert capability.sanitized_child_environment is True
    assert capability.bytecode_writes_allowed is False
    assert capability.worker_python_flags == ("-I", "-S", "-B")
    identity = transport.execution_identity()
    for key, value in identity.items():
        assert getattr(capability, key) == value
    source = Path(transport.__file__).resolve(strict=True)
    executable = Path(sys.executable).resolve(strict=True)
    assert source.suffix == ".py"
    assert capability.transport_source_sha256 == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert capability.transport_source_bytes == source.stat().st_size
    assert capability.worker_bootstrap_sha256 == hashlib.sha256(
        transport._WORKER_BOOTSTRAP.encode("utf-8")
    ).hexdigest()
    assert capability.interpreter_executable_sha256 == hashlib.sha256(
        executable.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "url",
    (
        "http://data.sec.gov/submissions/CIK0000320193.json",
        "HTTPS://data.sec.gov/submissions/CIK0000320193.json",
        "https://DATA.sec.gov/submissions/CIK0000320193.json",
        "https://data.sec.gov:443/submissions/CIK0000320193.json",
        "https://user@data.sec.gov/submissions/CIK0000320193.json",
        "https://www.sec.gov/Archives/a.txt#fragment",
        "https://www.sec.gov",
        "https://evil.example/Archives/a.txt",
        "https://www.sec.gov/has space",
        "https://www.sec.gov/back\\slash",
        "https://www.sec.gov/caf\N{LATIN SMALL LETTER E WITH ACUTE}",
        "https://data.sec.gov/submissions/CIK0000320193.json?x=1",
        "https://data.sec.gov/submissions//CIK0000320193.json",
        "https://data.sec.gov/submissions/../submissions/CIK0000320193.json",
        "https://data.sec.gov/submissions/%43IK0000320193.json",
        "https://data.sec.gov/submissions/CIK0000320193%.json",
        "https://data.sec.gov/submissions/CIK0000320193-submissions-0001.json",
        "https://www.sec.gov/Archives/edgar/full-index/1993/QTR4/master.gz",
        "https://www.sec.gov/Archives/edgar/full-index/2026/QTR4/master.gz",
        "https://www.sec.gov/Archives/edgar/full-index/2024/QTR5/master.gz",
        "https://www.sec.gov/Archives/edgar/full-index/1994/QTR1/master.idx",
        "https://www.sec.gov/Archives/edgar/daily-index/1994/QTR1/master.gz",
        "https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.gz/",
        "https://www.sec.gov/Archives/edgar/data/320193/000091205700023442.txt",
        "https://www.sec.gov/Archives/edgar/data/320193//0000912057-00-023442.txt",
    ),
)
def test_noncanonical_urls_are_rejected(url: str) -> None:
    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        transport.canonical_sec_url(url)
    assert captured.value.code == "url_mismatch"


@pytest.mark.parametrize(
    "url",
    (
        "https://data.sec.gov/submissions/CIK0000320193.json",
        "https://www.sec.gov/Archives/edgar/full-index/1994/QTR1/master.gz",
        "https://www.sec.gov/Archives/edgar/full-index/1994/QTR2/master.gz",
        "https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.gz",
        "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json",
        "https://www.sec.gov/Archives/edgar/data/320193/0000912057-00-023442.txt",
    ),
)
def test_canonical_urls_are_byte_stable(url: str) -> None:
    assert transport.canonical_sec_url(url) == url


@pytest.mark.parametrize(
    ("url", "role_class", "role_id", "body_limit"),
    (
        (
            "https://data.sec.gov/submissions/CIK0000320193.json",
            "main_submissions",
            "submissions/main",
            transport.MAIN_SUBMISSIONS_BODY_LIMIT,
        ),
        (
            "https://data.sec.gov/submissions/CIK0000320193-submissions-009.json",
            "historical_submissions",
            "submissions/historical/CIK0000320193-submissions-009.json",
            transport.HISTORICAL_SUBMISSIONS_BODY_LIMIT,
        ),
        (
            "https://www.sec.gov/Archives/edgar/full-index/1994/QTR1/master.gz",
            "quarterly_master",
            "master/1994/QTR1",
            transport.MASTER_COMPRESSED_BODY_LIMIT,
        ),
        (
            "https://www.sec.gov/Archives/edgar/full-index/1994/QTR2/master.gz",
            "quarterly_master",
            "master/1994/QTR2",
            transport.MASTER_COMPRESSED_BODY_LIMIT,
        ),
        (
            "https://www.sec.gov/Archives/edgar/data/320193/0000912057-00-023442.txt",
            "complete_submission",
            "complete/0000912057-00-023442",
            transport.COMPLETE_SUBMISSION_BODY_LIMIT,
        ),
    ),
)
def test_exact_url_derives_one_exact_journal_role_and_limit(
    url: str, role_class: str, role_id: str, body_limit: int
) -> None:
    binding = transport.derive_sec_role(url)
    assert binding == transport.SecRoleBinding(role_class, role_id, body_limit)


def test_head_preserves_duplicate_fields_in_wire_order() -> None:
    raw_head = (
        b"HTTP/1.1 200 OK\r\n"
        b"X-Test: one\r\n"
        b"x-test: two\r\n"
        b"Content-Length: 0\r\n\r\n"
    )
    head = transport.parse_http_response_head(raw_head)
    assert head.raw_head == raw_head
    assert head.headers == (
        (b"X-Test", b"one"),
        (b"x-test", b"two"),
        (b"Content-Length", b"0"),
    )
    assert head.values("X-Test") == (b"one", b"two")


def test_valid_content_length_identity_response() -> None:
    parsed = transport.parse_raw_http_response(
        _raw(
            b"hello",
            headers=(b"Content-Length: 5", b"Content-Encoding: identity"),
        ),
        body_limit=5,
    )
    assert parsed.body == b"hello"
    assert parsed.framing == "content-length"
    assert parsed.declared_content_length == 5
    assert parsed.content_encoding == "identity"


def test_valid_chunked_response_with_well_formed_extensions() -> None:
    parsed = transport.parse_raw_http_response(
        _raw(
            b'4;name=value\r\nWiki\r\n5; quoted="a\\\"b"\r\npedia\r\n0\r\n\r\n',
            headers=(b"Transfer-Encoding: chunked",),
        ),
        body_limit=9,
    )
    assert parsed.body == b"Wikipedia"
    assert parsed.framing == "chunked"
    assert parsed.declared_content_length is None


def test_valid_connection_eof_response() -> None:
    parsed = transport.parse_raw_http_response(
        _raw(b"eof-body"), body_limit=8
    )
    assert parsed.body == b"eof-body"
    assert parsed.framing == "eof"


@pytest.mark.parametrize(
    ("headers", "expected"),
    (
        ((b"Content-Length: 1", b"Content-Length: 1"), "content_length_rejected"),
        ((b"Content-Length: 1, 1",), "content_length_rejected"),
        ((b"Content-Length: +1",), "content_length_rejected"),
        ((b"Content-Length: -1",), "content_length_rejected"),
        ((b"Content-Length: 1.0",), "content_length_rejected"),
        ((b"Content-Length: 0x1",), "content_length_rejected"),
        (
            (b"Content-Length: 1", b"Transfer-Encoding: chunked"),
            "http_framing_rejected",
        ),
        (
            (b"Transfer-Encoding: chunked", b"Transfer-Encoding: chunked"),
            "transfer_encoding_rejected",
        ),
        ((b"Transfer-Encoding: gzip",), "transfer_encoding_rejected"),
        ((b"Transfer-Encoding: gzip, chunked",), "transfer_encoding_rejected"),
        (
            (b"Content-Encoding: identity", b"Content-Encoding: identity"),
            "content_encoding_rejected",
        ),
        ((b"Content-Encoding: gzip",), "content_encoding_rejected"),
        ((b"Content-Encoding: identity, gzip",), "content_encoding_rejected"),
    ),
)
def test_ambiguous_or_encoded_framing_is_rejected(
    headers: tuple[bytes, ...], expected: str
) -> None:
    assert _error_code(_raw(b"x", headers=headers)) == expected


@pytest.mark.parametrize(
    "payload",
    (
        b"HTTP/1.0 200 OK\r\n\r\n",
        b"HTTP/1.1 20 OK\r\n\r\n",
        b"HTTP/1.1 200 OK\n\n",
        b"HTTP/1.1 200 OK\r\n folded\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nNo-Colon\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nBad Name: x\r\n\r\n",
        b"HTTP/1.1 200 O\x01K\r\n\r\n",
    ),
)
def test_malformed_response_heads_are_rejected(payload: bytes) -> None:
    assert _error_code(payload) in {"http_framing_rejected", "premature_eof"}


def test_non_200_status_is_fixed_code_and_redirect_is_not_followed() -> None:
    payload = _raw(
        headers=(b"Location: https://evil.example/secret",), status=b"302 Found"
    )
    assert _error_code(payload) == "http_status_rejected"


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        (_raw(b"ab", headers=(b"Content-Length: 3",)), "premature_eof"),
        (
            _raw(b"abc", headers=(b"Content-Length: 2",)),
            "response_length_mismatch",
        ),
        (
            _raw(b"3\r\nab", headers=(b"Transfer-Encoding: chunked",)),
            "premature_eof",
        ),
        (
            _raw(
                b"2\r\nokX\r\n0\r\n\r\n",
                headers=(b"Transfer-Encoding: chunked",),
            ),
            "http_framing_rejected",
        ),
        (
            _raw(
                b"z\r\n",
                headers=(b"Transfer-Encoding: chunked",),
            ),
            "http_framing_rejected",
        ),
        (
            _raw(
                b"1;=bad\r\na\r\n0\r\n\r\n",
                headers=(b"Transfer-Encoding: chunked",),
            ),
            "http_framing_rejected",
        ),
        (
            _raw(
                b"1\r\na\r\n0\r\nX-Trailer: no\r\n\r\n",
                headers=(b"Transfer-Encoding: chunked",),
            ),
            "trailers_rejected",
        ),
        (
            _raw(
                b"1\r\na\r\n0\r\n\r\nx",
                headers=(b"Transfer-Encoding: chunked",),
            ),
            "response_length_mismatch",
        ),
    ),
)
def test_length_and_chunk_failures_are_exact(payload: bytes, expected: str) -> None:
    assert _error_code(payload) == expected


@pytest.mark.parametrize(
    "payload",
    (
        _raw(b"abc"),
        _raw(
            b"3\r\nabc\r\n0\r\n\r\n",
            headers=(b"Transfer-Encoding: chunked",),
        ),
    ),
)
def test_l_plus_one_is_the_exact_overflow_sentinel(payload: bytes) -> None:
    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        transport.parse_raw_http_response(payload, body_limit=2)
    assert captured.value.code == "body_limit_exceeded"
    assert captured.value.observed_body_bytes == 3


def test_declared_body_over_limit_is_rejected_before_body_read() -> None:
    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        transport.parse_raw_http_response(
            _raw(b"abc", headers=(b"Content-Length: 3",)), body_limit=2
        )
    assert captured.value.code == "body_limit_exceeded"
    assert captured.value.observed_body_bytes == 0


def test_every_private_contact_serialization_is_rejected_in_body() -> None:
    needles = transport.privacy_echo_needles(PRIVATE_CONTACT)
    assert len(needles) >= 5
    for needle in needles:
        with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
            transport.parse_raw_http_response(
                _raw(needle, headers=(f"Content-Length: {len(needle)}".encode(),)),
                body_limit=len(needle),
                private_contact=PRIVATE_CONTACT,
            )
        assert captured.value.code == "privacy_echo"


def test_private_contact_echo_is_rejected_in_raw_headers() -> None:
    payload = _raw(
        b"ok",
        headers=(
            b"Content-Length: 2",
            b"X-Echo: " + PRIVATE_CONTACT.encode("utf-8"),
        ),
    )
    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        transport.parse_raw_http_response(
            payload, body_limit=2, private_contact=PRIVATE_CONTACT
        )
    assert captured.value.code == "privacy_echo"
    assert PRIVATE_CONTACT not in str(captured.value)


def _rehash_receipt(receipt: dict[str, Any]) -> None:
    unsigned = dict(receipt)
    unsigned.pop("transport_receipt_sha256", None)
    receipt["transport_receipt_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _receipt_metadata(tmp_path: Path) -> dict[str, Any]:
    request_url = (
        "https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.gz"
    )
    role = transport.derive_sec_role(request_url)
    return {
        "body_bytes": 2,
        "body_limit_bytes": role.body_limit_bytes,
        "body_sha256": hashlib.sha256(b"ok").hexdigest(),
        "contact_fingerprint_sha256": hashlib.sha256(
            PRIVATE_CONTACT.encode()
        ).hexdigest(),
        "content_encoding": None,
        "declared_content_length": 2,
        "execution_identity": transport.execution_identity(),
        "framing": "content-length",
        "intent_event_sha256": INTENT_SHA256,
        "observed_url": request_url,
        "raw_headers_bytes": 38,
        "raw_headers_sha256": hashlib.sha256(b"safe head").hexdigest(),
        "request_url": request_url,
        "role_class": role.role_class,
        "role_id": role.role_id,
        "status_code": 200,
        "temporary_blob_path": str(tmp_path / "ignored.response.tmp"),
    }


def test_receipt_is_deterministic_redacted_and_self_hashed(tmp_path: Path) -> None:
    metadata = _receipt_metadata(tmp_path)
    first = transport.build_transport_receipt(metadata)
    second = transport.build_transport_receipt(dict(reversed(tuple(metadata.items()))))
    assert first == second
    transport.validate_transport_receipt(first, expected_metadata=metadata)
    round_trip = json.loads(json.dumps(first, sort_keys=True))
    assert round_trip == first
    transport.validate_transport_receipt(
        round_trip, expected_metadata=metadata
    )
    serialized = json.dumps(first, sort_keys=True)
    assert PRIVATE_CONTACT not in serialized
    assert "temporary_blob_path" not in serialized
    assert first["http"]["requested_url"] == first["http"]["observed_url"]
    assert first["http"]["status_code"] == 200
    assert first["deadline"]["duration_milliseconds"] == 30_000
    assert first["privacy_scan"]["forms"] == list(transport.PRIVACY_SCAN_FORMS)
    assert first["transport_capability"]["retries_allowed"] is False
    assert first["role"] == {
        "body_limit_bytes": transport.MASTER_COMPRESSED_BODY_LIMIT,
        "role_class": "quarterly_master",
    }
    assert first["journal_binding"] == {
        "intent_event_sha256": INTENT_SHA256,
        "role_id": "master/2024/QTR1",
    }
    assert first["execution_identity"] == transport.execution_identity()

    tampered = json.loads(json.dumps(first))
    tampered["body"]["byte_length"] = 3
    _rehash_receipt(tampered)
    with pytest.raises(transport.SecGemmaLeanV37TransportError):
        transport.validate_transport_receipt(tampered, expected_metadata=metadata)


@pytest.mark.parametrize(
    "mutation",
    (
        "root_extra",
        "body_missing",
        "body_hash",
        "body_length",
        "body_length_float",
        "header_extra",
        "header_hash",
        "header_length",
        "header_length_float",
        "deadline_extra",
        "http_extra",
        "http_url",
        "http_status",
        "http_status_float",
        "http_encoding",
        "http_framing",
        "http_declared",
        "privacy_body_scan",
        "privacy_bool_int",
        "privacy_header_scan",
        "privacy_forms",
        "privacy_fingerprint",
        "privacy_extra",
        "deadline_duration",
        "deadline_float",
        "deadline_scope",
        "capability_retry",
        "capability_false_int",
        "capability_source",
        "capability_extra",
        "execution_source",
        "execution_extra",
        "role_class",
        "role_limit",
        "role_extra",
        "journal_role",
        "journal_intent",
        "journal_extra",
        "schema",
    ),
)
def test_exact_receipt_rejects_hostile_recomputed_mutations(
    tmp_path: Path, mutation: str
) -> None:
    metadata = _receipt_metadata(tmp_path)
    receipt = transport.build_transport_receipt(metadata)
    hostile = copy.deepcopy(receipt)
    if mutation == "root_extra":
        hostile["extra"] = True
    elif mutation == "body_missing":
        del hostile["body"]["sha256"]
    elif mutation == "body_hash":
        hostile["body"]["sha256"] = "2" * 64
    elif mutation == "body_length":
        hostile["body"]["byte_length"] = 1
    elif mutation == "body_length_float":
        hostile["body"]["byte_length"] = 2.0
    elif mutation == "header_extra":
        hostile["raw_response_headers"]["value"] = "forbidden"
    elif mutation == "header_hash":
        hostile["raw_response_headers"]["sha256"] = "3" * 64
    elif mutation == "header_length":
        hostile["raw_response_headers"]["byte_length"] = 39
    elif mutation == "header_length_float":
        hostile["raw_response_headers"]["byte_length"] = 38.0
    elif mutation == "deadline_extra":
        hostile["deadline"]["extra"] = True
    elif mutation == "http_extra":
        hostile["http"]["extra"] = True
    elif mutation == "http_url":
        hostile["http"]["observed_url"] = (
            "https://data.sec.gov/submissions/CIK0000320193.json"
        )
    elif mutation == "http_status":
        hostile["http"]["status_code"] = 201
    elif mutation == "http_status_float":
        hostile["http"]["status_code"] = 200.0
    elif mutation == "http_encoding":
        hostile["http"]["content_encoding"] = "gzip"
    elif mutation == "http_framing":
        hostile["http"]["framing_mode"] = "eof"
    elif mutation == "http_declared":
        hostile["http"]["declared_content_length"] = 1
    elif mutation == "privacy_body_scan":
        hostile["privacy_scan"]["complete_body_scanned"] = False
    elif mutation == "privacy_bool_int":
        hostile["privacy_scan"]["complete_body_scanned"] = 1
    elif mutation == "privacy_header_scan":
        hostile["privacy_scan"]["raw_response_headers_scanned"] = False
    elif mutation == "privacy_forms":
        hostile["privacy_scan"]["forms"] = ["utf8"]
    elif mutation == "privacy_fingerprint":
        hostile["privacy_scan"]["contact_fingerprint_sha256"] = "4" * 64
    elif mutation == "privacy_extra":
        hostile["privacy_scan"]["extra"] = True
    elif mutation == "deadline_duration":
        hostile["deadline"]["duration_milliseconds"] = 30_001
    elif mutation == "deadline_float":
        hostile["deadline"]["duration_milliseconds"] = 30_000.0
    elif mutation == "deadline_scope":
        hostile["deadline"]["scope"] = "body_only"
    elif mutation == "capability_retry":
        hostile["transport_capability"]["retries_allowed"] = True
    elif mutation == "capability_false_int":
        hostile["transport_capability"]["retries_allowed"] = 0
    elif mutation == "capability_source":
        hostile["transport_capability"]["transport_source_sha256"] = "5" * 64
    elif mutation == "capability_extra":
        hostile["transport_capability"]["extra"] = True
    elif mutation == "execution_source":
        hostile["execution_identity"]["transport_source_sha256"] = "6" * 64
    elif mutation == "execution_extra":
        hostile["execution_identity"]["extra"] = True
    elif mutation == "role_class":
        hostile["role"]["role_class"] = "complete_submission"
    elif mutation == "role_limit":
        hostile["role"]["body_limit_bytes"] = (
            transport.COMPLETE_SUBMISSION_BODY_LIMIT
        )
    elif mutation == "role_extra":
        hostile["role"]["extra"] = True
    elif mutation == "journal_role":
        hostile["journal_binding"]["role_id"] = "submissions/main"
    elif mutation == "journal_intent":
        hostile["journal_binding"]["intent_event_sha256"] = "7" * 64
    elif mutation == "journal_extra":
        hostile["journal_binding"]["extra"] = True
    elif mutation == "schema":
        hostile["schema_version"] = "changed"
    else:
        raise AssertionError(mutation)
    _rehash_receipt(hostile)
    with pytest.raises(transport.SecGemmaLeanV37TransportError):
        transport.validate_transport_receipt(
            hostile, expected_metadata=metadata
        )


def test_default_tls_rejects_a_self_signed_loopback_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    certificate = tmp_path / "localhost-cert.pem"
    private_key = tmp_path / "localhost-key.pem"
    _write_pem_from_der_base64(
        certificate,
        label="CERTIFICATE",
        der_base64=_SELF_SIGNED_CERT_DER_BASE64,
    )
    _write_pem_from_der_base64(
        private_key,
        label="RSA PRIVATE KEY",
        der_base64=_SELF_SIGNED_KEY_DER_BASE64,
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certificate, private_key)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(3.0)
    port = listener.getsockname()[1]
    stopped = threading.Event()

    def serve() -> None:
        try:
            connection, _address = listener.accept()
            with connection:
                try:
                    with server_context.wrap_socket(connection, server_side=True):
                        pass
                except ssl.SSLError:
                    pass
        except (OSError, TimeoutError):
            pass
        finally:
            listener.close()
            stopped.set()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setattr(
        transport.socket,
        "getaddrinfo",
        lambda host, requested_port, **kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", port),
            )
        ],
    )
    try:
        with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
            transport._connect_default_tls(
                "localhost", deadline=time.monotonic() + 3.0
            )
        assert captured.value.code == "certificate_verification_failed"
        assert captured.value.__cause__ is None
    finally:
        listener.close()
        thread.join(timeout=3.0)
    assert stopped.is_set()
    assert not thread.is_alive()


def test_tls_socket_disables_ragged_eof_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class PlainSocket:
        def settimeout(self, value: float) -> None:
            calls.setdefault("timeouts", []).append(value)

        def connect(self, address: tuple[str, int]) -> None:
            calls["address"] = address

        def close(self) -> None:
            calls["plain_closed"] = True

    class WrappedSocket:
        def __init__(self, context: Any) -> None:
            self.context = context

        def close(self) -> None:
            calls["wrapped_closed"] = True

    class Context:
        verify_mode = ssl.CERT_REQUIRED
        check_hostname = True

        def wrap_socket(
            self,
            plain: Any,
            *,
            server_hostname: str,
            suppress_ragged_eofs: bool,
        ) -> WrappedSocket:
            calls["plain"] = plain
            calls["server_hostname"] = server_hostname
            calls["suppress_ragged_eofs"] = suppress_ragged_eofs
            return WrappedSocket(self)

    plain = PlainSocket()
    context = Context()
    monkeypatch.setattr(
        transport.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", 443),
            )
        ],
    )
    monkeypatch.setattr(transport.socket, "socket", lambda *_args: plain)
    monkeypatch.setattr(transport, "_default_tls_context", lambda: context)

    wrapped = transport._connect_default_tls(
        "www.sec.gov", deadline=time.monotonic() + 3.0
    )

    assert isinstance(wrapped, WrappedSocket)
    assert calls["server_hostname"] == "www.sec.gov"
    assert calls["suppress_ragged_eofs"] is False
    assert calls["address"] == ("127.0.0.1", 443)


def test_production_source_has_no_insecure_or_custom_ca_path() -> None:
    source = inspect.getsource(transport)
    assert source.count("ssl.create_default_context()") == 1
    assert "runpy." not in transport._WORKER_BOOTSTRAP
    assert transport._WORKER_BOOTSTRAP.count("source.read_bytes()") == 1
    assert "hashlib.sha256(source_bytes).hexdigest()" in transport._WORKER_BOOTSTRAP
    assert "compile(source_bytes" in transport._WORKER_BOOTSTRAP
    assert "suppress_ragged_eofs=False" in source
    assert ".pyc" not in transport._WORKER_BOOTSTRAP
    risky_header = "BEGIN RSA " + "PRIVATE KEY"
    assert risky_header not in Path(__file__).read_text(encoding="utf-8")
    for forbidden in (
        "ssl.CERT_NONE",
        "_create_unverified_context",
        "load_verify_locations(",
        "load_default_certs(",
        "cafile=",
        "capath=",
        "cadata=",
        "check_hostname = False",
        "ProxyHandler",
        "urllib.request",
        "requests.",
    ):
        assert forbidden not in source


def test_parent_launch_unit_injection_keeps_contact_out_of_argv_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fake child inspects launch inputs; it is not malicious-parent proof."""
    captured: dict[str, Any] = {}

    class FailedWorker:
        returncode = 2

        def communicate(self, *, input: bytes, timeout: float):
            captured["input"] = input
            return (
                json.dumps(
                    {
                        "code": "transport_error",
                        "observed_body_bytes": 0,
                        "schema_version": transport.TRANSPORT_SCHEMA_VERSION,
                        "status": "error",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii"),
                b"",
            )

        def poll(self) -> int:
            return self.returncode

    def fake_popen(command: tuple[str, ...], **kwargs: Any) -> FailedWorker:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return FailedWorker()

    monkeypatch.setattr(transport.subprocess, "Popen", fake_popen)
    client = transport.StrictSecTransport(
        private_contact=PRIVATE_CONTACT,
        temporary_directory=tmp_path.resolve(),
    )
    with pytest.raises(transport.SecGemmaLeanV37TransportError):
        client.fetch(
            "https://data.sec.gov/submissions/CIK0000320193.json",
            role_id="submissions/main",
            intent_event_sha256=INTENT_SHA256,
            body_limit=transport.MAIN_SUBMISSIONS_BODY_LIMIT,
        )
    assert PRIVATE_CONTACT not in repr(captured["command"])
    assert PRIVATE_CONTACT not in repr(captured["environment"])
    assert "-B" in captured["command"]
    assert captured["command"].count("-B") == 1
    identity = transport.execution_identity()
    assert captured["command"][-2:] == (
        identity["transport_source_sha256"],
        identity["interpreter_executable_sha256"],
    )
    worker_request = json.loads(captured["input"])
    assert set(worker_request) == {
        "body_limit",
        "intent_event_sha256",
        "output_path",
        "private_contact",
        "request_url",
        "role_id",
    }
    assert worker_request["private_contact"] == PRIVATE_CONTACT
    assert worker_request["role_id"] == "submissions/main"
    assert worker_request["intent_event_sha256"] == INTENT_SHA256
    assert worker_request["body_limit"] == transport.MAIN_SUBMISSIONS_BODY_LIMIT
    assert "proxy" not in " ".join(captured["environment"]).casefold()
    assert "cert" not in " ".join(captured["environment"]).casefold()


@pytest.mark.parametrize(
    ("role_id", "intent", "body_limit", "expected_code"),
    (
        (
            "master/2024/QTR2",
            INTENT_SHA256,
            transport.MASTER_COMPRESSED_BODY_LIMIT,
            "url_mismatch",
        ),
        (
            "master/2024/QTR1",
            "A" * 64,
            transport.MASTER_COMPRESSED_BODY_LIMIT,
            "url_mismatch",
        ),
        (
            "master/2024/QTR1",
            "1" * 63,
            transport.MASTER_COMPRESSED_BODY_LIMIT,
            "url_mismatch",
        ),
        (
            "master/2024/QTR1",
            INTENT_SHA256,
            transport.COMPLETE_SUBMISSION_BODY_LIMIT,
            "body_limit_exceeded",
        ),
    ),
)
def test_request_binding_and_derived_limit_mismatch_reject_before_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role_id: str,
    intent: str,
    body_limit: int,
    expected_code: str,
) -> None:
    monkeypatch.setattr(
        transport.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("child must not start")
        ),
    )
    client = transport.StrictSecTransport(
        private_contact=PRIVATE_CONTACT,
        temporary_directory=tmp_path.resolve(),
    )
    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        client.fetch(
            "https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.gz",
            role_id=role_id,
            intent_event_sha256=intent,
            body_limit=body_limit,
        )
    assert captured.value.code == expected_code


def test_parent_validation_unit_injected_success_requires_every_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit injection checks parent validation; it is not malicious-parent proof."""
    body = b"ok"
    raw_head = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n"

    class SuccessfulWorker:
        returncode = 0

        def communicate(self, *, input: bytes, timeout: float):
            request = json.loads(input)
            output_path = Path(request["output_path"])
            output_path.write_bytes(body)
            fingerprint = hashlib.sha256(
                request["private_contact"].encode("utf-8")
            ).hexdigest()
            return (
                json.dumps(
                    {
                        "metadata": {
                            "body_bytes": len(body),
                            "body_limit_bytes": request["body_limit"],
                            "body_sha256": hashlib.sha256(body).hexdigest(),
                            "contact_fingerprint_sha256": fingerprint,
                            "content_encoding": None,
                            "declared_content_length": len(body),
                            "execution_identity": transport.execution_identity(),
                            "framing": "content-length",
                            "intent_event_sha256": request[
                                "intent_event_sha256"
                            ],
                            "observed_url": request["request_url"],
                            "raw_headers_bytes": len(raw_head),
                            "raw_headers_sha256": hashlib.sha256(
                                raw_head
                            ).hexdigest(),
                            "request_url": request["request_url"],
                            "role_class": transport.derive_sec_role(
                                request["request_url"]
                            ).role_class,
                            "role_id": request["role_id"],
                            "status_code": 200,
                            "temporary_blob_path": str(output_path),
                        },
                        "schema_version": transport.TRANSPORT_SCHEMA_VERSION,
                        "status": "ok",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii"),
                b"",
            )

        def poll(self) -> int:
            return self.returncode

    monkeypatch.setattr(
        transport.subprocess,
        "Popen",
        lambda command, **kwargs: SuccessfulWorker(),
    )
    client = transport.StrictSecTransport(
        private_contact=PRIVATE_CONTACT,
        temporary_directory=tmp_path.resolve(),
    )
    result = client.fetch(
        "https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.gz",
        role_id="master/2024/QTR1",
        intent_event_sha256=INTENT_SHA256,
        body_limit=transport.MASTER_COMPRESSED_BODY_LIMIT,
    )
    assert result.temporary_blob_path.read_bytes() == body
    assert result.body_sha256 == hashlib.sha256(body).hexdigest()
    assert result.transport_receipt_sha256 == result.transport_receipt[
        "transport_receipt_sha256"
    ]
    metadata = {
        "body_bytes": result.body_bytes,
        "body_limit_bytes": result.body_limit_bytes,
        "body_sha256": result.body_sha256,
        "contact_fingerprint_sha256": result.contact_fingerprint_sha256,
        "content_encoding": result.content_encoding,
        "declared_content_length": result.declared_content_length,
        "execution_identity": dict(result.execution_identity),
        "framing": result.framing,
        "intent_event_sha256": result.intent_event_sha256,
        "observed_url": result.observed_url,
        "raw_headers_bytes": result.raw_headers_bytes,
        "raw_headers_sha256": result.raw_headers_sha256,
        "request_url": result.request_url,
        "role_class": result.role_class,
        "role_id": result.role_id,
        "status_code": result.status_code,
        "temporary_blob_path": str(result.temporary_blob_path),
    }
    transport.validate_transport_receipt(
        result.transport_receipt, expected_metadata=metadata
    )
    assert PRIVATE_CONTACT not in json.dumps(result.transport_receipt)


@pytest.mark.parametrize(
    "corruption",
    (
        "role_id",
        "role_class",
        "body_limit",
        "intent",
        "request_url",
        "observed_url",
        "source_identity",
        "bootstrap_identity",
        "interpreter_identity",
    ),
)
def test_parent_validation_unit_injection_rejects_worker_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    """A fake child tests parent checks, not cryptographic malicious-parent proof."""

    body = b"ok"
    raw_head = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n"

    class DriftedWorker:
        returncode = 0

        def communicate(self, *, input: bytes, timeout: float):
            request = json.loads(input)
            output_path = Path(request["output_path"])
            output_path.write_bytes(body)
            role = transport.derive_sec_role(request["request_url"])
            identity = transport.execution_identity()
            metadata = {
                "body_bytes": len(body),
                "body_limit_bytes": request["body_limit"],
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "contact_fingerprint_sha256": hashlib.sha256(
                    request["private_contact"].encode("utf-8")
                ).hexdigest(),
                "content_encoding": None,
                "declared_content_length": len(body),
                "execution_identity": identity,
                "framing": "content-length",
                "intent_event_sha256": request["intent_event_sha256"],
                "observed_url": request["request_url"],
                "raw_headers_bytes": len(raw_head),
                "raw_headers_sha256": hashlib.sha256(raw_head).hexdigest(),
                "request_url": request["request_url"],
                "role_class": role.role_class,
                "role_id": request["role_id"],
                "status_code": 200,
                "temporary_blob_path": str(output_path),
            }
            if corruption == "role_id":
                metadata["role_id"] = "master/2024/QTR2"
            elif corruption == "role_class":
                metadata["role_class"] = "complete_submission"
            elif corruption == "body_limit":
                metadata["body_limit_bytes"] = (
                    transport.COMPLETE_SUBMISSION_BODY_LIMIT
                )
            elif corruption == "intent":
                metadata["intent_event_sha256"] = "8" * 64
            elif corruption == "request_url":
                metadata["request_url"] = (
                    "https://data.sec.gov/submissions/CIK0000320193.json"
                )
            elif corruption == "observed_url":
                metadata["observed_url"] = (
                    "https://data.sec.gov/submissions/CIK0000320193.json"
                )
            elif corruption == "source_identity":
                metadata["execution_identity"] = dict(identity)
                metadata["execution_identity"]["transport_source_sha256"] = (
                    "9" * 64
                )
            elif corruption == "bootstrap_identity":
                metadata["execution_identity"] = dict(identity)
                metadata["execution_identity"]["worker_bootstrap_sha256"] = (
                    "a" * 64
                )
            elif corruption == "interpreter_identity":
                metadata["execution_identity"] = dict(identity)
                metadata["execution_identity"][
                    "interpreter_executable_sha256"
                ] = "b" * 64
            else:
                raise AssertionError(corruption)
            return (
                json.dumps(
                    {
                        "metadata": metadata,
                        "schema_version": transport.TRANSPORT_SCHEMA_VERSION,
                        "status": "ok",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii"),
                b"",
            )

        def poll(self) -> int:
            return self.returncode

    monkeypatch.setattr(
        transport.subprocess,
        "Popen",
        lambda command, **kwargs: DriftedWorker(),
    )
    client = transport.StrictSecTransport(
        private_contact=PRIVATE_CONTACT,
        temporary_directory=tmp_path.resolve(),
    )
    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        client.fetch(
            "https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.gz",
            role_id="master/2024/QTR1",
            intent_event_sha256=INTENT_SHA256,
            body_limit=transport.MASTER_COMPRESSED_BODY_LIMIT,
        )
    assert captured.value.code == "transport_error"
    assert list(tmp_path.iterdir()) == []


def test_real_isolated_worker_bootstrap_fails_before_network_on_occupied_path(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / ("b" * 32 + ".response.tmp")
    occupied.write_bytes(b"must remain")
    repo_root = Path(transport.__file__).resolve(strict=True).parents[1]
    identity = transport.execution_identity()
    command = (
        str(Path(sys.executable).resolve(strict=True)),
        "-I",
        "-S",
        "-B",
        "-c",
        transport._WORKER_BOOTSTRAP,
        str(repo_root),
        transport._WORKER_MODULE,
        transport._WORKER_FLAG,
        identity["transport_source_sha256"],
        identity["interpreter_executable_sha256"],
    )
    request = json.dumps(
        {
            "body_limit": transport.MAIN_SUBMISSIONS_BODY_LIMIT,
            "intent_event_sha256": INTENT_SHA256,
            "output_path": str(occupied),
            "private_contact": PRIVATE_CONTACT,
            "request_url": "https://data.sec.gov/submissions/CIK0000320193.json",
            "role_id": "submissions/main",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=(
            int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if sys.platform == "win32"
            else 0
        ),
        cwd=str(repo_root),
        env=transport._sanitized_worker_environment(),
    )
    try:
        stdout, _stderr = process.communicate(input=request, timeout=5.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)
    decoded = json.loads(stdout)
    assert process.returncode == 2
    assert decoded["status"] == "error"
    assert decoded["code"] == "blob_persistence_failed"
    assert PRIVATE_CONTACT not in stdout.decode("ascii")
    assert occupied.read_bytes() == b"must remain"


def test_parent_deadline_unit_injection_kills_and_reaps_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injected sleeper verifies parent cleanup, not malicious-parent proof."""
    real_popen = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def sleeping_popen(_command: tuple[str, ...], **_kwargs: Any):
        child = real_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=(
                int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if sys.platform == "win32"
                else 0
            ),
        )
        children.append(child)
        return child

    monkeypatch.setattr(transport, "REQUEST_DEADLINE_SECONDS", 0.15)
    monkeypatch.setattr(transport.subprocess, "Popen", sleeping_popen)
    client = transport.StrictSecTransport(
        private_contact=PRIVATE_CONTACT,
        temporary_directory=tmp_path.resolve(),
    )
    started = time.monotonic()
    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        client.fetch(
            "https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.gz",
            role_id="master/2024/QTR1",
            intent_event_sha256=INTENT_SHA256,
            body_limit=transport.MASTER_COMPRESSED_BODY_LIMIT,
        )
    elapsed = time.monotonic() - started
    assert captured.value.code == "hard_timeout"
    assert elapsed < 3.0
    assert len(children) == 1
    assert children[0].poll() is not None
    assert list(tmp_path.iterdir()) == []


def test_worker_termination_timeout_rekills_and_unconditionally_reaps() -> None:
    class DelayedProcess:
        returncode: int | None = None

        def __init__(self) -> None:
            self.kill_count = 0
            self.wait_timeouts: list[float | None] = []

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.kill_count += 1

        def wait(self, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = -9
            return self.returncode

    process = DelayedProcess()
    transport._terminate_worker(process)
    assert process.kill_count == 2
    assert process.wait_timeouts == [2.0, None]
    assert process.poll() == -9


def test_worker_termination_does_not_swallow_reap_failure() -> None:
    class BrokenProcess:
        def poll(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            raise OSError("injected reap failure")

    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        transport._terminate_worker(BrokenProcess())
    assert captured.value.code == "transport_error"
    assert captured.value.__cause__ is None


def test_fsynced_blob_is_exclusive_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ("a" * 32 + ".response.tmp")
    fsynced: list[int] = []
    real_fsync = transport.os.fsync
    monkeypatch.setattr(
        transport.os,
        "fsync",
        lambda descriptor: (fsynced.append(descriptor), real_fsync(descriptor))[1],
    )
    transport._write_fsynced_blob(path, b"exact body")
    assert path.read_bytes() == b"exact body"
    assert fsynced
    with pytest.raises(transport.SecGemmaLeanV37TransportError) as captured:
        transport._write_fsynced_blob(path, b"replacement")
    assert captured.value.code == "blob_persistence_failed"
    assert path.read_bytes() == b"exact body"
