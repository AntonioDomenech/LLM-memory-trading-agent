from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from agent_benchmark.sec_audit_transport import (
    MAX_REDIRECTS,
    SecAuditTransport,
    SecAuditTransportError,
    cache_key_for_url,
    canonical_sec_url,
)
from agent_benchmark.sec_point_in_time import (
    BudgetCounter,
    SecAuditLimitError,
)


PRIVATE_USER_AGENT = "Antonio SEC Audit private-contact@antoniodomenech.dev"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(
        self,
        status: int,
        *,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status_code = status
        self.headers = headers or {}
        self.chunks = chunks or []
        self.chunk_sizes: list[int] = []
        self.closed = False

    def iter_content(self, chunk_size: int):
        self.chunk_sizes.append(chunk_size)
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(
        self,
        responses: list[FakeResponse] | None = None,
        *,
        clock: FakeClock | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.clock = clock
        self.failure = failure
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "time": self.clock() if self.clock else None,
                **kwargs,
            }
        )
        if self.failure is not None:
            raise self.failure
        if not self.responses:
            raise AssertionError("unexpected network call")
        return self.responses.pop(0)


def _transport(
    cache_dir: Path,
    responses: list[FakeResponse] | None = None,
    *,
    max_bytes: int = 1024,
    max_requests: int = 100,
    session: FakeSession | None = None,
    max_retries: int = 3,
    max_redirects: int = 3,
) -> tuple[SecAuditTransport, FakeSession, BudgetCounter, FakeClock]:
    clock = FakeClock()
    budget = BudgetCounter(
        clock=clock,
        max_bytes=max_bytes,
        max_requests=max_requests,
    )
    resolved_session = session or FakeSession(responses, clock=clock)
    resolved_session.clock = clock
    client = SecAuditTransport(
        session=resolved_session,
        cache_dir=cache_dir,
        user_agent=PRIVATE_USER_AGENT,
        budget=budget,
        clock=clock,
        sleep=clock.sleep,
        max_retries=max_retries,
        max_redirects=max_redirects,
    )
    return client, resolved_session, budget, clock


@pytest.mark.parametrize(
    "url",
    (
        "http://www.sec.gov/file.txt",
        "https://evil.example/file.txt",
        "https://www.sec.gov:444/file.txt",
        "https://person@www.sec.gov/file.txt",
        "https://www.sec.gov/file.txt#fragment",
    ),
)
def test_nonofficial_or_non_https_urls_are_rejected_before_request(
    tmp_path: Path, url: str
) -> None:
    client, session, budget, _ = _transport(tmp_path)
    with pytest.raises(SecAuditTransportError, match="official SEC HTTPS"):
        client.fetch(url)
    assert session.calls == []
    assert budget.requests == 0


def test_private_user_agent_is_used_but_never_exposed_in_state_or_errors(
    tmp_path: Path,
) -> None:
    leaked_failure = RuntimeError(f"failed with {PRIVATE_USER_AGENT}")
    session = FakeSession(failure=leaked_failure)
    client, session, _, _ = _transport(tmp_path, session=session)
    safe = repr(client) + json.dumps(client.safe_state(), sort_keys=True)
    safe += json.dumps(client.__getstate__(), sort_keys=True)
    assert PRIVATE_USER_AGENT not in safe
    assert "private-contact" not in safe
    assert "antoniodomenech.dev" not in safe

    with pytest.raises(SecAuditTransportError) as captured:
        client.fetch("https://www.sec.gov/test.json")
    assert PRIVATE_USER_AGENT not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert session.calls[0]["headers"]["User-Agent"] == PRIVATE_USER_AGENT


def test_success_streams_caches_and_returns_only_safe_audit_metadata(
    tmp_path: Path,
) -> None:
    body = b'{"ok":true}'
    response = FakeResponse(
        200,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
        },
        chunks=[body[:4], body[4:]],
    )
    client, session, budget, _ = _transport(tmp_path, [response])
    url = "https://data.sec.gov/submissions/CIK0000320193.json"
    first_body, first = client.fetch(url)
    assert first_body == body
    assert first.url == url
    assert first.status_code == 200
    assert first.content_type == "application/json; charset=utf-8"
    assert first.size_bytes == len(body)
    assert first.content_sha256.startswith("sha256:")
    assert first.cache_hit is False
    assert first.network_requests == 1
    assert first.retries == first.redirects == 0
    assert PRIVATE_USER_AGENT not in json.dumps(asdict(first), sort_keys=True)
    assert response.closed is True
    assert budget.requests == 1
    assert budget.bytes_received == len(body)

    second_body, second = client.fetch(url)
    assert second_body == body
    assert second.cache_hit is True
    assert second.network_requests == second.retries == second.redirects == 0
    assert budget.bytes_received == 2 * len(body)
    assert len(session.calls) == 1
    key = cache_key_for_url(url)
    assert (tmp_path / f"{key}.body").read_bytes() == body
    metadata = (tmp_path / f"{key}.json").read_text()
    assert PRIVATE_USER_AGENT not in metadata
    assert "private-contact" not in metadata
    assert "user_agent_sha256" in metadata


def test_manual_redirect_stays_official_and_is_rate_limited(tmp_path: Path) -> None:
    redirect = FakeResponse(
        302,
        headers={"Location": "https://data.sec.gov/submissions/aapl.json"},
    )
    final = FakeResponse(
        200,
        headers={"Content-Type": "application/json", "Content-Length": "2"},
        chunks=[b"{}"],
    )
    client, session, budget, _ = _transport(tmp_path, [redirect, final])
    _, audit = client.fetch("https://www.sec.gov/start")
    assert audit.url == "https://data.sec.gov/submissions/aapl.json"
    assert audit.redirects == 1
    assert audit.network_requests == 2
    assert audit.retries == 0
    assert budget.requests == 2
    assert session.calls[1]["time"] - session.calls[0]["time"] >= 0.5
    assert all(call["allow_redirects"] is False for call in session.calls)
    assert redirect.closed and final.closed


@pytest.mark.parametrize(
    "location",
    ("http://www.sec.gov/downgrade", "https://attacker.example/escape"),
)
def test_redirect_downgrades_and_non_sec_hosts_are_rejected(
    tmp_path: Path, location: str
) -> None:
    redirect = FakeResponse(302, headers={"Location": location})
    client, session, budget, _ = _transport(tmp_path, [redirect])
    with pytest.raises(SecAuditTransportError, match="official SEC HTTPS"):
        client.fetch("https://www.sec.gov/start")
    assert len(session.calls) == 1
    assert budget.requests == 1
    assert redirect.closed


def test_redirect_ceiling_counts_every_network_attempt(tmp_path: Path) -> None:
    responses = [
        FakeResponse(302, headers={"Location": f"/hop-{position + 1}"})
        for position in range(MAX_REDIRECTS + 1)
    ]
    client, session, budget, _ = _transport(tmp_path, responses)
    with pytest.raises(SecAuditTransportError, match="redirect ceiling"):
        client.fetch("https://www.sec.gov/start")
    assert len(session.calls) == MAX_REDIRECTS + 1
    assert budget.requests == MAX_REDIRECTS + 1
    assert all(response.closed for response in responses)


def test_429_and_5xx_retries_are_bounded_backed_off_and_counted(
    tmp_path: Path,
) -> None:
    responses = [
        FakeResponse(429, headers={"Retry-After": "0"}),
        FakeResponse(503),
        FakeResponse(
            200,
            headers={"Content-Type": "text/plain", "Content-Length": "2"},
            chunks=[b"ok"],
        ),
    ]
    client, session, budget, clock = _transport(tmp_path, responses)
    body, audit = client.fetch("https://www.sec.gov/retry")
    assert body == b"ok"
    assert audit.retries == 2
    assert audit.network_requests == 3
    assert audit.redirects == 0
    assert budget.requests == 3
    assert clock.sleeps[:2] == [0.5, 1.0]
    assert [call["time"] for call in session.calls] == [0.0, 0.5, 1.5]


def test_retry_ceiling_fails_closed_and_counts_final_attempt(tmp_path: Path) -> None:
    responses = [FakeResponse(503) for _ in range(4)]
    client, session, budget, _ = _transport(tmp_path, responses)
    with pytest.raises(SecAuditTransportError, match="retry ceiling"):
        client.fetch("https://www.sec.gov/retry-exhausted")
    assert len(session.calls) == 4
    assert budget.requests == 4
    assert all(response.closed for response in responses)


def test_streaming_budget_stops_before_storing_overrun_chunk(tmp_path: Path) -> None:
    response = FakeResponse(
        200,
        headers={"Content-Type": "application/octet-stream"},
        chunks=[b"abc", b"def"],
    )
    client, _, budget, _ = _transport(tmp_path, [response], max_bytes=5)
    with pytest.raises(SecAuditLimitError, match="byte ceiling"):
        client.fetch("https://www.sec.gov/too-large")
    assert budget.bytes_received == 3
    assert response.closed
    assert not list(tmp_path.glob("*.body"))
    assert not list(tmp_path.glob("*.json"))


def test_declared_overrun_is_rejected_before_streaming(tmp_path: Path) -> None:
    response = FakeResponse(
        200,
        headers={"Content-Type": "text/plain", "Content-Length": "6"},
        chunks=[b"abcdef"],
    )
    client, _, budget, _ = _transport(tmp_path, [response], max_bytes=5)
    with pytest.raises(SecAuditTransportError, match="remaining byte budget"):
        client.fetch("https://www.sec.gov/declared-too-large")
    assert budget.bytes_received == 0
    assert response.chunk_sizes == []
    assert response.closed


def test_corrupt_cache_fails_closed_without_network(tmp_path: Path) -> None:
    url = canonical_sec_url("https://www.sec.gov/cached")
    key = cache_key_for_url(url)
    (tmp_path / f"{key}.body").write_bytes(b"tampered")
    (tmp_path / f"{key}.json").write_text("{}")
    client, session, budget, _ = _transport(tmp_path)
    with pytest.raises(SecAuditTransportError, match="schema changed"):
        client.fetch(url)
    assert session.calls == []
    assert budget.requests == 0
