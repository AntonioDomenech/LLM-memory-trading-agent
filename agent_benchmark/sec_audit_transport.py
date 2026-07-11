"""Bounded, cache-first SEC transport with no concrete network dependency."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

from .sec_point_in_time import (
    BudgetCounter,
    SecAuditLimitError,
    SecPointInTimeError,
    UserAgentAudit,
    content_sha256,
    validate_sec_user_agent,
)


OFFICIAL_SEC_HOSTS = frozenset({"www.sec.gov", "data.sec.gov"})
MAX_REQUESTS_PER_SECOND = 2.0
MIN_REQUEST_INTERVAL_SECONDS = 1.0 / MAX_REQUESTS_PER_SECOND
MAX_REDIRECTS = 3
MAX_RETRIES = 3
MAX_BACKOFF_SECONDS = 8.0
STREAM_CHUNK_BYTES = 64 * 1024
RETRYABLE_STATUSES = frozenset({429, *range(500, 600)})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class SecAuditTransportError(SecPointInTimeError):
    """A request, redirect, response, or cache violated the SEC audit contract."""


class ResponseLike(Protocol):
    status_code: int
    headers: Mapping[str, Any]

    def iter_content(self, chunk_size: int) -> Any: ...

    def close(self) -> None: ...


class SessionLike(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> ResponseLike: ...


@dataclass(frozen=True)
class ResponseAudit:
    url: str
    status_code: int
    content_type: str
    size_bytes: int
    content_sha256: str
    cache_hit: bool
    user_agent_sha256: str
    network_requests: int
    retries: int
    redirects: int


def canonical_sec_url(url: str) -> str:
    """Return a stable HTTPS SEC URL or reject it before any request occurs."""

    if not isinstance(url, str) or not url:
        raise SecAuditTransportError("SEC URL must be a non-empty string")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise SecAuditTransportError("SEC URL is malformed") from exc
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() != "https"
        or host not in OFFICIAL_SEC_HOSTS
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise SecAuditTransportError("Only official SEC HTTPS URLs are allowed")
    path = parsed.path or "/"
    return urlunsplit(("https", host, path, parsed.query, ""))


def cache_key_for_url(url: str) -> str:
    canonical = canonical_sec_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _private_header_factory(value: str) -> Callable[[], dict[str, str]]:
    # Keeping the raw contact in a closure prevents accidental dataclass/state
    # serialization.  __getstate__, repr, errors, cache, and audits expose only
    # the precomputed hash.
    def headers() -> dict[str, str]:
        return {
            "User-Agent": value,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json,text/plain,*/*",
        }

    return headers


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return str(value)
    return None


def _safe_content_type(headers: Mapping[str, Any]) -> str:
    value = (_header(headers, "Content-Type") or "application/octet-stream").strip()
    if len(value) > 160 or any(ord(char) < 32 or ord(char) > 126 for char in value):
        raise SecAuditTransportError("SEC response Content-Type is unsafe")
    return value


class SecAuditTransport:
    """Fetch official SEC bytes through an injected session and bounded cache."""

    def __init__(
        self,
        *,
        session: SessionLike,
        cache_dir: Path,
        user_agent: str,
        budget: BudgetCounter,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        timeout_seconds: float = 30.0,
        max_retries: int = MAX_RETRIES,
        max_redirects: int = MAX_REDIRECTS,
        allow_cache_reads: bool = True,
    ) -> None:
        if session is None:
            raise SecAuditTransportError("An injected SEC session is required")
        if not 0.0 < float(timeout_seconds) <= 60.0:
            raise SecAuditTransportError("SEC request timeout must be in (0, 60]")
        if isinstance(max_retries, bool) or not 0 <= max_retries <= MAX_RETRIES:
            raise SecAuditTransportError("SEC retry ceiling may only be tightened")
        if isinstance(max_redirects, bool) or not 0 <= max_redirects <= MAX_REDIRECTS:
            raise SecAuditTransportError("SEC redirect ceiling may only be tightened")
        if not isinstance(allow_cache_reads, bool):
            raise SecAuditTransportError("allow_cache_reads must be boolean")
        self._user_agent_audit = validate_sec_user_agent(user_agent)
        self._headers = _private_header_factory(user_agent)
        self._session = session
        self._cache_dir = Path(cache_dir).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._budget = budget
        self._clock = clock
        self._sleep = sleep
        self._timeout = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._max_redirects = int(max_redirects)
        self._allow_cache_reads = allow_cache_reads
        self._last_request_at: float | None = None

    def __repr__(self) -> str:
        return (
            "SecAuditTransport("
            f"cache_dir={str(self._cache_dir)!r}, "
            f"user_agent_sha256={self._user_agent_audit.sha256!r})"
        )

    def __getstate__(self) -> dict[str, Any]:
        return self.safe_state()

    def safe_state(self) -> dict[str, Any]:
        return {
            "cache_dir": str(self._cache_dir),
            "user_agent": asdict(self._user_agent_audit),
            "timeout_seconds": self._timeout,
            "max_retries": self._max_retries,
            "max_redirects": self._max_redirects,
            "allow_cache_reads": self._allow_cache_reads,
            "budget": self._budget.snapshot(),
        }

    @property
    def user_agent_audit(self) -> UserAgentAudit:
        return self._user_agent_audit

    def _cache_paths(self, request_url: str) -> tuple[Path, Path]:
        key = cache_key_for_url(request_url)
        return self._cache_dir / f"{key}.body", self._cache_dir / f"{key}.json"

    def _load_cache(self, request_url: str) -> tuple[bytes, ResponseAudit] | None:
        if not self._allow_cache_reads:
            return None
        body_path, metadata_path = self._cache_paths(request_url)
        if not body_path.exists() and not metadata_path.exists():
            return None
        if not body_path.is_file() or not metadata_path.is_file():
            raise SecAuditTransportError("SEC cache entry is incomplete")
        try:
            cached_size = body_path.stat().st_size
            remaining = self._budget.max_bytes - self._budget.bytes_received
            if cached_size < 0 or cached_size > remaining:
                raise SecAuditLimitError("SEC cached response exceeds remaining byte budget")
            body = body_path.read_bytes()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecAuditTransportError("SEC cache entry is unreadable") from exc
        expected = {
            "request_url",
            "url",
            "status_code",
            "content_type",
            "size_bytes",
            "content_sha256",
            "user_agent_sha256",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected:
            raise SecAuditTransportError("SEC cache metadata schema changed")
        if (
            metadata["request_url"] != request_url
            or metadata["size_bytes"] != len(body)
            or metadata["content_sha256"] != content_sha256(body)
            or not isinstance(metadata["status_code"], int)
            or not 200 <= metadata["status_code"] < 300
            or canonical_sec_url(metadata["url"]) != metadata["url"]
        ):
            raise SecAuditTransportError("SEC cache integrity check failed")
        self._budget.add_bytes(len(body))
        audit = ResponseAudit(
            url=metadata["url"],
            status_code=metadata["status_code"],
            content_type=str(metadata["content_type"]),
            size_bytes=len(body),
            content_sha256=metadata["content_sha256"],
            cache_hit=True,
            user_agent_sha256=self._user_agent_audit.sha256,
            network_requests=0,
            retries=0,
            redirects=0,
        )
        return body, audit

    def _store_cache(
        self,
        request_url: str,
        body: bytes,
        audit: ResponseAudit,
    ) -> None:
        body_path, metadata_path = self._cache_paths(request_url)
        metadata = {
            "request_url": request_url,
            "url": audit.url,
            "status_code": audit.status_code,
            "content_type": audit.content_type,
            "size_bytes": audit.size_bytes,
            "content_sha256": audit.content_sha256,
            "user_agent_sha256": audit.user_agent_sha256,
        }
        body_temp = body_path.with_suffix(".body.tmp")
        metadata_temp = metadata_path.with_suffix(".json.tmp")
        try:
            body_temp.write_bytes(body)
            metadata_temp.write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(body_temp, body_path)
            os.replace(metadata_temp, metadata_path)
        except OSError as exc:
            body_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
            raise SecAuditTransportError("SEC cache write failed") from exc

    def _rate_limited_request(self, url: str) -> ResponseLike:
        now = float(self._clock())
        if not math.isfinite(now):
            raise SecAuditTransportError("SEC transport clock is not finite")
        if self._last_request_at is not None:
            wait = MIN_REQUEST_INTERVAL_SECONDS - (now - self._last_request_at)
            if wait > 0.0:
                self._sleep(wait)
                now = float(self._clock())
            if now < self._last_request_at:
                raise SecAuditTransportError("SEC transport clock moved backwards")
            if now - self._last_request_at < MIN_REQUEST_INTERVAL_SECONDS - 1e-12:
                raise SecAuditTransportError("SEC transport rate-limit sleep was ineffective")
        self._budget.check()
        self._budget.begin_request()
        self._last_request_at = now
        response: ResponseLike | None = None
        try:
            response = self._session.request(
                "GET",
                url,
                headers=self._headers(),
                stream=True,
                allow_redirects=False,
                timeout=self._timeout,
            )
        except Exception:
            # Do not retain a third-party exception that may embed headers.
            pass
        if response is None:
            raise SecAuditTransportError("Injected SEC session request failed") from None
        return response

    def _retry_delay(self, response: ResponseLike, retry_number: int) -> float:
        delay = min(0.5 * (2 ** (retry_number - 1)), MAX_BACKOFF_SECONDS)
        retry_after = _header(response.headers, "Retry-After")
        if retry_after and re_full_nonnegative_number(retry_after):
            delay = max(delay, min(float(retry_after), MAX_BACKOFF_SECONDS))
        return delay

    def _stream_body(self, response: ResponseLike) -> bytes:
        try:
            remaining = self._budget.max_bytes - self._budget.bytes_received
            content_length = _header(response.headers, "Content-Length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError as exc:
                    raise SecAuditTransportError("SEC Content-Length is invalid") from exc
                if declared < 0 or declared > remaining:
                    raise SecAuditTransportError("SEC response exceeds remaining byte budget")
                if declared == 0:
                    return b""
            if remaining <= 0:
                raise SecAuditTransportError("SEC response exceeds remaining byte budget")
            chunks: list[bytes] = []
            iterator = response.iter_content(chunk_size=min(STREAM_CHUNK_BYTES, remaining))
            for chunk in iterator:
                if not chunk:
                    continue
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise SecAuditTransportError("SEC response chunk is not bytes")
                data = bytes(chunk)
                self._budget.add_bytes(len(data))
                chunks.append(data)
            return b"".join(chunks)
        finally:
            response.close()

    def fetch(self, url: str) -> tuple[bytes, ResponseAudit]:
        request_url = canonical_sec_url(url)
        cached = self._load_cache(request_url)
        if cached is not None:
            return cached
        current_url = request_url
        network_requests = retries = redirects = 0
        while True:
            response = self._rate_limited_request(current_url)
            network_requests += 1
            try:
                status = int(response.status_code)
            except (TypeError, ValueError):
                response.close()
                raise SecAuditTransportError("SEC response status is invalid") from None
            if status in REDIRECT_STATUSES:
                location = _header(response.headers, "Location")
                response.close()
                if location is None:
                    raise SecAuditTransportError("SEC redirect has no Location")
                if redirects >= self._max_redirects:
                    raise SecAuditTransportError("SEC redirect ceiling exceeded")
                current_url = canonical_sec_url(urljoin(current_url, location))
                redirects += 1
                continue
            if status in RETRYABLE_STATUSES:
                delay = self._retry_delay(response, retries + 1)
                response.close()
                if retries >= self._max_retries:
                    raise SecAuditTransportError(
                        f"SEC retry ceiling exceeded after status {status}"
                    )
                retries += 1
                self._sleep(delay)
                continue
            if not 200 <= status < 300:
                response.close()
                raise SecAuditTransportError(f"SEC request failed with status {status}")
            try:
                content_type = _safe_content_type(response.headers)
                body = self._stream_body(response)
            except Exception:
                response.close()
                raise
            audit = ResponseAudit(
                url=current_url,
                status_code=status,
                content_type=content_type,
                size_bytes=len(body),
                content_sha256=content_sha256(body),
                cache_hit=False,
                user_agent_sha256=self._user_agent_audit.sha256,
                network_requests=network_requests,
                retries=retries,
                redirects=redirects,
            )
            self._store_cache(request_url, body, audit)
            return body, audit


def re_full_nonnegative_number(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return bool(math.isfinite(number) and number >= 0.0)


__all__ = [
    "MAX_BACKOFF_SECONDS",
    "MAX_REDIRECTS",
    "MAX_REQUESTS_PER_SECOND",
    "MAX_RETRIES",
    "OFFICIAL_SEC_HOSTS",
    "ResponseAudit",
    "SecAuditTransport",
    "SecAuditTransportError",
    "cache_key_for_url",
    "canonical_sec_url",
]
