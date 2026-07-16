"""Opaque v2.1 external publication through one non-force annotated Git tag.

The publisher is intentionally narrow.  It accepts one already sealed artifact
hash, writes one annotated tag at the exact preregistered ref, pushes that ref
without force, and reads the remote tag object and peeled implementation commit
back before it can issue an opaque receipt.

No receipt can be created from a caller-supplied mapping or bare digest.  Local
tests may point the frozen origin URL at a local bare repository by using Git's
``url.*.insteadOf`` configuration; the production receipt still records and
checks the configured frozen origin URL.
"""

from __future__ import annotations

import copy
import hmac
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Final, Mapping

from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    validate_implementation_manifest,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    EXTERNAL_PUBLICATION_FIELDS,
    EXTERNAL_TAG_MESSAGE_FIELDS,
    EXTERNAL_TAG_REF_TEMPLATE,
    FINAL_ATTEMPT_ID,
    MAX_TOTAL_RUNTIME_SECONDS,
    canonical_json_bytes,
    canonical_sha256,
)


EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-external-tag-message-v1"
)
EXTERNAL_PUBLICATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-external-publication-v1"
)
EXTERNAL_PUBLISHER_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-git-tag-publisher-v1"
)
PUBLICATION_GENESIS_SHA256: Final[str] = "0" * 64

ACQUISITION_PASS: Final[str] = "acquisition_pass"
SCORED_PASS: Final[str] = "scored_pass"
SCORED_FAILED_GATE: Final[str] = "scored_failed_gate"
FINAL_REGISTRY_SUCCESSOR: Final[str] = "final_registry_successor"
REPORT_KINDS: Final[tuple[str, ...]] = (
    ACQUISITION_PASS,
    SCORED_PASS,
    SCORED_FAILED_GATE,
    FINAL_REGISTRY_SUCCESSOR,
)

TERMINAL_PASS: Final[str] = "terminal_pass"
TERMINAL_FAIL: Final[str] = "terminal_fail"
REGISTERED_UNRUN: Final[str] = "registered_unrun"

_REPORT_KIND_STATUS: Final[dict[str, str]] = {
    ACQUISITION_PASS: TERMINAL_PASS,
    SCORED_PASS: TERMINAL_PASS,
    SCORED_FAILED_GATE: TERMINAL_FAIL,
    FINAL_REGISTRY_SUCCESSOR: REGISTERED_UNRUN,
}
_SCORING_ATTEMPT_IDS: Final[frozenset[str]] = frozenset(
    {
        DEVELOPMENT_ATTEMPT_ID,
        CONFIRMATION_ATTEMPT_ID,
        FINAL_ATTEMPT_ID,
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_VERIFIED_EXTERNAL_PUBLICATION_SENTINEL = object()


class SecGemmaOnlineRiskOverlayPublisherError(RuntimeError):
    """An external tag could not be published and verified exactly."""


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} must be one detached plain mapping"
        )
    return copy.deepcopy(value)


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} must be a lowercase SHA-256"
        )
    return value


def _sha1(value: Any, location: str) -> str:
    if type(value) is not str or _SHA1_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} must be a lowercase Git SHA-1"
        )
    return value


def _commit(value: Any, location: str) -> str:
    return _sha1(value, location)


def _attempt_and_status(
    *,
    attempt_id: Any,
    terminal_status: Any,
    report_kind: Any,
) -> tuple[str, str, str]:
    if type(report_kind) is not str or report_kind not in REPORT_KINDS:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication report kind is not preregistered"
        )
    expected_status = _REPORT_KIND_STATUS[report_kind]
    if terminal_status != expected_status:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication terminal status differs from its report kind"
        )
    if report_kind == ACQUISITION_PASS:
        expected_attempts = {DEVELOPMENT_ACQUISITION_ID}
    elif report_kind == FINAL_REGISTRY_SUCCESSOR:
        expected_attempts = {FINAL_ATTEMPT_ID}
    else:
        expected_attempts = _SCORING_ATTEMPT_IDS
    if type(attempt_id) is not str or attempt_id not in expected_attempts:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication attempt differs from its report kind"
        )
    return attempt_id, terminal_status, report_kind


def _tag_ref(
    *,
    attempt_id: str,
    report_kind: str,
    artifact_sha256: str,
) -> str:
    return EXTERNAL_TAG_REF_TEMPLATE.format(
        attempt_id=attempt_id,
        report_kind=report_kind,
        artifact_sha256=artifact_sha256,
    )


def build_external_tag_message(
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_id: str,
    terminal_status: str,
    report_kind: str,
    artifact_sha256: str,
    predecessor_publication_sha256: str,
) -> dict[str, Any]:
    """Build the exact canonical annotated-tag message."""

    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    fixed_attempt, fixed_status, fixed_kind = _attempt_and_status(
        attempt_id=attempt_id,
        terminal_status=terminal_status,
        report_kind=report_kind,
    )
    artifact = _sha256(artifact_sha256, "publication artifact")
    predecessor = _sha256(
        predecessor_publication_sha256,
        "predecessor publication",
    )
    message = {
        "schema_version": EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_commit": implementation["implementation_commit"],
        "attempt_id": fixed_attempt,
        "terminal_status": fixed_status,
        "report_kind": fixed_kind,
        "artifact_sha256": artifact,
        "predecessor_publication_sha256": predecessor,
        "external_cost_usd": 0.0,
    }
    if tuple(message) != EXTERNAL_TAG_MESSAGE_FIELDS:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External tag message fields differ from the frozen contract"
        )
    return message


def validate_external_tag_message(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonically rebuild one external tag message."""

    observed = _mapping(value, "external tag message")
    if tuple(observed) != EXTERNAL_TAG_MESSAGE_FIELDS:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External tag message fields changed"
        )
    expected = build_external_tag_message(
        implementation_manifest=implementation_manifest,
        attempt_id=observed["attempt_id"],
        terminal_status=observed["terminal_status"],
        report_kind=observed["report_kind"],
        artifact_sha256=observed["artifact_sha256"],
        predecessor_publication_sha256=observed[
            "predecessor_publication_sha256"
        ],
    )
    if observed != expected:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External tag message differs from its canonical reconstruction"
        )
    return expected


def _publication_body(
    *,
    implementation_manifest: Mapping[str, Any],
    tag_message: Mapping[str, Any],
    tag_ref: str,
    remote_name: str,
    remote_url: str,
    remote_tag_object_sha1: str,
    remote_peeled_commit: str,
) -> dict[str, Any]:
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    message = validate_external_tag_message(
        tag_message,
        implementation_manifest=implementation,
    )
    expected_ref = _tag_ref(
        attempt_id=message["attempt_id"],
        report_kind=message["report_kind"],
        artifact_sha256=message["artifact_sha256"],
    )
    if tag_ref != expected_ref:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication tag ref changed"
        )
    if remote_name != "origin":
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication must use the frozen origin remote"
        )
    if remote_url != implementation["origin_url"]:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication remote URL changed"
        )
    tag_object = _sha1(
        remote_tag_object_sha1,
        "remote annotated-tag object",
    )
    peeled = _commit(remote_peeled_commit, "remote peeled commit")
    if peeled != implementation["implementation_commit"]:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Remote annotated tag does not peel to the implementation commit"
        )
    return {
        "schema_version": EXTERNAL_PUBLICATION_SCHEMA_VERSION,
        "publisher_id": EXTERNAL_PUBLISHER_ID,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_commit": implementation["implementation_commit"],
        "attempt_id": message["attempt_id"],
        "terminal_status": message["terminal_status"],
        "report_kind": message["report_kind"],
        "artifact_sha256": message["artifact_sha256"],
        "predecessor_publication_sha256": message[
            "predecessor_publication_sha256"
        ],
        "tag_ref": tag_ref,
        "tag_target_commit": implementation["implementation_commit"],
        "tag_message_sha256": canonical_sha256(message),
        "remote_name": remote_name,
        "remote_url": remote_url,
        "remote_tag_object_sha1": tag_object,
        "remote_peeled_commit": peeled,
        "external_cost_usd": 0.0,
    }


def build_external_publication(
    *,
    implementation_manifest: Mapping[str, Any],
    tag_message: Mapping[str, Any],
    tag_ref: str,
    remote_name: str,
    remote_url: str,
    remote_tag_object_sha1: str,
    remote_peeled_commit: str,
) -> dict[str, Any]:
    """Build the exact self-hashed external publication receipt payload."""

    body = _publication_body(
        implementation_manifest=implementation_manifest,
        tag_message=tag_message,
        tag_ref=tag_ref,
        remote_name=remote_name,
        remote_url=remote_url,
        remote_tag_object_sha1=remote_tag_object_sha1,
        remote_peeled_commit=remote_peeled_commit,
    )
    publication = {
        **body,
        "publication_sha256": canonical_sha256(body),
    }
    if tuple(publication) != EXTERNAL_PUBLICATION_FIELDS:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication fields differ from the frozen contract"
        )
    return publication


def validate_external_publication(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one publication payload without making it authoritative."""

    observed = _mapping(value, "external publication")
    if tuple(observed) != EXTERNAL_PUBLICATION_FIELDS:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication fields changed"
        )
    digest = _sha256(
        observed["publication_sha256"],
        "external publication self-hash",
    )
    message = build_external_tag_message(
        implementation_manifest=implementation_manifest,
        attempt_id=observed["attempt_id"],
        terminal_status=observed["terminal_status"],
        report_kind=observed["report_kind"],
        artifact_sha256=observed["artifact_sha256"],
        predecessor_publication_sha256=observed[
            "predecessor_publication_sha256"
        ],
    )
    expected = build_external_publication(
        implementation_manifest=implementation_manifest,
        tag_message=message,
        tag_ref=observed["tag_ref"],
        remote_name=observed["remote_name"],
        remote_url=observed["remote_url"],
        remote_tag_object_sha1=observed["remote_tag_object_sha1"],
        remote_peeled_commit=observed["remote_peeled_commit"],
    )
    if (
        not hmac.compare_digest(
            observed["tag_message_sha256"],
            canonical_sha256(message),
        )
        or not hmac.compare_digest(
            digest,
            expected["publication_sha256"],
        )
        or observed != expected
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication differs from its canonical reconstruction"
        )
    return expected


class VerifiedExternalPublication:
    """Opaque proof issued only after exact remote Git readback."""

    __slots__ = ("_publication", "_sentinel")

    def __init__(
        self,
        *,
        publication: Mapping[str, Any],
        _sentinel: object,
    ) -> None:
        if _sentinel is not _VERIFIED_EXTERNAL_PUBLICATION_SENTINEL:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Verified external publications can only be issued by the publisher"
            )
        self._publication = copy.deepcopy(dict(publication))
        self._sentinel = _sentinel

    @property
    def publication(self) -> dict[str, Any]:
        return copy.deepcopy(self._publication)

    @property
    def publication_sha256(self) -> str:
        return self._publication["publication_sha256"]

    @property
    def artifact_sha256(self) -> str:
        return self._publication["artifact_sha256"]

    def __repr__(self) -> str:
        return "VerifiedExternalPublication(<redacted>)"


def is_verified_external_publication(value: Any) -> bool:
    """Return whether ``value`` is an authority issued by this module."""

    return (
        type(value) is VerifiedExternalPublication
        and getattr(value, "_sentinel", None)
        is _VERIFIED_EXTERNAL_PUBLICATION_SENTINEL
    )


def _issue_verified_external_publication(
    publication: Mapping[str, Any],
    *,
    implementation_manifest: Mapping[str, Any],
) -> VerifiedExternalPublication:
    validated = validate_external_publication(
        publication,
        implementation_manifest=implementation_manifest,
    )
    return VerifiedExternalPublication(
        publication=validated,
        _sentinel=_VERIFIED_EXTERNAL_PUBLICATION_SENTINEL,
    )


class ExternalGitTagPublisher:
    """Publish and read back one exact annotated tag on ``origin``."""

    __slots__ = (
        "_repo_root",
        "_implementation",
        "_clock",
        "_environment",
        "_no_hook_path",
        "_test_only_allow_url_rewrite",
    )

    def __init__(
        self,
        *,
        repo_root: Path,
        implementation_manifest: Mapping[str, Any],
        clock: Any = time.monotonic,
        test_only_allow_url_rewrite: bool = False,
    ) -> None:
        if not isinstance(repo_root, Path) or not repo_root.is_absolute():
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher repo_root must be an absolute pathlib.Path"
            )
        try:
            resolved = repo_root.resolve(strict=True)
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher repository root is unavailable"
            ) from exc
        if not resolved.is_dir():
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher repository root is not a directory"
            )
        if not callable(clock):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher clock must be callable"
            )
        if type(test_only_allow_url_rewrite) is not bool:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher test-only URL rewrite flag must be Boolean"
            )
        self._repo_root = resolved
        self._implementation = validate_implementation_manifest(
            implementation_manifest
        )
        self._clock = clock
        self._test_only_allow_url_rewrite = (
            test_only_allow_url_rewrite
        )
        git_directory = resolved / ".git"
        if git_directory.is_symlink() or not git_directory.is_dir():
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher requires one exact non-worktree Git directory"
            )
        no_hook_path = (
            git_directory
            / "sec-gemma-online-risk-overlay-v2-1-disabled-hooks"
        )
        try:
            no_hook_path.mkdir(exist_ok=True)
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher no-hook directory could not be created"
            ) from exc
        self._no_hook_path = no_hook_path
        environment = os.environ.copy()
        environment["GIT_COMMITTER_NAME"] = (
            "SEC Gemma Online Risk Overlay Publisher"
        )
        environment["GIT_COMMITTER_EMAIL"] = (
            "sec-gemma-online-risk-overlay@localhost"
        )
        environment["GIT_TERMINAL_PROMPT"] = "0"
        environment["GCM_INTERACTIVE"] = "Never"
        self._environment = environment

    def _verify_no_hook_path(self) -> None:
        try:
            entries = list(self._no_hook_path.iterdir())
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher no-hook directory is unavailable"
            ) from exc
        if self._no_hook_path.is_symlink() or entries:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher no-hook directory is not exact and empty"
            )

    def _remaining(self, deadline_monotonic: float) -> float:
        if (
            type(deadline_monotonic) not in {int, float}
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher deadline must be one finite monotonic value"
            )
        try:
            now = float(self._clock())
        except Exception as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher monotonic clock failed"
            ) from exc
        remaining = float(deadline_monotonic) - now
        if not math.isfinite(now) or now < 0.0 or remaining <= 0.0:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher deadline has expired"
            )
        return min(remaining, float(MAX_TOTAL_RUNTIME_SECONDS))

    def _run(
        self,
        *args: str,
        deadline_monotonic: float,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        remaining = self._remaining(deadline_monotonic)
        self._verify_no_hook_path()
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-c",
                    f"core.hooksPath={self._no_hook_path}",
                    "-c",
                    "tag.gpgSign=false",
                    "-c",
                    "credential.interactive=never",
                    *args,
                ],
                cwd=self._repo_root,
                env=self._environment,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=remaining,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Git publication command failed or exceeded its deadline"
            ) from exc
        if check and completed.returncode != 0:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Git publication command failed closed"
            )
        return completed

    def _text(
        self,
        *args: str,
        deadline_monotonic: float,
        input_bytes: bytes | None = None,
    ) -> str:
        completed = self._run(
            *args,
            deadline_monotonic=deadline_monotonic,
            input_bytes=input_bytes,
        )
        try:
            return completed.stdout.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Git publication output is not UTF-8"
            ) from exc

    def _verify_repository(
        self,
        *,
        deadline_monotonic: float,
    ) -> tuple[str, str]:
        implementation_commit = self._implementation[
            "implementation_commit"
        ]
        head = self._text(
            "rev-parse",
            "--verify",
            "HEAD",
            deadline_monotonic=deadline_monotonic,
        )
        if head != implementation_commit:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher HEAD differs from the implementation commit"
            )
        dirty = self._text(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            deadline_monotonic=deadline_monotonic,
        )
        if dirty:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher repository is not clean"
            )
        remote_name = "origin"
        remote_url = self._text(
            "config",
            "--get",
            f"remote.{remote_name}.url",
            deadline_monotonic=deadline_monotonic,
        )
        if remote_url != self._implementation["origin_url"]:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher origin differs from the implementation manifest"
            )
        push_urls = self._run(
            "config",
            "--get-all",
            f"remote.{remote_name}.pushurl",
            deadline_monotonic=deadline_monotonic,
            check=False,
        )
        if push_urls.returncode not in {0, 1}:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher push-URL verification failed"
            )
        rewrites = self._run(
            "config",
            "--get-regexp",
            r"^url\..*\.(insteadof|pushinsteadof)$",
            deadline_monotonic=deadline_monotonic,
            check=False,
        )
        if rewrites.returncode not in {0, 1}:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher URL-rewrite verification failed"
            )
        if (
            push_urls.stdout.strip() or rewrites.stdout.strip()
        ) and not self._test_only_allow_url_rewrite:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher origin transport is redirected or rewritten"
            )
        branch_ref = f"refs/heads/{self._implementation['branch']}"
        remote_branch = self._text(
            "ls-remote",
            remote_name,
            branch_ref,
            deadline_monotonic=deadline_monotonic,
        )
        if remote_branch != f"{implementation_commit}\t{branch_ref}":
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Frozen origin branch does not equal the implementation commit"
            )
        return remote_name, remote_url

    def _ensure_ref_absent(
        self,
        *,
        remote_name: str,
        tag_ref: str,
        deadline_monotonic: float,
    ) -> None:
        local = self._run(
            "show-ref",
            "--verify",
            "--quiet",
            tag_ref,
            deadline_monotonic=deadline_monotonic,
            check=False,
        )
        if local.returncode not in {0, 1}:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Local tag existence check failed"
            )
        if local.returncode == 0:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "External publication tag ref cannot be reused"
            )
        remote = self._run(
            "ls-remote",
            "--exit-code",
            "--tags",
            remote_name,
            tag_ref,
            deadline_monotonic=deadline_monotonic,
            check=False,
        )
        if remote.returncode not in {0, 2}:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Remote tag existence check failed"
            )
        if remote.returncode == 0 or remote.stdout.strip():
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "External publication remote tag cannot be reused"
            )

    def _read_remote_tag(
        self,
        *,
        remote_name: str,
        tag_ref: str,
        deadline_monotonic: float,
    ) -> tuple[str, str]:
        output = self._text(
            "ls-remote",
            "--tags",
            remote_name,
            tag_ref,
            f"{tag_ref}^{{}}",
            deadline_monotonic=deadline_monotonic,
        )
        rows: dict[str, str] = {}
        for raw_line in output.splitlines():
            parts = raw_line.split("\t")
            if len(parts) != 2:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Remote tag readback is malformed"
                )
            digest, ref = parts
            rows[ref] = _sha1(digest, "remote tag readback")
        if set(rows) != {tag_ref, f"{tag_ref}^{{}}"}:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Remote annotated-tag readback is incomplete"
            )
        return rows[tag_ref], rows[f"{tag_ref}^{{}}"]

    def _canonical_tag_object(
        self,
        *,
        message: Mapping[str, Any],
        tag_ref: str,
        deadline_monotonic: float,
    ) -> tuple[bytes, str, str]:
        tag_name = tag_ref.removeprefix("refs/tags/")
        message_bytes = canonical_json_bytes(message)
        commit_timestamp = self._text(
            "show",
            "--no-patch",
            "--format=%ct",
            self._implementation["implementation_commit"],
            deadline_monotonic=deadline_monotonic,
        )
        if not commit_timestamp.isdigit():
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Implementation commit timestamp is not canonical"
            )
        tag_object_bytes = (
            f"object {self._implementation['implementation_commit']}\n"
            "type commit\n"
            f"tag {tag_name}\n"
            "tagger SEC Gemma Online Risk Overlay Publisher "
            "<sec-gemma-online-risk-overlay@localhost> "
            f"{commit_timestamp} +0000\n\n"
        ).encode("utf-8") + message_bytes + b"\n"
        local_object = self._text(
            "mktag",
            deadline_monotonic=deadline_monotonic,
            input_bytes=tag_object_bytes,
        )
        local_peeled = self._text(
            "rev-parse",
            "--verify",
            f"{local_object}^{{}}",
            deadline_monotonic=deadline_monotonic,
        )
        _sha1(local_object, "local annotated-tag object")
        if local_peeled != self._implementation["implementation_commit"]:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Local annotated tag does not target the implementation commit"
            )
        raw_tag = self._run(
            "cat-file",
            "-p",
            local_object,
            deadline_monotonic=deadline_monotonic,
        ).stdout
        separator = raw_tag.find(b"\n\n")
        if separator < 0:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Local annotated-tag object has no message"
            )
        observed_message = raw_tag[separator + 2 :].rstrip(b"\n")
        if observed_message != message_bytes:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Local annotated-tag message changed before publication"
            )
        return message_bytes, local_object, local_peeled

    def publish(
        self,
        *,
        attempt_id: str,
        terminal_status: str,
        report_kind: str,
        artifact_sha256: str,
        predecessor_publication_sha256: str,
        deadline_monotonic: float,
    ) -> VerifiedExternalPublication:
        """Push one new tag and issue a receipt only after remote readback."""

        message = build_external_tag_message(
            implementation_manifest=self._implementation,
            attempt_id=attempt_id,
            terminal_status=terminal_status,
            report_kind=report_kind,
            artifact_sha256=artifact_sha256,
            predecessor_publication_sha256=(
                predecessor_publication_sha256
            ),
        )
        tag_ref = _tag_ref(
            attempt_id=message["attempt_id"],
            report_kind=message["report_kind"],
            artifact_sha256=message["artifact_sha256"],
        )
        remote_name, remote_url = self._verify_repository(
            deadline_monotonic=deadline_monotonic
        )
        self._ensure_ref_absent(
            remote_name=remote_name,
            tag_ref=tag_ref,
            deadline_monotonic=deadline_monotonic,
        )
        _, local_object, local_peeled = self._canonical_tag_object(
            message=message,
            tag_ref=tag_ref,
            deadline_monotonic=deadline_monotonic,
        )
        self._run(
            "push",
            "--porcelain",
            remote_name,
            f"{local_object}:{tag_ref}",
            deadline_monotonic=deadline_monotonic,
        )
        remote_object, remote_peeled = self._read_remote_tag(
            remote_name=remote_name,
            tag_ref=tag_ref,
            deadline_monotonic=deadline_monotonic,
        )
        if (
            remote_object != local_object
            or remote_peeled != local_peeled
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Remote annotated-tag readback differs from the pushed object"
            )
        publication = build_external_publication(
            implementation_manifest=self._implementation,
            tag_message=message,
            tag_ref=tag_ref,
            remote_name=remote_name,
            remote_url=remote_url,
            remote_tag_object_sha1=remote_object,
            remote_peeled_commit=remote_peeled,
        )
        self._remaining(deadline_monotonic)
        return _issue_verified_external_publication(
            publication,
            implementation_manifest=self._implementation,
        )

    def _recover_exact_final_registry_publication(
        self,
        *,
        attempt_id: str,
        terminal_status: str,
        report_kind: str,
        artifact_sha256: str,
        predecessor_publication_sha256: str,
        deadline_monotonic: float,
    ) -> VerifiedExternalPublication:
        if (
            attempt_id != FINAL_ATTEMPT_ID
            or terminal_status != REGISTERED_UNRUN
            or report_kind != FINAL_REGISTRY_SUCCESSOR
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Only the exact final-registry successor can be recovered"
            )
        message = build_external_tag_message(
            implementation_manifest=self._implementation,
            attempt_id=attempt_id,
            terminal_status=terminal_status,
            report_kind=report_kind,
            artifact_sha256=artifact_sha256,
            predecessor_publication_sha256=(
                predecessor_publication_sha256
            ),
        )
        tag_ref = _tag_ref(
            attempt_id=attempt_id,
            report_kind=report_kind,
            artifact_sha256=artifact_sha256,
        )
        remote_name, remote_url = self._verify_repository(
            deadline_monotonic=deadline_monotonic
        )
        _, expected_object, expected_peeled = (
            self._canonical_tag_object(
                message=message,
                tag_ref=tag_ref,
                deadline_monotonic=deadline_monotonic,
            )
        )
        remote_object, remote_peeled = self._read_remote_tag(
            remote_name=remote_name,
            tag_ref=tag_ref,
            deadline_monotonic=deadline_monotonic,
        )
        if (
            remote_object != expected_object
            or remote_peeled != expected_peeled
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Existing final-registry tag differs from its canonical object"
            )
        publication = build_external_publication(
            implementation_manifest=self._implementation,
            tag_message=message,
            tag_ref=tag_ref,
            remote_name=remote_name,
            remote_url=remote_url,
            remote_tag_object_sha1=remote_object,
            remote_peeled_commit=remote_peeled,
        )
        self._remaining(deadline_monotonic)
        return _issue_verified_external_publication(
            publication,
            implementation_manifest=self._implementation,
        )

    def publish_or_recover_final_registry(
        self,
        *,
        attempt_id: str,
        terminal_status: str,
        report_kind: str,
        artifact_sha256: str,
        predecessor_publication_sha256: str,
        deadline_monotonic: float,
    ) -> VerifiedExternalPublication:
        """Publish once, or recover only the identical final-registry tag."""

        try:
            return self.publish(
                attempt_id=attempt_id,
                terminal_status=terminal_status,
                report_kind=report_kind,
                artifact_sha256=artifact_sha256,
                predecessor_publication_sha256=(
                    predecessor_publication_sha256
                ),
                deadline_monotonic=deadline_monotonic,
            )
        except SecGemmaOnlineRiskOverlayPublisherError:
            try:
                return self._recover_exact_final_registry_publication(
                    attempt_id=attempt_id,
                    terminal_status=terminal_status,
                    report_kind=report_kind,
                    artifact_sha256=artifact_sha256,
                    predecessor_publication_sha256=(
                        predecessor_publication_sha256
                    ),
                    deadline_monotonic=deadline_monotonic,
                )
            except SecGemmaOnlineRiskOverlayPublisherError as recovery_error:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Final-registry tag is neither new nor exactly recoverable"
                ) from recovery_error


__all__ = [
    "ACQUISITION_PASS",
    "EXTERNAL_PUBLICATION_SCHEMA_VERSION",
    "EXTERNAL_PUBLISHER_ID",
    "EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION",
    "ExternalGitTagPublisher",
    "FINAL_REGISTRY_SUCCESSOR",
    "PUBLICATION_GENESIS_SHA256",
    "REGISTERED_UNRUN",
    "REPORT_KINDS",
    "SCORED_FAILED_GATE",
    "SCORED_PASS",
    "SecGemmaOnlineRiskOverlayPublisherError",
    "TERMINAL_FAIL",
    "TERMINAL_PASS",
    "VerifiedExternalPublication",
    "build_external_publication",
    "build_external_tag_message",
    "is_verified_external_publication",
    "validate_external_publication",
    "validate_external_tag_message",
]
