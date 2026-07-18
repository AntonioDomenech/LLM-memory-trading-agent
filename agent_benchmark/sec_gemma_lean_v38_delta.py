"""Committed-blob-only implementation delta proof for SEC/Gemma v3.8.

The checker deliberately has a very small observation surface.  It invokes
Git only for raw tree differences and literal blob reads, parses Python with
``ast`` without importing it, and never consults the worktree.  A manifest is
created only after the implementation commit exists, so the manifest cannot
be an input to its own implementation hash.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tokenize
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


MANIFEST_VERSION = "aapl-sec-gemma-lean-v38-implementation-delta-v1"
PREREG_COMMIT = "c8a1f2d2ad7ec22d4f50600180908e9d499520f4"
PREREG_TREE = "5eaf18fa8e6f4ee715a66df6af010a312f50454a"
PREREG_DOC_PATH = "docs/aapl_sec_gemma_lean_evidence_v3_8.md"
PREREG_DOC_BLOB = "1740ddb65985fed0eb001623760c9a70425462ed"
PREREG_DOC_SHA256 = (
    "6d27fdc7f21668d0503a91b30c6b6e4699778031330f5c3a5205735b268102a2"
)
IMPLEMENTATION_BASE_COMMIT = "8d84ca6b1d8a5ef50d790beb9b4b53e51812a542"
IMPLEMENTATION_BASE_TREE = "1287094442f4ceecdaf0ca29b9bdb4980d65b7e7"
_COUNTERPART_MECHANICAL_REPLACEMENTS = (
    ("sec_gemma_lean_v37", "sec_gemma_lean_v38"),
    ("SecGemmaLeanV37", "SecGemmaLeanV38"),
    ("V37", "V38"),
    ("v37", "v38"),
    ("v3_7", "v3_8"),
    ("v3-7", "v3-8"),
    ("v3.7", "v3.8"),
)

_OID_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RAW_HEADER_RE = re.compile(
    rb":([0-7]{6}) ([0-7]{6}) ([0-9a-f]{40}) ([0-9a-f]{40}) ([A-Z][0-9]*)\Z"
)
_ZERO_OID = "0" * 40
_REGULAR_MODE = "100644"
_DIFF_TREE_ARGS = (
    "diff-tree",
    "--raw",
    "-r",
    "-z",
    "--no-renames",
    "--no-abbrev",
)


class DeltaValidationError(ValueError):
    """Raised when committed evidence is outside the preregistered delta."""


@dataclass(frozen=True)
class DeltaPins:
    prereg_commit: str
    prereg_tree: str
    prereg_doc_path: str
    prereg_doc_blob: str
    prereg_doc_sha256: str
    implementation_base_commit: str
    implementation_base_tree: str

    def validate(self) -> None:
        for name, value in (
            ("prereg_commit", self.prereg_commit),
            ("prereg_tree", self.prereg_tree),
            ("prereg_doc_blob", self.prereg_doc_blob),
            ("implementation_base_commit", self.implementation_base_commit),
            ("implementation_base_tree", self.implementation_base_tree),
        ):
            if not _OID_RE.fullmatch(value):
                raise DeltaValidationError(f"{name} must be a literal SHA-1 object id")
        if not _SHA256_RE.fullmatch(self.prereg_doc_sha256):
            raise DeltaValidationError("prereg_doc_sha256 must be canonical")
        _validate_path(self.prereg_doc_path)

    def as_json(self) -> dict[str, str]:
        return {
            "prereg_commit": self.prereg_commit,
            "prereg_tree": self.prereg_tree,
            "prereg_doc_path": self.prereg_doc_path,
            "prereg_doc_blob": self.prereg_doc_blob,
            "prereg_doc_sha256": self.prereg_doc_sha256,
            "implementation_base_commit": self.implementation_base_commit,
            "implementation_base_tree": self.implementation_base_tree,
        }


DEFAULT_PINS = DeltaPins(
    prereg_commit=PREREG_COMMIT,
    prereg_tree=PREREG_TREE,
    prereg_doc_path=PREREG_DOC_PATH,
    prereg_doc_blob=PREREG_DOC_BLOB,
    prereg_doc_sha256=PREREG_DOC_SHA256,
    implementation_base_commit=IMPLEMENTATION_BASE_COMMIT,
    implementation_base_tree=IMPLEMENTATION_BASE_TREE,
)


@dataclass(frozen=True)
class PathRule:
    """Exact static surface expected for one implementation-delta path."""

    status: str
    expected_symbol_ids: tuple[str, ...]
    expected_import_specs: tuple[str, ...] = ()
    expected_symbol_inventory_sha256: str | None = None
    expected_symbol_count: int | None = None
    expected_import_inventory_sha256: str | None = None
    expected_import_count: int | None = None

    def validate(self, path: str) -> None:
        if self.status not in {"A", "M"}:
            raise DeltaValidationError(f"{path}: rule status must be A or M")
        if tuple(sorted(set(self.expected_symbol_ids))) != self.expected_symbol_ids:
            raise DeltaValidationError(f"{path}: symbol ids must be unique and sorted")
        if tuple(sorted(set(self.expected_import_specs))) != self.expected_import_specs:
            raise DeltaValidationError(f"{path}: import specs must be unique and sorted")
        if self.expected_symbol_inventory_sha256 is not None:
            if self.expected_symbol_ids:
                raise DeltaValidationError(f"{path}: use exact ids or an inventory hash, not both")
            if not _SHA256_RE.fullmatch(self.expected_symbol_inventory_sha256):
                raise DeltaValidationError(f"{path}: symbol inventory hash is malformed")
            if (
                isinstance(self.expected_symbol_count, bool)
                or not isinstance(self.expected_symbol_count, int)
                or self.expected_symbol_count < 1
            ):
                raise DeltaValidationError(f"{path}: hashed symbol inventory needs a count")
        elif self.expected_symbol_count is not None:
            raise DeltaValidationError(f"{path}: symbol count requires an inventory hash")
        if self.expected_import_inventory_sha256 is not None:
            if self.expected_import_specs:
                raise DeltaValidationError(f"{path}: use exact imports or an import hash, not both")
            if not _SHA256_RE.fullmatch(self.expected_import_inventory_sha256):
                raise DeltaValidationError(f"{path}: import inventory hash is malformed")
            if (
                isinstance(self.expected_import_count, bool)
                or not isinstance(self.expected_import_count, int)
                or self.expected_import_count < 0
            ):
                raise DeltaValidationError(f"{path}: hashed import inventory needs a count")
        elif self.expected_import_count is not None:
            raise DeltaValidationError(f"{path}: import count requires an inventory hash")

    def as_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "expected_symbol_ids": list(self.expected_symbol_ids),
            "expected_import_specs": list(self.expected_import_specs),
            "expected_symbol_inventory_sha256": self.expected_symbol_inventory_sha256,
            "expected_symbol_count": self.expected_symbol_count,
            "expected_import_inventory_sha256": self.expected_import_inventory_sha256,
            "expected_import_count": self.expected_import_count,
        }


@dataclass(frozen=True)
class CounterpartRule:
    """Exact committed-v3.3 comparison contract for one v3.8 delta path.

    ``expected_candidate_sha256`` binds the complete v3.8 blob for ordinary
    files.  The verifier module is bound by a literal SHA-256 anchor in its
    focused test blob.  The verifier, in turn, binds that complete test blob
    with only the anchor literal masked.  This one-way construction has no
    cryptographic fixed point and leaves no executable byte unbound.
    """

    counterpart_path: str
    expected_mechanical_counts: tuple[int, ...]
    expected_candidate_sha256: str | None
    redacted_literal_assignment: str | None
    expected_redacted_candidate_sha256: str | None
    external_sha256_anchor_path: str | None
    external_sha256_anchor_assignment: str | None
    excluded_change_symbol_ids: tuple[str, ...]
    expected_changed_symbol_ids: tuple[str, ...]
    expected_changed_import_specs: tuple[str, ...]
    expected_changed_symbol_evidence_sha256: str

    def validate(self, path: str) -> None:
        _validate_path(self.counterpart_path)
        if not self.counterpart_path.endswith(".py"):
            raise DeltaValidationError(f"{path}: counterpart path must be Python")
        if len(self.expected_mechanical_counts) != len(
            _COUNTERPART_MECHANICAL_REPLACEMENTS
        ):
            raise DeltaValidationError(f"{path}: mechanical replacement count is incomplete")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.expected_mechanical_counts
        ):
            raise DeltaValidationError(f"{path}: mechanical counts must be nonnegative integers")
        if tuple(sorted(set(self.expected_changed_symbol_ids))) != (
            self.expected_changed_symbol_ids
        ):
            raise DeltaValidationError(f"{path}: counterpart symbol ids must be unique and sorted")
        if tuple(sorted(set(self.excluded_change_symbol_ids))) != (
            self.excluded_change_symbol_ids
        ):
            raise DeltaValidationError(f"{path}: excluded counterpart symbols must be unique and sorted")
        if tuple(sorted(self.expected_changed_import_specs)) != (
            self.expected_changed_import_specs
        ):
            raise DeltaValidationError(f"{path}: counterpart imports must be sorted")
        if not _SHA256_RE.fullmatch(self.expected_changed_symbol_evidence_sha256):
            raise DeltaValidationError(f"{path}: counterpart evidence hash is malformed")
        complete = self.expected_candidate_sha256
        redacted_name = self.redacted_literal_assignment
        redacted_hash = self.expected_redacted_candidate_sha256
        anchor_path = self.external_sha256_anchor_path
        anchor_name = self.external_sha256_anchor_assignment
        modes = (
            complete is not None,
            redacted_name is not None or redacted_hash is not None,
            anchor_path is not None or anchor_name is not None,
        )
        if sum(modes) != 1:
            raise DeltaValidationError(f"{path}: candidate binding mode is ambiguous")
        if modes[0]:
            if not _SHA256_RE.fullmatch(complete):
                raise DeltaValidationError(f"{path}: candidate blob hash is malformed")
        elif modes[1]:
            if (
                path != "tests/test_sec_gemma_lean_v38_delta.py"
                or redacted_name != "V38_DELTA_MODULE_SHA256"
                or not isinstance(redacted_hash, str)
                or not _SHA256_RE.fullmatch(redacted_hash)
            ):
                raise DeltaValidationError(f"{path}: invalid anchor-redaction contract")
        elif (
            path != "agent_benchmark/sec_gemma_lean_v38_delta.py"
            or anchor_path != "tests/test_sec_gemma_lean_v38_delta.py"
            or anchor_name != "V38_DELTA_MODULE_SHA256"
        ):
            raise DeltaValidationError(f"{path}: invalid external-anchor contract")
        expected_exclusions = {
            "agent_benchmark/sec_gemma_lean_v38_delta.py": (
                "assignment|$module._COUNTERPART_RULE_DATA|1",
            ),
            "tests/test_sec_gemma_lean_v38_delta.py": (
                "assignment|$module.V38_DELTA_MODULE_SHA256|1",
            ),
        }.get(path, ())
        if self.excluded_change_symbol_ids != expected_exclusions:
            raise DeltaValidationError(f"{path}: counterpart exclusions are not exact")

    def as_json(self) -> dict[str, Any]:
        return {
            "counterpart_path": self.counterpart_path,
            "expected_mechanical_counts": list(self.expected_mechanical_counts),
            "expected_candidate_sha256": self.expected_candidate_sha256,
            "redacted_literal_assignment": self.redacted_literal_assignment,
            "expected_redacted_candidate_sha256": (
                self.expected_redacted_candidate_sha256
            ),
            "external_sha256_anchor_path": self.external_sha256_anchor_path,
            "external_sha256_anchor_assignment": (
                self.external_sha256_anchor_assignment
            ),
            "excluded_change_symbol_ids": list(self.excluded_change_symbol_ids),
            "expected_changed_symbol_ids": list(self.expected_changed_symbol_ids),
            "expected_changed_symbol_count": len(self.expected_changed_symbol_ids),
            "expected_changed_import_specs": list(self.expected_changed_import_specs),
            "expected_changed_symbol_evidence_sha256": (
                self.expected_changed_symbol_evidence_sha256
            ),
        }


@dataclass(frozen=True)
class FrozenSymbolPin:
    path: str
    symbol_id: str
    physical_sha256: str
    literal_sha256: str
    semantic_sha256: str

    def validate(self) -> None:
        _validate_path(self.path)
        if not self.symbol_id:
            raise DeltaValidationError("frozen symbol id must be nonempty")
        for value in (
            self.physical_sha256,
            self.literal_sha256,
            self.semantic_sha256,
        ):
            if not _SHA256_RE.fullmatch(value):
                raise DeltaValidationError("frozen symbol hashes must be canonical")

    def as_json(self) -> dict[str, str]:
        return {
            "path": self.path,
            "symbol_id": self.symbol_id,
            "physical_sha256": self.physical_sha256,
            "literal_sha256": self.literal_sha256,
            "semantic_sha256": self.semantic_sha256,
        }


@dataclass(frozen=True)
class AllowSpec:
    path_rules: Mapping[str, PathRule]
    frozen_symbols: tuple[FrozenSymbolPin, ...] = ()
    counterpart_rules: Mapping[str, CounterpartRule] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.path_rules:
            raise DeltaValidationError("allow spec must name at least one exact path")
        for path in sorted(self.path_rules):
            _validate_path(path)
            if not path.endswith(".py"):
                raise DeltaValidationError(f"{path}: implementation paths must be Python")
            self.path_rules[path].validate(path)
        frozen_keys: set[tuple[str, str]] = set()
        for pin in self.frozen_symbols:
            pin.validate()
            key = (pin.path, pin.symbol_id)
            if key in frozen_keys:
                raise DeltaValidationError("duplicate frozen symbol pin")
            frozen_keys.add(key)
        if self.counterpart_rules and set(self.counterpart_rules) != set(self.path_rules):
            raise DeltaValidationError(
                "counterpart rules must cover the exact implementation path inventory"
            )
        for path in sorted(self.counterpart_rules):
            self.counterpart_rules[path].validate(path)

    def as_json(self) -> dict[str, Any]:
        return {
            "path_rules": {
                path: self.path_rules[path].as_json() for path in sorted(self.path_rules)
            },
            "frozen_symbols": [
                pin.as_json()
                for pin in sorted(
                    self.frozen_symbols, key=lambda value: (value.path, value.symbol_id)
                )
            ],
            "counterpart_rules": {
                path: self.counterpart_rules[path].as_json()
                for path in sorted(self.counterpart_rules)
            },
        }

    @property
    def sha256(self) -> str:
        self.validate()
        return _sha256(_canonical_json(self.as_json()))


@dataclass(frozen=True)
class _DiffEntry:
    path: str
    old_mode: str
    new_mode: str
    old_oid: str
    new_oid: str
    status: str


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_oid(value: bytes) -> str:
    header = b"blob " + str(len(value)).encode("ascii") + b"\0"
    try:
        digest = hashlib.sha1(header + value, usedforsecurity=False)
    except TypeError:  # pragma: no cover - older Python compatibility
        digest = hashlib.sha1(header + value)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DeltaValidationError("value is not canonical JSON") from exc
    return rendered.encode("ascii")


def _validate_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DeltaValidationError("Git paths must be nonempty canonical POSIX paths")
    pure = PurePosixPath(value)
    if pure.is_absolute() or str(pure) != value or any(part in {"", ".", ".."} for part in pure.parts):
        raise DeltaValidationError(f"unsafe Git path: {value!r}")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise DeltaValidationError("Git paths may not contain control characters")


def _git_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in ("SystemRoot", "WINDIR", "TEMP", "TMP"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _git(repo: Path, args: Sequence[str], *, git_binary: str) -> bytes:
    resolved_git = shutil.which(git_binary)
    if resolved_git is None:
        raise DeltaValidationError("Git executable is unavailable")
    try:
        result = subprocess.run(
            [resolved_git, *args],
            cwd=repo,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeltaValidationError("Git subprocess failed safely") from exc
    if result.returncode != 0 or result.stderr:
        raise DeltaValidationError("Git rejected a committed-object query")
    return result.stdout


def _raw_diff(
    repo: Path, old_object: str, new_object: str, *, git_binary: str
) -> tuple[_DiffEntry, ...]:
    if not _OID_RE.fullmatch(old_object) or not _OID_RE.fullmatch(new_object):
        raise DeltaValidationError("diff endpoints must be literal SHA-1 object ids")
    raw = _git(repo, (*_DIFF_TREE_ARGS, old_object, new_object), git_binary=git_binary)
    if not raw:
        return ()
    parts = raw.split(b"\0")
    if parts[-1] != b"" or len(parts) % 2 != 1:
        raise DeltaValidationError("Git raw diff is not exact NUL framing")
    entries: list[_DiffEntry] = []
    seen: set[str] = set()
    for index in range(0, len(parts) - 1, 2):
        header = parts[index]
        path_bytes = parts[index + 1]
        match = _RAW_HEADER_RE.fullmatch(header)
        if match is None:
            raise DeltaValidationError("Git raw diff header is malformed")
        try:
            path = path_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise DeltaValidationError("Git path is not canonical UTF-8") from exc
        _validate_path(path)
        if path in seen:
            raise DeltaValidationError("Git raw diff repeated a path")
        seen.add(path)
        old_mode, new_mode, old_oid, new_oid, status = (
            value.decode("ascii") for value in match.groups()
        )
        entries.append(
            _DiffEntry(path, old_mode, new_mode, old_oid, new_oid, status)
        )
    return tuple(sorted(entries, key=lambda value: value.path.encode("utf-8")))


def _cat_blob(
    repo: Path,
    object_id: str,
    *,
    path: str | None = None,
    git_binary: str,
) -> bytes:
    if not _OID_RE.fullmatch(object_id):
        raise DeltaValidationError("blob query requires a literal SHA-1 object id")
    specification = object_id
    if path is not None:
        _validate_path(path)
        specification = f"{object_id}:{path}"
    payload = _git(repo, ("cat-file", "blob", specification), git_binary=git_binary)
    if path is None and _git_blob_oid(payload) != object_id:
        raise DeltaValidationError("Git blob bytes disagree with their object id")
    return payload


def _assert_tree_pin(
    repo: Path, commit: str, tree: str, *, label: str, git_binary: str
) -> None:
    if _object_type(repo, tree, git_binary=git_binary) != "tree":
        raise DeltaValidationError(f"{label} pinned tree identity is not a tree object")
    observed_tree = _commit_tree_oid(
        repo, commit, label=label, git_binary=git_binary
    )
    if observed_tree != tree:
        raise DeltaValidationError(f"{label} commit does not match its pinned tree")


def _object_type(repo: Path, object_id: str, *, git_binary: str) -> str:
    if not _OID_RE.fullmatch(object_id):
        raise DeltaValidationError("object type query requires a literal SHA-1 object id")
    payload = _git(repo, ("cat-file", "-t", object_id), git_binary=git_binary)
    try:
        value = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:  # pragma: no cover - Git emits ASCII types
        raise DeltaValidationError("Git object type is malformed") from exc
    if value not in {"blob\n", "tree\n", "commit\n", "tag\n"}:
        raise DeltaValidationError("Git object type is malformed")
    return value[:-1]


def _commit_tree_oid(
    repo: Path, commit: str, *, label: str, git_binary: str
) -> str:
    if _object_type(repo, commit, git_binary=git_binary) != "commit":
        raise DeltaValidationError(f"{label} identity is not a commit object")
    payload = _git(repo, ("cat-file", "commit", commit), git_binary=git_binary)
    first_line, separator, _ = payload.partition(b"\n")
    if separator != b"\n" or not first_line.startswith(b"tree "):
        raise DeltaValidationError(f"{label} commit header is malformed")
    try:
        tree = first_line[5:].decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise DeltaValidationError(f"{label} commit tree identity is malformed") from exc
    if not _OID_RE.fullmatch(tree):
        raise DeltaValidationError(f"{label} commit tree identity is malformed")
    return tree


def _assert_commit_object(
    repo: Path, commit: str, *, label: str, git_binary: str
) -> None:
    _commit_tree_oid(repo, commit, label=label, git_binary=git_binary)


def _entry_with_hashes(
    repo: Path, entry: _DiffEntry, *, git_binary: str
) -> tuple[dict[str, Any], bytes | None, bytes | None]:
    if entry.status not in {"A", "M", "D"}:
        raise DeltaValidationError(f"{entry.path}: rename/type/copy status is forbidden")
    for mode in (entry.old_mode, entry.new_mode):
        if mode not in {"000000", _REGULAR_MODE}:
            raise DeltaValidationError(f"{entry.path}: symlink/submodule/nonregular mode")
    old_blob = None
    new_blob = None
    if entry.old_oid != _ZERO_OID:
        old_blob = _cat_blob(repo, entry.old_oid, git_binary=git_binary)
    if entry.new_oid != _ZERO_OID:
        new_blob = _cat_blob(repo, entry.new_oid, git_binary=git_binary)
    return (
        {
            "path": entry.path,
            "status": entry.status,
            "old_mode": entry.old_mode,
            "new_mode": entry.new_mode,
            "old_oid": entry.old_oid,
            "new_oid": entry.new_oid,
            "old_sha256": None if old_blob is None else _sha256(old_blob),
            "new_sha256": None if new_blob is None else _sha256(new_blob),
        },
        old_blob,
        new_blob,
    )


def _encoding_and_lines(blob: bytes) -> tuple[str, list[bytes], list[int]]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(blob).readline)
        blob.decode(encoding, errors="strict")
    except (LookupError, SyntaxError, UnicodeDecodeError) as exc:
        raise DeltaValidationError("Python blob encoding is not exact") from exc
    lines = blob.splitlines(keepends=True)
    if not lines:
        lines = [b""]
    starts: list[int] = []
    position = 0
    for line in lines:
        starts.append(position)
        position += len(line)
    return encoding, lines, starts


def _line_column_offset(
    blob: bytes,
    encoding: str,
    lines: Sequence[bytes],
    starts: Sequence[int],
    line_number: int,
    utf8_column: int,
) -> int:
    if line_number < 1 or line_number > len(lines) or utf8_column < 0:
        raise DeltaValidationError("AST position is outside the physical blob")
    raw_line = lines[line_number - 1]
    ending_length = 2 if raw_line.endswith(b"\r\n") else int(raw_line.endswith((b"\r", b"\n")))
    content = raw_line[:-ending_length] if ending_length else raw_line
    bom_length = 0
    decode_encoding = encoding
    if line_number == 1 and encoding.lower().replace("_", "-") == "utf-8-sig":
        bom_length = 3
        content = content[bom_length:]
        decode_encoding = "utf-8"
    try:
        text = content.decode(decode_encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise DeltaValidationError("AST line cannot be mapped to raw bytes") from exc
    consumed_utf8 = 0
    character_count = 0
    for character in text:
        if consumed_utf8 == utf8_column:
            break
        consumed_utf8 += len(character.encode("utf-8"))
        character_count += 1
        if consumed_utf8 > utf8_column:
            raise DeltaValidationError("AST column splits a Unicode character")
    if consumed_utf8 != utf8_column:
        raise DeltaValidationError("AST column exceeds its physical line")
    prefix = text[:character_count].encode(decode_encoding)
    return starts[line_number - 1] + bom_length + len(prefix)


def _node_span(
    blob: bytes,
    node: ast.AST,
    encoding: str,
    lines: Sequence[bytes],
    starts: Sequence[int],
    *,
    decorator: bool = False,
) -> tuple[int, int]:
    attributes = ("lineno", "col_offset", "end_lineno", "end_col_offset")
    if any(not hasattr(node, name) for name in attributes):
        raise DeltaValidationError("Python AST node lacks exact end positions")
    start_column = int(node.col_offset)
    if decorator:
        start_column -= 1
        if start_column < 0:
            raise DeltaValidationError("decorator has no physical at-sign")
    start = _line_column_offset(
        blob, encoding, lines, starts, int(node.lineno), start_column
    )
    end = _line_column_offset(
        blob, encoding, lines, starts, int(node.end_lineno), int(node.end_col_offset)
    )
    if not 0 <= start < end <= len(blob):
        raise DeltaValidationError("Python AST span is not a nonempty byte slice")
    if decorator and blob[start : start + 1] != b"@":
        raise DeltaValidationError("decorator byte slice is not exact")
    return start, end


def _literal_value_payload(value: Any) -> bytes:
    if value is None:
        rendered = {"type": "none", "value": None}
    elif value is Ellipsis:
        rendered = {"type": "ellipsis", "value": "..."}
    elif isinstance(value, bool):
        rendered = {"type": "bool", "value": value}
    elif isinstance(value, int):
        rendered = {"type": "int", "value": str(value)}
    elif isinstance(value, float):
        rendered = {"type": "float", "value": value.hex()}
    elif isinstance(value, complex):
        rendered = {
            "type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    elif isinstance(value, str):
        rendered = {"type": "str", "utf8_sha256": _sha256(value.encode("utf-8"))}
    elif isinstance(value, bytes):
        rendered = {"type": "bytes", "sha256": _sha256(value)}
    else:  # pragma: no cover - guarded by Python's Constant value contract
        raise DeltaValidationError("unsupported Python literal type")
    return _canonical_json(rendered)


def _symbol_record(
    blob: bytes,
    node: ast.AST,
    *,
    symbol_id: str,
    qualified_name: str,
    kind: str,
    occurrence: int,
    encoding: str,
    lines: Sequence[bytes],
    starts: Sequence[int],
    decorator: bool = False,
    import_spec: str | None = None,
) -> dict[str, Any]:
    start, end = _node_span(
        blob, node, encoding, lines, starts, decorator=decorator
    )
    literals: list[dict[str, Any]] = []
    literal_occurrence = 0
    for child in ast.walk(node):
        if not isinstance(child, ast.Constant):
            continue
        literal_occurrence += 1
        literal_start, literal_end = _node_span(
            blob, child, encoding, lines, starts
        )
        literals.append(
            {
                "occurrence": literal_occurrence,
                "physical_sha256": _sha256(blob[literal_start:literal_end]),
                "value_sha256": _sha256(_literal_value_payload(child.value)),
            }
        )
    literal_sha256 = _sha256(_canonical_json(literals))
    record: dict[str, Any] = {
        "symbol_id": symbol_id,
        "qualified_name": qualified_name,
        "kind": kind,
        "occurrence": occurrence,
        "byte_start": start,
        "byte_end": end,
        "physical_sha256": _sha256(blob[start:end]),
        "semantic_sha256": _sha256(
            ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
        ),
        "literal_sha256": literal_sha256,
        "literals": literals,
    }
    if import_spec is not None:
        record["import_spec"] = import_spec
    return record


def _assignment_names(
    node: ast.Assign | ast.AnnAssign | ast.AugAssign,
) -> tuple[str, ...]:
    targets: Sequence[ast.expr]
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = (node.target,)
    else:
        targets = (node.target,)
    names: list[str] = []

    def collect(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                collect(element)
        else:
            names.append(ast.dump(target, annotate_fields=False, include_attributes=False))

    for target in targets:
        collect(target)
    return tuple(names) or ("<anonymous>",)


def _import_spec(node: ast.Import | ast.ImportFrom) -> str:
    aliases = ",".join(
        f"{alias.name} as {alias.asname}" if alias.asname else alias.name
        for alias in node.names
    )
    if isinstance(node, ast.Import):
        return f"import:{aliases}"
    return f"from:{node.level}:{node.module or ''}:{aliases}"


def static_symbol_inventory(blob: bytes) -> tuple[dict[str, Any], ...]:
    """Return exact static symbols from Python bytes without importing them."""

    encoding, lines, starts = _encoding_and_lines(blob)
    try:
        source = blob.decode(encoding, errors="strict")
        tree = ast.parse(source, mode="exec", type_comments=True)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise DeltaValidationError("committed Python blob does not parse statically") from exc
    records: list[dict[str, Any]] = []
    occurrences: dict[tuple[str, str], int] = {}

    def add(
        node: ast.AST,
        qualified_name: str,
        kind: str,
        *,
        decorator: bool = False,
        import_spec: str | None = None,
    ) -> None:
        key = (kind, qualified_name)
        occurrence = occurrences.get(key, 0) + 1
        occurrences[key] = occurrence
        symbol_id = f"{kind}|{qualified_name}|{occurrence}"
        records.append(
            _symbol_record(
                blob,
                node,
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                kind=kind,
                occurrence=occurrence,
                encoding=encoding,
                lines=lines,
                starts=starts,
                decorator=decorator,
                import_spec=import_spec,
            )
        )

    def walk_statements(
        statements: Sequence[ast.stmt], scope: tuple[str, ...], in_function: bool
    ) -> None:
        def nested_statements(node: ast.AST) -> list[ast.stmt]:
            """Find statement children through non-statement AST containers.

            ``ExceptHandler`` (for both ``except`` and ``except*``) and
            ``match_case`` are AST nodes but are not ``ast.stmt`` subclasses.
            Stopping at immediate statement children would therefore omit
            their bodies.  Recursing only until the next statement keeps each
            statement in the inventory exactly once while covering every AST
            container, including future non-statement wrappers.
            """

            result: list[ast.stmt] = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    result.append(child)
                else:
                    result.extend(nested_statements(child))
            return result

        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*scope, statement.name))
                kind = "async_function" if isinstance(statement, ast.AsyncFunctionDef) else "function"
                add(statement, name, kind)
                for decorator_node in statement.decorator_list:
                    add(decorator_node, f"{name}.__decorator__", "decorator", decorator=True)
                walk_statements(statement.body, (*scope, statement.name), True)
                continue
            if isinstance(statement, ast.ClassDef):
                name = ".".join((*scope, statement.name))
                add(statement, name, "class")
                for decorator_node in statement.decorator_list:
                    add(decorator_node, f"{name}.__decorator__", "decorator", decorator=True)
                # A class body is its own static namespace even when the class
                # statement itself is nested in a function.
                walk_statements(statement.body, (*scope, statement.name), False)
                continue
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                scope_name = ".".join(scope) if scope else "$module"
                add(statement, f"{scope_name}.__import__", "import", import_spec=_import_spec(statement))
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and not in_function:
                scope_name = ".".join(scope) if scope else "$module"
                target = ",".join(_assignment_names(statement))
                add(statement, f"{scope_name}.{target}", "assignment")
            elif not scope:
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and statement is statements[0]
                ):
                    add(statement, "$module.__doc__", "module_docstring")
                else:
                    add(statement, f"$module.{type(statement).__name__}", "module_statement")
            child_statements = nested_statements(statement)
            if child_statements:
                walk_statements(child_statements, scope, in_function)

    walk_statements(tree.body, (), False)
    records.sort(key=lambda value: value["symbol_id"])
    if len({record["symbol_id"] for record in records}) != len(records):
        raise DeltaValidationError("static symbol inventory is not unique")
    return tuple(records)


def _symbol_changes(
    old_blob: bytes | None, new_blob: bytes | None
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    old_records = () if old_blob is None else static_symbol_inventory(old_blob)
    new_records = () if new_blob is None else static_symbol_inventory(new_blob)
    old_map = {record["symbol_id"]: record for record in old_records}
    new_map = {record["symbol_id"]: record for record in new_records}
    changed: list[dict[str, Any]] = []
    changed_ids: list[str] = []

    def identity(record: Mapping[str, Any] | None) -> Any:
        if record is None:
            return None
        # Physical offsets are evidence, not semantic identity: an insertion
        # earlier in the file may move a byte-identical symbol.  Its literal
        # byte slice, AST, imports, and physical byte slice must still agree.
        return {
            key: value
            for key, value in record.items()
            if key not in {"byte_start", "byte_end"}
        }

    for symbol_id in sorted(set(old_map) | set(new_map)):
        before = old_map.get(symbol_id)
        after = new_map.get(symbol_id)
        if identity(before) == identity(after):
            continue
        changed_ids.append(symbol_id)
        changed.append({"symbol_id": symbol_id, "before": before, "after": after})
    changed_id_set = set(changed_ids)
    imports = tuple(
        sorted(
            record["import_spec"]
            for record in new_records
            if record["kind"] == "import"
            and record["symbol_id"] in changed_id_set
        )
    )
    return changed, tuple(changed_ids), imports


def _sha256_literal_assignment(
    blob: bytes, name: str
) -> tuple[str, int, int]:
    """Read one canonical double-quoted lowercase-SHA literal assignment.

    The returned span covers only the 64 digest bytes, not the assignment,
    whitespace, quote characters, or newline.  Masked test-blob hashing can
    therefore leave every non-anchor byte bound.
    """

    if not isinstance(name, str) or not name.isidentifier():
        raise DeltaValidationError("literal assignment name is invalid")
    encoding, lines, starts = _encoding_and_lines(blob)
    try:
        tree = ast.parse(blob.decode(encoding, errors="strict"), mode="exec")
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise DeltaValidationError("literal-assignment blob does not parse") from exc
    matches: list[ast.Assign] = []
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ):
            matches.append(statement)
    if len(matches) != 1:
        raise DeltaValidationError(f"literal assignment {name!r} must occur exactly once")
    node = matches[0]
    if not isinstance(node.value, ast.Constant) or type(node.value.value) is not str:
        raise DeltaValidationError(
            f"literal assignment {name!r} must be one plain string constant"
        )
    value = node.value.value
    if not _SHA256_RE.fullmatch(value):
        raise DeltaValidationError(f"literal assignment {name!r} is not lowercase SHA-256")
    value_start, value_end = _node_span(
        blob, node.value, encoding, lines, starts
    )
    literal = blob[value_start:value_end]
    expected_literal = b'"' + value.encode("ascii") + b'"'
    if literal != expected_literal:
        raise DeltaValidationError(
            f"literal assignment {name!r} is not canonical double-quoted raw ASCII"
        )
    return value, value_start + 1, value_end - 1


def _masked_sha256_literal_assignment_sha256(
    blob: bytes, name: str
) -> tuple[str, str]:
    value, start, end = _sha256_literal_assignment(blob, name)
    if end - start != 64:  # pragma: no cover - implied by the regex
        raise DeltaValidationError("SHA-256 anchor span is not exactly 64 bytes")
    return _sha256(blob[:start] + (b"0" * 64) + blob[end:]), value


def _position_free(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _position_free(item)
            for key, item in value.items()
            if key not in {"byte_start", "byte_end"}
        }
    if isinstance(value, (list, tuple)):
        return [_position_free(item) for item in value]
    return value


def _counterpart_change_evidence_sha256(
    changes: Sequence[Mapping[str, Any]],
) -> str:
    """Bind exact before/after symbol content without position-only noise."""

    return _sha256(_canonical_json(_position_free(list(changes))))


def _mechanical_counterpart_blob(
    blob: bytes, expected_counts: Sequence[int]
) -> tuple[bytes, list[dict[str, Any]]]:
    if len(expected_counts) != len(_COUNTERPART_MECHANICAL_REPLACEMENTS):
        raise DeltaValidationError("counterpart mechanical count vector is incomplete")
    transformed = blob
    evidence: list[dict[str, Any]] = []
    for (before_text, after_text), expected_count in zip(
        _COUNTERPART_MECHANICAL_REPLACEMENTS, expected_counts, strict=True
    ):
        before = before_text.encode("ascii")
        after = after_text.encode("ascii")
        observed_count = transformed.count(before)
        if observed_count != expected_count:
            raise DeltaValidationError(
                "pinned counterpart mechanical replacement count changed"
            )
        transformed = transformed.replace(before, after)
        evidence.append(
            {
                "before": before_text,
                "after": after_text,
                "count": observed_count,
            }
        )
    return transformed, evidence


def _counterpart_evidence(
    repo: Path,
    candidate_path: str,
    candidate_blob: bytes,
    rule: CounterpartRule,
    pins: DeltaPins,
    implementation_commit: str,
    *,
    git_binary: str,
) -> dict[str, Any]:
    """Prove one committed v3.8 blob against its pinned v3.3 counterpart."""

    candidate_sha256 = _sha256(candidate_blob)
    binding: dict[str, Any]
    if rule.expected_candidate_sha256 is not None:
        if candidate_sha256 != rule.expected_candidate_sha256:
            raise DeltaValidationError(f"{candidate_path}: complete candidate blob drift")
        binding = {"kind": "complete_blob_sha256", "sha256": candidate_sha256}
    elif rule.redacted_literal_assignment is not None:
        redacted_sha256, literal_value = _masked_sha256_literal_assignment_sha256(
            candidate_blob, rule.redacted_literal_assignment
        )
        if redacted_sha256 != rule.expected_redacted_candidate_sha256:
            raise DeltaValidationError(f"{candidate_path}: anchor-masked candidate blob drift")
        if not isinstance(literal_value, str) or not _SHA256_RE.fullmatch(literal_value):
            raise DeltaValidationError(f"{candidate_path}: external anchor literal is malformed")
        binding = {
            "kind": "anchor_literal_masked_blob_sha256",
            "assignment": rule.redacted_literal_assignment,
            "redacted_sha256": redacted_sha256,
            "literal_value": literal_value,
        }
    else:
        anchor_path = rule.external_sha256_anchor_path
        anchor_name = rule.external_sha256_anchor_assignment
        if anchor_path is None or anchor_name is None:  # pragma: no cover - validated rule
            raise DeltaValidationError(f"{candidate_path}: external anchor is incomplete")
        anchor_blob = _cat_blob(
            repo, implementation_commit, path=anchor_path, git_binary=git_binary
        )
        anchor_value, _, _ = _sha256_literal_assignment(anchor_blob, anchor_name)
        if (
            not isinstance(anchor_value, str)
            or not _SHA256_RE.fullmatch(anchor_value)
            or anchor_value != candidate_sha256
        ):
            raise DeltaValidationError(f"{candidate_path}: external blob anchor mismatch")
        binding = {
            "kind": "external_literal_sha256",
            "anchor_path": anchor_path,
            "anchor_assignment": anchor_name,
            "sha256": candidate_sha256,
        }

    counterpart_blob = _cat_blob(
        repo,
        pins.implementation_base_commit,
        path=rule.counterpart_path,
        git_binary=git_binary,
    )
    transformed, replacements = _mechanical_counterpart_blob(
        counterpart_blob, rule.expected_mechanical_counts
    )
    changes, symbol_ids, import_specs = _symbol_changes(transformed, candidate_blob)
    changes_by_id = {change["symbol_id"]: change for change in changes}
    excluded: list[dict[str, Any]] = []
    for symbol_id in rule.excluded_change_symbol_ids:
        change = changes_by_id.pop(symbol_id, None)
        if change is None:
            raise DeltaValidationError(
                f"{candidate_path}: expected external-anchor data symbol is missing"
            )
        excluded.append(change)
    admitted_changes = [
        change for change in changes if change["symbol_id"] in changes_by_id
    ]
    admitted_ids = tuple(change["symbol_id"] for change in admitted_changes)
    if admitted_ids != rule.expected_changed_symbol_ids:
        raise DeltaValidationError(
            f"{candidate_path}: unknown or missing counterpart symbol; observed={admitted_ids}"
        )
    if import_specs != rule.expected_changed_import_specs:
        raise DeltaValidationError(
            f"{candidate_path}: counterpart import inventory drift; observed={import_specs}"
        )
    evidence_sha256 = _counterpart_change_evidence_sha256(admitted_changes)
    if evidence_sha256 != rule.expected_changed_symbol_evidence_sha256:
        raise DeltaValidationError(f"{candidate_path}: counterpart symbol content drift")
    return {
        "candidate_path": candidate_path,
        "candidate_blob_sha256": candidate_sha256,
        "candidate_binding": binding,
        "counterpart_commit": pins.implementation_base_commit,
        "counterpart_path": rule.counterpart_path,
        "counterpart_blob_oid": _git_blob_oid(counterpart_blob),
        "counterpart_blob_sha256": _sha256(counterpart_blob),
        "mechanical_replacements": replacements,
        "mechanically_transformed_blob_sha256": _sha256(transformed),
        "changed_symbol_ids": list(admitted_ids),
        "changed_symbol_count": len(admitted_ids),
        "changed_symbol_evidence_sha256": evidence_sha256,
        "changed_symbols": admitted_changes,
        "excluded_anchor_data_symbols": excluded,
        "changed_import_specs": list(import_specs),
    }


def _test_function_records(
    path: str, changes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not path.startswith("tests/"):
        return []
    bound: list[dict[str, Any]] = []
    for change in changes:
        before = change.get("before")
        after = change.get("after")
        record = after if isinstance(after, Mapping) else before
        if not isinstance(record, Mapping) or record.get("kind") not in {"function", "async_function"}:
            continue
        qualified_name = record.get("qualified_name")
        if not isinstance(qualified_name, str) or not qualified_name.split(".")[-1].startswith("test_"):
            continue

        def hashes(value: Any) -> dict[str, str] | None:
            if not isinstance(value, Mapping):
                return None
            return {
                "physical_sha256": value["physical_sha256"],
                "literal_sha256": value["literal_sha256"],
                "semantic_sha256": value["semantic_sha256"],
            }

        bound.append(
            {
                "path": path,
                "symbol_id": record["symbol_id"],
                "before": hashes(before),
                "after": hashes(after),
            }
        )
    return bound


def _frozen_symbol_evidence(
    repo: Path,
    pins: DeltaPins,
    frozen: Sequence[FrozenSymbolPin],
    implementation_commit: str,
    *,
    git_binary: str,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for pin in sorted(frozen, key=lambda value: (value.path, value.symbol_id)):
        records: list[dict[str, Any]] = []
        for label, revision in (
            ("prereg", pins.prereg_commit),
            ("implementation", implementation_commit),
        ):
            cache_key = (revision, pin.path)
            if cache_key not in cache:
                blob = _cat_blob(
                    repo, revision, path=pin.path, git_binary=git_binary
                )
                cache[cache_key] = {
                    record["symbol_id"]: record for record in static_symbol_inventory(blob)
                }
            record = cache[cache_key].get(pin.symbol_id)
            if record is None:
                raise DeltaValidationError(f"frozen symbol missing: {pin.path}:{pin.symbol_id}")
            for field in ("physical_sha256", "literal_sha256", "semantic_sha256"):
                if record[field] != getattr(pin, field):
                    raise DeltaValidationError(f"frozen symbol drift: {pin.path}:{pin.symbol_id}")
            records.append(record)
        evidence.append({**pin.as_json(), "prereg_equals_implementation": True})
    return evidence


def build_delta_manifest(
    repo: str | os.PathLike[str],
    implementation_commit: str,
    implementation_tree: str,
    *,
    pins: DeltaPins = DEFAULT_PINS,
    allow_spec: AllowSpec,
    git_binary: str = "git",
) -> dict[str, Any]:
    """Build a deterministic manifest entirely from immutable Git objects."""

    root = Path(repo).resolve(strict=True)
    if not root.is_dir():
        raise DeltaValidationError("repository root must be a directory")
    pins.validate()
    allow_spec.validate()
    if not _OID_RE.fullmatch(implementation_commit) or not _OID_RE.fullmatch(implementation_tree):
        raise DeltaValidationError("implementation commit/tree must be literal SHA-1 ids")
    _assert_commit_object(
        root, pins.prereg_commit, label="prereg", git_binary=git_binary
    )
    _assert_commit_object(
        root,
        pins.implementation_base_commit,
        label="implementation base",
        git_binary=git_binary,
    )
    _assert_commit_object(
        root, implementation_commit, label="implementation", git_binary=git_binary
    )
    _assert_tree_pin(
        root, pins.prereg_commit, pins.prereg_tree, label="prereg", git_binary=git_binary
    )
    _assert_tree_pin(
        root,
        pins.implementation_base_commit,
        pins.implementation_base_tree,
        label="implementation base",
        git_binary=git_binary,
    )
    _assert_tree_pin(
        root,
        implementation_commit,
        implementation_tree,
        label="implementation",
        git_binary=git_binary,
    )
    prereg_doc = _cat_blob(
        root, pins.prereg_tree, path=pins.prereg_doc_path, git_binary=git_binary
    )
    if _git_blob_oid(prereg_doc) != pins.prereg_doc_blob or _sha256(prereg_doc) != pins.prereg_doc_sha256:
        raise DeltaValidationError("preregistration document blob pin failed")

    inherited_diff = _raw_diff(
        root,
        pins.implementation_base_commit,
        pins.prereg_commit,
        git_binary=git_binary,
    )
    implementation_diff = _raw_diff(
        root, pins.prereg_commit, implementation_commit, git_binary=git_binary
    )
    parent_to_implementation = _raw_diff(
        root,
        pins.implementation_base_commit,
        implementation_commit,
        git_binary=git_binary,
    )
    inherited_paths = {entry.path for entry in inherited_diff}
    v38_paths = {entry.path for entry in implementation_diff}
    overlap = inherited_paths & v38_paths
    if overlap:
        raise DeltaValidationError(
            "inherited-at-prereg blobs changed in implementation: " + ",".join(sorted(overlap))
        )
    if v38_paths != set(allow_spec.path_rules):
        unknown = sorted(v38_paths - set(allow_spec.path_rules))
        missing = sorted(set(allow_spec.path_rules) - v38_paths)
        raise DeltaValidationError(f"implementation path inventory mismatch; unknown={unknown}; missing={missing}")

    inherited_json: list[dict[str, Any]] = []
    inherited_states: dict[str, dict[str, Any]] = {}
    for entry in inherited_diff:
        entry_json, _, _ = _entry_with_hashes(root, entry, git_binary=git_binary)
        entry_json["classification"] = "inherited_at_prereg_unchanged"
        inherited_json.append(entry_json)
        inherited_states[entry.path] = entry_json

    changes_json: list[dict[str, Any]] = []
    counterpart_json: list[dict[str, Any]] = []
    all_test_functions: list[dict[str, Any]] = []
    for entry in implementation_diff:
        rule = allow_spec.path_rules[entry.path]
        if entry.status != rule.status:
            raise DeltaValidationError(f"{entry.path}: status differs from allow spec")
        if entry.status == "D":
            raise DeltaValidationError(f"{entry.path}: implementation deletion is forbidden")
        entry_json, old_blob, new_blob = _entry_with_hashes(
            root, entry, git_binary=git_binary
        )
        if new_blob is None:
            raise DeltaValidationError("implementation path has no committed after blob")
        symbol_changes, symbol_ids, import_specs = _symbol_changes(old_blob, new_blob)
        if rule.expected_import_inventory_sha256 is None:
            imports_match = import_specs == rule.expected_import_specs
        else:
            imports_match = (
                len(import_specs) == rule.expected_import_count
                and _sha256(_canonical_json(list(import_specs)))
                == rule.expected_import_inventory_sha256
            )
        if not imports_match:
            raise DeltaValidationError(
                f"{entry.path}: unknown or missing import; observed={import_specs}"
            )
        if rule.expected_symbol_inventory_sha256 is None:
            symbols_match = symbol_ids == rule.expected_symbol_ids
        else:
            symbols_match = (
                len(symbol_ids) == rule.expected_symbol_count
                and _sha256(_canonical_json(list(symbol_ids)))
                == rule.expected_symbol_inventory_sha256
            )
        if not symbols_match:
            raise DeltaValidationError(
                f"{entry.path}: unknown or missing static symbol; observed={symbol_ids}"
            )
        entry_json["symbols"] = symbol_changes
        entry_json["classification"] = "v38_implementation_delta"
        changes_json.append(entry_json)
        all_test_functions.extend(_test_function_records(entry.path, symbol_changes))
        counterpart_rule = allow_spec.counterpart_rules.get(entry.path)
        if counterpart_rule is not None:
            counterpart_json.append(
                _counterpart_evidence(
                    root,
                    entry.path,
                    new_blob,
                    counterpart_rule,
                    pins,
                    implementation_commit,
                    git_binary=git_binary,
                )
            )

    classified_parent_paths: list[dict[str, Any]] = []
    for entry in parent_to_implementation:
        observed, _, _ = _entry_with_hashes(root, entry, git_binary=git_binary)
        if entry.path in inherited_paths:
            expected = inherited_states[entry.path]
            comparable = {
                key: observed[key]
                for key in (
                    "path",
                    "status",
                    "old_mode",
                    "new_mode",
                    "old_oid",
                    "new_oid",
                    "old_sha256",
                    "new_sha256",
                )
            }
            expected_comparable = {key: expected[key] for key in comparable}
            if comparable != expected_comparable:
                raise DeltaValidationError(f"{entry.path}: inherited blob proof changed")
            classification = "inherited_at_prereg_unchanged"
        elif entry.path in v38_paths:
            classification = "v38_implementation_delta"
        else:
            raise DeltaValidationError(f"{entry.path}: unclassified implementation-base delta")
        classified_parent_paths.append({**observed, "classification": classification})
    if {entry["path"] for entry in classified_parent_paths} != inherited_paths | v38_paths:
        raise DeltaValidationError("implementation-base classification is not exhaustive")

    frozen_evidence = _frozen_symbol_evidence(
        root,
        pins,
        allow_spec.frozen_symbols,
        implementation_commit,
        git_binary=git_binary,
    )

    # This ancestry and strict-addition gate is deliberately late.  The
    # detailed blob/path checks above retain their precise fail-closed
    # diagnostics for hostile synthetic commits, while a structurally valid
    # manifest still has to prove the exact preregistration history promised
    # before implementation.
    prereg_commit_payload = _git(
        root,
        ("cat-file", "commit", pins.prereg_commit),
        git_binary=git_binary,
    )
    prereg_headers, prereg_separator, _ = prereg_commit_payload.partition(b"\n\n")
    if prereg_separator != b"\n\n":
        raise DeltaValidationError("prereg commit headers are malformed")
    prereg_parents = [
        line for line in prereg_headers.split(b"\n") if line.startswith(b"parent ")
    ]
    if prereg_parents != [
        b"parent " + pins.implementation_base_commit.encode("ascii")
    ]:
        raise DeltaValidationError(
            "prereg commit must have the exact implementation-base parent"
        )

    implementation_commit_payload = _git(
        root,
        ("cat-file", "commit", implementation_commit),
        git_binary=git_binary,
    )
    implementation_headers, implementation_separator, _ = (
        implementation_commit_payload.partition(b"\n\n")
    )
    if implementation_separator != b"\n\n":
        raise DeltaValidationError("implementation commit headers are malformed")
    implementation_parents = [
        line
        for line in implementation_headers.split(b"\n")
        if line.startswith(b"parent ")
    ]
    if implementation_parents != [b"parent " + pins.prereg_commit.encode("ascii")]:
        raise DeltaValidationError(
            "implementation commit must have the exact prereg parent"
        )

    if pins == DEFAULT_PINS:
        if len(inherited_diff) != 1:
            raise DeltaValidationError(
                "prereg commit must add exactly its one preregistration document"
            )
        prereg_entry = inherited_diff[0]
        if (
            prereg_entry.path != pins.prereg_doc_path
            or prereg_entry.status != "A"
            or prereg_entry.old_mode != "000000"
            or prereg_entry.new_mode != _REGULAR_MODE
            or prereg_entry.old_oid != _ZERO_OID
            or prereg_entry.new_oid == _ZERO_OID
        ):
            raise DeltaValidationError(
                "prereg commit is not one regular document addition"
            )
        if len(implementation_diff) != 12 or any(
            entry.status != "A"
            or entry.old_mode != "000000"
            or entry.new_mode != _REGULAR_MODE
            or entry.old_oid != _ZERO_OID
            or entry.new_oid == _ZERO_OID
            for entry in implementation_diff
        ):
            raise DeltaValidationError(
                "implementation commit is not twelve strict regular additions"
            )

    body: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "observation_contract": {
            "worktree_reads": 0,
            "source_imports_or_execution": 0,
            "generation_phase": "after_implementation_commit",
            "git_commands": [
                "diff-tree --raw -r -z --no-renames --no-abbrev",
                "cat-file blob",
                "cat-file -t",
                "cat-file commit",
            ],
        },
        "pins": pins.as_json(),
        "implementation": {
            "commit": implementation_commit,
            "tree": implementation_tree,
        },
        "allow_spec_sha256": allow_spec.sha256,
        "prereg_document": {
            "path": pins.prereg_doc_path,
            "blob_oid": pins.prereg_doc_blob,
            "literal_sha256": pins.prereg_doc_sha256,
        },
        "inherited_at_prereg": inherited_json,
        "implementation_delta": changes_json,
        # Legacy field name retained for manifest-schema and focused-test
        # compatibility; the bound rules below target v3.4 counterparts.
        "committed_v32_counterpart_comparisons": counterpart_json,
        "implementation_base_classification": classified_parent_paths,
        "frozen_symbols": frozen_evidence,
        "bound_test_functions": sorted(
            all_test_functions, key=lambda value: (value["path"], value["symbol_id"])
        ),
    }
    body["manifest_sha256"] = _sha256(_canonical_json(body))
    return body


def serialize_delta_manifest(manifest: Mapping[str, Any]) -> bytes:
    """Serialize one manifest in its only accepted canonical representation."""

    return _canonical_json(dict(manifest)) + b"\n"


def _strict_json(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload or payload.startswith(b"\xef\xbb\xbf"):
        raise DeltaValidationError("manifest must be nonempty canonical UTF-8 bytes")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise DeltaValidationError("manifest contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise DeltaValidationError(f"manifest contains nonfinite JSON number {value}")

    try:
        decoded = payload.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeltaValidationError("manifest JSON is malformed") from exc
    if not isinstance(parsed, dict):
        raise DeltaValidationError("manifest root must be an object")
    if serialize_delta_manifest(parsed) != payload:
        raise DeltaValidationError("manifest bytes are not canonical")
    return parsed


def validate_delta_manifest(
    repo: str | os.PathLike[str],
    payload: bytes,
    *,
    expected_implementation_commit: str,
    expected_implementation_tree: str,
    pins: DeltaPins = DEFAULT_PINS,
    allow_spec: AllowSpec,
    git_binary: str = "git",
) -> dict[str, Any]:
    """Validate self-hash, trusted commit pin, and committed-blob recomputation."""

    parsed = _strict_json(payload)
    supplied_hash = parsed.get("manifest_sha256")
    if not isinstance(supplied_hash, str) or not _SHA256_RE.fullmatch(supplied_hash):
        raise DeltaValidationError("manifest self-hash is missing or malformed")
    without_hash = dict(parsed)
    del without_hash["manifest_sha256"]
    if _sha256(_canonical_json(without_hash)) != supplied_hash:
        raise DeltaValidationError("manifest self-hash mismatch")
    implementation = parsed.get("implementation")
    if not isinstance(implementation, dict):
        raise DeltaValidationError("manifest implementation identity is missing")
    commit = implementation.get("commit")
    tree = implementation.get("tree")
    if not isinstance(commit, str) or not isinstance(tree, str):
        raise DeltaValidationError("manifest implementation identity is malformed")
    if (
        commit != expected_implementation_commit
        or tree != expected_implementation_tree
    ):
        raise DeltaValidationError("manifest is not for the expected implementation commit/tree")
    expected = build_delta_manifest(
        repo,
        expected_implementation_commit,
        expected_implementation_tree,
        pins=pins,
        allow_spec=allow_spec,
        git_binary=git_binary,
    )
    if parsed != expected:
        raise DeltaValidationError("manifest differs from committed-blob recomputation")
    return parsed


# This table binds the exact twelve-file v3.8 projection derived from v3.7.
_V37_CLONE_PATH_RULES = {
    "agent_benchmark/sec_gemma_lean_v38_acquisition.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="ed557eb3a128ec362f4ad6974276c09299202b791a995a79d1a2f7441eeb8143",
        expected_symbol_count=216,
        expected_import_inventory_sha256="cd8d9c3fa9175929f7427d3b6d393f6685f264ef7666a199b0a47b86a12f7e1f",
        expected_import_count=29,
    ),
    "agent_benchmark/sec_gemma_lean_v38_delta.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="073e3042784a9e33eff13983695a1efba09b5023671ccf120f24b5b72f51290c",
        expected_symbol_count=145,
        expected_import_inventory_sha256="2a2dcd798027605f5dd9bb7907853c2a842052f926d92f3155908c7b4e2bafaa",
        expected_import_count=13,
    ),
    "agent_benchmark/sec_gemma_lean_v38_journal.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="f7476b2e57b55263b474c70a5b0fa11244653c502dfb5b00492976b09ae9a422",
        expected_symbol_count=228,
        expected_import_inventory_sha256="f0059f257f0add60f4bad87d53107921760a387ffc244d6b053f4a9c1b589e11",
        expected_import_count=16,
    ),
    "agent_benchmark/sec_gemma_lean_v38_preflight.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="b36e7d390bf0fbde5c7e02a886c0cb27b2439f19929f16c713a857e24787834c",
        expected_symbol_count=144,
        expected_import_inventory_sha256="fbf069973a07c59f9c0b4dc83daea919fbb6e304dfef07941e191031fc85fd0c",
        expected_import_count=25,
    ),
    "agent_benchmark/sec_gemma_lean_v38_source.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="c4c709948015bcf1623df8c2530b5072ce70d37594311849663515b618194d64",
        expected_symbol_count=333,
        expected_import_inventory_sha256="44a36f9953db408c55d2ae87c972b447792efe004fcb1371c6ad77f42086549f",
        expected_import_count=14,
    ),
    "agent_benchmark/sec_gemma_lean_v38_transport.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="7d461ffac7afcf1ea3e127b0ad26d8a08e021aa917000fb7fb273e45668a3d2a",
        expected_symbol_count=199,
        expected_import_inventory_sha256="6029a385a66bb8335e6ce9b53bde7c7c82dd24414da4ff9de25ab665b993a65d",
        expected_import_count=19,
    ),
    "tests/test_sec_gemma_lean_v38_acquisition.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="df1260404510debb93c28f305b187d193fd756ad553610cb58deb927cec81aa4",
        expected_symbol_count=155,
        expected_import_inventory_sha256="f7023b0c926558573cd26f86208d315d22bdbd4d713e374bba90933be4b5a490",
        expected_import_count=21,
    ),
    "tests/test_sec_gemma_lean_v38_delta.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="497a68daaf9874f5f1a7fcfe2b5214fe1b5aa0c1631fba06884a85bd86e8dc2e",
        expected_symbol_count=55,
        expected_import_inventory_sha256="f79992aae265307d579f39d9650f630e39fa60e03e4098b6f72cb49ac5e9a4f9",
        expected_import_count=10,
    ),
    "tests/test_sec_gemma_lean_v38_journal.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="a07aebd9025a1a78956b64381126eb7e4d78d9d94b384e40c07cb412c1c43ea8",
        expected_symbol_count=60,
        expected_import_inventory_sha256="3ed4a36e210c252ea5bfebb66083f695a4442e088ad9d8325c333f82eab00630",
        expected_import_count=9,
    ),
    "tests/test_sec_gemma_lean_v38_preflight.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="72ee992dad44ae40382e1e546a0c1459cb32ea881b82ca0055481148f36a0dfb",
        expected_symbol_count=64,
        expected_import_inventory_sha256="3ca55f4f2d16ac47fade84660f813ba87e22b545d6fa7daa7c714407f33e154c",
        expected_import_count=11,
    ),
    "tests/test_sec_gemma_lean_v38_source.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="b436af0daeccfd43b67ae1a3eac6614786bc0506b4fa830ab4a5d63184b6c2f9",
        expected_symbol_count=79,
        expected_import_inventory_sha256="8f39411737c5abb0a1bc06961a4a8667a9c73ee227a5c18056ad0badd7397a11",
        expected_import_count=12,
    ),
    "tests/test_sec_gemma_lean_v38_transport.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="404fcba807429dc521e6bcbea906f1adcb97768f20a95f5c27a8e84d2917a5f5",
        expected_symbol_count=101,
        expected_import_inventory_sha256="24f7eb09593b43f3b6e27840a9d1f801b6721b6e5abbaa0e319cf4d7ad238bf2",
        expected_import_count=16,
    ),
}
# These are exactly the twelve additive v3.8 paths.  No shared-file
# modification is admitted when DEFAULT_ALLOW_SPEC is built.
_V38_IMPLEMENTATION_PATHS = (
    "agent_benchmark/sec_gemma_lean_v38_acquisition.py",
    "agent_benchmark/sec_gemma_lean_v38_delta.py",
    "agent_benchmark/sec_gemma_lean_v38_journal.py",
    "agent_benchmark/sec_gemma_lean_v38_preflight.py",
    "agent_benchmark/sec_gemma_lean_v38_source.py",
    "agent_benchmark/sec_gemma_lean_v38_transport.py",
    "tests/test_sec_gemma_lean_v38_acquisition.py",
    "tests/test_sec_gemma_lean_v38_delta.py",
    "tests/test_sec_gemma_lean_v38_journal.py",
    "tests/test_sec_gemma_lean_v38_preflight.py",
    "tests/test_sec_gemma_lean_v38_source.py",
    "tests/test_sec_gemma_lean_v38_transport.py",
)

# This is inert, reviewable contract data.  Twelve ordinary candidates are
# bound by complete raw SHA-256.  The delta module and its focused test use
# the one-way external-anchor construction documented by CounterpartRule.
_COUNTERPART_RULE_DATA = {'agent_benchmark/sec_gemma_lean_v38_acquisition.py': {'counterpart_path': 'agent_benchmark/sec_gemma_lean_v37_acquisition.py',
                                                       'excluded_change_symbol_ids': (),
                                                       'expected_candidate_sha256': '8ba5b3dee285a5e12e631c5d5e8fbafce1a66d1c81b4c5736371ac8ceffbb3f6',
                                                       'expected_changed_import_specs': (),
                                                       'expected_changed_symbol_evidence_sha256': '3b7a76502cda4ab6da4e88585c4ee7d234b7d86fe80096384126236b88ed04d3',
                                                       'expected_changed_symbol_ids': ('class|DiskBackedSecAcquisition|1',
                                                                                       'function|DiskBackedSecAcquisition._run_locked|1'),
                                                       'expected_mechanical_counts': (12,
                                                                                      271,
                                                                                      0,
                                                                                      1,
                                                                                      2,
                                                                                      7,
                                                                                      2),
                                                       'expected_redacted_candidate_sha256': None,
                                                       'external_sha256_anchor_assignment': None,
                                                       'external_sha256_anchor_path': None,
                                                       'redacted_literal_assignment': None},
 'agent_benchmark/sec_gemma_lean_v38_delta.py': {'counterpart_path': 'agent_benchmark/sec_gemma_lean_v37_delta.py',
                                                 'excluded_change_symbol_ids': ('assignment|$module._COUNTERPART_RULE_DATA|1',),
                                                 'expected_candidate_sha256': None,
                                                 'expected_changed_import_specs': (),
                                                 'expected_changed_symbol_evidence_sha256': 'a6a1c2dcbc345bbed75ebad7a88c96b87e3ba05559aa4001083626eb68739fea',
                                                 'expected_changed_symbol_ids': ('assignment|$module.IMPLEMENTATION_BASE_COMMIT|1',
                                                                                 'assignment|$module.IMPLEMENTATION_BASE_TREE|1',
                                                                                 'assignment|$module.PREREG_COMMIT|1',
                                                                                 'assignment|$module.PREREG_DOC_BLOB|1',
                                                                                 'assignment|$module.PREREG_DOC_SHA256|1',
                                                                                 'assignment|$module.PREREG_TREE|1',
                                                                                 'assignment|$module._COUNTERPART_MECHANICAL_REPLACEMENTS|1',
                                                                                 'assignment|$module._DEFAULT_EXISTING_PATH_RULES|1',
                                                                                 'assignment|$module._V36_CLONE_PATH_RULES|1',
                                                                                 'assignment|$module._V37_CLONE_PATH_RULES|1'),
                                                 'expected_mechanical_counts': (43,
                                                                                1,
                                                                                11,
                                                                                11,
                                                                                2,
                                                                                1,
                                                                                7),
                                                 'expected_redacted_candidate_sha256': None,
                                                 'external_sha256_anchor_assignment': 'V38_DELTA_MODULE_SHA256',
                                                 'external_sha256_anchor_path': 'tests/test_sec_gemma_lean_v38_delta.py',
                                                 'redacted_literal_assignment': None},
 'agent_benchmark/sec_gemma_lean_v38_journal.py': {'counterpart_path': 'agent_benchmark/sec_gemma_lean_v37_journal.py',
                                                   'excluded_change_symbol_ids': (),
                                                   'expected_candidate_sha256': 'c433fcfa0b201546ed22bb7fed550c21eaf780170fc55336a82839df74d654b6',
                                                   'expected_changed_import_specs': (),
                                                   'expected_changed_symbol_evidence_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                                                   'expected_changed_symbol_ids': (),
                                                   'expected_mechanical_counts': (0,
                                                                                  105,
                                                                                  0,
                                                                                  0,
                                                                                  0,
                                                                                  3,
                                                                                  1),
                                                   'expected_redacted_candidate_sha256': None,
                                                   'external_sha256_anchor_assignment': None,
                                                   'external_sha256_anchor_path': None,
                                                   'redacted_literal_assignment': None},
 'agent_benchmark/sec_gemma_lean_v38_preflight.py': {'counterpart_path': 'agent_benchmark/sec_gemma_lean_v37_preflight.py',
                                                     'excluded_change_symbol_ids': (),
                                                     'expected_candidate_sha256': '07b1610a3c2eabfab1bc4edcc4c5a3b198a44cf5a4c311f006401ae8a63a5ad5',
                                                     'expected_changed_import_specs': (),
                                                     'expected_changed_symbol_evidence_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                                                     'expected_changed_symbol_ids': (),
                                                     'expected_mechanical_counts': (4,
                                                                                    142,
                                                                                    0,
                                                                                    0,
                                                                                    2,
                                                                                    12,
                                                                                    3),
                                                     'expected_redacted_candidate_sha256': None,
                                                     'external_sha256_anchor_assignment': None,
                                                     'external_sha256_anchor_path': None,
                                                     'redacted_literal_assignment': None},
 'agent_benchmark/sec_gemma_lean_v38_source.py': {'counterpart_path': 'agent_benchmark/sec_gemma_lean_v37_source.py',
                                                  'excluded_change_symbol_ids': (),
                                                  'expected_candidate_sha256': '8f624df2b28425b8e31db81997aeb0387d3c19418c0781e3b04fca66830d0f04',
                                                  'expected_changed_import_specs': (),
                                                  'expected_changed_symbol_evidence_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                                                  'expected_changed_symbol_ids': (),
                                                  'expected_mechanical_counts': (0,
                                                                                 318,
                                                                                 0,
                                                                                 16,
                                                                                 0,
                                                                                 0,
                                                                                 9),
                                                  'expected_redacted_candidate_sha256': None,
                                                  'external_sha256_anchor_assignment': None,
                                                  'external_sha256_anchor_path': None,
                                                  'redacted_literal_assignment': None},
 'agent_benchmark/sec_gemma_lean_v38_transport.py': {'counterpart_path': 'agent_benchmark/sec_gemma_lean_v37_transport.py',
                                                     'excluded_change_symbol_ids': (),
                                                     'expected_candidate_sha256': 'ec60f3868b1f5f3f4f638215e888d35bd3cae97ca442dedb3de4d28249252a26',
                                                     'expected_changed_import_specs': (),
                                                     'expected_changed_symbol_evidence_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                                                     'expected_changed_symbol_ids': (),
                                                     'expected_mechanical_counts': (5,
                                                                                    140,
                                                                                    0,
                                                                                    0,
                                                                                    0,
                                                                                    2,
                                                                                    3),
                                                     'expected_redacted_candidate_sha256': None,
                                                     'external_sha256_anchor_assignment': None,
                                                     'external_sha256_anchor_path': None,
                                                     'redacted_literal_assignment': None},
 'tests/test_sec_gemma_lean_v38_acquisition.py': {'counterpart_path': 'tests/test_sec_gemma_lean_v37_acquisition.py',
                                                  'excluded_change_symbol_ids': (),
                                                  'expected_candidate_sha256': '48f247952c7db6eede09b5b32b918b419e6c2e8f641028872907f29df6e9c2c8',
                                                  'expected_changed_import_specs': (),
                                                  'expected_changed_symbol_evidence_sha256': 'e81a3685164f185209b59251d7b9dc8b79246d6f4c3a37170aadc6948c6550be',
                                                  'expected_changed_symbol_ids': ('class|_FakeSource|1',
                                                                                  'function|_FakeSource.build_compact_checkpoint|1'),
                                                  'expected_mechanical_counts': (11,
                                                                                 38,
                                                                                 0,
                                                                                 0,
                                                                                 1,
                                                                                 1,
                                                                                 0),
                                                  'expected_redacted_candidate_sha256': None,
                                                  'external_sha256_anchor_assignment': None,
                                                  'external_sha256_anchor_path': None,
                                                  'redacted_literal_assignment': None},
 'tests/test_sec_gemma_lean_v38_delta.py': {'counterpart_path': 'tests/test_sec_gemma_lean_v37_delta.py',
                                            'excluded_change_symbol_ids': ('assignment|$module.V38_DELTA_MODULE_SHA256|1',),
                                            'expected_candidate_sha256': None,
                                            'expected_changed_import_specs': (),
                                            'expected_changed_symbol_evidence_sha256': '1eeace6f2626bb7ed7dc421f25eaf408dc2709fd0c847e8d0388d5f5149ce6e7',
                                            'expected_changed_symbol_ids': ('assignment|$module.EXPECTED_V38_COUNTERPARTS|1',
                                                                            'decorator|test_counterpart_baseline_count_import_and_evidence_tamper_fail_closed.__decorator__|1',
                                                                            'function|_make_counterpart_repo|1',
                                                                            'function|test_checker_rejects_checkpoint_main_receipt_omission_substitution_and_indirection|1',
                                                                            'function|test_committed_counterpart_rejects_same_symbol_import_hostile_body|1',
                                                                            'function|test_counterpart_baseline_count_import_and_evidence_tamper_fail_closed|1',
                                                                            'function|test_default_checkpoint_binding_counterparts_are_exact_and_only_expected_symbols_change|1',
                                                                            'function|test_default_git_pins_are_real_exact_objects_and_prereg_is_doc_only|1',
                                                                            'function|test_external_anchor_rejects_delta_body_with_stale_test_literal|1'),
                                            'expected_mechanical_counts': (19, 0, 8, 1, 1, 0, 0),
                                            'expected_redacted_candidate_sha256': 'c2360635181b2f8d4fc28f5418f771cfd557ef65a13a71f4952e96cda4066144',
                                            'external_sha256_anchor_assignment': None,
                                            'external_sha256_anchor_path': None,
                                            'redacted_literal_assignment': 'V38_DELTA_MODULE_SHA256'},
 'tests/test_sec_gemma_lean_v38_journal.py': {'counterpart_path': 'tests/test_sec_gemma_lean_v37_journal.py',
                                              'excluded_change_symbol_ids': (),
                                              'expected_candidate_sha256': 'e1e5fd68d0e125624cc7edbd891578f3afef1c5cb88288e4b0c103839bb4b16e',
                                              'expected_changed_import_specs': (),
                                              'expected_changed_symbol_evidence_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                                              'expected_changed_symbol_ids': (),
                                              'expected_mechanical_counts': (2, 43, 0, 0, 0, 0, 0),
                                              'expected_redacted_candidate_sha256': None,
                                              'external_sha256_anchor_assignment': None,
                                              'external_sha256_anchor_path': None,
                                              'redacted_literal_assignment': None},
 'tests/test_sec_gemma_lean_v38_preflight.py': {'counterpart_path': 'tests/test_sec_gemma_lean_v37_preflight.py',
                                                'excluded_change_symbol_ids': (),
                                                'expected_candidate_sha256': 'ad805271af364b835ed59dca93cdc322bb825bf42d51551aad9f2512ea14929b',
                                                'expected_changed_import_specs': (),
                                                'expected_changed_symbol_evidence_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                                                'expected_changed_symbol_ids': (),
                                                'expected_mechanical_counts': (2,
                                                                               28,
                                                                               0,
                                                                               0,
                                                                               0,
                                                                               2,
                                                                               0),
                                                'expected_redacted_candidate_sha256': None,
                                                'external_sha256_anchor_assignment': None,
                                                'external_sha256_anchor_path': None,
                                                'redacted_literal_assignment': None},
 'tests/test_sec_gemma_lean_v38_source.py': {'counterpart_path': 'tests/test_sec_gemma_lean_v37_source.py',
                                             'excluded_change_symbol_ids': (),
                                             'expected_candidate_sha256': '5c39dac99e2b8f73d2411bac18dff0cfe47218809dfebc11df32943e64c73e93',
                                             'expected_changed_import_specs': (),
                                             'expected_changed_symbol_evidence_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                                             'expected_changed_symbol_ids': (),
                                             'expected_mechanical_counts': (2, 56, 0, 1, 0, 0, 2),
                                             'expected_redacted_candidate_sha256': None,
                                             'external_sha256_anchor_assignment': None,
                                             'external_sha256_anchor_path': None,
                                             'redacted_literal_assignment': None},
 'tests/test_sec_gemma_lean_v38_transport.py': {'counterpart_path': 'tests/test_sec_gemma_lean_v37_transport.py',
                                                'excluded_change_symbol_ids': (),
                                                'expected_candidate_sha256': '965eba624ddcabc73dbc822ea970d8123cf79225d4c73dcec2bc83bced00b68b',
                                                'expected_changed_import_specs': (),
                                                'expected_changed_symbol_evidence_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                                                'expected_changed_symbol_ids': (),
                                                'expected_mechanical_counts': (1,
                                                                               15,
                                                                               0,
                                                                               0,
                                                                               0,
                                                                               0,
                                                                               0),
                                                'expected_redacted_candidate_sha256': None,
                                                'external_sha256_anchor_assignment': None,
                                                'external_sha256_anchor_path': None,
                                                'redacted_literal_assignment': None}}


def _counterpart_rule_from_data(value: Mapping[str, Any]) -> CounterpartRule:
    return CounterpartRule(
        counterpart_path=value["counterpart_path"],
        expected_mechanical_counts=tuple(value["expected_mechanical_counts"]),
        expected_candidate_sha256=value["expected_candidate_sha256"],
        redacted_literal_assignment=value["redacted_literal_assignment"],
        expected_redacted_candidate_sha256=value[
            "expected_redacted_candidate_sha256"
        ],
        external_sha256_anchor_path=value["external_sha256_anchor_path"],
        external_sha256_anchor_assignment=value[
            "external_sha256_anchor_assignment"
        ],
        excluded_change_symbol_ids=tuple(value["excluded_change_symbol_ids"]),
        expected_changed_symbol_ids=tuple(value["expected_changed_symbol_ids"]),
        expected_changed_import_specs=tuple(value["expected_changed_import_specs"]),
        expected_changed_symbol_evidence_sha256=value[
            "expected_changed_symbol_evidence_sha256"
        ],
    )


_DEFAULT_EXISTING_PATH_RULES = {
    path: _V37_CLONE_PATH_RULES[path] for path in _V38_IMPLEMENTATION_PATHS
}
_DEFAULT_COUNTERPART_RULES = {
    path: _counterpart_rule_from_data(_COUNTERPART_RULE_DATA[path])
    for path in _V38_IMPLEMENTATION_PATHS
}
_FROZEN_CONTENT_SYMBOLS: tuple[FrozenSymbolPin, ...] = ()

DEFAULT_ALLOW_SPEC = AllowSpec(
    path_rules=_DEFAULT_EXISTING_PATH_RULES,
    frozen_symbols=_FROZEN_CONTENT_SYMBOLS,
    counterpart_rules=_DEFAULT_COUNTERPART_RULES,
)


__all__ = [
    "AllowSpec",
    "CounterpartRule",
    "DEFAULT_ALLOW_SPEC",
    "DEFAULT_PINS",
    "DeltaPins",
    "DeltaValidationError",
    "FrozenSymbolPin",
    "MANIFEST_VERSION",
    "PathRule",
    "PREREG_COMMIT",
    "PREREG_DOC_BLOB",
    "PREREG_DOC_PATH",
    "PREREG_DOC_SHA256",
    "PREREG_TREE",
    "IMPLEMENTATION_BASE_COMMIT",
    "IMPLEMENTATION_BASE_TREE",
    "build_delta_manifest",
    "serialize_delta_manifest",
    "static_symbol_inventory",
    "validate_delta_manifest",
]
