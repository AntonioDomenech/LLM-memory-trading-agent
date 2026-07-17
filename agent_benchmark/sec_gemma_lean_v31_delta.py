"""Committed-blob-only implementation delta proof for SEC/Gemma v3.1.

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
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


MANIFEST_VERSION = "aapl-sec-gemma-lean-v31-implementation-delta-v1"
PREREG_COMMIT = "fe38fce300e858eeddb6894a540b8e52830d9287"
PREREG_TREE = "21090c026395595e6b9873402e0fc5954aec520a"
PREREG_DOC_PATH = "docs/aapl_sec_gemma_lean_evidence_v3_1.md"
PREREG_DOC_BLOB = "db5bf34d80a18affded60084c4d98b351dccfe1b"
PREREG_DOC_SHA256 = (
    "c066dc87cd30b15cfa517572f97c53fc028a8eb3fd7ed65913c122883512febd"
)
SCIENTIFIC_PARENT_COMMIT = "efbc481c57e480d48303763163676e64e87df49d"
SCIENTIFIC_PARENT_TREE = "bc91d619a4d04680d9600b1acd74f0a29662d0f9"

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
    scientific_parent_commit: str
    scientific_parent_tree: str

    def validate(self) -> None:
        for name, value in (
            ("prereg_commit", self.prereg_commit),
            ("prereg_tree", self.prereg_tree),
            ("prereg_doc_blob", self.prereg_doc_blob),
            ("scientific_parent_commit", self.scientific_parent_commit),
            ("scientific_parent_tree", self.scientific_parent_tree),
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
            "scientific_parent_commit": self.scientific_parent_commit,
            "scientific_parent_tree": self.scientific_parent_tree,
        }


DEFAULT_PINS = DeltaPins(
    prereg_commit=PREREG_COMMIT,
    prereg_tree=PREREG_TREE,
    prereg_doc_path=PREREG_DOC_PATH,
    prereg_doc_blob=PREREG_DOC_BLOB,
    prereg_doc_sha256=PREREG_DOC_SHA256,
    scientific_parent_commit=SCIENTIFIC_PARENT_COMMIT,
    scientific_parent_tree=SCIENTIFIC_PARENT_TREE,
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
        pins.scientific_parent_commit,
        label="scientific parent",
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
        pins.scientific_parent_commit,
        pins.scientific_parent_tree,
        label="scientific parent",
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
        pins.scientific_parent_commit,
        pins.prereg_commit,
        git_binary=git_binary,
    )
    implementation_diff = _raw_diff(
        root, pins.prereg_commit, implementation_commit, git_binary=git_binary
    )
    parent_to_implementation = _raw_diff(
        root,
        pins.scientific_parent_commit,
        implementation_commit,
        git_binary=git_binary,
    )
    inherited_paths = {entry.path for entry in inherited_diff}
    v31_paths = {entry.path for entry in implementation_diff}
    overlap = inherited_paths & v31_paths
    if overlap:
        raise DeltaValidationError(
            "inherited-at-prereg blobs changed in implementation: " + ",".join(sorted(overlap))
        )
    if v31_paths != set(allow_spec.path_rules):
        unknown = sorted(v31_paths - set(allow_spec.path_rules))
        missing = sorted(set(allow_spec.path_rules) - v31_paths)
        raise DeltaValidationError(f"implementation path inventory mismatch; unknown={unknown}; missing={missing}")

    inherited_json: list[dict[str, Any]] = []
    inherited_states: dict[str, dict[str, Any]] = {}
    for entry in inherited_diff:
        entry_json, _, _ = _entry_with_hashes(root, entry, git_binary=git_binary)
        entry_json["classification"] = "inherited_at_prereg_unchanged"
        inherited_json.append(entry_json)
        inherited_states[entry.path] = entry_json

    changes_json: list[dict[str, Any]] = []
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
        entry_json["classification"] = "v31_implementation_delta"
        changes_json.append(entry_json)
        all_test_functions.extend(_test_function_records(entry.path, symbol_changes))

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
        elif entry.path in v31_paths:
            classification = "v31_implementation_delta"
        else:
            raise DeltaValidationError(f"{entry.path}: unclassified scientific-parent delta")
        classified_parent_paths.append({**observed, "classification": classification})
    if {entry["path"] for entry in classified_parent_paths} != inherited_paths | v31_paths:
        raise DeltaValidationError("scientific-parent classification is not exhaustive")

    frozen_evidence = _frozen_symbol_evidence(
        root,
        pins,
        allow_spec.frozen_symbols,
        implementation_commit,
        git_binary=git_binary,
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
        "scientific_parent_classification": classified_parent_paths,
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


# This is the reviewed, exhaustive existing-file delta.  Keeping it in code
# makes manifest regeneration a verifier, not an authority to widen the
# preregistered surface.
_DEFAULT_EXISTING_PATH_RULES = {
    "agent_benchmark/sec_point_in_time.py": PathRule(
        "M",
        (
            "assignment|$module._SUBMISSIONS_ACCEPTANCE_RE|1",
            "function|_eastern_wall_clock|1",
            "function|_parse_submissions_acceptance|1",
            "function|parse_submissions_acceptance_datetime|1",
        ),
    ),
    "agent_benchmark/sec_filing_content.py": PathRule(
        "M",
        (
            "assignment|$module.LEGACY_FILENAME_LAST_DATE|1",
            "assignment|$module.LEGACY_PRIMARY_DOCUMENT_IDENTITY|1",
            "assignment|$module._LEGACY_DOCUMENT_IDENTITY_RE|1",
            "assignment|$module._SAFE_ARCHIVE_FILENAME_RE|1",
            "assignment|$module.__all__|1",
            "assignment|SGMLDocument.document_identity|1",
            "assignment|SGMLDocument.filename|1",
            "assignment|SGMLDocument.ordinal|1",
            "assignment|SGMLDocument.text_end_byte|1",
            "assignment|SGMLDocument.text_start_byte|1",
            "assignment|SGMLHeader.date_of_filing_date_change|1",
            "assignment|SequenceOnePrimarySelection.document_identity|1",
            "assignment|SequenceOnePrimarySelection.document|1",
            "assignment|SequenceOnePrimarySelection.sec_filename|1",
            "assignment|SequenceOnePrimarySelection.sgml_filename_missing|1",
            "assignment|SequenceOnePrimarySelection.submissions_filename_missing|1",
            "class|SGMLDocument|1",
            "class|SGMLHeader|1",
            "class|SequenceOnePrimarySelection|1",
            "decorator|SequenceOnePrimarySelection.__decorator__|1",
            "function|SequenceOnePrimarySelection.to_dict|1",
            "function|_parse_header|1",
            "function|_safe_filename|1",
            "function|parse_complete_submission|1",
            "function|select_sequence_one_primary_document|1",
        ),
    ),
    "agent_benchmark/sec_filing_gemma_corpus.py": PathRule(
        "M",
        (
            "assignment|$module._AAPL_ACCESSION_RE|1",
            "function|_deduplicate_columns|1",
            "function|_primary_document_url|1",
            "function|_universe_record|1",
            "function|acquire_official_sec_catalog|1",
            "function|validate_detached_catalog_replay|1",
        ),
    ),
    "agent_benchmark/sec_filing_gemma_contract.py": PathRule(
        "M",
        (
            "function|build_corpus_universe_manifest|1",
            "function|validate_live_lessons.normalize_bindings|1",
            "function|validate_live_lessons|1",
        ),
    ),
    "agent_benchmark/sec_filing_gemma_stage_access.py": PathRule(
        "M", ("assignment|$module._ACCESSION_RE|1",)
    ),
    "agent_benchmark/sec_filing_gemma_preprocessor.py": PathRule(
        "M", ("assignment|$module._AAPL_ACCESSION_RE|1",)
    ),
    "agent_benchmark/sec_filing_gemma_features.py": PathRule(
        "M", ("assignment|$module._ACCESSION_RE|1",)
    ),
    "agent_benchmark/sec_filing_gemma_prediction_evidence.py": PathRule(
        "M", ("assignment|$module._ACCESSION_RE|1",)
    ),
    "agent_benchmark/sec_filing_gemma_stage_authorization.py": PathRule(
        "M", ("assignment|$module._AAPL_ACCESSION_RE|1",)
    ),
    "agent_benchmark/sec_gemma_online_risk_overlay_acquisition.py": PathRule(
        "M", ("assignment|$module._ACCESSION_RE|1",)
    ),
    "agent_benchmark/sec_gemma_online_risk_overlay_features.py": PathRule(
        "M", ("assignment|$module._ACCESSION_RE|1",)
    ),
    "agent_benchmark/sec_gemma_online_risk_overlay_market_verifier.py": PathRule(
        "M", ("assignment|$module._ACCESSION_RE|1",)
    ),
    "tests/test_sec_point_in_time.py": PathRule(
        "M",
        (
            "decorator|test_submissions_acceptance_allows_only_zero_fraction.__decorator__|1",
            "decorator|test_submissions_acceptance_rejects_dst_gap_or_fold.__decorator__|1",
            "decorator|test_submissions_acceptance_rejects_noncanonical_iso.__decorator__|1",
            "decorator|test_user_agent_rejects_blank_or_placeholder_without_echo.__decorator__|1",
            "function|test_submissions_acceptance_allows_only_zero_fraction|1",
            "function|test_submissions_acceptance_rejects_dst_gap_or_fold|1",
            "function|test_submissions_acceptance_rejects_noncanonical_iso|1",
            "function|test_submissions_acceptance_z_preserves_sec_eastern_display_digits|1",
            "function|test_user_agent_rejects_blank_or_placeholder_without_echo|1",
            "function|test_user_agent_validation_returns_only_hash_and_never_contact|1",
            "import|$module.__import__|6",
            "import|$module.__import__|7",
            "import|$module.__import__|8",
        ),
        (
            "from:0:agent_benchmark.sec_point_in_time:AAPL_CIK,MAX_AUDIT_BYTES,MAX_AUDIT_REQUESTS,MAX_AUDIT_SECONDS,BudgetCounter,FilingRecord,SecAuditLimitError,SecPointInTimeError,archive_urls,content_sha256,parse_acceptance_datetime,parse_master_idx,parse_submissions_acceptance_datetime,parse_submissions_rows,validate_sec_user_agent",
            "from:0:zoneinfo:ZoneInfo",
            "import:pytest",
        ),
    ),
    "tests/test_sec_filing_content.py": PathRule(
        "M",
        (
            "decorator|test_document_text_boundaries_must_be_exactly_one_balanced_pair.__decorator__|1",
            "function|test_document_text_boundaries_must_be_exactly_one_balanced_pair|1",
            "function|test_header_preserves_exact_date_as_of_change|1",
            "function|test_iso_submissions_acceptance_reconciles_to_sgml_eastern_instant|1",
            "function|test_latin1_offsets_and_response_extracted_normalized_hashes_are_independent|1",
            "function|test_legacy_primary_filename_reconciles_when_only_one_source_supplies_it|1",
            "function|test_missing_filename_after_edgar7_boundary_rejects|1",
            "function|test_pre_edgar7_missing_filenames_use_reserved_sequence_identity|1",
            "function|test_primary_filename_reconciliation_rejects_conflict_and_bad_names|1",
            "function|test_sequence_one_selection_fails_closed|1",
            "function|test_sgml_filenames_must_be_safe_nonreserved_and_singular|1",
            "import|$module.__import__|7",
        ),
        (
            "from:0:agent_benchmark.sec_filing_content:MINIMUM_USABLE_TEXT_CHARACTERS,CompleteSubmission,SGMLDocument,audit_filing_content,conservative_availability_session,normalize_filing_text,parse_complete_submission,parse_sec_index_json,reconcile_filing_content,select_primary_document,select_sequence_one_primary_document",
        ),
    ),
    "tests/test_sec_filing_gemma_corpus.py": PathRule(
        "M",
        (
            "decorator|test_catalog_rejects_subject_form_date_and_primary_identity_attacks.__decorator__|1",
            "function|test_catalog_accepts_third_party_submitter_accession_for_apple_subject|1",
            "function|test_catalog_normalizes_timezone_qualified_submissions_acceptance_to_et|1",
            "function|test_catalog_rejects_subject_form_date_and_primary_identity_attacks|1",
            "function|test_cross_source_raw_rows_preserve_extra_columns_and_exact_types|1",
            "function|test_every_duplicate_accession_fails_closed|1",
            "function|test_exact_duplicates_are_deduplicated_but_conflicts_fail_closed|1",
            "function|test_historical_reference_count_range_and_subject_claim_are_checked|1",
        ),
    ),
    "agent_benchmark/sec_gemma_lean_v31_acquisition.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="300023eb0c1c8ffeb8913a1fcf8ca5b919990b6e20f8600682bc82def2e00b5f",
        expected_symbol_count=212,
        expected_import_inventory_sha256="c7cd4b7b90ab4078ed17edf95c2892b04aca1c56c4614682edf79c77d5cd2180",
        expected_import_count=28,
    ),
    "agent_benchmark/sec_gemma_lean_v31_delta.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="c7f81ea512bbcd700ff4f12bb72d524122965273d555076e370411334da96c81",
        expected_symbol_count=117,
        expected_import_inventory_sha256="6eb7d44e2e7e3f35bcc9835d64de8412af5a8cccc6aeed0594720eb6acf5aa2b",
        expected_import_count=13,
    ),
    "agent_benchmark/sec_gemma_lean_v31_journal.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="0a625508c1705a00d82f2fe77710fc5e05620ae0b2f02fd1876789c93580d8a3",
        expected_symbol_count=227,
        expected_import_inventory_sha256="f0059f257f0add60f4bad87d53107921760a387ffc244d6b053f4a9c1b589e11",
        expected_import_count=16,
    ),
    "agent_benchmark/sec_gemma_lean_v31_preflight.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="7fe17ea06c0d57a9bb94df4c20f04d509745d6ef9ef81dde935566bbb168bdba",
        expected_symbol_count=144,
        expected_import_inventory_sha256="7028ae04854221e9755327eb951dd26abfb1595b91c5d583b5df48382cc572d4",
        expected_import_count=25,
    ),
    "agent_benchmark/sec_gemma_lean_v31_source.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="a9fd92648d2407e89d9e1fa7b3539cc3faec7d3d51e1175b964d7a5c4b078aae",
        expected_symbol_count=330,
        expected_import_inventory_sha256="0de6ff93a23262c45e2996525906b40ced4342df6081654861781d67ab760ac0",
        expected_import_count=13,
    ),
    "agent_benchmark/sec_gemma_lean_v31_transport.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="c1d753a110e43d53ff9b7823fdbe876cf96ae0b2576dcc07b52828d89696ecea",
        expected_symbol_count=199,
        expected_import_inventory_sha256="6029a385a66bb8335e6ce9b53bde7c7c82dd24414da4ff9de25ab665b993a65d",
        expected_import_count=19,
    ),
    "tests/test_sec_gemma_lean_v31_acquisition.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="4be809cdc4ada470c0346286f12fe19e6d6f08226702a0617db5c0b6fb812fe6",
        expected_symbol_count=144,
        expected_import_inventory_sha256="a2df67778efe55b60bc45fc27df206d85797746d0ae6d880065aa129f0e27eef",
        expected_import_count=20,
    ),
    "tests/test_sec_gemma_lean_v31_delta.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="7d33efb436949196702ec7f24f5d0390bb99998380f8eb4bc4c847fea51b4d29",
        expected_symbol_count=36,
        expected_import_inventory_sha256="9e71af4feddaf527e7743466a2613ca8f35839b6a620be62efd620276f2d96cc",
        expected_import_count=9,
    ),
    "tests/test_sec_gemma_lean_v31_journal.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="473870868204fecb00cd32e275a08d667617a5b9ca9efa0bd6aae688a812806c",
        expected_symbol_count=54,
        expected_import_inventory_sha256="c8c51bd2796d5dcd9d8a3895c595092cbd65d263b427973396d8e803a49230c2",
        expected_import_count=9,
    ),
    "tests/test_sec_gemma_lean_v31_preflight.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="7232841d6bfff17e2eb2e2ef2a93d99a6597ada62fca4bda1db3b18ac60f9f8f",
        expected_symbol_count=58,
        expected_import_inventory_sha256="2d9cbbe9d9ea1905a334b0195fa0e22060791ffb2bef66bcb576dcc9bb70ec9a",
        expected_import_count=11,
    ),
    "tests/test_sec_gemma_lean_v31_source.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="63f01420a22ad6202fdcf1f568af122c8a65aab6352d5132ed24f1dff9e7005a",
        expected_symbol_count=66,
        expected_import_inventory_sha256="03ee1574614267ca1919a58ff50fa078a13413ea9f53b91604dcf92ddb6dc970",
        expected_import_count=12,
    ),
    "tests/test_sec_gemma_lean_v31_transport.py": PathRule(
        "A", (), (),
        expected_symbol_inventory_sha256="404fcba807429dc521e6bcbea906f1adcb97768f20a95f5c27a8e84d2917a5f5",
        expected_symbol_count=101,
        expected_import_inventory_sha256="3c9bbf6d0224d0fe15bf99d3b5f2f71d12433b1852d16edc6d51b2e82c37f1cb",
        expected_import_count=16,
    ),
}

_FROZEN_CONTENT_SYMBOLS = (
    FrozenSymbolPin(
        "agent_benchmark/sec_filing_content.py",
        "assignment|$module.CONTRACT_VERSION|1",
        "e4a358116679625d4c74f64c0c66b3704fadc2456d3c7b9c4d6b3fee53621233",
        "53c06841c4c6f2cde5052b359387404e19af9f8eeff7681145fc32014832548f",
        "819990108efc7d537f0ede8f537adcf4a7cb768b9feeef8a1650529fd165a98c",
    ),
    FrozenSymbolPin(
        "agent_benchmark/sec_filing_content.py",
        "assignment|$module.MINIMUM_USABLE_TEXT_CHARACTERS|1",
        "c1cbb2693cd7aa6e3e4c89610f00b3b5eb64cdb9124e8b106ee09a279bc8455b",
        "17e106d7325a43fd4c77c5c2a6c2599f71447de23c792fcbe43f1464fe533a06",
        "a3bbe80fb9615f6d357fba8f0af62be5d53291e44e90e294665e884f6cd59303",
    ),
    FrozenSymbolPin(
        "agent_benchmark/sec_filing_content.py",
        "function|normalize_document_text|1",
        "11c57caa2a62ea50f554e3e81a72ec50348e1d33c9707e5c637dec45b4539523",
        "b52d587842bf3165d3bf993211d8c5555fb8829b4566c4f96c81d046a7bea82f",
        "fd42dec59d996d96473da02f6efbd48770ee72c477fe39e6700d2c6d86f88167",
    ),
    FrozenSymbolPin(
        "agent_benchmark/sec_filing_content.py",
        "function|normalize_filing_text|1",
        "cad51c977d2bf23358097d46b0beeee379499a9b807946ce4ce82ea9a2bd8488",
        "4832a14409f30fec21751a8a708dabbd705e80841ba4c22206729d067e28a742",
        "14f0eb931504c47b9390fda5c83c1cabb7fe8d4cd4f434c9866f169611b72aa6",
    ),
)

DEFAULT_ALLOW_SPEC = AllowSpec(
    path_rules=_DEFAULT_EXISTING_PATH_RULES,
    frozen_symbols=_FROZEN_CONTENT_SYMBOLS,
)


__all__ = [
    "AllowSpec",
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
    "SCIENTIFIC_PARENT_COMMIT",
    "SCIENTIFIC_PARENT_TREE",
    "build_delta_manifest",
    "serialize_delta_manifest",
    "static_symbol_inventory",
    "validate_delta_manifest",
]
