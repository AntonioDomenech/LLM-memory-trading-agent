from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from agent_benchmark import sec_gemma_lean_v31_delta as delta


LOGIC_PATH = "agent_benchmark/logic.py"
TEST_PATH = "tests/test_logic.py"
DOC_PATH = "docs/prereg.md"


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def _write(repo: Path, path: str, payload: bytes) -> None:
    destination = repo / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def _commit(repo: Path, message: str) -> tuple[str, str]:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    commit = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git(repo, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    return commit, tree


def _commit_index(repo: Path, message: str) -> tuple[str, str]:
    _git(repo, "commit", "-q", "-m", message)
    commit = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git(repo, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    return commit, tree


def _blob(repo: Path, revision: str, path: str) -> bytes:
    return _git(repo, "cat-file", "blob", f"{revision}:{path}")


def _blob_oid(repo: Path, revision: str, path: str) -> str:
    return _git(repo, "rev-parse", f"{revision}:{path}").decode("ascii").strip()


def _changed_rule(
    repo: Path, prereg: str, implementation: str, path: str, status: str
) -> delta.PathRule:
    before = None if status == "A" else _blob(repo, prereg, path)
    after = _blob(repo, implementation, path)
    _, symbol_ids, import_specs = delta._symbol_changes(before, after)
    return delta.PathRule(status, symbol_ids, import_specs)


def _frozen_pin(repo: Path, prereg: str) -> delta.FrozenSymbolPin:
    records = {
        record["symbol_id"]: record
        for record in delta.static_symbol_inventory(_blob(repo, prereg, LOGIC_PATH))
    }
    record = records["function|normalize_filing_text|1"]
    return delta.FrozenSymbolPin(
        path=LOGIC_PATH,
        symbol_id=record["symbol_id"],
        physical_sha256=record["physical_sha256"],
        literal_sha256=record["literal_sha256"],
        semantic_sha256=record["semantic_sha256"],
    )


def _pins(
    repo: Path,
    parent: tuple[str, str],
    prereg: tuple[str, str],
) -> delta.DeltaPins:
    doc = _blob(repo, prereg[0], DOC_PATH)
    return delta.DeltaPins(
        prereg_commit=prereg[0],
        prereg_tree=prereg[1],
        prereg_doc_path=DOC_PATH,
        prereg_doc_blob=_blob_oid(repo, prereg[0], DOC_PATH),
        prereg_doc_sha256=hashlib.sha256(doc).hexdigest(),
        scientific_parent_commit=parent[0],
        scientific_parent_tree=parent[1],
    )


def _make_repo(
    tmp_path: Path,
    *,
    implementation_newline: bytes = b"\n",
    frozen: bool = True,
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Offline Test")
    _git(repo, "config", "user.email", "offline@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    parent_source = (
        b'"""Static test module."""\n'
        b"VALUE = 1\n\n"
        b"def normalize_filing_text(value):\n"
        b"    return value.strip()\n\n"
        b"def parse_value(value):\n"
        b"    return value + 1\n"
    )
    _write(repo, LOGIC_PATH, parent_source)
    parent = _commit(repo, "scientific parent")

    doc = b"# Exact preregistration\n"
    _write(repo, DOC_PATH, doc)
    _write(repo, "legacy/evidence.txt", b"inherited exact bytes\n")
    prereg = _commit(repo, "preregister")

    implementation_source = (
        b'"""Static test module."""\n'
        b"VALUE = 1\n\n"
        b"def normalize_filing_text(value):\n"
        b"    return value.strip()\n\n"
        b"def parse_value(value):\n"
        b"    return value + 2\n"
    ).replace(b"\n", implementation_newline)
    test_source = (
        b"from agent_benchmark.logic import parse_value\n\n"
        b"TRAP = lambda: __import__('pathlib').Path('executed').write_text('bad')\n\n"
        b"def test_parse_value():\n"
        b"    assert parse_value(1) == 3\n"
    ).replace(b"\n", implementation_newline)
    _write(repo, LOGIC_PATH, implementation_source)
    _write(repo, TEST_PATH, test_source)
    implementation = _commit(repo, "implementation")
    pins = _pins(repo, parent, prereg)
    rules = {
        LOGIC_PATH: _changed_rule(repo, prereg[0], implementation[0], LOGIC_PATH, "M"),
        TEST_PATH: _changed_rule(repo, prereg[0], implementation[0], TEST_PATH, "A"),
    }
    allow = delta.AllowSpec(
        path_rules=rules,
        frozen_symbols=(_frozen_pin(repo, prereg[0]),) if frozen else (),
    )
    return {
        "repo": repo,
        "parent": parent,
        "prereg": prereg,
        "implementation": implementation,
        "pins": pins,
        "allow": allow,
    }


def _manifest(state: dict[str, Any]) -> dict[str, Any]:
    return delta.build_delta_manifest(
        state["repo"],
        state["implementation"][0],
        state["implementation"][1],
        pins=state["pins"],
        allow_spec=state["allow"],
    )


def _validate_manifest(state: dict[str, Any], payload: bytes) -> dict[str, Any]:
    return delta.validate_delta_manifest(
        state["repo"],
        payload,
        expected_implementation_commit=state["implementation"][0],
        expected_implementation_tree=state["implementation"][1],
        pins=state["pins"],
        allow_spec=state["allow"],
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def test_static_inventory_names_assignments_imports_nested_and_decorators() -> None:
    blob = (
        b"import os\n"
        b"VALUE = 'literal'\n\n"
        b"@decorate(3)\n"
        b"def outer(value):\n"
        b"    @inside\n"
        b"    def nested():\n"
        b"        return value\n"
        b"    class Local:\n"
        b"        FIELD = 7\n"
        b"    return nested(), Local.FIELD\n"
    )
    records = {record["symbol_id"]: record for record in delta.static_symbol_inventory(blob)}
    assert {
        "import|$module.__import__|1",
        "assignment|$module.VALUE|1",
        "function|outer|1",
        "decorator|outer.__decorator__|1",
        "function|outer.nested|1",
        "decorator|outer.nested.__decorator__|1",
        "class|outer.Local|1",
        "assignment|outer.Local.FIELD|1",
    } <= set(records)
    assert records["function|outer|1"]["occurrence"] == 1
    assert records["function|outer|1"]["literal_sha256"]
    assert records["decorator|outer.__decorator__|1"]["physical_sha256"] == (
        hashlib.sha256(b"@decorate(3)").hexdigest()
    )


def test_static_inventory_covers_except_except_star_and_match_case_bodies() -> None:
    blob = (
        b"try:\n"
        b"    pass\n"
        b"except ValueError:\n"
        b"    import alpha\n"
        b"    HANDLED = 1\n"
        b"try:\n"
        b"    pass\n"
        b"except* RuntimeError:\n"
        b"    import beta\n"
        b"    STAR_HANDLED = 2\n"
        b"match subject:\n"
        b"    case {'kind': value}:\n"
        b"        import gamma\n"
        b"        CASE_VALUE = 3\n"
        b"        def case_function():\n"
        b"            return value\n"
    )
    records = {record["symbol_id"]: record for record in delta.static_symbol_inventory(blob)}
    assert {
        "import|$module.__import__|1",
        "import|$module.__import__|2",
        "import|$module.__import__|3",
        "assignment|$module.HANDLED|1",
        "assignment|$module.STAR_HANDLED|1",
        "assignment|$module.CASE_VALUE|1",
        "function|case_function|1",
    } <= set(records)
    assert {
        records["import|$module.__import__|1"]["import_spec"],
        records["import|$module.__import__|2"]["import_spec"],
        records["import|$module.__import__|3"]["import_spec"],
    } == {"import:alpha", "import:beta", "import:gamma"}


def test_default_new_file_allow_inventories_match_frozen_worktree_structure() -> None:
    for path, rule in delta.DEFAULT_ALLOW_SPEC.path_rules.items():
        if rule.status != "A":
            continue
        _, symbol_ids, import_specs = delta._symbol_changes(None, Path(path).read_bytes())
        assert len(symbol_ids) == rule.expected_symbol_count, path
        assert delta._sha256(delta._canonical_json(list(symbol_ids))) == (
            rule.expected_symbol_inventory_sha256
        ), path
        assert len(import_specs) == rule.expected_import_count, path
        assert delta._sha256(delta._canonical_json(list(import_specs))) == (
            rule.expected_import_inventory_sha256
        ), path


def test_committed_blob_manifest_round_trips_and_ignores_worktree(tmp_path: Path) -> None:
    state = _make_repo(tmp_path)
    manifest = _manifest(state)
    implementation_entries = {
        item["path"]: item for item in manifest["implementation_delta"]
    }
    assert set(implementation_entries) == {LOGIC_PATH, TEST_PATH}
    assert implementation_entries[LOGIC_PATH]["new_sha256"] == hashlib.sha256(
        _blob(state["repo"], state["implementation"][0], LOGIC_PATH)
    ).hexdigest()
    bound_test = manifest["bound_test_functions"][0]
    assert bound_test["symbol_id"] == "function|test_parse_value|1"
    assert bound_test["before"] is None
    assert bound_test["after"]["literal_sha256"]
    inherited = {
        item["path"]: item["classification"]
        for item in manifest["inherited_at_prereg"]
    }
    assert inherited == {
        DOC_PATH: "inherited_at_prereg_unchanged",
        "legacy/evidence.txt": "inherited_at_prereg_unchanged",
    }
    assert manifest["frozen_symbols"][0]["symbol_id"] == (
        "function|normalize_filing_text|1"
    )
    assert manifest["observation_contract"]["worktree_reads"] == 0
    assert not (state["repo"] / "executed").exists()

    # A hostile uncommitted worktree has no influence on the recomputation.
    _write(state["repo"], LOGIC_PATH, b"raise RuntimeError('worktree only')\n")
    payload = delta.serialize_delta_manifest(manifest)
    assert _validate_manifest(state, payload) == manifest
    assert not (state["repo"] / "executed").exists()


def test_checker_invokes_only_the_frozen_binary_git_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _make_repo(tmp_path)
    observed: list[tuple[str, ...]] = []
    real_run = delta.subprocess.run

    def recording_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed.append(tuple(args[1:]))
        return real_run(args, **kwargs)

    monkeypatch.setattr(delta.subprocess, "run", recording_run)
    _manifest(state)
    assert observed
    assert {args[0] for args in observed} == {"diff-tree", "cat-file"}
    for args in observed:
        if args[0] == "diff-tree":
            assert args[1:6] == (
                "--raw",
                "-r",
                "-z",
                "--no-renames",
                "--no-abbrev",
            )
        else:
            assert args[1] in {"blob", "-t", "commit"}


def test_unknown_path_symbol_and_import_each_fail_closed(tmp_path: Path) -> None:
    state = _make_repo(tmp_path / "path")
    repo = state["repo"]

    _write(repo, "agent_benchmark/rogue.py", b"ROGUE = True\n")
    state["implementation"] = _commit(repo, "unknown path")
    with pytest.raises(delta.DeltaValidationError, match="path inventory"):
        _manifest(state)

    state = _make_repo(tmp_path / "symbol")
    repo = state["repo"]
    clean_source = _blob(repo, state["implementation"][0], LOGIC_PATH)
    _write(repo, LOGIC_PATH, clean_source + b"\ndef surprise():\n    return 9\n")
    state["implementation"] = _commit(repo, "unknown symbol")
    with pytest.raises(delta.DeltaValidationError, match="unknown or missing static symbol"):
        _manifest(state)

    state = _make_repo(tmp_path / "import")
    repo = state["repo"]
    clean_source = _blob(repo, state["implementation"][0], LOGIC_PATH)
    _write(repo, LOGIC_PATH, b"import socket\n" + clean_source)
    state["implementation"] = _commit(repo, "unknown import")
    with pytest.raises(delta.DeltaValidationError, match="unknown or missing import"):
        _manifest(state)


def test_crlf_is_bound_as_physical_committed_bytes_not_normalized(tmp_path: Path) -> None:
    state = _make_repo(tmp_path, implementation_newline=b"\r\n", frozen=False)
    manifest = _manifest(state)
    entry = next(
        item for item in manifest["implementation_delta"] if item["path"] == LOGIC_PATH
    )
    raw = _blob(state["repo"], state["implementation"][0], LOGIC_PATH)
    assert b"\r\n" in raw
    assert entry["new_sha256"] == hashlib.sha256(raw).hexdigest()
    normalized_change = next(
        change
        for change in entry["symbols"]
        if change["symbol_id"] == "function|normalize_filing_text|1"
    )
    assert normalized_change["before"]["semantic_sha256"] == (
        normalized_change["after"]["semantic_sha256"]
    )
    assert normalized_change["before"]["physical_sha256"] != (
        normalized_change["after"]["physical_sha256"]
    )


def test_frozen_symbol_and_inherited_prereg_drift_are_terminal(tmp_path: Path) -> None:
    state = _make_repo(tmp_path / "frozen")
    repo = state["repo"]
    current = _blob(repo, state["implementation"][0], LOGIC_PATH)
    _write(repo, LOGIC_PATH, current.replace(b"return value.strip()", b"return value"))
    state["implementation"] = _commit(repo, "frozen drift")
    broad_rule = _changed_rule(
        repo, state["prereg"][0], state["implementation"][0], LOGIC_PATH, "M"
    )
    state["allow"] = delta.AllowSpec(
        path_rules={
            LOGIC_PATH: broad_rule,
            TEST_PATH: _changed_rule(
                repo, state["prereg"][0], state["implementation"][0], TEST_PATH, "A"
            ),
        },
        frozen_symbols=(_frozen_pin(repo, state["prereg"][0]),),
    )
    with pytest.raises(delta.DeltaValidationError, match="frozen symbol drift"):
        _manifest(state)

    state = _make_repo(tmp_path / "inherited")
    repo = state["repo"]
    _write(repo, "legacy/evidence.txt", b"mutated inherited bytes\n")
    state["implementation"] = _commit(repo, "inherited drift")
    with pytest.raises(delta.DeltaValidationError, match="inherited-at-prereg"):
        _manifest(state)


def test_manifest_rejects_noncanonical_tamper_and_rehashed_tamper(
    tmp_path: Path,
) -> None:
    state = _make_repo(tmp_path)
    manifest = _manifest(state)
    payload = delta.serialize_delta_manifest(manifest)

    with pytest.raises(delta.DeltaValidationError, match="expected implementation"):
        delta.validate_delta_manifest(
            state["repo"],
            payload,
            expected_implementation_commit=state["prereg"][0],
            expected_implementation_tree=state["prereg"][1],
            pins=state["pins"],
            allow_spec=state["allow"],
        )

    noncanonical = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(delta.DeltaValidationError, match="not canonical"):
        _validate_manifest(state, noncanonical)

    tampered = deepcopy(manifest)
    tampered["implementation_delta"][0]["classification"] = "forged"
    tampered.pop("manifest_sha256")
    tampered["manifest_sha256"] = hashlib.sha256(_canonical(tampered)).hexdigest()
    tampered_payload = _canonical(tampered) + b"\n"
    with pytest.raises(delta.DeltaValidationError, match="recomputation"):
        _validate_manifest(state, tampered_payload)

    damaged = bytearray(payload)
    damaged[-3] = ord("0") if damaged[-3] != ord("0") else ord("1")
    with pytest.raises(delta.DeltaValidationError):
        _validate_manifest(state, bytes(damaged))


def test_tree_pin_and_nonregular_mode_fail_closed(tmp_path: Path) -> None:
    state = _make_repo(tmp_path / "tree")
    wrong_tree = state["prereg"][1]
    with pytest.raises(delta.DeltaValidationError, match="implementation commit"):
        delta.build_delta_manifest(
            state["repo"],
            state["implementation"][0],
            wrong_tree,
            pins=state["pins"],
            allow_spec=state["allow"],
        )
    with pytest.raises(delta.DeltaValidationError, match="Git rejected|not a commit object"):
        delta.build_delta_manifest(
            state["repo"],
            state["implementation"][1],
            state["implementation"][1],
            pins=state["pins"],
            allow_spec=state["allow"],
        )
    with pytest.raises(delta.DeltaValidationError, match="not a tree object"):
        delta.build_delta_manifest(
            state["repo"],
            state["implementation"][0],
            state["implementation"][0],
            pins=state["pins"],
            allow_spec=state["allow"],
        )

    repo = state["repo"]
    link_blob = _git(repo, "hash-object", "-w", "--stdin", input_bytes=b"target").decode(
        "ascii"
    ).strip()
    _git(repo, "update-index", "--add", "--cacheinfo", f"120000,{link_blob},link.py")
    state["implementation"] = _commit_index(repo, "symlink mode")
    rules = dict(state["allow"].path_rules)
    rules["link.py"] = delta.PathRule("A", (), ())
    state["allow"] = delta.AllowSpec(
        path_rules=rules, frozen_symbols=state["allow"].frozen_symbols
    )
    with pytest.raises(delta.DeltaValidationError, match="symlink/submodule/nonregular"):
        _manifest(state)

    state = _make_repo(tmp_path / "submodule")
    repo = state["repo"]
    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{state['parent'][0]},nested.py",
    )
    state["implementation"] = _commit_index(repo, "submodule mode")
    rules = dict(state["allow"].path_rules)
    rules["nested.py"] = delta.PathRule("A", (), ())
    state["allow"] = delta.AllowSpec(
        path_rules=rules, frozen_symbols=state["allow"].frozen_symbols
    )
    with pytest.raises(delta.DeltaValidationError, match="symlink/submodule/nonregular"):
        _manifest(state)
