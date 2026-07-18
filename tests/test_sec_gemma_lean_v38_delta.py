from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from agent_benchmark import sec_gemma_lean_v38_delta as delta


V38_DELTA_MODULE_SHA256 = "4edb0804f964a1e0a415fe89fb304486cda2bb3af82fcc6fdc52b880a2fa82fe"
LOGIC_PATH = "agent_benchmark/logic.py"
TEST_PATH = "tests/test_logic.py"
DOC_PATH = "docs/prereg.md"
EXPECTED_V38_COUNTERPARTS = {
    "agent_benchmark/sec_gemma_lean_v38_acquisition.py": (
        "agent_benchmark/sec_gemma_lean_v37_acquisition.py"
    ),
    "agent_benchmark/sec_gemma_lean_v38_delta.py": (
        "agent_benchmark/sec_gemma_lean_v37_delta.py"
    ),
    "agent_benchmark/sec_gemma_lean_v38_journal.py": (
        "agent_benchmark/sec_gemma_lean_v37_journal.py"
    ),
    "agent_benchmark/sec_gemma_lean_v38_preflight.py": (
        "agent_benchmark/sec_gemma_lean_v37_preflight.py"
    ),
    "agent_benchmark/sec_gemma_lean_v38_source.py": (
        "agent_benchmark/sec_gemma_lean_v37_source.py"
    ),
    "agent_benchmark/sec_gemma_lean_v38_transport.py": (
        "agent_benchmark/sec_gemma_lean_v37_transport.py"
    ),
    "tests/test_sec_gemma_lean_v38_acquisition.py": (
        "tests/test_sec_gemma_lean_v37_acquisition.py"
    ),
    "tests/test_sec_gemma_lean_v38_delta.py": (
        "tests/test_sec_gemma_lean_v37_delta.py"
    ),
    "tests/test_sec_gemma_lean_v38_journal.py": (
        "tests/test_sec_gemma_lean_v37_journal.py"
    ),
    "tests/test_sec_gemma_lean_v38_preflight.py": (
        "tests/test_sec_gemma_lean_v37_preflight.py"
    ),
    "tests/test_sec_gemma_lean_v38_source.py": (
        "tests/test_sec_gemma_lean_v37_source.py"
    ),
    "tests/test_sec_gemma_lean_v38_transport.py": (
        "tests/test_sec_gemma_lean_v37_transport.py"
    ),
}


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
        implementation_base_commit=parent[0],
        implementation_base_tree=parent[1],
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
    parent = _commit(repo, "implementation base")

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


def _make_counterpart_repo(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Offline Test")
    _git(repo, "config", "user.email", "offline@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    old_path = "agent_benchmark/v37_logic.py"
    new_path = "agent_benchmark/v38_logic.py"
    source = b'def run():\n    return "source-only"\n'
    _write(repo, old_path, source)
    parent = _commit(repo, "counterpart base")
    _write(repo, DOC_PATH, b"# Exact preregistration\n")
    prereg = _commit(repo, "preregister")
    _write(repo, new_path, source)
    implementation = _commit(repo, "benign implementation")
    pins = _pins(repo, parent, prereg)
    changes, symbol_ids, imports = delta._symbol_changes(source, source)
    counterpart_rule = delta.CounterpartRule(
        counterpart_path=old_path,
        expected_mechanical_counts=(0, 0, 0, 0, 0, 0, 0),
        expected_candidate_sha256=hashlib.sha256(source).hexdigest(),
        redacted_literal_assignment=None,
        expected_redacted_candidate_sha256=None,
        external_sha256_anchor_path=None,
        external_sha256_anchor_assignment=None,
        excluded_change_symbol_ids=(),
        expected_changed_symbol_ids=symbol_ids,
        expected_changed_import_specs=imports,
        expected_changed_symbol_evidence_sha256=(
            delta._counterpart_change_evidence_sha256(changes)
        ),
    )
    allow = delta.AllowSpec(
        path_rules={
            new_path: _changed_rule(
                repo, prereg[0], implementation[0], new_path, "A"
            )
        },
        counterpart_rules={new_path: counterpart_rule},
    )
    return {
        "repo": repo,
        "parent": parent,
        "prereg": prereg,
        "implementation": implementation,
        "pins": pins,
        "allow": allow,
        "old_path": old_path,
        "new_path": new_path,
        "counterpart_rule": counterpart_rule,
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


def test_default_git_pins_are_real_exact_objects_and_prereg_is_doc_only() -> None:
    repo = Path(__file__).resolve().parents[1]
    prereg = "c8a1f2d2ad7ec22d4f50600180908e9d499520f4"
    prereg_tree = "5eaf18fa8e6f4ee715a66df6af010a312f50454a"
    base = "8d84ca6b1d8a5ef50d790beb9b4b53e51812a542"
    base_tree = "1287094442f4ceecdaf0ca29b9bdb4980d65b7e7"
    doc_path = "docs/aapl_sec_gemma_lean_evidence_v3_8.md"
    doc_blob = "1740ddb65985fed0eb001623760c9a70425462ed"
    doc_sha256 = "6d27fdc7f21668d0503a91b30c6b6e4699778031330f5c3a5205735b268102a2"
    assert delta.PREREG_COMMIT == prereg
    assert delta.PREREG_TREE == prereg_tree
    assert delta.IMPLEMENTATION_BASE_COMMIT == base
    assert delta.IMPLEMENTATION_BASE_TREE == base_tree
    assert delta.PREREG_DOC_PATH == doc_path
    assert delta.PREREG_DOC_BLOB == doc_blob
    assert delta.PREREG_DOC_SHA256 == doc_sha256
    assert _git(repo, "cat-file", "-t", prereg) == b"commit\n"
    assert _git(repo, "cat-file", "-t", prereg_tree) == b"tree\n"
    assert _git(repo, "cat-file", "-t", base) == b"commit\n"
    assert _git(repo, "cat-file", "-t", base_tree) == b"tree\n"
    prereg_commit = _git(repo, "cat-file", "commit", prereg).splitlines()
    assert prereg_commit[0] == f"tree {prereg_tree}".encode("ascii")
    assert [line for line in prereg_commit if line.startswith(b"parent ")] == [
        f"parent {base}".encode("ascii")
    ]
    assert _blob_oid(repo, prereg, doc_path) == doc_blob
    assert hashlib.sha256(_blob(repo, prereg, doc_path)).hexdigest() == doc_sha256
    inherited = delta._raw_diff(repo, base, prereg, git_binary="git")
    assert [(entry.path, entry.status) for entry in inherited] == [(doc_path, "A")]


def test_default_new_file_allow_inventories_match_frozen_worktree_structure() -> None:
    repo = Path(__file__).resolve().parents[1]
    expected_paths = tuple(sorted(EXPECTED_V38_COUNTERPARTS))
    assert tuple(sorted(delta.DEFAULT_ALLOW_SPEC.path_rules)) == expected_paths
    assert tuple(sorted(delta.DEFAULT_ALLOW_SPEC.counterpart_rules)) == expected_paths
    assert tuple(sorted(delta._V38_IMPLEMENTATION_PATHS)) == expected_paths
    assert len(delta.DEFAULT_ALLOW_SPEC.path_rules) == 12
    assert {
        path: rule.status
        for path, rule in delta.DEFAULT_ALLOW_SPEC.path_rules.items()
    } == {path: "A" for path in expected_paths}
    assert delta.DEFAULT_ALLOW_SPEC.frozen_symbols == ()
    assert {
        path: rule.counterpart_path
        for path, rule in delta.DEFAULT_ALLOW_SPEC.counterpart_rules.items()
    } == EXPECTED_V38_COUNTERPARTS
    for path, rule in delta.DEFAULT_ALLOW_SPEC.path_rules.items():
        before = None
        _, symbol_ids, import_specs = delta._symbol_changes(
            before, Path(path).read_bytes()
        )
        if rule.expected_symbol_inventory_sha256 is None:
            assert symbol_ids == rule.expected_symbol_ids, path
        else:
            assert len(symbol_ids) == rule.expected_symbol_count, path
            assert delta._sha256(delta._canonical_json(list(symbol_ids))) == (
                rule.expected_symbol_inventory_sha256
            ), path
        if rule.expected_import_inventory_sha256 is None:
            assert import_specs == rule.expected_import_specs, path
        else:
            assert len(import_specs) == rule.expected_import_count, path
            assert delta._sha256(delta._canonical_json(list(import_specs))) == (
                rule.expected_import_inventory_sha256
            ), path


def test_default_checkpoint_binding_counterparts_are_exact_and_only_expected_symbols_change() -> None:
    repo = Path(__file__).resolve().parents[1]
    expected_checkpoint_changes = {
        "agent_benchmark/sec_gemma_lean_v38_acquisition.py": {
            "counterpart": "agent_benchmark/sec_gemma_lean_v37_acquisition.py",
            "counts": (12, 271, 0, 1, 2, 7, 2),
            "raw_sha256": "8ba5b3dee285a5e12e631c5d5e8fbafce1a66d1c81b4c5736371ac8ceffbb3f6",
            "symbol_ids": (
                "class|DiskBackedSecAcquisition|1",
                "function|DiskBackedSecAcquisition._run_locked|1",
            ),
            "evidence_sha256": "3b7a76502cda4ab6da4e88585c4ee7d234b7d86fe80096384126236b88ed04d3",
        },
        "tests/test_sec_gemma_lean_v38_acquisition.py": {
            "counterpart": "tests/test_sec_gemma_lean_v37_acquisition.py",
            "counts": (11, 38, 0, 0, 1, 1, 0),
            "raw_sha256": "48f247952c7db6eede09b5b32b918b419e6c2e8f641028872907f29df6e9c2c8",
            "symbol_ids": (
                "class|_FakeSource|1",
                "function|_FakeSource.build_compact_checkpoint|1",
            ),
            "evidence_sha256": "e81a3685164f185209b59251d7b9dc8b79246d6f4c3a37170aadc6948c6550be",
        },
    }
    for path, expected in expected_checkpoint_changes.items():
        rule = delta.DEFAULT_ALLOW_SPEC.counterpart_rules[path]
        assert rule.counterpart_path == expected["counterpart"]
        assert rule.expected_mechanical_counts == expected["counts"]
        assert rule.expected_candidate_sha256 == expected["raw_sha256"]
        assert rule.expected_changed_symbol_ids == expected["symbol_ids"]
        assert rule.expected_changed_import_specs == ()
        assert (
            rule.expected_changed_symbol_evidence_sha256
            == expected["evidence_sha256"]
        )
        counterpart = _blob(
            repo, delta.IMPLEMENTATION_BASE_COMMIT, rule.counterpart_path
        )
        transformed, _ = delta._mechanical_counterpart_blob(
            counterpart, rule.expected_mechanical_counts
        )
        candidate = (repo / path).read_bytes()
        changes, symbol_ids, import_specs = delta._symbol_changes(
            transformed, candidate
        )
        assert hashlib.sha256(candidate).hexdigest() == expected["raw_sha256"]
        assert symbol_ids == expected["symbol_ids"]
        assert import_specs == ()
        assert delta._counterpart_change_evidence_sha256(changes) == (
            expected["evidence_sha256"]
        )

    mechanical_raw_hashes = {
        "agent_benchmark/sec_gemma_lean_v38_journal.py": "c433fcfa0b201546ed22bb7fed550c21eaf780170fc55336a82839df74d654b6",
        "agent_benchmark/sec_gemma_lean_v38_preflight.py": "07b1610a3c2eabfab1bc4edcc4c5a3b198a44cf5a4c311f006401ae8a63a5ad5",
        "agent_benchmark/sec_gemma_lean_v38_source.py": "8f624df2b28425b8e31db81997aeb0387d3c19418c0781e3b04fca66830d0f04",
        "agent_benchmark/sec_gemma_lean_v38_transport.py": "ec60f3868b1f5f3f4f638215e888d35bd3cae97ca442dedb3de4d28249252a26",
        "tests/test_sec_gemma_lean_v38_journal.py": "e1e5fd68d0e125624cc7edbd891578f3afef1c5cb88288e4b0c103839bb4b16e",
        "tests/test_sec_gemma_lean_v38_preflight.py": "ad805271af364b835ed59dca93cdc322bb825bf42d51551aad9f2512ea14929b",
        "tests/test_sec_gemma_lean_v38_source.py": "5c39dac99e2b8f73d2411bac18dff0cfe47218809dfebc11df32943e64c73e93",
        "tests/test_sec_gemma_lean_v38_transport.py": "965eba624ddcabc73dbc822ea970d8123cf79225d4c73dcec2bc83bced00b68b",
    }
    for path, raw_sha256 in mechanical_raw_hashes.items():
        rule = delta.DEFAULT_ALLOW_SPEC.counterpart_rules[path]
        assert rule.counterpart_path == path.replace("v38", "v37")
        assert rule.expected_candidate_sha256 == raw_sha256
        assert rule.expected_changed_symbol_ids == ()
        assert rule.expected_changed_import_specs == ()
        assert rule.expected_changed_symbol_evidence_sha256 == hashlib.sha256(
            b"[]"
        ).hexdigest()
        counterpart = _blob(
            repo, delta.IMPLEMENTATION_BASE_COMMIT, rule.counterpart_path
        )
        transformed, _ = delta._mechanical_counterpart_blob(
            counterpart, rule.expected_mechanical_counts
        )
        candidate = (repo / path).read_bytes()
        assert transformed == candidate
        assert hashlib.sha256(candidate).hexdigest() == raw_sha256


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
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_REPLACE_REF_BASE",
        "GIT_CONFIG_COUNT",
    ):
        monkeypatch.setenv(name, str(tmp_path / "hostile" / name))
    observed: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    real_run = delta.subprocess.run

    def recording_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed.append((tuple(args[1:]), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(delta.subprocess, "run", recording_run)
    _manifest(state)
    assert observed
    assert {args[0] for args, _ in observed} == {"diff-tree", "cat-file"}
    for args, kwargs in observed:
        environment = kwargs["env"]
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert environment["GIT_OPTIONAL_LOCKS"] == "0"
        assert environment["LC_ALL"] == "C"
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["timeout"] == 60
        for hostile_name in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_REPLACE_REF_BASE",
            "GIT_CONFIG_COUNT",
        ):
            assert hostile_name not in environment
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


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b":000000 100644 " + (b"0" * 40) + b" " + (b"1" * 40) + b" A\0x.py", "NUL framing"),
        (b":000000 100644 " + (b"0" * 40) + b" " + (b"1" * 40) + b" A\0\xff.py\0", "UTF-8"),
        (b":000000 100644 " + (b"0" * 40) + b" " + (b"1" * 40) + b" A\0../x.py\0", "unsafe Git path"),
        (b":000000 100644 " + (b"0" * 40) + b" " + (b"1" * 40) + b" A\0/x.py\0", "unsafe Git path"),
        (
            (b":000000 100644 " + (b"0" * 40) + b" " + (b"1" * 40) + b" A\0x.py\0") * 2,
            "repeated a path",
        ),
    ),
)
def test_raw_diff_hostile_framing_and_paths_fail_closed(
    payload: bytes, message: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(delta, "_git", lambda *args, **kwargs: payload)
    with pytest.raises(delta.DeltaValidationError, match=message):
        delta._raw_diff(tmp_path, "1" * 40, "2" * 40, git_binary="git")


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


def test_committed_counterpart_rejects_same_symbol_import_hostile_body(
    tmp_path: Path,
) -> None:
    state = _make_counterpart_repo(tmp_path)
    assert _manifest(state)["committed_v32_counterpart_comparisons"][0][
        "changed_symbol_count"
    ] == 0
    repo = state["repo"]
    original_path_rule = state["allow"].path_rules[state["new_path"]]
    benign = _blob(repo, state["implementation"][0], state["new_path"])
    commented = benign + b"# non-AST drift must still be bound\n"
    _write(repo, state["new_path"], commented)
    state["implementation"] = _commit(repo, "non-AST comment drift")
    _, comment_ids, comment_imports = delta._symbol_changes(None, commented)
    assert comment_ids == original_path_rule.expected_symbol_ids
    assert comment_imports == original_path_rule.expected_import_specs
    with pytest.raises(delta.DeltaValidationError, match="complete candidate blob drift"):
        _manifest(state)

    hostile = (
        b"def run():\n"
        b"    return __import__('urllib.request').request.urlopen("
        b"'https://example.invalid')\n"
    )
    _write(repo, state["new_path"], hostile)
    state["implementation"] = _commit(repo, "same symbol and import surface, hostile body")
    _, hostile_ids, hostile_imports = delta._symbol_changes(None, hostile)
    assert hostile_ids == original_path_rule.expected_symbol_ids
    assert hostile_imports == original_path_rule.expected_import_specs == ()
    with pytest.raises(delta.DeltaValidationError, match="complete candidate blob drift"):
        _manifest(state)

    # Even granting the hostile raw blob hash does not weaken the independent
    # committed-v3.7 semantic comparison.
    weakened_rule = replace(
        state["counterpart_rule"],
        expected_candidate_sha256=hashlib.sha256(hostile).hexdigest(),
    )
    state["allow"] = delta.AllowSpec(
        path_rules=state["allow"].path_rules,
        counterpart_rules={state["new_path"]: weakened_rule},
    )
    with pytest.raises(delta.DeltaValidationError, match="counterpart symbol"):
        _manifest(state)


def test_checker_rejects_blank_company_scope_expansion_row_drop_and_special_case(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Offline Test")
    _git(repo, "config", "user.email", "offline@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    old_path = "agent_benchmark/sec_gemma_lean_v32_source.py"
    new_path = "agent_benchmark/sec_gemma_lean_v38_source.py"
    baseline = (
        b"def parse_strict_master_gzip(cik, company, form, rows):\n"
        b"    is_apple_target = cik == '0000320193' and form in {'10-K', '10-Q'}\n"
        b"    if not company:\n"
        b"        raise ValueError('company required')\n"
        b"    rows.append((cik, company, form))\n"
        b"    return rows[-1] if is_apple_target else None\n"
    )
    approved = baseline.replace(
        b"    if not company:\n",
        b"    if is_apple_target and not company:\n",
    )
    _write(repo, old_path, baseline)
    parent = _commit(repo, "counterpart base")
    _write(repo, DOC_PATH, b"# Exact preregistration\n")
    prereg = _commit(repo, "preregister")
    _write(repo, new_path, approved)
    implementation = _commit(repo, "approved general parser rule")
    pins = _pins(repo, parent, prereg)
    approved_changes, approved_ids, approved_imports = delta._symbol_changes(
        baseline, approved
    )
    assert approved_ids == ("function|parse_strict_master_gzip|1",)
    assert approved_imports == ()
    approved_rule = delta.CounterpartRule(
        counterpart_path=old_path,
        expected_mechanical_counts=(0, 0, 0, 0, 0, 0, 0),
        expected_candidate_sha256=hashlib.sha256(approved).hexdigest(),
        redacted_literal_assignment=None,
        expected_redacted_candidate_sha256=None,
        external_sha256_anchor_path=None,
        external_sha256_anchor_assignment=None,
        excluded_change_symbol_ids=(),
        expected_changed_symbol_ids=approved_ids,
        expected_changed_import_specs=approved_imports,
        expected_changed_symbol_evidence_sha256=(
            delta._counterpart_change_evidence_sha256(approved_changes)
        ),
    )
    state = {
        "repo": repo,
        "prereg": prereg,
        "implementation": implementation,
        "pins": pins,
        "allow": delta.AllowSpec(
            path_rules={
                new_path: _changed_rule(
                    repo, prereg[0], implementation[0], new_path, "A"
                )
            },
            counterpart_rules={new_path: approved_rule},
        ),
    }
    assert _manifest(state)["committed_v32_counterpart_comparisons"][0][
        "changed_symbol_count"
    ] == 1

    hostile_variants = (
        approved.replace(
            b"    if is_apple_target and not company:\n",
            b"    if is_apple_target and form == '10-K' and not company:\n",
        ),
        approved.replace(
            b"    rows.append((cik, company, form))\n",
            b"    if not company:\n"
            b"        return None\n"
            b"    rows.append((cik, company, form))\n",
        ),
        approved.replace(
            b"    rows.append((cik, company, form))\n",
            b"    if not company and form == 'S-1':\n"
            b"        return None\n"
            b"    rows.append((cik, company, form))\n",
        ),
    )
    for index, hostile in enumerate(hostile_variants):
        _git(repo, "checkout", "-q", prereg[0])
        _write(repo, new_path, hostile)
        hostile_implementation = _commit(repo, f"hostile parser {index}")
        state["implementation"] = hostile_implementation
        state["allow"] = delta.AllowSpec(
            path_rules={
                new_path: _changed_rule(
                    repo,
                    prereg[0],
                    hostile_implementation[0],
                    new_path,
                    "A",
                )
            },
            counterpart_rules={
                new_path: replace(
                    approved_rule,
                    expected_candidate_sha256=hashlib.sha256(hostile).hexdigest(),
                )
            },
        )
        with pytest.raises(delta.DeltaValidationError, match="symbol content drift"):
            _manifest(state)


def test_checker_rejects_ims_alias_rewrite_scope_expansion_and_observed_special_case(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Offline Test")
    _git(repo, "config", "user.email", "offline@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    path = "agent_benchmark/sec_filing_content.py"
    baseline = (
        b"def parse_complete_submission(raw):\n"
        b"    if b'<SEC-DOCUMENT>' not in raw:\n"
        b"        raise ValueError('SEC envelope required')\n"
        b"    return raw\n"
    )
    approved = (
        b"def parse_complete_submission(raw):\n"
        b"    sec = b'<SEC-DOCUMENT>' in raw\n"
        b"    ims = b'<IMS-DOCUMENT>' in raw\n"
        b"    if sec == ims:\n"
        b"        raise ValueError('one exclusive envelope required')\n"
        b"    return raw\n"
    )
    _write(repo, path, baseline)
    parent = _commit(repo, "counterpart base")
    _write(repo, DOC_PATH, b"# Exact preregistration\n")
    prereg = _commit(repo, "preregister")
    _write(repo, path, approved)
    implementation = _commit(repo, "approved exclusive IMS envelope")
    pins = _pins(repo, parent, prereg)
    approved_changes, approved_ids, approved_imports = delta._symbol_changes(
        baseline, approved
    )
    assert approved_ids == ("function|parse_complete_submission|1",)
    assert approved_imports == ()
    approved_rule = delta.CounterpartRule(
        counterpart_path=path,
        expected_mechanical_counts=(0, 0, 0, 0, 0, 0, 0),
        expected_candidate_sha256=hashlib.sha256(approved).hexdigest(),
        redacted_literal_assignment=None,
        expected_redacted_candidate_sha256=None,
        external_sha256_anchor_path=None,
        external_sha256_anchor_assignment=None,
        excluded_change_symbol_ids=(),
        expected_changed_symbol_ids=approved_ids,
        expected_changed_import_specs=approved_imports,
        expected_changed_symbol_evidence_sha256=(
            delta._counterpart_change_evidence_sha256(approved_changes)
        ),
    )
    state = {
        "repo": repo,
        "implementation": implementation,
        "pins": pins,
        "allow": delta.AllowSpec(
            path_rules={
                path: _changed_rule(
                    repo, prereg[0], implementation[0], path, "M"
                )
            },
            counterpart_rules={path: approved_rule},
        ),
    }
    manifest = _manifest(state)
    assert manifest["implementation_delta"][0]["status"] == "M"

    hostile_variants = (
        approved.replace(
            b"    sec = b'<SEC-DOCUMENT>' in raw\n",
            b"    raw = raw.replace(b'<IMS-', b'<SEC-')\n"
            b"    sec = b'<SEC-DOCUMENT>' in raw\n",
        ),
        (
            b"def parse_complete_submission(raw):\n"
            b"    if b'<SEC-DOCUMENT>' in raw or b'<IMS-DOCUMENT>' in raw:\n"
            b"        return raw\n"
            b"    raise ValueError('some envelope required')\n"
        ),
        approved.replace(
            b"    if sec == ims:\n",
            b"    if raw == b'observed-failed-body':\n"
            b"        return raw\n"
            b"    if sec == ims:\n",
        ),
    )
    for index, hostile in enumerate(hostile_variants):
        _git(repo, "checkout", "-q", prereg[0])
        _write(repo, path, hostile)
        hostile_implementation = _commit(repo, f"hostile IMS parser {index}")
        state["implementation"] = hostile_implementation
        state["allow"] = delta.AllowSpec(
            path_rules={
                path: _changed_rule(
                    repo,
                    prereg[0],
                    hostile_implementation[0],
                    path,
                    "M",
                )
            },
            counterpart_rules={
                path: replace(
                    approved_rule,
                    expected_candidate_sha256=hashlib.sha256(hostile).hexdigest(),
                )
            },
        )
        with pytest.raises(delta.DeltaValidationError, match="symbol content drift"):
            _manifest(state)


def test_checker_rejects_family_asymmetric_hws_scope_expansion_and_observed_special_case(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Offline Test")
    _git(repo, "config", "user.email", "offline@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    old_path = "agent_benchmark/sec_gemma_lean_v34_source.py"
    new_path = "agent_benchmark/sec_gemma_lean_v38_source.py"
    baseline = (
        b"import re\n\n"
        b"def _exact_header_sgml_form(raw, envelope_family):\n"
        b"    pattern = (\n"
        b"        rb'^CONFORMED SUBMISSION TYPE: ([^\\r\\n]+)$'\n"
        b"        if envelope_family == 'sec'\n"
        b"        else rb'^[ \\t]*CONFORMED SUBMISSION TYPE:[ \\t]+([^\\r\\n]+)$'\n"
        b"    )\n"
        b"    matches = re.findall(pattern, raw, re.MULTILINE)\n"
        b"    return matches[0] if len(matches) == 1 else None\n"
    )
    approved = (
        b"import re\n\n"
        b"def _exact_header_sgml_form(raw, envelope_family):\n"
        b"    pattern = rb'^[ \\t]*CONFORMED SUBMISSION TYPE:[ \\t]+([^\\r\\n]+)$'\n"
        b"    matches = re.findall(pattern, raw, re.MULTILINE)\n"
        b"    return matches[0] if len(matches) == 1 else None\n"
    )
    _write(repo, old_path, baseline)
    parent = _commit(repo, "counterpart base")
    _write(repo, DOC_PATH, b"# Exact preregistration\n")
    prereg = _commit(repo, "preregister")
    _write(repo, new_path, approved)
    implementation = _commit(repo, "approved family-neutral ASCII HWS rule")
    pins = _pins(repo, parent, prereg)
    approved_changes, approved_ids, approved_imports = delta._symbol_changes(
        baseline, approved
    )
    assert approved_ids == ("function|_exact_header_sgml_form|1",)
    assert approved_imports == ()
    approved_rule = delta.CounterpartRule(
        counterpart_path=old_path,
        expected_mechanical_counts=(0, 0, 0, 0, 0, 0, 0),
        expected_candidate_sha256=hashlib.sha256(approved).hexdigest(),
        redacted_literal_assignment=None,
        expected_redacted_candidate_sha256=None,
        external_sha256_anchor_path=None,
        external_sha256_anchor_assignment=None,
        excluded_change_symbol_ids=(),
        expected_changed_symbol_ids=approved_ids,
        expected_changed_import_specs=approved_imports,
        expected_changed_symbol_evidence_sha256=(
            delta._counterpart_change_evidence_sha256(approved_changes)
        ),
    )
    state = {
        "repo": repo,
        "implementation": implementation,
        "pins": pins,
        "allow": delta.AllowSpec(
            path_rules={
                new_path: _changed_rule(
                    repo, prereg[0], implementation[0], new_path, "A"
                )
            },
            counterpart_rules={new_path: approved_rule},
        ),
    }
    assert _manifest(state)["committed_v32_counterpart_comparisons"][0][
        "changed_symbol_count"
    ] == 1

    hostile_variants = (
        baseline,
        approved.replace(b"[ \\t]", b"\\s"),
        approved.replace(
            b"    pattern = ",
            b"    raw = raw.replace(b'\\t', b' ')\n    pattern = ",
        ),
        approved.replace(
            b"    pattern = ",
            b"    if raw == b'observed-failed-body':\n"
            b"        return b'10-K'\n"
            b"    pattern = ",
        ),
    )
    for index, hostile in enumerate(hostile_variants):
        _git(repo, "checkout", "-q", prereg[0])
        _write(repo, new_path, hostile)
        hostile_implementation = _commit(repo, f"hostile HWS parser {index}")
        state["implementation"] = hostile_implementation
        state["allow"] = delta.AllowSpec(
            path_rules={
                new_path: _changed_rule(
                    repo,
                    prereg[0],
                    hostile_implementation[0],
                    new_path,
                    "A",
                )
            },
            counterpart_rules={
                new_path: replace(
                    approved_rule,
                    expected_candidate_sha256=hashlib.sha256(hostile).hexdigest(),
                )
            },
        )
        with pytest.raises(delta.DeltaValidationError, match="counterpart symbol"):
            _manifest(state)


def test_checker_rejects_fixed_offset_broad_timestamp_scope_and_observed_special_case(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Offline Test")
    _git(repo, "config", "user.email", "offline@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    old_path = "agent_benchmark/sec_gemma_lean_v35_source.py"
    new_path = "agent_benchmark/sec_gemma_lean_v38_source.py"
    baseline = (
        b"from datetime import datetime\n"
        b"import re\n"
        b"from .sec_point_in_time import (\n"
        b"    SecPointInTimeError,\n"
        b"    parse_submissions_acceptance_datetime,\n"
        b")\n\n"
        b"def consume(value):\n"
        b"    return parse_submissions_acceptance_datetime(value)\n"
    )
    approved = (
        b"from datetime import datetime, timezone\n"
        b"import re\n"
        b"from .sec_point_in_time import (\n"
        b"    SecPointInTimeError,\n"
        b"    parse_submissions_acceptance_datetime as "
        b"_parse_submissions_acceptance_wall_clock,\n"
        b")\n"
        b"from zoneinfo import ZoneInfo\n\n"
        b"def parse_submissions_acceptance_datetime(value):\n"
        b"    if not isinstance(value, str):\n"
        b"        raise SecPointInTimeError('acceptance must remain text')\n"
        b"    if re.fullmatch(r'[0-9]{14}', value):\n"
        b"        return _parse_submissions_acceptance_wall_clock(value)\n"
        b"    match = re.fullmatch(\n"
        b"        r'([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
        b"(?:\\.0{1,9})?Z',\n"
        b"        value,\n"
        b"    )\n"
        b"    if match is None:\n"
        b"        return _parse_submissions_acceptance_wall_clock(value)\n"
        b"    try:\n"
        b"        parsed = datetime.strptime(match.group(1), '%Y-%m-%dT%H:%M:%S')\n"
        b"    except ValueError as exc:\n"
        b"        raise SecPointInTimeError('invalid UTC acceptance') from exc\n"
        b"    return parsed.replace(tzinfo=timezone.utc).astimezone(\n"
        b"        ZoneInfo('America/New_York')\n"
        b"    )\n\n"
        b"def consume(value):\n"
        b"    return parse_submissions_acceptance_datetime(value)\n"
    )
    _write(repo, old_path, baseline)
    parent = _commit(repo, "counterpart base")
    _write(repo, DOC_PATH, b"# Exact preregistration\n")
    prereg = _commit(repo, "preregister")
    _write(repo, new_path, approved)
    implementation = _commit(repo, "approved representation-aware timestamp rule")
    pins = _pins(repo, parent, prereg)
    approved_changes, approved_ids, approved_imports = delta._symbol_changes(
        baseline, approved
    )
    assert approved_ids == (
        "function|parse_submissions_acceptance_datetime|1",
        "import|$module.__import__|1",
        "import|$module.__import__|3",
        "import|$module.__import__|4",
    )
    approved_rule = delta.CounterpartRule(
        counterpart_path=old_path,
        expected_mechanical_counts=(0, 0, 0, 0, 0, 0, 0),
        expected_candidate_sha256=hashlib.sha256(approved).hexdigest(),
        redacted_literal_assignment=None,
        expected_redacted_candidate_sha256=None,
        external_sha256_anchor_path=None,
        external_sha256_anchor_assignment=None,
        excluded_change_symbol_ids=(),
        expected_changed_symbol_ids=approved_ids,
        expected_changed_import_specs=approved_imports,
        expected_changed_symbol_evidence_sha256=(
            delta._counterpart_change_evidence_sha256(approved_changes)
        ),
    )
    state = {
        "repo": repo,
        "implementation": implementation,
        "pins": pins,
        "allow": delta.AllowSpec(
            path_rules={
                new_path: _changed_rule(
                    repo, prereg[0], implementation[0], new_path, "A"
                )
            },
            counterpart_rules={new_path: approved_rule},
        ),
    }
    assert _manifest(state)["committed_v32_counterpart_comparisons"][0][
        "changed_symbol_count"
    ] == 4

    hostile_variants = (
        approved.replace(
            b"return parsed.replace(tzinfo=timezone.utc).astimezone(\n"
            b"        ZoneInfo('America/New_York')\n"
            b"    )",
            b"return parsed.replace(tzinfo=timezone.utc).replace(\n"
            b"        hour=parsed.hour - 4\n"
            b"    )",
        ),
        approved.replace(
            b"r'([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
            b"(?:\\.0{1,9})?Z'",
            b"r'(.+)Z'",
        ),
        approved.replace(
            b"    if re.fullmatch(r'[0-9]{14}', value):\n",
            b"    if value == '2099-07-01T12:34:56Z':\n"
            b"        return _parse_submissions_acceptance_wall_clock('20990701123456')\n"
            b"    if re.fullmatch(r'[0-9]{14}', value):\n",
        ),
        approved.replace(
            b"return parsed.replace(tzinfo=timezone.utc).astimezone(\n"
            b"        ZoneInfo('America/New_York')\n"
            b"    )",
            b"return _parse_submissions_acceptance_wall_clock(\n"
            b"        match.group(1).replace('-', '').replace(':', '').replace('T', '')\n"
            b"    )",
        ),
    )
    for index, hostile in enumerate(hostile_variants):
        _git(repo, "checkout", "-q", prereg[0])
        _write(repo, new_path, hostile)
        hostile_implementation = _commit(repo, f"hostile timestamp parser {index}")
        _, hostile_ids, hostile_imports = delta._symbol_changes(baseline, hostile)
        assert hostile_ids == approved_ids
        assert hostile_imports == approved_imports
        state["implementation"] = hostile_implementation
        state["allow"] = delta.AllowSpec(
            path_rules={
                new_path: _changed_rule(
                    repo,
                    prereg[0],
                    hostile_implementation[0],
                    new_path,
                    "A",
                )
            },
            counterpart_rules={
                new_path: replace(
                    approved_rule,
                    expected_candidate_sha256=hashlib.sha256(hostile).hexdigest(),
                )
            },
        )
        with pytest.raises(delta.DeltaValidationError, match="symbol content drift"):
            _manifest(state)


def test_checker_rejects_calendar_suffix_weakening_caller_broadening_and_left_edge_special_case(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Offline Test")
    _git(repo, "config", "user.email", "offline@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    old_path = "agent_benchmark/sec_gemma_lean_v36_source.py"
    new_path = "agent_benchmark/sec_gemma_lean_v38_source.py"
    shared_path = "agent_benchmark/sec_session_calendar.py"
    shared = (
        b'EXPECTED_SESSIONS = ("2000-01-03",)\n'
        b'EXPECTED_MARKET_HISTORY_SESSIONS = ("1998-01-02", "2000-01-03")\n'
    )
    baseline = (
        b"from datetime import date, timedelta\n"
        b"from .sec_session_calendar import (\n"
        b"    EXPECTED_SESSIONS,\n"
        b"    validate_aapl_session_calendar,\n"
        b")\n\n"
        b'GLOBAL_AVAILABILITY_START = "2000-01-01"\n'
        b'STAGE_WINDOWS = {"development": ("2000-01-01", "2018-12-31")}\n'
        b'STAGE_SOURCE_CONFIGS = {"development": {"d_min": 72, "d_max": 80}}\n'
        b'STAGE_MODEL_CALL_CAPS = {"development": 80}\n'
        b"_EXPECTED_SESSION_SET = frozenset(EXPECTED_SESSIONS)\n\n"
        b"def _compact_authoritative_calendar_receipt():\n"
        b"    experiment = validate_aapl_session_calendar(EXPECTED_SESSIONS)\n"
        b'    return {"experiment_calendar": experiment}\n\n'
        b"_AUTHORITATIVE_CALENDAR_RECEIPT = "
        b"_compact_authoritative_calendar_receipt()\n\n"
        b"def _canonical_sessions(values):\n"
        b"    if tuple(values) != EXPECTED_SESSIONS:\n"
        b'        raise ValueError("calendar")\n'
        b"    return tuple(EXPECTED_SESSIONS)\n\n"
        b"def _stage_assignment(availability):\n"
        b"    if availability is None or "
        b"availability < GLOBAL_AVAILABILITY_START:\n"
        b"        return None\n"
        b'    return "development"\n\n'
        b"def reconcile_complete_submission(\n"
        b"    accession, filing_date, acceptance_date, change_date\n"
        b"):\n"
        b"    boundary = max(\n"
        b"        value\n"
        b"        for value in (filing_date, acceptance_date, change_date)\n"
        b"        if value is not None\n"
        b"    )\n"
        b"    availability = next(\n"
        b"        (\n"
        b"            session\n"
        b"            for session in EXPECTED_SESSIONS\n"
        b"            if session > boundary\n"
        b"        ),\n"
        b"        None,\n"
        b"    )\n"
        b"    return {\n"
        b'        "accession": accession,\n'
        b'        "filing_date": filing_date,\n'
        b'        "boundary": boundary,\n'
        b'        "availability": availability,\n'
        b'        "stage": _stage_assignment(availability),\n'
        b"    }\n\n"
        b"def _validate_target_row_binding(row):\n"
        b"    boundary = max(\n"
        b"        value\n"
        b"        for value in (\n"
        b'            row["filing_date"],\n'
        b'            row["acceptance_date"],\n'
        b'            row["change_date"],\n'
        b"        )\n"
        b"        if value is not None\n"
        b"    )\n"
        b"    availability = next(\n"
        b"        (\n"
        b"            session\n"
        b"            for session in EXPECTED_SESSIONS\n"
        b"            if session > boundary\n"
        b"        ),\n"
        b"        None,\n"
        b"    )\n"
        b'    if row["availability"] != availability or row["stage"] != '
        b"_stage_assignment(availability):\n"
        b'        raise ValueError("binding")\n'
        b"    return availability\n\n"
        b"def project_science_rows(rows):\n"
        b'    return tuple(row for row in rows if row["stage"] == "development")\n\n'
        b"def _build_stage_source_bundle(values):\n"
        b'    return {"session_dates": _canonical_sessions(values)}\n'
    )
    approved = (
        b"from datetime import date, timedelta\n"
        b"from .sec_session_calendar import (\n"
        b"    EXPECTED_MARKET_HISTORY_SESSIONS,\n"
        b"    EXPECTED_SESSIONS,\n"
        b"    validate_aapl_session_calendar,\n"
        b"    validate_market_history_session_calendar,\n"
        b")\n\n"
        b'GLOBAL_AVAILABILITY_START = "2000-01-01"\n'
        b'STAGE_WINDOWS = {"development": ("2000-01-01", "2018-12-31")}\n'
        b'STAGE_SOURCE_CONFIGS = {"development": {"d_min": 72, "d_max": 80}}\n'
        b'STAGE_MODEL_CALL_CAPS = {"development": 80}\n'
        b"_EXPECTED_SESSION_SET = frozenset(EXPECTED_MARKET_HISTORY_SESSIONS)\n\n"
        b"def _compact_authoritative_calendar_receipt():\n"
        b"    experiment = validate_aapl_session_calendar(EXPECTED_SESSIONS)\n"
        b"    history = validate_market_history_session_calendar(\n"
        b"        EXPECTED_MARKET_HISTORY_SESSIONS\n"
        b"    )\n"
        b"    suffix_start = len(EXPECTED_MARKET_HISTORY_SESSIONS) - "
        b"len(EXPECTED_SESSIONS)\n"
        b"    if (\n"
        b"        suffix_start != 504\n"
        b'        or history["sessions"][suffix_start:] != experiment["sessions"]\n'
        b"    ):\n"
        b'        raise ValueError("suffix")\n'
        b"    exact_boundary_start = (\n"
        b'        date.fromisoformat(history["start"]) - timedelta(days=1)\n'
        b"    ).isoformat()\n"
        b"    body = {\n"
        b'        "experiment_calendar": experiment,\n'
        b'        "availability_search_calendar": history,\n'
        b'        "suffix_relation": {\n'
        b'            "prefix_count": suffix_start,\n'
        b'            "ordered": True,\n'
        b"        },\n"
        b'        "availability_semantics": {\n'
        b'            "exact_boundary_start": exact_boundary_start,\n'
        b"        },\n"
        b"    }\n"
        b"    return body\n\n"
        b"_AUTHORITATIVE_CALENDAR_RECEIPT = "
        b"_compact_authoritative_calendar_receipt()\n\n"
        b"def _canonical_sessions(values):\n"
        b"    if tuple(values) != EXPECTED_SESSIONS:\n"
        b'        raise ValueError("calendar")\n'
        b"    return tuple(EXPECTED_SESSIONS)\n\n"
        b"def _stage_assignment(availability):\n"
        b"    if availability is None or "
        b"availability < GLOBAL_AVAILABILITY_START:\n"
        b"        return None\n"
        b'    return "development"\n\n'
        b"def _conservative_availability_session(boundary):\n"
        b"    if boundary < _AUTHORITATIVE_CALENDAR_RECEIPT[\n"
        b'        "availability_semantics"\n'
        b']["exact_boundary_start"]:\n'
        b"        return None\n"
        b"    return next(\n"
        b"        (\n"
        b"            session\n"
        b"            for session in EXPECTED_MARKET_HISTORY_SESSIONS\n"
        b"            if session > boundary\n"
        b"        ),\n"
        b"        None,\n"
        b"    )\n\n"
        b"def reconcile_complete_submission(\n"
        b"    accession, filing_date, acceptance_date, change_date\n"
        b"):\n"
        b"    boundary = max(\n"
        b"        value\n"
        b"        for value in (filing_date, acceptance_date, change_date)\n"
        b"        if value is not None\n"
        b"    )\n"
        b"    availability = _conservative_availability_session(boundary)\n"
        b"    return {\n"
        b'        "accession": accession,\n'
        b'        "filing_date": filing_date,\n'
        b'        "boundary": boundary,\n'
        b'        "availability": availability,\n'
        b'        "stage": _stage_assignment(availability),\n'
        b"    }\n\n"
        b"def _validate_target_row_binding(row):\n"
        b"    boundary = max(\n"
        b"        value\n"
        b"        for value in (\n"
        b'            row["filing_date"],\n'
        b'            row["acceptance_date"],\n'
        b'            row["change_date"],\n'
        b"        )\n"
        b"        if value is not None\n"
        b"    )\n"
        b"    availability = _conservative_availability_session(boundary)\n"
        b'    if row["availability"] != availability or row["stage"] != '
        b"_stage_assignment(availability):\n"
        b'        raise ValueError("binding")\n'
        b"    return availability\n\n"
        b"def project_science_rows(rows):\n"
        b'    return tuple(row for row in rows if row["stage"] == "development")\n\n'
        b"def _build_stage_source_bundle(values):\n"
        b'    return {"session_dates": _canonical_sessions(values)}\n'
    )
    project_line = (
        b'    return tuple(row for row in rows if row["stage"] == "development")\n'
    )
    approved_validator = (
        b"def _validate_target_row_binding(row):\n"
        b"    boundary = max(\n"
        b"        value\n"
        b"        for value in (\n"
        b'            row["filing_date"],\n'
        b'            row["acceptance_date"],\n'
        b'            row["change_date"],\n'
        b"        )\n"
        b"        if value is not None\n"
        b"    )\n"
        b"    availability = _conservative_availability_session(boundary)\n"
        b'    if row["availability"] != availability or row["stage"] != '
        b"_stage_assignment(availability):\n"
        b'        raise ValueError("binding")\n'
        b"    return availability\n"
    )
    baseline_validator = (
        b"def _validate_target_row_binding(row):\n"
        b"    boundary = max(\n"
        b"        value\n"
        b"        for value in (\n"
        b'            row["filing_date"],\n'
        b'            row["acceptance_date"],\n'
        b'            row["change_date"],\n'
        b"        )\n"
        b"        if value is not None\n"
        b"    )\n"
        b"    availability = next(\n"
        b"        (\n"
        b"            session\n"
        b"            for session in EXPECTED_SESSIONS\n"
        b"            if session > boundary\n"
        b"        ),\n"
        b"        None,\n"
        b"    )\n"
        b'    if row["availability"] != availability or row["stage"] != '
        b"_stage_assignment(availability):\n"
        b'        raise ValueError("binding")\n'
        b"    return availability\n"
    )

    _write(repo, old_path, baseline)
    _write(repo, shared_path, shared)
    parent = _commit(repo, "counterpart base")
    _write(repo, DOC_PATH, b"# Exact preregistration\n")
    prereg = _commit(repo, "preregister")
    _write(repo, new_path, approved)
    implementation = _commit(repo, "approved calendar correction")
    pins = _pins(repo, parent, prereg)
    approved_changes, approved_ids, approved_imports = delta._symbol_changes(
        baseline, approved
    )
    assert approved_ids == (
        "assignment|$module._EXPECTED_SESSION_SET|1",
        "function|_compact_authoritative_calendar_receipt|1",
        "function|_conservative_availability_session|1",
        "function|_validate_target_row_binding|1",
        "function|reconcile_complete_submission|1",
        "import|$module.__import__|2",
    )
    approved_rule = delta.CounterpartRule(
        counterpart_path=old_path,
        expected_mechanical_counts=(0, 0, 0, 0, 0, 0, 0),
        expected_candidate_sha256=hashlib.sha256(approved).hexdigest(),
        redacted_literal_assignment=None,
        expected_redacted_candidate_sha256=None,
        external_sha256_anchor_path=None,
        external_sha256_anchor_assignment=None,
        excluded_change_symbol_ids=(),
        expected_changed_symbol_ids=approved_ids,
        expected_changed_import_specs=approved_imports,
        expected_changed_symbol_evidence_sha256=(
            delta._counterpart_change_evidence_sha256(approved_changes)
        ),
    )
    state = {
        "repo": repo,
        "implementation": implementation,
        "pins": pins,
        "allow": delta.AllowSpec(
            path_rules={
                new_path: _changed_rule(
                    repo,
                    prereg[0],
                    implementation[0],
                    new_path,
                    "A",
                )
            },
            counterpart_rules={new_path: approved_rule},
        ),
    }
    assert _manifest(state)["committed_v32_counterpart_comparisons"][0][
        "changed_symbol_count"
    ] == 6

    hostile_variants = (
        ("d-max", approved.replace(b'"d_max": 80', b'"d_max": 128'), None),
        (
            "science-cap",
            approved.replace(
                b'STAGE_MODEL_CALL_CAPS = {"development": 80}',
                b'STAGE_MODEL_CALL_CAPS = {"development": 128}',
            ),
            None,
        ),
        (
            "stage-window",
            approved.replace(
                b'STAGE_WINDOWS = {"development": '
                b'("2000-01-01", "2018-12-31")}',
                b'STAGE_WINDOWS = {"development": '
                b'("1998-01-01", "2018-12-31")}',
            ),
            None,
        ),
        (
            "global-stage-start",
            approved.replace(
                b'GLOBAL_AVAILABILITY_START = "2000-01-01"',
                b'GLOBAL_AVAILABILITY_START = "1998-01-01"',
            ),
            None,
        ),
        (
            "sample",
            approved.replace(
                project_line,
                b'    return tuple(row for row in rows if row["stage"] == '
                b'"development")[::2]\n',
            ),
            None,
        ),
        (
            "truncate",
            approved.replace(
                project_line,
                b'    return tuple(row for row in rows if row["stage"] == '
                b'"development")[:80]\n',
            ),
            None,
        ),
        (
            "identity-drop",
            approved.replace(
                project_line,
                b"    return tuple(\n"
                b"        row\n"
                b"        for row in rows\n"
                b'        if row["stage"] == "development"\n'
                b'        and row["accession"] != "0000320193-94-000001"\n'
                b"    )\n",
            ),
            None,
        ),
        (
            "literal-d97",
            approved.replace(
                project_line,
                b"    selected = tuple(\n"
                b'        row for row in rows if row["stage"] == "development"\n'
                b"    )\n"
                b"    return selected[:80] if len(selected) == 97 else selected\n",
            ),
            None,
        ),
        (
            "greater-equal",
            approved.replace(
                b"            if session > boundary\n",
                b"            if session >= boundary\n",
            ),
            None,
        ),
        (
            "current-only-search",
            approved.replace(
                b"            for session in EXPECTED_MARKET_HISTORY_SESSIONS\n",
                b"            for session in EXPECTED_SESSIONS\n",
            ),
            None,
        ),
        (
            "left-edge-equality",
            approved.replace(
                b"    if boundary < _AUTHORITATIVE_CALENDAR_RECEIPT[\n",
                b"    if boundary <= _AUTHORITATIVE_CALENDAR_RECEIPT[\n",
            ),
            None,
        ),
        (
            "sentinel-clamp",
            approved.replace(
                b']["exact_boundary_start"]:\n'
                b"        return None\n",
                b']["exact_boundary_start"]:\n'
                b"        return EXPECTED_MARKET_HISTORY_SESSIONS[0]\n",
            ),
            None,
        ),
        (
            "filing-date-only",
            approved.replace(
                b"    boundary = max(\n"
                b"        value\n"
                b"        for value in (filing_date, acceptance_date, change_date)\n"
                b"        if value is not None\n"
                b"    )\n",
                b"    boundary = filing_date\n",
            ).replace(
                b"    boundary = max(\n"
                b"        value\n"
                b"        for value in (\n"
                b'            row["filing_date"],\n'
                b'            row["acceptance_date"],\n'
                b'            row["change_date"],\n'
                b"        )\n"
                b"        if value is not None\n"
                b"    )\n",
                b'    boundary = row["filing_date"]\n',
            ),
            None,
        ),
        (
            "old-accession-special-case",
            approved.replace(
                b"    availability = "
                b"_conservative_availability_session(boundary)\n"
                b"    return {\n"
                b'        "accession": accession,\n',
                b"    availability = (\n"
                b"        None\n"
                b'        if accession == "0000320193-94-000001"\n'
                b"        else _conservative_availability_session(boundary)\n"
                b"    )\n"
                b"    return {\n"
                b'        "accession": accession,\n',
            ),
            None,
        ),
        (
            "caller-history",
            approved.replace(
                b"    if tuple(values) != EXPECTED_SESSIONS:\n"
                b'        raise ValueError("calendar")\n'
                b"    return tuple(EXPECTED_SESSIONS)\n",
                b"    if tuple(values) not in (\n"
                b"        EXPECTED_SESSIONS,\n"
                b"        EXPECTED_MARKET_HISTORY_SESSIONS,\n"
                b"    ):\n"
                b'        raise ValueError("calendar")\n'
                b"    return tuple(values)\n",
            ),
            None,
        ),
        (
            "replay-history",
            approved.replace(
                b'    return {"session_dates": _canonical_sessions(values)}\n',
                b"    _canonical_sessions(values)\n"
                b'    return {"session_dates": '
                b"tuple(EXPECTED_MARKET_HISTORY_SESSIONS)}\n",
            ),
            None,
        ),
        (
            "unordered-suffix",
            approved.replace(
                b"    if (\n"
                b"        suffix_start != 504\n"
                b'        or history["sessions"][suffix_start:] '
                b'!= experiment["sessions"]\n'
                b"    ):\n",
                b"    if (\n"
                b"        suffix_start != 504\n"
                b'        or history["end"] != experiment["end"]\n'
                b'        or history["session_count"] '
                b"- experiment[\"session_count\"] != 504\n"
                b'        or set(history["sessions"][suffix_start:]) '
                b'!= set(experiment["sessions"])\n'
                b"    ):\n",
            ),
            None,
        ),
        (
            "omit-composite",
            approved.replace(b"    return body\n", b"    return experiment\n"),
            None,
        ),
        (
            "parse-only",
            approved.replace(approved_validator, baseline_validator),
            None,
        ),
        (
            "shared-calendar",
            approved,
            b'EXPECTED_SESSIONS = ("2000-01-03", "2000-01-04")\n',
        ),
    )
    assert len(hostile_variants) == 20
    for label, hostile, shared_payload in hostile_variants:
        assert hostile != approved or shared_payload is not None, label
        _git(repo, "checkout", "-q", prereg[0])
        _write(repo, new_path, hostile)
        if shared_payload is not None:
            _write(repo, shared_path, shared_payload)
        hostile_implementation = _commit(repo, f"hostile calendar {label}")
        state["implementation"] = hostile_implementation
        state["allow"] = delta.AllowSpec(
            path_rules={
                new_path: _changed_rule(
                    repo,
                    prereg[0],
                    hostile_implementation[0],
                    new_path,
                    "A",
                )
            },
            counterpart_rules={
                new_path: replace(
                    approved_rule,
                    expected_candidate_sha256=hashlib.sha256(hostile).hexdigest(),
                )
            },
        )
        with pytest.raises(
            delta.DeltaValidationError,
            match=r"counterpart symbol|path inventory",
        ):
            _manifest(state)


def test_checker_rejects_checkpoint_main_receipt_omission_substitution_and_indirection(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Offline Test")
    _git(repo, "config", "user.email", "offline@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    old_acquisition_path = "agent_benchmark/sec_gemma_lean_v37_acquisition.py"
    new_acquisition_path = "agent_benchmark/sec_gemma_lean_v38_acquisition.py"
    old_source_path = "agent_benchmark/sec_gemma_lean_v37_source.py"
    new_source_path = "agent_benchmark/sec_gemma_lean_v38_source.py"
    old_test_path = "tests/test_sec_gemma_lean_v37_acquisition.py"
    new_test_path = "tests/test_sec_gemma_lean_v38_acquisition.py"
    old_acquisition = (
        b"class DiskBackedSecAcquisition:\n"
        b"    def __init__(self, source):\n"
        b"        self.source = source\n\n"
        b"    def _run_locked(self, records, stage_output, phase_receipt_sha256):\n"
        b"        return self.source.build_compact_checkpoint(\n"
        b"            records=records,\n"
        b"            stage_output=stage_output,\n"
        b"            source_phase_receipt_sha256=phase_receipt_sha256,\n"
        b"        )\n\n"
        b"    def preserve(self):\n"
        b"        return True\n"
    )
    approved_acquisition = old_acquisition.replace(
        b"            stage_output=stage_output,\n"
        b"            source_phase_receipt_sha256=phase_receipt_sha256,\n",
        b"            stage_output=stage_output,\n"
        b"            main_parse_receipt_sha256=records[0].manifest[\n"
        b'                "parse_receipt_sha256"\n'
        b"            ],\n"
        b"            source_phase_receipt_sha256=phase_receipt_sha256,\n",
    )
    source = (
        b"class CheckpointSource:\n"
        b"    def build_compact_checkpoint(\n"
        b"        self,\n"
        b"        *,\n"
        b"        records,\n"
        b"        stage_output,\n"
        b"        source_phase_receipt_sha256,\n"
        b"        main_parse_receipt_sha256,\n"
        b"    ):\n"
        b'        if records[0].role != "submissions/main":\n'
        b'            raise ValueError("first role")\n'
        b"        if (\n"
        b"            type(main_parse_receipt_sha256) is not str\n"
        b"            or len(main_parse_receipt_sha256) != 64\n"
        b"        ):\n"
        b'            raise ValueError("canonical hash")\n'
        b"        if main_parse_receipt_sha256 != records[0].manifest[\n"
        b'            "parse_receipt_sha256"\n'
        b"        ]:\n"
        b'            raise ValueError("main parse receipt")\n'
        b"        return {\n"
        b'            "main_parse_receipt_sha256": main_parse_receipt_sha256,\n'
        b'            "stage_output": stage_output,\n'
        b'            "source_phase_receipt_sha256": source_phase_receipt_sha256,\n'
        b"        }\n"
    )
    old_test = (
        b"class _FakeSource:\n"
        b"    def build_compact_checkpoint(self, **kwargs):\n"
        b'        manifests = kwargs["manifests"]\n'
        b"        main_parse_receipt_sha256 = manifests[0][\n"
        b'            "parse_receipt_sha256"\n'
        b"        ]\n"
        b"        return {\n"
        b'            "main_parse_receipt_sha256": main_parse_receipt_sha256\n'
        b"        }\n"
    )
    approved_test = old_test.replace(
        b"        main_parse_receipt_sha256 = manifests[0][\n"
        b'            "parse_receipt_sha256"\n'
        b"        ]\n",
        b"        main_parse_receipt_sha256 = kwargs[\n"
        b'            "main_parse_receipt_sha256"\n'
        b"        ]\n"
        b"        assert main_parse_receipt_sha256 == manifests[0][\n"
        b'            "parse_receipt_sha256"\n'
        b"        ]\n",
    )
    base_blobs = {
        old_acquisition_path: old_acquisition,
        old_source_path: source,
        old_test_path: old_test,
    }
    for path, payload in base_blobs.items():
        _write(repo, path, payload)
    parent = _commit(repo, "counterpart base")
    _write(repo, DOC_PATH, b"# Exact preregistration\n")
    prereg = _commit(repo, "preregister")
    approved_candidates = {
        new_acquisition_path: approved_acquisition,
        new_source_path: source,
        new_test_path: approved_test,
    }
    for path, payload in approved_candidates.items():
        _write(repo, path, payload)
    implementation = _commit(repo, "approved explicit checkpoint binding")
    pins = _pins(repo, parent, prereg)
    expected_changed_ids = {
        new_acquisition_path: (
            "class|DiskBackedSecAcquisition|1",
            "function|DiskBackedSecAcquisition._run_locked|1",
        ),
        new_source_path: (),
        new_test_path: (
            "class|_FakeSource|1",
            "function|_FakeSource.build_compact_checkpoint|1",
        ),
    }
    counterpart_paths = {
        new_acquisition_path: old_acquisition_path,
        new_source_path: old_source_path,
        new_test_path: old_test_path,
    }
    approved_counterpart_rules = {}
    for path, payload in approved_candidates.items():
        counterpart = base_blobs[counterpart_paths[path]]
        changes, symbol_ids, import_specs = delta._symbol_changes(
            counterpart, payload
        )
        assert symbol_ids == expected_changed_ids[path]
        assert import_specs == ()
        approved_counterpart_rules[path] = delta.CounterpartRule(
            counterpart_path=counterpart_paths[path],
            expected_mechanical_counts=(0, 0, 0, 0, 0, 0, 0),
            expected_candidate_sha256=hashlib.sha256(payload).hexdigest(),
            redacted_literal_assignment=None,
            expected_redacted_candidate_sha256=None,
            external_sha256_anchor_path=None,
            external_sha256_anchor_assignment=None,
            excluded_change_symbol_ids=(),
            expected_changed_symbol_ids=symbol_ids,
            expected_changed_import_specs=import_specs,
            expected_changed_symbol_evidence_sha256=(
                delta._counterpart_change_evidence_sha256(changes)
            ),
        )
    state = {
        "repo": repo,
        "implementation": implementation,
        "pins": pins,
        "allow": delta.AllowSpec(
            path_rules={
                path: _changed_rule(
                    repo, prereg[0], implementation[0], path, "A"
                )
                for path in approved_candidates
            },
            counterpart_rules=approved_counterpart_rules,
        ),
    }
    approved_manifest = _manifest(state)
    assert {
        item["candidate_path"]: tuple(item["changed_symbol_ids"])
        for item in approved_manifest["committed_v32_counterpart_comparisons"]
    } == expected_changed_ids

    binding = (
        b"            main_parse_receipt_sha256=records[0].manifest[\n"
        b'                "parse_receipt_sha256"\n'
        b"            ],\n"
    )
    hostile_variants = (
        ("omitted-keyword", new_acquisition_path, old_acquisition),
        (
            "public-v37-digest",
            new_acquisition_path,
            approved_acquisition.replace(
                binding,
                b'            main_parse_receipt_sha256="31d0d7587a7263b294bb156d81ceaf6f06c073d66562baacf8330361747054a5",\n',
            ),
        ),
        (
            "literal-digest",
            new_acquisition_path,
            approved_acquisition.replace(
                binding,
                b'            main_parse_receipt_sha256="1111111111111111111111111111111111111111111111111111111111111111",\n',
            ),
        ),
        (
            "last-record",
            new_acquisition_path,
            approved_acquisition.replace(binding, binding.replace(b"[0]", b"[-1]")),
        ),
        (
            "second-record",
            new_acquisition_path,
            approved_acquisition.replace(binding, binding.replace(b"[0]", b"[1]")),
        ),
        (
            "phase-receipt",
            new_acquisition_path,
            approved_acquisition.replace(
                binding,
                b"            main_parse_receipt_sha256=phase_receipt_sha256,\n",
            ),
        ),
        (
            "manifest-get",
            new_acquisition_path,
            approved_acquisition.replace(
                binding,
                b"            main_parse_receipt_sha256=records[0].manifest.get(\n"
                b'                "parse_receipt_sha256"\n'
                b"            ),\n",
            ),
        ),
        (
            "manifest-fallback",
            new_acquisition_path,
            approved_acquisition.replace(
                binding,
                b"            main_parse_receipt_sha256=records[0].manifest.get(\n"
                b'                "parse_receipt_sha256", phase_receipt_sha256\n'
                b"            ),\n",
            ),
        ),
        (
            "string-cast",
            new_acquisition_path,
            approved_acquisition.replace(
                binding,
                b"            main_parse_receipt_sha256=str(\n"
                b"                records[0].manifest[\"parse_receipt_sha256\"]\n"
                b"            ),\n",
            ),
        ),
        (
            "dynamic-kwargs",
            new_acquisition_path,
            approved_acquisition.replace(
                b"        return self.source.build_compact_checkpoint(\n",
                b"        checkpoint_kwargs = {\n"
                b'            "main_parse_receipt_sha256": records[0].manifest[\n'
                b'                "parse_receipt_sha256"\n'
                b"            ]\n"
                b"        }\n"
                b"        return self.source.build_compact_checkpoint(\n",
            ).replace(binding, b"            **checkpoint_kwargs,\n"),
        ),
        (
            "new-helper",
            new_acquisition_path,
            approved_acquisition.replace(
                b"class DiskBackedSecAcquisition:\n",
                b"def _main_receipt(records):\n"
                b'    return records[0].manifest["parse_receipt_sha256"]\n\n'
                b"class DiskBackedSecAcquisition:\n",
            ).replace(
                binding,
                b"            main_parse_receipt_sha256=_main_receipt(records),\n",
            ),
        ),
        (
            "new-import",
            new_acquisition_path,
            b"import hashlib\n\n" + approved_acquisition,
        ),
        (
            "optional-source-parameter",
            new_source_path,
            source.replace(
                b"        main_parse_receipt_sha256,\n"
                b"    ):\n",
                b"        main_parse_receipt_sha256=None,\n"
                b"    ):\n",
            ),
        ),
        (
            "inferred-source-value",
            new_source_path,
            source.replace(
                b"    ):\n"
                b'        if records[0].role != "submissions/main":\n',
                b"    ):\n"
                b"        if main_parse_receipt_sha256 is None:\n"
                b"            main_parse_receipt_sha256 = records[0].manifest[\n"
                b'                "parse_receipt_sha256"\n'
                b"            ]\n"
                b'        if records[0].role != "submissions/main":\n',
            ),
        ),
        (
            "weakened-source-equality",
            new_source_path,
            source.replace(
                b"        if main_parse_receipt_sha256 != records[0].manifest[\n",
                b"        if False and main_parse_receipt_sha256 != records[0].manifest[\n",
            ),
        ),
        ("fake-silent-derivation", new_test_path, old_test),
        (
            "second-production-change",
            new_acquisition_path,
            approved_acquisition.replace(
                b"    def preserve(self):\n"
                b"        return True\n",
                b"    def preserve(self):\n"
                b"        return False\n",
            ),
        ),
    )
    assert len(hostile_variants) == 17
    for label, hostile_path, hostile_payload in hostile_variants:
        assert hostile_payload != approved_candidates[hostile_path], label
        _git(repo, "checkout", "-q", prereg[0])
        candidate_payloads = dict(approved_candidates)
        candidate_payloads[hostile_path] = hostile_payload
        for path, payload in candidate_payloads.items():
            _write(repo, path, payload)
        hostile_implementation = _commit(repo, f"hostile checkpoint {label}")
        hostile_counterpart_rules = dict(approved_counterpart_rules)
        hostile_counterpart_rules[hostile_path] = replace(
            approved_counterpart_rules[hostile_path],
            expected_candidate_sha256=hashlib.sha256(hostile_payload).hexdigest(),
        )
        state["implementation"] = hostile_implementation
        state["allow"] = delta.AllowSpec(
            path_rules={
                path: _changed_rule(
                    repo,
                    prereg[0],
                    hostile_implementation[0],
                    path,
                    "A",
                )
                for path in candidate_payloads
            },
            counterpart_rules=hostile_counterpart_rules,
        )
        with pytest.raises(
            delta.DeltaValidationError,
            match=r"counterpart symbol|counterpart import",
        ):
            _manifest(state)


def test_sha_anchor_masks_only_digest_and_rejects_noncanonical_forms() -> None:
    name = "V38_DELTA_MODULE_SHA256"
    first = "1" * 64
    second = "2" * 64
    canonical = f'{name} = "{first}"\n\ndef test_bound():\n    pass\n'.encode("ascii")
    canonical_hash, value = delta._masked_sha256_literal_assignment_sha256(
        canonical, name
    )
    assert value == first
    changed_digest = canonical.replace(first.encode("ascii"), second.encode("ascii"))
    assert delta._masked_sha256_literal_assignment_sha256(changed_digest, name)[0] == (
        canonical_hash
    )
    changed_spacing = canonical.replace(b" = ", b"  = ")
    assert delta._masked_sha256_literal_assignment_sha256(changed_spacing, name)[0] != (
        canonical_hash
    )
    changed_non_anchor = canonical.replace(b"pass", b"assert True")
    assert delta._masked_sha256_literal_assignment_sha256(changed_non_anchor, name)[0] != (
        canonical_hash
    )
    with pytest.raises(delta.DeltaValidationError, match="canonical double-quoted"):
        delta._masked_sha256_literal_assignment_sha256(
            canonical.replace(f'"{first}"'.encode("ascii"), f"'{first}'".encode("ascii")),
            name,
        )
    with pytest.raises(delta.DeltaValidationError, match="exactly once"):
        delta._masked_sha256_literal_assignment_sha256(
            canonical.replace(name.encode("ascii"), b"OTHER_SHA256"), name
        )
    with pytest.raises(delta.DeltaValidationError, match="plain string constant"):
        delta._masked_sha256_literal_assignment_sha256(
            canonical.replace(
                f'"{first}"'.encode("ascii"),
                f'"{first[:32]}" + "{first[32:]}"'.encode("ascii"),
            ),
            name,
        )
    with pytest.raises(delta.DeltaValidationError, match="exactly once"):
        delta._masked_sha256_literal_assignment_sha256(canonical + canonical, name)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"expected_mechanical_counts": (1, 0, 0, 0, 0, 0, 0)}, "mechanical replacement count"),
        ({"expected_changed_symbol_evidence_sha256": "1" * 64}, "symbol content drift"),
        ({"expected_changed_import_specs": ("import:socket",)}, "import inventory drift"),
        ({"counterpart_path": "agent_benchmark/missing_v37.py"}, "Git rejected"),
    ),
)
def test_counterpart_baseline_count_import_and_evidence_tamper_fail_closed(
    changes: dict[str, Any], message: str, tmp_path: Path
) -> None:
    state = _make_counterpart_repo(tmp_path)
    rule = replace(state["counterpart_rule"], **changes)
    state["allow"] = delta.AllowSpec(
        path_rules=state["allow"].path_rules,
        counterpart_rules={state["new_path"]: rule},
    )
    with pytest.raises(delta.DeltaValidationError, match=message):
        _manifest(state)


def test_external_anchor_rejects_delta_body_with_stale_test_literal(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Offline Test")
    _git(repo, "config", "user.email", "offline@example.invalid")
    old_path = "agent_benchmark/sec_gemma_lean_v37_delta.py"
    new_path = "agent_benchmark/sec_gemma_lean_v38_delta.py"
    anchor_path = "tests/test_sec_gemma_lean_v38_delta.py"
    benign = b"def verifier():\n    return True\n"
    hostile = b"def verifier():\n    return False\n"
    _write(repo, old_path, benign)
    parent = _commit(repo, "base")
    _write(repo, DOC_PATH, b"# Exact preregistration\n")
    prereg = _commit(repo, "preregister")
    _write(repo, new_path, hostile)
    benign_sha = hashlib.sha256(benign).hexdigest()
    _write(
        repo,
        anchor_path,
        f'V38_DELTA_MODULE_SHA256 = "{benign_sha}"\n'.encode("ascii"),
    )
    implementation = _commit(repo, "stale external anchor")
    pins = _pins(repo, parent, prereg)
    rule = delta.CounterpartRule(
        counterpart_path=old_path,
        expected_mechanical_counts=(0, 0, 0, 0, 0, 0, 0),
        expected_candidate_sha256=None,
        redacted_literal_assignment=None,
        expected_redacted_candidate_sha256=None,
        external_sha256_anchor_path=anchor_path,
        external_sha256_anchor_assignment="V38_DELTA_MODULE_SHA256",
        excluded_change_symbol_ids=(
            "assignment|$module._COUNTERPART_RULE_DATA|1",
        ),
        expected_changed_symbol_ids=(),
        expected_changed_import_specs=(),
        expected_changed_symbol_evidence_sha256=hashlib.sha256(b"[]").hexdigest(),
    )
    with pytest.raises(delta.DeltaValidationError, match="external blob anchor mismatch"):
        delta._counterpart_evidence(
            repo,
            new_path,
            hostile,
            rule,
            pins,
            implementation[0],
            git_binary="git",
        )


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
