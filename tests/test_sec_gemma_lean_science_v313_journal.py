from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_benchmark import sec_gemma_lean_science_v313_journal as journal_module
from agent_benchmark.sec_gemma_lean_science_v313_journal import (
    HashChainJournal,
    V313JournalError,
    ZERO_SHA256,
    canonical_json_bytes,
    canonical_sha256,
    replay_journal,
)


AUTHORITY_SHA256 = "a" * 64


def _root(tmp_path: Path, name: str = "journal") -> Path:
    root = (tmp_path / name).resolve()
    root.mkdir()
    return root


def test_canonical_json_is_utf8_sorted_and_rejects_native_floats() -> None:
    assert canonical_json_bytes({"z": "café", "a": [True, 2]}) == (
        '{"a":[true,2],"z":"café"}'.encode("utf-8")
    )
    assert canonical_sha256({"a": 1}) == canonical_sha256({"a": 1})
    with pytest.raises(V313JournalError) as caught:
        canonical_json_bytes({"binary64": 1.0})
    assert caught.value.code == "canonical_json_float_forbidden"


def test_append_is_gap_free_hash_chained_and_canonical(tmp_path: Path) -> None:
    root = _root(tmp_path)
    journal = HashChainJournal(root, authority_sha256=AUTHORITY_SHA256)
    first = journal.append("attempt_intent", {"value": 1})
    second = journal.append("request_intent", {"value": 2})

    replay = replay_journal(root, authority_sha256=AUTHORITY_SHA256)
    assert replay.event_count == 2
    assert first.previous_event_sha256 == ZERO_SHA256
    assert second.previous_event_sha256 == first.event_sha256
    assert replay.head_event_sha256 == second.event_sha256
    assert [path.name[:8] for path in sorted(root.iterdir())] == [
        "00000001",
        "00000002",
    ]
    for event in replay.events:
        encoded = event.path.read_bytes()
        assert encoded == canonical_json_bytes(json.loads(encoded))


def test_wrong_authority_tamper_gap_and_extra_entry_fail_closed(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    journal = HashChainJournal(root, authority_sha256=AUTHORITY_SHA256)
    journal.append("attempt_intent", {"value": 1})
    second = journal.append("request_intent", {"value": 2})

    with pytest.raises(V313JournalError) as caught:
        replay_journal(root, authority_sha256="b" * 64)
    assert caught.value.code == "journal_invalid"

    second.path.unlink()
    third_name = second.path.name.replace("00000002", "00000003", 1)
    (root / third_name).write_bytes(canonical_json_bytes(second.as_dict()))
    with pytest.raises(V313JournalError) as caught:
        replay_journal(root, authority_sha256=AUTHORITY_SHA256)
    assert caught.value.code == "journal_sequence_invalid"

    (root / third_name).unlink()
    (root / "foreign.txt").write_text("x", encoding="ascii")
    with pytest.raises(V313JournalError) as caught:
        replay_journal(root, authority_sha256=AUTHORITY_SHA256)
    assert caught.value.code == "journal_extra_entry"


def test_modified_event_bytes_break_hash_replay(tmp_path: Path) -> None:
    root = _root(tmp_path)
    event = HashChainJournal(
        root, authority_sha256=AUTHORITY_SHA256
    ).append("attempt_intent", {"value": 1})
    value = json.loads(event.path.read_text(encoding="utf-8"))
    value["payload"]["value"] = 2
    event.path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(V313JournalError) as caught:
        replay_journal(root, authority_sha256=AUTHORITY_SHA256)
    assert caught.value.code == "journal_hash_mismatch"


def test_replay_rechecks_size_of_descriptor_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    HashChainJournal(root, authority_sha256=AUTHORITY_SHA256).append(
        "attempt_intent", {"value": 1}
    )
    monkeypatch.setattr(
        journal_module,
        "read_regular_bytes",
        lambda _path: b"x" * (journal_module.MAX_EVENT_BYTES + 1),
    )
    with pytest.raises(V313JournalError) as caught:
        replay_journal(root, authority_sha256=AUTHORITY_SHA256)
    assert caught.value.code == "journal_invalid"


def test_every_append_fsyncs_file_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        journal_module.os, "fsync", lambda _descriptor: calls.append("file")
    )
    monkeypatch.setattr(
        journal_module,
        "fsync_directory",
        lambda directory: calls.append(f"directory:{directory.name}"),
    )
    HashChainJournal(root, authority_sha256=AUTHORITY_SHA256).append(
        "attempt_intent", {"value": 1}
    )
    assert calls == ["file", f"directory:{root.name}"]


def test_directory_fsync_failure_leaves_poison_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    journal = HashChainJournal(root, authority_sha256=AUTHORITY_SHA256)

    def fail(_directory: Path) -> None:
        raise V313JournalError("journal_directory_fsync_failed")

    monkeypatch.setattr(journal_module, "fsync_directory", fail)
    with pytest.raises(V313JournalError) as caught:
        journal.append("attempt_intent", {"value": 1})
    assert caught.value.code == "journal_directory_fsync_failed"
    assert list(root.glob(".append-failed-*.json"))
    with pytest.raises(V313JournalError) as replay_caught:
        replay_journal(root, authority_sha256=AUTHORITY_SHA256)
    assert replay_caught.value.code == "journal_append_incomplete"


@pytest.mark.parametrize(
    "name",
    [
        ".pending-00000001-" + "0" * 64 + ".json",
        ".append-failed-00000001-" + "0" * 64 + ".json",
    ],
)
def test_pending_or_failed_append_is_never_adopted(
    tmp_path: Path, name: str
) -> None:
    root = _root(tmp_path, name="poison")
    (root / name).write_bytes(b"{}\n")
    with pytest.raises(V313JournalError) as caught:
        replay_journal(root, authority_sha256=AUTHORITY_SHA256)
    assert caught.value.code == "journal_append_incomplete"


def test_relative_or_missing_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(V313JournalError) as caught:
        HashChainJournal(Path("relative"), authority_sha256=AUTHORITY_SHA256)
    assert caught.value.code == "journal_directory_invalid"
    with pytest.raises(V313JournalError) as caught:
        HashChainJournal(
            (tmp_path / "missing").resolve(), authority_sha256=AUTHORITY_SHA256
        )
    assert caught.value.code == "journal_directory_invalid"


def test_append_exclusive_create_never_follows_foreign_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    foreign = (tmp_path / "foreign.txt").resolve()
    foreign.write_bytes(b"do-not-touch")
    original = journal_module._write_exclusive_fsynced
    attacked = False

    def install_link_then_write(path: Path, payload: bytes) -> None:
        nonlocal attacked
        if not attacked and path.name.startswith(".pending-"):
            try:
                path.symlink_to(foreign)
            except OSError:
                os.link(foreign, path)
            attacked = True
        original(path, payload)

    monkeypatch.setattr(
        journal_module, "_write_exclusive_fsynced", install_link_then_write
    )
    with pytest.raises(V313JournalError) as caught:
        HashChainJournal(root, authority_sha256=AUTHORITY_SHA256).append(
            "attempt_intent", {"value": 1}
        )
    assert caught.value.code == "journal_append_durability_failed"
    assert foreign.read_bytes() == b"do-not-touch"


def test_append_publication_never_follows_swapped_staging_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    foreign = (tmp_path / "foreign.txt").resolve()
    foreign.write_bytes(b"do-not-touch")
    original = journal_module.publish_noreplace

    def swap_before_publish(source: Path, destination: Path, *, code: str) -> None:
        source.unlink()
        try:
            source.symlink_to(foreign)
        except OSError:
            os.link(foreign, source)
        original(source, destination, code=code)

    monkeypatch.setattr(journal_module, "publish_noreplace", swap_before_publish)
    with pytest.raises(V313JournalError):
        HashChainJournal(root, authority_sha256=AUTHORITY_SHA256).append(
            "attempt_intent", {"value": 1}
        )
    assert foreign.read_bytes() == b"do-not-touch"
    assert not any(path.name.startswith("00000001-") for path in root.iterdir())
