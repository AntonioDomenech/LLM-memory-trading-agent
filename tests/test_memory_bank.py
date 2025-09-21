import json

import pytest

from core.memory import MemoryBank


def test_memory_bank_save_without_parent_directory(tmp_path):
    """Saving memories should work when the target path has no parent folder."""

    path = tmp_path / "memory.json"
    bank = MemoryBank(str(path), emb_model="text-embedding-3-small")

    bank.add_item(
        "shallow",
        "Test memory entry",
        {"date": "2024-01-01", "source": "unit-test"},
        base_importance=1.0,
        seen_date="2024-01-01",
    )

    assert path.exists()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["shallow"], "Expected shallow layer to contain the saved entry"


def test_prune_layer_filters_by_importance_and_date(tmp_path):
    """Low-importance and stale memories should be removed when pruning."""

    path = tmp_path / "memory.json"
    bank = MemoryBank(str(path), emb_model="text-embedding-3-small")

    bank.add_item(
        "shallow",
        "Keep me: high importance",
        {"date": "2024-01-05"},
        base_importance=9.0,
        seen_date="2024-01-05",
    )
    bank.add_item(
        "shallow",
        "Discard: low importance",
        {"date": "2024-01-04"},
        base_importance=2.0,
        seen_date="2024-01-04",
    )
    bank.add_item(
        "shallow",
        "Discard: stale",
        {"date": "2023-12-28"},
        base_importance=8.0,
        seen_date="2023-12-28",
    )

    removed = bank.prune_layer("shallow", min_importance=5.0, before_date="2024-01-01")
    assert removed == 2

    reloaded = MemoryBank(str(path), emb_model="text-embedding-3-small")
    remaining_texts = {item["text"] for item in reloaded.layers["shallow"]}

    assert "Keep me: high importance" in remaining_texts
    assert "Discard: low importance" not in remaining_texts
    assert "Discard: stale" not in remaining_texts


def test_prune_layer_respects_max_items(tmp_path):
    """Pruning with a ``max_items`` limit should keep the most relevant items."""

    path = tmp_path / "memory.json"
    bank = MemoryBank(str(path), emb_model="text-embedding-3-small")

    bank.add_item(
        "shallow",
        "Alpha",
        {"date": "2024-01-01"},
        base_importance=3.0,
        seen_date="2024-01-01",
    )
    bank.add_item(
        "shallow",
        "Bravo",
        {"date": "2024-01-02"},
        base_importance=9.0,
        seen_date="2024-01-02",
    )
    bank.add_item(
        "shallow",
        "Charlie",
        {"date": "2024-01-03"},
        base_importance=6.0,
        seen_date="2024-01-03",
    )

    removed = bank.prune_layer("shallow", max_items=2)
    assert removed == 1

    remaining = [item["text"] for item in bank.layers["shallow"]]
    assert remaining == ["Bravo", "Charlie"]


def test_prune_layer_invalid_date_raises(tmp_path):
    """Invalid ``before_date`` inputs should raise a ``ValueError``."""

    path = tmp_path / "memory.json"
    bank = MemoryBank(str(path), emb_model="text-embedding-3-small")
    bank.add_item(
        "shallow",
        "Sample",
        {"date": "2024-01-01"},
        base_importance=5.0,
        seen_date="2024-01-01",
    )

    with pytest.raises(ValueError):
        bank.prune_layer("shallow", before_date="not-a-date")
