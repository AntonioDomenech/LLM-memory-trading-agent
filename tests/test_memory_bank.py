import json

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
