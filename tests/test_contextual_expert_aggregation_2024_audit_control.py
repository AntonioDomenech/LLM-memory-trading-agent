from __future__ import annotations

from pathlib import Path

import pytest

from agent_benchmark import contextual_expert_aggregation_2024_audit_control as control
from agent_benchmark import contextual_expert_aggregation_2024_audit_input as audit_input


def test_dependency_inventory_excludes_market_and_known_result_artifacts() -> None:
    values = {path.as_posix() for path in control.FROZEN_DEPENDENCY_PATHS}
    assert audit_input.INPUT_PATH.as_posix() not in values
    assert audit_input.RECEIPT_PATH.as_posix() not in values
    assert "e/APPROACH_COMPARISON.md" not in values
    assert not any("quarantine" in value.lower() for value in values)
    assert not any("through_2025" in value.lower() for value in values)
    assert not any("through_2026" in value.lower() for value in values)


def test_literal_git_path_rejects_parent_traversal() -> None:
    with pytest.raises(control.AuditControlError, match="unsafe"):
        control._literal("../outside")


def test_hardening_rejects_a_redirected_control_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    control_root = root / "e" / "aapl_causal_contextual_expert_aggregation_2024_audit_v2"
    control_root.parent.mkdir()
    try:
        control_root.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("test platform does not permit unprivileged directory symlinks")
    with pytest.raises(control.AuditControlError):
        control.harden_runtime_paths(root)
