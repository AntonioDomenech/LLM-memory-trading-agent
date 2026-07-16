from __future__ import annotations

import ast
from pathlib import Path

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    NEW_SOURCE_FILES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PREFIX = "aapl-sec-gemma-online-risk-overlay-"
V21_PREFIX = "aapl-sec-gemma-online-risk-overlay-v2-1-"


def _assigned_name(node: ast.AST) -> str | None:
    target: ast.AST | None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def test_every_overlay_schema_and_verifier_namespace_is_v2_1() -> None:
    observed = 0
    for relative_path in NEW_SOURCE_FILES.values():
        path = REPO_ROOT / relative_path
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            name = _assigned_name(node)
            if name is None or not (
                "SCHEMA_VERSION" in name
                or name.endswith("VERIFIER_ID")
            ):
                continue
            value_node = (
                node.value
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                else None
            )
            try:
                value = ast.literal_eval(value_node)
            except (TypeError, ValueError):
                continue
            if (
                type(value) is str
                and value.startswith(ARTIFACT_PREFIX)
            ):
                observed += 1
                assert value.startswith(V21_PREFIX), (
                    f"{relative_path}:{name} is not a v2.1 namespace: {value}"
                )
    assert observed >= 30
