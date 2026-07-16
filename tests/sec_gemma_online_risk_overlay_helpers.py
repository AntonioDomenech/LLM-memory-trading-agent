"""Shared deterministic fixtures for the SEC/Gemma overlay tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_synthetic_numerical_time_distributions(
    *,
    digest: Callable[[str], str],
    canonical_sha256: Callable[[Any], str],
) -> list[dict[str, Any]]:
    """Build the exact synthetic NumPy/tzdata manifest shape used in tests."""

    result: list[dict[str, Any]] = []
    for import_name, origin_path, additional_path in (
        (
            "numpy",
            "numpy/__init__.py",
            "numpy/_core/_multiarray_umath.pyd",
        ),
        (
            "tzdata",
            "tzdata/__init__.py",
            "tzdata/zoneinfo/America/New_York",
        ),
    ):
        runtime_files = sorted(
            [
                {
                    "path": origin_path,
                    "sha256": digest(
                        f"{import_name}:runtime:{origin_path}"
                    ),
                },
                {
                    "path": additional_path,
                    "sha256": digest(
                        f"{import_name}:runtime:{additional_path}"
                    ),
                },
            ],
            key=lambda item: item["path"],
        )
        origin_sha256 = next(
            item["sha256"]
            for item in runtime_files
            if item["path"] == origin_path
        )
        result.append(
            {
                "import_name": import_name,
                "distribution": import_name,
                "version": f"test-{import_name}-1",
                "module_origin": {
                    "path": origin_path,
                    "sha256": origin_sha256,
                },
                "runtime_files": runtime_files,
                "runtime_files_sha256": canonical_sha256(runtime_files),
                "direct_url_sha256": None,
            }
        )
    return result
